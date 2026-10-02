#!/usr/bin/env python3
"""
MCP Server for MERRA2 data analysis.
Exposes tools via HTTP endpoint and JSON-RPC protocol.
"""

import sys
import json
import uuid
import asyncio
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from mcp_downscaling_tools import MERRA2Analyzer, process_tool_call

# Initialize FastAPI app
app = FastAPI(title="MERRA2 MCP Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer
analyzer = MERRA2Analyzer()

CONFIG_PATH = CURRENT_DIR / "mcp-merra2-config.json"


def load_tools_from_config() -> list[dict]:
    """Load tool definitions from JSON config."""
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)
    tools = config.get("tools")
    if not isinstance(tools, list):
        raise ValueError(f"Invalid tools list in {CONFIG_PATH}")
    return tools


TOOLS = load_tools_from_config()
from config import state_dir  # noqa: E402

ARTIFACTS_DIR = state_dir() / "artifacts"

# Pydantic models
class ToolCall(BaseModel):
    name: str
    arguments: dict


def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Execute a tool and return the result as a string."""
    
    try:
        return process_tool_call(tool_name, tool_args, analyzer)
    
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


# HTTP Endpoints

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "MERRA2 MCP Server"}


@app.get("/tools")
async def list_tools():
    """List all available tools."""
    return {"tools": TOOLS}


@app.post("/call")
async def call_tool(tool_call: ToolCall):
    """Call a tool with given arguments."""
    logger.info(f"Calling tool: {tool_call.name} with args: {tool_call.arguments}")
    
    result = execute_tool(tool_call.name, tool_call.arguments)
    
    return {
        "tool": tool_call.name,
        "result": result
    }


@app.get("/artifacts/{filename}")
async def get_artifact(filename: str):
    """Serve generated plot artifacts."""
    path = (ARTIFACTS_DIR / filename).resolve()
    if not path.exists() or not path.is_file() or path.parent != ARTIFACTS_DIR.resolve():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path=path, media_type="image/png", filename=path.name)


@app.get("/jobs")
async def list_mcp_jobs():
    """List all subprocess jobs (training, inference, preprocessing, etc.)."""
    from mcp_downscaling_tools import _job_manager
    return json.loads(_job_manager.list_jobs())


@app.delete("/jobs/completed")
async def clear_mcp_completed_jobs():
    """Remove terminal jobs (done/failed/cancelled/blocked) from MCP job history."""
    from mcp_downscaling_tools import _job_manager
    result = json.loads(_job_manager.clear_completed_jobs())
    return result


@app.post("/jobs/{job_id}/stop")
async def stop_mcp_job(job_id: str):
    """Cancel a running subprocess job."""
    from mcp_downscaling_tools import _job_manager
    result = json.loads(_job_manager.cancel_job(job_id))
    if result.get("status") == "error":
        msg = result.get("message", "")
        code = 404 if "not found" in msg.lower() else 400
        raise HTTPException(status_code=code, detail=msg)
    return {"ok": True, "job_id": job_id}


@app.delete("/jobs/{job_id}")
async def delete_mcp_job(job_id: str):
    """Delete one terminal subprocess job from persisted history."""
    from mcp_downscaling_tools import _job_manager
    result = json.loads(_job_manager.delete_job(job_id))
    if result.get("status") == "error":
        msg = result.get("message", "")
        code = 404 if "not found" in msg.lower() else 400
        raise HTTPException(status_code=code, detail=msg)
    return {"ok": True, "job_id": job_id, "removed_log": bool(result.get("removed_log", False))}


@app.get("/jobs/{job_id}/logs")
async def get_mcp_job_logs(job_id: str, lines: int = 200):
    """Return the last N lines of a job's log file."""
    from mcp_downscaling_tools import _job_manager
    with _job_manager._lock:
        job = _job_manager._jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    log_path = Path(job.get("log_file", ""))
    if not log_path.exists():
        return {"job_id": job_id, "lines": [], "total_lines": 0, "log_file": str(log_path)}
    try:
        all_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return {"job_id": job_id, "lines": tail, "total_lines": len(all_lines), "log_file": str(log_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── MCP SSE Transport (standard MCP protocol for external IDEs) ───────────────
# Supports: Claude Desktop, Cursor, Windsurf, VS Code MCP extensions, etc.
# Connect via:  http://<host>:8001/sse

_sse_sessions: dict[str, asyncio.Queue] = {}


@app.get("/sse")
async def mcp_sse(request: Request):
    """SSE endpoint for standard MCP protocol. IDEs connect here."""
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _sse_sessions[session_id] = queue

    async def event_stream():
        # Tell the client where to POST its JSON-RPC messages
        yield f"event: endpoint\ndata: /messages/{session_id}\n\n"
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15)
                    if msg is None:
                        break
                    yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"  # keep connection alive
        finally:
            _sse_sessions.pop(session_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/messages/{session_id}")
async def mcp_message(session_id: str, request: Request):
    """Receive JSON-RPC messages from MCP clients; push responses to SSE stream."""
    queue = _sse_sessions.get(session_id)
    if not queue:
        raise HTTPException(status_code=404, detail="MCP session not found")

    body = await request.json()
    method = body.get("method", "")
    req_id = body.get("id")          # None for notifications
    params = body.get("params", {})

    # Notifications have no id — no response needed
    if req_id is None:
        return {"ok": True}

    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "Prithvi-WxC-Downscaling MCP", "version": "1.0.0"},
        }

    elif method == "tools/list":
        # TOOLS entries already use inputSchema (camelCase) from the config JSON
        result = {"tools": TOOLS}

    elif method == "tools/call":
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        raw = execute_tool(name, arguments)
        result = {"content": [{"type": "text", "text": raw}], "isError": False}

    else:
        await queue.put({
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        })
        return {"ok": True}

    await queue.put({"jsonrpc": "2.0", "id": req_id, "result": result})
    return {"ok": True}


# ── MCP Streamable-HTTP Transport ─────────────────────────────────────────────
# For use with mcp-remote or any client that supports streamable-HTTP.
# Connect via:  npx -y mcp-remote http://localhost:8001/mcp --transport http-only

def _handle_jsonrpc(body: dict) -> dict:
    method = body.get("method", "")
    req_id = body.get("id")
    params = body.get("params", {})

    if req_id is None:
        # Notification — no response
        return None

    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "Prithvi-WxC-Downscaling MCP", "version": "1.0.0"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        raw = execute_tool(name, arguments)
        result = {"content": [{"type": "text", "text": raw}], "isError": False}
    else:
        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}}

    return {"jsonrpc": "2.0", "id": req_id, "result": result}


@app.post("/mcp")
async def mcp_http(request: Request):
    """Streamable-HTTP MCP endpoint. Works with mcp-remote --transport http-only."""
    body = await request.json()

    # Handle JSON-RPC batch (array) or single message
    if isinstance(body, list):
        responses = [_handle_jsonrpc(msg) for msg in body]
        responses = [r for r in responses if r is not None]
        return responses
    else:
        resp = _handle_jsonrpc(body)
        if resp is None:
            return {}
        return resp


if __name__ == "__main__":
    logger.info("Starting MERRA2 MCP Server on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
