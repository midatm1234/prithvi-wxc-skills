#!/usr/bin/env python3
"""End-to-end smoke test of the stdio MCP server, as an MCP host would drive it.

Starts mcp/mcp_stdio.py against an empty temporary data root (a "fresh laptop"),
performs the MCP handshake, lists tools, and calls the read-only tools. Nothing
is downloaded.

    python tests/smoke_stdio.py            # newline-delimited JSON (MCP spec)
    python tests/smoke_stdio.py --framed   # legacy Content-Length framing
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "mcp" / "mcp_stdio.py"
FRAMED = "--framed" in sys.argv


class Client:
    def __init__(self, env: dict):
        self.proc = subprocess.Popen([sys.executable, str(SERVER)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        self.next_id = 0

    def send(self, payload: dict) -> None:
        data = json.dumps(payload).encode()
        if FRAMED:
            self.proc.stdin.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
        else:
            self.proc.stdin.write(data + b"\n")
        self.proc.stdin.flush()

    def recv(self) -> dict:
        line = self.proc.stdout.readline()
        if FRAMED:
            length = int(line.split(b":")[1])
            self.proc.stdout.readline()
            return json.loads(self.proc.stdout.read(length))
        if not line:
            raise RuntimeError("server closed stdout:\n" + self.proc.stderr.read().decode()[-3000:])
        return json.loads(line)

    def request(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}})
        reply = self.recv()
        assert reply.get("id") == self.next_id, reply
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["result"]

    def call(self, name: str, **arguments) -> tuple[dict, bool]:
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        return json.loads(result["content"][0]["text"]), result["isError"]

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=30)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="prithvi-smoke-") as tmp:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("GRANITE_WXC", "MERRA2_", "PRISM_", "NARR_"))}
        env.update(PIPELINE_DATA_ROOT=tmp, HOME=tmp)
        client = Client(env)
        try:
            init = client.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                                 "clientInfo": {"name": "smoke", "version": "0"}})
            assert init["protocolVersion"] == "2025-06-18", init
            client.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            names = {t["name"] for t in client.request("tools/list")["tools"]}
            for required in ("check_environment", "start_download_job", "setup_code", "setup_training_env",
                             "create_custom_yaml", "run_training_pipeline", "start_inference_job",
                             "get_run_manifest", "replay_run"):
                assert required in names, f"missing tool {required}"
            print(f"ok  handshake + {len(names)} tools ({'Content-Length' if FRAMED else 'newline'} framing)")

            env_report, is_err = client.call("check_environment", include_training_python=False)
            assert not is_err and env_report["data_root"] == str(Path(tmp).resolve()), env_report
            steps = [s.get("tool") or s["do"] for s in env_report["next_steps"]]
            assert "setup_code" in steps and "start_download_job" in steps, steps
            print(f"ok  check_environment on empty root -> next_steps {steps}")

            status, _ = client.call("check_raw_data_status")
            assert set(status["missing"]) >= {"prism", "elevation", "weights", "code"}, status["missing"]
            print(f"ok  check_raw_data_status missing={status['missing']}")

            bad, is_err = client.call("start_download_job", dataset="merra2")
            assert is_err and "start_date" in bad["message"], bad
            print("ok  start_download_job validates required dates")

            first, is_err = client.call("start_download_job", dataset="prism", start_date="2020-01-01", end_date="2020-01-01")
            assert not is_err, first
            clash, is_err = client.call("start_download_job", dataset="prism", start_date="2019-12-31", end_date="2020-01-02")
            assert is_err and first["job_id"] in clash["message"], clash
            client.call("cancel_job", job_id=first["job_id"])
            print("ok  overlapping downloads of the same dataset are refused")

            unknown, is_err = client.call("no_such_tool")
            assert is_err, unknown
            print("ok  unknown tool reported as isError")

            runs, _ = client.call("list_run_manifests")
            assert str(Path(tmp).resolve()) in runs["runs_dir"], runs
            assert [r["job_id"] for r in runs["runs"]] == [first["job_id"]], runs
            assert runs["runs"][0]["reproducible"], runs
            print("ok  manifests + state live under the data root; download manifest reproducible")
        finally:
            client.close()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
