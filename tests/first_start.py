#!/usr/bin/env python3
"""First-start behaviour of bin/prithvi-mcp, the way an MCP host sees it.

    python tests/first_start.py install      # empty venv: stand-in answers, background
                                             # install, hand-over to the full server
    python tests/first_start.py bad-env      # broken settings file is reported as a tool error
    python tests/first_start.py no-python    # no Python >= 3.11 and no uv is reported

Each case uses temporary directories; nothing outside them is touched.
"""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "bin" / "prithvi-mcp"


class Host:
    def __init__(self, env: dict):
        self.proc = subprocess.Popen([str(LAUNCHER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=env)
        self.next_id = 0
        self.buf = b""

    def send(self, payload: dict) -> None:
        self.proc.stdin.write(json.dumps(payload).encode() + b"\n")
        self.proc.stdin.flush()

    def read(self, timeout: float) -> dict | None:
        deadline = time.time() + timeout
        while b"\n" not in self.buf:
            left = deadline - time.time()
            if left <= 0 or not select.select([self.proc.stdout], [], [], left)[0]:
                return None
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                return None
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line)

    def request(self, method: str, params: dict | None = None, timeout: float = 60) -> dict:
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}})
        while True:
            msg = self.read(timeout)
            assert msg is not None, f"no reply to {method}"
            if msg.get("id") == self.next_id:
                return msg["result"]

    def call(self, name: str, **args) -> tuple[str, bool]:
        res = self.request("tools/call", {"name": name, "arguments": args})
        return res["content"][0]["text"], res["isError"]

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=30)


def base_env(tmp: Path) -> dict:
    env = dict(os.environ)
    env.update(PRITHVI_MCP_VENV=str(tmp / "cache" / "venv"), PIPELINE_DATA_ROOT=str(tmp / "data"),
               PRITHVI_WXC_ENV_FILE=str(tmp / "no-settings"))
    return env


def check(cond: bool, msg: str, detail: object = "") -> None:
    if not cond:
        raise AssertionError(f"{msg}: {detail}")
    print(f"ok  {msg}")


def handshake(host: Host) -> dict:
    start = time.time()
    init = host.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                       "clientInfo": {"name": "first-start", "version": "0"}}, timeout=30)
    host.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    check(time.time() - start < 10, f"handshake answered in {time.time() - start:.1f} s")
    return init


def case_install(tmp: Path) -> None:
    host = Host(base_env(tmp))
    try:
        init = handshake(host)
        check(init["serverInfo"]["version"] == "setup", "stand-in server answered while packages are missing")
        tools = [t["name"] for t in host.request("tools/list")["tools"]]
        check(tools == ["prithvi_setup_status"], "stand-in offers only the status tool", tools)
        text, err = host.call("prithvi_setup_status")
        check(err and "installing" in text, "status tool reports the install in progress", text[:200])
        text, err = host.call("check_raw_data_status")
        check(err and "not available yet" in text, "real tools explain they are not ready", text[:120])

        deadline = time.time() + 900
        notified = False
        while time.time() < deadline and not notified:
            msg = host.read(timeout=10)
            notified = bool(msg) and msg.get("method") == "notifications/tools/list_changed"
        check(notified, "full server took over and announced its tools (same session)")
        tools = [t["name"] for t in host.request("tools/list")["tools"]]
        check(len(tools) >= 40 and "start_download_job" in tools, f"tool list now has {len(tools)} tools")
        status = json.loads(host.call("check_raw_data_status")[0])
        check("missing" in status, "real tools work after the hand-over", list(status)[:5])
    finally:
        host.close()


def case_bad_env(tmp: Path) -> None:
    env_file = tmp / "env"
    env_file.write_text("PIPELINE_DATA_ROOT=/tmp/x\nBROKEN=(unclosed\n")
    env = base_env(tmp)
    env["PRITHVI_WXC_ENV_FILE"] = str(env_file)
    host = Host(env)
    try:
        handshake(host)
        text, err = host.call("start_download_job", dataset="prism")
        check(err and "settings file" in text and str(env_file) in text, "broken settings file is reported", text[:200])
    finally:
        host.close()


def case_no_python(tmp: Path) -> None:
    fake = tmp / "fakebin"
    fake.mkdir()
    real = shutil.which("python3")
    for name in ("python3", "python3.11", "python3.12", "python3.13", "python3.14"):
        script = fake / name
        script.write_text("#!/bin/bash\n"
                          'case "$*" in *"version_info >= (3, 11)"*) exit 1;; esac\n'
                          f'exec {real} "$@"\n')
        script.chmod(0o755)
    env = base_env(tmp)
    env["PATH"] = f"{fake}:/usr/bin:/bin"
    env["HOME"] = str(tmp / "home")  # hide ~/.local/bin and ~/.cargo/bin
    host = Host(env)
    try:
        handshake(host)
        text, err = host.call("prithvi_setup_status")
        check(err and "Python 3.11 or newer" in text, "missing Python is reported with the fix", text[:200])
    finally:
        host.close()


def main() -> int:
    cases = {"install": case_install, "bad-env": case_bad_env, "no-python": case_no_python}
    chosen = sys.argv[1:] or list(cases)
    for name in chosen:
        print(f"== {name}")
        with tempfile.TemporaryDirectory(prefix=f"prithvi-first-{name}-") as tmp:
            cases[name](Path(tmp))
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
