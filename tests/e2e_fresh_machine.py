#!/usr/bin/env python3
"""End-to-end "fresh machine" test that drives the MCP server only through tool calls.

Downloads real (small) data into an empty data root:
  * clones the pinned code (setup_code)
  * 2 days of PRISM ppt/tmax/tmin
  * 1 day of MERRA-2 (needs Earthdata credentials in env or ~/.netrc; skipped otherwise)
  * elevation from ELEVATION_SOURCE_FILE if set (checks the reference sha256)
then localizes a config, preflights it, and inspects run manifests + replay.

The 17 GB weights download and GPU training are not exercised.

    python tests/e2e_fresh_machine.py --root /path/to/empty/dir
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_stdio import Client  # noqa: E402


def wait(client: Client, job: dict, timeout: int = 3600) -> dict:
    job_id = job["job_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, _ = client.call("get_job_status", job_id=job_id)
        if status.get("status") in {"done", "failed", "cancelled", "blocked"}:
            return status
        time.sleep(5)
    raise TimeoutError(job_id)


def check(cond: bool, msg: str, detail: object = "") -> None:
    if not cond:
        raise AssertionError(f"{msg}: {detail}")
    print(f"ok  {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Empty directory to use as PIPELINE_DATA_ROOT")
    ap.add_argument("--skip-merra2", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    env = {k: v for k, v in os.environ.items() if not k.startswith(("GRANITE_WXC", "MERRA2_DATA", "PRISM_DATA", "NARR_DATA"))}
    env["PIPELINE_DATA_ROOT"] = str(root)
    client = Client(env)
    client.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                  "clientInfo": {"name": "e2e", "version": "0"}})
    try:
        job, err = client.call("setup_code")
        check(not err, "setup_code queued", job)
        status = wait(client, job)
        check(status["status"] == "done", "pinned code cloned", status.get("log_tail"))
        code_root = root / "code" / "Prithvi-UNet-stocahstic"

        job, _ = client.call("start_download_job", dataset="prism", start_date="2020-01-01", end_date="2020-01-02")
        status = wait(client, job)
        prism_files = sorted((root / "prism" / "prism_daily_800m_an").rglob("*.nc"))
        check(status["status"] == "done" and len(prism_files) == 6, "PRISM 2 days x 3 vars downloaded",
              [p.name for p in prism_files] or status.get("log_tail"))

        if not args.skip_merra2:
            job, _ = client.call("start_download_job", dataset="merra2", start_date="2020-01-01", end_date="2020-01-01")
            status = wait(client, job)
            merra = sorted((root / "merra2" / "daily_subset_with_H").glob("*.nc4"))
            check(status["status"] == "done" and len(merra) == 1, "MERRA-2 1 day downloaded + subset with H",
                  [p.name for p in merra] or status.get("log_tail"))

        if os.getenv("ELEVATION_SOURCE_FILE"):
            job, _ = client.call("start_download_job", dataset="elevation")
            status = wait(client, job)
            check(status["status"] == "done", "elevation fetched", status.get("log_tail"))
            manifest, _ = client.call("get_run_manifest", job_id=job["job_id"])
            check(manifest["status"] == "ok", "download job wrote a manifest", manifest)

        # A config whose input paths only exist on some other server.
        base = code_root / "examples" / "MERRA_PRISM" / "MERRA_PRISM.yaml"
        foreign = base.with_name("custom_foreign_paths.yaml")
        foreign.write_text(base.read_text()
                           .replace("/data/merra2/daily_subset_with_H", "/nonexistent/merra2/daily_subset_with_H")
                           .replace("/data/PRISM/prism_daily_800m_an", "/nonexistent/PRISM/prism_daily_800m_an"))
        made, err = client.call("create_custom_yaml", base_config=str(foreign), output_name="e2e",
                                training_start="2020-01-01", training_end="2020-01-02",
                                inference_start="2020-01-01", inference_end="2020-01-02")
        check(not err and any("localized" in c for c in made["changes"]), "create_custom_yaml localized paths",
              made.get("changes"))
        cfg_path = made["config_path"]

        pre, _ = client.call("preflight_check", config_path=cfg_path, stage="training")
        missing = " ".join(pre["missing"])
        check(not pre["ok"] and "weights" in missing and "predictor" not in missing
              and "PRISM" not in missing, "preflight sees local data, flags only what is absent", pre["missing"])

        blocked, err = client.call("run_training_pipeline", config_path=cfg_path, num_gpus=1)
        check(err and blocked.get("stage") == "preflight", "run_training_pipeline refuses to queue without inputs", blocked)

        runs, _ = client.call("list_run_manifests")
        check(len(runs["runs"]) >= 2, "every job has a run manifest", runs)
        replay, err = client.call("replay_run", job_id=runs["runs"][-1]["job_id"], dry_run=True)
        check(not err and replay["dry_run"], "replay_run produces a plan", replay)
    finally:
        client.close()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
