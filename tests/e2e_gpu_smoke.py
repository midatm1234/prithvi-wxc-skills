#!/usr/bin/env python3
"""GPU smoke test of the full pipeline through MCP tool calls only.

Downloads the 7 days the repo's MERRA_PRISM_smoke.yaml needs (training
2015-01-01..06, inference 2016-01-03), localizes the config to --root, then runs
run_training_pipeline (compute_scalars -> preprocessing -> 6 training steps ->
inference) and checks every job finished with a manifest.

Needs: setup_code + setup_training_env already done in --root, an NVIDIA GPU,
Earthdata credentials, and the weights (or MODEL_WEIGHTS_FILE pointing at them).

    python tests/e2e_gpu_smoke.py --root ~/prithvi-wxc-data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_fresh_machine import check, wait  # noqa: E402
from smoke_stdio import Client  # noqa: E402

RANGES = [("2015-01-01", "2015-01-06"), ("2016-01-03", "2016-01-03")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--force-local", action="store_true",
                    help="Pretend the base config's data paths do not exist (testing on a host that has them)")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()

    env = {k: v for k, v in os.environ.items() if not k.startswith(("GRANITE_WXC", "MERRA2_DATA", "PRISM_DATA"))}
    env["PIPELINE_DATA_ROOT"] = str(root)
    client = Client(env)
    client.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                  "clientInfo": {"name": "gpu-smoke", "version": "0"}})
    try:
        report, _ = client.call("check_environment")
        check(report["training_python"].get("torch") and report["code"].get("is_git"),
              "pinned code and training env present", report["next_steps"])

        jobs = []
        for start, end in RANGES:
            for dataset in ("merra2", "prism"):
                job, err = client.call("start_download_job", dataset=dataset, start_date=start, end_date=end)
                check(not err, f"queued {dataset} {start}..{end}", job)
                jobs.append(job)
        for job in jobs:
            status = wait(client, job)
            check(status["status"] == "done", f"download {job['job_id']} done", status.get("log_tail"))

        base = root / "code" / "Prithvi-UNet-stocahstic" / "examples" / "MERRA_PRISM" / "MERRA_PRISM_smoke.yaml"
        if args.force_local:
            foreign = base.with_name("custom_smoke_foreign_paths.yaml")
            text = base.read_text()
            for lab in ("/data/merra2/", "/data/PRISM/", "/data2/NARR/"):
                text = text.replace(lab, "/nonexistent" + lab)
            foreign.write_text(text)
            base = foreign
        made, err = client.call("create_custom_yaml", base_config=str(base), output_name="gpu_smoke",
                                case_name="mcp_gpu_smoke", num_gpus=args.num_gpus)
        check(not err, "smoke config created", made)
        cfg = made["config_path"]
        print("    changes:", *made["changes"], sep="\n      ")

        pre, _ = client.call("preflight_check", config_path=cfg, stage="training")
        check(pre["ok"], "preflight passes", pre)

        pipe, err = client.call("run_training_pipeline", config_path=cfg, num_gpus=args.num_gpus,
                                save_every=1, queue_inference=True)
        check(not err, "pipeline queued: " + " -> ".join(j["stage"] for j in pipe["jobs"]), pipe)
        for job in pipe["jobs"]:
            status = wait(client, job, timeout=6 * 3600)
            check(status["status"] == "done", f"{job['stage']} ({job['job_id']}) done", status.get("log_tail"))
            manifest, _ = client.call("get_run_manifest", job_id=job["job_id"])
            m = manifest["manifest"]
            print(f"    manifest reproducible={m['reproducible']} {m['non_reproducible_reasons']}")
    finally:
        client.close()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
