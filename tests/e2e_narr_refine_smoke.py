#!/usr/bin/env python3
"""GPU smoke test of NARR downscaling plus stochastic refinement, MCP tool calls only.

Checks out the stochastic_refinement code pin, downloads one training month and
one validation and one inference week of NARR and PRISM (plus elevation),
builds a separate smoke case from NARR_PRISM_subdomain.yaml, and runs
run_training_pipeline with a refiner: preprocessing -> scalars -> 2 short
Phase-1 epochs -> tiled inference -> evaluation -> Phase-1 cache -> refiner
training -> ensemble inference -> deterministic-vs-refined evaluation. Every
job must finish. Metrics are not checked; the run is far too short for them
to mean anything.

Needs an NVIDIA GPU and either setup_training_env in --root or
GRANITE_WXC_PYTHON, plus the weights (or MODEL_WEIGHTS_FILE). About 1.5 h on
one A100-class GPU after downloads.

    python tests/e2e_narr_refine_smoke.py --root ~/prithvi-wxc-data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_fresh_machine import check, wait  # noqa: E402
from smoke_stdio import Client  # noqa: E402

TRAINING = ("2014-01-01", "2014-01-31")
VALIDATION = ("2015-01-01", "2015-01-07")
INFERENCE = ("2016-01-01", "2016-01-07")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--refinement-type", default="diffusion_unet",
                    choices=["diffusion_unet", "diffusion_transformer", "flow_matching_unet",
                             "flow_matching_transformer"])
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()

    drop = ("GRANITE_WXC_REPO", "GRANITE_WXC_SCRIPT_DIR", "NARR_DATA_DIR", "PRISM_DATA_DIR",
            "ELEVATION_FILE", "MERRA2_DATA_DIR")
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop)}
    env["PIPELINE_DATA_ROOT"] = str(root)
    env["GRANITE_WXC_VARIANT"] = "stochastic_refinement"
    client = Client(env)
    client.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                  "clientInfo": {"name": "narr-refine-smoke", "version": "0"}})
    try:
        pins = json.loads((Path(__file__).resolve().parents[1] / "mcp" / "pins.json").read_text())
        pinned = pins["code"]["variants"]["stochastic_refinement"]["commit"]
        report, _ = client.call("check_environment")
        if not report["code"].get("is_git") or report["code"].get("commit") != pinned:
            job, err = client.call("setup_code", variant="stochastic_refinement")
            check(not err, "queued setup_code (stochastic_refinement)", job)
            status = wait(client, job)
            check(status["status"] == "done", "pinned stochastic_refinement code checked out", status.get("log_tail"))
            report, _ = client.call("check_environment")
        check(report["training_python"].get("torch"), "training env present", report["next_steps"])

        jobs = []
        for start, end in (TRAINING, VALIDATION, INFERENCE):
            for dataset in ("narr", "prism"):
                job, err = client.call("start_download_job", dataset=dataset, start_date=start, end_date=end)
                check(not err, f"queued {dataset} {start}..{end}", job)
                jobs.append(job)
        job, err = client.call("start_download_job", dataset="elevation")
        check(not err, "queued elevation", job)
        jobs.append(job)
        for job in jobs:
            status = wait(client, job, timeout=4 * 3600)
            check(status["status"] == "done", f"download {job['job_id']} done", status.get("log_tail"))

        base = root / "code" / "Prithvi-UNet-stocahstic" / "examples" / "NARR_PRISM" / "NARR_PRISM_subdomain.yaml"
        made, err = client.call(
            "create_custom_yaml", base_config=str(base), output_name="narr_refine_smoke",
            case_name="mcp_narr_refine_smoke", num_epochs=2, num_gpus=args.num_gpus,
            training_start=TRAINING[0], training_end=TRAINING[1],
            validation_start=VALIDATION[0], validation_end=VALIDATION[1],
            inference_start=INFERENCE[0], inference_end=INFERENCE[1],
            extra_overrides={"limit_steps_train": 300, "limit_steps_valid": 20})
        check(not err, "smoke config created", made)
        cfg = made["config_path"]
        print("    changes:", *made["changes"], sep="\n      ")

        pre, _ = client.call("preflight_check", config_path=cfg, stage="training")
        check(pre["ok"], "preflight passes", pre)

        options = {"refinement_type": args.refinement_type, "refinement_epochs": 20,
                   "ensemble_size": 4, "save_every": 1}
        if args.refinement_type.startswith("diffusion_"):
            options["refiner_clip_sample_range"] = 3
        pipe, err = client.call("run_training_pipeline", config_path=cfg, num_gpus=args.num_gpus, **options)
        check(not err, "pipeline queued: " + " -> ".join(j["stage"] for j in pipe["jobs"]), pipe)
        check(pipe["jobs"][-1]["stage"] == "refinement_evaluation", "refinement stages queued", pipe["jobs"])
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
