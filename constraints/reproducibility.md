---
type: Constraint
title: Reproducibility rules
description: Path policy, MCP pipeline order, and guardrails for reproducible PrithviWxC downscaling runs.
tags: [constraint, reproducibility, paths, mcp]
timestamp: 2026-10-01T06:00:00Z
---

Use MCP tools for all pipeline work. Do not invent file paths or run training by chaining shell commands when `run_training_pipeline` exists.

# Path policy

- New run writes go under the granite-wxc checkout (`GRANITE_WXC_REPO`).
- Raw downloads go under `PIPELINE_DATA_ROOT` or `~/prithvi-wxc-data`.
- Archived / historical reads may come from shared mounts when present.
- Do **not** hardcode another machine’s absolute paths in new configs. Prefer YAML from `create_custom_yaml`.

# Pipeline order

1. `list_available_configs` then `read_yaml_config` on the closest match.
2. `create_custom_yaml` for a new case (never overwrite without confirmation).
3. `start_compute_scalars_job` if scalars are missing.
4. `start_preprocessing_job` (depends on scalars).
5. Training: `run_training_pipeline` only. Inference-only: `start_inference_job`.
6. Monitor with `get_job_status`, `list_jobs`, `get_gpu_status`.

# Guardrails

- Confirm `num_epochs` and `num_gpus` before training.
- Never delete files in `mcp/artifacts/` unless the user asks.
- If a date is outside file coverage, report actual coverage from NetCDF metadata.
- If scalars are recomputed, warn that downstream training/inference must be re-run.

# Related

- [Downscale playbook](/playbooks/downscale-wxc.md)
- [Analyze playbook](/playbooks/analyze-wxc.md)
- [MCP pipeline tools](/tools/mcp-pipeline-tools.md)

# Citations

[1] [OKF constraint example](https://okf.md/examples)
[2] [OKF spec](https://okf.md/spec)
