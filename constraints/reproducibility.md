---
type: Constraint
title: Reproducibility rules
description: Path policy, MCP pipeline order, and guardrails for reproducible PrithviWxC downscaling runs.
tags: [constraint, reproducibility, paths, mcp]
timestamp: 2026-10-01T06:00:00Z
---

Use MCP tools for all pipeline work. Do not invent file paths or run training by chaining shell commands when `run_training_pipeline` exists.

# What must match for identical results

Pinned in `mcp/pins.json` and checked automatically; any deviation is listed in the run manifest's `non_reproducible_reasons`.

| Input | Pin |
|-------|-----|
| Training/inference code | `midatm1234/Prithvi-UNet-stocahstic` commit per variant (`setup_code`) |
| Backbone package | `NASA-IMPACT/Prithvi-WxC` commit |
| Pretrained weights | Hugging Face `ibm-granite/granite-geospatial-wxc-downscaling` revision + sha256 |
| Python packages | `env/training-requirements.lock.txt` (no xesmf: reference runs used the xarray-linear regridder) |
| Static orography | reference `prism_elevation.nc` on Zenodo (DOI 10.5281/zenodo.23096854) + sha256 |
| MERRA-2 / NARR / PRISM | provider versions + the config's date ranges |

Verified on 2026-10-01: a fresh data root downloaded through the MCP tools reproduced the lab's MERRA-2 daily subset value-for-value, and PRISM grids value-for-value (PRISM files differ only in the `history` attribute stamped at download time). Compare data by values, never by file bytes.

# Path policy

- Everything lives under `PIPELINE_DATA_ROOT` (default `~/prithvi-wxc-data`): data, pinned code, training env, and `.mcp-state/` (jobs, logs, manifests, plots).
- Existing data elsewhere can be reused with `MERRA2_DATA_DIR`, `PRISM_DATA_DIR`, `NARR_DATA_DIR`, `ELEVATION_FILE`, `MODEL_WEIGHTS_FILE`.
- Do **not** hardcode another machine’s absolute paths in new configs. Route configs through `create_custom_yaml`, which localizes missing input paths.

# Pipeline order

1. `list_available_configs` then `read_yaml_config` on the closest match.
2. `create_custom_yaml` for a new case (never overwrite without confirmation).
3. `start_compute_scalars_job` if scalars are missing.
4. `start_preprocessing_job` (depends on scalars).
5. Training: `run_training_pipeline` only. Inference-only: `start_inference_job`.
6. Monitor with `get_job_status`, `list_jobs`, `get_gpu_status`.

# Guardrails

- Confirm `num_epochs` and `num_gpus` before training.
- Never delete outputs, logs, or manifests under `.mcp-state/` unless the user asks.
- If a date is outside file coverage, report actual coverage from NetCDF metadata.
- If scalars are recomputed, warn that downstream training/inference must be re-run.

# Related

- [Downscale playbook](/playbooks/downscale-wxc.md)
- [Analyze playbook](/playbooks/analyze-wxc.md)
- [MCP pipeline tools](/tools/mcp-pipeline-tools.md)

# Citations

[1] [OKF constraint example](https://okf.md/examples)
[2] [OKF spec](https://okf.md/spec)
