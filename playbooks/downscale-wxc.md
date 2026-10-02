---
type: Playbook
title: Downscale weather/climate grids
description: Run PrithviWxC / granite-wxc downscaling via MCP — config, scalars, preprocess, train, infer.
tags: [playbook, downscaling, training, inference, mcp]
timestamp: 2026-10-01T06:00:00Z
---

Use when the user asks to downscale gridded weather/climate data (MERRA-2, NARR, ERA5, CORDEX, ACCESS-CM2, or any gridded NetCDF), or mentions PrithviWxC, granite-wxc, super-resolve, train downscaler, or run inference.

# Prerequisites

1. MCP server running on the machine that will do the work (plugin `bin/prithvi-mcp`, or `mcp/mcp_stdio.py`).
2. Machine prepared with the [setup playbook](/playbooks/setup-machine.md): `check_environment` reports pinned code and a training environment.
3. Raw inputs present — see step 0b.

Optional health check when running the HTTP servers:

```bash
curl -s http://localhost:8001/health || echo "MCP down"
curl -s http://localhost:8000/health || echo "Agent down"
```

# Steps

## 0b. Ensure raw inputs exist

Data is written on the machine that runs MCP.

1. Call `check_raw_data_status`.
2. If missing, `start_download_job`:
   - `merra2` → `mcp/downloaders/download_merra2_aws.py` (Earthdata Login)
   - `narr` → `mcp/downloaders/download_narr.py`
   - `prism` → `mcp/downloaders/download_prism.py` (CLI form of the PRISM notebook)
   - `elevation` → reference 800 m orography from Zenodo (DOI 10.5281/zenodo.23096854), sha256-verified
   - `weights` → pinned Hugging Face weights (~17.4 GB, sha256-verified)
   - `cordex` → CORDEX-ML-Bench (Zenodo)
   - `all` → merra2 + prism + elevation + weights
3. Pass `start_date` / `end_date` for time-bounded sources. Confirm large ranges first.
4. Poll `get_job_status` until downloads are `done`.

## 1. Explore existing configs

`list_available_configs` → `read_yaml_config` on the closest match. Prefer editing a known-good base over authoring from scratch. See [YAML data model](/concepts/dataset-yaml-model.md).

## 2. Build a new config

Call `create_custom_yaml` with a `base_config` and overrides. Never overwrite an existing config without confirmation — use a new `output_name`. Always route a base config through this tool on a new machine: input paths that do not exist locally (`data.predictor_dir`, `data.target_dir`, `data.static_elevation_file`, `path_model_weights`) are rewritten to the `PIPELINE_DATA_ROOT` layout. Then `preflight_check` the result.

Useful knobs: `predictor_variables`, `target_variables`, date ranges, bbox, `num_epochs`, `batch_size`, `learning_rate`, `num_gpus`, `case_name`, `extra_overrides`.

## 3. Compute scalars

`start_compute_scalars_job` with the new `config_path`. Skip if scalars already exist and inputs are unchanged. If scalars are recomputed, warn that training/inference must be re-run.

## 4. Preprocess

`start_preprocessing_job` with `mode: "both"` (or `"train"` / `"inference"`). Pass scalar job id via `depends_on`.

## 5. Train or infer

- Training: `run_training_pipeline` (`config_path`, `num_gpus`, `save_every`, `queue_inference`).
- Inference only: `start_inference_job` (`config_path`, `checkpoint`, `batch_size`).

Confirm `num_epochs` and `num_gpus` before training.

## 6. Monitor

`get_job_status`, `list_jobs`, `get_gpu_status`. Cancel with `cancel_job`.

## 7. Retrieve artifacts

Training and inference outputs land where the config says (`path_experiment`, `inference.output_dir`, repo-relative paths resolve against the code checkout). Job logs, run manifests, and plots live under `<PIPELINE_DATA_ROOT>/.mcp-state/`. Report paths; do not delete outputs. Each job's provenance is available via `get_run_manifest` — see the [reproduce playbook](/playbooks/reproduce-run.md).

# Related

- [Downscaling overview](/concepts/downscaling-overview.md)
- [YAML data model](/concepts/dataset-yaml-model.md)
- [MCP pipeline tools](/tools/mcp-pipeline-tools.md)
- [Reproducibility rules](/constraints/reproducibility.md)
- Follow-on analysis: [Analyze playbook](/playbooks/analyze-wxc.md)

# Citations

[1] [OKF playbook pattern](https://okf.md/examples)
[2] [OKF spec](https://okf.md/spec)
