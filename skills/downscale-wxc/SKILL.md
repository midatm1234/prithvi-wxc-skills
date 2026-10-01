---
name: downscale-wxc
description: Use when the user asks to downscale gridded weather/climate data (MERRA-2, NARR, ERA5, CORDEX, ACCESS-CM2, or any gridded NetCDF) with this repo's PrithviWxC / granite-wxc pipeline. Covers YAML config authoring, preprocessing, scalar computation, training, and inference through the MCP tool server. Trigger on "downscale", "super-resolve", "PrithviWxC", "granite-wxc", "NARR downscaling", "ERA5 downscaling", "MERRA-2 downscaling", "run inference", "train downscaler".
---

# Downscale gridded weather/climate data with PrithviWxC / granite-wxc

The downscaler is dataset-agnostic — any gridded NetCDF works as long as its paths and variable names are supplied via a YAML config. There is **no** per-source loader; NARR, ERA5, MERRA-2, CORDEX, ACCESS-CM2 etc. all flow through the same pipeline.

## Stack

- `agent_server.py` (port 8000) — chat + orchestration
- `mcp/mcp_server.py` (port 8001) — tool server
- Tool orchestrator: `mcp/mcp_merra2_tools.py` (name is legacy; tools are general)
- Reference configs: `/data2/aashishp/github_merra_test/granite-wxc/examples/CORDEX_ML/*.yaml`
- Artifacts: `mcp/artifacts/`

## How dataset selection actually works

The YAML `data:` block drives everything:

```yaml
data:
  type: cordex               # arbitrary label — CHANGE for NARR / ERA5 / MERRA-2
  training_predictor_paths: [/path/to/coarse_input.nc]     # e.g. ERA5 files
  training_target_paths:    [/path/to/highres_target.nc]   # e.g. NARR / station-obs
  validation_predictor_paths: [...]
  validation_target_paths:   [...]
  test_predictor_paths:      [...]
  test_target_paths:         [...]
  static_path: /path/to/Static_fields.nc
  input_vars:            [u, v, q, t, z]                   # predictor NetCDF var names
  input_levels:          [850.0, 700.0, 500.0]
  output_vars:           [pr, tasmax]                       # target NetCDF var names
  input_size_lat: 16                                        # coarse grid
  input_size_lon: 16
  target_size_lat: 128                                      # fine grid
  target_size_lon: 128
  downsample_factor: 8
  scalers:                                                  # written by compute_scalars
    inputs_mean:   /.../inputs_mean.npy
    inputs_std:    /.../inputs_std.npy
    targets_mean:  /.../targets_mean.npy
    targets_std:   /.../targets_std.npy
```

To switch data sources, point the `*_paths` at the new NetCDF files and update `input_vars` / `output_vars` to match those files' variable names. The model doesn't care whether they're NARR, ERA5, or MERRA-2.

## When invoked — clarify these before generating a config

1. **Predictor (coarse) source + paths** — NARR? ERA5? MERRA-2? Which NetCDF files?
2. **Target (fine) source + paths** — station obs? CORDEX? high-res reanalysis?
3. **Variable names** — predictor vars (`u`, `v`, `t`, `q`, `z`, `T2M`, …) and target vars (`pr`, `tasmax`, `TMP2m`, …). These must match what's *in* the files, not what the user calls them conceptually.
4. **Pressure levels** (if 3-D predictors) — e.g. `[850, 700, 500]`.
5. **Grid sizes** — coarse `input_size_lat/lon` and fine `target_size_lat/lon`; derive `downsample_factor`.
6. **Region / bbox** — for cropping.
7. **Date ranges** — separate for training and inference.
8. **Mode** — training run, inference only, or both.

If the user is vague on variable names or paths, call `list_files` / `github_repo_context` (query the granite-wxc repo) rather than guessing.

## Workflow

### 0. Verify services

```bash
curl -s http://localhost:8001/health || echo "MCP down"
curl -s http://localhost:8000/health || echo "Agent down"
```

If down, start each in its own background shell:
```bash
python mcp/mcp_server.py
python agent_server.py
```

### 0b. Make sure raw inputs exist (required on a new machine)

Data files are written **on the computer that runs MCP**, not into Cursor cloud.

- **User's laptop:** they install this plugin (or run `python mcp/mcp_stdio.py` locally). `start_download_job` then uses the bundled scripts in `mcp/downloaders/` and writes to `~/prithvi-wxc-data` (or `PIPELINE_DATA_ROOT`).
- **Remote hosted MCP:** files stay on the host. The server cannot write to the user's home directory.

Call `check_raw_data_status`. If inputs are missing, `start_download_job`:

- `merra2` → bundled `mcp/downloaders/download_merra2_aws.py` (or lab `/data/merra2/...` if present)
- `narr` → bundled `mcp/downloaders/download_narr.py` + `narr_daily_subset.yaml`
- `prism` → bundled `mcp/downloaders/download_prism_daily_800m.ipynb`
- `cordex` → [CORDEX-ML-Bench on Zenodo](https://zenodo.org/records/17517423)
- `elevation` → copy granite `prism_elevation.nc` (ETOPO fallback)
- `all` → merra2 + narr + prism + cordex + elevation

Pass `start_date` / `end_date` for merra2/narr/prism. For CORDEX, optional `domains` (`nz`, `alps`, `sa`, or `all`). Full multi-decade pulls are large — confirm the range first.

Then `get_job_status` until the download job is `done` before scalars/preproc.

### 1. Explore existing configs first

Use `list_available_configs`, then `read_yaml_config` on the closest match (usually `cordex_config*.yaml` or the NZ ESD variants). Prefer editing a known-good config over authoring from scratch.

### 2. Build the config

Call `create_custom_yaml` with a `base_config` and per-field overrides. Key knobs it exposes:

- `predictor_variables`, `target_variables` (map to `input_vars` / `output_vars`)
- `training_start`, `training_end`, `inference_start`, `inference_end`
- `lat_min/max`, `lon_min/max`
- `num_epochs`, `batch_size`, `learning_rate`, `num_gpus`
- `case_name` (drives output naming)
- `extra_overrides` — arbitrary deep YAML merge for anything the flat args don't cover (e.g. `data.training_predictor_paths`)

For a new data source, put the `*_paths` under `extra_overrides.data.*_paths` and the source label under `extra_overrides.data.type`.

### 3. Compute scalars (once per dataset combination)

`start_compute_scalars_job` with the new `config_path`. Writes `inputs_mean/std.npy` and `targets_mean/std.npy` to the paths referenced in `scalers:`. Skip if scalars already exist and the predictor/target files haven't changed.

### 4. Preprocess

`start_preprocessing_job` with `mode: "both"` (or `"train"` / `"inference"`). Depends on the scalar job — pass its job id via `depends_on`.

### 5. Train (optional) or infer

- Training: `run_training_pipeline` (config_path, num_gpus, save_every, queue_inference). Set `queue_inference=True` to auto-run inference after training.
- Inference only: `start_inference_job` (config_path, checkpoint, batch_size).

### 6. Monitor

`get_job_status <job_id>`, `list_jobs`, `get_gpu_status`. Cancel with `cancel_job`.

### 7. Retrieve artifacts

Outputs land in `mcp/artifacts/`. Report file paths back; don't open a browser unless asked.

## Common source-specific gotchas

- **NARR**: variable names differ from MERRA-2 (`TMP2m`, `APCP`, …); grid is Lambert Conformal — the pipeline expects regular lat/lon, so pre-regridding may be needed before pointing paths at it.
- **ERA5**: pressure levels use `level` not `lev`; single-level and pressure-level files are separate — list both under `training_predictor_paths` or concatenate.
- **MERRA-2**: variable names like `T2M`, `U10M`, `QV2M`, `PRECTOT`; already on regular lat/lon.
- **Mixed sources** (e.g. ERA5 predictor → NARR target): make sure the target grid actually covers the predictor bbox after crop, and that `downsample_factor` matches the true resolution ratio, not an assumed one.

## Guardrails

- **Never** overwrite an existing config without confirmation — save a new one via `create_custom_yaml`'s `output_name`.
- If scalars are recomputed, downstream inference/training must be re-run; warn the user before triggering this.
- Never delete files in `mcp/artifacts/` on the user's behalf.
- If the user asks for a date range outside the files' coverage, surface the actual coverage from the NetCDF metadata rather than silently clamping.
- Large training runs are expensive — confirm `num_epochs` and `num_gpus` before calling `run_training_pipeline`.

## Quick sanity check

Minimal end-to-end when unsure the stack is wired up: `check_raw_data_status` → `start_download_job` if inputs are missing → `list_available_configs` → `read_yaml_config` on one → `create_custom_yaml` with tiny date range and small bbox → `start_compute_scalars_job` → `start_preprocessing_job` → `start_inference_job`. Watch job statuses; confirm output appears in `mcp/artifacts/`.
