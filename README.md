# Prithvi WxC Downscaling — agent plugin, skills, and MCP server

Reproduce NASA/IBM **Prithvi WxC** downscaling (NARR or MERRA-2 at 32 km → PRISM 800 m daily precipitation, Tmax, Tmin) on your own machine, driven by an AI agent:

- **MCP server** (`bin/prithvi-mcp`): 40 tools that download data, fetch pinned code and weights, build the training env, configure, train, run inference, analyze, and replay runs.
- **Skills** (`skills/`): `prithvi-setup`, `prithvi-data`, `prithvi-downscale`, `prithvi-analyze`, `prithvi-reproduce` — tell the agent which tools to call, in which order.
- **Knowledge** ([OKF](https://okf.md/) bundle, `index.md` → `playbooks/`, `concepts/`, `constraints/`): the same workflows for any agent host.

Everything external is pinned in [`mcp/pins.json`](mcp/pins.json): training code commit, Prithvi WxC backbone commit, Hugging Face weights revision + sha256, the 800 m orography ([Zenodo, DOI 10.5281/zenodo.23096854](https://doi.org/10.5281/zenodo.23096854)) + sha256, and data sources. Python packages for training are pinned in [`env/training-requirements.lock.txt`](env/training-requirements.lock.txt). Every job writes a **run manifest** another machine can replay.

**Step-by-step guide:** [docs/RUNBOOK.md](docs/RUNBOOK.md) covers machine requirements, install, each pipeline stage with example prompts and measured times, where every file goes, and troubleshooting.

## Install

Requirements: Linux x86_64, Python ≥ 3.11, `git`. Training and inference need NVIDIA GPUs (reference: A100 80 GB, CUDA 13); downloads and analysis do not.

### Claude Code

```bash
claude plugin marketplace add midatm1234/prithvi-wxc-skills
```

```bash
claude plugin install prithvi-wxc-downscaling@prithvi-wxc
```

Then start Claude Code and ask, for example: *"Set up this machine for Prithvi WxC downscaling"*.

### VS Code

Use the Claude Code extension (same plugin commands as above), GitHub Copilot in agent mode (opening this repo picks up [`.vscode/mcp.json`](.vscode/mcp.json)), or the OpenAI Codex extension. Setup for each: [docs/RUNBOOK.md#using-vs-code](docs/RUNBOOK.md#using-vs-code). Agents without skills follow [`AGENTS.md`](AGENTS.md).

### Claude Desktop, Cursor, and other MCP hosts

Clone the repo and point the host at the launcher (see [`mcp.example.json`](mcp.example.json)):

```json
{ "mcpServers": { "prithvi-wxc-downscaling": { "command": "/ABS/PATH/TO/prithvi-wxc-skills/bin/prithvi-mcp" } } }
```

Cursor can load the repo as a plugin (`.cursor-plugin/`, `mcp.json`). Agents without skill support should be given [`index.md`](index.md).

### Settings and credentials

```bash
mkdir -p ~/.config/prithvi-wxc && cp env.example ~/.config/prithvi-wxc/env && chmod 600 ~/.config/prithvi-wxc/env
```

Edit it: `PIPELINE_DATA_ROOT` (default `~/prithvi-wxc-data`) and, for MERRA-2 only, a free [NASA Earthdata](https://urs.earthdata.nasa.gov/) login (or a `~/.netrc` entry for `urs.earthdata.nasa.gov`). NARR, PRISM, orography, and weights need no account.

On first start the server connects right away and installs its Python packages in the background (into `~/.cache/prithvi-wxc-mcp`, usually 1–3 minutes); its tools appear when the install finishes. If anything blocks setup (no Python 3.11+, a failed install, an error in the settings file), Claude can explain it through the `prithvi_setup_status` tool. To install ahead of time and see what the machine still needs, run the launcher with `--check` (for a plugin install it lives under `~/.claude/plugins/cache/prithvi-wxc/prithvi-wxc-downscaling/<version>/bin/`):

```bash
bin/prithvi-mcp --check
```

## What the agent does

```
check_environment ─► setup_code ─► setup_training_env            (prithvi-setup)
start_download_job: merra2 | narr | prism | elevation | weights    (prithvi-data)
create_custom_yaml (localizes paths) ─► preflight_check ─►
run_training_pipeline: scalars ─► preprocess ─► train ─► infer      (prithvi-downscale)
load_by_date / plot / climatology tools                            (prithvi-analyze)
get_run_manifest / replay_run                                      (prithvi-reproduce)
```

Long steps run as background jobs that survive server restarts and wait for free GPUs; poll with `get_job_status`.

## Data layout

```
$PIPELINE_DATA_ROOT/
  merra2/daily_subset_with_H/      narr/subset/      prism/prism_daily_800m_an/{ppt,tmax,tmin}/
  static/prism_elevation.nc        weights/ibm-granite/granite-geospatial-wxc-downscaling/...
  code/Prithvi-UNet-stocahstic/    envs/prithvi/
  .mcp-state/{jobs,logs,runs,artifacts}/     # job state, logs, run manifests, plots
```

Already have the data on a shared volume? Set `MERRA2_DATA_DIR`, `PRISM_DATA_DIR`, `NARR_DATA_DIR`, `ELEVATION_FILE`, or `MODEL_WEIGHTS_FILE` in the env file instead of re-downloading.

| Input | Size |
|-------|------|
| Weights | 17.4 GB (once) |
| PRISM ppt+tmax+tmin | ~25 GB per year; ~40 min per variable-year (PRISM rate limit) |
| MERRA-2 subset | ~0.5 GB per year kept; ~1.14 GB per day transferred, then deleted |

## Reproducibility

A run is reproducible when its manifest says `"reproducible": true`: the code is at a pinned commit with no local edits, the orography and weights match their pinned checksums, and the training env has no `xesmf` (the reference runs used the xarray-linear regridder; xESMF changes preprocessing).

Verified 2026-10-01 on a fresh, empty data root: MERRA-2 and PRISM files downloaded through these tools matched the reference lab data value-for-value. On the same root, the pinned training env built from the lock (torch 2.12.0+cu130, CUDA available) and the full smoke pipeline ran through MCP tool calls on an A100, with every run manifest marked reproducible. PRISM files differ only in a `history` attribute stamped at download time, so compare data by values, not bytes.

## Tests

```bash
python tests/smoke_stdio.py
```

```bash
python tests/e2e_fresh_machine.py --root /tmp/prithvi-e2e
```

```bash
python tests/e2e_gpu_smoke.py --root ~/prithvi-wxc-data
```

The smoke test runs the MCP handshake and read-only tools against an empty data root. The end-to-end test clones the pinned code and downloads 2 days of PRISM and 1 day of MERRA-2 into `--root`. The GPU smoke test (after `setup_code`, `setup_training_env`, and the weights) downloads 7 days and runs the repo's `MERRA_PRISM_smoke.yaml` through compute_scalars → preprocessing → 6 training steps → inference; it is the quickest way to confirm a machine can train before a multi-year run.

## Layout

```
.claude-plugin/        Claude Code plugin + marketplace manifests
.cursor-plugin/, mcp.json, plugin.json   Cursor / agent-plugins manifests
.mcp.json              MCP server registration for Claude Code
bin/prithvi-mcp        launcher (venv bootstrap + env file + stdio server)
skills/                Claude Code skills
mcp/                   MCP server, job manager, provenance, downloaders, pins.json
env/                   pinned training environment
index.md, playbooks/, concepts/, tools/, constraints/   OKF knowledge bundle
tests/                 smoke + end-to-end tests
```

`mcp/mcp_server.py` serves the same tools over HTTP/SSE on port 8001 for shared-server deployments (`pip install -r requirements-http.txt`).
