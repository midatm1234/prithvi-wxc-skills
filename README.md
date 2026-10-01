# PrithviWxC Downscaling — OKF + MCP

Portable package so **any** agent/tool host can:

1. Read the [OKF](https://okf.md/) knowledge (`index.md` → playbooks)
2. Run the **MCP tool server** to download data, preprocess, train, and run inference

Without the MCP server, playbooks are documentation only. This repo ships both.

## What you get

| Layer | Path | Role |
|-------|------|------|
| Knowledge (OKF) | `index.md`, `concepts/`, `playbooks/`, `tools/`, `constraints/` | What to do and in what order |
| Runtime (MCP) | `mcp/mcp_stdio.py`, `mcp/mcp_server.py`, tools, downloaders | Actually download / preprocess / train / infer |
| Wire-up | `mcp.json`, `mcp.example.json` | How to attach the server to Cursor / Claude / other MCP hosts |

## Quick start

```bash
git clone https://github.com/midatm1234/prithvi-wxc-skills.git
cd prithvi-wxc-skills

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.sample .env   # set Earthdata + optional GRANITE_WXC_* paths
```

### Run MCP (stdio — for Cursor, Claude Desktop, etc.)

```bash
PYTHONPATH=mcp python mcp/mcp_stdio.py
```

Or register via `mcp.example.json` (edit the absolute path to this clone).

### Run MCP (HTTP — optional)

```bash
PYTHONPATH=mcp python mcp/mcp_server.py   # http://0.0.0.0:8001
```

### Point an agent at the knowledge

Give the agent this repo’s `index.md` (OKF entry point). Playbooks tell it to call MCP tools such as:

- `check_raw_data_status` / `start_download_job` — get MERRA-2, NARR, PRISM, …
- `create_custom_yaml` — build training/inference configs
- `start_compute_scalars_job` / `start_preprocessing_job`
- `run_training_pipeline` / `start_inference_job`
- `load_by_date` / `plot_variable_2d` / climatology tools — analyze outputs

## Required on the user’s machine

| Need | Why |
|------|-----|
| Python deps (`requirements.txt`) | MCP + NetCDF / plots |
| `PIPELINE_DATA_ROOT` or `~/prithvi-wxc-data` | Where downloads and working data live |
| `GRANITE_WXC_REPO` | Checkout of [granite-wxc](https://github.com/) (training/inference scripts) |
| `GRANITE_WXC_PYTHON` (optional) | Conda/env Python used for GPU training jobs |
| Earthdata credentials | MERRA-2 downloads |

Data and training stay on **the machine that runs MCP**, not in this GitHub repo.

## Layout

```
.
├── index.md              # OKF entry — start here for agents
├── playbooks/            # downscale + analyze workflows
├── concepts/             # pipeline + YAML model
├── tools/                # tool catalog (docs)
├── constraints/          # reproducibility rules
├── mcp/                  # executable MCP server + downloaders
├── mcp.json              # Cursor / Agent Plugins style config
├── mcp.example.json      # generic host config (edit paths)
└── requirements.txt
```

## Note on granite-wxc

This package orchestrates jobs; the heavy training/inference code lives in your local `GRANITE_WXC_REPO`. Clone that separately and set the env var before training.
