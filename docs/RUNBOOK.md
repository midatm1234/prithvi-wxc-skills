# Prithvi WxC Downscaling Runbook

How to install this plugin on a GPU machine and use Claude to download data, preprocess, train, and run inference with Prithvi WxC. Predictors at 32 km (MERRA-2 or NARR) plus 800 m terrain go in; daily 800 m PRISM precipitation, Tmax, and Tmin come out.

Nothing is hosted. Each user runs their own MCP server on their own machine, so their GPUs do the work and their data stays on their disk.

```
Claude Code (chat + skills)  →  MCP server (bin/prithvi-mcp)  →  background jobs (local GPUs and disk)
└──────────────────────────── all on the user's GPU machine ────────────────────────────┘
```

## What's in this repo

| Component | What it does | Where | Works with |
|---|---|---|---|
| Plugin + marketplace | Installs the MCP server and the skills in one step | `.claude-plugin/`, `.mcp.json` | Claude Code |
| MCP server | 40 tools: setup, downloads, configs, background jobs, analysis, run provenance | `bin/prithvi-mcp`, `mcp/` | Any MCP client (stdio) |
| Skills | Tell Claude which tools to call and in what order: setup, data, downscale, analyze, reproduce | `skills/` | Claude Code |
| Knowledge bundle | The same workflows as plain Markdown playbooks | `index.md`, `playbooks/` | Any agent |
| Pins and lock file | Exact code commits, weights revision, data versions, Python packages | [`mcp/pins.json`](../mcp/pins.json), [`env/`](../env/) | Used automatically |
| Reference terrain | 800 m orography on the PRISM grid, checksum-verified on download | Zenodo [10.5281/zenodo.23096854](https://doi.org/10.5281/zenodo.23096854) | Downloaded automatically |

## What the GPU machine needs

| Item | Requirement |
|---|---|
| Operating system | Linux x86_64 |
| GPU | NVIDIA GPU with a driver that supports CUDA 13: `nvidia-smi` shows "CUDA Version" 13.0 or higher. Tested on A100 80 GB. |
| Python | 3.11 or newer for the MCP server. The training environment uses Python 3.14, which [uv](https://docs.astral.sh/uv/) downloads automatically, so install uv unless `python3.14` is already on the PATH. |
| Other tools | `git`, `curl` |
| Disk | 17.4 GB for the pretrained weights, about 25 GB per year of PRISM (3 variables), about 0.5 GB per year of MERRA-2, plus the training environment, checkpoints, and outputs |
| Accounts | A free [NASA Earthdata](https://urs.earthdata.nasa.gov/) login, for MERRA-2 only. NARR, PRISM, terrain, and weights need no account. |
| Network | HTTPS to GitHub, PyPI, the PyTorch package index, Hugging Face, Zenodo, NASA GES DISC, PRISM (Oregon State), and NOAA PSL |

The plugin also installs on macOS for downloads and analysis; training and inference need the Linux GPU machine.

## Install (once per machine)

1. **Install Claude Code and uv.**

   ```bash
   curl -fsSL https://claude.ai/install.sh | bash
   ```

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Add the marketplace and install the plugin.**

   ```bash
   claude plugin marketplace add midatm1234/prithvi-wxc-skills
   ```

   ```bash
   claude plugin install prithvi-wxc-downscaling@prithvi-wxc
   ```

3. **Create the settings file.** The server reads `~/.config/prithvi-wxc/env` every time it starts. Start from the template:

   ```bash
   mkdir -p ~/.config/prithvi-wxc
   curl -fsSL https://raw.githubusercontent.com/midatm1234/prithvi-wxc-skills/main/env.example -o ~/.config/prithvi-wxc/env
   chmod 600 ~/.config/prithvi-wxc/env
   ```

   Then edit it. The lines that matter:

   ```bash
   PIPELINE_DATA_ROOT=/big/disk/prithvi-wxc-data
   EARTHDATA_USERNAME=your_earthdata_username
   EARTHDATA_PASSWORD=your_earthdata_password
   ```

   Instead of the password lines, a `~/.netrc` entry for `urs.earthdata.nasa.gov` also works. In the Earthdata profile, authorize the "NASA GESDISC DATA ARCHIVE" application, or MERRA-2 downloads are refused.

4. **Start Claude Code and check the connection.** Run `claude` and type `/mcp`. The server `prithvi-wxc-downscaling` shows as connected right away. On the first start it installs its Python packages in the background (about 25 seconds on a fast connection) and its tools appear when that finishes. Until then, Claude answers Prithvi requests with the install progress.

**Updating later:**

```bash
claude plugin marketplace update prithvi-wxc
```

```bash
claude plugin update prithvi-wxc-downscaling@prithvi-wxc
```

Then restart Claude Code.

## Run the pipeline

Users talk to Claude in plain language; the skills turn each request into tool calls. Long steps run as background jobs that pick idle GPUs (under 20% busy and at least 20% of memory free), wait in a queue when all GPUs are busy, and keep running after Claude exits. Times below were measured on one A100 with a fast datacenter connection.

Skills can also be started directly, for example `/prithvi-wxc-downscaling:prithvi-setup`.

### 1. Set up the machine

> "Set up this machine for Prithvi WxC downscaling."

- **Tools:** `check_environment` → `setup_code` → `setup_training_env`
- **Result:** pinned training code under `code/`; training environment (PyTorch 2.12 for CUDA 13, Prithvi WxC) under `envs/prithvi/`. `check_environment` lists anything still missing, in order.
- **Time:** code clone in seconds; environment build several minutes.

### 2. Download inputs

> "Download MERRA-2, PRISM, terrain, and weights for 2015-01-01 to 2015-01-06."

- **Tools:** `start_download_job` with `merra2`, `narr`, `prism`, `elevation`, `weights`, or `all`; `check_raw_data_status`
- **Time:** MERRA-2 30 to 70 s per day (1.14 GB transferred, 1.4 MB kept). PRISM about 10 s per day for 3 variables. Terrain in seconds. Weights 17.4 GB, limited by bandwidth.
- **Scale:** the reference setup trains on 1996–2015 and evaluates 2016–2025, so full runs mean hundreds of GB transferred. Start with a few days.

### 3. Build a config

> "Make a config from MERRA_PRISM_smoke.yaml called smoke_test."

- **Tools:** `list_available_configs`, `read_yaml_config`, `create_custom_yaml`, `preflight_check`
- **Result:** a new YAML whose data paths point at this machine's data, plus a check that every input the run needs exists.

### 4. Train and run inference

> "Run the training pipeline on smoke_test with 1 GPU."

- **Tools:** `run_training_pipeline` queues scalars → preprocessing → training → inference, skipping finished stages. `start_inference_job` runs inference alone from a checkpoint.
- **Time:** the smoke config (6 training days, 1 inference day) takes about 6 minutes: scalars 0.2 min, preprocessing 0.6 min, training 1.9 min, inference 2.9 min.

### 5. Monitor jobs

> "How are my jobs doing?"

- **Tools:** `get_job_status`, `list_jobs`, `get_gpu_status`, `cancel_job`
- **Logs:** `.mcp-state/logs/` under the data root. Jobs survive Claude exiting; reconnect and ask again.

### 6. Analyze outputs

> "Plot Tmax on 2016-01-03 over LA County."

- **Tools:** `load_by_date`, `plot_variable_2d`, `compute_statistics`, `compute_monthly_climatology`, `compute_seasonal_climatology`, and more
- **Result:** numbers in chat; plots saved as PNG files under `.mcp-state/artifacts/`.

### 7. Reproduce or share a run

> "Show the manifest for the training job." / "Replay this manifest."

- **Tools:** `get_run_manifest`, `list_run_manifests`, `replay_run`
- **Result:** each job records its command, config, code commit, package versions, and inputs, and says whether it's reproducible. Another machine can replay the manifest file.

## Where files go

Almost everything goes under the data root, `PIPELINE_DATA_ROOT` (default `~/prithvi-wxc-data`):

```
~/prithvi-wxc-data/
├── merra2/
│   ├── daily_subset_with_H/          MERRA-2: one M2I3NPASM_subset_YYYYMMDD.nc4 per day
│   └── download_tmp/                 raw ~1.1 GB daily files, deleted after subsetting
├── narr/subset/                      NARR (raw files in narr/raw/ are removed after subsetting)
├── prism/prism_daily_800m_an/
│   └── {ppt,tmax,tmin}/YYYY/         PRISM: prism_<var>_us_30s_YYYYMMDD.nc
├── static/prism_elevation.nc         800 m terrain (from Zenodo)
├── weights/ibm-granite/granite-geospatial-wxc-downscaling/ECCC/weights/best_rmse_UNET.pt
├── cordex/                           CORDEX-ML-Bench, if downloaded
├── code/Prithvi-UNet-stocahstic/     setup_code: git clone of the pinned training code
├── envs/prithvi/                     setup_training_env: training Python environment
└── .mcp-state/
    ├── logs/                         one log per job (merra/, narr/, misc/ for downloads)
    ├── runs/                         run manifest per job (<job_id>.json)
    ├── artifacts/                    plots from the analysis tools
    └── jobs/                         job tracking
```

Training and inference outputs are written inside the code checkout, in a folder named after the config's `case_name`. For MERRA-2 runs that's under `code/Prithvi-UNet-stocahstic/examples/MERRA_PRISM/` (NARR runs use `examples/NARR_PRISM/`):

| Output | Path |
|---|---|
| Configs you create | `custom_<name>.yaml` |
| Scalars and preprocessed data | `preprocessed/<case_name>/` (`scalars/`, `training/`, `inference/`) |
| Checkpoints | `experiments/checkpoints/<case_name>/` (`best.ckpt`, `last.ckpt`, `epoch_NNN.ckpt`) |
| Downscaled results | `experiments/inference_output/<case_name>/<case_name>_inference_YYYYMMDD.nc` (ppt, tmax, tmin) |

Those locations come from the config (`preprocessed_dir`, `checkpoint_dir`, `inference.output_dir`); relative paths resolve against the code folder, so you can redirect them in the YAML.

Outside the data root:

- **The server's own Python packages:** `~/.cache/prithvi-wxc-mcp/venv`, with its `install.log`
- **The plugin itself:** `~/.claude/plugins/cache/prithvi-wxc/prithvi-wxc-downscaling/<version>/`

To put everything on another disk, set `PIPELINE_DATA_ROOT` in `~/.config/prithvi-wxc/env`. To reuse data that already exists elsewhere, set `MERRA2_DATA_DIR`, `NARR_DATA_DIR`, `PRISM_DATA_DIR`, `ELEVATION_FILE`, or `MODEL_WEIGHTS_FILE`.

## Using VS Code

Run VS Code on the GPU machine, or connect to it with the Remote - SSH extension, so the server starts where the GPUs are. Create the settings file first (Install, step 3). Then set up whichever agent you use in VS Code:

| Agent | Setup | What it gets |
|---|---|---|
| Claude Code extension | Install the extension, then run the two `claude plugin` commands from the Install section in VS Code's terminal | Tools and skills |
| GitHub Copilot (agent mode) | Clone this repo on the GPU machine and open the folder. VS Code finds [`.vscode/mcp.json`](../.vscode/mcp.json) and offers to start the server. To use the tools from other folders, run **MCP: Add Server…** from the Command Palette, choose a stdio command, and enter the path to `bin/prithvi-mcp`. | Tools, plus the workflow in [`AGENTS.md`](../AGENTS.md) |
| OpenAI Codex extension | Clone this repo on the GPU machine, then register the server with the command below | Tools, plus the workflow in [`AGENTS.md`](../AGENTS.md) |

```bash
git clone https://github.com/midatm1234/prithvi-wxc-skills.git ~/prithvi-wxc-skills
```

```bash
codex mcp add prithvi-wxc-downscaling -- ~/prithvi-wxc-skills/bin/prithvi-mcp
```

That command writes this to `~/.codex/config.toml`, which you can also add by hand:

```toml
[mcp_servers.prithvi-wxc-downscaling]
command = "/home/you/prithvi-wxc-skills/bin/prithvi-mcp"
```

Copilot and Codex don't load the Claude skills. They read [`AGENTS.md`](../AGENTS.md) when this repo is the open folder, and every MCP client receives the server's own short instructions. In Copilot, switch the chat to Agent mode so it can call tools.

## Other MCP clients

Any other MCP client that runs local servers works the same way: register `bin/prithvi-mcp` as a stdio server on the GPU machine. Most clients use this JSON shape:

```json
{
  "mcpServers": {
    "prithvi-wxc-downscaling": {
      "command": "/home/you/prithvi-wxc-skills/bin/prithvi-mcp"
    }
  }
}
```

- These clients get the tools but not the skills. Point the agent at [`AGENTS.md`](../AGENTS.md) or [`index.md`](../index.md) for the workflows.
- Cursor: the repo includes a Cursor plugin manifest (`.cursor-plugin/`, `mcp.json`). It hasn't been tested in Cursor yet.
- To check a machine without any client, run `bin/prithvi-mcp --check` from the clone. It installs the server's packages if needed and prints what the machine still needs.
- Don't run the HTTP server `mcp/mcp_server.py` on a network: it has no login and listens on all network interfaces.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| VS Code: the agent has no Prithvi tools | The MCP server isn't registered for that agent, or Copilot isn't in Agent mode | Follow "Using VS Code" for your agent; in Copilot, switch the chat to Agent mode |
| Claude says the Prithvi tools aren't available yet | The first-start install is still running, or setup is blocked: no Python 3.11+, a failed install, or an error in the settings file | Ask Claude to check the setup status; it reports the cause and the fix. Install log: `~/.cache/prithvi-wxc-mcp/install.log` |
| `/mcp` shows the server failed (plugin 2.0.0 only) | Version 2.0.0 installed its packages in a folder Claude Code never reused, so every start timed out | Update to 2.0.1 or later with the two update commands, then restart Claude Code |
| MERRA-2 download fails with a login or 401/403 error | Missing Earthdata credentials, or GES DISC not authorized | Fill in the settings file or `~/.netrc`, authorize "NASA GESDISC DATA ARCHIVE" in the Earthdata profile, then restart Claude Code |
| `setup_training_env` says it needs Python 3.14 | Neither uv nor `python3.14` is installed | Install uv and rerun the setup |
| `check_environment` says PyTorch can't see a GPU | NVIDIA driver older than CUDA 13 | Update the driver. A CUDA 12 build isn't supported yet. |
| A download is refused for overlapping dates | Another job is downloading the same days; two jobs would overwrite each other's files | Wait for it, or ask Claude to chain the new job after it |
| Training is refused at preflight | Inputs the config needs are missing | Rebuild the config with `create_custom_yaml`, then download what preflight lists |
| A manifest says `reproducible: false` | Code not at the pinned commit, edited code, different terrain file, or xesmf installed | Fix the reasons the manifest lists |
| Weights download stops for disk space | Less than about 18 GB free | Free space, or set `PIPELINE_DATA_ROOT` to a larger disk |

## Not included yet

- Phase 2 stochastic refinement (diffusion and flow-matching ensembles). The code exists in the training repo but isn't exposed as tools.
- CORDEX-ML-Bench training and evaluation. The data downloads, but nothing trains on it.
- GPUs whose drivers only support CUDA 12.
- Clusters that schedule jobs through SLURM or PBS.
- Training on macOS or Windows.

## Releasing a new version (maintainers)

1. Record changes in [`log.md`](../log.md). Change pins in [`mcp/pins.json`](../mcp/pins.json) deliberately, with a note.
2. Bump `version` in `.claude-plugin/plugin.json`, `.cursor-plugin/plugin.json`, and `plugin.json`. Users only receive an update when the version changes.
3. Run the tests: `tests/smoke_stdio.py`, `tests/first_start.py`, `tests/e2e_fresh_machine.py`, and on a GPU machine `tests/e2e_gpu_smoke.py`.
4. Validate the manifests: `claude plugin validate --strict .`
5. Commit, push to `main`, and tag the release with `claude plugin tag`.

---

Times and sizes were measured on 2026-10-01 and 2026-10-02 on a Linux VM with A100 80 GB GPUs, using Claude Code 2.1.178. They depend on network speed and GPU.
