# Bundle Update Log

## 2026-10-02

* **Create**: Apache-2.0 `LICENSE` (matching the training code repo); plugin author set to `midatm1234`.
* **Update**: The reference 800 m orography (`prism_elevation.nc`, Copernicus DEM GLO-30 on the PRISM grid) is published on Zenodo (DOI [10.5281/zenodo.23096854](https://doi.org/10.5281/zenodo.23096854)). `start_download_job dataset=elevation` fetches it and verifies the pinned sha256; a failed download or checksum mismatch fails the job instead of substituting other terrain. Verified on a fresh data root: identical to the lab file, run manifest reproducible.

## 2026-10-01 (v2.0.0)

* **Fix**: `mcp_stdio.py` used LSP `Content-Length` framing, so spec-compliant MCP hosts (Claude Code, Cursor) never got a reply. It now speaks newline-delimited JSON-RPC (legacy framing still accepted), negotiates the protocol version, and keeps tool output off stdout.
* **Fix**: PRISM downloads executed the notebook, which crashed on Jupyter-only `display()` and reported success on failures. Added `mcp/downloaders/download_prism.py` (same code, CLI, non-zero exit on failure).
* **Fix**: Removed host-specific defaults (`/data2/aashishp/...`, `/home/azureuser/...`, `/data/...` lab-script preference). Everything resolves from `PIPELINE_DATA_ROOT`; job state, logs, and plots moved out of the plugin into `.mcp-state/`.
* **Fix**: The pinned `stochastic_refinement` commit tracks symlinks (`examples/*_PRISM/{experiments,preprocessed,scalars_with_H}`) to `/data/granite-wxc/...`, which dangle on other machines. `setup_code` replaces out-of-repo symlinks with local directories marked `skip-worktree`, so the checkout stays clean at the pinned commit.
* **Create**: `mcp/pins.json` — training code commits, Prithvi WxC backbone commit, HF weights revision + sha256, orography sha256, data sources. `env/training-requirements.lock.txt` — training packages (verified to resolve from public indexes).
* **Create**: tools `check_environment`, `setup_code`, `setup_training_env`, `preflight_check`, `get_run_manifest`, `list_run_manifests`, `replay_run`; `start_download_job` gains `weights`, `code`, `env`; `create_custom_yaml` localizes input paths; stage tools refuse to queue when inputs are missing.
* **Create**: run manifests for every job (command, config text + sha256, code commit, pins, package versions, inputs, reproducibility verdict).
* **Create**: Claude Code plugin + marketplace (`.claude-plugin/`, `.mcp.json`, `skills/`), `bin/prithvi-mcp` launcher, Cursor manifest, playbooks [setup-machine](./playbooks/setup-machine.md) and [reproduce-run](./playbooks/reproduce-run.md), tests.
* **Verify**: fresh-root downloads through MCP matched lab MERRA-2 and PRISM data value-for-value. On that root, `setup_training_env` built the pinned env and `tests/e2e_gpu_smoke.py` ran scalars → preprocessing → training → inference on an A100; all manifests reproducible.
* **Fix**: `run_training_pipeline` looked for scalars only under `scalar_dir`; current code writes `<preprocessed_dir>/<case_name>/scalars`, so reruns recomputed scalars (invalidating downstream stages).
* **Create**: overlapping same-dataset downloads are refused (two jobs on the same day clobber each other's files).

## 2026-10-01

* **Update**: Added executable MCP runtime under [`mcp/`](./mcp/) so users can download, preprocess, train, and infer — not only read playbooks.
* **Update**: Published as a standalone OKF bundle (removed Cursor `SKILL.md` packaging). Entry point is root [index.md](./index.md).
* **Create**: Initial OKF v0.1 bundle from downscale / analyze workflows.
* **Create**: Added [downscaling overview](./concepts/downscaling-overview.md) and [YAML data model](./concepts/dataset-yaml-model.md).
* **Create**: Added playbooks [downscale-wxc](./playbooks/downscale-wxc.md) and [analyze-wxc](./playbooks/analyze-wxc.md).
* **Create**: Documented [MCP pipeline tools](./tools/mcp-pipeline-tools.md) and [MCP analysis tools](./tools/mcp-analysis-tools.md).
* **Create**: Captured [reproducibility constraints](./constraints/reproducibility.md).
