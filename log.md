# Bundle Update Log

## 2026-10-08

From an end-to-end NARR smoke test (separate `_smoke` case, one training month, 2 Phase-1 epochs, diffusion UNet refiner with 20 epochs and 4 members):

* **Fix**: `create_refinement_config` sorted the YAML keys while copying the base config, reordering `predictor_variables`; the Phase-1 cache then rejected every preprocessed product. Key order is now kept.
* **Fix**: `start_refinement_inference_job` never passed the refiner checkpoint (the generated config leaves `model.refinement.checkpoint` null), so ensemble inference failed at start. It now passes `--refinement-checkpoint` (the YAML value, else `<checkpoint_dir>/best.ckpt`).
* **Create**: `refiner_clip_sample_range` on `run_training_pipeline` / `create_refinement_config` (diffusion heads) enables x0 clipping as its own `_clip<range>` variant. Briefly trained refiners otherwise produce non-physical ensembles.
* **Update**: `stochastic_refinement` pin bumped from `7e2eb94` (branch `narr_prism`) to `904ba6f` (branch `aashish-dev-stochastic`), which adds three fixes on top of `7e2eb94`: Phase-1 training no longer tries to move collated tile dates to the device, and two evaluator fixes — Phase 1 is scored from the `*_phase1` baseline embedded in the refined products (bf16 deterministic inference never matched it to 2e-5, so `refinement_evaluation` always failed), and ensemble interval quantiles are vectorized (a week of evaluation took hours; now ~50 s, identical results).
* **Create**: `tests/e2e_narr_refine_smoke.py`, the NARR + refinement counterpart of `e2e_gpu_smoke.py`; smoke-test recipe and measured stage costs in the [refine-narr playbook](./playbooks/refine-narr.md) and `prithvi-refine` skill.

## 2026-10-06

* **Fix**: `run_training_pipeline` now follows the training repo's case-scoped order — training preprocessing → training-only scalars → validation/inference preprocessing → fine-tuning → tiled inference → evaluation. It previously computed scalars before any training products existed (which fails for a new case), never preprocessed validation dates, and skipped inference preprocessing whenever training data already existed. Scalars are looked up in `<preprocessed_dir>/<case_name>/scalars/`.
* **Create**: `start_evaluation_job` (`evaluate_prism_inference.py`: RMSE, correlation, bias, boundary errors vs PRISM); queued after inference by the pipeline (`evaluate`).
* **Create**: NARR-only stochastic residual refinement. `run_training_pipeline` with `refinement_type` (`diffusion_unet`, `diffusion_transformer`, `flow_matching_unet`, `flow_matching_transformer`; `refiner_attention: false` for a UNet without attention blocks) writes `custom_<base>_<type>.yaml` and queues Phase-1 residual cache → refiner training (Prithvi frozen, residual y_true - y_hat) → ensemble inference (y_hat + r_hat, mean, spread) → deterministic-vs-refined evaluation. `train_phase1: false` reuses an existing Phase-1 checkpoint. New tools `create_refinement_config`, `start_refinement_inference_job`, `start_refinement_evaluation_job`. MERRA-2 has no refinement.
* **Create**: [refine-narr playbook](./playbooks/refine-narr.md) and the `prithvi-refine` skill.
* **Fix**: shared scripts outside `examples/<DATASET>/` (e.g. `examples/evaluate_prism_inference.py`) got the wrong repo root and `PYTHONPATH`; the root is now the directory holding `granitewxc/`.
* **Create**: preprocessing and scalars jobs are refused when the case directory resolves outside the allowed roots (e.g. a case linked read-only from another checkout).
* **Update**: `stochastic_refinement` pin bumped from `c65e9f3` (branch `Prithvi-UNet-stochastic_refinement`) to `7e2eb94` (branch `narr_prism`), which adds `narr_prism_phase1_cache.py`, `evaluate_refinement.py`, and `narr_prism_refinement.py` `--num-gpus` / `--split` / `--resume-existing` that the refinement stages use.
* **Fix**: `setup_code` also localizes tracked symlinks that point inside the repo but dangle (`experiments -> artifacts/experiments` at the new pin, where `artifacts/` is an untracked per-machine link). Verified on a fresh clone: all six become local directories and the checkout stays clean.

## 2026-10-02

* **Create**: VS Code support. `.vscode/mcp.json` registers the server for GitHub Copilot when the repo is opened; `AGENTS.md` gives Copilot, Codex, and other agents without skills the workflow and rules; the runbook covers the Claude Code, Copilot, and Codex extensions (Codex registration verified with `codex mcp add`).
* **Create**: [docs/RUNBOOK.md](./docs/RUNBOOK.md), the step-by-step guide (requirements, install and update, each pipeline stage, file locations, troubleshooting, release steps), linked from the README.
* **Fix (v2.0.1)**: First start no longer fails in MCP hosts. If the server's packages are missing, `bin/prithvi-mcp` installs them in the background while a stand-in server (`mcp/setup_status_server.py`, standard library only) answers at once with a `prithvi_setup_status` tool, then hands over to the full server in the same session (`notifications/tools/list_changed`). Missing Python 3.11+, failed installs, and broken settings files are reported through that tool. The launcher also looks for Python in Homebrew/python.org/pyenv locations and can use uv to fetch one. The server's packages now always live in `~/.cache/prithvi-wxc-mcp` (previously Claude Code's plugin data folder, so a terminal `--check` install was invisible to Claude Code and every start reinstalled). Test: `tests/first_start.py`.
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
