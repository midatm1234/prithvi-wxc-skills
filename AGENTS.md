# Agent instructions: Prithvi WxC downscaling

This repository runs Prithvi WxC weather and climate downscaling on the user's own GPU machine: MERRA-2 or NARR predictors at 32 km, plus 800 m terrain, become daily 800 m PRISM precipitation, Tmax, and Tmin. The work is done by the MCP server `prithvi-wxc-downscaling` (`bin/prithvi-mcp`). Claude Code users get this guidance as skills (`skills/`); this file gives the same guidance to other agents, such as GitHub Copilot, OpenAI Codex, and Cursor.

## If the Prithvi tools are missing

- If the only Prithvi tool is `prithvi_setup_status`, call it and tell the user what it says: the server is still installing its packages, or setup is blocked and it names the fix.
- If there are no Prithvi tools at all, the MCP server isn't registered in this app. Point the user to [docs/RUNBOOK.md](docs/RUNBOOK.md), section "Using VS Code".

## Workflow, in this order

1. **Set up the machine.** Call `check_environment` and follow its `next_steps` (usually `setup_code`, then `setup_training_env`). Details: [playbooks/setup-machine.md](playbooks/setup-machine.md).
2. **Download inputs.** Call `check_raw_data_status`, then `start_download_job` with `merra2`, `narr`, `prism`, `elevation`, or `weights`. Confirm any date range longer than a month with the user first; MERRA-2 transfers about 1.1 GB per day and the weights are 17.4 GB. Details: [skills/prithvi-data/SKILL.md](skills/prithvi-data/SKILL.md).
3. **Build a config.** Call `list_available_configs` and `read_yaml_config`, then always pass the chosen base config through `create_custom_yaml`: it points data paths at this machine. Then call `preflight_check`.
4. **Train, run inference, evaluate.** Confirm `num_epochs`, `num_gpus`, and the date ranges with the user, then call `run_training_pipeline`. It runs training preprocessing, training-only scalars, validation/inference preprocessing, fine-tuning, tiled inference, and evaluation against PRISM, skipping finished stages. For inference from an existing checkpoint, call `start_inference_job`; for evaluation alone, `start_evaluation_job`. Details: [playbooks/downscale-wxc.md](playbooks/downscale-wxc.md).
   - **NARR only:** pass `refinement_type` (`diffusion_unet`, `diffusion_transformer`, `flow_matching_unet`, `flow_matching_transformer`) to add stochastic residual refinement on the frozen model: an ensemble y_hat + r_hat with mean and spread, compared against the deterministic output. Needs the `stochastic_refinement` code variant and a base config with `dates.validation`. MERRA-2 has no refinement. Details: [skills/prithvi-refine/SKILL.md](skills/prithvi-refine/SKILL.md), [playbooks/refine-narr.md](playbooks/refine-narr.md).
5. **Monitor.** Use `get_job_status`, `list_jobs`, and `get_gpu_status`. Report job ids and log paths. Call `cancel_job` only when the user asks.
6. **Analyze.** Use `load_by_date`, `plot_variable_2d`, `compute_statistics`, and the climatology tools. Details: [playbooks/analyze-wxc.md](playbooks/analyze-wxc.md).
7. **Reproduce or share.** Use `get_run_manifest`, `list_run_manifests`, and `replay_run`. Details: [playbooks/reproduce-run.md](playbooks/reproduce-run.md).

## Rules

- Never ask the user to paste a password into chat. NASA Earthdata credentials belong in `~/.config/prithvi-wxc/env` or `~/.netrc`.
- Don't train or download by running the scripts in a terminal. Use the MCP tools, so every run is tracked and gets a manifest.
- Don't install `xesmf` into the training environment; it changes preprocessing compared with the reference runs.
- Don't start the HTTP server `mcp/mcp_server.py` on a network; it has no login.
- Training and inference need an NVIDIA GPU on Linux. On other machines, only downloads and analysis work.

## Editing this repository

- MCP server code is in `mcp/`; tool definitions are in `mcp/mcp-merra2-config.json`; pinned versions are in `mcp/pins.json` (change pins deliberately and note it in `log.md`).
- Tests: `python tests/smoke_stdio.py`, `python tests/first_start.py`, `python tests/e2e_fresh_machine.py --root <empty dir>`, and on a GPU machine `tests/e2e_gpu_smoke.py`.
- Releases: see [docs/RUNBOOK.md](docs/RUNBOOK.md), "Releasing a new version".
