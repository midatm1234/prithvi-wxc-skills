---
name: prithvi-setup
description: Prepare a machine to run Prithvi WxC downscaling — check GPUs, disk, credentials, clone the pinned code, and build the pinned training environment through the prithvi-wxc-downscaling MCP tools. Use when the user is on a new machine, asks to "set up", "install", or "get started" with Prithvi WxC / Prithvi-UNet downscaling, or when another prithvi skill hits a missing-environment error.
---

# Set up a machine for Prithvi WxC downscaling

Everything runs on the machine where the `prithvi-wxc-downscaling` MCP server runs; files go under `PIPELINE_DATA_ROOT` (default `~/prithvi-wxc-data`).

1. Call `check_environment`. Read `next_steps` (ordered, each names the tool that fixes it) and `warnings`. Summarize them for the user in a short list before doing anything.
2. If code is missing or not at the pin: `setup_code` (variant `deterministic`, or `stochastic_refinement` for the diffusion / flow-matching residual corrector). Poll `get_job_status` until `done`. If the result lists `localized_symlinks`, mention that those output directories were made local because the upstream commit links them to another server.
3. If the training Python lacks torch/granitewxc: `setup_training_env` with `depends_on` = the setup_code job id. It installs `env/training-requirements.lock.txt` (torch 2.12 + CUDA 13.0, PrithviWxC pinned). Takes several minutes.
4. Credentials: MERRA-2 needs NASA Earthdata Login. Ask the user to put `EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD` in `~/.config/prithvi-wxc/env` (template: `env.example` in the plugin) or a `~/.netrc` entry for `urs.earthdata.nasa.gov`, then restart the MCP server. Never ask the user to paste a password into chat. NARR, PRISM, elevation, and weights need no account.
5. Call `check_environment` again and report what remains. Data downloads are the `prithvi-data` skill.

## Rules

- Do not install xesmf into the training env: the reference results used the xarray-linear regridder, and xesmf silently changes preprocessing.
- If `check_environment` warns that the code checkout is off-pin or dirty, tell the user runs will be marked non-reproducible; do not reset their checkout yourself.
- No GPU (`gpus` empty) is fine for downloading and analysis; training and inference need NVIDIA GPUs (the reference used A100 80 GB).
- If the server will not start, the user can run `bin/prithvi-mcp --check` in a terminal from the plugin directory to see install errors.
