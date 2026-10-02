---
type: Playbook
title: Set up a machine
description: Prepare any Linux machine to reproduce Prithvi WxC downscaling through MCP — environment check, pinned code, pinned training env, credentials.
tags: [playbook, setup, reproducibility, mcp]
timestamp: 2026-10-01T09:00:00Z
---

Use on a new machine, or whenever a pipeline tool fails for environment reasons.

# Steps

1. `check_environment`. It returns `next_steps` in order, each naming the tool that fixes it, plus `warnings`.
2. `setup_code` (variant `deterministic`, or `stochastic_refinement` for residual refinement). Clones `midatm1234/Prithvi-UNet-stocahstic` and checks out the pinned commit under `PIPELINE_DATA_ROOT/code`. It refuses to touch a checkout with local edits.
3. `setup_training_env` (`depends_on` = the setup_code job). Creates `PIPELINE_DATA_ROOT/envs/prithvi` from `env/training-requirements.lock.txt` and installs the pinned code; jobs use it automatically.
4. Credentials (MERRA-2 only): `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` in `~/.config/prithvi-wxc/env`, or a `~/.netrc` entry for `urs.earthdata.nasa.gov`. Restart the MCP server after editing.
5. `check_environment` again; then download inputs (step 0b of the [downscale playbook](/playbooks/downscale-wxc.md)).

# Requirements

- Linux x86_64; Python ≥ 3.11 for the MCP server; `git`.
- NVIDIA GPU(s) with CUDA 13 drivers for training and inference (reference: A100 80 GB). Downloads and analysis run without a GPU.
- Disk: ~18 GB for weights, ~25 GB per year of PRISM (3 variables), ~0.5 GB per year of MERRA-2 subsets (plus ~1.2 GB transient per day while subsetting).

# Related

- [Reproducibility rules](/constraints/reproducibility.md)
- [MCP pipeline tools](/tools/mcp-pipeline-tools.md)
