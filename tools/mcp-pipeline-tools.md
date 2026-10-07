---
type: Reference
title: MCP pipeline tools
description: MCP tools for downloads, YAML configs, preprocessing, scalars, training, inference, evaluation, NARR refinement, and jobs.
tags: [mcp, tools, pipeline, training, inference]
timestamp: 2026-10-01T06:00:00Z
resource: repo://mcp/mcp-merra2-config.json
---

Tools used by the [Downscale playbook](/playbooks/downscale-wxc.md). Exact schemas live in the plugin MCP tool manifest.

# Catalog

| Tool | Purpose |
|------|---------|
| `check_environment` | First call on a machine: GPUs, disk, code vs pin, training env, credentials, data; ordered next steps |
| `setup_code` | Clone + check out the pinned training/inference code |
| `setup_training_env` | Build the pinned training virtualenv |
| `check_raw_data_status` | Inventory local inputs |
| `start_download_job` | Download MERRA-2 / NARR / PRISM / elevation / weights / CORDEX |
| `preflight_check` | Verify a config's inputs exist for a stage |
| `list_available_configs` | List YAML configs under granite-wxc examples |
| `read_yaml_config` | Read a config |
| `create_custom_yaml` | Write a new config (do not overwrite blindly) |
| `list_files` / `find_file_by_date` / `github_repo_context` | Discover paths and repo context |
| `start_preprocessing_job` | Align / write preprocessed NetCDFs (`mode`: training, validation, inference) |
| `start_compute_scalars_job` | Training-only mean/std scalars (after training preprocessing) |
| `run_training_pipeline` | Preproc(train)→scalars→preproc(val/infer)→fine-tune→tiled inference→evaluation; NARR `refinement_type` adds Phase-2 refinement |
| `start_inference_job` | Inference-only job |
| `start_evaluation_job` | Evaluate inference vs PRISM on the exact grid (`evaluate_prism_inference.py`) |
| `create_refinement_config` | NARR: write `custom_<base>_<type>.yaml` (base config + refinement sections) |
| `start_refinement_inference_job` | NARR: ensemble inference (y_hat + r_hat, mean, spread) with a trained refiner |
| `start_refinement_evaluation_job` | NARR: deterministic vs refined metrics (`evaluate_refinement.py`) |
| `get_job_status` / `list_jobs` / `cancel_job` | Job lifecycle |
| `get_gpu_status` | GPU availability |
| `get_run_manifest` / `list_run_manifests` | Provenance of each job |
| `replay_run` | Re-run a manifest on this machine |

# Related

- [Downscale playbook](/playbooks/downscale-wxc.md)
- [Refinement playbook](/playbooks/refine-narr.md)
- [YAML data model](/concepts/dataset-yaml-model.md)
- Analysis side: [MCP analysis tools](/tools/mcp-analysis-tools.md)

# Citations

[1] `mcp/mcp-merra2-config.json`
[2] `mcp/mcp_stdio.py`
