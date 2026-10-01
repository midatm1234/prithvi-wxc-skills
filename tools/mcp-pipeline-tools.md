---
type: Reference
title: MCP pipeline tools
description: MCP tools for downloads, YAML configs, scalars, preprocessing, training, inference, and jobs.
tags: [mcp, tools, pipeline, training, inference]
timestamp: 2026-10-01T06:00:00Z
resource: repo://mcp/mcp-merra2-config.json
---

Tools used by the [Downscale playbook](/playbooks/downscale-wxc.md). Exact schemas live in the plugin MCP tool manifest.

# Catalog

| Tool | Purpose |
|------|---------|
| `check_raw_data_status` | Inventory local raw inputs |
| `start_download_job` | Download MERRA-2 / NARR / PRISM / CORDEX / elevation |
| `list_available_configs` | List YAML configs under granite-wxc examples |
| `read_yaml_config` | Read a config |
| `create_custom_yaml` | Write a new config (do not overwrite blindly) |
| `list_files` / `find_file_by_date` / `github_repo_context` | Discover paths and repo context |
| `start_compute_scalars_job` | Compute mean/std scalars |
| `start_preprocessing_job` | Align / write preprocessed NetCDFs |
| `run_training_pipeline` | Scalars→preproc→train (+ optional queued inference) |
| `start_inference_job` | Inference-only job |
| `get_job_status` / `list_jobs` / `cancel_job` | Job lifecycle |
| `get_gpu_status` | GPU availability |

# Related

- [Downscale playbook](/playbooks/downscale-wxc.md)
- [YAML data model](/concepts/dataset-yaml-model.md)
- Analysis side: [MCP analysis tools](/tools/mcp-analysis-tools.md)

# Citations

[1] `mcp/mcp-merra2-config.json`
[2] `mcp/mcp_stdio.py`
