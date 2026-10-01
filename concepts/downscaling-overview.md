---
type: Concept
title: Downscaling overview
description: Dataset-agnostic PrithviWxC / granite-wxc weather and climate downscaling stack.
tags: [prithvi, granite-wxc, downscaling, climate, merra2, narr]
timestamp: 2026-10-01T06:00:00Z
---

The downscaler is dataset-agnostic. Any gridded NetCDF works when paths and variable names are supplied via a YAML config. There is **no** per-source loader; NARR, ERA5, MERRA-2, CORDEX, ACCESS-CM2, and similar sources flow through the same pipeline.

# Stack

| Component | Role |
|-----------|------|
| Cursor plugin + MCP stdio (`mcp/mcp_stdio.py`) | Tool host for agents |
| Optional HTTP MCP (`mcp/mcp_server.py`, port 8001) | Local HTTP tool server |
| Optional agent UI (`agent_server.py`, port 8000) | Chat orchestration |
| granite-wxc checkout (`GRANITE_WXC_REPO`) | Training / inference scripts and example YAMLs |
| Local data root (`PIPELINE_DATA_ROOT` or `~/prithvi-wxc-data`) | Downloaded raw inputs |

# Related

- Config model: [YAML data model](/concepts/dataset-yaml-model.md)
- End-to-end run: [Downscale playbook](/playbooks/downscale-wxc.md)
- Post-run analysis: [Analyze playbook](/playbooks/analyze-wxc.md)
- Tool catalog: [MCP pipeline tools](/tools/mcp-pipeline-tools.md)
- Guardrails: [Reproducibility rules](/constraints/reproducibility.md)

# Citations

[1] [OKF spec](https://okf.md/spec)
[2] granite-wxc `examples/` reference configs (set via `GRANITE_WXC_REPO`)
