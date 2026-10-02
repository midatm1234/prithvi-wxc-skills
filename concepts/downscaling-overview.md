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
| MCP stdio server (`bin/prithvi-mcp` → `mcp/mcp_stdio.py`) | Tool host for Claude Code, Claude Desktop, Cursor, and other MCP clients |
| Optional HTTP MCP (`mcp/mcp_server.py`, port 8001) | Shared-server deployments |
| Pinned code (`PIPELINE_DATA_ROOT/code/Prithvi-UNet-stocahstic`, or `GRANITE_WXC_REPO`) | Training / inference scripts and example YAMLs |
| Training env (`PIPELINE_DATA_ROOT/envs/prithvi`, or `GRANITE_WXC_PYTHON`) | torch + PrithviWxC + granitewxc |
| Data root (`PIPELINE_DATA_ROOT`, default `~/prithvi-wxc-data`) | Inputs, weights, and `.mcp-state/` provenance |

# Reference setup

NARR (32 km) or MERRA-2 predictors — T, U, V, QV, H at 500/700/850 hPa, no near-surface fields — plus 800 m orography feed the Prithvi WxC 2.3B backbone with a UNet decoder, predicting daily 800 m PRISM precipitation, Tmax, and Tmin. Training 1996–2015, evaluation 2016–2025. The optional `stochastic_refinement` variant adds a diffusion / flow-matching model of the residual `y_true - y_hat` to turn the deterministic prediction into an ensemble.

# Related

- Config model: [YAML data model](/concepts/dataset-yaml-model.md)
- End-to-end run: [Downscale playbook](/playbooks/downscale-wxc.md)
- Post-run analysis: [Analyze playbook](/playbooks/analyze-wxc.md)
- Tool catalog: [MCP pipeline tools](/tools/mcp-pipeline-tools.md)
- Guardrails: [Reproducibility rules](/constraints/reproducibility.md)

# Citations

[1] [OKF spec](https://okf.md/spec)
[2] granite-wxc `examples/` reference configs (set via `GRANITE_WXC_REPO`)
