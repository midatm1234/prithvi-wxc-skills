# PrithviWxC Downscaling Knowledge

Agent-portable knowledge for NASA PrithviWxC / granite-wxc weather and climate downscaling: YAML config model, MCP tool workflows, analysis playbooks, and reproducibility rules.

Conformant with [OKF v0.1](https://okf.md/spec).

## Concepts

* [Downscaling overview](./concepts/downscaling-overview.md) - Dataset-agnostic PrithviWxC / granite-wxc downscaling stack
* [YAML data model](./concepts/dataset-yaml-model.md) - How `data:` paths and variables select predictors and targets

## Playbooks

* [Downscale weather/climate grids](./playbooks/downscale-wxc.md) - Config → scalars → preprocess → train/infer via MCP
* [Analyze NetCDF outputs](./playbooks/analyze-wxc.md) - Load by date, stats, maps, trends, and climatologies via MCP

## Tools

* [MCP pipeline tools](./tools/mcp-pipeline-tools.md) - Download, YAML, scalars, preprocess, train, infer, jobs
* [MCP analysis tools](./tools/mcp-analysis-tools.md) - Load, summarize, plot, compare, and climatology tools

## Constraints

* [Reproducibility rules](./constraints/reproducibility.md) - Path policy, pipeline order, and guardrails
