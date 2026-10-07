# PrithviWxC Downscaling Knowledge

Agent-portable knowledge for NASA PrithviWxC / granite-wxc weather and climate downscaling: YAML config model, MCP tool workflows, analysis playbooks, and reproducibility rules.

Conformant with [OKF v0.1](https://okf.md/spec).

**Runtime:** executable MCP tools live in [`mcp/`](./mcp/), launched by `bin/prithvi-mcp`. Install as a Claude Code plugin, or register with any MCP host (see README), so playbooks can download, preprocess, train, infer, and replay — not just describe those steps. Claude Code skills in [`skills/`](./skills/) are thin entry points into these playbooks.

## Concepts

* [Downscaling overview](./concepts/downscaling-overview.md) - Dataset-agnostic PrithviWxC / granite-wxc downscaling stack
* [YAML data model](./concepts/dataset-yaml-model.md) - How `data:` paths and variables select predictors and targets

## Playbooks

* [Set up a machine](./playbooks/setup-machine.md) - Environment check, pinned code, training env, credentials
* [Reproduce a run](./playbooks/reproduce-run.md) - Run manifests, sharing, and replay across machines
* [Downscale weather/climate grids](./playbooks/downscale-wxc.md) - Config → preprocess → scalars → train → tiled inference → evaluation via MCP
* [Refine NARR downscaling](./playbooks/refine-narr.md) - Diffusion / flow-matching residual refinement and ensembles (NARR only)
* [Analyze NetCDF outputs](./playbooks/analyze-wxc.md) - Load by date, stats, maps, trends, and climatologies via MCP

## Tools

* [MCP pipeline tools](./tools/mcp-pipeline-tools.md) - Download, YAML, preprocess, scalars, train, infer, evaluate, NARR refinement, jobs
* [MCP analysis tools](./tools/mcp-analysis-tools.md) - Load, summarize, plot, compare, and climatology tools

## Constraints

* [Reproducibility rules](./constraints/reproducibility.md) - Path policy, pipeline order, and guardrails
