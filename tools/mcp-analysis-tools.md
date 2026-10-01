---
type: Reference
title: MCP analysis tools
description: MCP tools for loading NetCDF by date, statistics, maps, difference plots, and climatologies.
tags: [mcp, tools, analysis, plotting, climatology]
timestamp: 2026-10-01T06:00:00Z
resource: repo://mcp/mcp-merra2-config.json
---

Tools used by the [Analyze playbook](/playbooks/analyze-wxc.md).

# Catalog

| Tool | Purpose |
|------|---------|
| `get_dataset_metadata` | Coverage and dataset metadata |
| `get_available_variables` | Variable inventory |
| `load_by_date` | Load daily NetCDF by natural-language date |
| `dataset_summary` | Dimensions / overview |
| `list_variables` / `list_coordinates` | Names and ranges |
| `variable_info` / `compute_statistics` / `slice_variable` | Stats and slices |
| `plot_variable_2d` / `plot_data` | Maps, heatmaps, lines, histograms |
| `compare_dates_difference_map` | Difference between two dates |
| `compute_trend_timeseries` | Trends |
| `compute_monthly_climatology` / `compute_seasonal_climatology` / `compute_period_composite` | Climatologies |

# Related

- [Analyze playbook](/playbooks/analyze-wxc.md)
- Pipeline side: [MCP pipeline tools](/tools/mcp-pipeline-tools.md)

# Citations

[1] `mcp/mcp-merra2-config.json`
[2] `mcp/analyzer.py`
