---
type: Playbook
title: Analyze NetCDF outputs
description: Analyze PrithviWxC / granite-wxc NetCDF outputs for a date, region, difference map, trend, or climatology via MCP.
tags: [playbook, analysis, netcdf, plotting, climatology, mcp]
timestamp: 2026-10-01T06:00:00Z
---

Use when the user asks to plot, summarize, compare dates, or compute monthly/seasonal climate stats from downscaled or training NetCDF files (MERRA-2, NARR, PRISM `ppt`/`tmax`/`tmin`).

All analysis goes through the `prithvi-wxc-downscaling` MCP server. Do not scrape directories by hand unless `list_files` / `find_file_by_date` fail.

# Dataset split

| Dates | Source |
|-------|--------|
| Before 2016 | Training preprocessed files |
| 2016 and later | Inference outputs (may be missing until inference has run) |

Override with `dataset_type` only when the user is explicit.

# Steps

1. `get_dataset_metadata` / `get_available_variables` if coverage or names are unknown.
2. `load_by_date` with a natural-language date (`Jan 1, 2024`, `2024-01-01`, `20240101`).
3. Pick a tool:
   - Overview: `dataset_summary`, `list_variables`, `list_coordinates`
   - Stats: `variable_info`, `compute_statistics`, `slice_variable`
   - Maps: `plot_variable_2d` or `plot_data` (`map`, `heatmap`, `line`, `histogram`)
   - Compare two dates: `compare_dates_difference_map`
   - Time: `compute_trend_timeseries`, `compute_monthly_climatology`, `compute_seasonal_climatology`, `compute_period_composite`
4. Clip spatially with `lat_min` / `lat_max` / `lon_min` / `lon_max` when the user names a region (LA County ≈ 33.3–34.9 N, 119.0–117.6 W).

# Variable names

| Kind | Names |
|------|-------|
| Predictors (500/700/850) | `predictor_QV_*`, `predictor_T_*`, `predictor_U_*`, `predictor_V_*`, `predictor_H_*` |
| Targets | `target_ppt`, `target_tmax`, `target_tmin` |
| Static | `static_elevation` |

# Artifacts

Plots land in `<PIPELINE_DATA_ROOT>/.mcp-state/artifacts/`; tool results include the local `path`. Report it (and `/artifacts/{filename}` if the HTTP server is up). Do not delete artifacts. If a date is outside file coverage, report actual coverage from NetCDF metadata instead of silently clamping.

# Related

- [MCP analysis tools](/tools/mcp-analysis-tools.md)
- Upstream pipeline: [Downscale playbook](/playbooks/downscale-wxc.md)
- [Reproducibility rules](/constraints/reproducibility.md)

# Citations

[1] [OKF playbook pattern](https://okf.md/examples)
[2] [OKF spec](https://okf.md/spec)
