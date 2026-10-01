---
name: analyze-wxc
description: Analyze PrithviWxC / granite-wxc NetCDF outputs (MERRA-2, NARR, PRISM ppt/tmax/tmin) for a date, region, difference map, trend, or climatology. Use when the user asks to plot, summarize, compare dates, or compute monthly/seasonal climate stats from downscaled or training files.
---

# Analyze downscaled and training NetCDF with MCP tools

All analysis goes through the `prithvi-wxc-downscaling` MCP server. Do not scrape directories by hand unless `list_files` / `find_file_by_date` fail.

## Dataset split

- Dates before 2016 → training preprocessed files
- Dates 2016 and later → inference outputs (may be missing until inference has run)

Override with `dataset_type` only when the user is explicit.

## Workflow

1. `get_dataset_metadata` / `get_available_variables` if you do not know coverage or names.
2. `load_by_date` with a natural-language date (`Jan 1, 2024`, `2024-01-01`, `20240101`).
3. Pick a tool:
   - Overview: `dataset_summary`, `list_variables`, `list_coordinates`
   - Stats: `variable_info`, `compute_statistics`, `slice_variable`
   - Maps: `plot_variable_2d` or `plot_data` (`map`, `heatmap`, `line`, `histogram`)
   - Compare two dates: `compare_dates_difference_map`
   - Time: `compute_trend_timeseries`, `compute_monthly_climatology`, `compute_seasonal_climatology`, `compute_period_composite`
4. Clip spatially with `lat_min` / `lat_max` / `lon_min` / `lon_max` when the user names a region (LA County ≈ 33.3–34.9 N, 119.0–117.6 W).

## Variable names

Predictors (levels 500/700/850): `predictor_QV_*`, `predictor_T_*`, `predictor_U_*`, `predictor_V_*`, `predictor_H_*`

Targets: `target_ppt`, `target_tmax`, `target_tmin`

Static: `static_elevation`

## Artifacts

Plots land in `mcp/artifacts/`. Report the path (and `/artifacts/{filename}` if the HTTP server is up). Do not delete artifacts.
