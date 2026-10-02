---
name: prithvi-analyze
description: Analyze Prithvi WxC downscaling inputs and outputs (preprocessed training NetCDF or inference output) through the prithvi-wxc-downscaling MCP tools — load a date, summarize variables, regional statistics, maps, date differences, trends, monthly/seasonal climatologies. Use when the user asks about values, plots, statistics, or climatology of downscaled precipitation, Tmax, Tmin, or the predictors.
---

# Analyze downscaled data

Full workflow: [analyze playbook](../../playbooks/analyze-wxc.md). Tool catalog: [analysis tools](../../tools/mcp-analysis-tools.md).

1. `load_by_date` for the requested date (before 2016 resolves to training data, 2016+ to inference output), or `load_dataset` with an explicit file.
2. `dataset_summary` / `list_variables` to confirm variable names (`target_ppt`, `target_tmax`, `target_tmin`, `predictor_T_850`, ...).
3. Pick the tool: `compute_statistics` or `variable_info` (numbers), `plot_variable_2d` or `plot_data` (maps, histograms, transects), `compare_dates_difference_map`, `compute_trend_timeseries`, `compute_monthly_climatology`, `compute_seasonal_climatology`, `compute_period_composite`. Named regions such as `la_county` and `socal` work via `region`.
4. Plot tools return a local `path` to a PNG under the MCP state directory; give the user that path (the `plot_url` only works with the optional HTTP server).

## Rules

- If a date is outside file coverage, report the actual coverage instead of guessing.
- Units: ppt in mm/day; tmax/tmin as stored in the file (check `variable_info`).
