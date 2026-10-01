---
type: Concept
title: YAML data model
description: How the YAML `data:` block selects predictors, targets, grids, and scalers.
tags: [yaml, config, netcdf, predictors, targets]
timestamp: 2026-10-01T06:00:00Z
---

The YAML `data:` block drives everything. Switch sources by pointing `*_paths` at new NetCDF files and updating `input_vars` / `output_vars` to match names **in the files**.

# Schema

```yaml
data:
  type: cordex               # arbitrary label — change for NARR / ERA5 / MERRA-2
  training_predictor_paths: [/path/to/coarse_input.nc]
  training_target_paths:    [/path/to/highres_target.nc]
  validation_predictor_paths: [...]
  validation_target_paths:   [...]
  test_predictor_paths:      [...]
  test_target_paths:         [...]
  static_path: /path/to/Static_fields.nc
  input_vars:            [u, v, q, t, z]
  input_levels:          [850.0, 700.0, 500.0]
  output_vars:           [pr, tasmax]
  input_size_lat: 16
  input_size_lon: 16
  target_size_lat: 128
  target_size_lon: 128
  downsample_factor: 8
  scalers:
    inputs_mean:   /.../inputs_mean.npy
    inputs_std:    /.../inputs_std.npy
    targets_mean:  /.../targets_mean.npy
    targets_std:   /.../targets_std.npy
```

# Clarify before authoring a config

1. Predictor (coarse) source + paths
2. Target (fine) source + paths
3. Variable names that match the NetCDF files
4. Pressure levels (if 3-D predictors)
5. Coarse and fine grid sizes → `downsample_factor`
6. Region / bbox
7. Training vs inference date ranges
8. Mode — training, inference only, or both

If paths or names are vague, use MCP `list_files` / `github_repo_context` rather than guessing.

# Source-specific gotchas

| Source | Notes |
|--------|-------|
| NARR | Names differ (`TMP2m`, `APCP`, …); Lambert grid may need regridding to lat/lon |
| ERA5 | Pressure dim is often `level` not `lev`; single-level and pressure files are separate |
| MERRA-2 | Names like `T2M`, `U10M`, `QV2M`, `PRECTOT`; regular lat/lon |
| Mixed sources | Confirm target grid covers predictor bbox after crop; `downsample_factor` must match real resolution ratio |

# Related

- Overview: [Downscaling overview](/concepts/downscaling-overview.md)
- How to build configs: [Downscale playbook](/playbooks/downscale-wxc.md)
- Tools: [MCP pipeline tools](/tools/mcp-pipeline-tools.md)

# Citations

[1] Plugin MCP config: `mcp.json` (`create_custom_yaml`, `read_yaml_config`)
