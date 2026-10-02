---
name: prithvi-data
description: Download the inputs for Prithvi WxC downscaling — MERRA-2 or NARR predictors, PRISM 800 m targets (ppt/tmax/tmin), static 800 m orography, pretrained weights, and optionally CORDEX-ML-Bench — via the prithvi-wxc-downscaling MCP download jobs. Use when the user asks to download, fetch, or check MERRA-2, NARR, PRISM, elevation/orography, weights, or training/inference data for a date range.
---

# Download pipeline inputs

All versions are pinned in `mcp/pins.json`; downloads are resumable and skip files already present.

1. `check_raw_data_status` — see what is already on disk and what is missing.
2. Agree the date range with the user. The reference setup trains on 1996-01-01..2015-12-31 and evaluates 2016-01-01..2025-12-31. Measured cost per year of daily data: MERRA-2 transfers ~1.14 GB per day (~415 GB/year) but keeps only a ~1.4 MB/day subset (raw files are deleted after subsetting); PRISM keeps ~23 MB per file, ~25 GB/year for ppt+tmax+tmin, and is rate-limited to one file every ~2 s (~40 min per variable-year). Confirm before anything longer than a month.
3. Start jobs with `start_download_job`. Different datasets can run in parallel; overlapping date ranges of the same dataset are refused (chain them with `depends_on` instead):

   | dataset | notes |
   |---------|-------|
   | `merra2` | needs Earthdata Login (see `prithvi-setup`); writes `merra2/daily_subset_with_H` |
   | `narr` | NOAA PSL, no login; alternative predictor set |
   | `prism` | PRISM AN daily 800 m; 2 s pause between requests per PRISM policy |
   | `elevation` | reference 800 m orography from Zenodo (DOI 10.5281/zenodo.23096854), sha256-verified |
   | `weights` | ~17.4 GB from Hugging Face, sha256-verified; check free disk first |
   | `all` | merra2 + prism + elevation + weights for one date range |

   Pass `start_date`/`end_date` (YYYY-MM-DD) or `config_path` to use that config's training dates.
4. Poll `get_job_status` (every minute or so for long jobs; it returns a log tail). On failure, read the log tail, explain the cause (auth, disk, network), and retry with the same arguments once the cause is fixed.
5. Finish with `check_raw_data_status` and report coverage (first/last date per dataset).

## Rules

- If the elevation download fails its checksum, do not substitute another terrain file; report the error. If the user points `ELEVATION_FILE` at their own file and the status shows `matches_reference: false`, tell them results will not be bit-identical to the published runs.
- To reuse data that already exists elsewhere (e.g. a shared `/data` volume), set `MERRA2_DATA_DIR` / `PRISM_DATA_DIR` / `NARR_DATA_DIR` / `ELEVATION_FILE` / `MODEL_WEIGHTS_FILE` in `~/.config/prithvi-wxc/env` instead of re-downloading.
