#!/usr/bin/env python3
"""
Download MERRA-2 files to ./merra2.

Config-driven (no long CLI). Edit CONFIG below.

This script supports two modes:
1) "earthaccess": uses NASA Earthdata Login via earthaccess, following the
   GESDISC tutorial pattern (search_data + download).
2) "s3": direct S3/HTTPS downloads using bucket/prefix/pattern or a manifest.
"""

from __future__ import annotations

import concurrent.futures as futures
import argparse
import os
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import boto3
import botocore
import requests


# -----------------------------
# CONFIG (edit these values)
# -----------------------------
CONFIG = {
    # Output directory
    "output_dir": "/data/merra2/download_tmp",
    "existing_download_dirs": ["/data/merra2"],

    # Select mode: "earthaccess" (download via earthaccess), "s3" (direct S3),
    # or "list_s3_urls" (query CMR and write S3 URLs).
    "mode": "earthaccess",

    # --- Mode: earthaccess ---
    # Uses Earthdata Login + search to get MERRA-2 granules, then downloads.
    # You must have Earthdata credentials set up (see earthaccess docs).
    "earthaccess": {
        "daac": "GESDISC",
        "doi": "",  # Example DOI (adjust for your collection)
        # MERRA-2 inst3_3d_asm_Np collection short_name
        "short_name": "M2I3NPASM",
        # Use start_date/end_date as canonical keys for day range.
        "start_date": "1995-01-01",
        "end_date": "2025-12-31",
        # Backward-compatible aliases.
        "temporal_start": "2020-01-01",
        "temporal_end": "2020-01-05",
        "max_results": None,  # set an int to cap results
        "require_us_west_2": False,  # Set True if you require in-region access
        # Avoid interactive prompts in batch/non-interactive runs. Supported
        # values include "environment", "netrc", "interactive", or a list.
        "login_strategies": ["environment", "netrc"],
        "persist_login": False,
    },

    # Output file for list_s3_urls mode
    "list_output": "./merra2_s3_urls.txt",

    # --- Mode: s3 ---
    # Option A: inline URL list (s3:// or https://)
    "urls": [
        "s3://gesdisc-cumulus-prod-protected/MERRA2/M2I3NPASM.5.12.4/2020/01/MERRA2_400.inst3_3d_asm_Np.20200101.nc4",
        "s3://gesdisc-cumulus-prod-protected/MERRA2/M2I3NPASM.5.12.4/2020/01/MERRA2_400.inst3_3d_asm_Np.20200102.nc4",
        "s3://gesdisc-cumulus-prod-protected/MERRA2/M2I3NPASM.5.12.4/2020/01/MERRA2_400.inst3_3d_asm_Np.20200103.nc4",
        "s3://gesdisc-cumulus-prod-protected/MERRA2/M2I3NPASM.5.12.4/2020/01/MERRA2_400.inst3_3d_asm_Np.20200104.nc4",
        "s3://gesdisc-cumulus-prod-protected/MERRA2/M2I3NPASM.5.12.4/2020/01/MERRA2_400.inst3_3d_asm_Np.20200105.nc4",
    ],

    # Option B: manifest file with one URL per line (s3:// or https://)
    # If set (non-empty), manifest is used and the S3 pattern settings below are ignored.
    "manifest": "",

    # Option C: generate S3 URLs from date range + pattern
    "s3_bucket": "YOUR_BUCKET",
    "s3_prefix": "YOUR_PREFIX",  # e.g. "MERRA2/M2I3NPASM.5.12.4/2020/01"
    "pattern": "MERRA2_400.inst3_3d_asm_Np.{YYYY}{MM}{DD}.nc4",
    "start_date": "2020-01-01",
    "end_date": "2020-01-31",

    # Auth + behavior
    "anonymous": False,  # True for public buckets
    "max_workers": 4,
    "dry_run": False,

    # Optional AWS profile name from ~/.aws/credentials
    "aws_profile": "default",
    "credentials_file": "./aws_credentials",


    # If True, use Earthdata Login to fetch temporary AWS credentials
    # for protected NASA S3 buckets (recommended for gesdisc-cumulus-prod-protected).
    "use_earthaccess_creds": True,

    # Post-processing after download
    "subset_compile": {
        "enabled": True,
        "bbox": (-127, 12.56, 23.26, 60),  # (west, south, east, north)
        "variables": ["QV", "T", "U", "V", "H"],
        "pressure_levels": [500, 700, 850],  # hPa
        "daily_input_dir": "/data/merra2/daily_subset",
        "daily_output_dir": "/data/merra2/daily_subset_with_H",
        "combined_output_template": "./merra2/M2I3NPASM_subset_{start}_{end}.nc4",
        "write_combined_file": False,
        "cleanup_downloaded_files": True,
        "cleanup_stale_downloads": True,
        "postprocess_redownload_attempts": 1,
        "overwrite_h": False,
    },

    # If True, skip this run completely when output files already exist.
    "skip_if_existing_files": True,
}


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def date_range(start: datetime, end: datetime) -> Iterable[datetime]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def read_manifest(path: Path) -> List[str]:
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def build_s3_urls(bucket: str, prefix: str, pattern: str, start: str, end: str) -> List[str]:
    if not (pattern and start and end):
        raise ValueError("pattern, start_date, and end_date are required to build keys")
    start_dt = parse_date(start)
    end_dt = parse_date(end)
    urls = []
    for d in date_range(start_dt, end_dt):
        y = d.strftime("%Y")
        m = d.strftime("%m")
        dd = d.strftime("%d")
        fname = pattern.format(YYYY=y, MM=m, DD=dd)
        key = "/".join([p for p in [prefix.strip("/"), fname] if p])
        urls.append(f"s3://{bucket}/{key}")
    return urls


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_s3(s3_client, bucket: str, key: str, out_path: Path) -> None:
    if out_path.exists():
        print(f"Skipping (already exists): {out_path.name}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, key, str(out_path))


def download_http(url: str, out_path: Path) -> None:
    if out_path.exists():
        print(f"Skipping (already exists): {out_path.name}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    # s3://bucket/key
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def list_netcdf_files(path: Path) -> List[Path]:
    return sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".nc", ".nc4"}])


def find_existing_download_file(filename: Optional[str], out_dir: Path) -> Optional[Path]:
    if not filename:
        return None
    candidates = [out_dir / filename]
    for dirname in CONFIG.get("existing_download_dirs", []):
        candidate = Path(dirname) / filename
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def extract_yyyymmdd(text: str) -> Optional[str]:
    m = re.search(r"(19|20)\d{6}", text)
    return m.group(0) if m else None


def get_date_bounds(mode: str, cfg: dict) -> Tuple[datetime, datetime]:
    if mode == "earthaccess":
        sub = cfg["earthaccess"]
        start = sub.get("start_date") or sub.get("temporal_start")
        end = sub.get("end_date") or sub.get("temporal_end")
        if not start or not end:
            raise ValueError("earthaccess.start_date/end_date (or temporal_start/temporal_end) are required")
        return parse_date(start), parse_date(end)
    return parse_date(cfg["start_date"]), parse_date(cfg["end_date"])


def detect_coord_name(ds, candidates: List[str], coord_kind: str) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"Could not find {coord_kind} coordinate. Tried: {candidates}")


def subset_dataset_bbox(ds, lat_name: str, lon_name: str, bbox: Tuple[float, float, float, float]):
    import numpy as np

    west, south, east, north = bbox

    # Convert 0..360 longitudes to -180..180 when bbox uses negative longitudes.
    lon_vals = ds[lon_name].values
    lon_max = float(np.nanmax(lon_vals))
    if (west < 0 or east < 0) and lon_max > 180:
        wrapped = ((ds[lon_name] + 180) % 360) - 180
        ds = ds.assign_coords({lon_name: wrapped}).sortby(lon_name)

    lat_vals = ds[lat_name].values
    if lat_vals[0] <= lat_vals[-1]:
        lat_slice = slice(south, north)
    else:
        lat_slice = slice(north, south)

    ds = ds.sel({lat_name: lat_slice})

    if west <= east:
        ds = ds.sel({lon_name: slice(west, east)})
    else:
        left = ds.sel({lon_name: slice(west, float(ds[lon_name].max()))})
        right = ds.sel({lon_name: slice(float(ds[lon_name].min()), east)})
        ds = left.combine_first(right).sortby(lon_name)

    return ds


def daily_subset_file_path(day_ymd: str, settings: Dict, create_dir: bool = True, kind: str = "output") -> Path:
    if kind == "input":
        daily_dir = Path(settings.get("daily_input_dir", settings.get("daily_output_dir", "./merra2/daily_subset")))
    else:
        daily_dir = Path(settings.get("daily_output_dir", "./merra2/daily_subset_with_H"))
    if create_dir:
        daily_dir.mkdir(parents=True, exist_ok=True)
    return daily_dir / f"M2I3NPASM_subset_{day_ymd}.nc4"


def combined_subset_file_path(start_dt: datetime, end_dt: datetime, settings: Dict) -> Path:
    out_tpl = settings.get(
        "combined_output_template",
        "./merra2/M2I3NPASM_subset_{start}_{end}.nc4",
    )
    return Path(
        out_tpl.format(
            start=start_dt.strftime("%Y%m%d"),
            end=end_dt.strftime("%Y%m%d"),
        )
    )


def subset_variables_with_h(settings: Dict) -> List[str]:
    variables = list(settings.get("variables", ["QV", "T", "U", "V"]))
    if "H" not in variables:
        variables.append("H")
    return variables


MERRA2_PRESSURE_LEVELS = [
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 650,
    600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 70, 50, 40, 30,
    20, 10, 7, 5, 4, 3, 2, 1, 0.7, 0.5, 0.4, 0.3, 0.1,
]


def contiguous_index_runs(indexes):
    indexes = [int(i) for i in indexes]
    if not indexes:
        return
    start = prev = indexes[0]
    positions = [0]
    for pos, idx in enumerate(indexes[1:], start=1):
        if idx == prev + 1:
            prev = idx
            positions.append(pos)
            continue
        yield start, prev + 1, positions
        start = prev = idx
        positions = [pos]
    yield start, prev + 1, positions


def merra2_lat_indexes(lat_values):
    import numpy as np

    indexes = np.rint((np.asarray(lat_values, dtype=float) + 90.0) / 0.5).astype(int)
    if np.any(indexes < 0) or np.any(indexes >= 361):
        raise ValueError("Latitude values are outside the expected MERRA-2 grid")
    return indexes.tolist()


def merra2_lon_indexes(lon_values):
    import numpy as np

    lon = np.asarray(lon_values, dtype=float)
    source_lon = np.where(lon < 0, lon + 360.0, lon)
    indexes = np.rint(source_lon / 0.625).astype(int) % 576
    return indexes.tolist()


def merra2_pressure_level_index(level) -> int:
    import numpy as np

    for idx, candidate in enumerate(MERRA2_PRESSURE_LEVELS):
        if np.isclose(float(level), float(candidate)):
            return idx
    raise ValueError(f"Unsupported MERRA-2 pressure level for fallback H read: {level}")


def build_h_daily_subset_dataset_fallback(src_file: Path, existing_ds):
    import numpy as np
    import xarray as xr
    from netCDF4 import Dataset

    lat_name = detect_coord_name(existing_ds, ["lat", "latitude", "LATITUDE", "y"], "latitude")
    lon_name = detect_coord_name(existing_ds, ["lon", "longitude", "LONGITUDE", "x"], "longitude")
    lev_name = detect_coord_name(existing_ds, ["lev", "level", "plev", "pressure"], "level")
    time_name = detect_coord_name(existing_ds, ["time", "TIME", "valid_time"], "time")

    lat_indexes = merra2_lat_indexes(existing_ds[lat_name].values)
    lon_indexes = merra2_lon_indexes(existing_ds[lon_name].values)
    lev_indexes = [merra2_pressure_level_index(level) for level in existing_ds[lev_name].values]

    with Dataset(src_file) as nc:
        h_var = nc.variables["H"]
        pieces = []
        for lev_idx in lev_indexes:
            level_data = np.ma.masked_all(
                (h_var.shape[0], len(lat_indexes), len(lon_indexes)),
                dtype=np.float32,
            )
            lat_start = lat_indexes[0]
            lat_stop = lat_indexes[-1] + 1
            for lon_start, lon_stop, positions in contiguous_index_runs(lon_indexes):
                block = h_var[:, lev_idx, lat_start:lat_stop, lon_start:lon_stop]
                level_data[:, :, positions] = block
            pieces.append(np.ma.mean(level_data, axis=0))

        data = np.ma.stack(pieces, axis=0).filled(np.nan).astype(np.float32)
        attrs = {name: getattr(h_var, name) for name in h_var.ncattrs() if name != "_FillValue"}

    data = data[np.newaxis, :, :, :]
    ds_h = xr.Dataset(
        {"H": ((time_name, lev_name, lat_name, lon_name), data, attrs)},
        coords={
            time_name: existing_ds[time_name].values,
            lev_name: existing_ds[lev_name].values,
            lat_name: existing_ds[lat_name].values,
            lon_name: existing_ds[lon_name].values,
        },
    )
    for coord_name in (time_name, lev_name, lat_name, lon_name):
        ds_h[coord_name].attrs = dict(existing_ds[coord_name].attrs)
    return ds_h


def h_status_for_daily_file(daily_file: Path, overwrite_h: bool = False) -> str:
    if not daily_file.exists():
        return "missing_file"
    try:
        import xarray as xr
    except Exception as e:
        raise ImportError("xarray is required to inspect daily subset files. Install it first.") from e

    with xr.open_dataset(daily_file) as ds:
        if "H" in ds.data_vars:
            return "overwrite_h" if overwrite_h else "h_present"
        return "h_missing"


def copy_existing_h_output(source_file: Path, target_file: Path, settings: Dict) -> bool:
    if not source_file.exists() or target_file.exists():
        return False
    if h_status_for_daily_file(source_file) != "h_present":
        return False
    if CONFIG.get("dry_run"):
        print(f"Dry run: would copy existing H daily file {source_file} -> {target_file}.")
        return True
    target_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = target_file.with_name(target_file.name + ".tmp")
    if tmp_file.exists():
        tmp_file.unlink()
    shutil.copy2(source_file, tmp_file)
    validate_daily_file(tmp_file, settings, emit_diagnostics=False)
    os.replace(tmp_file, target_file)
    validate_daily_file(target_file, settings)
    print(f"Copied existing H daily file to {target_file}")
    record_h_summary(settings, "created")
    return True


def print_h_diagnostics(ds, label: str = "H") -> None:
    import numpy as np

    h = ds["H"].load()
    values = h.values
    nan_count = int(np.isnan(values).sum())
    finite_values = values[np.isfinite(values)]
    if finite_values.size:
        h_min = float(finite_values.min())
        h_max = float(finite_values.max())
        h_mean = float(finite_values.mean())
    else:
        h_min = h_max = h_mean = float("nan")
    units = h.attrs.get("units", "unknown")
    print(
        f"{label} diagnostics: dims={h.dims}, shape={h.shape}, "
        f"min={h_min}, max={h_max}, mean={h_mean}, NaNs={nan_count}, units={units}"
    )


def values_equal(left, right) -> bool:
    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None and (hasattr(left, "shape") or hasattr(right, "shape")):
        try:
            return bool(np.array_equal(left, right, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(left, right))
    try:
        return bool(left == right)
    except ValueError:
        if np is None:
            return False
        try:
            return bool(np.array_equal(left, right, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(left, right))


def attrs_equal(left: Dict, right: Dict) -> bool:
    if set(left) != set(right):
        return False
    return all(values_equal(left[key], right[key]) for key in left)


def existing_dataset_snapshot(ds) -> Dict:
    return {
        "attrs": dict(ds.attrs),
        "coords": set(ds.coords),
        "dims": dict(ds.sizes),
        "data_vars": set(ds.data_vars),
        "var_dims": {name: tuple(ds[name].dims) for name in ds.data_vars},
        "var_shapes": {name: tuple(ds[name].shape) for name in ds.data_vars},
        "var_attrs": {name: dict(ds[name].attrs) for name in ds.data_vars},
        "coord_dims": {name: tuple(ds[name].dims) for name in ds.coords},
        "coord_shapes": {name: tuple(ds[name].shape) for name in ds.coords},
        "coord_attrs": {name: dict(ds[name].attrs) for name in ds.coords},
    }


def validate_daily_file(
    daily_file: Path,
    settings: Dict,
    existing_snapshot: Optional[Dict] = None,
    emit_diagnostics: bool = True,
) -> None:
    import numpy as np
    import xarray as xr

    pressure_levels = settings.get("pressure_levels", [500, 700, 850])

    with xr.open_dataset(daily_file) as ds:
        if "H" not in ds.data_vars:
            raise ValueError("H is missing after writing daily subset")

        if existing_snapshot is not None:
            if not existing_snapshot["data_vars"].issubset(set(ds.data_vars)):
                dropped = sorted(existing_snapshot["data_vars"] - set(ds.data_vars))
                raise ValueError(f"Existing variables were dropped: {dropped}")
            if set(ds.coords) != existing_snapshot["coords"]:
                raise ValueError("Coordinates changed while adding H")
            if dict(ds.sizes) != existing_snapshot["dims"]:
                raise ValueError("Dimensions changed while adding H")
            if not attrs_equal(dict(ds.attrs), existing_snapshot["attrs"]):
                raise ValueError("Global attributes changed while adding H")
            for name in existing_snapshot["data_vars"]:
                if tuple(ds[name].dims) != existing_snapshot["var_dims"][name]:
                    raise ValueError(f"Dimensions changed for existing variable {name}")
                if tuple(ds[name].shape) != existing_snapshot["var_shapes"][name]:
                    raise ValueError(f"Shape changed for existing variable {name}")
                if name != "H" and not attrs_equal(dict(ds[name].attrs), existing_snapshot["var_attrs"][name]):
                    raise ValueError(f"Attributes changed for existing variable {name}")
            for name in existing_snapshot["coords"]:
                if tuple(ds[name].dims) != existing_snapshot["coord_dims"][name]:
                    raise ValueError(f"Dimensions changed for coordinate {name}")
                if tuple(ds[name].shape) != existing_snapshot["coord_shapes"][name]:
                    raise ValueError(f"Shape changed for coordinate {name}")
                if not attrs_equal(dict(ds[name].attrs), existing_snapshot["coord_attrs"][name]):
                    raise ValueError(f"Attributes changed for coordinate {name}")

        lev_name = detect_coord_name(ds, ["lev", "level", "plev", "pressure"], "level")
        h_dims = set(ds["H"].dims)
        required_dims = {
            detect_coord_name(ds, ["time", "TIME", "valid_time"], "time"),
            lev_name,
            detect_coord_name(ds, ["lat", "latitude", "LATITUDE", "y"], "latitude"),
            detect_coord_name(ds, ["lon", "longitude", "LONGITUDE", "x"], "longitude"),
        }
        if h_dims != required_dims:
            raise ValueError(f"H has unexpected dimensions: {ds['H'].dims}")

        lev_values = ds[lev_name].values
        if len(lev_values) != len(pressure_levels) or not np.allclose(lev_values, pressure_levels):
            raise ValueError(
                f"H pressure levels are {lev_values.tolist()}, expected {pressure_levels}"
            )

        ds.load()
        if emit_diagnostics:
            print_h_diagnostics(ds)


def atomic_write_daily_dataset(
    ds,
    daily_file: Path,
    settings: Dict,
    existing_snapshot: Optional[Dict] = None,
) -> None:
    tmp_file = daily_file.with_name(daily_file.name + ".tmp")
    if tmp_file.exists():
        tmp_file.unlink()
    daily_file.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(tmp_file)
    try:
        validate_daily_file(tmp_file, settings, existing_snapshot=existing_snapshot, emit_diagnostics=False)
    except Exception:
        if tmp_file.exists():
            tmp_file.unlink()
        raise
    os.replace(tmp_file, daily_file)
    validate_daily_file(daily_file, settings, existing_snapshot=existing_snapshot)


def record_h_summary(settings: Dict, status: str) -> None:
    summary = settings.get("_h_summary")
    if not summary:
        return
    if status == "skipped_h_present":
        summary["skipped"] += 1
    elif status == "updated":
        summary["updated"] += 1
    elif status == "created":
        summary["created"] += 1
    elif status == "failed":
        summary["failed"] += 1


def print_h_summary(settings: Dict) -> None:
    summary = settings.get("_h_summary")
    if not summary:
        return
    print("Final H update summary:")
    print(f"  skipped because H already existed: {summary['skipped']}")
    print(f"  existing files updated with H: {summary['updated']}")
    print(f"  missing daily files newly created: {summary['created']}")
    print(f"  failures: {summary['failed']}")


def should_download_for_daily_h(day_ymd: Optional[str], settings: Dict) -> bool:
    if not day_ymd:
        return True

    daily_file = daily_subset_file_path(day_ymd, settings, create_dir=False)
    source_file = daily_subset_file_path(day_ymd, settings, create_dir=False, kind="input")
    overwrite_h = bool(settings.get("overwrite_h", False))
    status = h_status_for_daily_file(daily_file, overwrite_h=overwrite_h)

    if status in {"h_present", "overwrite_h"}:
        if status == "overwrite_h":
            print(f"{day_ymd}: H output exists; will replace because --overwrite-h is enabled.")
            return True
        print(f"{day_ymd}: H output already exists; skipped.")
        record_h_summary(settings, "skipped_h_present")
        return False

    if source_file.exists():
        source_status = h_status_for_daily_file(source_file, overwrite_h=False)
        if source_status == "h_present":
            copy_existing_h_output(source_file, daily_file, settings)
            return False
        print(f"{day_ymd}: source daily file found; H output missing; will download/subset H only.")
        return True

    print(f"{day_ymd}: source daily file missing and H output missing; will download/create full H output.")
    return True


def should_skip_for_existing_outputs(mode: str, cfg: dict, out_dir: Path) -> bool:
    if not cfg.get("skip_if_existing_files", True):
        return False

    subset_cfg = cfg.get("subset_compile", {})
    if not subset_cfg.get("enabled", False):
        return False

    start_dt, end_dt = get_date_bounds(mode, cfg)
    for d in date_range(start_dt, end_dt):
        daily_file = daily_subset_file_path(d.strftime("%Y%m%d"), subset_cfg, create_dir=False)
        if h_status_for_daily_file(daily_file) != "h_present":
            return False
    print("Found existing H daily subsets for the full requested range. Skipping download/subset.")
    return True


def _merge_and_dedup_time(ds_list, time_name: str):
    import xarray as xr

    merged = xr.concat(ds_list, dim=time_name).sortby(time_name)
    idx = merged.indexes.get(time_name)
    if idx is not None and idx.has_duplicates:
        keep = ~idx.duplicated()
        merged = merged.isel({time_name: keep})
    return merged


def build_daily_subset_dataset(
    src_file: Path,
    start_dt: datetime,
    end_dt: datetime,
    settings: Dict,
    variables: List[str],
):
    try:
        import numpy as np
        import xarray as xr
    except Exception as e:
        raise ImportError("xarray and numpy are required for subset/compile. Install them first.") from e

    bbox = tuple(settings.get("bbox", (-180, -90, 180, 90)))
    if len(bbox) != 4:
        raise ValueError("subset_compile.bbox must be a 4-tuple: (west, south, east, north)")

    pressure_levels = settings.get("pressure_levels", [500, 700, 850])

    with xr.open_dataset(src_file) as ds:
        lat_name = detect_coord_name(ds, ["lat", "latitude", "LATITUDE", "y"], "latitude")
        lon_name = detect_coord_name(ds, ["lon", "longitude", "LONGITUDE", "x"], "longitude")
        time_name = detect_coord_name(ds, ["time", "TIME", "valid_time"], "time")

        # Detect pressure level dimension
        lev_name = detect_coord_name(ds, ["lev", "level", "plev", "pressure"], "level")

        # Select only the requested variables (keep coords)
        available_vars = [v for v in variables if v in ds.data_vars]
        if not available_vars:
            print(f"Warning: none of {variables} found in {src_file.name}. Skipping.")
            return None
        ds_sub = ds[available_vars]

        # Select pressure levels
        ds_sub = ds_sub.sel({lev_name: pressure_levels})

        # Spatial subset
        ds_sub = subset_dataset_bbox(ds_sub, lat_name, lon_name, bbox)

        day_ymd = extract_yyyymmdd(src_file.name)
        if not day_ymd:
            first_day = np.datetime_as_string(ds_sub[time_name].values[0], unit="D")
            day_ymd = first_day.replace("-", "")

        day = datetime.strptime(day_ymd, "%Y%m%d")
        if not (start_dt.date() <= day.date() <= end_dt.date()):
            return None

        # Compute daily mean across the 8 3-hourly timesteps
        ds_sub = ds_sub.sortby(time_name).load()
        ds_daily = ds_sub.mean(dim=time_name, keep_attrs=True)

        # Add a single time coordinate for the day
        day_time = np.datetime64(f"{day_ymd[:4]}-{day_ymd[4:6]}-{day_ymd[6:8]}")
        ds_daily = ds_daily.expand_dims({time_name: [day_time]})

    return day_ymd, ds_daily


def subset_into_daily_file(src_file: Path, start_dt: datetime, end_dt: datetime, settings: Dict) -> Tuple[Optional[Path], str]:
    try:
        import xarray as xr
    except Exception as e:
        raise ImportError("xarray is required for subset/compile. Install it first.") from e

    if src_file.suffix.lower() not in {".nc", ".nc4"}:
        return None, "skipped"

    day_ymd = extract_yyyymmdd(src_file.name)
    daily_file = daily_subset_file_path(day_ymd, settings) if day_ymd else None
    source_daily_file = (
        daily_subset_file_path(day_ymd, settings, create_dir=False, kind="input")
        if day_ymd
        else None
    )
    overwrite_h = bool(settings.get("overwrite_h", False))

    if daily_file is not None and daily_file.exists():
        print(f"{day_ymd}: existing H output file found: {daily_file}")
        h_status = h_status_for_daily_file(daily_file, overwrite_h=overwrite_h)
        if h_status == "h_present":
            print(f"{day_ymd}: H already present; skipped.")
            return daily_file, "skipped_h_present"
        if h_status == "overwrite_h":
            print(f"{day_ymd}: H already present; re-downloading/replacing because --overwrite-h is enabled.")
        else:
            print(f"{day_ymd}: H missing; downloading/subsetting H only.")

    existing_daily_file = None
    if source_daily_file is not None and source_daily_file.exists():
        existing_daily_file = source_daily_file
    elif daily_file is not None and daily_file.exists():
        existing_daily_file = daily_file

    if existing_daily_file is not None:
        print(f"{day_ymd}: using existing daily subset as input: {existing_daily_file}")
        with xr.open_dataset(existing_daily_file) as existing:
            existing_loaded = existing.load()
            snapshot = existing_dataset_snapshot(existing_loaded)

        try:
            result = build_daily_subset_dataset(src_file, start_dt, end_dt, settings, ["H"])
            if result is None:
                existing_loaded.close()
                return None, "skipped"
            day_ymd, ds_h = result
        except RuntimeError as e:
            if "NetCDF: HDF error" not in str(e):
                existing_loaded.close()
                raise
            print(f"{day_ymd}: xarray H read failed ({e}); using scalar-level NetCDF fallback.")
            ds_h = build_h_daily_subset_dataset_fallback(src_file, existing_loaded)

        if "H" in existing_loaded.data_vars:
            existing_loaded = existing_loaded.drop_vars("H")
        merged = xr.merge([existing_loaded, ds_h[["H"]]], compat="override", combine_attrs="override")
        merged.attrs = snapshot["attrs"]

        try:
            atomic_write_daily_dataset(merged, daily_file, settings, existing_snapshot=snapshot)
        finally:
            merged.close()
            existing_loaded.close()
            ds_h.close()
        print(f"{day_ymd}: wrote H daily output: {daily_file}")
        return daily_file, "updated"

    if daily_file is not None:
        print(f"{day_ymd}: no existing source daily subset; creating full H daily output: {daily_file}")
    else:
        print(f"{src_file.name}: no existing source daily subset; creating full H daily output after reading source date.")

    variables = subset_variables_with_h(settings)
    result = build_daily_subset_dataset(src_file, start_dt, end_dt, settings, variables)
    if result is None:
        return None, "skipped"
    day_ymd, ds_daily = result
    daily_file = daily_subset_file_path(day_ymd, settings)

    try:
        atomic_write_daily_dataset(ds_daily, daily_file, settings)
    finally:
        ds_daily.close()
    print(f"{day_ymd}: created daily subset: {daily_file}")

    return daily_file, "created"


def process_downloaded_file(src_file: Path, start_dt: datetime, end_dt: datetime, settings: Dict) -> bool:
    try:
        daily_file, status = subset_into_daily_file(src_file, start_dt, end_dt, settings)
        if daily_file is None:
            print(f"Skipped post-process for {src_file.name} (outside date range or non-NetCDF).")
            return True

        if status in {"updated", "created", "skipped_h_present"} and settings.get("cleanup_downloaded_files", True) and src_file.exists():
            src_file.unlink()
            print(f"Removed original file: {src_file}")
        record_h_summary(settings, status)
        return True
    except Exception as e:
        record_h_summary(settings, "failed")
        print(f"Post-process failed for {src_file}: {e}")
        if src_file.exists() and not CONFIG.get("dry_run"):
            src_file.unlink()
            print(f"Removed unreadable original file so it can be re-downloaded: {src_file}")
        return False


def postprocess_redownload_attempts() -> int:
    subset_cfg = CONFIG.get("subset_compile", {})
    return max(0, int(subset_cfg.get("postprocess_redownload_attempts", 1)))


def write_combined_subset(start_dt: datetime, end_dt: datetime, settings: Dict) -> int:
    if not settings.get("write_combined_file", True):
        return 0

    try:
        import xarray as xr
    except Exception as e:
        raise ImportError("xarray is required for combined subset output. Install it first.") from e

    daily_files: List[Path] = []
    for d in date_range(start_dt, end_dt):
        p = daily_subset_file_path(d.strftime("%Y%m%d"), settings)
        if p.exists():
            daily_files.append(p)

    if not daily_files:
        print("No daily subset files found; skipping combined subset output.")
        return 0

    datasets = [xr.open_dataset(p) for p in daily_files]
    time_name = detect_coord_name(datasets[0], ["time", "TIME", "valid_time"], "time")
    combined = _merge_and_dedup_time(datasets, time_name).load()

    combined_out = combined_subset_file_path(start_dt, end_dt, settings)
    combined_out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_netcdf(combined_out)
    print(f"Wrote combined subset: {combined_out}")

    combined.close()
    for ds in datasets:
        ds.close()
    return 0


def compile_subsets(
    downloaded_files: List[Path],
    start_dt: datetime,
    end_dt: datetime,
    settings: Dict,
) -> int:
    errors = 0
    for fp in downloaded_files:
        ok = process_downloaded_file(fp, start_dt, end_dt, settings)
        if not ok:
            errors += 1

    if errors:
        return 1
    if settings.get("write_combined_file", False):
        return write_combined_subset(start_dt, end_dt, settings)
    return 0





def get_earthaccess_cfg(cfg: dict) -> dict:
    return cfg.get("earthaccess", cfg) if isinstance(cfg, dict) else {}


def earthaccess_login(earthaccess_module, cfg: dict):
    ea_cfg = get_earthaccess_cfg(cfg)
    strategies = ea_cfg.get("login_strategies", ["environment", "netrc"])
    persist = bool(ea_cfg.get("persist_login", False))
    if isinstance(strategies, str):
        strategies = [s.strip() for s in strategies.split(",") if s.strip()]
    if not strategies:
        strategies = ["environment", "netrc"]

    errors = []
    for strategy in strategies:
        try:
            return earthaccess_module.login(strategy=strategy, persist=persist)
        except OSError as e:
            errors.append(f"{strategy}: {e}")
            if strategy == "interactive":
                break
        except Exception as e:
            errors.append(f"{strategy}: {e}")

    raise RuntimeError(
        "Earthdata login failed without a usable interactive prompt. "
        "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD, set EARTHDATA_TOKEN, "
        "or configure ~/.netrc / NETRC for urs.earthdata.nasa.gov. "
        f"Tried strategies: {', '.join(strategies)}. Errors: {'; '.join(errors)}"
    )


def get_earthaccess_s3_client(cfg: dict):
    try:
        import earthaccess
    except Exception as e:
        raise ImportError("earthaccess is required for use_earthaccess_creds=True. Install it first.") from e

    # Login prompts for Earthdata credentials if not already configured.
    auth = earthaccess_login(earthaccess, cfg)

    daac = None
    if isinstance(cfg, dict):
        daac = cfg.get("earthaccess", {}).get("daac") or cfg.get("daac")

    creds = None

    # Try auth-bound credential helper if available
    try:
        if hasattr(auth, "get_s3_credentials"):
            creds = auth.get_s3_credentials(daac=daac) if daac else auth.get_s3_credentials()
    except Exception:
        creds = None

    # Fallback to module-level helper
    if creds is None:
        try:
            if daac:
                creds = earthaccess.get_s3_credentials(daac=daac)
            else:
                creds = earthaccess.get_s3_credentials()
        except Exception:
            creds = None

    # earthaccess may return different shapes depending on version.
    if isinstance(creds, dict) and "credentials" in creds:
        creds = creds["credentials"]
    if isinstance(creds, (list, tuple)) and creds:
        creds = creds[0]

    key_map = {
        "accessKeyId": "aws_access_key_id",
        "secretAccessKey": "aws_secret_access_key",
        "sessionToken": "aws_session_token",
        "access_key": "aws_access_key_id",
        "secret_key": "aws_secret_access_key",
        "token": "aws_session_token",
        "aws_access_key_id": "aws_access_key_id",
        "aws_secret_access_key": "aws_secret_access_key",
        "aws_session_token": "aws_session_token",
    }

    session_kwargs = {}
    if isinstance(creds, dict):
        for k, v in key_map.items():
            if k in creds:
                session_kwargs[v] = creds[k]

    region = None
    if isinstance(creds, dict):
        region = creds.get("region") or creds.get("regionName")

    if session_kwargs:
        session = boto3.session.Session(
            region_name=region or "us-west-2",
            **session_kwargs,
        )
        return session.client("s3")

    # Fallback: try earthaccess AWSSession if available
    try:
        if hasattr(earthaccess, "aws") and hasattr(earthaccess.aws, "AWSSession"):
            aws_sess = earthaccess.aws.AWSSession(daac=daac) if daac else earthaccess.aws.AWSSession()
            if hasattr(aws_sess, "get_session"):
                session = aws_sess.get_session()
            else:
                session = aws_sess.session
            return session.client("s3")
    except Exception:
        pass

    keys = list(creds.keys()) if isinstance(creds, dict) else [type(creds).__name__]
    raise KeyError(
        "Earthaccess did not return expected S3 credential keys. "
        f"Got keys: {keys}. If this persists, set AWS credentials via "
        "CONFIG['credentials_file'] or CONFIG['aws_profile'], or use mode='earthaccess' "
        "to download via HTTPS instead of direct S3."
    )





def get_boto3_session(cfg: dict) -> boto3.session.Session:
    # Optionally point boto3 to a local credentials file.
    creds_path = cfg.get("credentials_file")
    if creds_path:
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = creds_path

    profile = cfg.get("aws_profile")
    try:
        if profile:
            return boto3.session.Session(profile_name=profile)
        return boto3.session.Session()
    except botocore.exceptions.ProfileNotFound:
        # Fallback to default profile if the requested one is missing.
        return boto3.session.Session()


def check_region(require_us_west_2: bool) -> None:
    region = boto3.client("s3").meta.region_name
    if require_us_west_2 and region != "us-west-2":
        raise ValueError("This script is not running in us-west-2; direct S3 access may fail.")
    if region != "us-west-2":
        print(f"Warning: AWS region is {region}; NASA S3 access is typically in us-west-2.")


def run_earthaccess(
    cfg: dict,
    out_dir: Path,
    on_downloaded_file: Optional[Callable[[Path], bool]] = None,
) -> Tuple[int, List[Path]]:
    try:
        import earthaccess
    except Exception as e:
        raise ImportError("earthaccess is required for mode='earthaccess'. Install it first.") from e

    check_region(cfg.get("require_us_west_2", False))

    earthaccess_login(earthaccess, cfg)
    start = cfg.get("start_date") or cfg.get("temporal_start")
    end = cfg.get("end_date") or cfg.get("temporal_end")
    if not start or not end:
        raise ValueError("earthaccess.start_date/end_date (or temporal_start/temporal_end) are required")
    kwargs = {
        "temporal": (start, end),
    }
    if cfg.get("doi"):
        kwargs["doi"] = cfg["doi"]
    if cfg.get("short_name"):
        kwargs["short_name"] = cfg["short_name"]

    results = earthaccess.search_data(**kwargs)
    if cfg.get("max_results"):
        results = results[: cfg["max_results"]]

    if not results:
        print("No results found for the query.")
        return 2, []

    # Filter out granules whose output file already exists.
    # If a file already exists but has no corresponding daily subset, still post-process it.
    pending = []
    for granule in results:
        links = [l for l in granule.data_links() if isinstance(l, str)]
        filename = Path(links[0]).name if links else None
        subset_cfg = CONFIG.get("subset_compile", {})
        day_ymd = extract_yyyymmdd(filename or "")
        existing_path = find_existing_download_file(filename, out_dir)
        if on_downloaded_file is not None and day_ymd and not should_download_for_daily_h(day_ymd, subset_cfg):
            if (
                existing_path is not None
                and existing_path.exists()
                and subset_cfg.get("cleanup_stale_downloads", True)
                and not CONFIG.get("dry_run")
            ):
                existing_path.unlink()
                print(f"Removed stale original file: {existing_path}")
            continue
        if existing_path is not None:
            if on_downloaded_file is not None:
                if CONFIG.get("dry_run"):
                    print(f"Dry run: would post-process existing download for {filename}.")
                    continue
                print(f"Already downloaded: {filename}; checking daily H status ...")
                ok = on_downloaded_file(existing_path)
                if not ok:
                    print(f"Warning: post-processing failed for {filename}")
                    if not existing_path.exists():
                        pending.append(granule)
            else:
                print(f"Skipping (already exists): {filename}")
        else:
            pending.append(granule)

    if not pending:
        print("All granules already downloaded.")
        return 0, []

    if CONFIG.get("dry_run"):
        for r in pending:
            print(r)
        return 0, []

    # Download one granule at a time so each can be subsetted immediately.
    errors = 0
    all_downloaded: List[Path] = []
    for i, granule in enumerate(pending, 1):
        print(f"Downloading granule {i}/{len(pending)} ...")
        before = set(list_netcdf_files(out_dir))
        try:
            try:
                earthaccess.download([granule], path=str(out_dir))
            except TypeError:
                earthaccess.download([granule], local_path=str(out_dir))
        except Exception as e:
            print(f"Failed to download granule {i}: {e}")
            errors += 1
            continue

        after = set(list_netcdf_files(out_dir))
        new_files = sorted(after - before)
        if not new_files:
            print(f"Warning: no new files detected after downloading granule {i}.")
            continue

        all_downloaded.extend(new_files)
        if on_downloaded_file is not None:
            for path in new_files:
                ok = on_downloaded_file(path)
                for attempt in range(postprocess_redownload_attempts()):
                    if ok:
                        break
                    print(
                        f"Post-processing failed for {path.name}; "
                        f"re-downloading original file (attempt {attempt + 1}/{postprocess_redownload_attempts()}) ..."
                    )
                    before_retry = set(list_netcdf_files(out_dir))
                    try:
                        try:
                            earthaccess.download([granule], path=str(out_dir))
                        except TypeError:
                            earthaccess.download([granule], local_path=str(out_dir))
                    except Exception as e:
                        print(f"Failed to re-download {path.name}: {e}")
                        break
                    after_retry = set(list_netcdf_files(out_dir))
                    retry_files = sorted(after_retry - before_retry)
                    retry_path = retry_files[0] if retry_files else find_existing_download_file(path.name, out_dir)
                    if retry_path is None:
                        print(f"Warning: no file found after re-downloading {path.name}.")
                        break
                    if retry_path not in all_downloaded:
                        all_downloaded.append(retry_path)
                    ok = on_downloaded_file(retry_path)
                if not ok:
                    errors += 1

    return (1 if errors else 0), all_downloaded





def list_s3_urls(cfg: dict) -> list:
    try:
        import earthaccess
    except Exception as e:
        raise ImportError("earthaccess is required for list_s3_urls. Install it first.") from e

    earthaccess_login(earthaccess, cfg)
    start = cfg.get("start_date") or cfg.get("temporal_start")
    end = cfg.get("end_date") or cfg.get("temporal_end")
    if not start or not end:
        raise ValueError("earthaccess.start_date/end_date (or temporal_start/temporal_end) are required")
    kwargs = {
        "temporal": (start, end),
    }
    if cfg.get("doi"):
        kwargs["doi"] = cfg["doi"]
    if cfg.get("short_name"):
        kwargs["short_name"] = cfg["short_name"]

    results = earthaccess.search_data(**kwargs)
    if cfg.get("max_results"):
        results = results[: cfg["max_results"]]

    urls = []
    for g in results:
        for link in g.data_links():
            if isinstance(link, str) and link.startswith("s3://"):
                urls.append(link)
    return urls


def run_s3(
    cfg: dict,
    out_dir: Path,
    on_downloaded_file: Optional[Callable[[Path], bool]] = None,
) -> Tuple[int, List[Path]]:
    urls: List[str] = []
    if cfg.get("manifest"):
        urls = read_manifest(Path(cfg["manifest"]))
    elif cfg.get("urls"):
        urls = list(cfg["urls"])
    else:
        if not cfg.get("s3_bucket"):
            raise ValueError("CONFIG['s3_bucket'] must be set")
        urls = build_s3_urls(cfg["s3_bucket"], cfg["s3_prefix"], cfg["pattern"], cfg["start_date"], cfg["end_date"])

    if not urls:
        print("No URLs to download")
        return 2, []

    existing_paths: List[Path] = []
    if on_downloaded_file is None:
        missing_urls = []
        for url in urls:
            existing_path = find_existing_download_file(Path(url).name, out_dir)
            if existing_path is not None:
                print(f"Skipping download; original file already exists: {existing_path}")
                existing_paths.append(existing_path)
            else:
                missing_urls.append(url)
        urls = missing_urls

    if on_downloaded_file is not None:
        subset_cfg = CONFIG.get("subset_compile", {})
        filtered_urls = []
        for url in urls:
            day_ymd = extract_yyyymmdd(url)
            if should_download_for_daily_h(day_ymd, subset_cfg):
                existing_path = find_existing_download_file(Path(url).name, out_dir)
                if existing_path is not None:
                    if cfg.get("dry_run"):
                        print(f"Dry run: would post-process existing download for {existing_path.name}.")
                    else:
                        print(f"Already downloaded: {existing_path.name}; checking daily H status ...")
                        ok = on_downloaded_file(existing_path)
                        if not ok:
                            print(f"Warning: post-processing failed for {existing_path.name}")
                            filtered_urls.append(url)
                    continue
                filtered_urls.append(url)
            else:
                out_path = find_existing_download_file(Path(url).name, out_dir)
                if (
                    out_path is not None
                    and out_path.exists()
                    and subset_cfg.get("cleanup_stale_downloads", True)
                    and not cfg.get("dry_run")
                ):
                    out_path.unlink()
                    print(f"Removed stale original file: {out_path}")
        urls = filtered_urls

    if not urls:
        print("No downloads needed.")
        return 0, existing_paths

    if cfg.get("dry_run"):
        for u in urls:
            print(u)
        return 0, []

    if cfg.get("use_earthaccess_creds"):
        s3_client = get_earthaccess_s3_client(cfg)
    else:
        session = get_boto3_session(cfg)
        if cfg.get("anonymous"):
            s3_client = session.client("s3", config=botocore.client.Config(signature_version=botocore.UNSIGNED))
        else:
            s3_client = session.client("s3")

    def job(url: str) -> str:
        if url.startswith("s3://"):
            bucket, key = parse_s3_uri(url)
            out_path = out_dir / Path(key).name
            existing_path = find_existing_download_file(out_path.name, out_dir)
            if existing_path is not None:
                print(f"Skipping download; original file already exists: {existing_path}")
                return str(existing_path)
            download_s3(s3_client, bucket, key, out_path)
            return str(out_path)
        if url.startswith("http://") or url.startswith("https://"):
            out_path = out_dir / Path(url).name
            existing_path = find_existing_download_file(out_path.name, out_dir)
            if existing_path is not None:
                print(f"Skipping download; original file already exists: {existing_path}")
                return str(existing_path)
            download_http(url, out_path)
            return str(out_path)
        raise ValueError(f"Unsupported URL: {url}")

    errors = 0
    downloaded_paths: List[Path] = list(existing_paths)
    with futures.ThreadPoolExecutor(max_workers=cfg["max_workers"]) as ex:
        futs = {ex.submit(job, u): u for u in urls}
        for f in futures.as_completed(futs):
            u = futs[f]
            try:
                path = Path(f.result())
                downloaded_paths.append(path)
                print(f"Downloaded: {u} -> {path}")
                if on_downloaded_file is not None:
                    ok = on_downloaded_file(path)
                    for attempt in range(postprocess_redownload_attempts()):
                        if ok:
                            break
                        print(
                            f"Post-processing failed for {path.name}; "
                            f"re-downloading original file (attempt {attempt + 1}/{postprocess_redownload_attempts()}) ..."
                        )
                        try:
                            retry_path = Path(job(u))
                        except Exception as e:
                            print(f"Failed to re-download {path.name}: {e}")
                            break
                        if retry_path not in downloaded_paths:
                            downloaded_paths.append(retry_path)
                        ok = on_downloaded_file(retry_path)
                    if not ok:
                        errors += 1
            except Exception as e:
                errors += 1
                print(f"Failed: {u} ({e})")

    return (1 if errors else 0), downloaded_paths


def apply_cli_args() -> None:
    parser = argparse.ArgumentParser(description="Download and subset MERRA-2 daily files.")
    parser.add_argument(
        "--overwrite-h",
        action="store_true",
        help="Re-download and replace H even when H already exists in a daily subset file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads/updates without modifying files.",
    )
    parser.add_argument(
        "--earthdata-login-strategies",
        help=(
            "Comma-separated earthaccess login strategies. Default is environment,netrc "
            "to avoid broken interactive prompts in batch runs."
        ),
    )
    args = parser.parse_args()

    subset_cfg = CONFIG.setdefault("subset_compile", {})
    if args.overwrite_h:
        subset_cfg["overwrite_h"] = True
    if args.dry_run:
        CONFIG["dry_run"] = True
    if args.earthdata_login_strategies:
        CONFIG.setdefault("earthaccess", {})["login_strategies"] = [
            s.strip() for s in args.earthdata_login_strategies.split(",") if s.strip()
        ]


def main() -> int:
    apply_cli_args()

    out_dir = Path(CONFIG["output_dir"])
    ensure_output_dir(out_dir)

    mode = CONFIG.get("mode", "s3")
    if mode == "earthaccess":
        subset_cfg = CONFIG.get("subset_compile", {})
        subset_cfg["_h_summary"] = {"skipped": 0, "updated": 0, "created": 0, "failed": 0}
        start_dt, end_dt = get_date_bounds(mode, CONFIG)
        on_downloaded = None
        if subset_cfg.get("enabled", False):
            on_downloaded = lambda p: process_downloaded_file(p, start_dt, end_dt, subset_cfg)
        rc, _downloaded = run_earthaccess(CONFIG["earthaccess"], out_dir, on_downloaded_file=on_downloaded)
        if rc != 0:
            print_h_summary(subset_cfg)
            return rc
        if subset_cfg.get("enabled", False) and subset_cfg.get("write_combined_file", False):
            rc = write_combined_subset(start_dt, end_dt, subset_cfg)
            print_h_summary(subset_cfg)
            return rc
        print_h_summary(subset_cfg)
        return 0
    if mode == "list_s3_urls":
        urls = list_s3_urls(CONFIG["earthaccess"])
        if not urls:
            print("No S3 URLs found.")
            return 2
        list_path = Path(CONFIG["list_output"])
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text("\n".join(urls) + "\n")
        print(f"Wrote {len(urls)} S3 URLs to {list_path}")
        return 0
    if mode == "s3":
        subset_cfg = CONFIG.get("subset_compile", {})
        subset_cfg["_h_summary"] = {"skipped": 0, "updated": 0, "created": 0, "failed": 0}
        start_dt, end_dt = get_date_bounds(mode, CONFIG)
        on_downloaded = None
        if subset_cfg.get("enabled", False):
            on_downloaded = lambda p: process_downloaded_file(p, start_dt, end_dt, subset_cfg)
        rc, _downloaded = run_s3(CONFIG, out_dir, on_downloaded_file=on_downloaded)
        if rc != 0:
            print_h_summary(subset_cfg)
            return rc
        if subset_cfg.get("enabled", False) and subset_cfg.get("write_combined_file", False):
            rc = write_combined_subset(start_dt, end_dt, subset_cfg)
            print_h_summary(subset_cfg)
            return rc
        print_h_summary(subset_cfg)
        return 0
    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    raise SystemExit(main())
