#!/usr/bin/env python3
"""Download and subset NOAA PSL NARR NetCDF files.

Examples
--------
Download daily 2 m air temperature for 1995-2025 and subset with YAML:

    python download_narr.py --config narr_subset.yaml

Download 3-hourly pressure-level air temperature for January-March 2020:

    python download_narr.py --category pressure --variables air \
        --start 2020-01-01 --end 2020-03-31 --out-dir data/NARR

Download several variables with 4 parallel workers:

    python download_narr.py --category pressure --variables air hgt shum uwnd vwnd \
        --start 2020-01-01 --end 2020-12-31 --workers 4 --out-dir data/NARR

The default source is NOAA PSL THREDDS HTTP fileServer:
https://psl.noaa.gov/thredds/catalog/Datasets/NARR/catalog.html
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, NamedTuple


DEFAULT_BASE_URL = "https://psl.noaa.gov/thredds/fileServer/Datasets/NARR"
CHUNK_SIZE = 1024 * 1024
DAILY_ROOT = "Dailies"


class DownloadJob(NamedTuple):
    url: str
    destination: Path
    subset_destination: Path | None = None


class RemoteInfo(NamedTuple):
    exists: bool
    size: int | None
    accepts_ranges: bool


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r}; expected YYYY-MM-DD"
        ) from exc


def parse_config_date(value: Any, name: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return parse_date(value)
    raise ValueError(f"{name} must be a YYYY-MM-DD date")


def iter_months(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    if end < start:
        raise ValueError("--end must be on or after --start")

    year = start.year
    month = start.month
    while (year, month) <= (end.year, end.month):
        yield dt.date(year, month, 1)
        month += 1
        if month == 13:
            year += 1
            month = 1


def iter_years(start: dt.date, end: dt.date) -> Iterable[int]:
    if end < start:
        raise ValueError("--end must be on or after --start")
    yield from range(start.year, end.year + 1)


def infer_file_period(category: str, daily: bool, requested: str) -> str:
    if requested != "auto":
        return requested
    if not daily:
        return "monthly"
    return "yearly" if category == "monolevel" else "monthly"


def build_jobs(
    base_url: str,
    category: str,
    variables: list[str],
    start: dt.date,
    end: dt.date,
    out_dir: Path,
    flat: bool,
    daily: bool,
    file_period: str,
    subset_out_dir: Path | None = None,
) -> list[DownloadJob]:
    base = base_url.rstrip("/")
    jobs: list[DownloadJob] = []
    path_parts = [base]
    if daily:
        path_parts.append(DAILY_ROOT)
    path_parts.append(category)
    url_dir = "/".join(path_parts)

    for variable in variables:
        if file_period == "monthly":
            periods = [(f"{month:%Y%m}", month.year) for month in iter_months(start, end)]
        elif file_period == "yearly":
            periods = [(str(year), year) for year in iter_years(start, end)]
        else:
            raise ValueError(f"unsupported file period: {file_period}")

        for period, year in periods:
            filename = f"{variable}.{period}.nc"
            url = f"{url_dir}/{filename}"
            if flat:
                destination = out_dir / filename
            else:
                destination = out_dir / variable / filename
            subset_destination = None
            if subset_out_dir is not None:
                subset_destination = subset_out_dir / variable / filename
            jobs.append(
                DownloadJob(
                    url=url,
                    destination=destination,
                    subset_destination=subset_destination,
                )
            )

    return jobs


def load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required for --config. Install it with: python3 -m pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("YAML config must be a mapping at the top level")
    return config


def config_get(config: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    section_value = config.get(section, {})
    if not isinstance(section_value, dict):
        raise ValueError(f"{section} must be a mapping")
    return section_value.get(key, default)


def request_with_retries(
    request: urllib.request.Request,
    retries: int,
    timeout: int,
) -> urllib.response.addinfourl:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc

        if attempt < retries:
            time.sleep(min(30, 2**attempt))

    assert last_error is not None
    raise last_error


def remote_info(url: str, retries: int, timeout: int) -> RemoteInfo:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with request_with_retries(request, retries=retries, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            ranges = response.headers.get("Accept-Ranges", "").lower()
            return RemoteInfo(
                exists=True,
                size=int(length) if length and length.isdigit() else None,
                accepts_ranges="bytes" in ranges,
            )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return RemoteInfo(exists=False, size=None, accepts_ranges=False)
        raise


def should_skip(destination: Path, info: RemoteInfo, overwrite: bool) -> bool:
    if overwrite or not destination.exists():
        return False
    if info.size is None:
        return destination.stat().st_size > 0
    return destination.stat().st_size == info.size


def download_one(
    job: DownloadJob,
    retries: int,
    timeout: int,
    overwrite: bool,
    check_head: bool,
) -> tuple[str, str]:
    job.destination.parent.mkdir(parents=True, exist_ok=True)

    info = RemoteInfo(exists=True, size=None, accepts_ranges=False)
    if check_head:
        info = remote_info(job.url, retries=retries, timeout=timeout)
        if not info.exists:
            return ("missing", f"{job.url}")
        if should_skip(job.destination, info, overwrite):
            return ("skipped", f"{job.destination}")

    part_path = job.destination.with_suffix(job.destination.suffix + ".part")

    if overwrite:
        with contextlib.suppress(FileNotFoundError):
            part_path.unlink()

    for attempt in range(retries + 1):
        headers = {"User-Agent": "narr-downloader/1.0"}
        mode = "wb"
        resume_from = 0

        if part_path.exists() and part_path.stat().st_size > 0 and info.accepts_ranges:
            resume_from = part_path.stat().st_size
            headers["Range"] = f"bytes={resume_from}-"
            mode = "ab"

        request = urllib.request.Request(job.url, headers=headers)
        with request_with_retries(request, retries=retries, timeout=timeout) as response:
            if resume_from and getattr(response, "status", None) != 206:
                mode = "wb"
            with part_path.open(mode) as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)

        if info.size is None:
            break

        current_size = part_path.stat().st_size
        if current_size == info.size:
            break

        if current_size > info.size:
            part_path.unlink()

        if attempt < retries:
            time.sleep(min(30, 2**attempt))
    else:
        current_size = part_path.stat().st_size if part_path.exists() else 0
        raise RuntimeError(
            f"incomplete download for {job.url}: "
            f"got {current_size} bytes, expected {info.size}"
        )

    os.replace(part_path, job.destination)
    return ("downloaded", f"{job.destination}")


def subset_one(
    source: Path,
    destination: Path,
    start: dt.date,
    end: dt.date,
    subset: dict[str, Any],
    overwrite: bool,
) -> tuple[str, str]:
    if destination.exists() and not overwrite:
        return ("subset-skipped", f"{destination}")

    try:
        import xarray as xr
    except ImportError as exc:
        raise SystemExit(
            "xarray is required for subsetting. Install dependencies with: "
            "python3 -m pip install xarray netcdf4 pyyaml"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(source) as dataset:
        ds = dataset

        if "time" in ds.coords or "time" in ds.dims:
            ds = ds.sel(time=slice(start.isoformat(), end.isoformat()))

        variables = subset.get("variables")
        if variables:
            keep_vars = [name for name in variables if name in ds.data_vars]
            missing = sorted(set(variables) - set(keep_vars))
            if missing:
                print(f"warning: {source} missing variables: {', '.join(missing)}")
            if keep_vars:
                ds = ds[keep_vars]

        levels = subset.get("levels")
        if levels and "level" in ds.coords:
            ds = ds.sel(level=levels)

        bbox = subset.get("bbox")
        if bbox:
            ds = apply_bbox(ds, bbox)

        encoding = {
            name: {"zlib": True, "complevel": 4}
            for name in ds.data_vars
            if ds[name].ndim > 0
        }
        tmp = destination.with_suffix(destination.suffix + ".part")
        ds.to_netcdf(tmp, encoding=encoding)

    os.replace(tmp, destination)
    return ("subset", f"{destination}")


def apply_bbox(dataset: Any, bbox: dict[str, Any]) -> Any:
    import numpy as np

    required = {"lat_min", "lat_max", "lon_min", "lon_max"}
    missing = required - set(bbox)
    if missing:
        raise ValueError(f"subset.bbox missing keys: {', '.join(sorted(missing))}")

    lat_min = float(bbox["lat_min"])
    lat_max = float(bbox["lat_max"])
    lon_min = float(bbox["lon_min"])
    lon_max = float(bbox["lon_max"])

    lat_name = "lat" if "lat" in dataset.coords else "latitude"
    lon_name = "lon" if "lon" in dataset.coords else "longitude"
    if lat_name not in dataset.coords or lon_name not in dataset.coords:
        raise ValueError("dataset does not contain lat/lon coordinates")

    lat = dataset[lat_name]
    lon = dataset[lon_name]
    if lon_max <= 180 and float(lon.max()) > 180:
        lon_min = lon_min % 360
        lon_max = lon_max % 360
    elif lon_min >= 0 and float(lon.min()) < 0:
        lon_min = ((lon_min + 180) % 360) - 180
        lon_max = ((lon_max + 180) % 360) - 180

    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    if lon_min <= lon_max:
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
    else:
        lon_mask = (lon >= lon_min) | (lon <= lon_max)

    mask = lat_mask & lon_mask
    if not bool(mask.any()):
        raise ValueError("bbox does not overlap dataset grid")

    dims = list(mask.dims)
    indexers = {}
    for axis, dim in enumerate(dims):
        reduce_axes = tuple(i for i in range(mask.ndim) if i != axis)
        valid = np.asarray(mask.values).any(axis=reduce_axes)
        positions = np.flatnonzero(valid)
        indexers[dim] = slice(int(positions[0]), int(positions[-1]) + 1)

    subset = dataset.isel(indexers)
    return subset.where(mask.isel(indexers), drop=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and optionally subset NOAA PSL NARR NetCDF files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML config file. Values in the config override CLI defaults.",
    )
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Download daily NARR files from Datasets/NARR/Dailies.",
    )
    parser.add_argument(
        "--category",
        default="pressure",
        help="NARR folder/category, e.g. pressure or monolevel. Default: pressure",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Variable names as used by PSL filenames, e.g. air hgt shum uwnd vwnd.",
    )
    parser.add_argument("--start", type=parse_date, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", type=parse_date, help="End date YYYY-MM-DD.")
    parser.add_argument(
        "--file-period",
        choices=["auto", "monthly", "yearly"],
        default="auto",
        help=(
            "Remote file cadence. Daily monolevel is yearly; daily pressure is monthly. "
            "Default: auto"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/NARR"),
        help="Output directory. Default: data/NARR",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL for NARR files. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel downloads. Use modest values for large files. Default: 1",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries for transient network errors. Default: 3",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Socket timeout in seconds. Default: 120",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite complete existing files.",
    )
    parser.add_argument(
        "--no-head",
        action="store_true",
        help="Skip pre-download HEAD checks. Useful if HEAD is blocked or slow.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Save all files directly in --out-dir instead of category/variable subfolders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs and destinations without downloading.",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Subset downloaded files. Usually configured through --config.",
    )
    parser.add_argument(
        "--subset-out-dir",
        type=Path,
        default=Path("data/NARR_subset"),
        help="Subset output directory. Default: data/NARR_subset",
    )
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml_config(args.config) if args.config else {}

    start = config_get(config, "download", "start", args.start)
    end = config_get(config, "download", "end", args.end)
    if start is None or end is None:
        raise SystemExit("--start and --end are required unless provided in --config")

    variables = config_get(config, "download", "variables", args.variables)
    if not variables:
        raise SystemExit("--variables are required unless provided in --config")
    if isinstance(variables, str):
        variables = [variables]

    daily = bool(config_get(config, "download", "daily", args.daily))
    category = str(config_get(config, "download", "category", args.category))
    file_period = infer_file_period(
        category=category,
        daily=daily,
        requested=str(config_get(config, "download", "file_period", args.file_period)),
    )
    if file_period == "daily":
        raise SystemExit(
            "download.file_period controls the remote file cadence and must be "
            "'monthly', 'yearly', or 'auto'. For daily mean NARR pressure files, "
            "set download.daily: true and download.file_period: monthly."
        )

    subset_config = config.get("subset", {})
    if subset_config is None:
        subset_config = {}
    if not isinstance(subset_config, dict):
        raise ValueError("subset must be a mapping")

    subset_enabled = bool(config_get(config, "subset", "enabled", args.subset))
    subset_out_dir = Path(
        config_get(config, "subset", "out_dir", args.subset_out_dir)
    )

    return {
        "base_url": str(config_get(config, "download", "base_url", args.base_url)),
        "category": category,
        "variables": [str(variable) for variable in variables],
        "start": parse_config_date(start, "download.start"),
        "end": parse_config_date(end, "download.end"),
        "out_dir": Path(config_get(config, "download", "out_dir", args.out_dir)),
        "flat": bool(config_get(config, "download", "flat", args.flat)),
        "daily": daily,
        "file_period": file_period,
        "workers": int(config_get(config, "download", "workers", args.workers)),
        "retries": int(config_get(config, "download", "retries", args.retries)),
        "timeout": int(config_get(config, "download", "timeout", args.timeout)),
        "overwrite": bool(config_get(config, "download", "overwrite", args.overwrite)),
        "check_head": not bool(config_get(config, "download", "no_head", args.no_head)),
        "dry_run": bool(config_get(config, "download", "dry_run", args.dry_run)),
        "subset_enabled": subset_enabled,
        "subset_out_dir": subset_out_dir if subset_enabled else None,
        "remove_raw_after_subset": bool(
            config_get(config, "subset", "remove_raw_after_subset", False)
        ),
        "subset": subset_config,
    }


def main() -> int:
    args = parse_args()
    settings = resolve_settings(args)
    jobs = build_jobs(
        base_url=settings["base_url"],
        category=settings["category"],
        variables=settings["variables"],
        start=settings["start"],
        end=settings["end"],
        out_dir=settings["out_dir"],
        flat=settings["flat"],
        daily=settings["daily"],
        file_period=settings["file_period"],
        subset_out_dir=settings["subset_out_dir"],
    )

    if settings["dry_run"]:
        for job in jobs:
            print(f"{job.url} -> {job.destination}")
            if job.subset_destination is not None:
                print(f"  subset -> {job.subset_destination}")
        return 0

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, settings["workers"])) as pool:
        job_iter = iter(jobs)
        active: dict[concurrent.futures.Future[tuple[str, str]], DownloadJob] = {}

        def submit_next_job() -> bool:
            try:
                job = next(job_iter)
            except StopIteration:
                return False

            future = pool.submit(
                download_one,
                job,
                settings["retries"],
                settings["timeout"],
                settings["overwrite"],
                settings["check_head"],
            )
            active[future] = job
            return True

        for _ in range(max(1, settings["workers"])):
            if not submit_next_job():
                break

        while active:
            future = next(concurrent.futures.as_completed(active))
            job = active.pop(future)
            try:
                status, message = future.result()
                print(f"{status}: {message}")
                if (
                    settings["subset_enabled"]
                    and status in {"downloaded", "skipped"}
                    and job.subset_destination is not None
                ):
                    subset_status, subset_message = subset_one(
                        source=job.destination,
                        destination=job.subset_destination,
                        start=settings["start"],
                        end=settings["end"],
                        subset=settings["subset"],
                        overwrite=settings["overwrite"],
                    )
                    print(f"{subset_status}: {subset_message}")
                    if (
                        subset_status == "subset"
                        and settings["remove_raw_after_subset"]
                    ):
                        job.destination.unlink()
                        print(f"removed-raw: {job.destination}")
            except Exception as exc:
                failures += 1
                print(f"failed: {job.url} ({exc})", file=sys.stderr)
            submit_next_job()

    if failures:
        print(f"{failures} operation(s) failed.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
