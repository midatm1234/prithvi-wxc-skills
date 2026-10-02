#!/usr/bin/env python3
"""Download PRISM daily 800 m AN grids (CLI version of download_prism_daily_800m.ipynb).

The configuration and download functions below are copied verbatim from the
notebook so both produce identical files. The CLI overrides the configuration
globals and exits non-zero if any file fails, so the MCP job reports failure.

    python download_prism.py --start 1996-01-01 --end 2015-12-31 \
        --variables ppt,tmax,tmin --output-dir ~/prithvi-wxc-data/prism/prism_daily_800m_an
"""

# ---- notebook cell 1: configuration -----------------------------------------
from pathlib import Path
import os

# -----------------------------------------------------------------------------
# PRISM request parameters
# -----------------------------------------------------------------------------
# TIME_PERIOD options from the PRISM PDF: "daily", "monthly", "annual".
# START_DATE and END_DATE may be full ISO dates. For monthly downloads the month
# is used; for annual downloads the year is used.
START_DATE = "1996-01-03"
END_DATE = "2025-12-31"
TIME_PERIOD = "daily"

# Variable options from the PRISM PDF:
# "ppt", "tmin", "tmax", "tmean", "tdmean", "vpdmin", "vpdmax".
VARIABLES = ("ppt", "tmax", "tmin")

# Region options from the PRISM PDF: "us", "ak", "hi", "pr".
# Only "us" is currently implemented for the data service.
REGION = "us"

# Resolution options: "800m", "4km".

RESOLUTION = "800m"

# Dataset options: "an" for all-networks, or "lt" for long-term monthly 800 m.
# The PDF says the web service defaults to AN, and LT is only for 800 m monthly.
DATASET = "an"

# SOURCE_MODE options:
# "directory"   -> https://data.prism.oregonstate.edu/time_series/...
# "web_service" -> https://services.nacse.org/prism/data/get/...
SOURCE_MODE = "web_service"

# Web-service-only output format options from the PDF:
# None for default COG zip, or one of "nc", "asc", "bil".
DATA_FORMAT = "nc"

OUTPUT_DIR = Path(f"prism_{TIME_PERIOD}_{RESOLUTION}_{DATASET}")

# -----------------------------------------------------------------------------
# Downloader behavior
# -----------------------------------------------------------------------------
# Normal notebook runs download data. For smoke tests, launch with PRISM_RUN_DOWNLOAD=false.
RUN_DOWNLOAD = os.getenv("PRISM_RUN_DOWNLOAD", "true").strip().lower() in {"1", "true", "yes", "y"}

# PRISM's sample bulk script waits 2 seconds between requests.
REQUEST_DELAY_SECONDS = 2.0
TIMEOUT = (20, 600)  # connect timeout, read timeout in seconds
CHUNK_SIZE = 1024 * 1024
MAX_RETRIES = 3
STOP_ON_ERROR = False

# Existing files created by this notebook should be complete because downloads finish as .part files first.
# Turn this on to do a HEAD request and compare Content-Length before skipping existing zip files.
CHECK_REMOTE_SIZE_FOR_EXISTING = False
VERIFY_ZIP_AFTER_DOWNLOAD = False

# Extract each downloaded zip package after a successful transfer.
# With DATA_FORMAT = "nc", this keeps only the .nc file directly under the year folder.
# Sidecar files such as .txt, .xml, .stx, .csv, .prj, .aux.xml, and the .zip are removed.
EXTRACT_AFTER_DOWNLOAD = True
DELETE_ZIP_AFTER_EXTRACT = True
EXTRACT_DATA_FILE_ONLY = True
CLEANUP_OLD_EXTRACTED_FOLDERS = True

DIRECTORY_ROOT_URL = "https://data.prism.oregonstate.edu/time_series"
WEB_SERVICE_BASE_URL = "https://services.nacse.org/prism/data/get"
USER_AGENT = "PRISM bulk downloader notebook"


# ---- notebook cell 2: download functions ------------------------------------
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import calendar
import shutil
import time
import zipfile

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

try:
    import pandas as pd
except ImportError:
    pd = None

FILE_RESOLUTION_TOKEN = {
    "400m": "15s",
    "800m": "30s",
    "4km": "25m",
}

DATA_FILE_SUFFIX_BY_FORMAT = {
    None: ".tif",
    "nc": ".nc",
    "asc": ".asc",
    "bil": ".bil",
}

VALID_VARIABLES = {"ppt", "tmin", "tmax", "tmean", "tdmean", "vpdmin", "vpdmax"}
VALID_REGIONS = {"us", "ak", "hi", "pr"}
VALID_RESOLUTIONS = {"400m", "800m", "4km"}
VALID_TIME_PERIODS = {"daily", "monthly", "annual"}
VALID_DATASETS = {"an", "lt"}
VALID_DATA_FORMATS = {None, "nc", "asc", "bil"}


@dataclass(frozen=True)
class PrismItem:
    variable: str
    period_start: date
    date_token: str
    label: str
    url: str
    path: Path


def parse_config_date(value):
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(text, fmt).date()
            if fmt == "%Y-%m":
                return parsed.replace(day=1)
            if fmt == "%Y":
                return parsed.replace(month=1, day=1)
            return parsed
        except ValueError:
            pass
    raise ValueError(f"Could not parse date {value!r}; use YYYY-MM-DD, YYYY-MM, or YYYY")


def add_months(day, months):
    month_index = day.month - 1 + months
    year = day.year + month_index // 12
    month = month_index % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return day.replace(year=year, month=month, day=min(day.day, last_day))


def iter_period_starts(start_value, end_value):
    start = parse_config_date(start_value)
    end = parse_config_date(end_value)
    if start > end:
        raise ValueError("START_DATE must be on or before END_DATE")

    if TIME_PERIOD == "daily":
        current = start
        while current <= end:
            yield current
            current += timedelta(days=1)
    elif TIME_PERIOD == "monthly":
        current = start.replace(day=1)
        stop = end.replace(day=1)
        while current <= stop:
            yield current
            current = add_months(current, 1)
    elif TIME_PERIOD == "annual":
        current = start.replace(month=1, day=1)
        stop = end.replace(month=1, day=1)
        while current <= stop:
            yield current
            current = current.replace(year=current.year + 1)
    else:
        raise ValueError(f"Unsupported TIME_PERIOD: {TIME_PERIOD}")


def date_token_for(period_start):
    if TIME_PERIOD == "daily":
        return f"{period_start:%Y%m%d}"
    if TIME_PERIOD == "monthly":
        return f"{period_start:%Y%m}"
    if TIME_PERIOD == "annual":
        return f"{period_start:%Y}"
    raise ValueError(f"Unsupported TIME_PERIOD: {TIME_PERIOD}")


def label_for(period_start):
    if TIME_PERIOD == "daily":
        return f"{period_start:%Y-%m-%d}"
    if TIME_PERIOD == "monthly":
        return f"{period_start:%Y-%m}"
    if TIME_PERIOD == "annual":
        return f"{period_start:%Y}"
    raise ValueError(f"Unsupported TIME_PERIOD: {TIME_PERIOD}")


def filename_for(variable, date_token):
    token = FILE_RESOLUTION_TOKEN[RESOLUTION]
    return f"prism_{variable}_{REGION}_{token}_{date_token}.zip"


def data_file_suffix():
    try:
        return DATA_FILE_SUFFIX_BY_FORMAT[DATA_FORMAT]
    except KeyError as exc:
        raise ValueError(f"No data-file suffix is configured for DATA_FORMAT: {DATA_FORMAT}") from exc


def data_file_path_for(item):
    return item.path.with_suffix(data_file_suffix())


def old_extraction_dir_for(item):
    return item.path.with_suffix("")


def directory_period_path():
    # In the static 800 m tree, annual year zip files are stored under monthly/{year}/.
    if TIME_PERIOD == "annual":
        return "monthly"
    return TIME_PERIOD


def url_for(variable, period_start, date_token):
    if SOURCE_MODE == "directory":
        if DATA_FORMAT is not None:
            raise ValueError("DATA_FORMAT is only supported with SOURCE_MODE = 'web_service'")
        directory_base = f"{DIRECTORY_ROOT_URL.rstrip('/')}/{REGION}/{DATASET}/{RESOLUTION}"
        return f"{directory_base}/{variable}/{directory_period_path()}/{period_start:%Y}/{filename_for(variable, date_token)}"

    if SOURCE_MODE == "web_service":
        url = f"{WEB_SERVICE_BASE_URL.rstrip('/')}/{REGION}/{RESOLUTION}/{variable}/{date_token}"
        if DATASET == "lt":
            url += "/lt"
        if DATA_FORMAT is not None:
            url += f"?format={DATA_FORMAT}"
        return url

    raise ValueError('SOURCE_MODE must be either "directory" or "web_service"')


def path_for(variable, period_start, date_token):
    return OUTPUT_DIR / variable / f"{period_start:%Y}" / filename_for(variable, date_token)


def validate_config():
    unknown = set(VARIABLES) - VALID_VARIABLES
    if unknown:
        raise ValueError(f"Unknown PRISM variables: {sorted(unknown)}")
    if REGION not in VALID_REGIONS:
        raise ValueError(f"Unsupported REGION: {REGION}")
    if RESOLUTION not in VALID_RESOLUTIONS:
        raise ValueError(f"Unsupported RESOLUTION: {RESOLUTION}")
    if RESOLUTION not in FILE_RESOLUTION_TOKEN:
        raise ValueError(f"No filename token is configured for RESOLUTION: {RESOLUTION}")
    if TIME_PERIOD not in VALID_TIME_PERIODS:
        raise ValueError(f"Unsupported TIME_PERIOD: {TIME_PERIOD}")
    if DATASET not in VALID_DATASETS:
        raise ValueError(f"Unsupported DATASET: {DATASET}")
    if DATA_FORMAT not in VALID_DATA_FORMATS:
        raise ValueError(f"Unsupported DATA_FORMAT: {DATA_FORMAT}")
    if DATASET == "lt" and not (RESOLUTION == "800m" and TIME_PERIOD == "monthly"):
        raise ValueError("The PRISM PDF documents LT only for 800 m monthly data")
    if SOURCE_MODE == "directory" and DATA_FORMAT is not None:
        raise ValueError("DATA_FORMAT is only supported with SOURCE_MODE = 'web_service'")
    if DELETE_ZIP_AFTER_EXTRACT and not EXTRACT_AFTER_DOWNLOAD:
        raise ValueError("DELETE_ZIP_AFTER_EXTRACT requires EXTRACT_AFTER_DOWNLOAD = True")
    if EXTRACT_DATA_FILE_ONLY and DATA_FORMAT not in DATA_FILE_SUFFIX_BY_FORMAT:
        raise ValueError(f"EXTRACT_DATA_FILE_ONLY does not know how to handle DATA_FORMAT = {DATA_FORMAT!r}")


def build_manifest():
    validate_config()
    items = []
    for variable in VARIABLES:
        for period_start in iter_period_starts(START_DATE, END_DATE):
            date_token = date_token_for(period_start)
            items.append(
                PrismItem(
                    variable=variable,
                    period_start=period_start,
                    date_token=date_token,
                    label=label_for(period_start),
                    url=url_for(variable, period_start, date_token),
                    path=path_for(variable, period_start, date_token),
                )
            )
    return items


def build_session():
    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def remote_content_length(session, item):
    response = session.head(item.url, allow_redirects=True, timeout=TIMEOUT)
    response.raise_for_status()
    length = response.headers.get("Content-Length")
    return int(length) if length else None


def verify_zip(path):
    with zipfile.ZipFile(path) as zf:
        bad_member = zf.testzip()
    if bad_member:
        raise IOError(f"Zip verification failed at member {bad_member!r} in {path}")


def data_file_exists(item):
    path = data_file_path_for(item)
    return path.exists() and path.stat().st_size > 0


def cleanup_non_data_outputs(item):
    if not CLEANUP_OLD_EXTRACTED_FOLDERS or not EXTRACT_DATA_FILE_ONLY:
        return
    keep_path = data_file_path_for(item).resolve()
    for candidate in item.path.parent.glob(f"{item.path.stem}*"):
        resolved = candidate.resolve()
        if resolved == keep_path:
            continue
        if candidate.is_dir():
            if (candidate / ".extract_complete").exists():
                shutil.rmtree(candidate)
        elif candidate.is_file():
            candidate.unlink()


def find_data_member(zf, item):
    target_name = data_file_path_for(item).name
    suffix = data_file_suffix().lower()
    members = [m for m in zf.infolist() if not m.is_dir()]
    exact = [m for m in members if Path(m.filename).name == target_name]
    if exact:
        return exact[0]
    suffix_matches = [m for m in members if Path(m.filename).name.lower().endswith(suffix)]
    if not suffix_matches:
        names = ", ".join(Path(m.filename).name for m in members[:10])
        raise IOError(f"No {suffix} data file found in {item.path}; first members: {names}")
    if len(suffix_matches) > 1:
        names = ", ".join(Path(m.filename).name for m in suffix_matches)
        raise IOError(f"Expected one {suffix} file in {item.path}, found: {names}")
    return suffix_matches[0]


def extract_data_file_only(item, archive_path=None):
    archive_path = Path(archive_path) if archive_path is not None else item.path
    target_path = data_file_path_for(item)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = target_path.with_name(target_path.name + ".part")
    if part_path.exists():
        part_path.unlink()

    try:
        with zipfile.ZipFile(archive_path) as zf:
            bad_member = zf.testzip()
            if bad_member:
                raise IOError(f"Zip verification failed at member {bad_member!r} in {archive_path}")
            member = find_data_member(zf, item)
            with zf.open(member) as src, part_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=CHUNK_SIZE)
        part_path.replace(target_path)
    except Exception:
        if part_path.exists():
            part_path.unlink()
        raise


def extract_archive(item, archive_path=None):
    archive_path = Path(archive_path) if archive_path is not None else item.path
    if EXTRACT_DATA_FILE_ONLY:
        extract_data_file_only(item, archive_path)
    else:
        extract_dir = old_extraction_dir_for(item)
        extract_dir.mkdir(parents=True, exist_ok=True)
        root = extract_dir.resolve()
        with zipfile.ZipFile(archive_path) as zf:
            bad_member = zf.testzip()
            if bad_member:
                raise IOError(f"Zip verification failed at member {bad_member!r} in {archive_path}")
            for member in zf.infolist():
                target = (root / member.filename).resolve()
                if target != root and root not in target.parents:
                    raise IOError(f"Refusing to extract unsafe zip member {member.filename!r}")
            zf.extractall(root)

    if DELETE_ZIP_AFTER_EXTRACT and archive_path.exists():
        archive_path.unlink()
    cleanup_non_data_outputs(item)

def migrate_old_extraction(item):
    old_dir = old_extraction_dir_for(item)
    if not old_dir.is_dir():
        return False
    old_marker = old_dir / ".extract_complete"
    old_data_path = old_dir / data_file_path_for(item).name
    if not old_data_path.exists() or old_data_path.stat().st_size <= 0:
        return False

    target_path = data_file_path_for(item)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists():
        shutil.move(str(old_data_path), str(target_path))
    if CLEANUP_OLD_EXTRACTED_FOLDERS and old_marker.exists():
        shutil.rmtree(old_dir)
    cleanup_non_data_outputs(item)
    return True


def archive_is_complete(session, item):
    if not item.path.exists() or item.path.stat().st_size <= 0:
        return False
    if not CHECK_REMOTE_SIZE_FOR_EXISTING:
        return True
    length = remote_content_length(session, item)
    return length is None or item.path.stat().st_size == length


def existing_item_status(session, item):
    if EXTRACT_AFTER_DOWNLOAD and data_file_exists(item):
        cleanup_non_data_outputs(item)
        return "skipped"
    if EXTRACT_AFTER_DOWNLOAD and migrate_old_extraction(item):
        return "extracted"
    if not archive_is_complete(session, item):
        return None
    if EXTRACT_AFTER_DOWNLOAD:
        extract_archive(item)
        return "extracted"
    return "skipped"


def download_one(session, item):
    item.path.parent.mkdir(parents=True, exist_ok=True)
    part_path = item.path.with_name(item.path.name + ".part")
    if part_path.exists():
        part_path.unlink()

    try:
        with session.get(item.url, stream=True, timeout=TIMEOUT) as response:
            response.raise_for_status()
            expected_bytes = int(response.headers.get("Content-Length") or 0)
            with part_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        handle.write(chunk)

        actual_bytes = part_path.stat().st_size
        if expected_bytes and actual_bytes != expected_bytes:
            raise IOError(f"Expected {expected_bytes} bytes, downloaded {actual_bytes} bytes for {item.url}")
        if VERIFY_ZIP_AFTER_DOWNLOAD:
            verify_zip(part_path)

        if EXTRACT_AFTER_DOWNLOAD:
            extract_archive(item, archive_path=part_path)
        else:
            part_path.replace(item.path)
    except Exception:
        if part_path.exists():
            part_path.unlink()
        raise

def download_one_with_retries(session, item):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            download_one(session, item)
            return
        except Exception as exc:
            last_exc = exc
            if attempt > MAX_RETRIES:
                break
            time.sleep(min(60, 2 ** attempt))
    raise last_exc


def download_many(items):
    session = build_session()
    summary = {"downloaded": 0, "extracted": 0, "skipped": 0, "failed": 0, "failures": []}

    with tqdm(total=len(items), unit="file", desc=f"PRISM {TIME_PERIOD} grids") as bar:
        for item in items:
            try:
                status = existing_item_status(session, item)
                if status == "skipped":
                    summary["skipped"] += 1
                elif status == "extracted":
                    summary["extracted"] += 1
                else:
                    download_one_with_retries(session, item)
                    summary["downloaded"] += 1
                    if EXTRACT_AFTER_DOWNLOAD:
                        summary["extracted"] += 1
                    if REQUEST_DELAY_SECONDS > 0:
                        time.sleep(REQUEST_DELAY_SECONDS)
            except Exception as exc:
                summary["failed"] += 1
                summary["failures"].append({
                    "variable": item.variable,
                    "period": item.label,
                    "url": item.url,
                    "path": str(item.path),
                    "data_path": str(data_file_path_for(item)),
                    "error": repr(exc),
                })
                if STOP_ON_ERROR:
                    raise
            finally:
                bar.update(1)
                bar.set_postfix(
                    downloaded=summary["downloaded"],
                    extracted=summary["extracted"],
                    skipped=summary["skipped"],
                    failed=summary["failed"],
                )

    return summary


# ---- CLI --------------------------------------------------------------------

def main(argv=None):
    import argparse
    import json as _json
    import sys

    global START_DATE, END_DATE, VARIABLES, OUTPUT_DIR, RUN_DOWNLOAD, USER_AGENT

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--variables", default=",".join(VARIABLES))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true", help="List planned files without downloading")
    args = parser.parse_args(argv)

    START_DATE, END_DATE = args.start, args.end
    VARIABLES = tuple(v.strip() for v in args.variables.split(",") if v.strip())
    OUTPUT_DIR = Path(args.output_dir).expanduser()
    contact = os.getenv("PRISM_CONTACT_EMAIL", "").strip()
    if contact:
        USER_AGENT = f"{USER_AGENT} ({contact})"

    items = build_manifest()
    print(f"Prepared {len(items):,} PRISM {TIME_PERIOD} grid downloads -> {OUTPUT_DIR.resolve()}", flush=True)
    if args.dry_run:
        for item in items[:8]:
            print(f"  {item.url} -> {data_file_path_for(item)}")
        return 0

    summary = download_many(items)
    print(_json.dumps({k: v for k, v in summary.items() if k != "failures"}), flush=True)
    for failure in summary["failures"][:20]:
        print(f"FAILED {failure['variable']} {failure['period']}: {failure['error']}", file=sys.stderr)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
