#!/usr/bin/env python3
"""Dispatch pipeline data downloads to the existing host scripts.

  merra2   /data/merra2/download_merra2_aws.py
  narr     /data2/NARR/download_narr.py + narr_daily_subset.yaml
  prism    /data2/PRISM/download_prism_daily_800m.ipynb
  cordex   CORDEX-ML-Bench from https://zenodo.org/records/17517423
  elevation  write/copy prism_elevation.nc if missing (granite example, else ETOPO)
  all      merra2 + narr + prism + cordex
  status   inventory on-disk inputs
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

DEFAULT_GRANITE = Path(
    os.getenv("GRANITE_WXC_REPO", "/data2/aashishp/github_merra_test/granite-wxc")
)
DEFAULT_CONFIG = str(DEFAULT_GRANITE / "examples" / "MERRA_PRISM" / "MERRA_PRISM.yaml")
BUNDLED = Path(__file__).resolve().parent / "downloaders"
LAB_MERRA = Path("/data/merra2/download_merra2_aws.py")
LAB_NARR = Path("/data2/NARR/download_narr.py")
LAB_NARR_YAML = Path("/data2/NARR/narr_daily_subset.yaml")
LAB_PRISM = Path("/data2/PRISM/download_prism_daily_800m.ipynb")
ZENODO_RECORD = os.getenv("CORDEX_ZENODO_RECORD", "17517423")
ZENODO_PAGE = f"https://zenodo.org/records/{ZENODO_RECORD}"
PYTHON_JOB = os.getenv("GRANITE_WXC_PYTHON", sys.executable)
USER_DATA_ROOT = Path.home() / "prithvi-wxc-data"


def _first_existing(*paths: Path) -> Optional[Path]:
    for path in paths:
        if path and path.exists():
            return path
    return None


def resolve_merra_script() -> Path:
    env = os.getenv("MERRA2_DOWNLOAD_SCRIPT")
    if env:
        return Path(env).expanduser()
    found = _first_existing(LAB_MERRA, BUNDLED / "download_merra2_aws.py")
    return found or (BUNDLED / "download_merra2_aws.py")


def resolve_narr_script() -> Path:
    env = os.getenv("NARR_DOWNLOAD_SCRIPT")
    if env:
        return Path(env).expanduser()
    found = _first_existing(LAB_NARR, BUNDLED / "download_narr.py")
    return found or (BUNDLED / "download_narr.py")


def resolve_narr_yaml() -> Path:
    env = os.getenv("NARR_DOWNLOAD_YAML")
    if env:
        return Path(env).expanduser()
    found = _first_existing(LAB_NARR_YAML, BUNDLED / "narr_daily_subset.yaml")
    return found or (BUNDLED / "narr_daily_subset.yaml")


def resolve_prism_notebook() -> Path:
    env = os.getenv("PRISM_DOWNLOAD_NOTEBOOK")
    if env:
        return Path(env).expanduser()
    found = _first_existing(LAB_PRISM, BUNDLED / "download_prism_daily_800m.ipynb")
    return found or (BUNDLED / "download_prism_daily_800m.ipynb")


def using_lab_script(script: Path) -> bool:
    try:
        resolved = str(script.resolve())
    except Exception:
        resolved = str(script)
    return resolved.startswith("/data/") or resolved.startswith("/data2/NARR") or resolved.startswith("/data2/PRISM")


def data_root(script: Optional[Path] = None) -> Path:
    env = (os.getenv("PIPELINE_DATA_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if script is not None and using_lab_script(script):
        return Path("/data")
    return USER_DATA_ROOT.resolve()


MERRA_SCRIPT = resolve_merra_script()
NARR_SCRIPT = resolve_narr_script()
NARR_YAML = resolve_narr_yaml()
PRISM_NOTEBOOK = resolve_prism_notebook()
if (os.getenv("PIPELINE_DATA_ROOT") or "").strip():
    CORDEX_DEFAULT_DIR = data_root() / "cordex"
elif LAB_MERRA.exists():
    CORDEX_DEFAULT_DIR = DEFAULT_GRANITE / "granite-geospatial-wxc-downscaling" / "CORDEX"
else:
    CORDEX_DEFAULT_DIR = USER_DATA_ROOT / "cordex"

PRISM_VARS = ("ppt", "tmax", "tmin")
ETOPO_URLS = (
    "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/30s/"
    "ETOPO_2022_v1_30s_N90W180_surface.nc",
    "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/30s/"
    "30s_surface_elev_netcdf/ETOPO_2022_v1_30s_N90W180_surface.nc",
)
PRISM_NROWS, PRISM_NCOLS = 3105, 7025
PRISM_LAT_MAX = 49.937499999999503
PRISM_LON_MIN = -125.02083333333351
PRISM_RES = 0.0083333333329999992


def log(msg: str) -> None:
    print(msg, flush=True)


def load_config(config_path: Optional[str]) -> dict:
    if not config_path or yaml is None:
        return {}
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = (DEFAULT_GRANITE / "examples" / "MERRA_PRISM" / path).resolve()
        if not path.exists():
            path = (DEFAULT_GRANITE / "examples" / "NARR_PRISM" / Path(config_path).name).resolve()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def run_logged(cmd: List[str], cwd: Path) -> int:
    log(f"[cmd] cwd={cwd} {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode


# ---------------------------------------------------------------------------
# MERRA-2 — existing earthaccess script
# ---------------------------------------------------------------------------

def download_merra2(start: str, end: str) -> dict:
    script = resolve_merra_script()
    if not script.exists():
        raise SystemExit(f"MERRA-2 download script not found: {script}")
    relocate = (os.getenv("PIPELINE_DATA_ROOT") or "").strip() or not using_lab_script(script)
    merra_root = (data_root(script) / "merra2") if relocate else Path("/data/merra2")
    patch = ""
    if relocate:
        patch = f"""
mod.CONFIG["output_dir"] = {str(merra_root / "download_tmp")!r}
mod.CONFIG["existing_download_dirs"] = [{str(merra_root)!r}]
subset = mod.CONFIG.setdefault("subset_compile", {{}})
subset["daily_input_dir"] = {str(merra_root / "daily_subset")!r}
subset["daily_output_dir"] = {str(merra_root / "daily_subset_with_H")!r}
"""
    code = f"""
import importlib.util, sys
script = {str(script)!r}
sys.argv = [script]
spec = importlib.util.spec_from_file_location("download_merra2_aws", script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.CONFIG.setdefault("earthaccess", {{}})
mod.CONFIG["earthaccess"]["start_date"] = {start!r}
mod.CONFIG["earthaccess"]["end_date"] = {end!r}
mod.CONFIG["earthaccess"]["temporal_start"] = {start!r}
mod.CONFIG["earthaccess"]["temporal_end"] = {end!r}
{patch}
raise SystemExit(mod.main())
"""
    rc = run_logged([PYTHON_JOB, "-c", code], cwd=script.parent)
    return {
        "status": "ok" if rc == 0 else "error",
        "script": str(script),
        "output_root": str(merra_root / "daily_subset_with_H"),
        "on_this_machine": True,
        "returncode": rc,
        "start": start,
        "end": end,
    }


# ---------------------------------------------------------------------------
# NARR — existing NOAA PSL downloader + YAML
# ---------------------------------------------------------------------------

def download_narr(start: str, end: str, overwrite: bool = False) -> dict:
    script = resolve_narr_script()
    src_yaml = resolve_narr_yaml()
    if not script.exists():
        raise SystemExit(f"NARR download script not found: {script}")
    if not src_yaml.exists():
        raise SystemExit(f"NARR YAML not found: {src_yaml}")
    relocate = (os.getenv("PIPELINE_DATA_ROOT") or "").strip() or not using_lab_script(script)
    if relocate:
        if yaml is None:
            raise SystemExit("pyyaml is required to rewrite NARR output paths")
        cfg = yaml.safe_load(src_yaml.read_text()) or {}
        narr_root = data_root(script) / "narr"
        cfg.setdefault("download", {})["out_dir"] = str(narr_root / "raw")
        cfg.setdefault("subset", {})["out_dir"] = str(narr_root / "subset")
        yaml_path = narr_root / "narr_daily_subset.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        cwd = narr_root
    else:
        yaml_path = src_yaml
        cwd = script.parent
        narr_root = Path("/data2/NARR/data")
    cmd = [
        PYTHON_JOB,
        str(script),
        "--config",
        str(yaml_path),
        "--start",
        start,
        "--end",
        end,
    ]
    if overwrite:
        cmd.append("--overwrite")
    rc = run_logged(cmd, cwd=cwd)
    return {
        "status": "ok" if rc == 0 else "error",
        "script": str(script),
        "config": str(yaml_path),
        "output_root": str(narr_root),
        "on_this_machine": True,
        "returncode": rc,
        "start": start,
        "end": end,
    }


# ---------------------------------------------------------------------------
# PRISM — execute the existing notebook with date/output overrides
# ---------------------------------------------------------------------------

def download_prism(start: str, end: str, output_dir: Optional[Path], variables: Sequence[str]) -> dict:
    notebook = resolve_prism_notebook()
    if not notebook.exists():
        raise SystemExit(f"PRISM notebook not found: {notebook}")
    relocate = (os.getenv("PIPELINE_DATA_ROOT") or "").strip() or not using_lab_script(notebook)
    dest = output_dir
    if dest is None:
        dest = (
            data_root(notebook) / "prism" / "prism_daily_800m_an"
            if relocate
            else Path("/data/PRISM/prism_daily_800m_an")
        )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__main__"}
    patched = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source") or [])
        if not patched and "START_DATE" in src:
            src = re.sub(r'^START_DATE = .*$', f'START_DATE = "{start}"', src, flags=re.M)
            src = re.sub(r'^END_DATE = .*$', f'END_DATE = "{end}"', src, flags=re.M)
            src = re.sub(r'^VARIABLES = .*$', f"VARIABLES = {tuple(variables)!r}", src, flags=re.M)
            src = re.sub(r'^OUTPUT_DIR = .*$', f'OUTPUT_DIR = Path("{dest}")', src, flags=re.M)
            patched = True
        exec(compile(src, str(notebook), "exec"), ns)
    return {
        "status": "ok",
        "notebook": str(notebook),
        "output_dir": str(dest),
        "on_this_machine": True,
        "start": start,
        "end": end,
        "variables": list(variables),
    }


# ---------------------------------------------------------------------------
# CORDEX-ML-Bench — Zenodo
# ---------------------------------------------------------------------------

def _zenodo_files(record_id: str) -> list[dict]:
    url = f"https://zenodo.org/api/records/{record_id}"
    req = Request(url, headers={"User-Agent": "Prithvi-WxC-Downscaling/1.0"})
    with urlopen(req, timeout=60) as resp:
        rec = json.loads(resp.read().decode("utf-8"))
    files = rec.get("files") or rec.get("links", {})
    if isinstance(files, list):
        return files
    raise SystemExit(f"Unexpected Zenodo payload for record {record_id}")


def _http_download(url: str, dest: Path, skip_existing: bool = True) -> None:
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        log(f"[cordex] skip existing {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": "Prithvi-WxC-Downscaling/1.0"})
    log(f"[cordex] GET {url}")
    with urlopen(req, timeout=600) as resp, part.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    part.replace(dest)


def download_cordex(output_dir: Path, domains: Sequence[str], skip_existing: bool = True) -> dict:
    wanted = {d.lower() for d in domains}
    if "all" in wanted:
        wanted = {"nz", "alps", "sa"}
    files = _zenodo_files(ZENODO_RECORD)
    downloaded, extracted, skipped = [], [], []
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in files:
        key = str(entry.get("key") or entry.get("filename") or "")
        if not key.lower().endswith(".zip"):
            continue
        stem = Path(key).stem.lower()
        domain = next((d for d in ("nz", "alps", "sa") if d in stem), None)
        if domain is None or domain not in wanted:
            continue
        links = entry.get("links") or {}
        url = links.get("self") or links.get("download") or (
            f"https://zenodo.org/records/{ZENODO_RECORD}/files/{key}?download=1"
        )
        zip_path = output_dir / Path(key).name
        unpack_dir = output_dir / Path(key).stem
        if skip_existing and unpack_dir.exists() and any(unpack_dir.rglob("*.nc")):
            skipped.append(str(unpack_dir))
            log(f"[cordex] already unpacked {unpack_dir}")
            continue
        _http_download(url, zip_path, skip_existing=skip_existing)
        downloaded.append(str(zip_path))
        log(f"[cordex] unpack {zip_path} -> {output_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output_dir)
        extracted.append(str(unpack_dir if unpack_dir.exists() else output_dir))
    return {
        "status": "ok",
        "record": ZENODO_PAGE,
        "output_dir": str(output_dir),
        "downloaded": downloaded,
        "extracted": extracted,
        "skipped": skipped,
        "source": "Rampal et al. CORDEX-ML-Bench, https://doi.org/10.5281/zenodo.17517423",
    }


# ---------------------------------------------------------------------------
# Elevation fallback (no host script; granite already ships prism_elevation.nc)
# ---------------------------------------------------------------------------

def prism_target_grid(grid_from: Optional[Path] = None):
    import numpy as np

    if grid_from and Path(grid_from).exists():
        import xarray as xr

        with xr.open_dataset(grid_from) as ds:
            lat_name = next(n for n in ("lat", "latitude") if n in ds.coords or n in ds.dims)
            lon_name = next(n for n in ("lon", "longitude") if n in ds.coords or n in ds.dims)
            return np.asarray(ds[lat_name].values), np.asarray(ds[lon_name].values)
    lats = PRISM_LAT_MAX - (np.arange(PRISM_NROWS, dtype=np.float64) + 0.5) * PRISM_RES
    lons = PRISM_LON_MIN + (np.arange(PRISM_NCOLS, dtype=np.float64) + 0.5) * PRISM_RES
    return lats, lons


def download_elevation(output_file: Path, grid_from: Optional[Path] = None) -> dict:
    example = DEFAULT_GRANITE / "examples" / "MERRA_PRISM" / "prism_elevation.nc"
    if example.exists() and output_file.resolve() != example.resolve():
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(example.read_bytes())
        return {"status": "ok", "output_file": str(output_file), "source": str(example)}
    if output_file.exists():
        return {"status": "ok", "output_file": str(output_file), "source": "already present"}

    import numpy as np
    import xarray as xr

    cache_dir = output_file.parent / ".elevation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw = cache_dir / "ETOPO_2022_v1_30s_N90W180_surface.nc"
    if not raw.exists():
        last_err = None
        for url in ETOPO_URLS:
            log(f"[elevation] downloading {url}")
            try:
                req = Request(url, headers={"User-Agent": "Prithvi-WxC-Downscaling/1.0"})
                with urlopen(req, timeout=600) as resp:
                    raw.write_bytes(resp.read())
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                log(f"[elevation] failed {url}: {exc}")
        if last_err is not None and not raw.exists():
            raise SystemExit(f"Could not download ETOPO DEM and no granite elevation file at {example}")
    ds = xr.open_dataset(raw)
    z_name = next((n for n in ("z", "Band1", "elevation", "height") if n in ds.data_vars), list(ds.data_vars)[0])
    lat_name = next(n for n in ("lat", "latitude", "y") if n in ds.coords or n in ds.dims)
    lon_name = next(n for n in ("lon", "longitude", "x") if n in ds.coords or n in ds.dims)
    target_lat, target_lon = prism_target_grid(grid_from)
    da = ds[z_name].interp({lat_name: target_lat, lon_name: target_lon}, method="linear")
    elev = np.nan_to_num(np.asarray(da.values, dtype=np.float32), nan=-9999.0)
    out = xr.Dataset({"elevation": (("lat", "lon"), elev)}, coords={"lat": target_lat, "lon": target_lon})
    out["elevation"].attrs.update(units="m", long_name="surface elevation above mean sea level")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(output_file)
    ds.close()
    out.close()
    return {"status": "ok", "output_file": str(output_file), "source": "ETOPO 2022"}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def _count_dated_files(root: Path, pattern_substr: str = "") -> dict:
    if not root.exists():
        return {"exists": False, "files": 0, "first": None, "last": None, "path": str(root)}
    files = sorted(p for p in root.rglob("*") if p.is_file() and (not pattern_substr or pattern_substr in p.name))
    dates = []
    for path in files:
        for token in path.stem.replace("-", "").split("_"):
            if len(token) >= 8 and token[:8].isdigit():
                try:
                    dates.append(datetime.strptime(token[:8], "%Y%m%d").date())
                    break
                except ValueError:
                    continue
    return {
        "exists": True,
        "files": len(files),
        "first": min(dates).isoformat() if dates else None,
        "last": max(dates).isoformat() if dates else None,
        "path": str(root),
    }


def status_report(
    config_path: Optional[str],
    merra_dir: Optional[Path],
    prism_dir: Optional[Path],
    elev_file: Optional[Path],
) -> dict:
    cfg = load_config(config_path)
    data = cfg.get("data") or {}
    merra_dir = merra_dir or Path(data.get("predictor_dir") or "/data/merra2/daily_subset_with_H")
    prism_dir = prism_dir or Path(data.get("target_dir") or "/data/PRISM/prism_daily_800m_an")
    elev_file = elev_file or Path(
        data.get("static_elevation_file")
        or (DEFAULT_GRANITE / "examples/MERRA_PRISM/prism_elevation.nc")
    )
    if not elev_file.is_absolute():
        elev_file = (DEFAULT_GRANITE / elev_file).resolve()
    narr_subset = NARR_SCRIPT.parent / "data" / "subset"
    cordex_root = CORDEX_DEFAULT_DIR
    return {
        "scripts": {
            "merra2": str(resolve_merra_script()),
            "narr": str(resolve_narr_script()),
            "narr_yaml": str(resolve_narr_yaml()),
            "prism_notebook": str(resolve_prism_notebook()),
            "cordex_zenodo": ZENODO_PAGE,
            "bundled_downloaders": str(BUNDLED),
        },
        "writes_on": "the machine running this MCP process",
        "user_data_root": str(USER_DATA_ROOT),
        "pipeline_data_root": (os.getenv("PIPELINE_DATA_ROOT") or "").strip() or None,
        "merra2": _count_dated_files(merra_dir, "M2I3NPASM"),
        "narr": _count_dated_files(narr_subset if narr_subset.exists() else NARR_SCRIPT.parent / "data"),
        "prism": {var: _count_dated_files(prism_dir / var, var) for var in PRISM_VARS},
        "elevation": {
            "exists": elev_file.exists(),
            "path": str(elev_file),
            "bytes": elev_file.stat().st_size if elev_file.exists() else 0,
        },
        "cordex": {
            "exists": cordex_root.exists(),
            "path": str(cordex_root),
            "nc_files": len(list(cordex_root.rglob("*.nc"))) if cordex_root.exists() else 0,
        },
        "auth": {
            "EARTHDATA_USERNAME": bool(os.getenv("EARTHDATA_USERNAME")),
            "EARTHDATA_PASSWORD": bool(os.getenv("EARTHDATA_PASSWORD")),
            "netrc": (Path.home() / ".netrc").exists(),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_dates(p):
        p.add_argument("--config", default=os.getenv("GRANITE_WXC_CONFIG", DEFAULT_CONFIG))
        p.add_argument("--start")
        p.add_argument("--end")
        p.add_argument("--overwrite", action="store_true")

    for name in ("merra2", "narr", "prism", "elevation", "cordex", "all"):
        p = sub.add_parser(name)
        add_dates(p)
        if name in {"prism", "all"}:
            p.add_argument("--output-dir")
            p.add_argument("--variables", default="ppt,tmax,tmin")
            p.add_argument("--prism-dir")
        if name in {"merra2", "all"}:
            p.add_argument("--merra-dir")
        if name in {"elevation", "all"}:
            p.add_argument("--output-file")
            p.add_argument("--elevation-file")
            p.add_argument("--grid-from")
        if name in {"cordex", "all"}:
            p.add_argument("--cordex-dir")
            p.add_argument("--domains", default="all", help="Comma-separated: nz,alps,sa,all")

    p_s = sub.add_parser("status")
    p_s.add_argument("--config", default=os.getenv("GRANITE_WXC_CONFIG", DEFAULT_CONFIG))
    p_s.add_argument("--merra-dir")
    p_s.add_argument("--prism-dir")
    p_s.add_argument("--elevation-file")

    args = parser.parse_args(argv)
    cfg = load_config(getattr(args, "config", None))
    data = cfg.get("data") or {}
    results = {}

    if args.cmd == "status":
        print(json.dumps(status_report(
            args.config,
            Path(args.merra_dir) if args.merra_dir else None,
            Path(args.prism_dir) if args.prism_dir else None,
            Path(args.elevation_file) if args.elevation_file else None,
        ), indent=2))
        return 0

    need_dates = args.cmd in {"merra2", "narr", "prism", "all"}
    if need_dates and (not args.start or not args.end):
        dates = (cfg.get("dates") or {}).get("training") or {}
        args.start = args.start or dates.get("start")
        args.end = args.end or dates.get("end")
    if need_dates and (not args.start or not args.end):
        raise SystemExit("--start and --end are required (YYYY-MM-DD)")

    if args.cmd in {"merra2", "all"}:
        results["merra2"] = download_merra2(args.start, args.end)
    if args.cmd in {"narr", "all"}:
        results["narr"] = download_narr(args.start, args.end, overwrite=args.overwrite)
    if args.cmd in {"prism", "all"}:
        prism_dir = getattr(args, "prism_dir", None) or getattr(args, "output_dir", None) or data.get("target_dir")
        variables = [v.strip() for v in str(getattr(args, "variables", "ppt,tmax,tmin")).split(",") if v.strip()]
        results["prism"] = download_prism(args.start, args.end, Path(prism_dir) if prism_dir else None, variables)
    if args.cmd in {"cordex", "all"}:
        cordex_dir = Path(getattr(args, "cordex_dir", None) or CORDEX_DEFAULT_DIR)
        domains = [d.strip() for d in str(getattr(args, "domains", "all")).split(",") if d.strip()]
        results["cordex"] = download_cordex(cordex_dir, domains, skip_existing=not args.overwrite)
    if args.cmd in {"elevation", "all"}:
        elev_path = Path(
            getattr(args, "output_file", None)
            or getattr(args, "elevation_file", None)
            or data.get("static_elevation_file")
            or (DEFAULT_GRANITE / "examples/MERRA_PRISM/prism_elevation.nc")
        )
        if not elev_path.is_absolute():
            elev_path = (DEFAULT_GRANITE / elev_path).resolve()
        grid_from = getattr(args, "grid_from", None)
        results["elevation"] = download_elevation(elev_path, Path(grid_from) if grid_from else None)

    print(json.dumps(results, indent=2))
    failed = [k for k, v in results.items() if isinstance(v, dict) and v.get("status") not in {"ok", "partial"}]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
