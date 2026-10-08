#!/usr/bin/env python3
"""Download every input the PrithviWxC downscaling pipeline needs.

All files go under PIPELINE_DATA_ROOT (default ~/prithvi-wxc-data) using the
layout in config.py, on the machine that runs this script. Versions come from
pins.json.

  merra2     MERRA-2 M2I3NPASM via earthaccess (Earthdata Login) -> daily subset with H
  narr       NOAA PSL NARR pressure-level daily files -> subset
  prism      PRISM AN daily 800 m ppt/tmax/tmin (.nc)
  elevation  800 m orography on the PRISM grid (reference file from Zenodo 10.5281/zenodo.23096854, sha256-verified)
  weights    pretrained Prithvi WxC downscaling weights from Hugging Face (pinned revision)
  code       clone the pinned Prithvi-UNet training/inference code
  env        build envs/prithvi from the frozen training lock + pinned code
  cordex     CORDEX-ML-Bench from Zenodo (optional benchmark)
  all        merra2 + prism + elevation + weights   (add narr/cordex explicitly)
  status     inventory what is on disk
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence
from urllib.request import Request, urlopen

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402

BUNDLED = Path(__file__).resolve().parent / "downloaders"
PYTHON_JOB = sys.executable
USER_AGENT = "Prithvi-WxC-Downscaling-MCP/2.0"
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


def _script(env_name: str, bundled_name: str) -> Path:
    override = (os.getenv(env_name) or "").strip()
    return Path(override).expanduser() if override else BUNDLED / bundled_name


def run_logged(cmd: List[str], cwd: Path, env: Optional[dict] = None) -> int:
    log(f"[cmd] cwd={cwd} {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd), env=env).returncode


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def http_download(url: str, dest: Path, headers: Optional[dict] = None, resume: bool = True) -> None:
    """Stream url to dest via a .part file; resume a partial download when possible."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    offset = part.stat().st_size if (resume and part.exists()) else 0
    if offset:
        hdrs["Range"] = f"bytes={offset}-"
    log(f"[http] GET {url}" + (f" (resume at {offset:,} bytes)" if offset else ""))
    with urlopen(Request(url, headers=hdrs), timeout=600) as resp:
        mode = "ab" if offset and resp.status == 206 else "wb"
        total = resp.headers.get("Content-Length")
        done = offset if mode == "ab" else 0
        next_report = done + 1024 ** 3
        with part.open(mode) as fh:
            while True:
                block = resp.read(8 * 1024 * 1024)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if done >= next_report:
                    log(f"[http] {done / 1024 ** 3:.1f} GiB" + (f" of {(int(total) + offset) / 1024 ** 3:.1f} GiB" if total else ""))
                    next_report += 1024 ** 3
    part.replace(dest)


# ---------------------------------------------------------------------------
# MERRA-2 — bundled earthaccess script, outputs relocated under the data root
# ---------------------------------------------------------------------------

def download_merra2(start: str, end: str) -> dict:
    script = _script("MERRA2_DOWNLOAD_SCRIPT", "download_merra2_aws.py")
    out_dir = C.merra2_dir()
    root = out_dir.parent
    code = f"""
import importlib.util, sys
script = {str(script)!r}
sys.argv = [script]
spec = importlib.util.spec_from_file_location("download_merra2_aws", script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ea = mod.CONFIG.setdefault("earthaccess", {{}})
ea["start_date"] = ea["temporal_start"] = {start!r}
ea["end_date"] = ea["temporal_end"] = {end!r}
mod.CONFIG["output_dir"] = {str(root / "download_tmp")!r}
mod.CONFIG["existing_download_dirs"] = [{str(root)!r}]
subset = mod.CONFIG.setdefault("subset_compile", {{}})
subset["daily_input_dir"] = {str(root / "daily_subset")!r}
subset["daily_output_dir"] = {str(out_dir)!r}
raise SystemExit(mod.main())
"""
    root.mkdir(parents=True, exist_ok=True)
    rc = run_logged([PYTHON_JOB, "-c", code], cwd=root)
    return {"status": "ok" if rc == 0 else "error", "returncode": rc, "script": str(script),
            "output_dir": str(out_dir), "start": start, "end": end}


# ---------------------------------------------------------------------------
# NARR — bundled NOAA PSL downloader; YAML rewritten to the data root
# ---------------------------------------------------------------------------

def download_narr(start: str, end: str, overwrite: bool = False) -> dict:
    if yaml is None:
        raise SystemExit("pyyaml is required for NARR downloads")
    script = _script("NARR_DOWNLOAD_SCRIPT", "download_narr.py")
    src_yaml = _script("NARR_DOWNLOAD_YAML", "narr_daily_subset.yaml")
    out_dir = C.narr_dir()
    root = out_dir.parent
    cfg = yaml.safe_load(src_yaml.read_text()) or {}
    download = cfg.setdefault("download", {})
    download["out_dir"] = str(root / "raw")
    # download_narr.py prefers the YAML's start/end over --start/--end, and the
    # bundled YAML spans the whole archive, so the requested range goes in here.
    download["start"], download["end"] = start, end
    cfg.setdefault("subset", {})["out_dir"] = str(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_yaml = root / f"narr_daily_subset_{start}_{end}.yaml"
    run_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    cmd = [PYTHON_JOB, str(script), "--config", str(run_yaml), "--start", start, "--end", end]
    if overwrite:
        cmd.append("--overwrite")
    rc = run_logged(cmd, cwd=root)
    return {"status": "ok" if rc == 0 else "error", "returncode": rc, "script": str(script),
            "config": str(run_yaml), "output_dir": str(out_dir), "start": start, "end": end}


# ---------------------------------------------------------------------------
# PRISM — CLI version of the PRISM notebook
# ---------------------------------------------------------------------------

def download_prism(start: str, end: str, output_dir: Optional[Path], variables: Sequence[str]) -> dict:
    script = _script("PRISM_DOWNLOAD_SCRIPT", "download_prism.py")
    dest = Path(output_dir).expanduser() if output_dir else C.prism_dir()
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON_JOB, str(script), "--start", start, "--end", end,
           "--variables", ",".join(variables), "--output-dir", str(dest)]
    rc = run_logged(cmd, cwd=dest)
    return {"status": "ok" if rc == 0 else "error", "returncode": rc, "script": str(script),
            "output_dir": str(dest), "start": start, "end": end, "variables": list(variables)}


# ---------------------------------------------------------------------------
# Static elevation
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


def _elevation_result(path: Path, source: str) -> dict:
    pin = C.PINS["data"]["elevation"]
    digest = sha256_file(path)
    matches = digest == pin["sha256"]
    result = {"status": "ok", "output_file": str(path), "source": source,
              "sha256": digest, "matches_reference": matches}
    if not matches:
        result["warning"] = (
            "This elevation file differs from the reference used for the published models "
            f"(sha256 {pin['sha256'][:12]}…). Results will not be bit-identical."
        )
    return result


def _etopo_elevation(output_file: Path, grid_from: Optional[Path]) -> None:
    import numpy as np
    import xarray as xr

    cache_dir = output_file.parent / ".elevation_cache"
    raw = cache_dir / "ETOPO_2022_v1_30s_N90W180_surface.nc"
    if not raw.exists():
        last_err: Optional[Exception] = None
        for url in ETOPO_URLS:
            try:
                http_download(url, raw)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                log(f"[elevation] failed {url}: {exc}")
        if last_err is not None:
            raise SystemExit(f"Could not download ETOPO DEM: {last_err}")
    with xr.open_dataset(raw) as ds:
        z_name = next((n for n in ("z", "Band1", "elevation", "height") if n in ds.data_vars), list(ds.data_vars)[0])
        lat_name = next(n for n in ("lat", "latitude", "y") if n in ds.coords or n in ds.dims)
        lon_name = next(n for n in ("lon", "longitude", "x") if n in ds.coords or n in ds.dims)
        target_lat, target_lon = prism_target_grid(grid_from)
        da = ds[z_name].interp({lat_name: target_lat, lon_name: target_lon}, method="linear")
        elev = np.nan_to_num(np.asarray(da.values, dtype=np.float32), nan=-9999.0)
    out = xr.Dataset({"elevation": (("lat", "lon"), elev)}, coords={"lat": target_lat, "lon": target_lon})
    out["elevation"].attrs.update(units="m", long_name="surface elevation above mean sea level")
    out.attrs.update(title="Surface elevation on the PRISM 30 arc-second grid", source_dem="ETOPO 2022 30s")
    out.to_netcdf(output_file)


def download_elevation(output_file: Optional[Path] = None, grid_from: Optional[Path] = None,
                       overwrite: bool = False) -> dict:
    """Fetch the pinned reference file (Zenodo url or ELEVATION_SOURCE_FILE). ETOPO is used only when no url is pinned."""
    pin = C.PINS["data"]["elevation"]
    output_file = Path(output_file).expanduser() if output_file else C.elevation_file()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not overwrite:
        return _elevation_result(output_file, "already present")

    source_file = (os.getenv("ELEVATION_SOURCE_FILE") or "").strip()
    if source_file and Path(source_file).expanduser().exists():
        shutil.copyfile(Path(source_file).expanduser(), output_file)
        return _elevation_result(output_file, f"copied from {source_file}")

    if pin.get("url"):
        try:
            http_download(pin["url"], output_file, resume=False)
        except Exception as exc:
            raise SystemExit(f"Could not download the reference elevation from {pin['url']}: {exc}")
        result = _elevation_result(output_file, pin["url"])
        if not result["matches_reference"]:
            output_file.unlink()
            raise SystemExit(f"Checksum mismatch for {pin['url']}; deleted the download.")
        return result

    log("[elevation] no reference URL pinned; regenerating from ETOPO 2022 (not bit-identical)")
    _etopo_elevation(output_file, grid_from)
    return _elevation_result(output_file, "ETOPO 2022 fallback")


# ---------------------------------------------------------------------------
# Pretrained weights — Hugging Face, pinned revision
# ---------------------------------------------------------------------------

def download_weights(overwrite: bool = False) -> dict:
    pin = C.PINS["weights"]
    dest = C.weights_file()
    url = f"https://huggingface.co/{pin['hf_repo']}/resolve/{pin['revision']}/{pin['file']}"
    if dest.exists() and not overwrite and dest.stat().st_size == pin["approx_bytes"]:
        return {"status": "ok", "output_file": str(dest), "source": "already present",
                "bytes": dest.stat().st_size}
    free = shutil.disk_usage(dest.parent if dest.parent.exists() else C.data_root()).free
    if free < pin["approx_bytes"] * 1.05:
        raise SystemExit(f"Need ~{pin['approx_bytes'] / 1e9:.1f} GB free for weights; only {free / 1e9:.1f} GB at {dest.parent}")
    headers = {}
    token = (os.getenv("HF_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    http_download(url, dest, headers=headers)
    size = dest.stat().st_size
    if size != pin["approx_bytes"]:
        raise SystemExit(f"Weights size {size} != pinned {pin['approx_bytes']}; download is incomplete or the pin is wrong")
    result = {"status": "ok", "output_file": str(dest), "source": url, "bytes": size}
    if pin.get("sha256"):
        digest = sha256_file(dest)
        result["sha256"] = digest
        if digest != pin["sha256"]:
            raise SystemExit(f"Weights sha256 {digest} != pinned {pin['sha256']}")
    return result


# ---------------------------------------------------------------------------
# Pinned training/inference code
# ---------------------------------------------------------------------------

def git_state(repo: Path) -> dict:
    def _git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()

    if not (repo / ".git").exists():
        return {"exists": repo.exists(), "is_git": False, "path": str(repo)}
    return {
        "exists": True,
        "is_git": True,
        "path": str(repo),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "remote": _git("remote", "get-url", "origin"),
        "dirty_files": [line[3:] for line in _git("status", "--porcelain", "--untracked-files=no").splitlines()],
    }


def _git_lines(repo: Path, *args: str) -> List[str]:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return out.stdout.splitlines()


def _localize_external_symlinks(repo: Path) -> List[str]:
    """Replace tracked symlinks that point outside the checkout, or dangle, with local directories.

    Some pinned commits track symlinks such as examples/MERRA_PRISM/scalars_with_H ->
    /data/granite-wxc/...; they dangle on other machines (writes fail) and on the
    original host send outputs into a different checkout. Others point inside the
    repo at an untracked per-machine link (experiments -> artifacts/experiments)
    that a fresh clone lacks. Each one becomes an empty local directory marked
    skip-worktree, so the checkout still reads as clean at the pinned commit.
    """
    replaced = []
    root = repo.resolve()
    for line in _git_lines(repo, "ls-files", "-s"):
        meta, _, rel = line.partition("\t")
        if not meta.startswith("120000"):
            continue
        link = repo / rel
        if not link.is_symlink():
            continue
        target = os.readlink(link)
        resolved = Path(target) if os.path.isabs(target) else (link.parent / target).resolve()
        try:
            resolved.resolve().relative_to(root)
            if resolved.exists():
                continue  # symlink inside the repo is fine
        except ValueError:
            pass
        link.unlink()
        link.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(repo), "update-index", "--skip-worktree", rel], check=True)
        replaced.append(f"{rel} -> {target}")
    return replaced


def _restore_localized_symlinks(repo: Path) -> None:
    """Undo _localize_external_symlinks before switching commits (only if the dirs are still empty)."""
    for line in _git_lines(repo, "ls-files", "-v"):
        tag, _, rel = line.partition(" ")
        if tag != "S":
            continue
        path = repo / rel
        if path.is_dir() and not path.is_symlink():
            if any(path.iterdir()):
                raise SystemExit(f"{path} holds run outputs; move them out before switching code variants")
            path.rmdir()
        subprocess.run(["git", "-C", str(repo), "update-index", "--no-skip-worktree", rel], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--", rel], check=True)


def setup_code(variant: Optional[str] = None, dest: Optional[Path] = None) -> dict:
    pin = C.code_pin(variant)
    dest = Path(dest).expanduser() if dest else C.default_granite_repo()
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        rc = run_logged(["git", "clone", "--branch", pin["branch"], pin["repo"], str(dest)], cwd=dest.parent)
        if rc != 0:
            raise SystemExit(f"git clone failed ({rc})")
    state = git_state(dest)
    if state["dirty_files"]:
        raise SystemExit(f"{dest} has local modifications {state['dirty_files'][:5]}; refusing to switch commits")
    if state["commit"] != pin["commit"]:
        _restore_localized_symlinks(dest)
        run_logged(["git", "fetch", "origin", pin["branch"]], cwd=dest)
        rc = run_logged(["git", "checkout", "--detach", pin["commit"]], cwd=dest)
        if rc != 0:
            raise SystemExit(f"Could not check out pinned commit {pin['commit']}")
    replaced = _localize_external_symlinks(dest)
    result = {"status": "ok", "variant": pin["variant"], **git_state(dest), "pinned_commit": pin["commit"]}
    if replaced:
        result["localized_symlinks"] = replaced
    return result


def setup_env(code_dir: Optional[Path] = None) -> dict:
    """Create envs/prithvi from the frozen training lock and install the pinned code into it."""
    env_pin = C.PINS["environment"]
    lock = C._PLUGIN_ROOT / env_pin["lock_file"]
    venv = C.training_venv_python().parent.parent
    code_dir = Path(code_dir).expanduser() if code_dir else C._GRANITE_REPO_ROOT
    if not (code_dir / "pyproject.toml").exists():
        raise SystemExit(f"Pinned code not found at {code_dir}; run the 'code' step first")
    uv = shutil.which("uv")
    if uv and subprocess.run([uv, "--version"], capture_output=True).returncode != 0:
        uv = None  # present but unusable (e.g. a sandboxed snap install)
    python = C.training_venv_python()
    if not python.exists():
        venv.parent.mkdir(parents=True, exist_ok=True)
        system_python = shutil.which(f"python{env_pin['python']}")
        if not uv and not system_python:
            raise SystemExit(
                f"The training env needs Python {env_pin['python']}. Install uv "
                "(https://docs.astral.sh/uv/getting-started/installation/), which fetches it automatically, "
                f"or put python{env_pin['python']} on PATH, then rerun setup_training_env."
            )
        cmd = ([uv, "venv", "--python", env_pin["python"], str(venv)] if uv
               else [system_python, "-m", "venv", str(venv)])
        if run_logged(cmd, cwd=venv.parent) != 0:
            raise SystemExit(f"Could not create a Python {env_pin['python']} virtualenv at {venv}")
    pip = [uv, "pip", "install", "--python", str(python)] if uv else [str(python), "-m", "pip", "install"]
    index = ["--extra-index-url", env_pin["torch_index_url"]]
    if uv:
        index += ["--index-strategy", "unsafe-best-match"]
    if run_logged(pip + index + ["-r", str(lock)], cwd=venv) != 0:
        raise SystemExit(f"Installing {lock} failed")
    if run_logged(pip + ["--no-deps", "-e", str(code_dir)], cwd=venv) != 0:
        raise SystemExit(f"Installing {code_dir} failed")
    probe = subprocess.run([str(python), "-c", "import torch, granitewxc, PrithviWxC; "
                            "print(torch.__version__, torch.cuda.is_available())"],
                           capture_output=True, text=True)
    return {"status": "ok" if probe.returncode == 0 else "error", "python": str(python),
            "lock_file": str(lock), "probe": (probe.stdout or probe.stderr).strip(),
            "hint": "Set GRANITE_WXC_PYTHON to this python (it is picked up automatically when unset)."}


# ---------------------------------------------------------------------------
# CORDEX-ML-Bench — Zenodo
# ---------------------------------------------------------------------------

def _zenodo_files(record_id: str) -> list[dict]:
    req = Request(f"https://zenodo.org/api/records/{record_id}", headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=60) as resp:
        rec = json.loads(resp.read().decode("utf-8"))
    files = rec.get("files")
    if isinstance(files, list):
        return files
    raise SystemExit(f"Unexpected Zenodo payload for record {record_id}")


def download_cordex(output_dir: Path, domains: Sequence[str], skip_existing: bool = True) -> dict:
    record = C.PINS["data"]["cordex_ml_bench"]["zenodo_record"]
    wanted = {d.lower() for d in domains}
    if "all" in wanted:
        wanted = {"nz", "alps", "sa"}
    downloaded, extracted, skipped = [], [], []
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in _zenodo_files(record):
        key = str(entry.get("key") or entry.get("filename") or "")
        if not key.lower().endswith(".zip"):
            continue
        domain = next((d for d in ("nz", "alps", "sa") if d in Path(key).stem.lower()), None)
        if domain is None or domain not in wanted:
            continue
        links = entry.get("links") or {}
        url = links.get("self") or links.get("download") or f"https://zenodo.org/records/{record}/files/{key}?download=1"
        zip_path = output_dir / Path(key).name
        unpack_dir = output_dir / Path(key).stem
        if skip_existing and unpack_dir.exists() and any(unpack_dir.rglob("*.nc")):
            skipped.append(str(unpack_dir))
            continue
        if not (skip_existing and zip_path.exists() and zip_path.stat().st_size > 0):
            http_download(url, zip_path)
        downloaded.append(str(zip_path))
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output_dir)
        extracted.append(str(unpack_dir if unpack_dir.exists() else output_dir))
    return {"status": "ok", "record": f"https://zenodo.org/records/{record}", "output_dir": str(output_dir),
            "downloaded": downloaded, "extracted": extracted, "skipped": skipped}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def _count_dated_files(root: Path, pattern_substr: str = "") -> dict:
    if not root.exists():
        return {"exists": False, "files": 0, "first": None, "last": None, "path": str(root)}
    dates = []
    count = 0
    for path in root.rglob("*"):
        if not path.is_file() or (pattern_substr and pattern_substr not in path.name) or path.name.endswith(".part"):
            continue
        count += 1
        for token in path.stem.replace("-", "").split("_"):
            if len(token) >= 8 and token[:8].isdigit():
                try:
                    dates.append(datetime.strptime(token[:8], "%Y%m%d").date())
                    break
                except ValueError:
                    continue
    return {
        "exists": True,
        "files": count,
        "first": min(dates).isoformat() if dates else None,
        "last": max(dates).isoformat() if dates else None,
        "path": str(root),
    }


def status_report() -> dict:
    elev = C.elevation_file()
    weights = C.weights_file()
    return {
        "data_root": str(C.data_root()),
        "writes_on": "the machine running this MCP process",
        "merra2": _count_dated_files(C.merra2_dir(), "M2I3NPASM"),
        "narr": _count_dated_files(C.narr_dir()),
        "prism": {var: _count_dated_files(C.prism_dir() / var, var) for var in PRISM_VARS},
        "elevation": {"exists": elev.exists(), "path": str(elev),
                      "bytes": elev.stat().st_size if elev.exists() else 0},
        "weights": {"exists": weights.exists(), "path": str(weights),
                    "bytes": weights.stat().st_size if weights.exists() else 0,
                    "complete": weights.exists() and weights.stat().st_size == C.PINS["weights"]["approx_bytes"]},
        "code": git_state(C._GRANITE_REPO_ROOT),
        "cordex": {"exists": C.cordex_dir().exists(), "path": str(C.cordex_dir()),
                   "nc_files": len(list(C.cordex_dir().rglob("*.nc"))) if C.cordex_dir().exists() else 0},
        "auth": {
            "earthdata_env": bool(os.getenv("EARTHDATA_USERNAME") and os.getenv("EARTHDATA_PASSWORD")),
            "netrc": (Path.home() / ".netrc").exists(),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _dates_from_config(config_path: Optional[str]) -> dict:
    if not config_path or yaml is None or not Path(config_path).exists():
        return {}
    cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    return ((cfg.get("dates") or {}).get("training") or {}) if isinstance(cfg, dict) else {}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("merra2", "narr", "prism", "elevation", "weights", "code", "env", "cordex", "all"):
        p = sub.add_parser(name)
        p.add_argument("--config", help="YAML config; its dates.training range is the default date range")
        p.add_argument("--start")
        p.add_argument("--end")
        p.add_argument("--overwrite", action="store_true")
        p.add_argument("--output-dir")
        p.add_argument("--output-file")
        p.add_argument("--variables", default=",".join(PRISM_VARS))
        p.add_argument("--grid-from")
        p.add_argument("--domains", default="all", help="CORDEX domains: nz,alps,sa,all")
        p.add_argument("--variant", help="Code variant from pins.json")
    sub.add_parser("status")
    args = parser.parse_args(argv)

    if args.cmd == "status":
        print(json.dumps(status_report(), indent=2))
        return 0

    if args.cmd in {"merra2", "narr", "prism", "all"}:
        dates = _dates_from_config(args.config)
        args.start = args.start or dates.get("start")
        args.end = args.end or dates.get("end")
        if not args.start or not args.end:
            raise SystemExit("--start and --end are required (YYYY-MM-DD)")

    results: dict = {}
    steps = {
        "merra2": lambda: download_merra2(args.start, args.end),
        "narr": lambda: download_narr(args.start, args.end, overwrite=args.overwrite),
        "prism": lambda: download_prism(
            args.start, args.end, Path(args.output_dir) if args.output_dir else None,
            [v.strip() for v in args.variables.split(",") if v.strip()]),
        "elevation": lambda: download_elevation(
            Path(args.output_file) if args.output_file else None,
            Path(args.grid_from) if args.grid_from else None, overwrite=args.overwrite),
        "weights": lambda: download_weights(overwrite=args.overwrite),
        "code": lambda: setup_code(args.variant, Path(args.output_dir) if args.output_dir else None),
        "env": lambda: setup_env(Path(args.output_dir) if args.output_dir else None),
        "cordex": lambda: download_cordex(
            Path(args.output_dir) if args.output_dir else C.cordex_dir(),
            [d.strip() for d in args.domains.split(",") if d.strip()], skip_existing=not args.overwrite),
    }
    order = ["merra2", "prism", "elevation", "weights"] if args.cmd == "all" else [args.cmd]
    for name in order:
        log(f"===== {name} =====")
        try:
            results[name] = steps[name]()
        except SystemExit as exc:
            results[name] = {"status": "error", "message": str(exc)}
        log(json.dumps({name: results[name]}, indent=2))

    print(json.dumps(results, indent=2))
    return 1 if any(r.get("status") != "ok" for r in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
