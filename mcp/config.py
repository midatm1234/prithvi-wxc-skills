"""
Shared paths and pins for the PrithviWxC downscaling MCP tools.

Everything resolves from environment variables with portable defaults, so the
same plugin behaves the same on any machine. Nothing here may default to a
path that only exists on one host.

Data-root layout (PIPELINE_DATA_ROOT, default ~/prithvi-wxc-data):

    merra2/daily_subset_with_H/      MERRA-2 predictors (daily, subset, with H)
    narr/subset/                     NARR predictors
    prism/prism_daily_800m_an/{ppt,tmax,tmin}/YYYY/*.nc   PRISM targets
    static/prism_elevation.nc        800 m orography on the PRISM grid
    weights/<hf_repo>/<file>         pretrained Prithvi WxC downscaling weights
    cordex/                          CORDEX-ML-Bench (optional)
    code/Prithvi-UNet-stocahstic/    pinned training/inference code
    envs/prithvi/                    training virtualenv (optional)
    .mcp-state/                      jobs, logs, run manifests, plot artifacts
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

_MCP_DIR = Path(__file__).resolve().parent
_PLUGIN_ROOT = _MCP_DIR.parent
PINS_FILE = _MCP_DIR / "pins.json"


def load_pins() -> dict:
    with PINS_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


PINS = load_pins()


def _env_path(name: str) -> Optional[Path]:
    value = (os.getenv(name) or "").strip()
    return Path(value).expanduser() if value else None


def data_root() -> Path:
    return (_env_path("PIPELINE_DATA_ROOT") or Path.home() / "prithvi-wxc-data").resolve()


def state_dir() -> Path:
    """Writable MCP state (jobs, logs, manifests, plots). Never inside the plugin,
    which hosts may install read-only or replace on update."""
    path = _env_path("PRITHVI_MCP_STATE_DIR") or data_root() / ".mcp-state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def code_variant() -> str:
    return (os.getenv("GRANITE_WXC_VARIANT") or PINS["code"]["default_variant"]).strip()


def code_pin(variant: Optional[str] = None) -> dict:
    variants = PINS["code"]["variants"]
    name = variant or code_variant()
    if name not in variants:
        raise ValueError(f"Unknown code variant {name!r}; choose one of {sorted(variants)}")
    return {"repo": PINS["code"]["repo"], "variant": name, **variants[name]}


def default_granite_repo() -> Path:
    return data_root() / "code" / "Prithvi-UNet-stocahstic"


# Data layout --------------------------------------------------------------
# Each location can be overridden to reuse data that already exists elsewhere
# (e.g. MERRA2_DATA_DIR=/data/merra2/daily_subset_with_H on a shared server).

def merra2_dir() -> Path:
    return _env_path("MERRA2_DATA_DIR") or data_root() / "merra2" / "daily_subset_with_H"


def narr_dir() -> Path:
    return _env_path("NARR_DATA_DIR") or data_root() / "narr" / "subset"


def prism_dir() -> Path:
    return _env_path("PRISM_DATA_DIR") or data_root() / "prism" / "prism_daily_800m_an"


def elevation_file() -> Path:
    return _env_path("ELEVATION_FILE") or data_root() / "static" / "prism_elevation.nc"


def weights_file() -> Path:
    w = PINS["weights"]
    return _env_path("MODEL_WEIGHTS_FILE") or data_root() / "weights" / w["hf_repo"] / w["file"]


def cordex_dir() -> Path:
    return data_root() / "cordex"


def training_venv_python() -> Path:
    return data_root() / "envs" / "prithvi" / "bin" / "python"


# Code + interpreter --------------------------------------------------------

_GRANITE_REPO_ROOT = (_env_path("GRANITE_WXC_REPO") or default_granite_repo()).resolve()


def _resolve_granite_python() -> str:
    env = (os.getenv("GRANITE_WXC_PYTHON") or "").strip()
    if env:
        return env
    venv_python = training_venv_python()
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


_GRANITE_PYTHON = _resolve_granite_python()


def _as_resolved(path: Path | str) -> Optional[Path]:
    try:
        return Path(path).expanduser().resolve()
    except Exception:
        return None


def allowed_roots() -> Tuple[Path, ...]:
    """Directories MCP may read, write, or execute under.

    Defaults: the data root, the code checkout, this plugin, and the user's
    home. ``MERRA2_ALLOWED_ROOT`` (os.pathsep-separated) adds extra roots,
    e.g. a shared ``/data`` volume that already holds downloads.
    """
    roots: list[Path] = []

    def _add(path: Optional[Path | str]) -> None:
        resolved = _as_resolved(path) if path else None
        if resolved is None:
            return
        for existing in roots:
            try:
                resolved.relative_to(existing)
                return
            except ValueError:
                pass
        roots.append(resolved)

    _add(Path.home())
    _add(data_root())
    _add(_PLUGIN_ROOT)
    _add(_GRANITE_REPO_ROOT)
    _add(_env_path("GRANITE_WXC_SCRIPT_DIR"))
    _add(_env_path("PRITHVI_MCP_STATE_DIR"))

    extra = (os.getenv("MERRA2_ALLOWED_ROOT") or "").strip()
    if extra:
        for part in extra.split(os.pathsep):
            _add(part)

    return tuple(roots)


def path_is_allowed(path: Path | str) -> bool:
    resolved = _as_resolved(path)
    if resolved is None:
        return False
    for root in allowed_roots():
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def allowed_roots_message() -> str:
    return ", ".join(str(root) for root in allowed_roots())


def _resolve_script_dir() -> Path:
    return _env_path("GRANITE_WXC_SCRIPT_DIR") or _GRANITE_REPO_ROOT / "examples" / "MERRA_PRISM"


_SCRIPT_DIR = _resolve_script_dir()
_NARR_SCRIPT_DIR = _GRANITE_REPO_ROOT / "examples" / "NARR_PRISM"

_DEFAULT_CONFIG = str(_SCRIPT_DIR / "MERRA_PRISM.yaml")


def _detect_dataset_type(config_path: str) -> str:
    """Return 'narr' or 'merra' by inspecting the YAML config.

    Resolution order:
      1. Parent directory name contains 'narr_prism' or 'merra_prism'
      2. data.type field inside the YAML
      3. Config filename contains 'narr' or 'merra'
      4. Default: 'merra'
    """
    path = Path(config_path)

    # 1. Directory-based hint (most reliable for standard layouts)
    for part in path.parts:
        pl = part.lower()
        if "narr_prism" in pl:
            return "narr"
        if "merra_prism" in pl:
            return "merra"

    # 2. Parse YAML data.type
    if _YAML_AVAILABLE and path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                cfg = _yaml.safe_load(fh) or {}
            data_type = (cfg.get("data") or {}).get("type", "")
            if "narr" in data_type.lower():
                return "narr"
            if "merra" in data_type.lower():
                return "merra"
        except Exception:
            pass

    # 3. Filename hint
    if "narr" in path.name.lower():
        return "narr"

    return "merra"


def _script_dir_for_type(dataset_type: str) -> Path:
    """Return the examples script directory for 'narr' or 'merra'."""
    if dataset_type == "narr":
        return _NARR_SCRIPT_DIR
    return _SCRIPT_DIR
