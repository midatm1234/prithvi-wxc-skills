"""
Shared path constants for the MERRA2/NARR MCP tools.
All other modules import from here to avoid duplication.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
_USER_DATA_ROOT = Path.home() / "prithvi-wxc-data"

_GRANITE_REPO_ROOT = Path(
    os.getenv("GRANITE_WXC_REPO", "/data2/aashishp/github_merra_test/granite-wxc")
)


def _as_resolved(path: Path | str) -> Optional[Path]:
    try:
        return Path(path).expanduser().resolve()
    except Exception:
        return None


def allowed_roots() -> Tuple[Path, ...]:
    """Directories MCP may read, write, or execute under.

    Defaults are portable: this plugin, ``PIPELINE_DATA_ROOT`` or
    ``~/prithvi-wxc-data``, the user's home, and the granite-wxc checkout.
    ``MERRA2_ALLOWED_ROOT`` (os.pathsep-separated) adds extra roots; it is
    no longer required and no longer defaults to ``/data2/aashishp``.
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

    data_root = (os.getenv("PIPELINE_DATA_ROOT") or "").strip() or _USER_DATA_ROOT
    _add(Path.home())
    _add(data_root)
    _add(_PLUGIN_ROOT)
    granite = (os.getenv("GRANITE_WXC_REPO") or "").strip()
    if granite:
        _add(granite)
    elif _GRANITE_REPO_ROOT.exists():
        _add(_GRANITE_REPO_ROOT)
    script_dir = (os.getenv("GRANITE_WXC_SCRIPT_DIR") or "").strip()
    if script_dir:
        _add(script_dir)

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
    env_dir = os.getenv("GRANITE_WXC_SCRIPT_DIR")
    if env_dir:
        return Path(env_dir)
    return _GRANITE_REPO_ROOT / "examples" / "MERRA_PRISM"

_SCRIPT_DIR = _resolve_script_dir()
_NARR_SCRIPT_DIR = _GRANITE_REPO_ROOT / "examples" / "NARR_PRISM"

_DEFAULT_CONFIG = str(_SCRIPT_DIR / "MERRA_PRISM.yaml")

_GRANITE_PYTHON = os.getenv(
    "GRANITE_WXC_PYTHON",
    "/home/azureuser/miniforge3/envs/Prithvi/bin/python",
)


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
