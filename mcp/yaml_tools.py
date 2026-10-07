"""
YAML config read/write tools for MERRA2 MCP.

Contains:
  - read_yaml_config  — read and summarise an existing YAML config
  - create_custom_yaml — create a patched copy with user overrides
  - list_available_configs — list all available YAML configs (built-in + custom)
"""

import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from provenance import localize_config
from config import (
    _SCRIPT_DIR,
    _DEFAULT_CONFIG,
    _NARR_SCRIPT_DIR,
    allowed_roots_message,
    path_is_allowed,
)


def _validate_write_path(p: Path) -> Optional[str]:
    """Return an error string if p is outside the allowed write roots, else None."""
    if path_is_allowed(p):
        return None
    return (
        f"Write path {p} is outside the allowed roots ({allowed_roots_message()}). "
        "Set PIPELINE_DATA_ROOT, GRANITE_WXC_REPO, or MERRA2_ALLOWED_ROOT to add a directory."
    )


def _parse_config_entry(p: Path, dataset_type: str) -> Dict:
    """Parse a YAML config file and return a summary dict."""
    entry: Dict = {
        "name": p.name,
        "path": str(p),
        "dataset_type": dataset_type,
        "is_custom": p.name.startswith("custom_"),
    }
    try:
        with open(p, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        entry["case_name"] = cfg.get("case_name")
        entry["num_epochs"] = cfg.get("num_epochs")
        entry["batch_size"] = cfg.get("batch_size")
        entry["learning_rate"] = cfg.get("learning_rate")
        entry["num_gpus"] = (cfg.get("training") or {}).get("num_gpus")
        entry["predictor_variables"] = (cfg.get("data") or {}).get("predictor_variables", {})
        entry["target_variables"] = (cfg.get("data") or {}).get("target_variables", [])
        entry["training_dates"] = (cfg.get("dates") or {}).get("training", {})
        entry["inference_dates"] = (cfg.get("dates") or {}).get("inference", {})
        entry["spatial_subset"] = (cfg.get("data") or {}).get("spatial_subset", {})
    except Exception as exc:
        entry["parse_error"] = str(exc)
    return entry


def list_available_configs() -> List[Dict]:
    """Return a list of dicts describing every YAML config in MERRA_PRISM and NARR_PRISM dirs."""
    configs = []
    for p in sorted(_SCRIPT_DIR.glob("*.yaml")):
        configs.append(_parse_config_entry(p, "merra"))
    for p in sorted(_NARR_SCRIPT_DIR.glob("*.yaml")):
        configs.append(_parse_config_entry(p, "narr"))
    return configs


def read_yaml_config(config_path: str = _DEFAULT_CONFIG) -> str:
    """Return the full contents of a YAML config as a JSON payload."""
    path = Path(config_path)
    if not path.is_absolute():
        candidate = (_SCRIPT_DIR / path.name).resolve()
        if not candidate.exists():
            candidate = (_NARR_SCRIPT_DIR / path.name).resolve()
        if candidate.exists():
            path = candidate
        else:
            return json.dumps({"status": "error", "message": f"Config not found: {config_path}"})

    if not path.exists():
        return json.dumps({"status": "error", "message": f"Config not found: {path}"})

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw_text = fh.read()
        cfg = yaml.safe_load(raw_text) or {}
        summary = {
            "config_path": str(path),
            "case_name": cfg.get("case_name"),
            "num_epochs": cfg.get("num_epochs"),
            "batch_size": cfg.get("batch_size"),
            "learning_rate": cfg.get("learning_rate"),
            "dates": cfg.get("dates", {}),
            "predictor_variables": (cfg.get("data") or {}).get("predictor_variables", {}),
            "target_variables": (cfg.get("data") or {}).get("target_variables", []),
            "spatial_subset": (cfg.get("data") or {}).get("spatial_subset", {}),
            "num_gpus": (cfg.get("training") or {}).get("num_gpus"),
            "full_yaml": raw_text,
        }
        return json.dumps({"status": "ok", "config": summary})
    except Exception as exc:
        return json.dumps({"status": "error", "message": str(exc)})


_DATE_SPLITS = ("training", "validation", "inference")


def date_range_errors(cfg: dict) -> List[str]:
    """Problems with a config's dates: training must be set, every range needs
    start <= end, and training / validation / inference must not overlap."""
    dates = cfg.get("dates") or {}
    errors: List[str] = []
    ranges = {}
    for split in _DATE_SPLITS:
        rng = dates.get(split)
        if not rng:
            if split == "training":
                errors.append("dates.training.start and dates.training.end must be set.")
            continue
        start, end = rng.get("start"), rng.get("end")
        if not start or not end:
            errors.append(f"dates.{split} needs both start and end (got start={start!r}, end={end!r}).")
            continue
        try:
            start_d = datetime.strptime(str(start), "%Y-%m-%d").date()
            end_d = datetime.strptime(str(end), "%Y-%m-%d").date()
        except ValueError:
            errors.append(f"dates.{split} must use YYYY-MM-DD (got {start!r} .. {end!r}).")
            continue
        if start_d > end_d:
            errors.append(f"dates.{split} starts after it ends ({start} > {end}).")
            continue
        ranges[split] = (start_d, end_d)
    names = [s for s in _DATE_SPLITS if s in ranges]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (a0, a1), (b0, b1) = ranges[a], ranges[b]
            if a0 <= b1 and b0 <= a1:
                errors.append(
                    f"dates.{a} ({a0}..{a1}) overlaps dates.{b} ({b0}..{b1}); "
                    "training, validation and inference ranges must be disjoint."
                )
    unknown = sorted(set(dates) - set(_DATE_SPLITS))
    if unknown:
        errors.append(f"Unknown dates sections {unknown}; use training, validation, inference.")
    return errors


def _set_override(cfg: dict, key: str, value, record) -> None:
    """Apply one extra_overrides entry. Dotted keys (dates.validation.start) address
    nested values; dict values merge into existing sections instead of replacing them."""
    *parents, leaf = key.split(".")
    node = cfg
    for part in parents:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    path = key
    if isinstance(value, dict) and isinstance(node.get(leaf), dict):
        for sub_key, sub_value in value.items():
            _set_override(node[leaf], str(sub_key), sub_value,
                          lambda p, old, new: record(f"{path}.{p}", old, new))
        return
    if record(path, node.get(leaf), value):
        node[leaf] = value


def create_custom_yaml(
    base_config: str = _DEFAULT_CONFIG,
    output_name: Optional[str] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    training_start: Optional[str] = None,
    training_end: Optional[str] = None,
    inference_start: Optional[str] = None,
    inference_end: Optional[str] = None,
    validation_start: Optional[str] = None,
    validation_end: Optional[str] = None,
    num_gpus: Optional[int] = None,
    predictor_variables: Optional[dict] = None,
    target_variables: Optional[list] = None,
    case_name: Optional[str] = None,
    extra_overrides: Optional[dict] = None,
    localize_paths: bool = True,
) -> str:
    """Create a customised copy of a base YAML config and save it to disk.

    Returns JSON with the saved path, diff of changes, and full YAML text.
    """
    base_path = Path(base_config)
    if not base_path.is_absolute():
        candidate = (_SCRIPT_DIR / base_path.name).resolve()
        if not candidate.exists():
            candidate = (_NARR_SCRIPT_DIR / base_path.name).resolve()
        if candidate.exists():
            base_path = candidate
        else:
            return json.dumps({"status": "error", "message": f"Base config not found: {base_config}"})
    if not base_path.exists():
        return json.dumps({"status": "error", "message": f"Base config not found: {base_path}"})

    # Save the custom YAML alongside its base config so it stays with the correct dataset's scripts.
    output_dir = base_path.parent

    try:
        with open(base_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as exc:
        return json.dumps({"status": "error", "message": f"Failed to load base config: {exc}"})

    base_cfg_weights = cfg.get("path_model_weights")
    base_case_name = cfg.get("case_name")
    cfg = copy.deepcopy(cfg)
    changes: list = []

    def _record_change(path: str, old_val, new_val) -> bool:
        """Apply change reporting only when values actually differ."""
        if old_val != new_val:
            changes.append(f"{path}: {old_val!r} → {new_val!r}")
            return True
        return False

    # --- spatial subset -------------------------------------------------------
    spatial_enabled = any(v is not None for v in (lat_min, lat_max, lon_min, lon_max))
    if spatial_enabled:
        cfg.setdefault("data", {}).setdefault("spatial_subset", {})
        cfg["data"]["spatial_subset"]["enabled"] = True
        for key, val in [("lat_min", lat_min), ("lat_max", lat_max),
                         ("lon_min", lon_min), ("lon_max", lon_max)]:
            if val is not None:
                old = cfg["data"]["spatial_subset"].get(key)
                if _record_change(f"data.spatial_subset.{key}", old, val):
                    cfg["data"]["spatial_subset"][key] = val

    # --- training hyper-params ------------------------------------------------
    if num_epochs is not None:
        old = cfg.get("num_epochs")
        new_val = int(num_epochs)
        if _record_change("num_epochs", old, new_val):
            cfg["num_epochs"] = new_val

    if batch_size is not None:
        old = cfg.get("batch_size")
        new_val = int(batch_size)
        if _record_change("batch_size", old, new_val):
            cfg["batch_size"] = new_val

    if learning_rate is not None:
        old = cfg.get("learning_rate")
        new_val = float(learning_rate)
        if _record_change("learning_rate", old, new_val):
            cfg["learning_rate"] = new_val

    # --- dates ---------------------------------------------------------------
    if training_start is not None:
        old = (cfg.get("dates") or {}).get("training", {}).get("start")
        if _record_change("dates.training.start", old, training_start):
            cfg.setdefault("dates", {}).setdefault("training", {})["start"] = training_start

    if training_end is not None:
        old = (cfg.get("dates") or {}).get("training", {}).get("end")
        if _record_change("dates.training.end", old, training_end):
            cfg.setdefault("dates", {}).setdefault("training", {})["end"] = training_end

    if inference_start is not None:
        old = (cfg.get("dates") or {}).get("inference", {}).get("start")
        if _record_change("dates.inference.start", old, inference_start):
            cfg.setdefault("dates", {}).setdefault("inference", {})["start"] = inference_start

    if inference_end is not None:
        old = (cfg.get("dates") or {}).get("inference", {}).get("end")
        if _record_change("dates.inference.end", old, inference_end):
            cfg.setdefault("dates", {}).setdefault("inference", {})["end"] = inference_end

    for key, val in (("start", validation_start), ("end", validation_end)):
        if val is not None:
            old = (cfg.get("dates") or {}).get("validation", {}).get(key)
            if _record_change(f"dates.validation.{key}", old, val):
                cfg.setdefault("dates", {}).setdefault("validation", {})[key] = val

    # --- distributed training ------------------------------------------------
    if num_gpus is not None:
        old = (cfg.get("training") or {}).get("num_gpus")
        new_val = int(num_gpus)
        if _record_change("training.num_gpus", old, new_val):
            cfg.setdefault("training", {})["num_gpus"] = new_val

    # --- predictor variables -------------------------------------------------
    if predictor_variables is not None:
        # Defensive normalisation: the LLM occasionally delivers predictor_variables
        # as a JSON string (e.g. '{"QV":[500,700,850]}') or as a list of pairs
        # instead of a plain dict.  Normalise to {str: [int, ...]} here.
        if isinstance(predictor_variables, str):
            import json as _json
            try:
                predictor_variables = _json.loads(predictor_variables)
            except Exception:
                return json.dumps({"status": "error", "message": f"predictor_variables could not be parsed: {predictor_variables!r}"})

        if not isinstance(predictor_variables, dict):
            return json.dumps({"status": "error", "message": f"predictor_variables must be a dict mapping variable name to list of pressure levels, got {type(predictor_variables).__name__}"})

        # Ensure every key is a string and every value is a list of ints.
        normalized: dict = {}
        for k, v in predictor_variables.items():
            key = str(k)
            if isinstance(v, (list, tuple)):
                levels = [int(x) for x in v if str(x).lstrip("-").isdigit()]
            else:
                levels = [int(v)] if str(v).lstrip("-").isdigit() else []
            if levels:
                normalized[key] = levels
        predictor_variables = normalized

        old = (cfg.get("data") or {}).get("predictor_variables", {})
        if old != predictor_variables:
            cfg.setdefault("data", {})["predictor_variables"] = predictor_variables
            new_input_vars = [f"{var}_{lvl}" for var, levels in predictor_variables.items() for lvl in levels]
            elev_and_masks = ["elev"] + [f"mask_{v}" for v in new_input_vars] + ["mask_elev"]
            full_input_vars = new_input_vars + elev_and_masks
            cfg["data"]["input_vars"] = full_input_vars
            cfg["data"]["vertical_pres_vars"] = full_input_vars
            changes.append(f"data.predictor_variables: {list(old.keys())!r} → {list(predictor_variables.keys())!r}")
            changes.append(f"data.input_vars: rebuilt with {len(full_input_vars)} entries")

    # --- target variables ----------------------------------------------------
    if target_variables is not None:
        old_targets = (cfg.get("data") or {}).get("target_variables", [])
        if _record_change("data.target_variables", old_targets, target_variables):
            cfg.setdefault("data", {})["target_variables"] = target_variables
            cfg["data"]["output_vars"] = target_variables

    # --- case name -----------------------------------------------------------
    if case_name is not None:
        old = cfg.get("case_name")
        if _record_change("case_name", old, case_name):
            cfg["case_name"] = case_name

    # --- arbitrary extra overrides -------------------------------------------
    if extra_overrides:
        for k, v in extra_overrides.items():
            _set_override(cfg, str(k), v, _record_change)

    # A new case_name is a new run: never resume another case's checkpoint (the base
    # config may pin e.g. checkpoints/narr_prism_California/last.ckpt). Auto-resume
    # still picks up this case's own checkpoint_dir/<case_name>/last.ckpt.
    if cfg.get("case_name") != base_case_name and (
        cfg.get("resume_training") or cfg.get("resume_checkpoint_path")
    ):
        changes.append(
            f"resume: not resuming {base_case_name!r}'s checkpoint "
            f"({cfg.get('resume_checkpoint_path')!r}) for new case {cfg.get('case_name')!r}"
        )
        cfg["resume_training"] = False
        cfg["resume_checkpoint_path"] = None
        cfg["auto_resume_if_checkpoint_exists"] = True

    # Fine-tuning always starts from the pretrained Prithvi weights; never let an
    # override drop them (the training code cannot initialise without them).
    base_weights = base_cfg_weights
    if not cfg.get("path_model_weights") and base_weights:
        changes.append(f"path_model_weights: kept pretrained weights {base_weights!r} (cannot be removed)")
        cfg["path_model_weights"] = base_weights

    date_errors = date_range_errors(cfg)
    if date_errors:
        return json.dumps({
            "status": "error",
            "message": "Config not written — fix the dates: " + " ".join(date_errors),
            "dates": cfg.get("dates"),
        })

    # --- machine-local input paths -------------------------------------------
    # Base configs reference the lab server's data paths; point any that do not
    # exist here at this machine's PIPELINE_DATA_ROOT layout.
    if localize_paths:
        changes.extend(localize_config(cfg, str(base_path)))

    # If customisation resolved to a no-op, reuse the base config directly.
    if not changes:
        try:
            with open(base_path, "r", encoding="utf-8") as fh:
                base_text = fh.read()
        except Exception as exc:
            return json.dumps({"status": "error", "message": f"Failed to read base config: {exc}"})

        return json.dumps({
            "status": "ok",
            "config_path": str(base_path),
            "base_config": str(base_path),
            "changes": [],
            "reused_base_config": True,
            "full_yaml": base_text,
            "message": (
                "No effective config changes were requested. "
                "Reusing the selected base YAML without creating a custom file."
            ),
        })

    # --- determine output path -----------------------------------------------
    if output_name:
        safe_name = re.sub(r"[^\w\-]", "_", output_name).strip("_") or "custom"
    else:
        safe_name = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if not safe_name.endswith(".yaml"):
        safe_name += ".yaml"
    if not safe_name.startswith("custom_"):
        safe_name = f"custom_{safe_name}"

    out_path = output_dir / safe_name

    write_err = _validate_write_path(out_path)
    if write_err:
        return json.dumps({"status": "error", "message": write_err})

    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
        with open(out_path, "r", encoding="utf-8") as fh:
            saved_text = fh.read()
    except Exception as exc:
        return json.dumps({"status": "error", "message": f"Failed to save custom YAML: {exc}"})

    return json.dumps({
        "status": "ok",
        "config_path": str(out_path),
        "base_config": str(base_path),
        "changes": changes,
        "full_yaml": saved_text,
        "message": (
            f"Custom config saved to {out_path}. "
            f"{len(changes)} parameter(s) changed from base. "
            "Pass config_path to run_training_pipeline / start_inference_job when ready."
        ),
    })
