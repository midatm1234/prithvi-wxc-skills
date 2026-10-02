"""
Run manifests, preflight checks, and replay.

Every job gets a manifest under <state_dir>/runs/<job_id>.json recording the
resolved command, the full config text and hash, the code commit, pins, the
interpreter, and an inventory of input data. A manifest is self-contained: on
another machine, replay_run rewrites its config paths to that machine's data
root and launches the same job against the same pinned code.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

import config as C

MANIFEST_SCHEMA = "prithvi-wxc-run-manifest/1"


def runs_dir() -> Path:
    path = C.state_dir() / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_path(job_id: str) -> Path:
    return runs_dir() / f"{job_id}.json"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _plugin_version() -> Dict:
    info: Dict = {}
    try:
        info["version"] = json.loads((C._PLUGIN_ROOT / ".claude-plugin" / "plugin.json").read_text())["version"]
    except Exception:
        pass
    try:
        out = subprocess.run(["git", "-C", str(C._PLUGIN_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            info["commit"] = out.stdout.strip()
    except Exception:
        pass
    return info


def _python_info(python_bin: str) -> Dict:
    probe = (
        "import json, platform, sys\n"
        "out = {'version': platform.python_version()}\n"
        "for mod in ('torch', 'numpy', 'xarray', 'scipy', 'PrithviWxC', 'granitewxc', 'xesmf'):\n"
        "    try:\n"
        "        from importlib.metadata import version\n"
        "        out[mod] = version(mod)\n"
        "    except Exception:\n"
        "        pass\n"
        "print(json.dumps(out))\n"
    )
    try:
        res = subprocess.run([python_bin, "-c", probe], capture_output=True, text=True, timeout=60)
        info = json.loads(res.stdout) if res.returncode == 0 else {"error": res.stderr.strip()[-300:]}
    except Exception as exc:
        info = {"error": str(exc)}
    return {"path": python_bin, **info}


def repo_root_for_config(config_path: str) -> Path:
    """Configs live at <repo>/examples/<DATASET>_PRISM/<name>.yaml; relative paths in them are repo-relative."""
    path = Path(config_path).resolve()
    if len(path.parents) > 2 and path.parent.parent.name == "examples":
        return path.parents[2]
    return C._GRANITE_REPO_ROOT


def _resolve(value: Optional[str], repo_root: Path) -> Optional[Path]:
    if not value:
        return None
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (repo_root / p).resolve()


# ---------------------------------------------------------------------------
# Config localisation
# ---------------------------------------------------------------------------

def localize_config(cfg: dict, config_path: str) -> List[str]:
    """Point input paths that do not exist on this machine at the local data-root layout.

    Paths that already exist are left alone, so a shared server keeps using its
    data. Returns human-readable change notes and mutates cfg in place.
    """
    changes: List[str] = []
    repo_root = repo_root_for_config(config_path)
    data = cfg.setdefault("data", {})
    is_narr = C._detect_dataset_type(config_path) == "narr" or "narr" in str(data.get("type", "")).lower()
    targets = [
        (data, "predictor_dir", C.narr_dir() if is_narr else C.merra2_dir(), "data.predictor_dir"),
        (data, "target_dir", C.prism_dir(), "data.target_dir"),
        (data, "static_elevation_file", C.elevation_file(), "data.static_elevation_file"),
        (cfg, "path_model_weights", C.weights_file(), "path_model_weights"),
    ]
    for container, key, local, label in targets:
        current = container.get(key)
        if not current:
            continue
        resolved = _resolve(current, repo_root)
        if resolved is not None and resolved.exists():
            continue
        if str(current) != str(local):
            container[key] = str(local)
            changes.append(f"{label}: {current!r} → {str(local)!r} (localized to this machine)")
    return changes


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight(config_path: str, stage: str) -> Dict:
    """Check that the inputs a stage needs exist before a job is queued."""
    try:
        cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    except Exception as exc:
        return {"ok": False, "missing": [f"config unreadable: {exc}"]}
    repo_root = repo_root_for_config(config_path)
    data = cfg.get("data") or {}
    missing: List[str] = []
    fixes: List[str] = []

    def _need(label: str, value: Optional[str], fix: str, must_have_files: bool = False) -> None:
        p = _resolve(value, repo_root)
        if p is None:
            missing.append(f"{label} is not set in the config")
            fixes.append(fix)
        elif not p.exists() or (must_have_files and p.is_dir() and not any(p.iterdir())):
            missing.append(f"{label}: {p}")
            fixes.append(fix)

    if not repo_root.exists():
        missing.append(f"code checkout: {repo_root}")
        fixes.append("setup_code")
    if stage in {"compute_scalars", "preprocessing", "training", "inference"}:
        _need("predictors (data.predictor_dir)", data.get("predictor_dir"),
              "start_download_job dataset=merra2 (or narr)", must_have_files=True)
        _need("static elevation (data.static_elevation_file)", data.get("static_elevation_file"),
              "start_download_job dataset=elevation")
    if stage in {"compute_scalars", "preprocessing", "training"}:
        _need("PRISM targets (data.target_dir)", data.get("target_dir"),
              "start_download_job dataset=prism", must_have_files=True)
    if stage == "training" and cfg.get("backbone_use", True):
        _need("pretrained weights (path_model_weights)", cfg.get("path_model_weights"),
              "start_download_job dataset=weights")
    return {
        "ok": not missing,
        "missing": missing,
        "fix_with": sorted(set(fixes)),
        "hint": None if not missing else (
            "Run create_custom_yaml on this config to localize its paths to this machine's data root, "
            "then download whatever is still missing."
        ),
    }


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

def _elevation_check(cfg: dict, repo_root: Path) -> Dict:
    p = _resolve((cfg.get("data") or {}).get("static_elevation_file"), repo_root)
    if p is None or not p.exists():
        return {"path": str(p) if p else None, "exists": False}
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"path": str(p), "exists": True, "sha256": digest,
            "matches_reference": digest == C.PINS["data"]["elevation"]["sha256"]}


def _weights_check(cfg: dict, repo_root: Path) -> Dict:
    p = _resolve(cfg.get("path_model_weights"), repo_root)
    if p is None or not p.exists():
        return {"path": str(p) if p else None, "exists": False}
    size = p.stat().st_size
    return {"path": str(p), "exists": True, "bytes": size,
            "matches_pinned_size": size == C.PINS["weights"]["approx_bytes"]}


def write_manifest(job: Dict, python_bin: str) -> Optional[str]:
    """Write the manifest for a newly created job. Never raises: provenance must not block a run."""
    try:
        from download_pipeline_data import git_state, status_report

        config_path = job.get("config_path") or ""
        config_text = Path(config_path).read_text(encoding="utf-8") if config_path and Path(config_path).exists() else ""
        cfg = (yaml.safe_load(config_text) or {}) if config_text else {}
        repo_root = repo_root_for_config(config_path) if config_path else C._GRANITE_REPO_ROOT
        code = git_state(repo_root)
        pin = C.code_pin()
        reasons: List[str] = []
        is_download = str(job.get("type", "")).startswith("download")
        if is_download:
            pass  # downloads run the bundled downloaders; the training-code checkout is irrelevant
        elif code.get("is_git") and code.get("commit") not in {
            v["commit"] for v in C.PINS["code"]["variants"].values()
        }:
            reasons.append(f"code commit {code.get('commit', '')[:10]} is not a pinned commit")
        if not is_download and code.get("dirty_files"):
            reasons.append(f"code checkout has {len(code['dirty_files'])} uncommitted change(s)")
        if not is_download and not code.get("is_git"):
            reasons.append("code checkout is not a git repository")
        elevation = _elevation_check(cfg, repo_root) if cfg else {}
        if elevation.get("exists") and not elevation.get("matches_reference"):
            reasons.append("elevation file differs from the reference")
        weights = _weights_check(cfg, repo_root) if cfg else {}
        if weights.get("exists") and not weights.get("matches_pinned_size"):
            reasons.append("weights file size differs from the pinned file")

        python = _python_info(python_bin)
        if python.get("xesmf") and not is_download:
            reasons.append("xesmf is installed in the job interpreter; preprocessing will use xESMF "
                           "instead of the reference xarray-linear regridder")

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "job_id": job["job_id"],
            "job_type": job.get("type"),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": {"python": python_bin, "script": job.get("script"),
                        "script_relative_to_repo": _relative(job.get("script"), repo_root),
                        "args": job.get("args", [])},
            "depends_on": job.get("depends_on"),
            "config": {"path": config_path, "sha256": _sha256_text(config_text) if config_text else None,
                       "text": config_text},
            "code": {**code, "pinned_variant": pin["variant"], "pinned_commit": pin["commit"]},
            "plugin": _plugin_version(),
            "pins": C.PINS,
            "python": python,
            "platform": {"system": platform.system(), "machine": platform.machine()},
            "inputs": {"elevation": elevation, "weights": weights,
                       "data_inventory": status_report() if job.get("type", "").split("_")[0] != "download" else None},
            "reproducible": not reasons,
            "non_reproducible_reasons": reasons,
        }
        path = manifest_path(job["job_id"])
        path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        return str(path)
    except Exception as exc:  # pragma: no cover - defensive
        import logging
        logging.getLogger(__name__).warning("Could not write run manifest for %s: %s", job.get("job_id"), exc)
        return None


def _relative(path: Optional[str], root: Path) -> Optional[str]:
    if not path:
        return None
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return None


def get_run_manifest(job_id: str = "", path: str = "", include_config_text: bool = False) -> str:
    target = Path(path).expanduser() if path else manifest_path(job_id)
    if not target.exists():
        return json.dumps({"status": "error", "message": f"No manifest at {target}"})
    manifest = json.loads(target.read_text(encoding="utf-8"))
    if not include_config_text:
        manifest["config"] = {k: v for k, v in manifest.get("config", {}).items() if k != "text"}
        manifest.pop("pins", None)
    return json.dumps({"status": "ok", "manifest_path": str(target), "manifest": manifest}, default=str)


def list_run_manifests(limit: int = 20) -> str:
    rows = []
    for p in sorted(runs_dir().glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True)[:limit]:
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append({"job_id": m.get("job_id"), "job_type": m.get("job_type"), "created_utc": m.get("created_utc"),
                     "config": m.get("config", {}).get("path"), "code_commit": (m.get("code") or {}).get("commit"),
                     "reproducible": m.get("reproducible"), "path": str(p)})
    return json.dumps({"status": "ok", "runs_dir": str(runs_dir()), "runs": rows})


def replay_run(manifest_path_str: str = "", job_id: str = "", dry_run: bool = False) -> str:
    """Re-run a recorded job on this machine: same code commit, same config, local paths."""
    from job_manager import _job_manager

    src = Path(manifest_path_str).expanduser() if manifest_path_str else manifest_path(job_id)
    if not src.exists():
        return json.dumps({"status": "error", "message": f"Manifest not found: {src}"})
    m = json.loads(src.read_text(encoding="utf-8"))
    if m.get("schema") != MANIFEST_SCHEMA:
        return json.dumps({"status": "error", "message": f"Unsupported manifest schema {m.get('schema')!r}"})

    from download_pipeline_data import git_state

    job_type = m.get("job_type") or ""
    code = git_state(C._GRANITE_REPO_ROOT)
    wanted = (m.get("code") or {}).get("commit")
    warnings: List[str] = []
    # Download/setup jobs run the bundled downloaders, so the training-code commit is irrelevant.
    if not job_type.startswith("download") and code.get("commit") != wanted:
        return json.dumps({
            "status": "error",
            "message": (f"Local code is at {code.get('commit')}, the run used {wanted}. "
                        "Check out that commit (setup_code with the matching variant) before replaying."),
        })
    if code.get("dirty_files") and not job_type.startswith("download"):
        warnings.append("Local code checkout has uncommitted changes; results may differ.")

    args = list((m.get("command") or {}).get("args") or [])
    config_text = (m.get("config") or {}).get("text") or ""
    new_config = None
    changes: List[str] = []
    if config_text:
        original = (m.get("config") or {}).get("path") or ""
        dataset_dir = "NARR_PRISM" if C._detect_dataset_type(original) == "narr" else "MERRA_PRISM"
        new_config = C._GRANITE_REPO_ROOT / "examples" / dataset_dir / f"custom_replay_{m['job_id']}.yaml"
        cfg = yaml.safe_load(config_text) or {}
        changes = localize_config(cfg, str(new_config))
        if not dry_run:
            new_config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        args = [str(new_config) if i > 0 and args[i - 1] == "--config" else a for i, a in enumerate(args)]

    rel_script = (m.get("command") or {}).get("script_relative_to_repo")
    script = str(C._GRANITE_REPO_ROOT / rel_script) if rel_script else None
    if job_type.startswith("download"):
        script = str(Path(__file__).resolve().parent / "download_pipeline_data.py")
    if not script:
        return json.dumps({"status": "error", "message": "Manifest has no replayable script path"})

    plan = {"job_type": job_type, "script": script, "args": args,
            "config_path": str(new_config) if new_config else None, "config_changes": changes,
            "original_config_sha256": (m.get("config") or {}).get("sha256"), "warnings": warnings}
    if dry_run:
        return json.dumps({"status": "ok", "dry_run": True, "plan": plan})

    stage = job_type if job_type in {"compute_scalars", "preprocessing", "training", "inference"} else None
    if stage and new_config:
        check = preflight(str(new_config), stage)
        if not check["ok"]:
            return json.dumps({"status": "error", "stage": "preflight", "plan": plan, **check})

    import sys as _sys

    result = _job_manager.start_job(
        job_type, script, args, str(new_config) if new_config else "",
        python_bin=_sys.executable if job_type.startswith("download") else None,
        validate_config_outputs=not job_type.startswith("download"),
    )
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict) and parsed.get("status") == "error":
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    return json.dumps({"status": "ok", "job_id": result, "replay_of": m["job_id"], "plan": plan,
                       "message": f"Replay job '{result}' started. Compare outputs with the original run."})
