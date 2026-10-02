"""
Environment check for a fresh machine: what is installed, what is missing, and
which tool call fixes each gap, in order.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import config as C

MCP_MODULES = ("yaml", "numpy", "xarray", "netCDF4", "matplotlib", "requests", "tqdm", "earthaccess", "boto3")


def _training_python_probe(python_bin: str) -> Dict:
    if not Path(python_bin).exists() and not shutil.which(python_bin):
        return {"path": python_bin, "exists": False}
    probe = (
        "import json\n"
        "out = {}\n"
        "for mod in ('torch', 'granitewxc', 'PrithviWxC', 'xarray', 'h5netcdf', 'xesmf'):\n"
        "    try:\n"
        "        m = __import__(mod)\n"
        "        out[mod] = getattr(m, '__version__', 'installed')\n"
        "    except Exception:\n"
        "        out[mod] = None\n"
        "try:\n"
        "    import torch\n"
        "    out['cuda_available'] = torch.cuda.is_available()\n"
        "    out['cuda_devices'] = torch.cuda.device_count()\n"
        "except Exception:\n"
        "    pass\n"
        "print(json.dumps(out))\n"
    )
    try:
        res = subprocess.run([python_bin, "-c", probe], capture_output=True, text=True, timeout=120)
        info = json.loads(res.stdout.strip().splitlines()[-1]) if res.returncode == 0 else {"error": res.stderr[-400:]}
    except Exception as exc:
        info = {"error": str(exc)}
    return {"path": python_bin, "exists": True, **info}


def _gpus() -> List[Dict]:
    if not shutil.which("nvidia-smi"):
        return []
    res = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    gpus = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            gpus.append({"index": int(parts[0]), "name": parts[1],
                         "memory_total_mib": int(parts[2]), "memory_used_mib": int(parts[3])})
    return gpus


def check_environment(include_training_python: bool = True) -> str:
    from download_pipeline_data import status_report

    root = C.data_root()
    root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(root).free / 1e9
    data = status_report()
    code = data["code"]
    pin = C.code_pin()
    training = _training_python_probe(C._GRANITE_PYTHON) if include_training_python else {"skipped": True}
    gpus = _gpus()

    steps: List[Dict] = []
    warnings: List[str] = []

    missing_mcp = [m for m in MCP_MODULES if importlib.util.find_spec(m) is None]
    if missing_mcp:
        steps.append({"do": "Install MCP requirements", "detail": f"missing {missing_mcp}",
                      "how": "Restart the MCP via bin/prithvi-mcp (it installs requirements), or pip install -r requirements.txt"})

    if not code.get("is_git"):
        steps.append({"do": "Clone the pinned training/inference code", "tool": "setup_code",
                      "detail": f"{pin['repo']} @ {pin['commit'][:10]} ({pin['variant']}) -> {code['path']}"})
    elif code.get("commit") != pin["commit"]:
        warnings.append(f"Code checkout is at {code.get('commit', '')[:10]}, pin is {pin['commit'][:10]} ({pin['variant']}). "
                        "Results will not match the reference unless you run setup_code.")
    if code.get("dirty_files"):
        warnings.append(f"Code checkout has {len(code['dirty_files'])} uncommitted change(s); runs will be marked non-reproducible.")

    if include_training_python:
        if not training.get("exists") or not training.get("torch") or not training.get("granitewxc"):
            steps.append({"do": "Create the training environment", "tool": "setup_training_env",
                          "detail": f"{C.training_venv_python()} from env/training-requirements.lock.txt"})
        if training.get("xesmf"):
            warnings.append("xesmf is installed in the training env; preprocessing will differ from the reference "
                            "(which used the xarray-linear regridder). Uninstall xesmf for identical results.")
        if training.get("torch") and not training.get("cuda_available"):
            warnings.append("PyTorch cannot see a CUDA GPU; training and inference need NVIDIA GPUs.")

    if not data["auth"]["earthdata_env"] and not data["auth"]["netrc"]:
        steps.append({"do": "Provide NASA Earthdata credentials (MERRA-2 only)",
                      "detail": "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD, or add urs.earthdata.nasa.gov to ~/.netrc. "
                                "Register free at https://urs.earthdata.nasa.gov/"})

    if not data["merra2"]["files"] and not data["narr"]["files"]:
        steps.append({"do": "Download predictors", "tool": "start_download_job",
                      "args": {"dataset": "merra2", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}})
    if not any(v["files"] for v in data["prism"].values()):
        steps.append({"do": "Download PRISM targets", "tool": "start_download_job",
                      "args": {"dataset": "prism", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}})
    if not data["elevation"]["exists"]:
        steps.append({"do": "Fetch static orography", "tool": "start_download_job", "args": {"dataset": "elevation"}})
    if not data["weights"]["complete"]:
        steps.append({"do": "Download pretrained weights (~17.4 GB)", "tool": "start_download_job",
                      "args": {"dataset": "weights"}})
        if free_gb < 25:
            warnings.append(f"Only {free_gb:.0f} GB free under {root}; weights alone need ~17.4 GB.")
    if include_training_python and not gpus:
        warnings.append("No NVIDIA GPU detected (nvidia-smi). Downloads work here; training/inference need a GPU host.")

    if not C.PINS["data"]["elevation"].get("url"):
        warnings.append("No reference elevation URL is pinned yet; elevation falls back to ETOPO and will not be bit-identical.")

    return json.dumps({
        "status": "ok",
        "ready_for_training": not steps,
        "data_root": str(root),
        "state_dir": str(C.state_dir()),
        "free_disk_gb": round(free_gb, 1),
        "code": {**code, "pinned": pin},
        "training_python": training,
        "gpus": gpus,
        "data": data,
        "next_steps": steps,
        "warnings": warnings,
        "env_overrides": {k: os.environ[k] for k in (
            "PIPELINE_DATA_ROOT", "GRANITE_WXC_REPO", "GRANITE_WXC_PYTHON", "GRANITE_WXC_VARIANT",
            "MERRA2_DATA_DIR", "NARR_DATA_DIR", "PRISM_DATA_DIR", "ELEVATION_FILE", "MODEL_WEIGHTS_FILE",
        ) if os.environ.get(k)},
    }, default=str)
