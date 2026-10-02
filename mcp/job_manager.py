"""
Job management for MERRA2 MCP tools.

Contains:
  - JobManager  — launch, monitor, queue, and cancel subprocess jobs
  - run_training_pipeline (only training entrypoint), start_inference_job,
    start_preprocessing_job, start_compute_scalars_job
  - start_training_job is internal-only (used by run_training_pipeline)
  - Module-level _job_manager singleton
"""

import json
import logging
import os
import shlex
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    _SCRIPT_DIR,
    _NARR_SCRIPT_DIR,
    _DEFAULT_CONFIG,
    _GRANITE_PYTHON,
    _detect_dataset_type,
    _script_dir_for_type,
    allowed_roots_message,
    path_is_allowed,
    state_dir,
)
from provenance import preflight, write_manifest

logger = logging.getLogger(__name__)


class JobManager:
    """Launch and track background subprocess jobs (training, inference, etc.)."""

    # State lives under the data root, not the plugin: hosts may install plugins
    # read-only and replace them on update.
    STATUS_DIR = state_dir() / "jobs"
    JOBS_FILE = STATUS_DIR / "jobs.json"
    LOG_DIR = state_dir() / "logs"
    _GPU_CHECK_INTERVAL = 30  # seconds between GPU availability polls
    # Fraction of GPU memory that must be free for a GPU to be considered available
    _GPU_FREE_THRESHOLD = 0.2
    # A GPU is only "idle" if compute utilization is also below this (percent) —
    # a GPU can be compute-bound with low memory use and still be fully busy.
    _GPU_UTIL_IDLE_MAX = 20
    _RECONCILER_INTERVAL = 30  # seconds between background reconciler passes

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, dict] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._pending_queue: List[dict] = []   # jobs waiting for GPU
        self._dep_waiting: Dict[str, dict] = {}  # job_id -> pending dict; waiting for a dependency
        self._watcher_active = False
        self._reconciler_active = False
        self.JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.STATUS_DIR.mkdir(parents=True, exist_ok=True)
        # Keep default MERRA log directory available; per-job directories are
        # resolved dynamically from the selected config path.
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._load_jobs()
        self._ensure_reconciler()

    def _log_dir_for_config(self, config_path: str) -> Path:
        """Return the log directory for the dataset type of config_path."""
        return self.LOG_DIR / (_detect_dataset_type(config_path) if config_path else "misc")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_jobs(self) -> None:
        if not self.JOBS_FILE.exists():
            return
        try:
            with self.JOBS_FILE.open() as f:
                self._jobs = json.load(f)
            for job_id in list(self._jobs.keys()):
                self._reconcile_job_state(job_id, restarting=True)

            # Restore in-memory queues so queued jobs survive server restart.
            for job_id, job in self._jobs.items():
                status = job.get("status")
                if status == "waiting_for_gpu":
                    pending = self._reconstruct_pending(job)
                    if pending is None:
                        job["status"] = "cancelled"
                        job["note"] = "Could not reconstruct queued job on restart — resubmit"
                        self._write_status_file(job_id, "cancelled", note=job["note"])
                        continue
                    self._pending_queue.append(pending)
                    job["note"] = "Recovered queued state after server restart"
                elif status == "waiting_for_job":
                    pending = self._reconstruct_pending(job)
                    if pending is None:
                        job["status"] = "cancelled"
                        job["note"] = "Could not reconstruct queued job on restart — resubmit"
                        self._write_status_file(job_id, "cancelled", note=job["note"])
                        continue
                    self._dep_waiting[job_id] = pending
                    job["note"] = "Recovered dependency-wait state after server restart"

            self._save_jobs()
            if self._pending_queue:
                self._ensure_watcher()
        except Exception:
            self._jobs = {}

    def _reconstruct_pending(self, job: dict) -> Optional[dict]:
        """Rebuild the in-memory `pending` dict for a queued job after restart."""
        script = job.get("script") or self._reconstruct_script_path(
            job.get("type"), job.get("config_path")
        )
        if not script:
            return None
        env = os.environ.copy()
        try:
            repo_root = str(Path(script).resolve().parents[2])
            script_dir = str(Path(script).resolve().parent)
            env["PYTHONPATH"] = ":".join(
                filter(None, [repo_root, script_dir, env.get("PYTHONPATH", "")])
            )
        except Exception:
            pass
        return {
            "job_id": job["job_id"],
            "job_type": job.get("type", ""),
            "script": script,
            "args": job.get("args", []) or [],
            "config_path": job.get("config_path", ""),
            "log_file": job.get("log_file", ""),
            "env": env,
            "depends_on": job.get("depends_on"),
        }

    def _reconstruct_script_path(self, job_type: Optional[str], config_path: Optional[str]) -> Optional[str]:
        if job_type and job_type.startswith("download"):
            return str(Path(__file__).resolve().parent / "download_pipeline_data.py")
        if not job_type or not config_path:
            return None
        try:
            dataset_type = _detect_dataset_type(config_path)
        except Exception:
            return None
        script_dir = _script_dir_for_type(dataset_type)
        prefix = "narr" if dataset_type == "narr" else "merra"
        mapping = {
            "training":        f"{prefix}_prism_finetune.py",
            "inference":       f"{prefix}_prism_inference.py",
            "preprocessing":   f"preproc_{prefix}_prism.py",
            "compute_scalars": f"compute_scalars_{prefix}_prism.py",
        }
        name = mapping.get(job_type)
        if not name:
            return None
        return str(script_dir / name)

    def _save_jobs(self) -> None:
        try:
            with self.JOBS_FILE.open("w") as f:
                json.dump(self._jobs, f, indent=2, default=str)
        except Exception:
            pass

    def _status_file_path(self, job_id: str) -> Path:
        return self.STATUS_DIR / f"job_{job_id}.status.json"

    def _wrapper_script_path(self, job_id: str) -> Path:
        return self.STATUS_DIR / f"job_{job_id}.sh"

    def _write_status_file(self, job_id: str, status: str, return_code: Optional[int] = None, note: Optional[str] = None) -> None:
        payload = {
            "job_id": job_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if return_code is not None:
            payload["return_code"] = return_code
        if note:
            payload["note"] = note
        try:
            self._status_file_path(job_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _read_status_file(self, job_id: str) -> Optional[dict]:
        p = self._status_file_path(job_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _infer_terminal_from_log(self, log_path: str) -> Optional[tuple[str, Optional[int]]]:
        path = Path(log_path)
        if not path.exists():
            return None
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return None
        for line in reversed(lines[-300:]):
            if "[JOB_STATUS] CMD_EXIT" in line and "rc=" in line:
                try:
                    rc = int(line.rsplit("rc=", 1)[-1].strip().split()[0])
                except Exception:
                    rc = None
                if rc is None:
                    return None
                return ("done" if rc == 0 else "failed", rc)
            if "[JOB_STATUS] CANCELLED" in line:
                return ("cancelled", None)
        return None

    def _is_pid_alive(self, pid: Optional[int]) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    def _reconcile_job_state(self, job_id: str, restarting: bool = False) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return

        status_file = self._read_status_file(job_id)
        sf_status = (status_file or {}).get("status")
        sf_rc = (status_file or {}).get("return_code")

        if sf_status in {"done", "failed", "cancelled", "blocked"}:
            job["status"] = sf_status
            if sf_rc is not None:
                job["return_code"] = sf_rc
            if not job.get("end_time"):
                job["end_time"] = datetime.utcnow().isoformat()
            return

        inferred = self._infer_terminal_from_log(job.get("log_file", ""))
        if inferred is not None:
            terminal_status, rc = inferred
            job["status"] = terminal_status
            if rc is not None:
                job["return_code"] = rc
            if not job.get("end_time"):
                job["end_time"] = datetime.utcnow().isoformat()
            self._write_status_file(job_id, terminal_status, return_code=rc)
            return

        if job.get("status") in {"running", "unknown"}:
            if self._is_pid_alive(job.get("pid")):
                job["status"] = "running"
                if restarting:
                    job["note"] = "Recovered running state after server restart"
            else:
                if sf_status == "running":
                    job["status"] = "unknown"
                    if restarting:
                        job["note"] = "Server restarted and process could not be verified"
                elif restarting:
                    job["status"] = "unknown"
                    job["note"] = "Server restarted while job state was non-terminal"

    def _reconcile_all_jobs(self) -> None:
        with self._lock:
            for job_id in list(self._jobs.keys()):
                self._reconcile_job_state(job_id)
            self._save_jobs()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _monitor(self, job_id: str, proc: subprocess.Popen) -> None:
        """Background thread: wait for process exit, update status, and unblock dependents."""
        returncode = proc.wait()
        succeeded = returncode == 0
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job["status"] not in ("cancelled",):
                job["status"] = "done" if succeeded else "failed"
                job["return_code"] = returncode
                job["end_time"] = datetime.utcnow().isoformat()
                self._write_status_file(job_id, job["status"], return_code=returncode)
            self._save_jobs()
        self._handle_job_terminal(job_id, succeeded, returncode)

    def _handle_job_terminal(self, job_id: str, succeeded: bool, returncode: Optional[int]) -> None:
        """Unblock jobs waiting on `job_id`. Safe to call more than once — a
        second call finds no dependents and is a no-op."""
        with self._lock:
            unblocked = []
            blocked_ids = []
            for dep_job_id, pending in list(self._dep_waiting.items()):
                if pending.get("depends_on") == job_id:
                    if succeeded:
                        unblocked.append((dep_job_id, pending))
                    else:
                        blocked_ids.append(dep_job_id)
                    del self._dep_waiting[dep_job_id]

            for dep_job_id, pending in unblocked:
                dep_job = self._jobs.get(dep_job_id)
                if dep_job and dep_job["status"] == "waiting_for_job":
                    if self._requires_gpu(pending["job_type"]):
                        dep_job["status"] = "waiting_for_gpu"
                        dep_job["note"] = f"Dependency '{job_id}' completed — waiting for GPU"
                        self._pending_queue.append(pending)
                    else:
                        dep_job["status"] = "running"
                        dep_job["note"] = f"Dependency '{job_id}' completed — starting now"

            for dep_job_id in blocked_ids:
                dep_job = self._jobs.get(dep_job_id)
                if dep_job:
                    dep_job["status"] = "blocked"
                    dep_job["note"] = (
                        f"Dependency '{job_id}' failed (exit code {returncode}) — "
                        "this job will not run. Resubmit when ready."
                    )

            self._save_jobs()

        for dep_job_id, pending in unblocked:
            if self._requires_gpu(pending["job_type"]):
                continue
            logger.info(f"Dependency satisfied — starting job {dep_job_id} ({pending['job_type']})")
            self._launch_subprocess(
                dep_job_id,
                pending["job_type"],
                pending["script"],
                pending["args"],
                pending["config_path"],
                pending["log_file"],
                pending["env"],
                python_bin=pending.get("python_bin"),
            )

        if any(self._requires_gpu(pending["job_type"]) for _, pending in unblocked):
            self._ensure_watcher()

    def _tail_log(self, log_path: str, n: int = 30) -> List[str]:
        path = Path(log_path)
        if not path.exists():
            return []
        try:
            with path.open() as f:
                lines = f.readlines()
            return [ln.rstrip() for ln in lines[-n:]]
        except Exception:
            return []

    def _requires_gpu(self, job_type: str) -> bool:
        return job_type in {"training", "inference"}

    @staticmethod
    def _parse_num_gpus(args: List[str]) -> int:
        """Extract the requested GPU count from a job's CLI args (--num-gpus N)."""
        for i, a in enumerate(args):
            if a == "--num-gpus" and i + 1 < len(args):
                try:
                    return max(1, int(args[i + 1]))
                except ValueError:
                    return 1
        return 1

    def _free_gpu_indices(self) -> List[int]:
        """Return indices of physical GPUs that are both memory- and compute-idle."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:
            return []
        if result.returncode != 0:
            return []
        free = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            idx, used, total, util = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            mem_ok = total > 0 and (used / total) <= (1.0 - self._GPU_FREE_THRESHOLD)
            util_ok = util <= self._GPU_UTIL_IDLE_MAX
            if mem_ok and util_ok:
                free.append(idx)
        return free

    def _claimed_gpu_indices(self) -> set:
        """GPU indices already pinned to jobs this manager currently has running."""
        claimed = set()
        with self._lock:
            for job in self._jobs.values():
                if job.get("status") == "running":
                    claimed.update(job.get("gpu_indices") or [])
        return claimed

    def _select_free_gpus(self, num_gpus: int) -> Optional[List[int]]:
        """Pick `num_gpus` idle physical GPU indices, or None if not enough are free."""
        claimed = self._claimed_gpu_indices()
        free = [i for i in self._free_gpu_indices() if i not in claimed]
        if len(free) < num_gpus:
            return None
        return free[:num_gpus]

    def _gpus_available(self, num_gpus: int = 1) -> bool:
        """Return True if at least `num_gpus` physical GPUs are currently idle."""
        return self._select_free_gpus(num_gpus) is not None

    def _gpu_status_summary(self) -> str:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                lines = []
                for line in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 4:
                        idx, used, total, util = parts
                        pct = int(used) * 100 // int(total) if int(total) else 0
                        lines.append(f"GPU {idx}: {used}/{total} MiB ({pct}% mem, {util}% util)")
                return "; ".join(lines)
        except Exception:
            pass
        return "GPU info unavailable"

    def _launch_subprocess(
        self,
        job_id: str,
        job_type: str,
        script: str,
        args: List[str],
        config_path: str,
        log_file: str,
        env: dict,
        gpu_indices: Optional[List[int]] = None,
        python_bin: Optional[str] = None,
    ) -> None:
        """Actually spawn the subprocess and register it. Called with lock NOT held."""
        if gpu_indices:
            env = dict(env)
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["gpu_indices"] = gpu_indices
        quoted_args = " ".join(shlex.quote(str(a)) for a in args)
        wrapper_path = self._wrapper_script_path(job_id)
        status_file = self._status_file_path(job_id)
        wrapper_contents = "\n".join([
            "#!/usr/bin/env bash",
            "set -u",
            f"JOB_ID={shlex.quote(job_id)}",
            f"STATUS_FILE={shlex.quote(str(status_file))}",
            f"LOG_FILE={shlex.quote(log_file)}",
            f"PYTHON_BIN={shlex.quote(python_bin or _GRANITE_PYTHON)}",
            f"SCRIPT_PATH={shlex.quote(script)}",
            "timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }",
            "write_status() {",
            "  local st=\"$1\"",
            "  local rc=\"${2-}\"",
            "  local ts=\"$(timestamp)\"",
            "  local rc_field=\"null\"",
            "  if [[ -n \"$rc\" ]]; then rc_field=\"$rc\"; fi",
            "  cat > \"$STATUS_FILE\" <<JSON",
            "{",
            "  \"job_id\": \"$JOB_ID\",",
            "  \"status\": \"$st\",",
            "  \"timestamp\": \"$ts\",",
            "  \"pid\": $$,",
            "  \"return_code\": $rc_field",
            "}",
            "JSON",
            "}",
            "mkdir -p \"$(dirname \"$LOG_FILE\")\"",
            "echo \"[JOB_STATUS] START $(timestamp) job=$JOB_ID\" >> \"$LOG_FILE\"",
            "write_status running",
            "trap 'echo \"[JOB_STATUS] CANCELLED $(timestamp) job=$JOB_ID\" >> \"$LOG_FILE\"; write_status cancelled 143; exit 143' TERM INT",
            "echo \"[JOB_STATUS] CMD_START $(timestamp)\" >> \"$LOG_FILE\"",
            "set +e",
            f"nohup \"$PYTHON_BIN\" \"$SCRIPT_PATH\" {quoted_args} >> \"$LOG_FILE\" 2>&1",
            "rc=$?",
            "set -e",
            "echo \"[JOB_STATUS] CMD_EXIT $(timestamp) rc=$rc\" >> \"$LOG_FILE\"",
            "if [[ $rc -eq 0 ]]; then",
            "  write_status done $rc",
            "else",
            "  write_status failed $rc",
            "fi",
            "exit $rc",
            "",
        ])
        try:
            wrapper_path.write_text(wrapper_contents, encoding="utf-8")
            wrapper_path.chmod(0o755)
        except Exception as e:
            with self._lock:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["note"] = f"Failed to prepare wrapper script: {e}"
                self._save_jobs()
            return

        with self._lock:
            try:
                proc = subprocess.Popen(
                    ["/usr/bin/env", "bash", str(wrapper_path)],
                    cwd=str(Path(script).parent),
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self._processes[job_id] = proc
                self._jobs[job_id]["status"] = "running"
                self._jobs[job_id]["pid"] = proc.pid
                self._jobs[job_id]["status_file"] = str(status_file)
                self._jobs[job_id]["wrapper_script"] = str(wrapper_path)
                self._jobs[job_id].pop("note", None)
                self._write_status_file(job_id, "running")
                self._save_jobs()
            except Exception as e:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["note"] = str(e)
                self._write_status_file(job_id, "failed", note=str(e))
                self._save_jobs()
                return

        threading.Thread(
            target=self._monitor, args=(job_id, proc), daemon=True
        ).start()

    def _gpu_watcher_loop(self) -> None:
        import time
        while True:
            time.sleep(self._GPU_CHECK_INTERVAL)
            with self._lock:
                # Drop any cancelled jobs sitting at the head of the queue.
                while self._pending_queue:
                    head_id = self._pending_queue[0]["job_id"]
                    if self._jobs.get(head_id, {}).get("status") == "cancelled":
                        self._pending_queue.pop(0)
                    else:
                        break
                if not self._pending_queue:
                    self._watcher_active = False
                    return
                pending = self._pending_queue[0]

            num_gpus = self._parse_num_gpus(pending["args"])
            gpu_indices = self._select_free_gpus(num_gpus)
            if gpu_indices is None:
                continue

            with self._lock:
                if not self._pending_queue or self._pending_queue[0]["job_id"] != pending["job_id"]:
                    continue
                pending = self._pending_queue.pop(0)

            job_id = pending["job_id"]
            logger.info(
                f"GPU(s) {gpu_indices} available — starting queued job {job_id} ({pending['job_type']})"
            )
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["note"] = f"GPU(s) {gpu_indices} became available — starting now"
            self._launch_subprocess(
                job_id, pending["job_type"], pending["script"],
                pending["args"], pending["config_path"], pending["log_file"],
                pending["env"], gpu_indices=gpu_indices,
                python_bin=pending.get("python_bin"),
            )

    def _ensure_watcher(self) -> None:
        with self._lock:
            if not self._watcher_active:
                self._watcher_active = True
                threading.Thread(target=self._gpu_watcher_loop, daemon=True).start()

    def _reconciler_loop(self) -> None:
        """Background thread: detects running→terminal transitions for jobs whose
        in-process `_monitor` was lost (e.g., across a server restart), and
        unblocks their dependents so the pipeline can keep flowing."""
        import time
        while True:
            time.sleep(self._RECONCILER_INTERVAL)
            try:
                with self._lock:
                    active_ids = [
                        jid for jid, j in self._jobs.items()
                        if j.get("status") in ("running", "unknown")
                        and jid not in self._processes  # skip jobs owned by an in-process _monitor
                    ]
                for job_id in active_ids:
                    with self._lock:
                        prev = self._jobs.get(job_id, {}).get("status")
                    self._reconcile_job_state(job_id, restarting=False)
                    with self._lock:
                        job = self._jobs.get(job_id, {})
                        new = job.get("status")
                        rc = job.get("return_code")
                    if prev in ("running", "unknown") and new in ("done", "failed", "cancelled"):
                        self._handle_job_terminal(job_id, new == "done", rc)
                with self._lock:
                    self._save_jobs()
            except Exception as e:
                logger.warning(f"reconciler pass failed: {e}")

    def _ensure_reconciler(self) -> None:
        with self._lock:
            if not self._reconciler_active:
                self._reconciler_active = True
                threading.Thread(target=self._reconciler_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # Path validation
    # ------------------------------------------------------------------

    def _validate_path(self, p: str, label: str) -> Optional[str]:
        try:
            resolved = Path(p).expanduser().resolve()
        except Exception:
            return f"{label} path is invalid: {p!r}"
        if path_is_allowed(resolved):
            return None
        return (
            f"{label} path {resolved} is outside the allowed roots "
            f"({allowed_roots_message()}). Set PIPELINE_DATA_ROOT, "
            "GRANITE_WXC_REPO, or MERRA2_ALLOWED_ROOT to add a directory."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_job(
        self,
        job_type: str,
        script: str,
        args: List[str],
        config_path: str,
        depends_on: Optional[str] = None,
        python_bin: Optional[str] = None,
        validate_config_outputs: bool = True,
    ) -> str:
        """Launch (or queue) a subprocess job and return its job_id."""
        if config_path and not Path(config_path).is_absolute():
            candidate = (_SCRIPT_DIR / config_path).resolve()
            if not candidate.exists():
                candidate = (_NARR_SCRIPT_DIR / config_path).resolve()
            if not candidate.exists():
                candidate2 = (_SCRIPT_DIR / Path(config_path).name).resolve()
                if not candidate2.exists():
                    candidate2 = (_NARR_SCRIPT_DIR / Path(config_path).name).resolve()
                if candidate2.exists():
                    candidate = candidate2
            config_path = str(candidate)
            args = [config_path if (i > 0 and args[i - 1] == "--config") else a
                    for i, a in enumerate(args)]

        for label, p in [("script", script), ("config", config_path)]:
            if not p:
                continue
            err = self._validate_path(p, label)
            if err:
                return json.dumps({"status": "error", "message": err})

        if depends_on:
            with self._lock:
                dep = self._jobs.get(depends_on)
            if not dep:
                return json.dumps({"status": "error", "message": f"depends_on job '{depends_on}' not found"})
            if dep["status"] in ("done",):
                return json.dumps({"status": "error", "message": f"depends_on job '{depends_on}' already completed — start the job directly"})
            if dep["status"] in ("failed", "cancelled", "blocked"):
                return json.dumps({"status": "error", "message": f"depends_on job '{depends_on}' has status '{dep['status']}' — cannot chain from it"})

        if validate_config_outputs:
            try:
                import yaml as _yaml
                with open(config_path, "r", encoding="utf-8") as _f:
                    _cfg = _yaml.safe_load(_f) or {}
                _data = _cfg.get("data", {})
                _repo_root = Path(script).resolve().parents[2]
                _output_paths = {
                    "data.preprocessed_dir": _data.get("preprocessed_dir"),
                    "data.scalar_dir":       _data.get("scalar_dir"),
                    "path_experiment":       _cfg.get("path_experiment"),
                    "checkpoint_dir":        _cfg.get("checkpoint_dir"),
                    "run_dir":               _cfg.get("run_dir"),
                    "inference.output_dir":  (_cfg.get("inference") or {}).get("output_dir"),
                }
                for _label, _val in _output_paths.items():
                    if not _val:
                        continue
                    _p = Path(_val).expanduser()
                    if not _p.is_absolute():
                        _p = (_repo_root / _p).resolve()
                    _err = self._validate_path(str(_p), _label)
                    if _err:
                        return json.dumps({"status": "error", "message": _err})
            except Exception as e:
                return json.dumps({"status": "error", "message": f"Could not validate config: {e}"})

        job_id = str(uuid.uuid4())[:8]
        log_dir = self._log_dir_for_config(config_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / f"job_{job_id}.log")

        env = os.environ.copy()
        repo_root = str(Path(script).resolve().parents[2])
        script_dir = str(Path(script).resolve().parent)
        env["PYTHONPATH"] = ":".join(
            filter(None, [repo_root, script_dir, env.get("PYTHONPATH", "")])
        )

        pending = {
            "job_id": job_id,
            "job_type": job_type,
            "script": script,
            "args": args,
            "config_path": config_path,
            "log_file": log_file,
            "env": env,
            "depends_on": depends_on,
            "python_bin": python_bin,
        }

        gpu_indices: Optional[List[int]] = None
        if not depends_on and self._requires_gpu(job_type):
            gpu_indices = self._select_free_gpus(self._parse_num_gpus(args))

        if depends_on:
            with self._lock:
                self._jobs[job_id] = {
                    "job_id": job_id,
                    "type": job_type,
                    "status": "waiting_for_job",
                    "pid": None,
                    "start_time": datetime.utcnow().isoformat(),
                    "end_time": None,
                    "return_code": None,
                    "config_path": config_path,
                    "log_file": log_file,
                    "args": args,
                    "script": script,
                    "depends_on": depends_on,
                    "note": f"Waiting for job '{depends_on}' to complete successfully",
                }
                self._write_status_file(job_id, "waiting_for_job", note=f"Waiting for dependency {depends_on}")
                self._dep_waiting[job_id] = pending
                self._save_jobs()
        elif not self._requires_gpu(job_type):
            with self._lock:
                self._jobs[job_id] = {
                    "job_id": job_id,
                    "type": job_type,
                    "status": "running",
                    "pid": None,
                    "start_time": datetime.utcnow().isoformat(),
                    "end_time": None,
                    "return_code": None,
                    "config_path": config_path,
                    "log_file": log_file,
                    "args": args,
                    "script": script,
                }
                self._write_status_file(job_id, "running")
                self._save_jobs()
            self._launch_subprocess(
                job_id, job_type, script, args, config_path, log_file, env, python_bin=python_bin
            )
        elif gpu_indices is not None:
            with self._lock:
                self._jobs[job_id] = {
                    "job_id": job_id,
                    "type": job_type,
                    "status": "running",
                    "pid": None,
                    "start_time": datetime.utcnow().isoformat(),
                    "end_time": None,
                    "return_code": None,
                    "config_path": config_path,
                    "log_file": log_file,
                    "args": args,
                    "script": script,
                }
                self._write_status_file(job_id, "running")
                self._save_jobs()
            self._launch_subprocess(
                job_id, job_type, script, args, config_path, log_file, env,
                gpu_indices=gpu_indices, python_bin=python_bin,
            )
        else:
            gpu_info = self._gpu_status_summary()
            with self._lock:
                self._jobs[job_id] = {
                    "job_id": job_id,
                    "type": job_type,
                    "status": "waiting_for_gpu",
                    "pid": None,
                    "start_time": datetime.utcnow().isoformat(),
                    "end_time": None,
                    "return_code": None,
                    "config_path": config_path,
                    "log_file": log_file,
                    "args": args,
                    "script": script,
                    "note": f"Waiting for GPU — {gpu_info}",
                }
                self._write_status_file(job_id, "waiting_for_gpu", note=f"Waiting for GPU — {gpu_info}")
                self._pending_queue.append(pending)
                self._save_jobs()
            self._ensure_watcher()

        with self._lock:
            job_record = dict(self._jobs.get(job_id, {}))
        manifest = write_manifest(job_record, python_bin or _GRANITE_PYTHON)
        if manifest:
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["manifest"] = manifest
                    self._save_jobs()

        return job_id

    def get_status(self, job_id: str) -> str:
        self._reconcile_all_jobs()
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            return json.dumps({"status": "error", "message": f"Job '{job_id}' not found"})

        if job["status"] == "running":
            proc = self._processes.get(job_id)
            if proc and proc.poll() is not None:
                rc = proc.returncode
                with self._lock:
                    self._jobs[job_id]["status"] = "done" if rc == 0 else "failed"
                    self._jobs[job_id]["return_code"] = rc
                    self._jobs[job_id]["end_time"] = datetime.utcnow().isoformat()
                    self._save_jobs()
                job = dict(self._jobs[job_id])

        if job["status"] == "waiting_for_gpu":
            job = dict(job)
            job["gpu_status"] = self._gpu_status_summary()
            queue_pos = next(
                (i + 1 for i, p in enumerate(self._pending_queue) if p["job_id"] == job_id),
                None,
            )
            if queue_pos is not None:
                job["queue_position"] = queue_pos

        if job["status"] == "waiting_for_job":
            job = dict(job)
            dep_id = job.get("depends_on")
            if dep_id:
                with self._lock:
                    dep = self._jobs.get(dep_id, {})
                job["depends_on_status"] = dep.get("status", "unknown")

        result = dict(job)
        result["log_tail"] = self._tail_log(job["log_file"])
        return json.dumps(result, indent=2)

    def list_jobs(self) -> str:
        self._reconcile_all_jobs()
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.get("start_time", ""), reverse=True)
        summary = [{k: v for k, v in j.items() if k != "log_file"} for j in jobs]
        return json.dumps({"jobs": summary, "count": len(summary)}, indent=2)

    def clear_completed_jobs(self) -> str:
        """Delete terminal jobs from persistence and remove their log files."""
        terminal_statuses = {"done", "failed", "cancelled", "blocked"}

        with self._lock:
            remove_ids = [
                job_id for job_id, job in self._jobs.items()
                if job.get("status") in terminal_statuses
            ]
            log_paths = [
                Path(self._jobs[job_id].get("log_file", ""))
                for job_id in remove_ids
                if self._jobs[job_id].get("log_file")
            ]

            for job_id in remove_ids:
                self._jobs.pop(job_id, None)
                self._processes.pop(job_id, None)
                self._dep_waiting.pop(job_id, None)

            if remove_ids:
                self._pending_queue = [
                    pending for pending in self._pending_queue
                    if pending.get("job_id") not in remove_ids
                ]
                self._save_jobs()

        removed_logs = 0
        for p in log_paths:
            try:
                if p.exists() and p.is_file():
                    p.unlink()
                    removed_logs += 1
            except Exception:
                pass

            for job_id in remove_ids:
                try:
                    self._status_file_path(job_id).unlink(missing_ok=True)
                except Exception:
                    pass
                try:
                    self._wrapper_script_path(job_id).unlink(missing_ok=True)
                except Exception:
                    pass

        return json.dumps({
            "status": "ok",
            "removed_jobs": len(remove_ids),
            "removed_logs": removed_logs,
            "job_ids": remove_ids,
        }, indent=2)

    def delete_job(self, job_id: str) -> str:
        """Delete one terminal job from persistence and remove its log file."""
        terminal_statuses = {"done", "failed", "cancelled", "blocked"}

        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return json.dumps({"status": "error", "message": f"Job '{job_id}' not found"})

            status = job.get("status")
            if status not in terminal_statuses:
                return json.dumps({
                    "status": "error",
                    "message": f"Job '{job_id}' is '{status}' and cannot be deleted until it is terminal",
                })

            log_path = Path(job.get("log_file", "")) if job.get("log_file") else None
            self._jobs.pop(job_id, None)
            self._processes.pop(job_id, None)
            self._dep_waiting.pop(job_id, None)
            self._pending_queue = [
                pending for pending in self._pending_queue
                if pending.get("job_id") != job_id
            ]
            self._save_jobs()

        removed_log = False
        if log_path:
            try:
                if log_path.exists() and log_path.is_file():
                    log_path.unlink()
                    removed_log = True
            except Exception:
                removed_log = False
        try:
            self._status_file_path(job_id).unlink(missing_ok=True)
        except Exception:
            pass
        try:
            self._wrapper_script_path(job_id).unlink(missing_ok=True)
        except Exception:
            pass

        return json.dumps({
            "status": "ok",
            "job_id": job_id,
            "removed_log": removed_log,
        }, indent=2)

    def cancel_job(self, job_id: str) -> str:
        import signal, time

        with self._lock:
            job = self._jobs.get(job_id)
            proc = self._processes.get(job_id)

        if not job:
            return json.dumps({"status": "error", "message": f"Job '{job_id}' not found"})

        cancellable = ("running", "waiting_for_gpu", "waiting_for_job", "unknown")
        if job["status"] not in cancellable:
            return json.dumps({
                "status": "error",
                "message": f"Job '{job_id}' cannot be cancelled (current status: {job['status']})",
            })

        with self._lock:
            self._pending_queue = [p for p in self._pending_queue if p["job_id"] != job_id]
            self._dep_waiting.pop(job_id, None)

        should_kill = job["status"] in ("running", "unknown")
        pid = job.get("pid") if proc is None else (proc.pid if proc else None)

        if should_kill and (proc or pid):
            def _kill_group(sig):
                try:
                    pgid = os.getpgid(proc.pid if proc else pid)
                    os.killpg(pgid, sig)
                    return True
                except ProcessLookupError:
                    return False
                except Exception:
                    return False

            def _kill_direct(sig):
                target_pid = proc.pid if proc else pid
                if target_pid is None:
                    return False
                try:
                    os.kill(target_pid, sig)
                    return True
                except ProcessLookupError:
                    return False
                except Exception:
                    return False

            sent = _kill_group(signal.SIGTERM) or _kill_direct(signal.SIGTERM)
            if sent:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if proc is not None and proc.poll() is not None:
                        break
                    try:
                        check_pid = proc.pid if proc else pid
                        os.kill(check_pid, 0)
                        time.sleep(0.3)
                    except ProcessLookupError:
                        break
                    except Exception:
                        break
                else:
                    _kill_group(signal.SIGKILL) or _kill_direct(signal.SIGKILL)

        with self._lock:
            self._jobs[job_id]["status"] = "cancelled"
            self._jobs[job_id]["end_time"] = datetime.utcnow().isoformat()
            self._write_status_file(job_id, "cancelled")
            self._save_jobs()

        return json.dumps({"status": "ok", "message": f"Job '{job_id}' cancelled"})


# Module-level singleton — shared across all tool calls in one server process
_job_manager = JobManager()


# ---------------------------------------------------------------------------
# Convenience wrapper functions called by process_tool_call
# ---------------------------------------------------------------------------

def start_training_job(
    config_path: str = _DEFAULT_CONFIG,
    num_gpus: int = 1,
    save_every: int = 5,
    depends_on: Optional[str] = None,
) -> str:
    dataset_type = _detect_dataset_type(config_path)
    script_dir = _script_dir_for_type(dataset_type)
    script_name = "narr_prism_finetune.py" if dataset_type == "narr" else "merra_prism_finetune.py"
    script = str(script_dir / script_name)
    args = ["--config", config_path, "--num-gpus", str(num_gpus), "--save-every", str(save_every)]
    result = _job_manager.start_job("training", script, args, config_path, depends_on=depends_on)
    try:
        err = json.loads(result)
        if err.get("status") == "error":
            return result
        job_id = result
    except (json.JSONDecodeError, AttributeError):
        job_id = result
    status = _job_manager._jobs.get(job_id, {}).get("status", "queued")
    return json.dumps({
        "job_id": job_id,
        "type": "training",
        "status": status,
        "message": (
            f"Training job '{job_id}' is waiting for job '{depends_on}' to complete first."
            if depends_on else
            f"Training job '{job_id}' started (num_gpus={num_gpus}). "
            f"Call get_job_status with job_id='{job_id}' to check progress."
        ),
    })


def start_inference_job(
    config_path: str = _DEFAULT_CONFIG,
    checkpoint: Optional[str] = None,
    batch_size: int = 1,
    depends_on: Optional[str] = None,
) -> str:
    blocked = _preflight_error(config_path, "inference", depends_on)
    if blocked:
        return blocked
    dataset_type = _detect_dataset_type(config_path)
    script_dir = _script_dir_for_type(dataset_type)
    script_name = "narr_prism_inference.py" if dataset_type == "narr" else "merra_prism_inference.py"
    script = str(script_dir / script_name)
    args = ["--config", config_path, "--batch-size", str(batch_size)]
    if checkpoint:
        args += ["--checkpoint", checkpoint]
    result = _job_manager.start_job("inference", script, args, config_path, depends_on=depends_on)
    try:
        err = json.loads(result)
        if err.get("status") == "error":
            return result
        job_id = result
    except (json.JSONDecodeError, AttributeError):
        job_id = result
    status = _job_manager._jobs.get(job_id, {}).get("status", "queued")
    return json.dumps({
        "job_id": job_id,
        "type": "inference",
        "status": status,
        "message": (
            f"Inference job '{job_id}' is waiting for job '{depends_on}' to complete first."
            if depends_on else
            f"Inference job '{job_id}' started. "
            f"Call get_job_status with job_id='{job_id}' to check progress."
        ),
    })


def start_preprocessing_job(
    config_path: str = _DEFAULT_CONFIG,
    mode: str = "both",
    depends_on: Optional[str] = None,
) -> str:
    blocked = _preflight_error(config_path, "preprocessing", depends_on)
    if blocked:
        return blocked
    dataset_type = _detect_dataset_type(config_path)
    script_dir = _script_dir_for_type(dataset_type)
    script_name = "preproc_narr_prism.py" if dataset_type == "narr" else "preproc_merra_prism.py"
    script = str(script_dir / script_name)
    args = ["--config", config_path, "--mode", mode]
    result = _job_manager.start_job("preprocessing", script, args, config_path, depends_on=depends_on)
    try:
        err = json.loads(result)
        if err.get("status") == "error":
            return result
        job_id = result
    except (json.JSONDecodeError, AttributeError):
        job_id = result
    status = _job_manager._jobs.get(job_id, {}).get("status", "queued")
    return json.dumps({
        "job_id": job_id,
        "type": "preprocessing",
        "status": status,
        "message": (
            f"Preprocessing job '{job_id}' is waiting for job '{depends_on}' to complete first."
            if depends_on else
            f"Preprocessing job '{job_id}' started (mode={mode}). "
            f"Call get_job_status with job_id='{job_id}' to check progress."
        ),
    })


def start_compute_scalars_job(
    config_path: str = _DEFAULT_CONFIG,
    depends_on: Optional[str] = None,
) -> str:
    blocked = _preflight_error(config_path, "compute_scalars", depends_on)
    if blocked:
        return blocked
    dataset_type = _detect_dataset_type(config_path)
    script_dir = _script_dir_for_type(dataset_type)
    script_name = "compute_scalars_narr_prism.py" if dataset_type == "narr" else "compute_scalars_merra_prism.py"
    script = str(script_dir / script_name)
    args = ["--config", config_path]
    result = _job_manager.start_job("compute_scalars", script, args, config_path, depends_on=depends_on)
    try:
        err = json.loads(result)
        if err.get("status") == "error":
            return result
        job_id = result
    except (json.JSONDecodeError, AttributeError):
        job_id = result
    status = _job_manager._jobs.get(job_id, {}).get("status", "queued")
    return json.dumps({
        "job_id": job_id,
        "type": "compute_scalars",
        "status": status,
        "message": (
            f"Compute-scalars job '{job_id}' is waiting for job '{depends_on}' to complete first."
            if depends_on else
            f"Compute-scalars job '{job_id}' started. "
            f"Call get_job_status with job_id='{job_id}' to check progress."
        ),
    })


DOWNLOAD_DATASETS = {
    "merra2": "MERRA-2 M2I3NPASM via earthaccess (needs Earthdata Login)",
    "narr": "NOAA PSL NARR pressure-level daily files",
    "prism": "PRISM AN daily 800 m ppt/tmax/tmin",
    "elevation": "800 m orography on the PRISM grid (reference file from Zenodo 10.5281/zenodo.23096854, sha256-verified)",
    "weights": "Prithvi WxC downscaling weights from Hugging Face (~17.4 GB, sha256-verified)",
    "code": "Pinned Prithvi-UNet training/inference code (git clone + checkout)",
    "env": "Training virtualenv from env/training-requirements.lock.txt + pinned code",
    "cordex": "CORDEX-ML-Bench from Zenodo (optional)",
    "all": "merra2 + prism + elevation + weights",
}
_DATED = {"merra2", "narr", "prism", "all"}


_ACTIVE = {"running", "waiting_for_gpu", "waiting_for_job"}


def _overlapping_download(dataset: str, start: Optional[str], end: Optional[str]) -> Optional[str]:
    """Return the id of an active job writing the same dataset files, if any."""
    def _arg(args: List[str], flag: str) -> Optional[str]:
        return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None

    expand = {"all": {"merra2", "prism", "elevation", "weights"}}
    mine = expand.get(dataset, {dataset})
    with _job_manager._lock:
        jobs = [dict(j) for j in _job_manager._jobs.values()]
    for job in jobs:
        jtype = str(job.get("type", ""))
        if job.get("status") not in _ACTIVE or not jtype.startswith("download_"):
            continue
        theirs = expand.get(jtype[len("download_"):], {jtype[len("download_"):]})
        shared = mine & theirs
        if not shared:
            continue
        if not shared & {"merra2", "narr", "prism"}:
            return job["job_id"]  # undated outputs (weights, code, env, elevation): one writer at a time
        args = job.get("args") or []
        s2, e2 = _arg(args, "--start"), _arg(args, "--end")
        if not (start and end and s2 and e2) or (start <= e2 and s2 <= end):
            return job["job_id"]
    return None


def start_download_job(
    dataset: str = "all",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    elevation_file: Optional[str] = None,
    variables: Optional[str] = None,
    overwrite: bool = False,
    domains: Optional[str] = None,
    variant: Optional[str] = None,
    depends_on: Optional[str] = None,
) -> str:
    """Queue a download/setup step from download_pipeline_data.py as a background job."""
    dataset = (dataset or "all").lower().strip()
    if dataset not in DOWNLOAD_DATASETS:
        return json.dumps({
            "status": "error",
            "message": f"dataset must be one of {sorted(DOWNLOAD_DATASETS)}, got {dataset!r}",
        })
    if config_path and not Path(config_path).exists():
        return json.dumps({"status": "error", "message": f"config_path not found: {config_path}"})
    if dataset in _DATED and not (start_date and end_date) and not config_path:
        return json.dumps({
            "status": "error",
            "message": "start_date and end_date (YYYY-MM-DD) are required for merra2/narr/prism/all "
                       "(or pass config_path to use its dates.training range)",
        })

    overlap = _overlapping_download(dataset, start_date, end_date)
    if overlap:
        return json.dumps({
            "status": "error",
            "message": (f"Job '{overlap}' is already downloading {dataset} for overlapping dates; concurrent "
                        "downloads of the same day clobber each other's files. Wait for it, or pass "
                        f"depends_on='{overlap}', or use a non-overlapping range."),
        })

    script = str(Path(__file__).resolve().parent / "download_pipeline_data.py")
    args = [dataset]
    if config_path:
        args += ["--config", config_path]
    if start_date:
        args += ["--start", start_date]
    if end_date:
        args += ["--end", end_date]
    if overwrite:
        args.append("--overwrite")
    if output_dir:
        args += ["--output-dir", output_dir]
    if elevation_file:
        args += ["--output-file", elevation_file]
    if variables:
        args += ["--variables", variables]
    if domains:
        args += ["--domains", domains]
    if variant:
        args += ["--variant", variant]

    result = _job_manager.start_job(
        f"download_{dataset}",
        script,
        args,
        config_path or "",
        depends_on=depends_on,
        python_bin=sys.executable,
        validate_config_outputs=False,
    )
    try:
        err = json.loads(result)
        if err.get("status") == "error":
            return result
        job_id = result
    except (json.JSONDecodeError, AttributeError):
        job_id = result
    job = _job_manager._jobs.get(job_id, {})
    return json.dumps({
        "job_id": job_id,
        "type": f"download_{dataset}",
        "status": job.get("status", "queued"),
        "what": DOWNLOAD_DATASETS[dataset],
        "log_file": job.get("log_file"),
        "manifest": job.get("manifest"),
        "message": (
            f"Download job '{job_id}' started ({dataset}). Files are written on the machine running "
            "this MCP server, under PIPELINE_DATA_ROOT. "
            f"Call get_job_status with job_id='{job_id}' to follow progress."
        ),
    })


def setup_code(variant: Optional[str] = None, depends_on: Optional[str] = None) -> str:
    return start_download_job(dataset="code", variant=variant, depends_on=depends_on)


def setup_training_env(depends_on: Optional[str] = None) -> str:
    return start_download_job(dataset="env", depends_on=depends_on)


def preflight_check(config_path: str = _DEFAULT_CONFIG, stage: str = "training") -> str:
    return json.dumps({"status": "ok", "config_path": config_path, "stage": stage,
                       **preflight(config_path, stage)})


def _preflight_error(config_path: str, stage: str, depends_on: Optional[str]) -> Optional[str]:
    """Refuse to queue a stage whose inputs are missing, unless it waits on another job."""
    if depends_on:
        return None
    check = preflight(config_path, stage)
    if check["ok"]:
        return None
    return json.dumps({"status": "error", "stage": "preflight", "config_path": config_path, **check})


def check_raw_data_status() -> str:
    from download_pipeline_data import status_report

    report = status_report()
    missing = []
    if not report["merra2"]["files"] and not report["narr"]["files"]:
        missing.append("predictors (merra2 or narr)")
    if not any(report["prism"][v]["files"] for v in ("ppt", "tmax", "tmin")):
        missing.append("prism")
    if not report["elevation"]["exists"]:
        missing.append("elevation")
    if not report["weights"]["complete"]:
        missing.append("weights")
    if not report["code"].get("is_git"):
        missing.append("code")
    report["ready"] = not missing
    report["missing"] = missing
    report["next_step"] = (
        "Inputs are present. Next: create_custom_yaml (localizes paths), then run_training_pipeline."
        if not missing else
        f"Call start_download_job for: {', '.join(missing)} (use setup_code for code)."
    )
    return json.dumps({"status": "ok", **report})


# ---------------------------------------------------------------------------
# Smart pipeline helpers
# ---------------------------------------------------------------------------

_SCALAR_FILES = ("inputs_mean.npy", "inputs_std.npy", "targets_mean.npy", "targets_std.npy")


def _scalars_exist(config_path: str) -> Optional[Path]:
    try:
        import yaml as _yaml
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = _yaml.safe_load(fh) or {}
    except Exception:
        return None

    data = cfg.get("data", {}) or {}
    repo_root = Path(config_path).resolve().parents[2]
    case_name = cfg.get("case_name", "")
    candidates: List[Path] = []

    # Current code writes case-scoped scalars to <preprocessed_dir>/<case_name>/scalars.
    preproc_raw = data.get("preprocessed_dir")
    if preproc_raw and case_name:
        preproc = Path(preproc_raw).expanduser()
        if not preproc.is_absolute():
            preproc = (repo_root / preproc).resolve()
        candidates.append(preproc / case_name / "scalars")

    scalar_dir_raw = data.get("scalar_dir")
    if not scalar_dir_raw:
        return next((d for d in candidates if all((d / f).exists() for f in _SCALAR_FILES)), None)

    base = Path(scalar_dir_raw).expanduser()
    if not base.is_absolute():
        base = (repo_root / base).resolve()

    if case_name:
        case_dir = base if base.name == case_name else base / case_name
        candidates.append(case_dir)
    candidates.append(base)

    for d in candidates:
        if all((d / f).exists() for f in _SCALAR_FILES):
            return d
    return None


def _preprocessed_exists(config_path: str) -> Optional[Path]:
    try:
        import yaml as _yaml
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = _yaml.safe_load(fh) or {}
    except Exception:
        return None

    data = cfg.get("data", {}) or {}
    preproc_dir_raw = data.get("preprocessed_dir")
    if not preproc_dir_raw:
        return None

    repo_root = Path(config_path).resolve().parents[2]
    base = Path(preproc_dir_raw).expanduser()
    if not base.is_absolute():
        base = (repo_root / base).resolve()

    case_name = cfg.get("case_name", "")
    candidates: List[Path] = []
    if case_name:
        candidates.append(base / "training" / case_name)
        candidates.append(base / case_name / "training")
    candidates.append(base / "training")

    for d in candidates:
        if d.is_dir() and any(d.glob("*.nc")):
            return d
    return None


def run_training_pipeline(
    config_path: str = _DEFAULT_CONFIG,
    num_gpus: int = 1,
    save_every: int = 5,
    queue_inference: bool = True,
) -> str:
    """Smart pipeline: check scalars + preprocessed data, then submit all
    needed jobs in dependency order. Every job is visible via list_jobs /
    get_job_status and cancellable via cancel_job.

    Steps:
      1. compute_scalars        — skipped if scalar .npy files already exist
      2. preprocessing/training — skipped if training .nc files already exist
      3. preprocessing/inference — skipped if training was skipped
      4. training               — always runs (after its dependencies)
      5. inference              — queued; waits for training to finish
    """
    blocked = _preflight_error(config_path, "training", None)
    if blocked:
        return blocked

    jobs_submitted: List[dict] = []
    skipped: List[str] = []
    last_job_id: Optional[str] = None

    # Step 1 – Scalars
    scalar_dir = _scalars_exist(config_path)
    if scalar_dir:
        skipped.append(f"compute_scalars  (scalars already exist at {scalar_dir})")
    else:
        raw = start_compute_scalars_job(config_path=config_path, depends_on=last_job_id)
        parsed = json.loads(raw)
        if parsed.get("status") == "error":
            return json.dumps({"status": "error", "stage": "compute_scalars", "message": parsed.get("message")})
        last_job_id = parsed["job_id"]
        jobs_submitted.append({"stage": "compute_scalars", "job_id": last_job_id, "status": parsed.get("status")})

    # Step 2 – Preprocessing
    preproc_dir = _preprocessed_exists(config_path)
    if preproc_dir:
        nc_count = len(list(preproc_dir.glob("*.nc")))
        skipped.append(f"preprocessing    (training data already at {preproc_dir}, {nc_count} files)")
    else:
        raw = start_preprocessing_job(config_path=config_path, mode="training", depends_on=last_job_id)
        parsed = json.loads(raw)
        if parsed.get("status") == "error":
            return json.dumps({"status": "error", "stage": "preprocessing/training", "message": parsed.get("message")})
        last_job_id = parsed["job_id"]
        jobs_submitted.append({"stage": "preprocessing/training", "job_id": last_job_id, "status": parsed.get("status")})

        raw2 = start_preprocessing_job(config_path=config_path, mode="inference", depends_on=last_job_id)
        parsed2 = json.loads(raw2)
        if parsed2.get("status") == "error":
            return json.dumps({"status": "error", "stage": "preprocessing/inference", "message": parsed2.get("message")})
        last_job_id = parsed2["job_id"]
        jobs_submitted.append({"stage": "preprocessing/inference", "job_id": last_job_id, "status": parsed2.get("status")})

    # Step 3 – Training
    raw = start_training_job(config_path=config_path, num_gpus=num_gpus, save_every=save_every, depends_on=last_job_id)
    parsed = json.loads(raw)
    if parsed.get("status") == "error":
        return json.dumps({"status": "error", "stage": "training", "message": parsed.get("message")})
    last_job_id = parsed["job_id"]
    jobs_submitted.append({"stage": "training", "job_id": last_job_id, "status": parsed.get("status")})

    # Step 4 – Inference
    if queue_inference:
        raw = start_inference_job(config_path=config_path, depends_on=last_job_id)
        parsed = json.loads(raw)
        if parsed.get("status") == "error":
            return json.dumps({"status": "error", "stage": "inference", "message": parsed.get("message")})
        last_job_id = parsed["job_id"]
        jobs_submitted.append({"stage": "inference", "job_id": last_job_id, "status": parsed.get("status")})

    lines = ["Pipeline jobs submitted:"]
    for j in jobs_submitted:
        lines.append(f"  {j['stage']:<30} job_id={j['job_id']}  status={j['status']}")
    if skipped:
        lines.append("Skipped (already complete):")
        for s in skipped:
            lines.append(f"  {s}")
    lines += ["", "Monitor: list_jobs / get_job_status.  Cancel: cancel_job."]

    return json.dumps({
        "status": "ok",
        "jobs": jobs_submitted,
        "skipped": skipped,
        "message": "\n".join(lines),
    }, indent=2)
