"""One blocking library run at a time, owned by the ComfyUI execution queue."""

import copy
import fcntl
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

from .config import INT_LIMITS, cache_namespace, inside, validate
from .embedded import run as run_library
from .files import checkpoints, publish, read_jsonl, read_log

TERMINAL = {"completed", "failed", "stopped", "interrupted"}


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "start_ticks": fields[19],
            "pgid": int(fields[2]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Exited zombies do not hold models or GPU memory.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in ("Z", "X"):
                return True
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            return True
    return False


class Stopped(Exception):
    pass


class JobManager:
    def __init__(self, runtime, observer=None, interrupted=None):
        self.runtime = runtime
        self.observer = observer or (lambda event: None)
        self.interrupted = interrupted or (lambda: False)
        self.root = runtime.workspace.resolve()
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.active = None
        self.cancel = threading.Event()
        for file in (self.root / "jobs").glob("*/job.json"):
            job = json.loads(file.read_text(encoding="utf-8"))
            if file.parent.name != job["job_id"]:
                raise ValueError(
                    f"Job metadata id does not match its directory: {file}"
                )
            self.jobs[job["job_id"]] = job
        self._recover()

    def _recover(self):
        # Never signal an inherited pid: even a matching process cannot safely be adopted.
        for job in self.jobs.values():
            if job["status"] in TERMINAL:
                continue
            identity = job.get("process")
            current = process_identity(identity["pid"]) if identity else None
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            live = bool(
                identity
                and identity["boot_id"] == boot
                and (current == identity or group_alive(identity["pgid"]))
            )
            if live:
                job.update(
                    status="unknown",
                    error="A previous worker group may still be running. Stop it externally and restart ComfyUI before resuming; this process will not signal inherited PIDs.",
                )
            else:
                job.update(
                    status="interrupted",
                    process=None,
                    error="ComfyUI restarted before this job finished; resume from an existing complete checkpoint.",
                )
            self._save(job)

    def _save(self, job):
        job["updated_at"] = time.time()
        directory = self.root / "jobs" / job["job_id"]
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "job.json.tmp"
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(directory / "job.json")

    def _update(self, job, **values):
        with self.lock:
            job.update(values)
            self._save(job)

    def _phase(self, job, phase):
        now = time.time()
        with self.lock:
            if self.cancel.is_set():
                raise Stopped()
            if job["phases"] and "ended_at" not in job["phases"][-1]:
                job["phases"][-1]["ended_at"] = now
            job["phases"].append({"phase": phase, "started_at": now})
            job["status"] = phase
            self._save(job)
        self.observer({"job_id": job["job_id"], "phase": phase})
        logging.info("[KohakuMoss] Job %s: %s", job["job_id"], phase)

    def _claim(self):
        if self.active or any(j["status"] == "unknown" for j in self.jobs.values()):
            raise ValueError(
                "A KohakuMoss job is active or an inherited worker is unknown; only one GPU job may run"
            )
        file = (self.root / "gpu.lock").open("a")
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            file.close()
            raise ValueError(
                "KohakuMoss GPU is busy in another ComfyUI process"
            ) from None
        return file

    def submit(self, settings, submission_key):
        job, lease, resume_path = self._submit(settings, submission_key)
        if lease is not None:
            self._work(job, lease, resume_path)
        return self.get(job["job_id"])

    def _submit(self, settings, submission_key):
        with self.lock:
            for job in self.jobs.values():
                if submission_key in job["submission_keys"]:
                    return self.get(job["job_id"]), None, None
            if settings.get("start_mode") == "resume_latest":
                return self._resume(
                    settings.get("resume_job_id", ""),
                    settings.get("resume_checkpoint", "latest"),
                    submission_key,
                    additional_steps=settings.get("steps"),
                    output_path=settings.get("output_path"),
                )
            settings = validate(settings, self.runtime.dataset_root, self.root)
            settings["output_path"] = self._output_path(settings["output_path"])
            self.runtime.check()
            lease = self._claim()
            try:
                job_id = uuid.uuid4().hex
                name = f"{settings['run_name']}-{job_id[:12]}"
                directory = self.root / "jobs" / job_id
                directory.mkdir()
                job = dict(
                    job_id=job_id,
                    run_name=name,
                    settings=settings,
                    submission_keys=[submission_key],
                    created_at=time.time(),
                    status="created",
                    run_dir=str(Path(settings["output_root"]) / name),
                    cache_dir=str(self.root / "cache" / cache_namespace(settings)),
                    config=str(directory / "train.py"),
                    log=str(directory / "worker.log"),
                    log_generation=0,
                    commands=[],
                    phases=[],
                    error=None,
                    exit_code=None,
                    process=None,
                    resume_step=None,
                    published=None,
                    exports=[],
                    warnings=[],
                )
                self.jobs[job_id] = job
                self._save(job)
                return self._launch(job, lease)
            except BaseException:
                lease.close()
                raise

    def _output_path(self, value):
        if value is not None and not isinstance(value, str):
            raise ValueError("output_path must be a directory path")
        directory = inside(
            self.runtime.publish_dir / (value or ""), self.runtime.publish_dir
        )
        if directory.is_file() or directory.suffix.lower() == ".safetensors":
            raise ValueError("output_path must be a LoRA directory, not a file")
        return str(directory)

    def _launch(self, job, lease, resume_path=None):
        self.active = job["job_id"]
        self.cancel = threading.Event()
        return job, lease, resume_path

    def get(self, job_id):
        with self.lock:
            job = copy.deepcopy(self.jobs[job_id])
        rows = read_jsonl(Path(job["run_dir"]) / "metrics.jsonl", 1)
        job["metrics"] = {**(rows[-1] if rows else {}), **job.get("live_metrics", {})}
        job["checkpoints"] = [str(p) for p in checkpoints(job["run_dir"])]
        job["can_resume"] = job["status"] in {
            "completed",
            "stopped",
            "failed",
            "interrupted",
        } and bool(job["checkpoints"])
        return job

    def list(self):
        with self.lock:
            ids = sorted(
                self.jobs, key=lambda i: self.jobs[i]["created_at"], reverse=True
            )
        return [self.get(i) for i in ids]

    def logs(self, job_id, offset=0, limit=200, generation=None):
        job = self.get(job_id)
        if generation is not None and int(generation) != job["log_generation"]:
            offset = 0
        return {
            **read_log(job["log"], offset, limit),
            "generation": job["log_generation"],
        }

    def metrics(self, job_id, limit=1000):
        return read_jsonl(Path(self.get(job_id)["run_dir"]) / "metrics.jsonl", limit)

    def stop(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] in TERMINAL:
                return self.get(job_id)
            if job_id != self.active:
                raise ValueError(
                    "This worker cannot be safely controlled by this ComfyUI process"
                )
            self.cancel.set()
            self._update(
                job,
                status="stopping",
                stop_requested_at=time.time(),
                stop_note="Finishing the current optimizer step, then saving a complete checkpoint before stopping.",
            )
            return self.get(job_id)

    def _resume(
        self,
        job_id,
        checkpoint="latest",
        submission_key=None,
        additional_steps=None,
        output_path=None,
    ):
        if not isinstance(job_id, str) or not isinstance(checkpoint, str):
            raise ValueError("Job and checkpoint must be strings")
        with self.lock:
            if not job_id:
                job_id = next(
                    (
                        i
                        for i, j in self.jobs.items()
                        if checkpoint in [str(p) for p in checkpoints(j["run_dir"])]
                    ),
                    None,
                )
                if job_id is None:
                    raise ValueError(
                        "Choose a complete checkpoint from a KohakuMoss job"
                    )
            job = self.jobs[job_id]
            if submission_key and submission_key in job["submission_keys"]:
                return self.get(job_id), None, None
            if job["status"] not in {"completed", "stopped", "failed", "interrupted"}:
                raise ValueError(
                    "Resume requires a completed, stopped, failed or interrupted job"
                )
            available = checkpoints(job["run_dir"])
            chosen = available[-1] if available else None
            if checkpoint != "latest":
                chosen = next(
                    (p for p in available if checkpoint in (p.name, str(p))), None
                )
            if chosen is None:
                raise ValueError(
                    "No complete managed checkpoint is available; .partial directories cannot be resumed"
                )
            settings = validate(job["settings"], self.runtime.dataset_root, self.root)
            step = int(chosen.name.removeprefix("step_"))
            if additional_steps is not None:
                low, high = INT_LIMITS["steps"]
                if (
                    type(additional_steps) is not int
                    or not low <= additional_steps <= high
                    or step + additional_steps > high
                ):
                    raise ValueError(
                        "Additional steps must be a positive integer within the training limit"
                    )
                settings["steps"] = step + additional_steps
            if settings["steps"] <= step:
                raise ValueError(
                    "This checkpoint reached the previous target. Specify additional training steps"
                )
            settings["output_path"] = self._output_path(
                output_path if output_path is not None else settings["output_path"]
            )
            self.runtime.check()
            if not Path(job["config"]).is_file():
                raise ValueError("The original generated train config is missing")
            lease = self._claim()
            try:
                job.update(
                    status="created",
                    settings=settings,
                    error=None,
                    exit_code=None,
                    process=None,
                    resume_step=step,
                    live_metrics={},
                    progress={},
                )
                if submission_key:
                    job["submission_keys"].append(submission_key)
                self._save(job)
                return self._launch(job, lease, chosen)
            except BaseException:
                lease.close()
                raise

    def _event(self, job, event):
        event = dict(event)
        phase = event.pop("phase")
        if job["status"] == phase:
            event = {**job.get("progress", {}), **event}
        if phase == "log" and event.get("rotated"):
            self._update(job, log_generation=job["log_generation"] + 1)
        if phase != "log":
            with self.lock:
                if job["status"] != phase:
                    now = time.time()
                    if job["phases"] and "ended_at" not in job["phases"][-1]:
                        job["phases"][-1]["ended_at"] = now
                    job["phases"].append({"phase": phase, "started_at": now})
                job["status"] = phase
                if phase == "training":
                    job.setdefault("live_metrics", {}).update(event)
                job["progress"] = event
                self._save(job)
        self.observer(
            {
                "job_id": job["job_id"],
                "node_id": job["submission_keys"][-1].rsplit(":", 1)[-1],
                "phase": phase,
                **event,
            }
        )

    def _work(self, job, lease, resume_path):
        status, error = "completed", None
        try:
            from kohakumoss_common.events import Cancelled, CleanupFailed

            settings = job["settings"]
            Path(settings["output_root"]).mkdir(parents=True, exist_ok=True)
            Path(job["cache_dir"]).mkdir(parents=True, exist_ok=True)
            try:
                run_library(
                    self.runtime,
                    job,
                    resume_path,
                    lambda event: self._event(job, event),
                    lambda: self.cancel.is_set() or self.interrupted(),
                )
            except Cancelled:
                raise Stopped() from None
            except CleanupFailed as exception:
                status, error = "unknown", str(exception)
                raise
            if self.cancel.is_set():
                raise Stopped()
            self._phase(job, "exporting")
            export_root = Path(job["run_dir"]) / "export"
            exports = sorted(
                str(p) for p in export_root.glob("*") if p.is_file() or p.is_dir()
            )
            self._update(job, exports=exports)
            if settings["publish_to_loras"]:
                self._phase(job, "publishing")
                source = export_root / (job["run_name"] + "-comfy.safetensors")
                if not source.is_file() or source.stat().st_size == 0:
                    raise RuntimeError(
                        f"Native comfy export was not produced: {source}"
                    )
                self._update(
                    job,
                    published=str(
                        publish(
                            source,
                            Path(
                                settings.get("output_path") or self.runtime.publish_dir
                            ),
                        )
                    ),
                )
        except Stopped:
            status = "stopped"
        except (Exception, SystemExit) as exception:
            status, error = ("unknown" if status == "unknown" else "failed"), str(
                exception
            )
            logging.exception("[KohakuMoss] Library training failed")
        finally:
            with self.lock:
                self.active = None
                if status == "unknown":
                    self.orphan_lease = lease
                else:
                    lease.close()
                self._finish(job, status, error)
        if status in {"failed", "unknown"}:
            raise RuntimeError(error)

    def _finish(self, job, status, error=None):
        with self.lock:
            if job["phases"]:
                job["phases"][-1]["ended_at"] = time.time()
            self._update(job, status=status, error=error, finished_at=time.time())
        self.observer({"job_id": job["job_id"], "phase": status, "error": error})
        if error:
            logging.error("[KohakuMoss] Job %s %s: %s", job["job_id"], status, error)
        else:
            logging.info("[KohakuMoss] Job %s: %s", job["job_id"], status)
