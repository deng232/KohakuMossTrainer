"""Thread-safe, bounded training snapshots and native metric history."""

import json
import logging
import math
import os
import time
import uuid
from collections import OrderedDict, deque
from contextlib import contextmanager
from numbers import Integral, Real
from pathlib import Path
from threading import Lock, RLock

log = logging.getLogger(__name__)


def safe(value):
    """Copy CPU scalar data into strict JSON values; unsupported objects become null."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return None


def identity(path):
    try:
        stat = path.stat()
        return stat.st_dev, stat.st_ino
    except OSError:
        return None


class SessionStore:
    """One training lease; detached snapshots and a fixed history budget per job."""

    def __init__(self, send=None, points=10_000, jobs=20):
        self.send = send
        self.points = points
        self.limit = jobs
        self._lock = RLock()
        self._lease = Lock()
        self._jobs = OrderedDict()
        self._owners = {}

    def _emit(self, name, job):
        job["snapshot"]["event_seq"] += 1
        payload = safe(job["snapshot"])
        if name == "kohakumoss.metrics":
            payload.update(
                step=payload["last_metric_step"], metrics=payload["last_metrics"]
            )
        if self.send is not None:
            try:
                self.send(name, payload)
            except Exception:
                log.warning("Could not deliver KohakuMoss event", exc_info=True)

    @contextmanager
    def training(self, origin_node_id, workflow_id=""):
        if not self._lease.acquire(blocking=False):
            raise RuntimeError("KohakuMoss training is already running")
        job_id = uuid.uuid4().hex
        try:
            with self._lock:
                while len(self._jobs) >= self.limit:
                    old_id, old = self._jobs.popitem(last=False)
                    directory = old["snapshot"]["run_dir"]
                    if self._owners.get(directory) == old_id:
                        del self._owners[directory]
                self._jobs[job_id] = job = {
                    "snapshot": {
                        "job_id": job_id,
                        "origin_node_id": str(origin_node_id),
                        "workflow_id": str(workflow_id),
                        "run_name": "",
                        "run_dir": "",
                        "status": "INITIALIZING",
                        "phase": "INITIALIZING",
                        "last_metric_step": None,
                        "last_metrics": {},
                        "total_steps": None,
                        "started_at": time.time(),
                        "timestamp": time.time(),
                        "event_seq": 0,
                        "error": None,
                        "files": [],
                    },
                    "rows": deque(maxlen=self.points),
                    "count": 0,
                    "file": None,
                    "identity": None,
                }
                self._emit("kohakumoss.job_started", job)
            try:
                yield job_id
            except BaseException as error:
                self.update(
                    job_id,
                    status="FAILED",
                    phase="FAILED",
                    error=f"{type(error).__name__}: {error}",
                )
                raise
            else:
                self.update(job_id, status="COMPLETED", phase="COMPLETED")
        finally:
            self._lease.release()

    def configure(self, job_id, run_dir, run_name, total_steps):
        directory = str(Path(run_dir).expanduser().resolve())
        with self._lock:
            self._owners[directory] = job_id
            self.update(
                job_id, run_dir=directory, run_name=run_name, total_steps=total_steps
            )

    def attach_logger(self, run_dir, path=None):
        directory = str(Path(run_dir).expanduser().resolve())
        with self._lock:
            job_id = self._owners.get(directory)
            if job_id is None or self._jobs[job_id]["snapshot"]["status"] not in (
                "INITIALIZING",
                "TRAINING",
            ):
                raise RuntimeError("No active KohakuMoss session for logger directory")
            if path is not None:
                job = self._jobs[job_id]
                job["file"] = Path(path).resolve()
                job["identity"] = identity(job["file"])
            return job_id

    def update(self, job_id, **fields):
        with self._lock:
            job = self._jobs[job_id]
            job["snapshot"].update(safe(fields), timestamp=time.time())
            self._emit("kohakumoss.status", job)

    def metrics(self, job_id, step, metrics):
        with self._lock:
            job = self._jobs[job_id]
            values = safe(metrics)
            stamp = time.time()
            job["snapshot"].update(
                status="TRAINING",
                phase="TRAINING",
                last_metric_step=int(step),
                last_metrics=values,
                timestamp=stamp,
            )
            job["count"] += 1
            job["rows"].append(
                {
                    "step": int(step),
                    "timestamp": stamp,
                    "metrics": values,
                    "event_seq": job["snapshot"]["event_seq"] + 1,
                }
            )
            self._emit("kohakumoss.metrics", job)

    def resume_at(self, job_id, step):
        with self._lock:
            job = self._jobs[job_id]
            job["rows"] = deque(
                (r for r in job["rows"] if r["step"] <= step), maxlen=self.points
            )
            job["snapshot"].update(last_metric_step=None, last_metrics={})
            self._emit("kohakumoss.status", job)

    def snapshot(self, job_id):
        with self._lock:
            return safe(self._jobs[job_id]["snapshot"])

    def list_jobs(self, origin_node_id=None, workflow_id=None):
        with self._lock:
            return [
                safe(j["snapshot"])
                for j in reversed(self._jobs.values())
                if (
                    origin_node_id is None
                    or j["snapshot"]["origin_node_id"] == str(origin_node_id)
                )
                and (workflow_id is None or j["snapshot"]["workflow_id"] == workflow_id)
            ]

    def history(self, job_id):
        with self._lock:
            job = self._jobs[job_id]
            snapshot = safe(job["snapshot"])
            rows = safe(list(job["rows"]))
            file, token = job["file"], job["identity"]
            owner = self._owners.get(snapshot["run_dir"]) == job_id
            truncated = job["count"] > self.points
        candidates = []
        if file is not None:
            candidates = ([file] if owner else []) + sorted(
                file.parent.glob("metrics.*.jsonl")
            )
        disk = deque(maxlen=self.points)
        found = False
        for candidate in candidates:
            if token is None or identity(candidate) != token:
                continue
            try:
                with candidate.open(encoding="utf-8") as stream:
                    stat = stream.fileno()
                    # Verify the opened file too: native rotation may race the HTTP read.
                    actual = os.fstat(stat)
                    if (actual.st_dev, actual.st_ino) != token:
                        continue
                    found = True
                    for line in stream:
                        try:
                            row = json.loads(line)
                            step = row.pop("step")
                            if (
                                isinstance(step, bool)
                                or not isinstance(step, int)
                                or step < 0
                            ):
                                continue
                            stamp = row.pop("time", None)
                            if len(disk) == self.points:
                                truncated = True
                            disk.append(
                                {
                                    "step": step,
                                    "timestamp": safe(stamp),
                                    "metrics": safe(row),
                                }
                            )
                        except (ValueError, KeyError, AttributeError, TypeError):
                            continue
            except OSError:
                continue
            if candidate == file:
                with self._lock:
                    if self._owners.get(snapshot["run_dir"]) != job_id:
                        disk.clear()
                        found = False
            break
        merged = {row["step"]: row for row in disk}
        merged.update({row["step"]: row for row in rows})
        ordered = sorted(merged.values(), key=lambda row: row["step"])
        return {
            "job_id": job_id,
            "rows": ordered[-self.points :],
            "event_seq": snapshot["event_seq"],
            "snapshot": snapshot,
            "truncated": truncated
            or (file is not None and not found)
            or len(ordered) > self.points,
        }


STORE = SessionStore()
