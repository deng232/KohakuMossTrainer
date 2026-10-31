"""One local GPU job at a time; all heavy commands run in the trainer's venv."""

import copy
import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .config import INT_LIMITS, cache_namespace, configure_generated, inside, validate
from .files import checkpoints, publish, read_jsonl, read_log
from .runtime import SCRIPTS

TERMINAL = {"completed", "failed", "stopped", "interrupted"}
STOP_TIMEOUT = 30
KILL_TIMEOUT = 5
LOG_BYTES = 64 * 1024 * 1024


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


class GroupStillRunning(Exception):
    pass


class JobManager:
    def __init__(self, runtime):
        self.runtime = runtime
        self.root = runtime.workspace.resolve()
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.active = None
        self.cancel = threading.Event()
        self.process = None
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
        with self.lock:
            for job in self.jobs.values():
                if submission_key in job["submission_keys"]:
                    return self.get(job["job_id"])
            if settings.get("start_mode") == "resume_latest":
                return self.resume(
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
                self._launch(job, lease)
                return self.get(job_id)
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
        threading.Thread(
            target=self._work,
            args=(job, lease, resume_path),
            name=f"kohakumoss-{job['job_id'][:8]}",
            daemon=True,
        ).start()

    def get(self, job_id):
        with self.lock:
            job = copy.deepcopy(self.jobs[job_id])
        rows = read_jsonl(Path(job["run_dir"]) / "metrics.jsonl", 1)
        job["metrics"] = rows[-1] if rows else {}
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
                stop_note=f"SIGTERM requested; after {STOP_TIMEOUT}s the owned process group will be forcibly stopped. No new checkpoint is guaranteed.",
            )
            self._signal(signal.SIGTERM)
            return self.get(job_id)

    def _signal(self, signum):
        process = self.process
        if process is not None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    def _terminate_group(self):
        process = self.process
        if process is None:
            return True
        for signum, timeout in (
            (signal.SIGTERM, STOP_TIMEOUT),
            (signal.SIGKILL, KILL_TIMEOUT),
        ):
            self._signal(signum)
            deadline = time.monotonic() + timeout
            while True:
                process.poll()
                if process.returncode is not None and not group_alive(process.pid):
                    return True
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
        return False

    def resume(
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
                return self.get(job_id)
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
                )
                if submission_key:
                    job["submission_keys"].append(submission_key)
                self._save(job)
                self._launch(job, lease, chosen)
            except BaseException:
                lease.close()
                raise
            return self.get(job_id)

    def _provenance(self, job, env):
        try:
            commit = subprocess.run(
                ["git", "-C", str(self.runtime.repo), "rev-parse", "HEAD"],
                env=env,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            commit = "unknown (checkout has no git metadata)"
        code = "import sys,json,importlib.metadata as m; print(json.dumps({'python':sys.version,'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"
        result = subprocess.run(
            [str(self.runtime.venv / "bin/python"), "-c", code],
            env=env,
            cwd=self.runtime.repo,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        versions = json.loads(result.stdout)
        models = {}
        for key in ("transformer", "text_encoder", "vae"):
            path = Path(job["settings"][key])
            info = path.stat()
            models[key] = dict(
                path=str(path), size=info.st_size, mtime_ns=info.st_mtime_ns
            )
        warnings = [
            "Training and ComfyUI inference share GPU memory; avoid simultaneous inference."
        ]
        free_disk = shutil.disk_usage(self.root).free
        if free_disk < 20 * 1024**3:
            warnings.append(
                "Less than 20 GiB free disk space; caches/checkpoints may exhaust storage."
            )
        memory = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemAvailable:", "MemTotal:")):
                name, value, _ = line.split()
                memory[name.rstrip(":")] = int(value) * 1024
        if memory.get("MemAvailable", 0) < 16 * 1024**3:
            warnings.append(
                "Less than 16 GiB available host RAM; streamed weights/optimizer may not fit."
            )
        self._update(
            job,
            trainer_commit=commit,
            environment=versions,
            models=models,
            resources={"disk_free_bytes": free_disk, **memory},
            warnings=warnings,
        )

    def _command(self, job, phase, argv, env):
        if self.cancel.is_set():
            raise Stopped()
        self._phase(job, phase)
        record = {"phase": phase, "argv": argv, "started_at": time.time()}
        with self.lock:
            job["commands"].append(record)
            self._save(job)
        with Path(job["log"]).open("ab", buffering=0) as log:
            log.write(("\n" + json.dumps(argv, ensure_ascii=False) + "\n").encode())
            with self.lock:
                if self.cancel.is_set():
                    raise Stopped()
                process = subprocess.Popen(
                    argv,
                    cwd=self.runtime.repo,
                    env=env,
                    shell=False,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self.process = process
                self._update(job, process=process_identity(process.pid))
            while process.poll() is None:
                if self.cancel.is_set():
                    break
                if os.fstat(log.fileno()).st_size > LOG_BYTES:
                    os.ftruncate(log.fileno(), 0)
                    log.write(
                        b"[KohakuMoss: older console output truncated at 64 MiB]\n"
                    )
                    self._update(job, log_generation=job["log_generation"] + 1)
                time.sleep(0.1)
            if self.cancel.is_set() or group_alive(process.pid):
                if not self._terminate_group():
                    raise GroupStillRunning(
                        "The worker group has not exited after forced termination; GPU exclusion is retained. Stop it externally, then restart ComfyUI."
                    )
            with self.lock:
                record.update(exit_code=process.returncode, ended_at=time.time())
                self.process = None
                self._update(job, process=None, exit_code=process.returncode)
            if self.cancel.is_set():
                raise Stopped()
            if process.returncode:
                with Path(job["log"]).open("rb") as f:
                    f.seek(max(0, f.seek(0, 2) - 8192))
                    tail = f.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"{phase} exited with code {process.returncode}\n{tail}"
                )

    def _work(self, job, lease, resume_path):
        status, error = "completed", None
        try:
            env = self.runtime.environment()
            self._provenance(job, env)
            settings = job["settings"]
            config = Path(job["config"])
            config_cli = str(self.runtime.venv / "bin/kohakumoss-config")
            kogine = str(self.runtime.venv / "bin/kogine")
            if resume_path is None:
                Path(settings["output_root"]).mkdir(parents=True, exist_ok=True)
                Path(job["cache_dir"]).mkdir(parents=True, exist_ok=True)
                # WEIGHTS may be a single DiT file; every component is explicitly overridden.
                self._command(
                    job,
                    "validating",
                    [
                        config_cli,
                        "new",
                        "train",
                        "--weights",
                        settings["transformer"],
                        "--vae",
                        settings["vae"],
                        "--text-encoder",
                        settings["text_encoder"],
                        "--data",
                        settings["dataset_path"],
                        "--source",
                        settings["dataset_source"],
                        "--algo",
                        settings["algorithm"],
                        "--name",
                        job["run_name"],
                        "--steps",
                        str(settings["steps"]),
                        "--vram",
                        "12" if settings["memory_profile"] == "vram_12g" else "auto",
                        "--precision",
                        settings["precision"],
                        "--out",
                        str(config),
                    ],
                    env,
                )
                configure_generated(
                    config,
                    settings,
                    Path(job["run_dir"]).parent,
                    Path(job["cache_dir"]),
                )
            self._command(
                job,
                "validating",
                [
                    config_cli,
                    "check",
                    str(config),
                    str(config.with_name("train_data.py")),
                ],
                env,
            )
            for phase, script in SCRIPTS.items():
                if phase.startswith("caching_") and settings["cache_policy"] == "off":
                    continue
                data = config.with_name("train_data.py")
                argv = [
                    kogine,
                    "run",
                    script,
                    "--config",
                    str(config if phase == "training" else data),
                ]
                if phase == "training":
                    argv += ["--set", "SAMPLE_EVERY=0", "--set", "SAVE_MERGED=false"]
                if phase == "training" and resume_path is not None:
                    argv += [
                        "--set",
                        "RESUME=" + str(resume_path),
                        "--set",
                        "STEPS=" + str(settings["steps"]),
                        "--set",
                        "FINAL_CHECKPOINT=true",
                    ]
                self._command(job, phase, argv, env)
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
        except GroupStillRunning as exception:
            status, error = "unknown", str(exception)
        except Exception as exception:
            status, error = "failed", str(exception)
        finally:
            safe = status != "unknown" and self._terminate_group()
            with self.lock:
                self.active = None
                if safe:
                    self.process = None
                    lease.close()
                    self._finish(
                        job, "stopped" if self.cancel.is_set() else status, error
                    )
                else:
                    self.orphan_lease = lease
                    self._finish(
                        job,
                        "unknown",
                        "Worker group exit could not be confirmed. GPU exclusion is retained; stop the worker externally and restart ComfyUI. "
                        + (error or ""),
                    )

    def _finish(self, job, status, error=None):
        with self.lock:
            if job["phases"]:
                job["phases"][-1]["ended_at"] = time.time()
            self._update(job, status=status, error=error, finished_at=time.time())
        if error:
            logging.error("[KohakuMoss] Job %s %s: %s", job["job_id"], status, error)
        else:
            logging.info("[KohakuMoss] Job %s: %s", job["job_id"], status)
