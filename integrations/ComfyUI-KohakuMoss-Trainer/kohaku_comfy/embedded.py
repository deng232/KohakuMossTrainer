"""Library orchestration; no child interpreter, process group or background training."""

import logging
import threading
from pathlib import Path


class RunLog(logging.Handler):
    def __init__(self, path, observer):
        super().__init__()
        self.path = Path(path)
        self.observer = observer
        self.owner = threading.get_ident()

    def emit(self, record):
        if threading.get_ident() != self.owner:
            return
        line = self.format(record)
        rotated = self.path.exists() and self.path.stat().st_size > 64 * 1024 * 1024
        with self.path.open("w" if rotated else "a", encoding="utf-8") as file:
            file.write(line + "\n")
        self.observer({"phase": "log", "message": line, "rotated": rotated})


def run(runtime, job, resume_path, observer, cancelled):
    from kohakumoss.cli.config.main import write_train
    from kohakumoss.cli.config.starter import TrainChoice
    from kohakumoss.cli.config.validate import TRAIN, compose
    from kohakumoss.training.library import cache, session, train

    from .config import configure_generated

    settings, config = job["settings"], Path(job["config"])
    with session(observer, cancelled):
        handler = RunLog(job["log"], observer)
        for name in ("kohakumoss", "kohakumoss_data", "kohakumoss_kernels"):
            logger = logging.getLogger(name)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        observer({"phase": "validating"})
        if resume_path is None:
            write_train(
                runtime.repo,
                TrainChoice(
                    weights=settings["transformer"],
                    vae=settings["vae"],
                    text_encoder=settings["text_encoder"],
                    data=settings["dataset_path"],
                    source=settings["dataset_source"],
                    algo=settings["algorithm"],
                    name=job["run_name"],
                    steps=max(101, settings["steps"]),
                    vram=12 if settings["memory_profile"] == "vram_12g" else "auto",
                    precision=settings["precision"],
                    prompts=("",),
                ),
                config,
                False,
            )
            configure_generated(
                config, settings, Path(job["run_dir"]).parent, Path(job["cache_dir"])
            )
        k = compose(runtime.repo, TRAIN, [config], {})
        if resume_path is None and settings["steps"] <= 100:
            k["SCHEDULE"] = {**k["SCHEDULE"], "warmup": settings["steps"] // 10}
            with config.open("a", encoding="utf-8") as file:
                file.write(f"SCHEDULE = {k['SCHEDULE']!r}\n")
        # The host owns the process. Dataset workers may use threads, never processes.
        k.update(
            PARALLEL="none",
            ENCODE_DEVICES=[],
            NUM_WORKERS=0,
            DATA_SCAN_WORKERS=0,
            SAMPLE_EVERY=0,
            SAVE_MERGED=False,
            RESUME=str(resume_path) if resume_path else "",
            STEPS=settings["steps"],
            FINAL_CHECKPOINT=True,
        )
        if settings["cache_policy"] != "off":
            for kind, script in (("text", "cache_text"), ("latents", "cache_latents")):
                data = compose(
                    runtime.repo,
                    f"scripts/data/{script}.py",
                    [config.with_name("train_data.py")],
                    {},
                )
                data.update(NUM_WORKERS=0, DATA_SCAN_WORKERS=0)
                cache(data, kind)
        return train(k)
