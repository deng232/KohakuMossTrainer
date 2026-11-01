"""Native ConsoleLogger with an additional in-process metrics subscriber."""

from kohakumoss.logging import LOGGERS
from kohakumoss.logging.sinks import ConsoleLogger

from .session import STORE


class LiveLogger(ConsoleLogger):
    def __init__(self, output_dir, name, project, config, resume=False):
        self.job_id = STORE.attach_logger(output_dir)
        super().__init__(output_dir)
        STORE.attach_logger(output_dir, self.path)

    def log(self, metrics, step):
        super().log(metrics, step)
        STORE.metrics(self.job_id, step, metrics)

    def resume_at(self, step):
        super().resume_at(step)
        STORE.resume_at(self.job_id, step)


def register():
    LOGGERS.add("comfyui_live", LiveLogger)
