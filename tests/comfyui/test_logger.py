import tempfile
import unittest
from pathlib import Path

from kohaku_comfy.live_logger import register
from kohaku_comfy.session import STORE
from kohakumoss.logging import LOGGERS, build_logger


class LoggerTests(unittest.TestCase):
    def test_registry_publishes_before_return_and_preserves_resume_history(self):
        register()
        register()
        self.assertIn("comfyui_live", LOGGERS)
        with tempfile.TemporaryDirectory() as directory:
            with STORE.training("17") as job:
                STORE.configure(job, directory, "run", 10)
                logger = build_logger(
                    "comfyui_live", directory, "run", "test", {}, True
                )
                try:
                    logger.log({"loss": 0.8}, 1)
                    logger.log({"loss": 0.6}, 2)
                    self.assertEqual(STORE.snapshot(job)["last_metric_step"], 2)
                    self.assertEqual(STORE.snapshot(job)["status"], "TRAINING")
                    logger.resume_at(1)
                    logger.log({"loss": 0.7}, 2)
                finally:
                    logger.close()
                lines = (Path(directory) / "metrics.jsonl").read_text().splitlines()
                self.assertEqual(len(lines), 2)
                self.assertTrue((Path(directory) / "metrics.1.jsonl").exists())
                self.assertEqual(STORE.history(job)["rows"][-1]["metrics"]["loss"], 0.7)

    def test_unbound_logger_fails_before_opening_file(self):
        register()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "session"):
                build_logger("comfyui_live", directory, "run", "test", {}, True)
            self.assertFalse((Path(directory) / "metrics.jsonl").exists())
