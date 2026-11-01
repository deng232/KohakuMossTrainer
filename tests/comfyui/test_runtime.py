import tempfile
import threading
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import kohaku_comfy
from kohaku_comfy import runtime
from kohaku_comfy.live_logger import register
from kohaku_comfy.session import STORE
from kohakumoss.logging import build_logger


class RuntimeTests(unittest.TestCase):
    def test_synchronous_call_emits_real_logger_metrics_before_return(self):
        register()
        events = []
        calling_thread = threading.get_ident()
        with tempfile.TemporaryDirectory() as directory:

            def run(knobs):
                self.assertEqual(threading.get_ident(), calling_thread)
                logger = build_logger(
                    "comfyui_live", directory, "run", "test", {}, True
                )
                try:
                    logger.log({"loss": 0.42, "lr": 0.001}, 1)
                    job = STORE.list_jobs("17", "workflow")[0]
                    self.assertEqual(job["status"], "TRAINING")
                    self.assertTrue(
                        any(name == "kohakumoss.metrics" for name, _ in events)
                    )
                finally:
                    logger.close()
                return types.SimpleNamespace(step=1)

            native = types.SimpleNamespace(
                torch=None,
                run=run,
                run_name=lambda k: "run",
                configure=lambda s: {
                    "OUTPUT_DIR": str(Path(directory).parent),
                    "RUN_NAME": Path(directory).name,
                    "STEPS": 1,
                    "DEVICES": (0,),
                },
            )
            native.run_name = lambda k: Path(directory).name
            management = types.SimpleNamespace(
                unload_all_models=lambda: None, soft_empty_cache=lambda: None
            )
            comfy = types.ModuleType("comfy")
            comfy.model_management = management
            with (
                patch.dict("sys.modules", {"comfy": comfy}),
                patch.object(kohaku_comfy, "native", native, create=True),
                patch.object(runtime, "host_state", lambda *args: nullcontext()),
                patch.object(runtime, "prepare", lambda *args: None),
                patch.object(
                    STORE, "send", lambda name, data: events.append((name, data))
                ),
            ):
                result = runtime.execute(
                    {}, "17", {"workflow": {"extra": {"kohakumoss_id": "workflow"}}}
                )
            self.assertEqual(result["status"], "COMPLETED")
            self.assertEqual(result["run_dir"], directory)
            self.assertNotIn("state", result)

    def test_systemexit_becomes_node_error_and_session_is_reusable(self):
        def configure(settings):
            raise SystemExit("invalid knobs")

        native = types.SimpleNamespace(torch=None, configure=configure)
        comfy = types.ModuleType("comfy")
        comfy.model_management = types.SimpleNamespace(soft_empty_cache=lambda: None)
        with (
            patch.dict("sys.modules", {"comfy": comfy}),
            patch.object(kohaku_comfy, "native", native, create=True),
            patch.object(runtime, "host_state", lambda *args: nullcontext()),
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid knobs"):
                runtime.execute({}, "bad")
        self.assertEqual(STORE.list_jobs("bad")[0]["status"], "FAILED")
        with STORE.training("next"):
            pass
