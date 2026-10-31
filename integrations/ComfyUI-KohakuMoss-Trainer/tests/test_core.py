"""Small CPU checks; no ComfyUI, torch, weights or GPU required."""

import importlib.util
import io
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "kohaku_comfy",
    PLUGIN / "kohaku_comfy" / "__init__.py",
    submodule_search_locations=[str(PLUGIN / "kohaku_comfy")],
)
support = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = support
spec.loader.exec_module(support)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "\u6570\u636e \u96c6"
        self.data.mkdir()
        self.model = self.root / "weights.safetensors"
        self.model.write_bytes(b"weights")
        self.workspace = self.root / "jobs"
        self.values = dict(
            dataset_path=str(self.data),
            transformer=str(self.model),
            text_encoder=str(self.model),
            vae=str(self.model),
        )

    def test_output_browser_lists_only_directories_inside_lora_root(self):
        from kohaku_comfy.files import output_directories

        loras = self.root / "loras"
        (loras / "portraits").mkdir(parents=True)
        (loras / "model.safetensors").touch()
        (loras / "escape").symlink_to(self.data, target_is_directory=True)
        listing = output_directories(loras)
        self.assertEqual(listing["directories"], [str(loras / "portraits")])
        self.assertIsNone(listing["parent"])
        self.assertEqual(output_directories(loras, "portraits")["parent"], str(loras))
        for invalid in ("../", "escape", "model.safetensors"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                output_directories(loras, invalid)

    def test_rejects_traversal_unknown_options_and_nonfinite_numbers(self):
        from kohaku_comfy.config import validate

        for extra in (
            {"run_name": "../oops"},
            {"steps": 0},
            {"steps": True},
            {"learning_rate": float("nan")},
            {"output_root": "../escape"},
            {"algorithm": "full"},
            {"python": "arbitrary"},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate({**self.values, **extra}, self.root, self.workspace)
        outside = self.root / "outside"
        outside.mkdir()
        (self.data / "escape").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            validate(
                {**self.values, "dataset_path": str(self.data / "escape")},
                self.data,
                self.workspace,
            )

    def test_generated_train_and_cache_share_paths_and_compute_settings(self):
        from kohaku_comfy.config import configure_generated, validate

        settings = validate(
            {
                **self.values,
                "target_mp": 0.802816,
                "save_merged": True,
                "sample_every": 10,
                "sample_prompts": "legacy",
            },
            self.root,
            self.workspace,
        )
        train = self.root / "train.py"
        data = self.root / "train_data.py"
        train.write_text(
            '"""Drives scripts/train/moss.py."""\n'
            'CAPTION = {"template": "tags", "shuffle": True}\n'
            'TEXT_ENCODE = {"memory": "ring", "batch": 16}\n'
            'VAE_ENCODE = {"memory": "offload"}\n'
        )
        data.write_text(
            '"""Drives scripts/data/inspect_data.py."""\n'
            "from kohakuengine import use_config\n"
            f"use_config({str(train)!r})\n"
            'TEXT_ENCODER = "kohakumoss.encoders.text_encoder"\n'
            'TEXT_ENCODER_KWARGS = {"path": "old", "device": "cuda"}\n'
            'IMAGE_ENCODER = "kohakumoss.encoders.vae_encoder"\n'
            'IMAGE_ENCODER_KWARGS = {"path": "old", "device": "cuda"}\n'
        )
        configure_generated(
            train, settings, self.workspace / "runs", self.workspace / "cache"
        )
        engine = types.ModuleType("kohakuengine")

        def use_config(path):
            sys._getframe(1).f_globals.update(
                {k: v for k, v in runpy.run_path(path).items() if k.isupper()}
            )

        engine.use_config = use_config
        engine.load_config_file = lambda path: types.SimpleNamespace(
            globals_dict=runpy.run_path(path)
        )
        with patch.dict(sys.modules, kohakuengine=engine):
            t, d = runpy.run_path(str(train)), runpy.run_path(str(data))
        self.assertEqual(t["TEXT_ENCODER"], str(self.model))
        self.assertEqual(d["TEXT_ENCODER"], "kohakumoss.encoders.text_encoder")
        self.assertEqual(d["TEXT_ENCODER_KWARGS"]["path"], str(self.model))
        self.assertEqual(d["IMAGE_ENCODER_KWARGS"]["path"], str(self.model))
        self.assertEqual(d["TEXT_ENCODER_KWARGS"]["plan"], t["TEXT_ENCODE"])
        self.assertEqual(d["IMAGE_ENCODER_KWARGS"]["plan"], t["VAE_ENCODE"])
        self.assertEqual(d["CACHE_DIR"], str(self.workspace / "cache"))
        self.assertEqual(t["CAPTION"], {"template": "tags", "shuffle": True})
        self.assertEqual(t["SAVE_FORMAT"], "comfy")
        self.assertEqual(t["SAMPLE_EVERY"], 0)
        self.assertFalse(t["SAVE_MERGED"])
        self.assertEqual(t["TARGET_AREA"], 802816)

    def test_cache_plans_resolve_without_eager_use_config_globals(self):
        from kohaku_comfy.config import configure_generated, validate

        settings = validate(self.values, self.root, self.workspace)
        train = self.root / "train.py"
        train.write_text(
            'TEXT_ENCODE = {"memory": "ring"}\nVAE_ENCODE = {"memory": "offload"}\n'
        )
        data = self.root / "train_data.py"
        data.write_text(
            "from kohakuengine import use_config\n"
            f"use_config({str(train)!r})\n"
            "TEXT_ENCODER_KWARGS = {'device': 'cuda'}\n"
            "IMAGE_ENCODER_KWARGS = {'device': 'cuda'}\n"
        )
        configure_generated(
            train, settings, self.workspace / "runs", self.workspace / "cache"
        )
        engine = types.ModuleType("kohakuengine")
        engine.use_config = lambda path: None
        engine.load_config_file = lambda path: types.SimpleNamespace(
            globals_dict=runpy.run_path(path)
        )
        with patch.dict(sys.modules, kohakuengine=engine):
            loaded = runpy.run_path(str(data))
        self.assertEqual(loaded["TEXT_ENCODER_KWARGS"]["plan"], {"memory": "ring"})
        self.assertEqual(loaded["IMAGE_ENCODER_KWARGS"]["plan"], {"memory": "offload"})

    def test_legacy_job_pixel_area_is_preserved_as_mp(self):
        from kohaku_comfy.config import validate

        settings = validate(
            {**self.values, "target_area": 1048576}, self.root, self.workspace
        )
        self.assertEqual(settings["target_mp"], 1.048576)
        self.assertNotIn("target_area", settings)

    def test_folder_upload_preserves_images_sidecars_and_rejects_escape(self):
        from kohaku_comfy.files import save_dataset_upload

        result = save_dataset_upload(
            self.data, "", "dataset/10_person/photo.png", io.BytesIO(b"image")
        )
        second = save_dataset_upload(
            self.data,
            result["upload_id"],
            "dataset/10_person/photo.txt",
            io.BytesIO(b"caption"),
        )
        self.assertEqual(result["dataset_path"], second["dataset_path"])
        directory = Path(result["dataset_path"])
        self.assertEqual((directory / "10_person/photo.png").read_bytes(), b"image")
        self.assertEqual((directory / "10_person/photo.txt").read_bytes(), b"caption")
        with self.assertRaises(FileExistsError):
            save_dataset_upload(
                self.data,
                result["upload_id"],
                "dataset/10_person/photo.txt",
                io.BytesIO(b"overwrite"),
            )
        self.assertEqual((directory / "10_person/photo.txt").read_bytes(), b"caption")
        for path in (
            "../escape.png",
            "/absolute.png",
            "dataset/../escape.png",
            "dataset/script.py",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                save_dataset_upload(
                    self.data, result["upload_id"], path, io.BytesIO(b"bad")
                )
        with self.assertRaises(ValueError):
            save_dataset_upload(
                self.data, "../bad", "dataset/image.png", io.BytesIO(b"bad")
            )


class FileTests(unittest.TestCase):
    def test_metrics_partial_tail_rotation_and_bounded_logs(self):
        from kohaku_comfy.files import read_jsonl, read_log

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            path.write_bytes(b'{"step":1,"loss":0.5}\ninvalid\n{"step":2')
            self.assertEqual(read_jsonl(path), [{"step": 1, "loss": 0.5}])
            path.write_text('{"step":3,"loss":0.1}\n')
            self.assertEqual(read_jsonl(path)[0]["step"], 3)
            path.write_text('{"step":4,"loss":NaN}\n')
            self.assertIsNone(read_jsonl(path)[0]["loss"])
            log = Path(tmp) / "worker.log"
            log.write_bytes(b"one\ntwo\n\xffthree\n")
            first = read_log(log, 0, 2)
            self.assertEqual(first["text"], "one\ntwo\n")
            self.assertEqual(read_log(log, first["offset"], 2)["text"], "\ufffdthree\n")
            log.write_bytes(b"reset\n")
            self.assertEqual(read_log(log, first["offset"], 2)["text"], "reset\n")

    def test_resume_ignores_partial_and_incomplete_checkpoints(self):
        from kohaku_comfy.files import checkpoints

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("step_00000001", "step_00000002.partial", "step_00000003"):
                p = root / "checkpoints" / name
                p.mkdir(parents=True)
                (p / "state.json").write_text('{"step":1}')
                if name != "step_00000003":
                    (p / "trainable.safetensors").touch()
            self.assertEqual([p.name for p in checkpoints(root)], ["step_00000001"])

    def test_publication_never_overwrites_existing_lora(self):
        from kohaku_comfy.files import publish

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            source.write_bytes(b"new")
            dest = root / "loras"
            dest.mkdir()
            (dest / source.name).write_bytes(b"existing")
            result = publish(source, dest)
            self.assertEqual(result.read_bytes(), b"new")
            self.assertEqual((dest / source.name).read_bytes(), b"existing")
            self.assertEqual(list(dest.glob("*.partial")), [])


class JobTests(ConfigTests):
    def setUp(self):
        super().setUp()
        from kohaku_comfy.jobs import JobManager
        from kohaku_comfy.runtime import Runtime

        self.events = []
        self.runtime = Runtime(
            self.root, self.root / "workspace", self.root, self.root / "loras"
        )
        self.manager = JobManager(self.runtime, self.events.append)
        check = patch.object(Runtime, "check")
        check.start()
        self.addCleanup(check.stop)
        self.settings = {**self.values, "publish_to_loras": False, "steps": 2}

    def test_submit_blocks_on_library_and_deduplicates(self):
        def run(runtime, job, resume, observer, cancelled):
            self.assertEqual(self.manager.active, job["job_id"])
            self.assertIsNone(resume)
            observer({"phase": "training", "step": 1, "total": 2, "loss": 0.5})
            self.assertEqual(self.manager.get(job["job_id"])["metrics"]["step"], 1)

        with patch("kohaku_comfy.jobs.run_library", side_effect=run) as library:
            job = self.manager.submit(self.settings, "prompt:1")
            self.assertEqual(job["status"], "completed")
            self.assertIsNone(self.manager.active)
            self.assertEqual(
                self.manager.submit(self.settings, "prompt:1")["job_id"], job["job_id"]
            )
            library.assert_called_once()
        self.assertEqual(self.events[-1]["phase"], "completed")

    def test_library_failure_releases_lease_and_reaches_node(self):
        with patch(
            "kohaku_comfy.jobs.run_library", side_effect=RuntimeError("build failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "build failed"):
                self.manager.submit(self.settings, "prompt:1")
        self.assertIsNone(self.manager.active)
        self.assertEqual(self.manager.list()[0]["status"], "failed")
        from kohaku_comfy.jobs import JobManager

        restored = JobManager(self.runtime)
        self.assertEqual(restored.list()[0]["error"], "build failed")
        lease = self.manager._claim()
        lease.close()

    def test_stop_is_cooperative_and_does_not_return_before_cleanup(self):
        import threading

        started, release, cleaned = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def run(runtime, job, resume, observer, cancelled):
            started.set()
            release.wait(3)
            self.assertTrue(cancelled())
            cleaned.set()

        with patch("kohaku_comfy.jobs.run_library", side_effect=run):
            thread = threading.Thread(
                target=lambda: self.manager.submit(self.settings, "prompt:1")
            )
            thread.start()
            self.assertTrue(started.wait(2))
            job_id = self.manager.active
            self.manager.stop(job_id)
            self.assertTrue(thread.is_alive())
            with self.assertRaisesRegex(ValueError, "active"):
                self.manager._claim()
            release.set()
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertTrue(cleaned.is_set())
            self.assertEqual(self.manager.get(job_id)["status"], "stopped")

    def test_resume_reuses_managed_checkpoint_and_adds_steps(self):
        def first(runtime, job, resume, observer, cancelled):
            Path(job["config"]).touch()
            folder = Path(job["run_dir"]) / "checkpoints/step_00000002"
            folder.mkdir(parents=True)
            (folder / "state.json").write_text('{"step": 2}')
            (folder / "trainable.safetensors").touch()

        with patch("kohaku_comfy.jobs.run_library", side_effect=first):
            job = self.manager.submit(self.settings, "prompt:1")

        def resumed(runtime, job, resume, observer, cancelled):
            self.assertEqual(resume.name, "step_00000002")
            self.assertEqual(job["settings"]["steps"], 5)

        with patch("kohaku_comfy.jobs.run_library", side_effect=resumed):
            again = self.manager.submit(
                {
                    "start_mode": "resume_latest",
                    "resume_job_id": job["job_id"],
                    "resume_checkpoint": "latest",
                    "steps": 3,
                },
                "next:1",
            )
        self.assertEqual(again["job_id"], job["job_id"])
        self.assertEqual(again["status"], "completed")
        self.assertEqual(again["resume_step"], 2)

    def test_failed_cleanup_retains_gpu_exclusion(self):
        from kohakumoss_common.events import CleanupFailed

        with patch(
            "kohaku_comfy.jobs.run_library", side_effect=CleanupFailed("worker alive")
        ):
            with self.assertRaisesRegex(RuntimeError, "worker alive"):
                self.manager.submit(self.settings, "prompt:1")
        self.assertEqual(self.manager.list()[0]["status"], "unknown")
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.manager._claim()
        self.manager.orphan_lease.close()

    def test_output_path_cannot_escape_loras(self):
        with self.assertRaises(ValueError):
            self.manager.submit(
                {**self.settings, "output_path": "/outside"}, "prompt:1"
            )


if __name__ == "__main__":
    unittest.main()
