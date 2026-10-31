"""Small CPU checks; no ComfyUI, torch, weights or GPU required."""

import importlib.util
import io
import os
import runpy
import sys
import tempfile
import time
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


FAKE_CLI = r'''
import json, os, pathlib, subprocess, sys, time
args = sys.argv[1:]
if args[0] == "--version":
    print("fake runtime")
elif args[0] == "new":
    weights = pathlib.Path(args[args.index("--weights") + 1])
    if weights.is_file() and ("--vae" not in args or "--text-encoder" not in args):
        sys.exit("single-file WEIGHTS requires VAE and TEXT_ENCODER before validation")
    out = pathlib.Path(args[args.index("--out") + 1])
    name = args[args.index("--name") + 1]
    data = args[args.index("--data") + 1]
    out.write_text('"""Drives scripts/train/moss.py."""\nTEXT_ENCODE = {}\nVAE_ENCODE = {}\nCAPTION = {"template":"tags"}\n'
                   + "RUN_NAME = " + repr(name) + "\nDATASETS = " + repr([{"path": data}]) + "\n")
    out.with_name("train_data.py").write_text('"""Drives scripts/data/inspect_data.py."""\nTEXT_ENCODER_KWARGS = {}\nIMAGE_ENCODER_KWARGS = {}\nTEXT_ENCODE = {}\nVAE_ENCODE = {}\n')
elif args[0] == "check":
    print("configs checked")
elif args[0] == "run":
    script = args[1]
    print(script, flush=True)
    if script.endswith("moss.py"):
        values = {}
        exec(pathlib.Path(args[args.index("--config") + 1]).read_text(), values)
        for index, arg in enumerate(args):
            if arg == "--set":
                key, value = args[index + 1].split("=", 1)
                values[key] = int(value) if key == "STEPS" else value
        root = pathlib.Path(values["OUTPUT_DIR"]) / values["RUN_NAME"]
        root.mkdir(parents=True, exist_ok=True)
        (root / "metrics.jsonl").write_text('{"step":1,"loss":0.5,"lr":0.0001}\n')
        ckpt = root / "checkpoints" / "step_00000001"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "state.json").write_text('{"step":1}')
        (ckpt / "trainable.safetensors").write_bytes(b"checkpoint")
        marker = pathlib.Path(values["DATASETS"][0]["path"]) / "behavior"
        behavior = marker.read_text() if marker.exists() else "ok"
        if behavior == "wait":
            time.sleep(60)
        if behavior == "descendant":
            code = "import os,pathlib,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path(" + repr(str(marker.parent / "child.pid")) + ").write_text(str(os.getpid())); time.sleep(60)"
            subprocess.Popen([sys.executable, "-c", code])
            time.sleep(60)
        if behavior == "fail":
            print("CUDA out of memory (test fixture)", flush=True)
            sys.exit(7)
        export = root / "export"
        export.mkdir(exist_ok=True)
        final = root / "checkpoints" / ("step_%08d" % values["STEPS"])
        final.mkdir(parents=True, exist_ok=True)
        (final / "state.json").write_text(json.dumps({"step": values["STEPS"]}))
        (final / "trainable.safetensors").write_bytes(b"checkpoint")
        (export / (values["RUN_NAME"] + "-comfy.safetensors")).write_bytes(("export step " + str(values["STEPS"])).encode())
'''


class JobTests(unittest.TestCase):
    def setUp(self):
        from kohaku_comfy.jobs import JobManager
        from kohaku_comfy.runtime import Runtime

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "trainer"
        self.bin = self.root / "venv" / "bin"
        self.bin.mkdir(parents=True)
        for name in ("kohakumoss-config", "kogine", "kohakumoss-doctor"):
            file = self.bin / name
            file.write_text(f"#!{sys.executable}\n" + FAKE_CLI)
            file.chmod(0o755)
        (self.bin / "python").symlink_to(sys.executable)
        for script in (
            "train/moss.py",
            "data/inspect_data.py",
            "data/cache_text.py",
            "data/cache_latents.py",
        ):
            p = self.repo / "scripts" / script
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        self.data = self.root / "data"
        self.data.mkdir()
        model = self.root / "weights.safetensors"
        model.write_bytes(b"weights")
        self.settings = dict(
            dataset_path=str(self.data),
            transformer=str(model),
            text_encoder=str(model),
            vae=str(model),
        )
        self.runtime = Runtime(
            self.repo,
            self.bin.parent,
            self.root / "workspace",
            self.root,
            self.root / "loras",
        )
        self.manager = JobManager(self.runtime)
        self.addCleanup(self.cleanup_job)

    def cleanup_job(self):
        for job in self.manager.list():
            if job["status"] not in (
                "completed",
                "failed",
                "stopped",
                "interrupted",
                "unknown",
            ):
                self.manager.stop(job["job_id"])
                self.wait(job["job_id"], {"stopped", "failed"})

    def wait(self, job_id, statuses):
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            job = self.manager.get(job_id)
            if job["status"] in statuses:
                return job
            time.sleep(0.02)
        self.fail(f"job did not reach {statuses}: {job}")

    def test_background_pipeline_deduplicates_and_publishes(self):
        job = self.manager.submit(self.settings, "prompt1:node1")
        self.assertEqual(
            self.manager.submit(self.settings, "prompt1:node1")["job_id"], job["job_id"]
        )
        done = self.wait(job["job_id"], {"completed", "failed"})
        self.assertEqual(done["status"], "completed", done.get("error"))
        self.assertTrue(Path(done["published"]).is_file())
        self.assertEqual(
            [c["phase"] for c in done["commands"]],
            [
                "validating",
                "validating",
                "inspecting",
                "caching_text",
                "caching_latents",
                "training",
            ],
        )
        self.assertEqual(done["metrics"]["loss"], 0.5)
        generate = done["commands"][0]["argv"]
        self.assertEqual(generate[generate.index("--vae") + 1], self.settings["vae"])
        self.assertEqual(
            generate[generate.index("--text-encoder") + 1],
            self.settings["text_encoder"],
        )
        again = self.manager.submit(self.settings, "prompt2:node1")
        self.assertNotEqual(job["job_id"], again["job_id"])
        self.wait(again["job_id"], {"completed", "failed"})

    def test_stop_resume_and_single_job_exclusion(self):
        (self.data / "behavior").write_text("wait")
        job = self.manager.submit(self.settings, "prompt1:node1")
        self.wait(job["job_id"], {"training"})
        deadline = time.monotonic() + 3
        while (
            not self.manager.get(job["job_id"])["checkpoints"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        with self.assertRaisesRegex(ValueError, "active|running|busy"):
            self.manager.submit(self.settings, "prompt2:node1")
        self.manager.stop(job["job_id"])
        stopped = self.wait(job["job_id"], {"stopped"})
        self.assertTrue(stopped["checkpoints"])
        (self.data / "behavior").write_text("ok")
        self.manager.resume(job["job_id"])
        done = self.wait(job["job_id"], {"completed", "failed"})
        self.assertEqual(done["status"], "completed", done.get("error"))
        self.assertEqual(done["resume_step"], 1)

    def test_nonzero_exit_fails_and_history_survives_restart(self):
        from kohaku_comfy.jobs import JobManager

        (self.data / "behavior").write_text("fail")
        job = self.manager.submit(self.settings, "prompt1:node1")
        failed = self.wait(job["job_id"], {"failed"})
        self.assertIn("out of memory", failed["error"])
        self.assertEqual(failed["exit_code"], 7)
        reopened = JobManager(self.runtime)
        self.assertEqual(reopened.get(job["job_id"])["status"], "failed")

    def test_completed_checkpoint_can_continue_and_only_lora_is_published(self):
        output = self.runtime.publish_dir / "portraits"
        job = self.manager.submit(
            {**self.settings, "output_path": str(output)}, "first:node1"
        )
        done = self.wait(job["job_id"], {"completed", "failed"})
        self.assertEqual(done["status"], "completed", done.get("error"))
        self.assertTrue(done["can_resume"])
        previous = Path(done["published"])
        self.assertEqual(previous.parent, output)
        self.assertEqual(previous.read_bytes(), b"export step 2000")
        checkpoint = str(Path(done["run_dir"]) / "checkpoints/step_00002000")
        settings = {
            "start_mode": "resume_latest",
            "resume_checkpoint": checkpoint,
            "steps": 500,
            "output_path": str(output),
        }
        resumed = self.manager.submit(settings, "second:node1")
        again = self.manager.submit(settings, "second:node1")
        self.assertEqual(resumed["job_id"], again["job_id"])
        finished = self.wait(job["job_id"], {"completed", "failed"})
        self.assertEqual(finished["status"], "completed", finished.get("error"))
        self.assertEqual(finished["resume_step"], 2000)
        self.assertEqual(finished["settings"]["steps"], 2500)
        train = [
            command
            for command in finished["commands"]
            if command["phase"] == "training"
        ][-1]
        self.assertIn("STEPS=2500", train["argv"])
        self.assertIn("RESUME=" + checkpoint, train["argv"])
        self.assertEqual(Path(finished["published"]).read_bytes(), b"export step 2500")
        self.assertEqual(previous.read_bytes(), b"export step 2000")
        self.assertEqual(len(list(output.glob("*.safetensors"))), 2)
        self.assertFalse((output / "job.json").exists())
        self.assertTrue(Path(job["run_dir"]).is_relative_to(self.runtime.workspace))

    def test_lora_output_path_cannot_escape_registered_directory(self):
        with self.assertRaises(ValueError):
            self.manager.submit(
                {**self.settings, "output_path": str(self.root / "outside")},
                "first:node1",
            )

    def test_restart_marks_live_worker_unknown_and_blocks_new_jobs(self):
        from kohaku_comfy.jobs import JobManager

        (self.data / "behavior").write_text("wait")
        job = self.manager.submit(self.settings, "prompt1:node1")
        self.wait(job["job_id"], {"training"})
        deadline = time.monotonic() + 5
        while (
            not self.manager.get(job["job_id"]).get("process")
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertIsNotNone(self.manager.get(job["job_id"]).get("process"))
        reopened = JobManager(self.runtime)
        self.assertEqual(reopened.get(job["job_id"])["status"], "unknown")
        with self.assertRaisesRegex(ValueError, "active|unknown"):
            reopened.submit(self.settings, "prompt2:node1")
        with self.assertRaises(ValueError):
            reopened.stop(job["job_id"])
        self.manager.stop(job["job_id"])
        self.wait(job["job_id"], {"stopped"})

    def test_off_policy_skips_cache_commands(self):
        job = self.manager.submit(
            {**self.settings, "cache_policy": "off"}, "prompt1:node1"
        )
        done = self.wait(job["job_id"], {"completed", "failed"})
        self.assertEqual(done["status"], "completed", done.get("error"))
        self.assertEqual(
            [c["phase"] for c in done["commands"]],
            ["validating", "validating", "inspecting", "training"],
        )

    def test_monitor_failure_reaps_descendant_that_ignores_sigterm(self):
        import signal

        from kohaku_comfy import jobs

        (self.data / "behavior").write_text("descendant")
        original = self.manager._update
        groups = []

        def failing_monitor(job, **values):
            if values.get("process"):
                groups.append(values["process"]["pgid"])
            if "log_generation" in values and (self.data / "child.pid").exists():
                raise OSError("simulated monitoring failure")
            return original(job, **values)

        try:
            with (
                patch.object(jobs, "LOG_BYTES", 1),
                patch.object(jobs, "STOP_TIMEOUT", 0.2),
                patch.object(self.manager, "_update", side_effect=failing_monitor),
            ):
                job = self.manager.submit(self.settings, "prompt1:node1")
                done = self.wait(job["job_id"], {"failed"})
                self.assertIn("monitoring failure", done["error"])
                self.assertFalse(
                    jobs.group_alive(groups[-1]),
                    "loader descendant survived after GPU lease release",
                )
        finally:
            for group in groups:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_unconfirmed_group_exit_retains_cross_process_gpu_lock(self):
        from kohaku_comfy import jobs

        try:
            with (
                patch.object(jobs, "group_alive", return_value=True),
                patch.object(jobs, "STOP_TIMEOUT", 0),
                patch.object(jobs, "KILL_TIMEOUT", 0),
            ):
                job = self.manager.submit(self.settings, "prompt1:node1")
                done = self.wait(job["job_id"], {"unknown"})
                self.assertIn("exclusion is retained", done["error"])
            reopened = jobs.JobManager(self.runtime)
            with self.assertRaisesRegex(ValueError, "busy"):
                reopened.submit(self.settings, "prompt2:node1")
        finally:
            self.manager.orphan_lease.close()


if __name__ == "__main__":
    unittest.main()
