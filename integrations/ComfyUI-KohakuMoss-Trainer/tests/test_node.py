"""Exercise our node contract with lightweight ComfyUI boundary doubles."""

import asyncio
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

PLUGIN = Path(__file__).resolve().parents[1]


class NodeTests(unittest.TestCase):
    def setUp(self):
        for name in list(sys.modules):
            if name == "kohaku_test_plugin" or name.startswith("kohaku_test_plugin."):
                del sys.modules[name]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model = self.root / "model.safetensors"
        model.touch()
        folders = types.ModuleType("folder_paths")
        folders.get_filename_list = lambda category: ["model.safetensors"]
        folders.get_folder_paths = lambda category: [str(self.root)]
        folders.get_full_path = lambda category, name: str(self.root / name)
        folders.get_input_directory = lambda: str(self.root)
        folders.get_output_directory = lambda: str(self.root / "output")

        class Field:
            def __init__(self, id=None, **kwargs):
                self.id = id
                self.__dict__.update(kwargs)

        class Schema:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class Output:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.__dict__.update(kwargs)

        io = types.SimpleNamespace(ComfyNode=object, Schema=Schema, NodeOutput=Output)
        for name in ("String", "Int", "Float", "Boolean", "Combo"):
            setattr(io, name, types.SimpleNamespace(Input=Field, Output=Field))
        latest = types.ModuleType("comfy_api.latest")
        latest.io = io
        latest.ComfyExtension = object
        execution = types.ModuleType("comfy_execution.utils")
        execution.get_executing_context = lambda: types.SimpleNamespace(
            prompt_id="prompt-1", node_id="node-1", list_index=0
        )
        server = types.ModuleType("server")
        server.PromptServer = types.SimpleNamespace(instance=None)
        aiohttp = types.ModuleType("aiohttp")
        aiohttp.web = types.SimpleNamespace()
        management = types.SimpleNamespace(
            unload_all_models=Mock(),
            soft_empty_cache=Mock(),
            throw_exception_if_processing_interrupted=Mock(),
            processing_interrupted=lambda: False,
        )
        comfy = types.ModuleType("comfy")
        comfy.model_management = management
        self.modules = {
            "comfy": comfy,
            "comfy.model_management": management,
            "folder_paths": folders,
            "comfy_api": types.ModuleType("comfy_api"),
            "comfy_api.latest": latest,
            "comfy_execution": types.ModuleType("comfy_execution"),
            "comfy_execution.utils": execution,
            "server": server,
            "aiohttp": aiohttp,
        }
        self.context = patch.dict(sys.modules, self.modules)
        self.context.start()
        self.addCleanup(self.context.stop)
        spec = importlib.util.spec_from_file_location(
            "kohaku_test_plugin",
            PLUGIN / "__init__.py",
            submodule_search_locations=[str(PLUGIN)],
        )
        self.plugin = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.plugin
        spec.loader.exec_module(self.plugin)

    def test_only_one_output_node_and_unchanged_workflow_executes_again(self):
        extension = asyncio.run(self.plugin.comfy_entrypoint())
        nodes = asyncio.run(extension.get_node_list())
        self.assertEqual(len(nodes), 1)
        schema = nodes[0].define_schema()
        self.assertEqual(schema.node_id, "KohakuMossTrainer")
        self.assertTrue(schema.is_output_node)
        self.assertNotEqual(
            nodes[0].fingerprint_inputs(), nodes[0].fingerprint_inputs()
        )
        self.assertEqual([v.id for v in schema.outputs], ["job_id"])

    def test_repository_root_loads_node_sidebar_and_trainer_paths(self):
        repo = PLUGIN.parents[1]
        name = "kohaku_test_plugin.fork"
        spec = importlib.util.spec_from_file_location(
            name, repo / "__init__.py", submodule_search_locations=[str(repo)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        extension = asyncio.run(module.comfy_entrypoint())
        nodes = asyncio.run(extension.get_node_list())
        self.assertEqual(
            [node.define_schema().node_id for node in nodes], ["KohakuMossTrainer"]
        )
        self.assertTrue((repo / module.WEB_DIRECTORY / "trainer.js").is_file())
        self.assertTrue((repo / module.WEB_DIRECTORY / "dataset_browser.js").is_file())
        server = importlib.import_module(
            name + ".integrations.ComfyUI-KohakuMoss-Trainer.server"
        )
        with patch.dict("os.environ", {}, clear=True):
            runtime = server.get_manager().runtime
        self.assertEqual(runtime.repo, repo)
        self.assertFalse(hasattr(runtime, "venv"))

    def test_node_resolves_model_files_and_uses_prompt_identity(self):
        nodes = importlib.import_module("kohaku_test_plugin.nodes")

        class Manager:
            def submit(manager, settings, key):
                self.assertEqual(key, "prompt-1:node-1")
                self.assertEqual(
                    settings["transformer"], str(self.root / "model.safetensors")
                )
                return {"job_id": "abc", "status": "created"}

        with patch.object(nodes, "get_manager", return_value=Manager()):
            result = nodes.KohakuMossTrainer.execute(
                dataset_path=str(self.root),
                transformer="model.safetensors",
                text_encoder="model.safetensors",
                vae="model.safetensors",
            )
            self.assertEqual(result.args, ("abc",))
            with self.assertRaises(ValueError):
                nodes.KohakuMossTrainer.execute(transformer="../model.safetensors")

    def test_size_and_batch_controls_are_visible_without_advanced(self):
        nodes = importlib.import_module("kohaku_test_plugin.nodes")
        fields = {
            field.id: field for field in nodes.KohakuMossTrainer.define_schema().inputs
        }
        for name in (
            "target_mp",
            "resolution_mode",
            "max_tokens",
            "batch_size",
            "batching",
            "grad_accum",
        ):
            with self.subTest(name=name):
                self.assertFalse(fields[name].advanced)
        self.assertEqual(
            fields["dataset_path"].extra_dict["widgetType"], "KOHAKU_DATASET_FOLDER"
        )
        self.assertEqual(fields["target_mp"].default, 1.0)
        self.assertNotIn("target_area", fields)
        self.assertNotIn("save_merged", fields)
        self.assertEqual(fields["output_path"].default, str(self.root))
        self.assertFalse(fields["output_path"].advanced)
        self.assertEqual(
            fields["resume_checkpoint"].extra_dict["widgetType"],
            "KOHAKU_CHECKPOINT_FOLDER",
        )

    def test_learning_rate_keeps_small_decimal_values(self):
        nodes = importlib.import_module("kohaku_test_plugin.nodes")
        field = next(
            field
            for field in nodes.KohakuMossTrainer.define_schema().inputs
            if field.id == "learning_rate"
        )
        self.assertEqual(field.default, 0.0001)
        self.assertIs(field.round, False)
        self.assertEqual(field.step, 0.000001)
        with patch.object(nodes, "get_manager") as manager:
            manager.return_value.submit.return_value = {
                "job_id": "learning-rate",
                "status": "created",
            }
            nodes.KohakuMossTrainer.execute(
                dataset_path=str(self.root),
                transformer="model.safetensors",
                text_encoder="model.safetensors",
                vae="model.safetensors",
                learning_rate=0.0001,
            )
            settings = manager.return_value.submit.call_args.args[0]
            self.assertEqual(settings["learning_rate"], 0.0001)

    def test_all_advanced_fields_have_hover_explanations(self):
        nodes = importlib.import_module("kohaku_test_plugin.nodes")
        for field in nodes.KohakuMossTrainer.define_schema().inputs:
            if getattr(field, "advanced", False):
                with self.subTest(name=field.id):
                    self.assertTrue(getattr(field, "tooltip", "").strip())

    def test_resume_uses_checkpoint_without_requiring_workflow_model_inputs(self):
        nodes = importlib.import_module("kohaku_test_plugin.nodes")

        class Manager:
            def submit(manager, settings, key):
                self.assertEqual(
                    settings["resume_checkpoint"], "/run/checkpoints/step_00002000"
                )
                self.assertEqual(settings["steps"], 1000)
                return {"job_id": "resumed", "status": "created"}

        with patch.object(nodes, "get_manager", return_value=Manager()):
            result = nodes.KohakuMossTrainer.execute(
                start_mode="resume_latest",
                resume_checkpoint="/run/checkpoints/step_00002000",
                steps=1000,
            )
            self.assertEqual(result.args, ("resumed",))

    def test_example_widget_values_stay_aligned_after_seed(self):
        nodes = importlib.import_module("kohaku_test_plugin.nodes")
        fields = nodes.KohakuMossTrainer.define_schema().inputs
        widget_names = []
        for field in fields:
            widget_names.append(field.id)
            control = getattr(field, "control_after_generate", None)
            if control or (control is None and field.id in ("seed", "noise_seed")):
                widget_names.append("control_after_generate")
        workflow = json.loads(
            (PLUGIN / "examples/kohakumoss_single_node_workflow.json").read_text()
        )
        values = workflow["nodes"][0]["widgets_values"]
        self.assertEqual(len(values), len(widget_names))
        restored = dict(zip(widget_names, values))
        self.assertEqual(restored["caption_template"], "native")
        self.assertFalse(
            set(restored)
            & {
                "sample_prompts",
                "sample_every",
                "sample_steps",
                "sample_width",
                "sample_height",
            }
        )
        self.assertEqual(restored["save_format"], "comfy")

    def test_native_progress_defers_comfy_interrupt_until_driver_can_save(self):
        module = importlib.import_module("kohaku_test_plugin.server")
        management = self.modules["comfy"].model_management

        class Interrupt(BaseException):
            pass

        management.InterruptProcessingException = Interrupt
        management.processing_interrupted = lambda: False
        management.interrupt_current_processing = Mock()
        utils = types.ModuleType("comfy.utils")
        utils.ProgressBar = Mock()
        utils.ProgressBar.return_value.update_absolute.side_effect = Interrupt()
        module.PromptServer.instance = types.SimpleNamespace(send_sync=Mock())
        with patch.dict(sys.modules, {"comfy.utils": utils}):
            module.send_event(
                {"phase": "training", "step": 1, "total": 2, "node_id": "1"}
            )
        management.interrupt_current_processing.assert_called_once_with(True)

    def test_routes_report_errors_and_reject_cross_origin_requests(self):
        module = importlib.import_module("kohaku_test_plugin.server")
        handlers = {}

        class Routes:
            def get(self, path):
                return lambda fn: handlers.setdefault(("GET", path), fn)

            def post(self, path):
                return lambda fn: handlers.setdefault(("POST", path), fn)

        self_root = self.root

        class Manager:
            def get(self, job_id):
                if job_id == "missing":
                    raise KeyError(job_id)
                if job_id == "resume":
                    from kohaku_test_plugin.kohaku_comfy.config import DEFAULTS

                    settings = {
                        **DEFAULTS,
                        **{
                            key: str(self_root / "model.safetensors")
                            for key in ("transformer", "vae", "text_encoder")
                        },
                    }
                    return {"job_id": job_id, "can_resume": True, "settings": settings}
                return {"job_id": job_id, "status": "training"}

        module.web.json_response = lambda data, status=200: (status, data)
        module.PromptServer.instance = types.SimpleNamespace(routes=Routes())
        module.register_routes()
        self.assertEqual(len(handlers), 8)
        request = types.SimpleNamespace(
            headers={}, host="localhost:8188", match_info={"job_id": "ok"}
        )
        with patch.object(module, "get_manager", return_value=Manager()):
            handler = handlers["GET", "/kohakumoss/jobs/{job_id}"]
            self.assertEqual(
                asyncio.run(handler(request)),
                (200, {"job_id": "ok", "status": "training"}),
            )
            request.match_info["job_id"] = "resume"

            async def json_body():
                return {"additional_steps": 2, "checkpoint": "latest"}

            request.json = json_body
            code, body = asyncio.run(
                handlers["POST", "/kohakumoss/jobs/{job_id}/resume"](request)
            )
            self.assertEqual(code, 200)
            self.assertEqual(
                body["prompt"]["1"]["inputs"]["start_mode"], "resume_latest"
            )
            self.assertEqual(body["prompt"]["1"]["inputs"]["steps"], 2)
            request.match_info["job_id"] = "missing"
            self.assertEqual(asyncio.run(handler(request))[0], 404)
            request.headers["Origin"] = "http://untrusted.invalid"
            self.assertEqual(asyncio.run(handler(request))[0], 403)
