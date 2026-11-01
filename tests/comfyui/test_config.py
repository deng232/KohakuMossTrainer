import tempfile
import unittest
from pathlib import Path

from kohaku_comfy.config import overrides, resolve_models


class ConfigTests(unittest.TestCase):
    def test_final_host_constraints_cannot_be_overridden(self):
        settings = {
            "gpu": 2,
            "RUN_NAME": "test",
            "OUTPUT_DIR": "out",
            "advanced_json": '{"PARALLEL":"ddp","ENCODE_DEVICES":[1],"NUM_WORKERS":8,"DATA_SCAN_WORKERS":8,"DEVICES":[0,1],"LOGGER":"console"}',
            "SCHEDULE": '{"mode":"cosine","warmup":1,"end":-1}',
        }
        got = overrides(settings)
        self.assertEqual(got["DEVICES"], (2,))
        self.assertEqual(got["ENCODE_DEVICES"], ())
        self.assertEqual(got["PARALLEL"], "none")
        self.assertEqual(got["LOGGER"], "comfyui_live")
        self.assertEqual(got["NUM_WORKERS"], 0)
        self.assertEqual(got["DATA_SCAN_WORKERS"], 1)
        self.assertEqual(got["SCHEDULE"]["warmup"], 1)

    def test_json_shapes_and_run_path_are_validated(self):
        for settings in (
            {"advanced_json": "[]"},
            {"advanced_json": '{"steps":3}'},
            {"SCHEDULE": "null"},
            {"DATASETS": "{}"},
            {"RUN_NAME": "../escape"},
            {"RUN_NAME": "a\\b"},
            {"advanced_json": '{"LR":NaN}'},
            {"advanced_json": '{"TEXT_DEVICE":"cuda:1"}', "gpu": 0},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                overrides(settings)

    def test_explicit_dataset_json_is_native_and_paths_are_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            got = overrides(
                {
                    "DATASETS": '[{"source":"kohya","path":"images","include_root":true}]',
                    "WEIGHTS": directory,
                    "OUTPUT_DIR": directory,
                    "TEXT_ENCODE": '{"precision":"int8"}',
                }
            )
            self.assertEqual(got["DATASETS"][0]["source"], "kohya")
            self.assertTrue(got["DATASETS"][0]["include_root"])
            self.assertEqual(got["WEIGHTS"], str(Path(directory).resolve()))
            self.assertEqual(got["TEXT_ENCODE"], {"precision": "int8"})

    def test_discovery_uses_extra_model_paths(self):
        def find(folder, name):
            self.assertEqual((folder, name), ("diffusion_models", "moss.safetensors"))
            return "/extra/models/moss.safetensors"

        got = resolve_models(
            {
                "TRANSFORMER": "moss.safetensors",
                "TEXT_ENCODER": "/other/te.safetensors",
            },
            find,
        )
        self.assertEqual(got["TRANSFORMER"], "/extra/models/moss.safetensors")
        self.assertEqual(got["TEXT_ENCODER"], "/other/te.safetensors")

    def test_empty_optional_dataset_list_keeps_starter_dataset(self):
        self.assertNotIn("DATASETS", overrides({"DATASETS": "[]"}))
