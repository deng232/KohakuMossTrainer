import importlib
import math
import types
import unittest
from unittest.mock import patch


class NodeTests(unittest.TestCase):
    def test_two_nodes_discover_models_and_monitor_is_noop(self):
        folders = types.ModuleType("folder_paths")
        folders.models_dir = "models"
        folders.get_filename_list = lambda category: [f"{category}.safetensors"]
        folders.get_output_directory = lambda: "output"
        with patch.dict("sys.modules", {"folder_paths": folders}):
            nodes = importlib.import_module("kohaku_comfy.nodes")
            inputs = nodes.KohakuMossTraining.INPUT_TYPES()
            self.assertIn(
                "diffusion_models.safetensors", inputs["required"]["TRANSFORMER"][0]
            )
            self.assertEqual(len(nodes.NODE_CLASS_MAPPINGS), 2)
            self.assertTrue(nodes.KohakuMossTraining.OUTPUT_NODE)
            self.assertTrue(math.isnan(nodes.KohakuMossTraining.IS_CHANGED()))
            self.assertEqual(nodes.KohakuMossMonitor().monitor({"job_id": "done"}), ())
