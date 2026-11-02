"""Load the actual custom-node entry point without a ComfyUI installation."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web


class EntryPointTests(unittest.TestCase):
    def test_entrypoint_exports_nodes_and_registers_routes(self):
        server = types.ModuleType("server")
        server.PromptServer = types.SimpleNamespace(
            instance=types.SimpleNamespace(
                app=web.Application(),
                routes=web.RouteTableDef(),
                send_sync=lambda *a: None,
            )
        )
        folders = types.ModuleType("folder_paths")
        folders.get_input_directory = lambda: "input"
        folders.get_output_directory = lambda: "output"
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "test_kohaku_extension",
            root / "__init__.py",
            submodule_search_locations=[str(root)],
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules, {"server": server, "folder_paths": folders, spec.name: module}
        ):
            spec.loader.exec_module(module)
            self.assertEqual(len(module.NODE_CLASS_MAPPINGS), 3)
            paths = {route.path for route in server.PromptServer.instance.routes}
            self.assertIn("/kohakumoss/checkpoints", paths)
            self.assertIn("/kohakumoss/datasets", paths)
            self.assertIn("/kohakumoss/jobs", paths)


if __name__ == "__main__":
    unittest.main()
