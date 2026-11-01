import importlib.util
import unittest
from pathlib import Path


class InstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "kohaku_install", Path(__file__).parents[2] / "install.py"
        )
        cls.installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.installer)

    def test_incompatible_torch_is_rejected_before_installing(self):
        for version in ("2.12.0+cu128", "2.15.0", "2.14.0rc1"):
            with self.subTest(version=version), self.assertRaises(RuntimeError):
                self.installer.check_torch(version)
        self.installer.check_torch("2.13.1+cu130")

    def test_install_command_uses_same_interpreter_and_constraints(self):
        cmd = self.installer.install_command("python-comfy", "constraints.txt")
        self.assertEqual(cmd[:4], ["python-comfy", "-m", "pip", "install"])
        self.assertIn("constraints.txt", cmd)
        self.assertIn("--constraint", cmd)
        self.assertIn("only-if-needed", cmd)
        self.assertNotIn("--force-reinstall", cmd)
        self.assertNotIn("--no-deps", cmd)
        editable = cmd[cmd.index("--editable") + 1]
        self.assertEqual(editable, str(Path(__file__).resolve().parents[2]) + "[cuda]")
        self.assertFalse(any("git+" in arg for arg in cmd))
