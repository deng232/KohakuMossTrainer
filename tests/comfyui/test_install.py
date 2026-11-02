import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


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
        with (
            patch.object(
                self.installer.importlib.util,
                "find_spec",
                side_effect=lambda name: object() if name == "pip" else None,
            ),
            patch.object(self.installer.shutil, "which", return_value=None),
        ):
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
        editables = [cmd[i + 1] for i, value in enumerate(cmd) if value == "--editable"]
        self.assertEqual(
            editables,
            [str(self.installer.ROOT) + "[cuda]"],
        )

    def test_uv_module_targets_manager_interpreter_without_pip_only_flags(self):
        with patch.object(
            self.installer.importlib.util, "find_spec", return_value=object()
        ):
            cmd = self.installer.install_command("python-comfy", "constraints.txt")
            check = self.installer.package_command("python-comfy", "check")
        self.assertEqual(
            cmd[:7],
            ["python-comfy", "-m", "uv", "pip", "install", "--python", "python-comfy"],
        )
        self.assertEqual(
            check,
            ["python-comfy", "-m", "uv", "pip", "check", "--python", "python-comfy"],
        )
        self.assertNotIn("--upgrade-strategy", cmd)
        self.assertNotIn("--upgrade", cmd)
        self.assertIn("--constraint", cmd)
        self.assertNotIn(str(self.installer.ROOT / "LyCORIS"), cmd)

    def test_uv_executable_and_missing_tools(self):
        with (
            patch.object(self.installer.importlib.util, "find_spec", return_value=None),
            patch.object(self.installer.shutil, "which", return_value="/bin/uv"),
        ):
            self.assertEqual(
                self.installer.package_command("python-comfy", "check"),
                ["/bin/uv", "pip", "check", "--python", "python-comfy"],
            )
        with (
            patch.object(self.installer.importlib.util, "find_spec", return_value=None),
            patch.object(self.installer.shutil, "which", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "uv or pip"):
                self.installer.package_command("python-comfy", "install")

    def test_install_does_not_require_local_lycoris(self):
        with (
            patch.object(self.installer, "ROOT", Path("/missing-kohakumoss-checkout")),
            patch.object(
                self.installer,
                "package_command",
                return_value=["uv", "pip", "install", "--python", "python-comfy"],
            ),
        ):
            cmd = self.installer.install_command("python-comfy", "constraints.txt")
        self.assertIn("/missing-kohakumoss-checkout[cuda]", cmd)
        self.assertEqual(cmd.count("--editable"), 1)
