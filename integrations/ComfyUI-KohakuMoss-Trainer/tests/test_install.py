"""Manager shares ComfyUI Python; native CLI installer remains separate."""

import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN = Path(__file__).resolve().parents[1]


class InstallTests(unittest.TestCase):
    def test_root_hook_installs_into_current_python_and_pins_gpu_stack(self):
        from types import SimpleNamespace

        distributions = [
            SimpleNamespace(metadata={"Name": "torch"}, version="2.14.1+cu130"),
            SimpleNamespace(metadata={"Name": "nvidia-cublas"}, version="13.0"),
        ]
        commands = []

        def run(argv, **kwargs):
            constraints = Path(argv[argv.index("--constraint") + 1]).read_text()
            self.assertIn("torch==2.14.1+cu130", constraints)
            self.assertIn("nvidia-cublas==13.0", constraints)
            commands.append(argv)

        with (
            patch("importlib.metadata.version", return_value="2.14.1+cu130"),
            patch("importlib.metadata.distributions", return_value=distributions),
            patch("subprocess.run", side_effect=run),
            patch("importlib.util.find_spec", return_value=None),
        ):
            runpy.run_path(str(PLUGIN.parents[1] / "install.py"), run_name="__main__")
        self.assertEqual(commands[0][:4], [sys.executable, "-m", "pip", "install"])
        self.assertEqual(commands[0][-1], str(PLUGIN.parents[1]) + "[cuda]")

    def test_uv_uses_current_interpreter_without_creating_venv(self):
        with (
            patch("importlib.metadata.version", return_value="2.14.1+cu130"),
            patch("importlib.util.find_spec", return_value=object()),
            patch("subprocess.run") as run,
        ):
            runpy.run_path(str(PLUGIN / "install.py"), run_name="__main__")
        self.assertEqual(
            run.call_args.args[0][:7],
            [sys.executable, "-m", "uv", "pip", "install", "--python", sys.executable],
        )

    def test_incompatible_torch_is_rejected_without_replacement(self):
        with (
            patch("importlib.metadata.version", return_value="2.10.0"),
            patch("subprocess.run") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "will not replace torch"):
                runpy.run_path(str(PLUGIN / "install.py"), run_name="__main__")
            run.assert_not_called()

    def test_install_failure_is_propagated(self):
        with (
            patch("importlib.metadata.version", return_value="2.14.1"),
            patch(
                "subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip")
            ),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                runpy.run_path(str(PLUGIN / "install.py"), run_name="__main__")

    def run_native_installer(
        self, banner="CUDA Version: 13.0", driver_version=13000, status=0
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = root / "bin"
            tools.mkdir()
            venv = root / "trainer env"
            (venv / "bin").mkdir(parents=True)
            log = root / "commands"
            for name, body in {
                "nvidia-smi": f"echo '{banner}'",
                "uv": 'printf "%s\\n" "$*" >> "$INSTALL_LOG"',
            }.items():
                executable = tools / name
                executable.write_text(f"#!{shutil.which('sh')}\n" + body + "\n")
                executable.chmod(0o755)
            (tools / "python3").symlink_to(sys.executable)
            (root / "ctypes.py").write_text(
                "class c_int:\n    value = 0\n"
                "def byref(value):\n    return value\n"
                "class CDLL:\n"
                "    def __init__(self, name):\n        assert name == 'libcuda.so.1'\n"
                "    def cuDriverGetVersion(self, value):\n"
                f"        value.value = {driver_version}\n        return {status}\n"
            )
            doctor = venv / "bin/kohakumoss-doctor"
            doctor.write_text(f"#!{shutil.which('sh')}\nexit 0\n")
            doctor.chmod(0o755)
            env = dict(
                os.environ,
                PATH=str(tools) + os.pathsep + os.environ["PATH"],
                PYTHONPATH=str(root),
                VIRTUAL_ENV="/comfyui/venv",
                KOHAKUMOSS_VENV=str(venv),
                INSTALL_LOG=str(log),
            )
            result = subprocess.run(
                ["bash", str(PLUGIN.parents[1] / "scripts/install/install.sh")],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
            )
            commands = log.read_text().splitlines() if log.exists() else []
            return result, commands, venv

    def test_native_installer_targets_separate_venv_with_active_comfyui(self):
        result, commands, venv = self.run_native_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertIn(f"--python {venv}/bin/python", command)
            self.assertNotIn("/comfyui/venv", command)

    def test_missing_smi_cuda_header_uses_driver_api(self):
        for version, index in ((13000, "cu130"), (12080, "cu128"), (12060, "cu126")):
            with self.subTest(version=version):
                result, commands, _ = self.run_native_installer(
                    banner="", driver_version=version
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(commands), 2)
                self.assertIn(f"https://download.pytorch.org/whl/{index}", commands[0])

    def test_unknown_or_unsupported_driver_does_not_install(self):
        for version, status in ((12050, 0), (0, 0), (13000, 999)):
            with self.subTest(version=version, status=status):
                result, commands, _ = self.run_native_installer(
                    banner="", driver_version=version, status=status
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(commands, [])


if __name__ == "__main__":
    unittest.main()
