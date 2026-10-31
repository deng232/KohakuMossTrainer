"""Manager's install hook must keep dependencies out of ComfyUI's venv."""

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
    def test_repository_root_hook_runs_the_same_separate_venv_installer(self):
        with patch.dict(os.environ, {}, clear=True), patch("subprocess.run") as run:
            runpy.run_path(str(PLUGIN.parents[1] / "install.py"), run_name="__main__")
        args, kwargs = run.call_args
        self.assertEqual(
            Path(args[0][1]), PLUGIN.parents[1] / "scripts/install/install.sh"
        )
        self.assertEqual(kwargs["cwd"], PLUGIN.parents[1])
        self.assertEqual(
            kwargs["env"]["KOHAKUMOSS_VENV"], str(PLUGIN.parents[1] / ".venv")
        )
        self.assertTrue(kwargs["check"])

    def test_manager_hook_runs_native_installer_in_checkout(self):
        with patch.dict(os.environ, {}, clear=True), patch("subprocess.run") as run:
            runpy.run_path(str(PLUGIN / "install.py"), run_name="__main__")
        args, kwargs = run.call_args
        self.assertEqual(
            Path(args[0][1]), PLUGIN.parents[1] / "scripts/install/install.sh"
        )
        self.assertEqual(kwargs["cwd"], PLUGIN.parents[1])
        self.assertTrue(kwargs["check"])

    def test_hook_respects_paths_and_propagates_install_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            installer = repo / "scripts/install/install.sh"
            installer.parent.mkdir(parents=True)
            installer.touch()
            venv = repo / "training env"
            with (
                patch.dict(
                    os.environ, KOHAKUMOSS_REPO=str(repo), KOHAKUMOSS_VENV=str(venv)
                ),
                patch(
                    "subprocess.run",
                    side_effect=subprocess.CalledProcessError(1, "bash"),
                ) as run,
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    runpy.run_path(str(PLUGIN / "install.py"), run_name="__main__")
            self.assertEqual(run.call_args.kwargs["env"]["KOHAKUMOSS_VENV"], str(venv))
            self.assertEqual(run.call_args.kwargs["cwd"], repo)

    def test_hook_rejects_installing_into_comfyui_environment(self):
        with (
            patch.dict(os.environ, KOHAKUMOSS_VENV=sys.prefix),
            patch("subprocess.run") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "separate"):
                runpy.run_path(str(PLUGIN / "install.py"), run_name="__main__")
            run.assert_not_called()

    def test_native_installer_targets_separate_venv_with_active_comfyui(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = root / "bin"
            tools.mkdir()
            venv = root / "trainer env"
            (venv / "bin").mkdir(parents=True)
            log = root / "commands"
            for name, body in {
                "nvidia-smi": "echo 'CUDA Version: 13.0'",
                "uv": 'printf "%s\\n" "$*" >> "$INSTALL_LOG"',
            }.items():
                executable = tools / name
                executable.write_text(f"#!{shutil.which('sh')}\n" + body + "\n")
                executable.chmod(0o755)
            doctor = venv / "bin/kohakumoss-doctor"
            doctor.write_text(f"#!{shutil.which('sh')}\nexit 0\n")
            doctor.chmod(0o755)
            env = dict(
                os.environ,
                PATH=str(tools) + os.pathsep + os.environ["PATH"],
                VIRTUAL_ENV="/comfyui/venv",
                KOHAKUMOSS_VENV=str(venv),
                INSTALL_LOG=str(log),
            )
            subprocess.run(
                ["bash", str(PLUGIN.parents[1] / "scripts/install/install.sh")],
                cwd=root,
                env=env,
                check=True,
            )
            commands = log.read_text().splitlines()
            self.assertEqual(len(commands), 2)
            for command in commands:
                self.assertIn(f"--python {venv}/bin/python", command)
                self.assertNotIn("/comfyui/venv", command)


if __name__ == "__main__":
    unittest.main()
