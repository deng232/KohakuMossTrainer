"""ComfyUI-Manager install/repair hook for the separate trainer environment."""

import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    repo = (
        Path(os.environ.get("KOHAKUMOSS_REPO", Path(__file__).resolve().parents[2]))
        .expanduser()
        .resolve()
    )
    installer = repo / "scripts/install/install.sh"
    if sys.platform != "linux":
        raise RuntimeError(
            "KohakuMoss Trainer integration currently supports Linux only"
        )
    if not installer.is_file():
        raise RuntimeError(f"Missing trainer checkout at {repo}; set KOHAKUMOSS_REPO")
    venv = (
        Path(os.environ.get("KOHAKUMOSS_VENV", repo / ".venv")).expanduser().absolute()
    )
    if venv.resolve() == Path(sys.prefix).resolve():
        raise RuntimeError(
            "KOHAKUMOSS_VENV must be separate from ComfyUI's environment"
        )
    env = dict(os.environ, KOHAKUMOSS_VENV=str(venv))
    env["PATH"] = (
        str(Path(sys.executable).parent)
        + os.pathsep
        + os.environ.get("PATH", os.defpath)
    )
    subprocess.run(["bash", str(installer)], cwd=repo, env=env, check=True)
