"""Manager installer; runs in ComfyUI's interpreter and preserves its Torch stack."""

import importlib.metadata as metadata
import subprocess
import sys
import tempfile
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

ROOT = Path(__file__).resolve().parent
TORCH = SpecifierSet(">=2.13,<2.15")


def check_torch(version):
    if Version(version).is_prerelease or Version(version) not in TORCH:
        raise RuntimeError(
            f"KohakuMoss requires torch{TORCH}; ComfyUI has {version}. "
            "Installation stopped. This installer will not replace Torch."
        )


def install_command(interpreter, constraints):
    return [
        interpreter,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--upgrade-strategy",
        "only-if-needed",
        "--constraint",
        str(constraints),
        "--editable",
        str(ROOT) + "[cuda]",
    ]


def main():
    try:
        version = metadata.version("torch")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Install ComfyUI's supported Torch first; this installer does not install Torch"
        ) from error
    check_torch(version)
    constraints = []
    for name in ("torch", "torchvision", "torchaudio", "torchsde"):
        try:
            constraints.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            pass
    with tempfile.TemporaryDirectory(prefix="kohakumoss-install-") as directory:
        path = Path(directory) / "constraints.txt"
        path.write_text("\n".join(constraints) + "\n", encoding="utf-8")
        subprocess.run(install_command(sys.executable, path), check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; import kohakumoss; "
            "assert Path(kohakumoss.__file__).resolve().is_relative_to(Path(sys.argv[1]) / 'src'), "
            "'KohakuMoss must import from this checkout'; "
            "from kohakumoss.training.entry import run; "
            "from kohakumoss.logging import LOGGERS; "
            "from kohakumoss_data.cache.build import build_text, build_latents",
            str(ROOT),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
