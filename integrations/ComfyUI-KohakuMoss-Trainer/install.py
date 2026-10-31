"""Manager install/repair: install into the running ComfyUI Python environment."""

import importlib.metadata
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path


def install():
    from packaging.specifiers import SpecifierSet

    repo = Path(__file__).resolve().parents[2]
    torch = importlib.metadata.version("torch")
    if not SpecifierSet(">=2.13,<2.15").contains(torch):
        raise RuntimeError(
            f"ComfyUI has torch {torch}; trainer requires >=2.13,<2.15. "
            "Choose a compatible ComfyUI image; the installer will not replace torch."
        )
    # Pin installed GPU packages so the resolver cannot silently replace the host stack.
    with tempfile.TemporaryDirectory(prefix="kohakumoss-install-") as directory:
        constraints = Path(directory) / "constraints.txt"
        pins = []
        for dist in importlib.metadata.distributions():
            name = dist.metadata.get("Name", "").lower().replace("_", "-")
            if name in {
                "torch",
                "torchvision",
                "torchaudio",
                "triton",
            } or name.startswith(("nvidia-", "cuda-")):
                pins.append(f"{name}=={dist.version}")
        constraints.write_text("\n".join(pins) + "\n", encoding="utf-8")
        if importlib.util.find_spec("uv") is not None:
            command = [
                sys.executable,
                "-m",
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
            ]
        else:
            command = [sys.executable, "-m", "pip", "install"]
        subprocess.run(
            command
            + [
                "--constraint",
                str(constraints),
                "--editable",
                str(repo) + "[cuda]",
            ],
            check=True,
        )
    print("KohakuMoss installed into ComfyUI's Python. Restart ComfyUI.", flush=True)


if __name__ == "__main__":
    install()
