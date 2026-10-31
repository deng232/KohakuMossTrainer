"""The trainer uses the Python environment that loaded this ComfyUI node."""

import importlib.metadata
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Runtime:
    repo: Path
    workspace: Path
    dataset_root: Path
    publish_dir: Path

    def check(self):
        if sys.platform != "linux":
            raise ValueError(
                "KohakuMoss embedded training currently supports Linux only"
            )
        try:
            importlib.metadata.version("kohakumoss")
        except importlib.metadata.PackageNotFoundError:
            raise ValueError(
                "Trainer dependencies are missing from ComfyUI's Python environment; "
                "run this node's install.py through Manager repair, then restart ComfyUI"
            ) from None
