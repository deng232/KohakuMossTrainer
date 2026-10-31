"""ComfyUI-Manager install/repair entry point for this trainer checkout."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(
            Path(__file__).resolve().parent
            / "integrations/ComfyUI-KohakuMoss-Trainer/install.py"
        ),
        run_name="__main__",
    )
