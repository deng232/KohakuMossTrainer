"""Separate trainer environment and native command entry points."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPTS = {
    "inspecting": "scripts/data/inspect_data.py",
    "caching_text": "scripts/data/cache_text.py",
    "caching_latents": "scripts/data/cache_latents.py",
    "training": "scripts/train/moss.py",
}


@dataclass(frozen=True)
class Runtime:
    repo: Path
    venv: Path
    workspace: Path
    dataset_root: Path
    publish_dir: Path

    def check(self):
        if sys.platform != "linux":
            raise ValueError(
                "KohakuMoss Trainer integration currently supports Linux only"
            )
        if self.venv.resolve() == Path(sys.prefix).resolve():
            raise ValueError(
                "Use a separate KohakuMoss venv, not ComfyUI's Python environment"
            )
        for script in SCRIPTS.values():
            if not (self.repo / script).is_file():
                raise ValueError(f"Missing trainer script: {self.repo / script}")
        for name in ("python", "kohakumoss-config", "kogine"):
            path = self.venv / "bin" / name
            if not path.is_file() or not os.access(path, os.X_OK):
                raise ValueError(
                    f"Missing executable {path}; run this node's install.py via "
                    "ComfyUI-Manager install/repair or manually (see README)"
                )

    def environment(self):
        keys = (
            "HOME",
            "USER",
            "LOGNAME",
            "TMPDIR",
            "LANG",
            "LC_ALL",
            "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
            "CUDA_HOME",
            "CUDA_PATH",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "PYTORCH_CUDA_ALLOC_CONF",
            "PYTORCH_ALLOC_CONF",
        )
        env = {k: os.environ[k] for k in keys if k in os.environ}
        env.update(
            PATH=str(self.venv / "bin")
            + os.pathsep
            + os.environ.get("PATH", os.defpath),
            VIRTUAL_ENV=str(self.venv),
            PYTHONUNBUFFERED="1",
            PYTHONNOUSERSITE="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
        )
        for name, subdir in (
            ("KOHAKUMOSS_KERNEL_CACHE", "kohaku"),
            ("LYCORIS_KERNEL_CACHE_DIR", "lycoris"),
            ("TRITON_CACHE_DIR", "triton"),
            ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
            ("TILELANG_CACHE_DIR", "tilelang"),
        ):
            env[name] = str(self.workspace / "kernel-cache" / subdir)
        return env
