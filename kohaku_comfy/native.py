"""Optional GPU dependencies and existing Kohaku library APIs."""

import folder_paths
import torch

from kohakumoss.cli.config.starter import TrainChoice, train_starter
from kohakumoss.cli.config.validate import TRAIN, compose, repo_root
from kohakumoss.encoders import text_encoder, vae_encoder
from kohakumoss.encoders.text.encoder import resolve_text_plan
from kohakumoss.training.build.model import data_knobs, training_text_plan
from kohakumoss.training.entry import run
from kohakumoss.training.knobs import check, run_name
from kohakumoss.training.weights import component
from kohakumoss.weights import TEXT, VAE
from kohakumoss_data import DataConfig, MossDataset
from kohakumoss_data.cache.build import build_latents, build_text

from .config import overrides, resolve_models
from .live_logger import register

__all__ = [
    "torch",
    "text_encoder",
    "vae_encoder",
    "data_knobs",
    "training_text_plan",
    "resolve_text_plan",
    "run",
    "run_name",
    "component",
    "TEXT",
    "VAE",
    "DataConfig",
    "MossDataset",
    "build_latents",
    "build_text",
    "configure",
]


def configure(settings):
    register()
    root = repo_root()
    own = resolve_models(overrides(settings), folder_paths.get_full_path)
    device = torch.device("cuda", own["DEVICES"][0])
    if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        raise ValueError(f"Training GPU {device} is not available")
    if torch.distributed.is_initialized():
        raise RuntimeError(
            "An existing distributed process group is incompatible with single-GPU training"
        )
    choice = TrainChoice(
        weights=own.get("WEIGHTS", ""),
        data=settings.get("dataset_path", ""),
        source=settings.get("source", "kohya"),
        algo=(
            own.get("ALGO")
            if own.get("ALGO") in ("lora", "loha", "lokr", "oft", "full")
            else "lora"
        ),
        vram="auto",
        precision="fp16" if own.get("PRECISION") == "fp16" else "bf16",
        name=own.get("RUN_NAME", ""),
        steps=own.get("STEPS", 0),
        prompts=("",),
    )
    starter = train_starter(root, choice)
    knobs = compose(root, TRAIN, starter.layers, {**starter.own, **own})
    knobs = check(knobs)
    assert knobs["PARALLEL"] == "none" and knobs["ENCODE_DEVICES"] == ()
    assert knobs["DEVICES"] == (device.index,)
    assert knobs["NUM_WORKERS"] == 0 and knobs["DATA_SCAN_WORKERS"] == 1
    return knobs
