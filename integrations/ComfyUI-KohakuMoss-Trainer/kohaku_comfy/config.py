"""Validate workflow values and specialize configs written by the native generator."""

import hashlib
import json
import math
import re
from pathlib import Path

DEFAULTS = {
    "run_name": "moss-lora",
    "dataset_path": "",
    "dataset_source": "kohya",
    "transformer": "",
    "text_encoder": "",
    "vae": "",
    "algorithm": "lora",
    "rank": 32,
    "alpha": 32.0,
    "steps": 2000,
    "learning_rate": 1e-4,
    "memory_profile": "vram_12g",
    "save_every": 500,
    "cache_policy": "offline_both",
    "start_mode": "new",
    "resume_job_id": "",
    "resume_checkpoint": "latest",
    "precision": "bf16",
    "max_tokens": 0,
    "target_mp": 1.0,
    "batch_size": 1,
    "batching": "tokens",
    "grad_accum": 1,
    "microbatches": 4,
    "seed": 0,
    "caption_template": "native",
    "caption_variants": 4,
    "text_dropout": 0.1,
    "resolution_mode": "resize",
    "save_format": "comfy",
    "checkpoint_export": True,
    "keep_last": 3,
    "output_root": "",
    "publish_to_loras": True,
    "output_path": "",
}
CHOICES = {
    "dataset_source": ("kohya", "folder"),
    "algorithm": ("lora", "lokr", "loha", "dora", "oft"),
    "memory_profile": ("vram_12g", "auto"),
    "cache_policy": ("offline_both", "auto", "off"),
    "start_mode": ("new", "resume_latest"),
    "precision": ("bf16", "fp16"),
    "batching": ("tokens", "tokens_ffd", "count"),
    "caption_template": ("native", "tags", "nl"),
    "resolution_mode": ("resize", "bucket", "shifted_square", "square_crop"),
    "save_format": ("comfy", "hf", "both", "none"),
}
INT_LIMITS = {
    "rank": (1, 10000),
    "steps": (1, 10000000),
    "save_every": (0, 10000000),
    "max_tokens": (0, 1048576),
    "batch_size": (1, 4096),
    "grad_accum": (1, 65536),
    "microbatches": (1, 65536),
    "seed": (0, 2**63 - 1),
    "caption_variants": (1, 1024),
    "keep_last": (0, 10000),
}
FLOAT_LIMITS = {
    "target_mp": (0.000256, 16.777216),
    "alpha": (0.000001, 100000),
    "learning_rate": (1e-12, 1),
    "text_dropout": (0, 1),
}
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")


def inside(path, root):
    path, root = Path(path).expanduser().resolve(), Path(root).expanduser().resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path must be inside {root}: {path}")
    return path


def validate(values, dataset_root, workspace):
    if "target_area" in values:
        values = dict(values)
        area = values.pop("target_area")
        if (
            type(area) is not int
            or not 256 <= area <= 16777216
            or "target_mp" in values
        ):
            raise ValueError("Use target_mp instead of target_area")
        values["target_mp"] = area / 1_000_000
    # Ignore retired fields in saved jobs from older versions.
    values = {
        k: v
        for k, v in values.items()
        if k
        not in {
            "save_merged",
            "sample_prompts",
            "sample_every",
            "sample_steps",
            "sample_width",
            "sample_height",
        }
    }
    unknown = values.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    settings = {**DEFAULTS, **values}
    for key, default in DEFAULTS.items():
        value = settings[key]
        if isinstance(default, bool):
            valid = type(value) is bool
        elif isinstance(default, int):
            valid = (
                type(value) is int and INT_LIMITS[key][0] <= value <= INT_LIMITS[key][1]
            )
        elif isinstance(default, float):
            valid = (
                type(value) in (int, float)
                and math.isfinite(value)
                and FLOAT_LIMITS[key][0] <= value <= FLOAT_LIMITS[key][1]
            )
        else:
            valid = (
                isinstance(value, str) and "\x00" not in value and len(value) <= 65536
            )
        if not valid or (key in CHOICES and value not in CHOICES[key]):
            raise ValueError(f"Invalid {key}: {value!r}")
    if not SAFE_NAME.fullmatch(settings["run_name"]):
        raise ValueError(
            "run_name must use 1-80 letters, numbers, dots, dashes or underscores, starting with a letter/number"
        )
    if not settings["dataset_path"]:
        raise ValueError("Select a dataset directory")
    data = inside(settings["dataset_path"], dataset_root)
    if not data.is_dir():
        raise ValueError(f"Dataset directory does not exist: {data}")
    settings["dataset_path"] = str(data)
    for key in ("transformer", "text_encoder", "vae"):
        path = Path(settings[key]).expanduser().resolve()
        if not settings[key] or not path.is_file():
            raise ValueError(f"Missing {key} weight file: {path}")
        settings[key] = str(path)
    output = settings["output_root"]
    settings["output_root"] = str(
        inside(Path(workspace) / (output or "runs"), workspace)
    )
    if settings["publish_to_loras"] and settings["save_format"] not in (
        "comfy",
        "both",
    ):
        raise ValueError("Publishing requires save_format=comfy or both")
    return settings


def cache_namespace(settings):
    # Encoder ids/specs do not identify different fine-tuned weight files.
    identity = [settings["dataset_path"]]
    for key in ("text_encoder", "vae"):
        path = Path(settings[key])
        info = path.stat()
        identity.append((str(path), info.st_size, info.st_mtime_ns))
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]


def configure_generated(train, settings, output_parent, cache_dir):
    """Append only validated literals to newly generated, integration-owned configs."""
    knobs = {
        "TRANSFORMER": settings["transformer"],
        "TEXT_ENCODER": settings["text_encoder"],
        "VAE": settings["vae"],
        "OUTPUT_DIR": str(output_parent),
        "CACHE_DIR": str(cache_dir),
        "DIM": settings["rank"],
        "ALPHA": settings["alpha"],
        "STEPS": settings["steps"],
        "LR": settings["learning_rate"],
        "SAVE_EVERY": settings["save_every"],
        "TARGET_AREA": round(settings["target_mp"] * 1_000_000),
        "BATCH_SIZE": settings["batch_size"],
        "BATCHING": settings["batching"],
        "GRAD_ACC": settings["grad_accum"],
        "MICROBATCHES": settings["microbatches"],
        "SEED": settings["seed"],
        "CAPTION_VARIANTS": settings["caption_variants"],
        "TEXT_DROP": settings["text_dropout"],
        "RESOLUTION_MODE": settings["resolution_mode"],
        "SAMPLE_EVERY": 0,
        "SAVE_FORMAT": settings["save_format"],
        "SAVE_MERGED": False,
        "CHECKPOINT_EXPORT": settings["checkpoint_export"],
        "FINAL_CHECKPOINT": True,
        "KEEP_LAST": settings["keep_last"],
        "LOGGER": "console",
        "RESUME": "",
    }
    if settings["max_tokens"]:
        knobs["MAX_TOKENS"] = settings["max_tokens"]
    if settings["cache_policy"] == "off":
        knobs.update(TEXT_CACHE="off", LATENT_CACHE="off")
    elif settings["cache_policy"] == "offline_both":
        knobs.update(TEXT_CACHE="offline", LATENT_CACHE="offline")
    with train.open("a", encoding="utf-8") as f:
        f.write("\n# Validated ComfyUI workflow overrides.\n")
        for name, value in knobs.items():
            f.write(f"{name} = {value!r}\n")
        if settings["caption_template"] != "native":
            f.write(f"CAPTION = {{'template': {settings['caption_template']!r}}}\n")
    data = train.with_name(f"{train.stem}_data.py")
    with data.open("a", encoding="utf-8") as f:
        f.write(
            "\n# Keep native factories and use the same weights and plans as training.\n"
        )
        f.write(
            "TEXT_ENCODER_KWARGS = {**TEXT_ENCODER_KWARGS, "
            f"'path': {settings['text_encoder']!r}, 'plan': dict(TEXT_ENCODE)}}\n"
        )
        f.write(
            "IMAGE_ENCODER_KWARGS = {**IMAGE_ENCODER_KWARGS, "
            f"'path': {settings['vae']!r}, 'plan': dict(VAE_ENCODE)}}\n"
        )
