"""JSON widget values and enforced single-GPU host constraints."""

import json
import math
from pathlib import Path

JSON_FIELDS = {
    "SCHEDULE": dict,
    "CAPTION": dict,
    "TEXT_ENCODE": dict,
    "VAE_ENCODE": dict,
    "MEMORY_OVERRIDES": dict,
    "DATASETS": list,
}
HOST = {
    "PARALLEL": "none",
    "ENCODE_DEVICES": (),
    "NUM_WORKERS": 0,
    "DATA_SCAN_WORKERS": 1,
    "LOGGER": "comfyui_live",
    "DRIVER": "native",
}
MODEL_FOLDERS = {
    "TRANSFORMER": "diffusion_models",
    "TEXT_ENCODER": "text_encoders",
    "VAE": "vae",
}


def invalid_constant(value):
    raise ValueError(f"JSON contains a non-finite number: {value}")


def json_value(value, kind, name):
    result = (
        json.loads(value, parse_constant=invalid_constant)
        if isinstance(value, str)
        else value
    )
    if not isinstance(result, kind):
        raise ValueError(f"{name} must be a JSON {kind.__name__}")
    json.dumps(result, allow_nan=False)
    return result


def overrides(settings):
    """Native knob overrides; host execution constraints always win."""
    values = {key: value for key, value in settings.items() if key.isupper()}
    if "target_megapixels" in settings:
        area = settings["target_megapixels"]
        if (
            isinstance(area, bool)
            or not isinstance(area, (int, float))
            or not math.isfinite(area)
            or not 0.01 <= area <= 64
        ):
            raise ValueError("Target area must be between 0.01 and 64 MP")
        values["TARGET_AREA"] = round(area * 1_000_000)
    values.update(
        json_value(settings.get("advanced_json", "{}"), dict, "advanced_json")
    )
    if any(not key.isupper() or key.startswith("_") for key in values):
        raise ValueError("Advanced knob names must be UPPER_CASE")
    for name, kind in JSON_FIELDS.items():
        if name in values:
            values[name] = json_value(values[name], kind, name)
    if values.get("DATASETS") == []:
        values.pop("DATASETS")
    if values.get("CAPTION") == {}:
        values.pop("CAPTION")
    for name in ("WEIGHTS", "OUTPUT_DIR", "CACHE_DIR"):
        if values.get(name):
            values[name] = str(Path(values[name]).expanduser().resolve())
    name = values.get("RUN_NAME", "")
    if (
        not isinstance(name, str)
        or name in (".", "..")
        or any(c in name for c in ("/", "\\", "\x00"))
    ):
        raise ValueError("RUN_NAME must be a single directory name")
    gpu = settings.get("gpu", 0)
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ValueError("gpu must be a non-negative device index")
    if values.get("TEXT_DEVICE", "") not in ("", "cpu", "cuda", f"cuda:{gpu}"):
        raise ValueError(
            "TEXT_DEVICE must use the selected GPU (or CPU with offline caching)"
        )
    if values.get("FINAL_CHECKPOINT") == "true":
        values["FINAL_CHECKPOINT"] = True
    elif values.get("FINAL_CHECKPOINT") == "false":
        values["FINAL_CHECKPOINT"] = False
    values.update(HOST, DEVICES=(gpu,))
    return values


def resolve_models(values, find_file):
    """Resolve discovered filenames through ComfyUI's extra model paths too."""
    values = dict(values)
    values.pop("VARIANT", None)
    for name, folder in MODEL_FOLDERS.items():
        value = values.get(name)
        if not isinstance(value, str) or not value.strip() or value == "(auto)":
            raise ValueError(f"Select an explicit {name} model file")
        path = Path(value).expanduser()
        if not path.is_absolute():
            found = find_file(folder, value)
            if not found:
                raise ValueError(f"{name} model file not found: {value}")
            path = Path(found)
        if not path.is_file():
            raise ValueError(f"{name} model file not found: {path}")
        values[name] = str(path.resolve())
    return values
