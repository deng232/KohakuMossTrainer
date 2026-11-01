"""The synchronous training output node and the frontend monitor's no-op backend."""

import folder_paths

from .config import COMPONENT_DEFAULT
from .runtime import execute


def string(default="", multiline=False, tooltip=""):
    return ("STRING", {"default": default, "multiline": multiline, "tooltip": tooltip})


def integer(default, minimum=0):
    return ("INT", {"default": default, "min": minimum, "max": 2**31 - 1})


def floating(default, minimum=0):
    return ("FLOAT", {"default": default, "min": minimum, "max": 1e10, "step": 0.0001})


def models(kind):
    return ([COMPONENT_DEFAULT, *folder_paths.get_filename_list(kind)],)


class KohakuMossTraining:
    CATEGORY = "KohakuMoss"
    FUNCTION = "train"
    RETURN_TYPES = ("TRAINING_JOB",)
    RETURN_NAMES = ("training_job",)
    OUTPUT_NODE = True
    DESCRIPTION = "Runs training in ComfyUI's prompt worker. Other queued workflows wait until training finishes. Interrupt is not a safe training stop."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "WEIGHTS": string(
                    folder_paths.models_dir,
                    tooltip="ComfyUI models directory or Kohaku-supported weights root",
                ),
                "TRANSFORMER": models("diffusion_models"),
                "TEXT_ENCODER": models("text_encoders"),
                "VAE": models("vae"),
                "VARIANT": (["base", "sft"],),
                "dataset_path": string(
                    tooltip="Server dataset path; use Upload dataset folder to select a local folder. Ignored when DATASETS JSON is supplied."
                ),
                "source": (
                    [
                        "kohya",
                        "folder",
                        "folder_nl",
                        "jsonl",
                        "moss_shard",
                        "parquet",
                        "webdataset",
                    ],
                ),
                "target_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 64,
                        "step": 0.1,
                        "tooltip": "Target area in megapixels (MP). 1 MP = 1,000,000 pixels.",
                    },
                ),
                "CAPTION": string('{"template":"tags"}', True),
                "LATENT_CACHE": (["offline", "online", "off"],),
                "TEXT_CACHE": (["offline", "online", "off"],),
                "CACHE_DIR": string(),
                "CACHE_DTYPE": (["bf16", "fp16", "fp32", "fp8"],),
                "TEXT_ENCODE": string("{}", True),
                "VAE_ENCODE": string("{}", True),
                "VAE_DTYPE": (["auto", "bf16", "fp16", "fp32"],),
                "ALGO": (["lora", "loha", "lokr", "oft", "full"],),
                "DIM": integer(32, 1),
                "ALPHA": floating(32),
                "LR": floating(0.0001),
                "STEPS": integer(1000, 1),
                "MAX_TOKENS": integer(12288, 2),
                "GRAD_ACC": integer(1, 1),
                "SCHEDULE": string(
                    '{"mode":"cosine","warmup":100,"end":-1,"min_value":0.1}', True
                ),
                "MEMORY_PROFILE": (
                    ["auto", "balanced", "speed", "low", "minimal", "manual"],
                ),
                "MEMORY_OVERRIDES": string("{}", True),
                "PRECISION": (["bf16", "fp16"],),
                "gpu": integer(0),
                "SAVE_EVERY": integer(500),
                "SAVE_MINUTES": floating(0),
                "KEEP_LAST": integer(3),
                "FINAL_CHECKPOINT": (["auto", "true", "false"],),
                "SAVE_FORMAT": (["comfy", "hf", "both", "none"],),
                "METRICS_EVERY": integer(5, 1),
                "METRICS_LEVEL": (["bench", "train", "debug"],),
                "LOG_EVERY": integer(10, 1),
                "RUN_NAME": string("my-lora"),
                "OUTPUT_DIR": string(
                    folder_paths.get_output_directory() + "/kohakumoss"
                ),
                "RESUME": string(
                    tooltip="Empty, latest, or a native checkpoint path; applies to the next queued run"
                ),
                "run_token": integer(0),
                "advanced_json": string(
                    '{"SAMPLE_EVERY":0}',
                    True,
                    "Native UPPER_CASE overrides; single-GPU host constraints always win",
                ),
            },
            "optional": {
                "DATASETS": string(
                    "[]",
                    True,
                    "Optional native dataset list; empty uses dataset_path/source",
                )
            },
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def train(self, unique_id, extra_pnginfo=None, **settings):
        result = execute(settings, unique_id, extra_pnginfo)
        return (result,)


class KohakuMossMonitor:
    CATEGORY = "KohakuMoss"
    FUNCTION = "monitor"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Live frontend monitor. The backend runs only after training finishes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_job": ("TRAINING_JOB",),
                "show_lr": ("BOOLEAN", {"default": True}),
                "show_vram": ("BOOLEAN", {"default": True}),
                "smoothing": integer(1, 1),
                "history_points": (
                    "INT",
                    {"default": 10_000, "min": 10, "max": 10_000},
                ),
            }
        }

    def monitor(self, training_job, **kwargs):
        return ()


NODE_CLASS_MAPPINGS = {
    "KohakuMossTraining": KohakuMossTraining,
    "KohakuMossMonitor": KohakuMossMonitor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KohakuMossTraining": "KohakuMoss Training",
    "KohakuMossMonitor": "KohakuMoss uPlot Monitor",
}
