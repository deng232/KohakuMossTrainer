from pathlib import Path

import folder_paths
from comfy_api.latest import io
from comfy_execution.utils import get_executing_context

from .kohaku_comfy.config import CHOICES, DEFAULTS, FLOAT_LIMITS, INT_LIMITS
from .server import get_manager

MODELS = {
    "transformer": "diffusion_models",
    "text_encoder": "text_encoders",
    "vae": "vae",
}
BASIC = {
    "run_name",
    "dataset_path",
    "dataset_source",
    "transformer",
    "text_encoder",
    "vae",
    "algorithm",
    "rank",
    "alpha",
    "steps",
    "learning_rate",
    "memory_profile",
    "save_every",
    "cache_policy",
    "start_mode",
    "resolution_mode",
    "target_mp",
    "max_tokens",
    "batch_size",
    "batching",
    "grad_accum",
    "resume_checkpoint",
    "output_path",
}
TIPS = {
    "resume_job_id": "ID of the original training job. The checkpoint picker fills this automatically. Used only in resume_latest mode; leave empty when selecting a complete checkpoint by its full path.",
    "precision": "Training compute precision: bf16 or fp16. Weight storage and quantization come from the selected models and native memory profile. Applies to new jobs; resuming preserves the original precision.",
    "microbatches": "Micro-batches per optimizer step for native pipeline-parallel training only. Ordinary single-GPU training uses grad_accum instead; changing this does not increase single-GPU batch size.",
    "seed": "Random seed for training and dataset planning. The same seed helps reproduce a run; full checkpoint resume restores saved RNG state. This value does not automatically increment after queueing.",
    "caption_variants": "Number of cached caption renderings per image (default 4); training chooses a variant randomly. Useful with randomized caption templates. Does not generate new descriptions; 1 is sufficient for fixed captions. More variants can increase text cache work and size.",
    "text_dropout": "Probability of replacing an entire sample caption with the empty-prompt embedding during training. 0.1 means 10% of samples; 0 disables it and 1 drops all captions. Images still participate in training; this is not word dropout.",
    "save_format": "Native weight export format: comfy = ComfyUI safetensors; hf = Hugging Face format; both = both formats; none = no weight export. Publishing to the LoRA directory requires comfy or both. Full resume checkpoints are saved separately.",
    "checkpoint_export": "Also include inference-ready weight exports inside each full checkpoint's export directory. Increases checkpoint disk usage. Turning this off does not disable complete resume checkpoints or the final run export.",
    "keep_last": "Number of newest complete checkpoints to retain. Older complete checkpoints in this run are deleted as new ones are saved. 0 keeps all. Does not delete previously published LoRA files.",
    "publish_to_loras": "Copy the final ComfyUI LoRA export to output_path after successful training. Existing files are preserved using numbered names. When off, exports stay in the native run directory.",
    "dataset_path": "Existing dataset under KOHAKUMOSS_DATA_ROOT (default: ComfyUI input directory). Images and captions remain read-only.",
    "max_tokens": "0 keeps the native memory profile's token limit (6144 for vram_12g).",
    "target_mp": "Maximum image area in resize mode (small images are not upscaled), in megapixels (1 MP = 1000000 pixels). 1.048576 MP = 1024 squared; 0.589824 MP = 768 squared. Native resizing aligns dimensions to multiples of 16.",
    "resolution_mode": "resize preserves aspect ratio; square_crop crops to a square; bucket groups similar aspect ratios.",
    "batching": "tokens packs image and text tokens up to max_tokens; count uses batch_size images.",
    "grad_accum": "Number of micro-batches accumulated before one optimizer update (GRAD_ACC).",
    "memory_profile": "vram_12g uses the native 12 GiB preset. ComfyUI inference still shares GPU memory.",
    "batch_size": "Used only when batching=count; token batching uses max_tokens instead.",
    "caption_template": "native preserves the selected data preset's caption settings; tags selects the native tag template and nl selects the natural-language template. Controls caption rendering, not automatic caption generation.",
    "output_root": "Empty uses the workspace runs/ directory. Custom paths must remain inside the configured workspace.",
    "start_mode": "resume_latest loads a selected full checkpoint with its original settings. steps is the number of additional steps; output_path selects the exported LoRA directory.",
    "resume_checkpoint": "Choose a complete checkpoint from existing KohakuMoss jobs. Optimizer, step and RNG state are restored.",
    "steps": "New training: number of steps. Resuming: additional steps after the selected checkpoint (2000 + 1000 continues to 3000).",
    "output_path": "Directory for loadable exported LoRAs, inside the registered LoRA directory. Checkpoints, optimizer state, logs and configs stay in the default workspace.",
    "save_every": "0 disables periodic checkpoints. Stop does not guarantee a new checkpoint.",
}


def model_path(category, name):
    if (
        not isinstance(name, str)
        or Path(name).is_absolute()
        or ".." in Path(name).parts
    ):
        raise ValueError(f"Invalid {category} model selection")
    if name not in folder_paths.get_filename_list(category):
        raise ValueError(f"Unknown {category} model: {name}")
    found = folder_paths.get_full_path(category, name)
    if not found:
        raise ValueError(f"Missing {category} model: {name}")
    path = Path(found).resolve()
    roots = [Path(p).resolve() for p in folder_paths.get_folder_paths(category)]
    if not path.is_file() or not any(path.is_relative_to(root) for root in roots):
        raise ValueError(f"Model is outside registered {category} directories: {name}")
    return str(path)


class KohakuMossTrainer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = []
        for name, default in DEFAULTS.items():
            common = dict(default=default, advanced=name not in BASIC)
            if name in TIPS:
                common["tooltip"] = TIPS[name]
            if name == "output_path":
                common["default"] = str(
                    Path(folder_paths.get_folder_paths("loras")[0]).resolve()
                )
            if name in MODELS:
                options = folder_paths.get_filename_list(MODELS[name])
                inputs.append(
                    io.Combo.Input(
                        name,
                        options=options,
                        tooltip="Select existing local weights; no automatic downloads.",
                    )
                )
            elif name == "dataset_path":
                inputs.append(
                    io.String.Input(
                        name,
                        extra_dict={"widgetType": "KOHAKU_DATASET_FOLDER"},
                        **common,
                    )
                )
            elif name == "output_path":
                inputs.append(
                    io.String.Input(
                        name,
                        extra_dict={"widgetType": "KOHAKU_OUTPUT_FOLDER"},
                        **common,
                    )
                )
            elif name == "resume_checkpoint":
                inputs.append(
                    io.String.Input(
                        name,
                        extra_dict={"widgetType": "KOHAKU_CHECKPOINT_FOLDER"},
                        **common,
                    )
                )
            elif name in CHOICES:
                inputs.append(
                    io.Combo.Input(name, options=list(CHOICES[name]), **common)
                )
            elif type(default) is bool:
                inputs.append(io.Boolean.Input(name, **common))
            elif type(default) is int:
                low, high = INT_LIMITS[name]
                if name == "seed":
                    common["control_after_generate"] = False
                inputs.append(io.Int.Input(name, min=low, max=high, **common))
            elif type(default) is float:
                low, high = FLOAT_LIMITS[name]
                if name == "target_mp":
                    common.update(
                        display_name="Max target area (MP)", step=0.05, round=0.000001
                    )
                elif name == "learning_rate":
                    common.update(step=0.000001, round=False)
                inputs.append(io.Float.Input(name, min=low, max=high, **common))
            else:
                inputs.append(io.String.Input(name, **common))
        return io.Schema(
            node_id="KohakuMossTrainer",
            display_name="KohakuMoss Trainer",
            category="Kohaku/Training",
            is_output_node=True,
            description="Train in the ComfyUI queue until exports and cleanup finish. Open the KohakuMoss sidebar for live progress, Stop and Resume.",
            inputs=inputs,
            outputs=[io.String.Output("job_id")],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def execute(cls, **kwargs):
        context = get_executing_context()
        if context is None:
            raise ValueError(
                "KohakuMoss training must be submitted by workflow execution"
            )
        settings = dict(kwargs)
        if settings.get("start_mode") != "resume_latest":
            for name, category in MODELS.items():
                settings[name] = model_path(category, settings.get(name, ""))
        from comfy import model_management

        model_management.unload_all_models()
        model_management.soft_empty_cache()
        try:
            job = get_manager().submit(
                settings, f"{context.prompt_id}:{context.node_id}"
            )
        finally:
            model_management.soft_empty_cache()
        model_management.throw_exception_if_processing_interrupted()
        return io.NodeOutput(
            job["job_id"],
            ui={"kohakumoss_job": [job["job_id"]], "status": [job["status"]]},
        )
