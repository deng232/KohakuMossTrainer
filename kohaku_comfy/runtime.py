"""Synchronous prompt-worker execution and host resource cleanup."""

import gc
import importlib
import logging
import os
from contextlib import contextmanager
from pathlib import Path

from .cache import prepare
from .session import STORE

log = logging.getLogger(__name__)


@contextmanager
def host_state(torch, gpu):
    """Restore the caller's grad mode, selected RNG/device and compiler settings."""
    names = ("diffusers", "transformers", "huggingface_hub", "urllib3", "httpx")
    levels = {name: logging.getLogger(name).level for name in names}
    roots = [
        logging.getLogger(name)
        for name in ("kohakumoss", "kohakumoss_kernels", "kohakumoss_data")
    ]
    old_handlers = {handler for root in roots for handler in root.handlers}
    env_key = "TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL"
    old_env = os.environ.get(env_key)
    linalg = torch.backends.cuda.preferred_linalg_library()
    old_tag = torch.compiler.config.cache_key_tag
    inductor = importlib.import_module("torch._inductor.config")
    old_pool = inductor.quiesce_async_compile_pool
    try:
        with (
            torch.inference_mode(False),
            torch.enable_grad(),
            torch.random.fork_rng(devices=[gpu]),
            torch.cuda.device(gpu),
        ):
            yield
    finally:
        torch.compiler.config.cache_key_tag = old_tag
        inductor.quiesce_async_compile_pool = old_pool
        torch.backends.cuda.preferred_linalg_library(linalg)
        if old_env is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old_env
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)
        added = {handler for root in roots for handler in root.handlers} - old_handlers
        for root in roots:
            for handler in added:
                if handler in root.handlers:
                    root.removeHandler(handler)
        for handler in added:
            handler.close()


def execute(settings, origin_node_id, extra_pnginfo=None):
    workflow = (extra_pnginfo or {}).get("workflow", {})
    workflow_id = workflow.get("extra", {}).get("kohakumoss_id", workflow.get("id", ""))
    with STORE.training(origin_node_id, workflow_id) as job_id:
        try:
            from comfy import model_management

            from . import native

            knobs = native.configure(settings)
            with host_state(native.torch, knobs["DEVICES"][0]):
                run_dir = Path(knobs["OUTPUT_DIR"]) / native.run_name(knobs)
                STORE.configure(job_id, run_dir, native.run_name(knobs), knobs["STEPS"])
                model_management.unload_all_models()
                model_management.soft_empty_cache()
                prepare(knobs, job_id)
                state = native.run(knobs)
                STORE.update(
                    job_id, final_step=int(state.step), files=output_files(run_dir)
                )
                del state
        except ModuleNotFoundError as error:
            raise RuntimeError(
                f"Missing dependency {error.name}; run this extension's Manager installer in ComfyUI's Python environment"
            ) from error
        except SystemExit as error:
            raise RuntimeError(
                f"KohakuMoss rejected or ended the run: {error}"
            ) from error
        finally:
            gc.collect()
            try:
                from comfy import model_management

                model_management.soft_empty_cache()
            except Exception:
                log.warning("Could not release ComfyUI's free cache", exc_info=True)
    snapshot = STORE.snapshot(job_id)
    return {
        name: snapshot[name]
        for name in (
            "job_id",
            "origin_node_id",
            "workflow_id",
            "run_dir",
            "run_name",
            "status",
            "files",
        )
    }


def output_files(run_dir):
    """Checkpoint and export paths that actually exist in this run directory."""
    return sorted(
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file() and path.suffix in (".safetensors", ".pt", ".ckpt")
    )
