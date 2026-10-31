"""ComfyUI extension entry point for installing the complete trainer checkout."""

from importlib import import_module

WEB_DIRECTORY = "./integrations/ComfyUI-KohakuMoss-Trainer/web"


async def comfy_entrypoint():
    integration = import_module(".integrations.ComfyUI-KohakuMoss-Trainer", __name__)
    return await integration.comfy_entrypoint()
