"""ComfyUI entry point for KohakuMoss's two-node training integration."""

import logging

from server import PromptServer

from .kohaku_comfy.api import register as register_routes
from .kohaku_comfy.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

try:
    from .kohaku_comfy.live_logger import register

    register()
except ModuleNotFoundError as error:
    logging.getLogger(__name__).warning(
        "KohakuMoss dependency missing (%s); run the Manager installer before training",
        error.name,
    )

register_routes(PromptServer.instance)
