"""One asynchronous KohakuMoss training node; no training dependencies imported here."""

from comfy_api.latest import ComfyExtension

from .nodes import KohakuMossTrainer
from .server import register_routes

WEB_DIRECTORY = "./web"


class KohakuMossExtension(ComfyExtension):
    async def get_node_list(self):
        return [KohakuMossTrainer]


async def comfy_entrypoint():
    register_routes()
    return KohakuMossExtension()
