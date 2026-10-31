"""ComfyUI-local monitoring/control routes; creation is workflow execution only."""

import asyncio
import functools
import os
import threading
from pathlib import Path
from urllib.parse import urlsplit

import folder_paths
from aiohttp import web
from server import PromptServer

from .kohaku_comfy.files import output_directories, save_dataset_upload
from .kohaku_comfy.jobs import JobManager
from .kohaku_comfy.runtime import Runtime

_manager = None
_manager_lock = threading.Lock()
_registered = False


def get_manager():
    global _manager
    with _manager_lock:
        if _manager is None:
            repo = (
                Path(
                    os.environ.get(
                        "KOHAKUMOSS_REPO", Path(__file__).resolve().parents[2]
                    )
                )
                .expanduser()
                .resolve()
            )
            runtime = Runtime(
                repo=repo,
                venv=Path(os.environ.get("KOHAKUMOSS_VENV", repo / ".venv"))
                .expanduser()
                .absolute(),
                workspace=Path(
                    os.environ.get(
                        "KOHAKUMOSS_WORKSPACE",
                        Path(folder_paths.get_output_directory()) / "kohakumoss",
                    )
                )
                .expanduser()
                .resolve(),
                dataset_root=Path(
                    os.environ.get(
                        "KOHAKUMOSS_DATA_ROOT", folder_paths.get_input_directory()
                    )
                )
                .expanduser()
                .resolve(),
                publish_dir=Path(folder_paths.get_folder_paths("loras")[0]).resolve(),
            )
            _manager = JobManager(runtime)
        return _manager


def guarded(handler):
    @functools.wraps(handler)
    async def respond(request):
        # Retain ComfyUI's access control and reject browser cross-origin control requests.
        origin = request.headers.get("Origin")
        if request.headers.get("Sec-Fetch-Site") == "cross-site" or (
            origin and urlsplit(origin).netloc != request.host
        ):
            return web.json_response(
                {"error": "Cross-origin job access is not allowed"}, status=403
            )
        try:
            return web.json_response(await handler(request))
        except KeyError:
            return web.json_response({"error": "Unknown job"}, status=404)
        except (ValueError, OSError) as error:
            return web.json_response({"error": str(error)}, status=400)

    return respond


def register_routes():
    global _registered
    if _registered or PromptServer.instance is None:
        return
    routes = PromptServer.instance.routes

    @routes.post("/kohakumoss/datasets/upload")
    @guarded
    async def upload_dataset(request):
        root = Path(
            os.environ.get("KOHAKUMOSS_DATA_ROOT", folder_paths.get_input_directory())
        )
        post = await request.post()
        upload = post.get("file")
        if not isinstance(upload, web.FileField):
            raise ValueError("Select a dataset folder to upload")
        return await asyncio.to_thread(
            save_dataset_upload,
            root,
            post.get("upload_id", ""),
            post.get("relative_path", ""),
            upload.file,
        )

    @routes.get("/kohakumoss/outputs/directories")
    @guarded
    async def outputs(request):
        return await asyncio.to_thread(
            output_directories,
            Path(folder_paths.get_folder_paths("loras")[0]),
            request.query.get("path", ""),
        )

    @routes.get("/kohakumoss/jobs")
    @guarded
    async def jobs(request):
        return {"jobs": await asyncio.to_thread(get_manager().list)}

    @routes.get("/kohakumoss/jobs/{job_id}")
    @guarded
    async def status(request):
        return await asyncio.to_thread(get_manager().get, request.match_info["job_id"])

    @routes.get("/kohakumoss/jobs/{job_id}/logs")
    @guarded
    async def logs(request):
        return await asyncio.to_thread(
            get_manager().logs,
            request.match_info["job_id"],
            int(request.query.get("offset", 0)),
            int(request.query.get("limit", 200)),
            request.query.get("generation"),
        )

    @routes.get("/kohakumoss/jobs/{job_id}/metrics")
    @guarded
    async def metrics(request):
        return {
            "rows": await asyncio.to_thread(
                get_manager().metrics,
                request.match_info["job_id"],
                int(request.query.get("limit", 1000)),
            )
        }

    @routes.post("/kohakumoss/jobs/{job_id}/stop")
    @guarded
    async def stop(request):
        return await asyncio.to_thread(get_manager().stop, request.match_info["job_id"])

    @routes.post("/kohakumoss/jobs/{job_id}/resume")
    @guarded
    async def resume(request):
        values = await request.json()
        if not isinstance(values, dict) or set(values) - {
            "checkpoint",
            "additional_steps",
            "output_path",
        }:
            raise ValueError("Invalid resume options")
        return await asyncio.to_thread(
            get_manager().resume,
            request.match_info["job_id"],
            values.get("checkpoint", "latest"),
            additional_steps=values.get("additional_steps"),
            output_path=values.get("output_path"),
        )

    _registered = True
