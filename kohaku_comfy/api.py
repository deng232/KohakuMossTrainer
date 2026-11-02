"""Job queries, dataset uploads and ComfyUI's existing WebSocket queue bridge."""

import asyncio
import json
from pathlib import Path

from aiohttp import web

from .datasets import DATASET_ROOT, create_dataset, upload_file
from .session import STORE

OUTPUT_ROOT = web.AppKey("kohakumoss.output_root", Path)

_registered = False


def response(value):
    return web.Response(
        text=json.dumps(value, allow_nan=False), content_type="application/json"
    )


async def jobs(request):
    return response(
        {
            "jobs": STORE.list_jobs(
                request.query.get("origin_node_id"), request.query.get("workflow_id")
            )
        }
    )


async def snapshot(request):
    try:
        return response(STORE.snapshot(request.match_info["job_id"]))
    except KeyError:
        raise web.HTTPNotFound(text="Unknown KohakuMoss job") from None


async def history(request):
    try:
        result = await asyncio.to_thread(STORE.history, request.match_info["job_id"])
        return response(result)
    except KeyError:
        raise web.HTTPNotFound(text="Unknown KohakuMoss job") from None


def checkpoint_listing(root, selected):
    root = root.resolve()
    path = Path(selected) if selected else root
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_relative_to(root):
        raise web.HTTPForbidden(text="Checkpoint path must stay inside ComfyUI output")
    if not path.is_dir():
        raise web.HTTPNotFound(text="Checkpoint directory does not exist")
    directories = sorted(
        child.name
        for child in path.iterdir()
        if child.is_dir()
        and not child.name.endswith(".partial")
        and child.resolve().is_relative_to(root)
    )
    checkpoint = not path.name.endswith(".partial") and all(
        (path / name).is_file() and (path / name).resolve().is_relative_to(root)
        for name in ("state.json", "trainable.safetensors")
    )
    return {
        "path": str(path),
        "parent": None if path == root else str(path.parent),
        "directories": directories,
        "checkpoint": checkpoint,
    }


async def checkpoints(request):
    try:
        result = await asyncio.to_thread(
            checkpoint_listing, request.app[OUTPUT_ROOT], request.query.get("path", "")
        )
        return response(result)
    except PermissionError:
        raise web.HTTPForbidden(text="Checkpoint directory is not readable") from None


def register(server):
    global _registered
    if _registered:
        return
    import folder_paths

    server.app[OUTPUT_ROOT] = Path(folder_paths.get_output_directory()).resolve()
    server.routes.get("/kohakumoss/checkpoints")(checkpoints)
    server.app[DATASET_ROOT] = (
        Path(folder_paths.get_input_directory()).resolve() / "kohakumoss_datasets"
    )
    server.routes.post("/kohakumoss/datasets")(create_dataset)
    server.routes.post("/kohakumoss/datasets/{upload_id}/files")(upload_file)
    server.routes.get("/kohakumoss/jobs")(jobs)
    server.routes.get("/kohakumoss/jobs/{job_id}")(snapshot)
    server.routes.get("/kohakumoss/jobs/{job_id}/history")(history)
    STORE.send = server.send_sync
    _registered = True
