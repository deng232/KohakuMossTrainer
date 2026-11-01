"""Read-only routes and ComfyUI's existing WebSocket queue bridge."""

import asyncio
import json

from aiohttp import web

from .session import STORE

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


def register(server):
    global _registered
    if _registered:
        return
    server.routes.get("/kohakumoss/jobs")(jobs)
    server.routes.get("/kohakumoss/jobs/{job_id}")(snapshot)
    server.routes.get("/kohakumoss/jobs/{job_id}/history")(history)
    STORE.send = server.send_sync
    _registered = True
