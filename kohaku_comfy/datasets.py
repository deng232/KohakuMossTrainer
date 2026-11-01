"""Stream browser-selected dataset files into isolated ComfyUI input folders."""

import asyncio
import re
import uuid
from pathlib import Path

from aiohttp import web

DATASET_ROOT = web.AppKey("kohakumoss.dataset_root", Path)
EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".jxl",
    ".avif",
    ".txt",
    ".caption",
    ".json",
    ".jsonl",
)
MAX_FILE_BYTES = 100 * 1024 * 1024


def new_directory(root):
    upload_id = uuid.uuid4().hex
    directory = root / upload_id
    directory.mkdir(parents=True)
    return {
        "upload_id": upload_id,
        "path": str(directory.resolve()),
        "extensions": EXTENSIONS,
    }


def destination(root, upload_id, relative):
    if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
        raise ValueError("Invalid dataset upload ID")
    if (
        not relative
        or any(c in relative for c in ("\\", ":", "\x00"))
        or any(p in ("", ".", "..") for p in relative.split("/"))
    ):
        raise ValueError("Invalid relative dataset path")
    if Path(relative).suffix.lower() not in EXTENSIONS:
        raise ValueError("Only dataset images and captions can be uploaded")
    directory = root / upload_id
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Unknown dataset upload")
    target = directory / relative
    if not target.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Dataset path escapes its upload directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


async def create_dataset(request):
    try:
        result = await asyncio.to_thread(new_directory, request.app[DATASET_ROOT])
        return web.json_response(result)
    except OSError as error:
        raise web.HTTPInternalServerError(
            text=f"Cannot create dataset folder: {error}"
        ) from None


async def upload_file(request):
    try:
        target = await asyncio.to_thread(
            destination,
            request.app[DATASET_ROOT],
            request.match_info["upload_id"],
            request.query.get("path", ""),
        )
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error)) from None
    except OSError as error:
        raise web.HTTPInternalServerError(
            text=f"Cannot prepare dataset file: {error}"
        ) from None
    try:
        stream = await asyncio.to_thread(target.open, "xb")
    except FileExistsError:
        raise web.HTTPConflict(
            text="Dataset file already exists; it was not overwritten"
        ) from None
    except OSError as error:
        raise web.HTTPInternalServerError(
            text=f"Cannot open dataset file: {error}"
        ) from None
    complete = False
    try:
        size = 0
        async for chunk in request.content.iter_chunked(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_BYTES:
                raise web.HTTPRequestEntityTooLarge(
                    max_size=MAX_FILE_BYTES, actual_size=size
                )
            await asyncio.to_thread(stream.write, chunk)
        complete = True
    except OSError as error:
        raise web.HTTPInsufficientStorage(
            text=f"Cannot write dataset file: {error}"
        ) from None
    finally:
        try:
            await asyncio.to_thread(stream.close)
        finally:
            if not complete:
                await asyncio.to_thread(target.unlink, missing_ok=True)
    return web.json_response({"ok": True})
