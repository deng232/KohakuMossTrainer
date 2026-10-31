"""Bounded monitoring reads, native checkpoint discovery and atomic publication."""

import json
import os
import re
import shutil
import tempfile
import uuid
from collections import deque
from pathlib import Path

from .config import inside


def save_dataset_upload(root, upload_id, relative_path, source):
    root = Path(root).expanduser().resolve()
    if (
        not isinstance(relative_path, str)
        or len(relative_path) > 4096
        or "\\" in relative_path
        or "\x00" in relative_path
    ):
        raise ValueError("Invalid dataset file path")
    parts = relative_path.split("/")
    if not 2 <= len(parts) <= 32 or any(
        p in ("", ".", "..") or ":" in p for p in parts
    ):
        raise ValueError("Upload paths must be relative to a selected folder")
    suffixes = {
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
    }
    if Path(parts[-1]).suffix.lower() not in suffixes:
        raise ValueError("Only dataset images and caption sidecars may be uploaded")
    base = inside(root / ".kohaku_uploads", root)
    base.mkdir(parents=True, exist_ok=True)
    if upload_id:
        if not isinstance(upload_id, str) or not re.fullmatch(
            r"[a-f0-9]{32}", upload_id
        ):
            raise ValueError("Invalid upload id")
        session = inside(base / upload_id, base)
        if not session.is_dir() or (base / upload_id).is_symlink():
            raise ValueError("Unknown upload id")
    else:
        upload_id = uuid.uuid4().hex
        session = base / upload_id
        session.mkdir()
    target = inside(session.joinpath(*parts), session)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".upload-", suffix=".partial", dir=target.parent
    )
    try:
        with os.fdopen(fd, "wb") as output:
            shutil.copyfileobj(source, output)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"upload_id": upload_id, "dataset_path": str(session / parts[0])}


def read_jsonl(path, limit=1000):
    rows = deque(maxlen=max(1, min(int(limit), 2000)))
    try:
        with Path(path).open("rb") as f:
            # UI history is bounded; the complete native history stays on disk.
            f.seek(max(0, f.seek(0, 2) - 2 * 1024 * 1024))
            if f.tell():
                f.readline()
            for line in f:
                if not line.endswith(b"\n"):
                    break
                try:
                    row = json.loads(line, parse_constant=lambda value: None)
                except (ValueError, UnicodeDecodeError):
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except FileNotFoundError:
        pass
    return list(rows)


def read_log(path, offset=0, limit=200):
    try:
        with Path(path).open("rb") as f:
            size = f.seek(0, 2)
            offset = max(0, int(offset))
            f.seek(offset if offset <= size else 0)
            lines, remaining = [], 128 * 1024
            for _ in range(max(1, min(int(limit), 500))):
                line = f.readline(remaining)
                if not line:
                    break
                lines.append(line)
                remaining -= len(line)
                if remaining <= 0:
                    break
            return {
                "text": b"".join(lines).decode("utf-8", errors="replace"),
                "offset": f.tell(),
            }
    except FileNotFoundError:
        return {"text": "", "offset": 0}


def checkpoints(run_dir):
    root = (Path(run_dir) / "checkpoints").resolve()
    return sorted(
        p
        for p in root.glob("step_*")
        if re.fullmatch(r"step_\d{8}", p.name)
        and not p.is_symlink()
        and p.resolve().is_relative_to(root)
        and (p / "state.json").is_file()
        and (p / "trainable.safetensors").is_file()
    )


def publish(source, directory):
    """Link a completed temporary copy into place, without replacing any existing file."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".kohaku-", suffix=".partial", dir=directory
    )
    try:
        with os.fdopen(fd, "wb") as dst, Path(source).open("rb") as src:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        candidate = directory / Path(source).name
        n = 1
        while True:
            try:
                os.link(temporary, candidate)
                return candidate
            except FileExistsError:
                candidate = directory / f"{Path(source).stem}-{n}.safetensors"
                n += 1
    finally:
        Path(temporary).unlink(missing_ok=True)


def output_directories(root, path=""):
    root = Path(root).resolve()
    directory = inside(root / path, root)
    if not directory.is_dir():
        raise ValueError("Select an existing LoRA directory")
    children = []
    for child in directory.iterdir():
        if child.is_dir() and child.resolve().is_relative_to(root):
            children.append(str(child))
    return {
        "path": str(directory),
        "parent": str(directory.parent) if directory != root else None,
        "directories": sorted(children),
    }
