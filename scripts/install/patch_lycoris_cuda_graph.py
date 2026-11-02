#!/usr/bin/env python3
"""Apply the GPU-tested LyCORIS 4.1.3 Graph patch in the current interpreter."""

import hashlib
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

PATCH = Path(__file__).with_name("lycoris-4.1.3-cuda-graph.patch")
ORIGINAL = {
    "stream.py": "74d8f8411ae4b57b218864b1cd939a5ee1631f8233ff718b2bc97408f140efb3",
    "bypass.py": "cad91cacc3af41b7f6186a8ccd2d5384d7131516d10e76b5e2b148cf1a3e31e4",
}


def replace_file(path, data):
    # Atomic replacement keeps an interrupted write from truncating Python source.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temp = Path(output.name)
        try:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
            temp.chmod(path.stat().st_mode & 0o777)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def apply(package):
    paths = [package / "kernels/triton/lora" / name for name in ORIGINAL]
    command = ["git", "apply", "--check", "--reverse", str(PATCH)]
    if subprocess.run(command, cwd=package.parent, capture_output=True).returncode == 0:
        print("Already patched:", package)
        return
    before = {path: path.read_bytes() for path in paths}
    for path, data in before.items():
        if hashlib.sha256(data).hexdigest() != ORIGINAL[path.name]:
            raise RuntimeError(
                f"Refusing unrecognized or partially patched source: {path}"
            )
    with tempfile.TemporaryDirectory(prefix="lycoris-graph-patch-") as directory:
        staging = Path(directory)
        for path in paths:
            target = staging / path.relative_to(package.parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        subprocess.run(["git", "apply", str(PATCH)], cwd=staging, check=True)
        after = {}
        for path in paths:
            data = (staging / path.relative_to(package.parent)).read_bytes()
            compile(data, str(path), "exec")
            after[path] = data
        # Validate both files and backups before changing either installed file.
        for path, data in before.items():
            backup = path.with_suffix(".py.before-cuda-graph")
            if backup.exists() and backup.read_bytes() != data:
                raise RuntimeError(f"Refusing to replace a different backup: {backup}")
        for path, data in before.items():
            backup = path.with_suffix(".py.before-cuda-graph")
            if not backup.exists():
                with backup.open("xb") as output:
                    output.write(data)
        changed = []
        try:
            for path, data in after.items():
                replace_file(path, data)
                changed.append(path)
        except BaseException:
            for path in reversed(changed):
                replace_file(path, before[path])
            raise
    for path in paths:
        print("Patched:", path)
    print(
        "Restart ComfyUI before testing; original .py.before-cuda-graph backups retained."
    )


def main():
    version = importlib.metadata.version("lycoris-lora")
    if version != "4.1.3":
        raise RuntimeError(
            f"Patch validated only for lycoris-lora 4.1.3, found {version}"
        )
    spec = importlib.util.find_spec("lycoris")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate LyCORIS in this interpreter")
    apply(Path(next(iter(spec.submodule_search_locations))).resolve())


if __name__ == "__main__":
    main()
