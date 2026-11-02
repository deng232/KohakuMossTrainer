"""Source-only checks; run directly without Torch or pytest fixtures."""

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/install/patch_lycoris_cuda_graph.py"
)
spec = importlib.util.spec_from_file_location("graph_patch", SCRIPT)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class PatchTest(unittest.TestCase):
    def test_apply_repeat_and_reject_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "lycoris"
            folder = package / "kernels/triton/lora"
            folder.mkdir(parents=True)
            original = b"VALUE = 1\n"
            paths = [folder / name for name in ("stream.py", "bypass.py")]
            for path in paths:
                path.write_bytes(original)
            diff = root / "change.patch"
            diff.write_text(
                "".join(
                    f"--- a/lycoris/kernels/triton/lora/{p.name}\n"
                    f"+++ b/lycoris/kernels/triton/lora/{p.name}\n"
                    "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
                    for p in paths
                )
            )
            hashes = {p.name: hashlib.sha256(original).hexdigest() for p in paths}
            with (
                patch.object(installer, "PATCH", diff),
                patch.object(installer, "ORIGINAL", hashes),
            ):
                installer.apply(package)
                installer.apply(package)
                for path in paths:
                    self.assertEqual(path.read_bytes(), b"VALUE = 2\n")
                    self.assertEqual(
                        path.with_suffix(".py.before-cuda-graph").read_bytes(), original
                    )
                paths[0].write_bytes(b"VALUE = 3\n")
                before = {p: p.read_bytes() for p in paths}
                with self.assertRaises(RuntimeError):
                    installer.apply(package)
                self.assertEqual(before, {p: p.read_bytes() for p in paths})


if __name__ == "__main__":
    unittest.main()
