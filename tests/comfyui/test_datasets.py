import tempfile
import unittest
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kohaku_comfy.datasets import DATASET_ROOT, create_dataset, upload_file


class DatasetUploadTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_preserves_captions_and_subfolders_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            app = web.Application()
            app[DATASET_ROOT] = Path(directory)
            app.router.add_post("/datasets", create_dataset)
            app.router.add_post("/datasets/{upload_id}/files", upload_file)
            async with TestClient(TestServer(app)) as client:
                created = await (await client.post("/datasets")).json()
                endpoint = f"/datasets/{created['upload_id']}/files"
                for name, content in (
                    ("2_style/a.png", b"image"),
                    ("2_style/a.txt", b"caption"),
                ):
                    reply = await client.post(
                        endpoint, params={"path": name}, data=content
                    )
                    self.assertEqual(reply.status, 200, await reply.text())
                    self.assertEqual(
                        (Path(created["path"]) / name).read_bytes(), content
                    )
                reply = await client.post(
                    endpoint, params={"path": "2_style/a.txt"}, data=b"replacement"
                )
                self.assertEqual(reply.status, 409)
                self.assertEqual(
                    (Path(created["path"]) / "2_style/a.txt").read_bytes(), b"caption"
                )
                for name in (
                    "../outside.txt",
                    "/absolute.txt",
                    "x\\outside.txt",
                    "C:/outside.txt",
                    "code.py",
                ):
                    reply = await client.post(
                        endpoint, params={"path": name}, data=b"bad"
                    )
                    self.assertEqual(reply.status, 400, name)
                reply = await client.post(
                    "/datasets/unknown/files", params={"path": "a.txt"}, data=b"bad"
                )
                self.assertEqual(reply.status, 400)
