import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kohaku_comfy import api
from kohaku_comfy.session import SessionStore


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_known_jobs_only_and_strict_json(self):
        store = SessionStore()
        with store.training("17", "workflow") as job:
            store.metrics(job, 5, {"loss": float("inf")})
        app = web.Application()
        app.router.add_get("/kohakumoss/jobs", api.jobs)
        app.router.add_get("/kohakumoss/jobs/{job_id}", api.snapshot)
        app.router.add_get("/kohakumoss/jobs/{job_id}/history", api.history)
        with patch.object(api, "STORE", store):
            async with TestClient(TestServer(app)) as client:
                reply = await client.get(
                    "/kohakumoss/jobs?origin_node_id=17&workflow_id=workflow"
                )
                body = await reply.json()
                self.assertEqual(body["jobs"][0]["job_id"], job)
                reply = await client.get(
                    f"/kohakumoss/jobs/{job}/history?path=/etc/passwd"
                )
                history = await reply.json()
                self.assertIsNone(history["rows"][0]["metrics"]["loss"])
                reply = await client.get("/kohakumoss/jobs/unknown/history")
                self.assertEqual(reply.status, 404)
                reply = await client.post(f"/kohakumoss/jobs/{job}/stop")
                self.assertEqual(reply.status, 404)

    async def test_checkpoint_browser_stays_in_server_output_and_checks_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            checkpoint = root / "run" / "checkpoints" / "step_00000500"
            checkpoint.mkdir(parents=True)
            (checkpoint / "state.json").write_text("{}")
            (checkpoint / "trainable.safetensors").touch()
            (checkpoint.parent / "step_00001000.partial").mkdir()
            (root / "outside").symlink_to(Path(directory), target_is_directory=True)
            app = web.Application()
            app[api.OUTPUT_ROOT] = root
            app.router.add_get("/checkpoints", api.checkpoints)
            async with TestClient(TestServer(app)) as client:
                reply = await client.get("/checkpoints")
                data = await reply.json()
                self.assertEqual(data["directories"], ["run"])
                self.assertIsNone(data["parent"])
                self.assertFalse(data["checkpoint"])
                reply = await client.get(
                    "/checkpoints", params={"path": str(checkpoint)}
                )
                data = await reply.json()
                self.assertTrue(data["checkpoint"])
                self.assertEqual(data["path"], str(checkpoint))
                reply = await client.get(
                    "/checkpoints", params={"path": str(checkpoint.parent)}
                )
                self.assertEqual((await reply.json())["directories"], ["step_00000500"])
                for path in ("..", str(Path(directory)), "outside"):
                    reply = await client.get("/checkpoints", params={"path": path})
                    self.assertEqual(reply.status, 403)
                reply = await client.get("/checkpoints", params={"path": "missing"})
                self.assertEqual(reply.status, 404)
