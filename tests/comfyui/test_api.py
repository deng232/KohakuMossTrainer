import unittest
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
