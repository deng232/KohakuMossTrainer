import json
import tempfile
import unittest
from pathlib import Path

from kohaku_comfy.session import SessionStore


class SessionTests(unittest.TestCase):
    def test_metrics_are_copied_finite_and_ordered(self):
        events = []
        store = SessionStore(lambda name, data: events.append((name, data)))
        with store.training("17", "workflow") as job:
            row = {"loss": 0.42, "lr": float("nan")}
            store.metrics(job, 5, row)
            row["loss"] = 99
            snap = store.snapshot(job)
            self.assertEqual(snap["last_metrics"], {"loss": 0.42, "lr": None})
            self.assertEqual(snap["status"], "TRAINING")
            self.assertEqual(snap["origin_node_id"], "17")
            self.assertEqual(snap["workflow_id"], "workflow")
            json.dumps(events, allow_nan=False)
        self.assertEqual(store.snapshot(job)["status"], "COMPLETED")
        seq = [data["event_seq"] for _, data in events]
        self.assertEqual(seq, list(range(1, len(seq) + 1)))

    def test_failure_releases_lease_and_never_keeps_exception(self):
        store = SessionStore()
        with self.assertRaisesRegex(ValueError, "bad config"):
            with store.training("17") as job:
                with self.assertRaisesRegex(RuntimeError, "already"):
                    with store.training("18"):
                        self.fail("concurrent training")
                raise ValueError("bad config")
        self.assertEqual(store.snapshot(job)["status"], "FAILED")
        self.assertIn("bad config", store.snapshot(job)["error"])
        with store.training("18") as other:
            self.assertNotEqual(other, job)

    def test_disconnected_browser_does_not_fail_training(self):
        def disconnected(*args):
            raise ConnectionError("closed")

        store = SessionStore(disconnected)
        with self.assertLogs("kohaku_comfy.session", level="WARNING"):
            with store.training("17") as job:
                store.metrics(job, 1, {"loss": 1})
        self.assertEqual(store.snapshot(job)["status"], "COMPLETED")

    def test_history_skips_partial_rows_and_cannot_mix_reused_directory(self):
        store = SessionStore()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            with store.training("17") as old:
                store.configure(old, directory, "run", 10)
                path.write_text('{"step":1,"time":1,"loss":0.9}\n{"step":')
                store.attach_logger(directory, path)
                store.metrics(old, 1, {"loss": 0.9})
                rows = store.history(old)["rows"]
                self.assertEqual(rows[0]["metrics"]["loss"], 0.9)
            with store.training("17") as new:
                store.configure(new, directory, "run", 10)
                path.write_text('{"step":2,"time":2,"loss":0.1}\n')
                store.attach_logger(directory, path)
                self.assertEqual(store.history(old)["rows"][0]["step"], 1)
                self.assertTrue(store.history(old)["truncated"])
                self.assertEqual(store.history(new)["rows"][0]["step"], 2)
            with self.assertRaises(KeyError):
                store.history("../../etc/passwd")

    def test_rotated_native_history_remains_owned_by_old_job(self):
        store = SessionStore()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            with store.training("17") as old:
                store.configure(old, directory, "run", 10)
                path.write_text('{"step":1,"time":1,"loss":0.9}\n')
                store.attach_logger(directory, path)
            with store.training("17") as new:
                store.configure(new, directory, "run", 10)
                path.rename(path.with_name("metrics.1.jsonl"))
                path.write_text('{"step":2,"time":2,"loss":0.1}\n')
                store.attach_logger(directory, path)
                self.assertEqual(store.history(old)["rows"][0]["step"], 1)

    def test_bounds_history_and_recent_jobs(self):
        store = SessionStore(points=3, jobs=2)
        with store.training("17") as first:
            for step in range(5):
                store.metrics(first, step, {"loss": step})
            self.assertEqual(
                [r["step"] for r in store.history(first)["rows"]], [2, 3, 4]
            )
            self.assertTrue(store.history(first)["truncated"])
        for _ in range(2):
            with store.training("18"):
                pass
        self.assertEqual(len(store.list_jobs()), 2)
        self.assertEqual(store.list_jobs("17"), [])


if __name__ == "__main__":
    unittest.main()
