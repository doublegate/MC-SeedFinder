"""
Tests for the local app command contract and persistence.
"""

from __future__ import annotations

import unittest

import time

from mcseedfinder.app_contract import AsyncCommandHost, JobStore, LocalCommandHost
from mcseedfinder.engine import SearchSpec


class TestAppContract(unittest.TestCase):
    def test_start_search_persists_events_and_results(self) -> None:
        store = JobStore()
        host = LocalCommandHost(store)
        spec = SearchSpec(
            criteria={
                "nearby_structures": [
                    {"structure": "village", "max_distance": 1000},
                ],
            },
            start_seed=1,
            count=1,
            max_matches=1,
        )

        job_id = host.start_search(spec)

        job = store.get_job(job_id)
        self.assertIsNotNone(job)
        assert job is not None
        self.assertEqual(job["status"], "completed")
        self.assertEqual(store.list_results(job_id)[0]["seed"], 1)
        self.assertIn("completed", [event["type"] for event in store.list_events(job_id)])

    def test_command_status_transitions(self) -> None:
        store = JobStore()
        host = LocalCommandHost(store)
        spec = SearchSpec(
            criteria={
                "nearby_structures": [
                    {"structure": "village", "max_distance": 1000},
                ],
            },
            start_seed=1,
            count=1,
        )
        job_id = store.create_job(spec)

        host.pause_search(job_id)
        self.assertEqual(store.get_job(job_id)["status"], "paused")  # type: ignore[index]
        host.resume_search(job_id)
        self.assertEqual(store.get_job(job_id)["status"], "queued")  # type: ignore[index]
        host.cancel_search(job_id)
        self.assertEqual(store.get_job(job_id)["status"], "cancelled")  # type: ignore[index]

    def test_async_start_search_completes_in_background(self) -> None:
        store = JobStore()
        host = AsyncCommandHost(store)
        spec = SearchSpec(
            criteria={
                "nearby_structures": [
                    {"structure": "village", "max_distance": 1000},
                ],
            },
            start_seed=1,
            count=1,
            max_matches=1,
        )

        job_id = host.start_search(spec)
        host.wait(job_id, timeout=2)

        self.assertEqual(store.get_job(job_id)["status"], "completed")  # type: ignore[index]
        self.assertEqual(store.list_results(job_id)[0]["seed"], 1)
        host.shutdown()

    def test_async_cancel_search_records_cancelled_status(self) -> None:
        store = JobStore()
        host = AsyncCommandHost(store)
        spec = SearchSpec(
            criteria={
                "nearby_structures": [
                    {"structure": "village", "max_distance": 1},
                ],
            },
            start_seed=0,
            count=1_000_000,
            max_matches=1,
        )

        job_id = host.start_search(spec)
        host.cancel_search(job_id)
        host.wait(job_id, timeout=2)

        self.assertEqual(store.get_job(job_id)["status"], "cancelled")  # type: ignore[index]
        self.assertIn("cancelled", [event["type"] for event in store.list_events(job_id)])
        host.shutdown()

    def test_async_pause_and_resume(self) -> None:
        store = JobStore()
        host = AsyncCommandHost(store)
        spec = SearchSpec(
            criteria={
                "nearby_structures": [
                    {"structure": "village", "max_distance": 1},
                ],
            },
            start_seed=0,
            count=1000,
            max_matches=1,
        )

        job_id = host.start_search(spec)
        host.pause_search(job_id)
        time.sleep(0.01)
        self.assertEqual(store.get_job(job_id)["status"], "paused")  # type: ignore[index]
        host.resume_search(job_id)
        host.wait(job_id, timeout=2)
        self.assertIn(store.get_job(job_id)["status"], {"completed", "cancelled"})  # type: ignore[index]
        host.shutdown()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
