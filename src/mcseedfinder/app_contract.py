"""
Local app command contract and SQLite persistence.

This mirrors the Tauri command surface without requiring a desktop shell yet.
Commands are synchronous and local-first: they persist job metadata, events,
and verified results so the eventual Tauri layer can delegate to the same
contract or replace it with equivalent Rust commands.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .engine import SearchEvent, SearchSpec, run_staged_search


class JobStore:
    """SQLite store for seed-search jobs, events, results, and settings."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    timestamp_seconds REAL NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    def create_job(self, spec: SearchSpec, job_id: str | None = None) -> str:
        jid = job_id or str(uuid.uuid4())
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs(job_id, status, spec_json) VALUES (?, ?, ?)",
                (jid, "created", json.dumps(asdict(spec))),
            )
            self._conn.commit()
        return jid

    def set_status(self, job_id: str, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE job_id = ?",
                (status, job_id),
            )
            self._conn.commit()

    def record_event(self, event: SearchEvent) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO events(job_id, type, payload_json, timestamp_seconds)
                VALUES (?, ?, ?, ?)
                """,
                (
                    event.job_id,
                    event.type,
                    json.dumps(dict(event.payload)),
                    event.timestamp_seconds,
                ),
            )
            if event.type == "verified_match":
                self._conn.execute(
                    "INSERT INTO results(job_id, seed, report_json) VALUES (?, ?, ?)",
                    (
                        event.job_id,
                        int(event.payload["seed"]),
                        json.dumps(dict(event.payload)),
                    ),
                )
            self._conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT job_id, status, spec_json, created_at, updated_at FROM jobs "
                "WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "spec": json.loads(row["spec_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_events(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT type, payload_json, timestamp_seconds
                FROM events
                WHERE job_id = ?
                ORDER BY id
                """,
                (job_id,),
            ).fetchall()
        return [
            {
                "type": row["type"],
                "payload": json.loads(row["payload_json"]),
                "timestamp_seconds": row["timestamp_seconds"],
            }
            for row in rows
        ]

    def list_results(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT report_json FROM results WHERE job_id = ? ORDER BY id",
                (job_id,),
            ).fetchall()
        return [json.loads(row["report_json"]) for row in rows]


class LocalCommandHost:
    """Synchronous stand-in for the planned Tauri command surface."""

    def __init__(self, store: JobStore | None = None) -> None:
        self.store = store or JobStore()

    def start_search(self, spec: SearchSpec | Mapping[str, Any]) -> str:
        search_spec = spec if isinstance(spec, SearchSpec) else SearchSpec(**spec)
        job_id = self.store.create_job(search_spec)
        self.store.set_status(job_id, "running")
        try:
            for event in run_staged_search(search_spec, job_id=job_id):
                self.store.record_event(event)
                if event.type == "completed":
                    self.store.set_status(job_id, "completed")
            return job_id
        except Exception:
            self.store.set_status(job_id, "error")
            raise

    def pause_search(self, job_id: str) -> None:
        self.store.set_status(job_id, "paused")

    def resume_search(self, job_id: str) -> None:
        self.store.set_status(job_id, "queued")

    def cancel_search(self, job_id: str) -> None:
        self.store.set_status(job_id, "cancelled")

    def export_results(self, job_id: str, fmt: str = "json") -> str:
        results = self.store.list_results(job_id)
        if fmt == "json":
            return json.dumps(results, indent=2)
        if fmt == "plain":
            return "\n".join(str(row["seed"]) for row in results)
        if fmt == "csv":
            lines = ["seed,edition,version,dimension,score"]
            for row in results:
                lines.append(
                    f"{row['seed']},{row['edition']},{row['version']},"
                    f"{row['dimension']},{row['score']}"
                )
            return "\n".join(lines)
        raise ValueError(f"unknown export format {fmt!r}")

    def analyze_seed(self, request: Mapping[str, Any]) -> dict[str, Any]:
        seed = int(request["seed"])
        return {
            "seed": seed,
            "status": "candidate",
            "notes": ["full analyzer is reserved for the desktop/Rust phase"],
        }

    def render_tile(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "tile": dict(request),
            "status": "unsupported",
            "reason": "map tile rendering is not implemented in this backend yet",
        }

    def import_level_dat(self, path: str | Path) -> dict[str, Any]:
        return {
            "path": str(path),
            "status": "unsupported",
            "reason": "level.dat import requires the planned NBT backend",
        }


class AsyncCommandHost(LocalCommandHost):
    """Background job runner with cancellation and pause/resume controls."""

    def __init__(
        self,
        store: JobStore | None = None,
        max_workers: int = 2,
    ) -> None:
        super().__init__(store)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: dict[str, Future[None]] = {}
        self._cancel_flags: dict[str, threading.Event] = {}
        self._pause_flags: dict[str, threading.Event] = {}
        self._jobs_lock = threading.RLock()

    def start_search(self, spec: SearchSpec | Mapping[str, Any]) -> str:
        search_spec = spec if isinstance(spec, SearchSpec) else SearchSpec(**spec)
        job_id = self.store.create_job(search_spec)
        cancel_flag = threading.Event()
        pause_flag = threading.Event()
        with self._jobs_lock:
            self._cancel_flags[job_id] = cancel_flag
            self._pause_flags[job_id] = pause_flag
            self._futures[job_id] = self._executor.submit(
                self._run_job,
                job_id,
                search_spec,
                cancel_flag,
                pause_flag,
            )
        return job_id

    def _run_job(
        self,
        job_id: str,
        spec: SearchSpec,
        cancel_flag: threading.Event,
        pause_flag: threading.Event,
    ) -> None:
        self.store.set_status(job_id, "running")
        try:
            for event in run_staged_search(
                spec,
                job_id=job_id,
                cancel_callback=cancel_flag.is_set,
                pause_callback=pause_flag.is_set,
            ):
                self.store.record_event(event)
                if event.type == "cancelled":
                    self.store.set_status(job_id, "cancelled")
                    return
                if event.type == "completed":
                    self.store.set_status(job_id, "completed")
                    return
        except Exception:
            self.store.set_status(job_id, "error")
            raise

    def pause_search(self, job_id: str) -> None:
        with self._jobs_lock:
            flag = self._pause_flags.get(job_id)
            if flag:
                flag.set()
        self.store.set_status(job_id, "paused")

    def resume_search(self, job_id: str) -> None:
        with self._jobs_lock:
            flag = self._pause_flags.get(job_id)
            if flag:
                flag.clear()
        self.store.set_status(job_id, "running")

    def cancel_search(self, job_id: str) -> None:
        should_mark = True
        with self._jobs_lock:
            flag = self._cancel_flags.get(job_id)
            if flag:
                flag.set()
            future = self._futures.get(job_id)
            if future and future.done():
                should_mark = False
        if should_mark:
            self.store.set_status(job_id, "cancelling")

    def wait(self, job_id: str, timeout: float | None = None) -> None:
        with self._jobs_lock:
            future = self._futures[job_id]
        future.result(timeout=timeout)
        job = self.store.get_job(job_id)
        if job and job["status"] == "cancelling":
            self.store.set_status(job_id, "cancelled")

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
