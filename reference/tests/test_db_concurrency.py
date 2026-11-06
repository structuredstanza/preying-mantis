# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for knowledge-database behaviour under concurrent writers.

Field report: `--parallel 20` produced "PIPELINE CRITICAL ABORT IN TASK (...):
database is locked". SQLite serializes writers; the connection's busy
timeout is what separates a queued write from a dead campaign, and the
default (5s) was never raised for the module every campaign writes through.
These tests hold the write lock for real and check that writers wait for as
long as they are configured to, that the configuration reaches every
connection, and that the file is switched to write-ahead logging so readers
and the writer stop blocking each other.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

_REF_ROOT = str(Path(__file__).resolve().parent.parent)
if _REF_ROOT not in sys.path:
    sys.path.insert(0, _REF_ROOT)

from core import database  # noqa: E402
from core.database import DB_TIMEOUT_ENV, connect, init_db, read_findings, write_findings  # noqa: E402


def _finding(i: int) -> dict:
    return {"title": f"issue {i}", "description": "d", "severity": "LOW",
            "filepath": "app.py", "line_numbers": [i + 1]}


class _HoldWriteLock:
    """Holds SQLite's write lock on `db_path` for `seconds` from another
    thread, exactly the way a slow concurrent campaign would."""

    def __init__(self, db_path: str, seconds: float):
        self.db_path, self.seconds = db_path, seconds
        self._locked = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO findings (run_id, filepath, title, status) "
                         "VALUES ('holder', 'x.py', 'held', 'reported')")
            self._locked.set()
            time.sleep(self.seconds)
            conn.commit()
        finally:
            conn.close()

    def __enter__(self):
        self._thread.start()
        self._locked.wait(timeout=10)
        return self

    def __exit__(self, *exc):
        self._thread.join(timeout=30)
        return False


class BusyTimeoutTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mantis_dblock_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.db = os.path.join(self.tmp, "knowledge.db")
        init_db(self.db)
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)
        os.environ.pop(DB_TIMEOUT_ENV, None)

    def test_writer_waits_out_a_held_lock_instead_of_failing(self):
        # The lock is held for 1.5s; the configured timeout is 10s. Before
        # the fix this raised "database is locked" at the default 5s only if
        # the hold was longer; with the field's 20 writers it routinely was.
        os.environ[DB_TIMEOUT_ENV] = "10"
        with _HoldWriteLock(self.db, 1.5):
            t0 = time.time()
            write_findings(self.db, "app.py", [_finding(1)], run_id="r1")
            waited = time.time() - t0
        self.assertGreaterEqual(waited, 1.0, "the write did not actually wait on the lock")
        titles = {f["title"] for f in read_findings(self.db)}
        self.assertIn("issue 1", titles)
        self.assertIn("held", titles)

    def test_timeout_is_honoured_when_the_lock_outlasts_it(self):
        # Proves the knob is live: a 0.3s timeout against a 2s hold fails
        # the way the field saw it. Operators who want this can have it.
        os.environ[DB_TIMEOUT_ENV] = "0.3"
        with _HoldWriteLock(self.db, 2.0):
            with self.assertRaises(sqlite3.OperationalError) as ctx:
                write_findings(self.db, "app.py", [_finding(2)], run_id="r2")
        self.assertIn("locked", str(ctx.exception).lower())

    def test_default_and_invalid_values(self):
        self.assertEqual(database.db_timeout(), database.DEFAULT_DB_TIMEOUT)
        os.environ[DB_TIMEOUT_ENV] = "not-a-number"
        self.assertEqual(database.db_timeout(), database.DEFAULT_DB_TIMEOUT)
        os.environ[DB_TIMEOUT_ENV] = "-5"
        self.assertEqual(database.db_timeout(), database.DEFAULT_DB_TIMEOUT)
        os.environ[DB_TIMEOUT_ENV] = "120"
        self.assertEqual(database.db_timeout(), 120.0)

    def test_file_is_in_write_ahead_logging_mode(self):
        conn = sqlite3.connect(self.db)
        try:
            self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
        finally:
            conn.close()

    def test_readers_do_not_block_on_a_held_write_lock(self):
        # The point of WAL: a long write transaction elsewhere must not make
        # read_findings wait for it.
        write_findings(self.db, "app.py", [_finding(3)], run_id="r3")
        with _HoldWriteLock(self.db, 1.5):
            t0 = time.time()
            rows = read_findings(self.db)
            elapsed = time.time() - t0
        self.assertLess(elapsed, 1.0, f"a reader waited {elapsed:.2f}s on the writer")
        self.assertTrue(any(f["title"] == "issue 3" for f in rows))

    def test_many_concurrent_writers_all_land(self):
        # Twenty threads, each writing its own finding at once, the field's
        # shape in miniature. Every row must land; no "database is locked".
        errors: list = []

        def worker(i: int) -> None:
            try:
                write_findings(self.db, f"f{i}.py", [dict(_finding(i), filepath=f"f{i}.py")],
                               run_id=f"run-{i}")
            except Exception as exc:  # pragma: no cover - the assertion below reports it
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(errors, [])
        self.assertEqual(len(read_findings(self.db)), 20)

    def test_every_module_uses_the_shared_connection(self):
        # The fix is only as good as its coverage: no knowledge-database
        # connection may bypass connect() and fall back to the 5s default.
        import ast

        offenders = []
        for rel in ("core/database.py", "core/cost.py", "core/planner.py", "core/chains.py",
                    "core/custom_tools.py"):
            tree = ast.parse(Path(_REF_ROOT, rel).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "connect"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "sqlite3"):
                    if rel == "core/database.py" and any(k.arg == "timeout" for k in node.keywords):
                        continue  # connect() itself
                    offenders.append(f"{rel}:{node.lineno}")
        self.assertEqual(offenders, [])


class PipelineDefaultTest(unittest.TestCase):
    """main.pipeline scales the default with --parallel and never overrides
    an operator's explicit setting."""

    def _resolve(self, parallel: int, db_timeout=None, env_value=None) -> str:
        env = {k: v for k, v in os.environ.items() if k != DB_TIMEOUT_ENV}
        if env_value is not None:
            env[DB_TIMEOUT_ENV] = env_value
        # Only the prologue of pipeline() matters here; stop it right after.
        import main as mantis_main

        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(mantis_main, "find_workflow_json",
                                  side_effect=RuntimeError("stop"), create=True):
            import asyncio
            try:
                asyncio.run(mantis_main.pipeline("/nonexistent", workflow_path="/nonexistent/w.json",
                                                 parallel=parallel, db_timeout=db_timeout))
            except Exception:
                pass
            return os.environ.get(DB_TIMEOUT_ENV, "")

    def test_scales_with_parallelism(self):
        self.assertEqual(float(self._resolve(1)), database.DEFAULT_DB_TIMEOUT)
        self.assertEqual(float(self._resolve(20)), 300.0)

    def test_explicit_flag_and_env_win(self):
        self.assertEqual(float(self._resolve(20, db_timeout=7)), 7.0)
        self.assertEqual(self._resolve(20, env_value="42"), "42")


if __name__ == "__main__":
    unittest.main()
