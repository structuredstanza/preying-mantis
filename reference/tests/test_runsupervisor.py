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

"""The UI's only dangerous seam: turning request fields into a spawned process.

A browser is untrusted input, so the properties pinned here are the ones that
decide whether "start a scan" is a feature or a remote shell: argv is always a
list, only whitelisted fields survive, the target stays inside one root, and a
second run cannot fight the first for the SQLite write lock.
"""

import os
import shutil
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core import runsupervisor
from core.runsupervisor import Supervisor


class _FakeProc:
    def __init__(self, pid=4242, running=True):
        self.pid = pid
        self._running = running

    def poll(self):
        return None if self._running else 0


class SupervisorValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mantis_ui_sup_")).resolve()
        self.root = self.tmp / "root"
        self.root.mkdir()
        (self.root / "code").mkdir()
        (self.root / "code" / "app.py").write_text("x = 1\n")
        self.outside = self.tmp / "outside"
        self.outside.mkdir()
        (self.outside / "secret.py").write_text("y = 2\n")
        self.sup = Supervisor(
            allow_root=self.root,
            launch_py=self.tmp / "launch.py",
            state_dir=self.tmp / "state",
            db_path=str(self.tmp / "k.db"),
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ok(self, **fields):
        validated, err = self.sup.validate_fields(fields)
        self.assertEqual(err, "", err)
        self.assertIsNotNone(validated)
        return validated

    def test_accepts_a_target_inside_the_root(self):
        validated = self._ok(target=str(self.root / "code"), scan_mode="whole")
        self.assertEqual(validated["scan_mode"], "whole")
        self.assertTrue(validated["target"].startswith(str(self.root)))

    def test_refuses_a_target_outside_the_root(self):
        validated, err = self.sup.validate_fields({"target": str(self.outside)})
        self.assertIsNone(validated)
        self.assertIn("outside the allowed root", err)

    def test_refuses_parent_traversal(self):
        validated, err = self.sup.validate_fields(
            {"target": str(self.root / ".." / "outside")})
        self.assertIsNone(validated)
        self.assertIn("component", err)

    def test_refuses_a_symlinked_target(self):
        link = self.root / "link"
        link.symlink_to(self.outside)
        validated, err = self.sup.validate_fields({"target": str(link)})
        self.assertIsNone(validated)
        self.assertIn("symlink", err)

    def test_refuses_scan_mode_changed(self):
        validated, err = self.sup.validate_fields(
            {"target": str(self.root / "code"), "scan_mode": "changed"})
        self.assertIsNone(validated)
        self.assertIn("scan_mode", err)

    def test_refuses_an_unknown_sandbox(self):
        validated, err = self.sup.validate_fields(
            {"target": str(self.root / "code"), "sandbox": "firecracker"})
        self.assertIsNone(validated)
        self.assertIn("sandbox", err)

    def test_refuses_a_model_id_that_is_not_one(self):
        for bad in ("openai/gpt; rm -rf /", "a b", "$(id)", "x" * 200):
            with self.subTest(bad=bad[:20]):
                validated, err = self.sup.validate_fields(
                    {"target": str(self.root / "code"), "model": bad})
                self.assertIsNone(validated)
                self.assertIn("model", err)

    def test_refuses_a_non_http_api_base(self):
        validated, err = self.sup.validate_fields(
            {"target": str(self.root / "code"), "api_base": "file:///etc/passwd"})
        self.assertIsNone(validated)
        self.assertIn("api_base", err)

    def test_refuses_a_non_integer_token_budget(self):
        validated, err = self.sup.validate_fields(
            {"target": str(self.root / "code"), "max_tokens": "10M; id"})
        self.assertIsNone(validated)
        self.assertIn("max_tokens", err)

    def test_refuses_an_exclude_pattern_with_shell_metacharacters(self):
        validated, err = self.sup.validate_fields(
            {"target": str(self.root / "code"), "exclude": ["vendor", "x; rm -rf /"]})
        self.assertIsNone(validated)
        self.assertIn("exclude", err)

    def test_drops_unknown_fields(self):
        validated = self._ok(target=str(self.root / "code"),
                             command="rm -rf /", shell=True, argv=["-c", "id"])
        self.assertNotIn("command", validated)
        self.assertNotIn("shell", validated)
        self.assertNotIn("argv", validated)

    def test_strips_terminal_control_from_focus(self):
        validated = self._ok(target=str(self.root / "code"),
                             focus="look\x1b[31m here\nand here")
        self.assertNotIn("\x1b", validated["focus"])
        self.assertNotIn("\n", validated["focus"])


class SupervisorArgvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mantis_ui_argv_")).resolve()
        (self.tmp / "code").mkdir()
        self.sup = Supervisor(allow_root=self.tmp, launch_py=self.tmp / "launch.py",
                              state_dir=self.tmp / "state", db_path=str(self.tmp / "k.db"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_argv_is_a_list_and_never_a_shell_string(self):
        validated, _ = self.sup.validate_fields(
            {"target": str(self.tmp / "code"), "scan_mode": "auto",
             "focus": "look for IDOR"})
        argv = self.sup.build_argv(validated)
        self.assertIsInstance(argv, list)
        self.assertTrue(all(isinstance(a, str) for a in argv))
        self.assertEqual(argv[0], validated["target"])
        self.assertIn("--focus", argv)
        self.assertEqual(argv[argv.index("--focus") + 1], "look for IDOR")

    def test_resume_is_opt_in(self):
        validated, _ = self.sup.validate_fields({"target": str(self.tmp / "code")})
        self.assertNotIn("--resume", self.sup.build_argv(validated))
        validated, _ = self.sup.validate_fields(
            {"target": str(self.tmp / "code"), "resume": True})
        self.assertIn("--resume", self.sup.build_argv(validated))


class SupervisorProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mantis_ui_proc_")).resolve()
        (self.tmp / "code").mkdir()
        self.sup = Supervisor(allow_root=self.tmp, launch_py=self.tmp / "launch.py",
                              state_dir=self.tmp / "state", db_path=str(self.tmp / "k.db"))

    def tearDown(self):
        self.sup.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_second_start_is_refused_while_the_first_runs(self):
        fake = _FakeProc()
        with mock.patch.object(runsupervisor.subprocess, "Popen", return_value=fake):
            first = self.sup.start({"target": str(self.tmp / "code")})
            self.assertTrue(first["ok"], first)
            second = self.sup.start({"target": str(self.tmp / "code")})
        self.assertFalse(second["ok"])
        self.assertIn("still running", second["error"])

    def test_cancel_interrupts_the_process_group(self):
        fake = _FakeProc()
        with mock.patch.object(runsupervisor.subprocess, "Popen", return_value=fake):
            self.sup.start({"target": str(self.tmp / "code")})
        killed = []

        def fake_killpg(pgid, sig):
            killed.append(sig)
            fake._running = False

        with mock.patch.object(runsupervisor.os, "getpgid", return_value=999), \
                mock.patch.object(runsupervisor.os, "killpg", side_effect=fake_killpg):
            out = self.sup.cancel()
        self.assertTrue(out["ok"], out)
        # SIGINT first: it lets BudgetController checkpoint so --resume works.
        self.assertEqual(killed[0], signal.SIGINT)
        self.assertIn("--resume", out["resume_command"])

    def test_status_reports_a_finished_run_with_its_exit_code(self):
        fake = _FakeProc()
        with mock.patch.object(runsupervisor.subprocess, "Popen", return_value=fake):
            self.sup.start({"target": str(self.tmp / "code")})
        fake._running = False
        out = self.sup.status()
        self.assertFalse(out["running"])
        self.assertEqual(out["run"]["exit_code"], 0)


class RealProcessTests(unittest.TestCase):
    """Spawns an actual child, so the Popen wiring is not only mocked.

    A stub launcher is enough: what is pinned here is that the transcript path
    the UI hands over is the one the child writes, that the exit code reaches
    the status, and that the stdout log is readable afterwards.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mantis_ui_real_")).resolve()
        (self.tmp / "code").mkdir()
        self.launch = self.tmp / "launch.py"
        self.launch.write_text(
            "import json, sys\n"
            "args = sys.argv[1:]\n"
            "if '--dry-run' in args:\n"
            "    print('Dry-Run Execution Plan: stub'); raise SystemExit(0)\n"
            "out = args[args.index('--transcript-out') + 1]\n"
            "with open(out, 'w', encoding='utf-8') as fh:\n"
            "    fh.write(json.dumps({'type': 'text', 'text': 'stage one'}) + '\\n')\n"
            "print('stub run finished')\n"
        )
        self.sup = Supervisor(allow_root=self.tmp, launch_py=self.launch,
                              state_dir=self.tmp / "state", db_path=str(self.tmp / "k.db"))

    def tearDown(self):
        self.sup.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_start_really_spawns_and_records_the_transcript(self):
        import time

        out = self.sup.start({"target": str(self.tmp / "code")})
        self.assertTrue(out["ok"], out)
        deadline = time.time() + 30
        while time.time() < deadline:
            if not self.sup.status()["running"]:
                break
            time.sleep(0.1)
        status = self.sup.status()
        self.assertFalse(status["running"], "the stub child never exited")
        self.assertEqual(status["run"]["exit_code"], 0, status["run"]["stdout_tail"])
        transcript = Path(status["run"]["transcript_path"])
        self.assertTrue(transcript.is_file())
        self.assertIn("stage one", transcript.read_text())
        self.assertIn("stub run finished", status["run"]["stdout_tail"])


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mantis_ui_plan_")).resolve()
        (self.tmp / "code").mkdir()
        self.sup = Supervisor(allow_root=self.tmp, launch_py=self.tmp / "launch.py",
                              state_dir=self.tmp / "state", db_path=str(self.tmp / "k.db"))

    def tearDown(self):
        self.sup.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_plan_runs_the_launcher_with_dry_run_and_returns_its_output(self):
        seen = {}

        class _Done:
            returncode = 0
            stdout = "Dry-Run Execution Plan: target ok\n"
            stderr = ""

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            return _Done()

        with mock.patch.object(runsupervisor.subprocess, "run", side_effect=fake_run):
            out = self.sup.plan({"target": str(self.tmp / "code")})
        self.assertTrue(out["ok"], out)
        self.assertIn("Dry-Run Execution Plan", out["plan"])
        self.assertIn("--dry-run", seen["cmd"])
        self.assertIn("--yes", seen["cmd"])

    def test_plan_refuses_an_invalid_field_before_spawning_anything(self):
        with mock.patch.object(runsupervisor.subprocess, "run") as run:
            out = self.sup.plan({"target": str(self.tmp / "code"),
                                 "scan_mode": "changed"})
        self.assertFalse(out["ok"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
