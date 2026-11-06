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

"""Supervises `scripts/launch.py` as a child process: plan, spawn, cancel.

This is a supervisor, not a second launcher. Budget parsing, preflight, sandbox
downgrade refusal and scan-mode resolution stay in launch.py; duplicating any
of it here would drift from it. What this module adds is the part a browser
needs: a fixed field whitelist turned into an argv array (never a shell
string), a target jailed to one allowed root, and a process group that can be
interrupted so the run checkpoints and stays `--resume`-able.

No HTTP in this module: the request surface is `scripts/ui.py`, and everything
below is testable without a socket.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# "changed" is deliberately absent: it needs a --changed-files path, and
# exposing a caller-supplied path here would reopen the jail this module exists
# to hold.
SCAN_MODES = ("auto", "whole", "file-by-file", "cross-functional")
SANDBOXES = ("static-only", "gvisor", "microsandbox", "gce")

_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,119}$")
_EXCLUDE_RE = re.compile(r"^[A-Za-z0-9._/*?\[\]-]{1,200}$")
_MAX_TOKENS_CEILING = 10_000_000_000
_MAX_FOCUS = 500
_MAX_API_BASE = 300


def _clean(value: Any, limit: int) -> str:
    """Strips terminal control characters and caps length. Never raises."""
    text = "" if value is None else str(value)
    text = "".join(ch for ch in text if ch == "\t" or ch >= " ")
    return text[:limit]


def _within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass
class SupervisedRun:
    """One child run. `argv` is the validated array actually spawned."""

    run_id: str
    argv: List[str]
    proc: Any
    dir: Path
    target: str
    scan_mode: str
    started_at: float
    launch_py: Path
    exit_code: Optional[int] = None
    log: Any = None

    @property
    def transcript_path(self) -> Path:
        return self.dir / "transcript.jsonl"

    @property
    def stdout_path(self) -> Path:
        return self.dir / "stdout.log"

    def running(self) -> bool:
        return self.exit_code is None and self.proc is not None and self.proc.poll() is None

    def refresh(self) -> None:
        if self.proc is not None and self.exit_code is None:
            self.exit_code = self.proc.poll()
        if self.exit_code is not None and self.log is not None:
            try:
                self.log.close()
            finally:
                self.log = None

    def reap_in_background(self) -> None:
        """Collects the exit status without a request, so a finished child is
        never left a zombie until someone polls. Best-effort: the status is
        still collected by refresh() if this thread never runs."""
        def _wait():
            try:
                self.proc.wait()
            finally:
                self.refresh()
        threading.Thread(target=_wait, name=f"reap-{self.run_id}", daemon=True).start()

    def resume_command(self) -> str:
        """The command that picks this run back up after an interruption.

        Built from the argv this run was actually spawned with, so the database
        and every per-run override (sandbox, model, api-base, focus, budget,
        excludes) survive. A hand-assembled command dropped them: `--resume`
        with no `--db` resolves "latest" against the installation database, so
        the printed command resumed a different run in a different database.
        """
        parts = [str(Path(sys.executable)), str(self.launch_py)]
        skip_next = False
        for tok in self.argv:
            if skip_next:
                skip_next = False
                continue
            if tok == "--transcript-out":
                # The UI's own transcript destination; a resume writes its own.
                skip_next = True
                continue
            parts.append(tok)
        if "--resume" not in parts:
            parts.append("--resume")
        return " ".join(shlex.quote(p) for p in parts)


class Supervisor:
    """Owns at most one child run at a time, per database."""

    def __init__(self, allow_root: Path, launch_py: Path, state_dir: Path,
                 db_path: str = "", python: Optional[str] = None) -> None:
        self.allow_root = Path(allow_root).resolve()
        self.launch_py = Path(launch_py).resolve()
        self.state_dir = Path(state_dir)
        self.db_path = db_path
        self.python = python or sys.executable
        self.current: Optional[SupervisedRun] = None
        # start() is check-then-act and the UI serves each request on its own
        # thread: two POSTs (a double-click, two tabs) both saw `current` idle
        # and both spawned, leaving one child unmanaged.
        self._lock = threading.Lock()

    # --- validation ---------------------------------------------------------

    def validate_fields(self, fields: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        """Turns untrusted request fields into a validated dict, or explains why not.

        Every key not named here is dropped: the browser never composes argv.
        """
        from core.paths import validate_scan_target

        target_raw = _clean(fields.get("target"), 4096).strip()
        if not target_raw:
            return None, "A target file or directory is required."
        resolved, err = validate_scan_target(target_raw)
        if resolved is None:
            return None, err
        if not _within(self.allow_root, resolved):
            return None, (f"Target '{resolved}' is outside the allowed root "
                          f"'{self.allow_root}'.")
        if not (resolved.is_dir() or resolved.is_file()):
            return None, f"Target '{resolved}' is neither a file nor a directory."

        scan_mode = _clean(fields.get("scan_mode") or "auto", 40).strip().lower()
        if scan_mode not in SCAN_MODES:
            return None, f"scan_mode must be one of {list(SCAN_MODES)}."

        sandbox = _clean(fields.get("sandbox"), 40).strip().lower()
        if sandbox and sandbox not in SANDBOXES:
            return None, f"sandbox must be one of {list(SANDBOXES)}."

        model_raw = "" if fields.get("model") is None else str(fields.get("model"))
        if len(model_raw) > 120:
            return None, "model must be 120 characters or fewer."
        model = _clean(model_raw, 120).strip()
        if model and not _MODEL_RE.match(model):
            return None, ("model must be a provider id (letters, digits, "
                          "and . _ / : @ + - only).")

        api_base = _clean(fields.get("api_base"), _MAX_API_BASE).strip()
        if api_base and not re.match(r"^https?://[^\s]+$", api_base):
            return None, "api_base must be an http:// or https:// URL."

        focus = _clean(fields.get("focus"), _MAX_FOCUS).strip()

        max_tokens = fields.get("max_tokens")
        if max_tokens in (None, ""):
            max_tokens_val = None
        else:
            try:
                max_tokens_val = int(max_tokens)
            except (TypeError, ValueError):
                return None, "max_tokens must be an integer."
            if not 0 <= max_tokens_val <= _MAX_TOKENS_CEILING:
                return None, f"max_tokens must be between 0 and {_MAX_TOKENS_CEILING}."

        raw_exclude = fields.get("exclude") or []
        if isinstance(raw_exclude, str):
            raw_exclude = [p for p in raw_exclude.split(",") if p.strip()]
        if not isinstance(raw_exclude, (list, tuple)):
            return None, "exclude must be a list of patterns."
        exclude: List[str] = []
        for pattern in raw_exclude:
            if len(str(pattern)) > 200:
                return None, "exclude patterns must be 200 characters or fewer."
            pat = _clean(pattern, 200).strip()
            if not pat:
                continue
            if not _EXCLUDE_RE.match(pat):
                return None, f"exclude pattern {pat!r} has unsupported characters."
            exclude.append(pat)

        return {
            "target": str(resolved),
            "scan_mode": scan_mode,
            "sandbox": sandbox,
            "model": model,
            "api_base": api_base,
            "focus": focus,
            "max_tokens": max_tokens_val,
            "exclude": exclude,
            "resume": bool(fields.get("resume")),
        }, ""

    def build_argv(self, fields: Dict[str, Any]) -> List[str]:
        """argv for launch.py from already-validated fields. Always a list."""
        argv: List[str] = [fields["target"]]
        argv += ["--scan-mode", fields["scan_mode"]]
        if fields.get("sandbox"):
            argv += ["--sandbox", fields["sandbox"]]
        if fields.get("model"):
            argv += ["--model", fields["model"]]
        if fields.get("api_base"):
            argv += ["--api-base", fields["api_base"]]
        if fields.get("focus"):
            argv += ["--focus", fields["focus"]]
        if fields.get("max_tokens") is not None:
            argv += ["--token-budget", str(fields["max_tokens"])]
        for pattern in fields.get("exclude") or []:
            argv += ["--exclude", pattern]
        if self.db_path:
            argv += ["--db", str(self.db_path)]
        if fields.get("resume"):
            argv += ["--resume"]
        return argv

    def _env(self) -> Dict[str, str]:
        env = dict(os.environ)
        # The child is not attached to a terminal; nothing may block on a prompt.
        env["MANTIS_UI_CHILD"] = "1"
        env.pop("MANTIS_PROBE_LLM", None)
        return env

    # --- plan ---------------------------------------------------------------

    def plan(self, fields: Dict[str, Any], timeout: float = 180.0) -> Dict[str, Any]:
        """Runs `launch.py --dry-run` and returns its real plan.

        Two-phase on purpose: the confirmation the operator sees is the plan
        launch.py itself computed, so the UI cannot describe a different run
        from the one it then starts.
        """
        validated, err = self.validate_fields(fields)
        if validated is None:
            return {"ok": False, "error": err}
        argv = self.build_argv(validated) + ["--dry-run", "--yes"]
        try:
            proc = subprocess.run(
                [self.python, str(self.launch_py), *argv],
                cwd=str(self.launch_py.parent.parent),
                capture_output=True, text=True, timeout=timeout, env=self._env(),
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"The plan did not finish within {timeout:.0f}s."}
        except OSError as exc:
            return {"ok": False, "error": f"Could not run the launcher: {exc}"}
        output = (proc.stdout or "") + (proc.stderr or "")
        return {"ok": proc.returncode == 0, "exit_code": proc.returncode,
                "plan": output, "fields": validated,
                "error": "" if proc.returncode == 0 else output.strip()[-2000:]}

    # --- start / status / cancel -------------------------------------------

    def start(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Spawns the run. One at a time: a second top-level run would fight the
        same SQLite database for the write lock."""
        validated, err = self.validate_fields(fields)
        if validated is None:
            return {"ok": False, "error": err}
        with self._lock:
            return self._start_locked(validated)

    def _start_locked(self, validated: Dict[str, Any]) -> Dict[str, Any]:
        if self.current is not None:
            self.current.refresh()
            if self.current.running():
                return {"ok": False,
                        "error": f"Run {self.current.run_id} is still running. "
                                 "Cancel it before starting another."}

        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        run_dir = self.state_dir / "runs" / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return {"ok": False, "error": f"Cannot create the run directory: {exc}"}

        argv = self.build_argv(validated) + [
            "--yes", "--transcript-out", str(run_dir / "transcript.jsonl"),
        ]
        try:
            log = (run_dir / "stdout.log").open("a", encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": f"Could not open the run log: {exc}"}
        try:
            proc = subprocess.Popen(
                [self.python, str(self.launch_py), *argv],
                cwd=str(self.launch_py.parent.parent),
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                env=self._env(), start_new_session=True,
            )
        except OSError as exc:
            log.close()
            return {"ok": False, "error": f"Could not start the run: {exc}"}
        self.current = SupervisedRun(
            run_id=run_id, argv=argv, proc=proc, dir=run_dir,
            target=validated["target"], scan_mode=validated["scan_mode"],
            started_at=time.time(), launch_py=self.launch_py, log=log,
        )
        self.current.reap_in_background()
        return {"ok": True, "run": self.status()}

    def status(self) -> Dict[str, Any]:
        run = self.current
        if run is None:
            return {"running": False, "run": None}
        run.refresh()
        return {
            "running": run.running(),
            "run": {
                "run_id": run.run_id,
                "target": run.target,
                "scan_mode": run.scan_mode,
                "started_at": run.started_at,
                "elapsed_seconds": round(time.time() - run.started_at, 1),
                "exit_code": run.exit_code,
                "argv": run.argv,
                "stdout_tail": _tail(run.stdout_path, 40),
                "transcript_path": str(run.transcript_path),
                "resume_command": run.resume_command(),
            },
        }

    def close(self) -> None:
        """Releases the parent's handles. A running child keeps its own fds."""
        run = self.current
        if run is not None and run.log is not None:
            try:
                run.log.close()
            finally:
                run.log = None

    def cancel(self, grace: float = 25.0) -> Dict[str, Any]:
        """SIGINT the process group so the run checkpoints, then escalate."""
        run = self.current
        if run is None:
            return {"ok": False, "error": "No run to cancel."}
        run.refresh()
        if not run.running():
            return {"ok": True, "already_finished": True, "exit_code": run.exit_code}
        try:
            pgid = os.getpgid(run.proc.pid)
        except OSError:
            return {"ok": False, "error": "The run's process group is gone."}
        for sig, wait_for in ((signal.SIGINT, grace), (signal.SIGTERM, 10.0)):
            try:
                os.killpg(pgid, sig)
            except OSError:
                break
            deadline = time.time() + wait_for
            while time.time() < deadline:
                run.refresh()
                if not run.running():
                    return {"ok": True, "exit_code": run.exit_code,
                            "resume_command": run.resume_command()}
                time.sleep(0.2)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
        run.refresh()
        # SIGKILL may have landed before BudgetController wrote its checkpoint,
        # so the run may not be resumable. Say so instead of handing back the
        # same unqualified command as a graceful stop.
        return {"ok": True, "exit_code": run.exit_code, "killed": True,
                "resumable": False,
                "note": "Killed without a checkpoint; --resume may have nothing to pick up.",
                "resume_command": run.resume_command()}


def _tail(path: Path, lines: int) -> str:
    """Last `lines` lines of a log, for a browser that only wants the error."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            data = fh.read()
    except OSError:
        return ""
    return "\n".join(data.splitlines()[-lines:])
