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

import asyncio
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Optional, Union, List
import uuid

from google.adk.environment import ExecutionResult
from core.truncation import truncate_with_marker
from .base import BaseEnvironment
from .static_env import PROTECTED_VCS_DIRS, PROTECTED_METADATA_FILES

# Per-stream cap on what one execute() hands back. Must stay
# >= 10_000 and the cut must be announced (core.truncation): the reproducer
# reads this trace for the reached-sink sentinel, and a silent cut before the
# sentinel reads as "the PoC never got there".
MAX_OUTPUT = 16000


class GvisorEnvironment(BaseEnvironment):
    """OCI container sandbox executed under gVisor runtime (docker/podman --runtime=runsc).

    - ISOLATED FILESYSTEM: Target code is copied into a clean container workspace (/workspace).
    - NO HOST FS ACCESS: Container has no mounts to the host filesystem.
    - NETWORKLESS: --network=none guarantees strict network isolation.
    - STATEFUL EXECUTION: State persists across execute() calls within a file run.
    - ADK COMPLIANT: Implements execute(), read_file(), write_file(), initialize(), close().
    """

    def __init__(
        self,
        target_path: str = "",
        image: str = "mantis-sandbox:latest",
        runtime: str = "runsc",
        timeout_seconds: int = 30,
        container_tool: Optional[str] = None,
        workdir: str = "/workspace",
    ):
        super().__init__()
        self.target_path = os.path.abspath(target_path) if target_path else ""
        self.image = image
        self.runtime = runtime
        self.timeout = timeout_seconds
        self._workdir = Path(workdir)

        tool = container_tool
        if not tool:
            for candidate in ("docker", "podman"):
                if shutil.which(candidate) is not None:
                    tool = candidate
                    break
        elif tool not in ("docker", "podman"):
            raise ValueError(f"Invalid container_tool '{tool}'. Must be 'docker' or 'podman'.")

        if not tool or shutil.which(tool) is None:
            raise ValueError(
                f"sandbox type 'gvisor' requires 'docker' or 'podman' with '{runtime}' on PATH. "
                "Install a container engine with gVisor or set sandbox.type to 'static-only'."
            )
        self.tool = tool
        self.container_name = f"mantis-gv-{uuid.uuid4().hex[:12]}"
        self._started = False
        self._lock = asyncio.Lock()

    @property
    def working_dir(self) -> Path:
        return self._workdir

    def _run_cmd(self, argv: list[str], stdin: Optional[str] = None, timeout: Optional[int] = None) -> tuple[int, str]:
        t = timeout or self.timeout
        try:
            p = subprocess.run(
                argv,
                input=stdin or "",
                capture_output=True,
                text=True,
                timeout=t,
            )
            out = (p.stdout or "") + (p.stderr or "")
            return p.returncode, out
        except subprocess.TimeoutExpired:
            return -1, f"SANDBOX-TIMEOUT: command exceeded {t}s."
        except Exception as e:
            return -1, f"SANDBOX-ERROR: {e}"

    async def _ensure(self) -> None:
        fresh = False
        async with self._lock:
            if self._started:
                return

            def _init():
                # 1. Create container with runsc runtime, no network, and strict resource ceilings
                create_cmd = [
                    self.tool, "create",
                    "--pull=never",
                    "--name", self.container_name,
                    f"--runtime={self.runtime}",
                    "--network=none",
                    "--pids-limit=256",
                    "--memory=4g",
                    "--cpus=2.0",
                    "-w", str(self._workdir),
                    self.image,
                    "sleep", "infinity",
                ]
                rc, out = self._run_cmd(create_cmd)
                if rc != 0:
                    raise RuntimeError(f"Failed to create gVisor container '{self.container_name}': {out}")

                # 2. Start container
                rc, out = self._run_cmd([self.tool, "start", self.container_name])
                if rc != 0:
                    self._run_cmd([self.tool, "rm", "-f", self.container_name])
                    raise RuntimeError(f"Failed to start gVisor container '{self.container_name}': {out}")

                # 3. Stage target file or directory using canonical fail-closed staging helper
                if self.target_path and os.path.exists(self.target_path):
                    from .staging import stage_to_directory
                    with tempfile.TemporaryDirectory() as stage_dir:
                        count = stage_to_directory(self.target_path, stage_dir)
                        if count > 0:
                            rc, out = self._run_cmd([
                                self.tool, "cp",
                                f"{stage_dir}/.",
                                f"{self.container_name}:{self._workdir}/",
                            ])
                            if rc != 0:
                                self._run_cmd([self.tool, "rm", "-f", self.container_name])
                                raise RuntimeError(f"Failed to stage files into gVisor container: {out}")

            await asyncio.to_thread(_init)
            self._started = True
            self.is_initialized = True
            fresh = True
        if fresh:
            # Staging is done: anything newer than this was made by the
            # campaign (see core.poc_scaffold). Outside the lock: write_file
            # re-enters _ensure.
            from core.poc_scaffold import mark_staged
            await mark_staged(self.write_file)

    async def initialize(self) -> None:
        await self._ensure()

    async def execute(self, command: str, *, timeout: Optional[float] = None) -> ExecutionResult:
        await self._ensure()
        t = int(timeout) if timeout is not None else self.timeout

        def _exec():
            try:
                p = subprocess.run(
                    [
                        self.tool, "exec",
                        self.container_name,
                        "/bin/sh", "-c", command,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=t,
                )
                return ExecutionResult(
                    stdout=truncate_with_marker(p.stdout or "", MAX_OUTPUT, what="sandbox stdout"),
                    stderr=truncate_with_marker(p.stderr or "", MAX_OUTPUT, what="sandbox stderr"),
                    exit_code=p.returncode,
                    timed_out=False,
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    stdout="",
                    stderr=f"SANDBOX-TIMEOUT: command exceeded {t}s.",
                    exit_code=-1,
                    timed_out=True,
                )
            except Exception as e:
                return ExecutionResult(
                    stdout="",
                    stderr=f"SANDBOX-ERROR: {e}",
                    exit_code=1,
                    timed_out=False,
                )

        return await asyncio.to_thread(_exec)

    def _confine_to_workdir(self, candidate: Path, original: Union[Path, str]) -> Path:
        """Lexically normalizes candidate and requires it to stay inside the workdir."""
        norm = Path(os.path.normpath(str(candidate)))
        if norm != self._workdir and self._workdir not in norm.parents:
            raise PermissionError(
                f"Path '{original}' escapes the sandbox workspace '{self._workdir}'."
            )
        return norm

    def _resolve_guest_path(self, path: Union[Path, str]) -> Path:
        p = Path(path)
        if not p.is_absolute():
            return self._confine_to_workdir(self._workdir / p, path)
        try:
            if p == self._workdir or self._workdir in p.parents:
                return self._confine_to_workdir(p, path)
        except PermissionError:
            raise
        except Exception:
            pass
        if self.target_path:
            target_p = Path(self.target_path)
            if target_p.is_file():
                if p == target_p:
                    return self._confine_to_workdir(self._workdir / target_p.name, path)
                if target_p.parent in p.parents or p == target_p.parent:
                    try:
                        rel = p.relative_to(target_p.parent)
                        return self._confine_to_workdir(self._workdir / rel, path)
                    except ValueError:
                        pass
            elif target_p.is_dir():
                if target_p in p.parents or p == target_p:
                    try:
                        rel = p.relative_to(target_p)
                        return self._confine_to_workdir(self._workdir / rel, path)
                    except ValueError:
                        pass
        return self._confine_to_workdir(self._workdir / p.name, path)

    async def read_file(self, path: Union[Path, str]) -> bytes:
        await self._ensure()
        resolved_path = self._resolve_guest_path(path)

        def _read():
            try:
                p_exec = subprocess.run(
                    [self.tool, "exec", self.container_name, "cat", str(resolved_path)],
                    capture_output=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                raise TimeoutError(f"Read timed out in gVisor container: {path}")
            if p_exec.returncode != 0:
                raise FileNotFoundError(f"File not found in gVisor container: {path} ({p_exec.stderr.decode('utf-8', errors='replace')})")
            return p_exec.stdout

        return await asyncio.to_thread(_read)

    async def write_file(self, path: Union[Path, str], content: Union[str, bytes]) -> None:
        await self._ensure()
        resolved_path = self._resolve_guest_path(path)
        parent_dir = str(resolved_path.parent)
        data = content.encode("utf-8") if isinstance(content, str) else content

        def _write():
            mkdir_cmd = [self.tool, "exec", self.container_name, "mkdir", "-p", parent_dir]
            subprocess.run(mkdir_cmd, capture_output=True, timeout=self.timeout)

            # SECURITY: Pass destination as positional argument to prevent shell quoting injection
            write_cmd = [
                self.tool, "exec", "-i", self.container_name,
                "/bin/sh", "-c", 'cat > "$1"', "sh", str(resolved_path),
            ]
            p = subprocess.run(write_cmd, input=data, capture_output=True, timeout=self.timeout)
            if p.returncode != 0:
                raise RuntimeError(f"Failed to write file to gVisor container: {resolved_path} ({p.stderr.decode('utf-8', errors='replace')})")

        await asyncio.to_thread(_write)

    async def list_files(self, directory: str = "") -> List[str]:
        await self._ensure()
        resolved_dir = str(self._resolve_guest_path(directory)) if directory else str(self._workdir)

        def _list():
            find_cmd = [
                self.tool, "exec", self.container_name,
                "find", resolved_dir, "-maxdepth", "5", "-not", "-path", "*/.*", "-type", "f"
            ]
            rc, out = self._run_cmd(find_cmd)
            if rc != 0:
                raise RuntimeError(f"Failed to list files in gVisor container: {out}")
            file_list = []
            for line in out.strip().splitlines():
                line = line.strip()
                if line:
                    rel = os.path.relpath(line, str(self._workdir))
                    file_list.append(rel)
            return sorted(file_list)

        return await asyncio.to_thread(_list)

    async def preflight(self) -> None:
        def _check():
            out = subprocess.run(
                [self.tool, "info", "--format", "{{json .Runtimes}}"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode != 0:
                err_msg = (out.stderr or "").strip() or f"{self.tool} daemon not reachable"
                raise RuntimeError(
                    f"Could not connect to {self.tool} daemon ({err_msg}). "
                    f"Ensure {self.tool} is running, or set sandbox.type to 'static-only'."
                )

            try:
                runtimes = json.loads(out.stdout or "{}")
            except json.JSONDecodeError:
                runtimes = {}

            if self.runtime not in runtimes:
                raise RuntimeError(
                    f"'{self.runtime}' is not a registered {self.tool} runtime. "
                    f"Run `sudo runsc install -- --network=none && sudo systemctl restart {self.tool}`, "
                    "or set sandbox.type to 'microsandbox' or 'static-only'."
                )

            img_out = subprocess.run(
                [self.tool, "image", "inspect", self.image],
                capture_output=True, text=True, timeout=10,
            )
            if img_out.returncode != 0:
                raise RuntimeError(
                    f"sandbox image '{self.image}' not found in the local {self.tool} cache. "
                    f"Run ./install.sh to build it, or set sandbox.type to 'static-only'."
                )
        await asyncio.to_thread(_check)

    async def close(self) -> None:
        if self._started:
            def _cleanup():
                self._run_cmd([self.tool, "rm", "-f", self.container_name])
            await asyncio.to_thread(_cleanup)
            self._started = False
            self.is_initialized = False
