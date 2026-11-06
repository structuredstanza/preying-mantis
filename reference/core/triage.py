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

"""Operator triage: the one write path a human uses to close a single finding.

The MCP tool (`mantis_resolve_finding`) and the local UI both call this, so a
dismissal means the same thing whichever surface issued it -- one validated
resolution, the database's own monotonic guard, one audit line. Two surfaces
with two copies of this logic would drift, and the guard is the part that must
not.

Machine statuses (`dynamic_confirmed`) stay reserved for the verification
pipeline: this module never sets them, and the guard refuses to overwrite them
with a dismissal.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

RESOLUTION_STATUSES = ("false_positive",)

AUDIT_LOG_NAME = "resolutions.log"


def _append_audit(log_path: str, record: Dict[str, Any]) -> bool:
    """Appends one JSON line to the audit log.

    A symlink at the log path is refused rather than followed: the state
    directory is ours, and a link planted there would turn an operator verdict
    into a write anywhere the process can reach.
    """
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(log_path), flags, 0o600)
    except OSError:
        return False
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return True


def resolve_finding(
    db_path: str,
    finding_id: Any,
    resolution: str,
    reason: str,
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Closes one finding with an operator verdict and an audit trail.

    Returns `{finding_id, old_status, new_status, applied}` on success, plus a
    `note` naming why the monotonic guard refused the transition when it did.
    Returns `{error}` for bad input. Never raises.
    """
    res = str(resolution or "").strip().lower()
    if res not in RESOLUTION_STATUSES:
        return {
            "error": f"resolution must be one of {list(RESOLUTION_STATUSES)}: "
                     "'false_positive' means the report is wrong.",
        }
    if not str(reason or "").strip():
        return {"error": "A non-empty reason is required: it is the audit "
                         "trail for overriding the scanner."}
    try:
        fid = int(finding_id)
    except (TypeError, ValueError):
        return {"error": f"finding_id must be an integer, got {finding_id!r}."}

    from core.database import read_findings, update_finding_status_by_id

    def row_for(i):
        for f in read_findings(db_path):
            if f.get("id") == i:
                return f
        return None

    row = row_for(fid)
    if row is None:
        return {"error": f"No finding with id {fid}."}
    old_status = str(row.get("status") or "")
    update_finding_status_by_id(db_path, fid, str(row.get("run_id") or ""), res)
    new_status = str((row_for(fid) or {}).get("status") or "")
    applied = new_status.lower() == res
    out: Dict[str, Any] = {
        "finding_id": fid,
        "old_status": old_status,
        "new_status": new_status,
        "applied": applied,
    }
    if not applied:
        from core.database import MONOTONIC_TERMINAL_STATUSES

        if old_status.lower() == "dynamic_confirmed":
            out["note"] = ("The monotonic status guard refused the transition: "
                           "a dismissal never overwrites machine-verified "
                           "evidence (dynamic_confirmed).")
        elif old_status.lower() in MONOTONIC_TERMINAL_STATUSES:
            out["note"] = ("The monotonic status guard refused the transition: "
                           f"'{old_status}' is already a terminal status and "
                           "is preserved.")
        else:
            # The guard was not what stopped it: the UPDATE matches on
            # (id, run_id), and a row whose run_id is NULL (or differs) matches
            # nothing. Say that instead of asserting a terminal status the row
            # does not have.
            out["note"] = ("The row did not change: the status update matched no "
                           f"row for id {fid} with its recorded run_id. The "
                           f"finding is still '{old_status}'.")
    if log_path:
        _append_audit(str(log_path), {
            "at": time.time(),
            "finding_id": fid,
            "lineage_id": row.get("lineage_id"),
            "title": row.get("title"),
            "old_status": old_status,
            "new_status": new_status,
            "applied": applied,
            "resolution": res,
            "reason": str(reason).strip(),
        })
    return out
