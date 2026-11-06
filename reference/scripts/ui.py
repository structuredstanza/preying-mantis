#!/usr/bin/env python3
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

"""A local UI for one Mantis database: watch a run, triage findings, start one.

Three jobs, nothing else:

  * watch   -- tail the run's transcript JSONL, which `main.py --transcript-out`
               already writes and flushes per row
  * triage  -- close a finding through core/triage.py, so the UI and the MCP
               tool issue the same verdict through the same monotonic guard
  * launch  -- two-phase: `launch.py --dry-run` prints the plan, the operator
               confirms, then the validated argv is spawned and supervised

Every value in the findings table was written by a model reading a hostile
target, so it is escaped at render and never assigned to innerHTML. The server
binds loopback unless a token is supplied, and every mutating request needs a
custom header a cross-origin form cannot set.

Usage:
    python3 scripts/ui.py --db knowledge.db --allow-root /path/to/code
    python3 scripts/ui.py --db knowledge.db --host 0.0.0.0 --token "$(openssl rand -hex 16)"
"""

from __future__ import annotations

import argparse
import hmac
import html
import json
import os
import re
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runsupervisor import SCAN_MODES, SANDBOXES, Supervisor  # noqa: E402

MAX_BODY = 64 * 1024
MAX_TRANSCRIPT_ROWS = 400
_SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")


class App:
    """Everything a request needs, with no HTTP in it."""

    def __init__(self, db_path: str, allow_root: Path, state_dir: Path,
                 supervisor: Supervisor, token: str = "",
                 transcript: str = "", host: str = "127.0.0.1") -> None:
        self.db_path = db_path
        self.allow_root = allow_root
        self.state_dir = state_dir
        self.supervisor = supervisor
        self.token = token
        self.transcript = transcript
        self.host = host

    # --- findings -----------------------------------------------------------

    def db_present(self) -> bool:
        return bool(self.db_path) and os.path.exists(self.db_path)

    def findings(self, status: str = "", severity: str = "",
                 query: str = "", run_id: str = "") -> List[Dict[str, Any]]:
        """Reads findings through the canonical reader. Never creates the DB."""
        if not self.db_present():
            return []
        from core.database import read_findings

        rows = read_findings(self.db_path, status=(status or "").lower() or None,
                             run_id=run_id or None)
        sev = severity.upper()
        q = query.lower().strip()
        out = []
        for row in rows:
            if sev and str(row.get("severity") or "").upper() != sev:
                continue
            if q:
                haystack = " ".join(str(row.get(k) or "") for k in
                                    ("title", "filepath", "cwe", "description"))
                if q not in haystack.lower():
                    continue
            out.append({
                "id": row.get("id"),
                "title": row.get("title"),
                "severity": row.get("severity"),
                "status": row.get("status"),
                "filepath": row.get("filepath"),
                "cwe": row.get("cwe"),
                "score": row.get("mantis_risk_score"),
                "line_numbers": row.get("line_numbers"),
                "description": row.get("description"),
                "lineage_id": row.get("lineage_id"),
                "run_id": row.get("run_id"),
            })
        return out

    def resolve(self, finding_id: Any, resolution: str, reason: str) -> Dict[str, Any]:
        """Records an operator verdict through the shared triage path.

        The page offers only a dismissal control, and the endpoint accepts the
        same single resolution `mantis_resolve_finding` does: a finding is
        dismissed as false positive, or left alone. This tool reports defects;
        it does not adjudicate fixes.
        """
        if not self.db_present():
            return {"error": "No findings database."}
        from core.triage import AUDIT_LOG_NAME, resolve_finding

        self.state_dir.mkdir(parents=True, exist_ok=True)
        return resolve_finding(self.db_path, finding_id, resolution, reason,
                               log_path=self.state_dir / AUDIT_LOG_NAME)

    # --- transcript ---------------------------------------------------------

    def transcript_path(self) -> Optional[Path]:
        """Only the supervised run's transcript, or the one named at startup."""
        run = self.supervisor.current
        if run is not None and run.transcript_path.exists():
            return run.transcript_path
        if self.transcript:
            path = Path(self.transcript)
            if path.is_file():
                return path
        return None

    def transcript_since(self, offset: int) -> Dict[str, Any]:
        path = self.transcript_path()
        if path is None:
            return {"offset": 0, "rows": [], "present": False}
        try:
            size = path.stat().st_size
        except OSError:
            return {"offset": 0, "rows": [], "present": False}
        if offset < 0 or offset > size:
            offset = 0
        rows: List[Dict[str, Any]] = []
        try:
            with path.open("rb") as fh:
                fh.seek(offset)
                while len(rows) < MAX_TRANSCRIPT_ROWS:
                    line = fh.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        # The writer flushes per row; a line without its newline
                        # is still being written. Leave the offset before it so
                        # the next poll returns the complete row instead of
                        # dropping a half-row forever.
                        break
                    offset += len(line)
                    try:
                        rows.append(json.loads(line.decode("utf-8", "replace")))
                    except ValueError:
                        continue
        except OSError:
            return {"offset": offset, "rows": [], "present": False}
        return {"offset": offset, "rows": rows, "present": True}


def _parse_host(value: str) -> str:
    """Host header without its port; IPv6 brackets preserved."""
    value = (value or "").strip()
    if value.startswith("["):
        end = value.find("]")
        return value[:end + 1] if end != -1 else value
    return value.split(":")[0]


class Handler(BaseHTTPRequestHandler):
    server_version = "mantis-ui"
    app: App  # set on the server

    # --- plumbing -----------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter than default
        if os.environ.get("MANTIS_UI_VERBOSE"):
            sys.stderr.write("ui: " + (fmt % args) + "\n")

    def _host_ok(self) -> bool:
        # A wildcard bind answers on every address, and a browser sends the
        # address it dialled (192.168.1.5:8765), never "0.0.0.0": a fixed set
        # made the documented `--host 0.0.0.0 --token ...` mode answer 421 to
        # everything. Any syntactically valid Host is accepted there; the token
        # is what authorizes the request.
        if self.app.host in ("0.0.0.0", "::"):
            return bool(_parse_host(self.headers.get("Host", "")))
        allowed = {"127.0.0.1", "localhost", "[::1]", "::1", self.app.host}
        return _parse_host(self.headers.get("Host", "")) in allowed

    def _token_from_request(self, query: Dict[str, List[str]]) -> str:
        given = self.headers.get("X-Mantis-Token") or ""
        if not given:
            match = re.search(r"(?:^|;\s*)mantis_token=([^;]+)",
                              self.headers.get("Cookie", ""))
            if match:
                given = urllib.parse.unquote(match.group(1))
        if not given:
            given = (query.get("token") or [""])[0]
        return given

    def _authorized(self, query: Dict[str, List[str]]) -> bool:
        if not self.app.token:
            return True
        given = self._token_from_request(query)
        # compare_digest raises TypeError on non-ASCII str; header bytes decode
        # latin-1, so any high byte would otherwise 500 instead of 401.
        return bool(given) and hmac.compare_digest(
            given.encode("utf-8", "surrogateescape"),
            self.app.token.encode("utf-8", "surrogateescape"),
        )

    def _send(self, code: int, body: bytes, content_type: str,
              extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, code: int = 200) -> None:
        self._send(code, json.dumps(payload, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _error(self, code: int, message: str) -> None:
        self._json({"ok": False, "error": message}, code)

    def _body(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if length <= 0 or length > MAX_BODY:
            return {}
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8", "replace"))
        except (ValueError, OSError):
            return {}
        return payload if isinstance(payload, dict) else {}

    # --- routes -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._get()
        except Exception as exc:  # noqa: BLE001 - a route must answer, not reset
            # A DB the canonical reader refuses (symlinked path, schema-version
            # mismatch) or a filesystem error used to escape and drop the
            # connection, leaving the operator with a blank table and no reason.
            self._error(500, f"{type(exc).__name__}: {exc}")

    def _get(self) -> None:
        if not self._host_ok():
            return self._error(421, "Host header not allowed.")
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if not self._authorized(query):
            return self._send(401, b"token required", "text/plain; charset=utf-8")

        if parsed.path in ("/", "/index.html"):
            headers = {}
            if self.app.token and not self.headers.get("X-Mantis-Token"):
                headers["Set-Cookie"] = (
                    "mantis_token=" + urllib.parse.quote(self.app.token)
                    + "; HttpOnly; SameSite=Strict; Path=/"
                )
            return self._send(200, _page(self.app).encode("utf-8"),
                              "text/html; charset=utf-8", headers)

        if parsed.path == "/api/state":
            return self._json({
                "ok": True,
                "db_present": self.app.db_present(),
                "db_path": self.app.db_path,
                "allow_root": str(self.app.allow_root),
                "scan_modes": list(SCAN_MODES),
                "sandboxes": list(SANDBOXES),
                "severities": list(_SEVERITIES),
                **self.app.supervisor.status(),
            })

        if parsed.path == "/api/findings":
            rows = self.app.findings(
                status=(query.get("status") or [""])[0],
                severity=(query.get("severity") or [""])[0],
                query=(query.get("q") or [""])[0],
                run_id=(query.get("run_id") or [""])[0],
            )
            return self._json({"ok": True, "findings": rows, "count": len(rows)})

        if parsed.path == "/api/transcript":
            try:
                offset = int((query.get("offset") or ["0"])[0])
            except ValueError:
                offset = 0
            return self._json({"ok": True, **self.app.transcript_since(offset)})

        return self._error(404, "Not found.")

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._post()
        except Exception as exc:  # noqa: BLE001 - a route must answer, not reset
            self._error(500, f"{type(exc).__name__}: {exc}")

    def _post(self) -> None:
        if not self._host_ok():
            return self._error(421, "Host header not allowed.")
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if not self._authorized(query):
            return self._error(401, "token required")
        # A cross-origin HTML form cannot set a custom header; this server never
        # answers a CORS preflight, so requiring one closes CSRF.
        if self.headers.get("X-Mantis-UI") != "1":
            return self._error(403, "Missing X-Mantis-UI header.")

        payload = self._body()

        if parsed.path == "/api/plan":
            result = self.app.supervisor.plan(payload)
            return self._json(result, 200 if result.get("ok") else 400)

        if parsed.path == "/api/runs":
            result = self.app.supervisor.start(payload)
            return self._json(result, 200 if result.get("ok") else 409)

        if parsed.path == "/api/runs/cancel":
            result = self.app.supervisor.cancel()
            return self._json(result, 200 if result.get("ok") else 409)

        if parsed.path == "/api/findings/resolve":
            result = self.app.resolve(payload.get("finding_id"),
                                      payload.get("resolution", ""),
                                      payload.get("reason", ""))
            return self._json({"ok": "error" not in result, **result},
                              200 if "error" not in result else 400)

        return self._error(404, "Not found.")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mantis</title>
<style>
  :root { color-scheme: light dark; --fg:#1b1b1b; --muted:#666; --line:#d8d8d8;
          --bg:#fff; --surface:#f6f6f6; --accent:#0b6bcb; --bad:#b3261e;
          --warn:#8a5a00; --ok:#1a6b2f; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e8e8e8; --muted:#9a9a9a; --line:#3a3a3a; --bg:#161616;
            --surface:#1f1f1f; --accent:#6cb6ff; --bad:#ff8a80; --warn:#e0b050;
            --ok:#7ed493; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.45 ui-sans-serif, system-ui, sans-serif;
         color:var(--fg); background:var(--bg); }
  header { padding:10px 16px; border-bottom:1px solid var(--line);
           display:flex; gap:16px; align-items:baseline; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:650; letter-spacing:.02em; }
  .muted { color:var(--muted); }
  main { display:grid; grid-template-columns: minmax(0,3fr) minmax(0,2fr);
         gap:0; height:calc(100vh - 46px); }
  section { padding:14px 16px; overflow:auto; min-width:0; }
  #right { border-left:1px solid var(--line); }
  fieldset { border:1px solid var(--line); border-radius:6px; padding:10px 12px;
             margin:0 0 14px; }
  legend { font-weight:600; padding:0 6px; }
  label { display:block; font-size:12px; color:var(--muted); margin:8px 0 3px; }
  input, select, textarea { width:100%; padding:5px 7px; font:inherit;
    color:var(--fg); background:var(--surface); border:1px solid var(--line);
    border-radius:4px; }
  .row { display:grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr));
         gap:0 10px; }
  button { font:inherit; padding:5px 11px; border-radius:4px; cursor:pointer;
           border:1px solid var(--line); background:var(--surface); color:var(--fg); }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  button:disabled { opacity:.45; cursor:default; }
  .actions { display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; }
  pre { margin:8px 0 0; padding:9px 11px; background:var(--surface);
        border:1px solid var(--line); border-radius:4px; overflow:auto;
        max-height:38vh; white-space:pre-wrap; word-break:break-word;
        font:12px/1.4 ui-monospace, monospace; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:6px 7px; border-bottom:1px solid var(--line);
           vertical-align:top; }
  th { font-size:11px; text-transform:uppercase; letter-spacing:.05em;
       color:var(--muted); position:sticky; top:0; background:var(--bg); }
  td.path { font:12px ui-monospace, monospace; color:var(--muted);
            word-break:break-all; }
  .sev { font-weight:650; font-size:11px; letter-spacing:.04em; }
  .CRITICAL { color:var(--bad); } .HIGH { color:var(--bad); }
  .MEDIUM { color:var(--warn); } .LOW, .INFO { color:var(--muted); }
  .st { font:11px ui-monospace, monospace; color:var(--muted); }
  #transcript { max-height:none; height:100%; }
  #banner { padding:6px 16px; font-size:12px; background:var(--surface);
            border-bottom:1px solid var(--line); }
  .ok { color:var(--ok); } .bad { color:var(--bad); }
  td.act { white-space:nowrap; }
  td.act button { padding:2px 7px; font-size:12px; margin:1px 0; }
</style>
</head>
<body>
<header>
  <h1>Mantis</h1>
  <span id="db" class="muted"></span>
  <span id="runstate" class="muted"></span>
</header>
<div id="banner" class="muted">Idle.</div>
<main>
  <section id="left">
    <fieldset>
      <legend>Run</legend>
      <label for="target">Target (file or directory, inside the allowed root)</label>
      <input id="target" placeholder="/path/to/code" autocomplete="off">
      <div class="row">
        <div><label for="scan_mode">Scan mode</label>
          <select id="scan_mode"></select></div>
        <div><label for="sandbox">Sandbox (blank = configured)</label>
          <select id="sandbox"></select></div>
        <div><label for="model">Model (blank = configured)</label>
          <input id="model" autocomplete="off"></div>
        <div><label for="max_tokens">Token budget (blank = configured)</label>
          <input id="max_tokens" inputmode="numeric" autocomplete="off"></div>
      </div>
      <label for="focus">Focus (plain-language steer, optional)</label>
      <input id="focus" autocomplete="off">
      <label for="exclude">Exclude patterns (comma separated, optional)</label>
      <input id="exclude" autocomplete="off">
      <div class="actions">
        <button id="btn-plan">Plan</button>
        <button id="btn-start" class="primary" disabled>Start</button>
        <button id="btn-cancel" disabled>Cancel</button>
        <button id="btn-resume" disabled>Copy resume command</button>
      </div>
      <pre id="plan" hidden></pre>
    </fieldset>

    <fieldset>
      <legend>Findings <span id="fcount" class="muted"></span></legend>
      <div class="row">
        <div><label for="fstatus">Status</label><input id="fstatus" autocomplete="off"></div>
        <div><label for="fseverity">Severity</label>
          <select id="fseverity"><option value="">any</option></select></div>
        <div><label for="fq">Search</label><input id="fq" autocomplete="off"></div>
      </div>
      <table>
        <thead><tr><th>Sev</th><th>Title</th><th>File</th><th>Status</th>
          <th>Score</th><th></th></tr></thead>
        <tbody id="fbody"></tbody>
      </table>
    </fieldset>
  </section>

  <section id="right">
    <fieldset style="height:100%; margin:0; display:flex; flex-direction:column;">
      <legend>Transcript</legend>
      <pre id="transcript" style="flex:1;"></pre>
    </fieldset>
  </section>
</main>
<script>
const CONFIG = __CONFIG__;
const $ = (id) => document.getElementById(id);
let offset = 0, planFields = null, pollTimer = null;

function banner(text, cls) {
  const el = $('banner');
  el.textContent = text;
  el.className = cls || 'muted';
}

async function api(path, opts) {
  const res = await fetch(path, Object.assign({headers: {'X-Mantis-UI': '1'}}, opts || {}));
  const body = await res.json().catch(() => ({ok: false, error: 'bad response'}));
  return body;
}

function fillSelect(el, values, blankLabel) {
  el.replaceChildren();
  if (blankLabel !== undefined) {
    const o = document.createElement('option');
    o.value = ''; o.textContent = blankLabel; el.appendChild(o);
  }
  for (const v of values) {
    const o = document.createElement('option');
    o.value = v; o.textContent = v; el.appendChild(o);
  }
}

function formFields() {
  const exclude = $('exclude').value.split(',').map(s => s.trim()).filter(Boolean);
  const tokens = $('max_tokens').value.trim();
  return {
    target: $('target').value.trim(),
    scan_mode: $('scan_mode').value,
    sandbox: $('sandbox').value,
    model: $('model').value.trim(),
    focus: $('focus').value.trim(),
    exclude: exclude,
    max_tokens: tokens === '' ? null : tokens
  };
}

async function refreshState() {
  const s = await api('/api/state');
  $('db').textContent = s.db_present ? s.db_path : 'no database yet';
  const run = s.run;
  if (run && s.running) {
    $('runstate').textContent = 'running ' + run.run_id + ' · ' + run.elapsed_seconds + 's';
    $('runstate').className = 'ok';
    $('btn-cancel').disabled = false;
    $('btn-start').disabled = true;
    $('btn-resume').disabled = true;
  } else if (run) {
    $('runstate').textContent = 'finished ' + run.run_id + ' · exit ' + run.exit_code;
    $('runstate').className = run.exit_code === 0 ? 'ok' : 'bad';
    $('btn-cancel').disabled = true;
    $('btn-start').disabled = !planFields;
    $('btn-resume').disabled = false;
  } else {
    $('runstate').textContent = 'no run';
    $('runstate').className = 'muted';
    $('btn-cancel').disabled = true;
    $('btn-start').disabled = !planFields;
    $('btn-resume').disabled = true;
  }
}

async function planRun() {
  banner('Planning — launch.py is computing the real plan…');
  const res = await api('/api/plan', {method: 'POST', body: JSON.stringify(formFields())});
  const pre = $('plan');
  pre.hidden = false;
  pre.textContent = res.plan || res.error || '';
  if (res.ok) {
    planFields = res.fields;
    banner('Plan ready. Start runs exactly this.');
    $('btn-start').disabled = false;
  } else {
    planFields = null;
    banner('Plan refused: ' + (res.error || 'unknown error'), 'bad');
    $('btn-start').disabled = true;
  }
}

async function startRun() {
  if (!planFields) return;
  banner('Starting…');
  const res = await api('/api/runs', {method: 'POST', body: JSON.stringify(planFields)});
  if (!res.ok) { banner('Refused: ' + res.error, 'bad'); return; }
  banner('Run ' + res.run.run.run_id + ' started.');
  offset = 0;
  $('transcript').textContent = '';
  refreshState();
  pollTranscript();
}

async function cancelRun() {
  banner('Cancelling — SIGINT so the run checkpoints…');
  const res = await api('/api/runs/cancel', {method: 'POST', body: '{}'});
  banner(res.ok ? 'Cancelled. ' + (res.resume_command ? 'Resume: ' + res.resume_command : '')
                : 'Cancel failed: ' + res.error, res.ok ? 'muted' : 'bad');
  refreshState();
}

async function copyResume() {
  const s = await api('/api/state');
  const cmd = s.run && s.run.resume_command;
  if (!cmd) return;
  try { await navigator.clipboard.writeText(cmd); banner('Copied: ' + cmd, 'ok'); }
  catch (e) { banner('Resume command: ' + cmd); }
}

function cell(text, cls) {
  const td = document.createElement('td');
  if (cls) td.className = cls;
  td.textContent = text === null || text === undefined ? '' : String(text);
  return td;
}

async function loadFindings() {
  const params = new URLSearchParams({
    status: $('fstatus').value.trim(),
    severity: $('fseverity').value,
    q: $('fq').value.trim()
  });
  const res = await api('/api/findings?' + params.toString());
  const body = $('fbody');
  body.replaceChildren();
  if (!res.ok) { $('fcount').textContent = ''; return; }
  $('fcount').textContent = res.count + ' shown';
  for (const f of res.findings) {
    const tr = document.createElement('tr');
    const sev = (f.severity || '').toUpperCase();
    tr.appendChild(cell(sev, 'sev ' + sev));
    const title = cell(f.title || '(untitled)');
    if (f.description) title.title = f.description;
    tr.appendChild(title);
    tr.appendChild(cell(f.filepath || '', 'path'));
    tr.appendChild(cell(f.status || '', 'st'));
    tr.appendChild(cell(f.score === null || f.score === undefined ? '' : f.score));
    const act = document.createElement('td');
    act.className = 'act';
    const b = document.createElement('button');
    b.textContent = 'false +';
    b.title = 'Dismiss as a false positive (reason required)';
    b.onclick = () => resolveFinding(f, 'false_positive');
    act.appendChild(b);
    tr.appendChild(act);
    body.appendChild(tr);
  }
}

async function resolveFinding(f, resolution) {
  const reason = prompt('Reason for marking finding #' + f.id + ' ' + resolution +
                        ' (required — it is the audit trail):');
  if (reason === null) return;
  const res = await api('/api/findings/resolve', {
    method: 'POST',
    body: JSON.stringify({finding_id: f.id, resolution: resolution, reason: reason})
  });
  if (res.error) { banner('Refused: ' + res.error, 'bad'); return; }
  banner(res.applied ? ('#' + f.id + ' → ' + res.new_status)
                     : ('#' + f.id + ' unchanged: ' + (res.note || 'guard refused')),
         res.applied ? 'ok' : 'bad');
  loadFindings();
}

async function pollTranscript() {
  clearTimeout(pollTimer);
  const res = await api('/api/transcript?offset=' + offset);
  if (res.present && res.rows.length) {
    offset = res.offset;
    const pre = $('transcript');
    for (const row of res.rows) {
      const who = row.node || row.author || '';
      const kind = row.type || '';
      let text = '';
      if (row.type === 'tool_call') text = row.name + '(' + (row.args || '') + ')';
      else if (row.type === 'tool_response') text = row.name + ' → ' + (row.response || '');
      else if (row.type === 'error') text = 'ERROR ' + (row.name || '') + ': ' + (row.text || '');
      else text = row.text || '';
      pre.textContent += '[' + kind + '] ' + who + '  ' + text + '\n';
    }
    pre.scrollTop = pre.scrollHeight;
  }
  refreshState();
  pollTimer = setTimeout(pollTranscript, 2000);
}

async function init() {
  fillSelect($('scan_mode'), CONFIG.scan_modes);
  fillSelect($('sandbox'), CONFIG.sandboxes, '(configured)');
  fillSelect($('fseverity'), CONFIG.severities, 'any');
  $('btn-plan').onclick = planRun;
  $('btn-start').onclick = startRun;
  $('btn-cancel').onclick = cancelRun;
  $('btn-resume').onclick = copyResume;
  for (const id of ['fstatus', 'fseverity', 'fq']) {
    $(id).addEventListener('input', loadFindings);
  }
  await refreshState();
  await loadFindings();
  pollTranscript();
}
init();
</script>
</body>
</html>
"""


def _page(app: App) -> str:
    config = {
        "scan_modes": list(SCAN_MODES),
        "sandboxes": list(SANDBOXES),
        "severities": list(_SEVERITIES),
    }
    # json.dumps escapes </script> as \u003c, so the config cannot close the tag.
    return PAGE.replace("__CONFIG__", json.dumps(config))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local UI: watch a Mantis run, triage findings, start one.")
    parser.add_argument("--db", default="knowledge.db",
                        help="knowledge database (default: knowledge.db)")
    parser.add_argument("--allow-root", default=".",
                        help="only targets inside this directory may be scanned "
                             "(default: current directory)")
    parser.add_argument("--state-dir", default="",
                        help="UI state: run logs, transcripts, resolutions.log "
                             "(default: <db parent>/ui-state)")
    parser.add_argument("--transcript", default="",
                        help="tail an existing transcript JSONL when no supervised "
                             "run is active")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765,
                        help="bind port (default: 8765)")
    parser.add_argument("--token", default="",
                        help="required when --host is not loopback; compared in "
                             "constant time")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if not loopback and not args.token:
        print("Refusing to bind a non-loopback address without --token: the UI "
              "has no other authentication.", file=sys.stderr)
        return 2
    if not args.token:
        print("No --token: this UI is reachable by any local process. Set one "
              "if that matters.", file=sys.stderr)

    allow_root = Path(args.allow_root).resolve()
    if not allow_root.is_dir():
        print(f"--allow-root '{allow_root}' is not a directory.", file=sys.stderr)
        return 2
    db_path = str(Path(args.db).resolve())
    state_dir = (Path(args.state_dir).resolve() if args.state_dir
                 else Path(db_path).parent / "ui-state")
    launch_py = Path(__file__).resolve().parent / "launch.py"
    if not launch_py.is_file():
        print(f"launch.py not found next to ui.py ({launch_py}).", file=sys.stderr)
        return 2

    supervisor = Supervisor(allow_root=allow_root, launch_py=launch_py,
                            state_dir=state_dir, db_path=db_path)
    app = App(db_path=db_path, allow_root=allow_root, state_dir=state_dir,
              supervisor=supervisor, token=args.token,
              transcript=args.transcript, host=args.host)

    Handler.app = app
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        print(f"Cannot bind {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1
    url = f"http://{args.host}:{args.port}/"
    if args.token:
        url += "?token=" + urllib.parse.quote(args.token)
    print(f"Mantis UI on {url}")
    print(f"  database:   {db_path}")
    print(f"  allow-root: {allow_root}")
    print(f"  state:      {state_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
        if supervisor.current is not None:
            supervisor.current.refresh()
            if supervisor.current.running():
                print(f"A run is still going: {supervisor.current.run_id}. "
                      f"Cancel it from the UI or kill the process group.")
        supervisor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
