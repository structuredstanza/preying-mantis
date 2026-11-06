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

"""The UI's HTTP surface, exercised over a real socket.

Findings are written by a model reading a hostile target, so a finding title is
untrusted input rendered into a page. These tests pin that it stays data: the
page carries no finding text at all, the API returns it as JSON, and the
client-side renderer only ever assigns `textContent`. They also pin the three
controls that keep the surface local: Host allowlisting, a token, and the
custom header a cross-origin form cannot set.
"""

import http.client
import json
import shutil
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from core import runsupervisor
from core.runsupervisor import Supervisor
from scripts import ui

HOSTILE = "</script><script>alert('pwn')</script>"


def _make_server(app):
    """A Handler subclass per app, so two servers cannot share one config."""

    class Handler(ui.Handler):
        pass

    Handler.app = app
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class UiTestBase(unittest.TestCase):
    token = ""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mantis_ui_http_")).resolve()
        self.root = self.tmp / "root"
        self.root.mkdir()
        (self.root / "code").mkdir()
        self.db = str(self.tmp / "knowledge.db")
        from core.database import init_db, write_findings

        init_db(self.db)
        write_findings(self.db, "code/app.py", [{
            "title": "Open command injection",
            "severity": "HIGH",
            "description": "shell=True on request input",
            "cwe": "CWE-78",
            "status": "static_confirmed",
        }], run_id="run-1")
        write_findings(self.db, "code/app.py", [{
            "title": HOSTILE,
            "severity": "CRITICAL",
            "description": "hostile title from the target",
            "cwe": "CWE-78",
            "status": "dynamic_confirmed",
        }], run_id="run-1")

        self.sup = Supervisor(allow_root=self.root,
                              launch_py=self.tmp / "launch.py",
                              state_dir=self.tmp / "state", db_path=self.db)
        self.app = ui.App(db_path=self.db, allow_root=self.root,
                          state_dir=self.tmp / "state", supervisor=self.sup,
                          token=self.token, host="127.0.0.1")
        self.server = _make_server(self.app)
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def call(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        hdrs = dict(headers or {})
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
            hdrs.setdefault("Content-Length", str(len(data)))
        try:
            conn.request(method, path, body=data, headers=hdrs)
            resp = conn.getresponse()
            raw = resp.read()
            return resp.status, raw, dict(resp.getheaders())
        finally:
            conn.close()


class PageTests(UiTestBase):
    def test_the_page_carries_no_finding_text(self):
        status, raw, _ = self.call("GET", "/")
        self.assertEqual(status, 200)
        page = raw.decode("utf-8")
        self.assertIn("Mantis", page)
        # Findings arrive as JSON and are rendered with textContent, so a
        # hostile title must never appear in the document itself.
        self.assertNotIn("Open command injection", page)
        self.assertNotIn(HOSTILE, page)

    def test_the_page_embeds_config_as_escaped_json(self):
        status, raw, _ = self.call("GET", "/")
        self.assertEqual(status, 200)
        page = raw.decode("utf-8")
        self.assertIn("const CONFIG = {", page)
        # No finding data is interpolated into the document, so nothing from the
        # target can appear in a script context at all.
        self.assertNotIn("alert('pwn')", page)

    def test_findings_are_served_as_json_with_the_hostile_value_intact(self):
        status, raw, headers = self.call("GET", "/api/findings")
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers.get("Content-Type", ""))
        payload = json.loads(raw)
        titles = [f["title"] for f in payload["findings"]]
        self.assertIn(HOSTILE, titles)

    def test_state_reports_the_database_and_the_modes(self):
        status, raw, _ = self.call("GET", "/api/state")
        self.assertEqual(status, 200)
        payload = json.loads(raw)
        self.assertTrue(payload["db_present"])
        self.assertNotIn("changed", payload["scan_modes"])
        self.assertEqual(payload["sandboxes"],
                         ["static-only", "gvisor", "microsandbox", "gce"])


class TransportControlTests(UiTestBase):
    def test_a_foreign_host_header_is_refused(self):
        status, _, _ = self.call("GET", "/api/state", headers={"Host": "evil.example"})
        self.assertEqual(status, 421)

    def test_a_mutating_request_without_the_custom_header_is_refused(self):
        status, raw, _ = self.call("POST", "/api/runs/cancel", body={})
        self.assertEqual(status, 403)
        self.assertIn("X-Mantis-UI", json.loads(raw)["error"])

    def test_unknown_routes_are_404(self):
        status, _, _ = self.call("GET", "/api/nope")
        self.assertEqual(status, 404)


class TokenTests(UiTestBase):
    token = "s3cret-token"

    def test_a_request_without_the_token_is_refused(self):
        status, _, _ = self.call("GET", "/api/state")
        self.assertEqual(status, 401)

    def test_the_token_in_the_url_sets_a_strict_cookie(self):
        status, _, headers = self.call("GET", "/?token=s3cret-token")
        self.assertEqual(status, 200)
        cookie = headers.get("Set-Cookie", "")
        self.assertIn("mantis_token=", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)

    def test_a_header_token_is_accepted(self):
        status, _, _ = self.call("GET", "/api/state",
                                 headers={"X-Mantis-Token": "s3cret-token"})
        self.assertEqual(status, 200)

    def test_a_wrong_token_is_refused(self):
        status, _, _ = self.call("GET", "/api/state",
                                 headers={"X-Mantis-Token": "wrong"})
        self.assertEqual(status, 401)

    def test_the_cookie_alone_authenticates_a_follow_up_request(self):
        """The page sets the cookie so the operator never pastes the token
        again; a broken cookie branch would leave every later request 401."""
        status, _, headers = self.call("GET", "/?token=s3cret-token")
        self.assertEqual(status, 200)
        cookie = headers["Set-Cookie"].split(";")[0]
        self.assertNotIn("X-Mantis-Token", cookie)
        status, _, _ = self.call("GET", "/api/state", headers={"Cookie": cookie})
        self.assertEqual(status, 200)


class ResolveTests(UiTestBase):
    def _post(self, body):
        return self.call("POST", "/api/findings/resolve", body=body,
                         headers={"X-Mantis-UI": "1"})

    def _findings(self):
        status, raw, _ = self.call("GET", "/api/findings")
        self.assertEqual(status, 200)
        return json.loads(raw)["findings"]

    def test_dismissing_an_open_finding_applies_and_audits(self):
        target = next(f for f in self._findings() if f["status"] == "static_confirmed")
        status, raw, _ = self._post({"finding_id": target["id"],
                                     "resolution": "false_positive",
                                     "reason": "guarded by the upstream proxy"})
        self.assertEqual(status, 200, raw)
        payload = json.loads(raw)
        self.assertTrue(payload["applied"], payload)
        self.assertEqual(payload["new_status"], "false_positive")
        log = (self.tmp / "state" / "resolutions.log").read_text()
        self.assertIn("guarded by the upstream proxy", log)
        self.assertIn(str(target["id"]), log)

    def test_the_api_matches_the_mcp_tool_capability(self):
        """The endpoint accepts exactly what `mantis_resolve_finding` accepts.

        Both go through core/triage, so a verdict one surface refuses the other
        refuses too — a UI that silently supported more than the MCP tool would
        be a difference nobody would think to check.
        """
        target = next(f for f in self._findings() if f["status"] == "static_confirmed")
        for resolution in ("mitigated", "patch_verified"):
            with self.subTest(resolution=resolution):
                status, raw, _ = self._post({"finding_id": target["id"],
                                             "resolution": resolution,
                                             "reason": "trying it"})
                self.assertEqual(status, 400, raw)
                self.assertIn("false_positive", json.loads(raw)["error"])

    def test_the_page_renders_no_mitigate_control(self):
        status, raw, _ = self.call("GET", "/")
        self.assertEqual(status, 200)
        page = raw.decode("utf-8")
        self.assertNotIn("'mitigate'", page)
        self.assertNotIn("['mitigate'", page)

    def test_a_dismissal_never_overwrites_machine_evidence(self):
        target = next(f for f in self._findings() if f["status"] == "dynamic_confirmed")
        status, raw, _ = self._post({"finding_id": target["id"],
                                     "resolution": "false_positive",
                                     "reason": "I disagree"})
        self.assertEqual(status, 200, raw)
        payload = json.loads(raw)
        self.assertFalse(payload["applied"], payload)
        self.assertIn("monotonic", payload["note"])
        self.assertEqual(payload["new_status"], "dynamic_confirmed")

    def test_an_empty_reason_is_refused(self):
        target = self._findings()[0]
        status, raw, _ = self._post({"finding_id": target["id"],
                                     "resolution": "false_positive", "reason": "   "})
        self.assertEqual(status, 400, raw)
        self.assertIn("reason", json.loads(raw)["error"])

    def test_a_machine_status_cannot_be_claimed_from_the_ui(self):
        target = self._findings()[0]
        status, raw, _ = self._post({"finding_id": target["id"],
                                     "resolution": "dynamic_confirmed",
                                     "reason": "trust me"})
        self.assertEqual(status, 400, raw)
        self.assertIn("resolution", json.loads(raw)["error"])


class TranscriptTests(UiTestBase):
    def test_offset_resumes_without_repeating_or_dropping_rows(self):
        path = self.tmp / "transcript.jsonl"
        path.write_text(json.dumps({"type": "text", "text": "one"}) + "\n")
        self.app.transcript = str(path)

        status, raw, _ = self.call("GET", "/api/transcript?offset=0")
        self.assertEqual(status, 200)
        first = json.loads(raw)
        self.assertEqual(len(first["rows"]), 1)
        self.assertEqual(first["rows"][0]["text"], "one")

        status, raw, _ = self.call(f"GET", f"/api/transcript?offset={first['offset']}")
        self.assertEqual(json.loads(raw)["rows"], [])

        with path.open("a") as fh:
            fh.write(json.dumps({"type": "text", "text": "two"}) + "\n")
        status, raw, _ = self.call(f"GET", f"/api/transcript?offset={first['offset']}")
        rows = json.loads(raw)["rows"]
        self.assertEqual([r["text"] for r in rows], ["two"])

    def test_a_half_written_row_is_not_consumed(self):
        path = self.tmp / "transcript.jsonl"
        path.write_text('{"type": "text", "text": "complete"}\n{"type": "te')
        self.app.transcript = str(path)

        status, raw, _ = self.call("GET", "/api/transcript?offset=0")
        payload = json.loads(raw)
        self.assertEqual([r["text"] for r in payload["rows"]], ["complete"])

        # Finish the row; the offset left before it must let the next poll see it.
        with path.open("a") as fh:
            fh.write('xt", "text": "finished"}\n')
        status, raw, _ = self.call(f"GET", f"/api/transcript?offset={payload['offset']}")
        self.assertEqual([r["text"] for r in json.loads(raw)["rows"]], ["finished"])

    def test_no_transcript_yet_is_an_empty_answer_not_an_error(self):
        status, raw, _ = self.call("GET", "/api/transcript?offset=0")
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(raw)["present"])

class LaunchRouteTests(UiTestBase):
    def test_a_start_succeeds_and_the_resume_command_keeps_the_database(self):
        """The refusal branches below are not the whole route: a real spawn
        must answer 200, and the command the UI hands back for a resume must
        carry this run's own database and overrides. A hand-assembled resume
        command dropped --db, so it resumed "latest" in the installation
        database instead of the one the operator opened."""
        stub = self.tmp / "launch.py"
        stub.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
        self.sup.launch_py = stub
        status, raw, _ = self.call("POST", "/api/runs",
                                   body={"target": str(self.root / "code"),
                                         "sandbox": "static-only"},
                                   headers={"X-Mantis-UI": "1"})
        self.assertEqual(status, 200)
        payload = json.loads(raw)
        self.assertTrue(payload["ok"], payload)
        run = payload["run"]["run"]
        self.assertIn("--db", run["argv"])
        self.assertIn("--sandbox", run["argv"])
        self.assertIn("--db", run["resume_command"])
        self.assertIn(str(self.db), run["resume_command"])
        self.assertIn("--sandbox", run["resume_command"])
        self.assertIn("--resume", run["resume_command"])
        # The UI's own transcript destination belongs to this run, not to the
        # resume: a resumed run writes its own.
        self.assertNotIn("--transcript-out", run["resume_command"])

    def test_start_refuses_a_target_outside_the_root(self):
        elsewhere = self.tmp / "elsewhere"
        elsewhere.mkdir()
        status, raw, _ = self.call("POST", "/api/runs",
                                   body={"target": str(elsewhere)},
                                   headers={"X-Mantis-UI": "1"})
        self.assertEqual(status, 409)
        self.assertIn("outside the allowed root", json.loads(raw)["error"])

    def test_plan_surfaces_the_launchers_refusal(self):
        from unittest import mock

        class _Done:
            returncode = 1
            stdout = ""
            stderr = "Preflight Verification Failed: no sandbox\n"

        with mock.patch.object(runsupervisor.subprocess, "run", return_value=_Done()):
            status, raw, _ = self.call("POST", "/api/plan",
                                       body={"target": str(self.root / "code")},
                                       headers={"X-Mantis-UI": "1"})
        self.assertEqual(status, 400)
        payload = json.loads(raw)
        self.assertFalse(payload["ok"])
        self.assertIn("Preflight", payload["error"])


class HostAllowlistTests(UiTestBase):
    def test_a_wildcard_bind_accepts_the_address_the_browser_dialled(self):
        """`--host 0.0.0.0` is the documented remote mode, and a browser sends
        the address it connected to, never the wildcard: a fixed allowlist
        answered 421 to the page itself, making that mode unusable."""
        self.app.host = "0.0.0.0"
        status, _, _ = self.call("GET", "/api/state",
                                 headers={"Host": "192.168.1.5:8765"})
        self.assertEqual(status, 200)

    def test_a_loopback_bind_still_refuses_a_foreign_host(self):
        """DNS rebinding: a page on attacker.example resolving to 127.0.0.1
        still sends its own name, and that must not reach the API."""
        status, _, _ = self.call("GET", "/api/state",
                                 headers={"Host": "attacker.example"})
        self.assertEqual(status, 421)


class BindSafetyTests(unittest.TestCase):
    def test_a_non_loopback_bind_without_a_token_is_refused(self):
        self.assertEqual(ui.main(["--host", "0.0.0.0", "--port", "0"]), 2)


if __name__ == "__main__":
    unittest.main()
