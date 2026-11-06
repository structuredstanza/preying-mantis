# Local UI

`scripts/ui.py` is one local web page for one `knowledge.db`. It does three
things and nothing else:

- **watch** a run — tails the transcript JSONL that `main.py --transcript-out`
  already writes and flushes per row
- **triage** findings — dismisses a finding as `false_positive` with a required
  reason. The page offers only that control, and the endpoint accepts exactly
  what `mantis_resolve_finding` accepts — the same one verdict, so an API caller
  is not working with a reduced capability.
- **launch** a run — shows the launcher's own dry-run plan, then supervises the
  real process

It adds no data layer. Findings come from `core/database.py`, the verdict goes
through `core/triage.py` (the same path `mantis_resolve_finding` uses), and the
run is `scripts/launch.py` itself as a child process.

## Run it

```bash
cd reference
.venv/bin/python scripts/ui.py --db knowledge.db --allow-root /path/to/code
```

Open the printed URL. Useful flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--db` | `knowledge.db` | database to read and triage |
| `--allow-root` | current directory | only targets inside this directory may be launched |
| `--state-dir` | `<db dir>/ui-state` | run logs, transcripts, `resolutions.log` |
| `--transcript` | — | tail an existing transcript when no supervised run is active |
| `--host` / `--port` | `127.0.0.1` / `8765` | bind address |
| `--token` | — | required for any non-loopback bind |

## Launch is supervised, not reimplemented

The UI never parses budgets, runs preflight, or resolves a scan mode. It runs
`launch.py --dry-run`, shows the plan that command printed, and on confirmation
spawns the same argv without `--dry-run`. The plan the operator approves is
therefore the plan the launcher itself computed.

Only a fixed field whitelist crosses the boundary: `target`, `scan_mode`,
`sandbox`, `model`, `api_base`, `focus`, `max_tokens`, `exclude`, `resume`.
Everything else in a request body is dropped, and argv is always a list — never
a shell string.

`scan_mode: changed` is not offered: it needs a caller-supplied
`--changed-files` path, which would reopen the jail the target check exists to
hold. Use the CLI for a change-scoped run.

One run at a time per UI. A second start is refused while the first is live,
because two top-level runs would fight the same SQLite write lock.

Cancel sends `SIGINT` to the run's process group first, so `BudgetController`
checkpoints and the run stays `--resume`-able, then `SIGTERM`, then `SIGKILL`.
The UI shows the resume command to copy.

## Security model

The UI has one write path into a scan engine, so the controls are explicit:

- **Loopback by default.** A non-loopback `--host` is refused unless `--token`
  is set; the token is compared with `hmac.compare_digest`.
- **Host allowlist.** A request whose `Host` header is not the bound host or a
  loopback name is answered `421`, which closes DNS rebinding against a local
  server.
- **CSRF.** Every mutating request must carry `X-Mantis-UI: 1`. A cross-origin
  HTML form cannot set a custom header, and this server never answers a CORS
  preflight.
- **Target jail.** The target is validated by `core.paths.validate_scan_target`
  (no `..`, no symlinked leaf, no escaping intermediate symlink) and must sit
  inside `--allow-root`.
- **Escaping.** Every finding field was written by a model reading a hostile
  target. Values reach the page only as JSON, and the client assigns them with
  `textContent` — never `innerHTML`.
- **Audit.** A resolution appends one JSON line to
  `<state-dir>/resolutions.log`, and the database's monotonic guard still
  refuses a dismissal over `dynamic_confirmed`.

## Routes

| Route | Method | Purpose |
| --- | --- | --- |
| `/` | GET | the page |
| `/api/state` | GET | database presence, allowed root, run status |
| `/api/findings` | GET | findings, filtered by `status`, `severity`, `q`, `run_id` |
| `/api/transcript` | GET | rows since `offset` (byte offset, safe against a half-written row) |
| `/api/plan` | POST | runs `launch.py --dry-run`, returns its output |
| `/api/runs` | POST | spawns the run |
| `/api/runs/cancel` | POST | interrupts the run's process group |
| `/api/findings/resolve` | POST | `{finding_id, resolution, reason}` |

## What it does not do

No authentication beyond one shared token, no multi-user, no editing finding
text, no charts, no websockets, no scheduling, no changing `workflow.json`.
Model, sandbox and budget are per-run overrides only.
