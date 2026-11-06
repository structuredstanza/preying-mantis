# Mantis MCP Server: a secure development loop

`scripts/mcp_server.py` turns Mantis from a batch scanner into a live sidecar
for any MCP-capable coding agent (Gemini CLI, Claude Code, Cursor, Jetski, or
anything else speaking the Model Context Protocol over stdio). The intended
workflow is a loop that runs alongside ordinary development:

1. The agent edits code as usual.
2. For every change, it calls `mantis_check_change` with the unified diff: an
   instant, deterministic verdict built from everything Mantis already knows
   about the repository — open findings, the tree-sitter call graph, and scan
   coverage.
3. It resolves whatever the gate raises, and calls `mantis_scan_change` so the
   full multi-agent audit pipeline reviews the touched files in parallel,
   budget-capped, in the background.
4. The next `mantis_check_change` sees whatever that scan found, because both
   read the same per-repository knowledge database (see State).

The server itself **never calls an LLM**. Tier-1 tools are deterministic reads
over the structural catalog and the findings database; the Tier-2 scan spawns
the existing pipeline (`main.py`) as a subprocess that owns its own budget.

## Setup

The SDK dependency (`mcp`) is part of `requirements.txt`, so a normal install
covers it:

```bash
cd reference && ./install.sh
python3 scripts/configure.py --auto   # needed for the scan tier only
```

The deterministic gate works with no model configured at all. Configuration is
only consulted when `mantis_scan_change` launches the pipeline.

Start the server by hand to sanity-check it:

```bash
.venv/bin/python scripts/mcp_server.py --repo /path/to/your/repo
```

`--repo` defaults to the current directory (or `MANTIS_MCP_REPO`).

### Client configuration

Every client takes the same three facts: the venv's python, the server script,
and the repository to serve. Replace `/opt/mantis/reference` with your checkout.

**Gemini CLI** (`~/.gemini/settings.json`) and **Cursor** (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "mantis": {
      "command": "/opt/mantis/reference/.venv/bin/python",
      "args": [
        "/opt/mantis/reference/scripts/mcp_server.py",
        "--repo", "/path/to/your/repo"
      ]
    }
  }
}
```

**Claude Code**:

```bash
claude mcp add mantis -- /opt/mantis/reference/.venv/bin/python \
    /opt/mantis/reference/scripts/mcp_server.py --repo .
```

## Tools

| Tool                       | Tier | What it answers                                                                     |
| -------------------------- | ---- | ----------------------------------------------------------------------------------- |
| `mantis_check_change`      | 1    | The gate: PASS/REVIEW/BLOCK for a diff or file list, with reasons and blast radius. |
| `mantis_scan_change`       | 2    | Launches the full audit pipeline over the changed files in the background.          |
| `mantis_scan_status`       | 1    | Progress of a background scan, and findings recorded in its files so far.           |
| `mantis_reindex`           | 1    | Builds or incrementally refreshes the tree-sitter structural index.                 |
| `mantis_find_symbol`       | 1    | Where a function/class/macro is defined.                                            |
| `mantis_find_callers`      | 1    | Who can reach this code.                                                            |
| `mantis_find_callees`      | 1    | What this code reaches.                                                             |
| `mantis_function_at`       | 1    | The innermost function enclosing file:line, with its exact extent.                  |
| `mantis_get_findings`      | 1    | Recorded findings, filtered by file or status.                                      |
| `mantis_resolve_finding`   | 1    | Operator closure: mark a finding `false_positive`, with a reason.                   |
| `mantis_security_guidance` | 1    | Threat-model-informed guidance for a file: active threats, invariants, history.     |

## The gate

`mantis_check_change` returns one of three verdicts:

- **BLOCK** — an open HIGH or CRITICAL finding intersects the changed hunks or a
  modified function in a changed file (`relationship: "direct"`). Fix it, or
  mark it `false_positive` with evidence, before merging.
- **REVIEW** — something demands attention: open findings of any severity touch
  the change, sit elsewhere in a changed file outside the modified
  lines/functions (`relationship: "unrelated_in_file"`, pre-existing file debt),
  or sit in a direct caller (`relationship: "caller_radius"`); a changed file
  has never been covered by any recorded scan; or the index/database needed to
  decide is missing.
- **PASS** — every check ran and nothing known intersects the change.

Two properties are deliberate and load-bearing:

1. **The gate fails closed.** A missing structural index, a missing findings
   database, an unreadable coverage ledger, or an empty change set each yield
   REVIEW with instructions — never PASS. PASS means "every check ran and found
   nothing", not "secure": the parallel scan tier is what inspects new code, and
   the gate is what guarantees its results are never ignored.
2. **Read paths write nothing.** `sqlite3.connect()` creates files, so every
   read-only tool checks existence before opening. Only `mantis_reindex` and
   `mantis_scan_change` create the state directory; nothing the server does
   writes into the repository, ever (see State).

The blast radius is computed from the diff's changed lines: each line maps to
its enclosing function in the structural index, and each function to its direct
callers, so a finding in a *caller* of changed code surfaces at REVIEW even when
the finding's own file is untouched.

## Pre-commit and CI: `--check`

The same gate runs one-shot without an MCP client. It reads a unified diff on
stdin (or takes `--files`), prints the verdict JSON, and encodes the verdict in
its exit code: `0` PASS, `1` REVIEW, `2` BLOCK (`3` is a usage error).

```bash
# Pre-commit hook: block on BLOCK, allow REVIEW through with its reasons shown.
git diff --cached | .venv/bin/python scripts/mcp_server.py --repo . --check
test $? -lt 2

# CI: fail the job on anything but PASS.
git diff origin/main...HEAD | .venv/bin/python scripts/mcp_server.py --repo . --check
```

## Closing findings

`mantis_resolve_finding` is how a disputed finding stops gating:

- **`false_positive`** — the operator disputes the finding. The database's
  monotonic status guard refuses this over `dynamic_confirmed`: a dismissal
  never erases machine evidence. The result reports `applied: false` with the
  reason.

A resolution requires a non-empty `reason`, and every attempt is appended to
`resolutions.log` in the state directory for audit. Machine statuses
(`static_confirmed`, `dynamic_confirmed`) are reserved for the pipeline — the
server never lets an operator claim them. Mantis reports defects; it does not
adjudicate fixes, so there is no "mitigated" resolution.

## Scan lifecycle

`mantis_scan_status` (pass `wait_seconds`, up to `60`, to long-poll instead of
spinning on turns) reports `running`, then one of: `done` (all files clean
exit), `paused_at_budget` (the pipeline hit its LLM-call budget and paused
gracefully — findings written so far are already in the database and the run is
resumable), or `finished_with_errors`. The record also carries `index_restored`:
after the pipeline finishes, the server rebuilds the shared structural index
rooted at the repository, because the per-file scan re-roots it at the scanned
file, which would otherwise blind the gate's blast radius.

Scans pass `--path-root` so findings store repository-relative filepaths
(`routes/login.ts`, not the ambiguous basename `login.ts` that two same-named
files could cross-match).

## State

Everything lives in a per-repository directory **outside the repository**:
`<workspace>/mcp/<hash of the repository path>/` by default, or a directory
under `MANTIS_MCP_STATE_DIR`, or exactly `--state-dir`. It holds `knowledge.db`
(the same schema every `main.py` run writes and the MCP tools read),
`knowledge.structural_index/` (the catalog), `scans/*.log` (pipeline output per
background scan) and `resolutions.log` (the audit trail of
`mantis_resolve_finding`). Scan registries are in-memory — a `scan_id` does not
survive a server restart, but every finding a scan wrote does, which is the part
that matters: the gate reads the database, not the registry.

The state used to live in `.mantis/` at the repository root. It moved because
the repository is the code under audit, and in general it is somebody else's: a
committed symlink at a predictable path under `.mantis/` turned every state
write into a write to any file the server's user could reach, and a committed
`knowledge.db` or catalog was trusted as the server's own memory — enough to
hand `--check` a PASS on a never-scanned file. A state directory inside the
repository is refused outright. An old `.mantis/` in a checkout is simply
ignored; delete it.

This also means a fresh checkout — a CI runner, a new clone — starts with no
state, so `--check` there fails closed with REVIEW until a scan has run in that
environment. That is the intended answer: the gate reports what *this*
installation has audited, and a pull request cannot supply the audit.
