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

"""Leak tripwires and reflection parsing for the Mantis ADK reference implementation.

Adapted from open-source google-research/harness2 (Apache-2.0):
  - `assert_no_leak_paths`, `assert_no_grader_verdicts`, and
    `assert_tree_has_no_grader_verdicts`: fail-closed tripwires preventing
    benchmark ground-truth files or grader verdicts from entering model context.
  - `parse_reflection_blocks`: extracts structured `<experience>` lessons
    (`cue`, `lesson`, `evidence`, `confidence`, `target_stage`) from reflector
    output. Lessons without evidence are dropped: an invented attribution
    poisons memory for every later run. A bare `NO_ENTRY` line is the
    reflector's explicit "nothing citable this run" and is reported as such
    rather than as a parse that found nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import fnmatch
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_LEAK_GLOBS: tuple[str, ...] = (
    "*ground_truth*.json",
    "*synthetic_dataset*.json",
    "*review_dataset*.json",
    "*critic_dataset*.json",
    "*calibrate_dataset*.json",
)

GRADER_VERDICT_MARKERS: tuple[str, ...] = (
    "deterministic_eval_summary",
    "deterministic_eval_result",
    "judge_scope_and_deterministic_status",
    "Judge findings on the rendered result",
    '"ground_truth_total"',
    '"cross_file_recall"',
    '"unmatched_findings"',
)

_MARKER_SCAN_SUFFIXES: frozenset[str] = frozenset({
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
    ".html",
    ".htm",
    ".log",
    ".py",
    ".js",
    ".sh",
    ".xml",
    ".ini",
    ".toml",
    ".patch",
    ".diff",
    "",
})

_EXP_RE = re.compile(
    r"<experience>(.*?)</experience>", re.DOTALL | re.IGNORECASE
)

# The reflector's explicit "this run taught nothing citable" answer: the token
# alone on a line, case-insensitive, tolerating the backticks or emphasis a
# model tends to wrap a bare keyword in. `NO_ENTRY for the reviewer` is prose,
# not the token, and does not match.
_NO_ENTRY_RE = re.compile(r"^\s*[`*_]*\s*NO_ENTRY\s*[`*_]*\s*\.?\s*$", re.IGNORECASE | re.MULTILINE)

# The confidence vocabulary memory ranks by. Anything else a model writes
# ("very high", "certain", "0.9") is not a rank and collapses to the middle.
EXPERIENCE_CONFIDENCE_LEVELS: tuple[str, ...] = ("high", "medium", "low")
_DEFAULT_CONFIDENCE = "medium"


def normalize_experience_confidence(value: Any) -> str:
    """Maps a reflector-written confidence to one of EXPERIENCE_CONFIDENCE_LEVELS.

    Unknown or empty values become `medium`: a lesson whose confidence cannot be
    read is neither promoted above well-evidenced ones nor buried below them.
    """
    token = str(value or "").strip().lower()
    return token if token in EXPERIENCE_CONFIDENCE_LEVELS else _DEFAULT_CONFIDENCE


def assert_no_leak_paths(
    root: Path | str,
    patterns: Sequence[str] = DEFAULT_LEAK_GLOBS,
) -> None:
    """Rejects benchmark answer-key paths before a context reaches a model.

    Args:
        root: Directory root to scan for leaked ground-truth files.
        patterns: Glob patterns matching prohibited answer-key filenames.

    Raises:
        ValueError: If any file under `root` matches a pattern in `patterns`.
    """
    root_path = Path(root)
    if not patterns or not root_path.is_dir():
        return
    hits: list[str] = []
    for file_path in root_path.rglob("*"):
        if file_path.is_symlink():
            raise ValueError(
                f"symlink prohibited in model context: {file_path}"
            )
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(root_path).as_posix()
        for pattern in patterns:
            clean_pat = str(pattern).replace("\\", "/").lstrip("./")
            if (
                fnmatch.fnmatch(rel, clean_pat)
                or fnmatch.fnmatch(rel, f"*/{clean_pat}")
                or fnmatch.fnmatch(file_path.name, clean_pat)
            ):
                hits.append(rel)
                break
    if hits:
        shown = ", ".join(sorted(hits)[:8])
        raise ValueError(
            f"answer-key path(s) would enter model context: {shown}"
        )


def assert_no_grader_verdicts(
    root: Path | str,
    markers: Sequence[str] = GRADER_VERDICT_MARKERS,
) -> None:
    """Refuses a context tree containing grader verdict output.

    Args:
        root: Directory root to scan.
        markers: Substring markers identifying grader verdict output.

    Raises:
        ValueError: If any filename or text file body contains a grader marker.
    """
    root_path = Path(root)
    if not markers or not root_path.is_dir():
        return
    for file_path in sorted(root_path.rglob("*")):
        if file_path.is_symlink():
            raise ValueError(
                f"symlink prohibited in model context: {file_path}"
            )
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(root_path).as_posix()
        for marker in markers:
            if marker in rel:
                raise ValueError(
                    f"grader verdict would enter model context: {rel!r} "
                    f"matched marker {marker!r} in its NAME."
                )
        if file_path.suffix.lower() not in _MARKER_SCAN_SUFFIXES:
            continue
        try:
            body = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for marker in markers:
            if marker in body:
                where = body.index(marker)
                snippet = body[max(0, where - 60) : where + 120]
                raise ValueError(
                    f"grader verdict would enter model context: {rel!r} "
                    f"contains marker {marker!r} at offset {where} "
                    f"({snippet!r})."
                )


def assert_tree_has_no_grader_verdicts(
    tree: Mapping[str, str],
    markers: Sequence[str] = GRADER_VERDICT_MARKERS,
) -> None:
    """Checks an in-memory file mapping (`rel -> body`) for grader verdicts.

    Args:
        tree: Mapping of relative file paths to file contents.
        markers: Substring markers identifying grader verdict output.

    Raises:
        ValueError: If any path or body in `tree` contains a grader marker.
    """
    for rel in sorted(tree):
        for marker in markers:
            if marker in rel:
                raise ValueError(
                    f"harness file {rel!r} has grader-verdict marker "
                    f"{marker!r} in its NAME"
                )
        if Path(rel).suffix.lower() not in _MARKER_SCAN_SUFFIXES:
            continue
        body = str(tree[rel])
        for marker in markers:
            if marker in body:
                where = body.index(marker)
                snippet = body[max(0, where - 60) : where + 120]
                raise ValueError(
                    f"harness file {rel!r} contains grader-verdict marker "
                    f"{marker!r} at offset {where} ({snippet!r})"
                )


def parse_reflection_blocks(text: str) -> dict[str, Any]:
    """Extracts structured `<experience>` lessons from reflector text.

    Accepts both key-value blocks (`cue: ...`, `lesson: ...`, `evidence: ...`)
    and JSON array/object blocks. A lesson is kept only when `cue`, `lesson`
    and `evidence` are all present -- the same gate Harness² applies before
    anything enters its library. `confidence` is normalised to
    `high` | `medium` | `low` (anything else reads as `medium`) and
    `target_stage` defaults to `global`.

    A bare `NO_ENTRY` line is the reflector saying, explicitly, that this run
    produced nothing citable. It returns `{"experiences": [], "no_entry": True}`
    and wins over any `<experience>` block in the same text: a lesson emitted
    alongside a declaration that there is none is a contradiction, and the
    fail-safe reading of a contradiction is that nothing enters memory.

    Returns `{"experiences": [...]}` otherwise; never raises on malformed
    input, but raises `ValueError` if the parsed output would carry a grader
    verdict.
    """
    experiences: list[dict[str, str]] = []
    if not text or not isinstance(text, str):
        return {"experiences": experiences}
    if _NO_ENTRY_RE.search(text):
        return {"experiences": [], "no_entry": True}

    allowed_keys = ("cue", "lesson", "evidence", "confidence", "target_stage")
    required = ("cue", "lesson", "evidence")

    def _accept(entry: dict[str, str]) -> None:
        if all(entry.get(req) for req in required):
            entry["confidence"] = normalize_experience_confidence(entry.get("confidence"))
            entry.setdefault("target_stage", "global")
            experiences.append(entry)

    for exp_match in _EXP_RE.finditer(text):
        body = exp_match.group(1).strip()
        if not body:
            continue
        parsed_json: Any = None
        if body.startswith("[") or body.startswith("{"):
            try:
                parsed_json = json.loads(body)
            except ValueError:
                parsed_json = None
        if isinstance(parsed_json, dict):
            parsed_json = [parsed_json]
        if isinstance(parsed_json, list):
            for item in parsed_json:
                if isinstance(item, dict):
                    _accept({
                        k: str(item.get(k) or "").strip()
                        for k in allowed_keys
                        if item.get(k) is not None
                    })
            continue

        entry: dict[str, str] = {}
        for line in body.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                clean_key = key.strip().lower().replace("-", "_")
                if clean_key in allowed_keys:
                    entry[clean_key] = val.strip()
        _accept(entry)

    out: dict[str, Any] = {"experiences": experiences}
    assert_tree_has_no_grader_verdicts({"EXPERIENCE.md": json.dumps(out)})
    return out
