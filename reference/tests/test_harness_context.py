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

"""Unit tests for leak tripwires, `<experience>` parsing, and stage-targeted recall."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest
import unittest.mock

_REF_ROOT = Path(__file__).resolve().parent.parent
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from core.database import init_db, read_learnings, record_learning  # noqa: E402
from core.harness_context import (  # noqa: E402
    assert_no_grader_verdicts,
    assert_no_leak_paths,
    assert_tree_has_no_grader_verdicts,
    normalize_experience_confidence,
    parse_reflection_blocks,
)
from core.memory import (  # noqa: E402
    recall,
    recall_for_stage,
    record_reflection_output,
    render_stage_learnings_for_agent,
)


def _experience(cue: str, confidence: str, stage: str = "reviewer") -> str:
    """One evidence-complete `<experience>` block, as the reflector emits it."""
    return (
        "<experience>"
        + json.dumps([{
            "cue": cue,
            "lesson": f"lesson for {cue}",
            "evidence": f"finding #7 confirmed while {cue}",
            "confidence": confidence,
            "target_stage": stage,
        }])
        + "</experience>"
    )


def _set_timestamp(db: str, cue: str, timestamp: str) -> None:
    """Pins a learning row's write time so recency is deterministic in a test."""
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE learnings SET timestamp = ? WHERE learning LIKE ?",
            (timestamp, f"When {cue}:%"),
        )


class LeakTripwireTest(unittest.TestCase):
    def test_leak_and_grader_verdict_tripwires_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ground_truth.json").write_text('{"findings": []}', encoding="utf-8")
            with self.assertRaises(ValueError):
                assert_no_leak_paths(root)
            (root / "ground_truth.json").unlink()
            assert_no_leak_paths(root)

            (root / "trace.json").write_text('{"cross_file_recall": 0.75}', encoding="utf-8")
            with self.assertRaises(ValueError):
                assert_no_grader_verdicts(root)

        with self.assertRaises(ValueError):
            assert_tree_has_no_grader_verdicts({"eval.json": '{"ground_truth_total": 5}'})

    def test_symlink_in_context_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "real.txt").write_text("ok", encoding="utf-8")
            (root / "link.txt").symlink_to(root / "real.txt")
            with self.assertRaises(ValueError):
                assert_no_leak_paths(root)
            with self.assertRaises(ValueError):
                assert_no_grader_verdicts(root)


class ReflectionParsingTest(unittest.TestCase):
    SAMPLE = """
<experience>
[
  {
    "cue": "auditing cross-file SQL or coupon state transitions",
    "lesson": "call find_callers on DAO mutation methods before reading route handlers",
    "evidence": "turn 2 find_callers('redeem_coupon') immediately isolated the unguarded caller",
    "confidence": "high",
    "target_stage": "researcher"
  },
  {
    "cue": "triaging multi-file findings in reviewer",
    "lesson": "use get_function_boundary on both caller and callee before dismissing",
    "evidence": "saved 2 false dismissals on cross-module flows",
    "confidence": "medium",
    "target_stage": "reviewer"
  }
]
</experience>
"""

    def test_parses_json_and_key_value_blocks(self):
        parsed = parse_reflection_blocks(self.SAMPLE)
        self.assertEqual(set(parsed), {"experiences"})
        self.assertEqual(len(parsed["experiences"]), 2)
        self.assertEqual(parsed["experiences"][0]["target_stage"], "researcher")

        kv = parse_reflection_blocks(
            "<experience>\ncue: any endpoint\nlesson: verify auth decorator\nevidence: turn 1\n</experience>"
        )
        self.assertEqual(len(kv["experiences"]), 1)
        self.assertEqual(kv["experiences"][0]["confidence"], "medium")
        self.assertEqual(kv["experiences"][0]["target_stage"], "global")

    def test_lessons_without_evidence_are_dropped(self):
        """An invented attribution poisons memory for every later run; no evidence, no entry."""
        no_evidence = (
            '<experience>[{"cue": "x", "lesson": "y", "target_stage": "researcher"}]</experience>'
        )
        self.assertEqual(parse_reflection_blocks(no_evidence)["experiences"], [])
        self.assertEqual(parse_reflection_blocks("")["experiences"], [])
        self.assertEqual(parse_reflection_blocks("no blocks here")["experiences"], [])

    def test_edits_blocks_are_ignored(self):
        """Single-rollout runs have no edits to attribute; the block is not parsed or stored."""
        text = (
            "<edits>\n- `find_callers` — helped — found sink\n</edits>\n"
            "<experience>\ncue: c\nlesson: l\nevidence: e\n</experience>"
        )
        parsed = parse_reflection_blocks(text)
        self.assertNotIn("edits", parsed)
        self.assertEqual(len(parsed["experiences"]), 1)

    def test_record_and_stage_targeted_recall(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "knowledge.db")
            init_db(db)
            record_learning(db, "run-0", "general", "Always check auth middleware.")
            counts = record_reflection_output(db, "run-1", self.SAMPLE)
            self.assertEqual(counts, {"experiences": 2})

            researcher_lessons = recall_for_stage(db, "researcher", include_general=False)
            self.assertEqual(len(researcher_lessons), 1)
            self.assertIn("find_callers", researcher_lessons[0]["learning"])
            self.assertIn("evidence:", researcher_lessons[0]["learning"])

            reviewer_lessons = recall_for_stage(db, "reviewer", include_general=False)
            self.assertEqual(len(reviewer_lessons), 1)
            self.assertIn("get_function_boundary", reviewer_lessons[0]["learning"])

            # Untargeted lessons (target_stage=global) are recalled with include_general=True.
            record_reflection_output(
                db,
                "run-2",
                "<experience>\ncue: any endpoint\nlesson: verify auth decorator\nevidence: turn 1\n</experience>",
            )
            with_general = recall_for_stage(db, "reviewer", include_general=True)
            self.assertTrue(any("verify auth decorator" in item["learning"] for item in with_general))

    def test_legacy_causal_edit_rows_never_reach_general_recall(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "knowledge.db")
            init_db(db)
            record_learning(db, "run-old", "causal_edit", "[EDIT:helped] find_callers — found sink")
            record_learning(db, "run-old", "codebase", "Parameterized queries missing in dal.")
            mem = recall(db)
            categories = {item["category"] for item in mem["learnings"]}
            self.assertNotIn("causal_edit", categories)
            self.assertIn("codebase", categories)

    def test_no_entry_is_an_explicit_empty_result(self):
        """A bare NO_ENTRY line is the reflector saying there is nothing citable, not a parse miss."""
        for text in ("NO_ENTRY", "no_entry\n", "  `NO_ENTRY`  ", "**NO_ENTRY**.", "Reflection:\nNO_ENTRY\n"):
            with self.subTest(text=text):
                self.assertEqual(parse_reflection_blocks(text), {"experiences": [], "no_entry": True})
        # Prose that merely contains the word is not the token.
        self.assertEqual(parse_reflection_blocks("NO_ENTRY for the reviewer stage"), {"experiences": []})
        self.assertNotIn("no_entry", parse_reflection_blocks("nothing here"))
        self.assertNotIn("no_entry", parse_reflection_blocks(self.SAMPLE))
        # A lesson alongside a declaration that there is none is a contradiction;
        # the fail-safe reading is that nothing enters memory.
        both = "NO_ENTRY\n" + _experience("contradiction", "high")
        self.assertEqual(parse_reflection_blocks(both), {"experiences": [], "no_entry": True})

        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "knowledge.db")
            init_db(db)
            self.assertEqual(record_reflection_output(db, "run-1", "NO_ENTRY"), {"experiences": 0})
            self.assertEqual(record_reflection_output(db, "run-1", both), {"experiences": 0})
            self.assertEqual(read_learnings(db), [])

    def test_confidence_is_normalised_to_the_ranking_vocabulary(self):
        self.assertEqual(normalize_experience_confidence("HIGH"), "high")
        self.assertEqual(normalize_experience_confidence(" Low "), "low")
        for junk in ("very high", "certain", "0.9", "", None, 3):
            with self.subTest(value=junk):
                self.assertEqual(normalize_experience_confidence(junk), "medium")

        parsed = parse_reflection_blocks(
            _experience("a", "HIGH") + _experience("b", "certain") + _experience("c", "Low")
            + "<experience>\ncue: d\nlesson: l\nevidence: e\n</experience>"
        )
        self.assertEqual(
            [e["confidence"] for e in parsed["experiences"]], ["high", "medium", "low", "medium"]
        )

        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "knowledge.db")
            init_db(db)
            record_reflection_output(db, "run-1", _experience("x", "VERY HIGH"))
            tags = read_learnings(db)[0]["tags"]
            self.assertIn("confidence:medium", tags)
            self.assertNotIn("confidence:very high", tags)

    def test_stage_recall_ranks_by_confidence_then_recency(self):
        """high > medium > low; within a rank the newest lesson first; the cap keeps the top."""
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "knowledge.db")
            init_db(db)
            # Written in an order that agrees with none of the expected orderings,
            # with timestamps pinned so recency does not depend on the wall clock.
            for cue, confidence, when in (
                ("old-medium", "medium", "2026-01-01 00:00:00"),
                ("new-low", "low", "2026-03-01 00:00:00"),
                ("old-high", "high", "2026-01-01 00:00:00"),
                ("new-medium", "medium", "2026-03-01 00:00:00"),
                ("new-high", "high", "2026-03-01 00:00:00"),
                ("old-low", "low", "2026-01-01 00:00:00"),
            ):
                record_reflection_output(db, "run-1", _experience(cue, confidence))
                _set_timestamp(db, cue, when)
            # A later run's lesson for another stage must not leak into this one.
            record_reflection_output(db, "run-2", _experience("other-stage", "high", stage="reproducer"))

            items = recall_for_stage(db, "reviewer", include_general=False)
            cues = [item["learning"].split(":")[0].removeprefix("When ") for item in items]
            self.assertEqual(
                cues, ["new-high", "old-high", "new-medium", "old-medium", "new-low", "old-low"]
            )
            self.assertEqual([item["confidence"] for item in items][:2], ["high", "high"])

            # The cap drops from the bottom: the least-evidenced, oldest lessons go first.
            capped = recall_for_stage(db, "reviewer", include_general=False, max_learnings=3)
            self.assertEqual(
                [i["learning"].split(":")[0] for i in capped],
                ["When new-high", "When old-high", "When new-medium"],
            )

            rendered = render_stage_learnings_for_agent(db, "reviewer")
            lesson_lines = [ln for ln in rendered.splitlines() if ln.lstrip().startswith("- [")]
            self.assertEqual(
                [ln.split("]")[0].split("[")[1] for ln in lesson_lines],
                ["high", "high", "medium", "medium", "low", "low"],
            )
            self.assertTrue(lesson_lines[0].lstrip().startswith("- [high] When new-high:"))
            self.assertIn("(evidence:", lesson_lines[0])
            # Nothing targeting a stage renders nothing, as before.
            self.assertEqual(render_stage_learnings_for_agent(db, "planner"), "")

    def test_reflector_prompt_states_the_no_entry_contract(self):
        """The prompt and the parser must agree on the token, or NO_ENTRY is never emitted."""
        from core.prompts import STAGE_PROMPTS

        prompt = STAGE_PROMPTS["reflector"]
        self.assertIn("NO_ENTRY", prompt)
        self.assertIn("evidence", prompt)
        self.assertIn("dynamic_confirmed", prompt)
        self.assertNotIn("<edits>", prompt)
        self.assertNotIn("```", prompt)


class AppendedInstructionCapTest(unittest.TestCase):
    """The stage-experience block is bounded at the seam that appends it."""

    def _long_block(self, lessons: int = 40, cue_width: int = 100) -> tuple[str, str]:
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        db = str(Path(tmp) / "knowledge.db")
        init_db(db)
        # Highest confidence first in write order too, so a cut from the
        # bottom is unambiguous about which lessons it removes. Each rendered
        # line stays under memory's 400-char prose clip, so it ends in ")".
        for i in range(lessons):
            confidence = "high" if i < 5 else ("medium" if i < 20 else "low")
            record_reflection_output(
                db, f"run-{i}", _experience(f"cue{i:02d}-{'x' * cue_width}", confidence, stage="reproducer")
            )
        return db, render_stage_learnings_for_agent(db, "reproducer", max_learnings=lessons)

    def test_bound_keeps_fence_closed_and_announces_the_cut(self):
        from core.graph_loader import MAX_APPENDED_INSTRUCTION_CHARS, bound_appended_instruction
        from core.llm_gateway import UNTRUSTED_DATA_END, UNTRUSTED_DATA_START

        _, block = self._long_block()
        self.assertGreater(len(block), MAX_APPENDED_INSTRUCTION_CHARS)

        bounded, omitted = bound_appended_instruction(block)
        self.assertLessEqual(len(bounded), MAX_APPENDED_INSTRUCTION_CHARS)
        self.assertGreater(omitted, 0)
        marker = f"[STAGE EXPERIENCE truncated: {omitted} chars omitted; fewer, better-evidenced lessons are worth more than more]"
        self.assertTrue(bounded.endswith("\n" + marker), bounded[-200:])
        # The prose is still fenced, the fence is still closed, and the marker
        # (harness text, not prior-run prose) sits outside it.
        self.assertEqual(bounded.count(UNTRUSTED_DATA_START), 1)
        self.assertEqual(bounded.count(UNTRUSTED_DATA_END), 1)
        self.assertLess(bounded.index(UNTRUSTED_DATA_END), bounded.index(marker))
        # Whole lines only, cut from the bottom: every high-confidence lesson
        # survives and nothing is left half-said.
        kept_lines = [ln for ln in bounded.splitlines() if ln.lstrip().startswith("- [")]
        self.assertEqual(sum(ln.lstrip().startswith("- [high]") for ln in kept_lines), 5)
        self.assertTrue(all(ln.endswith(")") for ln in kept_lines), kept_lines[-1])
        # Omitted counts the original characters that are gone, exactly.
        self.assertEqual(omitted, len(block) - len(bounded.rsplit("\n", 1)[0]))

        # Under the cap nothing changes, and the object is passed through.
        short = render_stage_learnings_for_agent(self._long_block(lessons=2)[0], "reproducer")
        self.assertIs(bound_appended_instruction(short)[0], short)
        self.assertEqual(bound_appended_instruction(short)[1], 0)
        self.assertEqual(bound_appended_instruction("")[1], 0)

        # Holds for smaller caps too, down to a cap that admits nothing but the marker.
        for cap in (1000, 500, 300):
            with self.subTest(cap=cap):
                b, o = bound_appended_instruction(block, cap)
                self.assertLessEqual(len(b), cap)
                self.assertEqual(o, len(block) - len(b.rsplit("\n", 1)[0]))

    def test_callback_appends_a_bounded_block_and_logs_once(self):
        from core.context import RunContext, current_run_context
        from core.graph_loader import (
            MAX_APPENDED_INSTRUCTION_CHARS,
            _make_stage_experience_before_model_callback,
        )

        db, block = self._long_block()
        ctx = RunContext(jail_dir=str(Path(db).parent), db_path=db, run_id="run-cap")
        tok = current_run_context.set(ctx)
        try:
            class _Cfg:
                system_instruction = "Base reproducer prompt."

            class _Req:
                def __init__(self):
                    self.config = _Cfg()
                    self.appended: list[str] = []

                def append_instructions(self, items):
                    self.appended.extend(items)
                    self.config.system_instruction += "\n\n" + "\n\n".join(items)

            before_model = _make_stage_experience_before_model_callback("reproducer")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                req1, req2 = _Req(), _Req()
                before_model(None, req1)
                before_model(None, req2)
            self.assertEqual(len(req1.appended), 1)
            appended = req1.appended[0]
            self.assertLessEqual(len(appended), MAX_APPENDED_INSTRUCTION_CHARS)
            self.assertIn("[STAGE EXPERIENCE - REPRODUCER]", appended)
            self.assertIn("[STAGE EXPERIENCE truncated:", appended)
            self.assertEqual(req2.appended, [appended])
            # Logged once per node, not once per model turn.
            self.assertEqual(err.getvalue().count("stage experience for 'reproducer'"), 1)
            self.assertIn(str(MAX_APPENDED_INSTRUCTION_CHARS), err.getvalue())
        finally:
            current_run_context.reset(tok)


class SynthesizerPromptCapTest(unittest.TestCase):
    """A synthesizer-authored system_prompt over the cap is refused, never truncated."""

    def _spec(self, prompt: str) -> dict:
        return {
            "name": "prompt_cap_workflow",
            "nodes": [
                {"id": "custom_auditor", "type": "agent", "system_prompt": prompt, "tools": ["read_file", "report_findings"]},
                {"id": "reporter", "type": "agent", "skill": "mantis-report", "tools": ["generate_report", "get_findings"]},
            ],
            "edges": [
                {"from": "START", "to": "custom_auditor"},
                {"from": "custom_auditor", "to": "reporter"},
            ],
        }

    def test_over_long_prompt_is_a_validation_error_naming_node_length_and_cap(self):
        from core.graph_loader import MAX_NODE_SYSTEM_PROMPT_CHARS
        from core.synthesizer import ResearchGraphSynthesizer

        synth = ResearchGraphSynthesizer()
        too_long = "Audit every handler. " * (MAX_NODE_SYSTEM_PROMPT_CHARS // 20 + 1)
        self.assertGreater(len(too_long), MAX_NODE_SYSTEM_PROMPT_CHARS)
        with self.assertRaises(ValueError) as caught:
            synth.sanitize_and_validate_spec(self._spec(too_long), objective="Audit handlers")
        message = str(caught.exception)
        self.assertTrue(message.startswith("[SYNTHESIZER"), message)
        self.assertIn("custom_auditor", message)
        self.assertIn(str(len(too_long.strip())), message)
        self.assertIn(str(MAX_NODE_SYSTEM_PROMPT_CHARS), message)

        # Just under the cap passes through whole: a prompt the gate admits is
        # the prompt the model wrote, not a prefix of it.
        fits = ("Audit every handler. " * 600)[: MAX_NODE_SYSTEM_PROMPT_CHARS - 1].rstrip()
        self.assertGreater(len(fits), 8192)  # the length the old gate silently cut at
        spec = synth.sanitize_and_validate_spec(self._spec(fits), objective="Audit handlers")
        node = next(n for n in spec.nodes if n.id == "custom_auditor")
        self.assertEqual(node.system_prompt, fits)

    def test_shipped_workflow_and_archetypes_clear_the_cap_with_margin(self):
        from core.graph_loader import MAX_NODE_SYSTEM_PROMPT_CHARS
        from core.prompts import STAGE_PROMPTS
        from core.synthesizer import ResearchGraphSynthesizer

        synth = ResearchGraphSynthesizer()
        with open(_REF_ROOT / "workflow.json", encoding="utf-8") as fh:
            shipped = json.load(fh)
        spec = synth.sanitize_and_validate_spec(shipped, objective="shipped workflow round trip")
        self.assertEqual(len(spec.nodes), len(shipped["nodes"]))

        longest = 0
        for objective in (
            "audit kernel ioctl handlers",
            "Audit AWS IAM roles and policies",
            "Audit GitHub Actions CI/CD pipelines for injection",
            "Audit async Rust channels for deadlock hazards",
        ):
            for parallel in (False, True):
                arch = synth.synthesize_archetype(objective=objective, parallel_researchers=parallel)
                for node in arch.nodes:
                    longest = max(longest, len(getattr(node, "system_prompt", None) or ""))
        longest = max(longest, max(len(p) for p in STAGE_PROMPTS.values()))
        # "With margin": the longest prompt the harness itself writes uses
        # well under half the cap.
        self.assertLess(longest * 2, MAX_NODE_SYSTEM_PROMPT_CHARS, longest)


class StageModelCallbackTest(unittest.TestCase):
    def test_record_learning_deduplication_and_stage_model_callbacks(self):
        from core.context import RunContext, current_run_context
        from core.graph_loader import (
            _make_reflector_after_model_callback,
            _make_stage_experience_before_model_callback,
        )

        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "knowledge.db")
            init_db(db)
            ctx = RunContext(jail_dir=tmp, db_path=db, run_id="run-dedup")
            tok = current_run_context.set(ctx)
            try:
                after_model = _make_reflector_after_model_callback("reflector")

                class _Part:
                    text = (
                        "<experience>\n"
                        "cue: checking SQL sinks\n"
                        "lesson: run get_function_boundary before dismissing\n"
                        "evidence: turn 3\n"
                        "target_stage: reviewer\n"
                        "</experience>"
                    )
                    thought = False

                class _Content:
                    parts = [_Part()]

                class _Resp:
                    content = _Content()

                # Calling twice must deduplicate within the same run_id.
                after_model(None, _Resp())
                after_model(None, _Resp())
                rows = read_learnings(db, run_id="run-dedup")
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["category"], "experience:reviewer")

                rendered = render_stage_learnings_for_agent(db, "reviewer")
                self.assertIn("[STAGE EXPERIENCE - REVIEWER]", rendered)
                self.assertIn("get_function_boundary", rendered)
                self.assertEqual(render_stage_learnings_for_agent(db, "reproducer"), "")

                class _Cfg:
                    system_instruction = "Base reviewer prompt."

                class _Req:
                    config = _Cfg()

                    def append_instructions(self, items):
                        self.config.system_instruction += "\n\n" + "\n\n".join(items)

                req = _Req()
                before_model = _make_stage_experience_before_model_callback("reviewer")
                with unittest.mock.patch(
                    "core.memory.render_stage_learnings_for_agent",
                    wraps=render_stage_learnings_for_agent,
                ) as spy_render:
                    before_model(None, req)
                    self.assertIn("[STAGE EXPERIENCE - REVIEWER]", req.config.system_instruction)
                    # Second turn with a fresh LlmRequest uses the closure cache
                    # instead of re-querying SQLite.
                    req2 = _Req()
                    req2.config = _Cfg()
                    before_model(None, req2)
                    self.assertIn("[STAGE EXPERIENCE - REVIEWER]", req2.config.system_instruction)
                    self.assertEqual(spy_render.call_count, 1)
            finally:
                current_run_context.reset(tok)


if __name__ == "__main__":
    unittest.main()
