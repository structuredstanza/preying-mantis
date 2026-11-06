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

"""Evidence-first selection among candidate harnesses, with a blind tie-break.

Harness2 picks among finals by a blind read of their deliverables, because a
document task offers nothing firmer. A Mantis rollout does: every finding
carries the INV-1 tier the pipeline itself established -- a sandbox sentinel
(``dynamic_confirmed``), a static read, or a critic's dismissal. That is ground-truth-free, and it is hard to game with
prose. So selection here is:

1. **Drop rules** (upstream's): a candidate that crashed, delivered nothing,
   or has no evidence file is out before anything is compared.
2. **Non-regression against the base**, two ways. A candidate that loses a
   finding the base runs *all* upheld (any confirmed tier -- on a static-only
   workflow nothing reaches ``dynamic_confirmed``, and the floor must still
   exist) is not adoptable whatever it adds; nor is one whose evidence score
   fell below the weakest base run's. The intersection over base repeats,
   not the union, so one lucky base run cannot veto every candidate; and the
   weakest base score, so one lucky base run cannot raise the bar.
3. **Evidence score**, severity-weighted by tier, dismissals counted against;
   then fewer tokens as the tie-breaker among equals.
4. Only when the top candidates are within ``TIE_EPSILON`` of each other does
   a **blind listwise read** of their anonymised findings reports decide, via
   upstream ``run_selection`` with the pool restricted to the tied labels.

A selector failure leaves ``ok=False`` -- nothing is adopted, nothing enters
the library -- exactly as upstream. ``selection.json`` records which path
decided and why.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from evals.evolve._vendor import load_harness2

TIE_EPSILON = 0.5
SCORE_FLOOR_TOLERANCE = 1e-6


def base_floor(adapter, base_runs: Sequence[Any]) -> Dict[str, Any]:
    """What a candidate must keep: the findings every base run upheld, and the weakest base score."""
    key_sets = []
    scores = []
    for rv in base_runs:
        if not (rv.ok and rv.delivered):
            continue
        ev = adapter.evidence(rv.run_dir)
        if not ev:
            continue
        keys = ev.get("confirmed_keys")
        if keys is None:  # evidence written by an older adapter
            keys = ev.get("verified_keys") or []
        key_sets.append(set(keys))
        scores.append(float(ev.get("score") or 0.0))
    return {
        "keys": sorted(set.intersection(*key_sets)) if key_sets else [],
        "score": min(scores) if scores else 0.0,
        "base_runs": len(scores),
    }


def base_verified_keys(adapter, base_runs: Sequence[Any]) -> List[str]:
    """Kept for callers of the old name; the floor is now ``base_floor``."""
    return base_floor(adapter, base_runs)["keys"]


def rank_candidates(adapter, candidates: Dict[str, Any], base_runs: Sequence[Any]) -> Dict[str, Any]:
    """Pure ranking over the adapter's evidence files. Returns the table the selector acts on."""
    floor = base_floor(adapter, base_runs)
    rows: List[Dict[str, Any]] = []
    dropped: Dict[str, str] = {}
    for label, rv in candidates.items():
        if not rv.ok:
            dropped[label] = f"rollout failed: {rv.error or 'unknown'}"
            continue
        if not rv.delivered:
            dropped[label] = "rollout delivered no findings report"
            continue
        ev = adapter.evidence(rv.run_dir)
        if not ev:
            dropped[label] = "no evidence file"
            continue
        summary = _summary(rv.run_dir)
        kept_keys = ev.get("confirmed_keys")
        if kept_keys is None:
            kept_keys = ev.get("verified_keys") or []
        lost = sorted(set(floor["keys"]) - set(kept_keys))
        score = float(ev.get("score") or 0.0)
        below = score + SCORE_FLOOR_TOLERANCE < floor["score"]
        rows.append({
            "label": label,
            "score": score,
            "verified": int(ev.get("verified") or 0),
            "confirmed": int(ev.get("confirmed") or len(kept_keys)),
            "live": int(ev.get("live") or 0),
            "dismissed": int(ev.get("dismissed") or 0),
            "tokens": int(summary.get("tokens") or 0),
            "regressed": lost,
            "below_base_score": below,
            "non_regressing": not lost and not below,
        })
    rows.sort(key=lambda r: (not r["non_regressing"], -r["score"], r["tokens"], r["label"]))
    eligible = [r for r in rows if r["non_regressing"]]
    tie: List[str] = []
    if len(eligible) >= 2 and abs(eligible[0]["score"] - eligible[1]["score"]) < TIE_EPSILON:
        top = eligible[0]["score"]
        tie = [r["label"] for r in eligible if abs(r["score"] - top) < TIE_EPSILON]
    return {"base_floor": floor, "base_verified_floor": floor["keys"], "ranked": rows,
            "eligible": [r["label"] for r in eligible], "tie": tie, "dropped": dropped}


def _summary(run_dir: Path) -> Dict[str, Any]:
    try:
        return json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def select(task_id: str, k: int, mode: str, candidates: Dict[str, Any], base_runs: Sequence[Any], *,
           adapter, selector, work_dir: Path):
    """Returns an upstream ``Selection``. Evidence decides; a blind read only breaks ties."""
    load_harness2()
    from harness2.core import selection as h2sel

    table = rank_candidates(adapter, candidates, base_runs)
    sel = h2sel.Selection(task_id=task_id, mode=mode, k=k, n_candidates=len(candidates), dropped=sorted(table["dropped"]))
    out_dir = Path(work_dir) / "selection" / f"{mode}{k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evidence_ranking.json").write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    eligible = table["eligible"]
    if not eligible:
        reasons = "; ".join(
            f"{r['label']}: " + ", ".join(
                ([f"lost base-confirmed {r['regressed']}"] if r["regressed"] else [])
                + ([f"score {r['score']:.2f} below the weakest base run's {table['base_floor']['score']:.2f}"] if r["below_base_score"] else []))
            for r in table["ranked"] if not r["non_regressing"])
        sel.error = ("no candidate is adoptable: " + (reasons or "every rollout was dropped: " + json.dumps(table["dropped"])))
        sel.ok = False
        _persist(sel, out_dir, "", table, decided_by="none")
        return sel

    if len(table["tie"]) >= 2 and selector is not None:
        pool = {label: candidates[label] for label in table["tie"]}
        try:
            blind = h2sel.run_selection(task_id, k, mode, pool, adapter=adapter, prop=selector,
                                        work_dir=Path(work_dir) / "blind", pool_kind="finals")
        except Exception as exc:  # a selector crash is a failure, not a pick
            blind = None
            sel.error = f"blind tie-break failed: {type(exc).__name__}: {exc}"[:400]
        if blind is not None and blind.ok and blind.chosen_label in pool:
            sel.chosen_label = blind.chosen_label
            sel.chosen_id = blind.chosen_id
            sel.id_map = dict(blind.id_map)
            sel.why = f"[evidence tie among {table['tie']}; blind read decided] {blind.why}".strip()
            sel.usd = float(getattr(blind, "usd", 0.0) or 0.0)
            sel.ok = True
            decided_by = "blind_tie_break"
        else:
            sel.ok = False
            sel.error = sel.error or f"blind tie-break did not resolve: {getattr(blind, 'error', '')}"
            _persist(sel, out_dir, "", table, decided_by="blind_tie_break_failed")
            return sel
    else:
        winner = eligible[0]
        row = next(r for r in table["ranked"] if r["label"] == winner)
        sel.chosen_label = winner
        sel.chosen_id = winner
        sel.id_map = {label: label for label in eligible}
        sel.why = (f"evidence: score {row['score']:.2f} with {row['verified']} machine-verified, {row['confirmed']} confirmed and "
                   f"{row['live']} live findings ({row['dismissed']} dismissed) at {row['tokens']:,} tokens; "
                   f"keeps every finding all base runs confirmed {table['base_floor']['keys']} and scores at or above "
                   f"the weakest base run ({table['base_floor']['score']:.2f})")
        sel.ok = True
        decided_by = "evidence"

    sel.chosen_run_id = candidates[sel.chosen_label].run_id
    _persist(sel, out_dir, str(candidates[sel.chosen_label].run_dir), table, decided_by=decided_by)
    return sel


def _persist(sel, out_dir: Path, run_dir: str, table: Dict[str, Any], *, decided_by: str) -> None:
    payload = sel.to_dict()
    payload.update({"chosen_run_dir": run_dir, "decided_by": decided_by, "evidence_ranking": table})
    (out_dir / "selection.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
