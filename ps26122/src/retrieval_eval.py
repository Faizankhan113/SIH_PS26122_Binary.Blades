"""Retrieval-quality report. Needs no LLM and no API key.

    python -m src.retrieval_eval                 # human-readable table
    python -m src.retrieval_eval --json          # machine-readable summary
    python -m src.retrieval_eval --show-all      # also list the cases that are fine

For every case it asks the real retriever to rank the WHOLE schedule and reports
where the correct activity landed, and whether it made it into the shortlist the
LLM would see. It answers "did retrieval find the right activity?", which
`python -m src.calibrate` (the final decision) does not. Run both before and
after any change to retrieval and write the numbers in the README.

Cases: `data/retrieval_cases.json` (ID and thin-field cases) plus every
non-UNMATCHED case of `data/labelled_cases.json`, added as group "labelled-clear" or
"labelled-twin" (twins are ambiguous on purpose, so their rank is not an accuracy figure:
what matters for them is that they reach the shortlist).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "data" / "retrieval_cases.json"
DEFAULT_LABELLED = ROOT / "data" / "labelled_cases.json"
DEFAULT_SCHEDULE = ROOT / "data" / "schedule.json"
TOP_N = 3  # "found" means: ranked in the top N


def load_cases(path: Path = DEFAULT_CASES, labelled: Path | None = DEFAULT_LABELLED) -> list[dict[str, Any]]:
    cases = list(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    if labelled and Path(labelled).exists():
        for case in json.loads(Path(labelled).read_text(encoding="utf-8"))["cases"]:
            if case["expected"] != "UNMATCHED":
                cases.append({"id": case["id"], "group": "labelled-twin" if case.get("ambiguous") else "labelled-clear", "expected": [case["expected"]],
                              "event": {"raw_text": case.get("statement") or "", **case["event"]}})
    for case in cases:
        if not {"id", "group", "expected", "event"} <= set(case):
            raise ValueError(f"Retrieval case needs id, group, expected and event: {case!r}")
    return cases


def shortlist_for(index: Any, event: dict[str, Any], ranked: list[Any]) -> list[Any]:
    """The candidates the LLM would be shown. Uses the dynamic shortlist selector."""
    from . import semantic_match as sm

    select = getattr(sm, "select_shortlist", None)
    return select(event, ranked) if select else ranked[:5]


def evaluate(cases: list[dict[str, Any]], schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from .semantic_match import SemanticMatchIndex

    index = SemanticMatchIndex(schedule)
    rows = []
    for case in cases:
        event = {"event_id": case["id"], **case["event"]}
        ranked = index.retrieve(event, top_k=len(schedule))
        order = [c.activity_id for c in ranked]
        ranks = [order.index(a) + 1 for a in case["expected"] if a in order]
        shortlist = {c.activity_id for c in shortlist_for(index, event, ranked)}
        rows.append({
            "id": case["id"], "group": case["group"], "expected": case["expected"],
            "rank": min(ranks) if ranks else None,
            "all_expected_in_top_n": all(a in order[:TOP_N] for a in case["expected"]) if len(case["expected"]) > 1 else None,
            "in_shortlist": all(a in shortlist for a in case["expected"]),
            "shortlist_size": len(shortlist),
        })
    return rows


def _share(n: int, total: int) -> float | None:
    return round(n / total, 3) if total else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def block(sel: list[dict[str, Any]]) -> dict[str, Any]:
        ranks = [r["rank"] for r in sel if r["rank"] is not None]
        return {
            "cases": len(sel),
            "top1": _share(sum(1 for r in sel if r["rank"] == 1), len(sel)),
            f"top{TOP_N}": _share(sum(1 for r in sel if r["rank"] is not None and r["rank"] <= TOP_N), len(sel)),
            "in_shortlist": _share(sum(1 for r in sel if r["in_shortlist"]), len(sel)),
            "mean_rank": round(sum(ranks) / len(ranks), 2) if ranks else None,
            "avg_shortlist_size": round(sum(r["shortlist_size"] for r in sel) / len(sel), 1) if sel else None,
        }

    groups = sorted({r["group"] for r in rows})
    return {"all": block(rows), **{g: block([r for r in rows if r["group"] == g]) for g in groups}}


def format_report(rows: list[dict[str, Any]], summary: dict[str, Any], show_all: bool = False) -> str:
    lines = ["Retrieval rank of the correct activity (1 = best)", "-" * 78]
    for r in rows:
        problem = r["rank"] is None or r["rank"] > TOP_N or not r["in_shortlist"]
        if not (show_all or problem):
            continue
        lines.append(f"{'BAD' if problem else 'ok '} {r['group']:<9} {r['id']:<22} rank {str(r['rank']):>4}   "
                     f"in shortlist: {'yes' if r['in_shortlist'] else 'NO ':<3} (size {r['shortlist_size']})   "
                     f"expected {','.join(r['expected'])}")
    if len(lines) == 2:
        lines.append(f"(every case is in the top {TOP_N} and in the shortlist; use --show-all to list them)")
    lines += ["", f"{'group':<10}{'cases':>6}{'top-1':>8}{f'top-{TOP_N}':>8}{'shortlist':>11}{'mean rank':>11}{'avg list':>10}"]
    for name, b in summary.items():
        pct = lambda v: "n/a" if v is None else f"{v * 100:.0f}%"  # noqa: E731
        lines.append(f"{name:<10}{b['cases']:>6}{pct(b['top1']):>8}{pct(b[f'top{TOP_N}']):>8}"
                     f"{pct(b['in_shortlist']):>11}{str(b['mean_rank']):>11}{str(b['avg_shortlist_size']):>10}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report how well retrieval ranks the correct activity.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--labelled", type=Path, default=DEFAULT_LABELLED)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args(argv)
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    rows = evaluate(load_cases(args.cases, args.labelled), schedule)
    summary = summarize(rows)
    print(json.dumps({"summary": summary, "cases": rows}, indent=2) if args.json else format_report(rows, summary, args.show_all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
