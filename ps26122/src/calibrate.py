"""Calibration report for the match-decision thresholds.

    python -m src.calibrate                       # data/labelled_cases.json, current provider
    python -m src.calibrate --cases other.json --json

Runs the real matcher (`match_progress_event_v2`) on every labelled case and
prints the four numbers that describe how the thresholds behave:

  * auto-accept precision : of the auto-accepted cases, how many were right
  * auto-accept share     : how many cases the system would apply without a human
  * review share          : how many go to a supervisor
  * unmatched share       : how many are treated as possible new activities

plus the decision accuracy and the thresholds used. Real field data will not be
shared, so these numbers come from a SYNTHETIC labelled set and only show
whether a threshold change makes things safer or riskier. Record the baseline in
the README after changing any threshold in `semantic_match.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "data" / "labelled_cases.json"
DEFAULT_SCHEDULE = ROOT / "data" / "schedule.json"


def load_cases(path: Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    for case in cases:
        if "id" not in case or "event" not in case or "expected" not in case:
            raise ValueError(f"Labelled case needs id, event and expected: {case!r}")
    return cases


def is_correct_decision(case: dict[str, Any], result: dict[str, Any]) -> bool:
    """Was the final decision right for this case?

    * expected UNMATCHED       -> the status must be UNMATCHED.
    * ambiguous twins          -> the status must NOT be AUTO_ACCEPTED.
    * everything else          -> the right activity, AUTO_ACCEPTED or REVIEW.
    """
    status = result.get("review_status")
    if status == "ERROR":
        return False
    if case["expected"] == "UNMATCHED":
        return status == "UNMATCHED"
    if case.get("ambiguous"):
        return status != "AUTO_ACCEPTED"
    return status in {"AUTO_ACCEPTED", "REVIEW"} and result.get("matched_activity_id") == case["expected"]


def is_correct_auto_accept(case: dict[str, Any], result: dict[str, Any]) -> bool:
    """An auto-accept is only right if it hit the expected activity of an unambiguous case."""
    return (
        result.get("review_status") == "AUTO_ACCEPTED"
        and case["expected"] != "UNMATCHED"
        and not case.get("ambiguous")
        and result.get("matched_activity_id") == case["expected"]
    )


def summarize(cases: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure function: the four numbers plus accuracy, from cases and results."""
    total = len(cases)
    counts = {"AUTO_ACCEPTED": 0, "REVIEW": 0, "UNMATCHED": 0, "ERROR": 0}
    for result in results:
        counts[result.get("review_status") if result.get("review_status") in counts else "ERROR"] += 1
    auto_correct = sum(is_correct_auto_accept(c, r) for c, r in zip(cases, results))
    decision_correct = sum(is_correct_decision(c, r) for c, r in zip(cases, results))

    def share(n: int) -> float | None:
        return round(n / total, 3) if total else None

    return {
        "cases": total,
        "auto_accept_precision": round(auto_correct / counts["AUTO_ACCEPTED"], 3) if counts["AUTO_ACCEPTED"] else None,
        "auto_accept_share": share(counts["AUTO_ACCEPTED"]),
        "review_share": share(counts["REVIEW"]),
        "unmatched_share": share(counts["UNMATCHED"]),
        "errors": counts["ERROR"],
        "decision_accuracy": share(decision_correct),
        "counts": counts,
    }


def run_cases(
    cases: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    matcher: Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run the matcher on each case. A crash counts as ERROR, like in the app."""
    if matcher is None:
        from .semantic_match import match_progress_event_v2 as matcher
    results = []
    for case in cases:
        event = {"event_id": case["id"], "raw_text": case.get("statement") or "", **case["event"]}
        try:
            results.append(dict(matcher(event, schedule)))
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            results.append({"event_id": case["id"], "review_status": "ERROR", "matched_activity_id": None,
                            "confidence": 0.0, "reason": f"{type(exc).__name__}: {exc}"})
    return results


def _pct(value: float | None) -> str:
    return "n/a (nothing was auto-accepted)" if value is None else f"{value * 100:.1f}%"


def format_report(cases: list[dict[str, Any]], results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    from . import semantic_match as sm

    lines = ["Case results", "-" * 78]
    for case, result in zip(cases, results):
        ok = "ok " if is_correct_decision(case, result) else "BAD"
        lines.append(
            f"{ok} {case['id']:<10} expected {case['expected']:<9}"
            f"{' (ambiguous)' if case.get('ambiguous') else '':<13}"
            f"-> {result.get('review_status'):<13} {str(result.get('matched_activity_id')):<8} "
            f"{(result.get('confidence') or 0) * 100:5.1f}%"
        )
    lines += [
        "",
        f"Cases:                    {summary['cases']}",
        f"Auto-accept precision:    {_pct(summary['auto_accept_precision'])}",
        f"Auto-accept share:        {_pct(summary['auto_accept_share'])}",
        f"Review share:             {_pct(summary['review_share'])}",
        f"Unmatched share:          {_pct(summary['unmatched_share'])}",
        f"Errors:                   {summary['errors']}",
        f"Decision accuracy:        {_pct(summary['decision_accuracy'])}",
        "",
        "Thresholds used: "
        f"auto-accept {sm.CONF_AUTO_ACCEPT}, review {sm.CONF_REVIEW}, skip-LLM {sm.CONF_SKIP_LLM}, "
        f"min margin {sm.AUTO_ACCEPT_MIN_MARGIN}, ambiguity cap {sm.AMBIGUITY_CAP} "
        f"(near-duplicate {sm.AMBIGUITY_CAP_NEAR_DUPLICATE})",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print match-decision calibration numbers.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="labelled cases JSON")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE, help="schedule JSON")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    results = run_cases(cases, schedule)
    summary = summarize(cases, results)
    print(json.dumps(summary, indent=2) if args.json else format_report(cases, results, summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
