"""Seed a synthetic execution history for the Institutional Memory views.

This is a standalone, explicitly-invoked step, separate from the live
ingestion -> extraction -> matching flow (which writes to the database on its
own). It fills the report, statement, matching-result, decision and `main`
tables with a *synthetic* execution history, purely so the Institutional
Memory views have something realistic to show. In production this history
would instead accumulate naturally from real approved updates over the life of
the project.

Run it directly:

    python -m src.seed_history            # seed on top of current DB
    python -m src.seed_history --reset    # wipe report history first, then seed

It reuses the real `save_progress_update()` persistence function rather than
writing raw SQL, so the seeded data goes through the exact same rules real
updates do:

  - AUTO_ACCEPTED rows get a system AUTO_ACCEPT decision and are applied to
    `main`.
  - REVIEW rows stay PENDING on purpose, so the supervisor queue has content on
    first load. They do not change `main` and are left out of the
    institutional-memory statistics until a supervisor approves them.
"""

from __future__ import annotations

import argparse
import random
from datetime import date, datetime, time, timedelta
from typing import Any

from .database import get_connection, initialize_database, reset_history, save_progress_update
from .timeutil import format_actual

DELAY_CATEGORIES = [
    "MATERIAL_AVAILABILITY",
    "MANPOWER_SHORTAGE",
    "DESIGN_CHANGE",
    "WEATHER",
    "EQUIPMENT_BREAKDOWN",
    "PERMIT_APPROVAL_DELAY",
    "INTERFACE_DELAY",
    "REWORK_QUALITY",
]

_DELAY_REASON_TEXT = {
    "MATERIAL_AVAILABILITY": "the required material arrived late",
    "MANPOWER_SHORTAGE": "insufficient crew was available on site",
    "DESIGN_CHANGE": "a design revision was issued mid-work",
    "WEATHER": "adverse weather stopped outdoor work",
    "EQUIPMENT_BREAKDOWN": "a crane/equipment breakdown occurred on site",
    "PERMIT_APPROVAL_DELAY": "work permit approval was delayed",
    "INTERFACE_DELAY": "the crew was waiting on an interfacing discipline to clear the area",
    "REWORK_QUALITY": "rework was required after a QC inspection",
}

CONTRACTORS_BY_DISCIPLINE = {
    "Piping": ["ABC Piping", "XYZ Contractors", "Delta Pipeline Works"],
    "Mechanical": ["Apex Rotating Equipment", "Meridian Mechanical"],
    "Electrical": ["Volt Line Electrical", "PowerGrid Contractors"],
    "Instrumentation": ["Precision Instruments Co", "Signal Path Automation"],
    "Civil": ["Bedrock Civil Works", "Foundation Infra Ltd"],
    "QA/QC": ["Assure QA Services", "Integrity Inspection Co"],
}

_REPORTER_NAMES = ["R. Menon", "S. Iyer", "A. Sharma", "T. Kulkarni", "P. Nair", "V. Rao"]

# Rough "typical schedule variance in days" per discipline, purely to make
# the synthetic institutional-memory dataset feel differentiated across
# disciplines -- not a claim about real construction statistics.
DISCIPLINE_VARIANCE_BIAS = {
    "Piping": (1, 5),
    "Mechanical": (0, 4),
    "Electrical": (-1, 2),
    "Instrumentation": (-2, 1),
    "Civil": (-1, 3),
    "QA/QC": (-1, 1),
}


# Share of seeded start / finish values that carry an explicit time of day.
# The rest stay date-only on purpose, so the sample data shows a realistic mix and no
# screen ever has a made-up 00:00. Decided with a SEPARATE random generator so
# adding times did not change any earlier seeded choice (activities, statuses,
# dates, reporters).
TIMED_START_SHARE = 0.5
TIMED_FINISH_SHARE = 0.5


def _synthetic_statement(
    row: dict[str, Any],
    status: str,
    progress: float,
    actual_finish: str | None,
    contractor: str,
    delay_category: str | None,
    start_note: str = "",
) -> str:
    """The synthetic report sentence. `actual_finish` is text that may include a
    time ("2026-09-02 17:30"); `start_note` is an optional " Work started ..." sentence."""
    desc = row["description"]
    asset = row.get("asset") or ""
    if status == "COMPLETED":
        return f"{desc} on {asset} completed on {actual_finish} by {contractor}.{start_note}"
    if status == "DELAYED":
        reason_text = _DELAY_REASON_TEXT.get(delay_category, "an unspecified site issue")
        return f"{desc} on {asset} is delayed because {reason_text}.{start_note}"
    return f"{desc} on {asset} is approximately {progress:.0f}% complete, reported by {contractor}.{start_note}"


def _pick_time(trng: random.Random, first_hour: int, last_hour: int) -> time:
    return time(trng.randint(first_hour, last_hour), trng.choice((0, 15, 30, 45)))


def seed_synthetic_history(seed: int = 42, fraction: float = 0.65) -> dict[str, int]:
    rng = random.Random(seed)
    time_rng = random.Random(f"{seed}:time-of-day")  # Separate stream, see TIMED_*_SHARE

    conn = get_connection()
    try:
        # Archived activities (removed from schedule.json) get no synthetic history.
        # Dates come back as 'YYYY-MM-DD' text, and the "C" collation keeps the
        # order a plain byte-wise sort by id, so the same --seed always picks the
        # same activities whatever collation the database was created with.
        rows = [
            dict(r)
            for r in conn.execute(
                """
                SELECT l6_id, description, discipline, asset, location,
                       to_char(planned_start, 'YYYY-MM-DD')  AS planned_start,
                       to_char(planned_finish, 'YYYY-MM-DD') AS planned_finish
                FROM activities
                WHERE archived_at IS NULL
                ORDER BY l6_id COLLATE "C"
                """
            ).fetchall()
        ]
    finally:
        conn.close()

    seed_count = max(1, int(len(rows) * fraction))
    seeded_rows = rng.sample(rows, min(seed_count, len(rows)))

    summary = {"seeded": 0, "completed": 0, "delayed": 0, "in_progress": 0, "applied": 0, "pending_review": 0, "timed_start": 0, "timed_finish": 0, "timed_both": 0}

    for row in seeded_rows:
        discipline = row.get("discipline") or "Piping"
        contractor = rng.choice(CONTRACTORS_BY_DISCIPLINE.get(discipline, ["General Contractor Co"]))
        lo, hi = DISCIPLINE_VARIANCE_BIAS.get(discipline, (-1, 3))
        variance_days = rng.randint(lo, hi)

        planned_start = date.fromisoformat(row["planned_start"]) if row.get("planned_start") else date(2026, 6, 1)
        planned_finish = (
            date.fromisoformat(row["planned_finish"]) if row.get("planned_finish") else planned_start + timedelta(days=10)
        )

        outcome_roll = rng.random()
        if outcome_roll < 0.62:
            status, progress, delay_category = "COMPLETED", 100.0, None
            actual_start_d = planned_start + timedelta(days=rng.randint(-1, 2))
            actual_finish_d = planned_finish + timedelta(days=variance_days)
            summary["completed"] += 1
        elif outcome_roll < 0.80:
            status, progress = "DELAYED", float(rng.randint(20, 75))
            actual_start_d = planned_start + timedelta(days=rng.randint(0, 3))
            actual_finish_d = None
            delay_category = rng.choice(DELAY_CATEGORIES)
            summary["delayed"] += 1
        else:
            status, progress, delay_category = "IN_PROGRESS", float(rng.randint(15, 85)), None
            actual_start_d = planned_start + timedelta(days=rng.randint(-1, 2))
            actual_finish_d = None
            summary["in_progress"] += 1

        report_date = (actual_finish_d or actual_start_d or planned_start) + timedelta(days=rng.randint(0, 2))
        reporter = f"{rng.choice(_REPORTER_NAMES)} ({discipline} Supervisor)"

        # Give some values an explicit time of day (drawn even when unused, so
        # the stream stays aligned per row). Working hours: starts 06:00-10:45,
        # finishes 13:00-19:45. A finish that would not come after its start is left date-only.
        start_time = _pick_time(time_rng, 6, 10) if time_rng.random() < TIMED_START_SHARE else None
        finish_time = _pick_time(time_rng, 13, 19) if time_rng.random() < TIMED_FINISH_SHARE else None
        if actual_finish_d is None:
            finish_time = None
        if actual_start_d and actual_finish_d and actual_finish_d < actual_start_d:
            start_time = finish_time = None          # dates already odd; do not add times to them
        if start_time and finish_time and datetime.combine(actual_finish_d, finish_time) <= datetime.combine(actual_start_d, start_time):
            finish_time = None

        start_value = datetime.combine(actual_start_d, start_time or time(0, 0)) if actual_start_d else None
        finish_value = datetime.combine(actual_finish_d, finish_time or time(0, 0)) if actual_finish_d else None
        start_text = format_actual(start_value, bool(start_time)) if start_value else None
        finish_text = format_actual(finish_value, bool(finish_time)) if finish_value else None
        start_note = f" Work started at {start_time:%H:%M} on {actual_start_d.isoformat()}." if start_time else ""

        statement = _synthetic_statement(
            row, status, progress, finish_text, contractor, delay_category, start_note
        )

        event = {
            "event_id": f"EVT-SEED-{row['l6_id']}",
            "raw_text": statement,
            "activity_description": row["description"],
            "asset": row.get("asset"),
            "discipline": discipline,
            "location": row.get("location"),
            "status": status,
            "actual_start": start_value.isoformat() if start_value else None,
            "actual_finish": finish_value.isoformat() if finish_value else None,
            "actual_start_time_stated": bool(start_time),
            "actual_finish_time_stated": bool(finish_time),
            "progress_percent": progress,
            "contractor": contractor,
            "delay_reason_reported": _DELAY_REASON_TEXT.get(delay_category) if delay_category else None,
            "delay_reason_category": delay_category,
            "evidence_quote": statement,
        }
        match_confidence = round(rng.uniform(0.78, 0.99), 2)
        match_status = "AUTO_ACCEPTED" if match_confidence >= 0.85 else "REVIEW"
        match = {
            "matched_activity_id": row["l6_id"],
            "match_method": "SEMANTIC_HYBRID+LLM (synthetic seed)",
            "confidence": match_confidence,
            "reason": "Synthetic sample data for Institutional Memory, not a real field report.",
            "review_status": match_status,
        }

        saved = save_progress_update(
            event=event,
            match=match,
            reported_by=reporter,
            source_name="synthetic_seed.txt",
            report_date=report_date.isoformat(),
            report_id=f"SEED-{row['l6_id']}",
        )
        if saved["duplicate"]:
            continue  # identical statement already stored; nothing new was written
        summary["seeded"] += 1
        summary["applied" if saved["applied"] else "pending_review"] += 1
        summary["timed_start"] += 1 if start_time else 0
        summary["timed_finish"] += 1 if finish_time else 0
        summary["timed_both"] += 1 if (start_time and finish_time) else 0

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed synthetic sample history for Institutional Memory.")
    parser.add_argument("--reset", action="store_true", help="Wipe existing report history and reset every activity to NOT_STARTED first.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=0.65, help="Fraction of L6 activities to seed history for.")
    args = parser.parse_args()

    initialize_database()  # schema + schedule sync, once, before anything is seeded
    if args.reset:
        reset_history()
        print("Reset: report history cleared, every activity back to NOT_STARTED / 0%.")

    result = seed_synthetic_history(seed=args.seed, fraction=args.fraction)
    print("Seeded synthetic institutional-memory history:", result)


if __name__ == "__main__":
    main()
