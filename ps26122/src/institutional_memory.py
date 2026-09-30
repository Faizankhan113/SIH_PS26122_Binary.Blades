"""Institutional memory: queryable historical execution patterns.

Per the PS 26122 expected outcome, once real progress_updates accumulate
over a project's life, they should become "a growing, queryable repository
of real project execution patterns (actual durations, recurring delay
causes, discipline-wise productivity) that future projects can learn
from". This module implements that query layer on top of the existing
`progress_updates` / `planned_l6_activities` tables -- no new tables are
needed, since the audit schema already captures everything required.

Only updates that really reached the plan count as history: match_status
AUTO_ACCEPTED (accepted automatically, or approved by a supervisor). Pending
REVIEW, REJECTED and UNMATCHED rows are unconfirmed and are left out of the
productivity, delay-cause and search views. `summary_stats()` still
reports how many rows exist in each status.

This module is READ-ONLY. It never writes to the database. Population of
history happens either through real approved updates (future real-write
mode) or, for this demo, through `src/seed_history.py`, which is a
separate, explicitly-invoked step kept apart from the live 3-step
ingestion demo.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from .database import STATUS_AUTO_ACCEPTED, STATUS_REVIEW, get_connection
from .timeutil import actual_day, duration_hours

# SQL fragment: "this history row was applied to the plan".
_APPLIED = f"pu.match_status = '{STATUS_AUTO_ACCEPTED}'"


def _variance_days(actual_finish: str | None, planned_finish: str | None) -> int | None:
    """Finish variance in WHOLE DAYS. A stated time of day never changes it."""
    finish_day = actual_day(actual_finish)
    if finish_day is None or not planned_finish:
        return None
    try:
        return (finish_day - date.fromisoformat(planned_finish)).days
    except ValueError:
        return None


def summary_stats() -> dict[str, Any]:
    """High-level counters for the institutional memory dashboard header."""
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) c FROM progress_updates").fetchone()["c"]
        by_status = conn.execute(
            "SELECT match_status, COUNT(*) c FROM progress_updates GROUP BY match_status"
        ).fetchall()
        avg_conf_row = conn.execute("SELECT AVG(match_confidence) a FROM progress_updates").fetchone()
        # Archived activities (removed from schedule.json) are not part of the
        # live plan, so they are left out of both numbers; their history rows
        # stay visible in search_history() / the per-discipline stats.
        activities_touched = conn.execute(
            "SELECT COUNT(DISTINCT pu.l6_id) c FROM progress_updates pu "
            "JOIN planned_l6_activities pl ON pl.l6_id = pu.l6_id "
            "WHERE pl.archived_at IS NULL"
        ).fetchone()["c"]
        total_activities = conn.execute(
            "SELECT COUNT(*) c FROM planned_l6_activities WHERE archived_at IS NULL"
        ).fetchone()["c"]
        applied = sum(r["c"] for r in by_status if r["match_status"] == STATUS_AUTO_ACCEPTED)
        pending = sum(r["c"] for r in by_status if r["match_status"] == STATUS_REVIEW)
        return {
            "total_updates": total,
            "applied_updates": applied,
            "pending_review": pending,
            "by_match_status": {(r["match_status"] or "UNKNOWN"): r["c"] for r in by_status},
            "avg_match_confidence": round(avg_conf_row["a"], 3) if avg_conf_row["a"] is not None else None,
            "activities_with_history": activities_touched,
            "total_activities": total_activities,
        }
    finally:
        conn.close()


def discipline_productivity() -> list[dict[str, Any]]:
    """Per-discipline execution pattern: outcome mix, schedule variance, productivity ratio."""
    conn = get_connection()
    try:
        rows = conn.execute(
            f"""
            SELECT pu.discipline, pu.status, pu.actual_start, pu.actual_finish,
                   pu.actual_start_time_stated, pu.actual_finish_time_stated,
                   pl.planned_start, pl.planned_finish, pl.planned_duration
            FROM progress_updates pu
            LEFT JOIN planned_l6_activities pl ON pl.l6_id = pu.l6_id
            WHERE pu.discipline IS NOT NULL AND {_APPLIED}
            """
        ).fetchall()
    finally:
        conn.close()

    buckets: dict[str, dict[str, Any]] = {}
    for r in rows:
        d = r["discipline"]
        b = buckets.setdefault(
            d,
            {
                "discipline": d,
                "completed": 0,
                "delayed": 0,
                "in_progress": 0,
                "variances": [],
                "actual_durations": [],
                "planned_durations": [],
                "actual_hours": [],
            },
        )
        if r["status"] == "COMPLETED":
            b["completed"] += 1
        elif r["status"] == "DELAYED":
            b["delayed"] += 1
        elif r["status"] == "IN_PROGRESS":
            b["in_progress"] += 1

        variance = _variance_days(r["actual_finish"], r["planned_finish"])
        if variance is not None:
            b["variances"].append(variance)

        start_day, finish_day = actual_day(r["actual_start"]), actual_day(r["actual_finish"])
        if start_day and finish_day:
            # +1: planned_duration counts both the first and last day (as in
            # schedule.json), so the actual duration must be counted the same
            # way for the ratio below to compare like with like. Whole days only:
            # a stated time of day does not change this number.
            dur = (finish_day - start_day).days + 1
            if dur >= 1:
                b["actual_durations"].append(dur)
                if r["planned_duration"]:
                    b["planned_durations"].append(r["planned_duration"])

        # An actual duration in hours exists only when BOTH times were stated.
        hours = duration_hours(
            r["actual_start"], r["actual_start_time_stated"], r["actual_finish"], r["actual_finish_time_stated"]
        )
        if hours is not None:
            b["actual_hours"].append(hours)

    out = []
    for d, b in buckets.items():
        avg_variance = round(sum(b["variances"]) / len(b["variances"]), 1) if b["variances"] else None
        avg_actual_dur = round(sum(b["actual_durations"]) / len(b["actual_durations"]), 1) if b["actual_durations"] else None
        avg_planned_dur = round(sum(b["planned_durations"]) / len(b["planned_durations"]), 1) if b["planned_durations"] else None
        productivity_ratio = (
            round(avg_planned_dur / avg_actual_dur, 2) if avg_actual_dur and avg_planned_dur else None
        )
        out.append(
            {
                "discipline": d,
                "completed": b["completed"],
                "delayed": b["delayed"],
                "in_progress": b["in_progress"],
                "avg_variance_days": avg_variance,
                "avg_actual_duration_days": avg_actual_dur,
                "avg_planned_duration_days": avg_planned_dur,
                "productivity_ratio": productivity_ratio,
                # None unless at least one update had both times stated.
                "avg_timed_duration_hours": (
                    round(sum(b["actual_hours"]) / len(b["actual_hours"]), 1) if b["actual_hours"] else None
                ),
                "timed_updates": len(b["actual_hours"]),
            }
        )
    out.sort(key=lambda x: (x["discipline"] or ""))
    return out


def delay_reason_breakdown() -> list[dict[str, Any]]:
    """Recurring delay causes, overall and by discipline -- the 'lessons learned' view."""
    conn = get_connection()
    try:
        rows = conn.execute(
            f"""
            SELECT pu.delay_reason_category, pu.discipline, COUNT(*) c
            FROM progress_updates pu
            WHERE pu.delay_reason_category IS NOT NULL AND {_APPLIED}
            GROUP BY pu.delay_reason_category, pu.discipline
            ORDER BY c DESC
            """
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def distinct_filter_values() -> dict[str, list[str]]:
    conn = get_connection()
    try:
        disciplines = [
            r["discipline"]
            for r in conn.execute(
                f"SELECT DISTINCT pu.discipline FROM progress_updates pu WHERE pu.discipline IS NOT NULL AND {_APPLIED} ORDER BY pu.discipline"
            ).fetchall()
        ]
        categories = [
            r["delay_reason_category"]
            for r in conn.execute(
                f"SELECT DISTINCT pu.delay_reason_category FROM progress_updates pu "
                f"WHERE pu.delay_reason_category IS NOT NULL AND {_APPLIED} ORDER BY pu.delay_reason_category"
            ).fetchall()
        ]
        statuses = [
            r["status"]
            for r in conn.execute(
                f"SELECT DISTINCT pu.status FROM progress_updates pu WHERE pu.status IS NOT NULL AND {_APPLIED} ORDER BY pu.status"
            ).fetchall()
        ]
    finally:
        conn.close()
    return {"disciplines": disciplines, "delay_categories": categories, "statuses": statuses}


def search_history(
    discipline: str | None = None,
    delay_category: str | None = None,
    status: str | None = None,
    asset: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """The 'queryable repository' -- filterable list of historical execution facts."""
    conn = get_connection()
    try:
        clauses = [_APPLIED]
        params: list[Any] = []
        if discipline:
            clauses.append("pu.discipline = ?")
            params.append(discipline)
        if delay_category:
            clauses.append("pu.delay_reason_category = ?")
            params.append(delay_category)
        if status:
            clauses.append("pu.status = ?")
            params.append(status)
        if asset:
            clauses.append("pu.asset LIKE ?")
            params.append(f"%{asset}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"""
            SELECT pu.update_id, pu.l6_id, pu.report_date, pu.discipline, pu.asset, pu.location,
                   pu.activity_description, pu.status, pu.progress_percent, pu.actual_start,
                   pu.actual_finish, pu.actual_start_time_stated, pu.actual_finish_time_stated, pu.contractor, pu.delay_reason_category, pu.delay_reason_reported,
                   pu.match_confidence, pu.match_status, pu.reported_by, pu.approved_by,
                   pl.description AS planned_description, pl.planned_start, pl.planned_finish
            FROM progress_updates pu
            LEFT JOIN planned_l6_activities pl ON pl.l6_id = pu.l6_id
            {where}
            ORDER BY pu.report_date DESC, pu.created_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
    finally:
        conn.close()

    out = []
    for r in rows:
        d = dict(r)
        d["variance_days"] = _variance_days(d.get("actual_finish"), d.get("planned_finish"))
        d["actual_duration_hours"] = duration_hours(
            d.get("actual_start"), d.get("actual_start_time_stated"),
            d.get("actual_finish"), d.get("actual_finish_time_stated"),
        )
        out.append(d)
    return out
