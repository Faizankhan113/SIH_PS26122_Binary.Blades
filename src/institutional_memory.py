"""Institutional memory: queryable historical execution patterns.

Per the PS 26122 expected outcome, once real field updates accumulate over a
project's life, they should become "a growing, queryable repository of real
project execution patterns (actual durations, recurring delay causes,
discipline-wise productivity) that future projects can learn from". This
module is that query layer. It needs no tables of its own; it reads:

    main_updates          what each confirmed update did to the live state
    supervisor_decisions  the AUTO_ACCEPT / APPROVE decision behind it
    matching_results      the AI's answer (confidence, method) for the statement
    statement_revisions   the extracted facts (discipline, delay cause, ...)
    reports, activities   report date, reporter, and the planned dates

Only CONFIRMED updates count as history: a `main_updates` row exists only
after an AUTO_ACCEPT or APPROVE decision, whatever its outcome (APPLIED,
HISTORICAL_ONLY or NO_CHANGE; each is a real, confirmed report). Pending
REVIEW, REJECTED, NOT_FOUND and ERROR results are unconfirmed and are left out
of the productivity, delay-cause and search views. `summary_stats()` still
reports how many results exist in each status.

This module is READ-ONLY. It never writes to the database. History is filled
by real approved updates, or optionally with sample data by `src/seed_history.py`, a
separate, explicitly-invoked step.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from .database import (
    STATUS_AUTO_ACCEPTED,
    STATUS_REVIEW,
    _actual_text,
    _compat_status,
    _iso_day,
    get_connection,
)
from .timeutil import actual_day, duration_hours, project_timezone

# Confirmed updates, one row each, with everything the views need. Every query
# below starts from this and adds its own SELECT / WHERE / ORDER BY. (No literal
# percent signs in here: psycopg reads them as parameter markers.)
_CONFIRMED_FROM = """
    FROM main_updates mu
    JOIN supervisor_decisions d  ON d.decision_id = mu.decision_id
    JOIN matching_results mr     ON mr.result_id = d.result_id
    JOIN statement_revisions sr  ON sr.revision_id = mr.revision_id
    JOIN reports rp              ON rp.report_id = mu.report_id
    JOIN activities a            ON a.l6_id = mu.l6_id
    LEFT JOIN users du           ON du.user_id = d.decided_by_user_id
    WHERE d.decision_type IN ('AUTO_ACCEPT', 'APPROVE')
"""


def _as_text_dates(row: dict[str, Any]) -> dict[str, Any]:
    """Turn the timestamptz/date columns of a query row into the plain text the views use.

    actual_start / actual_finish -> 'YYYY-MM-DD' (or 'YYYY-MM-DDTHH:MM:SS' when a
    time of day was stated); planned dates -> 'YYYY-MM-DD'.
    """
    out = dict(row)
    for key in ("actual_start", "actual_finish"):
        out[key] = _actual_text(row.get(key), row.get(f"{key}_time_stated"))
    for key in ("planned_start", "planned_finish"):
        if key in row:
            out[key] = _iso_day(row[key])
    return out


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
        status_rows = conn.execute(
            "SELECT ai_status, final_status, COUNT(*) AS c FROM matching_results GROUP BY ai_status, final_status"
        ).fetchall()
        avg_conf = conn.execute("SELECT AVG(confidence) AS a FROM matching_results").fetchone()["a"]
        # Archived activities (removed from schedule.json) are not part of the
        # live plan, so they are left out of both numbers; their history rows
        # stay visible in search_history() / the per-discipline stats.
        activities_touched = conn.execute(
            "SELECT COUNT(DISTINCT mu.l6_id) AS c FROM main_updates mu "
            "JOIN activities a ON a.l6_id = mu.l6_id WHERE a.archived_at IS NULL"
        ).fetchone()["c"]
        total_activities = conn.execute(
            "SELECT COUNT(*) AS c FROM activities WHERE archived_at IS NULL"
        ).fetchone()["c"]
    finally:
        conn.close()

    by_status: dict[str, int] = {}
    for r in status_rows:
        label = _compat_status(r["ai_status"], r["final_status"])
        by_status[label] = by_status.get(label, 0) + r["c"]
    return {
        "total_updates": sum(by_status.values()),
        "applied_updates": by_status.get(STATUS_AUTO_ACCEPTED, 0),
        "pending_review": by_status.get(STATUS_REVIEW, 0),
        "by_match_status": by_status,
        "avg_match_confidence": round(float(avg_conf), 3) if avg_conf is not None else None,
        "activities_with_history": activities_touched,
        "total_activities": total_activities,
    }


def discipline_productivity() -> list[dict[str, Any]]:
    """Per-discipline execution pattern: outcome mix, schedule variance, productivity ratio."""
    conn = get_connection()
    try:
        rows = [
            _as_text_dates(r)
            for r in conn.execute(
                f"""
                SELECT sr.discipline, sr.status, sr.actual_start, sr.actual_finish,
                       sr.actual_start_time_stated, sr.actual_finish_time_stated,
                       a.planned_start, a.planned_finish, a.planned_duration
                {_CONFIRMED_FROM}
                  AND sr.discipline IS NOT NULL
                """
            ).fetchall()
        ]
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
            SELECT sr.delay_reason_category, sr.discipline, COUNT(*) AS c
            {_CONFIRMED_FROM}
              AND sr.delay_reason_category IS NOT NULL
            GROUP BY sr.delay_reason_category, sr.discipline
            ORDER BY c DESC, sr.delay_reason_category, sr.discipline
            """
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _distinct(conn, column: str) -> list[str]:
    """Distinct non-null values of one statement_revisions column among confirmed updates."""
    # `column` is one of the three fixed names below, never user input.
    rows = conn.execute(
        f"SELECT DISTINCT sr.{column} AS v {_CONFIRMED_FROM} AND sr.{column} IS NOT NULL ORDER BY v"
    ).fetchall()
    return [r["v"] for r in rows]


def distinct_filter_values() -> dict[str, list[str]]:
    conn = get_connection()
    try:
        return {
            "disciplines": _distinct(conn, "discipline"),
            "delay_categories": _distinct(conn, "delay_reason_category"),
            "statuses": _distinct(conn, "status"),
        }
    finally:
        conn.close()


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
        clauses: list[str] = []
        params: list[Any] = []
        if discipline:
            clauses.append("sr.discipline = %s")
            params.append(discipline)
        if delay_category:
            clauses.append("sr.delay_reason_category = %s")
            params.append(delay_category)
        if status:
            clauses.append("sr.status = %s")
            params.append(status)
        if asset:
            clauses.append("sr.asset ILIKE %s")
            params.append(f"%{asset}%")
        extra = "".join(f" AND {c}" for c in clauses)
        rows = conn.execute(
            f"""
            SELECT mr.result_id AS update_id, mu.l6_id, rp.report_datetime,
                   sr.discipline, sr.asset, sr.location, sr.activity_description, sr.status,
                   sr.progress_percent, sr.actual_start, sr.actual_finish,
                   sr.actual_start_time_stated, sr.actual_finish_time_stated,
                   sr.contractor, sr.delay_reason_category, sr.delay_reason_reported,
                   mr.confidence AS match_confidence, mr.matched_activity_id,
                   rp.reported_by_name AS reported_by,
                   CASE WHEN d.decision_type = 'AUTO_ACCEPT' THEN 'SYSTEM' ELSE du.name END AS approved_by,
                   mu.outcome,
                   a.description AS planned_description, a.planned_start, a.planned_finish
            {_CONFIRMED_FROM}
            {extra}
            ORDER BY rp.report_datetime DESC, mu.created_at DESC, mu.update_id
            LIMIT %s
            """,
            (*params, limit),
        ).fetchall()
    finally:
        conn.close()

    tz = project_timezone()
    out = []
    for r in rows:
        d = _as_text_dates(r)
        d["report_date"] = r["report_datetime"].astimezone(tz).date().isoformat()
        d["progress_percent"] = float(r["progress_percent"]) if r["progress_percent"] is not None else None
        d["match_confidence"] = float(r["match_confidence"]) if r["match_confidence"] is not None else None
        d["match_status"] = STATUS_AUTO_ACCEPTED
        d["variance_days"] = _variance_days(d.get("actual_finish"), d.get("planned_finish"))
        d["actual_duration_hours"] = duration_hours(
            d.get("actual_start"), d.get("actual_start_time_stated"),
            d.get("actual_finish"), d.get("actual_finish_time_stated"),
        )
        out.append(d)
    return out
