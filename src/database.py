from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg
from psycopg import errors as pg_errors
from psycopg.types.json import Jsonb

from . import db
from .timeutil import canonical_actual, parse_actual, project_timezone, today_local

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_PATH = PROJECT_ROOT / "data" / "schedule.json"

# Used only for a schedule row that has no project_id of its own.
DEFAULT_PROJECT_ID_ENV = "PS26122_DEFAULT_PROJECT_ID"
DEFAULT_PROJECT_ID = "GPC-001"

# The planned (baseline) columns that come from schedule.json. Execution state
# (actual_*, progress, status, contractor, update_ref) is never touched by a
# schedule sync.
PLANNED_FIELDS = (
    "project_id",
    "wbs_code",
    "activity_code",
    "description",
    "discipline",
    "asset",
    "location",
    "planned_start",
    "planned_finish",
    "planned_duration",
)

# The status words the matcher, the web app and the templates use. They are the
# "compatibility" vocabulary of this module (see PG_PLAN.md section 1).
STATUS_AUTO_ACCEPTED = "AUTO_ACCEPTED"   # applied (automatically, or after a supervisor approved it)
STATUS_REVIEW = "REVIEW"                 # waiting for a supervisor; nothing has changed in `main`
STATUS_REJECTED = "REJECTED"             # a supervisor said no; `main` was never changed
STATUS_UNMATCHED = "UNMATCHED"           # matcher's word for "no activity found"; stored as NOT_FOUND
STATUS_ERROR = "ERROR"                   # the matcher itself failed; stored as ERROR (never NOT_FOUND)

# What is stored in matching_results.ai_status. UNMATCHED (the matcher's word)
# is stored as NOT_FOUND; the compatibility layer turns it back on the way out.
AI_AUTO_ACCEPTED = "AUTO_ACCEPTED"
AI_REVIEW = "REVIEW"
AI_NOT_FOUND = "NOT_FOUND"
AI_ERROR = "ERROR"
_AI_STATUS_FROM_LABEL = {
    STATUS_AUTO_ACCEPTED: AI_AUTO_ACCEPTED,
    STATUS_REVIEW: AI_REVIEW,
    STATUS_UNMATCHED: AI_NOT_FOUND,
    "NOT_FOUND": AI_NOT_FOUND,
    STATUS_ERROR: AI_ERROR,
}

# supervisor_decisions.decision_type
DECISION_AUTO_ACCEPT = "AUTO_ACCEPT"
DECISION_APPROVE = "APPROVE"
DECISION_REJECT = "REJECT"
DECISION_DISMISS = "DISMISS"
DECISION_MARK_NEW = "MARK_NEW_ACTIVITY"
# A result gets at most one of these (same list as ux_supervisor_decisions_terminal).
_TERMINAL_DECISIONS = (DECISION_AUTO_ACCEPT, DECISION_APPROVE, DECISION_REJECT, DECISION_DISMISS, DECISION_MARK_NEW)

# main_updates.outcome
OUTCOME_APPLIED = "APPLIED"
OUTCOME_HISTORICAL = "HISTORICAL_ONLY"   # a newer report already set `main`
OUTCOME_NO_CHANGE = "NO_CHANGE"          # the report agrees with what `main` already holds

# Planner view of a NOT_FOUND item. It is derived from supervisor_decisions
# (no decision = open), so there is no planner column to keep in step.
PLANNER_OPEN = "open"
PLANNER_DISMISSED = "dismissed"
PLANNER_CONVERTED = "converted"      # "marked as new activity": a note only, the schedule is not edited
PLANNER_STATUSES = (PLANNER_OPEN, PLANNER_DISMISSED, PLANNER_CONVERTED)
_PLANNER_FROM_DECISION = {DECISION_DISMISS: PLANNER_DISMISSED, DECISION_MARK_NEW: PLANNER_CONVERTED}

DEGRADED_REASON = "Extraction failed, fields are keyword guesses."


class DegradedConfirmationRequired(ValueError):
    """Approving a degraded-extraction item needs an explicit extra confirmation."""


def is_degraded(event: dict[str, Any] | None) -> bool:
    return bool(event and event.get("extraction_degraded"))


def _degraded_reason(reason: str | None) -> str:
    """Put the degraded warning in front of the existing reason (once)."""
    if reason and DEGRADED_REASON in reason:
        return reason
    return DEGRADED_REASON + (f" {reason}" if reason else "")


# NOTE ON WHEN THE DATABASE IS INITIALIZED
# ----------------------------------------
# initialize_database() (schema + migrations + schedule sync) is meant to run
# ONCE per process start -- from the web app at startup, or from
# `python -m src.bootstrap` / the seed scripts. The getters and save functions
# below deliberately do NOT call it: doing so on every request re-read
# schedule.json and re-wrote every planned row for every matched event.


def get_connection() -> psycopg.Connection:
    """A new PostgreSQL connection (rows are dicts). The caller closes it.

    See src/db.py. Autocommit is off: call conn.commit() to save changes.
    """
    return db.get_connection()


def default_project_id() -> str:
    return (os.getenv(DEFAULT_PROJECT_ID_ENV) or DEFAULT_PROJECT_ID).strip()


def initialize_database() -> dict[str, int]:
    """Create any missing tables and sync the planned schedule.

    Safe to call repeatedly: it only writes when something actually changed.
    Returns the schedule-sync summary (see `_sync_schedule`).
    """
    db.init_schema()
    conn = get_connection()
    try:
        summary = _sync_schedule(conn)
        conn.commit()
        return summary
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def sync_schedule() -> dict[str, int]:
    """Re-read schedule.json into the database (insert / update / archive / restore)."""
    conn = get_connection()
    try:
        summary = _sync_schedule(conn)
        conn.commit()
        return summary
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Value adapters: PostgreSQL types -> the plain values the rest of the app expects
#
# The matcher, the templates and find_conflicts() were written against text
# dates, 0/1 flags and float progress. These keep that shape, so callers do not
# change when the storage does.
# --------------------------------------------------------------------------

def _iso_day(value: Any) -> str | None:
    """A date column -> 'YYYY-MM-DD' (or None)."""
    return value.isoformat() if value is not None else None


def _iso_stamp(value: datetime | None) -> str | None:
    """A timestamptz audit column -> ISO text in project-local time."""
    if value is None:
        return None
    return value.astimezone(project_timezone()).isoformat(timespec="seconds")


def _actual_text(value: datetime | None, time_stated: Any) -> str | None:
    """An actual start/finish -> 'YYYY-MM-DD', or 'YYYY-MM-DDTHH:MM:SS' when a time was stated.

    Same convention as timeutil.canonical_actual(): the time of day is only
    shown when the report really gave one.
    """
    if value is None:
        return None
    local = value.astimezone(project_timezone())
    if bool(time_stated):
        return local.replace(tzinfo=None, microsecond=0).isoformat(timespec="seconds")
    return local.date().isoformat()


def _load_schedule_rows() -> list[dict[str, Any]] | None:
    """The L6 rows from schedule.json, or None when the file does not exist."""
    if not SCHEDULE_PATH.exists():
        return None
    schedule = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(schedule):
        if item.get("level") != "L6":
            continue
        activity_id = item.get("activity_id")
        if not activity_id:
            raise ValueError(f"schedule.json entry #{index} has no activity_id")
        if activity_id in seen:
            raise ValueError(f"schedule.json has a duplicate activity_id: {activity_id}")
        seen.add(activity_id)
        rows.append(item)
    return rows


def _planned_values(item: dict[str, Any]) -> dict[str, Any]:
    """Map one schedule.json row to the planned columns stored in the database."""
    planned_duration = item.get("planned_duration")
    if planned_duration is None:
        # Only compute a duration when the file does not provide one.
        planned_duration = _duration(item.get("planned_start"), item.get("planned_finish"))
    return {
        "project_id": item.get("project_id") or default_project_id(),
        "wbs_code": item.get("wbs_code"),
        "activity_code": item.get("activity_code") or item["activity_id"],
        "description": item["description"],
        "discipline": item.get("discipline"),
        "asset": item.get("asset"),
        "location": item.get("location"),
        "planned_start": item.get("planned_start"),
        "planned_finish": item.get("planned_finish"),
        "planned_duration": planned_duration,
    }


def _pg_planned(planned: dict[str, Any]) -> dict[str, Any]:
    """schedule.json text values -> the types stored in `activities` (dates, integer)."""
    out = dict(planned)
    for col in ("planned_start", "planned_finish"):
        value = out.get(col)
        out[col] = date.fromisoformat(str(value)[:10]) if value else None
    if out.get("planned_duration") is not None:
        out["planned_duration"] = int(out["planned_duration"])
    return out


def _sync_schedule(conn: psycopg.Connection, source_rows: list[dict[str, Any]] | None = None) -> dict[str, int]:
    """Make `activities` match schedule.json, and make sure every activity has a `main` row.

    - new activity            -> inserted
    - changed planned fields  -> updated (live state in `main` is untouched)
    - unchanged               -> left completely alone (updated_at is NOT bumped)
    - missing from the file   -> archived (archived_at set; never deleted)
    - archived, back in file  -> restored (archived_at cleared)
    - activity without a main row -> main row created (NOT_STARTED, 0%)

    Does not commit: the caller does.
    """
    summary = {"inserted": 0, "updated": 0, "unchanged": 0, "archived": 0, "restored": 0, "main_created": 0}
    if source_rows is None:
        source_rows = _load_schedule_rows()
    if source_rows is None:
        return summary

    existing = {row["l6_id"]: row for row in conn.execute("SELECT * FROM activities").fetchall()}

    for item in source_rows:
        l6_id = item["activity_id"]
        planned = _pg_planned(_planned_values(item))
        current = existing.get(l6_id)

        if current is None:
            columns = ["l6_id", *planned]
            conn.execute(
                f"INSERT INTO activities ({', '.join(columns)}) "
                f"VALUES ({', '.join(['%s'] * len(columns))})",
                [l6_id, *planned.values()],
            )
            summary["inserted"] += 1
            continue

        was_archived = current["archived_at"] is not None
        changed = any(current[col] != value for col, value in planned.items())
        if not changed and not was_archived:
            summary["unchanged"] += 1
            continue

        assignments = ", ".join(f"{col} = %s" for col in planned)
        conn.execute(
            f"UPDATE activities SET {assignments}, archived_at = NULL, updated_at = now() WHERE l6_id = %s",
            [*planned.values(), l6_id],
        )
        summary["restored" if was_archived else "updated"] += 1

    if not source_rows:
        # An empty/wrong schedule file must not silently hide the whole plan.
        logger.warning("schedule.json contains no L6 activities; skipping archive step.")
    else:
        source_ids = {item["activity_id"] for item in source_rows}
        for l6_id, current in existing.items():
            if l6_id not in source_ids and current["archived_at"] is None:
                conn.execute(
                    "UPDATE activities SET archived_at = now(), updated_at = now() WHERE l6_id = %s",
                    (l6_id,),
                )
                summary["archived"] += 1

    # Every activity (archived ones too: their history stays) has exactly one live-state row.
    summary["main_created"] = conn.execute(
        "INSERT INTO main (l6_id) SELECT l6_id FROM activities ON CONFLICT (l6_id) DO NOTHING"
    ).rowcount
    return summary


def _duration(start: str | None, finish: str | None) -> int | None:
    """Planned duration in days, counting both the start and the finish day.

    Same convention as schedule.json (15 Jun to 24 Jun is 10 days). Only used
    when a schedule row does not carry its own `planned_duration`.
    """
    if not start or not finish:
        return None
    try:
        return (date.fromisoformat(finish) - date.fromisoformat(start)).days + 1
    except ValueError:
        return None


def list_active_schedule() -> list[dict[str, Any]]:
    """Planned L6 activities the matcher may choose from (archived ones excluded).

    Dates are 'YYYY-MM-DD' text. `status` is the live status from `main`.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT a.l6_id AS activity_id, 'L6' AS level, a.wbs_code, a.activity_code, a.description,
                   a.discipline, a.asset, a.location,
                   to_char(a.planned_start, 'YYYY-MM-DD')  AS planned_start,
                   to_char(a.planned_finish, 'YYYY-MM-DD') AS planned_finish,
                   COALESCE(m.status, 'NOT_STARTED')       AS status
            FROM activities a
            LEFT JOIN main m ON m.l6_id = a.l6_id
            WHERE a.archived_at IS NULL
            ORDER BY a.l6_id COLLATE "C"
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_planned_activity(l6_id: str | None) -> dict[str, Any] | None:
    """One activity with its live state, in the shape the app has always used.

    Planned fields come from `activities`, execution state from `main`.
    `update_ref` is the matching result whose decision last set the live state
    (the old "update id"), or None when nothing has been applied yet.
    """
    if not l6_id:
        return None
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT a.l6_id, a.project_id, a.wbs_code, a.activity_code, a.description,
                   a.discipline, a.asset, a.location,
                   a.planned_start, a.planned_finish, a.planned_duration,
                   a.archived_at, a.created_at, a.updated_at,
                   m.actual_start, m.actual_finish,
                   m.actual_start_time_stated, m.actual_finish_time_stated,
                   m.progress_percent, m.status, m.contractor,
                   d.result_id AS update_ref
            FROM activities a
            LEFT JOIN main m ON m.l6_id = a.l6_id
            LEFT JOIN supervisor_decisions d ON d.decision_id = m.last_decision_id
            WHERE a.l6_id = %s
            """,
            (l6_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "l6_id": row["l6_id"],
        "project_id": row["project_id"],
        "wbs_code": row["wbs_code"],
        "activity_code": row["activity_code"],
        "description": row["description"],
        "discipline": row["discipline"],
        "asset": row["asset"],
        "location": row["location"],
        "planned_start": _iso_day(row["planned_start"]),
        "planned_finish": _iso_day(row["planned_finish"]),
        "planned_duration": row["planned_duration"],
        "actual_start": _actual_text(row["actual_start"], row["actual_start_time_stated"]),
        "actual_finish": _actual_text(row["actual_finish"], row["actual_finish_time_stated"]),
        "actual_start_time_stated": int(bool(row["actual_start_time_stated"])),
        "actual_finish_time_stated": int(bool(row["actual_finish_time_stated"])),
        "progress_percent": float(row["progress_percent"]) if row["progress_percent"] is not None else 0.0,
        "status": row["status"] or "NOT_STARTED",
        "contractor": row["contractor"],
        "update_ref": row["update_ref"],
        "archived_at": _iso_stamp(row["archived_at"]),
        "created_at": _iso_stamp(row["created_at"]),
        "updated_at": _iso_stamp(row["updated_at"]),
    }


# --------------------------------------------------------------------------
# Writing progress updates
#
# The rules, in one place:
#   * `main` (the live state) changes only through _record_main_update(), and
#     only after a row exists in supervisor_decisions:
#       - a confident match         -> a system decision (AUTO_ACCEPT, no user)
#       - a supervisor's approval   -> an APPROVE decision (decide_review_update)
#     Both go through the same function, so there is one piece of update logic.
#   * REVIEW and NOT_FOUND matches are stored but never touch `main`.
#   * A confident match that conflicts with the live state is NOT applied; it
#     is stored as REVIEW with the reason spelled out.
#   * Newest report wins: a report older than the one that last set an
#     activity is recorded as HISTORICAL_ONLY; a report that agrees with the
#     live state is recorded as NO_CHANGE. Neither changes `main`.
#   * The same statement (report date + reporter + text) is stored once.
#   * Every save is ONE transaction: all rows are written, or none are.
#
# Public functions keep their old names and return shapes (the compatibility
# layer): "update_id" is the id of the matching result, and rows come back with
# the column names the web app and templates have always used.
# --------------------------------------------------------------------------

_NON_WORD = re.compile(r"[\W_]+")


def normalize_statement(text: str | None) -> str:
    """Case, punctuation and spacing do not make a statement 'different'."""
    return _NON_WORD.sub(" ", (text or "").casefold()).strip()


def compute_fingerprint(report_date: str | None, reporter: str | None, raw_text: str | None) -> str | None:
    """Hash of report date + reporter + normalized statement text.

    Returns None for an empty statement: there is nothing to compare, so such
    rows are never treated as duplicates of each other.
    """
    text = normalize_statement(raw_text)
    if not text:
        return None
    parts = [str(report_date or "").strip(), normalize_statement(reporter), text]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _day(value: Any) -> str:
    """The date part of a stored date or date-time (so 2026-09-22 == 2026-09-22T18:30)."""
    return str(value)[:10]


def find_conflicts(planned: dict[str, Any], event: dict[str, Any]) -> list[str]:
    """Reasons why applying `event` to the current state `planned` is unsafe.

    An empty list means it is a normal forward update. Checked:
      1. progress would go down,
      2. a different finish date than the one already recorded,
      3. status would move backwards from COMPLETED.
    """
    reasons: list[str] = []

    new_progress = event.get("progress_percent")
    current_progress = planned.get("progress_percent") or 0.0
    if new_progress is not None and float(new_progress) < float(current_progress):
        reasons.append(f"progress would drop from {float(current_progress):g}% to {float(new_progress):g}%")

    new_finish, current_finish = event.get("actual_finish"), planned.get("actual_finish")
    if new_finish and current_finish and _day(new_finish) != _day(current_finish):
        reasons.append(f"a finish date of {_day(current_finish)} is already recorded but this report says {_day(new_finish)}")

    new_status = event.get("status")
    if planned.get("status") == "COMPLETED" and new_status and new_status != "COMPLETED":
        reasons.append(f"status would move back from COMPLETED to {new_status}")

    return reasons


MAX_ALTERNATIVES = 3   # How many other candidates a reviewer is shown
MAX_TOP_CANDIDATES = 3  # How many candidates are stored with a matching result


def build_top_candidates(match: dict[str, Any]) -> list[dict[str, Any]]:
    """The top MAX_TOP_CANDIDATES entries of the match shortlist, with scores.

    Order is the shortlist order (best retrieval first, ID matches first). The
    activity the matcher chose is included when it is among them, so the stored
    list is the matcher's full evidence; `alternatives` (what a reviewer sees
    as "other candidates") leaves the chosen one out again.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cand in match.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        activity_id = cand.get("activity_id")
        if not activity_id or activity_id in seen:
            continue
        seen.add(activity_id)
        score = cand.get("retrieval_score")
        out.append({
            "activity_id": activity_id,
            "description": cand.get("description"),
            "score": round(float(score), 4) if isinstance(score, (int, float)) else None,
        })
        if len(out) == MAX_TOP_CANDIDATES:
            break
    return out


def build_alternatives(match: dict[str, Any], exclude_id: str | None = None) -> list[dict[str, Any]]:
    """Up to MAX_ALTERNATIVES candidates from the match shortlist, without `exclude_id`."""
    return _alternatives_from(build_top_candidates(match), exclude_id)


def _alternatives_from(top_candidates: Any, exclude_id: str | None) -> list[dict[str, Any]]:
    if isinstance(top_candidates, str):
        top_candidates = parse_alternatives(top_candidates)
    out = [c for c in (top_candidates or []) if isinstance(c, dict) and c.get("activity_id") != exclude_id]
    return out[:MAX_ALTERNATIVES]


def parse_alternatives(raw: Any) -> list[dict[str, Any]]:
    """A stored candidate list (jsonb list, or JSON text) -> a list of dicts. Damaged input gives []."""
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return []
    return [a for a in raw if isinstance(a, dict)] if isinstance(raw, list) else []


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10].upper()}"


def _report_datetime(report_date: Any) -> datetime:
    """D6: the report date at local midnight (project timezone) when no time is given."""
    parsed, _has_time = parse_actual(report_date)
    if parsed is None:
        parsed = datetime.combine(today_local(), datetime.min.time())
    return parsed.replace(tzinfo=project_timezone())


def _stamp(value: Any, time_stated: Any) -> tuple[datetime | None, bool]:
    """D7, the one adapter: a canonical actual start/finish -> (timestamptz value, time-stated flag).

    Goes through canonical_actual() first, so a time of day survives only when
    the report really stated one; a date alone becomes local midnight with the
    flag False.
    """
    text, flag = canonical_actual(value, time_stated)
    if text is None:
        return None, False
    parsed, _ = parse_actual(text)
    return parsed.replace(tzinfo=project_timezone()), bool(flag)


def _pct(value: Any) -> Decimal | None:
    """Progress as a 2-decimal Decimal (what numeric(5,2) stores), or None."""
    if value is None:
        return None
    return Decimal(str(value)).quantize(Decimal("0.01"))


def _reported_from_event(event: dict[str, Any]) -> dict[str, Any]:
    """The facts of a progress event in the form `main` stores them."""
    start, start_flag = _stamp(event.get("actual_start"), event.get("actual_start_time_stated"))
    finish, finish_flag = _stamp(event.get("actual_finish"), event.get("actual_finish_time_stated"))
    return {
        "actual_start": start, "actual_start_time_stated": start_flag,
        "actual_finish": finish, "actual_finish_time_stated": finish_flag,
        "progress_percent": _pct(event.get("progress_percent")),
        "status": event.get("status") or None,
        "contractor": event.get("contractor") or None,
    }


def _reported_from_revision(rev: dict[str, Any]) -> dict[str, Any]:
    """The same shape, read from a statement_revisions row."""
    return {
        "actual_start": rev["actual_start"], "actual_start_time_stated": bool(rev["actual_start_time_stated"]),
        "actual_finish": rev["actual_finish"], "actual_finish_time_stated": bool(rev["actual_finish_time_stated"]),
        "progress_percent": _pct(rev["progress_percent"]),
        "status": rev["status"] or None,
        "contractor": rev["contractor"] or None,
    }


_LIVE_FIELDS = (
    "actual_start", "actual_start_time_stated", "actual_finish", "actual_finish_time_stated",
    "progress_percent", "status", "contractor",
)


def _live_values(main_row: dict[str, Any]) -> dict[str, Any]:
    """The live state of an activity, in the same shape as a reported event."""
    return {
        "actual_start": main_row["actual_start"], "actual_start_time_stated": bool(main_row["actual_start_time_stated"]),
        "actual_finish": main_row["actual_finish"], "actual_finish_time_stated": bool(main_row["actual_finish_time_stated"]),
        "progress_percent": _pct(main_row["progress_percent"]),
        "status": main_row["status"],
        "contractor": main_row["contractor"],
    }


def plan_main_change(live: dict[str, Any], last_report_datetime: datetime | None,
                     reported: dict[str, Any], report_datetime: datetime) -> tuple[str, dict[str, Any]]:
    """Decide what a report does to an activity. Pure: no database, no clock.

    Returns (outcome, new_values):
      HISTORICAL_ONLY  the report is OLDER than the one that last set the
                       activity (newest report wins; the same moment counts as
                       newer, because the later approval wins a tie). new == live.
      NO_CHANGE        merging the report changes nothing. new == live.
      APPLIED          new = live with the report's non-empty fields on top.

    Empty fields in the report keep the live value; a time-stated flag moves
    together with its date (and stays as it was when the report has no date).
    """
    if last_report_datetime is not None and report_datetime < last_report_datetime:
        return OUTCOME_HISTORICAL, dict(live)
    new = dict(live)
    for key in ("actual_start", "actual_finish"):
        if reported[key] is not None:
            new[key] = reported[key]
            new[f"{key}_time_stated"] = reported[f"{key}_time_stated"]
    for key in ("progress_percent", "status", "contractor"):
        if reported[key] is not None:
            new[key] = reported[key]
    if all(new[k] == live[k] for k in _LIVE_FIELDS):
        return OUTCOME_NO_CHANGE, dict(live)
    return OUTCOME_APPLIED, new


def _lock_main(conn: psycopg.Connection, l6_id: str) -> dict[str, Any]:
    """Lock and read the live-state row of one activity (held until commit/rollback)."""
    row = conn.execute("SELECT * FROM main WHERE l6_id = %s FOR UPDATE", (l6_id,)).fetchone()
    if row is None:
        raise ValueError(f"Matched activity {l6_id!r} does not exist in the schedule")
    return row


def _record_main_update(
    conn: psycopg.Connection,
    *,
    main_row: dict[str, Any],
    decision_id: str,
    report_id: str,
    statement_id: str,
    report_datetime: datetime,
    reported: dict[str, Any],
) -> dict[str, Any]:
    """THE ONLY place that writes the live state. Needs a decision that already exists.

    `main_row` is the row returned by _lock_main() in this same transaction.
    Writes one main_updates row (previous, reported and new values) and, only
    when the outcome is APPLIED, the new values into `main`.
    """
    live = _live_values(main_row)
    outcome, new = plan_main_change(live, main_row["last_report_datetime"], reported, report_datetime)
    l6_id = main_row["l6_id"]
    update_id = _new_id("MUP")
    conn.execute(
        """
        INSERT INTO main_updates (
            update_id, decision_id, l6_id, report_id, statement_id, outcome,
            prev_actual_start, prev_actual_finish, prev_progress_percent, prev_status,
            reported_actual_start, reported_actual_finish, reported_progress_percent, reported_status,
            new_actual_start, new_actual_finish, new_progress_percent, new_status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            update_id, decision_id, l6_id, report_id, statement_id, outcome,
            live["actual_start"], live["actual_finish"], live["progress_percent"], live["status"],
            reported["actual_start"], reported["actual_finish"], reported["progress_percent"], reported["status"],
            new["actual_start"], new["actual_finish"], new["progress_percent"], new["status"],
        ),
    )
    if outcome == OUTCOME_APPLIED:
        conn.execute(
            """
            UPDATE main
            SET actual_start = %s, actual_start_time_stated = %s,
                actual_finish = %s, actual_finish_time_stated = %s,
                progress_percent = %s, status = %s, contractor = %s,
                last_report_id = %s, last_statement_id = %s, last_decision_id = %s,
                last_report_datetime = %s, updated_at = now()
            WHERE l6_id = %s
            """,
            (
                new["actual_start"], new["actual_start_time_stated"],
                new["actual_finish"], new["actual_finish_time_stated"],
                new["progress_percent"], new["status"], new["contractor"],
                report_id, statement_id, decision_id, report_datetime, l6_id,
            ),
        )
    return {"update_id": update_id, "outcome": outcome, "prev": live, "new": new}


def _insert_decision(
    conn: psycopg.Connection,
    *,
    result_id: str,
    decision_type: str,
    user_id: str | None,
    target: str | None = None,
    note: str | None = None,
    degraded_confirmed: bool = False,
) -> dict[str, Any]:
    """Append one row to supervisor_decisions. The caller is inside a transaction."""
    decision_id = _new_id("DEC")
    row = conn.execute(
        """
        INSERT INTO supervisor_decisions
            (decision_id, result_id, decision_type, decided_by_user_id, target_activity_id, note, degraded_confirmed)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING created_at
        """,
        (decision_id, result_id, decision_type, user_id, target, note, degraded_confirmed),
    ).fetchone()
    return {"decision_id": decision_id, "created_at": row["created_at"]}


def _set_verdict(conn: psycopg.Connection, result_id: str, final_status: str, final_activity_id: str | None) -> None:
    """Fill the verdict summary on a matching result. The AI columns are never touched."""
    conn.execute(
        "UPDATE matching_results SET final_status = %s, final_activity_id = %s, final_decided_at = now() "
        "WHERE result_id = %s",
        (final_status, final_activity_id, result_id),
    )


# --------------------------------------------------------------------------
# The compatibility layer: matching results read back as "progress update" dicts
# --------------------------------------------------------------------------

_TERMINAL_SQL = ", ".join(f"'{t}'" for t in _TERMINAL_DECISIONS)

# One row per matching result, joined to everything the web app shows about it.
_UPDATE_SELECT = f"""
    SELECT mr.result_id, mr.statement_id, mr.revision_id, mr.attempt_no, mr.ai_status,
           mr.matched_activity_id, mr.confidence, mr.match_method, mr.reason, mr.top_candidates,
           mr.final_status, mr.final_activity_id, mr.created_at AS result_created_at,
           st.fingerprint, st.raw_text, st.extraction_degraded, st.extraction_error,
           rp.report_id, rp.source_type, rp.source_name, rp.report_datetime,
           rp.submitted_by_user_id, rp.reported_by_name,
           rv.activity_description, rv.asset, rv.discipline, rv.location,
           rv.actual_start, rv.actual_finish, rv.actual_start_time_stated, rv.actual_finish_time_stated,
           rv.progress_percent, rv.status AS event_status, rv.contractor,
           rv.delay_reason_reported, rv.delay_reason_category, rv.evidence_quote, rv.activity_ref,
           d.decision_id, d.decision_type, d.decided_by_user_id, d.note AS decision_note,
           d.created_at AS decided_at, du.name AS decided_by_name,
           ta.description AS target_description, ta.planned_finish AS target_planned_finish,
           tm.status AS target_status, tm.progress_percent AS target_progress
    FROM matching_results mr
    JOIN report_statements st ON st.statement_id = mr.statement_id
    JOIN reports rp ON rp.report_id = st.report_id
    JOIN statement_revisions rv ON rv.revision_id = mr.revision_id
    LEFT JOIN supervisor_decisions d
           ON d.result_id = mr.result_id
          AND d.decision_type IN ({_TERMINAL_SQL})
    LEFT JOIN users du ON du.user_id = d.decided_by_user_id
    LEFT JOIN activities ta ON ta.l6_id = mr.matched_activity_id
    LEFT JOIN main tm ON tm.l6_id = mr.matched_activity_id
"""


def _compat_status(ai_status: str, final_status: str) -> str:
    """matching_results (ai_status, final_status) -> the old match_status word."""
    if ai_status == AI_ERROR:
        return STATUS_ERROR
    if ai_status == AI_NOT_FOUND:
        return STATUS_UNMATCHED
    if final_status in ("AUTO_ACCEPTED", "APPROVED"):
        return STATUS_AUTO_ACCEPTED
    if final_status == "REJECTED":
        return STATUS_REJECTED
    return STATUS_REVIEW


def _shape_update(row: dict[str, Any]) -> dict[str, Any]:
    """One _UPDATE_SELECT row -> the dict the web app and templates have always used."""
    approved = row["final_status"] in ("AUTO_ACCEPTED", "APPROVED")
    pending = row["ai_status"] == AI_REVIEW and row["final_status"] == "PENDING"
    decision_type = row["decision_type"]
    if decision_type == DECISION_AUTO_ACCEPT:
        approved_by: str | None = "SYSTEM"
    elif decision_type == DECISION_APPROVE:
        approved_by = row["decided_by_name"]
    else:
        approved_by = None
    planner_status = None
    if row["ai_status"] == AI_NOT_FOUND:
        planner_status = _PLANNER_FROM_DECISION.get(decision_type, PLANNER_OPEN)
    is_review_decision = decision_type in (DECISION_APPROVE, DECISION_REJECT)
    is_planner_decision = decision_type in _PLANNER_FROM_DECISION
    top = parse_alternatives(row["top_candidates"])
    progress = row["progress_percent"]
    target = row["matched_activity_id"]
    tz = project_timezone()
    return {
        "update_id": row["result_id"],
        "statement_id": row["statement_id"],
        "l6_id": row["final_activity_id"] if approved else None,
        "pending_l6_id": target if pending else None,
        "target_l6_id": target,
        "fingerprint": row["fingerprint"],
        "report_id": row["report_id"],
        "source_type": row["source_type"],
        "source_file": row["source_name"],
        "report_date": row["report_datetime"].astimezone(tz).date().isoformat(),
        "received_at": _iso_stamp(row["result_created_at"]),
        "reported_by": row["reported_by_name"],
        "reported_by_user_id": row["submitted_by_user_id"],
        "approved_by": approved_by,
        "approved_by_user_id": row["decided_by_user_id"] if decision_type == DECISION_APPROVE else None,
        "reviewed_by_user_id": row["decided_by_user_id"] if is_review_decision else None,
        "reviewed_at": _iso_stamp(row["decided_at"]) if is_review_decision else None,
        "raw_text": row["raw_text"],
        "activity_description": row["activity_description"],
        "asset": row["asset"],
        "discipline": row["discipline"],
        "location": row["location"],
        "actual_start": _actual_text(row["actual_start"], row["actual_start_time_stated"]),
        "actual_finish": _actual_text(row["actual_finish"], row["actual_finish_time_stated"]),
        "actual_start_time_stated": int(bool(row["actual_start_time_stated"])),
        "actual_finish_time_stated": int(bool(row["actual_finish_time_stated"])),
        "progress_percent": float(progress) if progress is not None else None,
        "status": row["event_status"],
        "contractor": row["contractor"],
        "delay_reason_reported": row["delay_reason_reported"],
        "delay_reason_category": row["delay_reason_category"],
        "evidence_quote": row["evidence_quote"],
        "activity_ref": row["activity_ref"],
        "match_confidence": float(row["confidence"]) if row["confidence"] is not None else None,
        "match_method": row["match_method"],
        "match_status": _compat_status(row["ai_status"], row["final_status"]),
        "match_reason": row["reason"],
        "created_at": _iso_stamp(row["result_created_at"]),
        "extraction_degraded": int(bool(row["extraction_degraded"])),
        "extraction_error": row["extraction_error"],
        "alternatives": _alternatives_from(top, target),
        "planner_status": planner_status,
        "planner_note": row["decision_note"] if is_planner_decision else None,
        "planner_decided_by": row["decided_by_name"] if is_planner_decision else None,
        "planner_decided_at": _iso_stamp(row["decided_at"]) if is_planner_decision else None,
        "matched_activity_description": row["target_description"],
        "matched_activity_planned_finish": _iso_day(row["target_planned_finish"]),
        "matched_activity_status": row["target_status"],
        "matched_activity_progress": float(row["target_progress"]) if row["target_progress"] is not None else None,
    }


def _fetch_updates(conn: psycopg.Connection, where: str, params: tuple = (), order: str = "") -> list[dict[str, Any]]:
    sql = _UPDATE_SELECT + (f" WHERE {where}" if where else "") + (f" ORDER BY {order}" if order else "")
    return [_shape_update(r) for r in conn.execute(sql, params).fetchall()]


def get_progress_update(update_id: str | None) -> dict[str, Any] | None:
    """One matching result in the old 'progress update' shape (or None)."""
    if not update_id:
        return None
    conn = get_connection()
    try:
        rows = _fetch_updates(conn, "mr.result_id = %s", (update_id,))
        return rows[0] if rows else None
    finally:
        conn.close()


def _save_result(update: dict[str, Any], *, duplicate: bool, conflicts: list[str] | None = None,
                 outcome: str | None = None) -> dict[str, Any]:
    """The return value of save_progress_update, built from a stored update dict."""
    return {
        "update_id": update["update_id"],
        # The activity this update targets: the one it was applied to, or the
        # one it is waiting to be applied to.
        "l6_id": update["l6_id"] or update["pending_l6_id"],
        "approved_by": update["approved_by"],
        "match_status": update["match_status"],
        "match_reason": update["match_reason"],
        "applied": outcome == OUTCOME_APPLIED,
        "outcome": outcome,
        "duplicate": duplicate,
        "conflicts": conflicts or [],
        "extraction_degraded": bool(update["extraction_degraded"]),
    }


def _source_type(source_name: str) -> str:
    name = (source_name or "").strip()
    if not name or name.lower() == "pasted text":
        return "PASTED"
    return "TEXT_FILE" if name.lower().endswith((".txt", ".log")) else "UPLOAD"


def _ensure_report(
    conn: psycopg.Connection, *, report_id: str, source_name: str, report_datetime: datetime,
    submitted_by_user_id: str | None, reported_by: str,
) -> None:
    """Create the report row if it is new; lock it either way (this serialises `seq`).

    A report id that already exists must be the SAME report (same moment and
    submitter); otherwise two different reports would silently merge.
    """
    conn.execute(
        """
        INSERT INTO reports (report_id, source_type, source_name, report_datetime,
                             submitted_by_user_id, reported_by_name, status)
        VALUES (%s, %s, %s, %s, %s, %s, 'COMPLETED')
        ON CONFLICT (report_id) DO NOTHING
        """,
        (report_id, _source_type(source_name), source_name, report_datetime, submitted_by_user_id, reported_by),
    )
    row = conn.execute("SELECT * FROM reports WHERE report_id = %s FOR UPDATE", (report_id,)).fetchone()
    if row["report_datetime"] != report_datetime or row["submitted_by_user_id"] != submitted_by_user_id:
        raise ValueError(
            f"Report id {report_id!r} is already used by a different report "
            "(another date or submitter). Use a different report id."
        )


def _find_live_statement(conn: psycopg.Connection, fingerprint: str | None) -> dict[str, Any] | None:
    """The statement that already holds this fingerprint (REJECTED ones do not count)."""
    if not fingerprint:
        return None
    return conn.execute(
        "SELECT statement_id FROM report_statements WHERE fingerprint = %s AND status <> 'REJECTED'",
        (fingerprint,),
    ).fetchone()


def _latest_result_id(conn: psycopg.Connection, statement_id: str) -> str | None:
    row = conn.execute(
        "SELECT result_id FROM matching_results WHERE statement_id = %s ORDER BY attempt_no DESC LIMIT 1",
        (statement_id,),
    ).fetchone()
    return row["result_id"] if row else None


def _duplicate_result(conn: psycopg.Connection, statement_id: str) -> dict[str, Any]:
    rows = _fetch_updates(conn, "mr.result_id = %s", (_latest_result_id(conn, statement_id),))
    return _save_result(rows[0], duplicate=True)


def save_progress_update(
    *,
    event: dict[str, Any],
    match: dict[str, Any],
    reported_by: str,
    source_name: str,
    report_date: str,
    report_id: str | None = None,
    reported_by_user_id: str | None = None,
) -> dict[str, Any]:
    """Store one progress event in ONE transaction; change `main` only for a safe AUTO_ACCEPTED match.

    Writes: report (if new), statement, revision 1, matching result and, for a
    confident match, a system AUTO_ACCEPT decision followed by the main_updates
    row and the `main` change.

    * AUTO_ACCEPTED -> applied (unless it conflicts, is outdated or changes nothing, see below).
    * REVIEW        -> stored only; a supervisor decides (decide_review_update).
    * UNMATCHED     -> stored as NOT_FOUND; shown on the planner list.
    * ERROR         -> stored as ERROR (the matcher crashed). Never NOT_FOUND.
                       Saving the same statement again adds a new attempt.
    * Conflicting AUTO_ACCEPTED -> stored as REVIEW with the reason.
    * Degraded extraction -> never AUTO_ACCEPTED: stored as REVIEW.
    * Older report than the one that last set the activity -> HISTORICAL_ONLY.
    * Report that agrees with the live state -> NO_CHANGE.
    * Same statement again -> the existing result is returned and nothing is
      written or applied a second time (result["duplicate"] is True).

    Returned keys: update_id, l6_id, approved_by, match_status (as stored),
    match_reason, applied (True only when `main` really changed), outcome
    (APPLIED / HISTORICAL_ONLY / NO_CHANGE, or None when `main` was not
    involved), duplicate, conflicts, extraction_degraded.
    """
    label = match.get("review_status")
    ai_status = _AI_STATUS_FROM_LABEL.get(label)
    if ai_status is None:
        raise ValueError(
            f"Cannot save a progress update with review_status={label!r}; "
            "only AUTO_ACCEPTED, REVIEW, UNMATCHED and ERROR are stored."
        )
    target = match.get("matched_activity_id") if ai_status in (AI_AUTO_ACCEPTED, AI_REVIEW) else None
    if ai_status in (AI_AUTO_ACCEPTED, AI_REVIEW) and not target:
        raise ValueError(f"A {ai_status} match must name the matched activity.")
    reason = match.get("reason")
    # The flag can arrive on the event (from extraction) or on the match
    # (already carried through matching); either one is enough.
    degraded = is_degraded(event) or is_degraded(match)
    extraction_error = event.get("extraction_error") or match.get("extraction_error")
    if degraded:
        # Last line of defence: even if a caller skipped the matching-stage
        # guard, a keyword guess can never reach `main` on its own.
        if ai_status == AI_AUTO_ACCEPTED:
            ai_status = AI_REVIEW
        reason = _degraded_reason(reason)
    fingerprint = compute_fingerprint(report_date, reported_by, event.get("raw_text"))
    top_candidates = build_top_candidates(match)
    activity_ref = str(event.get("activity_ref") or "").strip() or None
    reported = _reported_from_event(event)
    report_dt = _report_datetime(report_date)
    confidence = match.get("confidence")
    confidence = max(0.0, min(1.0, float(confidence))) if confidence is not None else None

    conn = get_connection()
    try:
        statement_id: str | None = None
        existing = _find_live_statement(conn, fingerprint)
        if existing:
            # Seen before. Only a statement whose last attempt CRASHED (ERROR)
            # may be matched again; anything else is a duplicate.
            conn.execute("SELECT 1 FROM report_statements WHERE statement_id = %s FOR UPDATE", (existing["statement_id"],))
            latest = conn.execute(
                "SELECT ai_status FROM matching_results WHERE statement_id = %s ORDER BY attempt_no DESC LIMIT 1",
                (existing["statement_id"],),
            ).fetchone()
            if latest is not None and latest["ai_status"] != AI_ERROR:
                result = _duplicate_result(conn, existing["statement_id"])
                conn.rollback()
                return result
            statement_id = existing["statement_id"]

        if statement_id is None:
            # A brand-new statement: report row, statement, revision 1.
            report_id = report_id or _new_id("RPT")
            _ensure_report(conn, report_id=report_id, source_name=source_name, report_datetime=report_dt,
                           submitted_by_user_id=reported_by_user_id, reported_by=reported_by)
            seq = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM report_statements WHERE report_id = %s", (report_id,)
            ).fetchone()["n"]
            statement_id = _new_id("STM")
            conn.execute(
                """
                INSERT INTO report_statements (
                    statement_id, report_id, seq, raw_text, fingerprint,
                    extraction_provider, extraction_model, prompt_version,
                    extraction_degraded, extraction_error, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    statement_id, report_id, seq, event.get("raw_text") or "", fingerprint,
                    event.get("extraction_provider"), event.get("extraction_model"), event.get("prompt_version"),
                    degraded, extraction_error,
                    "EXTRACTED" if ai_status == AI_ERROR else "MATCHED",
                ),
            )
            revision_id = _new_id("REV")
            conn.execute(
                """
                INSERT INTO statement_revisions (
                    revision_id, statement_id, revision_no, origin, created_by_user_id,
                    activity_description, asset, discipline, location,
                    actual_start, actual_finish, actual_start_time_stated, actual_finish_time_stated,
                    progress_percent, status, contractor, delay_reason_reported, delay_reason_category,
                    evidence_quote, activity_ref
                ) VALUES (%s, %s, 1, 'EXTRACTION', NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    revision_id, statement_id,
                    event.get("activity_description"), event.get("asset"), event.get("discipline"), event.get("location"),
                    reported["actual_start"], reported["actual_finish"],
                    reported["actual_start_time_stated"], reported["actual_finish_time_stated"],
                    reported["progress_percent"], reported["status"], reported["contractor"],
                    event.get("delay_reason_reported"), event.get("delay_reason_category"),
                    event.get("evidence_quote"), activity_ref,
                ),
            )
            attempt_no = 1
        else:
            # A retry after a matcher crash: a new attempt on the same statement, same revision.
            rev = conn.execute(
                "SELECT revision_id FROM statement_revisions WHERE statement_id = %s ORDER BY revision_no DESC LIMIT 1",
                (statement_id,),
            ).fetchone()
            revision_id = rev["revision_id"]
            attempt_no = conn.execute(
                "SELECT COALESCE(MAX(attempt_no), 0) + 1 AS n FROM matching_results WHERE statement_id = %s",
                (statement_id,),
            ).fetchone()["n"]
            rep = conn.execute(
                "SELECT rp.report_id, rp.report_datetime FROM report_statements st "
                "JOIN reports rp ON rp.report_id = st.report_id WHERE st.statement_id = %s",
                (statement_id,),
            ).fetchone()
            report_id, report_dt = rep["report_id"], rep["report_datetime"]
            if ai_status != AI_ERROR:
                conn.execute("UPDATE report_statements SET status = 'MATCHED' WHERE statement_id = %s", (statement_id,))

        # Decide whether a confident match may really be applied. The `main`
        # row is locked first, so what we check is what we then change.
        conflicts: list[str] = []
        main_row: dict[str, Any] | None = None
        if target:
            activity = conn.execute("SELECT archived_at FROM activities WHERE l6_id = %s", (target,)).fetchone()
            if activity is None:
                raise ValueError(f"Matched activity {target!r} does not exist in the schedule")
            if ai_status == AI_AUTO_ACCEPTED:
                main_row = _lock_main(conn, target)
                if activity["archived_at"] is not None:
                    conflicts = ["the matched activity has been removed from the schedule"]
                else:
                    outcome_now, _ = plan_main_change(_live_values(main_row), main_row["last_report_datetime"], reported, report_dt)
                    if outcome_now == OUTCOME_APPLIED:
                        # Only a real change can conflict; an outdated or identical report cannot.
                        planned_now = {
                            "progress_percent": float(main_row["progress_percent"]),
                            "actual_finish": _actual_text(main_row["actual_finish"], main_row["actual_finish_time_stated"]),
                            "status": main_row["status"],
                        }
                        conflicts = find_conflicts(planned_now, {
                            "progress_percent": event.get("progress_percent"),
                            "actual_finish": reported["actual_finish"] and _actual_text(reported["actual_finish"], reported["actual_finish_time_stated"]),
                            "status": event.get("status"),
                        })
                if conflicts:
                    ai_status = AI_REVIEW
                    reason = "Not applied automatically: " + "; ".join(conflicts) + "." + (
                        f" Original match note: {reason}" if reason else ""
                    )

        result_id = _new_id("UPD")
        conn.execute(
            """
            INSERT INTO matching_results (
                result_id, statement_id, revision_id, attempt_no, ai_status, matched_activity_id,
                confidence, match_method, reason, top_candidates, matcher_version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                result_id, statement_id, revision_id, attempt_no, ai_status, target,
                confidence, match.get("match_method"), reason, Jsonb(top_candidates),
                match.get("matcher_version"),
            ),
        )

        main_outcome: str | None = None
        if ai_status == AI_AUTO_ACCEPTED:
            # D1: even a confident match changes `main` only through a decision row.
            decision = _insert_decision(conn, result_id=result_id, decision_type=DECISION_AUTO_ACCEPT,
                                        user_id=None, target=target)
            change = _record_main_update(
                conn, main_row=main_row, decision_id=decision["decision_id"], report_id=report_id,
                statement_id=statement_id, report_datetime=report_dt, reported=reported,
            )
            main_outcome = change["outcome"]
            _set_verdict(conn, result_id, "AUTO_ACCEPTED", target)

        stored = _fetch_updates(conn, "mr.result_id = %s", (result_id,))[0]
        conn.commit()
        return _save_result(stored, duplicate=False, conflicts=conflicts, outcome=main_outcome)
    except pg_errors.UniqueViolation:
        # Lost a race with an identical submission (or a simultaneous retry).
        conn.rollback()
        existing = _find_live_statement(conn, fingerprint)
        if existing and _latest_result_id(conn, existing["statement_id"]):
            result = _duplicate_result(conn, existing["statement_id"])
            conn.rollback()
            return result
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def reset_history() -> None:
    """Clear all report history and put every activity back to NOT_STARTED / 0%.

    Keeps `users` and `activities`. (TRUNCATE does not fire the append-only
    triggers, so this is the one deliberate way to empty the audit tables.)
    """
    conn = get_connection()
    try:
        # One statement: PostgreSQL needs every table that references another
        # in the list to be truncated together with it.
        conn.execute(
            "TRUNCATE main_updates, supervisor_decisions, matching_results, statement_revisions, "
            "report_statements, reports, main RESTART IDENTITY"
        )
        conn.execute("INSERT INTO main (l6_id) SELECT l6_id FROM activities")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Review queue: REVIEW matches waiting for a supervisor
# --------------------------------------------------------------------------

def list_review_updates() -> list[dict[str, Any]]:
    """Matching results currently waiting for a supervisor (oldest first).

    `target_l6_id` is the activity the update would be applied to on approval.
    """
    conn = get_connection()
    try:
        return _fetch_updates(
            conn, "mr.ai_status = 'REVIEW' AND mr.final_status = 'PENDING'", order="mr.created_at, mr.result_id"
        )
    finally:
        conn.close()


def decide_review_update(
    update_id: str,
    *,
    decision: str,
    supervisor: dict[str, Any],
    confirm_degraded: bool = False,
) -> dict[str, Any] | None:
    """Approve or reject a REVIEW match. The ONLY way a REVIEW match ever reaches `main`.

    decision='approve': an APPROVE decision is written, then the update goes
      through the same code as an automatic one, so newest-report-wins and
      NO_CHANGE apply here too (`applied` is False for HISTORICAL_ONLY and
      NO_CHANGE; `outcome` says which). A supervisor's approval is a deliberate
      human decision, so the automatic conflict checks are not repeated.
    decision='reject': a REJECT decision is written and the statement is marked
      REJECTED (which frees its fingerprint, so it can be corrected and sent
      again). `main` is not touched.

    Exactly once: the matching result is row-locked, and a result can have only
    one terminal decision (unique index), so two supervisors clicking together
    produce one decision and one main_updates row; the second gets None.

    An item whose extraction was degraded (keyword guesses) can only be
    approved with confirm_degraded=True; otherwise DegradedConfirmationRequired
    is raised and nothing changes. Rejecting never needs the confirmation.

    Returns None when there is nothing to decide: the update does not exist or
    is not (or no longer) waiting in the review queue. Raises ValueError when
    approving an update whose activity has since been removed from the schedule.
    """
    if decision not in {"approve", "reject"}:
        raise ValueError(f"Invalid decision: {decision!r}")
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT mr.result_id, mr.statement_id, mr.revision_id, mr.matched_activity_id, mr.final_status,
                   st.extraction_degraded, st.report_id, rp.report_datetime
            FROM matching_results mr
            JOIN report_statements st ON st.statement_id = mr.statement_id
            JOIN reports rp ON rp.report_id = st.report_id
            WHERE mr.result_id = %s AND mr.ai_status = 'REVIEW'
            FOR UPDATE OF mr
            """,
            (update_id,),
        ).fetchone()
        if row is None or row["final_status"] != "PENDING":
            conn.rollback()
            return None
        target = row["matched_activity_id"]

        if decision == "reject":
            _insert_decision(conn, result_id=update_id, decision_type=DECISION_REJECT,
                             user_id=supervisor["user_id"], target=target)
            _set_verdict(conn, update_id, "REJECTED", None)
            # Same transaction as the decision: this frees the fingerprint.
            conn.execute("UPDATE report_statements SET status = 'REJECTED' WHERE statement_id = %s", (row["statement_id"],))
            conn.commit()
            return {"update_id": update_id, "match_status": STATUS_REJECTED, "applied": False, "l6_id": target}

        degraded = bool(row["extraction_degraded"])
        if degraded and not confirm_degraded:
            conn.rollback()
            raise DegradedConfirmationRequired(
                "This update comes from a failed extraction: its fields are keyword guesses. "
                "Approving it writes those guesses to the plan. Confirm to approve anyway."
            )
        activity = conn.execute("SELECT archived_at FROM activities WHERE l6_id = %s", (target,)).fetchone()
        if activity is None or activity["archived_at"] is not None:
            conn.rollback()
            raise ValueError(f"Activity {target} is no longer in the schedule, so this update cannot be approved.")

        main_row = _lock_main(conn, target)
        rev = conn.execute("SELECT * FROM statement_revisions WHERE revision_id = %s", (row["revision_id"],)).fetchone()
        decided = _insert_decision(conn, result_id=update_id, decision_type=DECISION_APPROVE,
                                   user_id=supervisor["user_id"], target=target, degraded_confirmed=degraded)
        change = _record_main_update(
            conn, main_row=main_row, decision_id=decided["decision_id"], report_id=row["report_id"],
            statement_id=row["statement_id"], report_datetime=row["report_datetime"],
            reported=_reported_from_revision(rev),
        )
        _set_verdict(conn, update_id, "APPROVED", target)
        conn.commit()
        return {
            "update_id": update_id,
            "match_status": STATUS_AUTO_ACCEPTED,
            "applied": change["outcome"] == OUTCOME_APPLIED,
            "outcome": change["outcome"],
            "l6_id": target,
        }
    except pg_errors.UniqueViolation:
        # The terminal-decision index caught a decision that slipped past the lock.
        conn.rollback()
        return None
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Planner list: NOT_FOUND items a planner has to look at
#
# `main` and the schedule are NEVER changed here (D3). "Dismiss" and "Mark as
# new activity" are stored as supervisor_decisions only. A decision is made
# once: repeating an action on an item that is already decided is refused and
# keeps the first decision.
# --------------------------------------------------------------------------

class PlannerItemNotFound(LookupError):
    """The update does not exist or is not a NOT_FOUND item."""


def planner_status_counts() -> dict[str, int]:
    """Number of NOT_FOUND items per planner status, plus 'all'."""
    counts = {status: 0 for status in PLANNER_STATUSES}
    conn = get_connection()
    try:
        for row in conn.execute(
            """
            SELECT d.decision_type, COUNT(*) AS n
            FROM matching_results mr
            LEFT JOIN supervisor_decisions d
                   ON d.result_id = mr.result_id AND d.decision_type IN ('DISMISS', 'MARK_NEW_ACTIVITY')
            WHERE mr.ai_status = 'NOT_FOUND'
            GROUP BY d.decision_type
            """
        ).fetchall():
            counts[_PLANNER_FROM_DECISION.get(row["decision_type"], PLANNER_OPEN)] += row["n"]
    finally:
        conn.close()
    counts["all"] = sum(counts.values())
    return counts


def count_open_planner_items() -> int:
    """For the navigation link."""
    return planner_status_counts()[PLANNER_OPEN]


def list_planner_items(status: str | None = PLANNER_OPEN) -> list[dict[str, Any]]:
    """NOT_FOUND items for the planner page, newest first.

    `status` is 'open', 'dismissed', 'converted', or None / 'all' for everything.
    Each item carries `alternatives` (the closest candidates stored with the
    matching result) and `planner_status` (never empty).
    """
    where = "mr.ai_status = 'NOT_FOUND'"
    if status in (None, "all"):
        pass
    elif status == PLANNER_OPEN:
        where += " AND d.decision_id IS NULL"
    elif status == PLANNER_DISMISSED:
        where += f" AND d.decision_type = '{DECISION_DISMISS}'"
    elif status == PLANNER_CONVERTED:
        where += f" AND d.decision_type = '{DECISION_MARK_NEW}'"
    else:
        raise ValueError(f"Unknown planner status: {status!r}")
    conn = get_connection()
    try:
        return _fetch_updates(conn, where, order="mr.created_at DESC, mr.result_id")
    finally:
        conn.close()


def decide_planner_item(
    update_id: str,
    *,
    decision: str,
    planner: dict[str, Any],
    note: str | None = None,
) -> dict[str, Any]:
    """Dismiss a NOT_FOUND item, or mark it as a new activity (a note only).

    decision: 'dismiss' -> DISMISS decision   (planner_status 'dismissed')
              'convert' -> MARK_NEW_ACTIVITY  (planner_status 'converted')

    Records who and when and the optional note. `main`, `activities` and the
    schedule are not touched.

    The first decision wins. Repeating an action on a decided item does not
    fail and does not change anything: the stored decision is returned with
    `changed=False`, so a double click can never overwrite who decided or when.

    Returns {update_id, planner_status, planner_note, planner_decided_by,
    planner_decided_at, changed}. Raises PlannerItemNotFound for an unknown id
    or an item that is not NOT_FOUND, and ValueError for an unknown decision.
    """
    types = {"dismiss": DECISION_DISMISS, "convert": DECISION_MARK_NEW}
    if decision not in types:
        raise ValueError(f"Invalid decision: {decision!r}")
    clean_note = (note or "").strip() or None
    conn = get_connection()
    try:
        locked = conn.execute(
            "SELECT result_id FROM matching_results WHERE result_id = %s AND ai_status = 'NOT_FOUND' FOR UPDATE",
            (update_id,),
        ).fetchone()
        if locked is None:
            conn.rollback()
            raise PlannerItemNotFound(update_id)
        current = _fetch_updates(conn, "mr.result_id = %s", (update_id,))[0]
        if current["planner_status"] != PLANNER_OPEN:
            conn.rollback()
            return {
                "update_id": update_id,
                "planner_status": current["planner_status"],
                "planner_note": current["planner_note"],
                "planner_decided_by": current["planner_decided_by"],
                "planner_decided_at": current["planner_decided_at"],
                "changed": False,
            }
        made = _insert_decision(conn, result_id=update_id, decision_type=types[decision],
                                user_id=planner["user_id"], note=clean_note)
        _set_verdict(conn, update_id, "DISMISSED" if decision == "dismiss" else "MARKED_NEW", None)
        conn.commit()
        return {
            "update_id": update_id,
            "planner_status": PLANNER_DISMISSED if decision == "dismiss" else PLANNER_CONVERTED,
            "planner_note": clean_note,
            "planner_decided_by": planner["name"],
            "planner_decided_at": _iso_stamp(made["created_at"]),
            "changed": True,
        }
    except PlannerItemNotFound:
        raise
    except pg_errors.UniqueViolation:
        # Another click got there first (the terminal-decision index): keep that decision.
        conn.rollback()
        latest = _fetch_updates(conn, "mr.result_id = %s", (update_id,))[0]
        return {
            "update_id": update_id,
            "planner_status": latest["planner_status"],
            "planner_note": latest["planner_note"],
            "planner_decided_by": latest["planner_decided_by"],
            "planner_decided_at": latest["planner_decided_at"],
            "changed": False,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
