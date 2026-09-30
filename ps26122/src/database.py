from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from .timeutil import canonical_actual, now_iso

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# PS26122_DB_PATH lets tests and deployments point at a different database file.
DB_PATH = Path(os.getenv("PS26122_DB_PATH") or PROJECT_ROOT / "data" / "ps26122.db")
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

# Match statuses (progress_updates.match_status).
STATUS_AUTO_ACCEPTED = "AUTO_ACCEPTED"   # applied to the plan (automatically, or after a supervisor approved it)
STATUS_REVIEW = "REVIEW"                 # stored as history only, waiting for a supervisor
STATUS_REJECTED = "REJECTED"             # a supervisor said no; the plan was never changed
STATUS_UNMATCHED = "UNMATCHED"           # stored as history only (possible new activity)

# Planner decision on an UNMATCHED item (progress_updates.planner_status).
PLANNER_OPEN = "open"
PLANNER_DISMISSED = "dismissed"
PLANNER_CONVERTED = "converted"      # "marked as new activity": a note only, the schedule is not edited
PLANNER_STATUSES = (PLANNER_OPEN, PLANNER_DISMISSED, PLANNER_CONVERTED)

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


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Extraction workers write their results concurrently; wait for the lock
    # (up to 30 s) instead of failing with "database is locked" after the default 5 s.
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def default_project_id() -> str:
    return (os.getenv(DEFAULT_PROJECT_ID_ENV) or DEFAULT_PROJECT_ID).strip()


def initialize_database() -> dict[str, int]:
    """Create the schema, migrate older databases, and sync the planned schedule.

    Safe to call repeatedly: it only writes when something actually changed.
    Returns the schedule-sync summary (see `sync_schedule`).
    """
    conn = get_connection()
    try:
        _create_schema(conn)
        _migrate_schema(conn)
        summary = _sync_schedule(conn)
        conn.commit()
        return summary
    finally:
        conn.close()


def sync_schedule() -> dict[str, int]:
    """Re-read schedule.json into the database (insert / update / archive / restore)."""
    conn = get_connection()
    try:
        summary = _sync_schedule(conn)
        conn.commit()
        return summary
    finally:
        conn.close()


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS planned_l6_activities (
            l6_id               TEXT PRIMARY KEY,
            project_id          TEXT NOT NULL,

            wbs_code            TEXT,
            activity_code       TEXT,
            description         TEXT NOT NULL,
            discipline          TEXT,
            asset               TEXT,
            location            TEXT,

            planned_start       TEXT,
            planned_finish      TEXT,
            planned_duration    INTEGER,

            -- Date-only text ('2026-09-22') or, when the report stated a time of
            -- day, 'YYYY-MM-DDTHH:MM:SS' in project-local time. The *_time_stated
            -- flag is 1 exactly when a time of day is present; it is never invented.
            actual_start        TEXT,
            actual_finish       TEXT,
            actual_start_time_stated   INTEGER NOT NULL DEFAULT 0,
            actual_finish_time_stated  INTEGER NOT NULL DEFAULT 0,
            progress_percent    REAL NOT NULL DEFAULT 0,
            status              TEXT NOT NULL DEFAULT 'NOT_STARTED',
            contractor          TEXT,

            update_ref          TEXT UNIQUE,
            archived_at         TEXT,
            created_at          TEXT NOT NULL,
            updated_at          TEXT NOT NULL,

            FOREIGN KEY (update_ref) REFERENCES progress_updates(update_id)
        );

        CREATE TABLE IF NOT EXISTS progress_updates (
            update_id               TEXT PRIMARY KEY,
            l6_id                   TEXT,

            report_id               TEXT,
            source_type             TEXT,
            source_file             TEXT,
            report_date             TEXT,
            received_at             TEXT NOT NULL,

            reported_by             TEXT NOT NULL,
            approved_by             TEXT,

            raw_text                TEXT NOT NULL,

            activity_description    TEXT,
            asset                   TEXT,
            discipline              TEXT,
            location                TEXT,

            actual_start            TEXT,
            actual_finish           TEXT,
            -- Same convention as planned_l6_activities (1 = a time of day was stated).
            actual_start_time_stated   INTEGER NOT NULL DEFAULT 0,
            actual_finish_time_stated  INTEGER NOT NULL DEFAULT 0,
            progress_percent       REAL,
            status                  TEXT,
            contractor              TEXT,

            delay_reason_reported   TEXT,
            delay_reason_category   TEXT,
            evidence_quote          TEXT,

            match_confidence        REAL,
            match_method            TEXT,
            match_status            TEXT,
            match_reason            TEXT,

            created_at              TEXT NOT NULL,

            -- The activity a REVIEW / conflicting match is waiting to be applied to.
            -- l6_id stays NULL until the update is really applied to the plan.
            pending_l6_id           TEXT,
            -- Hash of report date + reporter + normalized statement text.
            fingerprint             TEXT,
            -- 1 when extraction failed and the fields are keyword guesses.
            extraction_degraded     INTEGER NOT NULL DEFAULT 0,
            extraction_error        TEXT,
            -- What a reviewer needs. NULL on rows saved before these columns existed.
            -- alternatives_json: up to 3 other candidates the matcher considered,
            -- as [{"activity_id", "description", "score"}]. activity_ref: an ID or
            -- WBS code the report explicitly quoted.
            alternatives_json       TEXT,
            activity_ref            TEXT,
            -- Planner decision on an UNMATCHED item. NULL on every other row.
            -- planner_status: 'open' (saved as UNMATCHED, nobody looked yet),
            -- 'dismissed' or 'converted' (marked as a new activity; a note only,
            -- the schedule is never edited).
            planner_status          TEXT,
            planner_note            TEXT,
            planner_decided_by      TEXT,
            planner_decided_at      TEXT,

            FOREIGN KEY (l6_id) REFERENCES planned_l6_activities(l6_id)
        );

        CREATE TABLE IF NOT EXISTS users (
            user_id           TEXT PRIMARY KEY,
            name              TEXT NOT NULL,
            username          TEXT NOT NULL UNIQUE,
            password_hash     TEXT NOT NULL,
            role              TEXT NOT NULL CHECK (role IN ('contractor', 'supervisor')),
            status            TEXT NOT NULL DEFAULT 'active',
            created_at        TEXT NOT NULL
        );

        -- One row per contractor "run" (ingest -> extract -> match). The web
        -- session keeps only the run_id; everything else lives here. Scratch data:
        -- old runs are deleted by src.pipeline_runs.cleanup_old_runs().
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id          TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            ingestion_json  TEXT NOT NULL,
            match_json      TEXT
        );

        -- One row per statement of a run, saved as soon as its extraction finishes.
        CREATE TABLE IF NOT EXISTS pipeline_run_items (
            run_id       TEXT NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
            idx          INTEGER NOT NULL,
            statement    TEXT NOT NULL,
            state        TEXT NOT NULL DEFAULT 'pending'
                         CHECK (state IN ('pending', 'running', 'done', 'failed')),
            result_json  TEXT,
            error        TEXT,
            updated_at   TEXT NOT NULL,
            PRIMARY KEY (run_id, idx)
        );

        CREATE INDEX IF NOT EXISTS ix_pipeline_runs_user ON pipeline_runs(user_id);
        """
    )


def _add_missing_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    # SQLite has no "ADD COLUMN IF NOT EXISTS", so check first. This makes
    # initialize_database() safe to run against a database created by an older
    # version of the code, without losing existing rows.
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for col, decl in columns.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def _migrate_schema(conn: sqlite3.Connection) -> None:
    # progress_updates predates the users table; these columns tie a report to
    # a real identity (reported_by_user_id) and record who acted on a REVIEW
    # match (approved_by_user_id / reviewed_by_user_id / reviewed_at).
    _add_missing_columns(
        conn,
        "progress_updates",
        {
            "reported_by_user_id": "TEXT REFERENCES users(user_id)",
            "approved_by_user_id": "TEXT REFERENCES users(user_id)",
            "reviewed_by_user_id": "TEXT REFERENCES users(user_id)",
            "reviewed_at": "TEXT",
            "pending_l6_id": "TEXT",
            "fingerprint": "TEXT",
            "extraction_degraded": "INTEGER NOT NULL DEFAULT 0",
            "extraction_error": "TEXT",
            "alternatives_json": "TEXT",
            "activity_ref": "TEXT",
            "planner_status": "TEXT",
            "planner_note": "TEXT",
            "planner_decided_by": "TEXT",
            "planner_decided_at": "TEXT",
        },
    )
    _migrate_pending_targets(conn)
    _backfill_planner_status(conn)
    _backfill_fingerprints(conn)
    # One live record per fingerprint. Rejected rows are excluded so a
    # rejected statement can be corrected and submitted again.
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_progress_updates_fingerprint "
        "ON progress_updates(fingerprint) "
        "WHERE fingerprint IS NOT NULL AND match_status <> 'REJECTED'"
    )
    # archived_at: set when an activity disappears from schedule.json. Rows are
    # never deleted because progress_updates refer to them.
    _add_missing_columns(conn, "planned_l6_activities", {"archived_at": "TEXT"})
    # Time-of-day flags. Old rows hold date-only text, so 0 ("no time stated")
    # is exactly right for them and they load unchanged.
    time_flags = {
        "actual_start_time_stated": "INTEGER NOT NULL DEFAULT 0",
        "actual_finish_time_stated": "INTEGER NOT NULL DEFAULT 0",
    }
    _add_missing_columns(conn, "planned_l6_activities", time_flags)
    _add_missing_columns(conn, "progress_updates", time_flags)


def _backfill_planner_status(conn: sqlite3.Connection) -> None:
    """An UNMATCHED row saved before planner status existed has no planner status yet.
    Nobody has looked at it, so it starts as 'open'. Rows that already have a
    status (or are not UNMATCHED) are never touched, so this is safe to repeat."""
    conn.execute(
        "UPDATE progress_updates SET planner_status = 'open' "
        "WHERE match_status = 'UNMATCHED' AND planner_status IS NULL"
    )


def _migrate_pending_targets(conn: sqlite3.Connection) -> None:
    """Older databases stored the candidate of a REVIEW row in l6_id and had
    already applied it to the plan. Move the candidate to pending_l6_id
    for every row that the plan does not actually reference."""
    conn.execute(
        """
        UPDATE progress_updates
        SET pending_l6_id = COALESCE(pending_l6_id, l6_id)
        WHERE pending_l6_id IS NULL AND l6_id IS NOT NULL
          AND match_status IN ('REVIEW', 'REJECTED')
        """
    )
    conn.execute(
        """
        UPDATE progress_updates SET l6_id = NULL
        WHERE match_status IN ('REVIEW', 'REJECTED') AND l6_id IS NOT NULL
        """
    )


def _backfill_fingerprints(conn: sqlite3.Connection) -> None:
    """Give pre-existing rows a fingerprint; the first row of a group wins and
    later identical rows are left without one (they stay as history)."""
    rows = conn.execute(
        "SELECT update_id, report_date, reported_by, raw_text, match_status FROM progress_updates "
        "WHERE fingerprint IS NULL ORDER BY created_at, update_id"
    ).fetchall()
    if not rows:
        return
    taken = {
        r[0]
        for r in conn.execute(
            "SELECT fingerprint FROM progress_updates WHERE fingerprint IS NOT NULL AND match_status <> 'REJECTED'"
        )
    }
    for row in rows:
        fp = compute_fingerprint(row["report_date"], row["reported_by"], row["raw_text"])
        if fp is None:
            continue
        if row["match_status"] != STATUS_REJECTED:
            if fp in taken:
                continue
            taken.add(fp)
        conn.execute("UPDATE progress_updates SET fingerprint = ? WHERE update_id = ?", (fp, row["update_id"]))


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


def _sync_schedule(conn: sqlite3.Connection) -> dict[str, int]:
    """Make the planned columns of `planned_l6_activities` match schedule.json.

    - new activity            -> inserted
    - changed planned fields  -> updated (execution state is preserved)
    - unchanged               -> left completely alone (updated_at is NOT bumped)
    - missing from the file   -> archived (archived_at set; never deleted)
    - archived, back in file  -> restored (archived_at cleared)
    """
    summary = {"inserted": 0, "updated": 0, "unchanged": 0, "archived": 0, "restored": 0}
    source_rows = _load_schedule_rows()
    if source_rows is None:
        return summary

    existing = {row["l6_id"]: dict(row) for row in conn.execute("SELECT * FROM planned_l6_activities")}
    now = now_iso()

    # SQLite cannot safely create a circular pair of foreign keys in a single
    # insert, so planned rows are inserted first with update_ref NULL.
    for item in source_rows:
        l6_id = item["activity_id"]
        planned = _planned_values(item)
        current = existing.get(l6_id)

        if current is None:
            columns = ["l6_id", *planned, "created_at", "updated_at"]
            conn.execute(
                f"INSERT INTO planned_l6_activities ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                [l6_id, *planned.values(), now, now],
            )
            summary["inserted"] += 1
            continue

        was_archived = current.get("archived_at") is not None
        changed = any(current.get(col) != value for col, value in planned.items())
        if not changed and not was_archived:
            summary["unchanged"] += 1
            continue

        assignments = ", ".join(f"{col} = ?" for col in planned)
        conn.execute(
            f"UPDATE planned_l6_activities SET {assignments}, archived_at = NULL, updated_at = ? "
            "WHERE l6_id = ?",
            [*planned.values(), now, l6_id],
        )
        summary["restored" if was_archived else "updated"] += 1

    if not source_rows:
        # An empty/wrong schedule file must not silently hide the whole plan.
        logger.warning("schedule.json contains no L6 activities; skipping archive step.")
        return summary

    source_ids = {item["activity_id"] for item in source_rows}
    for l6_id, current in existing.items():
        if l6_id not in source_ids and current.get("archived_at") is None:
            conn.execute(
                "UPDATE planned_l6_activities SET archived_at = ?, updated_at = ? WHERE l6_id = ?",
                (now, now, l6_id),
            )
            summary["archived"] += 1
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
    """Planned L6 activities the matcher may choose from (archived ones excluded)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT l6_id AS activity_id, 'L6' AS level, wbs_code, activity_code, description, "
            "discipline, asset, location, planned_start, planned_finish, status "
            "FROM planned_l6_activities WHERE archived_at IS NULL ORDER BY l6_id"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Writing progress updates (safe writes and data integrity)
#
# The rules, in one place:
#   * Only two things may change the plan: an AUTO_ACCEPTED match, or a
#     supervisor approving a REVIEW match (decide_review_update). Both go
#     through _apply_to_plan(), so there is exactly one piece of update logic.
#   * REVIEW and UNMATCHED matches are stored as history only.
#   * An AUTO_ACCEPTED match that conflicts with the current plan is NOT
#     applied; it is downgraded to REVIEW with the reason spelled out.
#   * The same statement (report date + reporter + text) is stored once.
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


def _apply_to_plan(conn: sqlite3.Connection, l6_id: str, event: dict[str, Any], update_id: str, now: str) -> None:
    """THE ONLY place that writes execution state into planned_l6_activities.

    Empty fields in the event keep the existing value. Callers decide whether
    the update is allowed; this function just applies it.

    An actual start/finish is written through canonical_actual(), so a time
    of day is stored only when the report stated one. The time-stated flag moves
    together with its value (and stays as it was when the event has no value).
    """
    start, start_flag = canonical_actual(event.get("actual_start"), event.get("actual_start_time_stated"))
    finish, finish_flag = canonical_actual(event.get("actual_finish"), event.get("actual_finish_time_stated"))
    conn.execute(
        """
        UPDATE planned_l6_activities
        SET actual_start = COALESCE(?, actual_start),
            actual_start_time_stated = CASE WHEN ? IS NULL THEN actual_start_time_stated ELSE ? END,
            actual_finish = COALESCE(?, actual_finish),
            actual_finish_time_stated = CASE WHEN ? IS NULL THEN actual_finish_time_stated ELSE ? END,
            progress_percent = COALESCE(?, progress_percent),
            status = COALESCE(?, status),
            contractor = COALESCE(?, contractor),
            update_ref = ?,
            updated_at = ?
        WHERE l6_id = ?
        """,
        (
            start,
            start, start_flag,
            finish,
            finish, finish_flag,
            event.get("progress_percent"),
            event.get("status"),
            event.get("contractor"),
            update_id,
            now,
            l6_id,
        ),
    )


def _existing_record(conn: sqlite3.Connection, fingerprint: str | None) -> dict[str, Any] | None:
    if not fingerprint:
        return None
    row = conn.execute(
        "SELECT * FROM progress_updates WHERE fingerprint = ? AND match_status <> 'REJECTED'",
        (fingerprint,),
    ).fetchone()
    return dict(row) if row else None


def _result(row: dict[str, Any], *, duplicate: bool, conflicts: list[str] | None = None) -> dict[str, Any]:
    applied = row.get("match_status") == STATUS_AUTO_ACCEPTED and row.get("l6_id") is not None
    return {
        "update_id": row["update_id"],
        # The activity this update targets: the one it was applied to, or the
        # one it is waiting to be applied to. See "applied" to tell them apart.
        "l6_id": row.get("l6_id") or row.get("pending_l6_id"),
        "approved_by": row.get("approved_by"),
        "match_status": row.get("match_status"),
        "match_reason": row.get("match_reason"),
        "applied": applied,
        "duplicate": duplicate,
        "conflicts": conflicts or [],
        "extraction_degraded": bool(row.get("extraction_degraded")),
    }


MAX_ALTERNATIVES = 3   # How many other candidates are kept for a reviewer


def build_alternatives(match: dict[str, Any], exclude_id: str | None = None) -> list[dict[str, Any]]:
    """Up to MAX_ALTERNATIVES other candidates from the match shortlist.

    Order is the shortlist order (best retrieval first, ID matches first).
    `exclude_id` (the chosen activity) is left out; so is any repeated ID.
    """
    out: list[dict[str, Any]] = []
    seen = {exclude_id} if exclude_id else set()
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
        if len(out) == MAX_ALTERNATIVES:
            break
    return out


def parse_alternatives(raw: str | None) -> list[dict[str, Any]]:
    """Read `alternatives_json` back. Empty (old rows) or damaged text gives []."""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [a for a in data if isinstance(a, dict)] if isinstance(data, list) else []


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
    """Store one progress event; change the plan only for a safe AUTO_ACCEPTED match.

    * AUTO_ACCEPTED -> applied to the plan (unless it conflicts, see below).
    * REVIEW        -> history only; the candidate is kept in pending_l6_id.
    * UNMATCHED     -> history only.
    * anything else (for example ERROR) is refused with ValueError.
    * Conflicting AUTO_ACCEPTED -> stored as REVIEW with the reason.
    * Degraded extraction -> never AUTO_ACCEPTED: an AUTO_ACCEPTED match is
      stored as REVIEW and the plan is not touched until a supervisor approves.
    * Same statement again -> the existing record is returned, nothing is
      written or applied a second time (result["duplicate"] is True).

    The returned dict has: update_id, l6_id (target activity), approved_by,
    match_status (the status actually stored), match_reason, applied,
    duplicate, conflicts.
    """
    status = match.get("review_status")
    if status not in (STATUS_AUTO_ACCEPTED, STATUS_REVIEW, STATUS_UNMATCHED):
        raise ValueError(
            f"Cannot save a progress update with review_status={status!r}; "
            "only AUTO_ACCEPTED, REVIEW and UNMATCHED are stored."
        )
    target = match.get("matched_activity_id") if status != STATUS_UNMATCHED else None
    reason = match.get("reason")
    # The flag can arrive on the event (from extraction) or on the match
    # (already carried through matching); either one is enough.
    degraded = is_degraded(event) or is_degraded(match)
    extraction_error = event.get("extraction_error") or match.get("extraction_error")
    if degraded and status != STATUS_UNMATCHED:
        # Last line of defence: even if a caller skipped the matching-stage
        # guard, a keyword guess can never reach the plan on its own.
        if status == STATUS_AUTO_ACCEPTED:
            status = STATUS_REVIEW
        reason = _degraded_reason(reason)
    elif degraded:
        reason = _degraded_reason(reason)
    fingerprint = compute_fingerprint(report_date, reported_by, event.get("raw_text"))
    # The other candidates the matcher considered (never the chosen one) and
    # the ID / WBS code the report quoted. Filled on every save; old rows stay empty.
    alternatives = build_alternatives(match, exclude_id=match.get("matched_activity_id"))
    alternatives_json = json.dumps(alternatives) if alternatives else None
    activity_ref = str(event.get("activity_ref") or "").strip() or None
    # What is stored is the canonical form (time only when it was stated).
    actual_start, start_time_stated = canonical_actual(event.get("actual_start"), event.get("actual_start_time_stated"))
    actual_finish, finish_time_stated = canonical_actual(event.get("actual_finish"), event.get("actual_finish_time_stated"))

    update_id = f"UPD-{uuid.uuid4().hex[:8].upper()}"
    now = now_iso()

    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")  # one writer at a time: check + insert + apply are atomic

        existing = _existing_record(conn, fingerprint)
        if existing:
            conn.rollback()
            return _result(existing, duplicate=True)

        conflicts: list[str] = []
        planned: dict[str, Any] | None = None
        if target:
            row = conn.execute("SELECT * FROM planned_l6_activities WHERE l6_id = ?", (target,)).fetchone()
            if row is None:
                raise ValueError(f"Matched activity {target!r} does not exist in the schedule")
            planned = dict(row)

        if status == STATUS_AUTO_ACCEPTED:
            if not target:
                conflicts = ["no matched activity to apply the update to"]
            elif planned and planned.get("archived_at"):
                conflicts = ["the matched activity has been removed from the schedule"]
            elif planned:
                conflicts = find_conflicts(planned, event)
            if conflicts:
                status = STATUS_REVIEW
                reason = "Not applied automatically: " + "; ".join(conflicts) + "." + (
                    f" Original match note: {reason}" if reason else ""
                )

        applies_now = status == STATUS_AUTO_ACCEPTED
        approved_by = "SYSTEM" if applies_now else None

        conn.execute(
            """
            INSERT INTO progress_updates (
                update_id, l6_id, pending_l6_id, fingerprint, report_id, source_type, source_file,
                report_date, received_at, reported_by, approved_by, raw_text,
                activity_description, asset, discipline, location,
                actual_start, actual_finish, actual_start_time_stated, actual_finish_time_stated,
                progress_percent, status,
                contractor, delay_reason_reported, delay_reason_category,
                evidence_quote, match_confidence, match_method, match_status,
                match_reason, created_at, reported_by_user_id,
                extraction_degraded, extraction_error, alternatives_json, activity_ref,
                planner_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                update_id,
                target if applies_now else None,          # l6_id: only once really applied
                target if status == STATUS_REVIEW else None,  # pending_l6_id: the approval target
                fingerprint,
                report_id,
                "TEXT_FILE" if source_name.lower().endswith((".txt", ".log")) else "UPLOAD",
                source_name,
                report_date,
                now,
                reported_by,
                approved_by,
                event.get("raw_text", ""),
                event.get("activity_description"),
                event.get("asset"),
                event.get("discipline"),
                event.get("location"),
                actual_start,
                actual_finish,
                start_time_stated,
                finish_time_stated,
                event.get("progress_percent"),
                event.get("status"),
                event.get("contractor"),
                event.get("delay_reason_reported"),
                event.get("delay_reason_category"),
                event.get("evidence_quote"),
                match.get("confidence"),
                match.get("match_method"),
                status,
                reason,
                now,
                reported_by_user_id,
                1 if degraded else 0,
                extraction_error,
                alternatives_json,
                activity_ref,
                # Only an UNMATCHED row goes to the planner list.
                PLANNER_OPEN if status == STATUS_UNMATCHED else None,
            ),
        )

        if applies_now:
            _apply_to_plan(conn, target, event, update_id, now)

        conn.commit()
        return {
            "update_id": update_id,
            "l6_id": target,
            "approved_by": approved_by,
            "match_status": status,
            "match_reason": reason,
            "applied": applies_now,
            "duplicate": False,
            "conflicts": conflicts,
            "extraction_degraded": degraded,
        }
    except sqlite3.IntegrityError:
        # Lost a race with an identical submission from another process.
        conn.rollback()
        existing = _existing_record(conn, fingerprint)
        if existing:
            return _result(existing, duplicate=True)
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_progress_update(update_id: str | None) -> dict[str, Any] | None:
    if not update_id:
        return None
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM progress_updates WHERE update_id = ?", (update_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def reset_demo_data() -> None:
    """Reset execution fields/history while preserving the seeded L6 plan."""
    conn = get_connection()
    try:
        # Order matters: planned_l6_activities.update_ref points at
        # progress_updates, so the pointers must be cleared BEFORE the history
        # rows are deleted (otherwise the foreign key rejects the DELETE).
        conn.execute(
            """UPDATE planned_l6_activities
               SET actual_start = NULL, actual_finish = NULL,
                   actual_start_time_stated = 0, actual_finish_time_stated = 0,
                   progress_percent = 0, status = 'NOT_STARTED',
                   contractor = NULL, update_ref = NULL,
                   updated_at = ?""",
            (now_iso(),),
        )
        conn.execute("DELETE FROM progress_updates")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_planned_activity(l6_id: str | None) -> dict[str, Any] | None:
    if not l6_id:
        return None
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM planned_l6_activities WHERE l6_id = ?", (l6_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_review_updates() -> list[dict[str, Any]]:
    """Progress updates currently waiting for a supervisor (match_status REVIEW).

    `target_l6_id` is the activity the update would be applied to on approval.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT pu.*, COALESCE(pu.pending_l6_id, pu.l6_id) AS target_l6_id,
                   pla.description AS matched_activity_description,
                   pla.planned_finish AS matched_activity_planned_finish,
                   pla.status AS matched_activity_status,
                   pla.progress_percent AS matched_activity_progress
            FROM progress_updates pu
            LEFT JOIN planned_l6_activities pla ON pla.l6_id = COALESCE(pu.pending_l6_id, pu.l6_id)
            WHERE pu.match_status = 'REVIEW'
            ORDER BY pu.created_at
            """
        ).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            # The other candidates, ready for the review card ([] on old rows).
            item["alternatives"] = parse_alternatives(item.get("alternatives_json"))
            result.append(item)
        return result
    finally:
        conn.close()


def decide_review_update(
    update_id: str,
    *,
    decision: str,
    supervisor: dict[str, Any],
    confirm_degraded: bool = False,
) -> dict[str, Any] | None:
    """Approve or reject a REVIEW-status update. This is the ONLY way a REVIEW
    update ever reaches the plan, and it happens at most once.

    decision='approve': match_status becomes AUTO_ACCEPTED, approved_by is the
      real supervisor, l6_id is set, and the update is applied to the plan.
      A supervisor's approval is a deliberate human decision, so the automatic
      conflict checks are not repeated here.
    decision='reject': match_status becomes REJECTED. The plan is not touched.

    Both branches stamp reviewed_by_user_id / reviewed_at for the audit trail.

    An item whose extraction was degraded (keyword guesses) can only be
    approved with confirm_degraded=True; otherwise DegradedConfirmationRequired
    is raised and nothing changes. Rejecting never needs the confirmation.

    Returns None when there is nothing to decide: the update does not exist, is
    no longer in REVIEW (already decided), or has no target activity.
    Raises ValueError when approving an update whose activity has since been
    removed from the schedule.
    """
    if decision not in {"approve", "reject"}:
        raise ValueError(f"Invalid decision: {decision!r}")
    now = now_iso()
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM progress_updates WHERE update_id = ? AND match_status = 'REVIEW'", (update_id,)
        ).fetchone()
        if not row:
            conn.rollback()
            return None
        row = dict(row)
        target = row.get("pending_l6_id") or row.get("l6_id")

        if decision == "approve":
            if not target:
                conn.rollback()
                return None
            if row.get("extraction_degraded") and not confirm_degraded:
                conn.rollback()
                raise DegradedConfirmationRequired(
                    "This update comes from a failed extraction: its fields are keyword guesses. "
                    "Approving it writes those guesses to the plan. Confirm to approve anyway."
                )
            planned = conn.execute(
                "SELECT archived_at FROM planned_l6_activities WHERE l6_id = ?", (target,)
            ).fetchone()
            if planned is None or planned["archived_at"] is not None:
                conn.rollback()
                raise ValueError(f"Activity {target} is no longer in the schedule, so this update cannot be approved.")
            new_status, approved_by, approved_by_user_id, l6_id = (
                STATUS_AUTO_ACCEPTED, supervisor["name"], supervisor["user_id"], target,
            )
        else:
            new_status, approved_by, approved_by_user_id, l6_id = STATUS_REJECTED, None, None, None

        # The status guard makes "exactly once" hold even if two supervisors click at the same time.
        changed = conn.execute(
            """
            UPDATE progress_updates
            SET match_status = ?, approved_by = ?, approved_by_user_id = ?, l6_id = ?,
                reviewed_by_user_id = ?, reviewed_at = ?
            WHERE update_id = ? AND match_status = 'REVIEW'
            """,
            (new_status, approved_by, approved_by_user_id, l6_id, supervisor["user_id"], now, update_id),
        ).rowcount
        if changed != 1:
            conn.rollback()
            return None

        if decision == "approve":
            _apply_to_plan(conn, target, row, update_id, now)

        conn.commit()
        return {"update_id": update_id, "match_status": new_status, "applied": decision == "approve", "l6_id": target}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Planner list: UNMATCHED items a planner has to look at
#
# The plan and the schedule are NEVER changed here. "Mark as new activity" is a
# note only. A decision is made once: repeating an action on an item that is
# already decided is refused and keeps the first decision.
# --------------------------------------------------------------------------

class PlannerItemNotFound(LookupError):
    """The update does not exist or is not an UNMATCHED item."""


def planner_status_counts() -> dict[str, int]:
    """Number of UNMATCHED items per planner status, plus 'all'."""
    counts = {status: 0 for status in PLANNER_STATUSES}
    conn = get_connection()
    try:
        for row in conn.execute(
            "SELECT planner_status, COUNT(*) AS n FROM progress_updates "
            "WHERE match_status = 'UNMATCHED' GROUP BY planner_status"
        ):
            # An UNMATCHED row with no status can only exist if a database was
            # not migrated; it is still waiting for a planner, so count it as open.
            key = row["planner_status"] if row["planner_status"] in counts else PLANNER_OPEN
            counts[key] += row["n"]
    finally:
        conn.close()
    counts["all"] = sum(counts.values())
    return counts


def count_open_planner_items() -> int:
    """For the navigation link."""
    return planner_status_counts()[PLANNER_OPEN]


def list_planner_items(status: str | None = PLANNER_OPEN) -> list[dict[str, Any]]:
    """UNMATCHED items for the planner page, newest first.

    `status` is 'open', 'dismissed', 'converted', or None / 'all' for everything.
    Each item carries `alternatives` (the closest candidates stored with the update,
    [] on older rows) and `planner_status` (never empty for an UNMATCHED row).
    """
    if status in (None, "all"):
        where, params = "", ()
    elif status in PLANNER_STATUSES:
        # A row from an unmigrated database has NULL status and counts as open.
        where = " AND COALESCE(planner_status, 'open') = ?"
        params = (status,)
    else:
        raise ValueError(f"Unknown planner status: {status!r}")
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM progress_updates WHERE match_status = 'UNMATCHED'" + where +
            " ORDER BY created_at DESC, update_id",
            params,
        ).fetchall()
    finally:
        conn.close()
    items = []
    for row in rows:
        item = dict(row)
        item["planner_status"] = item.get("planner_status") or PLANNER_OPEN
        item["alternatives"] = parse_alternatives(item.get("alternatives_json"))
        items.append(item)
    return items


def decide_planner_item(
    update_id: str,
    *,
    decision: str,
    planner: dict[str, Any],
    note: str | None = None,
) -> dict[str, Any]:
    """Dismiss an UNMATCHED item, or mark it as a new activity (a note only).

    decision: 'dismiss' -> planner_status 'dismissed'
              'convert' -> planner_status 'converted'

    Records who and when (planner_decided_by / planner_decided_at) and the
    optional note. The schedule and the plan are not touched.

    The first decision wins. Repeating an action on a decided item does not
    fail and does not change anything: the stored decision is returned with
    `changed=False`, so a double click can never overwrite who decided or when.

    Returns {update_id, planner_status, planner_note, planner_decided_by,
    planner_decided_at, changed}. Raises PlannerItemNotFound for an unknown id
    or an item that is not UNMATCHED, and ValueError for an unknown decision.
    """
    targets = {"dismiss": PLANNER_DISMISSED, "convert": PLANNER_CONVERTED}
    if decision not in targets:
        raise ValueError(f"Invalid decision: {decision!r}")
    clean_note = (note or "").strip() or None
    now = now_iso()
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT update_id, planner_status, planner_note, planner_decided_by, planner_decided_at "
            "FROM progress_updates WHERE update_id = ? AND match_status = 'UNMATCHED'",
            (update_id,),
        ).fetchone()
        if row is None:
            conn.rollback()
            raise PlannerItemNotFound(update_id)
        current = row["planner_status"] or PLANNER_OPEN
        if current != PLANNER_OPEN:
            conn.rollback()
            return {
                "update_id": update_id,
                "planner_status": current,
                "planner_note": row["planner_note"],
                "planner_decided_by": row["planner_decided_by"],
                "planner_decided_at": row["planner_decided_at"],
                "changed": False,
            }
        # The status guard keeps "first decision wins" true even for two simultaneous clicks.
        changed = conn.execute(
            """
            UPDATE progress_updates
            SET planner_status = ?, planner_note = ?, planner_decided_by = ?, planner_decided_at = ?
            WHERE update_id = ? AND match_status = 'UNMATCHED'
              AND COALESCE(planner_status, 'open') = 'open'
            """,
            (targets[decision], clean_note, planner["name"], now, update_id),
        ).rowcount
        if changed != 1:
            conn.rollback()
            latest = conn.execute(
                "SELECT planner_status, planner_note, planner_decided_by, planner_decided_at "
                "FROM progress_updates WHERE update_id = ?", (update_id,)).fetchone()
            return {"update_id": update_id, "planner_status": latest["planner_status"],
                    "planner_note": latest["planner_note"], "planner_decided_by": latest["planner_decided_by"],
                    "planner_decided_at": latest["planner_decided_at"], "changed": False}
        conn.commit()
        return {
            "update_id": update_id,
            "planner_status": targets[decision],
            "planner_note": clean_note,
            "planner_decided_by": planner["name"],
            "planner_decided_at": now,
            "changed": True,
        }
    except PlannerItemNotFound:
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
