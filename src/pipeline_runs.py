"""Server-side storage for one contractor "run" and its per-statement
extraction progress.

Why: Flask's default session is a signed cookie that browsers cut off at about
4 KB, and a run (statements, extraction results, match rows) is far bigger.
The session now holds only ``run_id``; this module keeps everything else in
the ``pipeline_runs`` and ``pipeline_run_items`` tables.

Ownership: ``get_run`` only returns a run to the user who created it. The other
functions take a run_id that the caller has already obtained that way.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Any

from .database import get_connection
from .timeutil import now_iso, now_local

RUN_TTL_HOURS_ENV = "PS26122_RUN_TTL_HOURS"
DEFAULT_RUN_TTL_HOURS = 24.0

STATE_PENDING = "pending"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_FAILED = "failed"


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def create_run(user_id: str, ingestion: dict[str, Any]) -> str:
    """Store a new run (with one pending item per statement) and return its id."""
    run_id = f"RUN-{uuid.uuid4().hex[:12]}"
    now = now_iso()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO pipeline_runs (run_id, user_id, created_at, updated_at, ingestion_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, user_id, now, now, _dumps(ingestion)),
        )
        conn.executemany(
            "INSERT INTO pipeline_run_items (run_id, idx, statement, state, updated_at) VALUES (?, ?, ?, 'pending', ?)",
            [(run_id, i, text, now) for i, text in enumerate(ingestion.get("statements", []))],
        )
        conn.commit()
    finally:
        conn.close()
    return run_id


def get_run(run_id: str | None, user_id: str | None) -> dict[str, Any] | None:
    """The run's ingestion record, or None if it does not exist or is not this user's."""
    if not run_id or not user_id:
        return None
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT run_id, user_id, created_at, ingestion_json, match_json IS NOT NULL AS has_match "
            "FROM pipeline_runs WHERE run_id = ? AND user_id = ?",
            (run_id, user_id),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "run_id": row["run_id"],
        "user_id": row["user_id"],
        "created_at": row["created_at"],
        "ingestion": json.loads(row["ingestion_json"]),
        "has_match": bool(row["has_match"]),
    }


def _touch(conn, run_id: str) -> None:
    conn.execute("UPDATE pipeline_runs SET updated_at = ? WHERE run_id = ?", (now_iso(), run_id))


def delete_run(run_id: str | None) -> None:
    if not run_id:
        return
    conn = get_connection()
    try:
        conn.execute("DELETE FROM pipeline_runs WHERE run_id = ?", (run_id,))  # items cascade
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Extraction progress
# --------------------------------------------------------------------------

def claim_item(run_id: str, idx: int) -> bool:
    """Atomically move one item pending/failed -> running. True if THIS caller got it.

    Makes "start extraction" safe to call twice (double click, second tab): a
    statement is only ever sent to the LLM by one worker.
    """
    conn = get_connection()
    try:
        cur = conn.execute(
            "UPDATE pipeline_run_items SET state = 'running', error = NULL, updated_at = ? "
            "WHERE run_id = ? AND idx = ? AND state IN ('pending', 'failed')",
            (now_iso(), run_id, idx),
        )
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()


def save_item_result(run_id: str, idx: int, result: dict[str, Any]) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE pipeline_run_items SET state = 'done', result_json = ?, error = NULL, updated_at = ? "
            "WHERE run_id = ? AND idx = ?",
            (_dumps(result), now_iso(), run_id, idx),
        )
        _touch(conn, run_id)
        conn.commit()
    finally:
        conn.close()


def save_item_failure(run_id: str, idx: int, error: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE pipeline_run_items SET state = 'failed', error = ?, updated_at = ? "
            "WHERE run_id = ? AND idx = ?",
            (error[:2000], now_iso(), run_id, idx),
        )
        _touch(conn, run_id)
        conn.commit()
    finally:
        conn.close()


def list_items(run_id: str) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT idx, statement, state, result_json, error FROM pipeline_run_items "
            "WHERE run_id = ? ORDER BY idx",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "idx": r["idx"],
            "statement": r["statement"],
            "state": r["state"],
            "result": json.loads(r["result_json"]) if r["result_json"] else None,
            "error": r["error"],
        }
        for r in rows
    ]


def extraction_progress(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts for the progress screen. `complete` means nothing is pending or running."""
    counts = {STATE_PENDING: 0, STATE_RUNNING: 0, STATE_DONE: 0, STATE_FAILED: 0}
    for item in items:
        counts[item["state"]] += 1
    degraded = sum(1 for i in items if i["result"] and i["result"].get("extraction_degraded"))
    needs_correction = sum(1 for i in items if i["result"] and i["result"].get("needs_correction"))
    return {
        "total": len(items),
        **counts,
        # A "done" item that needs correction is finished but is not an event.
        STATE_DONE: counts[STATE_DONE] - needs_correction,
        "needs_correction": needs_correction,
        "degraded": degraded,
        "finished": counts[STATE_DONE] + counts[STATE_FAILED],  # counts still holds the raw 'done' total
        "complete": len(items) > 0 and counts[STATE_PENDING] == 0 and counts[STATE_RUNNING] == 0,
    }


def extracted_results(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extraction results of the statements that produced an event, in statement order.

    Statements that need correction are finished but are NOT events, so they
    are never matched.
    """
    return [
        i["result"]
        for i in items
        if i["state"] == STATE_DONE and i["result"] and not i["result"].get("needs_correction")
    ]


def recover_interrupted_items() -> int:
    """At startup: items left 'running' by a server that stopped can never finish. Mark them failed."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "UPDATE pipeline_run_items SET state = 'failed', error = ?, updated_at = ? WHERE state = 'running'",
            ("Extraction was interrupted (the server restarted). Retry to run it again.", now_iso()),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Match results
# --------------------------------------------------------------------------

def save_match_rows(run_id: str, rows: list[dict[str, Any]]) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE pipeline_runs SET match_json = ?, updated_at = ? WHERE run_id = ?",
            (_dumps(rows), now_iso(), run_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_match_rows(run_id: str) -> list[dict[str, Any]] | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT match_json FROM pipeline_runs WHERE run_id = ?", (run_id,)).fetchone()
    finally:
        conn.close()
    if row is None or row["match_json"] is None:
        return None
    return json.loads(row["match_json"])


# --------------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------------

def run_ttl_hours() -> float:
    try:
        return float(os.getenv(RUN_TTL_HOURS_ENV, "").strip() or DEFAULT_RUN_TTL_HOURS)
    except ValueError:
        return DEFAULT_RUN_TTL_HOURS


def cleanup_old_runs(max_age_hours: float | None = None) -> int:
    """Delete runs not touched for `max_age_hours` (default 24, env PS26122_RUN_TTL_HOURS).

    Age is measured from the last update, so a run that is still being worked on
    is never removed. Returns the number of runs deleted.
    """
    hours = run_ttl_hours() if max_age_hours is None else max_age_hours
    cutoff = now_local() - timedelta(hours=hours)
    conn = get_connection()
    try:
        stale = [
            r["run_id"]
            for r in conn.execute("SELECT run_id, updated_at FROM pipeline_runs").fetchall()
            if datetime.fromisoformat(r["updated_at"]) < cutoff
        ]
        conn.executemany("DELETE FROM pipeline_runs WHERE run_id = ?", [(rid,) for rid in stale])
        conn.commit()
        return len(stale)
    finally:
        conn.close()
