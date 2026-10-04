"""PostgreSQL connection and schema management for PS 26122.

    python -m src.db init      # create the tables (safe to re-run)
    python -m src.db status    # show the connection, schema version and tables
    python -m src.db drop      # delete ALL PS 26122 tables and their data (asks first)
    python -m src.db drop --yes

The connection comes from DATABASE_URL (see .env.example). Only the tables
listed in `ALL_TABLES` are ever touched, so the database can safely be shared
with other things.

Rows come back as dicts (`row["column"]`), and every session runs in the
project timezone (PS26122_TIMEZONE, default Asia/Kolkata), so timestamptz
values read back as project-local, timezone-aware datetimes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from .timeutil import DEFAULT_TIMEZONE, TIMEZONE_ENV_VAR, project_timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "db" / "schema.sql"
SCHEMA_VERSION = 1  # keep in step with the INSERT at the end of db/schema.sql

load_dotenv(PROJECT_ROOT / ".env")

# Parents before children. `drop_schema` removes them in reverse order.
ALL_TABLES = (
    "users",
    "activities",
    "reports",
    "report_statements",
    "statement_revisions",
    "matching_results",
    "supervisor_decisions",
    "main",
    "main_updates",
    "pipeline_runs",
    "pipeline_run_items",
    "schema_version",
)
ALL_FUNCTIONS = ("ps26122_forbid_change", "ps26122_keep_ai_output")


class DatabaseNotConfigured(RuntimeError):
    """DATABASE_URL is missing."""


def database_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise DatabaseNotConfigured(
            "DATABASE_URL is not set. Copy .env.example to .env and set it, for example:\n"
            "  DATABASE_URL=postgresql://ps26122_user:your_password@localhost:5432/ps26122"
        )
    return url


def describe_url() -> str:
    """The connection target without the password (safe to print)."""
    parts = urlsplit(database_url())
    host = parts.hostname or "localhost"
    port = f":{parts.port}" if parts.port else ""
    user = f"{parts.username}@" if parts.username else ""
    return f"{user}{host}{port}{parts.path}"


def get_connection() -> psycopg.Connection:
    """Open a new connection. The caller closes it (use `with` or try/finally).

    Autocommit is off: changes are saved with conn.commit(), or discarded with
    conn.rollback(). Closing without committing discards them.
    """
    project_timezone()  # fail early with a clear message if the timezone name is wrong
    tz_name = (os.getenv(TIMEZONE_ENV_VAR) or DEFAULT_TIMEZONE).strip()
    return psycopg.connect(
        database_url(),
        row_factory=dict_row,
        connect_timeout=10,
        options=f"-c timezone={tz_name}",
    )


def init_schema() -> list[str]:
    """Create any missing tables, indexes and guards. Returns the tables now present.

    Idempotent: re-running changes nothing. It never drops or alters an
    existing table (upgrades will be added as numbered SQL steps).
    """
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn = get_connection()
    try:
        # No parameters are passed, so the whole file runs as one script.
        conn.execute(sql)
        conn.commit()
        return existing_tables(conn)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def drop_schema() -> None:
    """Delete every PS 26122 table (and its data). Irreversible."""
    conn = get_connection()
    try:
        for table in reversed(ALL_TABLES):
            conn.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
        for function in ALL_FUNCTIONS:
            conn.execute(f"DROP FUNCTION IF EXISTS {function}() CASCADE")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def existing_tables(conn: psycopg.Connection | None = None) -> list[str]:
    """Which of the PS 26122 tables exist in the connected database (in design order)."""
    own = conn is None
    conn = conn or get_connection()
    try:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = current_schema() AND table_name = ANY(%s)",
            (list(ALL_TABLES),),
        ).fetchall()
        present = {r["table_name"] for r in rows}
        return [t for t in ALL_TABLES if t in present]
    finally:
        if own:
            conn.close()


def schema_version() -> int | None:
    """The installed schema version, or None when the schema is not installed."""
    conn = get_connection()
    try:
        if "schema_version" not in existing_tables(conn):
            return None
        row = conn.execute("SELECT max(version) AS v FROM schema_version").fetchone()
        return row["v"]
    finally:
        conn.close()


def missing_tables() -> list[str]:
    present = set(existing_tables())
    return [t for t in ALL_TABLES if t not in present]


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def _cmd_init(_args: argparse.Namespace) -> int:
    tables = init_schema()
    print(f"Schema ready on {describe_url()} (version {schema_version()}), {len(tables)} tables:")
    for t in tables:
        print(f"  {t}")
    return 0


def _cmd_status(_args: argparse.Namespace) -> int:
    conn = get_connection()
    try:
        # current_setting() with an alias: SHOW names its column "TimeZone" (mixed case).
        row = conn.execute(
            "SELECT current_setting('server_version') AS version, current_setting('TimeZone') AS tz"
        ).fetchone()
        version, tz = row["version"], row["tz"]
    finally:
        conn.close()
    print(f"Connected to {describe_url()} (PostgreSQL {version}, session timezone {tz})")
    installed = schema_version()
    print(f"Schema version installed: {installed if installed is not None else 'none'}")
    missing = missing_tables()
    if missing:
        print("Missing tables:", ", ".join(missing))
        print("Run:  python -m src.db init")
        return 1
    print(f"All {len(ALL_TABLES)} tables present.")
    return 0


def _cmd_drop(args: argparse.Namespace) -> int:
    if not args.yes:
        answer = input(f"Delete ALL PS 26122 tables and data on {describe_url()}? Type 'yes' to continue: ")
        if answer.strip().lower() != "yes":
            print("Cancelled.")
            return 1
    drop_schema()
    print("PS 26122 tables dropped.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PS 26122 PostgreSQL schema tools.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create missing tables (safe to re-run).").set_defaults(func=_cmd_init)
    sub.add_parser("status", help="Show connection and schema state.").set_defaults(func=_cmd_status)
    drop = sub.add_parser("drop", help="Delete all PS 26122 tables and data.")
    drop.add_argument("--yes", action="store_true", help="Do not ask for confirmation.")
    drop.set_defaults(func=_cmd_drop)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DatabaseNotConfigured as exc:
        print(exc, file=sys.stderr)
        return 2
    except psycopg.OperationalError as exc:
        print(f"Could not connect to PostgreSQL: {exc}", file=sys.stderr)
        print("Check that the server is running and DATABASE_URL (user, password, port, database) is right.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
