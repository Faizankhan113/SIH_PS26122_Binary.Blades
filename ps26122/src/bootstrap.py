"""One-command setup for a fresh checkout.

    python -m src.bootstrap             # create/migrate the DB, load the schedule,
                                        # create demo users, seed history if empty
    python -m src.bootstrap --fresh     # delete the database file first, then do the above
    python -m src.bootstrap --no-history  # skip the synthetic Institutional Memory history

It is safe to re-run: the schedule sync only writes real changes, existing
users are skipped, and history is only seeded when there is none yet (use
`--fresh` to rebuild everything from scratch).

The seeded history is SYNTHETIC demo data, not real field reports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import database
from .database import get_connection, initialize_database, reset_demo_data
from .seed_history import seed_synthetic_history
from .seed_users import TEST_ACCOUNTS
from .seed_users import main as seed_users


def _delete_database_files() -> None:
    db = Path(database.DB_PATH)
    for suffix in ("", "-journal", "-wal", "-shm"):
        candidate = Path(str(db) + suffix)
        if candidate.exists():
            candidate.unlink()


def _history_row_count() -> int:
    conn = get_connection()
    try:
        return conn.execute("SELECT COUNT(*) FROM progress_updates").fetchone()[0]
    finally:
        conn.close()


def bootstrap(*, fresh: bool = False, with_history: bool = True) -> dict[str, object]:
    if fresh:
        _delete_database_files()
        print(f"Deleted existing database at {database.DB_PATH}")

    sync = initialize_database()
    print(f"Database ready at {database.DB_PATH} (schedule sync: {sync})")

    seed_users()

    history: dict[str, int] | None = None
    if with_history:
        if _history_row_count() == 0:
            reset_demo_data()
            history = seed_synthetic_history()
            print("Seeded SYNTHETIC institutional-memory history:", history)
        else:
            print("History already present; not re-seeding (use --fresh to rebuild).")
    return {"sync": sync, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up the PS 26122 MVP database and demo accounts.")
    parser.add_argument("--fresh", action="store_true", help="Delete the database file and rebuild it.")
    parser.add_argument("--no-history", action="store_true", help="Do not seed synthetic history.")
    args = parser.parse_args()

    bootstrap(fresh=args.fresh, with_history=not args.no_history)

    print("\nDemo logins:")
    for account in TEST_ACCOUNTS:
        print(f"  {account['role']:<11} {account['username']} / {account['password']}")
    print("\nStart the app with:  python -m ui.app   (then open http://127.0.0.1:5000)")


if __name__ == "__main__":
    main()
