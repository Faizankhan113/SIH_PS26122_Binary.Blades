"""One-command setup for a fresh checkout (PostgreSQL).

    python -m src.bootstrap             # create the tables if missing, load the schedule
    python -m src.bootstrap --fresh     # drop every PS 26122 table first, then create them

Then, optionally (local accounts and sample data):
    python -m src.seed_users               # test accounts (contractor1, supervisor1)
    python -m src.seed_history --reset     # synthetic history for Institutional Memory
"""

from __future__ import annotations

import argparse

from . import database, db


def bootstrap(*, fresh: bool = False) -> dict[str, object]:
    print(f"Database: {db.describe_url()}")
    if fresh:
        db.drop_schema()
        print("Dropped all PS 26122 tables.")
    sync = database.initialize_database()  # creates missing tables, then syncs data/schedule.json
    tables = db.existing_tables()
    print(f"Schema version {db.schema_version()} ready, {len(tables)} tables.")
    print(f"Schedule sync: {sync}")
    return {"tables": tables, "sync": sync}


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up the PS 26122 PostgreSQL database.")
    parser.add_argument("--fresh", action="store_true", help="Drop all PS 26122 tables and rebuild them.")
    args = parser.parse_args()
    try:
        bootstrap(fresh=args.fresh)
    except db.DatabaseNotConfigured as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
