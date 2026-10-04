"""One-time manual seeding of test accounts.

Supervisor accounts are not self-signup (they can approve/act on contractor
submissions and shouldn't be self-granted), so a handful
are seeded directly with this script (there is no admin screen for creating supervisors).

Run it directly:

    python -m src.seed_users

Idempotent: re-running it skips any username that already exists rather
than erroring.
"""

from __future__ import annotations

from .auth import STATUS_ACTIVE, UsernameTakenError, create_user
from .database import initialize_database

TEST_ACCOUNTS = [
    {"name": "Rahul Sharma", "username": "contractor1", "password": "contractor123", "role": "contractor"},
    {"name": "Priya Nair", "username": "supervisor1", "password": "supervisor123", "role": "supervisor"},
]


def main() -> None:
    initialize_database()  # make sure the users table exists on a fresh checkout
    for account in TEST_ACCOUNTS:
        try:
            create_user(
                name=account["name"],
                username=account["username"],
                password=account["password"],
                role=account["role"],
                status=STATUS_ACTIVE,
            )
            print(f"Created {account['role']} account: {account['username']} / {account['password']}")
        except UsernameTakenError:
            print(f"Skipped {account['username']} -- already exists.")


if __name__ == "__main__":
    main()
