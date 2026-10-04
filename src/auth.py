"""Role-based authentication: users table, password hashing, login/signup logic.

Two roles. Each is the identity behind a "who did this" column in the
audit tables (`reports.submitted_by_user_id`,
`supervisor_decisions.decided_by_user_id`):

  - contractor  -- submits field-progress reports.
    Self-signup, but the account sits in `status='pending'` until a
    supervisor approves it (see `set_account_status`).
  - supervisor  -- reviews/approves REVIEW-status matches and decides planner
    items. Not self-signup; seeded directly (see `src/seed_users.py`).

This module owns everything about *who* a user is: the `users` table
(defined in db/schema.sql) and nothing else. User dicts keep the same keys as
before, with `created_at` as ISO text in project-local time.
"""

from __future__ import annotations

import uuid
from typing import Any

from psycopg import errors as pg_errors
from werkzeug.security import check_password_hash, generate_password_hash

from .database import get_connection
from .timeutil import to_local_iso

VALID_ROLES = {"contractor", "supervisor"}
STATUS_PENDING = "pending"
STATUS_ACTIVE = "active"
STATUS_REJECTED = "rejected"


class UsernameTakenError(ValueError):
    """Raised on signup when the username is already in use."""


# The unique constraint PostgreSQL names for `username text NOT NULL UNIQUE` on `users`.
_USERNAME_CONSTRAINT = "users_username_key"


def _user_dict(row: dict[str, Any]) -> dict[str, Any]:
    """A users row as a plain dict (created_at as project-local ISO text)."""
    user = dict(row)
    user["created_at"] = to_local_iso(user.get("created_at"))
    return user


def create_user(
    *, name: str, username: str, password: str, role: str, status: str = STATUS_ACTIVE
) -> dict[str, Any]:
    """Create a user row. Callers decide `status`: self-signup contractors
    start `pending` (see `signup_contractor`); seeded/admin-created accounts
    can be created `active` directly."""
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role: {role!r}")
    name = (name or "").strip()
    username = (username or "").strip().lower()
    if not name or not username or not password:
        raise ValueError("Name, username, and password are all required.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")

    user_id = f"USR-{uuid.uuid4().hex[:8].upper()}"
    conn = get_connection()
    try:
        row = conn.execute(
            """
            INSERT INTO users (user_id, name, username, password_hash, role, status)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING created_at
            """,
            (user_id, name, username, generate_password_hash(password), role, status),
        ).fetchone()
        conn.commit()
    except pg_errors.UniqueViolation as exc:
        conn.rollback()
        # Only the username constraint means "taken"; anything else is a real error.
        constraint = exc.diag.constraint_name
        if constraint and constraint != _USERNAME_CONSTRAINT:
            raise
        raise UsernameTakenError(f"Username '{username}' is already taken.") from exc
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {
        "user_id": user_id,
        "name": name,
        "username": username,
        "role": role,
        "status": status,
        "created_at": to_local_iso(row["created_at"]),
    }


def signup_contractor(*, name: str, username: str, password: str) -> dict[str, Any]:
    """Self-signup entry point. Always creates a `contractor` account in
    `pending` status -- a supervisor must approve it before first login
    (confirmed requirement)."""
    return create_user(name=name, username=username, password=password, role="contractor", status=STATUS_PENDING)


def get_user_by_username(username: str) -> dict[str, Any] | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = %s", ((username or "").strip().lower(),)
        ).fetchone()
        return _user_dict(row) if row else None
    finally:
        conn.close()


def get_user_by_id(user_id: str | None) -> dict[str, Any] | None:
    if not user_id:
        return None
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE user_id = %s", (user_id,)).fetchone()
        return _user_dict(row) if row else None
    finally:
        conn.close()


def verify_login(username: str, password: str) -> tuple[dict[str, Any] | None, str | None]:
    """Returns (user, None) on success, or (None, error_message) on failure.
    Distinguishes 'wrong credentials' from 'account not yet approved' so the
    login page can show the right message."""
    user = get_user_by_username(username)
    if not user or not check_password_hash(user["password_hash"], password):
        return None, "Incorrect username or password."
    if user["status"] == STATUS_PENDING:
        return None, "Your account is awaiting supervisor approval. Try again once it's approved."
    if user["status"] == STATUS_REJECTED:
        return None, "This account request was not approved. Contact a supervisor."
    return user, None


def list_pending_contractor_accounts() -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM users WHERE role = 'contractor' AND status = %s ORDER BY created_at, user_id",
            (STATUS_PENDING,),
        ).fetchall()
        return [_user_dict(r) for r in rows]
    finally:
        conn.close()


def set_account_status(user_id: str, status: str) -> None:
    if status not in {STATUS_ACTIVE, STATUS_REJECTED, STATUS_PENDING}:
        raise ValueError(f"Invalid status: {status!r}")
    conn = get_connection()
    try:
        conn.execute("UPDATE users SET status = %s WHERE user_id = %s", (status, user_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
