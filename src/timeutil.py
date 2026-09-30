"""Project-local, timezone-aware time helpers.

Every timestamp the app stores or shows should come from here, so that all of
them carry a UTC offset (e.g. ``2026-09-29T14:03:11+05:30``) and none of them
depend on the server's own timezone setting.

The project timezone comes from the ``PS26122_TIMEZONE`` environment variable
(an IANA name such as ``Asia/Kolkata``) and defaults to ``Asia/Kolkata``.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

TIMEZONE_ENV_VAR = "PS26122_TIMEZONE"
DEFAULT_TIMEZONE = "Asia/Kolkata"

# India has no daylight saving, so a fixed offset is an exact stand-in when the
# system has no timezone database (typical on a fresh Windows install without
# the `tzdata` package).
_IST_FALLBACK = timezone(timedelta(hours=5, minutes=30), "IST")


def project_timezone() -> tzinfo:
    """Return the project's timezone (read from the environment on every call)."""
    name = (os.getenv(TIMEZONE_ENV_VAR) or DEFAULT_TIMEZONE).strip()
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        if name == DEFAULT_TIMEZONE:
            return _IST_FALLBACK
        raise ValueError(
            f"Unknown timezone {name!r} in {TIMEZONE_ENV_VAR}. Use an IANA name such as "
            f"'{DEFAULT_TIMEZONE}' (on Windows you may also need `pip install tzdata`)."
        ) from exc


def now_local() -> datetime:
    """Current time as a timezone-aware datetime in the project timezone."""
    return datetime.now(project_timezone())


def now_iso() -> str:
    """Current project-local time as an ISO-8601 string with UTC offset."""
    return now_local().isoformat(timespec="seconds")


def today_local() -> date:
    """Today's date in the project timezone (not the server's)."""
    return now_local().date()


# ---------------------------------------------------------------------------
# Actual start / finish as date-times, with a "time was stated" flag
#
# Rules:
#   * Actual times are PROJECT-LOCAL wall-clock values. They are stored as naive
#     ISO text (no UTC offset), unlike the audit timestamps above.
#   * A time is never invented. A value only carries a time of day when the
#     report explicitly stated one; otherwise it is stored as a plain date
#     ("2026-08-30") and the flag is 0. A time of day therefore exists in the
#     stored text if and only if the flag is 1, so no screen can show a fake 00:00.
#   * Old rows (date-only text, flag 0) load unchanged.
# ---------------------------------------------------------------------------


def _local_naive(value: datetime) -> datetime:
    """Aware -> project-local wall clock without tzinfo; naive is returned unchanged."""
    if value.tzinfo is not None:
        return value.astimezone(project_timezone()).replace(tzinfo=None)
    return value


def parse_actual(value: Any) -> tuple[datetime | None, bool]:
    """Read a stored / incoming actual start-finish value.

    Returns ``(datetime, has_time_component)``. A date, or a string such as
    ``"2026-09-22"``, gives midnight and ``False``. ``"2026-09-22T18:30:00"`` (or
    with a space instead of the ``T``) gives that moment and ``True``.
    Anything unreadable gives ``(None, False)``.
    """
    if value is None or value == "":
        return None, False
    if isinstance(value, datetime):
        return _local_naive(value), True
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day), False
    text = str(value).strip()
    if not text:
        return None, False
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    if len(text) <= 10:
        try:
            parsed_day = date.fromisoformat(text)
        except ValueError:
            return None, False
        return datetime(parsed_day.year, parsed_day.month, parsed_day.day), False
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None, False
    return _local_naive(parsed), True


def canonical_actual(value: Any, time_stated: Any) -> tuple[str | None, int]:
    """The one way an actual start/finish is written to the database.

    Returns ``(text, flag)``: ``"YYYY-MM-DD"`` and 0, or ``"YYYY-MM-DDTHH:MM:SS"``
    and 1. A time is kept only when the flag is set AND the value really has a
    time component; otherwise the time part is dropped.
    """
    parsed, has_time = parse_actual(value)
    if parsed is None:
        return None, 0
    if bool(time_stated) and has_time:
        return parsed.replace(microsecond=0).isoformat(timespec="seconds"), 1
    return parsed.date().isoformat(), 0


def format_actual(value: Any, time_stated: Any = None) -> str:
    """Display text: ``2026-09-22`` or ``2026-09-22 18:30``. Empty string for no value.

    The time is shown only when ``time_stated`` is true. ``time_stated=None``
    means "unknown, decide from the value": a stored value that carries a time
    component is treated as timed (canonical values always do so consistently).
    """
    parsed, has_time = parse_actual(value)
    if parsed is None:
        return ""
    show_time = has_time if time_stated is None else (bool(time_stated) and has_time)
    return parsed.strftime("%Y-%m-%d %H:%M") if show_time else parsed.strftime("%Y-%m-%d")


def actual_day(value: Any) -> date | None:
    """The calendar day of a stored actual value (time ignored). Variance is in days."""
    parsed, _ = parse_actual(value)
    return parsed.date() if parsed else None


def duration_hours(start: Any, start_stated: Any, finish: Any, finish_stated: Any) -> float | None:
    """Actual duration in hours -- only when BOTH times were explicitly stated.

    Returns None when either time is missing or the finish is before the start.
    """
    s_val, s_flag = canonical_actual(start, start_stated)
    f_val, f_flag = canonical_actual(finish, finish_stated)
    if not (s_flag and f_flag):
        return None
    delta = datetime.fromisoformat(f_val) - datetime.fromisoformat(s_val)
    hours = delta.total_seconds() / 3600
    return round(hours, 1) if hours >= 0 else None


def canonicalize_event_dates(event: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of an event dict with actual_start/finish in canonical form.

    Used when an event leaves the extraction step (JSON from a date-time model
    field always carries ``T00:00:00`` for a date-only value; this strips it
    unless the matching ``*_time_stated`` flag is set). Also makes sure the two
    flags are present and are real booleans.
    """
    out = dict(event)
    for key in ("actual_start", "actual_finish"):
        flag_key = f"{key}_time_stated"
        value, flag = canonical_actual(out.get(key), out.get(flag_key))
        out[key] = value
        out[flag_key] = bool(flag)
    return out
