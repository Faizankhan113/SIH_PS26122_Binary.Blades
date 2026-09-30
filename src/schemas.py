from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .timeutil import parse_actual


Status = Literal[
    "NOT_STARTED",
    "STARTED",
    "IN_PROGRESS",
    "COMPLETED",
    "DELAYED",
    "BLOCKED",
    "CANCELLED",
    "UNKNOWN",
]
# ERROR means the matcher could not run (engine exception, schedule
# unavailable). It is a processing failure, not a finding about the report: it is
# never stored in the planner or supervisor queues and is offered a Retry instead.
ReviewStatus = Literal["AUTO_ACCEPTED", "REVIEW", "UNMATCHED", "ERROR"]


class ExtractedProgress(BaseModel):
    """Facts extracted from one contractor/site statement."""

    model_config = ConfigDict(extra="forbid")

    activity_description: str = Field(min_length=1)
    asset: str | None = None
    discipline: str | None = None
    location: str | None = None
    status: Status
    # Date-times in PROJECT-LOCAL wall-clock time (no UTC offset). A plain
    # date is stored as midnight, and the matching *_time_stated flag says
    # whether the report really gave a time of day. A time is never invented:
    # "morning shift" or "end of shift" leave the flag False.
    actual_start: datetime | None = None
    actual_finish: datetime | None = None
    actual_start_time_stated: bool = False
    actual_finish_time_stated: bool = False
    progress_percent: float | None = Field(default=None, ge=0, le=100)
    contractor: str | None = None
    delay_reason_reported: str | None = None
    activity_ref: str | None = Field(
        default=None,
        description=(
            "A schedule identifier EXPLICITLY written in the statement: an activity code "
            "(PIP-A100-103), an activity ID (L6-103) or a WBS code (A100.02). Not an asset tag. "
            "Null when the statement contains none."
        ),
    )
    evidence_quote: str | None = Field(
        default=None,
        description="Short verbatim quote from the source supporting the extracted facts.",
    )

    @field_validator("actual_start", "actual_finish", mode="before")
    @classmethod
    def _accept_dates_and_date_only_text(cls, value: Any) -> Any:
        """Accept a plain date / "YYYY-MM-DD" (as midnight) as well as a full date-time.

        Unreadable text is passed through so pydantic reports it as a normal
        validation error. An empty string means "not stated".
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        parsed, _has_time = parse_actual(value)
        return parsed if parsed is not None else value


class ProgressEvent(ExtractedProgress):
    """Canonical event after extraction, normalization, and validation."""

    event_id: str
    raw_text: str = Field(min_length=1)
    delay_reason_category: str | None = None


class PlannedActivity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    activity_id: str
    level: Literal["L5", "L6"]
    project_id: str | None = None
    wbs_code: str | None = None
    activity_code: str | None = None
    description: str
    discipline: str | None = None
    asset: str | None = None
    location: str | None = None
    planned_start: date | None = None
    planned_finish: date | None = None
    planned_duration: int | None = None


class MatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    matched_activity_id: str | None = None
    match_method: str
    confidence: float = Field(ge=0, le=1)
    reason: str
    review_status: ReviewStatus
