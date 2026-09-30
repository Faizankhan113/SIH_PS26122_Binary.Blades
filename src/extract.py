from __future__ import annotations

import uuid
from datetime import date, datetime
from pathlib import Path

from .normalize import ExtractionValidationError, normalize_progress_event
from .providers import get_provider
from .prompts import load_prompt
from .schemas import ExtractedProgress, ProgressEvent
from .timeutil import today_local


def extract_progress_event(
    raw_text: str,
    *,
    report_date: date | None = None,
    message_time: datetime | None = None,
) -> ProgressEvent:
    """Extract one contractor progress event and normalize it locally.

    When no report date is supplied, today's date in the PROJECT timezone is
    used so relative expressions such as "today" resolve to the actual day of
    execution.

    `message_time` is the timestamp of the message itself, for chat
    statements: "just now" resolves to it. Leave it None for pasted reports and
    files, where no such moment exists.
    """
    if not raw_text.strip():
        raise ExtractionValidationError("Contractor statement cannot be empty.")

    # A report date is optional. For the normal application flow, automatically
    # use today's date rather than relying on a hard-coded demo date.
    report_date = report_date or today_local()

    event_id = f"EVT-{uuid.uuid4().hex[:8].upper()}"
    prompt = load_prompt("extraction.txt").format(
        report_date=report_date.isoformat(),
        contractor_statement=raw_text.strip(),
    )

    provider = get_provider()
    extracted = provider.generate_structured(prompt, ExtractedProgress)
    return normalize_progress_event(
        extracted,
        event_id=event_id,
        raw_text=raw_text,
        report_date=report_date,
        message_time=message_time,
    )


if __name__ == "__main__":
    statement_path = Path(__file__).resolve().parents[1] / "data" / "examples.txt"
    statement = next(
        line.strip()
        for line in statement_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    result = extract_progress_event(statement)
    print(result.model_dump_json(indent=2))
