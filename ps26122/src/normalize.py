from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import re
from typing import Any

from pydantic import ValidationError

from .schemas import ExtractedProgress, ProgressEvent
from .timeutil import format_actual, parse_actual


class ExtractionValidationError(ValueError):
    """The statement itself cannot become a valid event.

    Raised for problems in the CONTENT of a report, for example a finish date
    earlier than the start date. It is deliberately different from a provider
    or network failure: a keyword guess cannot repair it, so the app shows the
    reason and the original text and asks for a correction instead of
    fabricating a degraded event.
    """


STATUS_MAP = {
    "not started": "NOT_STARTED",
    "not_started": "NOT_STARTED",
    "notstarted": "NOT_STARTED",
    "done": "COMPLETED",
    "complete": "COMPLETED",
    "completed": "COMPLETED",
    "finished": "COMPLETED",
    "finish": "COMPLETED",
    "started": "STARTED",
    "start": "STARTED",
    "in progress": "IN_PROGRESS",
    "in_progress": "IN_PROGRESS",
    "ongoing": "IN_PROGRESS",
    "underway": "IN_PROGRESS",
    "under way": "IN_PROGRESS",
    "delayed": "DELAYED",
    "delay": "DELAYED",
    "behind schedule": "DELAYED",
    "blocked": "BLOCKED",
    "stopped": "BLOCKED",
    "cancelled": "CANCELLED",
    "canceled": "CANCELLED",
}

DISCIPLINE_MAP = {
    "pipe": "Piping",
    "piping": "Piping",
    "piping work": "Piping",
    "mechanical piping": "Piping",
    "mech piping": "Piping",
    "elec": "Electrical",
    "electrical": "Electrical",
    "electrical work": "Electrical",
    # Acronyms: without these the fallback (.title()) would turn them into
    # "Hse" / "Qa/Qc", which no longer equals the schedule's discipline value.
    "hse": "HSE",
    "health and safety": "HSE",
    "qa/qc": "QA/QC",
    "qaqc": "QA/QC",
}


REASON_RULES = [
    ("MATERIAL_AVAILABILITY", lambda key: re.search(r"material", key) and re.search(r"late|delay|unavailable|arriv|short", key)),
    ("MANPOWER", lambda key: re.search(r"manpower|labou?r|workers?", key) and re.search(r"short|lack|unavailable", key)),
    ("EQUIPMENT", lambda key: re.search(r"equipment|crane|machine", key) and re.search(r"breakdown|failure|unavailable|fault", key)),
]


def normalize_status(value: str | None) -> str:
    if not value:
        return "UNKNOWN"
    key = re.sub(r"\s+", " ", value.strip().lower())
    return STATUS_MAP.get(key, "UNKNOWN")


def normalize_discipline(value: str | None) -> str | None:
    if not value:
        return None
    key = re.sub(r"\s+", " ", value.strip().lower())
    return DISCIPLINE_MAP.get(key, value.strip().title())


_REF_SEP_RE = re.compile(r"[\s_\u2010-\u2015\u2212]+")
_REF_LABEL_RE = re.compile(r"^(?:wbs|activity\s*(?:code|id)|code|id|ref(?:erence)?)\s*[:#\-]?\s*", re.I)


def _ref_key(text: str) -> str:
    return _REF_SEP_RE.sub("-", text.strip().upper())


def normalize_activity_ref(value: str | None, raw_text: str) -> str | None:
    """Clean the extractor's `activity_ref` and drop it unless the text really contains it.

    The prompt says "only when explicitly written", but a model can still invent
    an identifier. A reference the statement does not contain is discarded, so
    an invented ID can never pull the matcher towards the wrong activity.
    Returns the upper-case form with spaces turned into '-', or None.
    """
    if not value or not value.strip():
        return None
    cleaned = _REF_LABEL_RE.sub("", value.strip()).strip(" .,;:()[]")
    if not cleaned:
        return None
    key = _ref_key(cleaned)
    haystack = _ref_key(raw_text or "")
    if not re.search(rf"(?<![A-Z0-9]){re.escape(key)}(?![A-Z0-9])", haystack):
        return None
    return key


def normalize_asset(value: str | None) -> str | None:
    if not value:
        return None
    value = re.sub(r"\s+", " ", value.strip())
    value = re.sub(r"\bline\s*[- ]?\s*(\d+)\b", r"Line \1", value, flags=re.I)
    value = re.sub(
        r"\bvalve\s*[- ]?\s*([A-Za-z0-9]+(?:[- ][A-Za-z0-9]+)*)\b",
        lambda m: f"Valve {m.group(1)}",
        value,
        flags=re.I,
    )
    return value


def normalize_location(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"\s+", " ", value.strip()).title()


def normalize_percent(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if 0 <= number <= 100 else None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", str(value), flags=re.I)
    return float(match.group(1)) if match else None


def normalize_reason_text(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"\s+", " ", value.strip()) or None


def categorize_delay_reason(value: str | None) -> str | None:
    if not value:
        return None
    key = re.sub(r"\s+", " ", value.strip().lower())
    for category, rule in REASON_RULES:
        if rule(key):
            return category
    return "OTHER"


def parse_common_date(value: Any, *, reference_date: date | None = None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()

    text = str(value).strip()
    lowered = text.lower()
    if reference_date is not None:
        if lowered == "today":
            return reference_date
        if lowered == "yesterday":
            return reference_date - timedelta(days=1)
        if lowered == "tomorrow":
            return reference_date + timedelta(days=1)

    # An ISO date-time ("2026-08-30T18:30:00") is a date for this function;
    # the time of day is read by parse_common_datetime().
    if re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d", text):
        parsed_iso, _ = parse_actual(text)
        if parsed_iso is not None:
            return parsed_iso.date()

    for fmt in (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d/%m/%y",
        "%d-%m-%y",
        "%d %b %Y",
        "%d %B %Y",
        "%d %b %y",
        "%d %B %y",
        "%d %b",
        "%d %B",
    ):
        try:
            parsed = datetime.strptime(text, fmt).date()
            if fmt in {"%d %b", "%d %B"} and reference_date is not None:
                parsed = parsed.replace(year=reference_date.year)
            return parsed
        except ValueError:
            continue

    # Handle day/month without year only when a report date gives us a year.
    if reference_date is not None:
        for fmt in ("%d %b", "%d %B", "%d/%m", "%d-%m"):
            try:
                parsed = datetime.strptime(text, fmt)
                return date(reference_date.year, parsed.month, parsed.day)
            except ValueError:
                continue

    return None


# --------------------------------------------------------------------------
# Explicit times of day
#
# Rule: a time is never invented. Only a clock time that is WRITTEN in the
# statement ("6:30 pm", "18:30", "1830 hrs", "at noon") counts. Vague words such
# as "morning shift", "end of shift", "evening" or "overnight" are ignored, and
# so are times that look forward-looking ("will finish by 6 pm"), negated
# ("not started until 6 pm") or ranges ("between 6 and 7 pm").
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClockMention:
    """One explicit clock time found in a statement."""

    start: int
    end: int
    value: time
    text: str


_AMPM_RE = re.compile(r"(?<![\d:.])(\d{1,2})(?:[:.]([0-5]\d))?\s*([ap])\.?m\b\.?", re.I)
_H24_COLON_RE = re.compile(r"(?<![\d:.])([01]?\d|2[0-3]):([0-5]\d)(?::[0-5]\d)?(?![\d:])")
_H24_DOT_HRS_RE = re.compile(r"(?<![\d:.])([01]?\d|2[0-3])\.([0-5]\d)\s*(?:hrs?|hours?)\b", re.I)
_H24_COMPACT_RE = re.compile(r"(?<![\d:.])([01]\d|2[0-3])([0-5]\d)\s*(?:hrs?|hours?)\b", re.I)
_NOON_RE = re.compile(r"\bat\s+(noon)\b", re.I)

# Words that make a nearby time NOT an actual, past time.
_NOT_ACTUAL_RE = re.compile(
    r"\b(?:will|shall|expected?|expecting|planned?|planning|scheduled?|target(?:ed)?|"
    r"until|till|tomorrow|next|to be|not|never|yet to|hasn'?t|haven'?t|didn'?t|"
    r"has not|have not|did not)\b",
    re.I,
)
_RANGE_AFTER_RE = re.compile(r"^\s*(?:-|\u2013|\u2014|to|and)\s*\d{1,2}(?:[:.]\d{2})?\s*(?:[ap]\.?m\b|hrs?\b|hours?\b)", re.I)
_RANGE_BEFORE_RE = re.compile(r"\d{1,2}(?:[:.]\d{2})?\s*(?:[ap]\.?m\.?)?\s*(?:-|\u2013|\u2014|to|and)\s*$", re.I)
_SENTENCE_BREAK_RE = re.compile(r"[.;!?](?=\s|$)")

_START_WORDS = (
    r"started|starts?|commenced?|commencing|began|begun|beginning|kicked\s+off|"
    r"mobili[sz]ed|initiated"
)
_FINISH_WORDS = (
    r"completed?|completion|finished|finish(?:es)?|done|ended|handed\s+over|"
    r"closed\s+out|wrapped\s+up"
)
_START_KEYWORD_RE = re.compile(rf"\b(?:{_START_WORDS})\b", re.I)
_FINISH_KEYWORD_RE = re.compile(rf"\b(?:{_FINISH_WORDS})\b", re.I)
_JUST_NOW_RE = re.compile(r"\b(?:just\s+now|right\s+now)\b", re.I)

# Words that tie a time to some OTHER event ("crew arrived at 8 am", "work stopped at 6 pm").
# A start/finish keyword on the far side of such a word does not apply to the time.
_OTHER_EVENT_RE = re.compile(
    r"\b(?:arriv\w*|reach\w*|left|leav\w*|depart\w*|deliver\w*|receiv\w*|issued|permit|meeting|"
    r"briefing|toolbox|lunch|break|inspect\w*|rain\w*|stopp?ed|resum\w*|called)\b",
    re.I,
)

_KEYWORD_REACH_BEFORE = 50
_KEYWORD_REACH_AFTER = 30


def _to_time(hour: int, minute: int, meridiem: str | None) -> time | None:
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if meridiem.lower() == "p" else 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return time(hour, minute)


def _sentence_bounds(text: str, pos: int) -> tuple[int, int]:
    """[start, end) of the sentence that contains `pos` (breaks are . ; ! ? followed by space)."""
    lo = 0
    for m in _SENTENCE_BREAK_RE.finditer(text, 0, pos):
        lo = m.end()
    hi_match = _SENTENCE_BREAK_RE.search(text, pos)
    hi = hi_match.start() + 1 if hi_match else len(text)
    return lo, hi


def find_clock_times(text: str) -> list[ClockMention]:
    """Explicit, past-tense clock times written in `text`, in reading order."""
    found: list[ClockMention] = []
    taken: list[tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        return any(a < e and s < b for s, e in taken)

    def add(m: re.Match[str], value: time | None) -> None:
        if value is None or overlaps(m.start(), m.end()):
            return
        taken.append((m.start(), m.end()))
        found.append(ClockMention(m.start(), m.end(), value, m.group(0).strip()))

    for m in _AMPM_RE.finditer(text):
        add(m, _to_time(int(m.group(1)), int(m.group(2) or 0), m.group(3)))
    for m in _H24_DOT_HRS_RE.finditer(text):
        add(m, _to_time(int(m.group(1)), int(m.group(2)), None))
    for m in _H24_COMPACT_RE.finditer(text):
        add(m, _to_time(int(m.group(1)), int(m.group(2)), None))
    for m in _H24_COLON_RE.finditer(text):
        add(m, _to_time(int(m.group(1)), int(m.group(2)), None))
    for m in _NOON_RE.finditer(text):
        add(m, time(12, 0))

    kept: list[ClockMention] = []
    for mention in sorted(found, key=lambda c: c.start):
        if _RANGE_AFTER_RE.match(text[mention.end:mention.end + 24]):
            continue
        if _RANGE_BEFORE_RE.search(text[max(0, mention.start - 24):mention.start]):
            continue
        lo, _hi = _sentence_bounds(text, mention.start)
        if _NOT_ACTUAL_RE.search(text[max(lo, mention.start - 25):mention.start]):
            continue
        kept.append(mention)
    return kept


def _classify_position(text: str, start: int, end: int) -> str | None:
    """Is the date/time at text[start:end] the START or the FINISH of the work?

    The nearest start/finish keyword in the same sentence decides ("started
    6:30 pm on 30 Aug and finished 9 pm": the first time is the start, the
    second the finish). A keyword that is negated or forward-looking ("will
    start", "not started") does not count. Returns "start", "finish" or None.
    """
    lo, hi = _sentence_bounds(text, start)
    best: tuple[int, int, str] | None = None  # (distance, before-first tiebreak, kind)
    for regex, kind in ((_START_KEYWORD_RE, "start"), (_FINISH_KEYWORD_RE, "finish")):
        for m in regex.finditer(text, lo, hi):
            if m.end() <= start:
                distance, side = start - m.end(), 0
                if distance > _KEYWORD_REACH_BEFORE:
                    continue
            elif m.start() >= end:
                distance, side = m.start() - end, 1
                if distance > _KEYWORD_REACH_AFTER:
                    continue
            else:
                continue
            if _NOT_ACTUAL_RE.search(text[max(lo, m.start() - 14):m.start()]):
                continue
            between = text[m.end():start] if side == 0 else text[end:m.start()]
            if _OTHER_EVENT_RE.search(between):
                continue
            candidate = (distance, side, kind)
            if best is None or candidate < best:
                best = candidate
    return best[2] if best else None


def _single_sentence(text: str) -> bool:
    return len([p for p in _SENTENCE_BREAK_RE.split(text.strip()) if p.strip()]) <= 1


def parse_common_datetime(
    value: Any,
    *,
    reference_date: date | None = None,
) -> tuple[datetime | None, bool]:
    """Parse a date, or a date with an explicit time, into ``(datetime, time_stated)``.

    ``time_stated`` is True only when the value really carries a time of day
    (an ISO date-time with a non-midnight time, or text such as "30 Aug 6:30 pm"
    or "today 18:30"). A plain date is midnight with ``False``. Times are
    project-local; an offset, if present, is converted to project-local time.
    """
    if value is None or value == "":
        return None, False
    if isinstance(value, datetime):
        naive, _ = parse_actual(value)
        assert naive is not None
        return naive, naive.time() != time(0, 0)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day), False

    text = str(value).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d", text):
        parsed_iso, _ = parse_actual(text)
        if parsed_iso is not None:
            return parsed_iso, True

    mentions = find_clock_times(text)
    if len(mentions) == 1:
        mention = mentions[0]
        remainder = (text[:mention.start] + " " + text[mention.end:]).strip(" ,;-@")
        remainder = re.sub(r"\b(?:at|on)\s*$", "", remainder, flags=re.I).strip(" ,;-@")
        remainder = re.sub(r"^\s*(?:at|on)\b\s*", "", remainder, flags=re.I).strip(" ,;-@")
        day = parse_common_date(remainder, reference_date=reference_date)
        if day is not None:
            return datetime.combine(day, mention.value), True
        return None, False

    day = parse_common_date(text, reference_date=reference_date)
    return (datetime(day.year, day.month, day.day), False) if day else (None, False)


def extract_finish_date_from_raw(raw_text: str, *, report_date: date | None, status: str) -> date | None:
    """Deterministic safeguard for explicit completion dates the LLM may miss."""
    if status != "COMPLETED":
        return None

    text = raw_text.strip()
    lowered = text.lower()

    # Relative completion date.
    if report_date is not None and re.search(r"\b(?:completed|complete|finished|done)\b", lowered) and re.search(r"\btoday\b", lowered):
        return report_date

    patterns = [
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b",
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))\b",
        r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            candidate = match.group(1)
            # Prefer a date near completion wording when there are multiple dates.
            left = text[max(0, match.start() - 35):match.end() + 35].lower()
            if re.search(r"complete|finished|done", left) or len(list(re.finditer(pattern, text, flags=re.I))) == 1:
                parsed = parse_common_date(candidate, reference_date=report_date)
                if parsed:
                    return parsed
    return None


_DATE_PATTERNS = (
    r"\b(\d{4}-\d{2}-\d{2})\b",
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b",
    r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))\b",
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\b",
)


def extract_start_date_from_raw(raw_text: str, *, report_date: date | None) -> date | None:
    """Deterministic safeguard for an explicit START date the LLM may miss.

    Looks for a written date (or "today" / "yesterday" resolved from the report
    date) whose nearest keyword is a start word ("started", "commenced",
    "began"...). Forward-looking or negated wording ("will start on 5 Sep",
    "not started") is ignored. Returns None when nothing is explicit.
    """
    text = raw_text.strip()
    candidates: list[tuple[int, date]] = []
    taken: list[tuple[int, int]] = []
    for pattern in _DATE_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.I):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            if _classify_position(text, m.start(), m.end()) != "start":
                continue
            parsed = parse_common_date(m.group(1), reference_date=report_date)
            if parsed:
                candidates.append((m.start(), parsed))
    if report_date is not None:
        for m in re.finditer(r"\b(today|yesterday)\b", text, flags=re.I):
            if _classify_position(text, m.start(), m.end()) == "start":
                parsed = parse_common_date(m.group(1), reference_date=report_date)
                if parsed:
                    candidates.append((m.start(), parsed))
    if not candidates:
        return None
    return sorted(candidates)[0][1]


def resolve_actual_moments(
    raw_text: str,
    *,
    status: str,
    report_date: date | None,
    start_day: date | None,
    finish_day: date | None,
    llm_start: datetime | None = None,
    llm_start_stated: bool = False,
    llm_finish: datetime | None = None,
    llm_finish_stated: bool = False,
    message_time: datetime | None = None,
) -> tuple[tuple[date | None, time | None], tuple[date | None, time | None]]:
    """Combine the dates with the explicit times written in the statement.

    Returns ``((start_day, start_time), (finish_day, finish_time))``. A time is
    ``None`` unless the statement really wrote one, so a date-only report stays
    date-only. Steps:

      1. A time the LLM returned is kept ONLY when the same clock time appears
         in the statement (an invented time is dropped).
      2. Each remaining clock time goes to start or finish by its nearest
         keyword ("started 6:30 pm ... finished 9 pm").
      3. A time with no keyword is used only when the statement is a single
         sentence and only one of the two moments has a date (and the status
         agrees: COMPLETED -> finish, STARTED -> start). Otherwise it is left
         unassigned rather than guessed.
      4. "just now" / "right now" use ``message_time`` (a chat message's own
         timestamp) when that is the same day as the report date.

    A time can only be attached to a date that exists: nothing is invented.
    """
    start_time: time | None = None
    finish_time: time | None = None
    mentions = find_clock_times(raw_text)
    used: set[int] = set()

    def take_llm_time(moment: datetime | None, stated: bool) -> time | None:
        if moment is None:
            return None
        candidate = moment.time().replace(second=0, microsecond=0)
        if not stated and candidate == time(0, 0):
            return None
        for idx, mention in enumerate(mentions):
            if idx not in used and mention.value == candidate:
                used.add(idx)
                return candidate
        return None  # not written in the statement: dropped

    if start_day is not None:
        start_time = take_llm_time(llm_start, llm_start_stated)
    if finish_day is not None:
        finish_time = take_llm_time(llm_finish, llm_finish_stated)

    for idx, mention in enumerate(mentions):
        if idx in used:
            continue
        kind = _classify_position(raw_text, mention.start, mention.end)
        if kind == "start" and start_time is None and start_day is not None:
            start_time = mention.value
            used.add(idx)
        elif kind == "finish" and finish_time is None and finish_day is not None:
            finish_time = mention.value
            used.add(idx)

    leftovers = [i for i in range(len(mentions)) if i not in used]
    if len(leftovers) == 1 and _single_sentence(raw_text):
        mention = mentions[leftovers[0]]
        sentence_lo, _ = _sentence_bounds(raw_text, mention.start)
        about_other_event = _OTHER_EVENT_RE.search(raw_text[max(sentence_lo, mention.start - 40):mention.start])
        if _classify_position(raw_text, mention.start, mention.end) is None and not about_other_event:
            if status == "COMPLETED" and finish_day is not None and start_day is None and finish_time is None:
                finish_time = mention.value
            elif status == "STARTED" and start_day is not None and finish_day is None and start_time is None:
                start_time = mention.value

    if message_time is not None and (report_date is None or message_time.date() == report_date):
        now_naive, _ = parse_actual(message_time)
        assert now_naive is not None
        now_day, now_clock = now_naive.date(), now_naive.time().replace(microsecond=0)
        for m in _JUST_NOW_RE.finditer(raw_text):
            kind = _classify_position(raw_text, m.start(), m.end())
            if kind is None:
                kind = "finish" if status == "COMPLETED" else ("start" if status == "STARTED" else None)
            if kind == "start" and start_time is None and start_day in (None, now_day):
                start_day, start_time = now_day, now_clock
            elif kind == "finish" and finish_time is None and finish_day in (None, now_day):
                finish_day, finish_time = now_day, now_clock

    return (start_day, start_time), (finish_day, finish_time)


def extract_contractor_from_raw(raw_text: str, *, existing: str | None) -> str | None:
    """Deterministic safeguard for common explicit contractor wording."""
    if existing:
        return existing.strip()

    text = raw_text.strip()
    patterns = [
        r"\bby\s+([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,6})\b(?=\s*(?:[.,;]|$|\b(?:on|for|in|at|because|due|with)\b))",
        r"\bcontractor\s*[:\-]\s*([A-Z][^,.;]+)",
        r"\bcontractor\s+([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,6})\b",
    ]
    stop_words = {"the", "on", "for", "in", "at", "because", "due", "with", "and"}
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = re.sub(r"\s+", " ", match.group(1).strip(" \t\n.,;:-"))
            words = [w for w in value.split() if w.lower() not in stop_words]
            value = " ".join(words).strip()
            if value:
                return value
    return None


def _moment(day: date | None, clock: time | None) -> tuple[datetime | None, bool]:
    if day is None:
        return None, False
    return datetime.combine(day, clock or time(0, 0)), clock is not None


def normalize_progress_event(
    extracted: ExtractedProgress,
    *,
    event_id: str,
    raw_text: str,
    report_date: date | None = None,
    message_time: datetime | None = None,
) -> ProgressEvent:
    """Turn the LLM's extraction into the canonical event.

    `message_time` is the timestamp of the message itself (a chat message). It
    is a fact, so "just now" may use it; it is never used for anything else.
    """
    actual_start = parse_common_date(extracted.actual_start, reference_date=report_date)
    actual_finish = parse_common_date(extracted.actual_finish, reference_date=report_date)
    progress = normalize_percent(extracted.progress_percent)
    status = normalize_status(extracted.status)
    reported_reason = normalize_reason_text(extracted.delay_reason_reported)

    # Source-text safeguard for explicit status wording that the LLM may omit.
    if status == "UNKNOWN" and re.search(r"\b(?:delayed|behind schedule)\b", raw_text, flags=re.I):
        status = "DELAYED"

    if progress is not None and progress >= 100:
        status = "COMPLETED"
    elif status == "COMPLETED" and progress is None:
        progress = 100.0
    elif progress is not None and progress < 100 and status == "UNKNOWN":
        status = "IN_PROGRESS"

    explicit_finish = extract_finish_date_from_raw(
        raw_text, report_date=report_date, status=status
    )
    actual_finish = actual_finish or explicit_finish
    actual_start = actual_start or extract_start_date_from_raw(raw_text, report_date=report_date)
    contractor = extract_contractor_from_raw(
        raw_text, existing=extracted.contractor
    )

    if progress is None and status == "COMPLETED":
        progress = 100.0

    # Attach explicit times of day (never invented) to the dates.
    (start_day, start_clock), (finish_day, finish_clock) = resolve_actual_moments(
        raw_text,
        status=status,
        report_date=report_date,
        start_day=actual_start,
        finish_day=actual_finish,
        llm_start=extracted.actual_start,
        llm_start_stated=extracted.actual_start_time_stated,
        llm_finish=extracted.actual_finish,
        llm_finish_stated=extracted.actual_finish_time_stated,
        message_time=message_time,
    )
    start_dt, start_stated = _moment(start_day, start_clock)
    finish_dt, finish_stated = _moment(finish_day, finish_clock)

    try:
        event = ProgressEvent(
            event_id=event_id,
            raw_text=raw_text.strip(),
            activity_description=extracted.activity_description.strip(),
            asset=normalize_asset(extracted.asset),
            discipline=normalize_discipline(extracted.discipline),
            location=normalize_location(extracted.location),
            status=status,
            actual_start=start_dt,
            actual_finish=finish_dt,
            actual_start_time_stated=start_stated,
            actual_finish_time_stated=finish_stated,
            progress_percent=progress,
            contractor=contractor,
            delay_reason_reported=reported_reason,
            delay_reason_category=categorize_delay_reason(reported_reason),
            activity_ref=normalize_activity_ref(extracted.activity_ref, raw_text),
            evidence_quote=extracted.evidence_quote.strip() if extracted.evidence_quote else None,
        )
    except ValidationError as exc:
        raise ExtractionValidationError(
            "The extracted facts are not valid: "
            + "; ".join(f"{'.'.join(str(x) for x in e['loc']) or 'event'}: {e['msg']}" for e in exc.errors())
        ) from exc

    if event.actual_start and event.actual_finish:
        # Full date-times are compared only when BOTH times were stated. A
        # date-only finish is midnight in storage, and comparing that with a
        # stated start time ("started 18:30, finished 30 Aug") would be a false alarm.
        both_timed = event.actual_start_time_stated and event.actual_finish_time_stated
        earlier = (
            event.actual_finish < event.actual_start
            if both_timed
            else event.actual_finish.date() < event.actual_start.date()
        )
        if earlier:
            what = "date and time" if both_timed else "date"
            raise ExtractionValidationError(
                f"The finish {what} ({format_actual(event.actual_finish, event.actual_finish_time_stated)}) "
                f"is earlier than the start {what} "
                f"({format_actual(event.actual_start, event.actual_start_time_stated)})."
            )

    return event
