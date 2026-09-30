"""Hybrid semantic L5/L6 matching engine.

This replaces the old "hand the LLM all 70 activities and hope" matcher
with a two-stage pipeline that mirrors the target production architecture
described below:

    Sentence-Transformer embeddings -> pgvector ANN search -> Cross-Encoder
    reranking -> Python business rules -> confidence + review decision

Stage 1 (this module, `SemanticMatchIndex.retrieve`) is a *hybrid lexical +
semantic retriever* standing in for "Sentence-Transformer + pgvector". It
combines:

  - TF-IDF over character n-grams (robust to partial/substring overlap and
    domain vocabulary quirks, e.g. "24-inch CS spool" vs "Line 24-XX")
  - spaCy static word-vector cosine similarity (`en_core_web_md`), which
    gives genuine distributional semantics (captures "erect" ~ "erection"
    ~ "install") without needing a live Hugging Face model download
  - RapidFuzz token-set fuzzy matching (robust to word reordering/typos)
  - deterministic asset/discipline/location rule bonuses (never date-based,
    per the project's explicit design rule)

  A true sentence-transformer + pgvector retriever can be swapped in later
  by reimplementing `SemanticMatchIndex` with the same `retrieve()`
  signature -- nothing downstream needs to change.

Stage 2 (`llm_rerank`) stands in for the "Cross-Encoder reranking" step: an
LLM call scoped ONLY to the retrieved shortlist (not the full schedule),
which is both cheaper and far more reliable than asking an LLM to search
70 activities blind.

Stage 3 (`match_progress_event_v2`) does confidence calibration and the
AUTO_ACCEPTED / REVIEW / UNMATCHED decision, and returns the full ranked
candidate list as an audit trail.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .prompts import load_prompt
from .providers import get_provider, provider_label
from .schemas import MatchResult

# ==========================================================================
# MATCH DECISION CONFIG
#
# Every number that decides AUTO_ACCEPTED / REVIEW / UNMATCHED lives in this
# one block. They were set by hand: real field data will not be shared, so
# they cannot be tuned on real reports yet. Re-measure them with the labelled
# test set (`python -m src.calibrate`) after any change and record the
# result in the README ("Calibration baseline").
# ==========================================================================

# Final confidence at or above this -> AUTO_ACCEPTED (the plan is changed
# without a human). Everything below is a safety cap that must stay under it.
CONF_AUTO_ACCEPT = 0.85

# Final confidence at or above this (and below CONF_AUTO_ACCEPT) -> REVIEW.
# Below it the result is UNMATCHED (probably a new / unlisted activity).
CONF_REVIEW = 0.55

# If the best retrieval score is below this, the LLM is not even called: the
# search found nothing resembling the event.
CONF_SKIP_LLM = 0.20

# Auto-accept needs the chosen activity's retrieval score to beat EVERY
# rival (see `find_rivals`) by at least this much. A smaller margin means the
# event does not carry enough detail to tell the activities apart.
AUTO_ACCEPT_MIN_MARGIN = 0.15

# Confidence ceiling when the choice is ambiguous. Must be < CONF_AUTO_ACCEPT,
# otherwise the cap would never send anything to review.
AMBIGUITY_CAP = 0.80

# Stricter ceiling when the rival is a near-duplicate of the chosen
# activity (description similarity >= NEAR_DUPLICATE_SIMILARITY), for example
# "Header-A" vs "Header-B". Must be <= AMBIGUITY_CAP.
AMBIGUITY_CAP_NEAR_DUPLICATE = 0.75

# Description similarity (0..1, RapidFuzz token-set ratio) at which another
# candidate counts as a look-alike, and as a near-duplicate.
RIVAL_SIMILARITY = 0.50
NEAR_DUPLICATE_SIMILARITY = 0.70

# Sanity floor: if even the chosen activity's retrieval score is below
# WEAK_RETRIEVAL, confidence is capped at WEAK_RETRIEVAL_CAP (weak evidence).
WEAK_RETRIEVAL = 0.30
WEAK_RETRIEVAL_CAP = 0.60

# An UNMATCHED result reports at most this share of the best retrieval
# score, and never more than the LLM's own confidence. Keeps the number below
# CONF_REVIEW so the screen never says "no match, 99%".
UNMATCHED_RETRIEVAL_SHARE = 0.5


# --------------------------------------------------------------------------
# RETRIEVAL CONFIG. Same rule as above: hand-set, re-measure with
# `python -m src.retrieval_eval` and `python -m src.calibrate` after any change.
# --------------------------------------------------------------------------

# How the three text signals are mixed into the base retrieval score.
W_LEXICAL = 0.30
W_SEMANTIC = 0.40
W_FUZZY = 0.15

# Added to a candidate's retrieval score when the report names its activity
# code or activity ID exactly, or its WBS group. An exact code identifies ONE
# activity, so it outranks everything; a WBS code only names a group, so it
# lifts the whole group above the field, and the choice inside it stays open.
ID_MATCH_BONUS = 0.80
WBS_MATCH_BONUS = 0.50
ID_MATCH_NOTE = "matched by ID"  # rule-note prefix; the reason text is built from it

# The LLM sees at least SHORTLIST_MIN candidates, plus every ID match
# and every activity on the event's own asset, never more than SHORTLIST_MAX.
SHORTLIST_MIN = 8
SHORTLIST_MAX = 15

# Share of the base score that comes from the cleaned raw report text
# (the rest comes from the extracted fields). 0 turns the signal off.
# On the 16 synthetic thin-field cases the share of correct activities that
# reach the shortlist was 81% at 0.10 and 94% at 0.30, while clear cases and the
# "noisy" guard cases stayed at rank 1 and the match decisions did not change.
# Only 24 synthetic cases back this, so re-measure it whenever the case set grows.
RAW_TEXT_WEIGHT = 0.30


def validate_thresholds() -> None:
    """Fail loudly at import time if the config block is inconsistent."""
    if not (0 < CONF_SKIP_LLM < CONF_REVIEW < CONF_AUTO_ACCEPT <= 1):
        raise ValueError("Thresholds must satisfy 0 < CONF_SKIP_LLM < CONF_REVIEW < CONF_AUTO_ACCEPT <= 1")
    for name, cap in (("AMBIGUITY_CAP", AMBIGUITY_CAP), ("AMBIGUITY_CAP_NEAR_DUPLICATE", AMBIGUITY_CAP_NEAR_DUPLICATE)):
        if not (CONF_REVIEW <= cap < CONF_AUTO_ACCEPT):
            raise ValueError(f"{name}={cap} must sit in [CONF_REVIEW, CONF_AUTO_ACCEPT) or it cannot force REVIEW")
    if AMBIGUITY_CAP_NEAR_DUPLICATE > AMBIGUITY_CAP:
        raise ValueError("AMBIGUITY_CAP_NEAR_DUPLICATE must not exceed AMBIGUITY_CAP")
    if not (0 < UNMATCHED_RETRIEVAL_SHARE * 1.0 < CONF_REVIEW):
        raise ValueError("UNMATCHED_RETRIEVAL_SHARE must keep an UNMATCHED confidence below CONF_REVIEW")
    if not (1 <= SHORTLIST_MIN <= SHORTLIST_MAX):
        raise ValueError("Shortlist sizes must satisfy 1 <= SHORTLIST_MIN <= SHORTLIST_MAX")
    if not (0 <= RAW_TEXT_WEIGHT < 1):
        raise ValueError("RAW_TEXT_WEIGHT must be in [0, 1)")
    if not (0 < WBS_MATCH_BONUS <= ID_MATCH_BONUS):
        raise ValueError("Need 0 < WBS_MATCH_BONUS <= ID_MATCH_BONUS (an exact code is stronger than a group)")


validate_thresholds()

_NLP = None


def _get_nlp():
    """Lazily load the spaCy model once per process."""
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_md", disable=["ner", "parser", "lemmatizer"])
    return _NLP


def build_match_text(row: dict[str, Any], include_codes: bool = True) -> str:
    """Build the searchable text representation of one planned L6 activity.

    The activity code and WBS code are part of it, so a report that quotes
    `PIP-A100-103` or `A100.02` (even partly) can find the row. The index feeds
    the WITH-codes text only to the character n-gram signal: codes are lookup
    keys, not words, and mixing them into the word-vector and fuzzy signals
    measurably blurred ordinary matching (see README, "Retrieval baseline").
    """
    parts = [
        row.get("description") or "",
        row.get("discipline") or "",
        row.get("asset") or "",
        row.get("location") or "",
    ]
    if include_codes:
        parts += [row.get("activity_code") or "", row.get("wbs_code") or ""]
    return " | ".join(p for p in parts if p)


def build_event_text(event: dict[str, Any]) -> str:
    """Build the searchable text representation of one extracted progress event."""
    parts = [
        event.get("activity_description") or "",
        event.get("discipline") or "",
        event.get("asset") or "",
        event.get("location") or "",
        event.get("activity_ref") or "",  # An ID written in the report
    ]
    return " | ".join(p for p in parts if p)


# --------------------------------------------------------------------------
# The raw report text as a second, light retrieval signal
# --------------------------------------------------------------------------

_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
    re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b"),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTHS}(?:,?\s+\d{{4}})?", re.I),
    re.compile(rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b", re.I),
    re.compile(r"\b(?:today|yesterday|tomorrow)\b", re.I),
]
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent\b|per\s*cent\b)", re.I)
_BY_CONTRACTOR_RE = re.compile(r"\b(?:by|contractor\s*[:\-])\s+[A-Z][\w&.'-]*(?:\s+[A-Z][\w&.'-]*){0,3}")
_NOISE_RE = re.compile(r"[^A-Za-z0-9\-./&]+")


def clean_raw_text(raw_text: str | None, contractor: str | None = None) -> str:
    """Lightly clean a report sentence for use as a retrieval signal.

    Removes what never helps to identify the WORK: dates, percentages, the
    contractor's name and punctuation noise. Everything else (asset tags, area
    names, the activity wording) is kept. Returns "" when nothing useful is left.
    """
    text = str(raw_text or "")
    if contractor and contractor.strip():
        text = re.sub(re.escape(contractor.strip()), " ", text, flags=re.I)
    text = _BY_CONTRACTOR_RE.sub(" ", text)
    for pattern in _DATE_PATTERNS:
        text = pattern.sub(" ", text)
    text = _PERCENT_RE.sub(" ", text)
    text = _NOISE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" -./&")
    return text if len(text) >= 3 else ""


# --------------------------------------------------------------------------
# Activity code / activity ID / WBS named in the report
# --------------------------------------------------------------------------

_REF_SEPARATORS = re.compile(r"[\s_\u2010-\u2015\u2212]+")


def _normalize_ref_text(text: Any) -> str:
    """Upper-case and unify separators, so 'pip a100 103' and 'PIP-A100-103' compare equal."""
    return _REF_SEPARATORS.sub("-", str(text or "").upper())


_REF_TEXT_FIELDS = ("activity_ref", "raw_text", "activity_description", "asset", "evidence_quote")


@dataclass
class Candidate:
    activity_id: str
    row: dict[str, Any]
    lexical_score: float
    semantic_score: float
    fuzzy_score: float
    rule_bonus: float
    retrieval_score: float
    rule_notes: list[str] = field(default_factory=list)
    raw_score: float = 0.0  # How well the cleaned raw report text matched (0 when unused)

    def to_payload(self) -> dict[str, Any]:
        return {
            "activity_id": self.activity_id,
            "description": self.row.get("description"),
            "discipline": self.row.get("discipline"),
            "asset": self.row.get("asset"),
            "location": self.row.get("location"),
            "wbs_code": self.row.get("wbs_code"),
            "activity_code": self.row.get("activity_code"),
            "retrieval_score": self.retrieval_score,
            "retrieval_breakdown": {
                "lexical": self.lexical_score,
                "semantic": self.semantic_score,
                "fuzzy": self.fuzzy_score,
                "raw_text": self.raw_score,
                "rule_bonus": self.rule_bonus,
                "notes": self.rule_notes,
            },
        }


# --------------------------------------------------------------------------
# Identifier matching on TOKENS, not substrings
#
# "Partial match" used to mean "one string sits inside the other", which said
# that `Area A` ~ `Area AB`, `V-101A` ~ `V-101AB`, `V-101` ~ `TV-101` and
# `P-401` ~ `XP-401`. An identifier is now split into tokens (runs of letters
# and digits; every other character is a separator), and two identifiers match
# only through one of the explicit rules in IDENTIFIER_ALIAS_RULES below.
# Anything not covered by a rule does NOT match.
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: Any) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def _is_wildcard(token: str) -> bool:
    """`XX` in 'Line 24-XX' stands for "any sub-line"."""
    return len(token) >= 2 and set(token) == {"x"}


def _tokens_equal(a: str, b: str) -> bool:
    return a == b or _is_wildcard(a) or _is_wildcard(b)


_FAMILY_RE = re.compile(r"^(\d+)([a-z])?$")


def _same_family(a: str, b: str) -> bool:
    """`101` ~ `101a` (a number and the same number with ONE letter suffix).

    Not `101a` ~ `101b` (two different members) and not `101a` ~ `101ab`.
    """
    ma, mb = _FAMILY_RE.match(a), _FAMILY_RE.match(b)
    if not (ma and mb):
        return False
    return ma.group(1) == mb.group(1) and (ma.group(2) is None) != (mb.group(2) is None)


def _rule_contiguous_tokens(a: str, b: str) -> bool:
    """The shorter identifier's tokens appear, in order and next to each other, in the longer one.

    Whole tokens only, so `line 1` is not in `line 10`, and `v-101` is not in
    `tv-101`. Two intended relaxations: an `XX` token matches any token
    (`Line 24` ~ `Line 24-XX`), and the LAST token may be a number against the
    same number plus one letter (`V-101` ~ `V-101A`, the same vessel family).
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    n = len(short)
    for start in range(len(long_) - n + 1):
        window = long_[start:start + n]
        if all(_tokens_equal(s, w) for s, w in zip(short[:-1], window[:-1])) and (
            _tokens_equal(short[-1], window[-1]) or _same_family(short[-1], window[-1])
        ):
            return True
    return False


_RANGE_RE = re.compile(r"([a-z]+)[-\s]?(\d+)\s*(?:\.\.|\bto\b)\s*(?:[a-z]+[-\s]?)?(\d+)")
_SLASH_RE = re.compile(r"([a-z]+)[-\s]?(\d+)/(\d+)\b")


def _numbered_ids(text: str) -> set[tuple[str, int]]:
    """Every `prefix number` pair in the text, e.g. 'jb-703' -> ('jb', 703)."""
    toks = _tokens(text)
    return {(t, int(nxt)) for t, nxt in zip(toks, toks[1:]) if t.isalpha() and nxt.isdigit()}


def _rule_range_or_slash(a: str, b: str) -> bool:
    """`JB-701..JB-710` / `JB-701 to JB-710` cover any `JB-70x`; `FT-301/302` covers FT-301 and FT-302."""
    for spec_text, other in ((a, b), (b, a)):
        spec_text = str(spec_text or "").lower()
        members: list[tuple[str, int, int]] = [
            (m.group(1), int(m.group(2)), int(m.group(3))) for m in _RANGE_RE.finditer(spec_text)
        ]
        ids = _numbered_ids(other)
        if any(pre == p and lo <= n <= hi for p, lo, hi in members for pre, n in ids):
            return True
        if any(pre == m.group(1) and n in (int(m.group(2)), int(m.group(3)))
               for m in _SLASH_RE.finditer(spec_text) for pre, n in ids):
            return True
    return False


class _AliasRule(NamedTuple):
    name: str
    match: Callable[[str, str], bool]
    # (a, b) pairs that MUST match; documents the intent and is checked by the tests.
    examples: tuple[tuple[str, str], ...]


# The explicit table of intended equivalences. To allow a new kind of
# equivalence, add a rule here with examples; do not loosen the rules above.
IDENTIFIER_ALIAS_RULES: tuple[_AliasRule, ...] = (
    _AliasRule(
        "whole-token containment (XX wildcard, number ~ number + one letter)",
        _rule_contiguous_tokens,
        (("Line 24", "Line 24-XX"), ("Line 24-05", "Line 24-XX"), ("V-101", "V-101A"),
         ("V-101B", "V-101 Slug Catcher"), ("Valve V-101A", "V-101A"), ("Area 100", "Area 100 - Inlet & Separation"),
         ("K-201", "K-201/K-202 Vibration")),
    ),
    _AliasRule(
        "range and slash shorthand",
        _rule_range_or_slash,
        (("JB-703", "JB-701..JB-710"), ("JB-710", "JB-701 to JB-710"), ("FT-302", "FT-301/302")),
    ),
)


def _identifiers_partially_match(a: str, b: str) -> bool:
    """True only for the intended relationships in IDENTIFIER_ALIAS_RULES.

    False for accidental look-alikes: `Area A`/`Area AB`, `V-101A`/`V-101AB`,
    `V-101`/`TV-101`, `P-401`/`XP-401`, `line 1`/`line 10`.
    """
    if not a or not b:
        return False
    return any(rule.match(a, b) for rule in IDENTIFIER_ALIAS_RULES)


class SemanticMatchIndex:
    """Hybrid lexical + semantic retrieval index over one L6 plan snapshot."""

    def __init__(self, schedule: list[dict[str, Any]]):
        self.schedule = schedule
        self.match_texts = [build_match_text(r, include_codes=False) for r in schedule]  # semantic + fuzzy
        self.lexical_texts = [build_match_text(r) for r in schedule]  # n-grams, with codes

        self._tfidf: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        if self.lexical_texts:
            self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
            self._tfidf_matrix = self._tfidf.fit_transform(self.lexical_texts)

        nlp = _get_nlp()
        self._docs = [nlp(t) for t in self.match_texts]

        # Every ID this plan can be addressed by -> the rows it names.
        # key (normalized) -> [(row index, kind, original text)]
        self._refs: dict[str, list[tuple[int, str, str]]] = {}
        for i, row in enumerate(schedule):
            for column, kind in (("activity_id", "activity ID"), ("activity_code", "activity code"), ("wbs_code", "WBS group")):
                value = str(row.get(column) or "").strip()
                if value:
                    self._refs.setdefault(_normalize_ref_text(value), []).append((i, kind, value))
        self._ref_pattern = None
        if self._refs:
            keys = sorted(self._refs, key=len, reverse=True)
            self._ref_pattern = re.compile(r"(?<![A-Z0-9])(" + "|".join(re.escape(k) for k in keys) + r")(?![A-Z0-9])")

    def detect_references(self, event: dict[str, Any]) -> dict[int, tuple[str, str]]:
        """The rows whose activity code / activity ID / WBS group the report names.

        Looks for EXACT known codes (whole tokens only) in `activity_ref` and in
        the report text, so it works when the extractor filled `activity_ref`
        and when it did not. Returns {row index: (kind, code)}; an activity code
        or ID wins over a WBS group when a row is named both ways.
        """
        if self._ref_pattern is None:
            return {}
        haystack = " | ".join(_normalize_ref_text(event.get(k)) for k in _REF_TEXT_FIELDS if event.get(k))
        found: dict[int, tuple[str, str]] = {}
        for match in self._ref_pattern.finditer(haystack):
            for i, kind, original in self._refs[match.group(1)]:
                if i not in found or (found[i][0] == "WBS group" and kind != "WBS group"):
                    found[i] = (kind, original)
        return found

    def retrieve(self, event: dict[str, Any], top_k: int = 5) -> list[Candidate]:
        if not self.schedule:
            return []

        event_text = build_event_text(event)
        nlp = _get_nlp()
        event_doc = nlp(event_text)

        lexical_scores = [0.0] * len(self.schedule)
        if self._tfidf is not None and self._tfidf_matrix is not None:
            event_vec = self._tfidf.transform([event_text])
            lexical_scores = cosine_similarity(event_vec, self._tfidf_matrix)[0].tolist()

        # The cleaned raw text is scored the same way, as a second signal.
        raw_text = clean_raw_text(event.get("raw_text"), event.get("contractor")) if RAW_TEXT_WEIGHT > 0 else ""
        raw_lexical = [0.0] * len(self.schedule)
        raw_doc = None
        if raw_text:
            raw_doc = nlp(raw_text)
            if self._tfidf is not None and self._tfidf_matrix is not None:
                raw_lexical = cosine_similarity(self._tfidf.transform([raw_text]), self._tfidf_matrix)[0].tolist()

        references = self.detect_references(event)

        e_asset = (event.get("asset") or "").strip().lower()
        e_discipline = (event.get("discipline") or "").strip().lower()
        e_location = (event.get("location") or "").strip().lower()

        candidates: list[Candidate] = []
        for i, row in enumerate(self.schedule):
            semantic_score = 0.0
            cand_doc = self._docs[i]
            if event_doc.vector_norm and cand_doc.vector_norm:
                semantic_score = max(0.0, float(event_doc.similarity(cand_doc)))

            fuzzy_score = fuzz.token_set_ratio(event_text, self.match_texts[i]) / 100.0

            rule_bonus = 0.0
            notes: list[str] = []
            row_asset = (row.get("asset") or "").strip().lower()
            row_discipline = (row.get("discipline") or "").strip().lower()
            row_location = (row.get("location") or "").strip().lower()

            if e_asset and row_asset:
                if e_asset == row_asset:
                    rule_bonus += 0.25
                    notes.append("exact asset match")
                elif _identifiers_partially_match(e_asset, row_asset):
                    rule_bonus += 0.12
                    notes.append("partial asset match")

            if e_discipline and row_discipline:
                if e_discipline == row_discipline:
                    rule_bonus += 0.08
                    notes.append("discipline match")
                else:
                    rule_bonus -= 0.10
                    notes.append("discipline conflict")

            if e_location and row_location and (
                e_location == row_location or _identifiers_partially_match(e_location, row_location)
            ):
                rule_bonus += 0.05
                notes.append("location match")

            if i in references:
                kind, code = references[i]
                rule_bonus += WBS_MATCH_BONUS if kind == "WBS group" else ID_MATCH_BONUS
                notes.append(f"{ID_MATCH_NOTE} ({kind} {code})")

            field_base = W_LEXICAL * lexical_scores[i] + W_SEMANTIC * semantic_score + W_FUZZY * fuzzy_score
            raw_score = 0.0
            base = field_base
            if raw_doc is not None:
                raw_semantic = 0.0
                if raw_doc.vector_norm and cand_doc.vector_norm:
                    raw_semantic = max(0.0, float(raw_doc.similarity(cand_doc)))
                raw_fuzzy = fuzz.token_set_ratio(raw_text, self.match_texts[i]) / 100.0
                raw_score = W_LEXICAL * raw_lexical[i] + W_SEMANTIC * raw_semantic + W_FUZZY * raw_fuzzy
                base = (1 - RAW_TEXT_WEIGHT) * field_base + RAW_TEXT_WEIGHT * raw_score

            retrieval_score = max(0.0, min(1.0, base + rule_bonus))

            candidates.append(
                Candidate(
                    activity_id=row["activity_id"],
                    row=row,
                    lexical_score=round(float(lexical_scores[i]), 3),
                    semantic_score=round(semantic_score, 3),
                    fuzzy_score=round(fuzzy_score, 3),
                    rule_bonus=round(rule_bonus, 3),
                    retrieval_score=round(retrieval_score, 3),
                    rule_notes=notes,
                    raw_score=round(raw_score, 3),
                )
            )

        candidates.sort(key=lambda c: c.retrieval_score, reverse=True)
        return candidates[:top_k]


# Cache one index per distinct schedule (keyed by the set of activity IDs +
# count, so a stale index is never reused after the plan changes). 70 rows
# is small enough that rebuilding is cheap, but caching keeps a live demo
# snappy across repeated requests.
_INDEX_CACHE: dict[tuple, SemanticMatchIndex] = {}


def _schedule_cache_key(schedule: list[dict[str, Any]]) -> tuple:
    """Hash the actual matching fields, not just activity IDs, so an index
    is invalidated whenever description/asset/discipline/location change
    even though the ID set stayed the same."""
    import hashlib

    parts = []
    for r in sorted(schedule, key=lambda r: r["activity_id"]):
        fingerprint = "|".join(
            str(r.get(f) or "")
            for f in ("activity_id", "description", "asset", "discipline", "location", "activity_code", "wbs_code")
        )
        parts.append(hashlib.sha256(fingerprint.encode("utf-8")).hexdigest())
    return tuple(parts)


def _get_index(schedule: list[dict[str, Any]]) -> SemanticMatchIndex:
    key = _schedule_cache_key(schedule)
    idx = _INDEX_CACHE.get(key)
    if idx is None:
        idx = SemanticMatchIndex(schedule)
        _INDEX_CACHE.clear()  # only one schedule is ever live in this demo
        _INDEX_CACHE[key] = idx
    return idx


def llm_rerank(event: dict[str, Any], candidates: list[Candidate]) -> dict[str, Any]:
    """Cross-encoder-style rerank: ask the LLM to choose among the shortlist only."""
    payload = [c.to_payload() for c in candidates]
    # Keep the prompt lean: the LLM doesn't need the retrieval breakdown, only the score.
    lean_payload = [
        {k: v for k, v in c.items() if k != "retrieval_breakdown"} for c in payload
    ]
    prompt = load_prompt("matching_candidates.txt").format(
        event_json=json.dumps(event, indent=2, default=str),
        candidates_json=json.dumps(lean_payload, indent=2),
    )

    provider = get_provider()
    result = provider.generate_structured(prompt, MatchResult)

    valid_ids = {c.activity_id for c in candidates}
    if result.matched_activity_id is not None and result.matched_activity_id not in valid_ids:
        raise ValueError(
            f"LLM rerank returned {result.matched_activity_id!r}, which was not in the "
            "shortlisted candidates -- refusing to trust an invented activity ID."
        )

    result.match_method = f"CrossEncoderLLM:{provider_label(provider)}"
    return result.model_dump(mode="json")


def unmatched_confidence(llm_confidence: float, best_retrieval: float) -> float:
    """The one formula for the confidence of an UNMATCHED result.

    The lower of the LLM's own confidence and a share (default half) of the best
    retrieval score. Since retrieval scores are at most 1, this is at most 0.5,
    which is always below CONF_REVIEW.
    """
    return round(min(float(llm_confidence), best_retrieval * UNMATCHED_RETRIEVAL_SHARE), 3)


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _description_similarity(a: Candidate, b: Candidate) -> float:
    return fuzz.token_set_ratio(a.row.get("description") or "", b.row.get("description") or "") / 100.0


def _shares_event_asset(event: dict[str, Any], cand: Candidate) -> bool:
    """True when the event's asset text appears in the candidate's asset or description."""
    e_asset = _norm(event.get("asset"))
    if not e_asset:
        return False
    for text in (cand.row.get("asset"), cand.row.get("description")):
        text = _norm(text)
        if text and (e_asset == text or _identifiers_partially_match(e_asset, text)):
            return True
    return False


def _id_named_in_event(event: dict[str, Any], cand: Candidate) -> bool:
    """True when the event text quotes this activity's ID or activity code."""
    haystack = " ".join(
        str(event.get(k) or "") for k in ("raw_text", "activity_description", "asset", "evidence_quote", "activity_ref")
    ).upper()
    for key in ("activity_id", "activity_code"):
        ident = str(cand.row.get(key) or "").strip().upper()
        if ident and re.search(rf"(?<![A-Z0-9]){re.escape(ident)}(?![A-Z0-9])", haystack):
            return True
    return False


def _exact_support(event: dict[str, Any], chosen: Candidate, rival: Candidate) -> bool:
    """An exact asset match or an ID quoted in the event that favours `chosen` over `rival`.

    An exact asset match only helps when the rival does NOT have one too: if all
    the siblings match the asset exactly, the asset does not tell them apart.
    """
    if "exact asset match" in chosen.rule_notes and "exact asset match" not in rival.rule_notes:
        return True
    return _id_named_in_event(event, chosen) and not _id_named_in_event(event, rival)


def find_rivals(event: dict[str, Any], chosen: Candidate, ranked: list[Candidate]) -> list[Candidate]:
    """Every other activity the event could just as well refer to.

    A rival is any activity that
      * the retrieval step ranked ABOVE the chosen one (the LLM overruled the
        search, so the pick needs proof), or
      * reads like the chosen one (description similarity >= RIVAL_SIMILARITY), or
      * shares the event's asset text, or
      * sits in the same WBS group as the chosen one.
    `ranked` is the FULL ranked schedule, not only the shortlist, so a sibling that
    the shortlist cut off is still considered.
    """
    chosen_wbs = _norm(chosen.row.get("wbs_code"))
    chosen_rank = next((i for i, c in enumerate(ranked) if c.activity_id == chosen.activity_id), len(ranked))
    rivals: list[Candidate] = []
    for i, cand in enumerate(ranked):
        if cand.activity_id == chosen.activity_id:
            continue
        if (
            i < chosen_rank
            or _description_similarity(chosen, cand) >= RIVAL_SIMILARITY
            or _shares_event_asset(event, cand)
            or (chosen_wbs and _norm(cand.row.get("wbs_code")) == chosen_wbs)
        ):
            rivals.append(cand)
    return rivals


def assess_ambiguity(event: dict[str, Any], chosen: Candidate, ranked: list[Candidate]) -> dict[str, Any]:
    """Return {"ambiguous": bool, "cap": float | None, "notes": [str, ...]}.

    Ambiguous when some rival scored within AUTO_ACCEPT_MIN_MARGIN of the chosen
    activity (or above it) and no exact asset / ID evidence separates them.
    """
    notes: list[str] = []
    cap: float | None = None
    for rival in find_rivals(event, chosen, ranked):
        margin = round(chosen.retrieval_score - rival.retrieval_score, 3)
        if margin >= AUTO_ACCEPT_MIN_MARGIN or _exact_support(event, chosen, rival):
            continue
        near_duplicate = _description_similarity(chosen, rival) >= NEAR_DUPLICATE_SIMILARITY
        this_cap = AMBIGUITY_CAP_NEAR_DUPLICATE if near_duplicate else AMBIGUITY_CAP
        cap = this_cap if cap is None else min(cap, this_cap)
        if margin < 0:
            notes.append(f"{rival.activity_id} scored higher in retrieval ({rival.retrieval_score:.2f} vs {chosen.retrieval_score:.2f})")
        else:
            notes.append(f"{rival.activity_id} is too close (margin {margin:.2f}, needs {AUTO_ACCEPT_MIN_MARGIN:.2f})")
    return {"ambiguous": bool(notes), "cap": cap, "notes": notes}


def is_id_match(cand: Candidate) -> bool:
    """The report named this activity's code / ID / WBS group."""
    return any(note.startswith(ID_MATCH_NOTE) for note in cand.rule_notes)


def select_shortlist(
    event: dict[str, Any],
    ranked: list[Candidate],
    *,
    min_size: int | None = None,
    max_size: int | None = None,
) -> list[Candidate]:
    """The candidates the LLM gets to choose from.

    At least `min_size` (SHORTLIST_MIN), and always
      * every activity the report named by ID / activity code / WBS, then
      * every activity on the event's own asset (same-asset siblings),
    with the total capped at `max_size` (SHORTLIST_MAX). If the two lists are
    longer than the cap, ID matches keep their place first, then the best-ranked
    same-asset activities. ID matches are listed first, the rest in rank order.

    `ranked` must be the FULL ranked schedule, best first.
    """
    min_size = SHORTLIST_MIN if min_size is None else min_size
    max_size = SHORTLIST_MAX if max_size is None else max_size
    by_id = [c for c in ranked if is_id_match(c)]
    same_asset = [c for c in ranked if not is_id_match(c) and _shares_event_asset(event, c)]
    must_have = (by_id + same_asset)[:max_size]
    chosen = {c.activity_id for c in must_have}
    for cand in ranked:  # top up to the minimum, best first
        if len(chosen) >= min(min_size, max_size):
            break
        chosen.add(cand.activity_id)
    id_first = [c for c in by_id if c.activity_id in chosen]
    rest = [c for c in ranked if c.activity_id in chosen and not is_id_match(c)]
    return (id_first + rest)[:max_size]


def decide_match(
    event: dict[str, Any],
    ranked: list[Candidate],
    shortlist: list[Candidate],
    llm_result: dict[str, Any],
) -> dict[str, Any]:
    """Turn the LLM's pick plus the retrieval evidence into the final decision.

    Pure Python (no LLM, no I/O), so every rule can be tested with a stubbed
    `llm_result`. Returns matched_activity_id, confidence, review_status and
    `ambiguity` (why a close call was sent to review).
    """
    best_retrieval = max((c.retrieval_score for c in shortlist), default=0.0)
    llm_confidence = float(llm_result.get("confidence") or 0.0)
    matched_id = llm_result.get("matched_activity_id")
    llm_review_status = llm_result.get("review_status")
    by_id = {c.activity_id: c for c in shortlist}
    ambiguity: dict[str, Any] = {"ambiguous": False, "cap": None, "notes": []}

    if matched_id is None or matched_id not in by_id:
        # The LLM found no candidate (same formula in every UNMATCHED path).
        return {
            "matched_activity_id": None,
            "confidence": unmatched_confidence(llm_confidence, best_retrieval),
            "review_status": "UNMATCHED",
            "ambiguity": ambiguity,
        }

    chosen = by_id[matched_id]

    # Confidence starts from the LLM's own read of the evidence. Python then
    # applies CAPS; it never multiplies an already-low LLM confidence down
    # (that compounds two conservative signals into a falsely low result).
    confidence = llm_confidence

    ambiguity = assess_ambiguity(event, chosen, ranked)
    if ambiguity["ambiguous"]:
        confidence = min(confidence, ambiguity["cap"])
    if chosen.retrieval_score < WEAK_RETRIEVAL:
        confidence = min(confidence, WEAK_RETRIEVAL_CAP)  # weak evidence even for the winner

    confidence = round(confidence, 3)
    if confidence >= CONF_AUTO_ACCEPT:
        status = "AUTO_ACCEPTED"
    elif confidence >= CONF_REVIEW:
        status = "REVIEW"
    else:
        status = "UNMATCHED"

    # Defence in depth: an ambiguous pick can never be auto-accepted,
    # whatever the cap constants are later tuned to.
    if ambiguity["ambiguous"] and status == "AUTO_ACCEPTED":
        status = "REVIEW"
        confidence = min(confidence, round(CONF_AUTO_ACCEPT - 0.001, 3))

    # Hard floor: an explicit LLM REVIEW/UNMATCHED verdict is an ambiguity
    # safeguard. Python may downgrade a status the LLM was too generous about,
    # but must never relax one the LLM flagged as uncertain.
    if llm_review_status == "REVIEW" and status == "AUTO_ACCEPTED":
        status = "REVIEW"
        confidence = min(confidence, round(CONF_AUTO_ACCEPT - 0.001, 3))
    elif llm_review_status == "UNMATCHED" and status != "UNMATCHED":
        status = "UNMATCHED"

    if status == "UNMATCHED":
        # Never propose a low-confidence identity, and never show a high number.
        return {
            "matched_activity_id": None,
            "confidence": unmatched_confidence(min(llm_confidence, confidence), best_retrieval),
            "review_status": "UNMATCHED",
            "ambiguity": ambiguity,
        }

    return {
        "matched_activity_id": matched_id,
        "confidence": confidence,
        "review_status": status,
        "ambiguity": ambiguity,
    }


def match_progress_event_v2(
    event: dict[str, Any],
    schedule: list[dict[str, Any]],
    top_k: int | None = None,
) -> dict[str, Any]:
    """Full hybrid-retrieval + LLM-rerank matching pipeline. Works on plain dicts.

    The shortlist size is dynamic (`select_shortlist`). Pass `top_k` only to
    force a fixed size, for example in a test.
    """
    index = _get_index(schedule)
    # Rank the WHOLE schedule once: the shortlist comes from it, and the rest is
    # kept so ambiguity checks can see siblings the shortlist cut off.
    ranked = index.retrieve(event, top_k=max(len(schedule), top_k or 0))
    candidates = ranked[:top_k] if top_k else select_shortlist(event, ranked)
    candidate_payload = [c.to_payload() for c in candidates]

    if not candidates:
        return {
            "event_id": event.get("event_id"),
            "matched_activity_id": None,
            "match_method": "SEMANTIC_HYBRID+LLM",
            "confidence": 0.0,
            "reason": "No planned L6 activities are available to match against.",
            "review_status": "UNMATCHED",
            "candidates": [],
        }

    best_retrieval = max(c.retrieval_score for c in candidates)

    if best_retrieval < CONF_SKIP_LLM:
        return {
            "event_id": event.get("event_id"),
            "matched_activity_id": None,
            "match_method": "SEMANTIC_HYBRID (retrieval score too low, LLM rerank skipped)",
            "confidence": unmatched_confidence(1.0, best_retrieval),
            "reason": (
                f"No planned L6 activity resembles this event closely enough "
                f"(best retrieval score {best_retrieval:.2f}). Flagged as a possible "
                f"new/unlisted activity for planner review rather than force-matched."
            ),
            "review_status": "UNMATCHED",
            "candidates": candidate_payload,
        }

    llm_result = llm_rerank(event, candidates)
    decision = decide_match(event, ranked, candidates, llm_result)

    reason = llm_result.get("reason", "")
    chosen = next((c for c in candidates if c.activity_id == decision["matched_activity_id"]), None)
    if chosen is not None:
        for note in chosen.rule_notes:
            if note.startswith(ID_MATCH_NOTE):  # Say so when the report named the activity
                reason = f"{reason} The report names it: {note}.".strip()
                break
    if decision["ambiguity"]["ambiguous"]:
        reason = (
            f"{reason} Sent to review because the choice is not clear-cut: "
            + "; ".join(decision["ambiguity"]["notes"]) + "."
        ).strip()

    return {
        "event_id": event.get("event_id"),
        "matched_activity_id": decision["matched_activity_id"],
        "match_method": llm_result.get("match_method", "SEMANTIC_HYBRID+LLM"),
        "confidence": decision["confidence"],
        "reason": reason,
        "review_status": decision["review_status"],
        "candidates": candidate_payload,
        "ambiguity": decision["ambiguity"],
    }
