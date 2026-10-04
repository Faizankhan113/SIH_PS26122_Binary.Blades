-- PS 26122: PostgreSQL schema, v1 (the eight-table design from PG_PLAN.md).
--
-- Safe to run repeatedly: every statement is IF NOT EXISTS / CREATE OR REPLACE.
-- Applied by `python -m src.db init` (and by `python -m src.bootstrap`).
--
-- Layers
--   source material   reports, report_statements
--   AI output         statement_revisions (what was extracted / corrected),
--                     matching_results    (what the matcher decided)
--   human decisions   supervisor_decisions (append-only; the ONLY thing that
--                     may authorise a change to live data)
--   live state        main (one row per activity), main_updates (append-only)
--   reference         activities (planned fields only), users
--   scratch           pipeline_runs, pipeline_run_items
--
-- Rules enforced here, not just in Python
--   * audited rows are never cascade-deleted
--   * supervisor_decisions and main_updates cannot be updated or deleted
--   * matching_results keeps the AI's answer: only the verdict columns may change
--   * a result has at most ONE terminal decision (exactly-once approval)
--   * a decision is applied to main at most once (main_updates.decision_id UNIQUE)
--   * only the system AUTO_ACCEPT decision may have no human user

-- --------------------------------------------------------------------------
-- Reference tables
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS users (
    user_id        text PRIMARY KEY,
    name           text NOT NULL,
    username       text NOT NULL UNIQUE,
    password_hash  text NOT NULL,
    role           text NOT NULL CHECK (role IN ('contractor', 'supervisor')),
    status         text NOT NULL DEFAULT 'active'
                   CHECK (status IN ('pending', 'active', 'rejected')),
    created_at     timestamptz NOT NULL DEFAULT now()
);

-- Planned (baseline) fields only. Live values live in "main".
CREATE TABLE IF NOT EXISTS activities (
    l6_id             text PRIMARY KEY,
    project_id        text NOT NULL,
    wbs_code          text,
    activity_code     text,
    description       text NOT NULL,
    discipline        text,
    asset             text,
    location          text,
    planned_start     date,
    planned_finish    date,
    planned_duration  integer CHECK (planned_duration IS NULL OR planned_duration >= 0),
    -- Set when an activity disappears from schedule.json. Never deleted:
    -- history refers to it.
    archived_at       timestamptz,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_activities_discipline ON activities (discipline);
CREATE INDEX IF NOT EXISTS ix_activities_project    ON activities (project_id);

-- --------------------------------------------------------------------------
-- Source material
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS reports (
    report_id            text PRIMARY KEY,
    source_type          text NOT NULL,              -- e.g. TEXT_FILE, UPLOAD, PASTED
    source_name          text,                       -- original file name, or 'Pasted text'
    file_location        text,                       -- where the uploaded file is kept, if kept
    -- When the report is about. A date with no time is stored as local midnight.
    -- "Newest report wins" in main is decided on this column.
    report_datetime      timestamptz NOT NULL,
    submitted_by_user_id text REFERENCES users (user_id),
    reported_by_name     text NOT NULL,
    status               text NOT NULL DEFAULT 'RECEIVED'
                         CHECK (status IN ('RECEIVED', 'PROCESSING', 'COMPLETED', 'FAILED')),
    created_at           timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_reports_datetime  ON reports (report_datetime);
CREATE INDEX IF NOT EXISTS ix_reports_submitter ON reports (submitted_by_user_id);

CREATE TABLE IF NOT EXISTS report_statements (
    statement_id         text PRIMARY KEY,
    report_id            text NOT NULL REFERENCES reports (report_id),
    seq                  integer NOT NULL CHECK (seq >= 1),   -- position inside the report
    raw_text             text NOT NULL,
    -- Hash of report date + reporter + normalised text (see compute_fingerprint).
    fingerprint          text,
    extraction_provider  text,                       -- gemini | groq
    extraction_model     text,
    prompt_version       text,
    -- true when extraction failed and the fields are keyword guesses.
    extraction_degraded  boolean NOT NULL DEFAULT false,
    extraction_error     text,
    status               text NOT NULL DEFAULT 'EXTRACTED'
                         CHECK (status IN ('EXTRACTED', 'MATCHED', 'REJECTED', 'FAILED')),
    created_at           timestamptz NOT NULL DEFAULT now(),
    UNIQUE (report_id, seq)
);

-- One live statement per fingerprint. REJECTED rows are excluded, so a rejected
-- statement can be corrected and submitted again. Rejecting must therefore set
-- report_statements.status = 'REJECTED' in the same transaction as the decision.
CREATE UNIQUE INDEX IF NOT EXISTS ux_report_statements_fingerprint
    ON report_statements (fingerprint)
    WHERE fingerprint IS NOT NULL AND status <> 'REJECTED';

CREATE INDEX IF NOT EXISTS ix_report_statements_report ON report_statements (report_id);

-- --------------------------------------------------------------------------
-- AI output
-- --------------------------------------------------------------------------

-- The extracted facts of a statement. Revision 1 is the AI extraction; later
-- revisions are contractor or supervisor corrections. Matching always points
-- at the exact revision it used.
CREATE TABLE IF NOT EXISTS statement_revisions (
    revision_id                text PRIMARY KEY,
    statement_id               text NOT NULL REFERENCES report_statements (statement_id),
    revision_no                integer NOT NULL CHECK (revision_no >= 1),
    origin                     text NOT NULL
                               CHECK (origin IN ('EXTRACTION', 'CONTRACTOR_EDIT', 'SUPERVISOR_EDIT')),
    created_by_user_id         text REFERENCES users (user_id),   -- NULL for EXTRACTION

    activity_description       text,
    asset                      text,
    discipline                 text,
    location                   text,
    -- Project-local wall-clock moments. A date with no time is local midnight;
    -- the *_time_stated flag is true only when the report really gave a time.
    actual_start               timestamptz,
    actual_finish              timestamptz,
    actual_start_time_stated   boolean NOT NULL DEFAULT false,
    actual_finish_time_stated  boolean NOT NULL DEFAULT false,
    progress_percent           numeric(5, 2)
                               CHECK (progress_percent IS NULL OR progress_percent BETWEEN 0 AND 100),
    status                     text
                               CHECK (status IS NULL OR status IN (
                                   'NOT_STARTED', 'STARTED', 'IN_PROGRESS', 'COMPLETED',
                                   'DELAYED', 'BLOCKED', 'CANCELLED', 'UNKNOWN')),
    contractor                 text,
    delay_reason_reported      text,
    delay_reason_category      text,
    evidence_quote             text,
    -- An ID / WBS code the report explicitly quoted.
    activity_ref               text,

    created_at                 timestamptz NOT NULL DEFAULT now(),
    UNIQUE (statement_id, revision_no),
    CHECK ((origin = 'EXTRACTION') = (created_by_user_id IS NULL))
);

CREATE INDEX IF NOT EXISTS ix_statement_revisions_statement ON statement_revisions (statement_id);

-- One row per matching attempt.
CREATE TABLE IF NOT EXISTS matching_results (
    result_id            text PRIMARY KEY,
    statement_id         text NOT NULL REFERENCES report_statements (statement_id),
    revision_id          text NOT NULL REFERENCES statement_revisions (revision_id),
    attempt_no           integer NOT NULL DEFAULT 1 CHECK (attempt_no >= 1),

    -- The AI's answer. Never overwritten (enforced by trigger below).
    -- NOT_FOUND = the matcher ran and found nothing. ERROR = the matcher failed.
    ai_status            text NOT NULL
                         CHECK (ai_status IN ('AUTO_ACCEPTED', 'REVIEW', 'NOT_FOUND', 'ERROR')),
    matched_activity_id  text REFERENCES activities (l6_id),
    confidence           numeric(4, 3) CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    match_method         text,
    reason               text,
    -- Top three candidates with scores:
    -- [{"activity_id": "...", "description": "...", "score": 0.0}, ...]
    top_candidates       jsonb NOT NULL DEFAULT '[]'::jsonb,
    matcher_version      text,

    -- Summary of the human verdict, for measuring AI accuracy. The full
    -- history is in supervisor_decisions. These three columns may change.
    final_status         text NOT NULL DEFAULT 'PENDING'
                         CHECK (final_status IN ('PENDING', 'AUTO_ACCEPTED', 'APPROVED',
                                                 'REJECTED', 'DISMISSED', 'MARKED_NEW')),
    final_activity_id    text REFERENCES activities (l6_id),
    final_decided_at     timestamptz,

    created_at           timestamptz NOT NULL DEFAULT now(),
    UNIQUE (statement_id, attempt_no),
    -- A target activity exists exactly when the AI picked one.
    CHECK ((ai_status IN ('AUTO_ACCEPTED', 'REVIEW')) = (matched_activity_id IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS ix_matching_results_statement ON matching_results (statement_id);
CREATE INDEX IF NOT EXISTS ix_matching_results_revision  ON matching_results (revision_id);
-- The review queue and the planner list.
CREATE INDEX IF NOT EXISTS ix_matching_results_pending
    ON matching_results (ai_status, created_at)
    WHERE final_status = 'PENDING';

-- --------------------------------------------------------------------------
-- Human decisions (append-only)
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS supervisor_decisions (
    decision_id          text PRIMARY KEY,
    result_id            text NOT NULL REFERENCES matching_results (result_id),
    decision_type        text NOT NULL
                         CHECK (decision_type IN ('AUTO_ACCEPT', 'APPROVE', 'REJECT',
                                                  'DISMISS', 'MARK_NEW_ACTIVITY', 'EDIT')),
    -- NULL only for the system's AUTO_ACCEPT decision.
    decided_by_user_id   text REFERENCES users (user_id),
    -- The activity the decision applies to (required when it can change main).
    target_activity_id   text REFERENCES activities (l6_id),
    note                 text,
    -- true when a supervisor approved an update from a failed (degraded) extraction.
    degraded_confirmed   boolean NOT NULL DEFAULT false,
    created_at           timestamptz NOT NULL DEFAULT now(),

    CHECK ((decision_type = 'AUTO_ACCEPT') = (decided_by_user_id IS NULL)),
    CHECK (decision_type NOT IN ('AUTO_ACCEPT', 'APPROVE') OR target_activity_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS ix_supervisor_decisions_result ON supervisor_decisions (result_id);

-- Exactly once: a matching result gets at most one terminal decision.
-- (EDIT is not terminal, so a result can be edited any number of times.)
CREATE UNIQUE INDEX IF NOT EXISTS ux_supervisor_decisions_terminal
    ON supervisor_decisions (result_id)
    WHERE decision_type IN ('AUTO_ACCEPT', 'APPROVE', 'REJECT', 'DISMISS', 'MARK_NEW_ACTIVITY');

-- --------------------------------------------------------------------------
-- Live state
-- --------------------------------------------------------------------------

-- One row per activity, created when the schedule is synced.
CREATE TABLE IF NOT EXISTS main (
    l6_id                      text PRIMARY KEY REFERENCES activities (l6_id),
    actual_start               timestamptz,
    actual_finish              timestamptz,
    actual_start_time_stated   boolean NOT NULL DEFAULT false,
    actual_finish_time_stated  boolean NOT NULL DEFAULT false,
    -- Cumulative, 0 to 100.
    progress_percent           numeric(5, 2) NOT NULL DEFAULT 0
                               CHECK (progress_percent BETWEEN 0 AND 100),
    status                     text NOT NULL DEFAULT 'NOT_STARTED'
                               CHECK (status IN ('NOT_STARTED', 'STARTED', 'IN_PROGRESS', 'COMPLETED',
                                                 'DELAYED', 'BLOCKED', 'CANCELLED', 'UNKNOWN')),
    contractor                 text,
    -- What last set this row.
    last_report_id             text REFERENCES reports (report_id),
    last_statement_id          text REFERENCES report_statements (statement_id),
    last_decision_id           text REFERENCES supervisor_decisions (decision_id),
    -- Copy of that report's report_datetime, used for "newest report wins".
    last_report_datetime       timestamptz,
    updated_at                 timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_main_status ON main (status);

CREATE TABLE IF NOT EXISTS main_updates (
    update_id          text PRIMARY KEY,
    -- UNIQUE: one decision can be applied (or recorded) only once.
    decision_id        text NOT NULL UNIQUE REFERENCES supervisor_decisions (decision_id),
    l6_id              text NOT NULL REFERENCES activities (l6_id),
    report_id          text NOT NULL REFERENCES reports (report_id),
    statement_id       text NOT NULL REFERENCES report_statements (statement_id),
    outcome            text NOT NULL CHECK (outcome IN ('APPLIED', 'HISTORICAL_ONLY', 'NO_CHANGE')),

    prev_actual_start      timestamptz,
    prev_actual_finish     timestamptz,
    prev_progress_percent  numeric(5, 2),
    prev_status            text,

    reported_actual_start      timestamptz,
    reported_actual_finish     timestamptz,
    reported_progress_percent  numeric(5, 2),
    reported_status            text,

    -- For HISTORICAL_ONLY and NO_CHANGE the new values equal the previous ones.
    new_actual_start       timestamptz,
    new_actual_finish      timestamptz,
    new_progress_percent   numeric(5, 2),
    new_status             text,

    created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_main_updates_activity  ON main_updates (l6_id, created_at);
CREATE INDEX IF NOT EXISTS ix_main_updates_report    ON main_updates (report_id);
CREATE INDEX IF NOT EXISTS ix_main_updates_statement ON main_updates (statement_id);

-- --------------------------------------------------------------------------
-- Scratch data for one in-progress report run (cleaned up automatically)
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          text PRIMARY KEY,
    user_id         text NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    ingestion_json  jsonb NOT NULL,
    match_json      jsonb
);

CREATE TABLE IF NOT EXISTS pipeline_run_items (
    run_id       text NOT NULL REFERENCES pipeline_runs (run_id) ON DELETE CASCADE,
    idx          integer NOT NULL,
    statement    text NOT NULL,
    state        text NOT NULL DEFAULT 'pending'
                 CHECK (state IN ('pending', 'running', 'done', 'failed')),
    result_json  jsonb,
    error        text,
    updated_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, idx)
);

CREATE INDEX IF NOT EXISTS ix_pipeline_runs_user ON pipeline_runs (user_id);

-- --------------------------------------------------------------------------
-- Guards
-- --------------------------------------------------------------------------

-- Audit tables are append-only. (TRUNCATE does not fire row triggers, so a
-- deliberate history reset or `drop` can still clear them.)
CREATE OR REPLACE FUNCTION ps26122_forbid_change() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION '% is append-only: % is not allowed', TG_TABLE_NAME, TG_OP
        USING ERRCODE = 'integrity_constraint_violation';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_supervisor_decisions_append_only ON supervisor_decisions;
CREATE TRIGGER trg_supervisor_decisions_append_only
    BEFORE UPDATE OR DELETE ON supervisor_decisions
    FOR EACH ROW EXECUTE FUNCTION ps26122_forbid_change();

DROP TRIGGER IF EXISTS trg_main_updates_append_only ON main_updates;
CREATE TRIGGER trg_main_updates_append_only
    BEFORE UPDATE OR DELETE ON main_updates
    FOR EACH ROW EXECUTE FUNCTION ps26122_forbid_change();

-- A matching result keeps the AI's answer. Only the verdict summary
-- (final_status, final_activity_id, final_decided_at) may change.
CREATE OR REPLACE FUNCTION ps26122_keep_ai_output() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'matching_results rows cannot be deleted'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF (NEW.result_id, NEW.statement_id, NEW.revision_id, NEW.attempt_no, NEW.ai_status,
        NEW.matched_activity_id, NEW.confidence, NEW.match_method, NEW.reason,
        NEW.top_candidates, NEW.matcher_version, NEW.created_at)
       IS DISTINCT FROM
       (OLD.result_id, OLD.statement_id, OLD.revision_id, OLD.attempt_no, OLD.ai_status,
        OLD.matched_activity_id, OLD.confidence, OLD.match_method, OLD.reason,
        OLD.top_candidates, OLD.matcher_version, OLD.created_at)
    THEN
        RAISE EXCEPTION 'matching_results: the AI output is never overwritten (only the verdict columns may change)'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_matching_results_keep_ai_output ON matching_results;
CREATE TRIGGER trg_matching_results_keep_ai_output
    BEFORE UPDATE OR DELETE ON matching_results
    FOR EACH ROW EXECUTE FUNCTION ps26122_keep_ai_output();

-- --------------------------------------------------------------------------
-- Schema version
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS schema_version (
    version     integer PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);

INSERT INTO schema_version (version) VALUES (1) ON CONFLICT (version) DO NOTHING;
