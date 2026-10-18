# T2.1e: SQLite and Postgres lines judged against their effective expectations

On the phase head after T2.1d's landing (1a37efe2), before T2.2 lands; the judging-path changes T2.2's SQLite job needs, moved out of T2.2 so the gate judges T2.2 with a checkpoint that holds them (the orchestrator's decision).
Each change has one focused test, which fails before it.

## What changed

- **A database line's baseline** (`ratchet.py effective`): a `<case>@sqlite` or `<case>@postgres` line, which no baseline holds, takes python-ref's committed line of that key as its baseline for python-core and rust-host (rust-host's without the stage and error it does not observe), moved by an approved entry as a DuckDB baseline is; DuckDB's lines keep theirs; the SQLite and Postgres contract jobs hand the effective expectations to `compare.py` as the gate and cross-host do (`test_t21e_database_expectations.py`).
- **`statements_differ`, an entry's opt-in** (`register.schema.json`, `register.py`, `compare.py`): a `case_outcomes` row of an owner-approved entry may name the call its old outcome refuses at, on a move from that refusal to a success; for that line, runner and call the call's `sql.count` and `sql.result` against the reference are reported as diagnostic; without it every call gates as before (`test_t21e_approved_answer.py`).

## Decisions

The opt-in is the orchestrator's decision of 2026-10-18, after both round 1 reviews.
E112 (Phase 1) holds a moved runner's statements at the reference's stopping call to the reference's, and Phase 1's guard (`test_t132g_later_stage.py`) keeps that so under an approved success; E112 is left as recorded.
Where the owner approved the answer itself, as KO2-61 approves `time.feature-time-of-day` answering on SQLite where the reference refuses at materialize, the reference's statements up to its refusal are no expectation of that call's, so the entry may say so for that call alone; nothing else on the line, and no unapproved or later-refusal move, is relaxed.
