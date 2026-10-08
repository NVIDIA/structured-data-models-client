# Phase 7.7 acceptance contract

The checks in this directory hold the three sampling branches the gap list claimed unreachable to scenarios of the parity corpus, and the core to those scenarios.
This file is the interface they test through.
Everything it doesn't name is the implementer's choice.

Phase 7.5's contract (`tests/rust_core/sampling/driver/CONTRACT.md`), Phase 7.6's (`tests/rust_core/link/CONTRACT.md`) and Phase 8.5's (`tests/rust_core/request/CONTRACT.md`) still hold: the recordings, the canonical form, the hooks in `scripts/parity/capture_sampling.py`, the binding's runs and the request host.
Today's Python, in the phase venv, is the reference throughout.

## Names

The three **branches**, as `scripts/parity/coverage.py` numbers them (`_gaps.BRANCHES`, which also checks each line's text, so a moved line fails loudly):

| Name | Module (under `packages/sdm-relational-engine/src/sdm_relational_engine/rfm/`) | Branch | Taken when |
|---|---|---|---|
| `sqlite-declared-key` | `backend/sqlite/sampler.py` | 244->245 | `_seeded_order_by` finds no dtype on the entity key's source column |
| `decimal-mixed-indices` | `rfm.py` | 262->267 | `_first_uncastable` runs out of values without one that won't cast alone |
| `self-link` | `base/sampler.py` | 428->429 | `_get_query_time_offset_dict` finds other than one edge type from the aggregated table to the entity table |

A transcript **takes** a branch when the branch's second line runs, in the function it belongs to, while the transcript replays through `_parity.corpus.check`, as `_gaps.LineWatch` notes with `sys.monitoring`.

- A **SQLite declared-key transcript** is a `sqlite` transcript that takes `sqlite-declared-key` in a call of `_seeded_order_by` whose entity key is a column of the source table with no dtype, and that column is declared, as the transcript's recorded `PRAGMA table_info` result spells it (upper case, white space removed), `DECIMAL(10,0)`, `NUMERIC` or nothing at all.
- A **decimal mixed-indices transcript** is a `duckdb` transcript that takes `decimal-mixed-indices` and ends in the refusal `'indices' must hold values matching the primary key of '<table>', which is decimal128(...)`.
- A **self-link transcript** is a `memory` transcript that takes `self-link` and ends in the refusal `Could not find a unique foreign key from table '<table>' to '<table>'`, the same table twice.

## Scenarios (`accept_gaps_scenarios.py`, P77.scenarios)

- There are SQLite declared-key transcripts covering, between them, each of the three declared types.
  Each sends at least one payload, every one answered with status 200, and ends without an error.
- There is at least one decimal mixed-indices transcript.
  Its last call is `PredictiveQuery.materialize`, with `indices` holding at least one `int` (not a `bool`) and one `str`, for the query asked last, whose `FOR EACH` key `data.column_types` stores as `DECIMAL(10,0)`.
  It ends in `sdm_client.errors.StructuredDataError`, code `INVALID_REQUEST`, refusing that table's key, and the message names no index (holds no `; indices[`).
- There is at least one self-link transcript.
  Its schema is made by `Schema.from_data`; its last call is `PredictiveQuery.materialize`, for a query asking `COUNT(<table>.*, ...)` `FOR EACH <table>.<key>`.
  It ends in `sdm_client.errors.StructuredDataError`, code `INVALID_REQUEST`.
- Each is an offline scenario of `scripts/parity/_parity/scenarios.py` (`by_name()`), stored under its name, that replays and that `run_corpus.py record --only <them> --corpus DIR` writes again byte for byte.

## The core (`accept_gaps_core.py`, P77.scenarios)

- Each has a boundary recording (`tests/parity/sampling/`, the same relative path) naming its scenario and holding at least one call.
  A SQLite one's has a `sample_target` call that ran a statement; a self-link one's has a `sample_target` call whose outcome is the `ValueError` refusing the table linked to itself.
- The core, driven from those recordings alone in Phase 7.5's independent process (`_driver_run.py`), asks for exactly the recorded statements and gives exactly the recorded outputs or errors, call by call, every call compared.
- Replaying each with today's `KumoRelational._get_task_table` running as it does, the core's `task_table` run, driven as Phase 7.6's checks drive it (`tests/rust_core/link/_link.py`, `compare_runs`), gives what today's call gave or raises what it raised, class name and text.
  On the SQLite ones it finishes, and so does the core's `context` run, equal to today's `_get_context`.
  On the other two, today's call raises the `ValueError` refusal and so does the core's run, naming no index.
- With every `sample_target` and `sample_subgraph` call answered by the core, as Phase 7.5's `accept_driver_payload.py` does, each replays as recorded.
- Each has a request recording (`tests/parity/requests/`, the same relative path, `calls.json`), and replays through Phase 8.5's host (`replay_core`) with no problem of any aspect: statements, payloads, requests, result, frame, warnings, and the error's class, code, message and details, each given by the core.

## Gaps (`accept_gaps_list.py`, P77.gaps)

- `scripts/parity/sampling_coverage_gaps.json` lists none of the three branches under its module.
- `tests/parity/coverage.json` has each branch's module, and none of the three among its module's `missing_branches`.

Run with it: Phase 7.6's `accept_link_derived.py`, which holds `coverage.json` to what `coverage.py` writes; Phase 7.6's `accept_link_gaps.py`; and Phase 7.5's `accept_driver_coverage.py`, which holds every listed entry to a ported branch left untaken.

## What the implementer may change

`scripts/parity/_parity/scenarios.py` and `data.py` are not locked: the scenarios and their data go there.
`scripts/parity/_parity/runner.py` is not locked either: SQLite tables are written by `adbc_ingest`, which declares its own types, so declaring a key `DECIMAL(10,0)`, `NUMERIC` or without a type needs the runner to honor `data.column_types` on SQLite too (documented in `tests/parity/FORMAT.md`).
Nothing else about how a transcript is made or replayed may change: every existing transcript must still replay and record again byte for byte (`tests/parity/test_corpus.py`, `tests/parity/test_fixtures.py`).

Phase 7.5's `accept_driver_capture.py` (P77.derived) holds every scenario but the tabular ones to making both a `sample_target` and a `sample_subgraph` call.
A refused scenario therefore gathers a question over the same schema before the one refused, as `duckdb/regression#decimal-entity-keys` and `memory/link-prediction#multilabel` do.

## Running the checks

Phase 8.5's `accept_request_replay.py` passes in a pytest process of its own, as Phase 8.5's gate ran it, but at this phase's base it fails when run in one process after the other corpus-wide checks (the host's statements are then not credited to a core run).
So the gate runs Phase 8.5's corpus-wide request checks in a pytest process of their own; in `accept_gaps_core.py` the host replay is the last test.
