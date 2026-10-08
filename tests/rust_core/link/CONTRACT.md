# Phase 7.6 acceptance contract

The checks in this directory hold link prediction to the parity corpus and to Phase 7.5's sampling driver.
This file is the interface they test through.
Everything it doesn't name is the implementer's choice.

Phase 7.5's contract (`tests/rust_core/sampling/driver/CONTRACT.md`) still holds: the recording format, the canonical form, the hooks in `scripts/parity/capture_sampling.py` and the binding's runs.
Today's Python, in the phase venv, is the reference throughout.

## Names

- A **link-prediction transcript** is a transcript under `tests/parity/transcripts/` whose `PredictiveQuery` call asks a query with a `LIST_DISTINCT(...)` target and no `RANK`, and that is not a multilabel transcript.
  `sdm_client` decomposes only the ranked form (`sdm_client/decompose.py`); the plain form reaches the engine whole, which makes it a `TEMPORAL_LINK_PREDICTION` task.
- A **multilabel transcript** asks such a query and ends in an error whose message holds the start of the engine's multilabel text, `Multilabel-classification queries based on 'LIST_DISTINCT' are not supported yet.`
  The **multilabel text** for a column is what `KumoRelational._get_task_type` raises (`rfm.py` 2029->2030) when the listed column has other than one edge:
  `Multilabel-classification queries based on 'LIST_DISTINCT' are not supported yet. If you planned to write a link prediction query instead, make sure to register '<column>' as a foreign key.`
- A **DECIMAL-key transcript** is a completed link-prediction transcript whose `data.column_types` (`tests/parity/FORMAT.md`; the runner casts a generated column to it on DuckDB and declares it on PostgreSQL) gives a `DECIMAL...` or `NUMERIC...` type both to the column the query lists (`orders.product_id`, say) and to the primary key of the payloads' target table, as its boundary recording's `sample_subgraph` calls name it (`products.product_id`).
- A **completed** one ends without an error; a **refused** one ends in the recorded error of its last call.
- **The engine's text** is the message `KumoRelational._get_context` raises when the sampled subgraph holds no target entity (`rfm.py`, "no historical target entities"):
  `Cannot perform link prediction on subgraphs without any historical target entities. Please increase the number of hops and try again.`

## Scenarios (`accept_link_scenarios.py`, P76.scenarios)

- Every link-prediction transcript is a scenario of `scripts/parity/_parity/scenarios.py` (`by_name()`), not a live one, and replays (`_parity.corpus.check`).
- `run_corpus.py record --only <them> --corpus DIR` writes, for each, exactly the committed files: the recorder made them, and Phase 8's stand-in (`_parity/nim.py`) gave the stored responses.
- Among them, on each of `memory`, `duckdb`, `sqlite` and `postgres`, at least one is completed:
  - it sent at least one payload, and every payload is a link-prediction request: `task.kind` is `temporal_link_prediction`; `task.entity_table_names` is two tables, the first the query's entity table; `output.fields` holds `rankings`; the context's instance table has the target column, every value of it a list, some not empty; the predicted instance table has no target column;
  - it stores one response per payload, each with status 200, each prediction item carrying a non-empty `rankings`;
  - its result table has a `rankings` column, one row per predicted row, each ranking not empty.
- At least one is refused: it sent no payload, and its recorded error's message ends with the engine's text.

## Boundary (`accept_link_boundary.py`, P76.boundary)

- Every link-prediction transcript has a boundary recording (`tests/parity/sampling/`, the same relative path) holding at least one `sample_target` and one `sample_subgraph` call.
  In a completed one's, some `sample_target` call's `train.target` is a list type (`list<item>` by `_driver.type_name`) with values, and some `sample_subgraph` call samples both of the payload's `entity_table_names`.
- The core, driven from those recordings alone in Phase 7.5's independent process (`_driver_run.py`), asks for exactly the recorded statements and gives exactly the recorded outputs, call by call.
- Replaying each link-prediction transcript with today's `KumoRelational._get_task_table` and `_get_context` running as they do, each call described as `capture_sampling.describe` describes it, the core's `task_table` and `context` runs, driven from that description and answered from the statements today's call ran, ask for exactly those statements and give what today's call gave (frames by Arrow value and type, task type, entity tables, the warnings), or raise what it raised: the same class name and text, after the same warnings.
  Every completed one's task is `temporal_link_prediction`; every refused one's `_get_context` raised a `ValueError` with the engine's text, and so did the core's run.

## Gaps (`accept_link_gaps.py`, P76.gaps)

`scripts/parity/sampling_coverage_gaps.json`:

- No reason says what the retired link-prediction reasons said ("no task is link prediction", "no question reaching the engine has a LIST_DISTINCT target").
- No reason there, and no table row of Phase 8's `tests/parity/COVERAGE_GAPS.md` or `tests/parity/UNREACHABLE_DIAGNOSTICS.md`, both mentions link prediction (`link prediction`, `link-prediction` or `link_prediction`) and says the client never gathers it ("never gather...", "never be gathered").
  The request rows the scenarios now reach go (Phase 8's `accept_coverage.py` and `accept_messages.py` hold that), and those left get a reason that holds.
- No `rfm.py` entry is one of the link-prediction branches the scenarios take: `_get_task_type`'s `LIST_DISTINCT` test and its single-edge exit, `_get_task_table`'s link-prediction entity tables (the `is_link_pred` test, the loop over edge types and the edge match), and `_get_context`'s "no historical target entities" refusal.
- The multilabel `NotImplementedError` branch of `_get_task_type` (the `len(target_edge_types) != 1` test into its `raise`, 2029->2030) is not listed: the multilabel scenario takes it.
- No reason there, and no line of `COVERAGE_GAPS.md` or `UNREACHABLE_DIAGNOSTICS.md`, cites a file of the migration's harness, which is not in the repository (task 7): a phase's report, review, brief, status or audit (`Phase 7.5's report`, `the Phase 7.5 report`, `REPORT.md`), an `open question`, or the harness folder (`sdm-dbx`, `rust-core/phases`).
  A reason states the fact itself.

Phase 7.5's `accept_driver_coverage.py` runs with it: every listed entry is a ported branch left untaken, and every untaken one is listed or the module reaches 90%.

## Derived files (`accept_link_derived.py`, P76.derived)

- `scripts/parity/coverage.py --out PATH` writes `tests/parity/coverage.json` as committed, crate by crate and module by module (branch counts, covered counts and missing branches), and `rfm.py`'s missing branches there include none of the link-prediction branches above, nor the multilabel branch.
- Phase 6's `accept_capture.py` and `accept_calls.py` run with it: the dialect captures are what `capture_dialects.py` writes, every execute step of every SQL transcript is captured, and the core reproduces each captured call.

## Multilabel (`accept_link_multilabel.py`, P76.multilabel)

- There is at least one multilabel transcript, and each is an offline scenario of `scenarios.py`, stored under its name, that replays and that `run_corpus.py record --only <them>` writes again byte for byte.
- Each records today's behavior, error and all: it sent no payload, and its error is `sdm_client.errors.StructuredDataError` with code `INTERNAL_ERROR` and a message ending in `NotImplementedError: ` and the multilabel text for the listed column.
- Replaying each, today's `_get_task_table` raises the multilabel `NotImplementedError` with that text, and the sampler's `edge_types` link the listed column to two tables or more: the schema's foreign key links to two tables.
- The core's `task_table` run, driven from the same call as in "Boundary", raises the same class name and text.

A schema with one column linked to two tables needs explicit relationships: `Schema.from_data` and `Schema.from_source` take `relationships=` as `sdm_client.Relationship` objects, and name inference never links one column twice.
So `scripts/parity/_parity/runner.py` is not locked in this phase: the implementer may extend how the interpreter decodes a call's arguments so a transcript can give relationships (documented in `tests/parity/FORMAT.md`).
Nothing else about how a transcript is made or replayed may change; every existing transcript must still replay and record again byte for byte (`tests/parity/test_corpus.py`, `tests/parity/test_fixtures.py`).

## Decimal keys (`accept_link_decimal.py`, P76.decimal)

- On each of `duckdb` and `postgres` there is at least one DECIMAL-key transcript, and each replays.
- Replaying them, today's encoder writes some label item from a decimal's own digits on one at least: `payload.py`'s `_target_json_value` gets text from `_decimal_text` for an item (the branch 1018->1019).
  DuckDB reads a `DECIMAL` key as a decimal; PostgreSQL's sampler reads a `NUMERIC` one as a number, so its scenario holds the core to that instead.
- The core, driven from their boundary recordings alone in the independent process, asks for exactly the recorded statements and gives exactly the recorded outputs, call by call (as in "Boundary").
- The core's `task_table` and `context` runs, driven as in "Boundary", finish and give what today's `_get_task_table` and `_get_context` give.
- With every `sample_target` and `sample_subgraph` call answered by the core, as Phase 7.5's `accept_driver_payload.py` does, each replays: the payloads are today's.
