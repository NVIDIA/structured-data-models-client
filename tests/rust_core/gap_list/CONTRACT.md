# Phase 7.8 acceptance contract

The checks in this directory keep `scripts/parity/sampling_coverage_gaps.json` (the **gap list**) true by machine: the branches the Phase 7.7 verifier found reachable are held by a recorded case or a cited test and the core matches today's Python on them, every "reachable only" entry is backed by tests that take its branch, no other entry's branch is taken by anything, and the false clauses the verifier found are gone.
This file is the interface they test through.
Everything it doesn't name is the implementer's choice.

The verifier's findings are `investigations/P77-coverage-gaps-verifier.md` in the harness (`$SDM_RC_HARNESS`, by default `~/.local/share/sdm-dbx/rust-core`).
Phase 7.5's contract (`tests/rust_core/sampling/driver/CONTRACT.md`), Phase 7.7's (`tests/rust_core/gaps/CONTRACT.md`) and Phase 8.5's (`tests/rust_core/request/CONTRACT.md`) still hold.
Today's Python, in the phase venv, is the reference throughout.

## Names

Branches are numbered as `scripts/parity/coverage.py` numbers them (coverage.py's arcs, a multi-line statement by its first line); `_gap_list.py` also checks each one's two lines, so a moved line fails loudly.
Modules are under `packages/sdm-relational-engine/src/sdm_relational_engine/rfm/`.

The six **reachable branches** (`_gap_list.REACHABLE`):

| Name | Module | Branch | Taken when |
|---|---|---|---|
| `text-of-lists` | `base/sampler.py` | 1165->1166 | `_normalize_text` gets a text column whose first cell is list-like |
| `list-column-on-sql` | `base/sql_sampler.py` | 390->391 | `_tie_break_terms` skips a column of a list dtype |
| `no-valid-key` | `backend/local/graph_store.py` | 234->241 | a table without a time column has no row with a valid key |
| `entity-times-at-max` | `base/sampler.py` | 479->480 | `_no_entity_at_anchor` finds the entity table's earliest time `pd.Timestamp.max` |
| `sqlite-unseeded` | `backend/sqlite/sampler.py` | 283->284 | a SQLite entity draw with `random_seed=None` |
| `postgres-unseeded` | `backend/postgres/sampler.py` | 175->177 | a PostgreSQL entity draw with `random_seed=None` |

And `naive-utc`, `base/utils.py` 65->66, unreachable, whose reason held a false clause.

An entry **says "reachable only"** when its reason holds the words `reachable only`, in any case.
A reason **cites** a test by its pytest node id, relative to the repository root and written in backticks, such as `` `tests/rust_core/rfm/test_driver_fixtures.py::test_an_unseeded_context_on_a_fresh_model` ``.
A cited test must be one the host suites below collect (a `test_*.py` file), so `accept_*.py` files can't be cited.

## Measuring tests (`_gap_list.py`, `_gap_list_plugin.py`)

A **measured run** is `python -m coverage run` of `pytest` from the repository root, with branch coverage of the engine's `rfm` package, `-p no:cacheprovider` and the plugin `_gap_list_plugin.py`, which names coverage's context after the running test (`<node id>|setup`, `|call`, `|teardown`) and records each test's outcome.
A test **passes** when its setup, call and teardown all pass; a skipped test does not pass.
A test **takes** a branch when the branch is among coverage's executed branches for the contexts of that test, setup and teardown included.
A test **calls the core** when, in the pytest process and while it runs, Python code calls a function of `sdm_core._native`, a method of one of its objects, or one of its classes (`sys.monitoring` `CALL` events).
Work done in a subprocess is not measured, for coverage or for calls, nor is work done while a test runs a `coverage.Coverage` of its own, which takes over the tracer.

The **host suites** are those P78.suite-rest runs: `packages/sdm-api/tests`, `packages/sdm-pql/tests`, `packages/sdm-client/tests`, `packages/sdm-connectors/tests` and `tests`, with `-m "not live_nim and not live_warehouse"`, in the phase env (Postgres included).
These are the suites the verifier measured; the engine's own suite, which calls the engine's internals directly, is not among them.
The measure is shown to work on `control_gap_list.py`: one test there takes `base/utils.py` 65->66 and calls the core, the other neither.

## Routes (`_routes.py`)

The verifier's routes: offline public calls that take each reachable branch, over the parity corpus's generated data (`shop`, `shop-referrals`).

| Branch | Route |
|---|---|
| `text-of-lists` | `Schema.from_data`, then `users.country` set in place to one text and then tuples, typed `text`; a COUNT `materialize` |
| `list-column-on-sql` | a DuckDB `VARCHAR[]` or Postgres `text[]` column that `Schema.from_source` leaves out, put back with `columns['tags'] = Column('tags', 'string', 'multicategorical')`; a COUNT `materialize` |
| `no-valid-key` | `Schema.from_data`, then a key set to NaN in place (`products.product_id`, or `users.user_id`); a COUNT `materialize` |
| `entity-times-at-max` | `users.joined_at` all `pd.Timestamp.max` in memory, or the text `2262-04-11 23:47:16.854775807` on SQLite, set as the time column; a COUNT `materialize` |
| `sqlite-unseeded`, `postgres-unseeded` | a COUNT `materialize` with `random_seed=None` over a `Schema.from_source` |

## Taken (`accept_taken.py`, P78.taken)

For each reachable branch:

- Its route, through today's Python, takes it.
- It is held, one of two ways:
  - it is gone from the gap list, and `tests/parity/coverage.json` shows it taken (by a transcript, a fixture case or a PQL case);
  - or its entry says "reachable only", doesn't hold the word "unreachable" or the false clause the verifier quoted (`_gap_list.FALSE_CLAUSES`), and cites at least one test; in one measured run of the cited tests, every one passes, and at least one both takes the branch and calls the core.

The core matches today's Python on the routes:

- `list-column-on-sql`, DuckDB and Postgres: Phase 7.6's `task_table` and `context` comparisons (`tests/rust_core/rfm/test_rfm_runs.py`: `Boundary`, `_compare`), driven from today's calls on the route, find no problem, each over at least one call.
- `entity-times-at-max`, in memory and on SQLite: the `task_table` comparison finds no problem; today's `_get_task_table` refuses.
- `no-valid-key`, both keys: the route made through today's Python and through the request host (`scripts/parity/request_host.py`: `schema_from_data`, then the change, `predictive_query`, `materialize`) gives the same payloads or the same refusal (class, code, message, details), and the same warnings in order.
- `sqlite-unseeded`, `postgres-unseeded`: the core's `task_table` run, started from today's described call and answered with today's results, asks exactly today's statements, SQL byte for byte and parameters encoded alike, up to the end of today's first `_sample_entity_table` call, the unseeded entity read among them.
  What follows depends on draws from the system's entropy and is not compared.
- `text-of-lists` has no comparison here: no locked host or boundary can hand the core a text column of tuples, so the cited test that calls the core is what holds it (above), and the reviewer reads what it compares.

Whatever takes a branch the list no longer lists is held to the core by the corpus-wide checks (P78.p77-regress, P78.p77-scenarios) and, for driver fixture cases, by `tests/rust_core/rfm/test_driver_fixtures.py` (P78.suite-rest).

## Reachable only (`accept_reachable_only.py`, P78.reachable-only)

- No listed entry's branch is taken by the corpus (`tests/parity/coverage.json`).
- Every entry that says "reachable only" cites at least one test.
- In one measured run of the host suites (at least 4000 tests, with some branch taken in every module the list names):
  - every test an entry that says "reachable only" cites passes, and at least one of them takes its branch;
  - no other entry's branch is taken, in any context, test collection included.

So an entry a host test takes, such as one taken by calling the engine directly or over a recorded warehouse call, says "reachable only" and cites that test.

## Reasons (`accept_reasons.py`, P78.reasons)

- The clause the verifier's section C quotes for `base/utils.py` 65->66 is `_gap_list.FALSE_CLAUSES['naive-utc']`.
- That entry is still listed; its reason holds neither the quoted clause nor its text before ` (`, and names the `TaskTable` time column setter (`TaskTable` or `inference/task.py`), whose Series reach `to_naive_utc` too.
- No reason in the list holds any clause of `_gap_list.FALSE_CLAUSES`.

## What the implementer may change

Nothing in this directory, nor `tests/rust_core/rfm/test_rfm_runs.py`, whose `Boundary` and `_compare` the checks drive the core with.
The scenarios, fixture cases, host tests, the gap list, `scripts/parity/capture_sampling.py`'s hooks and `scripts/parity/request_host.py` are the implementer's, within Phase 7.7's and 8.5's contracts.
