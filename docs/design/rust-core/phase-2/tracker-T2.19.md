# T2.19: two-path sampling

Lane f, on the phase head after T2.3's landing (68f9b8e4).
`main`'s eb6c68eb changed the reference's SQL sampler (`sql_sampler.py`, `_sample_subgraph`) in two ways, and this package matches both in the core's walk (`sdm-core/src/subgraph.rs`).
The acceptance cases are the test author's (`T2.19 tests:` commits on `t219-tests`).

## What changed

- **A table reached by two paths is read only up to the frames it held when the hop began** (`Walk::run`, `Walk::visit`).
  At each hop the walk notes, for every table it has reached, how many frames (one per answer that added rows) the table holds, and a table visited at that hop reads its frames from its last visit up to that number, as the reference's `hop_end_dict` does.
  Before, a table read every frame it held when its turn came, so a table that a neighbor visited earlier in the same hop had just reached again (the second path) read those rows a hop early: their neighbors were fetched one hop too deep, an extra statement and extra rows in the payload.
  The rows are still numbered as before, by their place among the table's rows (`start`), so a keyless table's links and a keyed table's numbers are unchanged wherever no second path exists.
- **One column linking two tables** (a role-playing join: `orders.cid` to `customers` and to `buyers`): the reference now selects such a column once (`dict.fromkeys`), where it selected it once per link and failed.
  The core already read it once (`sample_columns` collects the columns as a set, and a frame is projected by name), and looks each link up on its own edge (`Graph::foreign_keys`); no code change, the walk's module doc now says so.

## Probes

Scratch cases in the contract's case format (never committed: the cases are the test author's), built from the reference's parity test's shapes (`test_sql_sampler_parity.py`, `_SHAPES` and `_shape_frames`) and eb6c68eb's `test_two_paths.py` tables, run through the public calls on python-ref and rust-host and compared by `compare.py`; questions `COUNT(... , 0, 30, days)` over three entities, anchor 2025-01-01, `until: predict`.

- DuckDB, before the change: `diamond` from `addresses` and the two-path shape from `customers` (keyed and keyless lines) differed: one statement more than the reference's at materialize and the payload's `related_tables` rows doubled (120 against 60; 237 against 120).
  `same-table-twice` (from `customers` and from `buyers`) and `star` (from `customers` and from `products`) already matched.
- DuckDB, after: every probe the core answers matches the reference at the payload, the statement count and the outcome.
- After the change, python-core (the test-ops wheel built from this tree) and rust-host on all three databases (SQLite through rust-host's `rusqlite` provider, Postgres 17 in the pinned image with rust-host replaying python-core's recorded fetches): no differences from python-ref on any probe the core answers; on Postgres only the diagnostic statement-parameter texts every Postgres case shows (KO2-58).
- `diamond` and `chain` from `customers` are refused by the core's not-yet rule `parent_table` (the entity table has a foreign key), which T2.11 lifts; the reference answers. Out of this package's scope; the orchestrator is told.

## Deviations and what is left

- None registered.
