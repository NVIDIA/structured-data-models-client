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
- `diamond` and `chain` from `customers` are refused by the core's not-yet rule `parent_table` (the entity table has a foreign key), which T2.11 lifts; the reference answers.
  The test author's cases sample each shape from its tables with no foreign key for that reason, and leave those questions to T2.11.

## Acceptance

The test author's three commits (`t219-tests`, 66e44670, 517db2a6, 30efc322) taken onto this package's commit in a scratch combined branch, `t219-combined`, never onto the lane:

- `contract/tests/test_t219_two_paths.py` (python-core, all three databases, Postgres 17 in the pinned image, `SDM_REQUIRE_POSTGRES=1`): 78 passed.
- `sdm-testkit`'s `tests/t219_two_paths.rs` (rust-host, DuckDB and SQLite): 2 passed.
- Every key `t219-pending.json` listed (the diamond from `addresses`, the `from_source` diamond and the keyed two-path case: 9 python-core keys, 6 rust-host keys) passed with the change, so `pending` is empty; `authored` is unchanged.

What the new acceptance cases need beside them, each as an earlier package made it for its cases (T2.0c, T2.2), on the combined branch:

- **Baselines**: `ratchet.py accept --checkpoint 68f9b8e4` over python-core's, python-old's and rust-host's DuckDB observations of the eleven cases; every other line kept.
- **SQLite goldens**: `sdm-data`'s `tests/sqlite_goldens.rs` pins the 96 requests the cases fetch on SQLite that no golden pinned, collected from both hosts' recorded fetches, which agree.
- **The keys map** (the cases' seam): each case a probe of `contract/refusal-maps/keys.json`, the case as committed, written by `refusal_map.py write` from python-core's and rust-host's runs; every probe answers on both hosts, so no entry; the keys family of `contract/pinned/families.probes.json` regenerated, eleven digests added, none changed.

Checks on the combined branch: `pytest contract/tests -n 4 --dist loadfile` all passed (4,590; the 28 that failed before the three items above were the ratchet's "an acceptance case with no proposed baseline", the SQLite golden check and the keys map's case list, and pass after them); `test_refusal_map.py` 57, `sdm-testkit`'s `refusal_map` 9 and `t22_sqlite` 9, `sdm-data`'s `sqlite_goldens` 1.
On the lane: `prepush.sh` (cheap tier), `cargo test --workspace --no-fail-fast` (818 passed), `test_refusal_map.py`, `test_wheel_smoke.py`, `test_release_contract.py`, `register.py validate` and `ratchet.py check --checkpoint 68f9b8e4`.

## Deviations and what is left

- None registered: on every case and probe the core answers, both hosts match the reference on all three databases.
- The judging-path files: none touched (`compare.py`, `run_python.py`, `ratchet.py` and the rest unchanged); the baselines, goldens, keys map and pin are data the cases need, as at T2.0c and T2.2.
