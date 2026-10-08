# Determinism fixes for the reference (T0.3)

This records which of the first attempt's 27 commits on `rust-core/p1-determinism` (GitLab layout, `packages/sdm-*`) were ported to `rust-core-reference`, and why.
The reference is `main` at `fa0637d` (tagged locally as `reference-baseline-fa0637d`) plus the kept fixes, on the GitHub layout (`packages/structured-data-models-*`).

## Categories

- (a) Needed for repeatable answers (seeded draws, stable ordering, independence from `PYTHONHASHSEED` and from whether torch is installed), checkable offline, and changing no answer under a fixed seed beyond removing nondeterminism. Kept.
- (b) Changes answers users see under a fixed seed. Owner decision.
- (c) Changes warehouse behavior (Snowflake, Postgres, Databricks sampling or inference). Kept only with a live comparison; none was done, so dropped.
- (d) A refusal or another behavior change that repeatability does not need. Dropped.
- (p) A process commit of the first attempt. Dropped.

## Every commit

| Commit | Subject | Category | Reason | Outcome |
|---|---|---|---|---|
| `3003ed1f` | Lock phase 1 acceptance checks | p | The first attempt's acceptance harness, not a fix; its payload runner was used, uncommitted, for the repeatability check below. | Dropped |
| `bdc4f2d4` | Sort each neighborhood stably, whether or not torch is installed | a | The in-memory graph store sorted with torch when importable and numpy quicksort otherwise, so installing torch changed sampled neighbors at timestamp ties; confirmed on `main` with the fake torch. | Kept as `d1590a1` |
| `508845fe` | Draw every client-side random choice from a RandomState stream | b | Moves the example shuffle from `default_rng` to `RandomState`, which reorders the recorded relational payload under a fixed seed. Only its `random_state` helper was ported, inside `f0f324c`, without the shuffle change. | Owner decision |
| `935cf99f` | Refuse a repeated batch index instead of sorting it into some order | d | Adds a refusal; with unique positions the old sort already gives one order. | Dropped |
| `f2f5b86c` | Re-pin the engine tests the RandomState shuffle moved | b | Test pins that follow `508845fe`. | Owner decision, with `508845fe` |
| `300bab01` | Order every SQL result whose order reaches a request | c | Adds `ORDER BY` to every SQL backend and replaces the Databricks oversample `LIMIT` with a seeded `xxhash64`, changing warehouse SQL. Its DuckDB and SQLite parts were not needed: the DuckDB and SQLite push-down payloads were repeatable without them. | Dropped |
| `2e48359f` | Seed the SQLite and Snowflake entity samples | a (SQLite), c (Snowflake) | A seeded SQLite request drew different context rows on every call (`ORDER BY RANDOM()`), confirmed on `main`; the Snowflake half changes warehouse sampling. | SQLite part kept in `f0f324c`; Snowflake part dropped |
| `da7e1dec` | Name where each SQL backend's seed goes in random_state's list | c | Documents seed routes of warehouse changes that were dropped; the ported helper has its own accurate docstring. | Dropped |
| `369bf44c` | Probe a SQLite time column's spelling on its first stored row | d | The probe is already deterministic for a given database file; not needed for repeatable runs. | Dropped |
| `c9695f3c` | Re-record the wire payloads for the RandomState shuffle | b | Golden re-record and changelog for `508845fe`. | Owner decision, with `508845fe` |
| `6d1e013b` | Ask a backend once which of a sampler's keys it enforces | c | Supports the warehouse ordering of `300bab01`. | Dropped |
| `7a01cd89` | Hash a Snowflake key in a form no session setting formats | c | Snowflake seeded draw. | Dropped |
| `100872a9` | Hash a SQLite integer key, not its rowid, for a seeded draw | a | Makes the seeded SQLite draw follow the data rather than physical rowids, which `VACUUM` may renumber. Ported with one adaptation: rows sharing an unenforced key are ordered by key and rowid, since the column tie-break of `300bab01` was not ported. | Kept in `f0f324c` |
| `f15d5bf4` | Infer warehouse column types from the rows a hash picks, in one order | c | Databricks, Postgres and Snowflake type inference. | Dropped |
| `f120f603` | Name the Databricks draw and the inference samples in the changelog | c | Changelog for dropped warehouse changes. | Dropped |
| `5a0d2e8e` | Teach the client's warehouse fakes the ordered inference sample | c | Test fakes for `f15d5bf4`. | Dropped |
| `e0ee2f46` | Count the rows of the Snowflake table named exactly, not the first one LIKE lists | c | Snowflake row count. | Dropped |
| `440b8d60` | Hash a warehouse table's key, or a few columns, to pick its inference sample | c | Databricks and Snowflake inference sample. | Dropped |
| `572085ad` | Pick an unkeyed PostgreSQL table's inference sample by each row's binary form | c | Postgres inference sample. | Dropped |
| `3367cdf7` | Refuse to guess which Snowflake table a row count belongs to | d | A refusal, on Snowflake. | Dropped |
| `16e96df8` | Hash a Snowflake DATE key by its date, which DATE_PART cannot give in nanoseconds | c | Snowflake seeded draw. | Dropped |
| `044cc192` | Say in the changelog which rows a table's types are inferred from, and that reading them scans the table | c | Changelog for dropped warehouse changes. | Dropped |
| `e8625fcb` | Order an unkeyed PostgreSQL table's unsupported columns by their text, as some have no binary form | c | Postgres inference sample. | Dropped |
| `d83afb95` | Say which values psycopg cannot read under other date and interval styles | c | Postgres server test for dropped changes. | Dropped |
| `25202c49` | Join the timeout tests' stalled handlers, which woke in later tests and printed a connection reset | d | Test hygiene; changes no answer and is not needed for repeatable runs. | Dropped |
| `70688f26` | Spread an unkeyed PostgreSQL table's inference sample by a hash of its text when it has no column the backend reads | c | Postgres inference sample. | Dropped |
| `890e8fe3` | Say that an unkeyed PostgreSQL table's binary sample order follows the client encoding | c | Postgres docstring for dropped changes. | Dropped |

Counts: 3 in (a), with `2e48359f` kept only in its SQLite part; 3 in (b); 16 in (c); 4 in (d); 1 process commit.

## What is on `rust-core-reference`

| Commit | Subject | From |
|---|---|---|
| `d1590a1` | Sort each neighborhood stably, whether or not torch is installed | `bdc4f2d4`, path-rewritten patch applied unchanged |
| `f0f324c` | Seed the SQLite entity sample | SQLite part of `2e48359f`, `100872a9`, and the `random_state` helper of `508845fe` |
| `70aa159` | Refuse to run as the reference if the Rust core can be loaded | New: `scripts/reference_guard.py` (see reference-environment.md) |

All three are signed off, with no co-author trailer.
No golden file changed: `scripts/snapshot_wire_payloads.py` matches `tests/golden/wire_payloads.json` on the reference.
Both offline suites pass on the reference (2623 passed and 136 skipped; engine 2181 passed and 106 skipped), as do `ruff check`, `ruff format --check`, `mypy` on the engine, and `scripts/check_repo_hygiene.py all`.

## Repeatability check

Each run was made in a fresh interpreter under `PYTHONHASHSEED=0` and `=1`, once with torch not installed and once with torch 2.14.1 installed, on both unpatched `main` and the reference.
Live and network tests were not run.
The outputs compared were:

- The payloads `scripts/snapshot_wire_payloads.py` collects, written as sorted JSON, and its own check against the golden file.
- The ported determinism checks: `tests/base/test_utils.py`, `tests/backend/sqlite/test_sqlite_graph.py` and `tests/backend/local` in the engine, as the list of test ids and outcomes.
- The first attempt's payload runner (`determinism_payloads.py` from `3003ed1f`, uncommitted): in-memory relational scenarios with timestamp ties, and SQLite and DuckDB push-down scenarios, each built three times per process.

Results on the reference:

- The wire payloads are byte-identical in all four configurations (sha256 `c180149d…`), and match the golden file.
- The ported checks give byte-identical result lists in all four configurations (54 passed each).
- The tie, SQLite and DuckDB payloads, compared parsed, are identical in all four configurations, and the tie payloads are also identical with the fake torch whose unstable sort breaks ties the other way.

Results on unpatched `main`:

- The wire payloads are byte-identical in all four configurations, and equal to the reference's.
- The tie payloads differ with and without torch (real and fake), in two scenarios.
- The SQLite push-down payloads differ between builds in one process, in both scenarios, so a seeded SQLite request is not repeatable.
- The DuckDB push-down payloads are identical in all configurations, and equal to the reference's.

The harness and run scripts are kept outside the repository, in `~/.local/share/sdm-dbx/rust-core/t03-harness/`.

## Changed outcomes: proposed register entries

These go into `register.yaml` in T0.8.

1. Neighbor ties in the in-memory backend (`d1590a1`).
   Old: neighbors sharing a parent and a timestamp were ordered by torch's sort when torch was importable and by numpy's quicksort otherwise.
   New: they are ordered by a stable numpy sort, so by source row, in both cases.
   Affected inputs: in-memory relational requests whose sampled neighbors tie on parent and timestamp, without torch installed.
   Observed: on the tie fixture, two of two scenarios send a different neighbor without torch; with torch 2.14.1 installed the old and new answers are equal.
   The recorded wire payloads have no such ties and are unchanged.
2. Seeded SQLite entity sample (`f0f324c`).
   Old: `ORDER BY RANDOM() LIMIT n`, ignoring the seed, so every call drew different rows.
   New: the rows whose integer primary key, or else rowid, hashes lowest under a cubic polynomial modulo `2**31 - 1`, with coefficients `random_state(seed).randint(0, 2**31 - 1, size=4)`; ties by key, then rowid.
   Affected inputs: SQLite push-down requests with a seed and without explicit entity ids; unseeded requests are unchanged.
   There was no repeatable old answer to preserve, but the new draw defines a stream the core must reproduce (T0.6).
3. SQLite seed warning (`f0f324c`).
   Old: a seeded SQLite sample warned once that `random_seed` is ignored.
   New: SQLite does not warn; Snowflake still does, and the two `random_seed` docstrings in `rfm.py` say so.
4. Seeds of `2**32` and above for the SQLite draw (`f0f324c`).
   New: `random_state` seeds `RandomState` with the seed's 32-bit words, least significant first, where `RandomState(seed)` would raise.

## Owner decisions needed

- Whether to adopt `508845fe` with `f2f5b86c` and `c9695f3c`: the example shuffle moves from `default_rng` to `RandomState`, which reorders the `relational/pql-count-aggregation` golden payload (same examples, labels and subgraphs, different order and renumbered ids).
  Until decided, the reference keeps `default_rng`.
- Whether to accept entry 2 as the reference's seeded SQLite draw, or to drop `f0f324c` and leave SQLite seeded draws unrepeatable in the reference.

## Known nondeterminism left in the reference

These come from dropped (c) commits and stay until a live comparison is done:

- A seeded Snowflake entity sample (`SAMPLE ROW` takes no seed; it still warns).
- Which oversampled rows a Databricks `LIMIT` keeps, and the order of other warehouse SQL results without `ORDER BY`.
- The rows Databricks, Postgres and Snowflake infer column types from.

Two client tests still say SQLite cannot seed the context draw (`test_tables_stay_where_they_are.py`, `test_a_keyless_table_samples_the_same_rows.py`); they pass, and could now compare the context rows.
