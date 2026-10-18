# Phase 2 tracker: T2.2

The design branch owns `tracker.md`; this file holds T2.2's section until it folds in.
Lane b, on the local branch `t2.2-impl`: on the phase head after T2.1d's landing (1a37efe2), then the test author's acceptance tests (`T2.2 tests:` commits), then this package and its fix round 1 (`T2.2 fix <n>:` commits).
The two judging fixes this package first made landed with T2.1a (its fixes 7 and 8); the contract additions landed with T2.1b (its fix 8).

## T2.2

T2.2 is SQLite: the dialect in `sdm-data`, the Python host's ADBC provider, the rust-host `rusqlite` provider, carried rows U12 and G6, and the exits the row names on SQLite.
By the orchestrator's scope change the first wave's SQLite legs (T2.17, T2.18, T2.8c, T2.19) are those packages', and the kit waits for T2.5.

### What changed

- **Two judging fixes, landed with T2.1a** (its fixes 7 and 8): the runner and comparator take the case's database route, and the comparator reads a row read's types in its database's driver's vocabulary.
- **The SQLite dialect** (`sdm-data/src/sqlite.rs`): each request rendered as the reference's SQLite backend runs it, so each answer is that statement's result (H9).
  The table list from `sqlite_master`; `PRAGMA table_info`, `index_list`, `index_info` and `foreign_key_list`, one table or index at a time; the preview in `rowid` order; the key counts' and time bounds' `UNION ALL`, the bounds over the stored text; the seeded draw of RS-2; the neighbor reads by `rowid`, windowed by the stored text; the key lookup by a plain join for a key the source keeps unique, by the first row's `rowid` otherwise, counting where the core reads the count.
  Relations of one or two parts, the first naming `main`, `temp` or an attached database, quoted alone; none or three refused by name (rule `relation`).
  A key set is bound as ADBC's ingest binds Arrow, a temporary table of its own name in the run (`__sdm_seeds_<attachment>__`), so the source's database is left as it was (the reference leaves its key tables in it).
  Each provider creates the table, and replaces it only where it made that name on the connection before, so a relation the caller made is never replaced or dropped (fix 1).
  The seeded draw hashes the key or the `rowid` once per column, by the type the key's declaration fixes, read in the statement from `pragma_table_info`, else by the schema's type the request names (fix 2b).
- **RS-2's stream**: numpy's legacy MT19937 and `randint`'s masked draw ported (`sdm-data/src/mt19937.rs`), held to `tests/rs2.json`, 24 seeds recorded from the reference's numpy from 0 past 2**126, both seedings.
- **What the core learns from the data layer** (dialect-free): `DataLayer::compares_stored_times`, SQLite's, makes the core read one stored value of a table's time column before its first windowed read in a run (a `scan` of one non-null value, the reference's probe statement) and name it on each bound's seed column (`SEED_STORED_TIME`), for the dialect to spell the bound as the reference does (`_time_spelling`).
  The core marks a lookup's seeds when the key is described unique (`SEED_KEY_UNIQUE`) and when it does not read the count (`SEED_COUNT_UNREAD`, the walk's lookups); DuckDB ignores both, and no pinned DuckDB trace changes.
- **Contract v26, fix 2a**: `SampleEntities.key_dtype`, optional and left out when none, so DuckDB's requests are unchanged (the orchestrator's deviation from all vocabulary in T2.1b); `DataLayer::samples_by_key_dtype` says which providers get it (SQLite's).
- **Contract v26** (T2.1b fix 8): `IndexAdvice::LeadingColumn` (the reference's SQLite rule: each table's indexes read one at a time, any index whose first column is the read's serves it, a foreign key with its table's time column wanted) and `DescribePart::IndexColumns` (an index's columns by its name, SQLite's `index_info`); T2.2's one schema change is fix 2a's.
- **The catalog read** (Rust-internal types): `CatalogColumn.unique` (SQLite's rowid alias, which no index lists, read as the reference reads it: the one primary key column declared `INTEGER`, fix 8), `CatalogColumn.dtype` (the data type a SQLite declaration fixes, the column's type whatever its values, as the reference types it, but for an `INTERVAL`, KO2-62; fix 7), `CatalogColumn.key_nullable` (a SQLite primary key keeps its catalog nullability, fix 3), `KeyKind::PartialUnique` (a partial unique index keeps its one column unique, fix 9), `CatalogKey.index` and `IndexColumns::Named` (read by name).
  The core reads a key's or index's columns by name in the reference's order, and `infer::preview` takes a rowid alias as unique.
- **Answers in SQLite's shape**: the key counts' position and the time bounds' position of either integer width; the time bounds' text read by the provider's declared reading (DT-7's list); a fetched time column holding no value read as no time, whatever type the driver gave it, after the zoned arms, so a zoned column keeps its unit (fix 6).
- **U12**: one `seed_integer` in `sdm-contract`, used by the core's seed check and both dialects; the duplicates in `compose.rs` and `duckdb.rs` are gone.
- **The Python host**: `sdm_client._rust.providers.sqlite.SqliteProvider(connection)` over ADBC, the driver's Arrow handed over unconverted; each bridge's `_provider` picks the provider and dialect by the source's backend; the routing seam's SQLite paths (`_routing.FROM_SOURCE`, `MATERIALIZE`, `PREDICT` by backend, `on_the_core`), forced as DuckDB's are; the bridge changes declared (`bridge_changes.json`, baseline rewritten).
- **rust-host**: `sdm-testkit/src/sqlite.rs`, the dev-only `rusqlite` provider typing every answer as ADBC's SQLite driver does (inference over the first 1,024 rows, its promotions and its failures) and binding key sets as ADBC ingests them; `sdm-contract-run --database sqlite`, each case loaded by the one contract loader through `contract/tools/load_case.py` (the Python `SDM_PYTHON` names), every call through its `until` (T2.1d's `through_until`), and keyed `<case>@sqlite`; the host's `Source` gains `Sqlite` beside DuckDB and the Postgres replay; `PROVIDERS` lists `sqlite` in both hosts.
- **rust-host's query text off DuckDB** (`pql_stack.rs` counts `normalized`, the adapter's third entry, fix 5): the Python host sends materialize the validated query as the PQL library writes it back (`to_string()`); rust-host off DuckDB sends `sdm-pql`'s port of that printer through the core's adapter (`sdm_core::pql::normalized`, which `sdm-host` passes on as `normalized_query`, so the test kit takes no edge to the parser), which writes all 102 recorded materialize queries as python-core sent them, `s0.boundary.deepest`'s 400 redundant parentheses dropped included.
- **G6**: `packages/structured-data-models-client/tests/test_t22_driver_class_reset.py` pins `_rust/run.py`'s per-step `driver_class` reset with a scripted core; a seeded fault (the reset removed) fails it.
- **L4 goldens**: `sdm-data/tests/sqlite_goldens.rs` pins every statement the dialect rendered for the cases and the key-pair vectors on both hosts, beside its request, each re-planned and compared.

### Probes

- python-core over all 86 cases on SQLite, compared with `compare.py --database sqlite` against python-ref's committed lines: every S1 case passes every gating rule but those listed under "Differences" and "For the test author".
- rust-host over the same cases on SQLite agrees with python-core on every core field and statement result (`cross-host`), but `refuse.unknown-column` (below).
- The 82 key-pair vectors on rust-host: every outcome is the reference's but the two UBIGINT-past-int64 ones, which answer with a right join where the reference fails INTERNAL_ERROR (P2B-12).
- ADBC SQLite's inference and ingest measured on the reference's driver (1.12.0), and the `rusqlite` provider held to them by its unit tests.
- No route change on DuckDB: DuckDB's statements, the pinned traces and the DuckDB sweep are unchanged.

### Differences (register)

- P2B-10: `empty.fact-table` on SQLite, P1B-11's named refusal (E66); approved by both round 1 reviewers.
- P2B-11: `time.text-one-bad-value` on SQLite, DT-15's named refusal (E68), UNSUPPORTED_REQUEST where the reference refuses INVALID_REQUEST; approved by the owner under E68 (DT-15 states this refusal).
- `time.text-us-date` on SQLite is now answered as the reference answers it: its refusal was no reading of DT-16 but an empty windowed fetch, whose columns SQLite's driver types as integers, read as no time column (`time_type`); the core now reads a column holding no value as holding no time.
- P2B-12: `key.uint64-above-int64-max` and the vectors `wide.ubigint.*`: a right join where the reference fails INTERNAL_ERROR (KO2-11 class (a)); approved by both round 1 reviewers.
- P2B-13: `time.feature-time-of-day` on SQLite answers where the reference reads a bare time as today's and refuses; approved by the owner, KO2-61.
- P2B-14: `time.feature-interval` on SQLite describes its INTERVAL text as text where the reference takes `INT` from the declared name; approved by the owner, KO2-62, its wording narrowed to an `INTERVAL` column.
- P2B-15: a SQLite seed past 2**64 - 1 refused (RS-7's refusal), the contract's `max_seed` a `u64`; approved by the owner, KO2-59.
- P2B-16 (temporary, KO2-56): on SQLite four S0 cases now reach the core's materialize (T2.1d) and are refused by E92-class rules, `target_filter` (`s0.boundary.deepest`, `digits-4300`, `start-past-64kb`) and `condition` (`escapes`), where the reference answers; T2.10 retires it.
- P2B-17 (proposed): the stored time's spelling is read once per call, where the reference keeps it for a materializer's life; one statement more on a second materialize of one query, the same answers; matching it needs host-held state across calls.
- P2B-10 to P2B-13 name their `<case>@sqlite` lines in `case_outcomes`, which the acceptance tests read.

### Round 1's measured differences, now the reference's

- A primary key that is not the rowid alias keeps its catalog nullability, so a null key is warned of, counted and filtered as on the reference (fix 3).
- A column declared `INTEGER` holding fractions is described int64 (fix 7); a composite key's one `INTEGER` column is the rowid alias (fix 8); a partial unique index keeps its column unique, and its two extra statements are gone (fix 9).
- The seeded draw chooses its hash once per column by the key's type (fix 2b).
- Still differing, not registered, for the orchestrator: the key counts' label (`k`) is each counted key's position in the request, where the reference numbers every keyed table, an enforced one skipped too; on a source whose enforced key comes first (a SQLite rowid alias, a DuckDB primary key) the later labels differ by the skipped ones, the counts equal. Matching it needs each key's label in the request (`TableKey`), a contract field; it predates T2.2 (DuckDB's rendering is the same).

### For the test author and the orchestrator

- T2.1d settled the S0 schema pin and the routes that stopped at `from_source`: on SQLite every call a case makes runs on the core on both hosts, which agree on every core field but `s0.boundary.depth-999`'s operations, a diagnostic case (python-core stops after validation on the host's RecursionError, as the reference does; rust-host goes on to materialize).
- With validation and materialize both run, the four P2B-16 cases answer an accepted query and then materialize's error; T2.1d's outcome rule judges the last call reached, so that pair is one error outcome.
- The `decimal.*` SQLite legs are T2.17's; the two S1 tests stay pending on them alone.
- `contract-python (ref-pandas3, sqlite)` compares with no effective expectations, so it fails on every registered SQLite difference until its expectations are wired (the round 1 review's P3-1).
- rust-host's SQLite pass needs a Python with DuckDB and pyarrow for the contract loader (`SDM_PYTHON`, else `python3`); this machine's `python3` has neither, and T2.1d's `rust-test` job sets one.

### Test results (this machine, load average near 30)

On the head after fix round 1, rebased onto 1a37efe2, under `nice`:

- `cargo test --workspace --no-fail-fast` (with `SDM_PYTHON` naming a Python with DuckDB and pyarrow): 129 test binaries, 805 tests, none failed; `t22_sqlite.rs` among them, its one pending test (`rust_host_runs_every_s1_case_on_sqlite`) failing as it must, on the nine `decimal.*` legs alone.
- `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --check`: clean.
- python-core over all 86 cases on SQLite against python-ref's committed lines (`compare.py --database sqlite`): 63 differences, every one a registered entry's (P2B-10 to P2B-16) or a `decimal.*` leg's (T2.17); with rust-host's run, the two hosts agree on every core field but `s0.boundary.depth-999`'s operations (diagnostic) and the `decimal.*` legs python-core does not route.
- The reviewers' probes (a null `TEXT PRIMARY KEY`, an `INT PRIMARY KEY`, `INTEGER` holding fractions, a composite key, a partial unique index) against the reference: the same schema, outcome and statement count each, the texts aside.
- `test_t22_sqlite.py`, `test_t22_row_types.py`, `test_t22_r1_regressions.py`, `test_refusal_map.py`, `test_wheel_smoke.py`, and the client's materialize-route, routing-seam and G6 tests: 316 pass; the one pending Python test fails as it must, on the `decimal.*` legs alone.
- `register.py validate`, `ratchet.py check --checkpoint 1a37efe2`, ruff on the changed files and `prepush.sh`'s cheap tier: every check passes.

### Protected and judging paths touched

- Judging: `contract/runners/run_python.py` and `scripts/rust_core/compare.py` (the two first commits only), `scripts/rust_core/bridge_changes.json` and `bridge_freeze.json` (declarations and baseline).
- Protected: `rust/crates/sdm-contract/schema/**` (v26 regenerated), `rust/crates/sdm-testkit/**`, `packages/**` (the client's routing calls, the seam's test, and the G6 test), `docs/design/rust-core/register.yaml` (lane B's entries), `contract/refusal-maps/**` (three line citations), `contract/tests/t22-pending.json` (names taken out of pending), `contract/tests/test_t22_row_types.py`.
