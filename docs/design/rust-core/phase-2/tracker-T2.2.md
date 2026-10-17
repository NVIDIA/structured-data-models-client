# Phase 2 tracker: T2.2

The design branch owns `tracker.md`; this file holds T2.2's section until it folds in.
Lane b, on the local branch `t2.2-impl`: on T2.1d (`t2.1d-combined`, which carries T2.1a, T2.1b and T2.1d), then the test author's acceptance tests (`T2.2 tests:` commits), then this package.
The two judging fixes this package first made landed with T2.1a (its fixes 7 and 8); the contract additions landed with T2.1b (its fix 8).

## T2.2

T2.2 is SQLite: the dialect in `sdm-data`, the Python host's ADBC provider, the rust-host `rusqlite` provider, carried rows U12 and G6, and the exits the row names on SQLite.
By the orchestrator's scope change the first wave's SQLite legs (T2.17, T2.18, T2.8c, T2.19) are those packages', and the kit waits for T2.5.

### What changed

- **Two judging fixes, landed with T2.1a** (its fixes 7 and 8): the runner and comparator take the case's database route, and the comparator reads a row read's types in its database's driver's vocabulary.
- **The SQLite dialect** (`sdm-data/src/sqlite.rs`): each request rendered as the reference's SQLite backend runs it, so each answer is that statement's result (H9).
  The table list from `sqlite_master`; `PRAGMA table_info`, `index_list`, `index_info` and `foreign_key_list`, one table or index at a time; the preview in `rowid` order; the key counts' and time bounds' `UNION ALL`, the bounds over the stored text; the seeded draw of RS-2; the neighbor reads by `rowid`, windowed by the stored text; the key lookup by a plain join for a key the source keeps unique, by the first row's `rowid` otherwise, counting where the core reads the count.
  Relations of one or two parts, the first naming `main`, `temp` or an attached database, quoted alone; none or three refused by name (rule `relation`).
  A key set is bound as ADBC's ingest binds Arrow, a temporary table replaced, so the source's database is left as it was (the reference leaves its key tables in it).
- **RS-2's stream**: numpy's legacy MT19937 and `randint`'s masked draw ported (`sdm-data/src/mt19937.rs`), held to `tests/rs2.json`, 24 seeds recorded from the reference's numpy from 0 past 2**126, both seedings.
- **What the core learns from the data layer** (dialect-free): `DataLayer::compares_stored_times`, SQLite's, makes the core read one stored value of a table's time column before its first windowed read in a run (a `scan` of one non-null value, the reference's probe statement) and name it on each bound's seed column (`SEED_STORED_TIME`), for the dialect to spell the bound as the reference does (`_time_spelling`).
  The core marks a lookup's seeds when the key is described unique (`SEED_KEY_UNIQUE`) and when it does not read the count (`SEED_COUNT_UNREAD`, the walk's lookups); DuckDB ignores both, and no pinned DuckDB trace changes.
- **Contract v26** (T2.1b fix 8): `IndexAdvice::LeadingColumn` (the reference's SQLite rule: each table's indexes read one at a time, any index whose first column is the read's serves it, a foreign key with its table's time column wanted) and `DescribePart::IndexColumns` (an index's columns by its name, SQLite's `index_info`); T2.2 changes nothing under the schema.
- **The catalog read** (Rust-internal types): `CatalogColumn.unique` (SQLite's rowid alias, which no index lists), `CatalogColumn.dtype` (a declared type's data type, used only for a column holding no value, which SQLite's driver types as integers), `CatalogKey.index` and `IndexColumns::Named` (read by name).
  The core reads a key's or index's columns by name in the reference's order, and `infer::preview` takes a rowid alias as unique.
- **Answers in SQLite's shape**: the key counts' position and the time bounds' position of either integer width; the time bounds' text read by the provider's declared reading (DT-7's list); a fetched time column holding no value read as no time, whatever type the driver gave it.
- **U12**: one `seed_integer` in `sdm-contract`, used by the core's seed check and both dialects; the duplicates in `compose.rs` and `duckdb.rs` are gone.
- **The Python host**: `sdm_client._rust.providers.sqlite.SqliteProvider(connection)` over ADBC, the driver's Arrow handed over unconverted; each bridge's `_provider` picks the provider and dialect by the source's backend; the routing seam's SQLite paths (`_routing.FROM_SOURCE`, `MATERIALIZE`, `PREDICT` by backend, `on_the_core`), forced as DuckDB's are; the bridge changes declared (`bridge_changes.json`, baseline rewritten).
- **rust-host**: `sdm-testkit/src/sqlite.rs`, the dev-only `rusqlite` provider typing every answer as ADBC's SQLite driver does (inference over the first 1,024 rows, its promotions and its failures) and binding key sets as ADBC ingests them; `sdm-contract-run --database sqlite`, each case loaded by the one contract loader through `contract/tools/load_case.py` (the Python `SDM_PYTHON` names), every call through its `until` (T2.1d's `through_until`), and keyed `<case>@sqlite`; the host's `Source` gains `Sqlite` beside DuckDB and the Postgres replay; `PROVIDERS` lists `sqlite` in both hosts.
- **rust-host's query text off DuckDB**: the Python host sends materialize the validated query as the PQL library writes it back (`to_string()`); rust-host off DuckDB sends `sdm-pql`'s port of that printer through the core's adapter (`sdm_core::pql::normalized`, which `sdm-host` passes on as `normalized_query`, so the test kit takes no edge to the parser), which writes all 102 recorded materialize queries as python-core sent them, `s0.boundary.deepest`'s 400 redundant parentheses dropped included.
- **G6**: `packages/structured-data-models-client/tests/test_t22_driver_class_reset.py` pins `_rust/run.py`'s per-step `driver_class` reset with a scripted core; a seeded fault (the reset removed) fails it.
- **L4 goldens**: `sdm-data/tests/sqlite_goldens.rs` pins every statement the dialect rendered for the cases and the key-pair vectors on both hosts, beside its request, each re-planned and compared.

### Probes

- python-core over all 86 cases on SQLite, compared with `compare.py --database sqlite` against python-ref's committed lines: every S1 case passes every gating rule but those listed under "Differences" and "For the test author".
- rust-host over the same cases on SQLite agrees with python-core on every core field and statement result (`cross-host`), but `refuse.unknown-column` (below).
- The 82 key-pair vectors on rust-host: every outcome is the reference's but the two UBIGINT-past-int64 ones, which answer with a right join where the reference fails INTERNAL_ERROR (P2B-12).
- ADBC SQLite's inference and ingest measured on the reference's driver (1.12.0), and the `rusqlite` provider held to them by its unit tests.
- No route change on DuckDB: DuckDB's statements, the pinned traces and the DuckDB sweep are unchanged.

### Differences (register, proposed)

- P2B-10: `empty.fact-table` on SQLite, P1B-11's named refusal (E66, reviewer).
- P2B-11: `time.text-one-bad-value` on SQLite, DT-15's named refusal (E68), UNSUPPORTED_REQUEST where the reference refuses INVALID_REQUEST.
- `time.text-us-date` on SQLite is now answered as the reference answers it: its refusal was no reading of DT-16 but an empty windowed fetch, whose columns SQLite's driver types as integers, read as no time column (`time_type`); the core now reads a column holding no value as holding no time.
- P2B-12: `key.uint64-above-int64-max` and the vectors `wide.ubigint.*`: a right join where the reference fails INTERNAL_ERROR (KO2-11 class (a), reviewer).
- P2B-13: `time.feature-time-of-day` on SQLite answers where the reference reads a bare time as today's and refuses; approved by the owner, KO2-61.
- P2B-14: `time.feature-interval` on SQLite describes its INTERVAL text as text where the reference takes `INT` from the declared name; approved by the owner, KO2-62.
- P2B-15: a SQLite seed past 2**64 - 1 refused (RS-7's refusal), the contract's `max_seed` a `u64`; approved by the owner, KO2-59.
- P2B-16 (temporary, KO2-56): on SQLite four S0 cases now reach the core's materialize (T2.1d) and are refused by E92-class rules, `target_filter` (`s0.boundary.deepest`, `digits-4300`, `start-past-64kb`) and `condition` (`escapes`), where the reference answers; T2.10 retires it.

### Deviations not measured by a case (kept, for review)

- A partial unique index is no key (the reference counts a single-column one as unique); its columns are still read, as the reference reads them.
- The rowid alias is SQLite's rule, a primary key of one column declared `INTEGER`; the reference also takes a composite key's one `INTEGER` column.
- A non-`INTEGER` primary key is held not null, as on DuckDB; SQLite lets it hold a null, which the reference keeps.
- The seeded draw decides between hashing the key and the `rowid` row by row by the value's storage class, where the reference decides by the key's declared type, which a typed request does not carry; the same for a column holding the integers it declares.
- The stored time's spelling is read once per run, where the reference keeps it per sampler across a materializer's runs; a second run on one materializer reads it again, one statement more.
- The key tables are temporary and replaced, never left in the source's database.

### For the test author and the orchestrator

- T2.1d settled the S0 schema pin and the routes that stopped at `from_source`: on SQLite every call a case makes runs on the core on both hosts, which agree on every core field but `s0.boundary.depth-999`'s operations, a diagnostic case (python-core stops after validation on the host's RecursionError, as the reference does; rust-host goes on to materialize).
- With validation and materialize both run, the four P2B-16 cases answer an accepted query and then materialize's error; T2.1d's outcome rule judges the last call reached, so that pair is one error outcome.
- The `decimal.*` SQLite legs are T2.17's.
- rust-host's SQLite pass needs a Python with DuckDB and pyarrow for the contract loader (`SDM_PYTHON`, else `python3`); this machine's `python3` has neither, and T2.1d's `rust-test` job sets one.

### Test results (this machine, load average near 30)

- `cargo test --workspace`: every test passes but the test author's still-pending ones and `t136_forecast`'s `a_forecast_of_many_steps_is_refused_at_once`, a two-second bound missed under the machine's load (3.3 s), which passes alone (2.7 s for the file).
- `t22_sqlite.rs` (with `SDM_PYTHON` naming a Python with DuckDB and pyarrow, which the contract loader needs): 8 pass; the two pending ones still fail as they must (`rust_host_runs_every_s1_case_on_sqlite`, on the registered differences and the case-level items below; `each_key_pair_on_sqlite_is_the_reference_s_on_rust_host`, on P2B-12 until approved).
- `contract/tests` (`-n 4 --dist loadfile`): 4,249 pass and one failed, three refusal-map citations moved by this package's edits, fixed in its own commit; `test_refusal_map.py` then passes whole.
- `test_t22_sqlite.py`: the provider marker passes; the three pending ones fail only on the registered differences, the case-level items below and the old route's statements.
- The client package's tests: 2,663 pass, 107 skipped.
- `prepush.sh` (cheap tier): every check passes.
- The root `tests` before the rebase: 1,605 pass; the 51 that failed were all `test_contract_reference.py`, on python-ref's six AdventureWorks lines off DuckDB recorded before T2.1b changed their case files, which T2.1b re-recorded since; after the rebase only the focused checks below ran (the machine's load rules).
- On T2.1d's outcome-rule commit (59f0a07d): `t22_sqlite.rs` 7 pass, and its two pending ones fail as they must, on the `decimal.*` legs (T2.17's), P2B-10, P2B-11 and P2B-12 (proposed), and nothing else.
- On the same base, `test_t22_sqlite.py`'s golden test passes and leaves `pending`; its two other pending ones fail on the same items.
- P2B-13's SQLite line is named by its `case_outcomes`, so the owner's approval covers it in both files' checks.
- After the rebase onto T2.1d: the SQLite cross-host check over the 24 S0 cases, `refuse.unknown-column`, the four cases T2.1d routes past `from_source`, `time.text-us-date` and `base.regression` agrees on every core field but `depth-999`'s operations (diagnostic).

### Protected and judging paths touched

- Judging: `contract/runners/run_python.py` and `scripts/rust_core/compare.py` (the two first commits only), `scripts/rust_core/bridge_changes.json` and `bridge_freeze.json` (declarations and baseline).
- Protected: `rust/crates/sdm-contract/schema/**` (v26 regenerated), `rust/crates/sdm-testkit/**`, `packages/**` (the client's routing calls, the seam's test, and the G6 test), `docs/design/rust-core/register.yaml` (lane B's entries), `contract/refusal-maps/**` (three line citations), `contract/tests/t22-pending.json` (names taken out of pending), `contract/tests/test_t22_row_types.py`.
