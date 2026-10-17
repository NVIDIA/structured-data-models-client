# Phase 2 tracker: T2.2

The design branch owns `tracker.md`; this file holds T2.2's section until it folds in.
Lane b, on the local branch `t2.2-impl`: T2.1b rebased onto T2.1a's head (`t2.1b-on-21a`), then the test author's acceptance tests (`T2.2 tests:` commits), then this package.
The orchestrator moves the commits onto the lane once T2.1a and T2.1b land.

## T2.2

T2.2 is SQLite: the dialect in `sdm-data`, the Python host's ADBC provider, the rust-host `rusqlite` provider, carried rows U12 and G6, and the exits the row names on SQLite.
By the orchestrator's scope change the first wave's SQLite legs (T2.17, T2.18, T2.8c, T2.19) are those packages', and the kit waits for T2.5.

### What changed

- **Two judging fixes, alone, first** (the orchestrator lands them with T2.1a as its fixes): `run_python.py`'s python-core forces a case's DuckDB paths as the database's, and `compare.py` starts a case's core operations on the database's dialect (7200d9b8, the T2.3 test author's patch); `compare.py` reads a row read's types in its database's driver's vocabulary, ADBC SQLite's Arrow names as themselves and nothing else, with one focused test (c172f1d5).
- **The SQLite dialect** (`sdm-data/src/sqlite.rs`): each request rendered as the reference's SQLite backend runs it, so each answer is that statement's result (H9).
  The table list from `sqlite_master`; `PRAGMA table_info`, `index_list`, `index_info` and `foreign_key_list`, one table or index at a time; the preview in `rowid` order; the key counts' and time bounds' `UNION ALL`, the bounds over the stored text; the seeded draw of RS-2; the neighbor reads by `rowid`, windowed by the stored text; the key lookup by a plain join for a key the source keeps unique, by the first row's `rowid` otherwise, counting where the core reads the count.
  Relations of one or two parts, the first naming `main`, `temp` or an attached database, quoted alone; none or three refused by name (rule `relation`).
  A key set is bound as ADBC's ingest binds Arrow, a temporary table replaced, so the source's database is left as it was (the reference leaves its key tables in it).
- **RS-2's stream**: numpy's legacy MT19937 and `randint`'s masked draw ported (`sdm-data/src/mt19937.rs`), held to `tests/rs2.json`, 24 seeds recorded from the reference's numpy from 0 past 2**126, both seedings.
- **What the core learns from the data layer** (dialect-free): `DataLayer::compares_stored_times`, SQLite's, makes the core read one stored value of a table's time column before its first windowed read in a run (a `scan` of one non-null value, the reference's probe statement) and name it on each bound's seed column (`SEED_STORED_TIME`), for the dialect to spell the bound as the reference does (`_time_spelling`).
  The core marks a lookup's seeds when the key is described unique (`SEED_KEY_UNIQUE`) and when it does not read the count (`SEED_COUNT_UNREAD`, the walk's lookups); DuckDB ignores both, and no pinned DuckDB trace changes.
- **Contract v26's working entry** gains `IndexAdvice::LeadingColumn` (the reference's SQLite rule: each table's indexes read one at a time, any index whose first column is the read's serves it, a foreign key with its table's time column wanted) and `DescribePart::IndexColumns` (an index's columns by its name, SQLite's `index_info`); the schema and lock regenerated, v25 untouched.
- **The catalog read** (Rust-internal types): `CatalogColumn.unique` (SQLite's rowid alias, which no index lists), `CatalogColumn.dtype` (a declared type's data type, used only for a column holding no value, which SQLite's driver types as integers), `CatalogKey.index` and `IndexColumns::Named` (read by name).
  The core reads a key's or index's columns by name in the reference's order, and `infer::preview` takes a rowid alias as unique.
- **Answers in SQLite's shape**: the key counts' position and the time bounds' position of either integer width; the time bounds' text read by the provider's declared reading (DT-7's list).
- **U12**: one `seed_integer` in `sdm-contract`, used by the core's seed check and both dialects; the duplicates in `compose.rs` and `duckdb.rs` are gone.
- **The Python host**: `sdm_client._rust.providers.sqlite.SqliteProvider(connection)` over ADBC, the driver's Arrow handed over unconverted; each bridge's `_provider` picks the provider and dialect by the source's backend; the routing seam's SQLite paths (`_routing.FROM_SOURCE`, `MATERIALIZE`, `PREDICT` by backend, `on_the_core`), forced as DuckDB's are; the bridge changes declared (`bridge_changes.json`, baseline rewritten).
- **rust-host**: `sdm-testkit/src/sqlite.rs`, the dev-only `rusqlite` provider typing every answer as ADBC's SQLite driver does (inference over the first 1,024 rows, its promotions and its failures) and binding key sets as ADBC ingests them; `sdm-contract-run --database sqlite`, each case loaded by the one contract loader through `contract/tools/load_case.py` (the Python `SDM_PYTHON` names) and keyed `<case>@sqlite`; `PROVIDERS` lists `sqlite` in both hosts.
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
- P2B-11: `time.text-us-date` and `time.text-one-bad-value` on SQLite, DT-15's and DT-16's readings (E68); `time.text-us-date` is a refusal of an input the reference answers on SQLite, so the owner's.
- P2B-12: `key.uint64-above-int64-max` and the vectors `wide.ubigint.*`: a right join where the reference fails INTERNAL_ERROR (KO2-11 class (a), reviewer).
- P2B-13: `time.feature-time-of-day` on SQLite answers where the reference reads a bare time as today's and refuses (owner).
- P2B-14: `time.feature-interval` on SQLite describes its INTERVAL text as text where the reference takes `INT` from the declared name (owner).
- P2B-15: a SQLite seed past 2**64 - 1 refused (RS-7's refusal), the contract's `max_seed` a `u64` (owner).

### Deviations not measured by a case (kept, for review)

- A partial unique index is no key (the reference counts a single-column one as unique); its columns are still read, as the reference reads them.
- The rowid alias is SQLite's rule, a primary key of one column declared `INTEGER`; the reference also takes a composite key's one `INTEGER` column.
- A non-`INTEGER` primary key is held not null, as on DuckDB; SQLite lets it hold a null, which the reference keeps.
- The seeded draw decides between hashing the key and the `rowid` row by row by the value's storage class, where the reference decides by the key's declared type, which a typed request does not carry; the same for a column holding the integers it declares.
- The stored time's spelling is read once per run, where the reference keeps it per sampler across a materializer's runs; a second run on one materializer reads it again, one statement more.
- The key tables are temporary and replaced, never left in the source's database.

### For the test author and the orchestrator

- The S0 cases and `refuse.unknown-column` pin `validate_query`'s schema as DuckDB describes it (`ts` a `time`); on SQLite the host's own schema is text (`string`), so `inputs` fails there, and the S0 cases' `materialize` runs the old route on SQLite (their routes name no materialize path), whose statements and temporary-table names are not the core's (the golden test cannot pin them).
- `nested.map`, `nested.struct`, `nested.union`, `nested.fixed-array` and `time.feature-interval` route only `from_source`, where DuckDB's reference refuses; on SQLite the reference answers, so their SQLite legs need the materialize and predict paths: with them, all pass but P2B-14.
- python-ref's `base.adventureworks*@sqlite` lines are of the case files before T2.1b (`identity.inputs.case`).
- The `decimal.*` SQLite legs are T2.17's.
- rust-host's SQLite pass needs a Python with DuckDB and pyarrow for the contract loader (`SDM_PYTHON`, else `python3`); this machine's `python3` has neither, and T2.2b's CI step must name one.

### Test results (this machine, load average near 30)

- `cargo test --workspace`: every test passes but the test author's still-pending ones and `t136_forecast`'s `a_forecast_of_many_steps_is_refused_at_once`, a two-second bound missed under the machine's load (3.3 s), which passes alone (2.7 s for the file).
- `t22_sqlite.rs` (with `SDM_PYTHON` naming a Python with DuckDB and pyarrow, which the contract loader needs): 8 pass; the two pending ones still fail as they must (`rust_host_runs_every_s1_case_on_sqlite`, on the registered differences and the case-level items below; `each_key_pair_on_sqlite_is_the_reference_s_on_rust_host`, on P2B-12 until approved).
- `contract/tests` (`-n 4 --dist loadfile`): 4,249 pass and one failed, three refusal-map citations moved by this package's edits, fixed in its own commit; `test_refusal_map.py` then passes whole.
- `test_t22_sqlite.py`: the provider marker passes; the three pending ones fail only on the registered differences, the case-level items below and the old route's statements.
- The client package's tests: 2,663 pass, 107 skipped.
- `prepush.sh` (cheap tier): every check passes.
- The root `tests`: 1,605 pass; the 51 that fail are all `test_contract_reference.py`, on python-ref's six AdventureWorks lines off DuckDB recorded before T2.1b changed their case files (`identity.inputs.case`), the base's, not this package's.

### Protected and judging paths touched

- Judging: `contract/runners/run_python.py` and `scripts/rust_core/compare.py` (the two first commits only), `scripts/rust_core/bridge_changes.json` and `bridge_freeze.json` (declarations and baseline).
- Protected: `rust/crates/sdm-contract/schema/**` (v26 regenerated), `rust/crates/sdm-testkit/**`, `packages/**` (the client's routing calls, the seam's test, and the G6 test), `docs/design/rust-core/register.yaml` (lane B's entries), `contract/refusal-maps/**` (three line citations), `contract/tests/t22-pending.json` (names taken out of pending), `contract/tests/test_t22_row_types.py`.
