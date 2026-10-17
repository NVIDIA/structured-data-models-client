# Phase 2 tracker: T2.1b

The design branch owns `tracker.md`; this file holds T2.1b's section until it folds in.
Built on the acceptance tests of `1cb95d6e` (`T2.1b tests: ...`), over `5e972867`; T2.0c and T2.1a land first and the orchestrator moves the branch.

## T2.1b

T2.1b is contract v26, the one bump of Phase 2 (design choice 1, lesson 14): the table source's `Relation` (choice 12) and the vocabulary the phase's later packages fill, with M1, M2 and M3.
v26 is the lock's working entry beside v25, which is merged and untouched.

### What changed, by exit

1. **Schema lock.**
   `CONTRACT_VERSION` is 26; `schema/sdm-contract-26.json` is the schema the types generate, locked by `contract.lock`'s new line `26`, and line `25` and its file are byte for byte as merged (`schema_lock.rs`, `contract_lock.py`, `tests/test_t21b_contract_v26.py`).
   `contract_lock.py --no-base`, which the root tests run on a shallow checkout, took every entry as working, so a phase's working entry beside a merged one failed by construction; with no base it now takes each entry but the newest as merged and checks the hashes, while the run with a base (prepush, CI's hygiene job) still holds the one-working-entry rule.
   Seeded under `--no-base`: v25's file edited in place fails, and an entry with no file fails.
2. **Both loaders.**
   The Python loader is the test author's (`tests/test_t21b_contract_v26.py`, against `sdm-contract-26.json`).
   The Rust loader is `sdm-contract/tests/contract_26_vectors.rs`: every accepted vector deserializes into the type it names and serializes back to the same JSON, and every refused one is refused.
3. **The binding jobs green on the bump.**
   The native module speaks 26 and refuses a v25 start by name (`INVALID_MESSAGE`, `contract` 25).
   `rust/bindings/sdm-py/tests/messages.py` writes 26 (row FX-6 derives it later) and its scan names a relation.
4. **SQL goldens unchanged.**
   A relation of one part renders exactly as v25 rendered the name: the statements recorded at v25 (`tests/golden/t21b-statements.json`) are the ones the core renders now, and the pinned statements of `sdm-data`'s family tests are unchanged, only their requests rewritten in v26's shape.
5. **A dotted source name still reads as one table.**
   `["a.b"]` renders `"a.b"`, never split, beside a decoy `b` of schema `a` (the acceptance tests, and `sdm-testkit/tests/relation_goldens.rs` executed on DuckDB).

### The interface

All plain data, as a Python or Java host builds it (KO2-46).

- `$defs/Relation` is `{"source": [string, ...]}`, nothing more; it replaces, field names kept, the `table` of `scan`, `preview`, `count_rows`, `column_ranges`, `repeated_key`, `sample_entities`, `neighbors` and `lookup`, the `source` of `NamedTable`, `DescribedTable` and `HeldTable`, `Describe.tables`' items, and `TableKey.table` and `TableColumn.table` (see the deviation below).
  The Python host sends `{"source": [name]}` and reads back the one part (`codec.relation`, `codec.relation_name`).
- `neighbors.key` and `lookup.key` are lists of column names; the seed key set holds `key` for a key of one column and `key_1` to `key_<n>` for several (`seed_key_columns`), and DuckDB joins each column to its seed column by the one-column rule, joined by `AND`.
- `ProviderCapabilities.max_seed` is a required integer, never null; DuckDB declares 2**31 - 1.
  A seed past what a `u64` holds is now always RS-7's refusal: the `seed_size` refusal it took under a provider declaring no limit is unreachable.
- Paging: `FetchOutcome::Data.continuation` (optional, opaque text) says more pages follow; the core asks again with `FetchStep.continuation` until a page comes without one, and the body reads the pages as one answer (`sdm-core`'s `Ask`); a page of another schema than the pages before ends the run, `PROVIDER_SCHEMA_MISMATCH`, check `pages`; a later page's provider error is the fetch's.
- Fallback limits: `ProviderCapabilities.fallback` (`{max_rows, max_bytes}`, optional); T2.6 reads it, and until then an undeclared optional request is still refused, `UNSUPPORTED_REQUEST`.
- Multi-call `predict`: `RequestState.batches` (each later call's `payload`, `rows` and `request_id`), `Materialize.most_entities` and `most_calls` (`main`'s `MOST_ENTITIES` and `MOST_CALLS`), `rows_per_call` among the call's values, and `CoreError.answered` (`{result, rows}`: an Arrow attachment of the answers received, `main`'s `answered` and `rows_answered`).
  T2.8c reads them; until then `predict` refuses a state with `batches` by name (`UNSUPPORTED_REQUEST`, field `state.batches`) and the core writes no `answered`.
- Ranking and forecast results: `Predicted.result`'s columns `scores`, `rankings` and `quantiles` are fixed in its documentation (Arrow carries its own schema), written from T2.8b and T2.9.
- `REQUEST_STATE_VERSION` is 2, `StatePayload`'s `x-request-state-version` with it; a version 1 state is refused by name.
- The relation's part count is the dialect's: `sdm-data` renders every `FROM` through `relation` (`lib.rs`), DuckDB taking 1 to 3 parts; SQLite (1 or 2) and Postgres (1 or 2) declare theirs with their dialects (T2.2, T2.3).
  A relation of no part is `INVALID_REQUEST` and one of more parts `UNSUPPORTED_REQUEST`, both rule `relation`, with `parts` and `most`; `describe_source` refuses a relation of no part itself, whatever the provider.
- DuckDB reads a relation of several parts' catalog where it resolves the name: `PRAGMA table_info("main"."t")` for its columns, and for its keys, foreign keys and the table list the catalog rows of its database, schema and table, two parts `[x, t]` as DuckDB reads `x.t` (schema `x` of the current database, else database `x`'s `main`).
  A foreign key's or an index's one table name matches only a relation of one part, so a relation of several parts takes no declared link and no index advice (Phase 2's hosts give one part).

### M1, M2, M3

- **M1**: `sdm-core/tests/attachments.rs` reads each `take_json` as a statement however rustfmt breaks it, from the request, the request state, a forecast's step or the sampler state; `SamplerState.memo` (the two-line decode) and `ForecastStep.query` are now held, each seeded (retyped, dropped).
  Phase 1's as-built claim that every decode was held overstated it; this closes it.
- **M2**: `schema_check` is its own feature, `schema-check`, which `test-ops` brings; `sdm-request` takes it alone, so `cargo check -p sdm-request -p sdm-core --tests` builds (E0004 at `dispatch.rs:66` before).
- **M3**: `schema_check` walks the whole definition first and fails on a keyword it does not read anywhere in it (under a property the value lacks, an `anyOf` option not tried), with tests; the contract's `StatePayload` reads whole.

### Deviation for the orchestrator

`TableKey.table` and `TableColumn.table` became relations, against the brief's decision that they name schema tables and stay strings.
The code says otherwise: the core fills both with the table's source (`sdm-core/src/describe.rs` `count_keys`, `sdm-core/src/materialize.rs` `read_bounds`), and DuckDB renders each as the `FROM` of `key_counts` and `time_bounds`.
Kept as strings, those two requests could not name a relation of several parts, and would bypass the one `FROM` function.

### Protected and judging paths touched

- Judging path: `scripts/rust_core/bridge_freeze.py` untouched; `scripts/rust_core/bridge_changes.json` declares three refactors (`_rust/schema.py::from_source`, `::build`, `_rust/materialize.py::description`) and `bridge_freeze.json` is rewritten with `--write`.
- Protected, for the landing's `Protected-paths`: `rust/crates/sdm-contract/schema/**` (the v26 schema and lock), `scripts/rust_core/contract_lock.py`, `scripts/rust_core/bridge_*.json`, and in `rust/crates/sdm-testkit/**` the v26 shape of `src/host.rs`, its tests' literals, `tests/neighbor-requests.json` (each request's `table` and `key` rewritten in v26's shape by script, nothing else) and the new `tests/relation_goldens.rs`.

### Open: judged data in v25's shape

These hold v25's wire shape, are the test author's or the gate's, and were not edited here:

- `contract/pinned/base.regression.core.json` and `edits.core.json` pin the core's inputs, fetches and description (string sources and tables, request state version 1), read by `contract/tests` and `sdm-testkit` tests;
- the three AdventureWorks cases' `core.describe_source.source` (`contract/cases/s1/base.adventureworks*.json`) name sources as strings, which `compare.py`'s `_case_inputs` holds the core's recorded inputs to, and which `sdm-testkit`'s host reads as a `SourceSpec`.
They need re-pinning and the three specs rewriting in v26's shape before the gate passes.

### Probes

- No route changes: the statements the core renders on the public route are byte for byte v25's (the golden), so no differential was rerun.
- Outside JSON: no new reader; a relation and a continuation are read by the existing bounded readers.
- The census (T2.0d, not on this base) will meet the new rule `relation`; its probe is the provider-level one the design names.
