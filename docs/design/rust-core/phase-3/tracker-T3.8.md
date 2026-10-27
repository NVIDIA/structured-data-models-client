# T3.8: the forced-route failures

Lane E, stack 1.
Brief: `$SDM_ORCH_STATE/phase-3/briefs/T3.8.md`; design row T3.8 (design.md, "Packages").

## What it found

The 118 client tests Phase 2's flip dry run failed with every core route forced (`phase-2/evidence/flip/forced-summary.txt`) were reproduced on the phase head d1e02d7c: the same 118, all in the three files the brief names.
They have five roots, not three families, and one test of the old engine's internals; the first is not a product fault:

| Root | Failures | Fix | The old client's rule |
|---|---|---|---|
| The client tests comparing the two routes' warnings counted the core's `[CODE] ` prefix as a difference | 46 | The prefix stays, on the orchestrator's decision (E41(i), E59(d)): it adds the code and loses nothing, registered as an improvement (P3E-2); those comparisons now compare the words after a leading `[CODE] ` | The words after the code are the reference's sentence, as the core gave them |
| `PostgresSource(schema=...)` was ignored: every read and the table listing used `current_schema()` | 47 | Contract v28's `ProviderChoice::Sql.namespace`, each part a name or `null`; `sdm-data` reads a one-part relation under it and lists tables in it; the host's provider choice carries the source's schema (`_rust/providers.for_source`) | `PostgresTable` and `Graph.from_postgres` read and list in the schema given, else `current_schema()` (`RE/backend/postgres/table.py:122-153`, `RE/graph.py:1306-1328`) |
| Key refusals and an id two rows hold said "the request is not one the core accepts" | 18 | The core names the rule (`key_not_unique`, `key_missing_values`, `held_by_several_rows`) and the values; the host words them | `_check_source_keys`, `_not_unique_enough`, `_a_key_missing_values`, `_SKIP_THE_CHECK` (`CL/schema.py:1517-1609`, `:2982-3030`); `_refuse_an_id_scored_twice` (`RE/base/sql_sampler.py:664-698`) |
| A table the source lacks, and a source with no table, were worded generically | 2 | Rules `no_such_table` (with the table as the source reads it and its database, through the data-layer port) and `no_tables` | `TableNotFoundError`'s sentence per backend; `engine_graph`'s `{source!r} holds no table to describe` (`CL/materializers/sources.py:157-163`) |
| A schema the core had not described raised `RouteError` at materialize | 4 | It is described on the core then, with the schema's keys and time columns and each column's type as the source holds it, and not kept (`_rust/materialize._described_now`) | `materialize` reads the catalog for any `Schema`, each column as the source holds it (`CL/materializers/relational.py:516-550`, `_copy_for_one_gather`) |
| `test_a_column_the_whole_table_could_not_check_is_kept_and_named` | 1 | None: it fakes a timeout by patching the old engine's `DuckDBTable._fetch_rows`, which the core route never calls | Candidate for T3.10's old-route-only list (KO3-37); the core's fallback on a failed `column_ranges` is held by `sdm-core`'s own tests |

The test author's 302 keys (`test_t38_forced_routes.py`) pass with every core route forced; its 141 pending keys all pass and are out of `t38-pending.json`.

## The validator (P2-MID2-1)

On the core route the core validates a query (`s0.validate`), and a query that a relational materialize gathers on the core is resolved by the core alone (`_rust.materialize.Unresolved`), so the old validator is never asked there.
Only where the old materializer gathers an accepted query (frames, or a route left old) does the old validator resolve it, as `s0.shadow`.
Held by `test_the_materialize_route.py::test_on_the_core_route_the_core_chooses_the_validator`, which makes `predictive_query.validate` raise.

## Contract v28

`ProviderChoice::Sql` gains `namespace: [string | null]`, left out when empty, so every v27 message reads the same.
A null part is the connection's current one; parts given after a null are refused by name (`UNSUPPORTED_REQUEST`, field `namespace`), as is any namespace on DuckDB or SQLite, whose connectors name none, and more than one part on Postgres.
T3.4 reuses the field for `DatabricksSource(catalog, schema)` (the orchestrator's decision, 2026-10-22); whichever of the two lands first makes the v28 working entry and the other rewrites it (H4).
Vectors: `sdm-contract/tests/messages.rs` (`a_provider_namespace_takes_names_and_null_parts`); rendering: `sdm-data/tests/namespace.rs`.

## Differences registered

- P3E-1: rejected as obsolete by Codex's round 2: since fix 2 a Postgres table missing from a source given no schema is named under the schema the read looked in, as the reference names it.
- P3E-2 (proposed, an improvement): a core warning's words follow its code in brackets, `[TIME_COLUMNS_GUESSED] Some time columns ...`; no entry recorded it before.

## Core hygiene (open question 3)

Done here, in files this package changes: P2-T2.2-2's duplicate `_provider` (now `_rust/providers.for_source`), and P2-WR-8's unreachable `SAID['CANDIDATES_UNSCORED']` (`_rust/errors.py`).
Left to T3.9: P2-WR-1, the rest of P2-WR-8 (`sdm-request/src/ranking.rs`, `sdm-contract/src/ops.rs`, `_rust/predict.py`, `client.py`), P2-WR-9, P2-T2.2-2's `_MADE_BY_ID` and `adbc-driver-sqlite` floor, and P2-T2.0c-1.

## Tests changed beside the product

Each pinned behaviour this package changes, and now pins the reference's:
`test_the_time_column_is_inferred_alike.py`, `test_a_key_is_counted_where_its_rows_are.py` and `test_tables_stay_where_they_are.py` (the two routes' warnings compared on the words after a leading `[CODE] `, P3E-2);
`test_the_rust_host.py` (`SAID['CANDIDATES_UNSCORED']` listed as predict's to word, P2-WR-8);
`test_the_materialize_route.py` (a schema not described on the core is described and materialized, as the reference materializes it; the validator test);
`test_the_schema_route.py` and `test_t133p3_big_integer_tier.py` (the provider patched where it is now made);
`tests/test_t21b_contract_v26.py` and `rust/bindings/sdm-py/tests/messages.py` (v28).

## Fix round history

- Fix 1: defbc6e0's handoff failed in the tooling self-test, whose seeds edit `_rust/schema.py`'s `from_source`; `from_source` keeps its run, and a schema the core never described is described through it.
  pandas 3.1.0 (released 2026-10-22) broke two things on both routes alike: the old engine's static anchor (`pd.Timestamp.min - pd.DateOffset(0)` now raises), fixed in the engine; and a request's description of a number pandas 3.0 could not hold, which 3.1 holds, so that test expects the old client's answer under the installed pandas.
- Fix 2 (Codex round 1, CHANGES REQUIRED; P3E-2 approved): a hand-built schema's left-out columns are no longer read (contract v28's `NamedTable.columns`); a query the core gathers alone keeps its day-first literal's warning; a missing Postgres table is named under the schema the read looked in, so P3E-1 has no difference left and awaits a reviewer's rejection.
- Fix 3 (Codex round 2 P1, Claude round 2 P1, P2s and P3s): a projected composite key no longer marks the column kept, and a named key is never refused by inference; the core route keeps the query as the core writes it back and names a forecast's step, through `CoreQuery` filled from the core's validation; P3E-1 closed as rejected by Codex's round 2.
