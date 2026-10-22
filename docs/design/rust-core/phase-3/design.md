# Phase 3: S2 live, Databricks first, and the Python switch

## Status

Draft for review (T3.1), written on 2026-10-22 from `origin/rust-core-phase-3` at 3ae4d1af (Phase 2 merged), with T3.0's process mechanisms on their topic branch and landing now.
The owner's answers so far are in [decisions.md](decisions.md), KO3-1 to KO3-26, the live-access answers of 2026-10-22 included; the open questions are at the end of this page, all together, each with a recommendation.
The acceptance checks below lock at T3.1's landing, after both reviewers approve one SHA and the day-1 questions are answered.
Everything not stated here is as in [Phase 2's design](../phase-2/design.md) and [Phase 1's design](../phase-1/design.md); their decision rows stay in force unless a row here or an owner answer changes them.
Choices marked "orchestrator's decision" are the orchestrator's, not the owner's; the owner sees them at the next check-in.
Lesson numbers are those of [learnings.md](../learnings.md) as T3.1 rewrites it.

## In one page

Phase 3 makes the Rust core answer on the three cloud warehouses the old client supports, and then makes the core the default for every user call it covers.
Today the core answers on DuckDB, SQLite and Postgres, but only when a test forces it; real users still run the old Python code on every path.

What gets built, in order:
1. Databricks first: a Databricks SQL dialect in the core, a Python provider for SQL warehouses and one for a notebook's Spark session, and the AdventureWorks quickstart notebook running on the core.
2. Databricks metric views and Snowflake semantic views, read from their YAML definitions by the core.
3. Snowflake, with its dialect and provider, and Lakebase (Databricks' managed Postgres), which reuses Phase 2's Postgres dialect.
4. Ranking and forecasting on all of them, the same five question kinds as offline.
5. The switch: every S0 (query validation), S1 (offline databases) and S2 (warehouse) path runs on the core by default, at once, after the 118 old-client tests that fail on the core today are fixed.

How it is tested without re-testing on slow warehouses after every change (the owner, KO3-22): one short live run at the start records the old client on each warehouse, every SQL statement it sends and every result it gets back, over Phase 2's test cases and AdventureWorks loaded into each warehouse (KO3-23).
All the coding is then done offline against that recording.
One live campaign at the end runs the core for real: the warehouse cases, the 21-job Databricks matrix, the quickstart notebook, live timing and the held-out set.
The early run is still needed because offline replay has nothing to compare against without it: the old client's answers on each warehouse, the statements it sends, the Arrow types each driver returns and each catalog's output exist only on the live engines.

The plan is 10 landings, T3.0 to T3.9, in three stacks of up to four packages plus the phase-end package.
The forecast from the design's landing to the switch gate is about 28 hours with four product packages building at once; the slip report compares the phase against it as it runs and never stops work.
The owner has answered the live-access questions (KO3-18 to KO3-26); what remains on day 1 is the orchestrator's: finding the sandbox catalog, schema, volume, Lakebase instance, metric views and serving endpoints the owner pointed to, which a discovery run is listing now in a private file.
No open owner question blocks day 1.
Java stays out of Phase 3 (KO2-34).

## Goal

Ship slice S2 for Python: `Schema.from_source` → `materialize` → `predict` live on Databricks (plain tables over a SQL warehouse and over Spark, and a metric view), Snowflake (plain tables and a semantic view) and Lakebase, including ranking and forecasting, with the Databricks job matrix across runtimes (migration-plan.md section 4, S2).
Then switch Python's S0, S1 and S2 paths to the core together, on the epic branch (plan section 4: "S0 (PQL validation), S1 and S2 paths switch together after S2, in Phase 3, so no call mixes old and new code").
Exit: the switch gate (plan section 7, "Python switch for a path"): S2's slice done, L6 green for its scenarios, the Databricks job matrix equal to the baseline, L7 within its limits with live timing; checked by the locked acceptance checks below.

## Order of work

1. **Before the clock starts** (no product code): T3.0 lands; this design is reviewed and locked, with the owner's day-1 answers; the full dry run of the checking tools over [dryrun-plan.yaml](dryrun-plan.yaml) runs on the landed T3.0 and its gaps join T3.2's list; one brief per package from [briefs/package.md](../orchestration/briefs/package.md); the test authors of T3.2, T3.3, T3.7 and T3.8 start.
2. **H0, four packages at once**: T3.2 (the tooling train, alone in lane A), T3.3 (Databricks), T3.7 (the forced-route failures) and T3.8 (the Python differences and coverage).
   T3.2 builds its recording mode and the warehouse loader first, so the orchestrator can run the phase's one early live capture, on all four warehouse legs at once, at about H3 while the rest of T3.2 is written.
   After it, no product package runs anything live (KO3-22).
3. **Stack 1** (T3.2 at the front, T3.7, T3.8) lands on one CI run at about H11.
   T3.5 (Snowflake), T3.6 (Lakebase and the live runners) and T3.4 (views) take the slots T3.2, T3.7 and T3.8 free.
4. **Stack 2** (T3.3, T3.6) at about H14, **stack 3** (T3.4, T3.5) at about H17.
5. **T3.9**, the switch and the switch gate: the flip, then the one live campaign at the end (the core on every warehouse case, the job matrix, the quickstart notebook, live timing, the held-out set's warehouse legs), the evidence pack and the whole-phase review, landing at about H28.
   The end-of-phase reports for S0 and S1 (L3, the census, offline L7) run beside stacks 2 and 3, as soon as stack 1 has landed.

## Scope

1. **Databricks** (E27, E31: first): the `databricks` dialect in `sdm-data` for every request kind (describe, preview, scan, counts, key counts, time bounds, column ranges, repeated keys, entity sampling, neighbors, lookup), its catalog reader into the neutral `CatalogAnswer`, key sets bound as the old client binds them (`posexplode(from_json(?, ...))`), sampling as `TABLESAMPLE (n PERCENT) REPEATABLE (seed)` with the percentage printed as Python prints it (RS-6, if the owner answers Q10 as recommended); the SQL-warehouse provider (`fetchall_arrow`) and the Spark provider (`spark.sql(...).toArrow()`, DBR 17.3 or later and serverless environment 4 or later, as the old client requires); the routes for every `databricks-sql` and `databricks-spark` path; and `examples/databricks/kumo_relational_quickstart.ipynb` running on the core.
2. **Views**: a Databricks metric view and a Snowflake semantic view each described through its YAML definition into a view model the core reads without dialect terms (Phase 2's choice 12), with renamed and expression columns and a row filter carried on the relation, measures and metrics left out as conversion notes, as the old client does; the expression allow-list of the old client's `view_expr.py`, with `trust_expressions`.
3. **Snowflake**: the `snowflake` dialect (double-quoted names with the old client's upper-case rule, `NUMBER(p,s)`, keys bound as `TABLE(FLATTEN(PARSE_JSON(?)))`, `QUALIFY`, `::TIMESTAMP_NTZ`), its catalog reader (`DESCRIBE TABLE`, `SHOW IMPORTED KEYS`, `SHOW TABLES`), the semantic view's retrieval, and the provider (`fetch_arrow_all`, on a connector connection or a Snowpark session's connection).
4. **Lakebase**: a `PostgresSource` over a Lakebase connection on Phase 2's Postgres dialect, the `sdm_connectors.databricks.lakebase()` connect path, the `lakebase` routes, and Postgres reads qualified by the relation's schema part (P2-MID2-6).
5. **Ranking and forecasting on every warehouse leg**, through the decomposition Phase 2 moved into the core; nothing new in the core's operations.
6. **The contract bump, v28, once** (lesson 11), in T3.3: the relation's optional `columns` (each a schema name with a source column or an SQL expression) and `filter`, a provider capability for such derived relations (a provider without it refuses one by name), and a failure class on `ProviderError` that the provider sets from its driver's codes (Phase 2's choice 12 reserved all four), with the contract rows Phase 2 carried for it (below).
7. **The job matrix**: the 21 cells of `phase-0/databricks-jobs.md` (J01 to J21: SQL warehouse, Spark and Lakebase, DBR 14.3 to 17.3 and serverless 1 to 6, x86_64 and aarch64, both in scope, KO3-26; CI's `native-wheel` already builds the aarch64 wheel, KO2-40), each a one-time job installing the candidate's own wheels (KO2-95's rule), run on the reference for the baseline and on the core, both in the end campaign, compared question by question, against the owner's existing serving endpoint and metric views (KO3-21).
   The DBR 14.3 install revalidation Phase 1 carried (T1.35, E154) is cells J01 and J02.
8. **The switch**: every S0, S1 and S2 default in `sdm_client/_routing.py` set to `core` in one commit (owner question Q5), the routing guard test changed with it, and the existing client test suites green on the default routes.
   `SDM_CORE_ROUTES` stays, test-only, until Phase 5 deletes the old engine.
9. **What the switch needs from Phases 1 and 2** (the carried rows, next section): the 118 forced-route failures, S1 coverage, the 12 unregistered user-visible differences, L3 and H24 on every database, DuckDB 2.0.
10. **The gate's live parts**: L6 on every warehouse leg, the job matrix, L7 live push-down timing (median of at least five interleaved runs, caches off, within 10% on Databricks and Snowflake; owner question Q8), and the held-out set's warehouse legs.
11. **The phase-end chores**: `main` merged and the reference rerun if `main` has moved (it has not since 43e55f4f), register reconciliation, the evidence pack, the whole-phase review (handbook, "Ending a phase").

### Explicitly out

- Java, including "Python and Java agree on the same Databricks data": deferred until S2 is switched for Python (KO2-34); its own track after this phase, unless the owner adds it (owner question G3). The Java host rows Phase 2 carried (P2-WR-2, P2-WR-5, P2-WR-6, P2-T2.14b-4, P2-MID2-4) go with it.
- S3: frames, files read from S3, link prediction, sessions, tabular requests, and the three refusals kept until Phase 4 (KO2-94); the fallback rows P2-T2.5-1 to -3 and FX-3 (Phase 4, placed there by owner rows).
- Release work (Phase 5): `publish.yml`, the SBOM, macOS, Windows and musl wheels, signing; ASan and LSan (KO2-25).
- `MEASURE()` and `SEMANTIC_VIEW()` queries: the old client never issues them (it reads a view's base tables), so the core does not either.

## The carried rows the switch needs

Migration-plan.md section 8, "Carried into Phase 3", holds 142 rows.
Every row ends in a package or a recorded "no" (lesson 18); the row text stays in the plan, and the package's tracker closes it.
"No" rows are the orchestrator's decision under KO2-52 (internal-tool hardening is P3 at most) and the owner's "product over gate machinery": they are reopened only when the dry run or a Phase 3 case hits them, and then go into T3.2 or the train.

| What | Rows | Package |
|---|---|---|
| **The 118 forced-route failures** (Phase 2's flip dry run: 2,789 client tests with every core route forced; 61 miss the "Some time columns..." warning, 38 count keys otherwise or refuse `INVALID_REQUEST`, 19 refuse on Postgres or raise `RouteError` where materialize needs a schema described on the core) | P2-T2.16-9, P2-T2.16-10, P2-T2.16-11 | T3.7 |
| **S0 fully on the core**: the host still picks the old validator for a query | P2-MID2-1 | T3.7 |
| **The 11 other unregistered user-visible differences** (with P2-MID2-1, the 12): `task=` wording and order; `datetime` as `context_anchor_time`; NaN `max_pq_iterations`; Postgres decimal inference through `ToDouble`; unsigned 64-bit `IN`; fractional `IN` against a scale-0 decimal; the fractional-iterations statement count; the generic composite-key message; `rows_per_call` naming the value; answers parsed twice; `RecursionError` on a deep answer | P2-WR-3, P2-WR-4, P2-WR-7, P2-T2.14b-1, P2-T2.10-1, P2-T2.1p-1, P2-T2.11-1, P2-T2.12-1, P2-T2.8c-1, P2-T2.14b-2, P2-T2.14b-3 | T3.8: each matched to the reference, or registered and sent to the owner |
| **S1 coverage**: 58 of 75 S0 and S1 paths and 22 of 44 cells with no acceptance case | P2-T2.16-3 (with G12, G16, G17) | T3.2 (the coverage rule: a case's SQLite and Postgres legs claim their own paths; a client test may claim a path the case format cannot express); T3.8 (the cases for the rest) |
| The S1 test rows that close coverage gaps | P2-T2.19-1, P2-T2.1b-2, P2-T2.1b-3, P2-T2.1b-4, P2-T2.5-4 | T3.8 |
| The Postgres numeric-spelling cases never run in CI; `SDM_REQUIRE_POSTGRES` on every Postgres step | P2-T2.14c-1 | T3.2 |
| **L3 on SQLite and Postgres**, the two classes it cannot name, the stale class list | P2-T2.16-4, P2-T2.16-5, P2-T2.11-2, M8 | T3.2 |
| **H24 locally**, the per-case table with database legs | P2-T2.16-7, P2-T2.16-6 | T3.2 |
| P2G-1 widened to every call the endpoint refuses | P2-T2.16-13 | T3.8 |
| **DuckDB 2.0** support and the `zip` advisory exceptions | P2-T2.16-12, OF-2 | T3.8 |
| The interim edit gate's retirement | P2-T2.16-8 | T3.8 |
| The overnight refusals that refuse input the reference answers (P2B-19, P2I-1, P2I-4) | OF-5 | T3.8, after owner question Q13 |
| **Contract rows for v28** | P2-MID2-2, P2-T2.2-1, P2-T2.1b-1, P2-T2.8c-2, P2-MID2-3, P2-MID2-5, FX-6 | T3.3 |
| Postgres reads qualified by the relation's schema part | P2-MID2-6 | T3.6 |
| Expression columns and metric views (KO2-25, KO2-39) | OF-4 | T3.3 (the contract fields), T3.4 (the views) |
| RS-6 on the warehouses | OF-3 | T3.3, after owner question Q10 |
| The live benchmark's "different requests" result fails the run | G8 | T3.6 |
| The held-out set's stored-NaN and `key_name_order` cases | OF-6 | HO3 |
| A bigger machine; branch protection on `rust-core` | OF-9, OF-7 | Owner questions Q15, Q14 |
| Judging rows Phase 3's work hits: owner approvals read by allowlist; the runner's statement log on psycopg copy and stream and the warehouse drivers; a stale installed wheel; a stale `origin/main`; refusal maps citing quoted text; time zones set per session and process | U4, P2-T2.1a-2, P2-MID2-7, P2-T2.0f-1, P2-T2.1b-6, C16, C18, M5 | T3.2 |
| Cheap CI rows on the path | P2-T2.1a-3, P2-T2.16a-1, P2-T2.11-4, C28, and P2-T2.16-14 if T3.0's split has not fixed the slow job | T3.2 |
| The test that names half `3a` as a half with no locked set | P2-T2.0a-1 | T3.1 (done in this draft: half `9a`) |
| Docs tidy: Phase 2's trackers, as-built and decision rows | P2-T2.16-1, P2-T2.16-2, P2-T2.1p-2 | T3.1 |
| Register text tidy | P2-T2.0c-3, P2-T2.8c-3, P2-T2.13b-1, P2-T2.21-2, P2-T2.14c-2 | T3.8 |
| Core hygiene, done by the package that next touches the file | U12, P2-T2.2-3 (`sdm-data`: T3.3); P2-WR-1, P2-WR-8, P2-WR-9, P2-T2.2-2, P2-T2.0c-1, P2-T2.7-1, P2-T2.7-2 (T3.7 or T3.8, whichever touches the file) | as listed |
| Covered by T3.0 | OF-1, P2-T2.1o-1, P2-T2.1n-1 (the tests' part), ENV-1, P2-T2.0b-3, P2-T2.0g-1 (the reviewer's part) | T3.0 |
| Checked once in a lane environment, closed if they pass | ENV-2, ENV-3 | T3.2 |
| **No**, until the dry run or a case hits them | FR P3-7, U3, U13, C2, C4, C5, C6, C7, C8, C9, C11, C22, G2, G4, G10, FX-2, M9, P2-WR-10, P2-WR-11, P2-T2.0-1, P2-T2.0b-1, P2-T2.0b-2, P2-T2.0b-4, P2-T2.0c-2, P2-T2.0e-1, P2-T2.0h-1, P2-T2.1a-1, P2-T2.1b-5, P2-T2.1b-7, P2-T2.1d-1, P2-T2.1e-1, P2-T2.1f-1, P2-T2.1g-1, P2-T2.1g-2, P2-T2.1h-1, P2-T2.1j-1, P2-T2.1j-2, P2-T2.1k-1, P2-T2.1k-2, P2-T2.1l-1, P2-T2.1m-1, P2-T2.4-1, P2-T2.7-3, P2-T2.11-3, P2-T2.21-1 | none |
| Phase 4 or the Java track (placed there by owner rows or their reviewers) | P2-T2.5-1, P2-T2.5-2, P2-T2.5-3, FX-3, OF-8; P2-WR-2, P2-WR-5, P2-WR-6, P2-T2.14b-4, P2-MID2-4 | none in Phase 3 |

L3 and H24 on all databases: L3 (`contract/tools/differential.py`) runs about 200 generated sources on each of DuckDB, SQLite and Postgres (KO2-26) and, if the owner answers Q12 as recommended, 20 on each of Databricks and Snowflake inside one live job each; H24, the core fed the reference's own fetched data, runs on Postgres locally (P2-T2.16-7) and on every warehouse leg by construction, since the warehouse legs replay the reference's captured results (choice 2).

## Architecture choices new in this phase

Phase 1's and Phase 2's mechanisms are reused as built: the contract and its lock, the data port and capabilities, `Relation`, runners and cells, comparator, ratchet and checkpoint, register, coverage, gate, the provider kit, the replay provider, the bridge freeze, the routing seam, the evidence pack; and T3.0's stacks, train, handoff stamp, CI split, lane environments, command hook, dry run and slip report.
Every new interface is held to KO2-46: only what this phase uses, as plain data a Python or Java host builds without language idioms.

### 1. Warehouse legs are database legs of the existing case format

**Recommendation:** add four warehouse legs, `databricks-sql`, `databricks-spark`, `snowflake` and `lakebase`, beside DuckDB, SQLite and Postgres, keyed `<case>@<leg>` exactly as Phase 2 keyed `<case>@sqlite`.
A case's data is still its base and setup (DuckDB statements), loaded into a sandbox schema of each warehouse once, through Arrow, by Phase 2's contract loader extended with a type map per engine (Databricks and Snowflake have no unsigned integers, no time-of-day type in tables, `DECIMAL` up to 38 digits, and their own timestamp kinds); a type an engine cannot store is listed and never counted as covered (KO2-47's rule).
An S2 case names its warehouse legs in a new field, so it runs only there and needs no exclusion; every S1 acceptance case runs on the warehouse legs too, as Phase 2's do on SQLite and Postgres, with the types an engine cannot store listed (the owner, KO3-23: "Can we replicate whatever dataset/testing you've done in phase 2 here?").
AdventureWorks is loaded the same way, and its cases ask the quickstart notebook's 5 questions (3 binary, 1 multiclass, 1 binary through a metric view), the deleted notebook's 8 (regression twice, forecast, binary, ranking, multiclass, and two through a metric view), and a ranking and a forecast on each leg, since the quickstart alone "is not enough for testing" (KO3-23).
The metric view and semantic view legs are cases on `databricks-sql`, `databricks-spark` and `snowflake` carrying a new `input.view` (the kind, the name and the YAML), and the connect paths carry `input.connect`.
Why: every S1 case already has reference lines and core observations on three databases, so a warehouse leg reuses the comparator, ratchet and coverage as they are, and one case set covers offline and live.
Rejected: separate S2 case files over a separate live dataset (a second corpus to keep in step); live-only cases (a live run per bug).

### 2. Capture the old client once live, replay it offline

**Recommendation:** the runner gains a recording mode for the reference on a warehouse leg: every statement the old client sends, with its parameters, and the full Arrow result, not only a digest, in a capture file per case and leg, with catalog, schema, host and user names replaced by placeholders before anything is written (the repository is public).
A replay provider (Phase 2's `sdm-testkit/src/replay.rs`, generalised, and its Python twin in the runner) answers the core's statement from the capture when its normalised text and parameters match a captured one, and fails by name, "statement not in the capture", when none does.
The core's dialects port the old client's statements one by one (the old samplers and table readers are the reference code), so a matching statement is the normal case and the whole S2 corpus replays offline from the one early capture.
A statement the core sends that the old client never sent has no recorded answer before the end campaign, so the design avoids them: a dialect package that needs one names it in its brief and registers it, and the end campaign confirms it live.
The end campaign records the core's own live run as well; that second capture is what rust-host and CI's warehouse jobs replay from then on, as Phase 2's rust-host replays Postgres (KO2-60); until it exists, both hosts replay the old client's capture.
The capture runs once, at about H3, as one short campaign over all four legs in parallel sessions and jobs; it is repeated for a case only if the case or its base changes.
Why: a live debugging cycle costs 15 to 30 minutes and the owner wants no re-testing on the warehouse after each change (KO3-22); an offline replay costs seconds, and it is H24's common-input comparison by construction (owner question Q11).
Rejected: a warehouse emulator (DuckDB with SQL transpiled) would judge SQL nobody runs in production; live-only development would put every fix through a remote job.

### 3. One contract bump, v28, in T3.3

**Recommendation:** v28 holds everything Phase 3 adds, as optional fields with defaults, so no existing field changes shape (Phase 2's choice 12 planned exactly this):
- `Relation.columns`: each a schema name with either a source column or an SQL text expression; `Relation.filter`: an SQL text row filter;
- the `derived_relations` provider capability; a provider without it refuses a derived relation by name;
- `ProviderError.failure_class`, set by the provider from its driver's codes (for example SQLSTATE), never from message text;
- the carried contract rows: the data layer's built-in-only flags folded into `ProviderCapabilities` (P2-MID2-2), seed meanings and answered parts as explicit fields rather than metadata strings (P2-T2.2-1, P2-T2.8c-2), the unread-text-columns warning with separate table and column (P2-MID2-3), `index_columns` documented with a vector (P2-T2.1b-1), the unused composite-key vocabulary recorded as groundwork or cut (P2-MID2-5), and the binding tests reading the version from the crate (FX-6).
SQL text stays opaque to the core and is rendered only by `sdm-data`'s dialects; the core never matches error text.
If a later package needs vocabulary, it rides in T3.3 before T3.3 lands, or the second package to land renumbers (lesson 11).

### 4. Two dialects, ported, in `sdm-data`

**Recommendation:** `databricks` and `snowflake` modules in `sdm-data` beside DuckDB, SQLite and Postgres, each rendering every request kind through Phase 2's one `FROM` function, declaring every capability with no fail-open default, and accepting the relation name parts its engine does (three for both).
Start from the old client's statements (`engine/rfm/backend/databricks/` and `snow/`, the `graph.py` builders, `SQLSampler`) and the first attempt's `dialects/src/databricks.rs` and `snow.rs` (349 and 358 lines) as a port source; never merged as is (decision 13).
The L0 guards already keep the two names out of every other crate.
Each dialect package runs the key-type pair matrix on its engine through replay before review (lesson 7), with `DECIMAL(38,x)` keys meeting Phase 2's exact decimals (KO2-35).

### 5. Host providers stay thin

**Recommendation:** three Python providers in `sdm_client/_rust/providers/`: Databricks SQL warehouse (`cursor.fetchall_arrow()`), Databricks Spark (`spark.sql(...).toArrow()`), Snowflake (`cursor.fetch_arrow_all()`, on a connector connection or `session.connection` of a Snowpark session); Lakebase reuses the Postgres provider.
Each hands Arrow to the core with no pandas round trip and passes the provider kit (Phase 2's T2.5) on its replayed fixture and once live.
Authentication, connection objects, Lakebase's hour-long database credential and backend detection stay in the host: they are language objects the core cannot compute (lesson 1).
Backend detection learns to tell a SQL warehouse, a Spark session and a metric view apart; both Databricks connections report `databricks` today.

### 6. Views: a crate for YAML, a port for the rules

**Recommendation:** each dialect retrieves a view's definition with the old client's statement (Databricks: `DESCRIBE TABLE EXTENDED`'s YAML; Snowflake: `SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)`) and hands the text to `sdm-schema`, which parses it with a YAML crate (`yaml-rust2`, maintained and pure Rust; `saphyr` is its younger successor) and interprets it into a view model with no dialect terms: tables, renamed and expression columns, joins as links, the filter, and measures or metrics as conversion notes.
YAML values are read as the old client's PyYAML reads them (YAML 1.1: `yes`, `on` and `off` are booleans where PyYAML makes them so) for the keys the view model reads, through PyYAML's documented implicit-type rules ported as a defined algorithm over the crate's plain scalars (lesson 3; owner question Q9).
The expression allow-list is a port of `view_expr.py` (1,744 lines), with `trust_expressions` as the old client has it; the first attempt's 2,150-line hand-written YAML reader is not ported.
A `phase-1/dependencies.md`-style row records the crate choice and its measured reason (lesson 3).

### 7. Sampling and session state on warehouses

**Recommendation:** match the old client's SQL: Databricks `TABLESAMPLE (p PERCENT) REPEATABLE (seed)` with the percentage written as Python's `str(float)` writes it, Snowflake's sampling statement as the old client sends it, and `ORDER BY` and `QUALIFY ROW_NUMBER()` forms unchanged, so the same statement draws the same rows on the same table and the capture answers it (owner question Q10, RS-6's coverage).
The session time zone is fixed per capture and named in it; a zoned time column on a warehouse is a case of its own on each leg, crossing the UTC instant and the session wall clock (lesson 6's "every reading").

### 8. The switch, in one commit

**Recommendation:** T3.9 sets every S0, S1 and S2 path's default to `core` in `_routing.py` at once, so no call mixes old and new code (plan section 4), and changes the guard test that holds the defaults (`test_the_routing_seam.py`) in the same commit.
Inputs the core still refuses by name on a switched path are only the three KO2-94 keeps, if any of them is reachable from S0, S1 or S2 paths at all (T3.2's census answers that); they stay refused by name on the epic branch until Phase 4 lifts them, which happens before the epic can merge into `main`, so no user of a released client sees them (owner question Q5).
Client tests that test old-engine internals rather than public behavior are marked to run on the old route only until Phase 5 deletes the engine, each listed in the evidence (owner question Q6); every other client test must pass on the default routes.

### 9. Host-held rules in scope (lesson 4)

| Rule | Where it lives after Phase 3 |
|---|---|
| Which validator checks a query (P2-MID2-1) | The core (T3.7) |
| View YAML parsing and interpretation, and the expression allow-list | The core (`sdm-schema`, T3.4) |
| Dialect SQL, catalog reading, key binding, sampling statements | The core (`sdm-data`, T3.3, T3.5) |
| Connections, authentication, Lakebase credentials, backend detection, the Spark and Snowpark sessions | The host: language objects |
| The serving transport (`core/databricks_serving.py`) | The host: I/O, as the plan's run loop puts it |
| The model and route choice from the schema's shape | The host until Phase 4's tabular requests (Phase 1's ledger) |

The rust-host leg runs every S0, S1 and S2 case, the warehouse legs on the core's captures, before the held-out run.

## The live-test plan

Live work runs only under the owner's live rules: one-time jobs only, uploads only to the authorized volume, no existing notebook or cluster touched, only the Lakebase instance the owner pointed to, and three live failures for non-product reasons stop live work (plan section 9).
The owner's shape (KO3-22): code against offline replay, and run the live tests at the end, because the warehouses are slow.
So there are exactly two live campaigns: one short capture at the start, and the tests at the end.
Every live run installs the candidate's own wheels, built by CI for x86_64 and aarch64 (KO2-40, KO3-26).
Nothing account-specific enters the repository: the profile, volume, catalog, warehouse, instance, endpoint and view names come from the environment and from the orchestrator's private notes; Snowflake's credentials come from the owner's env file, loaded into the run's environment and never read or printed (KO3-24).

**Why one early capture is still needed.**
Offline replay compares the core against the old client's answers; on a warehouse those answers, the statements the old client sends, the Arrow types each driver returns and each catalog's output exist only on the live engine.
Without them, the dialects would be written blind and the end campaign would find every difference at once.
It is one run, about 3 hours, all four legs in parallel, before any dialect exists, and nothing product-related waits on Databricks after it.

**Offline, from the early capture** (seconds per case, no account needed):
- every warehouse leg of every case, on python-core and rust-host, against the old client's captured statements and results (choice 2);
- every dialect's SQL text, pinned by goldens and checked statement by statement against the capture;
- metric and semantic view interpretation, from the captured view YAML;
- the provider kit on each provider, from a captured run of its fixture;
- Lakebase's whole data path on the local Postgres 17 that CI already runs, plus its replayed captures.

**Live only, at the end:**
- that the real engines accept the core's SQL and return the types the capture recorded;
- `TABLESAMPLE` repeatability and the warehouses' sampled rows;
- authentication and connect paths (`warehouse()`, `lakebase()`, Snowpark) and the serving endpoint;
- the quickstart notebook on the core, the job matrix, live L6, live L7 timing, and the held-out set's warehouse legs.

| Live run | When | What runs | Jobs or sessions | Wall time (forecast) |
|---|---|---|---|---|
| C-1 The early capture | once, at about H3, from T3.2's candidate | Phase 2's bases and AdventureWorks loaded into a sandbox schema of each warehouse; every S1 acceptance case and every S2 case run on the old client on each of the four legs and recorded; the existing metric view's and a semantic view's YAML recorded; the provider kit's fixture run once | SQL warehouse and Snowflake: sessions from this machine; Spark and Lakebase (reachable from serverless only): one one-time job each | about 3 h, the legs in parallel |
| E-1 The core on every warehouse case (L6) | the end campaign, on T3.9's switched candidate | Every case on python-core live on each leg, recorded as the core's capture; equal to C-1 | As C-1 | about 2 h, in parallel |
| E-2 The job matrix, both passes | the end campaign | 21 cells, 156 question runs per pass, once on the reference wheels and once on the candidate's, against the owner's existing endpoint and metric views | 42 one-time jobs, in parallel | about 1.5 h |
| E-3 The quickstart notebook on the core | the end campaign | `kumo_relational_quickstart.ipynb` on the candidate's wheels, against the existing endpoint | One one-time job | about 45 min |
| E-4 Live L7 timing | the end campaign, after E-1 | Push-down median over at least 5 interleaved runs, caches off, Databricks and Snowflake | Sessions from this machine | about 2 h |
| E-5 Held-out warehouse legs | the end campaign, once | HO3's warehouse cases on both routes | One job or session per warehouse | about 1 h |
| E-6 L3 on warehouses (if Q12 is yes) | the end campaign, beside E-2 | 20 generated sources per warehouse, both routes | One job or session per warehouse | about 1 h |

The end campaign's runs overlap: about 3.5 hours of wall clock, plus one forecast fix cycle (a fix, its offline replay, and a rerun of only the failing cases, about 2 hours).
Cost: about 48 one-time jobs (2 for the capture, 42 for the matrix, 1 notebook, 2 to 3 held-out and L3), each 10 to 20 minutes on a single-node cluster or serverless; the authorized SQL warehouse for about 6 hours of sessions in all (auto-stopping); a Snowflake warehouse for about 4 hours; the existing serving endpoint while the matrix and the notebook run, about 2 to 3 hours, scaling to zero after.
The orchestrator cannot see prices; the owner can turn these hours into money.

## Packages

M is mechanical (a port with reference recordings), N novel, R risky.
"Judging" means the package changes a path of `scripts/rust_core/judging-paths.txt`; after T3.0 only T3.2, the tooling train, may (T3.0's rule in `land.py`).
Every product package has a separate test author who writes its cases first; the cases' reference lines come from the early capture the orchestrator runs (C-1); the implementer never edits the test author's files.
Every package runs `handoff.py` before review and ports the old client's rule in each fix (`Rule:` lines).
Lanes: A head and tooling, B Databricks, C views, D Snowflake, E switch-ready, F differences and coverage, G Lakebase and live runners; at most four product packages build at once (the machine has 8 vCPUs, 30 GB and 412 GB of fast disk; T3.0's `heavy` admits by all three).
Hours are implementation plus one or two review rounds, as Phase 2 measured its planned work (`time-accounting.md`: its planned work came in on plan; its overrun was overhead T3.0 now removes).

| ID | Package | Output | Exit check | Depends on | Test author | Lane | Stack | Kind and hours |
|---|---|---|---|---|---|---|---|---|
| T3.0 | Process mechanisms | Stacks, the tooling train, the handoff stamp, CI's split and timing guard, lane environments, the command hook, the dry run, the slip report | Its own (T3.0's brief) | – | T3.0's | A | alone | landing now |
| T3.1 | This design | design.md, decisions.md, tracker.md, locked-set.yaml, forecast.yaml, dryrun-plan.yaml; learnings.md rewritten; `briefs/package.md`; `lane.py start --brief` refusing a brief missing a heading and refusing a package ID with a letter from Phase 3 on; `tests/test_learnings_enforced.py`; the test naming half `3a` moved to `9a`; docs tidy rows | Two APPROVEs on one SHA; the locked-set reader reads half `3a`; each new check's one focused test fails without it | T3.0 | T3.1's (for the two checks) | A | alone | docs, plus about 1 h of tooling |
| T3.2 | The tooling train | The case format's warehouse legs, `input.view` and `input.connect`, with malformed vectors for both loaders; the recording mode and the replay provider (choice 2) with placeholders for private names; the loader's warehouse type maps; `<case>@<leg>` keys in the comparator and ratchet; per-leg CI jobs replaying captures (`contract-python (ref-pandas3, <leg>)`, `cross-host (<leg>)`); coverage claiming database legs and client tests; L3's database axis; local H24; the per-case table by leg; `register.py` refusing an `approved_by: owner` entry with no owner-approved row (lesson 20) and U4's allowlist; the S2 case IDs added to half `3a`; the other judging rows of the carried table; every GAP of the full dry run at lock | Each change's one focused test fails without it; a stub of each kind in dryrun-plan.yaml passes the dry run with no GAP; a statement missing from a capture fails by name; a private name in a capture fails the write | T3.1 | T3.2's | A | 1, front | N, about 9 h (recording mode first, about 2.5 h) |
| T3.3 | Databricks and contract v28 | Choice 3's bump; the `databricks` dialect and catalog reader; the SQL-warehouse and Spark providers; routes and backend detection for `databricks-sql` and `databricks-spark`; RS-6's percentage text; failure classes from SQLSTATE; U12, P2-T2.2-3 | Every case's `databricks-sql` and `databricks-spark` legs equal to the old client's capture on python-core and rust-host, offline; every statement it sends found in the capture, or named and registered; the key-type pair matrix on Databricks; the provider kit on both providers, replayed; schema lock and both loaders at v28 | T3.2's recording mode (stacked on its candidate) and C-1 | T3.3's | B | 2 | M/R, about 9.5 h |
| T3.4 | Views | The view model in `sdm-schema`, the YAML crate with PyYAML's 1.1 reading, the expression allow-list port, metric view retrieval and interpretation, semantic view interpretation; its dependencies row | Every metric-view case on `databricks-sql` and `databricks-spark` equal to the capture, offline; semantic-view interpretation from the captured YAML equal to the old client's schema; YAML 1.1 value vectors (`yes`, `on`, `off`, `~`, octal-looking text) read as PyYAML reads them | T3.3 (v28's relation fields; stacked on its candidate) | T3.4's | C | 3 | M/R, about 9 h |
| T3.5 | Snowflake | The `snowflake` dialect and catalog reader, the semantic view's retrieval, the provider (connector and Snowpark), routes | Every case's `snowflake` leg and every semantic-view case equal to the capture on both hosts, offline; the key-type pair matrix and a session-zone case on Snowflake; the provider kit, replayed | T3.3; T3.4 for the semantic-view cases; C-1 | T3.5's | D | 3 | M, about 8 h |
| T3.6 | Lakebase and the live runners | The `lakebase()` connect path and routes on the Postgres dialect; reads qualified by schema (P2-MID2-6); the job-matrix runner and its one-time job template in `scripts/live/`, installing the candidate's wheels on each cell; the real-data benchmark failing on different requests (G8) | Every case's `lakebase` leg equal to the capture and on local Postgres; the runner's dry mode prints 21 cells' job specs, x86_64 and aarch64, and a cell installing a published wheel instead of the candidate's fails | T3.2 | T3.6's | G | 2 | M, about 6 h |
| T3.7 | Switch-ready: the forced-route failures | The 118 failures fixed in the core or the bridge's wording, each family ported from the old client's rule (the time-column warning, key counting and its refusals, Postgres refusals and the described-schema `RouteError`); the validator chosen in the core (P2-MID2-1) | `pytest packages/structured-data-models-client/tests` with every S0 and S1 route forced: 0 failures; a case per family on three databases | T3.1 | T3.7's | E | 1 | R, about 8 h |
| T3.8 | Differences, coverage and DuckDB 2.0 | The 11 user-visible rows matched or registered; OF-5 per Q13; acceptance cases for every S0 and S1 path and cell coverage still finds empty; the S1 test rows; P2G-1 widened; DuckDB 2.0 supported and the `zip` exceptions dropped if its build allows; the interim edit gate retired; register tidy; hygiene rows it touches | `coverage.py --gate S1` (T3.2) with no empty path or cell; each row's case fails before and passes after; CI green on DuckDB 1.5 and 2.0 | T3.1 | T3.8's | F | 1 | M, about 9 h |
| T3.9 | The switch and the switch gate | Choice 8's flip and guard test; the old-route-only list; register reconciled; `main` merged if it moved; the locked run; the end-of-phase reports; the end campaign E-1 to E-6 and its one fix cycle; the core's capture committed for rust-host and CI; HO3 once; the evidence pack; the whole-phase review and its fixes; the packet | The locked checks below | all | T3.9's (the guard test and the switched suites) | A | alone | R, about 8 h after stack 3 |
| HO3 | The held-out set | About 40 cases: about 25 offline across the three databases, about 15 on the warehouse legs, with a stored-NaN case and a `key_name_order` case (OF-6); written by a held-out author who reads no implementation, kept outside the repository | Written beside the lanes, which do not wait for it; frozen before its one use in T3.9; its converter tested and two dummy bundles run through `heldout.py counts` | T3.2 (the case format) | – | orchestrator | not a landing | – |

T3.2 is the phase's one planned train; a judging change found later rides at most one catch-up train at the front of the next stack (T3.0's rule), and any further one goes to the owner (lesson 13).
Owner decision rows and register entries ride with the package that needs them (KO3-1).

## Critical path and forecast

The forecast is the time from T3.1's landing (H0) to each landing, in landing order; [forecast.yaml](forecast.yaml) holds it for T3.0's slip report (`phase_stats.py --forecast`), which the heartbeat runs and which never stops work (KO3-12).

| Landing | Forecast | Holds the queue for | Why it takes that long |
|---|---|---|---|
| Stack 1: T3.2, T3.7, T3.8 | H11 | 11 h | T3.2's 9 h of work and two review rounds; T3.7 and T3.8 are written beside it and approved by about H10 |
| Stack 2: T3.3, T3.6 | H14 | 3 h | T3.3's dialect from H0, replayed against C-1 from about H6, review to about H13 |
| Stack 3: T3.4, T3.5 | H17 | 3 h | T3.4 from about H6 on T3.3's candidate; T3.5 from about H7, its semantic-view cases on T3.4's candidate |
| T3.9 | H28 | 11 h | The flip and the suites (2 h); the end campaign (3.5 h) and one fix cycle (2 h); the whole-phase review and its fixes (3.5 h); the evidence pack beside them |

**Critical path:** T3.1 → T3.2's recording mode and loader (2.5 h) → C-1 (3 h) → T3.3 against the capture → stack 2 → T3.4's views on v28 → stack 3 → T3.9 with the end campaign: about 28 hours of wall clock from the design's landing.
Before H0: T3.0's landing, this design's review rounds, the discovery of the owner's existing resources, the test authors' first cases and the dry run at lock.

This forecast misses the owner's one-day goal by about 4 hours plus the prep.
Putting all live testing at the end (KO3-22) moves about 2 hours of live runs out of the middle of the phase and into T3.9, where a real-engine surprise costs a fix cycle; the forecast holds one.
What would shorten it, in order: starting C-1 from T3.2's recording mode before T3.1 lands (about 2 hours); a bigger machine (owner question Q15) for a fifth build slot (about 2 hours).
None of them moves scope out of Phase 3.

## Machine and lanes

At most four product packages build at once (KO3-16), each in its lane's environment (`cells.sh`), with `heavy` admitting builds and gate legs by CPU, memory and disk; test authors and reviewers run no heavy suites (KO3-15).
The orchestrator runs the two live campaigns and the reference machine's work; implementers never run live jobs.

## The checking tools' dry run

[dryrun-plan.yaml](dryrun-plan.yaml) holds 21 stub cases in five groups: an S1 kind on each warehouse leg, what only a warehouse has (views, connect paths), warehouse values (`DECIMAL(38,6)` keys, a session-zone time column, a `STRUCT` feature), the switch (database legs claiming their paths, an uncovered edit path, a pickled request, the key-repeating family, S0 validation on the core) and the endpoint refusing a one-call request.
A first run of T3.0's `gate_dryrun.sh` from T3.0's topic branch, judged against Phase 2's last landing (the case reader, `coverage.py` and `ratchet.py`; no runners), found:

| Gap | Kinds | Goes into |
|---|---|---|
| The case format's `databases` names only DuckDB, SQLite and Postgres, and naming fewer needs an owner exclusion | all 15 S2 kinds | T3.2: warehouse legs (choice 1) |
| `input.view` is not a case field | the 4 view kinds | T3.2 |
| `input.connect` is not a case field | the 2 connect kinds | T3.2 |
| `input.pickle` is not a case field | the pickled-request kind | T3.2 (or the path is claimed by a client test, choice of T3.2's coverage rule) |
| Coverage's rule 6 wants a reference observation of every claimed cell | the 5 S1 kinds | Expected for a stub with no reference line; not a gap |
| `coverage.py --check` cannot see that a SQLite or Postgres leg covers a database path | (not reported: coverage S1 has no such rule yet) | T3.2 (P2-T2.16-3) |

The full run, with the runners (`--python`), rust-host (`--rust`) and the locked gate (`--gate 3a`) on the landed T3.0, runs before the design locks; its GAP lines join T3.2's list, and the tracker records them.

## Owner check-ins

- Kickoff: the questions below, on a readable page, all at once; the two day-1 questions first.
- Twice a day, a five-line check-in with the slip report's line, the register entries reviewers cannot approve, and any overnight decision written plainly when made (KO3-9).
- One design checkpoint when half the planned landings are in (after stack 2), and the whole-phase review at the end, both by both reviewers.
- End: the go or no-go packet (handbook, "Ending a phase", step 6).

## The evidence pack

As Phase 2's, on the final SHA, generated by the branch's own runs, plus:
- the per-case table by database and warehouse leg, by host;
- the captures' index (cases, legs, statement counts, the redaction check), not the captures' data, which stays outside the tree if it holds anything but the public bases;
- E-1 against C-1 per leg; the quickstart notebook's run; the job matrix's two passes, question by question; live L7; the held-out run;
- the switch: the routing defaults, the client suites on the default routes, and the old-route-only list;
- L3 per database, the census with real-data counts on every leg, offline L7, coverage S1 and S2;
- the host-held list before and after (choice 9).

## Gates

### Merge rules

As Phase 2's, with T3.0's mechanisms: packages land in stacks of up to four on one CI run (`land.py --stack`, `stack-land`; KO3-13); judging changes only in T3.2 or a train; two reviewers on every product round, both approving the same SHA (KO2-9), starting only on a SHA with a handoff stamp and reading the stamp and CI rather than probing (KO3-15); on docs-only fix rounds the reviewer who raised the finding confirms its fix, and rounds writing decision rows or register entries keep Claude (KO3-5); a finding family blocking twice gets a half-page rule note, not a redesign session (KO3-3).
Package IDs are integers, T3.0 to T3.9; register IDs come from per-lane blocks (`P3A-` for lane A, `P3B-` for lane B, and so on); each package writes its own tracker file; decisions.md takes only the owner's rows.

### Locked acceptance checks

Locked at T3.1's landing; the S2 case IDs join half `3a` once, in T3.2, from the kinds of dryrun-plan.yaml; any other change is the owner's (plan section 9).
Phase 3's locked set is [locked-set.yaml](locked-set.yaml), one half, `3a`, run as `gate.sh --locked 3a`: today's 309 S0 and S1 acceptance cases (19 of the 24 S0 cases and 290 of the 291 S1 cases; the other six are diagnostic), every S1 case on the databases its file names, steps `inventory`, `seeded` and `main`, checks 1, 2, 7 and 10 held by gate steps, and 12 the every check.
`gate.sh`'s reader reads it (run on this draft: it lists the half's steps and 309 cases).
The rest is judged outside the gate, as Phase 2's KO2-76 set it: CI-green parts by `land.py`'s exact-SHA rule, reports read by both reviewers, and the live and held-out runs once in T3.9.

| # | Check | Tool and machine | Fails on |
|---|---|---|---|
| 1 | Full L2 parity for S0, S1 and S2: every case on each database and warehouse leg it names, per field, on python-core and rust-host against the reference, the warehouse legs on the reference's captured results (H24 by construction); every difference has an approved register entry; coverage S1 and S2 with no empty path or cell | `gate.sh --locked 3a`; CI: `contract-python (<cell>, <database or leg>)`, `cross-host (<database or leg>)`; report: coverage | A changed result on one leg; a case missing on one leg; a statement not in its capture; an empty coverage cell |
| 2 | No unapproved refusal of an input the reference accepts, no panic, on any leg; the census with real-data counts per leg | `ratchet.py` in the gate; report: the census | A refusal with no entry; a kept rule hit on a switched path with no owner answer |
| 3 | L4: SQL goldens executed on DuckDB, SQLite and Postgres; the Databricks and Snowflake goldens checked statement by statement against the captures and executed live once (E-1) | CI: `l4-goldens (<database>)`; report: E-1 | A dialect change with no golden update; a golden the engine refused live |
| 4 | L5: each warehouse driver's Arrow types, as captured, reach the core without loss; the type maps list what each engine cannot store | CI: `l5-conversion*`; report | A unit or scale change in one driver's path |
| 5 | The provider kit: every built-in provider, the three new ones on replay and once live | CI: `rust-test*`, `contract-python (ref-pandas3, <leg>)`; report: the live kit run | A provider that truncates a page or fails a capability silently |
| 6 | L3: about 200 generated sources on each of DuckDB, SQLite and Postgres, and 20 per warehouse if Q12 is yes, every divergence classified | Report, reference machine and E-6 | An unclassified divergence |
| 7 | The reference and the phase branch at `main`'s head as of the gate run, the reference rerun, every changed line traced to a `main` commit | `ratchet.py`, the `main` step | `main` moved and not merged; a changed line with no commit |
| 8 | The switch: every S0, S1 and S2 default is `core`; the client suites pass on the default routes, except tests listed old-route-only under Q6; rust-host runs every S0, S1 and S2 case; the host-held list matches choice 9 | CI: `test (3.x)`, `installed-wheel`, `cross-host (*)`; `rust-lint` (the bridge freeze) | One path left `old`; a client test failing on the default route; a test marked old-route-only that tests public behavior |
| 9 | Held-out (HO3) under H11's threshold on python-core and rust-host, warehouse legs included | `phase_end.sh`'s held-out step and E-5, once | Before the one shot, two dummy bundles (a panic; differences over the threshold) each a no-go |
| 10 | Boundary and packaging: L0 clean, no `pyo3` in the pure-Rust host's tree, dialect words only in `sdm-data`, the wheel installing on x86_64, aarch64 and each runtime cell | CI: `rust-lint`, `rust-test*`, `wheel-smoke (*)`, `runtime-install (*)` | Phase 1's seeded L0 faults; a dialect word in `sdm-schema` |
| 11 | L7: offline within KO2-29's limits on DuckDB, SQLite and Postgres; live push-down median within 10% of the reference on Databricks and Snowflake (Q8) | Report: `contract/tools/l7.py`, E-4 | A scenario past 1.1 times; a live median past 10% |
| 12 | Every gate step the half runs passed | `gate.sh --locked 3a` | Any one step seeded to fail |
| 13 | L6: every case live on each warehouse leg on the core equal to the old client's live answers (E-1 against C-1); the quickstart notebook on the core; the 21-cell job matrix on the core equal to its reference baseline, question by question, on x86_64 and aarch64 cells | Report: E-1, E-2, E-3 | A question answered differently in one cell; a cell that does not install the candidate's wheels |

### Process rules that differ from Phase 2

- T3.0's mechanisms replace Phase 2's rules they encode (the handbook's "The Phase 3 process mechanisms (T3.0)").
- The two live campaigns are the orchestrator's work, like the reference machine's; no product package runs anything live (KO3-22); a capture is evidence for the commit that recorded it and is re-recorded only when the case or its base changes.
- Captures commit only what the public bases hold, with private names replaced; the write fails on a private name (T3.2's exit).
- An agent moves only its own lane branch; T3.0's hook blocks `--force` pushes, so a restack uses `land.py restack`.

## Risks

| Risk | Guard | Lesson |
|---|---|---|
| A dialect fix needs a live run per bug | Capture once, replay offline (choice 2); live tests only at the end (KO3-22) | 6, 14 |
| The end campaign finds real-engine problems all at once | Dialects send the old client's own statements, so the engines have already run them in C-1; any other statement is named and registered before the end; the capture records each driver's types; one fix cycle in the forecast | 6, 12 |
| The core grows driver or pandas behavior to match warehouse drivers | Providers hand Arrow to the core; differences go to the register; L5 from the captured types | 1, 3 |
| Copying the reference's warehouse accidents (sampling text, quoting) | RS-6's question (Q10); registered keeps and drops | 2 |
| A hand-written YAML reader or expression evaluator | A YAML crate; the allow-list ported from `view_expr.py` as a defined algorithm; a dependencies row | 3 |
| A warehouse rule is left in the Python host and Java later lacks it | Choice 9's list; rust-host runs every warehouse leg on the core's captures | 4 |
| Tests pin the mechanism (the replay) instead of the answer | Cases compare answers against the reference; a replay mismatch fails by name, never passes silently | 5 |
| Warehouse keys (`DECIMAL(38,x)`, `BIGINT` against `STRING`) join wrong rows | The key-type pair matrix on each engine before review | 7 |
| A fix makes the reviewer's example pass | `Rule:` lines, failing-first tests, the ratchet over every leg | 8 |
| A refusal of real warehouse data surfaces last | The census on every leg with real-data counts, run beside stacks 2 and 3 | 9 |
| The live benchmark compares nothing | G8 in T3.6; real datasets only | 10 |
| Two packages bump the contract | One bump in T3.3; a later need rides in it or renumbers | 11 |
| The phase overruns silently | The forecast file and the slip report; round 4 to the owner | 12 |
| Judging gaps found one at a time | The dry run before lock; one train; one catch-up at most | 13 |
| CI slows or a landing waits on serial runs | Stacks of four; the timing guard; the tooling split | 14 |
| A capture judged against the wrong commit or a stale base | Captures name their commit; CI's push-time base | 15 |
| Briefs leave the warehouse questions open (session zones, which side owns view YAML) | The package template's headings, refused by `lane.py start --brief` | 16 |
| The owner's existing resources are not found, or a metric view differs from what the cases need | The discovery run before H0; C-1 records the existing metric view's YAML, and a case needing another view creates it in the sandbox schema | 16 |
| Three non-product live failures stop live work | Quota and capacity errors retried once by the runner; owner question G5 | 9 |
| The 118 forced-route failures hide more than three families | T3.7 starts on day 1 with the whole failing list; its exit is 0 failures | 6 |
| Private names or credentials leak into a committed capture or a log | Placeholders written before any file; the write fails on a name from the environment; Snowflake's env file is loaded, never read or printed (KO3-24) | 19 |
| A serving endpoint's cold start slows the matrix | Warm the owner's existing endpoint before each pass | 14 |

## Owner questions

Each opens with what it means in plain words and a recommendation; IDs are footnotes.
Work proceeds as recommended until an answer arrives, except where a question says it blocks.

The live-access questions are answered (KO3-18 to KO3-26): the Databricks profile, the sandbox catalog, schema and volume the owner used before, a SQL warehouse, the existing metric views and serving endpoints, live testing at the end, Phase 2's datasets replicated from the quickstart notebook onwards, Snowflake through the owner's env file, a Lakebase instance in the same workspace, and both x86_64 and aarch64.
Their former numbers Q1 to Q4 are retired; the rest keep their numbers.
None of the questions below blocks day 1.

- **Q5. The switch.**
  Flip every S0, S1 and S2 path to the core at once, as the plan says?
  And if one of the three refusals you kept until Phase 4 (a tabular model request, text times read by the client's own rules, `LIST_DISTINCT` over a foreign key) can be reached from a switched path, it would refuse an input the old client answers.
  Recommendation: flip all at once; keep any such input refused by name on the epic branch until Phase 4 lifts it, since the epic does not reach users before then; T3.2's census says whether any is reachable at all.
- **Q6. Old-engine tests after the flip.**
  Some client tests check the old engine's internals rather than what a user sees, and cannot pass on the core.
  Recommendation: mark those to run on the old route only until Phase 5 deletes the old engine, each one listed in the evidence; every public-behavior test must pass on the core.
- **Q7. What the job matrix asks.**
  You said to start from today's quickstart notebook, which is not enough for testing (KO3-23); it asks 5 questions (no regression, ranking or forecast) and reads only through Spark, while the matrix's 21 cells were defined on the deleted notebook's 8 questions over a SQL warehouse, Spark and Lakebase.
  Recommendation: each of the 21 cells runs a one-time job template asking the quickstart's 5 questions plus the deleted notebook's 8 (with its ranking and forecast) on the cell's route, and the quickstart notebook itself runs once on the core as the AdventureWorks notebook check.
- **Q8. Live timing at the switch.**
  The plan asks for live push-down timing at the switch; it costs about 2 hours of runs on two warehouses.
  Recommendation: keep it at the switch, as written.
- **Q9. How view YAML values are read.**
  The old client reads view YAML with PyYAML, which reads `yes`, `on` and `off` as true and false (YAML 1.1); a strict YAML 1.2 reader keeps them as text.
  Recommendation: read them as the old client does, so no answer changes.
- **Q10. Warehouse sampling.**
  RS-6 approved sending the old client's sampling SQL on Postgres; Phase 2 left open whether it covers the warehouses.
  Recommendation: yes: the core sends the old client's sampling statements on Databricks and Snowflake, the percentage written as Python writes it, so the same rows are drawn.
- **Q11. The warehouse comparison on the old client's own data.**
  Replaying the old client's captured results into the core compares both on exactly the same rows, which is what H24 asked for on live warehouses.
  Recommendation: it gates (check 1), since it runs offline in minutes.
- **Q12. L3 on the warehouses.**
  The random-input comparison runs on DuckDB, SQLite and Postgres; on a warehouse it needs a live job.
  Recommendation: 20 generated sources per warehouse in one job each, beside the matrix.
- **Q13. Three refusals decided overnight in Phase 2** (a NaN `lag_timesteps`, `num_neighbors` text that is not a number, a negative `num_neighbors`) refuse inputs the old client answers.
  Recommendation: match the old client (lift all three) in T3.8.
- **Q14. Branch protection on `rust-core`.**
  Today any push could change it; only your merges should.
  Recommendation: protect it so only you can push.
- **Q15. A bigger build machine.**
  The machine allows four packages building at once; a fifth would save about 2 hours.
  Recommendation: yes if one is available; the plan does not depend on it.

**Process questions:**

- **G1. The standing "match" class.**
  You approved (KO3-7) that statement counts, warning repeat counts, tie order among equal scores and hash order are matched only when a user can see them, otherwise one register class you approve once.
  Recommendation: T3.2 writes that one entry; approve its text when it lands.
- **G2. One severity bar for both reviewers** (you asked "Has this really been a problem?").
  Yes: on T2.14b, Claude rated rare mismatches with the old client as minor three rounds running while Codex blocked on them, and Codex approved decision and register wording Claude then found wrong (three packages and the design); each disagreement cost a round.
  Recommendation: one bar: any measured difference from the old client on a public call, any red CI job the change caused, and any decision row wider than your words blocks; hardening internal tools stays minor (KO2-52).
- **G3. When the Java track runs.**
  Java is deferred until S2 is switched (KO2-34), which is this phase's exit.
  Recommendation: as its own track right after Phase 3, about 12 to 15 packages.
- **G4. The design's "no" list.**
  About 45 carried checking-tool and test rows are closed as "no" unless a Phase 3 case or the dry run hits them (the carried-rows table).
  Recommendation: accept the list; any row hit later goes into T3.2 or the train.
- **G5. Live failures for non-product reasons.**
  Three such failures stop live work (plan section 9); quota and capacity errors are common on serverless and are not about the product.
  Recommendation: the runner retries a quota or capacity error once, and only a failure after the retry counts.
