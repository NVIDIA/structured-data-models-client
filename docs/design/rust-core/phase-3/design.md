# Phase 3: S2 live, Databricks first, the Python switch, and Java's start

## Status

Draft for review round 4 (T3.1), written on 2026-10-22 from `origin/rust-core-phase-3` at 3ae4d1af (Phase 2 merged), against T3.0's process mechanisms as they stand at 503740ab on T3.0's topic branch.
Review rounds 1 to 3's findings, the owner's Java answer, and the owner's answers to the Java questions and to the design questions are applied ([tracker.md](tracker.md), "Design review rounds").
The owner's answers so far are in [decisions.md](decisions.md), KO3-1 to KO3-58; none is open ("Owner questions", at the end of this page).
The acceptance checks below lock at T3.1's landing, after both reviewers approve one SHA.
T3.1 lands stacked on T3.0 (T3.0's "Stacked landings"), so everything this page names from T3.0 exists when T3.1 lands.
Everything not stated here is as in [Phase 2's design](../phase-2/design.md) and [Phase 1's design](../phase-1/design.md); their decision rows stay in force unless a row here or an owner answer changes them.
Choices marked "orchestrator's decision" are the orchestrator's, not the owner's; the owner sees them at the next check-in.
Lesson numbers are those of [learnings.md](../learnings.md) as T3.1 rewrites it.

## In one page

Phase 3 makes the Rust core answer on the three cloud warehouses the old client supports, then makes the core the default for every user call it covers, and starts Java on the same core.
Today the core answers on DuckDB, SQLite and Postgres, but only when a test forces it; real users still run the old Python code on every path.

What gets built:
1. Databricks first: a Databricks SQL dialect in the core, a Python provider for SQL warehouses and one for a notebook's Spark session.
2. Databricks metric views and Snowflake semantic views, read from their YAML definitions by the core.
3. Snowflake, with its own dialect and provider, and Lakebase (Databricks' managed Postgres), which reuses Phase 2's Postgres dialect.
4. Ranking and forecasting on all of them, the same five question kinds as offline.
5. The switch: every S0 (query validation), S1 (offline databases) and S2 (warehouse) path runs on the core by default, at once, after the 118 old-client tests that fail on the core today are fixed.
6. Java (the owner, KO3-27): a Java library on the same core, answering query validation, then DuckDB, SQLite and Postgres through JDBC, then Databricks through JDBC, on the same test cases as Python; it never holds up the Python switch.

How the warehouse work is tested without re-testing on slow warehouses after every change (the owner, KO3-22):
- One short live run at the start records, for every test case on each warehouse, the old client's answers and the exact tables the warehouse stored and returned.
- All coding is then judged offline: the core runs each warehouse case against a local database loaded with those recorded tables, and its answers must equal the old client's recorded answers.
- One live campaign at the end runs the core for real on the commit that lands: every warehouse case, the 21-job Databricks matrix, both Databricks notebooks, live timing and the held-out set.

The early run is still needed because the offline judge has nothing to compare against without it: the old client's warehouse answers, the warehouses' own column types and catalog output, and the rows a warehouse's random sampling picked exist only on the live engines.

The plan is 14 landings: T3.0 and this design, nine Python packages, three Java packages; up to four Python packages build at once, as the machine allows (KO3-16), with Java on a fifth lane of its own (KO3-28).
The forecast from this design's landing is about 41 hours to the Python switch, which is also the phase's end: Java's packages land before it, and Java's live check runs in the same end campaign.
It assumes one fix cycle in the live end campaign, without which the switch lands at about H33.
The slip report compares the phase against the forecast as it runs and never stops work (KO3-12).
The owner approved the offline judge and answered the design's other questions (KO3-36 to KO3-49), including the rule for differences: the core matches the old client or does better, where better means a better error or more support, and only a worse difference blocks.
No owner question is open, and nothing blocks day 1: the live resources the early run needs are ready (KO3-50 to KO3-53): the Lakebase instance is running, one dedicated test schema exists on Databricks and one on Snowflake, a write guard lets the tests write only there, and a warm serving endpoint for model version 26 is up.

## Goal

Ship slice S2 for Python: `Schema.from_source` → `materialize` → `predict` live on Databricks (plain tables over a SQL warehouse and over Spark, and a metric view), Snowflake (plain tables and a semantic view) and Lakebase, including ranking and forecasting, with the Databricks job matrix across runtimes (migration-plan.md section 4, S2).
Then switch Python's S0, S1 and S2 paths to the core together, on the epic branch (plan section 4: "S0 (PQL validation), S1 and S2 paths switch together after S2, in Phase 3, so no call mixes old and new code").
Exit: the switch gate (plan section 7, "Python switch for a path"): S2's slice done, L6 green for its scenarios, the Databricks job matrix equal to the baseline, L7 within its limits with live timing; checked by the locked acceptance checks below.

Start Java (KO3-27; KO2-34 had deferred it): a Java library on the same core passing the same contract cases as Python on Java's scope (`phase-0/scope-java.yaml`, 66 paths: S0, S1 on DuckDB, SQLite and Postgres, and S2 on a Databricks SQL warehouse and a metric view, with binary, multiclass and regression questions), and agreeing with Python on the same Databricks data (plan section 4, S2; Phase 0 decision J2).
Java follows Python and never holds it back (plan section 2, decision 1): Java's own checks block only Java's packages, while a shared-core defect a Java leg shows blocks the path it breaks (plan section 6).

## Order of work

1. **Before the clock starts** (no product code):
   - T3.0 lands, and this design is reviewed and locked; T3.1 lands stacked on T3.0;
   - the full dry run of the checking tools over [dryrun-plan.yaml](dryrun-plan.yaml) runs on T3.0's tree, with the runners, rust-host and the locked gate, and its gaps join T3.2's list;
   - the live resources ready (done: KO3-50 to KO3-53);
   - one brief per package from [briefs/package.md](../orchestration/briefs/package.md), each passing `scripts/rust_core/brief_check.py`;
   - every warehouse test author (T3.4 to T3.7, and T3.13 for Java's Databricks cases) writes the input files of its S2 cases, so that every case kind in dryrun-plan.yaml has its cases before the early capture; S1's cases exist already.
2. **H0, four Python packages and the Java lane at once**: T3.2 (the tooling train), T3.3 (the warehouse harness), T3.4 (Databricks) and T3.8 (the forced-route failures), and T3.11 (the Java binding) on its own lane (KO3-28).
   T3.2 and T3.3 write the recording mode and the warehouse loader first; a review round by both reviewers on those commits approves them before the capture uses them (about H4.5).
3. **C-1, the early capture** (about H5 to H8), from that reviewed SHA: the one live run before the end; after it, no product package runs anything live (KO3-22).
4. **Stack 1** (T3.2 at the front, T3.3, T3.8) lands on one CI run at about H12.
   As slots free, T3.9 (differences and coverage) starts at about H6, T3.6 (Snowflake, landing after the views it needs) at H7, T3.7 (Lakebase, live runners, reports) at H8, and T3.5 (views) at H10, on T3.4's candidate.
5. **Stack 2** (T3.4, T3.7, T3.9) at about H20; **stack 3** (T3.5, then T3.6, whose semantic-view cases need T3.5, and Java's T3.11 and T3.12) at about H25; T3.6's review runs on a SHA stacked on T3.5's approved candidate, so the SHA approved is the one that lands.
6. **Java** on its lane: T3.11 (binding, API, JAR, Java S0) from H0, T3.12 (Java S1 through JDBC) from about H9, T3.13 (Java on Databricks) from about H18, landing alone at about H29.
7. **T3.10**, the switch and the switch gate: the flip, the whole-phase review beside a first run of the live end campaign, the fixes, then the end campaign on the final candidate, Java's live check (E-7) included; it lands at about H41 and ends the phase.
   The end-of-phase reports for S0 and S1 (L3, the census, offline L7) run as soon as stack 2 has landed.
   Java never holds the switch: if T3.13 has not landed when the final end campaign starts, T3.10 lands without it, and T3.13 lands after with E-7 run on its own landing commit.

## Scope

1. **Databricks** (E27, E31: first): the `databricks` dialect in `sdm-data` for every request kind (describe, preview, scan, counts, key counts, time bounds, column ranges, repeated keys, entity sampling, neighbors, lookup), its catalog reader into the neutral `CatalogAnswer`, key sets bound as the old client binds them (`posexplode(from_json(?, ...))`), sampling as `TABLESAMPLE (n PERCENT) REPEATABLE (seed)` with the percentage printed as Python prints it (KO3-41); the SQL-warehouse provider (`fetchall_arrow`) and the Spark provider (`spark.sql(...).toArrow()`, DBR 17.3 or later and serverless environment 4 or later, as the old client requires); the routes for every `databricks-sql` and `databricks-spark` path.
2. **Views**: a Databricks metric view and a Snowflake semantic view each described through its YAML definition into a view model the core reads without dialect terms (Phase 2's choice 12), with renamed and expression columns and a row filter carried on the relation, measures and metrics left out as conversion notes, as the old client does; the expression allow-list of the old client's `view_expr.py`, with `trust_expressions`.
3. **Snowflake**: the `snowflake` dialect (double-quoted names with the old client's upper-case rule, `NUMBER(p,s)`, keys bound as `TABLE(FLATTEN(PARSE_JSON(?)))`, `QUALIFY`, `::TIMESTAMP_NTZ`), its catalog reader (`DESCRIBE TABLE`, `SHOW IMPORTED KEYS`, `SHOW TABLES`), the semantic view's retrieval, and the provider (`fetch_arrow_all`, on a connector connection or a Snowpark session's connection).
4. **Lakebase**: a `PostgresSource` over a Lakebase connection on Phase 2's Postgres dialect, the `sdm_connectors.databricks.lakebase()` connect path, the `lakebase` routes, and Postgres reads qualified by the relation's schema part (P2-MID2-6).
5. **Ranking and forecasting on every warehouse leg**, through the decomposition Phase 2 moved into the core; nothing new in the core's operations.
6. **The contract bump, v28, once** (lesson 11), in T3.4 (choice 3).
7. **The job matrix**: the 21 cells of `phase-0/databricks-jobs.md` (J01 to J21: SQL warehouse, Spark and Lakebase; DBR 14.3 to 17.3 and serverless 1 to 6; x86_64 and aarch64, both in scope, KO3-26), each a one-time job installing the wheels under test (KO2-95's rule: the reference's for the baseline pass, the candidate's for the core pass), compared question by question against the model-version-26 serving endpoint (KO3-53) and the owner's existing metric views (KO3-21).
   CI's `native-wheel` already builds the aarch64 wheel (KO2-40). On Databricks a cell's architecture comes from its node type (Graviton for aarch64), not from a separate runtime string, so the aarch64 cells (J02, J04, J06 and J11) keep their runtimes (14.3, 15.4, 16.4, 17.3) on Graviton nodes; C-1 runs one small probe job per such runtime on a Graviton node, and a cell whose probe fails moves to the newest runtime line on Graviton, recorded in the evidence and listed for the owner (KO3-26: "both x86 and aarch64").
   The DBR 14.3 install revalidation Phase 1 carried (T1.35, E154) is cells J01 and J02.
8. **Both Databricks notebooks on the core**: `kumo_relational_quickstart.ipynb` (the AdventureWorks notebook of E27 and E31, which replaced the deleted one) and `lakebase_relational.ipynb`, once each in the end campaign.
9. **Java** (KO3-27), choice 10: the `sdm-jni` binding; a JAR built with Maven, with natives for Linux x86_64 and aarch64 and macOS arm64 and x86_64 (Phase 0 decision G13, KO3-32); Java 17 bytecode, tested on JDK 17 and 21 (KO3-34); a Java runner for the contract cases; Java S0; Java S1 on DuckDB, SQLite and Postgres through JDBC with the JDBC-to-Arrow conversion table and L5 for Java; Java on a Databricks SQL warehouse and a metric view through the Databricks JDBC driver, answering Phase 0's J2 questions live at the end against the same data as Python.
10. **The switch**: every S0, S1 and S2 default in `sdm_client/_routing.py` set to `core` in one commit (choice 8), the routing guard test changed with it, and the existing client test suites green on the default routes.
    `SDM_CORE_ROUTES` stays, test-only, until Phase 5 deletes the old engine.
11. **What the switch needs from Phases 1 and 2** (the carried rows, next section): the 118 forced-route failures, S1 coverage, the 12 unregistered user-visible differences, L3 and H24 on every database, DuckDB 2.0.
12. **The gate's live parts**: L6 on every warehouse leg, the job matrix, both notebooks, L7 live push-down timing (median of at least five interleaved runs, caches off, within 10% on Databricks and Snowflake), run in the end campaign (KO3-39), and the held-out set's warehouse legs.
13. **The phase-end chores**: `main` merged and the reference rerun if `main` has moved (it has not since 43e55f4f), with the warehouse reference lines re-recorded in the end campaign if it has (choice 2), register reconciliation, the evidence pack, the whole-phase review (handbook, "Ending a phase").

### Explicitly out

- Java beyond Phase 0's scope: Snowflake, Lakebase and Spark in Java, frames, ranking and forecasting in Java (J2), and publishing the JAR to a repository (Phase 5's release work; KO3-31). The Java host rows Phase 2 carried for ranking (P2-WR-2, P2-WR-5, P2-T2.14b-4) wait until the owner adds ranking to Java's scope.
- S3: frames, files read from S3, link prediction, sessions, tabular requests, and the three refusals kept until Phase 4 (KO2-94); the fallback rows P2-T2.5-1 to -3 and FX-3 (Phase 4, placed there by owner rows).
- Release work (Phase 5): `publish.yml`, the SBOM, macOS, Windows and musl Python wheels, signing; ASan and LSan (KO2-25).
- `MEASURE()` and `SEMANTIC_VIEW()` queries: the old client never issues them (it reads a view's base tables), so the core does not either.

## The carried rows

Migration-plan.md section 8, "Carried into Phase 3", holds 142 rows naming 145 items.
Every item ends in a package or a recorded "no" (lesson 18); a script compared the set of item IDs in the plan's table with the IDs below, and every ID is placed exactly once.
The row text stays in the plan, and the package's tracker closes it.
"No" items are accepted by the owner (KO3-48), under KO2-52 (internal-tool hardening is P3 at most) and the owner's "product over gate machinery": they are reopened only when the dry run or a Phase 3 case hits them, and then go into T3.2 or the train.
Since T3.0's fix round, `scripts/rust_core/judging-paths.txt` lists only the gate, the ratchet, the comparator, coverage, the register check, `phases.py`, the L0 guards, the census, `land.py`, `lane.py`, `judging.py`, `run_python.py`, `capture_check.py`, the schemas and `sdm-testkit`'s `case.rs` and `observation.rs`; so `ci.yml`, the bridge declarations, the other runners and tools, vectors and fixtures ride with the package that needs them, and only those paths go into T3.2.

| What | Items | Package |
|---|---|---|
| **The 118 forced-route failures** (Phase 2's flip dry run: 2,789 client tests with every core route forced; 61 miss the "Some time columns..." warning, 38 count keys otherwise or refuse `INVALID_REQUEST`, 19 refuse on Postgres or raise `RouteError` where materialize needs a schema described on the core) | P2-T2.16-9, P2-T2.16-10, P2-T2.16-11 | T3.8 |
| **S0 fully on the core**: the host still picks the old validator for a query | P2-MID2-1 | T3.8 |
| **The 11 other unregistered user-visible differences** (with P2-MID2-1, the 12): `task=` wording and order; `datetime` as `context_anchor_time`; NaN `max_pq_iterations`; Postgres decimal inference through `ToDouble`; unsigned 64-bit `IN`; fractional `IN` against a scale-0 decimal; the fractional-iterations statement count; the generic composite-key message; `rows_per_call` naming the value; answers parsed twice; `RecursionError` on a deep answer | P2-WR-3, P2-WR-4, P2-WR-7, P2-T2.14b-1, P2-T2.10-1, P2-T2.1p-1, P2-T2.11-1, P2-T2.12-1, P2-T2.8c-1, P2-T2.14b-2, P2-T2.14b-3 | T3.9: each matched to the reference, or registered and sent to the owner |
| **S1 coverage**: 58 of 75 S0 and S1 paths and 22 of 44 cells with no acceptance case | P2-T2.16-3, G12, G16, G17 | T3.2 (coverage: a case's database and warehouse legs claim their own paths; a named client test may claim a path the case format cannot express; the S1, S2 and Java rules); T3.9 (the cases for what is still empty) |
| The S1 test rows that close coverage gaps | P2-T2.19-1, P2-T2.1b-2, P2-T2.1b-3, P2-T2.1b-4, P2-T2.5-4 | T3.9 |
| The two case loaders disagreeing on `fail_call: -0` and non-finite `rows_per_call` | P2-T2.1i-1 | T3.2 (both loaders and `contract/tests/malformed-cases.json`) |
| **L3 on SQLite and Postgres**, the two classes it cannot name, the stale class list, the `_by_time` seeds | P2-T2.16-4, P2-T2.16-5, P2-T2.11-2, M8, P2-T2.11-3 | T3.7 (`contract/tools/differential.py`) |
| **H24 locally**, the per-case table with database legs | P2-T2.16-7, P2-T2.16-6 | T3.7 |
| P2G-1 widened to every call the endpoint refuses | P2-T2.16-13 | T3.9 |
| **DuckDB 2.0** support and the `zip` advisory exceptions | P2-T2.16-12, OF-2 | T3.9 |
| The interim edit gate's retirement | P2-T2.16-8 | T3.9 |
| The overnight decisions: three refusals of input the reference answers, lifted (P2B-19, P2I-1, P2I-4; KO3-44); "Postgres key equality" (P2G-4, P2G-5) and "the data time-range wording in errors" (KO2-89), confirmed (KO3-57); P2D-2 to P2D-5, still under the overnight delegation and on the owner's morning list | OF-5 | T3.9 (and the owner's morning list) |
| **Contract rows for v28**, and the interfaces a Java host sees | P2-MID2-2, P2-T2.2-1, P2-T2.1b-1, P2-T2.8c-2, P2-MID2-3, P2-MID2-5, FX-6, P2-MID2-4, P2-WR-6 | T3.4 |
| Postgres reads qualified by the relation's schema part | P2-MID2-6 | T3.7 |
| Expression columns and metric views (KO2-25, KO2-39) | OF-4 | T3.4 (the contract fields), T3.5 (the views) |
| RS-6 on the warehouses | OF-3 | T3.4 and T3.6, under KO3-41 |
| The live benchmark's "different requests" result fails the run (a Phase 1 row the table does not list) | G8 | T3.7 |
| The held-out set's stored-NaN and `key_name_order` cases | OF-6 | HO3 |
| A bigger machine | OF-9 | Closed: no (KO3-55) |
| Branch protection on `rust-core`; `ci.yml`'s phase predicate | OF-7 | T3.0 (KO3-56: the push refusal goes into T3.0's command hook and pre-push hook in its next fix round; the design cites T3.0's landing SHA when it lands); T3.3 (the predicate) |
| Judging rows Phase 3's work hits: owner approvals read by allowlist; the statement log on psycopg copy and stream and the warehouse drivers; a stale installed wheel; a stale `origin/main`; result rows hashed as text so a case's predictions are compared again | U4, P2-T2.1a-2, P2-MID2-7, P2-T2.0f-1, P2-T2.21-1 | T3.2 |
| Refusal maps citing quoted text, not line numbers (they broke four Phase 2 landings) | P2-T2.1b-6 | T3.7 |
| Time zones set per session and process | C16, C18, M5 | T3.3 (`ci.yml`, the vectors); C18's static check in T3.2's tests |
| CI rows: the Postgres numeric-spelling cases in CI and `SDM_REQUIRE_POSTGRES` on every Postgres step; `pytest-timeout` pinned; a forced rebuild; a stale comment; the skip step's `jq`; the slowest job's time if T3.0's split has not fixed it | P2-T2.14c-1, P2-T2.1a-3, P2-T2.16a-1, P2-T2.11-4, C28, P2-T2.16-14, P2-T2.1o-1 | T3.3 |
| Checked once in a lane environment, closed if they pass | ENV-2, ENV-3 | T3.3 |
| The test naming half `3a` as a half with no locked set | P2-T2.0a-1 | T3.0 (done in its fix round: half `99a`) |
| Docs tidy: Phase 2's trackers, as-built and decision rows | P2-T2.16-1, P2-T2.16-2, P2-T2.1p-2 | T3.1 |
| Register text tidy | P2-T2.0c-3, P2-T2.8c-3, P2-T2.13b-1, P2-T2.21-2, P2-T2.14c-2 | T3.9 |
| Core hygiene, done by the package that next touches the file | U12, P2-T2.2-3 (`sdm-data`: T3.4); P2-WR-1, P2-WR-8, P2-WR-9, P2-T2.2-2, P2-T2.0c-1 (T3.8 or T3.9, whichever touches the file); P2-T2.7-1, P2-T2.7-2 (`sdm-testkit`: T3.3) | as listed |
| Covered by T3.0 | OF-1, P2-T2.1n-1 (the tests' part), ENV-1, P2-T2.0b-3, P2-T2.0g-1 (the reviewer's part) | T3.0 |
| **No**, until the dry run or a case hits them | FR P3-7, U3, U13, C2, C4, C5, C6, C7, C8, C9, C11, C22, G2, G4, G10, FX-2, M9, P2-WR-10, P2-WR-11, P2-T2.0-1, P2-T2.0b-1, P2-T2.0b-2, P2-T2.0b-4, P2-T2.0c-2, P2-T2.0e-1, P2-T2.0h-1, P2-T2.1a-1, P2-T2.1b-5, P2-T2.1b-7, P2-T2.1d-1, P2-T2.1e-1, P2-T2.1f-1, P2-T2.1g-1, P2-T2.1g-2, P2-T2.1h-1, P2-T2.1j-1, P2-T2.1j-2, P2-T2.1k-1, P2-T2.1k-2, P2-T2.1l-1, P2-T2.1m-1, P2-T2.4-1, P2-T2.7-3 | none |
| Phase 4, or Java's ranking once the owner adds it (placed there by owner rows or their reviewers) | P2-T2.5-1, P2-T2.5-2, P2-T2.5-3, FX-3, OF-8; P2-WR-2, P2-WR-5, P2-T2.14b-4 | none in Phase 3 |

L3 and H24 on all databases: L3 (`contract/tools/differential.py`) runs about 200 generated sources on each of DuckDB, SQLite and Postgres (KO2-26) and 20 on each of Databricks and Snowflake in the end campaign (KO3-43); H24, the core fed the reference's own fetched data, runs on Postgres locally (P2-T2.16-7) and on every warehouse leg by construction, since the offline judge serves the reference's recorded tables and random draws (choice 2).

## Architecture choices new in this phase

Phase 1's and Phase 2's mechanisms are reused as built: the contract and its lock, the data port and capabilities, `Relation`, runners and cells, comparator, ratchet and checkpoint, register, coverage, gate, the provider kit, the bridge freeze, the routing seam, the evidence pack; and T3.0's stacks, tooling train, handoff stamp, CI split and timing guard, lane environments, command hook, dry run and slip report.
Every new interface is held to KO2-46: only what this phase uses, as plain data a Python or Java host builds without language idioms.

### 1. Warehouse legs are database legs of the existing case format

**Recommendation:** add four warehouse legs, `databricks-sql`, `databricks-spark`, `snowflake` and `lakebase`, keyed `<case>@<leg>` exactly as Phase 2 keyed `<case>@sqlite`.
A case's data is still its base and setup (DuckDB statements), loaded into a dedicated test schema of each warehouse through Arrow by Phase 2's contract loader (`contract/runners/databases.py`), extended with a type map per engine: Databricks and Snowflake have no unsigned integers, no time-of-day type in tables, `DECIMAL` up to 38 digits and their own timestamp kinds; a type an engine cannot store is listed, and its cell is never counted as covered (KO2-47's rule).
Every S1 acceptance case runs on every warehouse leg whose engine stores its data, by the same default that runs it on DuckDB, SQLite and Postgres, so no S1 case file changes (the owner, KO3-23: "Can we replicate whatever dataset/testing you've done in phase 2 here?").
The two cases that build 100,001-row tables to test the core's call batching (`multicall.entities.100000` and `.100001`) are left off the warehouse legs, which add nothing to what they test; that is an exclusion, which the owner approved with the offline judge (KO3-45).
An S2 case names its warehouse legs in a new `warehouses` field and runs only there, with no exclusion; the metric and semantic view cases carry a new `input.view` (the kind, the name and the YAML, or the name of an existing view), and the connect paths `input.connect` (`warehouse`, `spark`, `snowpark`, `lakebase`).
AdventureWorks is loaded the same way, and its cases ask the quickstart notebook's 5 questions (3 binary, 1 multiclass, 1 binary through a metric view), the deleted notebook's 8 (regression twice, forecast, binary, ranking, multiclass, and two through a metric view), and a ranking and a forecast on each leg, since the quickstart alone "is not enough for testing" (KO3-23).
Why: every S1 case already has reference lines and core observations on three databases, so a warehouse leg reuses the comparator, ratchet and coverage, and one case set covers offline and live.
Rejected: separate S2 case files over a separate live dataset (a second corpus to keep in step); live-only cases (a live run per bug).

### 2. The offline judge: the old client's answers, a hybrid stand-in, and a live-only list

Round 1 found that the first draft's judge, answering the core's SQL from a capture by matching statement text, could not judge a new dialect before the end, and made SQL text a gate where Phase 1's H9 (owner-approved) keeps it diagnostic.
Round 2 found that a local engine cannot run a view's own warehouse SQL (`IFF(...)`, `named_struct(...)`, backtick names), which the old client pastes verbatim into its queries.
**Recommendation:** judge a warehouse leg offline by its answers against the old client's recorded answers, serving the core's data requests with a hybrid stand-in, and name what neither part can judge as "live only", judged in the end campaign:

- **What C-1 records**, per case and leg, from the old client running live: its public outcome as a reference line (schema, payload, predictions, warnings, refusal), in the observation format Phase 2's lines use; every table of the case as the warehouse returns it through the leg's driver (Arrow, with the warehouse's own column types); for each view, one table per relation the old client derives from it (the fact table and each joined table, each with the expression columns the warehouse accepted and the filter on the table it applies to), evaluated by the warehouse; a column the warehouse refused, which the old client drops after retrying column by column, is served only by exact replay; every random draw the old client's sampler made (the table, the seed, the fraction as written, and the rows it picked); and every statement it sent with its parameters and result.
  Names of catalogs, schemas, hosts and users are replaced by placeholders before anything is written, and the write fails on any private name from the environment.
  A capture file carries a format version (`sdm-capture/1`), the recording commit, the reference commit, each leg's runtime, Python, pandas and driver versions, and a hash of each case's input, so `capture_check.py` finds a case changed after C-1 mechanically.
  Captures live under `contract/captures/`, a protected path (T3.2 adds the pattern to `contract/protected-paths.txt`), so a change to one passes only in a landing whose `Protected-paths` trailer names it; T3.3 commits C-1's captures, and T3.10 commits E-0's and the core's live capture.
- **The stand-in provider** answers the core's data requests for a warehouse leg offline, as Phase 2's replay provider answers Postgres for rust-host (KO2-60): it receives each request with the plan the leg's own dialect rendered for it, and answers in three parts, in this order:
  1. **Exact replay.** When the plan's statement and its parameters equal, after normalising placeholders and whitespace, a statement the old client sent in C-1 for the same case and leg, it answers with that statement's recorded result. This covers what the core is designed to copy from the old client verbatim: the catalog statements (so each dialect's catalog reader runs offline on the warehouse's real output), the view retrieval statements, and the reads that paste a view's expressions and filter.
  2. **Local engine.** Otherwise it renders the request through the DuckDB dialect and runs it on a local DuckDB loaded with the leg's recorded tables (each derived relation of a view served from its warehouse-evaluated table when the relation's columns and filter are the recorded view's), with DuckDB's session set to the warehouse's NULL order and to the capture's time zone; for `lakebase` the local engine is the local Postgres 17 CI already runs, so the core's real Postgres SQL runs offline.
     An entity-sampling request is answered with the old client's recorded draw only when its table, seed and fraction text equal the draw's, and refused by name otherwise, so a wrong percentage text or seed fails offline.
  3. **Live only.** Where neither part can answer, the stand-in returns `LIVE_ONLY` with a reason code, each a row of the blind-spot table below (B1 to B4): a statement the old client never sent over SQL the local engine cannot run (B1), a derived relation holding a column the warehouse refused (B2), keys of different recorded types compared in one request (B4). Only the stand-in assigns the state; no package may declare a case live-only by its own judgement, and a difference the stand-in did not flag fails as any difference does.
  **Where it lives.** One Rust module in `sdm-testkit` (`src/standin.rs`, with its capture reader), within `sdm-testkit`'s allowed edges (`sdm-host`, `sdm-contract`) and its dev-only DuckDB, which rust-host calls in process, and a small `sdm-testkit` binary, `sdm-standin`, that serves it over a pipe (Arrow IPC with JSON control messages) to the Python and Java test harnesses, each of which plugs a thin forwarding provider into its host (`contract/runners/standin_client.py`; the Java runner's, T3.13).
  So no shipped crate reaches `sdm-testkit` or DuckDB, and `l0_deps.py` at 503740ab (bindings only over `sdm-host` and `sdm-contract`; nothing shipped reaching `sdm-testkit` or a database driver) holds unchanged: no L0 rule changes for the stand-in.
  Both `rust/crates/sdm-testkit/**` and `contract/runners/**` are protected paths already, so the judge changes only in a landing that names it in its trailer: T3.3 writes it, and T3.4 extends it for v28's relation fields under that trailer.
  **The `live_only` state in the tools (T3.2).** The comparator, the ratchet and the gate learn a `live_only` outcome for a case and leg: admitted only when the stand-in returned `LIVE_ONLY` with a reason code in the blind-spot table, which `compare.py` checks against the list in `contract/compare-policy.yaml`; neither passed nor failed offline; listed in the package's exit, counted by the tracker and by `handoff.py`, which stamps a SHA with live-only items but names them.
  It is discharged only by E-1 (E-7 for Java) on the final candidate: the core's live outcome against the old client's live answers, recorded with the core's capture, after which the stand-in answers that case by exact replay of the core's own committed statements, so CI keeps judging it offline; the locked run fails while any `live_only` item has no E-1 verdict for the final candidate.
- **What gates offline**: the core's public outcome on the leg against C-1's reference line, field by field, under the same comparator and register as every other leg; a live-only case is pending, never passed.
  On a stand-in leg the statement count and each statement's result are diagnostic, since most statements are the stand-in's; a different number of random draws, or a draw with another table, seed or fraction, fails by name.
  That narrows H9's statement rule on stand-in legs only, which the owner approved (KO3-45); H9 applies in full on every live leg.
- **The dialect's SQL** is judged by goldens per request kind, reviewed against the old client's recorded statement for the same request (a non-gating diff in the package's evidence), by exact replay wherever the core copies the old client's statement, and executed live in the end campaign (E-1), where H9 applies in full.
- **The stand-in's blind spots, each covered live**:

| Row | Not judged offline | Why | Covered by |
|---|---|---|---|
| B1 | The warehouse accepting a statement the old client never sent, over SQL the local engine cannot run | Neither replay nor the local engine runs it; the stand-in returns `LIVE_ONLY` (B1) | E-1, every case on every leg, H9 in full |
| B2 | A view relation with a column the warehouse refused in C-1 | No warehouse-evaluated table holds it; `LIVE_ONLY` (B2) unless exact replay serves it | E-1 |
| B3 | The three providers' fetch code (`fetchall_arrow`, `toArrow`, `fetch_arrow_all`) on a live driver; computed results' types (an aggregate's type is DuckDB's) | The stand-in replaces the provider and computes locally | Offline, each provider's own code over a fake cursor returning C-1's recorded Arrow (T3.4, T3.6); live, the provider kit and every case on each warehouse in E-1 |
| B4 | A `BIGINT` key compared with a `STRING` key, and other cross-type comparisons | Engine semantics; the stand-in returns `LIVE_ONLY` (B4) when one request compares keys of different recorded types | E-1, with the key-type pair cases on each leg |
| B5 | NULL order, identifier case folding, the session time zone | Engine semantics | Set offline (NULL order and time zone from the capture; recorded names as the warehouse returned them); E-1 confirms |
| B6 | Repeatable sampling (`TABLESAMPLE ... REPEATABLE`) | Only the warehouse draws | The recorded draws offline, checked by table, seed and fraction; E-1 live |
| B7 | The Databricks JDBC driver's types for Java | No JDBC capture | E-7, every Java S2 case live |

Rejected: matching every core statement to recorded ones (it cannot answer a statement the old client never sent, and the inherited Postgres dialect already sends one, `rust/crates/sdm-data/src/postgres.rs`'s relation-identity predicate); a warehouse emulator by transpiled SQL (it judges SQL nobody runs); a live check per dialect mid-phase (cheap, but against KO3-22; the owner chose otherwise, KO3-45).

**When the capture runs and what it covers.**
- C-1 runs once, from a SHA both reviewers approved in a review round on T3.2's and T3.3's recording and loader commits, so a finding on the recording format cannot invalidate the one capture every dialect package works from; a later format change converts captures offline with T3.3's converter, never re-records.
- Its precondition is that every case kind in dryrun-plan.yaml has its case files: S1's exist, and the S2 test authors write theirs before H0.
- A warehouse case added or changed after C-1 (a failing-first test, a reviewer's request) is "pending live" on that leg: it does not pass, its package's exit lists it, the tracker counts it, and E-0 records its reference line before E-1 judges it; its offline evidence meanwhile is the same case's DuckDB, SQLite and Postgres legs.
- The SQL-warehouse and Snowflake legs record from this machine, in the reference's own environment (`~/code/sdmc-reference`, the `ref-pandas3` cell); the Spark and Lakebase legs record in one-time serverless jobs (Lakebase is reachable from serverless only), which install the reference environment's Python packages at their recorded versions where the runtime allows, record the versions they ran, and register any per-runtime difference under H1.
- If `main` moves during the phase, the reference is updated (check 7) and E-0 re-records every warehouse reference line on the new reference commit before E-1.
- Size: the bases are small (8 to 120 rows, AdventureWorks' subset 1,000), each distinct table is stored once per leg as compressed Arrow, and the statement transcripts are text: about 20 MB for all legs, against Phase 2's 25 to 45 MB of reference observations; the per-leg CI jobs run the same cases as the DuckDB job, so each takes about as long, in parallel.

### 3. One contract bump, v28, in T3.4

**Recommendation:** v28 holds everything Phase 3 adds, as optional fields with defaults, so no existing field changes shape (Phase 2's choice 12 planned it):
- `Relation.columns`: each a schema name with either a source column or an SQL text expression; `Relation.filter`: an SQL text row filter;
- the `derived_relations` provider capability; a provider without it refuses a derived relation by name;
- option values as typed values, not the host's JSON text echoed back (P2-MID2-4), which a Java host needs (choice 10), and Python key equality written once (P2-WR-6);
- the carried contract rows: the data layer's built-in-only flags folded into `ProviderCapabilities` (P2-MID2-2), seed meanings and answered parts as explicit fields rather than metadata strings (P2-T2.2-1, P2-T2.8c-2), the unread-text-columns warning with separate table and column (P2-MID2-3), `index_columns` documented with a vector (P2-T2.1b-1), the unused composite-key vocabulary recorded as groundwork or cut (P2-MID2-5), and the binding tests reading the version from the crate (FX-6).
The provider failure class Phase 2 reserved is left out: nothing in Phase 3 reads it, and Phase 2 cut it once under KO2-46 for the same reason.
SQL text stays opaque to the core and is rendered only by `sdm-data`'s dialects; the core never matches error text.
If a later package needs vocabulary, it rides in T3.4 before T3.4 lands, or the second package to land renumbers (lesson 11).

### 4. Two dialects, ported, in `sdm-data`

**Recommendation:** `databricks` and `snowflake` modules in `sdm-data` beside DuckDB, SQLite and Postgres, each rendering every request kind through Phase 2's one `FROM` function, declaring every capability with no fail-open default, and accepting three relation name parts.
Port the old client's statements (`engine/rfm/backend/databricks/` and `snow/`, the `graph.py` builders, `SQLSampler`), with the first attempt's `dialects/src/databricks.rs` and `snow.rs` (349 and 358 lines) as a port source, never merged as is (decision 13).
The L0 guards already keep the two names out of every other crate.
Each dialect package runs the key-type pair matrix and every spelling of the same key value on its engine's recorded tables before review (lesson 7), with `DECIMAL(38,x)` keys meeting Phase 2's exact decimals (KO2-35).

### 5. Host providers stay thin

**Recommendation:** three Python providers in `sdm_client/_rust/providers/`: Databricks SQL warehouse (`cursor.fetchall_arrow()`), Databricks Spark (`spark.sql(...).toArrow()`), Snowflake (`cursor.fetch_arrow_all()`, on a connector connection or `session.connection` of a Snowpark session); Lakebase reuses the Postgres provider.
Each hands Arrow to the core with no pandas round trip; offline, its own code runs over a fake cursor returning C-1's recorded Arrow, and live, it passes the provider kit in E-1.
Authentication, connection objects, Lakebase's hour-long database credential and backend detection stay in the host: they are language objects the core cannot compute (lesson 1).
Backend detection learns to tell a SQL warehouse, a Spark session and a metric view apart; both Databricks connections report `databricks` today.

### 6. Views: a crate for YAML, a port for the rules

**Recommendation:** each dialect retrieves a view's definition with the old client's statement (Databricks: `DESCRIBE TABLE EXTENDED`'s YAML; Snowflake: `SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)`) and hands the text to `sdm-schema`, which parses it with a YAML crate (`yaml-rust2`, maintained and pure Rust) and interprets it into a view model with no dialect terms: tables, renamed and expression columns, joins as links, the filter, and measures or metrics as conversion notes.
YAML values are read as the old client's PyYAML reads them (YAML 1.1: `yes`, `on` and `off` are booleans where PyYAML makes them so) for the keys the view model reads, by PyYAML's documented implicit-type rules ported as a defined algorithm over the crate's plain scalars (lesson 3; KO3-40).
The expression allow-list is a port of `view_expr.py` (1,744 lines), with `trust_expressions` as the old client has it; the first attempt's 2,150-line hand-written YAML reader is not ported.
The existing metric view over AdventureWorks and the twin metric and semantic views over a small shop dataset (KO3-21; their names stay private) are recorded in C-1 and used as cases beside views the cases create in the test schema.
A `phase-1/dependencies.md`-style row records the crate choice and its measured reason (lesson 3).

### 7. Sampling and session state on warehouses

**Recommendation:** match the old client's SQL: Databricks `TABLESAMPLE (p PERCENT) REPEATABLE (seed)` with the percentage written as Python's `str(float)` writes it, Snowflake's sampling statement as the old client sends it, and `ORDER BY` and `QUALIFY ROW_NUMBER()` forms unchanged, so the same statement draws the same rows on the same table (KO3-41, RS-6's coverage).
The session time zone is fixed per capture and named in it; a zoned time column on a warehouse is a case of its own on each leg, crossing the UTC instant and the session wall clock (lesson 6's "every reading").

### 8. The switch, in one commit

**Recommendation:** T3.10 sets every S0, S1 and S2 path's default to `core` in `_routing.py` at once, so no call mixes old and new code (plan section 4), and changes the guard test that holds the defaults (`test_the_routing_seam.py`) in the same commit.
Inputs the core still refuses by name on a switched path are only the three KO2-94 keeps, if any of them is reachable from S0, S1 or S2 paths at all (T3.2's census measures it); they stay refused by name on the epic branch until Phase 4 lifts them, before the epic can merge into `main` (KO3-36).
Client tests that test old-engine internals rather than public behavior are marked to run on the old route only until Phase 5 deletes the engine, each listed in the evidence (KO3-37); every other client test must pass on the default routes.

### 9. Host-held rules in scope (lesson 4)

| Rule | Where it lives after Phase 3 |
|---|---|
| Which validator checks a query (P2-MID2-1) | The core (T3.8) |
| View YAML parsing and interpretation, and the expression allow-list | The core (`sdm-schema`, T3.5) |
| Dialect SQL, catalog reading, key binding, sampling statements | The core (`sdm-data`, T3.4, T3.6) |
| Connections, authentication, Lakebase credentials, backend detection, the Spark and Snowpark sessions, JDBC connections | The host: language objects |
| The serving transport (`core/databricks_serving.py`, and its Java twin) | The host: I/O, as the plan's run loop puts it |
| The model and route choice from the schema's shape | The host until Phase 4's tabular requests (Phase 1's ledger) |

The rust-host leg runs every S0, S1 and S2 case, the warehouse legs on the stand-in, before the held-out run; Java's legs run the same cases on Java's scope.

### 10. Java: a thin binding, Python's API in Java's words, the same cases

**Recommendation:** Java is a second host on the same core, built the way plan section 3 ("Bindings") describes, with no Java rule in the core:
- **Binding.** A new crate `rust/bindings/sdm-jni` on `jni-rs`, over `sdm-host` exactly as `sdm-py` is: control messages as versioned JSON bytes, data as Arrow through Arrow Java's C Data module with no copy, a panic contained at each exported entry, one owner and an explicit release for every native resource; L0's dependency guard admits `jni` only in that crate.
- **JAR.** One JAR, built by Maven from one `pom.xml` (the standard build for a single-module library with bundled natives, which publishes to Maven Central directly when Phase 5 publishes; Gradle would add a wrapper and plugins for no gain here, answering KO3-31's question), group `com.nvidia`, artifact `structured-data-models-client`, Java package `com.nvidia.sdm`, following Java's naming convention (KO3-31); nothing published until Phase 5.
  It holds the natives for Linux x86_64 and aarch64 (built where CI builds the wheels) and macOS arm64 and x86_64 (built on the `macos-15` and `macos-15-intel` runners CI already uses for the engine wheels, so macOS costs no new CI machinery, KO3-32), loaded from the JAR at first use.
  Java 17 bytecode, tested on JDK 17 and 21 on Linux x86_64 and on JDK 21 on aarch64 and macOS (KO3-34; plan section 6, L8).
- **API.** It mirrors Python's public API, with Java names and plain data: `Schema.fromSource(source, ...)`, `PredictiveQuery.parse(pql).materialize(...)`, `SdmClient.predict(request)`, sources built from a `java.sql.Connection` (`DuckDbSource`, `SqliteSource`, `PostgresSource`, `DatabricksSource`), results as Arrow `VectorSchemaRoot` with small values as Java records, options as builders, errors as one `StructuredDataException` carrying the contract's error code, as Python's `StructuredDataError` does (KO3-29). The names land in `phase-0/scope-java.yaml`'s `api` fields with T3.11.
- **Data access.** JDBC to Arrow in the host, through a reviewed conversion table (plan section 3: "native types to Arrow ... JDBC types: host, as a reviewed table"), with Arrow Java's JDBC adapter where it agrees with the table; L5 for Java checks the same tables reach the core as identical Arrow through Java's JDBC drivers and Python's drivers. The SQL is the core's own dialects', so Java's Databricks leg runs on the same stand-in as Python's.
- **Model calls.** The Java host sends the core's request bytes to the endpoint over `java.net.http` (and to a Databricks serving endpoint through its REST API), never parsing or re-encoding the body, as the Python host does since T2.14b.
- **Logging.** `tracing` bridged to the SLF4J 2 API, which the JAR declares and does not bind; the application chooses its logger (KO3-30).
- **Runtime flags.** Java needs `--enable-native-access=ALL-UNNAMED` and Arrow Java's `--add-opens=java.base/java.nio=ALL-UNNAMED`; the JAR's documentation names both, every Java test and the live check run with them, and the orchestrator asks the Java consumer (the Pyramid team) to confirm both, as Phase 0's G13 asked (KO3-35).
- **Tests.** `contract/runners/run_java.py` reads each case with the Python case reader and hands the Java program only the calls to make and the database to open; the Java program writes an observation in `contract/observation.schema.json`'s shape, judged by the checkpoint's comparator. T3.2 teaches the schemas, the comparator, the ratchet, coverage and the gate the runner name `java` and Java's manifest; the runner itself is T3.11's, as rust-host's driver lives in `sdm-testkit`.
  A Java leg runs exactly where the Java host reports the provider for it, as Phase 2 staged each database on its provider: S0 cases once the JAR builds (T3.11), the offline databases once their JDBC providers exist (T3.12), `databricks-sql` once the Databricks provider does (T3.13); T3.2 defines the rule, and once a leg's baselines are admitted the ratchet's "baselined, and not observed" rule stops a later candidate dropping it.
Rejected: a hand-written C interface with a Java FFM layer (plan section 3 chose `jni-rs`); generating Java from the Python API (Java idioms differ, and the Python API carries pandas).

## The live-test plan

Live work runs only under the owner's live rules: one-time jobs only, uploads only to the authorized volume, no existing notebook or cluster touched, only the Lakebase instance the owner names, and three live failures for non-product reasons stop live work (plan section 9; a quota or capacity error is retried once and only a failure after the retry counts, KO3-49).
The owner's shape (KO3-22): code against the offline judge, and run the live tests at the end.
So there is one live run at the start (C-1) and one live campaign at the end (E-0 to E-7), run on the commit that lands.
Nothing account-specific enters the repository: the profile, the volume, the catalog and schemas, the warehouses, the Lakebase instance, the endpoint and the views come from the environment and the orchestrator's private notes; Snowflake's credentials come from the owner's env file, loaded into the run's environment and never read or printed (KO3-24).
On Databricks the tests read the existing schema where AdventureWorks is loaded (KO3-52); on every warehouse they write only to the one dedicated test schema, through a write guard that refuses any other target (KO3-51): the client's own logic only reads.
The guard exists as a tested script (about 50 tests) outside the repository, which C-1 uses; T3.7 brings it into `scripts/live/` with its tests, and every later live run goes through it.
Wheels: C-1 and E-2's baseline pass install the reference's wheels; every other live run installs the candidate's own wheels, built by CI for x86_64 and aarch64 (KO2-40, KO3-26).

**Offline, from C-1** (seconds per case, no account): every warehouse leg of every case on python-core and rust-host, on the stand-in (choice 2); the dialects' goldens with their diff against the recorded statements; view interpretation from the recorded YAML; each warehouse provider's own code over a fake cursor returning C-1's recorded Arrow; Lakebase's whole data path on local Postgres.

**Live, at the end**: that the engines accept the core's SQL and give the old client's answers; `TABLESAMPLE` repeatability; the connect paths (`warehouse()`, `lakebase()`, Snowpark) and the serving endpoint; both notebooks; the job matrix; live timing; the held-out set's warehouse legs; Java's Databricks questions.

| Live run | When | What runs | Jobs or sessions | Wall time (forecast) |
|---|---|---|---|---|
| C-1 The early capture | once, about H5 to H8, from the reviewed recording SHA | Phase 2's bases and AdventureWorks loaded into each warehouse's test schema; one probe job per aarch64 cell's runtime on a Graviton node, installing the reference's aarch64 wheel and importing it; every S1 acceptance case and every S2 case on the old client on each of the four legs, recorded (choice 2); the existing views' YAML; the provider kit's fixture | SQL warehouse and Snowflake: sessions from this machine; Spark and Lakebase: one one-time job each | about 3 h, the legs in parallel |
| E-0 Reference top-up | the end campaign, first | The old client on every case added or changed after C-1, and on every case if `main` moved | As C-1 | 0.5 h (2 h if `main` moved) |
| E-1 The core on every warehouse case (L6) | after E-0 | Every case on python-core live on each leg, recorded as the core's capture, compared with E-0 and C-1 under H9 in full; the provider kit once live | As C-1 | about 2 h, in parallel |
| E-2 The job matrix, both passes | after E-1 | 21 cells: the 15 Unity Catalog cells ask the quickstart's 5 questions and the deleted notebook's 8 (195 runs per pass), the 6 Lakebase cells the 6 base questions (36 runs); once on the reference's wheels, once on the candidate's; against the model-version-26 endpoint (KO3-53) and the owner's existing views | 42 one-time jobs, in parallel | about 1.5 h |
| E-3 Both notebooks on the core | beside E-2 | `kumo_relational_quickstart.ipynb` and `lakebase_relational.ipynb` on the candidate's wheels | Two one-time jobs | about 1 h |
| E-4 Live L7 timing | after E-2, on quiet warehouses | Push-down median over at least 5 interleaved runs per scenario, caches off (`use_cached_result` off on Databricks, `USE_CACHED_RESULT` off on Snowflake), the core against the old client | Sessions from this machine | about 1.5 h |
| E-5 Held-out warehouse legs | inside E-1's window, once | HO3's warehouse cases: the old client and python-core live, recorded; rust-host on the stand-in loaded from E-5's own recording | One session or job per leg | about 1 h |
| E-6 L3 on warehouses (KO3-43) | inside E-2's window | 20 generated sources per warehouse, both routes | One session or job per warehouse | about 1 h |
| E-7 Java on Databricks | inside E-2's window, on the same candidate | Every Java S2 case, and Phase 0's J2 questions (the four base questions and the two metric-view questions, KO3-35), through the Databricks JDBC driver on the SQL warehouse, compared with Python's answers on the same data in E-1 and E-2 | A session from this machine, and one job running the JAR on x86_64 (DBR 17.3, Java 17) and one on aarch64 (DBR 17.3 on a Graviton node) | about 45 min |

The end campaign's serial chain is E-0 (0.5 h), E-1 (2 h), E-2 with E-3 and E-7 (1.5 h), then E-4 (1.5 h): about 5.5 hours.
It runs twice in the forecast: once on the first switched candidate, beside the whole-phase review, and again on the final candidate after their fixes, since live evidence counts only for the commit that lands (lesson 15); any change after the final run under `rust/`, `packages/` or `contract/` runs it again.
If the first run and the review find nothing, the first run is the evidence and the switch lands about 8 hours earlier.
Cost: C-1 takes about 6 one-time jobs (2 for the capture, 4 aarch64 probes); one end-campaign run takes about 50 (42 matrix jobs, 2 notebook runs, 2 to 3 for the held-out set and L3, 2 for Java), so the forecast's two runs take about 100, or about 56 in all if the first run is the evidence; each 10 to 20 minutes on a single-node cluster or serverless; the authorized SQL warehouse for about 10 hours of sessions in all, auto-stopping; Snowflake's smallest warehouse for about 6 hours; the warm model-version-26 endpoint during the matrix and notebook runs (KO3-53).
The orchestrator cannot see prices; the owner can turn these hours into money.

## Packages

M is mechanical (a port with reference recordings), N novel, R risky.
"Judging" means the package changes a path of `scripts/rust_core/judging-paths.txt`; only T3.2, the tooling train, does, and a train may change only judging paths, their tests and docs (T3.0's `judging.py`).
Every product package has a separate test author who writes its cases first; the warehouse cases' reference lines come from C-1, and its pending-live cases' from E-0.
Every package runs `handoff.py` before review and ports the old client's rule in each fix (`Rule:` lines).
Lanes: A head and tooling, B Databricks, C views, D Snowflake, E switch-ready, F differences and coverage, G Lakebase and live runners, H harness, J Java.
Hours are the implementation hours a package holds a build slot for; its review rounds and their queue, about 3 hours more, run outside the slots (the forecast below adds them), as Phase 2's planned work measured once its overhead is removed.

| ID | Package | Output | Exit check | Depends on | Test author | Lane | Stack | Kind, hours |
|---|---|---|---|---|---|---|---|---|
| T3.0 | Process mechanisms | Stacks, the tooling train, the handoff stamp, CI's split and timing guard, lane environments, the command hook, the dry run, the slip report | Its own (T3.0's brief) | – | T3.0's | A | 0 | landing now |
| T3.1 | This design | design.md, decisions.md, tracker.md, locked-set.yaml, forecast.yaml, dryrun-plan.yaml; learnings.md rewritten; `briefs/package.md` and `scripts/rust_core/brief_check.py`; `tests/test_learnings_enforced.py`; the design reviewer's brief and handbook on the new lesson numbers; docs tidy rows | Two APPROVEs on one SHA; the locked-set reader reads half `3a`; `brief_check.py`'s and the learnings check's focused tests each fail with their rule removed | T3.0 | T3.1's | A | 0 | docs and a small check |
| T3.2 | The tooling train (judging paths only) | Case format: `warehouses`, `input.view`, `input.connect`, both loaders (`run_python.py`, `case.rs`) and malformed vectors, P2-T2.1i-1 aligned; the recording mode in `run_python.py` (choice 2) and its completeness in `capture_check.py`, with each case's input hash; `contract/captures/**` added to `contract/protected-paths.txt`; the `live_only` outcome in `compare.py`, the ratchet and `gate.sh`, its reason codes in `contract/compare-policy.yaml`, and the locked run failing on a `live_only` item with no E-1 verdict; `l0_deps.py` classifying `sdm-jni` as a binding allowed only `sdm-host` and `sdm-contract`, absent until it exists, and `l0_scan.py` scanning it for dialect words once it does; the Java legs' admission, each run where the Java host reports the provider (S0 cases once the JAR builds, the offline databases once JDBC providers exist, `databricks-sql` once the Databricks provider does), held by the ratchet's no-removal rule; `<case>@<leg>` keys and the runner `java` in the observation schema, `observation.rs`, `compare.py` and `ratchet.py`; H9's diagnostic statement rule on stand-in legs in `compare-policy.yaml` (KO3-45); `coverage.py`: legs claim their paths, client tests may claim a path, the S1, S2 and Java rules; the census's real-data mode per leg; `register.py` refusing an `approved_by: owner` entry with no owner-approved row and U4's allowlist; `lane.py start --brief` running `brief_check.py`, and refusing a package ID with a letter from Phase 3 on; `gate.sh` running the stand-in and Java legs; P2-T2.1a-2, P2-MID2-7, P2-T2.0f-1, P2-T2.21-1; every GAP of the full dry run | Each change's one focused test fails without it (for `live_only`: a case marked live-only with no stand-in reason is refused, a pending item fails the locked run, and an item with an E-1 verdict passes; for `l0_deps.py`, a seeded workspace holding an unclassified binding, and one holding `sdm-jni` with a `pyo3` edge); every stub the tools T3.2 changes judge (format, coverage, ratchet, compare) passes the dry run with no GAP, while the stand-in and Java kinds are judged once T3.3 and T3.11 exist; `land.py --train` accepts its file list | T3.1 | T3.2's | A | 1, front | N, about 6 h (recording first, about 2.5 h) |
| T3.3 | The warehouse harness | The loader into the four warehouses with each engine's type map (`databases.py`, `contract/type-map-vectors.json`); the stand-in provider of choice 2 (`sdm-testkit`'s `src/standin.rs`, its capture reader, the `sdm-standin` binary, and `contract/runners/standin_client.py`); C-1's captures, committed under the protected-path trailer; `handoff.py` running each touched case's warehouse legs on the stand-in; the capture writer's placeholders, its format version and converter; the C-1 and E-0 job templates (`scripts/live/`); the per-leg CI jobs (`contract-python (ref-pandas3, <leg>)`, `cross-host (<leg>)`) and the CI rows of the carried table; P2-T2.7-1, P2-T2.7-2 | A private name in a capture fails the write; the stand-in fails by name on a recorded table dropped, on a random draw dropped, and on a draw asked with another seed or fraction text; an exact-replay statement serves its recorded result; a request neither part answers is `LIVE_ONLY` with its reason code; `l0_deps.py` passes with the stand-in in the workspace; the converter reads a version-1 capture; a warehouse-only difference stops `handoff.py`'s stamp; CI's leg jobs run the cases they name | T3.1 | T3.3's | H | 1 | N, about 6 h (loader first, about 3 h) |
| T3.4 | Databricks and contract v28 | Choice 3's bump; the `databricks` dialect and catalog reader; the SQL-warehouse and Spark providers; routes and backend detection for `databricks-sql` and `databricks-spark`; RS-6's percentage text; the stand-in extended for v28's relation fields (`derived_relations`), under the protected-path trailer; U12, P2-T2.2-3 | Every case's `databricks-sql` and `databricks-spark` legs equal to C-1's reference lines on python-core and rust-host on the stand-in; the goldens of every request kind, with their diff against the recorded statements reviewed; the key-type pair matrix on Databricks' recorded tables; the catalog reader's output equal on the replayed catalog statements; both providers' own code over a fake cursor returning C-1's recorded Arrow; schema lock and both loaders at v28; pending-live and live-only cases listed | T3.2, T3.3 (stacked on their candidates); C-1 | T3.4's | B | 2 | M/R, about 10 h |
| T3.5 | Views | The view model in `sdm-schema`, the YAML crate with PyYAML's 1.1 reading, the expression allow-list port, metric view retrieval and interpretation, semantic view interpretation; its dependencies row | Every metric-view case on `databricks-sql` and `databricks-spark` equal to C-1's reference lines on the stand-in, the view reads served by exact replay or from the warehouse-evaluated view tables, any other case listed live-only; semantic-view interpretation from the recorded YAML equal to the old client's schema; YAML 1.1 value vectors (`yes`, `on`, `off`, `~`, octal-looking text) read as PyYAML reads them | T3.4 (v28's relation fields; stacked on its candidate) | T3.5's | C | 3 | M/R, about 9 h |
| T3.6 | Snowflake | The `snowflake` dialect and catalog reader, the semantic view's retrieval, the provider (connector and Snowpark), routes | Every case's `snowflake` leg and every semantic-view case equal to C-1's reference lines on both hosts on the stand-in, live-only cases listed; goldens reviewed against the recorded statements; the key-type pair matrix and a session-zone case on Snowflake's recorded tables; the provider's own code over a fake cursor returning C-1's recorded Arrow | T3.4; T3.5 (below it in stack 3: T3.6's review rounds run on a SHA stacked on T3.5's approved candidate); C-1 | T3.6's | D | 3 | M, about 8 h |
| T3.7 | Lakebase, the live runners and the phase-end reports | The `lakebase()` connect path and routes on the Postgres dialect; reads qualified by schema (P2-MID2-6); the write guard in `scripts/live/` with its tests, through which every live run writes only to the dedicated test schema (KO3-51); the matrix runner and job template, both notebooks' driver, installing the wheels under test on each cell; live timing in `contract/tools/l7.py --live`; offline warehouse L7 in `l7.py --warehouse` (choice 2's comparable measure: the old client replaying its own C-1 statements through a fake driver, against the core with the stand-in out of process, both measured as client CPU with no engine time in the measured process); G8; `differential.py`'s database axis and classes; `evidence.py`'s per-leg table; the local H24 tool; refusal maps citing quoted text | Every case's `lakebase` leg equal to C-1's reference lines on local Postgres; the write guard refuses a write outside the test schema on each warehouse's statement forms, and every live runner goes through it; the matrix runner's dry mode prints 21 cells' job specs, and a cell installing a published wheel instead of the one under test fails; `l7.py --live --dry` prints the interleaved plan with caches off; `l7.py --warehouse` reports every warehouse leg's scenarios for both clients, and a seeded slow scenario fails each report; L3 runs on SQLite and Postgres; the per-case table shows every leg | T3.2, T3.3; C-1 | T3.7's | G | 2 | M, about 8 h |
| T3.8 | Switch-ready: the forced-route failures | The 118 failures fixed in the core or the bridge's wording, each family ported from the old client's rule (the time-column warning, key counting and its refusals, Postgres refusals and the described-schema `RouteError`); the validator chosen in the core (P2-MID2-1); the bridge declarations its bridge edits need | `pytest packages/structured-data-models-client/tests` with every S0 and S1 route forced: 0 failures; a case per family on three databases | T3.1 | T3.8's | E | 1 | R, about 8 h |
| T3.9 | Differences, coverage and DuckDB 2.0 | The 11 user-visible rows matched or registered; OF-5: the three overnight refusals lifted (KO3-44), P2G-4, P2G-5 and KO2-89 pointed at KO3-57, P2D-2 to P2D-5 left as they stand until the owner reads them; acceptance cases for every S0 and S1 path and cell coverage still finds empty; the S1 test rows; P2G-1 widened; DuckDB 2.0 supported, its CI cell, and the `zip` exceptions dropped if its build allows; the interim edit gate retired; register tidy | `coverage.py` S1 rule (T3.2) with no empty path or cell; each row's case fails before and passes after; CI green on DuckDB 1.5 and 2.0 | T3.2 (coverage); T3.8 for the shared files | T3.9's | F | 2 | M, about 9 h |
| T3.10 | The switch and the switch gate | Choice 8's flip and guard test; the old-route-only list; the S2 case IDs added to half `3a` (the one planned change to the locked set); register reconciled; `main` merged if it moved; the locked run; the end-of-phase reports; the whole-phase review; the end campaign E-0 to E-7 on the final candidate; E-0's reference lines and the core's live capture committed under the protected-path trailer, which discharge every `live_only` item; HO3 once; the evidence pack; the packet | The locked checks below | all Python packages; the Java packages that have landed by its final candidate | T3.10's (the guard test and the switched suites) | A | alone | R, about 17 h on the critical path |
| T3.11 | Java binding, API and Java S0 | `sdm-jni`; the Maven build and the JAR with natives for Linux x86_64 and aarch64 and macOS arm64 and x86_64; the Java API of choice 10 and its exception mapping; the SLF4J bridge; `contract/runners/run_java.py` and the Java runner program; `handoff.py` running each touched case's Java legs; CI jobs building the JAR and running the Java legs (JDK 17 and 21 on x86_64, JDK 21 on aarch64 and macOS); a panic test and a JAR-load test in a clean container | Every S0 case on the Java host equal to the reference, and no S1 or S2 Java leg admitted yet (no Java provider exists); a Java-only difference stops `handoff.py`'s stamp; a panic inside the core reaches Java as an exception and leaves the JVM running; the JAR loads on every JDK and platform listed; `l0_deps.py` (T3.2) holds `sdm-jni` to `sdm-host` and `sdm-contract` and refuses `pyo3` in it | T3.2 (the `java` runner name; stacked on its candidate) | T3.11's | J | 3 | N, about 9 h |
| T3.12 | Java S1 through JDBC | JDBC providers for DuckDB, SQLite and Postgres; the JDBC-to-Arrow conversion table; L5 for Java; the model transport over `java.net.http` | Every S1 case on Java's 46 S1 paths equal to the reference on each database on the Java host; L5: the same tables through Java's and Python's drivers reach the core as identical Arrow, and a seeded unit or scale change in one JDBC path fails; `coverage.py`'s Java rule with no empty S0 or S1 path | T3.11 (stacked on its candidate); T3.8, T3.9 (the shared-core fixes) | T3.12's | J | 3 | M/N, about 9 h |
| T3.13 | Java on Databricks | The Databricks JDBC provider, the metric-view route through the same connection, the serving transport, Java's S2 routes; the Java runner's forwarding provider to `sdm-standin`; E-7's runner | Every Java S2 case's `databricks-sql` leg equal to C-1's reference lines on the Java host on the stand-in, live-only cases listed; E-7's dry mode prints every Java S2 case and the JDBC session plan; E-7 itself runs in T3.10's end campaign (check 14) | T3.12; T3.4, T3.5 (the dialect and the views) | T3.13's | J | alone | M, about 7 h |
| HO3 | The held-out set | About 40 cases: about 25 offline across the three databases, about 15 on the warehouse legs, a stored-NaN case and a `key_name_order` case (OF-6), and about 5 on Java's scope; written by a held-out author who reads no implementation, kept outside the repository | Written beside the lanes, which do not wait for it; frozen before its one use in T3.10; its converter tested and two dummy bundles run through `heldout.py counts` | T3.2 (the case format) | – | orchestrator | not a landing | – |

T3.2 is the phase's one planned train; a judging change found later rides at most one catch-up train at the front of the next stack (T3.0's rule), and any further one goes to the owner (lesson 13).
Owner decision rows and register entries ride with the package that needs them (KO3-1).
The Python package beyond KO3-1's "about 7 to 10" is the harness, which T3.0's train rule keeps out of the train (its files are not judging paths); the three Java packages are the owner's addition (KO3-27).

## Critical path and forecast

The forecast is the time from T3.1's landing (H0) to each landing, in landing order; [forecast.yaml](forecast.yaml) holds it for T3.0's slip report (`phase_stats.py --forecast`), which the heartbeat runs and which never stops work (KO3-12).
It is built from four Python build slots (KO3-16) and the Java lane (KO3-28): a package holds a slot while it is being written and its fix rounds queue through `heavy`; reviews and CI hold none.
The fifth lane's builds queue behind the others in `heavy`, which the forecast counts as about an hour on the Python path.

| Hours | Slot 1 | Slot 2 | Slot 3 | Slot 4 | Java lane |
|---|---|---|---|---|---|
| H0 to H6 | T3.2 (to H6) | T3.3 (to H6) | T3.4 | T3.8 (to H8) | T3.11 |
| H6 to H10 | T3.9 (from H6) | T3.6 (from H7) | T3.4 (to H10) | T3.8, then T3.7 (from H8) | T3.11 (to H9), then T3.12 |
| H10 to H18 | T3.9 (to H15) | T3.6 (to H15) | T3.5 (from H10) | T3.7 (to H16) | T3.12 (to H18) |
| H18 to H26 | free | free | T3.5 (to H19) | free | T3.13 (to H25) |
| H23 to H34 | T3.10's flip and fixes | | | | |

| Landing | Forecast | Holds the queue for | Why |
|---|---|---|---|
| Stack 1: T3.2, T3.3, T3.8 | H12 | 12 h | T3.2's and T3.3's recording commits reviewed at about H4.5; T3.2 approved about H9, T3.3 about H9, T3.8 about H11; the stack's CI run, about 1 h |
| Stack 2: T3.4, T3.7, T3.9 | H20 | 8 h | T3.4 judged on the stand-in from C-1's end (H8), approved about H13; T3.9 about H18; T3.7 about H19; CI about 1 h |
| Stack 3: T3.5, T3.6, T3.11, T3.12 | H25 | 5 h | T3.5 written from H10 on T3.4's candidate, approved about H22; T3.6 written H7 to H15, then restacked on T3.5's approved candidate at H22 and reviewed there, approved about H23.5; T3.11 approved about H12 and T3.12 about H21, waiting for the stack; CI about 1 h |
| T3.13 alone | H29 | 4 h | Written from H18, its metric-view part on T3.5's candidate, approved about H28 |
| T3.10 | H41 | 12 h | The flip and the switched suites (H23 to H25, on stack 3's candidates); the first switched candidate's wheels (H26); the end campaign's first run (H26 to H31.5) beside the whole-phase review (H25 to H29), with the hour of fifth-lane queueing on this path; fixes (H31.5 to H34.5); the final candidate's wheels (H35.5); the end campaign on it, E-7 included (H35.5 to H41) |

**Critical path:** T3.1 → T3.2's and T3.3's recording commits and their review (4.5 h) → C-1 (3 h) → T3.4 on the stand-in → stack 2 → T3.5 → T3.10's flip, review and two end-campaign runs: about 41 hours to the Python switch, which ends the phase.
Java is off that path: its lane runs beside the Python slots from H0, and its last package lands about 11 hours before the switch.
Before H0: T3.0's landing, this design's review rounds, the S2 test authors' case files, and the full dry run.

This forecast misses the owner's one-day goal by about 17 hours.
The round-1 draft's 28 hours double-booked build slots, under-counted the end campaign, and ran live evidence on a commit before the one that lands; this one does not.
With the fifth lane, Java ends inside the phase rather than after it; with only four slots it would have ended about 3 hours after the switch.
What would shorten it: the end campaign passing on its first run with nothing for the whole-phase review to fix (8 hours); the owner declined a bigger machine (KO3-55).
None of them moves scope out of Phase 3.

## Machine and lanes

At most four Python packages build at once (KO3-16), with Java's lane as a fifth (KO3-28), each in its lane's environment (`cells.sh`), with `heavy` admitting builds and gate legs by CPU, memory and disk; test authors and reviewers run no heavy suites, and reviewers do not live-test (KO3-15; T3.0's reviewer brief has them read the handoff stamp and CI).
The Java lane adds a JDK 17 and a JDK 21 to its environment, about 1 GB of disk.
The orchestrator runs C-1, the end campaign and the reference machine's work; implementers never run live jobs.

## The checking tools' dry run

[dryrun-plan.yaml](dryrun-plan.yaml) holds 30 stub cases in six groups: an S1 kind on each warehouse leg, AdventureWorks on a warehouse and a three-part name; what only a warehouse has (views, the four connect paths); warehouse values (`DECIMAL(38,6)` keys on Databricks and Snowflake, a session-zone time column, a `STRUCT` feature, a schema-qualified Lakebase read); the switch (legs claiming their paths, an uncovered edit path, a pickled request, the key-repeating family, S0 validation on the core); the endpoint refusing a one-call request; and Java (S0, S1 through JDBC, Databricks through JDBC).
Its `reports` section names each end-of-phase report command the locked checks need, so the full dry run shows which do not exist yet.
A first run of T3.0's `gate_dryrun.sh` from T3.0's topic branch, judged against Phase 2's last landing (the case reader, `coverage.py` and `ratchet.py`; no runners), over the first 21 kinds, found:

| Gap | Kinds | Goes into |
|---|---|---|
| The case format's `databases` names only DuckDB, SQLite and Postgres, and naming fewer needs an owner exclusion | every S2 kind | T3.2: the `warehouses` field (choice 1) |
| `input.view` and `input.connect` are not case fields | the view and connect kinds | T3.2 |
| `input.pickle` is not a case field | the pickled-request kind | T3.2: the path is claimed by a named client test instead (coverage) |
| Coverage's rule 6 wants a reference observation of every claimed cell | the S1 kinds | Expected for a stub with no reference line; not a gap |
| `coverage.py --check` cannot see that a SQLite, Postgres or warehouse leg covers its path, and reads only Python's manifest | (no rule yet) | T3.2 (P2-T2.16-3, the Java rule) |

The full run, with the runners (`--python`), rust-host (`--rust`) and the locked gate (`--gate 3a`) on T3.0's tree, runs before the design locks; its GAP lines join T3.2's list, and the tracker records them.

## Owner check-ins

- Kickoff: done; every design question is answered ("Owner questions").
- Twice a day, a five-line check-in with the slip report's line, the pending-live count, the register entries reviewers cannot approve, and any overnight decision written plainly when made (KO3-9).
- One design checkpoint when stack 2 has landed (half the planned landings), and the whole-phase review in T3.10, both by both reviewers.
- End: the go or no-go packet (handbook, "Ending a phase", step 6), Java included.

## The evidence pack

As Phase 2's, on the final SHA, generated by the branch's own runs, plus:
- the per-case table by database and warehouse leg, by host, Java included;
- the captures' index (cases, legs, tables, draws, the redaction check, each leg's runtime versions);
- E-1 against E-0 and C-1 per leg; both notebooks' runs; the job matrix's two passes, question by question; live L7; the held-out run; E-7 beside Python's answers;
- the switch: the routing defaults, the client suites on the default routes, and the old-route-only list;
- L3 per database, the census with real-data counts on every leg, offline L7 on every database and warehouse leg, live L7, coverage S1, S2 and Java;
- the host-held list before and after (choice 9);
- Java: L5 for Java, and the JAR's load test per JDK and architecture.

## Gates

### Merge rules

As Phase 2's, with T3.0's mechanisms: packages land in stacks of up to four on one CI run (`land.py --stack`, `stack-land`; KO3-13), a cancelled run rerun as T3.0's rule says; judging changes only in T3.2 or one catch-up train; two reviewers on every product round, both approving the same SHA (KO2-9), starting only on a SHA with a handoff stamp; a difference from the old client blocks only where the core is worse (a wrong answer, a crash, lost support, a worse error), while one where the core is better (a better error, or more support) is registered as an improvement, reviewer-approvable, and does not block; a red CI job the change caused, and a decision row wider than the owner's words, still block (KO3-41, KO3-47; `briefs/reviewer.md`); on docs-only fix rounds the reviewer who raised the finding confirms its fix, and rounds writing decision rows or register entries keep Claude (KO3-5); a finding family blocking twice gets a half-page rule note, not a redesign session (KO3-3).
Package IDs are integers, T3.0 to T3.13; register IDs come from per-lane blocks (`P3A-` for lane A, `P3B-` for lane B, and so on); each package writes its own tracker file; decisions.md takes only the owner's rows.

### Locked acceptance checks

Locked at T3.1's landing; the one planned change is T3.10 adding the S2 case IDs to half `3a` from the S2 case files then landed, of the kinds in dryrun-plan.yaml; any other change is the owner's (plan section 9).
Phase 3's locked set is [locked-set.yaml](locked-set.yaml), one half, `3a`, run as `gate.sh --locked 3a`: the 309 S0 and S1 acceptance cases at the lock (19 of the 24 S0 cases and 290 of the 291 S1 cases; the other six are diagnostic), each S1 case on every database and warehouse leg its file and its engines' type maps give, steps `inventory`, `seeded` and `main`, checks 1, 2, 7 and 10 held by gate steps, and 12 the every check.
`gate.sh`'s reader reads it (run on this draft: it lists the half's steps and 309 cases, equal to today's acceptance set).
The rest is judged outside the gate, as Phase 2's KO2-76 set it: CI-green parts by `land.py`'s exact-SHA rule, reports read by both reviewers, and the live and held-out runs once in T3.10, on the commit that lands.

| # | Check | Tool and machine | Fails on |
|---|---|---|---|
| 1 | Full L2 parity for S0, S1 and S2: every case on each database and warehouse leg it runs on, per field, on python-core and rust-host against the reference, the warehouse legs on the stand-in (choice 2: public outcome gating, statements diagnostic) and H24 by construction; every difference has an approved register entry; coverage S1 and S2 with no empty path or cell; no case pending live, and no `live_only` item without an E-1 (or E-7) verdict on the final candidate, at the final run | `gate.sh --locked 3a`; CI: `contract-python (<cell>, <database or leg>)`, `cross-host (<database or leg>)`; report: coverage | A changed answer on one leg; a case missing on one leg; a random draw the old client did not make; an empty coverage cell; a `live_only` item with no E-1 verdict; a live-only mark with no stand-in reason |
| 2 | No unapproved refusal of an input the reference accepts, and no panic, on any leg; the census's real-data mode (T3.2) over AdventureWorks at full size and RelBench H&M on DuckDB, SQLite and Postgres and over C-1's AdventureWorks tables on each warehouse leg's stand-in, with per-leg counts; a kept rule reachable on a switched path only under an owner answer | `ratchet.py` in the gate; report: `refusal_census.py --real-data`, reference machine | A refusal with no entry; a kept rule hit on a switched path with no owner answer; the census's control, a seeded refusal on a real source |
| 3 | L4: SQL goldens executed on DuckDB, SQLite and Postgres (Lakebase's dialect is Postgres's); the Databricks and Snowflake goldens reviewed against the recorded statements, and executed live in E-1 with H9 in full | CI: `l4-goldens (<database>)`; report: E-1 | A dialect change with no golden update; a statement the engine refused or answered differently live |
| 4 | L5: each warehouse driver's Arrow types, as recorded, reach the core without loss; the type maps list what each engine cannot store | CI: `l5-conversion*`; report | A unit or scale change in one driver's path |
| 5 | The provider kit: every offline provider and the non-SQL provider pass it in CI; each warehouse provider's own code runs offline over a fake cursor returning C-1's recorded Arrow, and the kit passes on each warehouse live in E-1 | CI: `rust-test*`, `contract-python (ref-pandas3, <database or leg>)`; report: E-1 | A provider that truncates a page or fails a capability silently |
| 6 | L3: about 200 generated sources on each of DuckDB, SQLite and Postgres, and 20 per warehouse (KO3-43), every divergence classified | Report: `differential.py` (T3.7), reference machine, and E-6 | An unclassified divergence |
| 7 | The reference and the phase branch at `main`'s head as of the gate run, the reference rerun, the warehouse reference lines re-recorded in E-0 if it moved, every changed line traced to a `main` commit | `ratchet.py`, the `main` step | `main` moved and not merged; a changed line with no commit |
| 8 | The switch: every S0, S1 and S2 default is `core`; the client suites pass on the default routes, except tests listed old-route-only under KO3-37; rust-host runs every S0, S1 and S2 case; the host-held list matches choice 9 | CI: `test (3.x)`, `installed-wheel`, `cross-host (*)`; `rust-lint` (the bridge freeze) | One path left `old`; a client test failing on the default route; a test marked old-route-only that tests public behavior |
| 9 | Held-out (HO3) under H11's threshold on python-core and rust-host, warehouse legs included (E-5), and on the Java host for its Java cases | `phase_end.sh`'s held-out step and E-5, once | Before the one shot, two dummy bundles (a panic; differences over the threshold) each a no-go |
| 10 | Boundary and packaging: L0 clean, no `pyo3` in the pure-Rust host's tree, `jni` only in `sdm-jni`, dialect words only in `sdm-data`, the wheel installing on x86_64, aarch64 and each runtime cell | CI: `rust-lint`, `rust-test*`, `wheel-smoke (*)`, `runtime-install (*)` | Phase 1's seeded L0 faults; a dialect word in `sdm-schema` |
| 11 | L7: offline within KO2-29's limits on DuckDB, SQLite and Postgres, and on every warehouse leg by `l7.py --warehouse` (the old client replaying its C-1 statements against the core on the stand-in, both client CPU only); live push-down median within 10% of the old client's on Databricks and Snowflake, at least 5 interleaved runs, caches off, in the end campaign (KO3-39) | Report: `contract/tools/l7.py` offline, `l7.py --warehouse` (T3.7), and `l7.py --live` (T3.7) in E-4 | A scenario past 1.1 times; a live median past 10%; the live report's control, a seeded slow scenario |
| 12 | Every gate step the half runs passed | `gate.sh --locked 3a` | Any one step seeded to fail |
| 13 | L6: every case live on each warehouse leg on the core equal to the old client's live answers (E-1 against E-0 and C-1, H9 in full); both notebooks on the core; the 21-cell job matrix on the core equal to its reference baseline, question by question, on x86_64 and aarch64 cells; all on the commit that lands | Report: E-1, E-2, E-3 | A question answered differently in one cell; a cell that does not install the wheels under test; evidence naming another commit |
| 14 | Java (KO3-27): every S0 and S1 case on Java's paths on the Java host equal to the reference on DuckDB, SQLite and Postgres, and Java's `databricks-sql` legs on the stand-in, each leg admitted when the Java host has its provider; L5 for Java; the JAR loads on JDK 17 and 21 on Linux x86_64, and on JDK 21 on Linux aarch64 and macOS arm64 and x86_64 (KO3-32, KO3-34); E-7: every Java S2 case live on Databricks equal to Python's answers on the same data, run with the flags of choice 10 (KO3-35). Judged for Java's packages and the phase's end; a Java-only failure never holds a Python package, and a shared-core defect a Java leg shows blocks the path it breaks (plan section 6) | CI: `java (<jdk>, <platform>, <database or leg>)`; report: L5 for Java, E-7 | A Java leg differing from the reference; a seeded unit change in one JDBC path; a JAR missing one architecture's native |

### Process rules that differ from Phase 2

- T3.0's mechanisms replace Phase 2's rules they encode (the handbook's "The Phase 3 process mechanisms (T3.0)").
- C-1 and the end campaign are the orchestrator's work, like the reference machine's; no product package runs anything live (KO3-22).
- A capture is evidence for the commit that recorded it; its format changes only through T3.3's converter; a case added after C-1 is pending live until E-0.
- Live evidence names the commit that lands: the end campaign's final run is on the final candidate, and any later change under `rust/`, `packages/` or `contract/` runs it again.
- An agent moves only its own lane branch; T3.0's hook blocks `--force` pushes, so a restack uses `land.py restack`.

## Risks

| Risk | Guard | Lesson |
|---|---|---|
| A dialect fix needs a live run per bug | The stand-in judges answers offline (choice 2); live tests only at the end (KO3-22) | 6, 14 |
| The end campaign finds real-engine problems all at once | Dialects port the old client's statements, which the engines ran in C-1; goldens reviewed against them; the recorded types feed the stand-in; one fix cycle and a second campaign run in the forecast | 6, 12 |
| The stand-in answers differently from the warehouse | The comparison is against the old client's answers from the warehouse, so a stand-in difference fails rather than passes; E-1 judges the real engine | 2, 6 |
| The core grows driver or pandas behavior to match warehouse drivers | Providers hand Arrow to the core; differences go to the register; L5 from the recorded types | 1, 3 |
| Copying the reference's warehouse accidents (sampling text, quoting) | Q6 on RS-6; registered keeps and drops | 2 |
| A hand-written YAML reader or expression evaluator | A YAML crate; the allow-list ported from `view_expr.py`; a dependencies row | 3 |
| A warehouse rule is left in the Python host and Java lacks it | Choice 9's list; rust-host and Java run every warehouse leg on the stand-in | 4 |
| Java finds Python assumptions in the core (option values echoed as Python text, Python's `str()` of dates) | Typed option values in v28 (P2-MID2-4); Java runs the same cases from T3.11 on; a Java-found core defect is fixed in the core | 1, 4 |
| JNI misuse crashes the JVM | `jni-rs`; panics contained per entry; a crash test in a subprocess; one owner per native resource | 1 |
| Java's JDBC types reach the core differently from Python's drivers | The reviewed conversion table and L5 for Java on the same tables, before T3.12's review | 3, 5 |
| Tests pin the mechanism (the stand-in) instead of the answer | Cases compare public answers against the reference; a missing table or draw fails by name, never passes silently | 5 |
| Warehouse keys (`DECIMAL(38,x)`, `BIGINT` against `STRING`) join wrong rows | The key-type pair matrix on each engine's recorded tables before review | 7 |
| A fix makes the reviewer's example pass | `Rule:` lines, failing-first tests, the ratchet over every leg | 8 |
| A refusal of real warehouse data surfaces last | The census's real-data mode on every leg (T3.2), run after stack 2 | 9 |
| The live benchmark compares nothing | G8 in T3.7; real datasets only | 10 |
| Two packages bump the contract | One bump in T3.4; a later need rides in it or renumbers | 11 |
| The phase overruns silently | The forecast file and the slip report; the pending-live count at check-ins; round 4 to the owner | 12 |
| Judging gaps found one at a time | The dry run with its report commands before lock; one train; one catch-up at most | 13 |
| CI slows or a landing waits on serial runs | Stacks of four; the timing guard; the leg jobs in parallel; captures about 20 MB | 14 |
| A capture or live run judged on the wrong commit or reference | Captures name their commits and versions; live evidence on the landing commit; CI's push-time base | 15 |
| Briefs leave the warehouse questions open | The package template, checked by `brief_check.py` before each launch | 16 |
| A live resource is not ready for C-1 | Done before H0: the test schemas exist, Lakebase is running, the v26 endpoint is up (KO3-50 to KO3-53) | 16 |
| Tests run with Snowflake's account-admin role touch other teams' data | The write guard: tests write only to the dedicated test schema (KO3-51), and the client itself only reads | 19 |
| Private names or credentials leak into a committed capture or a log | Placeholders written before any file; the write fails on a name from the environment; Snowflake's env file loaded, never read or printed (KO3-24) | 19 |
| Three non-product live failures stop live work | Quota and capacity errors retried once by the runner (KO3-49) | 9 |
| The Java lane slows the Python path on an 8-CPU machine | `heavy` admission; Java's builds share the Rust cache; the forecast counts an hour of queueing (KO3-28) | 14 |
| The 118 forced-route failures hide more than three families | T3.8 starts on day 1 with the whole failing list; its exit is 0 failures | 6 |
| An older Databricks runtime does not run on a Graviton node | C-1's probe per runtime installs and imports the reference's aarch64 wheel; a runtime whose probe fails moves to the newest runtime line on Graviton, recorded and listed for the owner | 14 |

## Owner questions

**Answered** (KO3-18 to KO3-58): live access; Java and its questions; and on 2026-10-22 (night) the switch all at once with KO2-94's refusals kept until Phase 4 (KO3-36), old-engine tests on the old route until Phase 5 (KO3-37), the job matrix's questions (KO3-38), live timing in the end campaign (KO3-39), view YAML read as PyYAML reads it (KO3-40), the old client's sampling SQL on the warehouses and the rule "parity or better", better meaning a better error or more support (KO3-41), the comparison on the old client's recorded data gating (KO3-42), L3 on the warehouses (KO3-43), the three overnight refusals lifted (KO3-44), the offline judge, option (a) (KO3-45), one standing register class for what a user cannot see, whose entry T3.9 writes (KO3-46), the severity rule: only a worse difference blocks (KO3-47), the "no" list (KO3-48), and one retry of a quota or capacity error before it counts as a live failure (KO3-49).
The discovery run found a candidate for every resource the live-access answers point to; the names stay in the orchestrator's private notes.

**Live resources** (KO3-50 to KO3-53): Lakebase may be started and is running; on Databricks the tests read the existing AdventureWorks schema and write only to one dedicated test schema; on Snowflake they write only to one dedicated test schema, through a write guard, with the client's own logic read-only; the endpoint serves model version 26 and stays warm.
Also answered: the overnight decisions the simplified question named, confirmed under the rule of parity or better (KO3-57; the four it summarised as "four others" stand under the overnight delegation, below); protecting `rust-core` by a check that stops agents pushing it, which T3.0's command hook and pre-push hook gain in T3.0's next fix round (KO3-56); no bigger machine (KO3-55); and the Java packages as proposed (KO3-54).

**No question is open.**

**The orchestrator's decisions, listed for the owner's reading** (under the owner's overnight delegation, KO3-9; each reversible):
- aarch64 on Databricks (corrected in round 2): the aarch64 cells keep their runtimes on Graviton nodes, each proved by a probe job in C-1; a runtime whose probe fails moves to the newest runtime line on Graviton (KO3-26).
- The JAR's macOS natives (KO3-32) are the core's first macOS build, since Phase 2 built only the old engine on macOS and the client's macOS wheels stay in Phase 5; they reuse the macOS runners Phase 2's CI already has.
- Phase 2's overnight decisions not confirmed by name (KO3-57): "error wording" names no entry exactly, so P2D-2, P2D-3, P2D-4 and P2D-5 stand under the overnight delegation (KO2-88) and are on the owner's morning list; each is the old client's behavior or better under KO3-41.
