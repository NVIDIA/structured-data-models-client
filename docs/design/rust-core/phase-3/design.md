# Phase 3: S2 live, Databricks first, the Python switch, and Java's start

## Status

Draft for review round 2 (T3.1), written on 2026-10-22 from `origin/rust-core-phase-3` at 3ae4d1af (Phase 2 merged), against T3.0's process mechanisms as they stand at 36e0779f on T3.0's topic branch.
Round 1's findings, the owner's Java answer and the owner's answers to the Java questions are applied ([tracker.md](tracker.md), "Design review rounds").
The owner's answers so far are in [decisions.md](decisions.md), KO3-1 to KO3-35; the open questions are at the end of this page, all together, each with a recommendation, the day-1 ones first.
The acceptance checks below lock at T3.1's landing, after both reviewers approve one SHA and the day-1 questions are answered.
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

The plan is 14 landings: T3.0 and this design, ten Python packages, three Java packages; up to four Python packages build at once, as the machine allows (KO3-16), with Java on a fifth lane of its own (KO3-28).
The forecast from this design's landing is about 40 hours to the Python switch, which is also the phase's end: Java's packages land before it, and Java's live check runs in the same end campaign.
It assumes one fix cycle in the live end campaign, without which the switch lands at about H32.
The slip report compares the phase against the forecast as it runs and never stops work (KO3-12).
Five questions block day 1, because the early run needs them: where test tables may be written on Databricks, starting the stopped Lakebase instance, Snowflake's test role, the serving endpoint's model version, and approving the offline judge for the warehouse legs.

## Goal

Ship slice S2 for Python: `Schema.from_source` → `materialize` → `predict` live on Databricks (plain tables over a SQL warehouse and over Spark, and a metric view), Snowflake (plain tables and a semantic view) and Lakebase, including ranking and forecasting, with the Databricks job matrix across runtimes (migration-plan.md section 4, S2).
Then switch Python's S0, S1 and S2 paths to the core together, on the epic branch (plan section 4: "S0 (PQL validation), S1 and S2 paths switch together after S2, in Phase 3, so no call mixes old and new code").
Exit: the switch gate (plan section 7, "Python switch for a path"): S2's slice done, L6 green for its scenarios, the Databricks job matrix equal to the baseline, L7 within its limits with live timing; checked by the locked acceptance checks below.

Start Java (KO3-27; KO2-34 had deferred it): a Java library on the same core passing the same contract cases as Python on Java's scope (`phase-0/scope-java.yaml`, 66 paths: S0, S1 on DuckDB, SQLite and Postgres, and S2 on a Databricks SQL warehouse and a metric view, with binary, multiclass and regression questions), and agreeing with Python on the same Databricks data (plan section 4, S2; Phase 0 decision J2).
Java follows Python and never holds it back (plan section 2, decision 1): Java's own checks block only Java's packages, while a shared-core defect a Java leg shows blocks the path it breaks (plan section 6).

## Order of work

1. **Before the clock starts** (no product code):
   - T3.0 lands, and this design is reviewed and locked with the owner's day-1 answers; T3.1 lands stacked on T3.0;
   - the full dry run of the checking tools over [dryrun-plan.yaml](dryrun-plan.yaml) runs on T3.0's tree, with the runners, rust-host and the locked gate, and its gaps join T3.2's list;
   - one brief per package from [briefs/package.md](../orchestration/briefs/package.md), each passing `scripts/rust_core/brief_check.py`;
   - every warehouse test author (T3.4 to T3.7, and T3.13 for Java's Databricks cases) writes the input files of its S2 cases, so that every case kind in dryrun-plan.yaml has its cases before the early capture; S1's cases exist already.
2. **H0, four Python packages and the Java lane at once**: T3.2 (the tooling train), T3.3 (the warehouse harness), T3.4 (Databricks) and T3.8 (the forced-route failures), and T3.11 (the Java binding) on its own lane (KO3-28).
   T3.2 and T3.3 write the recording mode and the warehouse loader first; a review round by both reviewers on those commits approves them before the capture uses them (about H4.5).
3. **C-1, the early capture** (about H5 to H8), from that reviewed SHA: the one live run before the end; after it, no product package runs anything live (KO3-22).
4. **Stack 1** (T3.2 at the front, T3.3, T3.8) lands on one CI run at about H12.
   As slots free, T3.9 (differences and coverage) starts at about H6, T3.6 (Snowflake) at H7, T3.7 (Lakebase, live runners, reports) at H8, and T3.5 (views) at H10, on T3.4's candidate.
5. **Stack 2** (T3.4, T3.6, T3.7, T3.9) at about H20; **stack 3** (T3.5 and Java's T3.11 and T3.12) at about H23.
6. **Java** on its lane: T3.11 (binding, API, JAR, Java S0) from H0, T3.12 (Java S1 through JDBC) from about H9, T3.13 (Java on Databricks) from about H18, landing alone at about H29.
7. **T3.10**, the switch and the switch gate: the flip, the whole-phase review beside a first run of the live end campaign, the fixes, then the end campaign on the final candidate, Java's live check (E-7) included; it lands at about H40 and ends the phase.
   The end-of-phase reports for S0 and S1 (L3, the census, offline L7) run as soon as stack 2 has landed.
   Java never holds the switch: if T3.13 has not landed when the final end campaign starts, T3.10 lands without it, and T3.13 lands after with E-7 run on its own landing commit.

## Scope

1. **Databricks** (E27, E31: first): the `databricks` dialect in `sdm-data` for every request kind (describe, preview, scan, counts, key counts, time bounds, column ranges, repeated keys, entity sampling, neighbors, lookup), its catalog reader into the neutral `CatalogAnswer`, key sets bound as the old client binds them (`posexplode(from_json(?, ...))`), sampling as `TABLESAMPLE (n PERCENT) REPEATABLE (seed)` with the percentage printed as Python prints it (if the owner answers Q6 as recommended); the SQL-warehouse provider (`fetchall_arrow`) and the Spark provider (`spark.sql(...).toArrow()`, DBR 17.3 or later and serverless environment 4 or later, as the old client requires); the routes for every `databricks-sql` and `databricks-spark` path.
2. **Views**: a Databricks metric view and a Snowflake semantic view each described through its YAML definition into a view model the core reads without dialect terms (Phase 2's choice 12), with renamed and expression columns and a row filter carried on the relation, measures and metrics left out as conversion notes, as the old client does; the expression allow-list of the old client's `view_expr.py`, with `trust_expressions`.
3. **Snowflake**: the `snowflake` dialect (double-quoted names with the old client's upper-case rule, `NUMBER(p,s)`, keys bound as `TABLE(FLATTEN(PARSE_JSON(?)))`, `QUALIFY`, `::TIMESTAMP_NTZ`), its catalog reader (`DESCRIBE TABLE`, `SHOW IMPORTED KEYS`, `SHOW TABLES`), the semantic view's retrieval, and the provider (`fetch_arrow_all`, on a connector connection or a Snowpark session's connection).
4. **Lakebase**: a `PostgresSource` over a Lakebase connection on Phase 2's Postgres dialect, the `sdm_connectors.databricks.lakebase()` connect path, the `lakebase` routes, and Postgres reads qualified by the relation's schema part (P2-MID2-6).
5. **Ranking and forecasting on every warehouse leg**, through the decomposition Phase 2 moved into the core; nothing new in the core's operations.
6. **The contract bump, v28, once** (lesson 11), in T3.4 (choice 3).
7. **The job matrix**: the 21 cells of `phase-0/databricks-jobs.md` (J01 to J21: SQL warehouse, Spark and Lakebase; DBR 14.3 to 17.3 and serverless 1 to 6; x86_64 and aarch64, both in scope, KO3-26), each a one-time job installing the wheels under test (KO2-95's rule: the reference's for the baseline pass, the candidate's for the core pass), compared question by question against the owner's existing serving endpoint and metric views (KO3-21).
   CI's `native-wheel` already builds the aarch64 wheel (KO2-40); which Databricks runtimes run on aarch64 nodes in this workspace is probed once in C-1 (owner question Q3).
   The DBR 14.3 install revalidation Phase 1 carried (T1.35, E154) is cells J01 and J02.
8. **Both Databricks notebooks on the core**: `kumo_relational_quickstart.ipynb` (the AdventureWorks notebook of E27 and E31, which replaced the deleted one) and `lakebase_relational.ipynb`, once each in the end campaign.
9. **Java** (KO3-27), choice 10: the `sdm-jni` binding; a JAR built with Maven, with natives for Linux x86_64 and aarch64 and macOS arm64 and x86_64 (Phase 0 decision G13, KO3-32); Java 17 bytecode, tested on JDK 17 and 21 (KO3-34); a Java runner for the contract cases; Java S0; Java S1 on DuckDB, SQLite and Postgres through JDBC with the JDBC-to-Arrow conversion table and L5 for Java; Java on a Databricks SQL warehouse and a metric view through the Databricks JDBC driver, answering Phase 0's J2 questions live at the end against the same data as Python.
10. **The switch**: every S0, S1 and S2 default in `sdm_client/_routing.py` set to `core` in one commit (choice 8), the routing guard test changed with it, and the existing client test suites green on the default routes.
    `SDM_CORE_ROUTES` stays, test-only, until Phase 5 deletes the old engine.
11. **What the switch needs from Phases 1 and 2** (the carried rows, next section): the 118 forced-route failures, S1 coverage, the 12 unregistered user-visible differences, L3 and H24 on every database, DuckDB 2.0.
12. **The gate's live parts**: L6 on every warehouse leg, the job matrix, both notebooks, L7 live push-down timing (median of at least five interleaved runs, caches off, within 10% on Databricks and Snowflake; owner question Q4), and the held-out set's warehouse legs.
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
"No" items are the orchestrator's decision under KO2-52 (internal-tool hardening is P3 at most) and the owner's "product over gate machinery": they are reopened only when the dry run or a Phase 3 case hits them, and then go into T3.2 or the train (owner question G3).
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
| The overnight decisions the owner has not read: three refusals of input the reference answers (P2B-19, P2I-1, P2I-4) and P2D-2 to P2D-5, P2G-4, P2G-5, KO2-89 | OF-5 | T3.9, after owner questions Q9 and Q10 |
| **Contract rows for v28**, and the interfaces a Java host sees | P2-MID2-2, P2-T2.2-1, P2-T2.1b-1, P2-T2.8c-2, P2-MID2-3, P2-MID2-5, FX-6, P2-MID2-4, P2-WR-6 | T3.4 |
| Postgres reads qualified by the relation's schema part | P2-MID2-6 | T3.7 |
| Expression columns and metric views (KO2-25, KO2-39) | OF-4 | T3.4 (the contract fields), T3.5 (the views) |
| RS-6 on the warehouses | OF-3 | T3.4 and T3.6, after owner question Q6 |
| The live benchmark's "different requests" result fails the run | G8 | T3.7 |
| The held-out set's stored-NaN and `key_name_order` cases | OF-6 | HO3 |
| A bigger machine | OF-9 | Owner question Q12 |
| Branch protection on `rust-core`; `ci.yml`'s phase predicate | OF-7 | Owner question Q11 (protection); T3.3 (the predicate) |
| Judging rows Phase 3's work hits: owner approvals read by allowlist; the statement log on psycopg copy and stream and the warehouse drivers; a stale installed wheel; a stale `origin/main`; result rows hashed as text so a case's predictions are compared again | U4, P2-T2.1a-2, P2-MID2-7, P2-T2.0f-1, P2-T2.21-1 | T3.2 |
| Refusal maps citing quoted text, not line numbers (they broke four Phase 2 landings) | P2-T2.1b-6 | T3.7 |
| Time zones set per session and process | C16, C18, M5 | T3.3 (`ci.yml`, the vectors); C18's static check in T3.2's tests |
| CI rows: the Postgres numeric-spelling cases in CI and `SDM_REQUIRE_POSTGRES` on every Postgres step; `pytest-timeout` pinned; a forced rebuild; a stale comment; the skip step's `jq`; the slowest job's time if T3.0's split has not fixed it | P2-T2.14c-1, P2-T2.1a-3, P2-T2.16a-1, P2-T2.11-4, C28, P2-T2.16-14, P2-T2.1o-1 | T3.3 |
| Checked once in a lane environment, closed if they pass | ENV-2, ENV-3 | T3.3 |
| The test naming half `3a` as a half with no locked set | P2-T2.0a-1 | T3.1 (done: half `9a`) |
| Docs tidy: Phase 2's trackers, as-built and decision rows | P2-T2.16-1, P2-T2.16-2, P2-T2.1p-2 | T3.1 |
| Register text tidy | P2-T2.0c-3, P2-T2.8c-3, P2-T2.13b-1, P2-T2.21-2, P2-T2.14c-2 | T3.9 |
| Core hygiene, done by the package that next touches the file | U12, P2-T2.2-3 (`sdm-data`: T3.4); P2-WR-1, P2-WR-8, P2-WR-9, P2-T2.2-2, P2-T2.0c-1 (T3.8 or T3.9, whichever touches the file); P2-T2.7-1, P2-T2.7-2 (`sdm-testkit`: T3.3) | as listed |
| Covered by T3.0 | OF-1, P2-T2.1n-1 (the tests' part), ENV-1, P2-T2.0b-3, P2-T2.0g-1 (the reviewer's part) | T3.0 |
| **No**, until the dry run or a case hits them | FR P3-7, U3, U13, C2, C4, C5, C6, C7, C8, C9, C11, C22, G2, G4, G10, FX-2, M9, P2-WR-10, P2-WR-11, P2-T2.0-1, P2-T2.0b-1, P2-T2.0b-2, P2-T2.0b-4, P2-T2.0c-2, P2-T2.0e-1, P2-T2.0h-1, P2-T2.1a-1, P2-T2.1b-5, P2-T2.1b-7, P2-T2.1d-1, P2-T2.1e-1, P2-T2.1f-1, P2-T2.1g-1, P2-T2.1g-2, P2-T2.1h-1, P2-T2.1j-1, P2-T2.1j-2, P2-T2.1k-1, P2-T2.1k-2, P2-T2.1l-1, P2-T2.1m-1, P2-T2.4-1, P2-T2.7-3 | none |
| Phase 4, or Java's ranking once the owner adds it (placed there by owner rows or their reviewers) | P2-T2.5-1, P2-T2.5-2, P2-T2.5-3, FX-3, OF-8; P2-WR-2, P2-WR-5, P2-T2.14b-4 | none in Phase 3 |

L3 and H24 on all databases: L3 (`contract/tools/differential.py`) runs about 200 generated sources on each of DuckDB, SQLite and Postgres (KO2-26) and, if the owner answers Q8 as recommended, 20 on each of Databricks and Snowflake in the end campaign; H24, the core fed the reference's own fetched data, runs on Postgres locally (P2-T2.16-7) and on every warehouse leg by construction, since the offline judge serves the reference's recorded tables and random draws (choice 2).

## Architecture choices new in this phase

Phase 1's and Phase 2's mechanisms are reused as built: the contract and its lock, the data port and capabilities, `Relation`, runners and cells, comparator, ratchet and checkpoint, register, coverage, gate, the provider kit, the bridge freeze, the routing seam, the evidence pack; and T3.0's stacks, tooling train, handoff stamp, CI split and timing guard, lane environments, command hook, dry run and slip report.
Every new interface is held to KO2-46: only what this phase uses, as plain data a Python or Java host builds without language idioms.

### 1. Warehouse legs are database legs of the existing case format

**Recommendation:** add four warehouse legs, `databricks-sql`, `databricks-spark`, `snowflake` and `lakebase`, keyed `<case>@<leg>` exactly as Phase 2 keyed `<case>@sqlite`.
A case's data is still its base and setup (DuckDB statements), loaded into a dedicated test schema of each warehouse through Arrow by Phase 2's contract loader (`contract/runners/databases.py`), extended with a type map per engine: Databricks and Snowflake have no unsigned integers, no time-of-day type in tables, `DECIMAL` up to 38 digits and their own timestamp kinds; a type an engine cannot store is listed, and its cell is never counted as covered (KO2-47's rule).
Every S1 acceptance case runs on every warehouse leg whose engine stores its data, by the same default that runs it on DuckDB, SQLite and Postgres, so no S1 case file changes (the owner, KO3-23: "Can we replicate whatever dataset/testing you've done in phase 2 here?").
The two cases that build 100,001-row tables to test the core's call batching (`multicall.entities.100000` and `.100001`) are left off the warehouse legs, which add nothing to what they test; that is an exclusion, so it is in owner question D5.
An S2 case names its warehouse legs in a new `warehouses` field and runs only there, with no exclusion; the metric and semantic view cases carry a new `input.view` (the kind, the name and the YAML, or the name of an existing view), and the connect paths `input.connect` (`warehouse`, `spark`, `snowpark`, `lakebase`).
AdventureWorks is loaded the same way, and its cases ask the quickstart notebook's 5 questions (3 binary, 1 multiclass, 1 binary through a metric view), the deleted notebook's 8 (regression twice, forecast, binary, ranking, multiclass, and two through a metric view), and a ranking and a forecast on each leg, since the quickstart alone "is not enough for testing" (KO3-23).
Why: every S1 case already has reference lines and core observations on three databases, so a warehouse leg reuses the comparator, ratchet and coverage, and one case set covers offline and live.
Rejected: separate S2 case files over a separate live dataset (a second corpus to keep in step); live-only cases (a live run per bug).

### 2. The offline judge: the old client's answers, a local engine over the warehouse's own tables

Round 1 found that the first draft's judge, answering the core's SQL from a capture by matching statement text, could not judge a new dialect before the end, and made SQL text a gate where Phase 1's H9 (owner-approved) keeps it diagnostic.
**Recommendation:** judge a warehouse leg offline by its answers, with the data a warehouse would give, and judge the dialect's SQL by goldens and once live:

- **What C-1 records**, per case and leg, from the old client running live: its public outcome as a reference line (schema, payload, predictions, warnings, refusal), in the observation format Phase 2's lines use; every table of the case as the warehouse returns it through the leg's driver (Arrow, with the warehouse's own column types); the catalog's answers (key declarations, column types, view YAML); every random draw the old client's sampler made (the table, the seed, the fraction and the rows it picked); and, as diagnostics only, every statement it sent with its parameters and result.
  Names of catalogs, schemas, hosts and users are replaced by placeholders before anything is written, and the write fails on any private name from the environment.
  A capture file carries a format version (`sdm-capture/1`), the recording commit, the reference commit, and each leg's runtime, Python, pandas and driver versions.
- **The stand-in provider** answers the core's typed data requests for a warehouse leg offline: a local DuckDB loaded with the leg's recorded tables, with their recorded Arrow types, rendering through the DuckDB dialect and declaring the warehouse dialect's capabilities; catalog requests answered from the recorded catalog; and every entity-sampling request answered with the old client's recorded draw for that table, in order (H24's common inputs), since a local engine cannot reproduce a warehouse's `TABLESAMPLE`.
  For the `lakebase` leg the stand-in is the local Postgres 17 CI already runs, loaded with the recorded tables, so the core's real Postgres SQL runs offline.
  python-core uses `contract/runners/standin.py` and rust-host `sdm-testkit`'s stand-in, both built in T3.3.
- **What gates offline**: the core's public outcome on the leg against C-1's reference line, field by field, under the same comparator and register as every other leg.
  On a stand-in leg the statement count and each statement's result are diagnostic, since the statements are the stand-in's, not the warehouse's; a different number of random draws than the old client made fails by name, because the draws decide the sample a user's answer comes from.
  That narrows H9's statement rule on stand-in legs only, so it is owner question D5; H9 applies in full on every live leg.
- **The dialect's SQL** is judged by goldens per request kind, reviewed against the old client's recorded statement for the same request (a non-gating diff in the package's evidence), and executed once live in the end campaign (E-1), where H9 applies in full: the same answers, the same statement count, each statement's result.
- **What this cannot judge before the end**: whether a warehouse accepts the dialect's SQL and returns what the stand-in returned; the end campaign judges it, as the owner chose (KO3-22), and the forecast holds one fix cycle for it.
  A false pass needs the stand-in and the dialect to differ from the warehouse in exactly compensating ways and survive E-1; a stand-in that answers differently from the warehouse shows as a difference from the old client's answers, which is a failure, not a pass.

Rejected: matching the core's statements to recorded ones (the draft's choice: it cannot answer a statement the old client never sent, and the inherited Postgres dialect already sends one, `rust/crates/sdm-data/src/postgres.rs`'s relation-identity predicate); a warehouse emulator by transpiled SQL (it judges SQL nobody runs); a live check per dialect mid-phase (cheap, about 20 minutes per warehouse, but against KO3-22; offered as owner question D5's option b).

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
Each hands Arrow to the core with no pandas round trip and passes the provider kit on the stand-in and once live in E-1.
Authentication, connection objects, Lakebase's hour-long database credential and backend detection stay in the host: they are language objects the core cannot compute (lesson 1).
Backend detection learns to tell a SQL warehouse, a Spark session and a metric view apart; both Databricks connections report `databricks` today.

### 6. Views: a crate for YAML, a port for the rules

**Recommendation:** each dialect retrieves a view's definition with the old client's statement (Databricks: `DESCRIBE TABLE EXTENDED`'s YAML; Snowflake: `SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)`) and hands the text to `sdm-schema`, which parses it with a YAML crate (`yaml-rust2`, maintained and pure Rust) and interprets it into a view model with no dialect terms: tables, renamed and expression columns, joins as links, the filter, and measures or metrics as conversion notes.
YAML values are read as the old client's PyYAML reads them (YAML 1.1: `yes`, `on` and `off` are booleans where PyYAML makes them so) for the keys the view model reads, by PyYAML's documented implicit-type rules ported as a defined algorithm over the crate's plain scalars (lesson 3; owner question Q5).
The expression allow-list is a port of `view_expr.py` (1,744 lines), with `trust_expressions` as the old client has it; the first attempt's 2,150-line hand-written YAML reader is not ported.
The existing metric view over AdventureWorks and the twin metric and semantic views over a small shop dataset (KO3-21; their names stay private) are recorded in C-1 and used as cases beside views the cases create in the test schema.
A `phase-1/dependencies.md`-style row records the crate choice and its measured reason (lesson 3).

### 7. Sampling and session state on warehouses

**Recommendation:** match the old client's SQL: Databricks `TABLESAMPLE (p PERCENT) REPEATABLE (seed)` with the percentage written as Python's `str(float)` writes it, Snowflake's sampling statement as the old client sends it, and `ORDER BY` and `QUALIFY ROW_NUMBER()` forms unchanged, so the same statement draws the same rows on the same table (owner question Q6, RS-6's coverage).
The session time zone is fixed per capture and named in it; a zoned time column on a warehouse is a case of its own on each leg, crossing the UTC instant and the session wall clock (lesson 6's "every reading").

### 8. The switch, in one commit

**Recommendation:** T3.10 sets every S0, S1 and S2 path's default to `core` in `_routing.py` at once, so no call mixes old and new code (plan section 4), and changes the guard test that holds the defaults (`test_the_routing_seam.py`) in the same commit.
Inputs the core still refuses by name on a switched path are only the three KO2-94 keeps, if any of them is reachable from S0, S1 or S2 paths at all (T3.2's census measures it); they stay refused by name on the epic branch until Phase 4 lifts them, before the epic can merge into `main` (owner question Q1).
Client tests that test old-engine internals rather than public behavior are marked to run on the old route only until Phase 5 deletes the engine, each listed in the evidence (owner question Q2); every other client test must pass on the default routes.

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
  A Java leg runs exactly when the candidate builds the JAR (Phase 2's staging rule), and once its baselines are admitted the ratchet's "baselined, and not observed" rule stops a later candidate dropping it.
Rejected: a hand-written C interface with a Java FFM layer (plan section 3 chose `jni-rs`); generating Java from the Python API (Java idioms differ, and the Python API carries pandas).

## The live-test plan

Live work runs only under the owner's live rules: one-time jobs only, uploads only to the authorized volume, no existing notebook or cluster touched, only the Lakebase instance the owner names, and three live failures for non-product reasons stop live work (plan section 9; owner question G4).
The owner's shape (KO3-22): code against the offline judge, and run the live tests at the end.
So there is one live run at the start (C-1) and one live campaign at the end (E-0 to E-7), run on the commit that lands.
Nothing account-specific enters the repository: the profile, the volume, the catalog and schemas, the warehouses, the Lakebase instance, the endpoint and the views come from the environment and the orchestrator's private notes; Snowflake's credentials come from the owner's env file, loaded into the run's environment and never read or printed (KO3-24).
Wheels: C-1 and E-2's baseline pass install the reference's wheels; every other live run installs the candidate's own wheels, built by CI for x86_64 and aarch64 (KO2-40, KO3-26).

**Offline, from C-1** (seconds per case, no account): every warehouse leg of every case on python-core and rust-host, on the stand-in (choice 2); the dialects' goldens with their diff against the recorded statements; view interpretation from the recorded YAML; the provider kit on the stand-in; Lakebase's whole data path on local Postgres; offline L7 over the stand-in legs.

**Live, at the end**: that the engines accept the core's SQL and give the old client's answers; `TABLESAMPLE` repeatability; the connect paths (`warehouse()`, `lakebase()`, Snowpark) and the serving endpoint; both notebooks; the job matrix; live timing; the held-out set's warehouse legs; Java's Databricks questions.

| Live run | When | What runs | Jobs or sessions | Wall time (forecast) |
|---|---|---|---|---|
| C-1 The early capture | once, about H5 to H8, from the reviewed recording SHA | Phase 2's bases and AdventureWorks loaded into each warehouse's test schema; every S1 acceptance case and every S2 case on the old client on each of the four legs, recorded (choice 2); the existing views' YAML; the provider kit's fixture; one small aarch64 job per runtime the matrix needs, installing nothing but Python, to learn which runtimes run on aarch64 nodes here (Q3) | SQL warehouse and Snowflake: sessions from this machine; Spark and Lakebase: one one-time job each; about 6 tiny aarch64 probe jobs | about 3 h, the legs in parallel |
| E-0 Reference top-up | the end campaign, first | The old client on every case added or changed after C-1, and on every case if `main` moved | As C-1 | 0.5 h (2 h if `main` moved) |
| E-1 The core on every warehouse case (L6) | after E-0 | Every case on python-core live on each leg, recorded as the core's capture, compared with E-0 and C-1 under H9 in full; the provider kit once live | As C-1 | about 2 h, in parallel |
| E-2 The job matrix, both passes | after E-1 | 21 cells: the 15 Unity Catalog cells ask the quickstart's 5 questions and the deleted notebook's 8 (195 runs per pass), the 6 Lakebase cells the 6 base questions (36 runs); once on the reference's wheels, once on the candidate's; against the owner's existing endpoint and views | 42 one-time jobs, in parallel | about 1.5 h |
| E-3 Both notebooks on the core | beside E-2 | `kumo_relational_quickstart.ipynb` and `lakebase_relational.ipynb` on the candidate's wheels | Two one-time jobs | about 1 h |
| E-4 Live L7 timing | after E-2, on quiet warehouses | Push-down median over at least 5 interleaved runs per scenario, caches off (`use_cached_result` off on Databricks, `USE_CACHED_RESULT` off on Snowflake), the core against the old client | Sessions from this machine | about 1.5 h |
| E-5 Held-out warehouse legs | inside E-1's window, once | HO3's warehouse cases: the old client and python-core live, recorded; rust-host on the stand-in loaded from E-5's own recording | One session or job per leg | about 1 h |
| E-6 L3 on warehouses (if Q8 is yes) | inside E-2's window | 20 generated sources per warehouse, both routes | One session or job per warehouse | about 1 h |
| E-7 Java on Databricks | inside E-2's window, on the same candidate | Phase 0's J2 questions, the four base questions and the two metric-view questions (KO3-35), through the Databricks JDBC driver on the SQL warehouse, compared with Python's answers on the same data in E-1 and E-2 | A session from this machine, and one job on DBR 17.3 (Java 17) running the JAR on x86_64 and on aarch64 | about 45 min |

The end campaign's serial chain is E-0, E-1, E-2 with E-3, then E-4: about 5 hours.
It runs twice in the forecast: once on the first switched candidate, beside the whole-phase review, and again on the final candidate after their fixes, since live evidence counts only for the commit that lands (lesson 15); any change after the final run under `rust/`, `packages/` or `contract/` runs it again.
If the first run and the review find nothing, the first run is the evidence and the switch lands about 8 hours earlier.
Cost: about 60 one-time jobs (2 for C-1, about 6 aarch64 probes, 84 matrix jobs over two campaign runs counted once if the first run is the evidence, 4 notebook runs, 2 to 3 for the held-out set and L3, 2 for Java), each 10 to 20 minutes on a single-node cluster or serverless; the authorized SQL warehouse for about 10 hours of sessions in all, auto-stopping; Snowflake's smallest warehouse for about 6 hours; the owner's always-warm endpoint during the matrix and notebook runs.
The orchestrator cannot see prices; the owner can turn these hours into money.

## Packages

M is mechanical (a port with reference recordings), N novel, R risky.
"Judging" means the package changes a path of `scripts/rust_core/judging-paths.txt`; only T3.2, the tooling train, does, and a train may change only judging paths, their tests and docs (T3.0's `judging.py`).
Every product package has a separate test author who writes its cases first; the warehouse cases' reference lines come from C-1, and its pending-live cases' from E-0.
Every package runs `handoff.py` before review and ports the old client's rule in each fix (`Rule:` lines).
Lanes: A head and tooling, B Databricks, C views, D Snowflake, E switch-ready, F differences and coverage, G Lakebase and live runners, H harness, J Java.
Hours are implementation plus two review rounds, Phase 2's measured median for planned work once its overhead is removed.

| ID | Package | Output | Exit check | Depends on | Test author | Lane | Stack | Kind, hours |
|---|---|---|---|---|---|---|---|---|
| T3.0 | Process mechanisms | Stacks, the tooling train, the handoff stamp, CI's split and timing guard, lane environments, the command hook, the dry run, the slip report | Its own (T3.0's brief) | – | T3.0's | A | 0 | landing now |
| T3.1 | This design | design.md, decisions.md, tracker.md, locked-set.yaml, forecast.yaml, dryrun-plan.yaml; learnings.md rewritten; `briefs/package.md` and `scripts/rust_core/brief_check.py`; `tests/test_learnings_enforced.py`; the design reviewer's brief and handbook on the new lesson numbers; the test naming half `3a` moved to `9a`; docs tidy rows | Two APPROVEs on one SHA; the locked-set reader reads half `3a`; `brief_check.py`'s and the learnings check's focused tests each fail with their rule removed | T3.0 | T3.1's | A | 0 | docs and a small check |
| T3.2 | The tooling train (judging paths only) | Case format: `warehouses`, `input.view`, `input.connect`, both loaders (`run_python.py`, `case.rs`) and malformed vectors, P2-T2.1i-1 aligned; the recording mode in `run_python.py` (choice 2) and its completeness in `capture_check.py`; `<case>@<leg>` keys and the runner `java` in the observation schema, `observation.rs`, `compare.py` and `ratchet.py`; H9's diagnostic statement rule on stand-in legs in `compare-policy.yaml` (if D5 is yes); `coverage.py`: legs claim their paths, client tests may claim a path, the S1, S2 and Java rules; the census's real-data mode per leg; `register.py` refusing an `approved_by: owner` entry with no owner-approved row and U4's allowlist; `lane.py start --brief` running `brief_check.py`, and refusing a package ID with a letter from Phase 3 on; `gate.sh` running the stand-in and Java legs; P2-T2.1a-2, P2-MID2-7, P2-T2.0f-1, P2-T2.21-1; every GAP of the full dry run | Each change's one focused test fails without it; every stub in dryrun-plan.yaml passes the dry run with no GAP; `land.py --train` accepts its file list | T3.1 | T3.2's | A | 1, front | N, about 6 h (recording first, about 2.5 h) |
| T3.3 | The warehouse harness | The loader into the four warehouses with each engine's type map (`databases.py`, `contract/type-map-vectors.json`); the stand-in provider for python-core (`contract/runners/standin.py`) and rust-host (`sdm-testkit`); the capture writer's placeholders, its format version and converter; the C-1 and E-0 job templates (`scripts/live/`); the per-leg CI jobs (`contract-python (ref-pandas3, <leg>)`, `cross-host (<leg>)`) and the CI rows of the carried table; P2-T2.7-1, P2-T2.7-2 | A private name in a capture fails the write; a stand-in with one recorded table dropped fails the case by name, and with one random draw dropped fails "draw not recorded"; the converter reads a version-1 capture; CI's leg jobs run the cases they name | T3.1 | T3.3's | H | 1 | N, about 6 h (loader first, about 3 h) |
| T3.4 | Databricks and contract v28 | Choice 3's bump; the `databricks` dialect and catalog reader; the SQL-warehouse and Spark providers; routes and backend detection for `databricks-sql` and `databricks-spark`; RS-6's percentage text; U12, P2-T2.2-3 | Every case's `databricks-sql` and `databricks-spark` legs equal to C-1's reference lines on python-core and rust-host on the stand-in; the goldens of every request kind, with their diff against the recorded statements reviewed; the key-type pair matrix on Databricks' recorded tables; the provider kit on both providers on the stand-in; schema lock and both loaders at v28; pending-live cases listed | T3.2, T3.3 (stacked on their candidates); C-1 | T3.4's | B | 2 | M/R, about 10 h |
| T3.5 | Views | The view model in `sdm-schema`, the YAML crate with PyYAML's 1.1 reading, the expression allow-list port, metric view retrieval and interpretation, semantic view interpretation; its dependencies row | Every metric-view case on `databricks-sql` and `databricks-spark` equal to C-1's reference lines on the stand-in; semantic-view interpretation from the recorded YAML equal to the old client's schema; YAML 1.1 value vectors (`yes`, `on`, `off`, `~`, octal-looking text) read as PyYAML reads them | T3.4 (v28's relation fields; stacked on its candidate) | T3.5's | C | 3 | M/R, about 9 h |
| T3.6 | Snowflake | The `snowflake` dialect and catalog reader, the semantic view's retrieval, the provider (connector and Snowpark), routes | Every case's `snowflake` leg and every semantic-view case equal to C-1's reference lines on both hosts on the stand-in (the semantic-view cases once T3.5's interpreter is under it); goldens reviewed against the recorded statements; the key-type pair matrix and a session-zone case on Snowflake's recorded tables; the provider kit on the stand-in | T3.4; T3.5 for the semantic-view cases; C-1 | T3.6's | D | 2 | M, about 8 h |
| T3.7 | Lakebase, the live runners and the phase-end reports | The `lakebase()` connect path and routes on the Postgres dialect; reads qualified by schema (P2-MID2-6); the matrix runner and job template, both notebooks' driver, installing the wheels under test on each cell; live timing in `contract/tools/l7.py --live`; G8; `differential.py`'s database axis and classes; `evidence.py`'s per-leg table; the local H24 tool; refusal maps citing quoted text | Every case's `lakebase` leg equal to C-1's reference lines on local Postgres; the matrix runner's dry mode prints 21 cells' job specs, and a cell installing a published wheel instead of the one under test fails; `l7.py --live --dry` prints the interleaved plan with caches off, and a seeded slow scenario fails its report; L3 runs on SQLite and Postgres; the per-case table shows every leg | T3.2, T3.3; C-1 | T3.7's | G | 2 | M, about 8 h |
| T3.8 | Switch-ready: the forced-route failures | The 118 failures fixed in the core or the bridge's wording, each family ported from the old client's rule (the time-column warning, key counting and its refusals, Postgres refusals and the described-schema `RouteError`); the validator chosen in the core (P2-MID2-1); the bridge declarations its bridge edits need | `pytest packages/structured-data-models-client/tests` with every S0 and S1 route forced: 0 failures; a case per family on three databases | T3.1 | T3.8's | E | 1 | R, about 8 h |
| T3.9 | Differences, coverage and DuckDB 2.0 | The 11 user-visible rows matched or registered; OF-5 per Q9 and Q10; acceptance cases for every S0 and S1 path and cell coverage still finds empty; the S1 test rows; P2G-1 widened; DuckDB 2.0 supported, its CI cell, and the `zip` exceptions dropped if its build allows; the interim edit gate retired; register tidy | `coverage.py` S1 rule (T3.2) with no empty path or cell; each row's case fails before and passes after; CI green on DuckDB 1.5 and 2.0 | T3.2 (coverage); T3.8 for the shared files | T3.9's | F | 2 | M, about 9 h |
| T3.10 | The switch and the switch gate | Choice 8's flip and guard test; the old-route-only list; the S2 case IDs added to half `3a` (the one planned change to the locked set); register reconciled; `main` merged if it moved; the locked run; the end-of-phase reports; the whole-phase review; the end campaign E-0 to E-7 on the final candidate; the core's live capture committed for rust-host and CI; HO3 once; the evidence pack; the packet | The locked checks below | all Python packages; the Java packages that have landed by its final candidate | T3.10's (the guard test and the switched suites) | A | alone | R, about 17 h on the critical path |
| T3.11 | Java binding, API and Java S0 | `sdm-jni`; the Maven build and the JAR with natives for Linux x86_64 and aarch64 and macOS arm64 and x86_64; the Java API of choice 10 and its exception mapping; the SLF4J bridge; `contract/runners/run_java.py` and the Java runner program; CI jobs building the JAR and running the Java legs (JDK 17 and 21 on x86_64, JDK 21 on aarch64 and macOS); a panic test and a JAR-load test in a clean container | Every S0 case on the Java host equal to the reference; a panic inside the core reaches Java as an exception and leaves the JVM running; the JAR loads on every JDK and platform listed; L0 admits `jni` only in `sdm-jni` | T3.2 (the `java` runner name; stacked on its candidate) | T3.11's | J | 3 | N, about 9 h |
| T3.12 | Java S1 through JDBC | JDBC providers for DuckDB, SQLite and Postgres; the JDBC-to-Arrow conversion table; L5 for Java; the model transport over `java.net.http` | Every S1 case on Java's 46 S1 paths equal to the reference on each database on the Java host; L5: the same tables through Java's and Python's drivers reach the core as identical Arrow, and a seeded unit or scale change in one JDBC path fails; `coverage.py`'s Java rule with no empty S0 or S1 path | T3.11 (stacked on its candidate); T3.8, T3.9 (the shared-core fixes) | T3.12's | J | 3 | M/N, about 9 h |
| T3.13 | Java on Databricks | The Databricks JDBC provider, the metric-view route through the same connection, the serving transport, Java's S2 routes; E-7's runner | Every Java S2 case's `databricks-sql` leg equal to C-1's reference lines on the Java host on the stand-in; E-7's dry mode prints the questions and the JDBC session plan; E-7 itself runs in T3.10's end campaign (check 14) | T3.12; T3.4, T3.5 (the dialect and the views) | T3.13's | J | alone | M, about 7 h |
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
| Stack 2: T3.4, T3.6, T3.7, T3.9 | H20 | 8 h | T3.4 judged on the stand-in from C-1's end (H8), approved about H13; T3.6 and T3.9 about H18; T3.7 about H19; CI about 1 h |
| Stack 3: T3.5, T3.11, T3.12 | H23 | 3 h | T3.5 written from H10 on T3.4's candidate, approved about H22; T3.11 approved about H12 and T3.12 about H21, waiting for the stack |
| T3.13 alone | H29 | 6 h | Written from H18, its metric-view part on T3.5's candidate, approved about H28 |
| T3.10 | H40 | 11 h | The flip and the switched suites (H23 to H25); the first switched candidate's wheels (H26); the end campaign's first run (H26 to H31) beside the whole-phase review (H25 to H29), with the hour of fifth-lane queueing on this path; fixes (H32 to H34); the final candidate's wheels (H35); the end campaign on it, E-7 included (H35 to H40) |

**Critical path:** T3.1 → T3.2's and T3.3's recording commits and their review (4.5 h) → C-1 (3 h) → T3.4 on the stand-in → stack 2 → T3.5 → T3.10's flip, review and two end-campaign runs: about 40 hours to the Python switch, which ends the phase.
Java is off that path: its lane runs beside the Python slots from H0, and its last package lands about 11 hours before the switch.
Before H0: T3.0's landing, this design's review rounds and the owner's day-1 answers, the S2 test authors' case files, and the full dry run.

This forecast misses the owner's one-day goal by about 16 hours.
The round-1 draft's 28 hours double-booked build slots, under-counted the end campaign, and ran live evidence on a commit before the one that lands; this one does not.
With the fifth lane, Java ends inside the phase rather than after it; with only four slots it would have ended about 3 hours after the switch.
What would shorten it, in order: the end campaign passing on its first run with nothing for the whole-phase review to fix (8 hours); the mid-phase live check of each dialect's SQL (owner question D5, option b), which makes that first-run pass likelier; a bigger machine (Q12), which would remove the fifth lane's queueing (about an hour).
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

- Kickoff: the questions below, on a readable page, all at once; the day-1 questions first.
- Twice a day, a five-line check-in with the slip report's line, the pending-live count, the register entries reviewers cannot approve, and any overnight decision written plainly when made (KO3-9).
- One design checkpoint when stack 2 has landed (half the planned landings), and the whole-phase review in T3.10, both by both reviewers.
- End: the go or no-go packet (handbook, "Ending a phase", step 6), Java included.

## The evidence pack

As Phase 2's, on the final SHA, generated by the branch's own runs, plus:
- the per-case table by database and warehouse leg, by host, Java included;
- the captures' index (cases, legs, tables, draws, the redaction check, each leg's runtime versions);
- E-1 against E-0 and C-1 per leg; both notebooks' runs; the job matrix's two passes, question by question; live L7; the held-out run; E-7 beside Python's answers;
- the switch: the routing defaults, the client suites on the default routes, and the old-route-only list;
- L3 per database, the census with real-data counts on every leg, offline L7 on every leg, coverage S1, S2 and Java;
- the host-held list before and after (choice 9);
- Java: L5 for Java, and the JAR's load test per JDK and architecture.

## Gates

### Merge rules

As Phase 2's, with T3.0's mechanisms: packages land in stacks of up to four on one CI run (`land.py --stack`, `stack-land`; KO3-13), a cancelled run rerun as T3.0's rule says; judging changes only in T3.2 or one catch-up train; two reviewers on every product round, both approving the same SHA (KO2-9), starting only on a SHA with a handoff stamp; on docs-only fix rounds the reviewer who raised the finding confirms its fix, and rounds writing decision rows or register entries keep Claude (KO3-5); a finding family blocking twice gets a half-page rule note, not a redesign session (KO3-3).
Package IDs are integers, T3.0 to T3.13; register IDs come from per-lane blocks (`P3A-` for lane A, `P3B-` for lane B, and so on); each package writes its own tracker file; decisions.md takes only the owner's rows.

### Locked acceptance checks

Locked at T3.1's landing; the one planned change is T3.10 adding the S2 case IDs to half `3a` from the S2 case files then landed, of the kinds in dryrun-plan.yaml; any other change is the owner's (plan section 9).
Phase 3's locked set is [locked-set.yaml](locked-set.yaml), one half, `3a`, run as `gate.sh --locked 3a`: the 309 S0 and S1 acceptance cases at the lock (19 of the 24 S0 cases and 290 of the 291 S1 cases; the other six are diagnostic), each S1 case on every database and warehouse leg its file and its engines' type maps give, steps `inventory`, `seeded` and `main`, checks 1, 2, 7 and 10 held by gate steps, and 12 the every check.
`gate.sh`'s reader reads it (run on this draft: it lists the half's steps and 309 cases, equal to today's acceptance set).
The rest is judged outside the gate, as Phase 2's KO2-76 set it: CI-green parts by `land.py`'s exact-SHA rule, reports read by both reviewers, and the live and held-out runs once in T3.10, on the commit that lands.

| # | Check | Tool and machine | Fails on |
|---|---|---|---|
| 1 | Full L2 parity for S0, S1 and S2: every case on each database and warehouse leg it runs on, per field, on python-core and rust-host against the reference, the warehouse legs on the stand-in (choice 2: public outcome gating, statements diagnostic) and H24 by construction; every difference has an approved register entry; coverage S1 and S2 with no empty path or cell; no case pending live at the final run | `gate.sh --locked 3a`; CI: `contract-python (<cell>, <database or leg>)`, `cross-host (<database or leg>)`; report: coverage | A changed answer on one leg; a case missing on one leg; a random draw the old client did not make; an empty coverage cell |
| 2 | No unapproved refusal of an input the reference accepts, and no panic, on any leg; the census's real-data mode (T3.2) over AdventureWorks at full size and RelBench H&M on DuckDB, SQLite and Postgres and over C-1's AdventureWorks tables on each warehouse leg's stand-in, with per-leg counts; a kept rule reachable on a switched path only under an owner answer | `ratchet.py` in the gate; report: `refusal_census.py --real-data`, reference machine | A refusal with no entry; a kept rule hit on a switched path with no owner answer; the census's control, a seeded refusal on a real source |
| 3 | L4: SQL goldens executed on DuckDB, SQLite and Postgres (Lakebase's dialect is Postgres's); the Databricks and Snowflake goldens reviewed against the recorded statements, and executed live in E-1 with H9 in full | CI: `l4-goldens (<database>)`; report: E-1 | A dialect change with no golden update; a statement the engine refused or answered differently live |
| 4 | L5: each warehouse driver's Arrow types, as recorded, reach the core without loss; the type maps list what each engine cannot store | CI: `l5-conversion*`; report | A unit or scale change in one driver's path |
| 5 | The provider kit: every built-in provider, the three new ones on the stand-in and once live in E-1 | CI: `rust-test*`, `contract-python (ref-pandas3, <leg>)`; report: E-1 | A provider that truncates a page or fails a capability silently |
| 6 | L3: about 200 generated sources on each of DuckDB, SQLite and Postgres, and 20 per warehouse if Q8 is yes, every divergence classified | Report: `differential.py` (T3.7), reference machine, and E-6 | An unclassified divergence |
| 7 | The reference and the phase branch at `main`'s head as of the gate run, the reference rerun, the warehouse reference lines re-recorded in E-0 if it moved, every changed line traced to a `main` commit | `ratchet.py`, the `main` step | `main` moved and not merged; a changed line with no commit |
| 8 | The switch: every S0, S1 and S2 default is `core`; the client suites pass on the default routes, except tests listed old-route-only under Q2; rust-host runs every S0, S1 and S2 case; the host-held list matches choice 9 | CI: `test (3.x)`, `installed-wheel`, `cross-host (*)`; `rust-lint` (the bridge freeze) | One path left `old`; a client test failing on the default route; a test marked old-route-only that tests public behavior |
| 9 | Held-out (HO3) under H11's threshold on python-core and rust-host, warehouse legs included (E-5), and on the Java host for its Java cases | `phase_end.sh`'s held-out step and E-5, once | Before the one shot, two dummy bundles (a panic; differences over the threshold) each a no-go |
| 10 | Boundary and packaging: L0 clean, no `pyo3` in the pure-Rust host's tree, `jni` only in `sdm-jni`, dialect words only in `sdm-data`, the wheel installing on x86_64, aarch64 and each runtime cell | CI: `rust-lint`, `rust-test*`, `wheel-smoke (*)`, `runtime-install (*)` | Phase 1's seeded L0 faults; a dialect word in `sdm-schema` |
| 11 | L7: offline within KO2-29's limits on DuckDB, SQLite, Postgres and every warehouse leg's stand-in (client CPU, which the stand-in measures as well as a warehouse); live push-down median within 10% of the old client's on Databricks and Snowflake, at least 5 interleaved runs, caches off (Q4) | Report: `contract/tools/l7.py` offline, and `l7.py --live` (T3.7) in E-4 | A scenario past 1.1 times; a live median past 10%; the live report's control, a seeded slow scenario |
| 12 | Every gate step the half runs passed | `gate.sh --locked 3a` | Any one step seeded to fail |
| 13 | L6: every case live on each warehouse leg on the core equal to the old client's live answers (E-1 against E-0 and C-1, H9 in full); both notebooks on the core; the 21-cell job matrix on the core equal to its reference baseline, question by question, on x86_64 and aarch64 cells; all on the commit that lands | Report: E-1, E-2, E-3 | A question answered differently in one cell; a cell that does not install the wheels under test; evidence naming another commit |
| 14 | Java (KO3-27): every S0 and S1 case on Java's paths on the Java host equal to the reference on DuckDB, SQLite and Postgres, and Java's `databricks-sql` legs on the stand-in; L5 for Java; the JAR loads on JDK 17 and 21 on Linux x86_64, and on JDK 21 on Linux aarch64 and macOS arm64 and x86_64 (KO3-32, KO3-34); E-7's answers equal to Python's on the same data, run with the flags of choice 10 (KO3-35). Judged for Java's packages and the phase's end; a Java-only failure never holds a Python package, and a shared-core defect a Java leg shows blocks the path it breaks (plan section 6) | CI: `java (<jdk>, <platform>, <database or leg>)`; report: L5 for Java, E-7 | A Java leg differing from the reference; a seeded unit change in one JDBC path; a JAR missing one architecture's native |

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
| The end campaign finds real-engine problems all at once | Dialects port the old client's statements, which the engines ran in C-1; goldens reviewed against them; the recorded types feed the stand-in; one fix cycle and a second campaign run in the forecast; owner question D5's option b moves the SQL check earlier | 6, 12 |
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
| A test schema, the Lakebase instance, Snowflake's role or the endpoint is not ready for C-1 | Day-1 questions D1 to D4; the discovery already found each candidate (privately) | 16 |
| Tests run with Snowflake's account-admin role touch other teams' data | A least-privilege test role (D3); otherwise every statement confined to the test schema | 19 |
| Private names or credentials leak into a committed capture or a log | Placeholders written before any file; the write fails on a name from the environment; Snowflake's env file loaded, never read or printed (KO3-24) | 19 |
| Three non-product live failures stop live work | Quota and capacity errors retried once by the runner (G4) | 9 |
| The Java lane slows the Python path on an 8-CPU machine | `heavy` admission; Java's builds share the Rust cache; the forecast counts an hour of queueing (KO3-28) | 14 |
| The 118 forced-route failures hide more than three families | T3.8 starts on day 1 with the whole failing list; its exit is 0 failures | 6 |
| An older Databricks runtime does not run on aarch64 nodes | C-1's aarch64 probes; owner question Q3 before E-2 | 14 |

## Owner questions

Each opens with what it means in plain words and a recommendation; IDs are footnotes.
Work proceeds as recommended until an answer arrives, except where a question says it blocks.
The live-access answers (KO3-18 to KO3-26) are recorded; the discovery run found a candidate for every resource they point to, and the names stay in the orchestrator's private notes.

**Day 1 (they block C-1, the early capture, at about H5):**

- **D1. Where the tests may write on Databricks.**
  The tests create tables and a few views; your sandbox catalog has an empty schema marked safe to drop and the wheels volume from earlier runs.
  Recommendation: a new dedicated Phase 3 schema in the sandbox catalog for tables and views, and a new folder in the existing wheels volume; nothing written anywhere else.
- **D2. The Lakebase instance.**
  The instance you created earlier for parity recordings is stopped.
  Recommendation: start it for C-1 and the end campaign, and stop it after each.
- **D3. Snowflake's test role.**
  The credentials you gave reach Snowflake with the account's admin role, which could drop any team's data.
  Recommendation: create a test role that can use the smallest warehouse, owns one new test schema (tables, views, semantic views) and can read the existing shop tables and semantic view; if you prefer not to, the tests run with the admin role but every statement stays in that one new schema.
- **D4. The serving endpoint.**
  Your newest always-warm endpoint serves model version 24, while the registered model has versions up to 26 that no endpoint serves.
  Recommendation: test against that endpoint, so no run waits on a cold start.
- **D5. The offline judge for the warehouse legs** (new acceptance infrastructure, plan section 9).
  Before the end, a warehouse case is judged by running the core against a local database holding the exact tables the warehouse returned in C-1, with the warehouse's random samples replayed, and comparing its answers with the old client's recorded answers (choice 2); on those offline legs the SQL statements are the local stand-in's, so their count and per-statement results are only diagnostics, while every live run keeps H9's full statement rule; and the two 100,001-row batching cases stay off the warehouse legs.
  (a) As described, the dialects' SQL first runs on a warehouse in the end campaign; (b) also one short live check of each dialect's SQL when its package is approved, about 20 minutes per warehouse, catching SQL a warehouse rejects about a day earlier.
  Recommendation: (a), as you asked for live tests at the end; choose (b) if a late surprise worries you more than three short mid-phase runs.

**Python and live questions:**

- **Q1. The switch.**
  Flip every S0, S1 and S2 path to the core at once, as the plan says?
  And if one of the three refusals you kept until Phase 4 (a tabular model request, text times read by the client's own rules, `LIST_DISTINCT` over a foreign key) can be reached from a switched path, it refuses an input the old client answers.
  Recommendation: flip all at once; keep any such input refused by name on the epic branch until Phase 4 lifts it, since the epic does not reach users before then; T3.2's census says whether any is reachable at all.
- **Q2. Old-engine tests after the flip.**
  Some client tests check the old engine's internals rather than what a user sees, and cannot pass on the core.
  Recommendation: run those on the old route only until Phase 5 deletes the old engine, each listed in the evidence; every public-behavior test must pass on the core.
- **Q3. aarch64 Databricks runtimes.**
  In your workspace only the newest runtime line has explicit aarch64 versions; whether DBR 14.3, 15.4 and 16.4 run on aarch64 nodes is unproven, and cells J02, J04 and J06 need them.
  Recommendation: C-1 probes it with tiny jobs; any runtime that cannot run on aarch64 there has its aarch64 cell run on the newest aarch64 runtime instead, recorded in the evidence.
- **Q4. Live timing at the switch.**
  The plan asks for live push-down timing at the switch; it costs about 1.5 hours of runs.
  Recommendation: keep it at the switch, as written.
- **Q5. How view YAML values are read.**
  The old client reads view YAML with PyYAML, which reads `yes`, `on` and `off` as true and false (YAML 1.1); a strict YAML 1.2 reader keeps them as text.
  Recommendation: read them as the old client does, so no answer changes.
- **Q6. Warehouse sampling.**
  RS-6 approved sending the old client's sampling SQL on Postgres; Phase 2 left open whether it covers the warehouses.
  Recommendation: yes: the core sends the old client's sampling statements on Databricks and Snowflake, the percentage written as Python writes it, so the same rows are drawn.
- **Q7. What the job matrix asks.**
  You said to start from today's quickstart notebook, which is not enough for testing (KO3-23).
  Recommendation: each of the 15 Unity Catalog cells asks the quickstart's 5 questions and the deleted notebook's 8 (with its ranking and forecast) on the cell's route, each of the 6 Lakebase cells the 6 base questions, and both notebooks run once on the core.
- **Q8. L3 on the warehouses.**
  The random-input comparison runs on DuckDB, SQLite and Postgres; on a warehouse it needs a live run.
  Recommendation: 20 generated sources per warehouse, in the end campaign.
- **Q9. Three refusals decided overnight in Phase 2** refuse inputs the old client answers: a NaN `lag_timesteps`, `num_neighbors` text that is not a number, and a negative `num_neighbors`.
  Recommendation: match the old client (lift all three) in T3.9.
- **Q10. The other Phase 2 overnight decisions, for your confirmation** (each was the orchestrator's under your delegation; each can be reversed):
  - a key or time column given as a value that is not a name, with a very long printed form, is refused with a shorter message than the old client's (P2D-2);
  - a Postgres JSON cell holding a broken character escape is refused by name, where the old client crashes (P2D-3);
  - a refusal of a key past the 64-bit range names the table, column and row but not the value read from the data (P2D-4);
  - a Postgres `json[]` column is left out of the schema with a notice, where the old client refuses the whole table (P2D-5);
  - Postgres keys stored as numbers without a declared precision are compared as exact numbers, so `1.0` and `1.00` are one key, as the database itself compares them (P2G-4, P2G-5);
  - warnings and refusals about the anchor and context times may name the time column's earliest and latest times, as the old client does (KO2-89).
  Recommendation: confirm all of them.
- **Q11. Branch protection on `rust-core`.**
  Today any push could change it; only your merges should.
  Recommendation: protect it so only you can push.
- **Q12. A bigger build machine.**
  Today's machine takes four Python packages and the Java lane, with about an hour of queueing.
  Recommendation: only if one is easy to get; the plan does not depend on it.

**Java questions:** answered (KO3-28 to KO3-35): a fifth lane for Java; Python's API in Java's words; the SLF4J 2 API only; coordinates in Java's convention, built with Maven and published in Phase 5; Linux x86_64 and aarch64, with macOS on the runners Phase 2 already uses; Java 17 bytecode tested on JDK 17 and 21; the runtime flags documented and used in every test, with the Java consumer asked to confirm them.
Their former numbers J1 to J7 (and Q16 to Q22 before round 2) are retired.

**Process questions:**

- **G1. The standing "match" class.**
  You approved (KO3-7) that statement counts, warning repeat counts, tie order among equal scores and hash order are matched only when a user can see them, otherwise one register class you approve once.
  Recommendation: T3.2 writes that one entry; approve its text when it lands.
- **G2. One severity bar for both reviewers** (you asked "Has this really been a problem?").
  Yes: on T2.14b, Claude rated rare mismatches with the old client as minor three rounds running while Codex blocked on them, and Codex approved decision and register wording Claude then found wrong (three packages and the design); each disagreement cost a round.
  Recommendation: one bar: any measured difference from the old client on a public call, any red CI job the change caused, and any decision row wider than your words blocks; hardening internal tools stays minor (KO2-52).
- **G3. The design's "no" list.**
  About 45 carried checking-tool and test items are closed as "no" unless a Phase 3 case or the dry run hits them (the carried-rows table).
  Recommendation: accept the list; any item hit later goes into T3.2 or the train.
- **G4. Live failures for non-product reasons.**
  Three such failures stop live work (plan section 9); quota and capacity errors are common on serverless and are not about the product.
  Recommendation: the runner retries a quota or capacity error once, and only a failure after the retry counts.
