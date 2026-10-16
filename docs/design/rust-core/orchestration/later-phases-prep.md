# Phases 3 to 5: preparation, plans and owner questions

Phase 3 is S2 live, Databricks first, and the Python switch for S0, S1 and S2. Phase 4 is S3: frames, link prediction, sessions and tabular. Phase 5 deletes the old engine and produces the release candidate, and the owner decides the epic merge (migration-plan.md sections 4, 7 and 8).
Each phase still starts with its own reviewed design, as the handbook's "Every phase starts with its design" describes. This page is starting material for each `phase-N/design.md`, not a substitute: the kickoff writes, reviews and locks the design, and only then do product packages start.
It was drafted on 2026-10-14, before Phase 1's merge, and updated after it. Phase 1 merged into `rust-core` as `91a00319` on 2026-10-15. All 58 of its owner-to-confirm decision rows were settled, and the register has no proposed entries left.

Package IDs follow land.py's pattern (`T<n>.<n>[a-z]` or `CI<n>`): `T3.<n>`, `T4.<n>`, `T5.<n>`. Each design lands as `T<N>.0a`, as Phase 2's does. If moving the tooling onto the new phase branch still needs a code change after T2.0, that change is `T<N>.0` and lands first, alone.

## The short answer

- **Phase 3 fits in one day**, if its live work is prepared before the clock starts: the reference's warehouse statements captured for offline replay, and the live baselines recorded.
- **Phase 4, as written, is about 1.5 days**: about 50 hours of package work. It fits in one day only with the cuts in Q4.3 and Q4.4.
- **Phase 5 fits in one day**, if the L8 platform jobs were built earlier as a background lane. If a Windows or macOS build runs for the first time on Phase 5's day, one failing target can take the whole day.
- **Java has no slot** in any phase. It needs its own track of about 12 to 15 packages (owner decision; see the end of this page).
- **The process questions are asked once**, at Phase 2's kickoff (G1, G3, G4 and G6 to G9 in [phase-2-prep.md](phase-2-prep.md)). Later phases keep those answers unless the owner changes them, and ask only their own questions.

## What every phase reuses

| What | Where | How later phases use it |
|---|---|---|
| Contract types and JSON Schema, locked by `contract.lock` | `rust/crates/sdm-contract`, `scripts/rust_core/contract_lock.py` | Each phase adds its vocabulary under one version bump, in its first product package |
| The run state machine and dispatch | `sdm-core` | Unchanged; new operations are new dispatch arms |
| The data port: `DataLayer`, one `Arc<dyn DataLayer>` at the composition root, provider capabilities | `sdm-contract::data`, `sdm-data`, `sdm-host` | Each new dialect is a module plus its capability declarations |
| A dialect-neutral catalog answer (E54) | `sdm-data` | Databricks, Snowflake and Lakebase readers produce the same `CatalogAnswer` |
| L0 guards, including the `dialects` guard that keeps SQL words out of the other crates | `scripts/rust_core/l0_scan.py`, `l0_deps.py`, `bridge_freeze.py` | New dialects can't leak into the domain crates |
| Runners `python-ref`, `python-core`, `python-old`, `rust-host`; the comparator and its policy | `contract/runners/`, `sdm-testkit`, `scripts/rust_core/compare.py`, `contract/compare-policy.yaml` | Phase 2 adds the backend axis. Live warehouses need the nondeterminism rules (RS-6's notes). `python-old` goes in Phase 5 |
| Ratchet, register, gate, coverage, evidence pack | `ratchet.py`, `register.py`, `gate.sh`, `coverage.py`, `evidence.py` | Phase 2 adds `coverage.py --gate S1` (T2.15); each later phase adds its slice's gate |
| The old-against-core differential | `contract/tools/differential.py` | The proposed basis of L3 (Q2.8) |
| Real-data harness | `contract/tools/relbench.py`, `contract/fixtures/adventureworks`, `scripts/live/` | Real-data parity in every phase |
| Landing, lanes, the watchdog | `scripts/rust_core/` (`land.py`, `lane.py`, `prepush.sh`) and `scripts/rust_core/orchestration/` | Approved packages stack and land on their own evidence without new CI |
| CI | `.github/workflows/ci.yml` | Private images, timeouts, no CI on landing commits, cold `rust-test` in about 34 minutes; a Postgres 17 service already runs in CI |
| Brief rules | [briefs/](briefs/) | Appended to every brief |

## How a one-day phase runs

"One day" is read as 24 hours of wall clock, with agents running overnight (question G1). If the owner means a working day of about 10 hours, each phase needs about half the scope below.

| Window | What happens |
|---|---|
| **Prep, before H0** (overlaps the previous phase's last 6 to 8 hours; no product code) | The design, reviewed and locked; the owner's answers recorded; one brief per package. Test authors write the new cases and record the reference observations. Live baselines are recorded where the phase is live. The update from `main` and the reference rerun start |
| **H0 to H2** | The phase's first package, run alone: its single contract bump plus any judging-path change. Every other lane starts on top of its candidate as a stack |
| **H2 to H14** | 5 or 6 lanes. Each package goes through implementer, push, CI and both reviews in parallel, a fix round, then `land.py`. About 3 packages per lane |
| **About H8** | One design checkpoint over the landed range, against `learnings.md` |
| **H14 to H20** | The full local gate, the register reconciled, the evidence pack |
| **H20 to H24** | The fresh whole-phase review and its fixes, then the owner's merge |

A package's target cycle is 3.5 to 4.5 hours: tests already written, 1 to 2 hours of implementation, CI overlapping the reviews, one fix round, the landing.
After each merge, check CI on `rust-core` and on the next phase's first run (Phase 1's merge left `rust-core` red; see T2.0 in [phase-2-prep.md](phase-2-prep.md)).

## Phase 3: S2 live, Databricks first, and the Python switch

### Scope as written

From the plan (section 8 Phase 3, section 4 S2, decisions E27 and E31):
- the Databricks dialect, provider and AdventureWorks notebook on the core first, then Snowflake (with a semantic view) and Lakebase;
- ranking and forecasting on all three;
- the Databricks job matrix across runtimes (21 cells, J01 to J21, `phase-0/databricks-jobs.md`);
- "Python and Java agree on the same Databricks data";
- the Python switch for the S0, S1 and S2 paths together.

Exit, the switch gate (plan section 7): S2 slice-done with L6 green for its scenarios; the job matrix equal to the baseline; L7 within its limits, live timing included (push-down median over 5 or more interleaved runs, caches off, within 10% on Databricks and Snowflake).

Assigned to Phase 3 by Phase 1:
- S0's L3 run and the update from `main` before any S0 path switches;
- the Databricks sampling percentage formatted as Python prints it (RS-6, now approved);
- H24's live common-input comparison;
- the `_routing.py` test that fails if a default changes before the switch;
- the YAML crate for metric and semantic views (`saphyr` or `yaml-rust2`) and a ruling on YAML 1.1 values such as `yes` and `on` (plan section 5);
- the DBR 14.3 install revalidation that T1.35 carried (not done in Phase 1; it must happen before the epic merge, so in Phase 3 or Phase 5).

Reused: Phase 2's Postgres dialect (Lakebase is Postgres) and its conformance kit; the comparator's nondeterminism rules; the archive's dialects (`dialects/src/databricks.rs`, `snow.rs`, `view_expr.rs`, `schema/src/{metric_view,semantic_view,yaml}.rs`); the old samplers (`databricks/sampler.py` 598 lines, `snow/sampler.py` 460, `metric_view.py` 637); and the orchestrator's live tooling, kept outside this repo (wheel builds, Lakebase load and clean-up, the job-matrix scripts).

### Work breakdown (about 12 packages, plus live runs)

Live runs are not landings. Their results go into T3.9's evidence pack.

| ID | Package | Kind | Estimate | Depends on |
|---|---|---|---|---|
| prep | Live baselines on the **reference** wheel: the 21-cell job matrix and the L6 scenarios. **Capture the reference's warehouse statements and results** for offline replay. Load the data and create the views (AdventureWorks in Unity Catalog exists; Snowflake needs its tables and a semantic view; Databricks a metric view; Lakebase through the load tooling). Build the wheels for the matrix's architectures | M | 6–10 h of wall clock before H0, mostly waiting on remote jobs | Live access (Q3.1) |
| T3.1 | Contract part 1: what S2 adds (view-backed sources, Spark and Snowpark provider kinds), plus the L7 harness's judging pieces if approved (Q3.4) | N | 2 h | prep |
| T3.2 | **Databricks dialect** in `sdm-data` (catalog, preview, sampling with `TABLESAMPLE … REPEATABLE` and the Python-printed percentage, neighbors, lookup) and the SQL-warehouse provider (`fetchall_arrow`), checked offline against the captured reference statements | M (port) | 4 h | T3.1 |
| T3.3 | Databricks Spark provider (`toArrow`, DBR 17.3 and later, or serverless 4 and later) | M | 2 h | T3.2 |
| T3.4 | Metric view (Databricks) and semantic view (Snowflake), interpreted in `sdm-schema` over the YAML crate | M/R (YAML differences) | 4 h | T3.1 |
| T3.5 | Snowflake dialect and provider (`fetch_arrow_all`, Snowpark's `session.connection`), offline against captured statements | M | 4 h | T3.1 |
| T3.6 | Lakebase: Phase 2's Postgres dialect, the `lakebase` connect path, routes | M | 2 h | T3.1 |
| T3.7 | **The switch**: flip `_routing.py`'s defaults for every S0, S1 and S2 path at once, update the guard test, run the **existing client test suites** on the core route and sort their failures | R | 4–6 h | T3.2–T3.6 |
| T3.8 | L7 offline: query counts, CPU per scenario at most 1.1 times today's, import time, library size, the control-message cap, zero-copy, no pandas in providers | N (new acceptance infrastructure) | 3–4 h | T3.1 |
| Live 1 | AdventureWorks notebook on the core (Databricks; E27, E31) | Run | 1 h | T3.2 and a wheel |
| Live 2 | The 21-cell job matrix on the core, compared with the prep baseline | Run | 1–1.5 h of parallel jobs | Live 1 |
| Live 3 | L6 for the Snowflake and Lakebase scenarios; L7 live push-down timing (5 or more interleaved runs on each of 2 warehouses) | Run | 2 h | T3.4–T3.6 |
| T3.9 | Switch gate: evidence pack, register, review | M | 4 h | all |

**Critical path**: prep → T3.1 → T3.2 (4 h) → wheel, upload, Live 1 (1.5 h) → Live 2 (1.5 h) → T3.7's suite triage, overlapping → Live 3 → T3.9 (4 h). About 15 or 16 hours with one live fix cycle of about 1.5 hours. It fits only if the dialects are right before the first remote run, which is why prep captures the reference's statements.

### A one-day schedule (5 code lanes plus a live lane the orchestrator runs)

| Lane | H0–H4 | H4–H8 | H8–H12 | H12–H18 | H18–H24 |
|---|---|---|---|---|---|
| A (head) | T3.1, then T3.8 | T3.8 | | Switch gate | Review and merge |
| B Databricks | T3.2 | T3.3 | Live fixes | | |
| C views | T3.4 | T3.4 fixes | | | |
| D Snowflake | T3.5 | T3.5 fixes | Live fixes | | |
| E switch | T3.6 | T3.7 flip and suites | T3.7 triage | | |
| Live (orchestrator) | Wheels ready | Live 1 at about H5 | Live 2 | Live 3 | Clean-up |

### Owner questions to settle before Phase 3

- **Q3.1 Live access for the whole day, without asking again.**
  - Databricks: how many one-time job runs (21 cells, for the core and the reference, plus about 10 L7 runs); creating a metric view in a sandbox schema; which database instance to use for Lakebase (the standing rule is to ask before creating one); and which serving endpoint answers live predictions (a scale-to-zero endpoint's cold start matters for a matrix run).
  - Snowflake: an account, a warehouse, credentials, and permission to load the test tables and create a semantic view. None is set up yet.
- **Q3.2 Java.** S2 says "Java follows on Databricks" (plan section 4), and Java "never holds Python back" (section 2). Confirm Java isn't part of Phase 3's day, and schedule its track (below).
- **Q3.3 The switch.** Flip every S0, S1 and S2 path at once, as written? And inputs still refused by name at the switch, if Q2.4 deferred any: approved temporary regressions until Phase 4; a per-input fallback to the old route (not recommended: "no call mixes old and new code"); or no switch for paths that still have refusals.
- **Q3.4 L7.** Approve the L7 harness as new acceptance infrastructure and **set its numbers now**: the import-time limit, the native-library size limit and the per-run boundary overhead. The plan names the checks but not the limits. And: live timing at the switch, as written, or only at the release gate?
- **Q3.5 Job matrix size at the switch.** All 21 cells, or one cell per runtime family and architecture (about 8), with all 21 at the epic merge and the release?
- **Q3.6 YAML.** Pick the crate. Are YAML 1.1 values (`yes`, `on`) read as 1.2 strings or as booleans?
- **Q3.7 Client tests of old internals.** After the flip they run on the core. May tests of old-engine internals be marked old-route-only until Phase 5 deletes them?
- **Q3.8 H24's live common-input comparison.** Information only, or a gate?

### Risks

| Risk | Mitigation |
|---|---|
| Live debugging cycles take 15 to 30 minutes each | Replay the captured reference statements offline; reproduce locally first (plan section 6). Live work starts only when a dialect matches the captured statements |
| The existing client suites fail in large numbers after the flip | **Dry-run the flip in Phase 2's last hours** (force every route with `SDM_CORE_ROUTES`) to measure it, and fix or classify in Phase 2's spare hours |
| Snowflake access isn't ready | Q3.1. Without it the Snowflake paths can't switch, and the gate passes for Databricks and Lakebase only (owner decision) |
| Three live failures for non-product reasons stop live work | G7, plus retries that recognise quota errors, as CI does for registry pulls |
| L7 CPU above 1.1 times today's on some scenario | Measure offline during Phase 2's prep. The first attempt measured the core at about half today's CPU |
| A serving endpoint's cold start slows the matrix | Warm it before the matrix. L7's live timing measures push-down only |

### Disproportionate items (owner decisions)

- **Live L7 timing at the switch**: keep the offline L7 at the switch and run live timing once, at the release gate.
- **The full 21-cell matrix at every gate** (switch, epic merge, release): run it in full at the switch and at the release, and a subset at the epic merge.
- **"Python and Java agree on Databricks data" inside S2**: move it to the Java track.

## Phase 4: S3, frames, link prediction, sessions and tabular

### Scope as written

From the plan (section 8 Phase 4, section 4 S3):
- in-memory frames, including files read from S3 (the reading stays in the host);
- their ranking and forecasting;
- link prediction;
- sessions;
- tabular requests;
- "moving the rest of the code as its paths need it".

Exit: the slice gate and the switch gate.

The path count in `phase-0/scope-python.yaml` is 121 S3 paths: `client.fit` 31 and `fitted.predict` 31, on **every backend** including the live warehouses; `query.materialize` 16; `client.predict` 16; `connectors.read-table` 14; `connectors.read` 4; `schema.from-data` 2; `request.tabular` 2; `result.read` 2; `schema.construct` 1; `fitted.close` 1; `result.merge` 1.

Assigned to Phase 4 by Phase 1:
- reading frame text times. PK-7 keeps pandas 3's text-timestamp detection, and the owner's E66(h) (2026-10-11) rules out pandas emulation in the core: the core's text-time reader parses a fixed list of formats and refuses the rest by name (DT-15; DT-6 was withdrawn). Q4.1 settles how frames meet both;
- the in-memory sampler's semantics (Phase 1 design, "Explicitly out");
- settling the disagreement between the SQL and in-memory samplers (F2; ORD-2, approved for now as "each path keeps its own policy until settled"; ORD-3);
- DT-3's public in-memory proof (H24);
- the random streams RS-2 to RS-5, all approved, including RS-3's state across calls and RS-5's tabular context sample;
- column-wise conversion of pandas object columns;
- the ledger row "model and route choice, and the tabular refusals";
- E92's two S3 rules, `model` and `text_time_rules`.

Reused: Phase 2's in-memory fallback (T2.6); E52's state as a value (for sessions); the archive's `request/src/{frames,sessions,tabular,tabular_adapter}.rs`, `sampling/src/frame.rs` and `schema/src/frame.rs`; the old `local/graph_store.py` and `local/sampler.py`; `adapters/tabular.py` (580 lines) and `materializers/tabular.py` (339).

### Work breakdown (about 15 packages)

| ID | Package | Kind | Estimate |
|---|---|---|---|
| T4.1 | Contract part 1: frames as a host provider over Arrow tables, the session state value, tabular request types, link-prediction outputs | N | 2 h |
| T4.2 | The frame provider: the core answers data requests over host-supplied Arrow, reusing T2.6; the `schema.from-data` and `schema.construct` routes | M | 4 h |
| T4.3 | The host's conversion table for pandas object columns, column-wise (L5 Python) | M/R | 3 h |
| T4.4 | Frame text times, as Q4.1 decides | **R** if it drifts toward emulation (lesson 3) | 2–6 h |
| T4.5 | The in-memory sampler settled (F2, ORD-2, ORD-3); DT-3's public proof | N | 4 h |
| T4.6 | Random streams RS-2 to RS-4 (numpy streams, bounded emulation); RS-3's state across calls as a value | M | 3 h |
| T4.7 | Link prediction: materialize and predict | N/R | 4–5 h |
| T4.8 | Frames' ranking and forecasting, reusing T2.8a, T2.8b and T2.9 over the frame provider | M | 3 h |
| T4.9a | Sessions on the core and frames: `client.fit`, `fitted.predict`, `fitted.close`, state as a value | N/R | 4 h |
| T4.9b | Sessions on the SQL backends | N/R | 3 h |
| T4.10 | Tabular requests: the KumoTabular model, `request.tabular`, RS-5, the model and route choice in `sdm-core` | N/R | 5 h |
| T4.11 | `result.merge`, `predict-frame`, and routes for `connectors.read` and `read-table` (the reads stay in the host) | M | 2 h |
| T4.12 | Live L6 for warehouse-backed S3 paths (fit and fitted predict on Databricks, Snowflake and Lakebase), on Phase 3's live lane | Run | 2–3 h |
| T4.13 | S3 slice gate and switch: flip the S3 defaults, gate, review | M | 5 h |

About 50 hours of package work plus 2 to 3 hours live. **With 6 lanes that's about 1.5 days as written.**

### A one-day schedule (6 lanes; needs the cuts in Q4.3 and Q4.4)

| Lane | H0–H5 | H5–H10 | H10–H15 | H15–H24 |
|---|---|---|---|---|
| A (head) | T4.1, then T4.11 | T4.13 prep | Live lane | Gate |
| B frames | T4.2 | T4.3 | T4.8 | |
| C inference and sampler | T4.4 | T4.5 | T4.6 | |
| D link prediction | T4.7 | T4.7 fixes | | |
| E sessions | T4.9a | T4.9b | Fixes | |
| F tabular | T4.10 | T4.10 fixes | | |

**Critical path**: T4.1 → T4.9a → T4.9b → live sessions on the warehouses → gate, about 16 to 18 hours.

### Owner questions to settle before Phase 4

- **Q4.1 Frame text times.** (a) The host detects text times with real pandas while it converts a frame to Arrow. That is the host's "native types to Arrow" job, keeps PK-7, and puts no emulation in the core. (b) The core reads frame text by the same fixed list as SQL text (DT-15). That changes answers and needs a register entry. Emulating pandas' inference in the core is ruled out by E66(h). Recommendation: (a). Decide before H0: it decides whether T4.4 takes 2 hours or 6.
- **Q4.2 The sampler disagreement** (F2, ORD-2). Frames follow the SQL sampler's policy (NULL times dropped, newest first), which gives one behavior and a register entry? Or each path keeps its own policy, as ORD-2 now records?
- **Q4.3 Sessions on the warehouses.** Run live L6 for fit and fitted predict on Databricks, Snowflake and Lakebase within Phase 4's day? Or prove them offline on DuckDB, SQLite, Postgres and frames, and run the warehouse session cells once, at Phase 5's release L6?
- **Q4.4 Tabular.** Tabular requests (and RS-5) are a separate model family. Keep them in Phase 4's day, or give them their own half-day? That is the clearest cut if Phase 4 has to fit.
- **Q4.5 Object columns** of mixed Python types: refuse by name, or convert as pandas would?
- **Q4.6 Rules Q2.4 deferred**, if any land here. Lifting them before Phase 5 deletes the old engine avoids a regression on switched paths.

### Risks

| Risk | Mitigation |
|---|---|
| Frame inference becomes pandas emulation | Q4.1 (a) |
| Sessions' state across calls, and RS-3 | E52 already makes state a value; test authors record call sequences in prep |
| Link prediction and tabular are both new model families | Separate lanes; payload vectors from the reference in prep; Q4.4 |
| 121 paths make coverage's rule 7 large | Generate case variants by Phase 2's backend axis: one case per mode per backend |

### Disproportionate items (owner decisions)

- **Live L6 on the warehouses for sessions at Phase 4's switch**: move it to Phase 5's release L6 (Q4.3).
- **Tabular inside the same day**: move it to its own half-day (Q4.4).

## Phase 5: deleting the old engine, the release candidate, the epic merge

### Scope as written

From the plan (section 8 Phase 5, section 7's epic-merge and release gates, section 6 L8):
- delete the old engine on the epic branch;
- the release candidate;
- the owner decides the epic merge into `main`.

Epic-merge gate: every slice done and switched; the full L8 platform matrix green; L6 and the job matrix green on the epic commit; the existing Python test suites green.

Release gate: every applicable layer green on the release commit with the old engine deleted, L6 and the job matrix included; live test resources cleaned up.

L8: the real wheel (and the JAR, if Java ships) installed in clean containers on every platform in the support matrix (Linux x86_64 and aarch64 manylinux2014, musllinux_1_2 if feasible, macOS x86_64 10.12 and arm64 11.0, Windows x86_64); exported symbols and minimum system library versions; `sdm-host` under sanitizers; panic, crash and leak tests; HTTP retry, timeout and redaction tests.

Assigned to Phase 5 by Phase 1:
- delete the routing seam and `SDM_CORE_ROUTES` (B-D7);
- remove the shadow check and the old Python validator (PQL-2, PQL-3);
- seed the failure families of retired tests before deleting them (H24; plan section 6, "Retiring old tests");
- run L3 before and after the deletion;
- one independent release audit (plan section 9);
- the DBR 14.3 install revalidation, if Phase 3 didn't do it.

### What gets deleted

- `packages/structured-data-models-relational-engine/src`: 25,019 lines of Python, plus `csrc/neighbor_sampler.cpp` and vendored assets. Its tests are 29,114 lines. (Every line count in this section is `wc -l` over the `.py` files at Phase 1's merge commit 91a00319; this handoff changes none of them. The PQL count is its `src` only.)
- Before deleting it, move what are really **host** functions into the client: the serving client (`client/snowflake_serving.py`, 349 lines), `utils/progress_logger.py`, and any HTTP or auth code. The plan keeps HTTP and auth in the host (section 3).
- The client's old route: `materializers/relational.py` (1,597 lines), parts of `schema.py`, `client.py`, `request.py`, `decompose.py` and `adapters/*`, `_routing.py` (366 lines), the bridge freeze, and `python-old` with every judging rule that names it (E110, E112, E114).
- `structured-data-models-pql` (13,208 lines): the runtime dependency goes. The ANTLR grammar stays only as CI's cross-check (plan section 3, PQL).

### Work breakdown (about 10 packages; the deletions are mechanical, the platform matrix is the risk)

| ID | Package | Kind | Estimate |
|---|---|---|---|
| prep | **The L8 platform CI jobs, built during Phases 3 and 4 as a background lane** (maturin wheels for the 6 targets, clean-container installs, symbol and glibc checks), so Phase 5 isn't their first run. They change `ci.yml`, a protected path, so each lands alone as a CI package (CI6 on). Also: the list of failure families to seed, and the release audit's brief | M/R | Background |
| T5.1 | Judging-path part 1: remove `python-old` from the runners, the comparator and the ratchet rules; `python-core` against the reference becomes the only Python comparison | N | 3 h |
| T5.2 | Move the host functions out of the engine package (serving, progress, HTTP) | M | 3 h |
| T5.3 | Delete the engine package, its C++ build and its tests; drop the client's dependency on it | M | 2 h |
| T5.4 | Delete the old route in the client: the routing seam, the shadow check, the bridge freeze, the old validator path; the PQL package becomes test-only | M | 3 h |
| T5.5 | Retire the old tests after seeding their failure families (H24) | M | 3 h |
| T5.6 | L8: sanitizers on `sdm-host` (if Q2.7 moved them here); panic, crash and leak tests; HTTP retry, timeout and redaction | M | 3 h |
| T5.7 | Release candidate wheels on every platform; clean installs; notices and SBOM | Run | 2–3 h |
| T5.8 | Release live run: L6, the 21-cell job matrix, the DBR 14.3 install revalidation, warehouse sessions if Q4.3 moved them here; live clean-up | Run | 3 h |
| T5.9 | L3 after the deletion; the independent release audit; the evidence pack; the epic-merge decision | M | 4 h |

**Critical path**: T5.1 → T5.3 and T5.4 → T5.7 → T5.8 → T5.9, about 15 hours. Comfortable if the platform jobs already work.

### Owner questions to settle before Phase 5

- **Q5.1 Deletion scope.** Delete all of `structured-data-models-relational-engine`, or keep a deprecated shim distribution? Is the PQL package kept test-only, or removed?
- **Q5.2 Platforms required for release.** musllinux in or out ("if feasible")? Windows? Which hosted runners can the repository use for macOS, Windows and native aarch64? Wheels are built locally today; can CI build them all?
- **Q5.3 Java in this release.** Does a JAR ship, and does it need L8 on JDK 17 and 21?
- **Q5.4 Seeding the retired tests' failure families** (H24). A bounded version (the 10 families the postmortems list), or waive it?
- **Q5.5 After the merge.** The version number, the contract version, the PyPI release, who merges the epic into `main`, and when.
- **Q5.6 Live resources** to clean up, and confirm the DBR 14.3 job is still pre-authorized.

### Risks

| Risk | Mitigation |
|---|---|
| A platform build fails for the first time on the day | The background L8 lane from Phase 3 on |
| Client tests that went through the old route fail once it's gone | Phase 3's flip already moved them to the core; Phase 5 only removes the old route |
| Host functions found inside the engine late | T5.2 runs first, from a grep-based inventory made in prep |
| The release audit finds a gap | Run it as a checkpoint review at about H8, not only at the end |

## Proposals that change what the plan or the locked checks require

All are owner decisions. None is assumed in the schedules above, except where a schedule says it needs one to fit in a day. Several are asked at Phase 2's kickoff; the last column says where.

| # | As written | Proposal | Why it matters for one day | Asked as |
|---|---|---|---|---|
| D1 | Two reviewers on every round (E25, E29) | Keep two on round 1 and on every product or judging-path fix round; one reviewer for confirmations, docs-only and tiny fixes, the tier chosen by a script; one Codex per lane. Phase 1's data supports only this narrower cut: in 31 of 40 blocking rounds one reviewer alone would have approved a P1 or P2 | Removes the largest serializer without weakening substantive review | G3 |
| D2 | A checkpoint every 4 or 5 packages | One mid-phase and one at the end | Saves 2 or 3 checkpoint cycles a day | G4 |
| D3 | A difference waits for an approved entry | Standing classes reviewers may approve, plus a parked list for the owner at the phase end | Removes owner waits | G6 |
| D4 | Two landings per case family | One landing when the reference observations are committed first | Halves those landings | G8 |
| D5 | The dataframe-engine comparison at Phase 2's entry | Waive it, or a 2-hour spike | Removes an entry gate | Q2.3 |
| D6 | ASan and LSan in Phase 2 | Phase 5's L8 | Removes a toolchain detour | Q2.7 |
| D7 | L3 as a new generator | `differential.py` plus seed sets against the reference | No new infrastructure | Q2.8 |
| D8 | L7 live timing at the switch | At the release only | About 2 hours of live runs | Q3.4 |
| D9 | The full job matrix at every gate | Full at the switch and the release, a subset at the epic merge | Live time | Q3.5 |
| D10 | Java inside S2 | Its own track | Phase 3 can't hold it | Q3.2 |
| D11 | Sessions live on the warehouses in Phase 4 | At the release L6 | Phase 4 fits | Q4.3 |
| D12 | Tabular inside Phase 4 | Its own half-day | Phase 4 fits | Q4.4 |
| D13 | Large phase documents | Designs of 3 pages or fewer | Hours of writing and review per phase | G9 |

## Java, as its own track

The plan's S2 includes Java, and Phase 1b's row says Java's S0 work starts at Phase 1's go. Phase 1 has merged, and nothing Java exists yet: no `sdm-jni` crate, no JAR, no JDBC conversion table. It can't fit inside Phase 3's day. The track needs:
- `sdm-jni` (jni-rs) and the JAR with natives for linux-x86_64, linux-aarch64 and both macOS architectures;
- the JDBC conversion table and L5 for Java;
- a Java runner for the contract cases;
- the SLF4J bridge decision;
- Java S0, then S1 on DuckDB, SQLite and Postgres, then Databricks over JDBC;
- confirmation with the consumer about `--enable-native-access` and Arrow Java's flag (plan sections 3 and 10).

At Phase 1's cost per package that is about 12 to 15 packages: one more day after Phase 3, or a day between Phases 2 and 3. Its packages need IDs in land.py's pattern too, so where it runs (its own phase branch or inside a phase) is part of the owner's decision. It doesn't block Python: "Python sets the pace" (plan section 4).
