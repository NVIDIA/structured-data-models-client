# Phase 2: slice S1 offline

## Status

Approved by both reviewers at f6eb5f05 (round 11, closure check round 12), then revised with the owner's answers of 2026-10-16 (fix 12, [tracker.md](tracker.md)); the revision is reviewed by both reviewers (KO2-9) before it lands as T2.0a.
The owner's answers are in [decisions.md](decisions.md), KO2-1 to KO2-44; only KO2-20 (Q2.2) waits for the owner's confirmation, and nothing in the locked checks depends on it.
The acceptance checks below lock at T2.0a's landing.
Everything not stated here is as in [Phase 1's design](../phase-1/design.md), cited as `P1 d:<line>`; Phase 1's decision rows are cited by ID and stay in force unless a row here or an owner answer changes them.
Question names: `Q-G<n>` (process), `Q2.<n>` (product), `K<n>` and `Q-<name>` (new here), never shared with plan section 8's carried rows (G4, G6, G8 and so on).

## Goal

Ship slice S1 for Python: `Schema.from_source` → `materialize` → `predict` on DuckDB, SQLite and Postgres, offline, including ranking and forecasting, through the core (migration-plan.md section 4, S1).
Prove the data layer is swappable: a minimal non-SQL test provider passes the provider conformance kit, and the same prediction runs with an alternate data layer swapped in at the composition root, with no domain-crate edits (plan section 3, "Data access").
Exit: the slice gate (plan section 7, "Slice done"), checked by the locked acceptance checks below.
Paths stay unswitched for users: S0, S1 and S2 switch together in Phase 3 (plan section 4).

## Order of work

1. **`main` first** (KO2-3): T2.0e, the gate rules a merge of `main` and a new reference need, and T2.0a, this design with the owner's rows, each land alone in whichever order both reviewers approve; then T2.0c merges `main` 327fda03 into the phase branch with the reference re-recorded at 6298913c (KO2-1), one landing.
   `main` reaches `rust-core` only through the phase branch (KO2-7).
2. **The judging path**: T2.1a, then the single contract bump T2.1b; HO2 is frozen beside T2.1b.
3. **Product work** in six lanes, at most four at once, with lane H's tooling and CI packages beside them from the start; then T2.14a to T2.14c and the slice gate, T2.16, which merges `main` again if it has moved.

## Scope

1. **Backends.** SQLite and Postgres dialects in `sdm-data`, beside DuckDB; Python providers for both that hand Arrow to the core with no pandas round trip (plan section 3, "Performance rules"); rust-host providers for both (a dev-only `rusqlite` provider for SQLite; Postgres by replaying the reference's recorded fetch results, KO2-20).
   Every S1 case runs on all three databases (KO2-19, choice 2).
2. **Operations.** Ranking (`top_k`, candidates, `prediction_statistic`, NaN ranks last under PD-1) with its decomposition moved into `sdm-request`; forecasting's remaining options (`lag_timesteps`, quantiles) on top of the split already in the core (E153); `main`'s multi-call `predict` (1fc5a686, T2.8c).
3. **The 37 named refusals of E92's class** (`phase-1/not-yet-refusals.md`; `contract/tests/test_t133_umbrellas.py` counts 37): the 35 S1 reaches are lifted, `model` and `text_time_rules` kept until Phase 4 provisionally (KO2-22). `float_nan` comes first (plan section 8, FR2 P3-1).
4. **The provider conformance kit** in `sdm-testkit` (fixture dataset, multi-page results, limits, expected answers, capability failure and retryability), passed by every built-in provider; the in-memory fallback for optional requests with declared limits that fail loudly; a minimal non-SQL test provider; a swapped `DataLayer`.
5. **The host-held semantics S1 reaches move into the core** (lesson 17). Each of Phase 1's ledger rows still marked for a later phase, with its disposition:
   - the ranking decomposition, its refusals and reassembly, and the forecast's `top_k` and `MOST_PARTS` refusals and parts' wording (`P1 d:263`): into `sdm-request` (T2.8a, T2.8b);
   - the answer re-encoding in `_encoded` and the response-depth coupling (`P1 d:277-278`): removed, the bytes cross untouched (T2.14b);
   - the model and route choice from the schema's shape (`P1 d:264`): stays in the host until tabular requests, Phase 4 (S3);
   - `main`'s multi-call `predict` (1fc5a686: batches, row numbering across calls, answered parts carried on a failed call, the 100,000-entity and 100-call ceilings, the per-batch 30 MiB body refusal naming what to change, the pre-send context check per call, the batched-ranking and batched-forecast refusals): written in the core from the start (T2.8c), never in the host.
   The bridge freeze is widened to its full ledger first (FR P3-2, T2.14a, alone), the rows move (T2.14b), and the unreached functions leave `COVERED` in a later landing (T2.14c, alone), as `bridge_freeze.py:107-110` requires.
6. **Reference `Request` pickles load** in the new client, and an unknown `RequestState` version is refused by name (H23; plan section 6, L2). `REQUEST_STATE_VERSION` and the version field exist (`sdm-contract/src/ops.rs:833-845`); what is left is loading the reference's pickles and versioning the state the new vocabulary adds.
7. **The full L4 and L5 for the three databases** (A5's deferral): SQL goldens executed on all three; the conversion tables on pandas 2 and 3; the same tables through three drivers reach the core as identical Arrow.
8. **Offline L7**, from plan section 6, with KO2-29's limits fixed in the locked-set file: query counts equal to the reference's; median scenario CPU at most the reference's and none above 1.1 times; boundary overhead per in-memory run within a fixed bound; the control-message size cap; zero-copy and "no pandas in providers"; import time and native library size within limits.
   Live timing stays at the switch gate (Phase 3).
9. **`main`'s other S1 changes, better than `main` where `main` loses information**: exact decimals with the core's three decimal refusals lifted (KO2-35, T2.17); any column name kept as data and nameable in a query (KO2-36, T2.18); two-path sampling (eb6c68eb, T2.19, early).
10. **The slice-gate chores**: the update from `main` and the reference rerun, at the start (T2.0c) and again at the gate if `main` has moved (T2.16), each through the phase branch (KO2-7); L3 on agreed seed sets; register reconciliation; the evidence pack; a fresh whole-phase review.
11. **Plan section 8's carried rows**, each assigned to a package or closed by a recorded decision (tracker, "Carried rows"), with T2.16 as the backstop for any row no lane reached (lesson 11).
12. **For information, not gating**: a dry run of Phase 3's flip in the phase's last hours (every route forced with `SDM_CORE_ROUTES`, the existing client suites run, failures counted), as later-phases-prep.md asks.

### Explicitly out

- Live warehouses, metric and semantic views beyond v26's table source (choice 12), L6, live L7 timing and the Databricks job matrix (re-run on the epic before it merges, KO2-38): Phase 3 (`P1 d:41`).
- Frames and S3 reads, link prediction, sessions, tabular requests, the in-memory sampler, frame text inference (DT-6, PK-7): Phase 4 (`P1 d:41`, `P1 d:46`); E92's `model` and `text_time_rules` too (KO2-22).
- Java: deferred until S2 is switched for Python (KO2-34).
- The full L8 (`P1 d:43`); ASan and LSan move to Phase 5's L8 (KO2-25); release work, Phase 5 (below).
- The Python switch of any path: Phase 3.
- Rendering expression columns and row filters: Phase 3 (KO2-25, KO2-39).
- P1B-86 and P1B-89 stay as they are (E152); FX-3 waits for Phase 4 (KO2-31).

## Architecture choices new in this phase

Phase 1's mechanisms are reused as built: the contract and its lock (`P1 d:100-125`), the data port (`P1 d:136-167`), runners and cells (`P1 d:319-360`), comparator (`P1 d:362-378`), ratchet and checkpoint (`P1 d:380-396`, H2, E131(f)), register v2 (`P1 d:398-404`), coverage (`P1 d:406-411`), gate (`P1 d:436`), lanes and landing (E43, E130, E131), the bridge freeze (`P1 d:249-258`), the evidence pack (`P1 d:528-541`).
What changes:

1. **One contract bump, v26, in T2.1b** (lesson 14; H4). All of Phase 2's vocabulary goes in it:
   - paging: a `Fetch` result may carry a continuation, and the core asks again until the provider says it is done; nothing is truncated silently (plan section 3, "The provider contract"), replacing Phase 1's "drained whole" (`P1 d:145`);
   - multi-column keys in `Neighbors`, `Lookup` and key sets (for `composite_key`), replacing the single `key: String` (`sdm-contract/src/data.rs:261-279`);
   - ranking and forecast result fields;
   - `main`'s multi-call `predict`: several model calls per request in `RequestState`, the per-call and per-request limits, and an error's answered-parts fields;
   - the in-memory fallback's limits as declared capabilities; an undeclared optional request then takes the fallback instead of `UNSUPPORTED_REQUEST` (`P1 d:160`), and a declared capability that fails is still an error (`P1 d:161`);
   - `max_seed` required of every provider, with no "any seed" default (`P1 d:142`);
   - the table source and the provider failure class (choice 12, KO2-39);
   - the `RequestState` version raised for the new state.
   A later need is a second bump renumbered by the lane that lands second (lesson 14; the renumbering script is T2.0b's, U13).
2. **Every case on all three databases** (KO2-19; T2.1a).
   A case runs on DuckDB, SQLite and Postgres unless its file says otherwise, and no case file may say otherwise without an owner-approved exclusion; the case format gains the fields Phase 1's could not carry (`candidates`, `top_k` on a forecast; E153), with accept and refuse vectors in `contract/tests/malformed-cases.json` run by both loaders (`P1 d:581`).
   - **Data.** SQLite and Postgres bases are loaded from the DuckDB-built tables through Arrow by one contract loader, with a per-engine type map total over the type set (part of T2.1a's vectors); the reference and both core hosts read the same loaded database.
     A type an engine lacks (no unsigned or 128-bit integers on Postgres; no decimals, nested values or intervals on SQLite) lands as the map says, for example `numeric`, text or JSON text, and the case then compares both sides on that engine's version of the input, with coverage counting the mapped cell, not the original.
     No case is skipped: an input T2.1a finds no engine can load at all is a proposed exclusion for the owner, named in check 1's evidence, and the locked run fails on that case until the owner answers.
   - **Reference.** The recorder (test-author role) records the reference for every S1 case on each database on this Linux machine (KO2-2), one reference line per case and database; the three runs are independent and run in parallel.
     The core's SQLite and Postgres observations arrive with T2.2 and T2.3, and the ratchet holds each from its first admission.
   - **CI.** Each `contract-python` cell runs once per database as its own job, `contract-python (<cell>, <database>)`, and `cross-host` once per database, `cross-host (<database>)`: nine and three parallel jobs where there are three and one today.
     Only the Postgres `contract-python` jobs start the pinned Postgres service; `cross-host (postgres)` replays (KO2-20).
     Each job runs today's case count, so a push's wall time stays about today's (43 to 88 minutes, plus a minute for the service), while those jobs' runner minutes triple; T2.1a records the timings.
     A job that runs no case on its database fails, so a leg can't pass empty.
   - **Locally**, `gate.sh` runs the three databases as parallel processes on the reference machine.
3. **The reference capture extends to the reference's SQLite and Postgres drivers** (T2.1a), so the reference's statements and results on those databases are recorded like DuckDB's.
   Postgres runs locally in a pinned container (`POSTGRES_IMAGE`, already pinned in CI).
4. **Every new dialect renders every request kind**, including Phase 1's late ones (`column_ranges`, `repeated_key`, the typed seed join), through the one `FROM` function of choice 12, and declares every capability with no fail-open default: `max_seed`, `null_order`, `index_advice`, `text_times`, `opaque_vendor`, `derived_relations` (`P1 d:142`).
   L0 already keeps dialect text inside `sdm-data` and the drivers off shipped crates (`P1 d:167`).
5. **Keys are typed per relationship** (lesson 19): a link joins in the joined key's declared type, both ends decided together; composite keys follow the same rule column by column.
   Every package that binds or casts keys runs the key-type pair matrix on its own engine before review: T2.2 (SQLite), T2.3 (Postgres), T2.12 (composite keys, all three), T2.17 (decimal keys, all three).
6. **Ranking's decomposition and `main`'s multi-call `predict` live in `sdm-request`** (lesson 17), the way E153 moved the forecast split: the core splits a request into its questions and calls, refuses what the reference refuses, and puts the answer back together; the Python host keeps presentation only.
7. **The response bytes cross untouched**: the host stops re-encoding the answer (`_encoded`) and stops knowing the core's depth limit (`_ENCODED_DEPTH`), per plan section 3's "The host never parses or re-encodes the model request or response body".
8. **Sampling on SQLite and Postgres** follows the reference's registered streams: seeded SQLite draws use RS-2's stream (DF-2), and unseeded SQLite keeps `ORDER BY RANDOM()`, compared on common inputs only (KO2-23); Postgres keeps the reference's `setseed` SQL under RS-6, which the reviewers approved at T1.36 fix 7 (KO2-24).
9. **No dataframe-engine comparison** (KO2-21): Phase 2's new data work (paging, the fallback's ordering and limits, the psycopg rows-to-Arrow path, the decimal sums) is each a `phase-1/dependencies.md`-style row naming "use a library", "port a defined algorithm" or "keep in the host" with its measured reason (lesson 4), added by the package that writes it.
10. **Refusals of inputs the reference accepts are generated, not surveyed** (lessons 8 and 18).
    T2.0d adds a census tool (`contract/tools/refusal_census.py`, new; `contract/runners/refusal_map.py` stays frozen at bug fixes, `P1 d:418`):
    - its list of refusal sites is generated from the code, never written by hand: every named rule, every rule-less refusal constructor (`unsupported_type`, `sdm-contract/src/types.rs:521`; admission's `unsupported`, `sdm-host/src/admission.rs:84`; predict's `unsupported`, `sdm-request/src/predict.rs:187`) and the Python host's own refusals; it fails when a site has no probe;
    - it runs each probe on both routes (python-core against the reference) and on every database the probe applies to;
    - it runs the public calls over the phase's real datasets (AdventureWorks and the RelBench sources T1.33 used) on both routes and flags every case where the core refuses and the reference answers, keyed by rule where there is one and by site and code otherwise, with how many real sources hit each.
    A database the core can't run yet is reported as unmeasured, never as zero hits.
    Its first run is on DuckDB; the two kept rules (KO2-22) stay kept only while their real-data counts are zero on all three databases, measured after T2.2 and T2.3, and a non-zero count goes back to the owner.
    It is judging: T2.0d lands alone, the census and its probe list become protected paths, and every landing's gate runs the checkpoint's census probes (minutes); the real-data part (an hour or more on this machine) runs at each lane's first landing, after T2.2 and T2.3, and at T2.16. Locked check 2 fails on a refusal with no register entry and on a kept rule with a non-zero real-data count that no owner answer covers.
    A census hit outside E92's 37 rules is never left for the gate (KO2-28): the package whose code refuses lifts it if it owns the code, and otherwise registers it as a proposed entry marked owner-pending, sent at the next check-in. T2.2 and T2.3 each exit with no such hit on their database that has neither a fix nor an owner-pending entry.
11. **Crash replacements cross every reading** (lesson 21): a refusal or check that stands in for a reference crash (E66(b)'s class) carries vectors across the UTC instant, the wall clock in the session zone, with and without optional libraries such as pytz, and each supported pandas, recorded from the reference and run on both core hosts, with a seeded fault that drops one reading.
    This is an exit of every package that adds such a refusal and a line of every test-author brief.
12. **The table source** (metric-view groundwork, KO2-39; in v26).
    Today a schema table is a source name (`Table.source: String`, `sdm-engine/src/graph.rs:20-30`), quoted as one identifier (`sdm-data/src/duckdb.rs:725-757`).
    A metric view needs four things more (`main`'s `engine/rfm/backend/databricks/table.py:136-205`, 18e26f1c): a name of up to three parts, each quoted alone; columns renamed to the view's spelling; columns that are SQL expressions; a SQL row filter.
    v26 replaces the string with one dialect-free type in `sdm-contract`, plus one capability and one error field:
    ```rust
    pub struct Relation {                // where a schema table's rows come from
        pub source: Vec<String>,         // 1 to 3 name parts, each quoted alone, never split on dots
        pub columns: Vec<RelationColumn>,// empty: every source column under its own name
        pub filter: Option<SqlText>,     // a row filter, as the source wrote it
    }
    pub struct RelationColumn { pub name: String, pub from: ColumnFrom } // `name`: the schema's spelling
    pub enum ColumnFrom { Source(String), Expr(SqlText) }                // a source column or an expression
    pub struct SqlText(pub String);      // opaque: rendered verbatim by the dialect that described it
    // Capabilities: `derived_relations: bool`, declared by every provider, no default.
    // ProviderError: `class: Option<FailureClass>`, FailureClass = Missing | Denied | Unreadable,
    // set by the provider from its driver's codes, beside today's `sqlstate` and `vendor_code`.
    ```
    - Requests keep naming the schema table and the schema's column names, so renames, expressions and the filter are invisible above `sdm-data`; two schema tables over one physical table (a role-playing join) are two relations with the same `source`.
    - `sdm-data` renders every dialect's `FROM` through one function: a plain relation (no `columns`, no `filter`) is its quoted parts, as today; a derived one is `(SELECT src."a" AS "b", (<expr>) AS "c" FROM <parts> AS src WHERE (<filter>))`.
    - A provider that does not declare `derived_relations` refuses a derived relation by name (`UNSUPPORTED_REQUEST`, rule `relation`); a non-SQL provider never declares it.
    - The core never reads `SqlText` and never matches error text: a provider sets `class` from codes in the host, where its driver's message is (Phase 3's Databricks provider parses SQLSTATE and the error class there), and only Phase 3's view describe treats `Missing`, `Denied` or `Unreadable` on a joined table as an answer, dropping the join with a warning; every other provider error still ends the run (`sdm-contract/src/data.rs:818-829`).
    - Dialect logic stays in `sdm-data` (L0 keeps dialect terms out of `sdm-schema`, `l0_scan.py:1050-1086`): in Phase 3, `sdm-data` reads a view definition into a dialect-free view model (tables as relations, joins by key columns, dimensions and measures as `SqlText` with the columns they read), and `sdm-schema` interprets only that model, which is how plan section 3 splits "view retrieval per source" from "source-neutral view interpretation". The model is internal to the Rust crates and crosses no provider boundary, so it needs no bump.
    Phase 2 implements `source` (one part from every S1 host, so no S1 answer moves; a golden shows a dotted name still reads as one table), the one `FROM` function in all three dialects, `derived_relations: false` everywhere, the `relation` refusal with a provider-level census probe (no public S1 input reaches it), and `class` from SQLite and Postgres codes, not acted on.
    It defines `columns` and `filter` and only refuses them; Phase 3 renders them, adds the view model and the describe's tolerance.
    Why this shape: it is the smallest that carries the four needs without SQL entering the typed request vocabulary, which a non-SQL provider (T2.7) could not evaluate.
    A Snowflake semantic view is the same shape (base tables, expression facts and dimensions, a filter) rendered by its own dialect; Lakebase is Postgres with plain relations; frames (Phase 4) take plain relations only.
    It needs no dialect tag, since a schema reads one source.

## Tasks

M is mechanical (ported, with reference vectors), N novel, R risky. "Judging" means the package changes a file or data `gate.sh` takes from the checkpoint (handbook, "Landing"); those land alone, first in their chain (lesson 13).
Each product package has a separate test author (briefs/test-author.md; lesson 16) who writes its new cases and records their reference vectors, on the reference T2.0c records, before the implementer's first push; the cases and the code land together in one landing, and the implementer never edits the test author's files (KO2-13).
Lanes: A head, B dialects, C data layer, D ranking, E forecast, targets and names, F schema and sampling, G calls and decimals, H tooling and CI; T2.0c on its topic branch; the reference machine is the orchestrator's.

| ID | Package | Output | Exit check | Depends on | Lane | Judging |
|---|---|---|---|---|---|---|
| T2.0 | Tooling phase-aware; first checkpoint; rust-core's CI green | Landed 292f8d8a, merged into `rust-core` as 3b81df85 (KO2-6) | – | – | A | Yes |
| T2.0e | Gate rules for `main` first (split out of T2.0b) | The ratchet accepts a replaced reference file only under a reference approval row (decisions.md's format) naming the SHA that the file's name and `contract/reference-env.txt` name, the new recording carrying every case the old one did and each per-case difference listed for the reviewers; the bridge freeze's `main` reason, valid only for a commit the candidate's merge of `origin/main` brings in; `main`'s commits brought in by one merge commit titled with the package ID exempt from the subject rule | Seeded faults: a row without the SHA, with another or an abbreviated SHA, or not the owner's; two reference files left; a `main` reason naming a commit outside the merge or already in the checkpoint; a non-`main` commit without the ID behind the merge; the post-landing state judged against itself; Phase 1's landings still verify | T2.0 | A | Yes |
| T2.0a | This design, decisions, tracker; learnings 17-22 | The owner's rows; the locked checks; `phase-2/locked-set.yaml`, added to `contract/protected-paths.txt` | Two APPROVEs on one SHA, closure check | T2.0 | – | Yes (the locked-set file) |
| T2.0c | `main` 327fda03 into Phase 2, the reference re-recorded | The merge with the audit's conflict resolutions; `uv.lock` regenerated with a CI check that it is current; the deleted AdventureWorks notebook frozen as a contract fixture (KO2-41); KO2-37's floors with the floor cell, `wheel_smoke.py` and the release-contract test; the core sends only the inference options the caller set (26ecdd28, 27 of the 52 cases' payloads), its pins regenerated; the reference and every vector citing it re-recorded by a separate recorder; each changed line and moved baseline registered citing its `main` commit, the interim refusals naming the package that lifts them (T2.8c, T2.17, T2.18, T2.19; Q-I); `bridge_changes.json` under T2.0e's `main` reason; the two credential-shaped test literals built at run time; the epic's actions pinned by digest; a Rust leg in CodeQL | `gate.sh` against the new reference under T2.0e's rule; every changed line traced to a `main` commit or "identity only"; CI green on the exact SHA | T2.0e, T2.0a | topic | Yes (reference data) |
| T2.0d | Refusal census | `contract/tools/refusal_census.py` (choice 10) and its probe list, implementing the census step the locked-set file names, and its first run on DuckDB | Seeded faults: a refusal site with no probe, a rule-less refusal on a real source, a probe skipped on one database, a real-data run with a source left out each fail; a candidate that weakens the census is still judged by the checkpoint's | T2.0 | H | Yes |
| T2.0b | Carried tooling rows | U4 (approvals by allowlist, not a `WITHDRAWN` denylist), U3, C2, C5, C22 (re-checked first), C28, G4, G10, FX-6 (the binding tests read the contract version), U13 (renumbering script), M5 (`TZ` in CI), FR P3-7's ratchet parts, `lane.py` adopting an existing lane's package (T2.0's landing) | Each row's test fails before and passes after; seeded fault: a declined approval | T2.0e | H | Yes |
| T2.1a | The S1 judging path | Choice 2 whole: the case format's database axis and new fields with malformed vectors; the type map; reference capture on SQLite and Postgres and the reference recorded for all 52 cases on all three; runners and comparator for three databases; comparator and both encoders holding a vector for every type in the type set and every stage a case can stop at (lesson 20); `coverage.py --gate S1` with G12, G16 and G17; L3 wiring (`differential.py` over three databases, M8, seed sets); the scripted endpoint answering several calls, numbering each from zero, and failing a chosen call (T2.8c's cases); `gate.sh --locked` reading the locked set's new step names, `limits` and `ci` keys; `--ci-run` through `final_gate.sh` and `phase_end.sh`, and the `ci` step checking each job pattern's conclusions on the candidate's exact-SHA run; the per-database CI jobs, the pinned Postgres service in the Postgres `contract-python` jobs and `rust-test*`, with the reference's and the core's SQLite and Postgres drivers installed by one shared step, and tests reading `SDM_REQUIRE_POSTGRES` that fail, never skip, when the service is unreachable (H26): `contract/tests/test_postgres_required.py` and an `sdm-testkit` test; the common-input comparison (H24), the reference's captured fetch results replayed into the core's run, built and controlled here on DuckDB and applied to Postgres in T2.3; C6, C18 | Seeded faults: a missing database, a retyped field, an empty evidence file, an uncovered cell, an unreachable Postgres service with `SDM_REQUIRE_POSTGRES` set, a named step whose tool is missing, a CI pattern matching no job, a matching job failed, skipped or cancelled, a database job that runs no case, and a run of another tree each fail, each CI check's evidence seeded to fail on its own; an empty landing commit resolves to its tested candidate's run; on common inputs, a different legitimate draw is tolerated while a payload defect on the same rows fails; the post-landing gate passes judged against itself | T2.0c, T2.0a | A | Yes |
| T2.1b | Contract v26 | The vocabulary of choice 1 with the table source of choice 12, M1, M2, M3 | Schema lock; both loaders; the binding jobs green on the bump (FX-6 landed); SQL goldens unchanged for plain relations; a derived relation refused by name | T2.1a | A | No |
| T2.2 | SQLite | Dialect, Python provider (Arrow), rust-host `rusqlite` provider; U12; G6 | L4 goldens executed; the key-type pair matrix on SQLite; every S1 case on SQLite; the kit once T2.5 lands | T2.1b | B | No |
| T2.3 | Postgres | Dialect, psycopg provider (Arrow without pandas), rust-host replay provider (KO2-20); key sets bound as the dialect binds them (`P1 d:131`) | As T2.2, on Postgres, with every S1 case also compared on common inputs (H24) and T2.1a's common-input control repeated on Postgres | T2.1b | B | No |
| T2.4 | Full L4 and L5 | The goldens on all three databases; the Arrow-to-pandas presentation table on pandas 2 and 3 (never built in Phase 1, as-built T1.31); three-driver Arrow equality; the `l4-goldens (<database>)` and `l5-conversion*` CI jobs the locked set names | The new jobs green; seeded type drift fails; the post-landing state judged against itself | T2.2, T2.3 | B | Yes (`ci.yml`) |
| T2.5 | Conformance kit | Fixture dataset, paging, limits, expected answers (some hand-checked), capability failure, retryability, run from `sdm-testkit` against the rust-host providers and from `contract/tests` against each database's Python provider in its `contract-python (ref-pandas3, <database>)` job, both sides reading one fixture dataset and one expected-answers file under `contract/kit/`, written by T2.5's test author; M9, G5 | Every provider landed by then passes on both sides (T2.2 and T2.3 run it on theirs if they land after T2.5); a seeded truncation and a seeded silent capability failure in each host's provider path each fail the kit | T2.1b | C | No |
| T2.6 | In-memory fallback | Ordering, row limits, memory and transfer limits that fail loudly (Phase 4 reuses it) | Kit cases with the capability undeclared give the pushdown's answer; a limit exceeded is a named error | T2.1b | C | No |
| T2.7 | Non-SQL provider and swapped data layer | A minimal non-SQL test provider; a second `DataLayer` composed at the composition root | The provider passes the kit; the same prediction runs on the swapped layer; a guard shows no domain crate changed for it | T2.5 | C | No |
| T2.8a | Ranking, materialize side | `top_k`, candidates, the decomposition in `sdm-request`; `problem_type` for ranking; the forecast's `top_k` and `MOST_PARTS` refusals in the core | S1 ranking cases at materialize, both hosts | T2.1b | D | No |
| T2.8b | Ranking, predict side | Scores, rankings, `prediction_statistic`, PD-1, reassembly; `response_field` for rankings; `instance_id_type`, `prediction_type`; M10 | S1 ranking cases at predict, both hosts | T2.8a | D | No |
| T2.8c | Multi-call `predict` (`main`'s 1fc5a686), 6–8 h | Up to 10,000 entities a call in place of the core's 1,000 (`PREDICT_ROWS`, `sdm-core/src/materialize.rs:160`), larger requests split with a per-batch walk and context, `rows_per_call`, rows numbered across calls, answered parts on a failed call, the 100,000-entity and 100-call ceilings, the per-batch 30 MiB body refusal and its remedy texts, the pre-send context check per call, the batched-forecast refusal; the batched-ranking refusal lands with whichever of T2.8a and T2.8c lands second | Cases: `rows_per_call=3` over 8 entities, a failure on call 2, invalid `rows_per_call` values, each ceiling's boundary, a forecast over one call, the 30 MiB refusal with one row and with several, both hosts | T2.1b | G | No |
| T2.9 | Forecasting | `num_forecasts`, `lag_timesteps`, quantiles; `response_field` for quantiles | Forecast cases on rust-host and python-core | T2.1b | E | No |
| T2.10 | PQL targets | `aggregation`, `open_window`, `condition`, `logical_target`, `target_filter`, `static_target` | Label vectors from the reference; cases per rule | T2.1b | E | No |
| T2.12 | Schema rules | `float_nan` first, `float_signed_zero`, `key_tie`, `key_name_order`, `composite_key`; FR P3-8's key-cast vectors; the nine `edit.*` S1 cases, diagnostic in Phase 1 (as-built T1.24b) though reached on both core hosts, admitted as acceptance | The key-type pair matrix for composite keys on all three engines; cases per rule; the nine `edit.*` cases acceptance, and, in a later landing of its own (protected files), their interim gate retired (`contract/tests/test_edit_cases.py`, `sdm-testkit`'s `tests/edit_cases.rs`, `contract/pinned/edits.core.json`) once the ratchet holds them, with an exit showing which replacement assertions catch each failure the interim gate caught (its `core.edits` pin, which `compare.py` does not compare, and the edit-history mutations); any assertion with no shown replacement is kept | T2.1b | F | Yes (the retirement, its own landing) |
| T2.11 | Sampling options | `anchor_entity`, `context_anchor_time`, `exclude_cols_dict`, `num_hops`, `num_neighbors`, `max_pq_iterations`, `use_prediction_time`, `return_embeddings`, `unseeded`, `parent_table` (links looked up by exact value, so distinct decimal keys never merge, KO2-35); `response_field` for embeddings; M6 | Cases per option; `parent_table` on the RelBench sources that hit it and on two decimal keys one float64 apart | T2.12 | F | No |
| T2.13 | Query and option long tail | `entity_filter`, `entity_ids`, `assuming`, `evaluate`, `explain`, `target_transform_null`, `inference_config_text`; `response_field` for explanations and item metadata; U1 | Cases per rule | T2.1b | D | No |
| T2.17 | Exact decimals (KO2-35) | The core's three decimal refusals lifted: a decimal class target (one class per distinct exact value, written as the reference writes it), a decimal sum past 38 digits (computed wider, rounded once), a decimal feature float64 can't hold (sent as the nearest double); keys stay exact; the differences from `main` registered | Re-recorded `decimal-vectors.json` and decimal `key-cast-vectors.json` answered where the reference answers; new cases (a `DECIMAL(10,2)` sum, a sum past 38 digits, an inexact `DECIMAL(38,18)` feature, a scaled key with predict's id, a wide key) on all three databases; the key-type pair matrix for decimal keys | T2.1a, HO2 | G | No |
| T2.18 | Any column name (KO2-36) | Step 1: the host stops checking column names on the core route and the core keeps every column as data; step 2: `(table, column)` carried separately through the parsed query (`sdm-pql/src/query.rs:42,341`, `ast.rs:310,709`, `sdm-core/src/pql.rs:559`), a doubled backtick inside a quoted name, line breaks inside quotes; the differences from `main` registered, the PQL corpus's new accepts as core-only | Cases on all three databases: columns `a.b`, a backtick and a line break, one of them the primary key, each also as target, entity, filter and aggregation column; vectors per character | T2.1b | E | No |
| T2.19 | Two-path sampling (`main`'s eb6c68eb) | A table reached by two paths in one hop read only up to the slices it held when the hop began (`sdm-core/src/subgraph.rs:235-266`); one column linking two tables | Cases on all three databases: `diamond`, `same-table-twice`, `chain`, `star` (`main`'s `test_sql_sampler_parity.py:304-330`), a `from_source` diamond compared at the payload, one column linking two tables | T2.1a, HO2 | F | No |
| T2.14a | Bridge freeze widened | FR P3-2: `_send_body`, `_rust/codec.py`, `edits.py`, `errors.py` brought under the freeze, re-baselined | The freeze's self-test; its post-landing state judged against itself | T2.1a | A | Yes |
| T2.14b | Host clean-up | Response bytes untouched; `Request` pickles (H23) with their fixture test, and a `main`-era `Schema` pickle with `conversion_notes`; ledger rows deleted with `moved` declarations and the baseline `--write`; the host wording rows G15, M4, FX-1, FX-4 (KO2-32) | Pickle fixtures from the reference load; an unknown state version refused by name; `rust-host` runs every S1 case (lesson 17) | T2.14a, T2.8b, T2.8c | A | No |
| T2.14c | Bridge freeze lowered | The functions the core route no longer reaches leave `COVERED` (`bridge_freeze.py:107-110`) | The freeze's self-test; its post-landing state judged against itself | T2.14b | A | Yes |
| T2.15 | Offline L7 | `contract/tools/l7.py`, added to `contract/protected-paths.txt` with the census, a measurement tool over recorded runs (statement counts, CPU per scenario, in-memory boundary overhead, message sizes, zero-copy and no-pandas checks, import time, native library size), its step in the gate, judged against the limits in the locked-set file (KO2-29) | Seeded faults: an extra statement, a pandas round trip in a provider, a scenario slowed past 1.1 times, median CPU above the reference's, boundary overhead over its bound; a candidate that weakens the tool is still judged by the checkpoint's | T2.1b | C | Yes |
| T2.16 | The slice gate | Register reconciled; `main` merged again if it moved since T2.0c (KO2-7); `gate.sh --locked 2a`; L3; the census; offline L7; the held-out run; the evidence pack; the flip dry run; the whole-phase review; any carried row no lane reached | The locked checks below | all | A | No |
| HO2 | The new held-out set | About 40 seeded cases across the three databases, written by a held-out author (briefs/heldout-author.md) who reads no implementation, in T2.1a's case format, kept outside the repository and every worktree; a tested converter; its digest recorded; check 9's two dummy bundles run through `heldout.py counts` | Frozen before the first product package starts, and those lanes wait for it; only the orchestrator and the phase-end run read it | T2.1a | orchestrator | No |
| CI5 | Faster CI | `rust-host` job split from `rust-test`; sharded `cargo test` and contract tests, each shard checking its list adds up to the whole; C4 | Same tests run (a seeded missing shard fails); the locked set's job patterns still match; timings recorded | T2.0b | H | Yes (`ci.yml`) |
| CI6 | Merge-queue landing | `land_chain.sh` reading a queue file, ejecting only the failing package and those above, `auto_land.py` running `mech_confirm.py` | Self-tests: an ejection rebuilds the stack; a clean move requeues | T2.0 | H | No (orchestration) |
| CI7 | aarch64 wheels and runtime cells (KO2-38, KO2-40) | `native-wheel` on native `ubuntu-24.04-arm` beside x86_64 (a mirrored `manylinux2014_aarch64` image by digest, rustup-init's checksum, the per-arch tags in `native_wheel.py`, `rust_notices.py` over both targets); `wheel-smoke (aarch64-floor)` and `(aarch64-newest)`; six `runtime-install (<runtime>)` jobs, each preinstalling one Databricks runtime's Python, pandas, NumPy, pyarrow and PyYAML, running the documented install line verbatim, `pip check`, then the binding and client tests | All new jobs green; seeded faults: a wheel set missing one arch, an install that moves the runtime's NumPy, a resolution failure | T2.0c | H | Yes (`ci.yml`) |

**Critical path** (one fix round and a two-reviewer confirmation each): T2.0e (3 h) → T2.0c (3 h; its merge and recording run beside T2.0e) → T2.1a (8 h) → T2.1b (4 h) → the product packages → T2.14b (3 h) → T2.14c (1 h) → T2.16 (7 h).
The product packages are about 79 hours of work in six lanes, four at once: B 14 h (T2.2 5, T2.3 6, T2.4 3), C 15 h (T2.5 4, T2.6 4, T2.7 3, T2.15 4), D 13 h (T2.8a 4.5, T2.8b 4, T2.13 4.5), E 13.5 h (T2.9 3.5, T2.10 4, T2.18 6), F 12 h (T2.19 2, T2.12 5, T2.11 5), G 11.5 h (T2.8c 7, T2.17 4.5), so about 20 to 21 hours of wall clock; T2.17 and T2.19 start beside T2.1b, once HO2 is frozen.
In all, about 49 to 50 hours, about two days of wall clock with agents running overnight (KO2-8), against the approved design's 37.
What grew: `main` first (T2.0e, T2.0c), all cases on three databases (T2.1a), the table source (T2.1b), T2.8c at 6–8 h (now beside T2.8a and T2.8b, not after them) and three new packages (T2.17, T2.18, T2.19, about 12.5 h); the owner's answers are off the path.
Lane H (T2.0d 3 h, T2.0b 3 h, CI5 3 h, CI6 3 h, CI7 7 h) starts at once and ends well before T2.16; a fifth product slot would save about 4 hours at the cost of saturating the machine.
The table has 34 landings (T2.0's done; T2.12's retirement of the interim edit gate is its second), plus a correction package after any design checkpoint that finds something (KO2-10, about seven checkpoints) and HO2, which is not a landing: about 17 landings a day, within the handbook's 15 to 20.

### Lanes and machine

The machine has 8 cores and 30 GB of memory, and a cold `cargo build --workspace` takes about 6 minutes; six lanes building at once would saturate it.
Run at most four product lanes at once (D first, since T2.14b waits for it; then B, C, E, F and G as slots free), plus lanes A and H, with a shared sccache, and keep each lane's Cargo target across landings (`SDM_CLEAN_AFTER_LAND` off, phase-2-prep.md item 3).

## Owner check-ins

- Kickoff: done (decisions.md); Q-I below and KO2-20's confirmation go to the next check-in.
- Design checkpoints every 4 or 5 landings (KO2-10), by both reviewers, of the landed range against learnings.md and this design, run beside the lanes; each one's corrections are their own package.
- End: the go or no-go packet (handbook, "Ending a phase", step 6).
- Twice a day, a five-line check-in, with the batch of register entries reviewers can't approve (KO2-11); questions found mid-phase are marked owner-pending and work continues around them.

## The evidence pack

As Phase 1's (`P1 d:528-541`), on the final SHA, generated by the branch's own runs, plus:
- the per-case table by database (DuckDB, SQLite, Postgres) and by host (python-core, rust-host);
- the refusal census (choice 10) with real-data hit counts;
- the conformance-kit results for every provider, the non-SQL provider and the swapped layer;
- the reference updates: the `main` commits merged and each changed reference line with the commit it traces to;
- L3: seed sets, counts and every divergence classified;
- offline L7 against its limits;
- the held-out run;
- the host-held ledger before and after (lesson 17);
- the flip dry run's counts, for information.

## Gates

### Merge rules

As Phase 1's (E29, E43, E73(c), E130, E131, H2, H3), with these owner rules: two reviewers on every round, confirmations and docs included, both approving the same SHA (KO2-9, replacing E25's one-reviewer option); a new case lands with its code, written by its test author (KO2-13); register classes (a) to (c) approved by reviewers, the rest batched to the owner (KO2-11); a check failing twice the same way goes to the owner while other lanes continue (KO2-12).
Package IDs are `T2.<n>[a-z]` and `CI<n>` (land.py's pattern); HO2 is not a landing.
Register IDs come from per-lane blocks: lane A `P2A-1..19`, lane B `P2B-…`, and so on to lane H; the IDs start with a capital letter, as `capture_check.py` and `coverage.py` read them.
Each lane writes only in its own register banner, and each package writes its own tracker file (KO2-17); decisions.md takes only the owner's rows (KO2-14).

### Locked acceptance checks

Locked at T2.0a's landing; a third change goes to the owner (plan section 9).
Each is falsifiable: it passes on the candidate and fails on the seeded fault named, as Phase 1's were (`P1 d:555-564`; H26).
Phase 2's locked set is [`locked-set.yaml`](locked-set.yaml), in the format T2.0 landed (`sdm-locked-set/1`; Phase 1's file is `phase-1/locked-set.yaml`). Phase 2 has one half, `2a`, run as `gate.sh --locked 2a`, holding:
- `cases`: all 52 of today's S1 cases, each run on all three databases (KO2-19), so the nine `edit.*` cases that are diagnostic today fail the locked run until T2.12 admits them; a floor no case may leave, while cases admitted later are held by the ratchet and by coverage S1's rule that every in-scope cell has an acceptance case;
- `checks`: checks 1 to 8, 10 and 11 below, each mapped to the gate steps that judge it (12 is the `every` check and never in `checks`); check 9, the held-out run, stays outside the gate, because it runs once, after the locked gate, through `phase_end.sh`'s held-out step (`heldout_run.sh`, then `heldout.py counts`), and its verdict goes into the packet beside the gate's evidence file;
- `steps`: from T2.0's closed list (`inventory`, which the reader requires beside `cases`, and `seeded`) plus the ones T2.1a adds to it: `census` (T2.0d's tool), `coverage-s1` and `l3` (T2.1a), `l7` (T2.15) and `ci`;
- `every`: `'12'`; `reference` is left to the reader's default, check 1 holding `compare`;
- `limits`: KO2-29's L7 limits, a key T2.1a adds to the reader;
- `ci`: for each check run in CI, job-name patterns fixed now: per database for checks 1, 3 and 5, so no database's leg can be missing; checks 8 and 10 as below.
  CI5 keeps these names when it splits or shards: a shard appends `, shard <n>` inside the parentheses, which the patterns' trailing `*` takes, and its split `rust-host` job joins check 5 only if the owner approves that change to the locked checks.
  The `ci` step resolves the run to read as `land.py verify` and `final_gate.sh` do under E131: for a candidate, its own exact-SHA run; for an empty landing commit, the run of the first non-landing commit below it, after checking the two trees are equal; for a moved landing (E130, `land.py --moved-from`), the moved-from commit's run, after the same checks `land.py verify` makes; it records both the judged SHA and the tested SHA and run. The step reports per check: a failure names the check number and the pattern. Every pattern must match at least one job, and every matching job must have concluded `success`; a skipped or cancelled job fails, and a run of any other tree fails.

T2.0a writes the file whole, and its landing adds it to `contract/protected-paths.txt` and names it in `Protected-paths`, so the ratchet refuses any later change to it that no landing declares; no package edits it. Until T2.1a lands, T2.0's reader refuses the half by name for its new step names and keys; no landing runs `--locked 2a` before T2.16, and T2.1a's tests show the half is read. A step whose tool has not landed fails the locked run by name. The closure check compares the file with this table.

| # | Check | Tool and machine | Seeded fault that must fail it |
|---|---|---|---|
| 1 | Full S1 L2 parity: every S1 case in scope, on DuckDB, SQLite and Postgres, per field, on python-core and rust-host against the reference, Postgres also on common inputs (H24); every difference has an approved register entry; `coverage.py --gate S1` with no empty cell | `gate.sh --locked 2a` on the reference machine, plus CI's `contract-python (<cell>, <database>)` jobs and `cross-host (<database>)` | A changed SQL result, a dropped warning, a case missing on one database, an empty coverage cell, a payload defect on common Postgres inputs |
| 2 | No unapproved refusal of an input the reference accepts, and no panic; each of E92's 37 rules lifted or kept by an owner answer that quotes its real-data count | `ratchet.py` and the census (choice 10), reference machine | A rule re-added on one input; a refusal with no entry; a kept rule hit by a real source with no owner answer |
| 3 | L4: SQL goldens executed on DuckDB, SQLite and Postgres | CI: `l4-goldens (<database>)` | A seeded SQL change without a golden update |
| 4 | L5: conversion tables on pandas 2 and 3; three drivers give identical Arrow | CI: `l5-conversion*` | A seeded unit or scale change in one driver's path |
| 5 | Conformance kit: every built-in provider, the non-SQL provider and the swapped layer pass; the same prediction on the swapped layer with no domain-crate change | CI: `sdm-testkit` in `rust-test*`; the Python kit in `contract-python (ref-pandas3, <database>)`, against that database's provider | A provider that truncates a page; a capability that fails silently; a domain-crate edit in the swap |
| 6 | L3: `differential.py` on the agreed seed sets per database against the reference, every divergence classified (lesson 12) | Reference machine | An unclassified divergence |
| 7 | The reference and the phase branch at `main`'s head as of the gate run, the reference rerun; each changed line traced to a `main` commit and registered | `ratchet.py`'s reference rule (T2.0e), reference machine | A new reference commit with no owner row; a changed line citing no commit |
| 8 | Host-held rows in scope moved: `rust-host` runs every S1 case; the freeze lowered; reference-pickled `Request` objects load and an unknown `RequestState` version is refused by name | CI: `contract-python (*)` (the pickle fixtures), `cross-host (*)`, `rust-lint` (the bridge freeze) | A rule re-added on the Python side only; a corrupted pickle fixture; a bumped version |
| 9 | Held-out (HO2): the set under H11's threshold, on python-core and rust-host; outside `gate.sh --locked 2a` | `phase_end.sh`'s held-out step once, at the phase end (`heldout_run.sh`, then `heldout.py counts` for the verdict) | Before the one shot, two dummy bundles in a disposable output and stamp directory, each run for real through `heldout.py counts`: one holding a panic and one with differences over the threshold, each asserted a no-go (the dry mode only prints commands, `heldout_run.sh:40-45`) |
| 10 | Boundary and packaging: L0 clean, no `pyo3` in the pure-Rust host's tree, the wheel installs and loads in a clean container on x86_64 and aarch64 and on each promised runtime's package set (KO2-38, KO2-40) | CI: `rust-lint` (L0), `rust-test*` (the `pyo3`-free tree), `wheel-smoke (*)`, `wheel-smoke (aarch64*)`, `installed-wheel`, the six `runtime-install (<runtime>)` jobs | Phase 1's seeded L0 faults (`selftest.py`); a wheel set missing one arch |
| 11 | Offline L7 within the locked-set file's limits (scope item 8, KO2-29) | T2.15's tool, reference machine | An extra statement; a pandas round trip in a provider; a scenario slowed past 1.1 times; median CPU above the reference's; in-memory boundary overhead over its bound |
| 12 | Every gate step the half runs passed (T2.0's `every` check) | `gate.sh --locked 2a`, reference machine | Any one step seeded to fail |

### Process rules that differ from Phase 1

- The design and docs land through the topic branch as T2.0a; every phase-branch commit is a landing (handbook, step 2). T2.0c lands from its topic branch, its merge of `main` admitted by T2.0e's rule.
- Judging-path packages and parts (T2.0, T2.0e, T2.0a, T2.0c, T2.0d, T2.0b, T2.1a, T2.4's CI jobs, T2.12's retirement of the interim edit gate, T2.14a, T2.14c, T2.15, CI5, CI7) land alone, before anything relies on them (lesson 13). T2.0c carries one core change (the inference options) with the re-recorded lines because each judges the other.
- The orchestrator runs the reference machine's work and HO2; implementers and test authors never open held-out data.
- E49(a) and E56(a) freeze tooling at bug fixes; T2.0 was a demonstrated integration failure, and T2.0e, T2.0b, T2.0d, T2.15, CI5, CI6 and CI7 run under KO2-15, KO2-18 and KO2-38 to KO2-40.

## Risks

| Risk | Guard | Lesson |
|---|---|---|
| The core grows pandas or driver behavior to match the reference on new databases | Providers hand Arrow to the core; differences go to the register; L5 checks three-driver equality | 1, 3 |
| Copying the reference's accidents (SQLite's unseeded order, NaN ranks, `main`'s float-cast decimal keys) | Registered keeps and drops: RS-2, RS-6, PD-1; KO2-23; exact decimals under KO2-35 | 2 |
| Hand-rolled paging, fallback or wide decimal sums | A `dependencies.md` row with a measured reason for each (choice 9); the kit judges paging and the fallback | 4 |
| Ranking, multi-call predict or the new databases meet the real host only at the gate | Both hosts run each case from its first landing; `cross-host (<database>)` in CI; L3 before the gate | 5, 17 |
| Tests that check internals | Public-route cases by a separate test author; reference-recorded vectors | 6, 16 |
| A fix breaks neighbors (a dialect fix changes DuckDB) | Ratchet over every case on every database | 7 |
| Hand-judged refusal lists | The census from the code (choice 10); coverage generated | 8, 18 |
| Process without brakes; lanes fragmenting | Stop-rule counters in the tracker; one lane per package; at most four product lanes; design checkpoints every 4 or 5 landings | 9, 10 |
| Review agreements lost | Every finding ends as a fix, a carried row or a recorded "no"; T2.16 is the backstop | 11 |
| Differences found by review | Each implementer runs `differential.py` and the census and classifies every mismatch before review | 12 |
| Judging and product changes together | Judging packages land alone; T2.1a settles the judging path before admissions | 13, 20 |
| Contract bumps collide, or Phase 3 needs one for metric views | One bump (T2.1b), holding `main`'s new vocabulary and the table source; U13's renumbering script | 14 |
| Evidence on the wrong SHA | `land.py` exact-SHA rules; `final_gate.sh` | 15 |
| Key types across a relationship give wrong rows | Per-relationship typing; the key-type pair matrix in T2.2, T2.3, T2.12 and T2.17 | 19 |
| A replacement for a reference crash misses a reading | Choice 11's crossed vectors, an exit of each such package | 21 |
| A benchmark on generated data | An accuracy benchmark only on real data (KO2-27); the live benchmark's `different` comparison fails the run (carried row G8) before any live rerun | 22 |
| Ranking or multi-call needs two or more fix rounds | Split T2.8a, T2.8b, T2.8c; vectors in the test-author batch; result shape fixed in T2.1b | 7 |
| `composite_key` reaches key sets, the graph and the sampler | Its contract shape in T2.1b; if it slips, it stays refused by name and goes to the owner with its census count | 19 |
| The update from `main` changes behavior | `main` first (T2.0c); each new difference traced to a `main` commit; a second update at the gate | 2 |
| A database leg of CI passes empty | Per-database job patterns in the locked set; a job that runs no case fails | 9 |
| The machine saturates with parallel builds | At most four product lanes; shared sccache; CI5 | 10 |

## Release work deferred to Phase 5

The owner left release work for the end, with this note (KO2-42); none of it blocks Phase 2, and CI7 brings aarch64 forward:
- `publish.yml` rewritten for a native client: wheels per platform with the pinned toolchain, a (distribution, platform) set check, no client sdist unless the owner wants one, the SBOM taken from the built wheels, `twine check` over all of it;
- Rust in the open source inventory and the SBOM (`cargo cyclonedx` or `cargo auditable`, per target), `THIRD-PARTY-NOTICES-RUST.txt` per target set, and whether nSpect reads `rust/Cargo.lock` and how the dev-only DuckDB tree is declared;
- macOS, Windows and musl client wheels from the support matrix, each smoke-tested;
- `.gitlab-ci.yml`'s jobs that can't build the epic, retired or left before the epic merges (GitHub only, KO2-43);
- CodeQL's c-cpp leg dropped with the engine (T2.0c adds a Rust leg);
- the CHANGELOG entry and `stability.md`: the platform set, the raised floors (KO2-37), `_native`, and whether the release is minor or major;
- signing native binaries, if release rules require it;
- the live Databricks job matrix re-run on the epic before it merges (KO2-38).

## Owner questions

The kickoff's and the audits' questions are answered (decisions.md); KO2-20 (Q2.2, replay) proceeds as recommended until the owner confirms it.
One new question, with a recommendation and its cost either way:
- **Q-I.** Between T2.0c and the package that lifts it, the core refuses some inputs the new reference answers: decimal class targets, decimal sums past 38 digits and inexact decimal features (T2.17), columns with a dot, backtick or line break on python-core (T2.18), and 1,001 to 10,000 entities and `rows_per_call` (T2.8c).
  Recommended: the owner approves, in one answer, T2.0c's interim register entries for these, each naming its lifting package and void once that package lands (T2.16 fails on any left).
  Without it, T2.0c's landing waits for the owner to approve each entry (about ten), likely one check-in cycle.
