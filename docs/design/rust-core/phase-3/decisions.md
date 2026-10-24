# Phase 3 decisions

The owner's decisions for Phase 3, newest last.
As in Phase 2 (KO2-14), this file records only the owner's own decisions; the orchestrator's and implementers' choices go in [design.md](design.md) and each package's tracker.
Phase 1's and Phase 2's rows ([phase-1/decisions.md](../phase-1/decisions.md), [phase-2/decisions.md](../phase-2/decisions.md)) stay in force unless a row here changes them; cite them by ID, never copy them.
A status `approved by owner <date>` records an answer the owner wrote; an agent never writes that status for itself (plan section 2, decision 12).
Each Decision cell says no more than the owner's words, which the Owner's words cell quotes exactly, with the question as it was asked; a row that would need more is an open question in design.md instead.
A row is at most 1,000 characters.
Row IDs start `KO3-`: a capital letter, as `capture_check.py` and `coverage.py` read IDs, and a `3` so they never collide with earlier phases' rows.

**Reference approvals.**
A row approves a new reference commit exactly as Phase 2's do (phase-2/decisions.md, "Reference approvals"); Phase 3 has none yet.

## Before the design: package size (2026-10-20)

Source: the owner's answer on 2026-10-20 to the orchestrator's package-granularity question, recorded in the orchestrator's state file `phase-2/owner-answers-2026-10-17.md`, section "2026-10-20 (UTC): package granularity".
The question, as asked: "1 owner decision rows that approve recorded differences ride with the package that needs them (reference approvals and comparison loosenings go in the landing just ahead, never alone); 2 CI-config changes ride with the package that needs them; 3 Phases 3-5 at about 7-10 landings each, integer IDs (T3.1, T3.2), groups capped at about 10 hours of implementation."

| # | Decision | Owner's words | Status |
|---|---|---|---|
| KO3-1 | Owner decision rows that approve recorded differences ride with the package that needs them; reference approvals and comparison loosenings go in the landing just ahead, never alone; CI-config changes ride with the package that needs them; Phases 3 to 5 at about 7 to 10 landings each, with integer IDs (T3.1, T3.2), groups capped at about 10 hours of implementation | "yes to all" | approved by owner 2026-10-20 |

## The root-cause analysis's decisions (2026-10-21, evening)

Source: the owner's answers to the nine decisions at the end of Phase 2's root-cause analysis (`$SDM_ORCH_STATE/phase-3/rca/RCA.md`, "Decisions only the owner can make"), recorded in `$SDM_ORCH_STATE/phase-3/owner-answers-2026-10-21.md`.
Each question is summarized in the Decision cell as it was asked; the RCA file holds its full text and recommendation.

| # | Decision | Owner's words | Status |
|---|---|---|---|
| KO3-2 | RCA decision 1, one severity bar for both reviewers (any measured difference from the reference on a public call, any red gate or CI job the change caused, and any decision row wider than the owner's words is at least P2): not decided; the owner asked a question back | "1. Severity: Has this really been a problem?" | answered by KO3-47 |
| KO3-3 | RCA decision 2, stop and redesign: when the same family of finding blocks two rounds in a row, or a fix's own change causes a new blocking finding twice in a row, stop patching; without a large redesign session for a simple missed case | "2. Yes to Stop and redesign, but sometimes you might've just missed some dumb case so we shouldn't need a large redesign sesion for that..." | approved by owner 2026-10-21 |
| KO3-4 | RCA decision 3, speed infrastructure (the stacked queue, the CI fix, environment and disk hygiene, the command hook) is schedule work built in Phase 3's first package, not "machinery" | "3. Yes please." | approved by owner 2026-10-21 |
| KO3-5 | RCA decision 4, docs-only fix rounds: the reviewer who raised the finding confirms its fix; any round writing decision rows or register entries keeps Claude | "4. Sure..." | approved by owner 2026-10-21 |
| KO3-6 | RCA decision 5, Phase 3's budget (at most 10 landings and a 16-hour planned critical path): approved, then replaced the same evening by KO3-12, which rejects hard budgets | "5. Yes" | superseded by KO3-12 |
| KO3-7 | RCA decision 6, what "match the reference" covers: answers, refusal kinds and user-visible messages; statement counts, warning repeat counts, tie order among equal scores and hash order only when a user can see them, otherwise one standing register class the owner approves once | "6. Yes" | approved by owner 2026-10-21 |
| KO3-8 | RCA decision 7, a blocking hook for the banned shell commands for every Claude agent; Codex keeps the watcher | "7. Yes" | approved by owner 2026-10-21 |
| KO3-9 | RCA decision 8, overnight delegation for Phase 3, on condition each decision is written plainly when made and put to the owner in the morning, before the final candidate | "8. Yes" | approved by owner 2026-10-21 |
| KO3-10 | RCA decision 9, process lessons live in `learnings.md` with the product lessons | "9. Yes" | approved by owner 2026-10-21 |
| KO3-11 | The lessons are enforced by mechanisms, not by updating a Markdown file | "The problem is all these "lessons" aren't being enforced, you just seem to be updating some random md file and not officially enforcing them. How can you enforce? Think carefully. Could be templates, scripts, github actions, pre-commit hooks, etc. You have a lot of lessons learned / RCAs found but they don't translate to your decisions above" | the owner's instruction, 2026-10-21 |

## The five Phase 3 process changes (2026-10-21, evening)

Source: the owner's answers to the orchestrator's first proposal of five process changes and then to the revised five, recorded in `$SDM_ORCH_STATE/phase-3/owner-answers-2026-10-21.md`; T3.0 built the revised five.

| # | Decision | Owner's words | Status |
|---|---|---|---|
| KO3-12 | No hard budgets of landings or hours; change the process itself | "1. The problem with setting "budgets" like the 10 landings and 16-hour critical path is it can lead to shoddy work. Is there a way to change the process itself? Imagine 16 hours go by and it just throws its hands in the air" | approved by owner 2026-10-21 (with KO3-17) |
| KO3-13 | Stack all the time, with CI mandatory every N packages, for example 4 | "2. Yes we should be stacking all the time. But maybe this can be limited to every N packages (e.g. 4 packages) before CI is mandatory" | approved by owner 2026-10-21 (with KO3-17) |
| KO3-14 | Fix the slow checking-tool tests in CI, with a warning and fix system | "3. Yes please. What even are these tool tests? and yes we need some warning + fix system." | approved by owner 2026-10-21 (with KO3-17) |
| KO3-15 | Reviewers do not live-test; checking against the old client is the implementer's, the tests' and CI's | "4. Do we think this "compare with the old client" will add too many slowdowns in the review phases? The reviewers shouldn't really need to live test, that's the point of CI or the implementer and the tests we write." | approved by owner 2026-10-21 (with KO3-17) |
| KO3-16 | Test environments must fit the machine's cores, memory and disk | "5. This "own test environment" thing will blow up ssd and memory, remember that we can't run too many jobs in parallel due to limited cpu cores, memory, and disk." | approved by owner 2026-10-21 (with KO3-17) |
| KO3-17 | The revised five changes are built (T3.0), then Phase 3's design begins | "Yes, do these. After this is done, we will begin the phase 3 design doc" | approved by owner 2026-10-21 |

## Live access (2026-10-22)

Source: the owner's answers on 2026-10-22 to the orchestrator's nine live-access questions, recorded in `$SDM_ORCH_STATE/phase-3/owner-answers-2026-10-21.md`, section "2026-10-22: live access answers".
The questions, as asked: 1 the Databricks CLI profile; 2 the catalog, schema and volume; 3 the SQL warehouse; 4 the metric view and GPU serving endpoint; 5 the job-run budget; 6 which notebook; 7 Snowflake access; 8 the Lakebase instance; 9 whether aarch64 is in scope.
The repository is public, so private names in the owner's words are replaced by a bracketed description; the unredacted words stay in the state file.

| # | Decision | Owner's words | Status |
|---|---|---|---|
| KO3-18 | The Databricks CLI profile for live work is the one the owner named | "1. [the owner's profile name]" | approved by owner 2026-10-22 |
| KO3-19 | The catalog, schema and volume are the ones the owner has used before, found by the orchestrator | "2. Just whatever I have used in the past, find it. Might be some [workspace name] sandbox thing related to kumo relational" | approved by owner 2026-10-22 |
| KO3-20 | A SQL warehouse may be used | "3. Yes" | approved by owner 2026-10-22 |
| KO3-21 | Existing metric views and GPU serving endpoints the owner created are reused | "4. Yes, there should already be one or more that I have created" | approved by owner 2026-10-22 |
| KO3-22 | Most of the implementation is written without incremental checks on Databricks; the live tests run at the end | "5. I suggest coding the majority of the implementation instead of running incremental checks in databricks. Then run the tests at the "end". Does that make sense? Databricks warehouse can be slow and we want to avoid spending a long time re-testing after every change" | approved by owner 2026-10-22 |
| KO3-23 | Phase 2's datasets and testing are replicated on the warehouses, starting from the quickstart notebook, which alone is not enough for testing | "6. Can we replicate whatever dataset/testing you've done in phase 2 here? and yes there's already the quickstart for databricks in the https://github.com/NVIDIA/structured-data-models-client/blob/main/examples/databricks/kumo_relational_quickstart.ipynb that you can start with, but it's not enough for testing" | approved by owner 2026-10-22 |
| KO3-24 | Snowflake is tested with the credentials in the owner's env file, which is not read directly | "7. I have added [a private env file] which contains three env vars: SNOWFLAKE_URL, SNOWFLAKE_USERNAME, SNOWFLAKE_ACCESS_TOKEN. Avoid reading it directly as it has credentials. You can use this for testing." | approved by owner 2026-10-22 |
| KO3-25 | The Lakebase instance is one in the owner's workspace, found by the orchestrator | "8. You should be able to find another one in our [workspace name] workspace" | approved by owner 2026-10-22 |
| KO3-26 | Both x86_64 and aarch64 are in scope | "9. Yes, both x86 and aarch64. I'm not sure you're right, I believe we support both, but double-check" | approved by owner 2026-10-22 |

## Java in Phase 3 (2026-10-22)

Source: the owner's message on 2026-10-22, recorded in `$SDM_ORCH_STATE/phase-3/owner-answers-2026-10-21.md`, section "2026-10-22: Java in Phase 3", written after reading the draft design, which kept Java out under Phase 2's KO2-34.

| # | Decision | Owner's words | Status |
|---|---|---|---|
| KO3-27 | Start adding Java support in Phase 3 (Phase 2's KO2-34 had deferred Java until S2 is switched for Python) | "BTW side note - is it possible to begin the Java implementation as well? I'm looking at our migration-plan.md at https://github.com/NVIDIA/structured-data-models-client/blob/rust-core/docs/design/rust-core/migration-plan.md and it includes "Java follows on Databricks" so I'm wondering why you've not considered it here. I think we should start adding Java support as well in phase 3" | approved by owner 2026-10-22 |

## The Java questions (2026-10-22)

Source: the owner's answers on 2026-10-22 to the design's Java questions Q16 to Q22 (as numbered in the design at 8f9f4b1f; design.md numbers them J1 to J7 since round 2), and to the follow-up on JDK 17, recorded in `$SDM_ORCH_STATE/phase-3/owner-answers-2026-10-21.md`, sections "2026-10-22: Java questions Q16 to Q22" and "2026-10-22: Q21 follow-up (JDK 17)".
Each question is summarized in the Decision cell as it was asked.

| # | Decision | Owner's words | Status |
|---|---|---|---|
| KO3-28 | Q16, a fifth lane for Java: Java on its own lane from the start, a fifth package in flight, with `heavy` queueing its builds | "16: Yes" | approved by owner 2026-10-22 |
| KO3-29 | Q17, the Java API's shape: mirror Python's public API with Java names and plain data | "17: Yes" | approved by owner 2026-10-22 |
| KO3-30 | Q18, logging: depend on the SLF4J 2 API only, bind nothing | "18: Sure" | approved by owner 2026-10-22 |
| KO3-31 | Q19, Maven coordinates and publishing: the orchestrator picks coordinates that follow Java's naming convention; the owner asks whether Maven is the right tool | "19: pick whatever makes sense and aligns with java naming convention. Side note - is Maven the right tool to be using, or something else?" | approved by owner 2026-10-22 |
| KO3-32 | Q20, platforms for the natives: Linux x86_64 and aarch64 at least; macOS only if there is an easy way that Phase 2 also used | "20: Do linux x86 and ARM minimum, not sure how to test macOS. If there's an easy way (e.g. github actions) then sure, as long as we also did it in phase 2. If we didn't, don't worry about it" | approved by owner 2026-10-22 |
| KO3-33 | Q21, JDKs in CI: the owner first leaned to JDK 21 only | "21: Maybe do Java 21, no need for 17 right? It's old." | superseded by KO3-34 |
| KO3-34 | Q21's follow-up: build for Java 17 and test on JDK 17 and 21, as recommended | "Q21, ok do as you recommend and support 17" | approved by owner 2026-10-22 |
| KO3-35 | Q22, runtime flags and Databricks questions: the orchestrator documents the flags, runs every Java test with them, asks the Java consumer to confirm, and includes the two metric-view questions in Java's live check | "22: Sure, you do it." | approved by owner 2026-10-22 |

## The design questions (2026-10-22, night)

Source: the owner's answers on 2026-10-22 (night) to the design's open questions as the orchestrator sent them, simplified, recorded in `$SDM_ORCH_STATE/phase-3/owner-answers-2026-10-21.md`, section "2026-10-22 (night): the design questions, simplified".
The questions are numbered as they were sent; their text, as sent, is recorded in the same file, section "Record: the simplified question list the owner answered on 2026-10-22 (night)"; each Decision cell gives the question as asked and its recommendation.

| # | Decision | Owner's words | Status |
|---|---|---|---|
| KO3-36 | Q5, the switch: flip every S0, S1 and S2 path to the core at once; an input still refused by name on a switched path (one of KO2-94's three) stays refused on the epic branch until Phase 4 lifts it | "5: Yes." | approved by owner 2026-10-22 |
| KO3-37 | Q6, old-engine tests after the flip: tests of old-engine internals run on the old route only until Phase 5, each listed in the evidence | "6: Sure." | approved by owner 2026-10-22 |
| KO3-38 | Q7, the job matrix: the 21 cells ask the quickstart's questions and the deleted notebook's, and the notebooks run once on the core | "7: Sure." | approved by owner 2026-10-22 |
| KO3-39 | Q8, live timing at the switch: kept, run at the end because it is slow | "8: Sure, but at the end probably since it's slow." | approved by owner 2026-10-22 |
| KO3-40 | Q9, view YAML values read as the old client's PyYAML reads them (YAML 1.1) | "9: Sure." | approved by owner 2026-10-22 |
| KO3-41 | Q10, the old client's sampling statements on Databricks and Snowflake (RS-6); and the owner's rule for differences: feature parity with the old client or better, where better means a better error or more support | "10: Sure, although you might potentially find bugs in the old client. Ideally our new client has feature parity or is even better than old client. Better means better error or more support." | approved by owner 2026-10-22 |
| KO3-42 | Q11, comparing the core with the old client on the old client's own recorded data gates | "11: Sure." | approved by owner 2026-10-22 |
| KO3-43 | Q12, L3 on the warehouses: 20 generated sources per warehouse | "12: Yes." | approved by owner 2026-10-22 |
| KO3-44 | Q13, the three refusals decided overnight in Phase 2 that refuse input the old client answers (a NaN `lag_timesteps`, `num_neighbors` text that is not a number, a negative `num_neighbors`) are lifted to match the old client | "13: Ok." | approved by owner 2026-10-22 |
| KO3-45 | D5, the offline judge for the warehouse legs, option (a): judged offline on the stand-in, statements diagnostic there, the dialects' SQL first run live in the end campaign; the two 100,001-row batching cases off the warehouse legs | "D5: Sure." | approved by owner 2026-10-22 |
| KO3-46 | G1, one standing register class the owner approves once for statement counts, warning repeat counts, tie order and hash order a user cannot see | "G1: Sure." | approved by owner 2026-10-22 |
| KO3-47 | G2, the severity bar, amended: a difference from the old client does not block a landing when the core does better, as KO3-41 defines better | "G2: I don't want to "block a landing" - discrepancies are ok as long as our rust core does better. Again, see above for my definition of "better"." | approved by owner 2026-10-22 |
| KO3-48 | G3, the design's "no" list of carried items, reopened only when the dry run or a Phase 3 case hits one | "G3: Accepted." | approved by owner 2026-10-22 |
| KO3-49 | G4, a quota or capacity error in a live run is retried once, and only a failure after the retry counts toward the stop rule | "G4: Sure." | approved by owner 2026-10-22 |

## Live resources, and the night's other answers (2026-10-22)

Sources: the owner's answers on 2026-10-22 to the orchestrator's four live-resource questions (1 starting the stopped Lakebase instance; 2 write safety for Snowflake, whose credentials carry the account's admin role; 3 where the tests read and write on Databricks; 4 the model serving endpoint), in `$SDM_ORCH_STATE/phase-3/owner-answers-2026-10-21.md`, section "2026-10-22: live resource answers"; and, in section "2026-10-22 (night): overnight mandate and Q14, Q15", the owner's mandate for the night and answers to the design's questions on a bigger machine and on protecting `rust-core` (Q12 and Q11 as numbered at 050f68f7).
Private names in the owner's words are replaced by a bracketed description; the unredacted words stay in the state file.

| # | Decision | Owner's words | Status |
|---|---|---|---|
| KO3-50 | The stopped Lakebase instance may be started for the tests | "1. Lakebase: Yes you can start it" | approved by owner 2026-10-22 |
| KO3-51 | A helper or safety lets the tests write only to one schema used for all testing; the client's own logic is read-only | "2. Create some helper script or safety that only writes to a certain schema (maybe a schema that we use for all our testing purpose). Then all our actual logic should be read-only within the client, so it's less of a concern." | approved by owner 2026-10-22 |
| KO3-52 | On Databricks, the tests read the existing schema where AdventureWorks is already loaded (not the schema marked safe to drop), found by the orchestrator | "3. We should already have a schema in databricks [workspace name] I believe. Find it. It's not the "safe to drop" schema. Find it, it might be a kumo relational or kumo rfm naming or something else in which adventureworks tables are already loaded" | approved by owner 2026-10-22 |
| KO3-53 | The tests use a serving endpoint for the Kumo relational model version 26, kept warm | "4. Use the Kumo relational / rfm v26 I believe. You can serve endpoint, no need to scale to zero since we need our testing to be quick" | approved by owner 2026-10-22 |
| KO3-54 | The Java packages as the design proposed them | "As for the Java packages - it looks good. I'm really hopeful that this can go by faster, since Python already did a lot of the work with interface planning." | approved by owner 2026-10-22 |
| KO3-55 | Q12, a bigger build machine: no | "Bigger machine: no" | approved by owner 2026-10-22 |
| KO3-56 | Q11, protecting `rust-core`: a pre-commit check or similar stops agents pushing it | "You should add pre-commit check or something to avoid pushing rust-core" | approved by owner 2026-10-22 |
| KO3-57 | Q10 of the simplified list also listed "7 of my Phase 2 overnight decisions for you to confirm: error wording; Postgres key equality; the data time-range wording in errors; and four others"; the owner's answer confirms the items it named that map to register entries exactly: "Postgres key equality" (P2G-4 and P2G-5, the two entries on Postgres numeric key equality) and "the data time-range wording in errors" (KO2-89); "error wording" names no entry exactly, so it confirms nothing, and P2D-2 to P2D-5 stand under the overnight delegation (KO2-88), listed for the owner's morning reading | "10: Sure, although you might potentially find bugs in the old client. Ideally our new client has feature parity or is even better than old client. Better means better error or more support." | approved by owner 2026-10-22 (P2G-4, P2G-5 and KO2-89 only) |
| KO3-58 | The night's mandate: by the morning, T3.0 done and this design reviewed and approved with the Java work; Phase 3 may start in full if it looks good | "Ok - I'm going to bed. By the morning: ensure that T3.0 is done, the design doc is ready (after reviews and approval), includes the java work. In fact, you can even get started on phase 3 fully if it looks good. Add this to your memory. Any questions before I leave?" | approved by owner 2026-10-22 |

## Warehouse table shape (2026-10-24)

Source: the owner's answers on 2026-10-24 to the orchestrator's question on how a schema reused across materialize calls reads a warehouse table that changed (A, keep what the first read found, as the old client does; or B, read each table as it stands at every run), and the items listed with it.
These rows record the owner's decisions, which the owner made on 2026-10-24.
The register's entries P3A-1 to P3A-5, the differences from the old client those decisions lead to, name these rows and stay proposed until the owner approves each entry.

| # | Decision | Owner's words | Status |
|---|---|---|---|
| KO3-59 | Plan B: no remembered table shape between materialize runs; each materialize reads each table's location, columns and types as they stand (rows were always read fresh) | "Ok, I'm convinced. Let's plan B." | approved by owner 2026-10-24 |
| KO3-60 | Item 1: a column the user adds that the source lacks is a named error, not silently dropped | "1. Sure" | approved by owner 2026-10-24 |
| KO3-61 | Item 3: a column retyped against the source is a named error | "3. Sure" | approved by owner 2026-10-24 |
| KO3-62 | Item 6: one query object freezing its graph at first materialize, on the old client, is registered and confirmed | "6. Hmm ok" | approved by owner 2026-10-24 |

## View conversion answers (2026-10-24)

Source: the owner's direct continuation answers recorded in `$SDM_ORCH_STATE/phase-3/owner-answers-2026-10-24-codex.md`, items 11 and 16 and the later answer retiring P3J-7 and P3J-8.
These decisions cover only the entries listed below; they grant no approval to P3J-12 through P3J-15 or the proposed Python emulation allowlist additions.

| # | Decision | Owner's words | Status |
|---|---|---|---|
| T35-OWNER-2026-10-24-ERRORS | Item 11: approve the clearer view errors P3J-1 through P3J-5 and P3J-9 through P3J-11 as presented | "Approved" | approved by owner 2026-10-24 |
| T35-OWNER-2026-10-24-YAML | Item 16: approve the YAML scanner differences P3J-6 as presented | "Approved" | approved by owner 2026-10-24 |
| T35-OWNER-2026-10-24-RETIRED | Retire P3J-7 and P3J-8 as obsolete proposals replaced by P3J-9; the register uses rejected for a retired entry, without claiming restored Python parity | "Ok retire 7/8 as superseded." | approved by owner 2026-10-24 |

## Kickoff answers

None yet beyond the rows above: the open questions are in [design.md](design.md), "Owner questions", each with a recommendation.
