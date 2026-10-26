# Phase 3 validation tracker

Updated 2026-10-26, 11:16 UTC.
Phase 4 is paused at the owner's request; current agents focus on Phase 3.

## Acceptance

- Compare the same data, queries, model, settings and prediction entities against the frozen Python client.
- Check schemas, values, nulls, row multiplicities, logical request payloads and required errors under the approved compatibility decisions.
- Record exact prediction parity when obtained.
- Where the frozen client already rejects an existing model case, a matching or closely matching rejection with the same underlying cause counts as parity; compare accuracy where both return predictions.
- For differing predictions, record the task's quality metric, Python baseline, Rust result and degradation; the owner accepts at most 5% worse prediction accuracy, with parity or a 1–2% difference preferred.
- Keep prediction disagreement separate from measured accuracy degradation.
- Record latency and throughput separately; the 5% accuracy threshold does not set a speed threshold.
- Reference recordings, captured observations, successful model calls and completed comparisons are distinct stages.
- Preserve original source/build IDs and failed evidence; no tolerance or comparison exclusion is implied by this tracker.

## Current matrix

Databricks SQL, Databricks Spark, Snowflake and Lakebase are four execution paths.
Metric views and semantic views are tracked separately below as warehouse feature coverage.
The coordinator alone edits this tracker; warehouse agents report evidence and next actions without editing it.
Use the existing common cases, pinned data, seeds, anchors and model settings across paths; retain backend-specific cases explicitly without adding scenarios.

| Area | Data and payload comparison | Model comparison | Quality verdict | Owner | Next action |
|---|---|---|---|---|---|
| Databricks SQL | All 302 original cases executed: 301 non-connector cases plus the repaired original connector; zero operator failures and cleanup clear; strict reconciliation pending | Complete corrected-client runtime comparison passes on actual `76eb785b91` source | All corresponding prediction rows and logical payloads exact, approved matching refusals retained | Databricks agent | Finish finite data reconciliation; wide-key error correction is landed and both original public cases now match the reference |
| Databricks Spark | Original monolithic run stopped after 108 complete cases when the serverless process ran out of memory; all complete bundles are retained; the missing 192 are running in eight serial 24-case jobs on actual `80c763a7be` | Complete original runtime comparison passes on actual `76eb785b91` source | Exact prediction/payload parity and approved matching refusals on that source | Databricks agent | Collect first bounded job `818601119704735`, immediately continue the remaining original batches, and verify all 300 cases plus cleanup |
| Snowflake | All 304 original cases captured and cleaned; finite reconciliation continues; all 16 stale digests have exact historical-input proofs and all 131 positional schema differences have raw catalog-order proofs | Fifteen ordinary/semantic pairs have exact payloads and nine exact positive predictions plus six approved matching refusals; original diamond also returns predictions | Diamond RMSE is 1.54% worse, within the owner's 5% limit; it is not exact payload/prediction parity | Snowflake agent | Single-case sampling oracle landed; finish finite reconciliation; original 20-source and 33-provider checks complete with explicit evidence |
| Lakebase | All 282 existing cases reconciled on recorded source; pinned 173106-row fixture intact | All six corrected-client runtime profiles pass the full paired comparison on actual `76eb785b91` source | Exact predictions and logical payloads; approved matching refusals | Runtime agent and coordinator | Preserve all producing source/build IDs and finish final-source acceptance after the sampling correction |
| Databricks metric views | One SQL smoke payload matches exactly; 19 cases per SQL/Spark path remain in the full comparison | All three original SQL model questions have exact payloads and either exact predictions or matching endpoint refusal | SQL model parity; Spark and full data coverage pending | Databricks agent | Finish Spark metric-view coverage; the missing time-column choice warning is restored and landed |
| Snowflake semantic views | All original cases captured; repaired wide-decimal and semantic-twin boundaries execute correctly | Five original semantic model pairs have exact logical payloads and predictions | Exact prediction parity | Snowflake agent | Timing complete but 31.51% slower; finish final evidence assessment |

## Active blockers and next steps

1. Current Phase 3 head is `aff7e32d2e353153197badeb863faa659f9e6d4e`, pushed and explicitly verified after two exact code approvals and normal unscoped handoff checks.
   `80c763a7be` landed the inference-option wording/register correction and explicit Pydantic dependency; `2d073d5b57` registered three already-reviewed Python bridge fingerprints.
   `aff7e32d2e` admits the PostgreSQL builtin ARRAY_AGG in guarded read queries, fixing the original Lakebase notebook metadata read without permitting writes.
   Its lint and whole write-guard test module passed in the normal 258-second handoff.
   The protected `rust-core` branch remains Phase 2 until every exit requirement passes.
   `451aa3bde4` landed the minimal Databricks key-overflow classification repair; both original wide-key public cases now return the same named invalid request as the frozen client.
   `b2b447c4fb` lands T3.9's approved NaN lag, SQL-expression and negative-neighbor support with explicit typed boundaries.
   Its complete handoff passed in 371 seconds, including 136 contract tests, all 29 structural vocabulary controls, 133 public materialize tests, Rust and Java checks.
   Genuine production and test-ops artifacts retain their actual `8a7323cb66` producer identity.
2. Databricks SQL's 302 original functional cases are all captured with zero operator failures and verified cleanup.
   The strict comparator's 5599 raw differences remain preserved, not silently waived.
   All 95 schema-order cases have exact named table contents and matching full link multisets, plus the same actual catalog query, typed parameters and unordered typed rows; raw catalog order differs without an ORDER BY.
   Six decimal/key cases have exact mathematical proofs of correct current conversion; finite register applicability still needs reconciliation.
   All six ranking request lists are exact after only relationship permutation and the existing volatile request ID; all 72 lag cells independently match their own source windows.
   These finite proofs retain the original failures and do not introduce automatic comparator exclusions.
   The original Databricks diamond real-model pair now has exact payloads and all three predictions on current `8a7323cb66`, with independently derived truth and 0% RMSE degradation.
   Earlier saved catalog-order payload differences remain preserved; the fresh pair does not relabel them.
   Four warning categories and the multicall context order remain separate.
   SQL/Spark fixture leases are released; no duplicate SQL campaign is needed.
3. Snowflake completed all 304 data captures, 15 ordinary/semantic model pairs and the original diamond accuracy comparison.
   Nine positive pairs have exact typed predictions; six fully materialized same-code endpoint refusals count as parity under existing approval, with response bodies explicitly unrecorded.
   The diamond's original model pair has RMSE 4.668331 versus 4.740305, 1.54% degradation and within budget; its strict differences remain preserved.
   The original 20-source differential has finite proofs for all nine successful payload pairs and all catalog-order differences; raw classifier failures remain attached.
   The existing provider kit passed all 33 normal cases and both fault controls after the twice-reviewed six-alias oracle correction; cleanup verified zero owned objects.
   The existing single-case sampling oracle landed at `91b7f348f8` after both fresh code approvals and a passing 200-second normal unscoped handoff.
   The original negative-neighbor Snowflake case also passed on genuine current `8a7323cb66`: success, exact logical payload and typed prediction result, with the old refusal preserved.
4. Lakebase's 282 data cases remain reconciled on their recorded source; the pinned 173106-row fixture is intact.
   All 21 corrected-client model/runtime profiles have completed and their full paired consumer passes all 255 questions.
   There are 114 positive prediction pairs with 7485 typed-exact rows, 141 approved matching endpoint refusals, exact logical payloads, zero actual differences and zero operator errors.
   Receipt SHA256 is `cfabf1853ccc01d722da21d0f9261d7a970d95cbd6f3566ad9a8ee2b380e28ce`.
   All 42 cloud executions are terminal; actual `76eb785b91` source/build identities and 147 stricter diagnostics remain preserved.
5. Full locked-dependency CI [40313141056](https://github.com/NVIDIA/structured-data-models-client/actions/runs/40313141056) was dispatched once at 11:11 UTC on exact integrated `aff7e32d2e`, then cancelled while still queued with zero jobs after the preceding run exposed additional failures.
   Avoid running the known-bad source again; dispatch the next full CI after the reviewed fixture correction lands.
   Previous run [40308222963](https://github.com/NVIDIA/structured-data-models-client/actions/runs/40308222963) on `80c763a7be` is not green: 10 jobs have failed while root tooling remains active.
   The three missing bridge registrations are already corrected at `2d073d5b57`.
   The remaining diagnosed failures are missing approved negative-neighbor SQL in statement/golden expectations, an incorrect phase-checkpoint value in the register, and a stale refusal-map source-line citation; the cross-host failure cascades from the missing Rust observation artifact.
   Contract tooling also ended with a runner shutdown before producing complete assertion details.
   The runtime agent owns the narrow correction and actual public-path reproduction, with canonical regeneration for generated files and both reviews before checks.
   No dependency or Pydantic runtime failure was identified in these logs.
   The older `91b7f348f8` run is terminal failure; its option-formatting and invalid-register failures were repaired at `80c763a7be`.
   Official `80c763a7be` x86/ARM production/test-ops artifacts, complete Java/JNI and the full source bootstrap are authenticated and published with original producer IDs.
   The Spark serverless process ran out of memory after 108 complete original cases; all 108 are retained and the automatic retry was stopped without duplicate case execution.
   Both reviewers approved fresh 24-case processes for only the remaining original positions 108–299, using actual `80c763a7be` artifacts and unchanged original cases.
   The first of eight serial jobs is `818601119704735`; the Databricks agent owns its exclusive fixture lease and immediate continuation.
   The original Lakebase notebook failed before its data/model journey because the read guard rejected ARRAY_AGG in its original metadata query.
   Its failed job is terminal and its read lease is explicitly released; no competing Lakebase reader remains.
   That narrow guard defect is now fixed at `aff7e32d2e`; rerun the original notebook through a fresh immutable output prefix with proper source identity.
   The original Databricks notebook is reviewed and staged, and waits for the Spark fixture lease.
   The read-only local Java comparison completed on genuine `91b7f348f8`: all 18 request pairs are exact after the existing request-ID exception; 15 endpoint-refusal code pairs match, and 75 ordered prediction rows with 150 binary64 probabilities are exact.
   Python DataFrame probability columns and Java probability maps are fully accounted for in finite evidence; the original strict report remains preserved for final independent review.
   The Java warehouse cohort is exactly 249 existing SQL cases, not the earlier incorrect shorthand of 157: 215 S1 plus 34 S2, with identical IDs and case blobs across the inspected sources.
   Each architecture also requires the existing six J2 shapes in three zones.
   The Databricks agent froze bounded execution with fresh Python/Java child processes, retaining all cases and avoiding the same monolithic-memory failure.
   One focused code approval is complete; the independent broad review is active, and no changed runner has executed before both approvals.
   The sole artifact collector retains original producer IDs for all proven unchanged reused assets.
   No completed model or SQL campaign is relabeled as final-source execution.
6. The runtime agent owns missing original local generated-source checks for DuckDB, SQLite and PostgreSQL, plus full-data census and memory readiness.
   The fresh DuckDB 200-source run completed with output byte-identical to its earlier execution: 166 strict agreements, 26 approved empty-table outcomes and eight existing finite decimal/date proofs with no residual.
   The fresh SQLite 200-source run also completed with all 200 public records and canonical verdicts identical; its original 34 finite failures and 17 SQL-numbering rows remain under exact proof review.
   PostgreSQL also completed all 200 original sources with zero operator errors.
   Its raw report has 151 agreements, 26 approved empty-table outcomes and 23 unclassified rows; finite existing-authority assessment covers the latter as 22 approved PostgreSQL wide-decimal improvements and one strict text-time case.
   Its two g99 SQL result differences have all six typed read hashes independently reproduced across 38400 seeds and 896 rows, with exact public payloads and results.
   Raw reports remain unchanged and final independent evidence review is required.
   The original pinned census caches are present and verified, with no download or disk-space blocker.
   The full census completed all six original questions on both DuckDB and SQLite, with no ordinary refusal-control hits; only the deliberate invalid control fires.
   Do not repeat either completed backend.
   The original full PostgreSQL census remains active on its owned local database, with no lock wait or duplicate run; the runtime agent owns collection.
   The separate SQLite indexing draft passed its bounded controls but is off the critical path because the original unindexed census finished; no indexed full-data retry was run or claimed.
   All 20 existing native and host memory/ownership controls passed.
   The coordinator owns final CI dispatch, finite Databricks reconciliation, integration, performance disposition and this tracker.
   The missing metric-view time-choice warning is fixed and landed at `fc643df9fe` after both exact code approvals, eight passing public controls, all 17 assembly tests and the complete normal unscoped handoff.
   Genuine native artifacts retain their `2cf4beb833` producer identity across the test-only repairs.
   The fix keeps warning selection in Rust after final table/time overrides; Python remains a formatting adapter.
   The inference-wording/register repair is landed at `80c763a7be`, with a passing 291-second normal handoff and both exact reviews.
   Its Python formatter uses the actual installed Pydantic library; the explicit client requirement is `pydantic>=2.7`, and all 149 resolved dependency identities and artifacts remain unchanged.
   Active lanes are the runtime agent for original PostgreSQL census and CI diagnosis, the Databricks agent for bounded cloud validation and original notebooks, and the Snowflake agent for independent review and source-impact evidence.
   The coordinator owns integration, finite comparison assessment and this tracker.
   Phase 4 stays paused.
7. Three owner policy decisions remain unanswered: the measured latency/native-size/boundary performance costs; extending the existing one-case Snowflake random-sampling comparison rule to the original multicall case; and reuse of proven-unaffected completed live results after later client changes.
   The second sampling case is not authorized by KO3-63; the protected comparison guard explicitly requires the owner for a second case.
   Source-domain proofs preserve original producing IDs and require both final independent reviewers; they do not themselves authorize reuse.
   Databricks multicall has the same sampled set in a different order and is assessed separately.
   No performance waiver, extra sampling allowance, source waiver or model-accuracy waiver is inferred.

## Measured performance blocker

The genuine corrected client completed five alternating paired live Databricks runs with caches disabled and the original slow control.
The original slow control was detected correctly.
All four measured cases fail the plan's unchanged 10% live slowdown limit:

| Existing case | Python median | Rust median | Ratio |
|---|---:|---:|---:|
| Ordinary no-order-90d | 6.42 s | 12.82 s | 2.00× |
| Metric-view no-order-90d | 5.29 s | 10.87 s | 2.06× |
| Snowflake ordinary | 1.418 s | 1.718 s | 1.211× |
| Snowflake semantic view | 0.931 s | 1.225 s | 1.315× |

Existing functional records show 12–13 extra fresh-source catalog, key and preview reads; sampling and statistics reads otherwise match.
The owner previously approved re-reading current source facts on every run, whereas the old client keeps table facts.
Two independent source assessments found no safe toggle that meets the limit while retaining the freshness, type, empty-table and error contracts.
Per-table metadata fusion could remove some calls but does not demonstrate the required 10% bound; a broader typed batching design would need separate review.
Keep this failed result visible; do not disable freshness, repeat unchanged runs to select better samples or call Phase 3 complete while the performance decision remains unresolved.
This latency result is separate from the owner's 5% prediction-accuracy limit.
Both Snowflake timing runs also detected their slow controls and completed cleanup; no performance failure is waived.
Offline performance and all remaining exit checks remain required.

## Finite comparison evidence and record correction

Snowflake's two lag cases have 17 strict payload differences covered by the existing approved own-row alignment correction, P2F-1.
Two independent assessments recomputed all 72 lag cells from the frozen source rows and each entity's exact time windows; every Rust cell matches its own source rows.
The Python value 10.0 in the one both-populated mismatch belongs to a different entity; Rust's 11.5 is correct for the requested entity and window.
The existing comparison rule already admits both scalar corrections and null-on-one-side differences; only two exact Snowflake case entries need carrying under the existing approval.
No product fix, new matcher or broad lag exception is authorized by this evidence.
The finite approved Snowflake punctuation applicability is carried in `130bd0f94c`; the diamond catalog-order evidence remains separate and does not waive actual model accuracy.

The coordinator's `76eb785b91` landing trailer mistakenly named `tests/test_t22_core_legs_off_duckdb.py`.
The actual reviewed and tested file was `contract/tests/test_t21d_routes_off_duckdb.py`, alongside `contract/runners/run_python.py`.
Both exact code reviews, both patch-preserving confirmations and the passing handoff cover those actual files.
This records the correction without rewriting the protected branch's history or changing reviewed code.
The next landing must derive every changed protected path from the actual Git diff and name those exact paths.

## Full CI and phase exit

- **Required: green full CI on the final integrated Phase 3 source before closing T3.10 or merging into `rust-core`.**
- [Run 40258337766](https://github.com/NVIDIA/structured-data-models-client/actions/runs/40258337766) is terminal with **81 passing jobs and one failed job**, `tooling (contract)`, on older source `2a492e889c8a3b0e601a5df252a1f4465596148a`.
- The previously identified contract-fixture repair is already landed at `f57b4ca19330fb122041d3fb22fcc14ac743aab6` and passed its normal package checks; this older red run does not prove the repaired final source is green.
- The failing test is `contract/tests/test_t20f_gate.py::test_compare_takes_database_keys_and_refuses_malformed_ones_by_name`, which the landed fixture repair addresses.
  Those corrections are now integrated; the newer full run below is the current diagnostic source.
- [Run 40271951963](https://github.com/NVIDIA/structured-data-models-client/actions/runs/40271951963) on older `0d2b435082` is terminal with 23 failed, 38 passed and 10 skipped jobs.
  Its root tooling suite passed all 978 tests; the failures identify the repaired stale fixtures, prior-checkpoint schema/trailer issue and missing-observation cascades.
  Preserve the original failures and producing source IDs.
- [Run 40277736422](https://github.com/NVIDIA/structured-data-models-client/actions/runs/40277736422) is terminal with failure on exact integrated source `76eb785b91aeaa319e6537ac8017a70b6f186ed4`.
  It was dispatched once with locked dependencies and the full workflow, without a topic subset.
  Nine jobs failed: four stale wheel-smoke inventories, three DuckDB contract cells, the Rust job and the root-tooling inventory check.
  The contract-tooling job passed; root tooling completed 977 passed and one failed test in 71 minutes, with its sole stale inventory assertion already corrected in `130bd0f94c`.
  The Rust job passed observations, comparison and coverage; only its landing ratchet failed on the already-corrected trailer pathname.
  The third DuckDB cell additionally exposes a stale test rejecting approved database-scoped Lakebase register entries; the one-file test correction passed both reviews and normal checks, and is landed and verified at `b549507b87`.
  All four smoke jobs passed their 12 production and 59 native tests before the stale inventory check failed.
  The required-ID and append-only landing-record corrections are now landed at `130bd0f94c`; this does not prove the older run green or replace final-source full CI.
- Fix every remaining failure or flake, obtain both required code approvals, rerun the affected checks and finish with green full CI; no failure is waived or accepted as complete.
- Genuine `76eb785b91` x86/ARM production client artifacts and Java artifacts are verified and available; fresh final-source natives and Java/JNI remain required after the landed product corrections.
- Offline DuckDB performance completed 178 existing scenarios with five repetitions per client: 175 fully completed pairs have a 0.732 median CPU ratio, but the deepest case fails its individual limit at 1.241; three incomplete cases remain explicit.
  Import, boundary overhead, control size and provider pandas checks pass; native baseline size and statement inputs are not yet measured in this report.
  A bounded same-case diagnostic shows the DuckDB outlier uses the old materializer under its scoped test routing: actual default-core median is 62.94 ms against 109.54 ms for that mixed route; all ten calls succeed, but the original failed report is not overwritten.
  SQLite completed 160 scenarios and 150 complete pairs: median CPU ratio 0.345 with no individual slowdown, but five ranking cases exceed the 2 ms boundary budget, worst 2.403 ms.
  PostgreSQL completed 176 scenarios and 167 complete pairs: median CPU ratio 0.524, no individual CPU or boundary failures, and import/control/provider checks pass.
  Its stock report still explicitly lacks native baseline and statement inputs, so this is not an overall performance PASS.
  The fresh official `91b7f348f8` production native is 15863640 bytes against the immutable recorded Phase 1 baseline 9562016, ratio 1.659 above the 1.25 limit; the current official `80c763a7be` x86 native is 15876472 bytes and also exceeds that limit.
  No synthetic baseline wheel or size waiver is used.
- Also finish existing model/runtime comparisons, original notebook/provider checks, direct paired performance checks, memory checks and the sealed final check, then final evidence review and merge/push.

## Finish sequence

1. Keep the landed shared repair fixed while finishing the remaining reviewed corrections.
2. Resolve the concrete blockers and finish every matrix row, recording the producing source and reference for each result.
3. Complete direct performance checks, memory checks and final CI on the accepted source.
4. Run the sealed check last, complete final acceptance, then merge and push Phase 3 into `rust-core`.
5. Keep Phase 4 paused until the owner resumes it.

The earlier Python reference inventory is present for all selected warehouse cases.
It does not by itself establish that every saved reference digest is current or every final Rust comparison passes.
Detailed commands, original logs, hashes and private runtime receipts stay in the existing orchestration state directory.
