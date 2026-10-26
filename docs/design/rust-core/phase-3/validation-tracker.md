# Phase 3 validation tracker

Updated 2026-10-26, 07:23 UTC.
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
| Databricks SQL | Corrected client has 257/301 original non-connector observations, zero operator failures; completed 24-case processes each cleaned and verified empty; original connector case remains pending | Thirteen original model cases reconcile; all 15 corrected-client Databricks runtime jobs succeeded, final paired collection pending | Completed model pairs have exact predictions or approved matching refusals; data reconciliation remains open | Databricks agent | Finish 301 cases, diagnose the genuine wide-decimal INTERNAL_ERROR, run the original connector and reconcile all 302 |
| Databricks Spark | Original 300 cases prepared with verified final-source production wheels and original serverless ARM runtime; not submitted while SQL owns the fixture | Prior 13 original model pairs reconciled; corrected runtime job succeeded, paired collection pending | Original parity retained; corrected full data verdict pending | Databricks agent | Submit the original 300-case run immediately after SQL/connector cleanup |
| Snowflake | All 304 original cases captured and cleaned; approved lag/name differences recorded; diamond has differing raw catalog orders | Fifteen ordinary/semantic pairs have exact payloads and nine exact positive predictions plus six approved matching refusals; original diamond also returns predictions | Diamond RMSE is 1.54% worse, within the owner's 5% limit; it is not exact payload/prediction parity | Snowflake agent | Finish existing 20-source differential/provider checks on the final native, with unchanged frozen-reference preparation underway |
| Lakebase | All 282 existing cases reconciled: 128 original differences match existing approvals, two type-capture differences repaired, one retained count difference has two exact-equivalence approvals | All ten model pairs pass: four exact prediction matches (285 rows), six exact refusal matches, all payloads exact except request IDs | Exact prediction/refusal parity for this local Python pair; remaining runtime cells pending | Lakebase agent and coordinator | Complete final source integration and remaining runtime cells; pinned 173106-row fixture is restored exactly |
| Databricks metric views | One SQL smoke payload matches exactly; 19 cases per SQL/Spark path remain in the full comparison | All three original SQL model questions have exact payloads and either exact predictions or matching endpoint refusal | SQL model parity; Spark and full data coverage pending | Databricks agent | Complete SQL and Spark metric-view coverage |
| Snowflake semantic views | All original cases captured; repaired wide-decimal and semantic-twin boundaries execute correctly | Five original semantic model pairs have exact logical payloads and predictions | Exact prediction parity | Snowflake agent | Timing complete but 31.51% slower; finish final evidence assessment |

## Active blockers and next steps

1. Current Phase 3 head is `b549507b87f8896a7be80da0ce834d17a259305e`, landed and verified after two exact code approvals and normal checks.
   It additionally repairs the stale Postgres-family scope test after both exact code reviews and the normal unscoped handoff, including its whole-file check.
   Prior `130bd0f94c` carries the smoke inventory, finite Snowflake applicability and integrated tracker; shipping code/build inputs remain identical to `76eb785b91aeaa319e6537ac8017a70b6f186ed4`.
   The protected `rust-core` branch remains Phase 2 until every exit requirement passes.
2. Databricks owns the exclusive SQL fixture lease and the original 301-case continuation.
   The unbounded process grew to 14.6 GiB RSS; 24-case processes preserve all 146 earlier complete observations and bound process lifetime, with cleanup and zero-table verification between them.
   `decimal.key-wide` still has a genuine materialization INTERNAL_ERROR against a reference INVALID_REQUEST; the cause is unproven and must be diagnosed, not waived.
   The original connector case follows cleanup, then Spark's original 300-case campaign.
3. Snowflake completed all 304 data captures and 15 original ordinary/semantic model pairs.
   Nine positive pairs have exact typed predictions; six fully materialized same-code endpoint refusals count as parity under existing approval, with response bodies explicitly unrecorded.
   Full same-input diamond controls prove identical payloads for each catalog order; the one actual original model pair has differing catalog orders and predictions, with RMSE 4.668331 versus 4.740305, 1.54% degradation and within budget.
   Its strict 46-difference report remains preserved; matching task accuracy does not imply exact request parity.
   Both five-pair timing runs completed with detected slow controls and cleared fixtures; existing differential/reference preparations are next.
4. Lakebase's 282 data cases and ten local model pairs remain reconciled on their recorded source.
   All six Lakebase reference runtime jobs succeeded and were collected; all six corrected-client Lakebase core jobs succeeded and final artifacts are being collected.
   The canonical 173106-row fixture remains protected by the existing runtime read lease.
5. All 21 corrected-client model/runtime profiles remain required, totaling 255 questions per client pass.
   All 21 corrected-client runtime jobs succeeded, including all six Lakebase profiles; final paired artifact collection is active.
   Twenty references are complete; the remaining J11 reference uses the authorized on-demand driver after two preserved spot-driver interruptions.
   The reviewed partial diagnostic accounts for 169 questions, 5005 typed-exact prediction rows and 91 approved matching refusals, with zero actual differences; eight incomplete profiles prevent a full matrix PASS.
   Original source/build IDs remain attached to every reused artifact; no completed job is relabeled as new source evidence.
6. The runtime agent also owns the genuine T3.9 closure miss: the approved NaN lag, SQL-text neighbor limit and negative-neighbor refusals remain in the current product and must be lifted.
   Closest public old/new reproductions are recorded; typed core/provider boundaries and explicit SQL-expression intent are the approved design.
   Implementation and canonical working-contract generation are complete with tests-first commits; the candidate is being frozen on the actual phase head for two exact independent code reviews before checks.
7. The coordinator owns CI, offline performance, integration and this tracker.
   Snowflake owns its remaining model/timing work; Databricks owns SQL/Spark, wide-decimal diagnosis and the sole artifact collector; the runtime agent owns cloud jobs and the T3.9 correction.
   These are the three active lanes; Phase 4 stays paused.

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
- [Run 40277736422](https://github.com/NVIDIA/structured-data-models-client/actions/runs/40277736422) is the new full CI run, actively executing on exact integrated source `76eb785b91aeaa319e6537ac8017a70b6f186ed4`.
  It was dispatched once with locked dependencies and the full workflow, without a topic subset.
  It currently has eight failed jobs: four stale wheel-smoke inventories, three DuckDB contract cells and one Rust test job.
  The Rust job passed observations, comparison and coverage; only its landing ratchet failed on the already-corrected trailer pathname.
  The third DuckDB cell additionally exposes a stale test rejecting approved database-scoped Lakebase register entries; the one-file test correction passed both reviews and normal checks, and is landed and verified at `b549507b87`.
  All four smoke jobs passed their 12 production and 59 native tests before the stale inventory check failed.
  The required-ID and append-only landing-record corrections are now landed at `130bd0f94c`; this does not prove the older run green or replace final-source full CI.
- Fix every remaining failure or flake, obtain both required code approvals, rerun the affected checks and finish with green full CI; no failure is waived or accepted as complete.
- Genuine `76eb785b91` x86/ARM production client artifacts and Java artifacts are verified and available; preserve their actual source IDs when later product corrections land.
- Offline DuckDB performance completed 178 existing scenarios with five repetitions per client: 175 fully completed pairs have a 0.732 median CPU ratio, but the deepest case fails its individual limit at 1.241; three incomplete cases remain explicit.
  Import, boundary overhead, control size and provider pandas checks pass; native baseline size and statement inputs are not yet measured in this report.
  A bounded same-case diagnostic shows the DuckDB outlier uses the old materializer under its scoped test routing: actual default-core median is 62.94 ms against 109.54 ms for that mixed route; all ten calls succeed, but the original failed report is not overwritten.
  SQLite completed 160 scenarios and 150 complete pairs: median CPU ratio 0.345 with no individual slowdown, but five ranking cases exceed the 2 ms boundary budget, worst 2.403 ms.
  PostgreSQL reference is complete and core measurement is active.
  The measured production native is 15831864 bytes against the immutable recorded Phase 1 baseline 9562016, ratio 1.656 above the 1.25 limit; no synthetic baseline wheel or size waiver is used.
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
