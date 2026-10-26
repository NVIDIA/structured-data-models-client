# Phase 3 validation tracker

Updated 2026-10-26, 18:23 UTC.
Phase 3 / T3.10 is unfinished.
Phase 4 is paused by the owner.
The coordinator alone edits this tracker; agents report results and next actions.

## Acceptance and owner decisions

- Compare the frozen Python client and Rust client using the existing cases, pinned data, queries, seeds, anchors, model and settings.
- Preserve schemas, values, nulls, row multiplicities, logical payloads and required errors under explicitly approved compatibility decisions.
- Matching or closely matching endpoint rejections with the same underlying cause count as compatibility parity where the old client already rejects the case.
- Where both return predictions, at most 5% worse prediction accuracy passes; parity or a 1–2% difference is preferred.
- Prediction disagreement, accuracy degradation and execution speed are separate measurements.
- Owner approved the measured warehouse latency, native size and small SQLite boundary-overhead costs, with optimization afterwards.
- Owner approved retaining proven-unaffected completed results after later changes only when two independent reviewers verify their applicability; rerun every affected case, preserve original build IDs and require green final-source full CI.
- Owner directed sampling to remain unchanged and Snowflake to be compared on identical original rows/contexts.
  Preserve the earlier independent-draw 26.51% loss as a historical measurement; it does not establish controlled-input migration accuracy.
- Preserve failed attempts, raw strict differences and original source/native/build identities; recording an observation is not comparison acceptance.
- Use existing coverage only; do not add scenarios or new validation machinery.

## Warehouse comparison matrix

These are comparisons of old versus new clients on each execution path, not comparisons between warehouses.
Metric and semantic views are feature coverage within their warehouses.
Completed results below retain their actual producing sources; bounded source applicability is accepted only where both independent reviews establish it.

| Area | Data comparison | Model comparison and accuracy | Remaining |
|---|---|---|---|
| Databricks SQL | All 302 originals executed; nine affected originals rerun on genuine d90 with zero operator failures; raw differences retained | Original model profiles have exact predictions/payloads or approved matching refusals; diamond has 0% degradation | Final finite acceptance; all generated/provider checks executed and source-based differences reviewed |
| Databricks Spark | All 300 original executions terminal; final batch cleanup verified | Original runtime model comparison complete; stress-case aggregate RMSE +0.0002943%, within 5%; final single-row batch +11.4946% diagnostic retained | Final finite data/error review; original four-request model comparison completed |
| Snowflake | All 304 originals captured; exact finite reconciliation packet assembled; raw strict failures retained | Ordinary/semantic pairs and diamond pass; historical independent-draw stress pair is 26.51% worse | Identical-input control passes with all 10001 predictions exact; final finite data/error/warning review remains |
| Lakebase | All 282 originals reconciled on their recorded source; original notebook passed all 32 commands | All six runtime profiles included in the completed paired matrix; exact predictions/payloads or approved matching refusals | Final evidence acceptance; notebook/source applicability complete |
| Databricks metric views | SQL smoke exact; original cases remain included in SQL/Spark coverage | All three original SQL model questions have exact payloads and exact predictions or matching refusals | Finish comparison assembly and source-impact assessment of the completed Spark batches |
| Snowflake semantic views | Original cases captured; wide-decimal and semantic-twin boundaries repaired | Five original semantic model pairs have exact payloads/predictions | Final finite acceptance; measured latency cost already accepted |

## Exact remaining T3.10 checklist

Each unfinished row is required before closing T3.10; completed rows remain explicitly marked DONE.
Counts describe existing cases only.

| ID | Pending item | Current state | Owner / next action |
|---|---|---|---|
| R1 | **DONE: Databricks foreign-key metadata fix landed** | Pushed and independently verified `dd5b3f37ef`; both exact carry approvals and normal 188-second package handoff passed | Coordinator consumed handoff; original native producer IDs retained |
| R2 | **DONE: execute all nine affected originals** | Genuine d90: 9/9 observed, zero operator failures, SDK0 cleanup; corrected key32 and odd-name cases succeed, metric warning restored, negative-neighbor has zero strict differences; 186 raw differences remain for finite attribution | Coordinator consumes actual results and exact owner-approved key32 support; final comparison acceptance remains R8 |
| R3 | Finish Spark comparison acceptance | **300/300 executions assembled**, original whole-case model accuracy passes; general finite review passes observed product semantics while retaining raw read-attribution limits | Obtain independent focused final evidence opinion; do not rerun the 300 cases or claim raw strict PASS |
| R4 | Complete Java Databricks SQL and timezone comparisons on x86 and ARM | Both on-demand jobs FAILED with Java heap exhaustion in ContractRunner observation JSON serialization; each retained 96/249 Python and 72/249 Java cases, all 72 paired outcomes/codes match; 0/18 timezone checks completed | Cloud owner prepares reviewed missing-only continuation with measured JVM resource admission; both clusters terminated, exact owned schemas verified empty and removed; retain failed logs and original producers |
| R5 | **DONE: original Lakebase notebook** | Parent `1028333862593563` passed all 32 original commands on genuine 7e41; full 173106-row data and PK/FK post-check exact; read lease released | Coordinator consumed authentic success/export/fixture receipt; both source-impact approvals retain original IDs |
| R6 | **DONE: original Databricks notebook** | Parent `435766366545516` passed all 39 original commands on genuine 7e41; source/export verified; only the verified owned metric view removed, SDK0 cleanup | Coordinator consumed authentic success and cleanup receipt; original failed attempts retained |
| R7 | Final independent acceptance of Databricks generated-source and provider checks | All 20 paired generated journeys and all 33 provider cases plus both fault controls executed; general finite review accepts proven row equivalence and contract-valid count header; raw strict failures retained | Obtain independent focused final evidence opinion; no further warehouse runs presently required |
| R8 | Finalize finite warehouse comparison acceptance and scoped metadata | Java diagnostic-budget fix landed at `17db946bc3` after both CODE approvals and passing package checks; three pending proposals resolved, no proposed entries remain; final source `15ec2ad916` | Snowflake one-original-case diagnostic frozen for two CODE reviews before controls/live; close final error/warning evidence and obtain both complete evidence opinions |
| R9 | **DONE: two reviews approve bounded retention of completed cohorts** | Both independently approve retaining all 255 model questions, 18 census questions, 20 memory controls and 600 local generated-source results with original IDs; model warning/raw-FK equality is not claimed | All nine affected originals are executed in R2; final finite comparison acceptance remains in R8 |
| R10 | Obtain green full CI on final integrated source | **Previous full run 40355886004 GREEN: 81 successful jobs**; fresh [final-source run 40375303944](https://github.com/NVIDIA/structured-data-models-client/actions/runs/40375303944) started at 18:10 UTC on `15ec2ad916`; 28 passed, zero failures at 18:20 snapshot | Collect authentic final-source wheels as producers finish; collect complete final CI result, fix actual failures if any |
| R11 | **DONE: Snowflake identical-input comparison** | Full 10101-row source and original 10000-row draw verified; both complete current requests match the originals; all 10001 actual model26 predictions exactly equal old, RMSE ratio 1.0 on both batches and aggregate; SDK0 cleanup | Coordinator consumed receipt `8f48d1c7`; sampling unchanged; historical independent-draw 26.51% loss and failed recorder-mode attempt preserved |
| R12 | Run locked acceptance `3a` and assemble pre-heldout evidence | Previous-source run retained all 315 observations per host and successful ratchet/compare/coverage; SQLite L1 child interpreter issue corrected and exact failed test passes; superseded run stopped before sealed checks | Coordinator prepares fresh final-source acceptance with authentic 15e wheels, explicit SDM_PYTHON and warm Cargo dependency cache; no copied stamps or relabeled observations |
| R13 | Close final acceptance, run the sealed check last, merge and push Phase 3 into `rust-core` | Not started; protected `rust-core` remains Phase 2; sealed started marker absent | Coordinator obtains both final evidence approvals, closes performance dispositions, runs the sealed check last, then merges and pushes |

R9 supplies the two independent applicability opinions; R2 now supplies the actual nine affected reruns.
Their execution is complete; finite acceptance of the remaining raw differences stays in R8.
R4 uses 249 SQL cases plus six existing J2 shapes in three timezones on each architecture.
There is no new requirement to rerun accepted performance measurements or optimize their accepted costs before closure.
The original performance reports still contain incomplete pairs and missing stock-report inputs; final acceptance must account for them explicitly rather than calling those reports blanket PASS.

## Current execution and integration

- Pushed and explicitly verified Phase 3 head: `15ec2ad916def7fd55a8390128ed91ca25e7a90f`.
- The generated FK matrix repair previously landed at `1637948d6a`: 11 alias-only rows updated by its documented writer, 2961 other rows unchanged; both CODE approvals, whole matrix 2/2 and normal 157-second handoff passed.
  Product code, build inputs and dependencies are unchanged from d90.
- Exact 23-case compatibility metadata and both owner approvals previously landed at `d90ac9e3e7` after both independent CODE approvals, passing register validation and a normal 192-second handoff.
  Both landings have terminal explicit verification.
- The Databricks FK alias-only fix previously landed at `dd5b3f37ef`, with both exact carry approvals, a normal 188-second handoff and terminal explicit landing verification.
  It retains the genuine `aa27490b64` native producer identity; no rebuilt or relabeled native is claimed.
- The case-map CI timeout fix previously landed at `c7e38c6fa2` after both code approvals and a passing normal 477-second handoff, including the whole runner test module in 322 seconds.
  It loads the existing 394-case map once instead of repeatedly revalidating the complete corpus inside the comparison loop; assertions and coverage are unchanged.
- Prior CI `40315122377` on `ad28ef4287` finished with 80 successful jobs and one contract-tooling timeout; its 201 other contract tests passed.
- Earlier integrated [CI 40336101930](https://github.com/NVIDIA/structured-data-models-client/actions/runs/40336101930) runs the whole workflow on integrated `d90ac9e3e7`, dispatched once with locked dependencies at 14:03:35 UTC.
  The run is terminal: 79 jobs succeeded and two failed; Rust stopped on the stale generated matrix and the cross-host job lacked its artifact.
  All eight official d90 producer groups, Java/JNI and source publication are authenticated and retained.
  Full CI `40355886004` finished green on `42dbb880f9`; final-source run `40375303944` covers the subsequent Java fix and validator optimization.
  The sole cloud collector retains all artifact identities and starts dependent runs as producers finish, without waiting for full CI to complete.
- Superseded `7e41bb73a5` CI `40322768900` was cancelled after preserving its 78 successful jobs, remaining job states and all eight authenticated official artifact groups.
  It lacked the timeout repair and FK fix; cancellation released workflow concurrency so the corrected run started without the previous 36-minute queue.
- Cloud author owns final-source artifact collection and the missing-only Java recovery proposal; both prior Java jobs are terminal and cleaned.
- Two independent reviewers are reviewing the frozen one-case Snowflake original-error diagnostic; its author awaits both approvals before controls and the single baseline.
- Coordinator owns final runtime preparation, locked acceptance, integration and this tracker; superseded session6081 is terminal and its results remain preserved.
- Java diagnostic-budget repair landed at `17db946bc3`; complete package checks passed after both CODE approvals.
- Reference-validator optimization landed at `02c733594f`; one local read replaces three reads of the same 60 MB input, with unchanged outputs and approximately 44% lower measured local elapsed time.
- Final empty freeze `15ec2ad916` has the same tree as `02c733594f`; both mechanical CODE approvals and the normal handoff passed.
- Both independent notebook-only source-impact reviews approve genuine `7e41bb73a5` reuse for the original Lakebase and Databricks notebooks, with original identities and no raw-FK record equality claim.
  The no-retry carry received both approvals and all three controls passed; both original notebooks have now actually succeeded and their cleanup/fixture checks are verified.
- Canonical complete-Git publication of `d90ac9e3e7`, production wheels and Java/JNI assets are complete and verified; the cloud author remains the sole publisher and official-artifact collector.
- Failed Java attempts and their partial Python observations are preserved; both owned empty schemas were verified and removed without force.
- Maximum three active package lanes, two heavy local slots, Cargo jobs capped at two.
  Both independent Sol High code approvals precede package tests and builds; changed code requires renewed approvals.

The final compatibility records landed at `42dbb880f9` on passing exact `ebd02c1a5` evidence; no product/build/dependency changes or sampler approval are implied.
The owner reported reminders arriving repeatedly.
The coordinator disabled the reminder script and removed its exact cron entry at 15:59 UTC; the independent nightly CI entry is unchanged.
Do not re-enable reminders automatically.

## Completed evidence retained for final review

- Full model/runtime matrix: 21 profiles, 42 terminal cloud executions, 255 questions, 114 positive pairs with 7485 exact typed rows and 141 approved matching endpoint refusals; all logical payloads exact, zero actual differences or operator errors on actual `76eb785b91`.
  Receipt: `cfabf1853ccc01d722da21d0f9261d7a970d95cbd6f3566ad9a8ee2b380e28ce`.
- Full-data census: all 18 original questions pass across DuckDB, SQLite and PostgreSQL; AdventureWorks 173106 and H&M 16931173 rows retained, no unexpected refusal and all owned PostgreSQL fixtures cleaned.
  Actual source/native identity is `8a7323cb66`; receipt: `ac73037cf1b3c96f6bfeee4e56ec26dc173e754251abdab5c4469c4ba4b9c0f5`.
- All 20 native/host memory and ownership controls pass on their recorded source.
- All three local 200-source campaigns complete; finite decimal/date, SQL numbering and text-time evidence remains attached to the raw reports for final review.
- Snowflake's original 20 generated-source checks and 33 provider cases plus both existing fault controls are complete; Databricks's corresponding executions also completed, with final independent acceptance pending.
- Original Databricks 10001-entity multicall accuracy comparison passes: aggregate RMSE 2.6043078493 old versus 2.6043036069 Rust; the one affected row improves by 14.91%.
  The first 10000 predictions and their eight context examples are identical; the final single-row call differs slightly, so the complete paired requests are not claimed identical.
- Original Spark 10001-entity saved-input comparison passes the aggregate limit: RMSE ratio 1.00000294257; first 10000 predictions exact, final single-row error 11.4946% worse, explicitly retained.
  Receipt: `049e0b48a5f55fc438d596309f31c2502dcac40317be53cc3c3f6b90eff243e8`.
- Original Snowflake 10001-entity controlled comparison passes with identical original inputs and requests: all predictions exact, aggregate and both batch RMSE ratios 1.0; sampling unchanged.
  Receipt: `8f48d1c726ea4904f2cfec81390fd7be010cfd63da645efdd7a460232d096361`.
- Original Databricks diamond accuracy comparison passes with 0% degradation; Snowflake diamond is 1.54% worse and passes the owner's 5% limit.

## Accepted performance costs

The owner explicitly accepted these measured costs with optimization afterwards, retaining correctness and accuracy requirements.
Keep the raw failed target measurements visible; do not describe acceptance as meeting the original numerical target.

| Measurement | Recorded result | Disposition |
|---|---|---|
| Databricks ordinary / metric-view live latency | 2.00× / 2.06× old-client median | Accepted cost; optimize afterwards |
| Snowflake ordinary / semantic-view live latency | 1.211× / 1.315× old-client median | Accepted cost; optimize afterwards |
| Native size | Approximately 15.9 MB versus 9.6 MB Phase 1 baseline | Accepted cost; optimize afterwards |
| Five SQLite boundary cases | At most 0.403 ms above the 2 ms target | Accepted cost; optimize afterwards |

Local performance campaigns completed 178 DuckDB, 160 SQLite and 176 PostgreSQL scenarios, with incomplete pairs explicitly retained.
DuckDB's raw deepest-case 1.241 ratio used mixed legacy routing; the actual default-core diagnostic is faster than the old client, without overwriting the original result.
Only the three affected original negative-neighbor scenarios were rerun with five repetitions on genuine corrected native artifacts.
DuckDB CPU ratio 0.897 and SQLite 0.384 meet their scenario limits; PostgreSQL remains a refusal/incomplete pair on both sides and is not a timing PASS.
Twenty local performance pairs remain unjudged: 15 ordinary incomplete/refusal pairs and five non-comparable approved support improvements; none is promoted to a numerical PASS.
No new warehouse CPU recorder or replay machinery is required on the critical path.

## Evidence and history

Both completed-cohort applicability reports are preserved as `T3.10-completed-evidence-d90-source-impact-broad-r2.md` and `T3.10-final-source-applicability-dd5-focused-r1.md` in the existing private review directory.
Detailed original logs, hashes, source-impact proposals and failed attempts remain in the existing orchestration state directory.
The immediately preceding uncommitted tracker draft is preserved there as `phase-3/handoffs/validation-tracker-before-20261026-1341.md`.
Git retains the earlier CI history and decision trail.
This tracker changes status only; it does not grant comparator tolerances, native identity exceptions or new compatibility approval.
