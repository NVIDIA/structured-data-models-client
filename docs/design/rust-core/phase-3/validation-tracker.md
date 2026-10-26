# Phase 3 validation tracker

Updated 2026-10-26, 16:35 UTC.
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
| Databricks Spark | All 300 original executions terminal; final batch cleanup verified | Original runtime model comparison complete; existing multicall stress quality carry awaiting both code reviews | Final finite review and the existing four-request model comparison |
| Snowflake | All 304 originals captured; exact finite reconciliation packet assembled; raw strict failures retained | Ordinary/semantic pairs and diamond pass; historical independent-draw stress pair is 26.51% worse | Controlled identical-input comparison, with sampling unchanged per owner; final finite acceptance |
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
| R3 | Finish Spark comparison assembly | **300/300 original executions terminal**, final 24 cleanup verified; authentic raw results retained | All 300 assembled and strict-compared; raw 4780 differences retained; finite public/input proofs prepared; original four-request quality carry under independent code review |
| R4 | Complete Java Databricks SQL and timezone comparisons on x86 and ARM | Both genuine d90 attempts failed when AWS terminated their Spot drivers; no completed case chunks recovered; only two verified owned empty schemas removed and leases released | Both code approvals consumed; x86 `522943989994657` and ARM `630212305896219` are running after cluster startup/runtime installation, with zero completed case claims so far |
| R5 | **DONE: original Lakebase notebook** | Parent `1028333862593563` passed all 32 original commands on genuine 7e41; full 173106-row data and PK/FK post-check exact; read lease released | Coordinator consumed authentic success/export/fixture receipt; both source-impact approvals retain original IDs |
| R6 | **DONE: original Databricks notebook** | Parent `435766366545516` passed all 39 original commands on genuine 7e41; source/export verified; only the verified owned metric view removed, SDK0 cleanup | Coordinator consumed authentic success and cleanup receipt; original failed attempts retained |
| R7 | Run the existing Databricks generated-source and provider checks | All 20 old/core generated journeys and all 33 provider cases plus both fault controls executed, SDK0 cleanup verified; strict failures preserved | General review proves generated-source private rows via 49 original tables and 22450 answer rows; provider sole normal mismatch is a contract-valid count header, with both fault sensitivities evidenced; final independent acceptance remains |
| R8 | Finalize finite Databricks/Snowflake comparison acceptance and scoped compatibility metadata | Earlier 23-case metadata landed at d90; final narrow decimal/lag/clock/UTC/FK records landed and verified at `42dbb880f9` after both exact renewed approvals, register validation and normal 173-second handoff; original schema failure retained | Final finite evidence acceptance remains; existing Java P2 proposals P3S-5/6/8 are still unapproved and under source classification; no generic tolerances are admitted |
| R9 | **DONE: two reviews approve bounded retention of completed cohorts** | Both independently approve retaining all 255 model questions, 18 census questions, 20 memory controls and 600 local generated-source results with original IDs; model warning/raw-FK equality is not claimed | All nine affected originals are executed in R2; final finite comparison acceptance remains in R8 |
| R10 | Obtain green full CI on the final integrated branch | Full integrated [run 40355886004](https://github.com/NVIDIA/structured-data-models-client/actions/runs/40355886004) dispatched at 16:06 UTC on exact `42dbb880f9`, locked dependencies; both earlier failures are repaired in this source; previous run had 79 successful jobs and two failures | Latest: 66 successful jobs, zero failures; Rust lint passed and Rust tests remain active; coordinator consumes terminal results and fixes any genuine failure under normal review rules |
| R11 | Snowflake identical-input comparison | Owner explicitly keeps sampling unchanged; earlier independent-draw pair remains recorded; a narrow original-sample control is being authored against genuine current core | Author freezes one-case strict provider interception, obtains both code approvals, then proves full generated requests/contexts and measures current predictions against the same original inputs |
| R12 | Run locked acceptance `3a` and assemble pre-heldout evidence | Fresh genuine42 runtime cells prepared from authenticated existing wheels; preflight passed; phase_end through step2 is active, with both Python observation sets saved and Rust host compiling | Coordinator owns live session6081; Cargo2/one heavy slot; supported --allow-proposed keeps P3S-5/6/8 unapproved; sealed content/run untouched |
| R13 | Close final acceptance, run the sealed check last, merge and push Phase 3 into `rust-core` | Not started; protected `rust-core` remains Phase 2; sealed started marker absent | Coordinator obtains both final evidence approvals, closes performance dispositions, runs the sealed check last, then merges and pushes |

R9 supplies the two independent applicability opinions; R2 now supplies the actual nine affected reruns.
Their execution is complete; finite acceptance of the remaining raw differences stays in R8.
R4 uses 249 SQL cases plus six existing J2 shapes in three timezones on each architecture.
There is no new requirement to rerun accepted performance measurements or optimize their accepted costs before closure.
The original performance reports still contain incomplete pairs and missing stock-report inputs; final acceptance must account for them explicitly rather than calling those reports blanket PASS.

## Current execution and integration

- Pushed and explicitly verified Phase 3 head: `42dbb880f9b95aec806f99e0735075c79c3e595a`.
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
  Renewed full CI `40355886004` runs on integrated `42dbb880f9` with the generated snapshot and final compatibility records, without topic-level CI or reduced scope.
  The sole cloud collector retains all artifact identities and starts dependent runs as producers finish, without waiting for full CI to complete.
- Superseded `7e41bb73a5` CI `40322768900` was cancelled after preserving its 78 successful jobs, remaining job states and all eight authenticated official artifact groups.
  It lacked the timeout repair and FK fix; cancellation released workflow concurrency so the corrected run started without the previous 36-minute queue.
- Cloud author owns Spark finite proofs, reviewed quality execution and the two live Java jobs; the SF author owns the identical-input control; two independent reviewers own code/evidence confirmations; coordinator owns integration, live acceptance session6081 and this tracker.
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
- Snowflake's original 20 generated-source checks and 33 provider cases plus both existing fault controls are complete; this does not substitute for the pending Databricks checks.
- Original Databricks 10001-entity multicall accuracy comparison passes: aggregate RMSE 2.6043078493 old versus 2.6043036069 Rust; the one affected row improves by 14.91%.
  The first 10000 predictions and their eight context examples are identical; the final single-row call differs slightly, so the complete paired requests are not claimed identical.
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
