# Phase 3 validation tracker

Updated 2026-10-27, 00:20 UTC.
Phase 3 / T3.10 is unfinished.
The owner assigned existing Phase4work to a separate new agent; this coordinator continues Phase3.
The coordinator alone edits this tracker; agents report results and next actions.

## Acceptance and owner decisions

- Compare the frozen Python client and Rust client using the existing cases, pinned data, queries, seeds, anchors, model and settings.
- Preserve schemas, values, nulls, row multiplicities, logical payloads and required errors under explicitly approved compatibility decisions.
- Matching or closely matching endpoint rejections with the same underlying cause count as compatibility parity where the old client already rejects the case.
- Where both return predictions, at most 5% worse prediction accuracy passes; parity or a 1–2% difference is preferred.
- Prediction disagreement, accuracy degradation and execution speed are separate measurements.
- Owner approved the previously named warehouse latency, native size and five small SQLite boundary-overhead costs, with optimization afterwards.
- Two additional affected SQLite timing rows exceed the 2 ms boundary target by 0.16 ms and 0.08 ms; total CPU is faster than the reference, but extending the cost acceptance to these two rows is awaiting the owner’s answer.
- Owner approved retaining proven-unaffected completed results after later changes only when two independent reviewers verify their applicability; rerun every affected case, preserve original build IDs and require green final-source full CI.
- Owner directed sampling to remain unchanged and Snowflake to be compared on identical original rows/contexts.
  Preserve the earlier independent-draw 26.51% loss as a historical measurement; it does not establish controlled-input migration accuracy.
- Owner explicitly approved keeping all valid sampled links instead of the old partial-link behavior; exact decision row is `T310-OWNER-2026-10-26-COMPLETE-SAMPLED-LINKS-1` on the current repair candidate.
- Preserve failed attempts, raw strict differences and original source/native/build identities; recording an observation is not comparison acceptance.
- Use existing coverage only; do not add scenarios or new validation machinery.

## Warehouse comparison matrix

These are comparisons of old versus new clients on each execution path, not comparisons between warehouses.
Metric and semantic views are feature coverage within their warehouses.
Completed results below retain their actual producing sources; bounded source applicability is accepted only where both independent reviews establish it.
The complete-link repair is landed at `cc51e6b6eb`; all selected affected warehouse and Java executions are now complete and are receiving independent finite result review.
A test-only follow-up at `ebe4cffde` corrects stale raw-parity assertions and one source citation, without changing client/build/dependency bytes; it needs both exact CODE approvals, package checks, landing and final-source full CI.
All retained artifacts keep their actual producing source IDs.

| Area | Data comparison | Model comparison and accuracy | Remaining |
|---|---|---|---|
| Databricks SQL | All 302 originals executed; final affected nine-case execution completed with zero operator failures and owned cleanup | Current original diamond quality refresh is 2.4323% worse, within 5%; original 255-question matrix retained where proven unaffected | Independent finite review of the nine actual observations and quality pair; raw strict differences retained |
| Databricks Spark | All 300 original executions terminal; final affected nine-case run completed SUCCESS with owned cleanup | Original runtime/model evidence and aggregate +0.0002943% remain retained, with the single-row diagnostic limit preserved | Independent finite acceptance of the final nine observations, including existing metric-view coverage |
| Snowflake | All 304 originals captured; final affected nine-case execution complete, zero operator failures, owned cleanup | Current original diamond quality refresh is 1.9583% worse, within 5%; controlled 10001-row original-input predictions remain exact | Independent finite review of nine observations and actual quality pair; exact joint relationship-order delta remains explicit |
| Lakebase | All 282 originals reconciled; final affected nine-case execution complete with zero operator failures and owned cleanup; original notebook 32/32 | All six original runtime profiles retained with exact payloads/predictions or approved matching refusals | Independent finite acceptance of the final nine observations |
| Databricks metric views | Existing SQL/Spark coverage executed; exact original SQL smoke retained | Original three SQL model questions have exact predictions or matching refusals | Include affected Spark/SQL actual results in the final finite evidence assembly; no new campaign |
| Snowflake semantic views | Original cases complete; original wide-decimal and semantic-twin boundaries repaired | Five original model pairs exact; source applicability retained under both input proofs | Final evidence assembly and acceptance, retaining original producers |

## Exact remaining T3.10 checklist

Each unfinished row is required before closing T3.10; completed rows remain explicitly marked DONE.
Counts describe existing cases only.

| ID | Pending item | Current state | Owner / next action |
|---|---|---|---|
| R1 | **DONE: Databricks foreign-key metadata fix landed** | Pushed and independently verified `dd5b3f37ef`; both exact carry approvals and normal 188-second package handoff passed | Coordinator consumed handoff; original native producer IDs retained |
| R2 | **DONE: execute all nine affected originals** | Genuine d90: 9/9 observed, zero operator failures, SDK0 cleanup; corrected key32 and odd-name cases succeed, metric warning restored, negative-neighbor has zero strict differences; 186 raw differences remain for finite attribution | Coordinator consumes actual results and exact owner-approved key32 support; final comparison acceptance remains R8 |
| R3 | **DONE: Spark comparison assembly and finite review** | All 300 original executions assembled; both independent opinions support observed product semantics and aggregate quality, retaining raw read-attribution limits and final single-row diagnostic regression | Original producer IDs and all 4780 raw differences retained; no strict comparator PASS claimed |
| R4 | Finish finite Java graph parity acceptance | Both corrective nine-case runs on x86/ARM are now TERMINATED/SUCCESS; all 18 Java plus paired Python observations collected, clusters terminated and owned schemas verified empty/deleted. Each architecture has eight successes and the same existing Avito refusal | Independent finite result reviews of actual bodies/errors and approved FK changes; raw comparison EXIT1 retained, no all-case success claim |
| R5 | **DONE: original Lakebase notebook** | Parent `1028333862593563` passed all 32 original commands on genuine 7e41; full 173106-row data and PK/FK post-check exact; read lease released | Coordinator consumed authentic success/export/fixture receipt; both source-impact approvals retain original IDs |
| R6 | **DONE: original Databricks notebook** | Parent `435766366545516` passed all 39 original commands on genuine 7e41; source/export verified; only the verified owned metric view removed, SDK0 cleanup | Coordinator consumed authentic success and cleanup receipt; original failed attempts retained |
| R7 | **DONE: Databricks generated-source and provider checks** | Both finite opinions accept all 20 classified generated journeys; after both CODE approvals, corrected directCountRows oracle rechecked all 99 complete saved live answers: normal 33/33, truncate detected, silent-capability detected | Receipt `712a0c21`; all 99 original judgments reproduced first, only direct count-header judgment changed per pass; original failures preserved, no new queries |
| R8 | **DONE: finite warehouse comparison evidence and affected Lakebase original** | Both final DK302/SF304 opinions accept bounded evidence; final15e Lakebase negative-neighbor original now actually returns PostgreSQL PROVIDER_ERROR/2201W under P2I-4, with one observed case and no operator failure | Full original AW173106 typed rows, ordered columns/types and PK/FK restored exactly; SDK0 between case and restoration, lease released; receipt `4ff68507` retained for final pack sign-off |
| R9 | Accept selected affected comparisons | All 36 warehouse and 18 Java plus paired Python observations complete. Four actual DK/SF quality calls complete within 5%; broad quality result accepted and focused result review active. Original unaffected cohorts retained under both applicability proofs | Two independent finite result reviews of actual selected observations; preserve raw strict failures and genuine cc51 producers |
| R10 | Finish green full CI on corrected final source | [Run40418671928](https://github.com/NVIDIA/structured-data-models-client/actions/runs/40418671928): 75 successful jobs, three failures and two tooling jobs running. Root causes are stale Python/Rust raw-parity expectations after approved FK correction, one shifted citation, and a downstream missing Rust artifact | Test-only R2 `ebe4cffde` frozen: broad CODE approved, independent focused CODE pending. After both approvals, required checks and landing, dispatch one full corrected-source CI; no product rebuild or warehouse rerun inferred |
| R11 | **DONE: Snowflake identical-input comparison** | Full 10101-row source and original 10000-row draw verified; both complete current requests match the originals; all 10001 actual model26 predictions exactly equal old, RMSE ratio 1.0 on both batches and aggregate; SDK0 cleanup | Coordinator consumed receipt `8f48d1c7`; sampling unchanged; historical independent-draw 26.51% loss and failed recorder-mode attempt preserved |
| R12 | Finish corrected-source acceptance and performance disposition | Current cc51 stock3a remains live collecting seeded checks; its L1 failed the same stale Rust parity assertions. All 27 existing L7 rows actually measured with five repeats; every total CPU ratio is below 1 | Finish test-only repair and required final-source acceptance; two additional SQLite boundary costs await owner disposition, raw timing report EXIT1 preserved |
| R13 | Final sealed check, evidence package and authorized integration | Sealed40-case check remains unstarted. H3 cc51 external binding has both CODE approvals, but must be truthfully reconciled with the upcoming final test-only source and any required producer mapping | Ordinary finite results, cost disposition, final-source CI and acceptance first; then fresh stock admissions/dry and sealed-last once, evidence pack, two final GO opinions, merge/push Phase3 |
| R14 | **DONE: existing Spark and Lakebase live provider kits** | Spark saved-answer correction has both CODE approvals, five controls and actual 99-answer readjudication PASS; Lakebase actually completed all 99 normal/fault cases on final15e, normal33 clean and both faults detected | Spark receipt `08a8d2c7`, Lakebase `d3f3825e`; original failures, all driver captures and source IDs preserved; cleanup verified and lease released |
| R15 | **DONE: approved shared graph correction landed** | Owner-approved valid-link correction landed/frozen as cc51 with genuine official artifacts; local N3 quality is 2.4701% worse, current original DK/SF quality refreshes are 2.4323%/1.9583% worse, all within 5% | No sampling change; finite quality reviews and subsequent test-only source attribution remain separate acceptance steps |

R9 supplies the two independent applicability opinions; R2 now supplies the actual nine affected reruns.
Their execution is complete; finite acceptance of the remaining raw differences stays in R8.
R4 uses 249 SQL cases plus six existing J2 shapes in three timezones on each architecture.
Explicit design scope requires 20 generated sources for Databricks SQL and Snowflake; those are complete.
It does not require new Spark or Lakebase generated-source campaigns.
The existing provider kit is required live on all four warehouse legs; R14 has closed the two missing receipts found in the exit audit.
There is no new requirement to rerun accepted performance measurements or optimize their accepted costs before closure.
Both final count reviewers independently close completed comparable scenarios using actual existing traces and approved P3A-1/KO2-58 rules.
The original raw count differences, incomplete timing pairs and PostgreSQL Rust missing count measurement remain explicit; no blanket numerical PASS is claimed.

The owner requested a two-hour completion focus, targeting approximately 22:15 UTC.
That target was missed after the Java tail exposed a real shared Rust graph defect.
The graph decision is approved; the critical path is its landing, affected-result acceptance and corrected-source validation.
The deadline does not lower correctness requirements.

## Current execution and integration

- Pushed and explicitly verified final Phase3head: `cc51e6b6eb8c799fbeea31a737d7456fddd67c2b`, tree `1fb4aec15789b2b8893465045b8e908a89621465`, actual checkpoint530d3c253.
- New full CI40418671928 is running; all eight official producer groups and immutable source/production/Java publications are authenticated.
- Final local runtime setup passed259.59seconds; stock locked3a and the targeted27-row timing refresh are active on separate controlled heavy groups.
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
- All six residual Java cloud jobs have finished; the cloud owner collected the complete original249-case assembly and all timezone model outputs, verified SDK0, and removed only the six owned schemas.
  Raw strict failures remain preserved; one graph discrepancy requires the separate R15 correction.
  All four live provider kits are complete; prior failures and duplicate Spark attempt remain preserved with owned cleanup verified.
- The original Snowflake rank diagnostic completed after both CODE approvals and four passing controls: actual old int8 index overflow precedes lookup; all seven materialization SQL texts match the historical capture.
  It is fresh corroboration, with exact historical body unknown; cleanup verified zero tables/views.
- Java missing-only recovery source and execution binding had both CODE approvals and 23 passing controls.
  Failed 8 GiB attempts honestly stopped at memory admission; current reviewed 32 GiB jobs pass the unchanged fixed 2 GiB heap and resource guards.
- Both final Databricks SQL finite opinions are complete; the one affected Lakebase original and full fixture restoration are complete after renewed reviews.
- Provider oracle controls and actual saved-answer re-adjudication passed after both approvals.
- Coordinator owns the final packaging continuation, integration and this tracker.
  Session71947 is terminal: its locked gate passed, and its subsequent packaging failure is preserved; no successful wrapper return is claimed.
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

Both independent opinions cover all249 Java SQL cases per architecture, preserving actual d90/final15e producing sources, raw SQL-count failures, the10001-case sampled-body difference and warning-observation limits.
All cases other than the x86 diamond graph discrepancy have bounded finite acceptance; completing execution does not waive that discrepancy.

## Exact Java model-reference gap and closure

All six current Java/Python model requests agree, but earlier old-client J21 runs used different product/customer selections and a different historical anchor.
The exact gap is now closed by six genuine frozen43e public-client predictions using the same saved requests, without rematerializing or changing sampling.
Both CODE reviews preceded calls; both actual-result reviews verify three INVALID_REQUEST rejections, two UNSUPPORTED_DTYPE rejections, and one successful25-row result with all50 probability bits exactly matching the current client.
Actual old post-adapter requests were captured and checked before forwarding to the model.
Receipt `81fb6351` preserves source, request, response and model26 identities; no old-client local-materializer or current-wire-capture claim is inferred.

Both final reviewers confirmed publishing the generated evidence on a separate evidence lane while retaining the eventual tested source unchanged.
The later documentation commit receives its own review/checks and does not inherit the product source's CI identity.

## Shared graph repair and bounded accuracy experiment

The old Python client and source15e each mark a logical foreign key complete after the first partial map, dropping valid links supplied by later sampled paths.
The public original diamond reproduces21 or22 links out of43 solely from table-discovery order.
Candidate `f83375b1` merges per sampled row, preserving the old last-valid duplicate rule within each partial map and adding no samples, rows or source reads.
Both independent CODE reviews approve; the genuine production native has SHA `531ff395`, all nine Rust payload tests pass, and both public discovery-order tests retain all43 links.
The actual existing model26 experiment compares original21-link and corrected43-link requests with identical other values.
RMSE changes from4.668331 to4.783642, or2.4701% worse, on just three predictions; this is within5% and does not establish an accuracy improvement.
Both finite reviewers verify the actual experiment, including unchanged model settings and independently computed labels.
The discarded-link step has no temporal filter; this exact experiment adds no future-dated rows, which does not prove universal leakage safety.
The owner approved keeping all valid links; the literal answer is preserved privately and recorded in the current candidate's Phase3decisions.

Actual315-case DuckDB impact receipt `4511ff3e` identifies exactly nine changed materialized/public payloads: five sample.parent-table variants, two diamond variants and two keyed/keyless two-path variants.
The other306 match under the existing canonical policy; all case/base hashes agree and207 raw section differences remain recorded.
Both reviewers identified the exact affected cohorts and proved bounded reuse for unaffected cohorts; all selected corrective warehouse/runtime executions and both original quality pairs are now complete.
Independent finite result acceptance remains separate from successful execution; a generic non-diamond or acyclic-graph exemption is insufficient.

## Separate Phase4ownership, ongoing Phase3execution

The owner clarified: "Keep Phase 3 here; new agent handles Phase 4".
The earlier Phase3transfer interpretation was incorrect and is superseded.
This coordinator remains responsible for Phase3, its tracker, landing, CI and warehouse validation; current-candidate normal handoff is running.
The cloud owner completed every selected remote/local run and verified cleanup; immutable actual results are in independent finite review.
The new Phase4agent prompt has been rebuilt from verified branches/worktrees, package status, exact interrupted work and owner decisions; this coordinator retains Phase3.
The earlier private Phase3handoff files are marked superseded; they do not transfer ownership or halt Phase3.
The selective-Java execution binding was twice approved before launch; actual completed results now need finite acceptance.
There is no pending owner decision for the valid-link repair.

## Completed evidence retained for final review

- Full model/runtime matrix: 21 profiles, 42 terminal cloud executions, 255 questions, 114 positive pairs with 7485 exact typed rows and 141 approved matching endpoint refusals; all logical payloads exact, zero actual differences or operator errors on actual `76eb785b91`.
  Receipt: `cfabf1853ccc01d722da21d0f9261d7a970d95cbd6f3566ad9a8ee2b380e28ce`.
- Full-data census: all 18 original questions pass across DuckDB, SQLite and PostgreSQL; AdventureWorks 173106 and H&M 16931173 rows retained, no unexpected refusal and all owned PostgreSQL fixtures cleaned.
  Actual source/native identity is `8a7323cb66`; receipt: `ac73037cf1b3c96f6bfeee4e56ec26dc173e754251abdab5c4469c4ba4b9c0f5`.
- All 20 native/host memory and ownership controls pass on their recorded source.
- All three local 200-source campaigns complete; finite decimal/date, SQL numbering and text-time evidence remains attached to the raw reports for final review.
- Snowflake's original 20 generated-source checks and 33 provider cases plus both existing fault controls are complete; Databricks's corresponding executions also completed, with both finite acceptance opinions complete.
- Original Databricks 10001-entity multicall accuracy comparison passes: aggregate RMSE 2.6043078493 old versus 2.6043036069 Rust; the one affected row improves by 14.91%.
  The first 10000 predictions and their eight context examples are identical; the final single-row call differs slightly, so the complete paired requests are not claimed identical.
- Original Spark 10001-entity saved-input comparison passes the aggregate limit: RMSE ratio 1.00000294257; first 10000 predictions exact, final single-row error 11.4946% worse, explicitly retained.
  Receipt: `049e0b48a5f55fc438d596309f31c2502dcac40317be53cc3c3f6b90eff243e8`.
- Original Snowflake 10001-entity controlled comparison passes with identical original inputs and requests: all predictions exact, aggregate and both batch RMSE ratios 1.0; sampling unchanged.
  Receipt: `8f48d1c726ea4904f2cfec81390fd7be010cfd63da645efdd7a460232d096361`.
- Historical pre-correction diamond quality measurements were DK0% and SF1.54% worse.
  The actual complete-FK refresh is DK2.4323% and SF1.9583% worse, both within5%; preserve the old producer results without substituting them for current affected evidence.

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
Twenty local performance pairs remain explicitly non-comparable: 15 ordinary incomplete/refusal pairs and five non-comparable approved support improvements; none is promoted to a numerical PASS.
No new warehouse CPU recorder or replay machinery is required on the critical path.
Both final auditors accept the shipping-default DuckDB deepest-case diagnostic CPU ratio 0.6354, preserving the earlier mixed-route raw failure.

## Evidence and history

Both completed-cohort applicability reports are preserved as `T3.10-completed-evidence-d90-source-impact-broad-r2.md` and `T3.10-final-source-applicability-dd5-focused-r1.md` in the existing private review directory.
Detailed original logs, hashes, source-impact proposals and failed attempts remain in the existing orchestration state directory.
The immediately preceding uncommitted tracker draft is preserved there as `phase-3/handoffs/validation-tracker-before-20261026-1341.md`.
Git retains the earlier CI history and decision trail.
This tracker changes status only; it does not grant comparator tolerances, native identity exceptions or new compatibility approval.
