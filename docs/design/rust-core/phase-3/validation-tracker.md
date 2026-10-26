# Phase 3 validation tracker

Updated 2026-10-26, 13:54 UTC.
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
- One owner choice remains pending: extending the existing Snowflake random-sampling allowance to the second original multicall case.
- Preserve failed attempts, raw strict differences and original source/native/build identities; recording an observation is not comparison acceptance.
- Use existing coverage only; do not add scenarios or new validation machinery.

## Warehouse comparison matrix

These are comparisons of old versus new clients on each execution path, not comparisons between warehouses.
Metric and semantic views are feature coverage within their warehouses.
Completed results below retain their actual producing sources until the final applicability review.

| Area | Data comparison | Model comparison and accuracy | Remaining |
|---|---|---|---|
| Databricks SQL | All 302 originals executed; zero operator failures; finite reconciliation packet assembled | Original model profiles have exact predictions/payloads or approved matching refusals; diamond has 0% degradation | Final finite acceptance; FK-affected reruns; generated-source/provider checks |
| Databricks Spark | 228/300 originals completed with zero operator failures in retained batches | Original runtime model comparison complete on its recorded source | Remaining 72 original cases, including remaining metric-view coverage |
| Snowflake | All 304 originals captured; exact finite reconciliation packet assembled; raw strict failures retained | 15 ordinary/semantic pairs: nine exact positive pairs and six matching refusals; diamond RMSE 1.54% worse, within 5% | Final finite acceptance and the second sampling decision |
| Lakebase | All 282 originals reconciled on their recorded source | All six runtime profiles included in the completed paired matrix; exact predictions/payloads or approved matching refusals | Final source applicability and original notebook |
| Databricks metric views | SQL smoke exact; original cases remain included in SQL/Spark coverage | All three original SQL model questions have exact payloads and exact predictions or matching refusals | Finish remaining Spark coverage and final applicability |
| Snowflake semantic views | Original cases captured; wide-decimal and semantic-twin boundaries repaired | Five original semantic model pairs have exact payloads/predictions | Final finite acceptance; measured latency cost already accepted |

## Exact remaining T3.10 checklist

Each unfinished row is required before closing T3.10; completed rows remain explicitly marked DONE.
Counts describe existing cases only.

| ID | Pending item | Current state | Owner / next action |
|---|---|---|---|
| R1 | **DONE: Databricks foreign-key metadata fix landed** | Pushed and independently verified `dd5b3f37ef`; both exact carry approvals and normal 188-second package handoff passed | Coordinator consumed handoff; original native producer IDs retained |
| R2 | Rerun every case affected by the FK fix | Full 302-case setup inventory identifies six stored-FK cases, two with missing public edges; proposed regression scope is those six plus one empty-result control; both impact reviews still required | Coordinator and two reviewers identify affected original cases; author reruns them, including the original dot-entity case |
| R3 | Finish Spark originals 228–299 | 228/300 complete; job `1066106807422509` is running positions 228–251; then 252–275 and 276–299 | Cloud author collects each terminal result and verified cleanup, then runs the next original batch |
| R4 | Complete Java Databricks SQL and timezone comparisons on x86 and ARM | Both first attempts failed before any Java case: JDBC configuration omitted required options; zero Java cases and zero J2 cases completed | Both corrected configurations now pass actual JDBC/privacy consumer validations; cloud author freezes both carry reviews, then runs 249 SQL cases plus 18 J2 comparisons per architecture on the corrected FK client |
| R5 | Complete the original Lakebase notebook | Prior attempts stopped on validation guard omissions; ARRAY_AGG and MD5 fixes landed; no notebook completion claim | Genuine `7e41bb73a5` production assets are published and readback-verified; cloud author obtains both exact carry/source-impact reviews and fresh fixture admission, then runs the original notebook |
| R6 | Complete the original Databricks notebook | Original cells verified; MD5/COUNT_IF guard fixes landed; no completed run | Cloud author obtains both exact carry reviews and runs at a clean shared Spark boundary |
| R7 | Run the existing Databricks generated-source and provider checks | 20 seeded generated-source checks plus 33 provider cases and two existing fault controls remain unexecuted in this cloud lane | Cloud author executes original checks with honest source/native binding and verifies cleanup |
| R8 | Finalize finite Databricks/Snowflake comparison acceptance and scoped compatibility metadata | Both reconciliation packets assembled; 23-case metadata candidate `eefe28c013` is pushed on actual FK landing and in two independent CODE reviews; raw differences retained | Coordinator batches exact entries under existing approvals, obtains required reviews/checks, and resolves the owner-pending sampling entry |
| R9 | Obtain two final-source applicability reviews and rerun all affected completed results | Owner has now approved bounded reuse; completed 255-question model matrix, 18 census questions, 20 memory controls and local campaigns retain original IDs | Two independent reviewers verify source impact; coordinator retains only proven-unaffected results and dispatches every affected rerun |
| R10 | Obtain green full CI on the final integrated branch | Current run `40322768900` is on older `7e41bb73a5`: 74 successful jobs, zero failed, three running as checked at 13:38 UTC; it lacks the now-landed timeout fix and pending FK fix | Coordinator integrates repairs and dispatches full final-source CI; fix any remaining failure under normal review rules |
| R11 | Resolve the second Snowflake sampling allowance | Owner requested an explanation; allowance remains unapproved | Owner decides whether different random 10000-row samples from identical SQL may pass this original case; all other checks and the 5% accuracy limit remain required |
| R12 | Close final acceptance, run the sealed check last, merge and push Phase 3 into `rust-core` | Not started; protected `rust-core` remains Phase 2 | Coordinator obtains both final evidence approvals, closes all remaining performance-report dispositions, runs the sealed check last, then merges and pushes |

R2 and R9 overlap on source-impact analysis but have separate required outputs: affected reruns and independently justified retained evidence.
R4 uses 249 SQL cases plus six existing J2 shapes in three timezones on each architecture.
There is no new requirement to rerun accepted performance measurements or optimize their accepted costs before closure.
The original performance reports still contain incomplete pairs and missing stock-report inputs; final acceptance must account for them explicitly rather than calling those reports blanket PASS.

## Current execution and integration

- Pushed Phase 3 head: `dd5b3f37ef07f80800c7d31be2e4373a93976ec0`.
- The Databricks FK alias-only fix is landed there, with both exact carry approvals, a normal 188-second handoff and terminal explicit landing verification.
  It retains the genuine `aa27490b64` native producer identity; no rebuilt or relabeled native is claimed.
- The case-map CI timeout fix previously landed at `c7e38c6fa2` after both code approvals and a passing normal 477-second handoff, including the whole runner test module in 322 seconds.
  It loads the existing 394-case map once instead of repeatedly revalidating the complete corpus inside the comparison loop; assertions and coverage are unchanged.
- Prior CI `40315122377` on `ad28ef4287` finished with 80 successful jobs and one contract-tooling timeout; its 201 other contract tests passed.
- Current [CI 40322768900](https://github.com/NVIDIA/structured-data-models-client/actions/runs/40322768900) began after a 36-minute queue and still runs code before that timeout repair.
  Its current running jobs are Rust tests, root tooling and contract tooling.
- FK author owns R1 and affected-source handoff; cloud author owns Spark, Java, notebooks and remaining Databricks checks; coordinator owns integration, finite acceptance and this tracker.
- All eight official `7e41bb73a5` artifact groups are authenticated locally; production assets are published with all remote SHA readbacks verified.
  Notebook carry reviews remain pending; one independent FK source-impact proof is frozen, with the second requested.
- Failed Java attempts and their partial Python observations are preserved; both owned empty schemas were verified and removed without force.
- Maximum three active package lanes, two heavy local slots, Cargo jobs capped at two.
  Both independent Sol High code approvals precede package tests and builds; changed code requires renewed approvals.

## Completed evidence retained for final review

- Full model/runtime matrix: 21 profiles, 42 terminal cloud executions, 255 questions, 114 positive pairs with 7485 exact typed rows and 141 approved matching endpoint refusals; all logical payloads exact, zero actual differences or operator errors on actual `76eb785b91`.
  Receipt: `cfabf1853ccc01d722da21d0f9261d7a970d95cbd6f3566ad9a8ee2b380e28ce`.
- Full-data census: all 18 original questions pass across DuckDB, SQLite and PostgreSQL; AdventureWorks 173106 and H&M 16931173 rows retained, no unexpected refusal and all owned PostgreSQL fixtures cleaned.
  Actual source/native identity is `8a7323cb66`; receipt: `ac73037cf1b3c96f6bfeee4e56ec26dc173e754251abdab5c4469c4ba4b9c0f5`.
- All 20 native/host memory and ownership controls pass on their recorded source.
- All three local 200-source campaigns complete; finite decimal/date, SQL numbering and text-time evidence remains attached to the raw reports for final review.
- Snowflake's original 20 generated-source checks and 33 provider cases plus both existing fault controls are complete; this does not substitute for the pending Databricks checks.
- Original Databricks 10001-entity multicall accuracy comparison passes: aggregate RMSE 2.6043078493 old versus 2.6043036069 Rust; the one affected row improves by 14.91%.
  Both use identical SQL, source-row multiset and positional RNG draws, but unordered warehouse input changes the selected context entities; requests are not claimed identical.
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
No new warehouse CPU recorder or replay machinery is required on the critical path.

## Evidence and history

Detailed original logs, hashes, source-impact proposals and failed attempts remain in the existing orchestration state directory.
The immediately preceding uncommitted tracker draft is preserved there as `phase-3/handoffs/validation-tracker-before-20261026-1341.md`.
Git retains the earlier CI history and decision trail.
This tracker changes status only; it does not grant comparator tolerances, native identity exceptions or new compatibility approval.
