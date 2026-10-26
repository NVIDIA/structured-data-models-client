# Phase 3 validation tracker

Updated 2026-10-26, 06:08 UTC.
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
| Databricks SQL | Ordinary SQL smoke payload matches exactly; Older-client run accounted for all 302 cases: 301 observations and one diagnosed runner failure; corrected-client 301-case run is active and the connector case remains separately pending | All 13 model cases pass the existing comparison: 385 prediction rows across six successes match exactly; seven endpoint refusals match exact codes | Exact predictions and refusal parity for this Python model pair; remaining runtime cells pending | Databricks agent and coordinator | Finish corrected-client 301 cases, run the connector case with the landed runner fix, and reconcile all 302 |
| Databricks Spark | Full 300-case run prepared; needs the original ARM/serverless runtime and genuine new ARM client artifact | All 13 original managed-runtime model pairs compared: 385 exact prediction rows, seven exact-code refusals, exact logical payloads | Exact parity on prior client; corrected final-client runtime comparison pending | Databricks agent | Build the truthful ARM client in final CI, then run all 300 original cases after SQL cleanup |
| Snowflake | Alignment correction landed; real wide-decimal prediction now matches Python, oversized-key refusal remains correct, semantic twin matches exactly; current 304-case run active | Final comparison pending | Not measured | Snowflake agent | Finish the current original data campaign and ten ordinary plus five existing semantic model cases |
| Lakebase | All 282 existing cases reconciled: 128 original differences match existing approvals, two type-capture differences repaired, one retained count difference has two exact-equivalence approvals | All ten model pairs pass: four exact prediction matches (285 rows), six exact refusal matches, all payloads exact except request IDs | Exact prediction/refusal parity for this local Python pair; remaining runtime cells pending | Lakebase agent and coordinator | Complete final source integration and remaining runtime cells; pinned 173106-row fixture is restored exactly |
| Databricks metric views | One SQL smoke payload matches exactly; 19 cases per SQL/Spark path remain in the full comparison | All three original SQL model questions have exact payloads and either exact predictions or matching endpoint refusal | SQL model parity; Spark and full data coverage pending | Databricks agent | Complete SQL and Spark metric-view coverage |
| Snowflake semantic views | Twin now executes; its stale reference was identified as type-label-only case edits and refreshed with frozen Python | Pending | Not measured | Snowflake agent | Fresh frozen-Python references collected; finish comparisons on corrected final client |

## Active blockers and next steps

1. The reviewed Arrow alignment and CI fixture repair landed at `f531938047e48dd3a462360b78b2c91c9c958fe6` after normal checks.
   Its genuine production client is built at candidate `b049980d4a295f393344bef7efa65b72828232fb`; the stable live worktree preserves that source and its native artifact IDs.
2. The two-file runner correction landed and verified at current Phase 3 head `76eb785b91aeaa319e6537ac8017a70b6f186ed4`.
   It excludes exactly three host-owned connection paths from forced Rust routing, retains all core calls and unknown-route refusals, and changes no shipped client code, warehouse case or reference.
   Both independent reviews and normal checks passed.
3. Databricks SQL's older-client comparison exposed decimal refusals covered by the shared alignment cause; its 5541 raw differences remain preserved and are not accepted as parity.
   The corrected genuine client is now running the original 301 non-connector cases with exclusive fixture ownership; the remaining original connector case follows with the fixed runner.
   Spark's 300-case campaign waits for fixture cleanup and the genuine corrected ARM CI artifact.
4. Snowflake's current original 304-case campaign is running after three successful real boundaries; the older 122 observations remain distinct.
   Strict schema projection, warning and SQL diagnostics remain visible; no broad schema, relationship or payload ordering exception is implied.
5. Lakebase's 282 data cases and ten local model pairs are reconciled on their recorded source.
   The canonical 173106-row fixture remains read-leased by six reference runtime jobs.
   Fourteen reference profiles are staged; one ARM Spark reference is retrying a confirmed cloud spot-driver interruption without changing its runtime or source.
   Future classic core jobs will use an on-demand driver under the existing policy, preserving node type, runtime, architecture and dataset settings.
6. All 21 corrected-client model/runtime profiles remain required, totaling 255 questions per client pass.
   Their plans are being bound to actual `76eb785b91` source publication and genuine CI artifacts.
   The bounded report correction passed both independent reviews and 26 controls; those controls are not model acceptance evidence.
7. The coordinator owns CI, integration and this tracker.
   Snowflake owns its data/model campaign and source publication; Databricks owns SQL/Spark coverage and the sole Python/Java artifact collector; the runtime agent owns cloud references and final core profiles.

## Measured performance blocker

The genuine corrected client completed five alternating paired live Databricks runs with caches disabled and the original slow control.
The original slow control was detected correctly.
Both measured cases fail the plan's unchanged 10% live slowdown limit:

| Existing case | Python median | Rust median | Ratio |
|---|---:|---:|---:|
| Ordinary no-order-90d | 6.42 s | 12.82 s | 2.00× |
| Metric-view no-order-90d | 5.29 s | 10.87 s | 2.06× |

Existing functional records show 12–13 extra fresh-source catalog, key and preview reads; sampling and statistics reads otherwise match.
The owner previously approved re-reading current source facts on every run, whereas the old client keeps table facts.
Two independent source assessments found no safe toggle that meets the limit while retaining the freshness, type, empty-table and error contracts.
Per-table metadata fusion could remove some calls but does not demonstrate the required 10% bound; a broader typed batching design would need separate review.
Keep this failed result visible; do not disable freshness, repeat unchanged runs to select better samples or call Phase 3 complete while the performance decision remains unresolved.
This latency result is separate from the owner's 5% prediction-accuracy limit.
Snowflake timing, offline performance and all remaining exit checks remain required.

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
  Green status is still unproven.
- Fix every remaining failure or flake, obtain both required code approvals, rerun the affected checks and finish with green full CI; no failure is waived or accepted as complete.
- Collect the genuine final-source ARM client artifact for the reference-compatible Spark data run; do not substitute the obsolete client.
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
