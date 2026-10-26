# Phase 3 validation tracker

Updated 2026-10-26, 04:30 UTC.
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
| Databricks SQL | Ordinary SQL smoke payload matches exactly; full 302-case run active, comparison follows | All 13 model cases pass the existing comparison: 385 prediction rows across six successes match exactly; seven endpoint refusals match exact codes | Exact predictions and refusal parity for this Python model pair; remaining runtime cells pending | Databricks agent and coordinator | Finish active SQL run and compare all 302 cases |
| Databricks Spark | Full 300-case run prepared; needs the original ARM/serverless runtime and genuine new ARM client artifact | Reference/core model jobs active on their original managed runtime | Not yet compared | Databricks agent | Build the truthful ARM client in final CI, then run all 300 original cases after SQL cleanup |
| Snowflake | Prior observations retained; capture and large-key fixes have both approvals; boundary tests pass after restoring the proven optional engine, with one test-only assertion carry in final review | Final comparison pending | Not measured | Snowflake agent | Finish the reviewed test carry and normal checks, then all 304 data cases and model comparisons |
| Lakebase | All 282 existing cases reconciled: 128 original differences match existing approvals, two type-capture differences repaired, one retained count difference has two exact-equivalence approvals | All ten model pairs pass: four exact prediction matches (285 rows), six exact refusal matches, all payloads exact except request IDs | Exact prediction/refusal parity for this local Python pair; remaining runtime cells pending | Lakebase agent and coordinator | Complete final source integration and remaining runtime cells; pinned 173106-row fixture is restored exactly |
| Databricks metric views | One SQL smoke payload matches exactly; 19 cases per SQL/Spark path remain in the full comparison | All three original SQL model questions have exact payloads and either exact predictions or matching endpoint refusal | SQL model parity; Spark and full data coverage pending | Databricks agent | Complete SQL and Spark metric-view coverage |
| Snowflake semantic views | Twin now executes; its stale reference was identified as type-label-only case edits and refreshed with frozen Python | Pending | Not measured | Snowflake agent | Validate the shared capture fix, then compare all ten existing semantic-view cases |

## Active blockers and next steps

1. Shared warehouse repair is landed and pushed at `df21cf66a35d38679d71c8dafa317cad35822163` after two independent approvals and all normal package checks.
   Genuine live-client artifacts retain their producing source `ee71d0cf53a1d615580d459b4cf2c4d1a31eef1a`; later test-only corrections do not relabel them.
2. Snowflake's substantive fixes passed both reviews and initial boundary checks; a final test-only property-read assertion carry is being completed before the normal handoff and live campaign.
3. Lakebase's completed run exposes mostly existing approved Postgres differences that do not yet apply to its logical case names, plus one unstable composite-type identifier and one extra catalog read.
   The type-name fix, explicit existing-approval rows and stale register grammar correction passed both reviews and normal package checks.
   One original OID-sensitive case is being refreshed with the corrected capture, preserving all earlier evidence and restoring the exact owned AdventureWorks fixture afterward.
   Two independent reviews classified the exact extra empty-catalog read as covered by existing approved query-batching equivalence; its original strict failure is retained.
   Each needs exact evidence or a correction; none is accepted by ignoring the difference.
4. Databricks SQL is active with exclusive fixture ownership; Spark follows after cleanup using the original serverless ARM runtime.
   The corrected final source needs a genuine ARM native artifact; final CI supplies it, avoiding use of an obsolete client.
   Direct paired performance checks wait for SQL fixture traffic to finish so the timing comparison is interpretable.
5. Model comparison uses the same ten existing ordinary questions across warehouses, plus Databricks' three existing metric-view questions.
   Retain endpoint refusals and compare their underlying cause under the owner's approval.
   Databricks SQL's thirteen-question model pair now has exact public-result/payload parity; Spark reference/core model jobs are active separately from the serialized data writers.


## Full CI and phase exit

- **Required: green full CI on the final integrated Phase 3 source before closing T3.10 or merging into `rust-core`.**
- [Run 40258337766](https://github.com/NVIDIA/structured-data-models-client/actions/runs/40258337766) is terminal with **81 passing jobs and one failed job**, `tooling (contract)`, on older source `2a492e889c8a3b0e601a5df252a1f4465596148a`.
- The previously identified contract-fixture repair is already landed at `f57b4ca19330fb122041d3fb22fcc14ac743aab6` and passed its normal package checks; this older red run does not prove the repaired final source is green.
- Recheck the failed test names against that repair, integrate the reviewed Snowflake and Lakebase corrections, then dispatch full phase-end CI and inspect every terminal job.
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
