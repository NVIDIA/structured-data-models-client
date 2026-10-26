# Phase 3 validation tracker

Updated 2026-10-26, 04:01 UTC.
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
| Databricks SQL | Ordinary SQL smoke payload matches exactly; full 302-case run active, comparison follows | One ordinary SQL smoke: all 75 prediction rows match exactly | Exact parity on this smoke; full matrix pending | Databricks agent and coordinator | Finish active SQL run and compare all 302 cases |
| Databricks Spark | Full 300-case corrected-client comparison pending | Corrected-client comparison pending | Not measured | Databricks agent | Use the reference-compatible Spark runtime and complete comparisons |
| Snowflake | Prior observations retained; combined capture-status and large-key error fix reviewed with one optional-driver dependency finding | Final comparison pending | Not measured | Snowflake agent | Correct the review finding, obtain both approvals, validate fixes, then finish all 304 cases |
| Lakebase | 282/282 observed with no operator failures; 255 cases have no gating differences; 27 need approved-policy applicability, stable type identity or catalog-read correction | Original six-question reference/core pair active, followed by four existing common quickstarts | Pending | Lakebase agent and coordinator | Finish model pairs and resolve the 27 differing data cases |
| Databricks metric views | One SQL smoke payload matches exactly; 19 cases per SQL/Spark path remain in the full comparison | SQL smoke: all 25 prediction rows match exactly | Exact parity on this smoke; broader coverage pending | Databricks agent | Complete SQL and Spark metric-view coverage |
| Snowflake semantic views | Twin now executes; its stale reference was identified as type-label-only case edits and refreshed with frozen Python | Pending | Not measured | Snowflake agent | Validate the shared capture fix, then compare all ten existing semantic-view cases |

## Active blockers and next steps

1. Shared warehouse repair is landed and pushed at `df21cf66a35d38679d71c8dafa317cad35822163` after two independent approvals and all normal package checks.
   Genuine live-client artifacts retain their producing source `ee71d0cf53a1d615580d459b4cf2c4d1a31eef1a`; later test-only corrections do not relabel them.
2. Snowflake's combined capture fallback and large-key error correction needs one review fix to retain optional-driver support, then two fresh approvals and normal checks before live execution.
3. Lakebase's completed run exposes mostly existing approved Postgres differences that do not yet apply to its logical case names, plus one unstable composite-type identifier and one extra catalog read.
   Each needs exact evidence or a correction; none is accepted by ignoring the difference.
4. Databricks SQL is active with exclusive fixture ownership; Spark follows after cleanup using the reference-compatible runtime.
5. Model comparison uses the same ten existing ordinary questions across warehouses, plus Databricks' three existing metric-view questions.
   Retain endpoint refusals and compare their underlying cause under the owner's approval.

## Finish sequence

1. Keep the landed shared repair fixed while finishing the remaining reviewed corrections.
2. Resolve the concrete blockers and finish every matrix row, recording the producing source and reference for each result.
3. Complete direct performance checks, memory checks and final CI on the accepted source.
4. Run the sealed check last, complete final acceptance, then merge and push Phase 3 into `rust-core`.
5. Keep Phase 4 paused until the owner resumes it.

The earlier Python reference inventory is present for all selected warehouse cases.
It does not by itself establish that every saved reference digest is current or every final Rust comparison passes.
Detailed commands, original logs, hashes and private runtime receipts stay in the existing orchestration state directory.
