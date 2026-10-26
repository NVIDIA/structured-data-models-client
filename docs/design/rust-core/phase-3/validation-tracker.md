# Phase 3 validation tracker

Updated 2026-10-26, 03:30 UTC.
Phase 4 is paused at the owner's request; current agents focus on Phase 3.

## Acceptance

- Compare the same data, queries, model, settings and prediction entities against the frozen Python client.
- Check schemas, values, nulls, row multiplicities, logical request payloads and required errors under the approved compatibility decisions.
- Record exact prediction parity when obtained.
- For differing predictions, record the task's quality metric, Python baseline, Rust result and degradation; the owner accepts at most 5% worse prediction accuracy, with parity or a 1–2% difference preferred.
- Keep prediction disagreement separate from measured accuracy degradation.
- Record latency and throughput separately; the 5% accuracy threshold does not set a speed threshold.
- Reference recordings, captured observations, successful model calls and completed comparisons are distinct stages.
- Preserve original source/build IDs and failed evidence; no tolerance or comparison exclusion is implied by this tracker.

## Current matrix

Databricks SQL, Databricks Spark, Snowflake and Lakebase are four execution paths.
Metric views and semantic views are tracked separately below as warehouse feature coverage.

| Area | Data and payload comparison | Model comparison | Quality verdict | Owner | Next action |
|---|---|---|---|---|---|
| Databricks SQL | Ordinary SQL smoke payload matches exactly; full 302-case comparison pending | One ordinary SQL smoke: all 75 prediction rows match exactly | Exact parity on this smoke; full matrix pending | Databricks agent and coordinator | Finish shared repair checks; run full corrected-client comparison |
| Databricks Spark | Full 300-case corrected-client comparison pending | Corrected-client comparison pending | Not measured | Databricks agent | Use the reference-compatible Spark runtime and complete comparisons |
| Snowflake | Prior 282 observations retained; fresh isolated cases exposed capture-status and large-key error issues | Final comparison pending | Not measured | Snowflake agent | Resolve the concrete capture/error causes before another full campaign |
| Lakebase | Two saved representative cases compare exactly; full 282-case corrected-client comparison pending | Pending | Not measured | Lakebase agent | Check reference identities, then run full data and model comparisons |
| Databricks metric views | One SQL smoke payload matches exactly; 19 cases per SQL/Spark path remain in the full comparison | SQL smoke: all 25 prediction rows match exactly | Exact parity on this smoke; broader coverage pending | Databricks agent | Complete SQL and Spark metric-view coverage |
| Snowflake semantic views | Nine prior observations retained; remaining twin case now executes, but its reference identity and comparison need resolution | Pending | Not measured | Snowflake agent | Resolve reference identity and capture-status issues, then complete all ten cases |

## Active blockers and next steps

1. Shared repair: the Databricks batch index now uses the required int64 transport, and comparator expectations use canonical warehouse dialects and source namespaces.
   Both substantive reviews passed, the Rust batch and generated SQL checks passed, and fresh production/test-operation clients were built.
   Package checks exposed two test-formatting issues and then two local test-fixture declarations; the latest small fixture correction awaits review and checks.
2. Snowflake: an unsupported Arrow-fetch attempt followed by a successful row-fetch fallback remains marked as a failed statement in captured evidence.
   Diagnose and correct terminal-status recording without hiding failed execution or changing comparison policy.
3. Snowflake: the isolated large-key case returns `INTERNAL_ERROR` where the reference returns `INVALID_REQUEST`.
   Inspect the actual core error and fix its cause or classification.
4. Semantic view: the remaining twin case's saved reference digest differs from the current case.
   Establish the exact source difference before deciding whether a bounded reference refresh is necessary.
5. Model inventory: the first reference endpoint run produced six successes and seven refusals across thirteen questions.
   Distinguish endpoint-rejected inputs from useful accuracy comparisons; two successful ordinary/metric-view comparisons now match exactly.

## Finish sequence

1. Complete and land the shared repair with its required reviews and checks.
2. Resolve the concrete blockers and finish every matrix row, recording the producing source and reference for each result.
3. Complete direct performance checks, memory checks and final CI on the accepted source.
4. Run the sealed check last, complete final acceptance, then merge and push Phase 3 into `rust-core`.
5. Keep Phase 4 paused until the owner resumes it.

The earlier Python reference inventory is present for all selected warehouse cases.
It does not by itself establish that every saved reference digest is current or every final Rust comparison passes.
Detailed commands, original logs, hashes and private runtime receipts stay in the existing orchestration state directory.
