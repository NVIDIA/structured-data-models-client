# Phase 3 validation tracker

Updated 2026-10-26, 05:45 UTC.
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
| Databricks SQL | Ordinary SQL smoke payload matches exactly; 263/302 observations retained; interrupted process cleaned up safely, remaining 39 cases running, comparison follows | All 13 model cases pass the existing comparison: 385 prediction rows across six successes match exactly; seven endpoint refusals match exact codes | Exact predictions and refusal parity for this Python model pair; remaining runtime cells pending | Databricks agent and coordinator | Finish active SQL run and compare all 302 cases |
| Databricks Spark | Full 300-case run prepared; needs the original ARM/serverless runtime and genuine new ARM client artifact | All 13 original managed-runtime model pairs compared: 385 exact prediction rows, seven exact-code refusals, exact logical payloads | Exact parity on prior client; corrected final-client runtime comparison pending | Databricks agent | Build the truthful ARM client in final CI, then run all 300 original cases after SQL cleanup |
| Snowflake | Generic contract correction landed; 122/304 cases observed before a real valid-Arrow alignment failure halted the campaign safely | Final comparison pending | Not measured | Snowflake agent | Complete the twice-reviewed Arrow alignment fix and normal checks, then resume the original data and model cases |
| Lakebase | All 282 existing cases reconciled: 128 original differences match existing approvals, two type-capture differences repaired, one retained count difference has two exact-equivalence approvals | All ten model pairs pass: four exact prediction matches (285 rows), six exact refusal matches, all payloads exact except request IDs | Exact prediction/refusal parity for this local Python pair; remaining runtime cells pending | Lakebase agent and coordinator | Complete final source integration and remaining runtime cells; pinned 173106-row fixture is restored exactly |
| Databricks metric views | One SQL smoke payload matches exactly; 19 cases per SQL/Spark path remain in the full comparison | All three original SQL model questions have exact payloads and either exact predictions or matching endpoint refusal | SQL model parity; Spark and full data coverage pending | Databricks agent | Complete SQL and Spark metric-view coverage |
| Snowflake semantic views | Twin now executes; its stale reference was identified as type-label-only case edits and refreshed with frozen Python | Pending | Not measured | Snowflake agent | Fresh frozen-Python references collected; finish comparisons on corrected final client |

## Active blockers and next steps

1. Phase 3 head `0d2b435082c39350b0b6b327df783b70eecb4f95` includes the reviewed generic contract and Lakebase corrections.
   The frozen local campaign source `2a4b42033c9658c34f717700db5cee60c05356cd` has the same product tree; original source/build IDs stay recorded.
2. Snowflake exposed a real native ingestion bug: a valid Decimal128 value buffer can be unaligned.
   Candidate `a74e9eae95c4f1b0a99834ae83f0613fe84ca52f` has two independent CODE approvals; the genuine native build passed; boundary checks exposed four test-fixture setup failures from writing a read-only sliced buffer.
   The author is collecting remaining required groups, then correcting those fixture writes without changing product code and obtaining fresh reviews.
   Preserve malformed-input refusals and exact values; no compatibility waiver replaces this fix.
3. Lakebase's 282 data cases and ten local model pairs are reconciled on the recorded prior client.
   Its restored 173106-row fixture is read-leased by six reference runtime jobs; final corrected-client runtime coverage remains required.
4. Databricks SQL retained 263 observations before a SIGKILL whose cause is unproven.
   The owner verified and cleared only its four owned fixture tables, then launched the remaining 39 original cases once.
   Spark's 300-case campaign waits for that exclusive fixture lease and a genuine corrected ARM artifact.
5. SQL and managed Spark model pairs both have 13/13 parity on the recorded prior client, including 385 exact prediction rows each and seven matching endpoint refusals.
   Remaining original runtime references are being collected in parallel; final coverage is 21 profiles and 255 questions per client pass.
   Corrected-client matrix jobs remain held until the actual landed source and artifact receipts exist.
6. The coordinator owns CI fixture repair, integration and this tracker.
   The Snowflake agent owns native alignment checks and Snowflake continuation; the Databricks agent owns remaining SQL/Spark coverage and artifact collection; the runtime agent owns cloud references and the bounded matrix report correction.
   Direct paired timing waits for quiet warehouse fixtures.

## Full CI and phase exit

- **Required: green full CI on the final integrated Phase 3 source before closing T3.10 or merging into `rust-core`.**
- [Run 40258337766](https://github.com/NVIDIA/structured-data-models-client/actions/runs/40258337766) is terminal with **81 passing jobs and one failed job**, `tooling (contract)`, on older source `2a492e889c8a3b0e601a5df252a1f4465596148a`.
- The previously identified contract-fixture repair is already landed at `f57b4ca19330fb122041d3fb22fcc14ac743aab6` and passed its normal package checks; this older red run does not prove the repaired final source is green.
- The failing test is `contract/tests/test_t20f_gate.py::test_compare_takes_database_keys_and_refuses_malformed_ones_by_name`, which the landed fixture repair addresses.
  Those corrections are now integrated; the newer full run below is the current diagnostic source.
- [Run 40271951963](https://github.com/NVIDIA/structured-data-models-client/actions/runs/40271951963) on `0d2b435082` has 23 failed jobs and one active job at this update.
  Actual logs show stale Databricks SQL fixture dispatch and Rust offset parsing, repeated across Python/runtime suites, plus downstream missing-observation artifacts.
  Two contract cells also fail against the prior checkpoint’s old case-key schema and an imprecise protected-path landing trailer; retain those failures and verify the next normal checkpoint and exact protected-path record.
  One Python key-set fixture repair is already twice reviewed and passed its whole file with real PostgreSQL; the additional three-file correction `4dd372365` has both independent CODE approvals and its whole Python/Rust checks are running before the next final-source CI run.
  The new Arrow product correction also requires a fresh full run; this current run is not final acceptance.
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
