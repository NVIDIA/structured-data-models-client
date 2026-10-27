# Phase 3 final disposition

This record covers the acceptance candidate before its authorized integration into `rust-core`.
The [current validation tracker](https://github.com/NVIDIA/structured-data-models-client/blob/rust-core-phase-3-topic-T3.10-validation-tracker/docs/design/rust-core/phase-3/validation-tracker.md) records the eventual merge identity and current execution status.
The [earlier evidence packet](README.md) is the reviewed HOLD snapshot; its pending statements describe that earlier point in time.

## Candidate and CI

| Item | Verified identity or result |
|---|---|
| Phase 3 candidate | `ff86cd1c9e5ca53d0f5e9bc28399916294631549` |
| Tested source tree | `ff4402e3c5864985d33767801bf23b36d7300b7d` |
| Frozen Python reference | `43e55f4f58924384c3cb52f67c8ed2e73ba45fa1` |
| Final full CI | [40473118676](https://github.com/NVIDIA/structured-data-models-client/actions/runs/40473118676), attempt 2, SUCCESS: 81 successful jobs and two intentional skips |
| Final CI receipt SHA-256 | `2553d3aca0d6f287a6d5111cc3647b31cec44d4eb259f124d87dfe8db77993bf` |

The single failed Java ARM SQLite job was retried after an artifact-list connection reset before setup or testing.
The retry passed in 145 seconds, and its dependent timing report passed.
The other 79 successful jobs retain their original execution times; they were not executed again.
GitHub changed the two skipped jobs' metadata timestamps on retry, including inconsistent start/end ordering; their raw records remain preserved and neither is represented as executed.
The original failure, original job identities, successful retry and all 83 final job records are retained.

## Actual owner decisions

The owner explicitly approved both final compatibility choices after the comparisons, conditional on final results remaining close.
The existing maximum of 5% prediction-accuracy degradation continues to apply.
The verbatim answer is preserved separately from the unchanged delivered questions and original run-start register.

| Approved scope | Boundary |
|---|---|
| Verified internal SQL and metadata differences | The exact reviewed helper-column and alias correspondences, one additional empty lookup before the same refusal, Java column-type preflight queries, and recreated-view owner/time metadata |
| More accurate decimals on Databricks SQL and Spark | The two reviewed cases use `-0.0875` rather than `-0.08750000000000001`, and `±3e-15` rather than `±3.0000000000000002e-15` |

The SQL disposition preserves the verified source values, row multiplicities, predicates and table/column/link content, apart from the explicitly approved decimal corrections.
Catalog discovery continues to preserve each provider's actual reply order.
These are bounded post-run dispositions, not approval before the draw or a general numeric tolerance.
Owner-answer SHA-256: `eeb6784d5b7a2387a263b9e19f8b8f6886c2a695577419b125de7d9ea697acf8`.
Delivered-question SHA-256: `a6344e4309946bdd88fe336b8c9e2e3c5904bc24623bd4d284c8b12d9b1c9fe7`.

## Comparisons and accuracy

All 40 original cases and 169 supported case/backend pairs remain covered, including both reference passes, Python core and Rust, plus the original 20 Java and 25 old-client DuckDB observations.
All 114 successful reference observations remain successful, with no panic, missing observation or new refusal of a supported input.
The original frozen inputs, derived execution inputs, one-shot marker and failed attempts remain preserved.

The raw H11 counter remains NO-GO at 33 differing cases across 82 legs.
Its threshold remains four, and its approval list was not changed to manufacture a passing counter result.
Under the actual bounded post-run owner decisions, the independently reviewed unapproved remainder is one native-validation trace case, within the unchanged four-case allowance.
That trace remains outside the SQL exception.

The existing model comparisons meet the owner's aggregate accuracy limit.
The corrected Databricks and Snowflake diamond comparisons are respectively 2.4323311659790736% and 1.9582898136461635% worse than the old client, within 5%.
The local corrected graph comparison is approximately 2.4701% worse, also within 5%.
The controlled Snowflake 10001-entity comparison has identical original rows, contexts, requests and predictions, with RMSE ratio 1.0.
The Spark stress comparison retains its aggregate RMSE ratio 1.00000294257 and its separately disclosed single-row 11.4946% worse error; the aggregate result is not a claim of identical individual predictions.
The earlier independently sampled Snowflake 26.51% loss remains historical evidence rather than a controlled-input migration result.

## Retained evidence and integration

The complete warehouse, view, Java, model, notebook, provider, memory, conversion, differential and performance evidence remains in the earlier packet with its actual producing commits and hashes.
Source234 and earlier binaries and live observations keep their authentic identities under the two independent source-applicability opinions.
Current ff86 changes only the reviewed testkit and exact numeric judging paths; shipped client, wire, dependency and build inputs remain unchanged from source234.
The genuine ordinary-gate main/check7/check12 failures remain recorded beside the owner's explicit frozen-baseline exception and both policy reviews.
Previously accepted performance and size costs, unsupported storage pairings, incomplete measurements and diagnostic limits retain their exact scope.

Both independent final reviewers mapped all 14 locked exits to actual evidence and gave FINAL GO for exact ff86 and tree ff4402.
They verified the literal owner approvals, the existing aggregate accuracy condition, green final CI and the bounded remaining trace.

| Final source-acceptance opinion | SHA-256 |
|---|---|
| General review | `d303aed8038ffb184219abd6400b778b57e36b96dc60c5766449ed8a6e3bc6b3` |
| Independent focused review | `1e4808d227875894bac7c845ccffd18380e67c9a2f3e183b7761be285e6c7778` |

After those opinions and the separate documentation checks pass, the authorized integration is a signed non-fast-forward merge with the current `rust-core` tip as first parent, exact ff86 as second parent, and tree ff4402 unchanged.
The merge must pass the existing epic/provenance checks and be pushed normally; automatic core-push CI is observed separately.
Phase 4 remains owned by its separate coordinator, who rebases it after the actual core merge is available.
