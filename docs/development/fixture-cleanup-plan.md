<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fixture cleanup

Base: PR #75, `b450c957f`; branch: `rust-fixture-cleanup`.
Step 2 is complete, including final independent coverage review and reconciliation of all file/group totals.
Step 3 is active: simplify obsolete test plumbing while retaining useful correctness and compatibility gates.
Do not merge this branch without instruction.

## Result so far

The 747-file inventory now has a decision for every file: 184 removed, 13 reduced, and 550 retained.
These counts include test source and schemas, not just JSON fixtures.
Sizes below are measured from the tracked inventory and current files, excluding replacement test code.

| Inventory measure | Before | Now | Removed |
| --- | ---: | ---: | ---: |
| Bytes | 33,169,701 | 5,218,869 | 27,950,832 (84.3%) |
| Text lines | 515,268 | 128,656 | 386,612 (75.0%) |

Large captured view, request, SQL, payload and result outputs now have independently expected behavioral assertions.
Large repeated inputs are constructed in test code, preserving their original cases and boundaries.
Expected answers come from authored values, raw source rows, or independent external algorithms, never the production function under test.
No product behavior was changed; production-source edits only correct two obsolete documentation comments.
Test-only input factory and consumer changes support the replacements.

## Coverage kept

| Area | What the replacements check |
| --- | --- |
| SQL, all five dialects | Request kinds, capability/source/read-as probes, identifiers and namespaces, exact/composite keys, parameters, schemas, windows/order and refusal semantics. Local execution supplements warehouse rendering checks. |
| Sampling and requests | Independent source labels, complete related histories, leakage prevention, batch sizes/order, ranking scores/top-k, forecast steps and request lifecycle. |
| Python, Rust and Java | Existing language consumers remain; shared input factories work in each affected language. Java native-binding output recordings were replaced with direct assertions. |
| Types and responses | Exact numeric identity, nulls, decimals, temporal values, typed frames, response alignment, errors and result interpretation. |
| Saved requests | Historical serialized inputs remain; explicit metadata/frame checks, saved-data-to-wire continuity and scripted HTTP answers replace opaque output digests. |
| PR #75 | Source lifetime, foreign-key alias and other existing regression cases remain. This cleanup does not revert product changes. |

Retained files have a specific reason: authored test inputs, versioned protocol schemas/locks, API/capability inventories, or independently obtained interoperability oracles.
Examples include timezone data, historical pickles, Unicode lowercase mappings and exact NumPy/RNG/sort behavior.
Small outcome inventories still serve current consumers; broad runner simplification is step 3.
Retention does not claim those unchanged suites were freshly rerun.

Three inputs remain explicitly protected because their runtime consumers are unresolved:

- `contract/baselines/ref-pandas3/python-old.jsonl`
- `contract/census-words/duckdb.jsonl`
- `contract/fixtures/adventureworks/notebook-fa0637d4.ipynb`

They remain byte-for-byte unchanged.
The notebook's documented reader is not present in the current classification test, so documentation alone is not treated as proof of execution.

## Evidence and tracking

- [File inventory](fixture-cleanup-inventory.csv): all 747 original paths, decisions, actual remaining sizes, preserved behaviors and per-file validation/review evidence.
- [Dependency map](fixture-cleanup-dependencies.csv): baseline fixture → loader → test → CI/registry edges, plus current `replacement-assertion` edges.
- [Group ledger](fixture-cleanup-ledger.csv): 17 groups, reconciled totals, implementation status and validation limits.

Deleted paths in baseline dependency edges are deliberate history, not current runtime dependencies.
Only the three protected retained inputs have unresolved edges.
The map is static evidence; it does not prove a test ran, or that every conditional selector executed.
These are temporary cleanup records, not a new permanent capability framework.

Affected suites were validated at implementation boundaries and coverage decisions received independent review.
Representative results below are separate runs with possible overlap, not an additive test total.

| Batch | Evidence |
| --- | --- |
| Request transcript replacements | 749 passed, 158 skipped; PostgreSQL was unavailable for that batch. |
| Five-dialect SQL replacements | 36 Rust matrix tests; 42 Rust embedded-SQL tests and 3 SQLite tests passed. |
| SQL statement replacements | 368 Python/PostgreSQL 17 checks and 20 Rust DuckDB tests passed. |
| Java shared query factory | 84 `BindingTest` tests passed. |
| Wire results | 36 tests passed against scripted HTTP responses. |
| Recent PostgreSQL replacements | Exact keys: 9; conversion: 25; temporal candidates: 17 passed against local PostgreSQL 17. |
| Historical pickles | 10 passed after recording deletion. |
| Final request-kind replacement | 29 Python provider tests passed; post-prune 2 Python and 3 Rust tests passed across all five readers. |

The final independent review is recorded in `/tmp/fixture-step2-completion-review.md`; the task-created PostgreSQL container has been stopped.
Ruff and Clippy were run for affected code; detailed logs and review paths are recorded per file.
Some earlier PostgreSQL variants were skipped and remain unverified by those runs.
Later PostgreSQL runs cover their named batches only.
Scripted HTTP tests exercise real client transport but are not live model validation.
No claim is made here of exhaustive live warehouse/model coverage or a fresh integrated run.

## Remaining steps

| Step | Status and scope |
| --- | --- |
| 1. Map and prioritize | Complete: inventory, consumer mapping and independent audit. |
| 2. Replace and delete | Complete: replacements, focused validation, reconciled tracking and final independent review. |
| 3. Simplify plumbing | In progress: unused loaders/recorders, obsolete CI comparison steps, manifests/generators and contributor rules. Keep useful API/protocol enforcement. |
| 4. Review and validate | Then run integrated Rust/Python/Java suites, connector checks, lint/type checks and schema/capability gates. Run live checks where changed behavior or maintained requirements require them. |

Step 2 completion requires actual file totals, no unresolved deleted-input consumers, independent review of coverage decisions and honest affected-test evidence.
It does not mean every recorded artifact should be deleted or that steps 3/4 have already passed.
Report fixture savings separately from replacement test-code growth; do not count minification or moving data as savings.
Keep connector/model/language coverage and PR #75 fixes throughout the remaining work.

### Step 3 decisions

Keep API snapshot enforcement: it detects accidental exported-name/signature changes.
Keep current wire assertions and pinned-schema validation, but rename the former snapshot script to `scripts/check_wire_behavior.py`.
Move the former SQL recorder to `tests/support/sql_source_scenarios.py`: it only supplies current source scenarios to correctness tests.
Retain case-observation uploads for failure diagnosis; their old SQL-golden comments no longer apply.
No execution job is removed based only on a stale name or comment.

Removed the unused shared-fixture compression format and its two format-only tests.
No current JSON uses that format; Snowflake inputs now load directly, while repository size, schema-lock and named-consumer checks remain.
Removed the uncalled reference-client guard, warehouse `capture_line` exporter and Arrow IPC reader.
The exporter was the only non-test consumer of its driver-version helper, so that helper and its six isolated recorder-metadata tests were removed together.
Active current-client observation, capture conservation and Rust replay paths remain; their data/transport checks are separate from the retired exporter.
Compression cleanup: 43 tests passed and the repository-size gate passed (`/tmp/fixture-step3-compression-removal.log`, `/tmp/fixture-step3-size-gate.log`).
