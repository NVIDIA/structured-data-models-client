<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fixture cleanup plan

Base: PR #75, `b450c957f`; working branch: `rust-fixture-cleanup`.
Planning inventory: 959,126 tracked text lines, including 499,837 JSON lines.
These are text counts, not production-code counts.

## Step 1 result

The initial inventory and consumer mapping are complete; replacement decisions remain provisional until their assertions are implemented and checked.
The [file inventory](fixture-cleanup-inventory.csv) covers 747 files (33.2 MB, 515,268 text lines), including recorded test source and schemas.
The [dependency map](fixture-cleanup-dependencies.csv) records 3,940 edges, and the [ledger](fixture-cleanup-ledger.csv) tracks 17 groups with proposed actions and coverage to preserve.
File sizes, evidence locations, referenced symbols and ledger totals were checked; product tests were not run.

Start implementation with the large view vectors, request recordings, SQL recordings and client payload recordings.
Keep real interoperability inputs, protocol schemas and compatibility inventories unless their individual review justifies replacement.
The inventory describes proposed actions, not completed deletions or guaranteed line savings.

Three inputs remain on hold because their current runtime consumers are unresolved:

- `contract/baselines/ref-pandas3/python-old.jsonl`
- `contract/census-words/duckdb.jsonl`
- `contract/fixtures/adventureworks/notebook-fa0637d4.ipynb`

The AdventureWorks README claims a notebook reader that is absent from the current classification test; documentation alone does not establish coverage.
These files stay untouched until that uncertainty is resolved.

The map is static evidence, not proof that tests ran or passed.
Shared-helper edges and conditional selectors do not establish that every fixture reaches every caller.
The full live API registry is selected through its manual CLI, not by the committed hosted workflow; the workflow runs a bounded selection instead.

An independent audit confirmed the totals and found a missing dynamic Java selector plus underspecified SQL coverage.
Both are now recorded; group-level replacement proposals still require file-specific assertions before deletion.

## Step 2 progress

The view-expression and view-step groups are implemented and independently reviewed.
Both recordings are deleted: **4,393,707 bytes and 96,943 text lines** removed.
Direct assertions cover expression safety, parameter conversion, view descriptions, substitutions, filters/measures, time selection and join/rollback decisions.
The combined affected suites passed 73 tests; final integrated validation remains step 4.
Request transcript readers are also replaced, and the shared request and batch-reconstruction recordings are removed.
The seven affected Python modules passed 749 tests with 158 skips; the reviewed context-label correction passed its targeted rerun (42 tests, 21 skips).
Postgres was not configured and is not claimed tested; local execution used the current branch’s isolated Python/Rust build.
Total recording removal so far: **25,398,745 bytes and 311,332 text lines**.
The SQL dialect matrix recording and its writer are removed after independent review.
Its replacements cover five-dialect probes, scopes/refusals, schema and key bindings, quoting, aggregates, sampling seeds, and window/order behavior; 36 affected Rust tests and Clippy passed.
The four embedded SQL arrays and their shared snapshot helper are also removed; their replacement batch passed 42 Rust tests and the retained SQLite join tests passed all three tests.
The client materialization and payload-byte recordings are removed after independent review; 39 materialization tests and 12 typed payload-byte scenarios passed.
Forced-route replacements passed 199 tests with 104 PostgreSQL skips and independent review; their recording is removed.
Four SQL family suites now execute all 39 inputs directly and assert current outcomes, result encoding and query-log consistency; all passed with Clippy and independent review.
The large date recording is removed after review; explicit calendar, refusal, warning-format and timezone assertions passed 24 tests and Clippy.
Both SQL statement recordings are removed after independent review: 368 Python guard/PostgreSQL17 tests and 20 Rust DuckDB tests passed, plus the final query-log assertion rerun.
Custom-profile PQL recording is removed; direct tests retain all eleven capability flags, sixteen diagnostic codes and mode/precedence rules, with successful forecast/rank/entity facts.
PQL corpus/quirks recordings and their unused tree comparator are removed; replacements preserve 74 diagnostic codes, 29 grammar categories and 15,000 generated fuzz inputs.
Nine affected integration tests and 28 retained PQL unit tests passed with Clippy; independent reviews identified small syntax controls that were added and passed.
Client payload replacements are complete; other mapped groups remain pending.

The scalar/SIMD sorting controls remain as independently acquired interoperability inputs (about 1 MB combined).
Their exact equal-key permutations protect sampled-row identity; ordinary sorting assertions would lose that check.
This retention was independently reviewed from source and bounded metadata, without rerunning those tests.

The dependency CSV preserves the audited baseline and adds `replacement-assertion` edges to current tests.
The inventory and ledger track removals and validation; deleted baseline paths are intentional historical entries, not current runtime dependencies.

## Outcome

Remove the majority of recorded-fixture lines and replace historical output comparisons with independent behavioral assertions.
Preserve supported behavior, recent PR fixes, Python/Java coverage, and engine boundaries.
Do not move giant expected objects into test source, compress them into archives, or introduce a new contract framework.

## Start with a dependency map

Build a small edge table, `fixture-cleanup-dependencies.csv`, before deleting fixtures.
It represents the graph below; no graph database or viewer is needed.

```text
fixture -> loader/helper -> exact test -> CI selection
   +----> generator, consumer registry, or compatibility-ledger entry
```

Each edge records source, destination, relation, evidence location, and confidence (`confirmed` or `unresolved`).
Expand fixture directories into individual files and identify test functions/methods, not just test modules.
Include Rust `include_str!`/`include_bytes!`, Python file reads and parametrization, Java resources, glob/directory loaders, and generated inputs.
Seed this map from `contract/fixtures/maintained/consumers.json`, existing compatibility mappings, and targeted searches.
Verify those declarations against code; they are not assumed complete.
Missing search hits do not prove that a fixture is unused, especially with dynamic loaders.
Resolve uncertain consumers before deleting their inputs.

Use file sizes, line counts, JSON keys/counts and bounded snippets to investigate.
Do not load entire large fixtures into agent context.

## Track every change

The [cleanup ledger](fixture-cleanup-ledger.csv) starts with the audited groups below.
Expand or split rows during mapping so every changed fixture, test, loader and CI selection belongs to a row.
The dependency table supplies exact file/test detail without making the ledger unreadable.

Each row records scope, action (`delete`, `replace`, `retain`, or a stated combination), behavior protected, replacement tests, related plumbing, baseline/remaining bytes and lines, review, and validation evidence.
Statuses advance through `scoped`, `mapped`, `implemented`, and `verified`.
`mapped` records identified consumers and a proposed group-level action, not deletion readiness; `verified` requires review and actual test results, not just a test name or passing collection.
Before changing each file, record its exact replacement assertions or retention decision, including independently expected date/time/type and boundary results.
Broad groups must be split or annotated as implementation reaches them.
Deletion without replacement needs a reason, such as duplicate coverage or a fixture-format-only test.
Record test removals, additions and retained exceptions explicitly.
Update the ledger in the same change as the affected tests.

These are lightweight working records for this cleanup, not a second permanent capability ledger.
Continue using the existing consumer registry and compatibility manifest for maintained enforcement.

## Implementation order

| Step | Work |
| --- | --- |
| 1. Map and prioritize | Complete dependency edges and ledger decisions, largest bytes/lines first; distinguish redundant permutations from meaningful regressions. |
| 2. Replace and delete | Add small, independently expected assertions; remove the corresponding recordings and obsolete tests together. |
| 3. Simplify plumbing | Remove unused loaders/recorders and obsolete CI comparison steps; update existing manifests, generators and contributor rules. |
| 4. Review and validate | Review coverage decisions, then run affected suites and integrated checks once; rerun only for changes or failures. |

Start with view vectors (~97k lines), request recordings (4.8 MB), SQL transcripts/matrices (~3.6 MB), and client payload recordings.
API/wire snapshot replacement is a separate tracked group because it changes inventory/CI plumbing despite being smaller.
Keep only justified small interoperability inputs, versioned schema contracts, and API inventory checks.
Do not discard valid boundary coverage solely because its input is large: construct it in test code when it protects a real limit, safety property or regression.

Examples: assert batching emits 10,000 and 1 rows in order without label leakage; assert top-k from supplied probabilities; execute SQL and check rows/types rather than whole SQL text.
Expected answers must not be calculated by the production helper being tested.
SQL replacement must retain the dialect matrix input builders and independently assert request kinds, namespaces/quoting, composite and typed keys, window/order rules, bindings, expected schemas, capability/source-database/read-as probes and refusals (including unknown dialects) for all five dialects.
Local database execution supplements these checks; it cannot replace Databricks or Snowflake rendering coverage.
Preserve connector-specific representations, ordering/tie rules, exact numbers, lifecycle/ownership checks, and the fixes introduced on PR #75.

## Completion checks

- Every removed or changed file/test is accounted for; no unresolved dependency edges remain for deleted inputs.
- No orphan consumers, obsolete fixture-count checks, stale CI selections, or silently narrowed language/connector coverage.
- Preserve API inventory and compatibility enforcement when replacing recorded behavior snapshots; update generated artifacts through their generators.
- Run affected Rust/Python/Java tests, local connector execution tests, lint/type checks and retained schema/coverage gates after integration.
  Run live checks only where affected behavior or maintained requirements call for them.
- Report before/after fixture bytes and lines separately from test code and production code, plus test results and justified retained fixtures.
  Do not count minification or moving data between files as cleanup.

The prediction-result recording is removed after 150 tests passed and independent review.
Replacement assertions check full frame layout and dtypes, typed values, response alignment, output options and refusal behavior against authored inputs and scripted HTTP responses.

The shared payload-rules recording is removed after 46 Python and three Rust tests passed, with Ruff, Clippy and independent review.
Direct checks preserve source-derived classification labels and related-instance membership, class ordering, leakage, payload byte budget, inference transport and exact decimal features.

The shared response recording is removed after independent review: two Rust reader tests, 24 Python binding tests and all 57 generated public boundary cases passed.
Assertions retain typed parsing, exact numeric identity, frame layout, endpoint versus local error provenance, and depth/width handling without stored response bodies or historical outputs.

The engine label recording and its comparator-only self-test are removed after independent review and 29 direct/retained Rust tests passed with Clippy.
Calendar arithmetic, aggregation masks/windows, typed comparisons and numerical precision remain covered; four external Arrow conversion bit sentinels remain as small explicit assertions.

The YAML value recording is removed after all five Rust consumers were replaced and independently reviewed.
The combined suites passed 43 tests, with ten semantic tests passing after the final composite-relationship addition; Clippy passed.
Typed YAML, diagnostic rendering, semantic definitions and generated fuzz coverage remain, including negative timezone and lazy composite-relationship controls.

Native admission now retains only 229 independently acquired timezone cases, with unchanged answers and provenance verified against the original.
The duplicate test-only parser/probes are removed; actual PQL and host resolver controls passed 18 tests.
Quoted-name and tabular-order recordings are removed after 22 and six tests passed using independent CPython/pandas oracles.
Small SQL datasets and the bounded schema vocabulary inventory remain useful unchanged inputs.

The client payload group is complete: the final label recording is removed after 55 direct public DuckDB tests passed and independent review closed.
Four small SQL-input/vocabulary files remain intentionally; exact decimal, calendar, numeric type, leakage and refusal checks use independent assertions.

Two large S1 boundary cases now generate their repeated indices through the existing narrow factory mechanism in both loaders.
Their case IDs and batching/ranking assertions remain; 612 loader tests, 20 expanded factory controls, two Rust tests and 72 runtime checks passed (35 Postgres skips).
Independent review verified exact expanded-input equivalence; the two cases now occupy 4,662 bytes.
Generating the arrays removes 615,532 bytes but adds two explanatory factory lines, included in the totals above.

Warehouse-view replacement is in progress: ordinary description and refresh paths use small authored Arrow inputs, with existing refusal/override assertions preserved.
All 13 affected Rust tests pass; the historical conversion matrix and strict alias/override descriptions still depend on the recording, which remains retained pending their replacement and review.

The nested join-drop test now uses authored catalog and preview data too; all 13 Rust view tests pass (`/tmp/fixture-view-nested.log`).
Independent review is identifying the exact assertions needed to replace the remaining historical conversion matrix.

Alias and caller-override description tests now use authored inputs.
The added schema assertion checks column identity/order, time/key selection, join endpoints and measure exclusion; all 14 Rust view tests pass.
Only the historical conversion matrix still replays captured responses; file deletion remains pending its replacement and review.

Warehouse matrix replacement now includes explicit source/filter refusals, unsupported-join omission and successful nested relationship assertions; all 17 Rust view tests pass.
The recording remains until the remaining unique matrix behaviors are accounted for and reviewed.

Warehouse review found four remaining assertion areas before deletion: dtype/source facts, derived-month/raw-time selection, wildcard expansion and two sibling joins.
Typed parameter defaults and unsafe/untrusted dimension omission now have direct checks; all 19 Rust view tests pass (`/tmp/fixture-view-lexical.log`).

Wildcard/key/measure, derived-month/raw-time, and dtype/stype/source checks are implemented; all 20 Rust view tests pass (`/tmp/fixture-view-types.log`).
The remaining matrix coverage gap is a successful two-sibling-join topology, followed by replay removal and independent review.

Sibling joins are covered and the historical matrix/SQL/Arrow replay is removed from the Rust view test.
All 20 remaining tests pass without recorded responses; the capture file itself remains pending final independent review and cleanup.

`contract/view-captures.json` and its registration are removed: every reader now uses authored inputs and independent assertions.
All 20 Rust tests and Clippy pass; final review corrections distinguish month/raw timestamps and restrict key-count answers to known keys.
Independent review closed; the removed 746,640 bytes and 778 lines are included in the verified total.

Text-time recordings and unused writer are removed after independent review: eight direct Reader/inference tests, nine retained parser tests and Clippy passed.
The tests cover format/epoch arithmetic, ordering, timezone normalization, calendar/pivot limits, warning formats and actual schema inference including the 100-row boundary.

Snowflake public schema comparisons now use independent key/type/column/relationship assertions; 38 local connector tests and Ruff pass.
Public payload comparisons and Python/Rust input replays remain, so neither Snowflake JSON nor its Arrow files are deleted yet.

Snowflake public snapshot is removed after source-derived label/history and exact wire-schema assertions replaced full payload equality.
Review added users-instance uniqueness and context pair cardinality checks; final rerun/review closure are tracked before counting the 560,348-byte reduction in the verified total.
The separate Snowflake SQL-response recording and Arrow input directory still have active Python/Rust readers.

Snowflake public review is closed and its removal is included in the verified total.
Rust catalog planning checks now assert namespace, case-preserved identifiers and exact request kinds directly; the rest of the Rust replay suite is still being replaced.

Snowflake Rust tests now assert catalog, SQL and bound-key behavior directly; all 30 tests and Clippy pass.
The Python suite still uses 14 replay input cases, with independent schema/payload expectations; all 38 tests pass.
Removed 13 unreachable replay cases and 72 orphan Arrow streams after independent consumer review, preserving all four Rust connector samples.
This removes another 854,845 bytes and 1,023 text lines; replacing the remaining Python replay provider is still pending.

Python Snowflake key reads now derive response rows from source previews with explicit seed/window/order rules; 32 sampled-response recordings and 18 orphan Arrow streams are removed (260,311 bytes).
All 38 Python tests pass with query-shape guards, including independent public label/history assertions and the missing-anchor journey.
Catalog/preview recordings remain (41,036-byte index and 69 Arrow streams including Rust inputs); those replacements are still pending.

Snowflake table listings now use authored names, counts and flags, retaining view exclusion and hybrid-evidence tests.
All 38 tests pass after removing 28 listing responses and 26 Arrow files (98,096 bytes); independent review confirmed retained coverage.

Snowflake session identity and table discovery now use authored values and exact query/parameter assertions; all 38 Python tests pass after removing those responses and three Arrow files.

The empty/null snapshot has no behavioral consumer and is removed after independent audit.
Code-built shared helpers and current Rust case execution remain; 21 affected Python tests and the Rust empty/null suite pass, as does Ruff.
Native-binding snapshots still have active Java consumers and remain pending replacement.

Java native-binding output recording and its unused writer/mocked writer tests are removed.
Direct assertions retain all maintained validation inputs, Unicode/text transport, explicit diagnostics/severity, 23 errors, warning ownership and actual large warning attachments.
All 84 binding and five panic-recovery tests pass, including review fixes for returned Unicode text and packed-date error rules.
Three packaging tests pass using clean JDK 17/21 processes, including Java 17 bytecode and the missing-native error; independent review findings are addressed.

The host2-zone recording was behavior-orphaned and is removed after independent audit; 14 generic case/harness checks pass.
SQLite key-pair comparisons are replaced in code with independent type rules, complete shop histories/labels and exact result cells; strengthened checks and independent review are running before deleting its recording.

SQLite key-pair output archive is removed; all 82 source-built inputs remain.
Replacement asserts schema storage families, exact joins/results, all labels/history, task/wire schema and metadata counts; all three tests covering 82 inputs pass after deletion, including review fixes; Ruff passes.
The reviewed wire foreign-key dtype is the target primary dtype, distinct from the separately checked public source declaration.

SQLite source-edge snapshots are removed; nine source-native scenarios now assert values, labels/history, schema/wire differences, scratch-table safety and source refresh/cache counts directly.
All nine tests pass after deletion; independent review confirmed source/wire/value and refresh-count coverage.
The composite-key scenario retains its existing generic INVALID_REQUEST behavior, without claiming a specific duplicate-key diagnostic.

Forecast Python identity tests no longer load the output recording merely to check a question/count inventory; their real empty-step, ordering and tampered-ID assertions remain.
The Rust forecast suite still consumes the recording and is being audited for direct replacement.

Timestamp storage assertions now check exact physical units, signed infinity buffers, five finite imported values, conversion refusals and duration/date representations without recorded expectations.
The targeted test and Ruff pass; public-path timestamp comparisons still need replacement before deleting host2-vectors.json.
Forecast resource-boundary inputs now come from the retained regression source case; the bounded-refusal Rust test passes.

Oversized-buffer tests now construct their four inputs directly and assert supported outcomes without historical records.
The recording-only giant-buffer test is removed; all four optional cases collect, and the Rust widening suite passes.
The approximately 7 GB allocations and optional live endpoint were not run.

Timestamp public tests no longer read historical outcomes: all 63 source scenarios assert type/inference and failure-stage rules directly.
All 110 tests pass in 15.10 seconds, and Ruff passes.
Independent coverage review is pending before deleting the recording.

Forecast observation coverage now executes the Rust host and independently asserts three actual calls, last-call capture, result column types and every entity/step/prediction/answer-index cell.
This replaces a test that only compared two archived observations; the new targeted Rust test passes.
Remaining forecast digest consumers still require replacement before removing forecast-vectors.json.

Timestamp review closed after restoring explicit time-inference warning-code assertions.
All 110 tests and Ruff pass; host2-vectors.json is retired (152,240 bytes and 7,752 lines).
Its four optional giant cases remain source-built and were not executed; 13 scaled-capacity Rust tests passed.

Inventory totals after timestamp retirement: 25,550,985 bytes and 319,084 fixture lines removed or reduced.

The forecast observation comparator and archived inventory self-check are removed.
Current execution retains observation-schema validation, exact result grid and last-call capture; the six-test suite passed before removing its inventory-only test.
Success/refusal payload digest comparisons remain the next replacement.

Forecast five refusal cases now use authored queries and explicit INVALID_REQUEST assertions: zero/excess steps, outside-data window, boolean and static targets. Targeted Rust test passed (/tmp/fixture-forecast-refusals-direct.log). Corrected test input to set both public and typed core anchor; no product changes. Does not claim endpoint-call instrumentation.

Three-step seven-day SUM now directly verifies source-query windows, task target dtype, complete training entity/anchor grid, every label by arithmetic over80 known shop orders, prediction entities/anchors and exact result grid. Targeted Rust test passed /tmp/fixture-forecast-labels-direct.log; DuckDB only converts calendar strings for expected arithmetic.

Direct SUM assertions now cover four scenarios: three-step7-day, single-step30-day, offset7..14-day and early-anchor3-day forecasts. All exact labels/grid/window/result checks pass in3.17s; /tmp/fixture-forecast-sums-direct.log. Boundary expectations use (start,end], consistent with source temporal contract; fixed test expectation, no production changes.

Direct COUNT assertions now cover four14-day steps and three filtered30-day steps, calculating every count from source order identities/timestamps/prices, retaining empty-window rows and full forecast result identity. Targeted test passed /tmp/fixture-forecast-count-direct.log; Clippy -D warnings passed /tmp/fixture-forecast-direct-clippy.log. Updated stale module prose.

Entity-filter and ASSUMING-input direct cases added, preserving exact labels/grid/results. Important existing limitation verified in ForecastPlan::step: ASSUMING is accepted but omitted from per-step queries; tests explicitly assert existing behavior, not counterfactual execution. No product change. Test case runner now uses unique per-invocation temporary paths to avoid concurrent-test collisions.

Direct AVG/weeks, MIN/months and MAX/hours tests pass against an independently authored80-order aggregate and complete training-grid oracle; /tmp/fixture-forecast-aggregates-direct.log. Monthly sampler stride uses epoch-window lengths31/28 days, while label boundaries use calendar months, verified against documented timeframe contract. No production change.

Removed all forecast recording readers, hashing/canonicalization and replay-only tests. Six direct Rust tests passed together in10.04s (/tmp/fixture-forecast-no-replay.log). Clippy -D warnings passed. Exact result-column/grid helper additionally applied to AVG/MIN/MAX after suite run; requires next validation. Independent review pending before deleting recording; no live path references remain.

Added direct full related-table histories for every forecast call: user fields, exact order prices/timestamps/linkage/descending order, no future leakage, complete per-instance coverage and key/relationship shapes. Six Rust tests passed in11.49s (/tmp/fixture-forecast-history.log). Subsequently added schema dtypes, metadata counts and nonempty request-id type; Clippy passed, runtime verification pending. Independent reviewer independently identified this payload coverage gap; follow-up review underway.

Forecast recording retirement: Replaced by authored forecast queries with exact windows/labels/full entity-anchor grids/results; independent source SQL for AVG/MIN/MAX; complete related histories and no future leakage, schema/key/relationship types, metadata/request-id type, five named refusals and bounded resource tests. Six Rust tests passed /tmp/fixture-forecast-reviewed.log. Independent review requested feature/schema gap now addressed. ASSUMING remains accepted-but-stripped existing behavior; no false counterfactual claim. Rust IDs are scripted; real identity checks remain Python.
Removed 188,228 bytes and 9,438 lines; the recording has no remaining code readers.

Inventory totals after forecast retirement: 25,739,213 bytes and 328,522 fixture lines removed or reduced.

Affected Python forecast suites: 32 passed, 6 skipped; /tmp/fixture-forecast-python-final.log.
Final forecast Clippy with warnings denied passes.

Started metric-view replacement with authored source definition asserting typed parameter substitution/filter, parameter note value, dimensions, join path/source/keys/cardinality and measure retention; targeted Rust test passed /tmp/fixture-metric-direct.log. Existing183106-byte recording and replay tests remain pending full category audit/replacement.

Added passing direct nested metric-join test: exact parent/path/table/join keys, unsupported cardinality with descendant removal, composite USING refusal, duplicate alias and invalid/nameless entry messages as typed variants. /tmp/fixture-metric-joins-direct.log. Metric recording still retained pending remaining categories.

Five authored metric integration tests pass: parameter/filter/join dimensions, nested skip paths, legacy fields fallback and dimensions precedence, exact measure input/aggregation metadata with window-order deduplication, typed mapping/source refusals and unsafe terminal filter refusal. /tmp/fixture-metric-sections-direct.log. Recording retained pending parameter/remaining coverage review.

Added explicit described-type override and missing-default/type checks; removed two metric recorded-output replay tests and their helpers. Seven direct Rust tests passed /tmp/fixture-metric-no-replay.log. Recording retained pending final coverage audit, particularly descendant dimensions/wildcards and actual source-column parameter shadowing orchestration.

Added passing wildcard exclusions/skipped descendant dimension assertion and real orchestration source-column shadowing test (PRICE default999 must not replace physical price). Logs /tmp/fixture-metric-wildcard.log and /tmp/fixture-metric-shadow.log. Final independent metric review requested; recording retained.

Combined view tests passed29(21routine+8metric). Added ninth metric test asserting joined-table/missing-default terminal filter refusal context and correct remediation hatch; targeted pass /tmp/fixture-metric-refusal-context.log. Review pending; recording remains. Combined evidence /tmp/fixture-metric-combined.log.

Metric independent review found user-warning/judged-join/refusal composition gaps. Added exact omission warning sentences, invalid source/ON skip continuation, missing-default filter sentence and unsafe-filter context. All11 metric tests pass /tmp/fixture-metric-review-fixes.log. Final review confirmation pending before deletion.

Metric recording retirement removes 183,106 bytes and 709 lines.
Replaced with11 direct metric definition tests and actual source-column parameter shadowing in view_captures. Preserves nested paths/skips/descendant dimensions, fields fallback, wildcard exclusions, typed defaults/override/refusal, measure metadata/window orders, user omission warnings and terminal filter diagnostics. Reviewer bounded findings addressed; /tmp/fixture-metric-direct-review.md and /tmp/fixture-metric-review-fixes.log. Clippy -D warnings passed. Lower-layer scalar/dialect/wording coverage retained.

Post-deletion metric validation: 11 metric and 21 view-routine tests pass.
Inventory totals: 25,922,319 bytes and 329,231 fixture lines removed or reduced.

Databricks public schema/source suite now builds both shop preview tables directly with explicit source values/Arrow types instead of decoding captured preview bytes. All70 tests and Ruff pass /tmp/fixture-databricks-preview-direct.log. Recording still supplies other response inputs and other consumers; nothing deleted yet. Independent full capture audit underway.

Databricks base shop catalog response rows now authored: ordered table/column metadata with exact physical declarations/nullability, typed empty key/FK replies retaining duplicate legacy header input and current alias validation. All70 source suite tests and Ruff pass /tmp/fixture-databricks-catalog-direct.log. SQL index/sampling responses and other readers still use capture; retirement not complete.

Databricks SQL/Spark adapter module detached from capture: small authored mixed Arrow/null/Unicode, catalog, repeated lookup and typed-empty inputs preserve exact pass-through/binding/order/ownership/error checks. All11 adapter tests and Ruff pass /tmp/fixture-databricks-adapters-direct.log. Public source and Rust catalog readers remain.

Rust Databricks catalog reader detached from capture: five direct tests cover19 explicit declarations, ordered names/nullability, informational single-column keys/composite omission, empty and qualified FK roles, listings and metric-view refusal. Five tests pass /tmp/fixture-databricks-rust-catalog.log. Source suite remains final capture reader; independent final review needed.

Final Python Databricks source reader detached: retained14 exact protocol SQL input spellings with independently constructed catalog/preview/count/extrema/entity rows; removed three seeded response arrays, using existing real local query execution for completing materializations. All70 tests and Ruff pass /tmp/fixture-databricks-source-detached.log. All3readers detached; independent final review underway before deleting capture.

Databricks capture retirement: 370,313 bytes and 5,370 lines removed.
All3readers replaced: authored public-source shop/catalog/count/bounds responses with exactSQL inputs, tiny adapter Arrow/null/Unicode/identity controls, direct Rust catalog declaration/key/FK/list/refusal assertions. Source lifetime/FK alias/PR75 regression tests retained. Independent review /tmp/fixture-databricks-direct-review.md found no blocking gaps. Rust declarations include all20previous spellings plus3controls. No standaloneArrow inputs needed.

After Databricks deletion: 81 Python tests, 5 Rust catalog tests and 4 Rust plan tests pass; Ruff passes.
Removed stale maintained-consumer entry.
Inventory totals: 26,292,632 bytes and 334,601 fixture lines removed or reduced.

Regression recording replacement started: existing explicit result/cardinality checks were reading the recording itself.
They now check actual prediction rows and materialized context/prediction/history counts; the targeted Rust host test passes (/tmp/fixture-regression-actual.log).
The 212,205-byte recording remains until its fetch, schema, state and full payload coverage has independent replacement assertions.

Regression replacement now shares source-derived shop history assertions with forecasts: exact related values/order, keys/types/relationships, instance uniqueness and no future leakage.
Description fields and inference provenance are asserted independently; the full snapshot comparison for description has been removed.
Targeted regression passes (/tmp/fixture-regression-schema-final.log), and all six forecast tests pass after helper extraction (/tmp/fixture-forecast-shared.log).
Independent review requested; the recording remains for remaining state/fetch/payload comparisons.
An earlier full DuckDB test invocation is still running the payload corpus (session 59115); its regression failure predates corrected test field spellings and age type, so it is not a clean suite result.

Regression prediction snapshot comparison replaced: complete 112-example grid and source-arithmetic SUM labels, prediction entities/date, result column schema, task/query and explicit materialize-to-wire transformations now pass (/tmp/fixture-regression-independent-body.log).
Output gains median at prediction; internal inference_config is removed from the wire while fast mode remains.
The earlier full DuckDB run finished with 19 passing tests and the already-corrected schema expectation failure; no product failure found.
Recording still protects remaining materialized/fetch/input comparisons pending replacement and independent review.

Regression recording retired: 212,205 bytes and 75 encoded text lines removed after independent coverage review closed.
Direct assertions preserve description/provenance, typed operation inputs, selected fetch wiring, complete grid and arithmetic labels, related histories, task/options, state handoff and result identity.
Targeted test and Clippy pass; no maintained consumer registry entry was present.
Inventory totals: 26,504,837 bytes and 334,676 fixture lines removed or reduced.

Type-set cleanup started: unused historical reference answers, migration notes and Arrow-setting recording removed (112,846 to 63,057 bytes).
All 110 vectors and 30 reached inputs plus active assertions remain; this is not yet full replacement of the group.
Five host type tests pass (/tmp/fixture-type-set-pruned.log); testkit consumers are running in session 1616, and independent bounded review is pending.

Type-set pruning validation completed: five host tests plus both driver/type-reach testkit tests pass, preserving every selected case (/tmp/fixture-type-consumers-pruned.log).

Type-set review completed: retain the authored 140-case admission/inference specification, reduced from 112,846 to 52,518 bytes.
Removed unused provenance/recorded answers, unused reached metadata and redundant display-spelling comparisons; preserved all 41 Arrow variants, parameter boundaries and wide/after-preview SQL witnesses.
The refusal diagnostic test now checks actual engine results instead of validating only fixture expectations.
Five host tests pass; final testkit checks running in session 91457 after the additional field removal.
Independent review: /tmp/fixture-type-set-review.md.

Final type-set consumers pass: both driver reads and type reach, in addition to five host tests (/tmp/fixture-type-consumers-final.log).

Ranking replacement started: three independent tests calculate ranking order, stable ties and top_k from authored scripted endpoint scores; all pass (/tmp/fixture-ranking-direct.log).
Recording retained for remaining refusal, response-field, unusual candidate and connector assertions; bounded independent audit requested.

Ranking direct error coverage added: unscored/duplicate candidates avoid extra calls; NaN scores stop at the offending candidate; endpoint failures identify candidate; transport exceptions preserve completed part count on DuckDB/SQLite.
Nine tests pass, two PostgreSQL tests skip without configured DSN (/tmp/fixture-ranking-errors.log); Ruff passes.
Recording remains until remaining response-field/candidate/control checks are replaced.

Ranking response/type replacement: authored carried scores/rankings (including retained NaN member), scalar prediction types, mean/numeric statistic forwarding and exact typed candidate rankings on DuckDB/SQLite pass.
Twenty-one tests pass, five PostgreSQL checks skipped without DSN; /tmp/fixture-ranking-fields-final.log, Ruff passes.
Expected binary class 1 converts to True; candidate outputs obey the query top-2 cutoff.
Remaining ranking work includes full score associations, additional refusal/control checks and removing replay-only plumbing after review.

Ranking refusal/control checks now execute current client calls rather than inspect recorded control outcomes.
All ten invalid/changed-argument control pairs pass; fixed their helper to consume expanded authored cases so rank.batched actually tests 10,001 entities (/tmp/fixture-ranking-controls-final.log).
Eight explicit stage/code/call-boundary refusal cases also pass (/tmp/fixture-ranking-refusals.log); Ruff passes.

Ranking recording readers/replay-only helpers removed; source scenarios remain.
Nine output/score/identity/HTTP status checks pass (/tmp/fixture-ranking-output.log); full ranking suite running session 69290, final independent review requested.
Fixture retained until those checks complete.

Ranking recording retired: 99,646 bytes and 299 lines removed after independent review closed.
Full detached suite: 76 passed, 21 PostgreSQL skips; final review additions: 24 passed, 5 PostgreSQL skips.
Preserves independently scored results, typed connector roundtrips, output/type contracts, candidate omissions/deduplication, errors/partial progress, and current refusal/control pairs.
No code readers or consumer registry entry remain; Ruff passes.

Named diagnostics now derive outcomes from authored cases and required semantic details, not captured outcomes.
Snapshot-conditioned case selection removed: explicit SQLite decimal materialization-type refusals and now-successful wide-key path are checked (4 connector tests pass); initial expanded run passed88 with46Postgres skips before those four expectation corrections.
Logs /tmp/fixture-wording-direct.log and /tmp/fixture-wording-connectors-final.log; warning comparisons and record-only controls still pending, recording retained.

Diagnostic fixture fully detached: warning class/count/date bounds/index counts/remedy/overflow assertions replace recorded sentences; 16pass8Postgres skips (/tmp/fixture-wording-warnings.log).
Removed recording-only controls and loader, retained selector completeness assertions.
Full suite session24240 and final independent review pending before deletion; Ruff passes.

Diagnostic recording retired: 96,991 bytes and 2,239 lines removed.
Direct suite113pass54PostgreSQL skips, independent review closed, Ruff passes; no registry entry remains.
PostgreSQL execution requires a configured DSN and is not claimed passed.

Label recording replacement started: direct Rust test checks complete temporal grids and arithmetic regression/binary/decimal labels from source values; passes /tmp/fixture-labels-direct.log.
Static multiclass assertions and replay-checker retirement remain before removing label-expectations.json.

Static multiclass Rust source checks pass for all eight entities and the source-derived latest anchor (/tmp/fixture-labels-static.log).
Added matching independent Python label arithmetic/type/grid assertions for all four cases; final run /tmp/fixture-labels-python-direct-final.log.
Label fixture has both Python and Rust readers; recording stays until both old comparison helpers are removed and reviewed.

Label readers detached in both languages; obsolete comparison checker/mutation tests removed, direct source assertions and Python decimal boundary checks retained.
Python9pass Rust2pass; independent final ordering/identity review pending before recording deletion.
Logs /tmp/fixture-labels-python-detached.log and /tmp/fixture-labels-rust-detached.log.

Host label recording retired: 111,176 bytes and 3,982 lines removed with its registry entry.
Direct source arithmetic, full grids, newest-first complete entity blocks and typed static labels preserve both language routes; independent review closed.
Python9Rust2pass, Ruff/Clippy pass; /tmp/fixture-host-labels-review.md and final label logs.

Text-normalization recording replacement started: five direct behavior tests cover whitespace/punctuation/br parsing/Unicode casing and length ordering; full independent Unicode mapping sweep retained.
Initial5pass; final boundary expansion review pending before removing772 recorded normalized values.

Words corpus reduced109,412 to48,627bytes: removed772 normalized-output recordings, kept independent Unicode mapping/provenance and full scalar sweep.
Five direct tests cover punctuation,29whitespace points, malformed/valid HTML breaks, character299/300/301 and distinct word49/50/51 limits; reviewed correction proves first50 order.
Postprune validation /tmp/fixture-words-pruned.log and Clippy /tmp/fixture-words-clippy.log.

Temporal-range corpus reduced63,036 to51,529bytes by removing unused pandas comparison outputs; generator now needs only stdlib zoneinfo.
Regenerated and verified all252 active boundary expectations unchanged; both Rust tests pass.
Retain independent zoneinfo versus Rust oracle for18vendored zones, rather than duplicating timezone logic in handwritten expectations; independent review requested.

Option recording replacement started: Python per-item metadata assertions derive expected call/local-row values from authored batching, verify exact order and global row identity; absent explanation checks no longer consult recording.
Five selected result-observation tests pass (/tmp/fixture-option-metadata.log); other option/explanation readers remain.
Timezone retention review closed /tmp/fixture-time-ranges-review.md.

Result explanation readers detached in Python and Rust: two authored endpoint answers preserve nested explanation/forecast-step wrapping, per-item metadata exact order and global row indices.
Python8pass (/tmp/fixture-explanation-python.log); Rust test session16543 rebuilding affected crates (/tmp/fixture-explanation-rust.log).
Option fixture remains for other consumers.

### Option recording retirement

Replace `contract/option-vectors.json` with authored outcomes and exact call counts across the existing connector cases.
Preserve query/output routing, requested row identity, columns/count/types, score-derived ranking, forecast step order, raw explanations and global/local metadata identities.
Assert warning classes/counts, filter selection scope and remedy, missing-explanation effect, and refusal details.
Independent review: `/tmp/fixture-options-review.md`; warning semantic feedback incorporated.
Pre-deletion Python query-option suite: 255 passed, 121 PostgreSQL skips (`/tmp/fixture-options-direct.log`).
Deleted the 82,953-byte / 4,150-line recording after independent review.
Affected Python suites: 263 passed, 121 PostgreSQL skips (`/tmp/fixture-options-deleted.log`); Rust explanation suite: 3 passed (`/tmp/fixture-options-rust-deleted.log`); Ruff passes.

### Batching recording retirement

Replace `contract/prediction-batch-vectors.json` with authored call counts, refusal details, request-size remedies and exact scored rows.
Check every row of the 10,001-row result, entity-major forecast results, materialize-only success, and partial answer IDs/order/typed scores.
HTTP failures include the submitted failing call; transport exceptions occur before submission.
Preserve existing native-routing, context/related-row identity and training-leakage checks.
Independent coverage review: `/tmp/fixture-batching-review.md`.
Focused replacement suite passed: 46 tests, 23 PostgreSQL skips (`/tmp/fixture-batching-direct.log`).
Deleted the 59,866-byte recording after review; full affected suite: 102 passed, 44 PostgreSQL skips (`/tmp/fixture-batching-deleted.log`); Ruff passed.

### Column-name diagnostic recording retirement

Replace `contract/column-name-regression-vectors.json` with execution of all authored scenarios, including ordinary and unusual names.
Assert refusal stage/class/code and the relevant query, misspelled column, task override or forecast call limit plus actionable remedy.
Local DuckDB/SQLite: 30 passed; 15 PostgreSQL cases skipped (`/tmp/fixture-column-diagnostics-direct.log`).
Independent review closed `/tmp/fixture-column-diagnostics-review.md`; deleted the 47,145-byte recording.

### Large query attachment input

Construct the 12,000 repeated literals at runtime instead of storing them in `s0.boundary.start-past-64kb.json`.
Python and Rust loader checks preserve the original query hash and 72,082-byte length; Java's direct loader also expands the case.
The public native Python route still materializes successfully (`/tmp/fixture-long-query-current.jsonl`).
Python focused loader check passed; Rust loader checks: 4 passed.
Initial full Python loader run had 624 passes and one schema-placeholder failure; fixed with a nonempty placeholder and verified by the focused test.
Java BindingTest: 84 passed (`/tmp/fixture-query-factory-java.log`); independent review closed (`/tmp/fixture-query-factory-review.md`), including the Java loader fix.

### Random streams and PQL journeys

Retain `rs1.json`, `rs1-draws.json` and `legacy_network64_controls.json` as independent NumPy interoperability oracles, following `/tmp/fixture-random-network-review.md`.
They protect exact streams, entropy boundaries, cached halves, bounded draws and sorting tie/sentinel behavior.
Replace PQL journey recordings with authored schema variants and explicit query acceptance plus unknown-column error/span assertions.
The 34 recorded cases reduce to 11 schema shapes; duplicate data-level cases provide no additional coverage in this schema-only consumer.
PQL replacement: four tests and strict Clippy pass; independent review closed (`/tmp/fixture-pql-journey-direct.log`, `/tmp/fixture-pql-journey-review.md`).
Deleted the 48,087-byte / 2,575-line journey recording.

### Protected inventory reconciliation

Marked five protocol schema/lock files, thirteen compatibility input/inventory files and the three explicitly unresolved inputs as retained, rather than pending replacement.
Verified their files exist and checked schema-lock/state-payload consumers and compatibility fixture/ledger consumers.
This is a retention decision, not a claim of new runtime validation.
The unseeded sampling recording remains pending replacement: derive source membership from authored shop inputs, retain schema/row-count/relationship checks, and check both fresh draws rather than only the first.

### Unseeded source oracle replacement

Unseeded tests now obtain source tuples from authored SQL inputs and compute SUM labels independently over `(anchor, anchor + 30 days]`.
Both random draws receive the same full checks, rather than checking only the second draw's outcome.
Freshness checks ordered examples, not different membership; this is a nondeterminism smoke check.
Deleted the redundant recorded source-data section after six tests passed (two PostgreSQL skips); schema/result expectations remain pending direct replacement.
Validation: `/tmp/fixture-unseeded-source.log`; review: `/tmp/fixture-unseeded-review.md`.

### Unseeded recording retirement

Replace remaining schema/result recordings with authored type/semantic/nullability/relationship checks, exact prediction identity/counts and independently expected scripted result values.
Check the complete 112 entity/anchor pairs, independent labels and source membership on both draws.
Also enforce unique per-instance keys, same-instance user/order/line links and no future orders.
Independent review `/tmp/fixture-unseeded-review.md` confirms preserved coverage; row-link strengthening incorporated.
Direct and row-key suites each passed six tests with two PostgreSQL skips; final post-deletion run is `/tmp/fixture-unseeded-deleted.log`.

### Remaining authored-case audit

The 391 pending case files total about 1.1 MB and primarily hold test inputs, small expected outcomes and routing metadata, rather than captured results.
A metadata-only scan found roughly 232 KB of route lists, 184 KB of core inputs and 175 KB of public inputs before pretty-print whitespace.
Do not classify all of these as unfinished output-snapshot replacements; audit input duplication separately and leave broad routing-tool simplification to step 3.
The PostgreSQL regression recording has no discovered runtime loader; only unused recording constants remained in the shared scenario helper.
Removed those constants while retaining all authored scenarios/helpers; eight case-input validation tests passed (`/tmp/fixture-postgres-inputs.log`).
Independent review closed (`/tmp/fixture-postgres-orphan-review.md`); deleted the 41,851-byte orphan recording.
No behavioral reader was removed; this does not establish fresh live coverage of all 21 historical scenarios.

### Remaining corpus classification

`malformed-cases.json` contains authored field mutations and accepted/refused input rules shared by both loaders, not captured API output.
`request-kinds.json` remains a genuine replacement task: nine request kinds over three source datasets, with keyed variants, and recorded Arrow schema/rows consumed by Rust provider tests.
`validator-options.json` also remains a replacement task: seven queries across two boolean options and two modes (56 combinations); training differs from inference in 18 cases, so preserve both mode semantics and ordered diagnostic spans.
Independent case-inventory review closed `/tmp/fixture-case-retention-review.md`: retain authored cross-host inputs and small explicit expected outcomes; keep identified repeated boundary-array cases pending factory reduction.

### Remaining boundary index arrays

Use the existing Python/Rust input factories for 100/101-call cases and 500/501-entity ranking-product cases.
Asserted the original public and typed core arrays exactly matched `range(count)` before replacing them; candidate lists and expected outcomes remain unchanged.
Python loader checks: 40 passed (`/tmp/fixture-small-factories-final.log`); Rust loader checks: four passed; independent review closed (`/tmp/fixture-small-factories-rust.log`, `/tmp/fixture-small-factories-review.md`).
Removed 42,264 bytes of repeated input arrays.
Both batching and ranking consumers use the canonical expanded loader; reviewing for any bypassing consumer.

### Validator-option recording retirement

Replace `rust/crates/sdm-pql/tests/validator-options.json` with authored schema input and seven queries across both boolean options and both modes.
Assert acceptance and ordered diagnostic codes/spans for all 56 combinations, including training problem-type requirements, timestamp restrictions and comparison errors.
The same four tables and relationship groups are retained; no validator output is used to compute expected answers.
Test and strict Clippy pass; independent review closed: `/tmp/fixture-validator-options-direct.log`, `/tmp/fixture-validator-options-clippy.log`, `/tmp/fixture-validator-options-review.md`.
Deleted the 26,026-byte recording after confirming all 56 expected tuples and 18 training distinctions are preserved.

### Host and wire inventory follow-up

Review pending timezone and host-interoperability inputs as groups, preserving serialized compatibility data and genuine driver encodings.
The generated `tests/golden/wire_payloads.json` is still pending replacement, not automatically retained: nine canonical journeys feed both snapshot equality and schema-validation/mutation tests.
Its replacement must preserve actual transport capture and invalid-wire rejection; update its generator/consumers rather than manually editing generated JSON.
Independent input-retention review: `/tmp/fixture-host-retention-review.md`.

Host retention review closed: retain all 19 timezone inputs and 33 host input/specification files.
Exception: `tests/fixtures/saved-requests/expected.json` is an active 42 KB product-output snapshot, so move it to replacement work.
Keep all ten historical pickle inputs; replace expected answers while preserving caller-payload precedence and both saved-request generations.
Evidence: `/tmp/fixture-host-retention-review.md`.

### Saved-request result assertions

Added independent scripted expected predictions, probabilities, ranking scores, metadata, explanations and warnings across both historical versions and missing-native-state restoration.
Inventory now checks actual pickle protocol headers and authored request names rather than a recorded claim.
All 17 saved-request tests pass (`/tmp/fixture-saved-request-direct-items.log`); lint passes.
Keep the expected-output recording until transport body continuity, DataFrame rendering and output-field comparisons are independently covered; review pending `/tmp/fixture-saved-request-review.md`.

### Saved-request snapshot retirement

All 17 saved-request tests now pass without reading expected-output JSON.
Outgoing parsed bodies are compared to deep-copied persisted input payloads with named envelope changes; payload inputs remain unchanged after prediction.
Direct frame columns/rows and scripted outputs replace frame/body digests.
Caller-truncated batches independently require four entities and two calls, both before and after clearing native state.
Final review pending `/tmp/fixture-saved-request-review.md`; passing test evidence `/tmp/fixture-saved-request-direct.log`.

Saved-request review feedback addressed: explicit item prediction types and DataFrame dtypes guard against Python numeric/bool equality masking type changes.
All 17 tests pass (`/tmp/fixture-saved-request-typed.log`); deleted the 41,999-byte expected-output snapshot, retaining all ten historical pickle inputs.

### Branch checkpoint

User authorized committing and pushing this cleanup branch.
Step 2 remains in progress; this checkpoint preserves completed replacements, retained-input decisions and the current conversion work.
The result-conversion recording remains present because `test_postgres_values.py` still reads its PostgreSQL section.
Independent review approved the presentation replacement; remaining PostgreSQL reader must be detached before full retirement.
Integrated cross-language validation remains step 4; focused test evidence is recorded per group.

### Conversion recording section cleanup

Independent review approved all twelve pandas presentation dtype expectations and retained actual driver checks.
Removed unused presentation/driver/provenance sections from `result-conversion-vectors.json`; its PostgreSQL value section is unchanged because `test_postgres_values.py` still reads it.
Local presentation validation: seven passed, four explicitly skipped for absent PostgreSQL DSN.
Recorded the separate completed retention review for all 40 Snowflake Arrow files and eight historical pickle inputs; no orphan inputs were found.
Evidence: `/tmp/fixture-result-conversion-review.md`, `/tmp/fixture-other-input-retention-review.md`.

PostgreSQL value tests now report explicit skips when the DSN is absent instead of silently returning as passes.
Combined result/value validation: `/tmp/fixture-conversion-pruned.log`; no new live PostgreSQL execution claimed.

Schema-pickle output replacement preserves the three historical serialized inputs and directly checks declarations, edits, links, detached state and round trips.
Independent review approved removing the 4,265-byte output JSON; all six affected tests passed.

Numeric-text recording removed after independent review: all 41 inputs remain covered by direct values, IEEE special-value rules and syntax refusals, with negative-zero coverage added.
Four Rust tests and Clippy passed; the independent NumPy temporal-sort and MT19937 coefficient oracles remain because their exact ordering and seeded values protect interoperability.

Wire-output snapshot replacement keeps all nine current transport journeys and checks task/output rules, typed source values, context exclusion, full relational histories/COUNT labels and schema links.
Independent NumPy permutations preserve seeded row order; pinned-schema and malformed-request tests now consume current captures, with ten intentional-corruption controls for assertion quality.
The 194,865-byte generated output recording is removed with its writer path; public API inventory remains pending, and broad tooling consolidation stays in step 3.
Post-deletion validation passed all 36 affected tests against the available pinned model schema; transport responses remain locally scripted, not live model results.

The public API inventory, case/observation schemas, source SQL bases and active four-fact connector outcome maps remain after consumer audit.
Two inactive native-runner baseline JSONL outputs (178,161 bytes) are deleted; the explicitly protected old-client baseline is unchanged.
This was a source/CI consumer audit, not another product-test run.

Request-state output recording (7,866 bytes) replaced by explicit entity/window/decomposition facts, failed-part attribution and typed first-candidate/ranking assertions.
Scores come from the independently written endpoint rule; review tightened these to exact float equality.
Shared input helpers remain for temporal and encoding tests; only the obsolete recording loader is removed.
Request-state post-deletion validation: 27 passed, one existing skip, with independent review closed.

Seven small source/grammar/registry inputs remain after dynamic-consumer audit; the unused 20,711-byte Databricks statement recording is removed.
This classification used source/CI evidence and does not claim a fresh warehouse run.

The 9,291-byte SQL golden is removed after review: tests execute plain/dotted/decoy sources and independently check labels, complete histories, newest-first order, keys and prediction order.
The current SQL rename/catalog check remains; the obsolete recording writer is removed.
