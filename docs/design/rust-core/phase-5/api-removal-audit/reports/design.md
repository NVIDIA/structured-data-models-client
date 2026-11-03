<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 2. Schema reports and explanation conveniences

**Status: design only. Support both Python and Java; pandas and notebook formatting stay Python-specific.**
Expose readable information already held by schemas/results, without another warehouse or model call.
Phase 6 boundary review: [decision and evidence](../../../phase-6/boundary-review.md).

```text
Schema fields / prediction explanation
        ├─ records and text → Python / Java reporting helpers
        └─ numeric attribution aggregation → Rust → report records
```

| Layer | Responsibility |
|---|---|
| Python / Java | Metadata/link records, explanation accessors, text printing and local display |
| Rust result-processing code | Shared feature-importance aggregation and numeric/error rules; reuse the existing result path |
| Existing schema/result descriptors | Source of truth; no inference or schema mutation during reporting |

## Proposed user API

Both languages expose schema/table metadata records, relationship records and a text rendering.
Python additionally offers pandas views and `print_metadata()` / `print_links()`; Java returns read-only records and text suitable for its own UI.
Records contain the information available now: table/column names, types, keys, time columns and relationships.
Do not invent inferred-link provenance or sanitization statistics; those have separate contracts.

Keep `PredictionResult.explanation` / `explanation()` unchanged.
Add a read-only explanation view for summary, details, warnings, cohorts, subgraphs and feature importance, with print/text conveniences.
Do not reintroduce the old return type that switched between a DataFrame and an explanation object.
Preserve unknown explanation fields and define absent/malformed fields explicitly; absence is not a zero-valued explanation.
Numerical attribution interpretation may use a small shared Rust helper because its model-specific meaning must agree across languages, not simply because it is common code.
Before adding that helper, define supported payload formats, signed/nonfinite scores, missing values and normalization.
Compute on demand when feature importance is requested; do not add another call to ordinary prediction or copy the entire prediction result for a report.

## Acceptance

Compare metadata records and attribution values across languages using the same fixtures.
Cover absent explanations, nested cohorts/subgraphs, empty attribution, repeated columns and malformed numeric values.
Test host text/pandas views separately; verify reporting does not mutate inputs or perform I/O.
No GPU tests are needed for wrappers over captured result payloads.
