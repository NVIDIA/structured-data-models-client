<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Raw inputs and type parity

**Goal:** prove that ordinary reader output works through prediction, and cover important types absent from the shop fixture.
The existing 582 comparisons remain valid for their prepared inputs; they do not establish comprehensive type coverage.

## Two additions

```text
Same versioned rows in each supported source
  → public Python / Java reader
  → unchanged DataFrame / Arrow table
  → public schema or direct request → materialize → real model prediction

Frozen Python is the reference for each new Python and Java journey.
```

No test-only casts, boolean repair, timestamp parsing, decimal narrowing, Arrow reconstruction or row sorting between reader and prediction.
Ordinary public schema declarations (keys, relationships, explicit semantic types) are allowed and recorded identically in meaning.
Use explicit ordered queries where the public reader supports them; otherwise compare by stable identity without altering the input.
Record raw reader schema/values, inferred or declared semantic types, actual model inputs and results separately.
A reader representation difference is acceptable only if it preserves meaning through the public API; do not erase it before testing.

| Addition | Representative data | What must be checked |
|---|---|---|
| Free text | Sentences, Unicode, empty strings and longer descriptions | Actual `text` features reach the model unchanged; test inference separately from explicit declaration |
| Missing values | Null numeric, boolean, categorical and text features; an all-null feature | Missingness, omission/refusal rules and model inputs agree with the old client or a documented improvement |
| Multicategorical | String lists, empty lists, null lists and null elements | Input features, not merely generated link targets; retain list structure where supported |
| Sequence | Numeric lists with empty/null cases | Verify actual old-client/model support first; test prediction or the correct refusal |
| Numeric boundaries | Nullable narrow integers, values beyond float's exact integer range, decimals and float32/64 | No silent rounding, overflow or unintended categorical/numerical reinterpretation |
| Temporal boundaries | Dates, timezone-aware timestamps and fractional precision around a timezone transition | Preserve instants, precision and nulls; verify supported conversion or explicit refusal |

## Scope and execution

1. Inventory old-client, model and connector support before choosing expected results.
   Add a compact coverage table distinguishing source-read checks, real-model prediction, verified refusals and untested cases.
2. Add versioned common fixtures to the existing `tests/parity` harness, retaining the same logical rows and keys across languages and connectors.
   Use one ordinary scalar raw-reader journey per supported reader/input family and model, plus targeted type journeys where the source can represent that type.
   Cover memory, SQL readers, Spark, files/S3 and semantic-view reads where public APIs support them; retain source-backed Relational checks for affected provider types.
   Do not invent list support for CSV or turn unsupported nested types into strings just to make tests pass.
3. Capture the frozen Python baseline, then implement the Python/Java cases together using existing public APIs and installed artifacts.
   Reproduce any mismatch end to end before fixing product code; keep shared policy in Rust and language-specific conversion at host boundaries.
   Preserve warehouse sampling design and the single-table Tabular API.
4. Review contract/type decisions early, then run focused regressions and real-source/model comparisons after cohesive implementation.
   Test supported nullable/text inputs through fit/predict/close on session-capable transports as well as ordinary prediction.
   Reuse immutable completed evidence; rerun only affected neighbors, recording build and endpoint identities.
5. Publish the coverage/results table, fix discovered regressions, and integrate signed-off commits into `rust-core-phase-4`.
   Unsupported old behavior must have refusal evidence; a blocked or missing test stays incomplete.

## Completion and cost

Every listed family has explicit Python/Java and Relational/Tabular coverage or a verified capability limitation.
Every supported reader family has an unmodified read-to-predict journey; connector-specific datatype checks remain targeted rather than a full Cartesian expansion.
Compare both new clients directly with frozen Python, and report cross-language differences in raw representation separately from prediction parity.
Do not require identical predictions across different models or inherently different connector sampling semantics.
Keep permanent checks in the existing manifest/runners/comparator; add no separate testing framework.
Local checks need no model; live checks reuse authorized endpoints and run locally for this work.
Dedicated CI access remains deferred, and live tests do not run on every commit.
Preserve the original [parity plan](parity-completion-plan.md), [Tabular API](tabular-api-flow.md), [Java Tabular](java-tabular.md) and [Relational boundaries](relational-in-memory.md).
