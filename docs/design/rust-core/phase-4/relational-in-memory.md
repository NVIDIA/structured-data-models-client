<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# In-memory Relational: what happens where

**Implemented:** Rust prepares related tables for prediction through the existing Python API.
Database push-down sampling stays unchanged.

## The user journey

```mermaid
flowchart TD
    A["Python: tables in memory + schema + query"]
    A --> B["materialize(): Rust finds examples, labels and related rows"]
    B --> C["Prepared request: no model call yet"]
    C --> D["predict(): send examples and rows to score"]
    C --> E["fit(): upload examples once; server returns session ID"]
    E --> F["fitted.predict(): score rows using saved examples"]
    F --> G["close(): release server session"]
    D --> H["Rust matches answers to input rows; Python returns result"]
    F --> H
```

`fit()` does not train the model.
Sessions require an endpoint that supports them; this design reuses the existing session API.
For each network call, Rust prepares/checks model data and Python performs HTTP and presents the result.
This work primarily replaces the in-memory preparation behind `materialize()`; it does not rebuild warehouse sessions.

## Inside materialize()

| Previous implementation | Current implementation |
|---|---|
| Python and the legacy engine build the graph and sample related rows | Python supplies typed columns; Rust builds the graph and samples related rows |
| Sampling runs through the old engine | Rust keeps the graph and sampling state together until their owner is released |

Python keeps DataFrames and handles their conversion.
Rust owns row selection, labels, relationship traversal and sampling decisions.
It does not ask Python which neighbor to choose.
Prepared intermediate data stays in Rust instead of repeatedly moving to Python and back.

## Four rules

1. **Correct examples:** preserve row identities, relationship order, time cutoffs and held-out/target exclusions.
2. **Correct randomness:** each materializer/model keeps its own continuing random state, including after failures; concurrent access is serialized.
3. **Clear memory ownership:** define when input is snapshotted; protect it from later frame mutation without copying at every step, and release it with its owner.
4. **No warehouse changes:** reuse existing prediction machinery without modifying SQL sampling or provider fallback.

## Parity means correct behavior, not copying bugs

Preserve supported user workflows and meaningful results, not every incidental behavior of the old implementation.
When a difference appears, reproduce it through the public API and decide whether it is a regression or a confirmed old-client bug.
For a bug, document the intended behavior and add a regression check for the improved result; do not stall the migration trying to recreate it.
Faster execution and lower memory use are welcome when correctness is preserved and measurements support the claim.
Do not weaken temporal cutoffs, target exclusions or row attribution just to make comparisons pass.

## Verification

Independent architecture and compatibility reviews cover graph ownership, randomness, temporal cutoffs and sessions.
Paired live regression and temporal-link journeys matched the old client's result values, identities and order through prediction, fit, two scores and close.
The live candidate preceded the final diagnostic/advisory fixes; consolidated tests and focused checks of the final installed package are complete.
All 13 findings from the consolidated run were resolved and passed together on rerun; the separate legacy engine suite and 190 targeted Rust tests also passed.

One 60,000-row comparison measured repeat materialization at 3.7 ms versus 22.7 ms previously, but first materialization was slower (149.4 versus 137.9 ms) and whole-process peak RSS higher (150.7 versus 136.9 MiB).
This is one workload, not a general speed or memory improvement claim.

See [Java Tabular](java-tabular.md) for the parallel language implementation.
