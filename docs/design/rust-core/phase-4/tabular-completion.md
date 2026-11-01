<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Next: finish Tabular sessions

**Keep session information in the client, but let Rust decide how to prepare requests and interpret answers.**
We do not need a persistent Rust session object to move this logic out of Python.

## What these calls do

| Call | Meaning |
| --- | --- |
| `materialize()` | Choose example rows and rows to score. No model call. |
| `fit(request)` | Upload the example rows once. The server keeps them and returns a session ID. This does not train a model. |
| `fitted.predict(request)` | Send new rows to score using those saved examples. |
| `fitted.close()` | Ask the server to release the examples. |

Direct prediction and PQL materialization already have Rust implementations.
Session preparation and answer handling still use the old Python implementation.

## Where things live

Three different things were blurred together in the previous draft:

| What | Where it should live | Why |
| --- | --- | --- |
| Uploaded example rows | Model server | They are reused across predictions. |
| Session ID, column types, task, outputs and keys needed to detect overlap | Client's fitted object | Needed between calls; no second copy of the uploaded rows. |
| Validation, request encoding and matching answers to input rows | Rust functions | Shared correctness and behavior across Python and other languages. |

The client passes its saved session information into Rust on each call.
Rust returns results and any updated information; it need not retain either after the call finishes.
The saved key list can still grow with the number of examples, so “no retained rows” does not mean zero memory cost.

## One prediction, before and after

```text
TODAY
Python prepares request → Python sends it → Server
Python interprets reply ← Python receives it ← Server

PROPOSED
Client supplies: new rows + saved session information
                        |
                        v
             Rust checks and prepares request
                        |
                        v
             Python sends it to the server
                        |
                 server replies
                        |
                        v
             Rust checks and interprets reply
                        |
                        v
             Python returns the public result
```

Here Python is the **host**: the language layer that owns DataFrames and network connections.
Another language supplies its own host and reuses the same Rust logic.
Rust does not call Python to compare values, choose rows or make model decisions.
The network exchange remains necessary regardless of where session information is stored.

## Does this reverse the earlier state decision?

**No. Recommend preserving host-held state.**
The migration design explicitly says “typed, versioned values the host holds”; the implemented host already passes stored request facts back to Rust.
Rust also has temporary state while one operation runs, which is different from retaining sessions between public calls.
I have not located the specific historical decision attributing this choice to Phase 3 callback delays.

| Choice | Assessment |
| --- | --- |
| Client holds a defined session value; Rust consumes it | Recommended. Fits existing design, works across languages, avoids a second session registry. |
| Rust retains sessions behind handles | Adds ownership, release and synchronization work. Consider only if measuring repeated metadata transfer shows a meaningful bottleneck. |

Extend the existing “send this model request” mechanism to name create, score and delete operations.
Keep the existing network methods and retry behavior; do not add a new callback framework.
Exact message fields need a small contract review before implementation.

## What we will change

1. **Reuse Rust's Tabular encoder and answer handling for sessions.**
   Keep the public API and warehouse push-down paths unchanged.
2. **Stop building data that is immediately discarded.**
   The old session adapter encodes both examples and scoring rows, then throws away half.
   Fit should encode examples; later predictions should encode only new rows.
   Fit must still consider both frames when deciding column types.
3. **Verify the complete journey, then land.**
   Cover held-out keys, incompatible column types, empty predictions and cleanup failures, plus files loaded through the existing CSV/Parquet reader.
   Run independent reviews and consolidated old/new tests, including a real session-capable server and a Rust-only policy test.

Correctness requirements: use the schema saved at fit; exclude held-out examples; match answers to the right rows; leave failed deletes retryable; make repeated successful close harmless.
Preserve client cleanup after individual failures and explicitly check custom transports that modify requests.
Removing failures caused only by encoding discarded data is a proposed improvement, not something to imitate automatically.
Measure preparation time and memory before claiming a speedup.

This completes the remaining Tabular integration; relational sampling and default-route switching remain separate.

## Evidence

Source audit: `0320960ce6`; no product code changed for this plan.

- State ownership: [migration design](../migration-plan.md), “State and memory”; `sdm_client/_rust/results.py` module documentation.
- Existing sessions: `sdm_client/client.py`, `FittedModel` and `SDMClient.fit`.
- Discarded encoding: `sdm_client/adapters/tabular.py`, `session_plan` and `session_prediction_plan`.
- Rust reuse: `sdm-request/src/tabular.rs` and `tabular/response.rs`; existing model-call mechanism: `sdm-contract/src/run.rs`.
- Constraints: [boundaries](boundaries.md) and [restart](restart.md).

Implementation acceptance note: exact key comparison intentionally avoids rounding entity IDs through the active Decimal precision.
A frozen-client public reproduction showed a named high-precision holdout remaining in the uploaded context; the new regression requires removing that exact ID while retaining its distinct neighbor.
The implementation remains opt-in until consolidated validation and landing are complete.
