<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 4 implementation boundaries

This narrows the [restart plan](restart.md); its behavior requirements still apply.
Scope is Kumo Tabular internals, followed by source-less frame sampling.
Public APIs, warehouse push-down planning, SQL, providers, provider fallback, sampler state and session lifecycle remain unchanged.
No universal sampler, new crate or local DuckDB replacement is needed.

```text
Public Request / PredictiveQuery / SDMClient / PredictionResult
                         |
Python native observations -> sdm-host -> sdm-core composition
                                          |
                       schema / request / engine / wire policies
                                          |
                         existing host transport -> model

Source-backed materialization -> existing Phase 3 path (unchanged)
```

## Ownership

| Package or module | Responsibility | Must not own |
| --- | --- | --- |
| Python client, `_rust/direct.py` | Public objects, actual frame/dtype owners, reached native conversions, transport and presentation. | Sampling policy, model policy or a generic callback interpreter. |
| `sdm-contract` | Closed typed operation inputs/results and attachment references. | Algorithms, Python objects or a serialized execution script. |
| `sdm-host`, bindings | Existing message/attachment transport and run lifecycle. | Model or sampling policy; bindings never bypass the host. |
| `sdm-core::direct` | Compose direct operations and existing model-call machinery. | Duplicate schema, encoding or sampling algorithms. |
| `sdm-schema` | Existing schema inference/compatibility policies with bounded direct-input additions. | Model transport or model-specific encoding. |
| `sdm-request::{direct,tabular}` | Generic request validity; Tabular binding, encoding and response attribution. | Warehouse sampling or native object ownership. |
| `sdm-wire` | Existing model wire types and response decoding. | Public pandas construction. |
| `sdm-engine::tabular` | Later: pure filtering and conditional sample-then-sort, returning source row positions. | SQL/provider selection or transport. |

## Contracts

- Construction validates a model-neutral Request in the existing error order.
  Prediction consumes current public frames/task/schema, not a stale constructor snapshot.
- Preflight precedes empty-result handling; oversized direct context is refused, never sampled.
  Encoding produces both model bytes and their exact row/key layout; response attribution uses that layout.
- Native observations occur only where the old client consumes them.
  Preserve conversion failures, owner lifetimes and mutation effects; combine pure Rust steps only when no observable work separates them.
- Use existing attachments for input-sized columns and response data, explicit tagged alternatives for direct versus gathered requests, and canonical generators for message changes.
  Reject mixed alternatives; preserve existing gathered-call behavior.
  Direct Arrow fields carry small native storage observations under the versioned `sdm.direct.storage.v1` metadata key, defined and validated in `sdm-contract`.
  Capture these facts during reached conversion, preserve them across batches, and reject malformed or incompatible metadata; warehouse consumers do not use this contract.
  Direct inference options that require native transport encoding use a closed deferred-options alternative.
  Rust constructs the logical envelope and its fixed inference slot; Python binds the retained native value at send time and calls the existing native transport.
  For this alternative, native transport owns serialization, endpoint wrapping and wrapped-size enforcement, preserving open-check/error order and custom transport behavior.
  Built-in byte calls with ordinary options and warehouse paths remain unchanged; custom transports retain their native `predict` method and decoded response owners.
  Response attribution still belongs to Rust.
  Native transports retain the mutable payload object; after response parsing, observe only the layout fields the reference consumes from that object, including transport mutations.
  Keep presentation identities and actual key owners in Python; unconsumed response text never crosses a UTF-8-only boundary.
  Closed response-field observations preserve native parse order without JSON-serializing opaque result values.
  Rust selects target restoration and probability-key collision policy; Python performs the reached `as_declared` casts, native equality/hash operations and presentation replacements.
  Requested-output presence uses reached native comparisons, with Rust selecting the fields and missing-output notice.
- Later PQL selection preserves scoring-key expansion/exclusion and known-target filtering.
  If reduction is needed, sample with fresh seed 42, then sort the sampled index labels; otherwise do not read/sort the index.
  Infer task from selected context, then use the same generic Request construction.
- Later relational work extends existing frame graph/task/reference owners to plans/contexts where justified.
  Preserve adjacency order, continuing RNG and failure progression; do not change provider `fallback.rs`.

## Reconcile the stopped implementation

Keep the existing public entrypoints, direct-input alternative and narrow sender/model-call extractions.
Do not adopt the dirty implementation as already correct.
Remove or justify the shared serializer's new `classification` acceptance; Tabular support must not silently broaden gathered prediction behavior.
Review the response conversion operations against actual Python consumption points before retaining or combining them.
Keep Tabular policy local; shared helper changes require unchanged Phase 3 behavior, not a wider redesign.

First finish one connected direct Tabular increment, then extend supported families and add PQL selection before relational work.
An integer/Boolean pilot is an increment, not completion of Tabular.
Independent design review against the RCA precedes implementation.
Two independent exact-source architecture/compatibility reviews precede executable checks; then verify public parity and relevant shared-path regressions and land the increment.
