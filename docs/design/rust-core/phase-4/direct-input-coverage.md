<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Direct Tabular input coverage

This extends the existing [boundaries](boundaries.md) and [restart contracts](restart.md).
The numeric/Boolean pilot, broader datatype conversion, and native options/response increments are landed.
The public-input admission and ownership increment is implemented and independently reviewed; final handoff and landing remain pending.
All listed families now have connected constructor/prediction paths and recorded parity cases.
The consolidated installed-wheel run passes 320 public/shared-regression tests, with 12 environment-dependent skips.
The frozen-reference evidence includes 74 datatype observations, 22 native options/response cases, and 17 public-input/ownership cases.
The full Rust contract, schema, request and host suites pass 463 tests; strict Clippy passes.
These results establish the listed coverage, not exhaustive behavior of every possible user-written extension.
PQL sampling, fitted Tabular use, file/result/state journeys, live acceptance and the default-route switch remain separate follow-on work.

| Input family | Implemented coverage and decisive parity cases |
| --- | --- |
| NumPy float32/64 | Construction, declared widths, NaN versus null, infinities, signed zero and mixed-width context/scoring. |
| Nullable and Arrow-backed numeric/Boolean | Validity masks, present NaN, typed-empty columns and lossless widening. |
| Strings and categoricals | Actual scalar values, nulls, categories versus labels, inferred categorical/text meaning and advice before encoding. |
| Datetime and timezone-aware columns | Native timestamp conversion, UTC normalization, microsecond wire spelling, NaT and range failures. |
| Object columns | Value-based inference, all-null untyped columns, homogeneous scalars and mixed scalar fallback. |
| Decimal and wide integers | Integral versus fractional inference, precision/range checks, fixed-point text and native conversion failures. |
| Other extension/scalar families | Preserve the reference's actual scalar fallback and refusal behavior; do not infer support from Arrow conversion alone. |
| Inference options | Nested mappings/sequences, NumPy scalars and native conversion failures, consumed after table encoding. |
| Mutable/public native inputs | Retain owners and consumption order across construction, prediction and diagnostics; preserve explicit schema edits and custom effects. |

Reference paths are `wire/dtypes.py::infer_tfm_dtype`, `serialize_column` and `serialize_cell`, plus `adapters/tabular.py::_envelope` and the direct `Request` constructor.
Schema inference and wire serialization are separate consumption points; an eager Arrow snapshot cannot silently stand in for both.
Container-valued cells remain refusals where the old client refuses them.
Direct row indexes remain irrelevant to encoding; this work does not implement PQL index sorting.

Implement the agreed coverage cohesively, then review the full source and run consolidated old/core parity and shared-path regression checks.
New contract fields must carry observed storage or completed native observations, with semantic decisions in the existing Rust policy modules.
Any boundary amendment must be specific to a demonstrated datatype requirement.

## Reviewed conversion contract

Keep constructor inference, prediction dtype resolution and post-advice encoding distinct.
Carry native storage family, object-content classification, validity/provenance and declared decimal metadata beside value buffers.
Rust chooses schema meaning and wire types from these facts.
Numeric categorical storage does not imply numeric wire dtype; all-null object storage does not imply a typed string column.
Reuse existing string/decimal/temporal primitives only where their direct-frame semantics agree.

Encoding retains the native column owner, validates scalar cells, takes its native value list, and converts that column before observing the next column.
Custom conversion, equality/hash and extension iteration effects must occur only at their reference consumption points.
Do not eagerly stringify objects or copy warehouse inference behavior into the direct route.
Native inference-option conversion stays after both tables are encoded.

Early review qualifications: effectful cells are consumed sequentially, so an earlier rejection prevents later native conversions.
Unknown extensions remain unknown until scalar admission; content classification does not replace separately reached distinct-count, hashing/equality or key observations.
Bulk Arrow conversion is reserved for ordinary families whose conversion is observationally equivalent.

For effectful cells, use closed conversion observations at the reference's actual short circuits: missingness, finite check, coercion, then required exactness checks.
Python retains the native value and exception; Rust decides acceptance and encoding.
Do not collect later observations before an earlier rejection, including diagnostic `repr` conversions.
This extends the existing direct-response observation pattern, without a callback registry or changes to warehouse paths.

Custom-object construction also needs reached observations instead of a whole-frame snapshot.
Preserve initial scalar admission, ordered dtype/stype inference, repeated table-construction admission, then key/time selection and schema advice.
Distinct counts for stype, key selection and advice are separate native reads and cannot substitute for one another.
Scoring compatibility precedes the target's `dropna`/`nunique` observations; the task-kind warning filter follows those observations.
Use closed schema observations with Rust-owned decisions at these boundaries, retaining bulk preparation for proven-passive input families.

Native time selection reuses the existing Rust selector with separately reached population and UTC-start observations.
The complete-frame counts use an internal facts adapter; this does not query a warehouse or change source-backed inference.
Only an unresolved earliest-of-several choice reaches native datetime parsing, and its Arrow attachment retains nanoseconds before wire precision is applied.
The direct single-table schema retains the selected column; it does not retain time-choice advice history, which the reference suppresses for this path.
