<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tabular PQL acceptance

Scope: both frame-backed PQL materialization entrypoints through the existing direct Request → KumoTabular → PredictionResult path.
Follow [boundaries](boundaries.md) and [restart](restart.md).
PQL completion does not include fitted/file-fed Tabular, relational migration or changing default routes.
The development route remains opt-in.

## Required behavior and evidence

| Requirement | Implementation / public evidence |
| --- | --- |
| Requested indices | Expand repeated requested keys to all matching source positions in request order; preserve missing-key diagnostics before target inspection. `test_tabular_pql_core.py` and canonical `tabular_pql_reference` receipts. |
| Supplied scoring frame | Exclude matching entity keys, including duplicates, while retaining the supplied frame and native owners. Same public suite and receipts. |
| Context filtering | Remove unknown targets after scoring exclusion; keep candidate order and do not inspect index labels when reduction is unnecessary. Same public suite; `test_tabular_sorting_scope.py`. |
| Sampling | Fresh seed 42, sample without replacement, then sort the actual sampled labels. Verify selected and encoded row order through prediction, including 320-row duplicate/missing boundaries. Same public suite and receipts. |
| Counts and task | Actual cap, cap plus one, invalid/empty cases, explicit overrides and task inference from selected context. Count/task landing `447498a66`: 11 frozen public journeys and 44 installed checks. |
| Native effects | Preserve reached key/hash/frame observations, substantive errors and ordinary numeric/Decimal signals. Canonical numeric and Decimal groups, with explicit owner decisions below. |
| Prediction | Reuse direct path landed at `391723fb`; both PQL entrypoints first landed at `60335b57c`. Public fixtures compare context order, encoded order, errors and attributed predictions. |
| Isolation | Warehouse source dispatch, SQL/providers/fallback, continuing warehouse RNG and fitted lifecycle remain unchanged. No universal sampler, new crate or numeric callback protocol. |

Python retains actual frames and native owners and applies source-row positions returned by Rust.
Rust composes ordinary scoring expansion, exclusion, known-target filtering and cap policy in `sdm-core`, using `sdm-engine::tabular` for selection.
Native-effect paths retain reached key/mask observations.
Reduction warnings precede sampling reads; sampling observes the retained candidate population afterward.
Task inference follows final selection and uses the generic Request builder.

## Finite index and runtime matrix

| Index family | Coverage and status |
| --- | --- |
| Range, signed/unsigned numeric, floating, Boolean, string, nullable and primitive Arrow | Frozen public `indices` and supplied-frame journeys, including duplicates, missing labels and sampled ties. |
| Complex, finite Decimal/integer, date, fixed-offset/naive datetime | Landed `b62256b51`; generic object comparison semantics and reached failures are retained. |
| NumPy scalar objects and mixed precision | Exact owners and widths; source-bound promotion semantics separate from sorting profile. Landed `6e78eb1`; ordinary ignore/raise/warn/print behavior completed at `4b9369bc`. |
| Timestamp, standard ZoneInfo/pytz, temporal MultiIndex | Landed `87ca02c27`: identity/fold, precision, wide-year failures, unused-level guards and non-total mixed temporal lexsort. |
| Decimal/float context effects | Landed `bee780b9`: exact finite comparisons, FloatOperation/InvalidOperation traps and ordered sticky flags, including failures. Canonical Decimal public receipts and Rust controls. |
| Categorical and MultiIndex | Category order/codes, levels/codes, missingness, duplicates, nested owner addressing and actual sampled-subset ordering; existing public suites. |
| Arrow decimal/date/binary | Landed `5060b299`: six runtimes each passed 728 public checks. Exact coefficients/scales, physical dates, explicit unsigned bytes, chunks, nulls and MultiIndex levels. |
| Object bytes, Decimal infinities and Python timedelta | Final batch implementation; frozen public witnesses prove the old client succeeds on both entrypoints in current/floor runtimes. Consolidated validation and landing remain required. |

Supported producer bindings are explicit official NumPy binaries, not a guess from a CPU name or version alone.
The local six-runtime matrix covers NumPy 1.23.5, 1.26.4, 2.1.3, 2.3.4 and 2.5.4; pandas 2.2.2 through 3.1.0; Arrow 14 through 26; Python 3.10, 3.12 and 3.14.
Per-runtime canonical receipts retain exact binary hashes and effective CPU dispatch features.
Arm floor/newest baseline acceptance landed with `a4f7295d8`; the final runtime runner also includes Arrow and ordinary-object completion suites.
Live AVX512 acceptance is still required separately from AVX2/Arm results and independent source vectors.
The legacy 1.26 network64 profile landed at `7b022ce92`; source vectors alone do not establish native execution.

Object values keep their actual comparison family and generic sorting behavior; they are not silently converted into Arrow or typed numeric ordering.
Ordering profiles are bound only when needed, using retained observations without rereading native owners.
Stable Arrow arrays use stable ordering; generic object comparisons retain their source-bound algorithm and substantive failures.
Byte vectors are distinct from Unicode, durations use normalized components and widened arithmetic, and Decimal infinity retains its Decimal owner for contextual float signals.

## Explicit compatibility decisions

- User-defined comparison/sorting functions are unsupported by the October 31 owner decision.
  This supersedes the former custom-sort Python fallback exception; removal landed at `eb26d7c2`.
  Inspect the actual sampled owner only when sorting is required; do not execute custom sorting or inspect unused indexes.
- Synthetic custom NumPy handler mutations do not require emulation.
  Numeric conversion/comparison stays in Rust through one-way facts/results; no callback interpreter or numeric Run step.
  Ordinary precision, rounding, overflow modes, errors and required diagnostic order remain covered.
- Incidental pandas warnings need not match one-for-one.
  Weak-and-warn admission landed at `eb26d7c2`; the exact reference-only int64-to-int8 promotion advisory from MultiIndex code arithmetic is excluded from mode comparison.
  Avoiding that obsolete warning-as-error failure is an intentional improvement; every other warning/count and error remains checked.
- A bounded supplied-frame Decimal failure comparison allows the informational reduction advisory when both clients have identical substantive error and Decimal flags.
  Raw reference observations remain retained.
- Wide Arrow dates deliberately avoid the old monotonic probe's incidental Python-date range failure.
  Both public entrypoints match the complete ordinary-date reference journey after a constant physical-day shift, including reconstructed indexes, encoded rows and predictions.

## Final delivery gate

Finish the remaining connected code and finite coverage before broad testing, per the owner's October 31 direction.
Obtain two independent exact-source reviews, generate canonical contracts/receipts, review generated artifacts, then run one consolidated public/regression pass and required handoff.
Retain raw evidence and fix substantive failures; do not turn an unsupported ordinary family into a silent fallback.
Completion requires verified landing, live runtime evidence and reconciliation of every row above.
