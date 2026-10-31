<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tabular PQL acceptance

The direct-input path landed at `391723fb`.
This work connects both PQL entrypoints to that path under [boundaries](boundaries.md) and [restart](restart.md).
Both public entrypoints landed in the ordinary-input increment at `60335b57c`.
The frozen-reference pilot passes five public journeys, and a generated reference fixture records exact sampled contexts or failures for 16 ordinary index families.

| Requirement | Public evidence required |
| --- | --- |
| Indices | Repeated requested keys expand to every matching source position in request order; missing keys preserve diagnostics. |
| Supplied scoring frame | Exclude matching entity keys, including duplicates; preserve the supplied frame and its owners. |
| Context | Remove unknown targets after scoring exclusion; retain candidate order when no reduction is needed. |
| Sampling | Fresh legacy seed 42, sample without replacement, then sort only the sampled index; verify final encoded context order. |
| Counts | Explicit and default caps, invalid counts, empty candidates, exact cap and cap plus one; preserve warning timing. |
| Task | Explicit overrides and inference from the selected context, including binary whole-number targets. |
| Index families | Range, signed/unsigned numeric, float with missing labels, string/object, temporal, categorical and MultiIndex; duplicates and mixed labels where supported. |
| Native effects | Reached key equality/hash, frame/index operations and substantive failures; approved custom sorting override exception only. |
| Prediction | Materialized Request through KumoTabular to attributed PredictionResult, compared with frozen Python. |
| Isolation | Existing source-backed materialization, warehouse sampling and fitted lifecycle remain unchanged. |

Python retains actual frames and native owners and applies returned row positions.
Rust composes scoring expansion, exclusion, known-target filtering and conditional sampling in `sdm-core`, with selection policy in `sdm-engine::tabular`.
Index observations are acquired only on the reduction path at their actual consumption point.
The sort profile must match the supported producer; the existing datetime introsort is not evidence for numeric SIMD tie ordering.
Task inference follows selection and uses the same generic Request builder as direct calls.

Before implementation, independently review the concrete observations and native failure boundaries against the frozen materializer.
Then freeze one connected candidate, obtain two exact-source approvals, run consolidated parity and regression checks, and land immediately.
Each row above remains pending until its public evidence is recorded.

## Early review decisions

Both independent reviewers verified that the current and frozen materializers differ only in import placement.
The source path uses dictionary lookup for indices, but set construction followed by pandas membership for supplied frames.
These are distinct observations, especially for missing entities; a shared equality shortcut is not justified.

Selection returns source positions, with scoring expansion separate from candidate filtering.
Only a reduced context enters the sampled-index ordering operation, with labels in draw order.
The selected-context task decision precedes schema snapshot and generic Request construction.
Sampling advice precedes the sampling read and can raise or mutate native data, so it is a real observation boundary.

The engine owns a small legacy seed-42 generator implementation.
The existing warehouse MT19937 coefficient implementation and crate dependencies remain untouched: the dependency rules forbid coupling the data and engine crates for reuse.
Archived sorting kernels are source material for review, not inherited validation or a completed public route.

The capacity decision occurs before warning delivery, but permutation generation consumes the retained candidate population afterward.
This preserves warning-driven changes in candidate length without reconstructing candidates from a potentially different source frame.
Missing-key refusal must also precede target-validity observation; bulk admission is limited to proven passive inputs.

## Current increment and remaining scope

The connected ordinary-input increment uses the existing direct prediction path, native owner observations, Rust selection and sampled-index ordering.
Its candidate requires two independent exact-source reviews, canonical contract generation, installed-wheel parity, regressions and verified landing.
The recorded index expectations come from the frozen public client through `materialize`, not from candidate sorting kernels.

Typed complex, finite Decimal/integer, exact date and fixed-offset/naive datetime labels landed at `b62256b51` through the existing sampled-index operation.
Its frozen public evidence covers sampled indices and unreduced supplied frames separately; supplied frames do not accept an explicit context count.
The host retains exact owners and observes their closed fields, while Rust compares exact coefficients and civil values and reports reached comparison failures.
The increment landed at `795dba504` delays producer binding until Rust reports that the sampled facts require it.
Pure total comparisons with monotonic or distinct keys can finish without a NumPy binary binding; stable Arrow paths keep their own profile.
When ties or comparison failures need the producer, Python binds the retained facts and retries once without rereading native owners.
The older scalar producer bindings landed at `b719a516`, with frozen public receipts selected by the actual runtime rather than reusing another NumPy version's tie order.

The increment landed at `6e78eb1` retains exact NumPy integer and floating scalar owners inside object indexes.
Its `ScalarObject` facts carry a source-bound comparison-semantics tag independently of the sorting profile.
NumPy 1.23 legacy, NumPy 1.26 legacy and audited NumPy 2.5 weak-scalar comparisons differ; NumPy 1.26's promotion mode is observed rather than inferred from its binary.
The engine uses generic object comparison and sorting for these facts, without claiming a total order or applying the profile-free unique-key shortcut to mixed-precision comparisons.
Reached integer-to-float overflow has a closed result; native warning-handler effects remain an explicit unfinished obligation.
Its public parity passed 379 cases in each of the prepared NumPy 1.23.5, 1.26.4 and 2.5.4 runtimes.

The increment landed at `b5c8a403` binds the official NumPy 2.1.3 and 2.3.4 deployment wheels.
The former requires unconditional four- and eight-lane SIMD entry profiles; only the original-array sorted shortcut differs, while the outer monotonic check remains unchanged.
Frozen public materialization confirms a tie-order difference at 320 sampled rows with sorted duplicate present keys and a missing label.
Compiler/source evidence supports the existing GNU fallback family, and receipt identity includes effective AVX2 and AVX512 dispatch groups.
The current host provides live AVX2 evidence; source compatibility does not claim live AVX512 execution.
Two independent exact-source reviews, canonical contract 40 generation and the required handoff passed before verified landing.
Installed-wheel public parity passed 394 cases in each of the current, NumPy 2.1.3 and NumPy 2.3.4 runtimes; six runtime-specific frozen receipts contain 183 journeys each.

Full PQL completion remains pending for custom comparison-label ownership, Decimal context signals, remaining numeric promotion modes and effects, additional temporal owners, broader producer bindings, and remaining native frame/count effects.
These are retained obligations, not silently unsupported public inputs or an expansion of the approved custom-sort exception.
The development route remains opt-in; existing defaults and warehouse paths are unchanged.
A frozen public custom-label comparison witness has been presented for an owner decision; no broader fallback is implemented.

The ordinary integer-key/primitive-target path uses one `SelectTabularRows` composition for both entrypoints.
That operation composes expansion or key exclusion, known-target filtering and capacity policy in Rust.
Only native-effect paths retain staged key/mask observations; sampling stays after warning delivery for both.
Forced frame routes explicitly refuse relational materialization until its separate core consumer exists, avoiding a silent legacy route.

The increment landed at `a4f7295d8` binds the official NumPy 2.5.4 cp314 binary and the Arm floor/newest binaries to existing audited sorting and comparison profiles.
Frozen reference recording uses Python 3.14.8, pandas 3.1.0 and Arrow 26.0.0.
Date/datetime comparison refusals retain Rust-selected operand positions; Python presents the native exception for that exact admitted pair because the diagnostic changed in Python 3.13.
This single failure-presentation operation does not provide a sorting callback or broaden custom-owner admission.
Two independent source reviews and handoff passed before verified landing.
Installed-wheel parity passed 394 cases each on current x86, Python 3.14 x86, Arm floor and Arm newest; both native Arm lanes ran on the exact candidate `fa9b13d25`.
The targeted Arm run is PQL evidence and does not replace full CI.

The count/task increment landed at `447498a66` with 11 canonical frozen public journeys and 44 installed-client checks.
It covers the actual 10,000-row cap and cap plus one with default/explicit counts, both empty selected-context causes, binary task inference after scoring exclusion on both entrypoints, an integer-subclass count and reduction-warning handlers that record or raise.
The dedicated frozen acceptance compares context and encoded row order, results, errors and event sequences in the matching current runtime.
The ordinary semantic checks also run without runtime-specific receipts; neither path silently substitutes a different runtime oracle.
No product change was needed for these cases.

The increment landed at `7b022ce92` addresses NumPy 1.26 AVX512 with a distinct eight-lane profile: unconditional entry, a 64-element recursive network cutoff and zero-filled masked index lanes.
Its existing partition and GNU depth fallback remain shared; the host alone selects the audited producer profile.
Independent vectors are transcribed from the pinned upstream network headers and cover ordinary duplicates, maximum/infinity keys and nonzero-index subranges.
Those vectors establish network behavior, not native AVX512 public execution; live AVX512 evidence is still absent on this host.

Object-held Timestamp support is the next connected temporal increment.
Frozen public witnesses distinguish instant comparison from mixed Timestamp/datetime wall-clock comparison, including timezone identity, fold, submicrosecond precision and years outside Python datetime range.
Native tick values and resolution preserve wide instants without forcing nanoseconds into int64 or eagerly calling range-sensitive timezone methods.
The core must preserve separate LT/GT/EQ outcomes and reached year-range refusals; a total-order shortcut is not valid for ambiguous mixed temporal owners.
Standard pandas 2 pytz and pandas 3 ZoneInfo owners are part of the planned coverage; arbitrary custom timezone callbacks remain unresolved.

Temporal MultiIndex acceptance must cover three distinct boundaries, verified through frozen public calls on pandas 2 and 3.
First, ordered used codes can return before sorting an unused level containing a wide Timestamp and datetime; forcing normalization with alternating secondary codes reaches the year-range ValueError.
Second, reached level errors carry a level path plus operand positions, so Python presents the error from the retained level owners rather than indexing top-level tuples.
Third, mixed ambiguous-time owners can be ordering ties without being equal; expanded-value stable lexsort can preserve alternating sampled codes that distinct integer ranks would reorder.
The native object stable-sort helper family is identical across the audited NumPy 1.23.5, 1.26.4, 2.1.3, 2.3.4 and 2.5.4 sources.
This source evidence establishes the implementation target, not candidate validation or a completed temporal landing.

## Ordinary numeric warning acceptance

The owner does not require incidental pandas warning counts or their interleaving.
For numeric overflow RuntimeWarnings from pandas index/sorting sites, acceptance retains raw frozen receipts and compares the warning sites, while ignoring repetition and placement among other warnings.
Selected rows, encoded rows, predictions, exceptions (including warning-as-error), unrelated warnings and printed diagnostics remain exact.
This does not authorize a different comparison algorithm, Python callbacks or a broader fallback.

## Decimal comparison increment

The one-way numeric increment landed at `4b9369bc`, with six-runtime frozen-reference acceptance and the required reviewed handoff.
Mixed finite Decimal/Python-float labels are the next demonstrated gap: the reference sets `FloatOperation` and sorts or raises its trap, while the landed core refuses admission.
Python retains the actual standard Decimal context and supplies its trap settings with exact scalar facts.
Rust owns exact comparison, stage-specific trap suppression and sorting; it returns ordered closed numeric diagnostics and the result or refusal.
Python presents diagnostics in order, ORs reached Decimal flags into the retained context, and then constructs the result or raises the selected exception.
An earlier warning-as-error prevents later flag presentation, without callbacks, a new Run step or changes to warehouse paths.
Both public entrypoints cover ordinary finite values, precision boundaries, float infinity/NaN, sticky flags and mixed-level warning/error ordering.
Previously matching all-Decimal NaN cases remain regression controls; arbitrary Decimal subclasses and mixed Decimal-NaN owners are not claimed by this increment.
The old client remains a reference for useful behavior, not a requirement to copy demonstrated bugs or incidental implementation details; intentional improvements require explicit justification and verification.

For supplied-frame failures with identical Decimal exceptions and flags, acceptance ignores only the presence of the exact informational default-sampling advisory.
The old pandas Boolean-Series conversion can compare row labels and raise before that advisory; positional filtering avoids this redundant comparison and reaches the same terminal error during sorting.
Raw reference evidence, successful-call warnings, unrelated warnings and exceptions remain exact; this allowance does not treat a warning promoted to an exception as equivalent to a Decimal error.


## Weak-and-warn comparison binding

Mixed Decimal/Python-float support landed and was verified at `bee780b9`.
The next binding admits NumPy 1.26 and 2.1 weak-and-warn comparisons through their existing weak comparison profiles, retaining binary provenance checks and observing the mode on each call.
Pinned generic dispatch checks promotion warnings against output dtypes; comparison outputs remain Boolean, so this does not import arithmetic promotion-warning behavior into sorting.
The source audit is supported by 2,520 paired native LT/GT/EQ comparisons with identical values, errors and warnings, plus 32 frozen public journeys.
Public acceptance explicitly requests weak-and-warn and compares with the unchanged frozen weak receipts; the canonical recorder's default remains weak.
No Rust algorithm, contract, diagnostic event or warehouse path changes.
