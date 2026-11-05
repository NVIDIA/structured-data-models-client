<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PR #63: explanation coverage audit

Audited Phase 9 at `6cf0bf0c1` against all 48 inline [PR #63 review comments](https://github.com/NVIDIA/structured-data-models-client/pull/63), review bodies (including outside-diff findings), replies and issue comments.
**The port has substantial lifecycle and numerical coverage, but is not free of the problems raised in review.**
The previous completion summary overstated confidence by presenting broad suite totals without this review-by-review mapping.

## What was actually tested

| Layer | Evidence and limitation |
|---|---|
| Independent numerical expectations | Public constructor checks for cohort means/shares, class/category limits, missing values, anchored time, exact labels and empty classification; Rust checks for ordering, overflow and invalid input. These did not cover all combinations below. |
| Request/session integration | One-shot, cached fit analysis, holdout exclusion, late EXPLAIN, persistence, partial results, merge refusal and SQLite prediction after closing its source connection. Most use controlled model replies. |
| Python/Java boundary | Thin result wrappers, native computation, Java Arrow ownership/allocation cleanup and exact mixed/wide/decimal labels. |
| Real model execution | Previous final-source CPU run: four Python cases and three Java methods, both model families and regression/classification/session paths. Checks establish integration, not an exhaustive numerical oracle or connector matrix. |
| Selected old-versus-new comparison | Four exact constructor comparisons: categorical regression/classification, numeric quantiles, multicategorical. Empty classification improves the old exception. Not a full differential suite. |
| Broad suites | The reported 577 Rust tests and 202 Python checks include unrelated coverage; they are not explanation-specific counts. |

This audit reran the public column-analysis, Tabular-explanation, Relational-explanation and gradient-explanation files: **81 passed, 4 opt-in cases skipped in 8.42 seconds**.
No real model or warehouse endpoint was rerun in this audit.

## Review concerns mapped to the port

“Code only” means inspected implementation, not a demonstrated regression test for that precise concern.
Withdrawn bot findings are not treated as confirmed product bugs.

| Review concern | Current evidence / remaining work |
|---|---|
| Later forecast steps introduce tables/relationships | Rust `columns::forecast` unions schemas and renumbers instances; public forecast test checks nonempty columns and gradient step identity. **Missing a numerical assertion for a table first reached in a later step.** |
| Timestamp bins must use age and half-open boundaries | Implemented; public test verifies 0.5h/1.5h, future/null exclusions. **Exact-edge and old-but-narrow timestamp-span cases are not explicitly tested.** |
| Combine every batch | Original bot finding was withdrawn after the author clarified single-draw semantics. Port analyzes the first batch, consistent with merged #63; open #73 proposes a different union-of-sent-instances contract. |
| Missing target labels / related rows of dropped entities | Direct public tests cover filtering and stable class means, including supplied relational context. Improvement over the narrow path the author deferred. |
| Fit must survive unavailable statistics | Recoverable analysis returns diagnostics; public prediction test confirms predictions survive. Session caching/late EXPLAIN is tested, but **a dedicated fit-failure survival assertion is still desirable**. Malformed protocol remains an error. |
| List-valued multicategorical cells | Reported ordinary-client failure was withdrawn: supported payloads use separator-joined strings. Separator behavior is tested; no claim of a supported arbitrary-list dataframe path. |
| Large numeric labels collide | Formatting avoids the original `.4g` large-value case, but there is no dedicated close-large-value regression assertion. |
| Small numeric labels collide | **Reproduced bug:** 11 values from 0 to 0.001 produce 10 cohorts but only two printed labels (`[0 - 0]`, `(0 - 0]`). Existing quantile tests missed this. |
| Missing categories disappear into OTHER | **Reproduced bug:** 12 categories plus null produces no `N/A`; its outcome is mixed into `OTHER(*)`. Existing missing-value test has only one nonmissing category. |
| Unique ID-like categories dominate rankings | Code retains merged-main eligibility and score formula. Public probe confirms 300 singleton categories are admitted. **No informative-feature-versus-ID ranking test; text semantic type itself is excluded.** |
| Score disagrees with displayed cohorts / favors more bins | Code calculates score before tail merging and normalizes it. Formula tests do not establish statistical validity. **Known inherited limitation**, not an improvement; #73 proposes removing score entirely. |
| Pre-trim population differs from fitted population | Intentional merged-#63 behavior, tested through materialize/save/load/predict. Public reference still needs an explicit population explanation; design docs alone are insufficient. #73 changes this contract. |
| Every fit pays analysis cost | Retained deliberately for later EXPLAIN without retaining raw context. Caching is tested; **no explanation-specific latency or peak-memory benchmark was run**. |
| Every request keeps a full context copy | Improved: retain removed-row delta, not another full context. Persistence and exclusion from model body tested; no measured memory saving claim. #73 proposes removing even this need by changing population. |
| Entity table is analyzed at hop 0 | Engine does so. Gradient documentation says it, but the separate column-association section should explicitly describe question-table exclusion and hop 0 too. |
| Empty one-shot result loses analysis | Engine computes before the empty return. Constructor empty-label tests are not the same case; **add an explicit one-shot empty-predict/nonempty-context assertion**. |
| Forecast validates too late | **Still present by code inspection:** combined analysis precedes per-step `prepare` validation. No test proves invalid outputs avoid analysis. |
| Forecast loses request IDs/metadata | **Still present by code inspection:** `by_step` retains first request ID and builds empty metadata. Existing gradient step-identity test does not verify request IDs. |
| Wide integer labels restored | Direct public constructor and Java exact-label tests pass. **Still need a model-wire string label compared to the prediction's restored integer**, not just an already-typed constructor input. |
| Mixed timestamp formats | Rust parses each value independently, avoiding first-value format inference. **Code only for the precise mixed-fractional-format review example.** |
| EXPLAIN defaults spuriously treated as explicitly named outputs | Outside-diff review concern remains plausible: Rust expansion sets `default_outputs=false`. **Needs a public regression probe with omitted default outputs; not certified fixed.** |

## Follow-up PR is a different contract

[PR #73](https://github.com/NVIDIA/structured-data-models-client/pull/73) is open, not merged at this audit.
It proposes adaptive numeric labels, separate N/A, earlier validation and forecast request IDs, plus two substantive changes: remove score/ranking and analyze sent rows across batches instead of the pre-trim population.
Those proposals must be reconciled explicitly across Rust, Python, Java and public documentation; silently porting an unreviewed contract would be wrong.
Its adaptive precision proposal also deserves tests below six decimal places rather than assuming a fixed maximum guarantees distinct labels.

## Recommended next step

First add public regression tests for the reproduced label/N/A bugs and missing review cases above.
Then fix confirmed defects in the Rust engine, with shared Python/Java contract tests, and reconcile the score/population decisions with #73.
Keep hosts responsible for dataframe/Arrow presentation and I/O; do not add Python callbacks or change sampling SQL.
Finish with focused lifecycle and bounded CPU-model tests; another broad warehouse matrix is not needed solely for cohort formatting.
This audit changes documentation only; it does not claim the identified defects are fixed.
