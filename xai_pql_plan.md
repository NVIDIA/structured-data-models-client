# `EXPLAIN` in PQL: plan and tracker

Goal: `EXPLAIN PREDICT …` asks for gradient explanations, the same as `outputs=['explanation']`
on `predict()`. Branch `jg-explain-pql`, on top of `jg-enable-explain` (PR #34). Only work that
lands on this branch is listed.

**Status values:** `NEEDS DECISION` (Jana chooses; the end-state column holds a proposal) ·
`TO DO` (agreed, not built) · `WAITING (on X)` · `DONE` (with the commit)

| Issue | End state | Status |
|---|---|---|
| **P1** The parser already lowers `EXPLAIN` to `ParsedQuery.explain`, but the field says `# TODO: deprecate` (`predictive_query/query.py:76`). `EVALUATE` is the one to deprecate, not `EXPLAIN` | The comment is removed; `EXPLAIN` is a supported prefix | TO DO |
| **P2** After `materialize()` the flag survives only as the `EXPLAIN` prefix in `request.query` (verified 2026-10-29: the prefix is kept on both paths, `materializers/tabular.py:94`, `materializers/relational.py:311`) | `predict()` knows the request asked `EXPLAIN`; the smallest form that works, no new public field unless unavoidable | TO DO |
| **P3** `client.predict` ignores it. Output lists enter at `_prepare`, `fit`, `FittedModel.predict` and `_answer_in_parts` (`client.py`); other sites (`_ranked`, `_by_step`, `_empty_answer`, the adapters' `default_output` and output sections) inherit from those or rebuild their own list (verified 2026-10-30) | When the query says `EXPLAIN`, `'explanation'` is injected at the entry points: added to the model's defaults when the caller gave none, added to the caller's list when missing, no-op when present. Every other site inherits it, and T1 asserts the output actually reaches the request on each path. `EXPLAIN` in the query and `'explanation'` in `outputs` are interchangeable and give the same result | TO DO (agreed: Jana, 2026-10-29: `EXPLAIN` in the query is enough; `'explanation'` is added to whatever outputs are sent, never required from the caller) |
| **P4** Sessions pin outputs at `fit()` (`FittedModel._outputs`) and refuse a request whose query text differs from the pinned one (`client.py:1328`); `fit(PREDICT …)` + `fitted.predict(EXPLAIN PREDICT …)` is refused today (verified 2026-10-29) | The `EXPLAIN` prefix is ignored when comparing the question; the stored text is unchanged; that call asks for `'explanation'` as in P3 | TO DO (agreed: Jana, 2026-10-30) |
| **P5** Rankings (from `LIST_DISTINCT`) already refuse `'explanation'` (`_refuse_outputs_a_ranking_cannot_give`, `client.py`). Forecasts are also split into parts but support explanations per step, warning when only some steps carry one (verified 2026-10-30) | `EXPLAIN` on a ranking is refused before sending, with the existing ranking refusal message. `EXPLAIN` on a forecast flows through P3 like any query and keeps the per-step warning | TO DO (agreed: Jana, 2026-10-30) |
| **P6** Since #34 the in-process backend supports `'explanation'` for classification and regression (it refused with `UNSUPPORTED_OUTPUT` before; stale row corrected 2026-10-30) | `EXPLAIN` in-process returns explanations through P3, the same as `outputs=['explanation']`; nothing new | TO DO (verify by test) |
| **T1** Unit tests | Flag survives `materialize()` on both models; outputs gain `'explanation'` with and without a caller list, on each entry-point path (P3); session path; ranking refusal; forecast pass-through with the per-step warning; in-process success. Written per the `unittesting` skill | TO DO |
| **T2** Live tests | One `EXPLAIN` case per model in `test_live_nim.py` and `test_live_relational.py`, asserting `result.explanations` is filled | WAITING (on the box and the NIM images with the `refs/v1.0.1` alias workaround) |
| **D1** Skills say `outputs=['explanation']` only | `skills/tabular/SKILL.md`, `skills/relational/SKILL.md`, `skills/tabular/references/pql.md` and `skills/relational/pql.md` (there is no `skills/relational/references/pql.md`): one sentence, `EXPLAIN PREDICT …` is the same as `outputs=['explanation']`, with an example | TO DO |
| **D2** `docs/reference/prediction-output.md` "asking" section | Mentions the `EXPLAIN` prefix as the PQL way to ask | TO DO |
| **D3** CHANGELOG `## Unreleased` | Added: `EXPLAIN` prefix | TO DO |
| **D4** Public API snapshot | `tests/golden/public_api.json` unchanged unless P2 ends up adding a public field; then regenerated with only that diff | TO DO |
| **V1** Local gate | ruff, format, mypy, hygiene, client suite and root tests green; counts recorded here | TO DO |
| **N1** The relational envelope carries the query text, prefix included, in `metadata.source_query` (`payload.py:120`), which the NIM only logs (verified 2026-10-29); tabular sends no text | Nothing to change | DONE: no work |
| **B1** The branch is based on `jg-enable-explain`, not `main` | After #34 merges, rebased onto `main`; the PR base switched to `main` | WAITING (on #34 merging) |
