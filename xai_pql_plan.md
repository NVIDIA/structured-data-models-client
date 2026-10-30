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
| **P3** `client.predict` ignores it. Output lists are built in four places (verified 2026-10-29): `client.py` `_prepare` :748, `fit` :700-716, `FittedModel.predict` :184-199, `_answer_in_parts` :528-547 | When the query says `EXPLAIN`, `'explanation'` is in the outputs sent, in all four places: added to the model's defaults when the caller gave none, added to the caller's list when missing, no-op when present. `EXPLAIN` in the query and `'explanation'` in `outputs` are interchangeable and give the same result | TO DO (agreed: Jana, 2026-10-29: `EXPLAIN` in the query is enough; `'explanation'` is added to whatever outputs are sent, never required from the caller) |
| **P4** Sessions pin outputs at `fit()` (`FittedModel._outputs`) and refuse a request whose query text differs from the pinned one (`client.py:1328`); `fit(PREDICT …)` + `fitted.predict(EXPLAIN PREDICT …)` is refused today (verified 2026-10-29) | The `EXPLAIN` prefix is ignored when comparing the question; the stored text is unchanged; that call asks for `'explanation'` as in P3 | TO DO (agreed: Jana, 2026-10-30) |
| **P5** Queries the client splits into parts (rankings, `LIST_DISTINCT`) cannot carry an explanation; rankings already refuse `'explanation'` | `EXPLAIN` on such a query is refused before sending, with the existing ranking refusal message | TO DO |
| **P6** The in-process backend refuses `'explanation'` with `UNSUPPORTED_OUTPUT` | `EXPLAIN` there gets the same refusal, through P3; nothing new | TO DO (verify by test) |
| **T1** Unit tests | Flag survives `materialize()` on both models; outputs gain `'explanation'` with and without a caller list; session path; ranking refusal; in-process refusal. Written per the `unittesting` skill | TO DO |
| **T2** Live tests | One `EXPLAIN` case per model in `test_live_nim.py` and `test_live_relational.py`, asserting `result.explanations` is filled | WAITING (on the box and the NIM images with the `refs/v1.0.1` alias workaround) |
| **D1** Skills say `outputs=['explanation']` only | `skills/tabular/SKILL.md`, `skills/relational/SKILL.md` and both `references/pql.md`: one sentence, `EXPLAIN PREDICT …` is the same as `outputs=['explanation']`, with an example | TO DO |
| **D2** `docs/reference/prediction-output.md` "asking" section | Mentions the `EXPLAIN` prefix as the PQL way to ask | TO DO |
| **D3** CHANGELOG `## Unreleased` | Added: `EXPLAIN` prefix | TO DO |
| **D4** Public API snapshot | `tests/golden/public_api.json` unchanged unless P2 ends up adding a public field; then regenerated with only that diff | TO DO |
| **V1** Local gate | ruff, format, mypy, hygiene, client suite and root tests green; counts recorded here | TO DO |
| **N1** The relational envelope carries the query text, prefix included, in `metadata.source_query` (`payload.py:120`), which the NIM only logs (verified 2026-10-29); tabular sends no text | Nothing to change | DONE: no work |
| **B1** The branch is based on `jg-enable-explain`, not `main` | After #34 merges, rebased onto `main`; the PR base switched to `main` | WAITING (on #34 merging) |
