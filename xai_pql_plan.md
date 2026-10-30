# `EXPLAIN` in PQL: plan and tracker

Goal: `EXPLAIN PREDICT …` asks for gradient explanations, the same as `outputs=['explanation']`
on `predict()`. Branch `jg-explain-pql`, on top of `jg-enable-explain` (PR #34). Only work that
lands on this branch is listed.

**Status values:** `NEEDS DECISION` (Jana chooses; the end-state column holds a proposal) ·
`TO DO` (agreed, not built) · `WAITING (on X)` · `DONE` (with the commit)

| Issue | End state | Status |
|---|---|---|
| **P1** The parser already lowers `EXPLAIN` to `ParsedQuery.explain`, but the field says `# TODO: deprecate` (`predictive_query/query.py:76`). `EVALUATE` is the one to deprecate, not `EXPLAIN` | The comment is removed; `EXPLAIN` is a supported prefix | TO DO |
| **P2** `materialize()` drops the flag: a `Request` keeps only the query text (`request.query`) | `Request` carries `explain: bool`, set from the parsed query on both the relational and the tabular PQL paths; `False` for requests built without PQL | TO DO |
| **P3** `client.predict` ignores it | When `request.explain` is true, `'explanation'` is in the outputs sent: added to the model's default outputs when the caller gave none, and added to the caller's list when it is missing there | TO DO (agreed: Jana, 2026-10-29: `EXPLAIN` in the query is enough; `'explanation'` is added to whatever outputs are sent, never required from the caller) |
| **P4** Sessions pin outputs at `fit()` (`FittedModel._outputs`), so an `EXPLAIN` request on `fitted.predict()` would not ask for one | Same rule as P3, applied per `fitted.predict()` call | TO DO |
| **P5** Queries the client splits into parts (rankings, `LIST_DISTINCT`) cannot carry an explanation; rankings already refuse `'explanation'` | `EXPLAIN` on such a query is refused before sending, with the existing ranking refusal message | TO DO |
| **P6** The in-process backend refuses `'explanation'` with `UNSUPPORTED_OUTPUT` | `EXPLAIN` there gets the same refusal, through P3; nothing new | TO DO (verify by test) |
| **T1** Unit tests | Flag survives `materialize()` on both models; outputs gain `'explanation'` with and without a caller list; session path; ranking refusal; in-process refusal. Written per the `unittesting` skill | TO DO |
| **T2** Live tests | One `EXPLAIN` case per model in `test_live_nim.py` and `test_live_relational.py`, asserting `result.explanations` is filled | WAITING (on the box and the NIM images with the `refs/v1.0.1` alias workaround) |
| **D1** Skills say `outputs=['explanation']` only | `skills/tabular/SKILL.md`, `skills/relational/SKILL.md` and both `references/pql.md`: one sentence, `EXPLAIN PREDICT …` is the same as `outputs=['explanation']`, with an example | TO DO |
| **D2** `docs/reference/prediction-output.md` "asking" section | Mentions the `EXPLAIN` prefix as the PQL way to ask | TO DO |
| **D3** CHANGELOG `## Unreleased` | Added: `EXPLAIN` prefix | TO DO |
| **D4** `Request` gains a public field (P2) | `tests/golden/public_api.json` regenerated; the diff is only that field | TO DO |
| **V1** Local gate | ruff, format, mypy, hygiene, client suite and root tests green; counts recorded here | TO DO |
| **B1** The branch is based on `jg-enable-explain`, not `main` | After #34 merges, rebased onto `main`; the PR base switched to `main` | WAITING (on #34 merging) |
