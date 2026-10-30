# `EXPLAIN` in PQL: design and tracker

Branch `jg-explain-pql`, on top of `jg-enable-explain` (PR #34, which added explanation
support itself). Only work that lands on this branch is listed.

## What this adds

A query may start with the word `EXPLAIN`: `EXPLAIN PREDICT users.churn FOR EACH users.user_id`.
It means exactly the same as passing `outputs=['explanation']` to `predict()`: the answer carries
gradient explanations. Passing `outputs=['explanation']` as well is allowed and redundant; the two
spellings are interchangeable and give the same result (agreed: Jana, 2026-10-29).

## Design: what happens to an `EXPLAIN` query, level by level

**1. Parsing.** The parser already strips the `EXPLAIN` prefix and records it as a boolean on the
parsed query (`ParsedQuery.explain`). Nothing new to build; the field just stops being marked
"TODO: deprecate" — `EXPLAIN` is a supported prefix.

**2. Materializing (turning the query into a request).** Today the flag survives only as the
literal `EXPLAIN` prefix inside the request's query text. That is not good enough, because
rankings and forecasts are split into parts whose text is rebuilt as plain `PREDICT …` — the
prefix is gone before anything reads it. So the decision "was this asked as `EXPLAIN`" is made
here, at materialize time, from the parse flag, before any splitting. The request already keeps
the original text verbatim (`request.asked`), so the smallest form is to read the prefix from
that; no new public field unless unavoidable.

**3. Entering `predict()`.** At the top of the call, if the request was asked as `EXPLAIN`,
`'explanation'` is added to the outputs: appended to the model's defaults when the caller passed
no `outputs`, appended to the caller's list when it lacks `'explanation'`, and nothing changes
when it is already there. The addition is made to the one outputs list the rest of the call keeps
reading, so the later steps — the ranking check, the "answered without" warning — see it too.
Adding it from the get-go, on the real list, is the whole trick: everything downstream then
behaves exactly as if the caller had typed `outputs=['explanation']`, with no special cases:

- **Plain predictions** (classification, regression): the request goes out asking for
  `'explanation'`; the answer carries `result.explanations`.
- **Rankings** (`LIST_DISTINCT … RANK TOP k`): the client answers a ranking by asking one
  yes/no classification per candidate and sorting the probabilities itself — the model never
  produces a ranking, so there is no such thing as an explanation of one. The existing output
  check (which allows only `prediction` and `rankings`) sees the injected `'explanation'` and
  refuses with the existing message, before anything is sent. Nothing new is built (agreed:
  Jana, 2026-10-30).
- **Forecasts**: each step is its own request and is explained on its own, exactly as
  `outputs=['explanation']` does today, including the existing warning when only some steps
  come back explained (agreed: Jana, 2026-10-30).
- **Sessions**: `fit('EXPLAIN PREDICT …')` pins `'explanation'` into the session's outputs, so
  every later `predict()` on that session explains — that is what `fit` means: it explains on
  all calls with the same query (agreed: Jana, 2026-10-30). The other direction,
  `fit('PREDICT …')` then `fitted.predict('EXPLAIN PREDICT …')`, is refused today only because
  the session compares the question text literally; the comparison ignores the `EXPLAIN`
  prefix, the stored text stays unchanged, and the call asks for `'explanation'` as above
  (agreed: Jana, 2026-10-30).

**4. Backends.** No changes. Both backends already serve `'explanation'` for classification and
regression since #34: the in-process backend computes gradient explanations, and the wire path
passes the output list through untouched. A served model that answers without the explanation
(today's shipping relational build does not serve them) triggers the existing "answered without"
warning — which works because step 3 put `'explanation'` into the list that warning reads.

**5. One error message.** The relational preflight that refuses a target column holding only one
value names the column by extracting it from the query text with a pattern anchored to a leading
`PREDICT`. The pattern is extended to accept an optional `EXPLAIN` prefix, so the message keeps
naming the column instead of quoting the whole query.

**6. Docs.** One sentence — `EXPLAIN PREDICT …` is the same as `outputs=['explanation']` — with
an example, everywhere explanations are asked for: both skills, both PQL references, the
prediction-output reference, the architecture page, the beyond-one-answer reference, CHANGELOG.
A skill example that runs `predict` must carry verified output (`tests/test_skills.py` enforces
this).

## Status values

`NEEDS DECISION` (Jana chooses; the end-state column holds a proposal) · `TO DO` (agreed, not
built) · `WAITING (on X)` · `DONE` (with the commit)

## Work items

| Item | End state | Status |
|---|---|---|
| **W1** Parser field (design §1) | The `# TODO: deprecate` comment on `ParsedQuery.explain` is removed; `EXPLAIN` is a supported prefix | DONE |
| **W2** Detect at materialize (design §2) | `predict()` knows the request asked `EXPLAIN`, decided before any splitting; read from `request.asked` when set (rankings, forecasts), else from `request.query` (plain requests never set `asked`); no new public field | DONE |
| **W3** Inject at `predict()` entry (design §3) | `'explanation'` is added to the caller-named outputs list at the top of `SDMClient.predict` (before the split, so `_answer_in_parts` receives it), `fit`, and `FittedModel.predict`; every other site inherits that list; rankings refuse, forecasts and sessions flow through, as in §3 | DONE |
| **W4** Session text comparison (design §3, sessions) | The `EXPLAIN` prefix is ignored when a session compares the question text; stored text unchanged | DONE |
| **W5** Column-name pattern (design §5) | The relational preflight pattern accepts an optional `EXPLAIN` prefix and keeps naming the column | DONE |
| **T1** Unit tests | `tests/test_a_query_asked_to_explain_is_explained.py`: 18 pass, 2 in-process cases skip without the `sdm` extra (the same skip as every in-process test; they run in CI). Mutation-checked: disabling each of the three changes fails the matching tests. Written per the `unittesting` skill | DONE |
| **T2** Live tests | One `EXPLAIN` case per model in `test_live_nim.py` and `test_live_relational.py`, asserting `result.explanations` is filled | WAITING (on the box and the NIM images with the `refs/v1.0.1` alias workaround) |
| **D1** Docs (design §6) | `skills/tabular/SKILL.md`, `skills/relational/SKILL.md`, `skills/tabular/references/pql.md`, `skills/relational/pql.md`, `skills/relational/references/beyond-one-answer.md`, `docs/reference/prediction-output.md`, `docs/about/architecture.md`, CHANGELOG `## Unreleased`. One sentence each, no new runnable examples | DONE |
| **D2** Public API snapshot | `tests/golden/public_api.json` unchanged (W2 added no public field); the snapshot test passes untouched | DONE |
| **V1** Local gate | 2026-10-30, repo venv: ruff clean; 402 files formatted; mypy clean on 187 source files; hygiene OK; api+pql+client+connectors+root suites 2250 passed, 406 skipped, 90 deselected, 0 failed | DONE |
| **B1** Branch base | After #34 merges, rebased onto `main`; the PR base switched to `main` | WAITING (on #34 merging) |

## Settled questions (kept for the record)

- The relational request envelope carries the query text, prefix included, in
  `metadata.source_query`; the NIM only logs it. Tabular sends no query text. Nothing to change.
- A `LIST_DISTINCT` without `RANK` goes to the model as a link-prediction task and is refused by
  every backend with `UNSUPPORTED_TASK` before explanations ever matter. Nothing to change.
- The in-process backend supports `'explanation'` since #34; an earlier version of this plan
  expected a refusal there (stale, corrected 2026-10-30).
