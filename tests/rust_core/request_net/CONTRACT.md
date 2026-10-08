# Phase 8 acceptance contract

The acceptance checks in `tests/rust_core/request_net/` test the Phase 8 safety net through a small, fixed interface.
This file is that interface.
Everything it doesn't name is the implementer's choice, to be described in `tests/parity/FORMAT.md`.

The checks run in the phase venv, with the packages installed editable from this checkout: a check fails if `sdm_client` or `sdm_relational_engine` is imported from anywhere else, because every site is named by its file and line here.
P8.hashseed records the PostgreSQL scenarios through `SDM_POSTGRES_TEST_DSN`, as `run_corpus.py record` does.
Nothing under `packages/*/src` changes in this phase, so a line number names the same statement from the lock to the end of the phase.

## The engine raises in the catalog (P8.catalog)

An engine raise that reaches users is a `raise` statement in `sdm_relational_engine/rfm/inference/` or `sdm_relational_engine/rfm/rfm.py` whose class is `ValueError`, `KeyError`, `TypeError`, or `KumoRelationalError` or a class deriving from it.
`materializers/relational.py::_engine_errors` turns each of these into an `INVALID_REQUEST` carrying its text.

`packages/sdm-client/diagnostics.json` lists each one as an entry of its `diagnostics` array with:

- `kind`: `raise`;
- `class`: the class as the source names it, such as `ValueError` (a dotted name is read by its last part);
- `template`: holding every literal part of the message, as for the other entries;
- `file` and `line`: the repository-relative file, and a line of the `raise` statement.

`code` is free: `null`, or the `INVALID_REQUEST` users see.
The catalog keeps every entry it has today, lists no entry that is not a real site, and is exactly what `scripts/diagnostics.py --out` writes.

## Message fixtures (P8.messages)

### Which sites

The sites are every `raise` of `StructuredDataError` or a class deriving from it, and every `warnings.warn`, in these files under `packages/sdm-client/src/sdm_client/`: `decompose.py`, `request.py`, `client.py`, `result.py`, `base.py`, `models.py`, `query.py`, `errors.py`, and every file under `materializers/`, `adapters/` and `wire/`.
Under `packages/sdm-relational-engine/src/sdm_relational_engine/`, they are every `warnings.warn` and every engine raise that reaches users (above) in `rfm/inference/` and `rfm/rfm.py`.
Any other catalog entry in those files is a site too.
A site is named `path:line`, with the line where its statement starts, as the catalog gives it.

### What a message fixture records

A message fixture is a case of any file `tests/parity/fixtures/*.json`, run by the Python runner `tests/parity/_fixture_ops.py` (`run(operation, input)`).
Besides the keys `FORMAT.md` already names, it records:

- `expected_error`, for a case that raises: an object with
  - `class`: the module-qualified class of what the operation raised, such as `sdm_client.errors.StructuredDataError` or `builtins.ValueError`;
  - `code`: as today, the error's `code` where it is a non-empty string, otherwise its class name;
  - `message`: as today, the error's `message` attribute where it is a string, otherwise `str(error)`;
  - `details_keys`: the sorted keys of the error's `details` mapping, `[]` when it has none.
- `expected_warnings`: every warning of a `UserWarning` class the operation's caller sees, in order, repeats included, each `{"category": "module.Class", "message": text}`, such as `sdm_client.core.warnings.SDMWarning`.
  A case that raises lists the warnings given before it raised.

"What the operation raised" is the exception `run` raises from, that is the operation's own exception, not the runner's `FixtureError` wrapper.
The checks run each case under `warnings.catch_warnings(record=True)` with `simplefilter('always')`, so a warning the operation catches or silences itself is not seen.

### How a fixture is credited with a site

A case is credited with a site when the case gives what it records and:

- for a raise site, the case records all four keys of `expected_error`, and the site is the `raise` statement that raised the operation's exception, or one it was raised from: its `__cause__`, or its `__context__` when that is shown;
- for a warning site, the case records `expected_warnings`, and the site's `warnings.warn` call gave one of the warnings the caller saw, by category and text.

An operation must therefore let a refusal propagate and let warnings reach its caller, rather than turning them into output.

### Sites without a fixture

`tests/parity/UNREACHABLE_DIAGNOSTICS.md` holds a Markdown table:

| Site | Kind | Class | Reason |
|---|---|---|---|
| `packages/sdm-client/src/sdm_client/example.py:123` | raise | StructuredDataError | Why no input reaches it, or why its text never reaches a user |

- `Site` is `path:line`, in backticks or not; `Kind` is `raise` or `warning`; `Class` is the class as the catalog names it.
- The reason has at least three words; the reviewer checks it holds.
- A listed site is credited to no fixture, is listed once, and is a site of the list above.

Every site is credited to a fixture or listed there.
Every case of every fixture file gives what it records, and every case at BASE is still there.

## Transcripts (P8.nim-v2, P8.scenarios)

A transcript keeps the version-1 keys and adds:

- `responses`: the bodies the stand-in model answered, in the order it answered them, one per request it was sent.
  A response is JSON; a stand-in answer that is an HTTP error stores its status as an integer under `status` (or `status_code`) somewhere in its entry.
  Every transcript that sent a payload stores `responses`, and every transcript of BASE stores one per payload.
  A replay answers from them: changing a predicted value alike in `responses` and `result.arrow` still replays, and changing it in `responses` alone does not.
- `error`: present when the run's last call raised, an object with `class` (or `type`) holding the module-qualified class and `message` holding the text, and whatever else the format needs.
  A replay compares it: a changed message does not replay.
- The result's outcome that `to_pandas()` does not show, where the scenario needs it: the merged result's `unscored_candidates` (a JSON list under that key) and its explanation (an object under the key `by_forecast_step`, keyed by step), anywhere in the transcript outside `payloads` and `responses`.

`FORMAT.md` has a row for the key `responses` in its table of transcript keys.

Calls keep the version-1 names, and the session calls are named `SDMClient.fit` (whose arguments include `holdout` where the run names one) and `FittedModel.predict`.

## Scenarios (P8.scenarios)

The new scenarios are found by what their transcripts hold, not by their names, among the transcripts added since BASE whose `backend` is `memory`, `duckdb`, `sqlite` or `postgres`, or `tabular` where the scenario isn't a ranking or forecast.
Each scenario found must replay.

| Scenario | The transcript |
|---|---|
| An unscored candidate | `task_kind` `ranking`; no `error`; a warning of category `…SDMWarning` containing `left out of this ranking, unscored`; at least one payload and fewer payloads than `candidates` in its `PredictiveQuery.materialize` call; result rows; a non-empty `unscored_candidates` list |
| All candidates cold | `task_kind` `ranking`; no payloads; an `error` whose class ends `UnsupportedRequestError` and whose message contains `nothing to rank and nothing was sent` |
| Tied ranking scores | `task_kind` `ranking`; no `error`; a result row whose `rankings` hold two equal scores |
| More than 1,000 entities | `task_kind` neither `ranking` nor `forecast`; no `error`; at least two payloads; more than 1,000 result rows |
| A missing positive-class probability | `task_kind` `ranking`; an `error` whose message contains `without saying how sure it was` |
| An absent field | no `error`; a warning containing `answered without` and `which this request asked for` |
| Explanations by step | `task_kind` `forecast`; no `error`; an `SDMClient.predict` call whose `outputs` include `explanation`; a `by_forecast_step` object with at least two steps |
| A NIM 4xx body | an `error`, and a stored response whose status is 400–499 |
| A NIM 5xx body | an `error`, and a stored response whose status is 500–599 |
| A session, on each of `memory`, `duckdb`, `sqlite` and `postgres` | no `error`; calls `SDMClient.fit` and `FittedModel.predict`; result rows |
| A holdout, on each of the same four | as a session, with a `holdout` in the `SDMClient.fit` call and a warning containing `named as holdout` |

## The stand-in's answers (P8.nim-v2)

In every ranking transcript of BASE, live ones included, each result row's `rankings` hold pairwise different scores, the floats of the parts' responses differ between parts, and at least one row's `prediction` is not the first candidates in the order the `materialize` call named them.
In every forecast transcript of BASE, each row's predictions differ from step to step.
`run_corpus.py replay` passes every transcript on disk.

## Coverage (P8.coverage)

Branch coverage is measured by running, under `coverage run` with `branch = True` and `source_pkgs = sdm_client, sdm_relational_engine`, the corpus replay (`scripts/parity/run_corpus.py replay`), the fixture runner (`tests/parity/test_fixtures.py`) and the PQL corpus runner (`tests/parity/test_pql_corpus.py`), as `scripts/parity/coverage.py` runs them.

The modules are the Phase 8.5 modules above without `rfm/rfm.py`, plus `sdm_relational_engine/exceptions.py`.
`rfm/rfm.py` is left out because most of it is the sampling driver, which Phase 7.5 ports; its sites are still covered by P8.messages.

A module passes when the corpus reaches at least 90% of its branches, or when every branch it misses is a row of `tests/parity/COVERAGE_GAPS.md`:

| Module | Branch | Reason |
|---|---|---|
| `packages/sdm-client/src/sdm_client/example.py` | `140->146` | Why the corpus cannot reach it, or why it need not |

- `Branch` is `from->to` as coverage's JSON report gives a missing branch, a negative `to` being an exit.
- The reason has at least three words; the reviewer checks it holds.
- A listed branch is one the corpus misses, in a module this check measures, listed once.

## Hash seeds (P8.hashseed)

`run_corpus.py record --corpus DIR` writes the same bytes under `PYTHONHASHSEED` 1 and 4242, and those are the stored files of every offline transcript (every backend but `databricks`, `lakebase` and `snowflake`): no stored offline file is left unwritten, and nothing written is unstored.
`make_fixtures.py --check` passes under both seeds, and so do `tests/parity/test_corpus.py` and `tests/parity/test_fixtures.py`.

## A NaN ranking score (P8.nan-order)

Some fixture case's `expected` output holds a ranking: a list of at least three objects with `candidate` and `score` keys, in which one score is NaN (`{"$float": "nan"}`, or a JSON `NaN`) and at least two are numbers.
The case gives what it records today, so the order it records is today's.
