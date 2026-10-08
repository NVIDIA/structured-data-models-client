# Phase 8.5 acceptance contract

The acceptance checks in this directory, and the Rust test `rust/crates/core/tests/accept_request_fixtures.rs`, test the port of requests, decomposition and answers through a small, fixed interface.
This file is that interface.
Everything it doesn't name is the implementer's choice.

The checks build the binding's wheel from this checkout with maturin and install it into the phase venv before calling it (Phase 7's `_sampling.native`).
No call into the binding may end in a Rust panic: a panic fails the check that made it.
Today's Python, in the phase venv (pandas 3.0.6, numpy 2.5.3), is the reference throughout.

`_request.py`, beside this file, holds what the checks share: the guard, the recorder that sits on the binding, the replay of a transcript through the host, and the recordings.

## Names

- A **transcript** is any `tests/parity/transcripts/**/transcript.json`, the live ones included; they replay offline.
- A **request fixture case** is a case of any `tests/parity/fixtures/*.json` whose `operation` starts with `client.`, `decompose.`, `payload.`, `tabular.` or `wire.`.
- **Ported code** is every function, method and lambda whose innermost enclosing function or class the porting ledger sends to a core crate (`pql`, `schema`, `dialects`, `sampling`, `labels`, `request` or `core`, or `triage:` of one).
  The ledger decides twice and either is enough: `ported.tsv` beside this file (the ledger's rows when these checks were locked), and the harness's `ledger/rules.tsv` when the check runs (`$SDM_RC_HARNESS`, by default `~/.local/share/sdm-dbx/rust-core`).
- **Plain data** is `None`, `bool`, `int`, `float`, `str`, `bytes`, a `pyarrow.Table`, `RecordBatch`, `Array` or `ChunkedArray`, or a `list`, `tuple` or `dict` (with `str` keys) of plain data.
- A **warning** is written `{"category": "module.QualifiedName", "message": text}`, as in a transcript's `warnings`.

## The binding

`sdm_core._native` holds, besides what earlier phases put there:

### Runs

`materialize_run`, `predict_run`, `fit_run` and `session_predict_run`, the runs the brief names, each returning a run of Phase 4's run loop (`tests/rust_core/CONTRACT.md`), with one more kind of step.
What they take is the implementer's design, and is plain data.

`run.next()` of every run the binding returns, those of earlier phases' functions included when a host call drives them, returns a step with:

- `kind`: `run_queries`, `call_model` or `done`;
- `warnings`: a list of warnings, those the run gave since its previous step, in order.

A `run_queries` step is Phase 4's: `queries`, each with `id`, `sql` and `params`.
The host runs each query of the step, in order, as `source._run(query.sql, query.params or None)` on the source the call was given, and submits `Outcome.data(table)` with what that returned, or, where it raised, `Outcome.error(sqlstate, vendor_code, message)` with the error's `sqlstate` (when a `str`), its `errno` (when an `int`) and its text.

A `call_model` step asks the host to call the model once:

| Attribute | Holds |
|---|---|
| `id` | What `submit` takes back |
| `endpoint` | `predict`, `session_create`, `session_predict` or `session_delete` |
| `session_id` | The session's id, a `str`, for `session_predict` and `session_delete`; otherwise `None` |
| `body` | The request body as JSON text; `None` for `session_delete` |

The host sends it through the client's transport (`sdm_client.core.transport`): `predict(json.loads(body))`, `create_session(json.loads(body))`, `session_predict(session_id, json.loads(body))` or `delete_session(session_id)`.
It then submits `Outcome.response(text)`, `text` the JSON text of what the transport returned (`null` for `delete_session`), or, where the transport raised, `Outcome.model_error(text)`, `text` a JSON object of the error: `class` (its module-qualified class, such as `sdm_client.errors.NimRequestError`), `status` (its `status_code` where that is an `int`, otherwise `null`), `code` (its `code` where that is a non-empty `str`, otherwise `null`), `message` (its `message` where that is a `str`, otherwise its text) and `details` (its `details` mapping, `{}` when it has none).

A `done` step has `value`, plain data.

Where today's call refuses, `next()` raises an `Exception` that carries `warnings`, as a step does, besides what the host needs to raise today's error.

Every argument of a binding function and of an `Outcome` constructor, every value a binding function returns other than a run, every step attribute and every `done` value is plain data.

### Functions

- `flatten`, `merge` and `absent`: the pure functions the brief names, for `PredictionResult.to_pandas`, `PredictionResult.merge` and `PredictionResult.absent`, taking and returning plain data.
  `flatten`'s value is the result's **layout**: a `pyarrow.Table` (or `RecordBatch`) whose columns are the frame's, in order (below, "The layout").
- `payload_size_bytes(payload_json)`: `payload_size_bytes(json.loads(payload_json))` of `sdm_relational_engine.rfm.inference.payload`, an `int`.
  However the text is written, its value is measured: Python's `json.dumps` with `separators=(',', ':')` and `ensure_ascii`, floats by `repr`.
  Text that isn't a JSON object, or that holds `NaN` or an infinity, raises `ValueError`.
- `request_fixture_op(operation, input_json)`: the JSON text `sdm_core::fixture_op` gives (below, "Fixtures").

## The host

`scripts/parity/request_host.py` is a thin Python host of the core: what Phase 9's wrappers will be, in miniature, for the checks.
It defines:

| Name | Stands in for | Returns |
|---|---|---|
| `schema_from_source(source, **args)` | `Schema.from_source(source, **args)` | a schema handle |
| `schema_from_data(frames, **args)` | `Schema.from_data(frames, **args)` | a schema handle |
| `predictive_query(query, data)` | `PredictiveQuery(query, data)`, `data` a schema handle or a `dict` of frames | a query handle |
| `materialize(query, **args)` | `query.materialize(**args)` | a request handle |
| `request(**args)` | `Request(**args)` | a request handle |
| `Client(transport)` | an `SDMClient` that sends through `transport` | a client handle |
| `client.predict(request, *, model, outputs=None)` | `SDMClient.predict` | a result |
| `client.fit(request, *, model, outputs=None, holdout=None)` | `SDMClient.fit` | a fitted handle |
| `client.close()` | `SDMClient.close` | |
| `fitted.predict(request)` | `FittedModel.predict` | a result |
| `fitted.close()` | `FittedModel.close` | |
| `merge(results)` | `PredictionResult.merge(results)` | a result |
| `flatten(result)` | the layout `to_pandas()` is built from | the value `sdm_core._native.flatten` returned |
| `frame(layout)` | `to_pandas()`, built from the layout alone | a pandas `DataFrame` |
| `absent(result, requested)` | `result.absent(requested)` | a tuple of `str` |

Arguments are what a user would pass: frames as pandas `DataFrame`s, times as `pd.Timestamp`s, a model as its id.
A **result** has the attributes `model`, `request_id`, `metadata`, `outputs`, `unscored_candidates` and `explanation`, as a `PredictionResult` has them.
`merge`, `flatten` and `absent` also take `sdm_client.PredictionResult`s, which users can build themselves.

The host keeps these rules, and the checks hold it to them:

- **The core does the work.**
  While a host function or method runs, no ported code runs, in any thread, except beneath `Source._run`.
  Nothing goes through `Source._run` but a query of a `run_queries` step, run once, in order, before the run's `next()` is called again.
  Nothing goes through the transport but a `call_model` step's call, made once, before the run's `next()` is called again, with the step's body, parsed equal; a `DELETE` of a session is the one call the host may also make on its own.
- **Runs end.**
  Every run a host call starts has finished, with `done` or by raising, when the call returns.
- **Frames go in as Arrow.**
  Every `pyarrow` table or batch passed to a binding function is one that `sdm_core.describe.describe_frames` returned, or one the binding itself returned (the same object, unchanged), and a host call given frames calls `describe_frames` (task 6, DV19).
  `Outcome.data` takes what `Source._run` returned for that query.
- **The core speaks.**
  Every warning the host gives is one the core gave: carried by a step or a raised error, or held in a value a binding function returned (a `{"category", "message"}` object anywhere in it, JSON text included).
  The host gives each with `warnings.warn`, of the class the category names, in the core's order.
  An error the host raises is today's class (`sdm_client.errors`), with today's `code`, `message`, `details` and, for `NimRequestError`, `status_code`; its message is text the core gave (an error's text or attributes, or a returned value).
- **The pure functions are the core's.**
  `flatten` calls `sdm_core._native.flatten` once and returns its value unchanged; `merge` calls `sdm_core._native.merge` and `absent` calls `sdm_core._native.absent`.

## The layout

`flatten`'s value has the columns of the transcript's `result.arrow`, by name and in order, with the same number of rows.
Each column's type is that column's type as `tests/rust_core/sampling/driver/_driver.py`'s `type_name` names it (a dictionary by its values, any string as `string`, any list as `list<...>`), and its values are equal in that file's canonical form: floats bit for bit, NaN and null alike.
Schema and field metadata are free, so the layout may carry what `frame` needs.

`frame(layout)` is equal to today's `to_pandas()` by `pandas.testing.assert_frame_equal` with exact values: the same columns in the same order, the same dtypes, the same index and values.

## Replaying a transcript

`_request.replay_core(path)` replays one transcript through the host, as `scripts/parity/run_corpus.py replay` replays it through today's Python, with the same tape and the same stored responses:

| Call | Made as |
|---|---|
| `Schema.from_source` | `schema_from_source(source, **args)`, `source` the scenario's source, or `source.semantic_model(name)` where `semantic_model` names one |
| `Schema.from_data` | `schema_from_data({name: frame, ...}, **args)` |
| `PredictiveQuery` | `predictive_query(query, data)` |
| `PredictiveQuery.materialize` | `materialize(query, **args)` |
| `Request` | `request(**args)` |
| `SDMClient.predict`, `SDMClient.fit` | the client's `predict` and `fit`, one `Client` for the run, over the transport of `SDMClient(url='http://localhost:8000')` sending to the stored responses |
| `FittedModel.predict`, `FittedModel.close` | the fitted handle's |
| `PredictionResult.merge` | `merge(results)`, every result so far |
| `PredictionResult.to_pandas` | `flatten(result)`, then `frame(layout)` |

When the calls are done, the client is closed.
A replay passes when:

- **the statements** are the recorded steps, SQL byte for byte and parameters encoded alike, in order, every one taken (P85.sql);
- **the requests** to the model are the stored ones, by method, path and session body, every one taken;
- **the payloads** equal the stored ones, and the golden ones (`tests/golden/wire_payloads.json`) of a scenario of the same name, parsed, by value and JSON type, with `metadata.request_id` normalized (P85.payloads);
- **the error**, the result's frame, the answer (`unscored_candidates`, `explanation`) and the warnings equal the transcript's (P85.results, P85.decompose, P85.diagnostics, P85.sessions);
- the layout equals `result.arrow`, and `frame(layout)` equals today's frame (P85.results);
- the host kept the rules above.

## Recordings

Every binding call a replay makes is recorded, and `python tests/rust_core/request/_request.py capture --out DIR` writes the recordings of every transcript under `DIR`, one directory per transcript at its path under `tests/parity/transcripts/` (`#` written `--`).
The committed `tests/parity/requests/` is what it writes, under any `PYTHONHASHSEED` (P85.capture).
Each directory holds `calls.json`; the Arrow IPC files the recordings name are shared, in `tests/parity/requests/tables/`, each named by the SHA-256 of its bytes:

```
{
 "format": 1,
 "scenario": NAME,
 "transcript": "tests/parity/transcripts/...",
 "calls": [CALL, ...]
}
```

A call is `{"host": NAME, "function": F, "args": [V, ...], "kwargs": {K: V}}` and one of:

- `"returned": V`, what a function gave;
- `"raised": E`, what it raised;
- `"steps": [STEP, ...]`, for a run, each step as the run gave it, with what the host submitted.

A step is:

- `{"kind": "run_queries", "queries": [{"sql", "params"}, ...], "warnings": [...], "outcomes": [O, ...]}`, one outcome per query, in order: `{"data": V}` or `{"error": {"sqlstate", "vendor_code", "message"}}`;
- `{"kind": "call_model", "endpoint", "session_id", "body": JSON, "warnings": [...], "outcome": O}`, the body parsed, and `{"response": TEXT}` or `{"model_error": TEXT}`;
- `{"kind": "done", "value": V, "warnings": [...]}`;
- `{"kind": "raised", "error": E}`, where `next()` raised; it is the run's last.

`params` are encoded as a transcript step's are (`null` for none).
A value `V` is JSON, with:

| Tag | Means |
|---|---|
| `{"$table": SHA}`, `{"$batch": SHA}` | A `pyarrow.Table`, or `RecordBatch`, in `tables/SHA.arrow` |
| `{"$array": SHA}`, `{"$chunked": SHA}` | A `pyarrow.Array`, or `ChunkedArray`: the one column `values` of the table in `tables/SHA.arrow` |
| `{"$tuple": [...]}`, `{"$bytes": BASE64}`, `{"$int": DIGITS}` | A tuple, bytes, an integer outside int64 |
| `{"$float": "nan"}`, `"inf"`, `"-inf"` | NaN and the infinities |
| `{"$dict": {...}}` | A mapping one of whose keys starts with `$` |

An error `E` is `{"type": its class's __name__, "message": its text, "attributes": {...}}`, the attributes being its `args` and every plain-data attribute in its `__dict__`.
In every string, recorded or compared, each request id the core makes (`sdm-` and 12 lowercase hex digits) is written `sdm-<request-id>`.

**On its own (P85.independent).**
Every recording replays through the binding in a process where `pandas`, `dateutil`, `pytz`, `sdm_client`, `sdm_relational_engine`, `predictive_query`, `sdm_connectors` and `sdm_api` cannot be imported: each call, made with its recorded arguments and given its recorded outcomes, gives the recorded steps and values (tables by `_driver.py`'s canonical form, everything else by value and JSON type).

**In Rust (P85.payloads).**
`cargo test -p sdm-core --test request_replay`, a test the implementer writes in `rust/crates/core/tests/request_replay.rs`, replays every recording without Python: it reads the recordings under `$SDM_REQUEST_RECORDINGS` when that is set, and under `tests/parity/requests/` otherwise; it makes each call with the Rust API the binding wraps, from the recorded arguments, submits the recorded outcomes, and requires the recorded steps (queries, call bodies parsed equal, warnings), values and errors.
It prints the line `request_replay: N recordings, M calls`.
The check proves it compares by changing, in a copy of the recordings, a call's body, a query's SQL, a model's response, a warning, a `done` value and an error's message, each alone, and requiring the test to fail on each.

## Fixtures (P85.fixtures, P85.diagnostics)

The run loop's crate exports `fixture_op(operation: &str, input_json: &str) -> Result<String, E>`, `E` implementing `Display`.
For a request fixture case, `input_json` is the case's `input`, and `Ok` holds JSON text:

- `{"output": X, "warnings": W}`, `X` what the Python runner (`tests/parity/_fixture_ops.py`) gives, as fixture JSON;
- `{"error": {"class", "code", "message", "details_keys"}, "warnings": W}`, as a case's `expected_error` writes them;

with `W` the warnings the operation gave its caller, in order.
`accept_request_fixtures.rs` requires every request fixture case to give what it records: its `expected` output by value and JSON type, or its `expected_error` (every key it records), and its `expected_warnings` where it records them.

The sole governed correction in `fixture_corrections.json` is `request-gaps/client.run/a-tabular-target-dropped-after-its-schema`, the Phase 9 task 7b dropped-target fix.
Its complete original case (operation, input and old oracle included) remains unchanged and is bound by value and JSON type before either core runner uses the separately frozen corrected outcome.
The core must give that corrected error and ordered warnings, never the old-or-corrected alternative; every other case keeps the original comparison.
The corrected case runs exactly once and may not be excluded.
The Python acceptance additionally executes its original public fixture on the old selector in a fresh process, which must still give the original KeyError.
The original `tests/parity/test_fixtures.py` uses the same exact entry for this case only, choosing the old or corrected expectation from the actual imported product selector.
No original fixture, exclusion or minimum case count is changed.

A case whose input the core cannot be given as Arrow and content tags, or whose behavior stays in Python (a type check naming a Python type, a Python-only object), may be excluded: `scripts/parity/request_fixture_exclusions.json` is a JSON object of case ids to reasons.
A listed id is a request fixture case, listed once, with a reason of at least three words; that each reason holds is the reviewer's check (`P85.fixture-exclusions`).

Every warning and refusal site Phase 8 credited to a fixture case (`tests/rust_core/request_net/CONTRACT.md`) whose innermost enclosing symbol is ported code is credited to at least one case that is not excluded.

## Sizes (P85.size)

- 2,000 payloads generated from seed 20261023, holding text with control characters, non-ASCII, astral and escaped characters, and floats whose `repr` is awkward (exponents, subnormals, the largest double, `-0.0`, integral values), written as JSON in other spellings than Python's, give through `payload_size_bytes` what Python's gives; text holding `NaN` or an infinity raises `ValueError`.
- Frames whose relational payload is 30 MiB minus one byte, exactly 30 MiB, and 30 MiB plus one byte, by Python's count, give through the host what they give today: the first two are gathered and send the same payloads, the third is refused with today's error.

## Generated results (P85.results, P85.decompose)

- 600 `PredictionResult`s generated from seed 20261022 (ids of several types, forecast steps, probabilities, quantiles, scores, embeddings, rankings, missing fields, empty results) give through `frame(flatten(result))` what `to_pandas()` gives today, and through `absent` what `absent` gives today.
- 300 merges of them, some refused, give through `merge` today's merged result: the same error, or the same `model`, `request_id`, `metadata`, `outputs`, `unscored_candidates`, `explanation` and frame.
- 3,000 `client.ranked` inputs generated from seed 20261021 (ties, top-k from none to more than the candidates, integer, text and mixed candidates, no NaN) give through `request_fixture_op` what the Python runner gives.

## Coverage (P85.coverage)

As Phase 7.5's (`tests/rust_core/sampling/driver/CONTRACT.md`, "Coverage"), for the code the ledger sends to `request` or `core`: each module's ported branches are at least 90% taken under `scripts/parity/coverage.py`'s measure, or every untaken one is in `scripts/parity/request_coverage_gaps.json`, in the same format, each entry a ported branch left untaken with a reason.
A reviewer reads the reasons (`P85.coverage-gaps`).

## The protocol crate (P85.protocol)

- A workspace library other than `sdm-core`, with no workspace dependency and which `sdm-core` depends on, holds the protocol types: a crate depending on it alone compiles using `{Diagnostic, Outcome, Query, Run, Step}` from its root, with `Diagnostic`'s fields `class`, `category`, `code: Option<String>`, `message: String` and `details` (their types otherwise free).
- `sdm-core`, `sdm-request`, `sdm-sampling`, `sdm-schema` and `sdm-dialects` depend on it, directly or through each other.
- `sdm-core` depends on `sdm-request`, `sdm-sampling` and `sdm-schema`, and no workspace crate but the binding depends on `sdm-core`, directly or not.
- `sdm_core::{Diagnostic, Outcome, Query, Run, Step}` are the protocol crate's own, re-exported, and Phase 4's run-loop checks (`tests/rust_core/accept_loop.py`) still pass.

## MR !39 (P85.mr39)

When `tests/golden/public_api.json` no longer lists `ModelSpec`, `RuntimeContract` or `Request.decomposition` as public, the line of the harness's `ledger/rules.tsv` that decides each such symbol (the first that matches it) does not call it public in its note.
While it lists them all and `predictive_query` too, MR !39 is not in the stack: the check is skipped, and says so.

## Operation-local host callbacks (DV126)

The plain-argument rule excludes only the existing authenticated operation-local callback slots below.
Slots are zero-based positional arguments; `None` remains plain data.
All other arguments, callback arguments and returned values remain subject to the unchanged plain-data and ported-code guards.

| Native entrypoint | Slots and exact existing provider |
| --- | --- |
| `flatten` | 1 `ResultValues.infer_column`, 2 `ResultValues.column_order`, 3 `ResultValues.protocol` |
| `merge` | 1 `ResultValues.equal`, 2 `ResultValues.truth`, 3 `ResultValues.metadata_entry`, 4 `ResultValues.protocol` |
| `absent` | 2 `ResultValues.protocol` |
| `materialize_run` | 2 and 3 `SequenceValue` |
| `request_materialized`, `predict_run`, `session_predict_run` | 2 `HeldValues.protocol` |
| `fit_run` | 2 `_FitProtocols`, including its optional `sequence` `SequenceValue`; optional 3 `SequenceValue` |
| `request_op` | 3 `HeldValues.protocol` |

Provider module origins, installed helper bytes, loaded class and method code must match the current canonical source.
Callbacks remain lazy, synchronous and owned by their actual host call.
The recorder neither invokes a callback to inspect its value nor pauses the guard around it.
It retains each actual requested operation, actual plain arguments and actual return or original raised exception in ordered auxiliary `protocol_calls`, with its positional slot and optional `sequence` attribute path.
The existing sequence and result/held protocol vocabularies are closed; an unknown operation or changed owner, slot, source or code fails closed.

These are auxiliary public-capture facts, not replay expectations or new raw `calls.json` input variants.
Public replay recomputes from sealed raw recipe options, never from recorded callback results.
Callback-free raw recordings retain their existing shape.
Attempting to serialize a callback-bearing call through the raw recording encoder is refused rather than emitting an unreplayable recording.
