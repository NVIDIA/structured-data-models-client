# Parity transcript format, version 1

A transcript records one scenario run through today's Python client: how its data was made, the public API calls it made, every database step those calls took, the payloads sent to the model and what it answered, and the decoded answer or the error the run ended in.
Later phases replay transcripts through new code and require the same steps, payloads and answer.
The format is language-neutral: JSON, with database results and decoded answers as Arrow IPC files.

The JSON Schema is `tests/parity/schema/transcript.schema.json`.
The tools that read and write it live in `scripts/parity/`.

Version 1 changed in Phase 8 without a new number, because the corpus is replaced as a whole and nothing outside this repository reads it.
Since then a transcript that sent a request stores `responses`, and a replay answers from them and refuses a transcript without them; a transcript may also hold `stand_in`, `error` and `answer`.
A reader written before Phase 8 does not read these transcripts, and a transcript recorded before Phase 8 does not replay.

## Layout

Recorded transcripts live under `tests/parity/transcripts/`, one directory per scenario.
The directory is the scenario's name as a path, with `#` written as `--`: `sqlite/count` is in `sqlite/count/`, and `tabular/pql-predict-frame#default-outputs` is in `tabular/pql-predict-frame--default-outputs/`.
Each directory holds:

- `transcript.json`, the transcript;
- `step-NNNN.arrow`, the result of step `NNNN` (0-based, four digits), for each step that returned rows;
- `result.arrow`, the decoded answer.

Arrow files are written in the IPC file format, uncompressed.
A reader should also accept the IPC stream format.

The JSON is UTF-8, with one key per line and arrays of scalars, such as a payload's row, kept on one line.
Layout and key order carry no meaning.

## The transcript

| Key | Holds |
|---|---|
| `format_version` | The integer `1` |
| `name` | The scenario's name, matching `^[A-Za-z0-9_./#-]+$` |
| `backend` | `sqlite`, `duckdb`, `postgres` or `memory` offline; `databricks`, `lakebase` or `snowflake` live; `tabular` for the tabular cases |
| `task_kind` | The kind of question: `count`, `classification`, `regression`, `forecast`, `ranking`, `inactivity` or `link-prediction` (a `LIST_DISTINCT` without `RANK`, which reaches the engine whole) |
| `data` | How the data is made (below) |
| `stand_in` | How the stand-in model misbehaves for this scenario ([Responses](#responses)); absent for one that behaves |
| `calls` | The public API calls, in order (below) |
| `steps` | The database steps, in the order the run took them (below) |
| `payloads` | The prediction request bodies sent to the model (`POST /v1/predictions`), parsed, in the order they were sent |
| `responses` | What the stand-in model answered, one entry per request the run sent it, in order ([Responses](#responses)) |
| `error` | Present when the run's last call raised: what it raised (below) |
| `result` | The decoded `PredictionResult.to_pandas()` answer: the path of an Arrow file, relative to the transcript's directory |
| `answer` | What the decoded result holds that its frame does not show (below); absent when it holds nothing more |
| `warnings` | The warnings the run gave its caller, in order |

### `data`

| Key | Holds |
|---|---|
| `generator` | The name of a generator in `scripts/parity/_parity/data.py`, such as `shop` or `wire-snapshot` (below) |
| `seed` | The generator's seed |
| `namespace` | Where the tables are in the database, such as `{"schema": "parity_shop"}`; empty when the database has no namespace to name |
| `connection` | What the run read off the connection without a query, such as psycopg's `{"autocommit": true}`; a replay answers the same |
| `view` | For a live scenario over a semantic model: `metric-view` or `semantic-view` |
| `session` | The session's settings the run reads under, set once the data is loaded, such as `{"TimeZone": "America/New_York"}`: on PostgreSQL with `set_config`, on DuckDB as the connection's `config`; absent for the database's defaults |
| `column_types` | The type a column is stored as, by `table.column`, instead of the one its frame gives it, such as `{"orders.ordered_at": "timestamptz"}` or `{"orders.product_id": "DECIMAL(10,0)"}`; PostgreSQL loads into that type under UTC, DuckDB casts the column to it, SQLite declares the column with it (an empty type declaring none), and DuckDB stores `TIMESTAMPTZ` as `timezone('UTC', column)`, so a naive time, which is UTC, is the instant it names |
| `twin` | For a session-zone scenario, the name of its twin: the same backend, task kind, calls and data but for these three keys, with naive columns and a UTC session; the two must send equal payloads and decode an equal answer |

No data file is checked in.
Generators draw from numpy's legacy `RandomState` (MT19937), whose streams NEP 19 freezes.

- `shop` is a small shop: 40 `users`, 12 `products` in three categories, and a year of `orders` lines in 2024.
  Lines of one basket share a user and an hour, so `ordered_at` has ties.
  A quarter of the users stop ordering partway through the year.
- `shop-catalog` is the shop with a supplier's `catalog`, one row per product: its `sku`, which is the product's id, and its `supplier`.
- `shop-referrals` is the shop with each user's `referrer_id`, the id of another user, and `joined_at`, a day in the shop's first two months.
- `shop-categorical-keys` is the shop with `users.user_id` and `orders.user_id` held as a pandas `category` of the users' ids.
- `wire-snapshot` is the `people` and `orders` of `scripts/snapshot_wire_payloads.py`; it draws nothing, and its seed is 0.
- `marketplace` is a small marketplace of customers, items, reviews, stores and purchases, for what the shop never makes the sampler do; `marketplace-garbled` stores its purchase times as text, a few of which pandas does not read.
- `marketplace-large`, `marketplace-extras` and `marketplace-wide` are what the driver fixtures (below) ask about: 12,000 customers, text sequences and keys of two columns, and twelve more tables.
- `marketplace-untimed` is the marketplace with purchases that have no key and no time.

The tables are loaded into the database before anything is recorded, so loading takes no steps.

The session-zone scenarios, such as `postgres/count#timestamptz-new-york`, ask a question as of a midday anchor in a session whose time zone isn't UTC (`session`), over `orders.ordered_at` stored zoned or naive (`column_types`).
Each names its `twin`, such as `postgres/count#midday`, the same question over naive columns in UTC.
A zoned scenario sent other payloads than its twin before the time-zone fix, and must send the same ones since; a naive one sent the same ones before and since.
The link-prediction scenarios, such as `duckdb/link-prediction`, ask `PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) FOR EACH users.user_id`, which reaches the engine as a temporal link prediction.
`#multilabel` links `orders.product_id` to both `products` and the `catalog` of `shop-catalog`, gathers a count over that schema, then records the core's refusal of the list over a key linked to two tables (`UNSUPPORTED_REQUEST`, which today's engine ended in as `INTERNAL_ERROR`), which comes before any sampling; `#decimal-keys` stores `orders.product_id` and `products.product_id` as decimals (`column_types`).
`duckdb/regression#decimal-entity-keys` stores the product ids as `DECIMAL(10,0)`, gathers the count question for each user, then the regression question for each product and its prediction: the core reads the whole decimals as the integers they hold, where today's engine refused the task table's decimal key (`tests/rust_core/shadow/fix_changes.json`).
`duckdb/regression#decimal-mixed-indices` does the same but asks the regression question for the indices `[1, "2"]`, each of which casts to the decimal key alone but not together, and records the engine's refusal of the indices, which names none of them.
`sqlite/count#key-decimal`, `#key-numeric` and `#key-untyped` declare `users.user_id` and `orders.user_id` as `DECIMAL(10,0)`, `NUMERIC` and with no type, from none of which the engine reads a dtype.
`memory/count#self-link` links `users.referrer_id` of `shop-referrals` to `users.user_id`, gathers the count of orders for each user, then records the engine's refusal of a count of users for each user, which finds no unique foreign key from the table to itself.
`memory/count#categorical-keys` gathers the count question over `shop-categorical-keys`, then records the engine's refusal of the same question for the indices `["1", "2"]`, which the cast to the key's `category` dtype makes missing.

SQLite tables are written with `adbc_ingest`, and made again with the types `column_types` declares, DuckDB tables from registered frames, and PostgreSQL tables with `BIGINT`, `DOUBLE PRECISION`, `TEXT` and `TIMESTAMP` columns, in the schema `namespace` names.

### `calls`

Each call is `{"call": NAME, "args": {...}}`, and each acts on what the calls before it made.

| `call` | Does |
|---|---|
| `Schema.from_source` | `Schema.from_source(source, **args)`, over the scenario's database; `semantic_model` names a metric or semantic view to read instead |
| `Schema.from_data` | `Schema.from_data({name: frame for name in tables})`, over the generated frames |
| `PredictiveQuery` | `PredictiveQuery(query, schema)`, with the schema made above, or with `schema` given |
| `PredictiveQuery.materialize` | `query.materialize(**args)` |
| `Request` | `Request(**args)` |
| `SDMClient.predict` | `client.predict(request, model=model, outputs=outputs)`; the result is kept with every earlier one |
| `SDMClient.fit` | `client.fit(request, model=model, outputs=outputs, holdout=holdout)`, keeping the fitted model |
| `FittedModel.predict` | `fitted.predict(request)`, with the request made last; the result is kept like a prediction's |
| `FittedModel.close` | `fitted.close()` |
| `PredictionResult.merge` | `PredictionResult.merge(results)`, over every result kept so far, in order |
| `PredictionResult.to_pandas` | `result.to_pandas()` |

Every model call of a run goes through one `SDMClient`, which is closed when the calls are done, releasing what it pinned.

Arguments are JSON values, plus these tagged objects:

| Tag | Means |
|---|---|
| `{"$frame": {"table": T, "rows": [A, B], "head": N, "drop": [C, ...]}}` | Generated frame `T`, its rows from position `A` up to `B` (`iloc[A:B]`), then its first `N` rows, without columns `C`; `rows`, `head` and `drop` are optional |
| `{"$frames": [T, ...]}` | `{T: frame}` for each generated frame named |
| `{"$timestamp": "2024-10-01T00:00:00"}` | A pandas timestamp |
| `{"$relationship": [S, C, T, K]}` | `sdm_client.Relationship(S, C, T, K)`: the column `S.C` points at `T.K`, as `relationships=` takes one |

### `steps`

A step is an object with an `op`, one of `execute`, `ingest`, `register`, `commit` and `paramstyle`.

#### `execute`

The run executed SQL.

| Key | Holds |
|---|---|
| `sql` | The exact SQL text |
| `params` | Its parameters as an encoded value (below); `null` when there were none |
| `result` | The Arrow file holding what the run fetched, or `null` when it fetched nothing |
| `error` | `null`, or the error the call raised (below) |
| `description` | The driver's `cursor.description` after the statement (below); `null` when it gave none; absent when the statement raised |
| `fetch` | How the run fetched the result: `rows` (`fetchall`, `fetchone`, `fetchmany`), `arrow` (the driver's Arrow fetch), or `none` (an Arrow fetch the driver answered with nothing) |
| `fetch_call` | For `arrow` and `none`: the Arrow fetch the run made, `{"method": NAME, "args": [...], "kwargs": {...}}`, such as Snowflake's `fetch_arrow_all` with `{"force_return_table": true}` |
| `fetch_errors` | Fetches that raised before the result was fetched, in order, each `{"fetch": "rows" or "arrow", "call": ..., "error": ...}`; absent when there were none |

Parameters are the same whether the run passed them by position or by the driver's keyword (`parameters=` for databricks-sql, ADBC and DuckDB, `params=` for psycopg and Snowflake).
The recorder passes them to the driver as the run did.

A result fetched as rows is stored as an Arrow table with one column per result column, named as the driver described it.
A column Arrow cannot hold exactly, such as one mixing value types, is stored as text holding each value's encoding, and its field metadata says `parity.encoding: json`.
A result fetched as Arrow is stored as the driver returned it.

On replay, a fetch must be the recorded one: rows where rows were fetched, and the same Arrow method with the same arguments.
A recorded failed fetch is raised again when the run makes it, and the run's next fetch is checked against the next one.

A `description` holds one object per column: `name`, and each other DB-API field the driver set, encoded, when JSON holds it exactly (`type_code`, `display_size`, `internal_size`, `precision`, `scale`, `null_ok`).
A `type_code` JSON cannot hold, such as ADBC's Arrow types or DuckDB's type objects, is kept as text under `type_text`, and a replay gives `None` for it.
psycopg's `type_display`, such as `int8` or `varchar(4)`, is kept too, and a replay gives it as the column's `type_display` attribute.
A replay answers `cursor.description` from this, not from the result.

An error is `{"type": "module.Class", "message": text}`, with:

- `args`, the error's arguments, encoded, when they are not just the message;
- `attributes`, its public attributes that JSON holds exactly, such as the Snowflake connector's `errno` and `sqlstate`.

A replay raises the same class, built from its arguments, with the attributes set again; a class whose constructor wants more is built without it.

#### `ingest`

The run wrote a table with ADBC's `adbc_ingest`: `table`, `mode`, `data` (below) and `error`.

#### `register`

The run registered a table with DuckDB's `register`: `name`, `data` and `error`.

#### `commit`

The run called `commit()` on the connection.

Since Phase 3 the engine passes key lists as JSON parameters and writes nothing, so no recording made since holds an `ingest`, `register` or `commit` step; the format keeps them, and a replay checks them, for recordings that do.

#### `paramstyle`

The run set the Snowflake connector's parameter style to `style`.
The Snowflake backend's `run` (`sdm_connectors/backends/snowflake.py`, beneath `Source._run`) sets `qmark` for a statement with parameters and puts the caller's style back, so such a statement takes two steps around it.

#### Inline tables

`data` on `ingest` and `register` is `{"columns": [...], "types": [...], "rows": [[...], ...]}`: column names, Arrow type names, and rows of encoded values.

### Encoded values

Parameters and inline rows are JSON, with a one-key object for what JSON lacks:

| Value | Written as |
|---|---|
| NaN and infinities | `{"$float": "nan"}`, `{"$float": "inf"}`, `{"$float": "-inf"}` |
| Timestamp | `{"$datetime": "2024-01-01T08:00:00"}`, ISO 8601, with its offset when it has one; `{"$datetime": null}` for `NaT` |
| Date, time | `{"$date": "2024-01-01"}`, `{"$time": "08:00:00"}` |
| Duration | `{"$timedelta_us": 3600000000}` |
| Decimal | `{"$decimal": "12.50"}` |
| Bytes | `{"$bytes": BASE64}` |
| UUID | `{"$uuid": "…"}` |
| A mapping whose one key starts with `$` | `{"$object": {...}}` |

Integers, including numpy's, are JSON integers; floats are JSON numbers; a tuple is an array.

### `payloads`

The bodies the client's transport sent, parsed from the bytes it would have put on the wire.
The only normalized field is `metadata.request_id`, set to `<request-id>`.

### `result`

`PredictionResult.to_pandas()`, as an Arrow table without pandas metadata.
The frame's index is kept as a column only when it is not the default `0..n-1`.

### `error`

The run's last call raised, and what it raised is the scenario's outcome, compared on replay as a result is.
An error any earlier call raises fails the recording instead.

| Key | Holds |
|---|---|
| `class` | The module-qualified class, such as `sdm_client.errors.UnsupportedRequestError` |
| `code` | The error's `code` where it is a non-empty string, otherwise `null` |
| `message` | The error's `message` where it is a string, otherwise its `str()` |
| `details` | Its `details` mapping, encoded; absent when it has none |
| `status_code` | The HTTP status of a `NimRequestError`; absent otherwise |

### `answer`

What the decoded result, the last one the calls made, holds that `to_pandas()` does not show:

| Key | Holds |
|---|---|
| `unscored_candidates` | The candidates a ranking left out, encoded, in the order they were named; absent when there are none |
| `explanation` | The result's explanation, encoded, with every mapping key written as text; a forecast's is `{"by_forecast_step": {"1": ..., "2": ...}}`; absent when there is none |

### `warnings`

Each warning of a `UserWarning` class the run gave, as `{"category": "module.Class", "message": text}`, in order, repeats included.

## Responses

No NIM is reached.
The client's own transport sends each request to a stand-in session (`scripts/parity/_parity/nim.py`).
When a scenario is recorded, the stand-in answers from the request alone, and each answer is stored under `responses`.
A replay answers each request with the next stored response and computes nothing, so a reader in another language needs no hash.

A response is `{"method": M, "path": P, "status": S, "body": B}`: the request's HTTP method and path, such as `POST` and `/v1/predictions`, and the status and JSON body the stand-in answered with; `body` is `null` for a `204`.
A session call's own body, which is not a prediction request and so is not among `payloads`, is kept with its response under `request`, with only `metadata.request_id` normalized.
On replay, a request whose method, path or session body is not the next stored response's is a divergence, and so is a run that sends fewer requests than are stored.

The stand-in serves:

- `POST /v1/predictions`: a prediction, below;
- `POST /v1/sessions`: pins the request under the id `synthetic-session-N`, counting from 1, and answers `{"session_id": id}`;
- `POST /v1/sessions/{id}/predictions`: a prediction of the pinned request with the call's own sections over it;
- `DELETE /v1/sessions/{id}`: releases it, answering `204`.

A prediction (version 2 of the stand-in):

- each predicted row is answered under its id: `instance_id` on the relational wire, the stringified primary-key list on the tabular one;
- `unit(parts...)` is a number in [0, 1): the first 8 bytes, big-endian, of the SHA-256 of the parts' text joined by `|`, over 2^64;
- each draw is seeded with the question the payload asks, its `metadata.source_query`, when it has one, then the row's id: so each candidate of a ranking and each step of a forecast, asked as its own question, is answered differently;
- a regression predicts `round(100 × unit(query, id, "value"), 6)`;
- a classification weighs each class `1 + unit(query, id, "class", label)`, over the classes the context's labels hold, and predicts the heaviest; its probabilities are the weights normalized and rounded to 6 places;
- a link prediction asked for `rankings` ranks the target entities the context's label lists name, spelled as ids, each `{"id": target, "score": round(unit(query, id, "rank", target), 6)}`, highest score first (ties by id), cut at the task's `top_k` when it gives one;
- an item asked for `scores` carries `[round(unit(query, id, "score"), 6)]`, one asked for `embeddings` carries `round(unit(query, id, "embedding", i), 6)` for `i` of 0, 1 and 2, and one asked for `explanation` carries `{"synthetic": true, "weight": round(unit(query, id, "explanation"), 6)}`.

Version 1 seeded every draw with the row's id alone, so every candidate of a ranking scored the same and every step of a forecast predicted the same.

`stand_in` makes the stand-in misbehave the way a scenario needs:

| Key | Does |
|---|---|
| `seed` | `"row"` seeds every draw with the row's id alone, as version 1 did, so the parts of a question answer alike |
| `omit` | Fields left out of every item, such as `["probabilities"]` |
| `omit_at` | A list of `{"at": N, "fields": [...]}`: those fields are left out of the answer to request `N` alone, counting every request from 0 |
| `items_at` | A list of `{"at": N, "items": [...]}`: the answer to request `N` holds these items instead, each `{"from": I, "set": {...}, "drop": [...]}`, item `I` of the answer with those fields set and dropped |
| `fail` | A list of `{"at": N, "status": S, "body": B}`: request `N`, counting every request from 0, is answered with status `S` and body `B` instead |

## Recording and replaying

`scripts/parity/run_corpus.py record` loads each scenario's data, runs its calls with a recording connection, and writes the transcript.
`scripts/parity/run_corpus.py replay` runs the calls again from the transcript, with a replaying connection and no database:

- each database call must match the next step: the same `op`, then SQL text identical byte for byte, then parameters whose encodings are identical;
- the step's recorded result is what the run fetches, and its recorded error is raised again;
- the run must take every recorded step, no more;
- each request to the model must be the next stored response's, and is answered with it;
- the payloads must be equal after parsing, value and JSON type, with only `metadata.request_id` normalized;
- the error the run ended in, if any, must be the recorded `error`;
- the decoded answer must be equal, column types and values, and so must `answer`;
- the warnings must be equal.

The first difference fails the scenario and is named: a step as `step k`, a model request as `response k`, a payload by its path.

`scripts/parity/run_corpus.py reanswer` answers the model calls of stored transcripts again from today's stand-in, replaying their database steps rather than reaching a database.
It rewrites only what follows from the answers (`responses`, `error`, `result`, `answer` and `warnings`), keeps the step files byte for byte, and stores nothing for a transcript whose steps or payloads would change or that would not replay.
That is how the live recordings take a change to the stand-in without credentials; with no names, it answers every live recording.

### What the recorder adjusts

- **DuckDB's connection class.**
  DuckDB's connection is a native class whose instances cannot be made without a database, so its stand-in is a plain class, and the engine's DuckDB backend is told to accept it for the run.
  Every other stand-in is a subclass of its driver's connection class.
- **DuckDB threads.**
  The recording database runs one thread, so a query without `ORDER BY` returns its rows in one order and re-recording does not change stored results.

- **Row fetches.**
  The recorder reads a result's rows with one `fetchall`, whichever of `fetchall`, `fetchone` and `fetchmany` the run calls, and serves the run from those rows.
- **Fetches the stand-ins do not offer.**
  A stand-in cursor offers the row fetches and its driver's Arrow fetches only.
  Snowflake's `fetch_pandas_all` and DuckDB's `fetchdf` are absent, so code choosing a fetch by what the cursor has, such as `sdm_connectors.reader.read_table`, falls through to the row fetch where the real cursor would have offered them.
  The relational push-down path the corpus records asks for neither.

Nothing under `packages/*/src` is changed; each adjustment is undone when the run ends.

`record` replays each recording before storing it, and stores nothing for a scenario whose recording does not replay.
One scenario that cannot be recorded is reported, and the others are still recorded.

### Live scenarios

The Databricks, Lakebase and Snowflake scenarios need credentials to record, and one with no stored transcript is `pending-live`.
`run_corpus.py record --live` records them with the credentials `scripts/parity/_parity/live.py` lists, loads the shop into a scratch namespace first, and scrubs each recording (`scripts/parity/scrub.py`) before it is stored.
The scrubbed recording must replay, or it is not stored.
`scrub.py scan` checks every stored transcript for secrets.
A live recording holds what the warehouse answered when it was recorded, such as a Databricks table's id and creation time in `DESCRIBE DETAIL`, so recording it again changes those results even when the run's SQL, payloads and answer are the same.
Snowflake folds unquoted names to upper case, so its table scenarios read the shop written with quoted lower-case names, as every other backend has them.
Its semantic view reads a second copy of the shop written unquoted, in upper case, in the same schema, and its scenarios ask their question with the tables and columns in upper case, such as `PREDICT COUNT(ORDERS.*, 0, 30, days) FOR EACH USERS.USER_ID`.
That is because the engine reads a semantic view's quoted names with their quotes, so a semantic view over the lower-case tables can't be read.

The scrubber replaces, with `<redacted:WHAT>`:

- the values of the recording environment's credential variables;
- tokens, keys, passwords in URLs, warehouse and cloud storage hosts, bucket and container URLs (`s3://`, `gs://`, `abfss://`, ...), workspace ids, warehouse paths and email addresses, by their shape;
- values under a key that names a secret, an owner or a location: a JSON key, an Arrow column's name, a struct member or map key, or the first cell of a key-value row such as `DESCRIBE TABLE EXTENDED` returns (`Owner`, `Location`).

It reads Arrow strings at any depth: nested lists, structs and maps, dictionary-encoded columns, field names, and schema and field metadata.

`tests/parity/test_live_call_shapes.py` runs the engine's Databricks and Snowflake tables and samplers over stub drivers with the real drivers' signatures, directly and through a recording and a replaying stand-in, so the live cells' call shapes are checked without credentials.

## Fixtures

`tests/parity/fixtures/*.json` hold cases converted from test files, one file per test file, and the message fixtures, `messages-*.json`.
Each file is `{"source": PATH, "cases": [...]}`, and each case:

| Key | Holds |
|---|---|
| `id` | Unique across all fixtures: `<file>/<operation>/<label>` |
| `source` | The repository-relative path of the test file the case comes from; for a message fixture, the catalog of warnings and refusals, `packages/sdm-client/diagnostics.json` |
| `operation` | The operation to run, e.g. `dates.day_comes_first` or `labels.aggregation_type` |
| `input` | The operation's input, as JSON with the tags below |
| `expected` | The output, as JSON with the tags below |
| `expected_error` | Instead of `expected`, what the operation raised (below) |
| `expected_warnings` | Every warning of a `UserWarning` class the operation gave its caller, in order, repeats included, each `{"category": "module.Class", "message": text}`; for a case that raises, those given before it raised |

`expected_error` is an object with:

- `class`: the module-qualified class of what the operation raised, such as `sdm_client.errors.StructuredDataError` or `builtins.ValueError`;
- `code`: the error's `code` where it is a non-empty string (every `StructuredDataError` has one), otherwise its class name;
- `message`: the error's `message` where it is a string, otherwise its `str()`;
- `details_keys`: the sorted keys of the error's `details` mapping, `[]` when it has none.

The operations, and how each reads its input and writes its output, are defined by the Python runner, `tests/parity/_fixture_ops.py`; a runner in another language implements the same ones.
The runner runs each case with every warning shown (`simplefilter('always')`).
`scripts/parity/make_fixtures.py` writes the cases: each takes the inputs of a test case, and its expectation is what today's code gives; where the test states the answer, the generator checks today's code still gives it.
Outputs compare by JSON type as well as value.

Besides the converted cases, two groups hold the sampling driver's:

- `sampler-sequences.json`: `sampling.to_list`, how `Table._sanitize` reads a column of text typed as a list or a sequence.
- `driver.json`: `driver.materialize`, a question asked with `PredictiveQuery.materialize` of tables built for the case: generated as a scenario's `data` says, loaded into the case's `backend` as a recording loads them, then changed by its `setup` statements (keys, constraints and indexes the corpus's tables never have).
  The output is the number of payloads, their SHA-256 as canonical JSON with the request id fixed, and the warnings given, in order.
  Where the question is refused, `expected_error` also holds `warnings`: those given before the error, in the same form.
  `tests/rust_core/rfm/test_driver_fixtures.py` holds the Rust core to the same cases.

### Message fixtures

`scripts/parity/_fixture_messages.py` writes a case for each warning and refusal of the modules Phase 8.5 ports, and each engine raise that reaches users, as listed in `packages/sdm-client/diagnostics.json`, the way a user meets it.
Most are `client.run` cases: public calls over small frames, with the model answered by given responses.
A site no input reaches is listed, with the reason, in `tests/parity/UNREACHABLE_DIAGNOSTICS.md`, and `tests/rust_core/request_net/accept_messages.py` checks that every site is one or the other.

`client.run` takes:

| Key | Holds |
|---|---|
| `tables` | The frames the calls are made over, by name |
| `calls` | The public calls, as a transcript's `calls` are written ([`calls`](#calls)), over `tables` |
| `responses` | What the model answers, in order, as a transcript's `responses` are written ([Responses](#responses)), without the session bodies |
| `show_bodies` | When `true`, the output also holds `bodies`: every request body posted, parsed, in order, with `metadata.request_id` written `<request-id>` |

Besides a transcript's calls, it makes `Schema.from_duckdb` (`Schema.from_source` over an in-memory DuckDB holding the tables named in `load`, all by default), `Schema.link` (append a relationship), `Schema.set` (set an attribute of a column), `Schema.set_table` (set an attribute of a table), `Schema.delete` (leave a column out, as `del schema[table][column]` does), `Request.assign` (assign a half of the request), and `SDMClient.for_databricks_serving` (make the client a serving endpoint's, through a workspace client that is never called).
`SDMClient.predict`, `SDMClient.fit` and `FittedModel.predict` take `request` to send something other than the request made last, and `PredictionResult.merge` takes `parts`, positions among the results kept, to merge those alone.
Some calls change, in place, what earlier calls made, as a caller's statement between two calls does:

- `Frame.set_columns` relabels the columns of a frame of `tables` (`table`, `columns`), as `frame.columns = columns` does; labels given as lists make the labels a `MultiIndex` of those tuples.
- `Frame.drop` drops columns of a frame of `tables` (`table`, `columns`).
- `Schema.put_source_table` puts a `SourceTable` of a table's name, columns and keys into the `tables` of the schema made last (`table`), in place of the table.

A schema keeps the frames it was made from by reference, so it sees a frame changed after it.
`PredictionResult.absent` asks the last result which of the names in `requested` it lacks, and `Request.payloads` reads the envelopes of the request made last.

Its arguments use `$frame` and `$frames` as a transcript's do, over `tables`, and:

| Tag | Means |
|---|---|
| `{"$model": NAME}` | An instance of the model handle `sdm_client.models.NAME`, such as `KumoTabular` or the bare `Model` |
| `{"$schema": null}` | The schema made last |
| `{"$cyclic": KEY}` | A mapping holding itself under `KEY`, which JSON cannot write and a caller's Python can |
| `{"$inference_config": {"class": C, "fields": {...}}}` | The engine's inference-config dataclass `C` of `sdm_relational_engine.rfm.inference.config`, made with those fields, which a caller may give in place of a mapping |

It returns the requests sent, each `[method, path]`, the last frame `to_pandas()` gave, as `{"columns": {NAME: series}}`, what the last result holds beyond its frame: `unscored_candidates` and `explanation`, each when there is one, and the bodies posted when `show_bodies` asks for them.
It also returns what `PredictionResult.absent` gave, as `absent`, and the envelopes `Request.payloads` read, as `payloads`, each with `metadata.request_id` written `<request-id>`, when they were asked for.
`make_fixtures.py` records a case's `responses` by running it first against the parity stand-in (`scripts/parity/_parity/nim.py`), misbehaving as the case needs.

`request-shapes.json` holds the data shapes, options and endpoint answers no transcript holds, such as decimal and Arrow columns, text time columns, an inference config given as the dataclass, and ids an endpoint may answer under; it records the bodies posted where they are the point.
`request-branches.json` holds the cases that take branches of the request path Phase 8.5's coverage review found reachable and untaken: an inference config given as a mapping for a classification, a forecast step whose window outran the data, a key, a column or a composite key of a type the path refuses, probabilities keyed otherwise than a boolean column spells them, a ranking of 150 candidates with NaN scores, and two of the payload's value readers asked directly.
`request-gaps.json` holds the cases that take branches of the request path Phase 8.5's coverage-gap verifier found ordinary public calls reach and no recorded case took: quantiles and embeddings asked for, a bool target, a plain link prediction, a fit refused for a column's cardinality, a tabular fit with held-out keys with and without a key column, a tabular target typed as an ID, `absent` asked of a name the result does not know, rankings over a table and a column whose names need backticks, frames relabelled after their schema read them, or a column, a foreign key, a tabular target or a composite key's part dropped from them, and a `SourceTable` put into a schema over frames.
`payload.class_label` takes a `value` and the target's `dtype`, and returns the class label a relational payload spells it as.

`client.ranked` takes `candidates`, `scores` (one list per candidate, in that order, of the positive-class probability each row was answered with) and `top`, and returns each row's `row_index`, `prediction` and `rankings` as a merged ranking holds them.
It records how ties and a NaN score are ordered today.

A series is `{"dtype": D, "values": [...]}`, with `D` pandas' name for the dtype; a frame is `{"columns": {NAME: series}}`.
Besides the tags of [Encoded values](#encoded-values), fixture values use:

| Tag | Means |
|---|---|
| `{"$int": "digits"}` | An integer outside int64 |
| `{"$timestamp": ISO}` | A pandas timestamp, where a `$datetime` is a Python `datetime` |
| `{"$interval": {"left": L, "right": R, "closed": C}}` | A pandas interval with recursively encoded endpoints and explicit `left`, `right`, `both` or `neither` closure |
| `{"$pandas_timedelta": {"value": N, "unit": U}}` | A pandas Timedelta interval endpoint with its exact integer count and native unit; ordinary timedelta values retain their existing encoding |
| `{"$complex": [R, I]}` | A Python complex scalar with separately encoded real and imaginary components, preserving signed zeros and nonfinite values |
| `{"$clongdouble": [R, I]}` | A NumPy extended complex scalar with separately encoded `longdouble` components, avoiding conversion through Python complex |
| `{"$period": {"ordinal": N, "freq": F}}` | A pandas Period with its exact ordinal and frequency |
| `{"$na": null}`, `{"$nat": null}` | pandas' `NA` and `NaT` |
| `{"$np": {"type": T, "value": V}}` | A numpy scalar of type `T`, unit included: `datetime64[ns]` is a nanosecond datetime whatever `V` spells |
| `{"$ndarray": {"dtype": D, "values": [...]}}` | A numpy array; `D` is `str` for text |
| `{"$tuple": [...]}` | A tuple |

The `$np` tag uses type `longdouble` and a round-trippable scientific decimal string for extended floating scalars.
It preserves extended precision and signed nonfinite values without storing platform memory padding or converting through a Python float.

## PQL corpus

`tests/parity/pql/valid.json` and `invalid.json` hold `{"cases": [...]}`, and each case:

| Key | Holds |
|---|---|
| `id` | Unique across the corpus: `<group>/<query as a slug>` |
| `query` | The PQL text |
| `schema` | The `SchemaDefinition` it is resolved against, as `predictive_query.schema.json_serde` writes it |
| `profile` | The profile it is resolved under: `RELATIONAL`, `TABULAR`, `TIMESERIES`, `ANOMALY` or `FINE_TUNING` |
| `expected_tree` | For a query that resolves: the `ValidatedPredictiveQuery`, as `json_serde.to_json_dict` writes it, spans and types included |
| `expected_errors` | For one that does not: every diagnostic, in order, as `{"code", "span", "message"}`; `span` is `null` or `{"start_row", "start_col", "end_row", "end_col"}` |
| `expected_warnings` | Each `UserWarning` resolving the query gives, in order, as `{"category": "module.Class", "message": text}`; absent when there are none |

Resolution is `predictive_query.validate(query, schema, profile)`.
`scripts/parity/make_pql_corpus.py` writes the corpus and `tests/parity/test_pql_corpus.py` runs it.

## Dialect calls

`tests/parity/dialects/<backend>.json` hold every call into the SQL dialects that the Rust port is checked against, one file per backend: `databricks`, `duckdb`, `postgres`, `snow` and `sqlite`.
`scripts/parity/capture_dialects.py` writes them: it replays every SQL transcript, the live ones too, recording each call into a backend's `dialect.py`, into the SQL fragments of `base/utils.py` and `sdm_connectors.sql.quote_ident` it imports, into `view_expr`'s checks, and into the helpers of the tables and samplers that compute SQL or its parameters.
It then makes the calls no scenario makes, such as `holds_any`, composite keys, the scan caps and zoned time bounds, itself (`scripts/parity/_dialect_synthetic.py`).
The format, the names of the functions and the tags of their arguments are those of `tests/rust_core/dialects/CONTRACT.md`.

Each call lists where it was made: `scenario:<name>` for a transcript's replay, `synthetic:<label>` for a call the capture makes.
A statement's `params` are encoded as a transcript step's are, so every recorded step is a captured statement of its scenario.
A moved helper's `args` describe what it decides on, such as a column's dtype or the rows a catalog query returned, as `scripts/parity/_dialect_helpers.py` says; `capture_dialects.replay_helper` calls the helper again from them alone.

The capture needs no database or credentials, and writes the same bytes on every run, whatever `PYTHONHASHSEED` is.

## Coverage

`tests/parity/coverage.json` is written by `scripts/parity/coverage.py`: the branch coverage of `sdm_relational_engine`, `sdm_client`, `sdm_connectors` and `predictive_query` while the corpus replays and the fixture and PQL runners run.
Under `crates`, each core crate the porting ledger's rules name holds its modules, by repository-relative path, each with `num_branches`, `covered_branches`, `missing_branches` (`[from_line, to_line]` per uncovered branch), `num_statements` and `covered_lines`.
A module is in the crate of the first ledger rule that matches it as a whole, with `triage:` dropped.

`tests/parity/COVERAGE_GAPS.md` lists, for each module Phase 8.5 ports whose branch coverage under the same three runs is below 90%, every branch the corpus misses, with the reason; `tests/rust_core/request_net/accept_coverage.py` measures it afresh rather than reading this report.
