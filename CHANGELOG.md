# Changelog

Kept in the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) shape, and
versioned by [semantic versioning](https://semver.org) as
[`docs/reference/stability.md`](docs/reference/stability.md) defines it for
these packages. All five distributions share one version and one entry.

## Unreleased

Everything below is the first release, which all five distributions
declare as `1.0.0` and none has published. The heading becomes
`## 1.0.0 - <date>` when the tag is cut, and nothing else moves.

### Added

- The five distributions are named for the repository that publishes them:
  `structured-data-models-client`, `-api`, `-pql`, `-connectors` and
  `-relational-engine`. The modules stay `sdm_client`, `sdm_api`,
  `predictive_query`, `sdm_connectors` and `sdm_relational_engine`, so no
  import changes. Nothing had published under the earlier names.
- `docs/reference/stability.md`: the compatibility contract, saying what is public,
  what a version bump means, and the deprecation path. Enforced rather than
  stated: `tests/golden/` records the exported surface of every public package and
  the wire payloads of four canonical requests, and `tests/test_release_contract.py`
  compares them on every run.
- `RELEASING.md`, and a publish workflow using PyPI Trusted Publishing with PEP
  740 attestations and a CycloneDX SBOM. No upload token exists.
- A blocking `mypy` job. All five distributions ship `py.typed`; nothing checked
  them until now.
- Branch coverage, measured to be read rather than gated on a number.
- CI jobs that run the suite against the *installed wheel*, audit dependencies,
  and generate an SBOM.
- A contract check over the request bodies. The rest of the gate works on
  digests and the response models, so nothing read what the client actually
  sends. The recorded payloads in `tests/golden/` are now validated against the
  pinned spec, which catches a renamed key, a dropped section, a value off an
  enum, and an invented key in any section the contract declares closed. Six of
  the seventeen schemas a request reaches set `additionalProperties: true`, so
  a key invented inside one of those stays contractual and is not reported. A
  test pins that, so the check is not mistaken for coverage it does not give.
  Rules the spec states in prose rather than in a schema are not read either,
  and neither are OpenAPI's own formats, which are not JSON Schema assertions.
- `FittedModel` carries the question its context was pinned for, so a session
  can refuse a request that asks something else rather than answering the
  pinned question in its place.
- `Request.query`: the predictive query a request was built from, when it was
  built from one. It is what tells one request from another on a scoring call,
  where only the rows travel.
- Linux aarch64 wheels for the relational engine, beside x86-64, on a
  manylinux_2_28 baseline. Databricks serverless runs on either architecture
  and classic clusters on Graviton, so an x86-64-only wheel left
  `structured-data-models-client[relational]` uninstallable there. A GitLab `wheels` job builds the
  full cp310 to cp313 matrix from the same `[tool.cibuildwheel]` table the
  release reads.
- A PostgreSQL backend for the relational engine, so Lakebase and any other
  PostgreSQL can be sampled where the tables live. It came across after the
  rest of the engine had, so only the connector's reader had arrived:
  `Graph.from_postgres`, with columns, keys and foreign keys read from the
  catalog, and a sampler that fetches only the rows a prediction needs. Its
  tests run in the `engine` job against a real `postgres:17`, and fail rather
  than skip there if the server or the driver is missing.
- `SDMClient.for_databricks_serving(endpoint)`: a client for a model served on
  Databricks Model Serving, reached by the endpoint's name with the
  workspace's own identity, so a notebook supplies no URL or key. `predict()`
  works as it does against a NIM; `fit()` is refused, since a serving endpoint
  has no sessions. Needs the new `databricks-serving` extra. It sends the
  frozen `request_json`/`response_json` envelope every deployed endpoint
  reads and writes.
- Every Databricks connection and API request sdm-client opens names it as
  `nvidia_sdm-client/<version>`, as Databricks requires of a partner product.
  Prediction requests from `SDMClient.for_databricks_serving` carry it in the
  User-Agent. So do SQL warehouse connections from `warehouse()` and the
  engine's own, ahead of any `user_agent_entry` you pass. Lakebase
  connections from `lakebase()` carry it as `application_name`, ahead of any
  you pass. The `WorkspaceClient` requests these make carry it too, through a
  copy of your client rather than your client itself.
  Warehouse sessions structured-data-models-client opens also tag their statements
  `sdm_client_purpose: sampling`, beside your own query tags, so query history
  tells sampling apart. An `sdm_client_purpose` of your own is kept instead.
  A connection or Spark session you open yourself and pass to a source is used
  as it is, with neither.
  `structured-data-models-connectors[databricks]` now needs `databricks-sql-connector>=4.1.3`,
  the first release whose SEA route sends a session's query tags.
- `Schema.from_source` and the sources in `sdm_connectors`: a schema over
  tables that stay in their source, sampled there with SQL, so only the rows a
  prediction needs are read. `DatabricksSource` takes a SQL warehouse or a
  notebook's own `spark`, `SnowflakeSource` a connection or a Snowpark
  session, `PostgresSource` PostgreSQL or Lakebase, and `DuckDBSource` and
  `SQLiteSource` their own; `Source(connection)` picks the class for you.
  A type checker sees the same: each class takes, and its `connection` is,
  the driver's own type, and `Source(connection)` is typed as the class it
  returns, with no driver imported to say so. Leaving `tables` out takes
  every table the source holds. The engine could always do this; nothing on
  `Schema` reached it, so a warehouse table had to be read whole into pandas
  first. A notebook's `spark` samples on the notebook's own compute, as its
  user, with no warehouse to name; it needs `DataFrame.toArrow()`, so
  Databricks Runtime 17.3 LTS or serverless environment 4, and an older
  session is refused with the warehouse route named instead. One source per
  schema, and `Table.data` stays a DataFrame: tables in place are a
  `SourceTable`, which holds none.
  `del schema['t']['column']` leaves a column of a table in place out of both
  the read and the request.
  Every source takes the driver's own connection, or a Spark or Snowpark session, and never closes it: the caller opens it and closes it.
  `read(path)` reads a CSV or Parquet file, from a local path, a `file:` URI or an `s3://` URI, and `read_table(connection, ...)` reads a table over an open connection.
  `warehouse()` and `lakebase()` in `sdm_connectors.databricks`, below, open a Databricks connection with the caller's `WorkspaceClient`.
- `source.semantic_model(name)` on `DatabricksSource` (a metric view) and
  `SnowflakeSource` (a semantic view): a source whose tables, keys and
  relationships the model declares, read by `Schema.from_source` like any
  other. Its dimensions become columns, an expression dimension computed in
  the warehouse when it is sampled; measures and metrics are left to the
  predictive query. `Schema` itself names no warehouse.
- `sdm_connectors.databricks`: `warehouse(warehouse_id, workspace=w)` and `lakebase(endpoint, database=..., workspace=w)` open a SQL warehouse or Lakebase connection as your own `WorkspaceClient`, and return the driver's native connection for a source to take and you to close.
  Each authenticates only through that client, so a `DATABRICKS_TOKEN` in the environment cannot override it.
  `lakebase()` mints the short-lived password and looks up the endpoint's host itself; an open connection outlives the password, and a new one needs a fresh call.
  The new `lakebase` extra brings `databricks-sdk>=0.89` and plain `psycopg`, which uses the system's libpq: the `postgres` extra's `psycopg[binary]` bundles an OpenSSL that fails its FIPS self-test on Databricks' aarch64 machines and aborts the process.
- `examples/databricks/`: AdventureWorks over Unity Catalog and over Lakebase,
  sampled in place and answered by a Model Serving endpoint. Checked against
  the public API by `test_databricks_notebooks.py`; they run only in a
  workspace.
- What installing on Databricks needs: the engine's `pyarrow` floor is 10.0.1
  rather than 8, since its warehouse backends read results as `pd.ArrowDtype`
  and Databricks Runtime 14.3 LTS ships 8; `structured-data-models-connectors[databricks]` brings
  `pyyaml`, which a metric view's definition is parsed with, and
  `databricks-sdk>=0.89`, which `warehouse()` looks a warehouse up with; and
  the engine has a `postgres` extra for its PostgreSQL backend.
- A schema over tables in their source reads each table's columns, keys and
  preview once, not once per question. Each question samples through fresh
  copies of what was read, so what one writes onto its tables cannot reach
  another. Against Databricks, a demand question after the first took 4.6 s
  rather than 14.7 s. The time range is still read per question: it decides
  the anchor a question defaults to, and must see new rows.
- `time_columns=` on `Schema.from_data` and `Schema.from_source` takes
  `None`, as in `time_columns={'users': None}`, to say a table has no time
  column. Naming no column was refused before.

### Fixed

- **A schema may infer a different time column than before.** `from_data`
  and `from_source` now choose it by one internal rule, so the same rows get
  the same time column either way. It never takes a timestamp that is empty
  in more than half the table's rows, or holds one value in every row that
  holds one (unless the table has a single row), nor one whose name says
  when a row was loaded or synced, such as `_fivetran_synced` or
  `EXTRACTDATE`, nor a date of birth, such as `BirthDate` or `dob`. A stamp
  of the last edit, such as `ModifiedDate`, `updated_at` or `last_login`, is
  never taken on a table another table links to, and on one nothing links
  to, read as an append-only log of changes, only when no other timestamp
  is left. Of several left, a `created` stamp is taken, and otherwise the
  one whose values start earliest, read over the whole table on a source
  and leaving out placeholders before 1901 or at 1970-01-01T00:00:00. A
  source table with a timestamp and at least as many rows as the preview
  now costs one more query when it is described: a `MIN`/`MAX`/`COUNT` over
  the whole table for all its timestamps at once (0.18 s for 8 million rows
  on Postgres). When that query fails, as a timeout on a large view can, a
  column the rows read would refuse is kept, behind any they do not, since
  the first rows of a table kept in time order often share one day, and the
  warning names it when it is taken.
  Before, `from_data` took a lone timestamp unchecked and gave a table with
  several none, and `from_source` compared only the rows it read and never
  took a name such as `last_seen`. On AdventureWorks `customers` now gets no
  time column, where `ModifiedDate`, one date later than every order, made
  every windowed question about a customer fail, and `sales_order_details`
  gets `OrderDate` instead of `ModifiedDate`. RelBench's `rel-f1` `drivers`
  gets none rather than `dob`. One `SDMWarning` per schema of more than one
  table names every column taken as the earliest of several or as a lone
  edit stamp, and every column refused for a table left without a time
  column, each with its reason and the `time_columns=` override that settles
  it.
- **A metric view dimension keeps its string literals.** Resolving a
  dimension rewrote text inside a literal as if it were SQL: `'source.web'`
  lost its `source.`, `'customers.x'` moved a source dimension to the
  `customers` join or dropped it as referencing two tables, and a literal
  equal to another dimension's name was replaced by that dimension's
  expression. Literals are now left as written.
- **A metric view parameter named `Type`, `View Text`, `Catalog` or `Database` no longer stands in for the view's own detail.**
  `DESCRIBE TABLE EXTENDED` lists parameters before the details, and the first row of each name was read, so such a view was refused as not a metric view.
  The description is now read section by section, on a SQL warehouse and a Spark session alike.
- **A metric view joined to a table that cannot be read converts without that join.**
  A joined table that was missing, empty, unreadable or itself a metric view failed the whole conversion when a dimension referenced it, and so did one failing dimension expression or a column of a type with no graph equivalent.
  Now the join, or the column, is left out with a conversion message; only a source table that cannot be read fails the conversion, and that failure now closes a connection the conversion opened.
- **A metric view join whose key repeats is left out, rather than making one entity of several rows.**
  A join's key became the joined table's primary key unchecked, and neither Databricks nor the view checks it.
  It is now counted where the rows are, through the view's filter for the source table, with one query per join, and a key that repeats, or holds no value at all, drops the join with a message showing a repeated value.
  Rows with no key are left alone: no join reaches them, and sampling leaves them out.
- **A metric view's wildcard dimensions become columns.**
  `- expr: source.*`, `- expr: <join>.*` and either with `EXCEPT (a, b)` were dropped as missing a name.
  They now stand for the columns of their table, each named as the column, as Databricks expands them.
- **A metric view parameter is read as its default.**
  A filter or dimension referencing a parameter, such as `source.qty >= min_qty`, ran with the bare name, which no table has.
  The default is now written in as a literal of the parameter's type, checked against that type, and the expression guard runs on the result.
  A dimension referencing a parameter with no default, or one whose type has no literal, is left out with a message naming it, and a filter referencing one refuses the view.
  A source column of the same name wins over a parameter, as a SQL warehouse reads it.
- **A metric view's time column is the source column its dates truncate.**
  A view that holds a date only as `DATE_TRUNC('MONTH', OrderDate)`, or a cast of it to a date, got that truncation as its time column, since it starts at or before the column it truncates, so every windowed question counted by month.
  Each such dimension is now offered to the time-column rule as the column it truncates or casts, added to the table if the rule takes it, and a physical timestamp is preferred to an expression derived from it.
  A dimension a window measure orders by is offered first, as the time axis the view's author chose.
  The conversion message and the schema's time-column warning both name the column taken and the dimensions it was taken in place of.
- **The columns a metric view's measures aggregate are columns of the graph.**
  A measure such as `SUM(LineTotal)` left `LineTotal` out of every table unless a dimension also named it, so no predictive query could say what the measure says.
  Each column a measure reads, inside a `FILTER (WHERE ...)`, a ratio or a composed measure too, is now added to its table where missing, for measures that read one table; a dimension of the same name is kept, and the column left out with a message.
  Measures stay out of the graph, and are only read, never run.
  One conversion message lists them, with the aggregation each simple one is, such as `SUM(order_lines.LineTotal, 0, 30, days)`.
- **A metric view's names are matched without regard to case, as Databricks matches them.**
  A view naming `source.SALESORDERID` or joining `on: source.CUSTOMERID = CUSTOMERS.CUSTOMERID` over columns spelled `SalesOrderID` and `CustomerID` failed the conversion with `Column 'SALESORDERID' does not exist`.
  Dimensions, join keys and join names now match their columns and tables in any case, and a column keeps the view's spelling, the one a question about the view uses, so every read names it so; a key no dimension names keeps its table's.
- **An unqualified column in a metric view join's `on` is the joined table's, as Databricks reads it.**
  `on: source.CustomerID = CustomerID` was refused as unresolvable when both tables have a `CustomerID`, dropping the join and its entity table.
- **A semantic model gets only the relationships it declares, and a table reached by two paths samples.**
  `Schema.from_source(source.semantic_model(...))` also inferred relationships from column names, so a snowflake view's lines, holding `CustomerID`, gained a link straight to `customers` beside the path through orders, and churn questions failed with an `IndexError`.
  No relationship is now inferred beside a semantic model's joins; pass `relationships=` to add one.
  The SQL sampler also read, in one hop, rows a second path had just added for the next, so any schema reaching a table by two paths failed the same way; each hop now reads only the rows of the hop before.
- **A column linking to two tables no longer fails sampling.**
  A metric view that joins one table twice under two names, such as a buyer and a recipient both read from `customers`, links one column to two tables, and every prediction failed with `operands could not be broadcast`, since the sampler selected the column once per link.
  It is now selected once.
- **What a semantic model loses, or why it is refused, reaches `Schema.from_source` as the client says everything else.**
  The engine's conversion messages arrived as its own `ViewConversionWarning`, which filtering `SDMWarning` let through, attributed to this client's module rather than the call; they are now one `SDMWarning` at the caller.
  A model the engine refuses, for a filter on a joined table, a query as its source or any other reason, raised the engine's `ValueError`; it is now a `StructuredDataError` (`INVALID_REQUEST`) carrying the engine's message, with the model's name in `details`.
- **A metric view built on another metric view is refused for what it is.**
  A source or joined table that is itself a metric view got the advice to read it with `semantic_model`, which is what the caller had done.
  The message now says a metric view on a metric view is not supported and to point the source, or the join, at the underlying table; such a join is dropped, such a source refuses the view.
  A query as the source, or as a join's source, is still refused, now with the way out: a regular view over the query.
  A dropped join now names the graph tables and dimensions lost with it, whether it is dropped for its condition, such as one on two columns, or while converting, for a table that cannot be read, a key that repeats or a parent join dropped before it.
- **A ranking leaves out the candidates no sampled context can teach, and
  ranks the rest.** The endpoint refuses a candidate whose context holds one
  answer, usually because no entity in it linked to the candidate in the
  window, so the ranking failed after every call before that candidate had
  been sent. Every candidate is now checked first. Those that cannot be
  scored get no call and no score: they are named on the new
  `PredictionResult.unscored_candidates` and in one `SDMWarning`, and each
  row may rank fewer than `k`. A ranking with no candidate left is refused
  before any call. `docs/reference/prediction-output.md` explains why, and
  how a table of (entity, candidate) pairs scores every candidate.
- **A relational regression now reports the median, so its values change.**
  The endpoint reports the mean of the predicted distribution unless told
  otherwise, and this client dropped the engine's `output_type`, so answers
  were the mean where the engine and kumo-relational-client give the median.
  On a right-skewed target such as a count the mean is higher. Requests now
  carry `output.prediction_statistic`, `'median'` by default; pass
  `materialize(..., prediction_statistic='mean')` for the mean.
  `inference_config={'output_type': ...}` is still refused, and now names the
  option to use instead.
- A `Schema` pickled before sources were added loads again, as the schema of frames it was.
  Restoring it left out the source and the engine's reading of it, so asking for either raised `AttributeError`.
- **Foreign keys a PostgreSQL or Databricks catalog declares now link the
  tables.** `Schema.from_source` compared the engine's qualified name for the
  referenced table with the caller's bare one, so no declared key matched and
  links fell back to name matching.
- **A declared foreign key could attach to a table whose name differs only in case.**
  PostgreSQL and Snowflake hold `"Parent"` and `parent` as two tables, and `Schema.from_source` matched the referenced name without regard to case, so the key landed on whichever was listed last.
  The exact name now wins, and a case-insensitive match counts only when one table answers to it.
- **DuckDB discovery read every schema and attached database.**
  A table named like one in the current schema was listed twice, which the graph refused, and a namesake's keys and foreign keys were read as the table's own.
  Discovery and the constraint lookups now stay in `current_database()` and `current_schema()`, where a bare table name is read from.
- **A declared foreign key to a unique column, not the primary key, was dropped with a warning that blamed the dtypes.**
  When the referenced table has no primary key, `Schema.from_source` now keys it by that column, as the catalog says it identifies a row, and checks it like any other key.
  When the table has another primary key, the warning says so and names the `primary_keys=` that would let the key join.
- **SQLite timestamps stored with nanoseconds put a row on a window's edge on the wrong side of it.**
  A window bound was spelled with microseconds and zeros after them, so `.123456000` sorted before a stored `.123456789`, dropping the row at the window's end and keeping the one at its start.
  The bound now carries all nine digits before it is cut to the stored length.
- **Reading a Postgres database held a transaction open.** Without
  autocommit, psycopg keeps the first statement's transaction, and its locks,
  until a commit, so a graph kept writers to its tables waiting, and one failed
  read made every later one fail too. Each read of the Postgres backend now
  ends its own transaction, and `lakebase()` connects with autocommit by default.
- **`read_table` over a psycopg connection left it idle in transaction.** The
  connector's reader ran its query in the transaction psycopg opened for it and
  never ended that, so the caller's connection held its locks, and after one
  failed read refused every later statement. A read now ends its own
  transaction, or a savepoint inside one the caller opened, exactly as the
  engine's Postgres backend does.
- **A keyless Postgres table with a json or xml column failed to sample.**
  The tie-break that orders a keyless table on every scalar column included
  json and xml, which Postgres reads as strings but cannot order, so the whole
  neighbour query failed. The tie-break now orders every string column by its
  text, so it stays total.
- **A table with no key was sampled in whatever order the engine read it.**
  Neighbours are kept newest first, and a keyless table had nothing after the
  timestamp to order by, while the rows kept came back unordered. Rows sharing
  a timestamp, such as AdventureWorks order headers recorded by the day, then
  reached the request in a different order on Databricks serverless Spark than
  on a SQL warehouse, and the model's answer moved with it. A keyless table is
  now ordered on every scalar column after its timestamp, and the SQL samplers
  return neighbours in that order.
- **A Databricks schema of every table recorded names it could not read
  back.** Leaving `tables` out took each table's name as the engine reports it,
  which on Databricks is qualified by catalog and schema, and the next read
  asked the source's namespace for that qualified name.
- **SQLite windows lost or gained rows on their edges when timestamps kept
  fractional seconds.** SQLite compares a window bound with the stored text,
  and the bound was written without the `.000000` pandas writes through ADBC,
  so an order exactly on the end of a 30-day window was left out of the label
  and one exactly on the anchor was counted. Found by checking every sampled
  label against the same sum in pandas: one in six hundred was wrong. The bound
  is now spelled exactly as the column stores it, fraction and UTC suffix
  included.
- **The release could not have smoke-tested the engine.** The sibling wheels
  its exact pins resolve against were staged where the build container never
  looked, and the uv frontend ignored the variable pointing at them, so the
  first tag would have stopped at "structured-data-models-api was not found". Reproduced with the
  pinned cibuildwheel, and fixed in both places.
- **Eleven engine test modules never ran in CI**, fakes included, because the
  Databricks and Snowflake drivers they import were not installed; twenty more
  tests skipped for want of `databricks-sdk` and `openai`. The engine job now
  installs them, and the tests that need a real warehouse are marked
  `live_warehouse`, a marker the engine's and the client's test configs now
  register, and deselected by name instead of skipping.
- **A session tells one question from another by what it asks.** The guard
  compared the query text, which a request built from frames has not got, which
  `task=` does not change, and which two spellings of one string literal write
  differently. A frame-built request scored against a session answered the
  pinned question. The question is now assembled from the request, and
  nothing visible changes: `fit()` builds the handle, and `FittedModel` is not
  constructed directly.
- **A ranking candidate is written as its column travels**, not as its own
  value, so a small key drawn from a column whose values straddle int64 is
  still written as that column's text rather than as a number it does not hold.
- **A year-first date is recognised by its shape**, including the
  space-separated numeric offset a warehouse writes by default. A column
  written in no date shape, and one written in two, are each told apart from
  one that is ambiguous; all three used to get the ambiguity message, whose
  remedy re-dates half the column.
- **A bool target's probability keys** are restored the way its prediction is,
  so `probability_True` reads on both models rather than on whichever answered.
- **Nine notices about a schema** said things that were not true of it: the
  repeated-key notice fired on an ordinary foreign key with a remedy that
  deletes the working edge; two notices about one keyless table contradicted
  each other; a keyless table was said to reach the model unjoined; the table a
  question is about was said to be left out; a table pointing at itself was
  called an ambiguous pair; a lone table was told a window over it is refused.
  Every route that adds a relationship, `append` included, now goes through one
  place and is checked and de-duplicated alike.
- **A failure in one part of a ranking or a forecast** is described as that
  part. The wording was written for a forecast and every clause of it was false
  of a ranking; a forecast named the total rather than the failing step; and
  both dropped the exception type, losing a status and details on the one path
  every ranking and every forecast takes.
- **A window pooled from two of different units is bracketed.** Ranking two
  offsets by adding them to one fixed date orders them the wrong way round at
  every other anchor, and a bound inside the calendar window loses rows.
- **The transport reads a body, and reports a failure, one way.** A 200 carrying
  a JSON array escaped `models()` as a bare `AttributeError`; a session id of
  `..` turned a scoring call against a pinned context into a one-shot
  prediction elsewhere; a read timeout was reported as exceeded retries to a
  caller who had asked for none; and a 500 on the prediction path is no longer
  retried, since the payload and the model are the same on every attempt.
- **A connector keeps a value in the column it was read from.** The row-based
  fetch, which is the only path the postgres backend has, collapsed two result
  columns of one name. Connection defaults are taken from the environment only
  where the caller named none.
- **A request is described as what the model does with it.** A 0/1 integer
  target is spoken of on both routes, with the remedy that works on each; a
  single-class classification is remarked on rather than refused, because the
  tabular endpoint answers it; a relational context this client emptied is
  refused rather than sent; and a column of raw bytes is not called nested and
  multi-label.

- **A column the model cannot read is named rather than silently dropped.**
  A `text` feature column reaches the relational model and is ignored there,
  because the model has no text processor. Holding content fixed and varying
  only the semantic type, the same determining column scores 1.00 as
  `categorical` and chance as `text`. The request still succeeds, because a
  model that answers from graph structure alone should not refuse over an
  unreadable attribute, but an `SDMWarning` now names each `text` feature
  column a request for `KumoRelational` carries. It says to type the column
  categorical if its values are labels rather than prose, or to leave it out.

- **A time column the caller never named can no longer be acquired.** The
  engine guessed keys and time columns from column *names*, and the guess could
  fail while it was being made: a string column of dates called `release` was
  parsed during graph construction and overflowed, before any schema could undo
  it. Nothing is inferred now; the schema decides every key, time column and
  semantic type.

- **Selecting no rows is the same no-op however it is written.** `indices=[]`
  was refused while an empty `predict` frame was answered with zero rows, so a
  filter that matched nothing had to be handled two ways. Naming no candidates
  to rank stays refused: that is a question with no answer, not an empty one.

- **A failed request no longer names a cause the client cannot know.** Every
  5xx and every dropped connection reported GPU memory pressure. Measured, the
  GPU was at one percent while that appeared, six identical retries failed, and
  only a smaller neighbourhood sample succeeded. The NIM's own detail is quoted
  instead, and both remedies are named.

- **A binary target asked through a predictive query is a classification.**
  `PREDICT users.churn FOR EACH users.user_id`, over a column of 0 and 1, was
  gathered as a regression: 0 and 1 are integers, integers are numerical, and a
  numerical target meant regression. The endpoint answered it with continuous
  values near 0.5, which read like probabilities and are not. Nothing warned.

  A target holding exactly two whole numbers is now asked as a classification,
  and the client says so. It is the values that decide, not how pandas holds
  them: a 0/1 column with one missing value arrives as `float64` and is still
  a label. Only that case changes: three whole numbers may be a count, and a
  fractional pair like 0.5/1.5 is not a label, so both remain regressions.

  **This changes what goes over the wire** for such a request. `task.kind`
  becomes `classification`, and the target column is declared `categorical`
  rather than `numerical`, which is the correction the tabular adapter already
  made for classification targets and could not make while the task was wrong.
  If you were relying on the old reading, pass `task='regression'` to keep it.

  **Say plainly what this costs:** the rule is two whole numbers, so a genuine
  count that happens to hold two (`items_bought` in {0, 1}, `visits` in
  {1, 2}, `rating` in {7, 9}) is now asked as a classification and its request
  moves on the wire. That is a change of default rather than a repair: those
  targets were being read correctly before. The warning names the other
  reading and `task='regression'` restores it, and the judgement is that
  reading every 0/1 label as a quantity was the more common and the more
  damaging mistake.

- **A context that cannot teach is refused where the endpoint would refuse it.**
  Score every row of one class and the context left behind holds only the
  other. A classification over that is rejected by the endpoint
  (`TASK_SCHEMA_MISMATCH`), so it is now rejected locally in the same terms,
  before a table is encoded and sent. A regression over a constant context is
  accepted by the endpoint, so it is still only warned about. Previously the
  same degeneracy produced a working request on one path and a guaranteed
  server error on two others.
- **`materialize()` takes `task=`.** A query names a target, not what is asked
  of it, and the single-table path had no way to say. It is authoritative, and
  stating it silences the warning above. A relational query resolves to its own
  kind in the engine, so passing `task=` with one is refused rather than
  silently ignored.
- `predictive_query` attached no logging handler, so its warnings reached the
  stderr of any application that had not configured logging. It now installs a
  `NullHandler`; applications that *have* configured logging still receive the
  records, unchanged.
- `structured-data-models-api`, `structured-data-models-pql` and `structured-data-models-client` each declared their version in two
  places. There is now one literal per distribution, in `_version.py`, and a
  test that the five agree.
- The four pure-Python distributions admitted Python 3.14 through
  `requires-python` but did not classify it. They now do, and the narrower
  range of the `[relational]` extra, which is a native build, is documented
  where users meet it rather than surfacing as an unexplained pip resolver
  error.
- **An entity table whose time column starts after the anchor now raises
  `NoEligibleEntityError` instead of returning an empty label table.**
  `get_train_table()` reads an empty sample as an answer of no labels, which
  it is when the query's filter selects nothing, and it matched on the opening
  of the sampler's message. That opening also carries the case where no entity
  existed at the anchor, so a time column recording an edit rather than a
  creation came back as a legitimate empty table and nothing said to look at
  it. The new error is a `ValueError`, so a caller already catching one is
  unaffected.
- **`Graph.from_relbench` failed for every dataset with a 404.**
  It read RelBench's `hashes.json` from upstream's `main`, which RelBench 3 deleted when it moved its data from `relbench.stanford.edu` to Hugging Face.
  The loader now reads the tables from those Hugging Face repositories at a commit pinned in the package, and checks each against a SHA-256 recorded with it, so what loads cannot change upstream unseen.
  `scripts/pin_relbench.py` writes the pins, and `rfm/assets/VENDORED.md` says how the digests were obtained.
  Downloads land in a new cache directory, so a dataset fetched before is fetched once more, and the old `<cache>/rel-*` folders (and the other archive folders beside them) can be deleted.
  `rel-mimic`, `tgbn-genre`, `tgbn-reddit` and `tgbn-token` are no longer available, as RelBench does not publish them on Hugging Face.
- **A Databricks metric view named as a table failed with the warehouse's `METRIC_VIEW_MISSING_MEASURE_FUNCTION`.**
  `Schema.from_source(DatabricksSource(...), {'m': '<metric view>'})` read its rows like a table's, and Databricks refuses that for a measure.
  The column look-up now reads the relation's type from `information_schema` too, so a metric view is refused before any row is read, with a `StructuredDataError` (`INVALID_REQUEST`) that says to use `source.semantic_model('<metric view>')`.
  The engine raises the new `MetricViewAsTableError`, a `ValueError`, pointing to `Graph.from_databricks_metric_view`.

### Behaviour worth knowing before you rely on it

**`materialize()` takes the model it is gathering for.** It used to pick by
counting tables, and the model was named later at `predict()`. For a one-table
schema those two decisions could disagree with no way to reconcile them: the
request was always built for `kumo-tabular`, so `predict(model='kumo-relational')`
refused it and no argument could say otherwise. `materialize(model=…)` now takes
the same handle or id `predict()` does. Left unsaid, the shape still decides,
so existing calls are unchanged: one table with no relationships is tabular, and
anything wider is relational. That now counts a relationship as well as a second
table, matching `Request.is_relational`. A lone table pointing at itself used to
be gathered tabularly and then refused by the model for needing relationships.

Gathering and sending for different models is now refused where it used to raise
a bare `KeyError` from inside the adapter: a request carrying relational payloads
is not shaped into a tabular envelope, and the message names the model it was
gathered for.

**`id` holds the key you asked about, on both models.** The relational path
already returned the caller's own key; the tabular path returned what the
endpoint echoed, which is the stringified list of primary-key values, `'[1]'`
where you asked about `1`. Both now return `'1'`, so a result joins back onto
your rows whichever model answered. As with the relational path, a response that
names a row outside the batch, or answers one twice, is refused rather than
silently re-keyed. A composite key, or rows supplied without the key column, are
left as the endpoint sent them: there is no single value to put back.

A session answers the same way, which needed the adapter to be handed the whole
request rather than the half that just went over the wire. That also settles a
latent one: a relational session could not tell a forecast's repeated rows from
a duplicate answer, because the task kind was pinned at fit time and the adapter
only ever saw the scoring call. The wire is unchanged: a session's scoring
call still carries only what the contract allows there.

**A `DECIMAL` column travels as the number it is, and a fractional one rounds.**
A warehouse `DECIMAL`/`NUMBER` arrives as `Decimal`, which is exact and
unbounded, and the wire has no such type. So: a whole-number decimal is sent as
`int64`; one too wide for `int64` is sent as digits in a string, which keeps
every one of them; and a **fractional** decimal is sent as `float64`, which
rounds it. Below 2**53 that rounding is ordinary float behaviour and is
accepted silently. Above it a float cannot hold even the integer part, so the
value is refused rather than quietly mangled. If you need a fractional decimal
kept exactly, send the column as strings.

This applies however the column reaches us. Read through Arrow (which the
connectors prefer, because it is the path that keeps a wide integer exact) a
`DECIMAL` used to be announced as `string`, so an id column stopped being an id
to the model. It is now named the same way on both paths.

**A CSV column with blanks keeps its type instead of becoming `float64`.**
`read` on a local CSV now reads through pandas' nullable dtypes, so an
integer column containing a blank comes back `Int64` rather than being promoted
to `float64` and losing exactness past 2**53. Columns with no nulls are
narrowed back to the plain numpy dtypes, as `read_table` does whether a driver
returns Arrow or plain rows, so the common case is unchanged, but a
column **with** nulls now arrives as `Int64`, `string` or `boolean` where it
previously arrived as `float64` or `object`. Code that tests `frame.dtypes`
against numpy types for such a column will see the difference.

**`Schema.from_source` checks every key against every row, so a schema that built before may now warn or raise.**
Databricks and Snowflake do not enforce a declared primary key, and a key inferred from a 1,000-row preview has only been seen to hold there.
A repeated value used to pass silently: a scored id matched one of its rows at random, so a prediction changed between seeds, and the context could hold one entity twice at one anchor time, with labels that disagreed.
The rules are the ones `Schema.from_data` applies to a frame.
A declared key, or one named in `primary_keys`, warns when up to 1% of rows repeat a value and is refused above that, naming the count, an example and what to do instead.
A key named in `primary_keys` is refused for a row with no value in it too; a declared or inferred one is warned about.
An inferred key that repeats more than 1% is declined, with the warning `from_data` gives.
The count is one `UNION ALL` statement per source, a full scan of each table, about 0.8 s per AdventureWorks table on a cold SQL warehouse; approximate counts were too inaccurate to judge 1% by.
A key the source keeps unique and never null is not scanned at all: a PostgreSQL or DuckDB primary key, a SQLite `INTEGER PRIMARY KEY`, and a Snowflake hybrid table's primary key.
A key the source keeps unique but lets be null is still scanned, for its nulls alone: a unique constraint or index on a nullable column, and every other SQLite primary key, since SQLite lets one hold a null.
A semantic model's tables are not counted, since the model derives them.
`check_keys=False` skips the scan and every check with it.
`SourceTable.duplicate_keys` now holds what the count found, and `None` where nothing was counted, rather than always zero.
Sampling holds the line too, whatever was checked: an id asked to be scored that more than one row holds is refused, and context sampling keeps one row per key, so no entity appears twice at one anchor time.

**A semantic model's expressions may call only standard SQL functions, unless you trust it.**
A metric view or semantic view is written by someone else, and its expressions run as SQL under your credentials every time a request is gathered.
Any function call used to pass, so a view could make structured-data-models-client call a user-defined function, `secret()`, `http_request()`, `ai_query()` or `SYSTEM$WAIT()` as you.
Now a column whose expression calls anything but a pure, deterministic built-in function is left out, with a conversion message naming the function.
Qualified and quoted function names, `IDENTIFIER()`, and session values such as `current_user` and `current_date` are left out the same way.
`source.semantic_model(name, trust_expressions=True)`, and `trust_expressions=True` on `Graph.from_databricks_metric_view` and `Graph.from_snowflake_semantic_view`, keep such a column if you trust whoever authored the view.
Sub-queries, comments and statement separators are refused either way, and that check now also catches the forms Databricks runs without `SELECT`: `TABLE t`, a query opening with `FROM`, and the `|>` pipe.
It also refuses parentheses that do not balance outside quoted text, since a stray `)` could close the query the expression is spliced into.
Some allowed built-ins are costly or read the session time zone; `docs/about/architecture.md` lists them.
A keyword before `(`, as in `x IN (1)`, counts as grammar only after something known to be a value: a literal, a closing bracket, quoted text, a qualified name, or a column of the table the expression runs on.
Databricks lets almost every keyword name a function, so `WHERE from(y)` or `CASE WHEN then(x)` calls one, and a word the check cannot place now leaves the column out instead.
A column named like a keyword that takes an expression next, such as `where` or `then`, is never taken for a value; one named `region`, `date` or `year` still is.
`zone` is treated the same way out of caution: no warehouse parses `AT TIME ZONE` today, but Databricks keeps the word as a keyword.
Bare `user` and `current_path` are left out like `current_user`, since Databricks answers both with the session's identity and search path.
On Databricks a session word that also names a column of the table reads that column, so it is kept, except `current_path`, which Databricks answers with the search path even then.
Snowflake keeps refusing every session word, column or not.
Each table's columns are read before its expressions are checked, and the metric view's `filter` is checked against the source table's columns before any row is read.

**A metric view's `filter` is applied, or the view is refused.**
The filter decides which rows a metric view holds, and it used to be reported as unrepresentable and dropped, so a schema and every request read rows the view leaves out.
A filter on the source table's own columns now limits every read of that table: its preview, its row count, its time range, entity sampling and each lookup read `(SELECT * FROM <source> AS source WHERE (<filter>))` instead of the table.
A filter that references a joined table has no single table to apply to, so `Graph.from_databricks_metric_view` refuses the view with a `ValueError`, and `Schema.from_source` with a `StructuredDataError` (`INVALID_REQUEST`), as they do a filter with a sub-query or a call to a function other than a pure built-in.
The message names the way out: create a regular view with that `WHERE` clause and point the metric view's source at it.
A parameter the filter or a dimension references is read as its default, as a query of the view without arguments reads it, and a filter on a parameter with no default refuses the view.
`materialization` is still left out with a message.

**`predictive_query` is not public API.**
`structured-data-models-pql` installs it because `structured-data-models-client` parses and validates PQL with it, and any release may change it.
Write PQL with `sdm_client.PredictiveQuery`.
`ModelSpec` is not exported either, since it carries a `predictive_query.Profile`, and `KumoTabular` and `KumoRelational` have no public `spec` attribute.
`RuntimeContract`, the part of a `ModelSpec` that describes a deployment, is internal too, since nothing public takes or returns one.
`Model.name` and `SDMClient.capabilities()` describe a model.

**Nothing public hands out a type that is not public.**
`Request.decomposition` is gone: it returned the client's own record of how a ranking or a forecast is asked in parts, and `Request.parts` and `Request.asked` say what was asked.
`FittedModel` is returned by `SDMClient.fit()` and is not constructed directly, since what it is built from is internal; calling the class raises `TypeError`.
`tests/test_release_contract.py` now fails when a public signature, dataclass field, property or class attribute names a type this project does not export.
Two exceptions remain and are listed there with their reasons: `PredictionResult.items` holds the contract's generated `PredictionItem`, and a `DatabricksSource` built from a Spark session returns the session wrapped as a connection.

### Notes for a reviewer

**Graph visualization is kept, and so is its provenance.** `Graph.visualize()`
renders a diagram offline because the notebook environments this client is used
from, Snowflake and Databricks, block external scripts, so a diagram that
fetched from a CDN would not draw at all. The mermaid bundle that makes that
possible is vendored as upstream bytes with its SHA-256 pinned, its components
enumerated in `LICENSE`, and a test that re-checks the digest. That test exists
because a repo-wide rename once rewrote two identifiers inside the minified
file and silently turned a vendored artifact into a modified one.
`Graph.to_mermaid()` returns the diagram source for anyone who would rather
render it themselves.

**The relational engine's example scripts and notebooks were removed.** They
came across with the port written against `RelationalClient` and
`client.relational(...)`, an interface this client does not export, and they
set a `run_mode` that is no longer a knob. `python
scripts/check_repo_hygiene.py retired` now fails on any shipped file
naming a retired interface, notebooks
and per-package example directories included. The one quickstart this
repository ships was written against the shipped API and every cell was
executed against a live `kumo-tabular` and `kumo-relational` NIM.
