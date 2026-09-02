# Bugs

Findings from an end-to-end QA pass against both live NIMs, real public
datasets (UCI Adult, UCI Wine Quality, Titanic, Northwind) and real
warehouses (Snowflake, Databricks, S3).

Each file states the problem, where it is, why it matters, and the fix. Every
one was reproduced before it was written down, and then re-examined against a
single test — **does the repository itself already say this should be
otherwise?** Two reports failed that test and were deleted; two more were
re-scoped. What is left is seventeen, and all seventeen are fixed on this branch.

The reports are kept as the record of what was wrong and why each fix is shaped
the way it is; the probe that found each one, in [`../qa`](../qa/README.md), is
its regression test. Where a fix differs from the proposal the file started
with, the file says so and why. The last of them,
[18](018-an-integer-entity-key-can-be-declared-float64.md), was found after the
rest by testing the built wheels rather than the source tree — a reminder that
the check which finds a bug is usually the one nobody had run yet.

| # | Title | Severity |
|---|---|---|
| [1](001-decimal-columns-lose-precision.md) | A `Decimal` column loses precision, and walks past the guard that exists to stop it | High |
| [9](009-csv-reader-rounds-large-integers.md) | The CSV reader rounds large integers when the column has a blank | High |
| [14](014-a-sum-over-integers-in-duckdb-is-rounded.md) | A `SUM` over an integer column in DuckDB comes back rounded | High |
| [15](015-the-schemas-relationships-are-additive-only.md) | A schema's `relationships` are additive only, so the model sees a graph the caller did not describe | High |
| [18](018-an-integer-entity-key-can-be-declared-float64.md) | An integer entity key can be declared `float64`, and the endpoint refuses the request | High |
| [19](019-a-list-valued-column-crashes-or-is-stringified.md) | A list-valued column crashes in one place and is silently stringified in another | High |
| [2](002-non-unique-primary-key-accepted.md) | A primary key that is knowingly not unique is used to join and correlate, and nothing says so | Medium |
| [3](003-engine-exceptions-escape-the-error-surface.md) | Engine exceptions escape the SDK's error surface, one advertising an API this SDK does not have | Medium |
| [4](004-uint64-declared-as-int64.md) | A `uint64` column is declared `int64` while carrying values `int64` cannot hold | Medium |
| [5](005-an-empty-result-has-no-columns.md) | An empty `predict` frame returns a result with no columns at all | Medium |
| [6](006-s3-parquet-prefix-needs-a-trailing-slash.md) | A Parquet prefix on S3 is refused unless it ends in `/` | Medium |
| [8](008-endpoint-limits-are-not-declared-locally.md) | One endpoint limit is not declared locally, and one refusal does not name the column it is about | Medium |
| [10](010-all-null-context-column-stringifies-real-values.md) | A column empty in the context turns real values into strings in the rows being scored | Medium |
| [11](011-numpy-scalars-in-inference-raise-a-raw-typeerror.md) | A numpy scalar in `inference` raises a raw `TypeError` from the JSON encoder | Medium |
| [17](017-schema-checks-that-a-column-exists-not-that-it-fits.md) | `Schema`'s table attributes validate nothing, where the class they wrap validates on assignment | Medium |
| [7](007-materialize-prints-to-stdout.md) | `materialize()` prints progress to stdout | Low |
| [13](013-a-malformed-response-is-coerced-rather-than-refused.md) | A malformed `200` response is coerced rather than refused | Low |

## What was rejected

Two reports were written and then deleted, because neither survived the test
*"does the repository itself already say this should be otherwise"*.

- **An explicit `pd.Categorical` becomes `text` at 65 levels.** Surprising, but
  the relational engine does the same thing, no test or doc says the pandas
  dtype should be authoritative over cardinality, and the threshold's comment
  explains the reasoning. That is a design preference, not a defect. Its one
  real consequence — the endpoint then refusing `text` — is recorded in
  [8](008-endpoint-limits-are-not-declared-locally.md).
- **`visualize()`/`to_mermaid()` are unreachable.** They are not. The README
  says `Graph.visualize()`, `sdm_relational_engine.rfm.Graph` is exported, and
  it works. The genuine observation — that the only obtainable picture shows the
  fabricated edge — belongs to
  [15](015-the-schemas-relationships-are-additive-only.md), where it now is.

Two others were re-scoped rather than deleted. The `_KEY_UNIQUENESS = 0.99`
threshold in [2](002-non-unique-primary-key-accepted.md) is deliberate and
well-argued, so that report is now about the silence rather than the threshold
— and about the fact that the engine already counts the duplicates and the
client throws the count away. And the contract puts no `minItems` on `rows`, so
a zero-row `predict` is legal and
[5](005-an-empty-result-has-no-columns.md) is about the shapeless result
instead of the acceptance.

Two of the fifteen ([8](008-endpoint-limits-are-not-declared-locally.md) and
[17](017-schema-checks-that-a-column-exists-not-that-it-fits.md)) looked at
first like they contained a judgement call. The open-source predecessor,
`kumo-relational-client`, settled both, and neither hedge survived:

- **8** — the predecessor declares `_MAX_PRED_SIZE`, `_MAX_TEST_SIZE`,
  `_MAX_CONTEXT_SIZE` and `_MAX_SIZE` locally, and **nothing else**: sizes are
  the client's business, what a model can represent is the server's. So the
  class ceiling belongs in `limits` and a `semantic_types` field does not — that
  proposal is withdrawn. What is left of the `text` half is that the refusal
  does not name the column, which the engine already has a named pattern for
  (`_cardinality_guidance`) and simply was not applied here.
- **17** — the engine's `Table.primary_key` and `Table.time_column` *are*
  property setters that validate on assignment, and `Column.__setattr__`
  rejects an stype the dtype cannot support. All of it is in this repo, one
  package over, and in the predecessor unchanged. `sdm_client.Table` is a plain
  dataclass with bare attributes, so this is a regression rather than a
  question of how strict to be — the rule is "whatever the engine will reject
  anyway", which leaves every re-typing the docs invite still legal.

## Themes

Four of these are the same defect wearing different clothes. The wire layer has
a rule — an integer that `float64` cannot hold exactly must not travel as a
number — and it enforces that rule correctly for a Python `int`. A `Decimal`
walks past it ([1](001-decimal-columns-lose-precision.md)) because the guard
tests `Integral`; a CSV blank defeats it before the guard runs
([9](009-csv-reader-rounds-large-integers.md)) because pandas has already
promoted the column; a DuckDB `SUM` defeats it in the Arrow conversion
([14](014-a-sum-over-integers-in-duckdb-is-rounded.md)) because the repair
written for that hazard does not cover decimals; and a `uint64` satisfies it
while still exceeding the dtype being declared
([4](004-uint64-declared-as-int64.md)). Fixing one does not fix the others, and
the rule is only as strong as its weakest path.

What was checked and *is* exact, for contrast: Parquet `Int64` with a null,
Parquet `int64`, DuckDB `BIGINT` with a null, SQLite `integer` with a null, CSV
without a blank, Snowflake `NUMBER(38,0)`, Databricks `BIGINT`, and a Python
`int` list. The repair works; it just has holes.

Two more are the same shape: the SDK works out something careful and then
discards it one function later — `UNTYPED` for a column with no values
([10](010-all-null-context-column-stringifies-real-values.md)), and the
schema's own relationships, overruled by a second inference inside the engine
([15](015-the-schemas-relationships-are-additive-only.md)). The second is the
most serious thing in this list: `Schema` is the object the SDK asks you to
trust, and on Northwind it reports three relationships while four are sent.

And four are error-surface holes, all failing the same sentence in
`errors.py` — *"Base class for every error this client raises"*: engine
exceptions ([3](003-engine-exceptions-escape-the-error-surface.md)), a schema
edit whose incompatibility surfaces later as a `ValueError` or a bare
`KeyError` ([17](017-schema-checks-that-a-column-exists-not-that-it-fits.md)),
a JSON encoder `TypeError`
([11](011-numpy-scalars-in-inference-raise-a-raw-typeerror.md)), and a response
coerced instead of refused
([13](013-a-malformed-response-is-coerced-rather-than-refused.md)).

## What was checked and is clean

Worth recording, because it is most of the surface. Scoped precisely: where a
claim has an exception, the exception is named.

**Connectors.** Local Parquet (file and directory) and gzip; S3 Parquet as an
object or a prefix **with** a trailing slash ([6](006-s3-parquet-prefix-needs-a-trailing-slash.md)
is the prefix without one); CSV **without a blank in an integer column**
([9](009-csv-reader-rounds-large-integers.md) is the one with); SQLite
including a table name with a space and a quoted identifier; DuckDB over
Parquet and in memory, and `BIGINT` including with nulls (but not `sum()`,
which is [14](014-a-sum-over-integers-in-duckdb-is-rounded.md)); missing file,
missing key, missing bucket, unknown suffix, unknown backend, unknown keyword,
wrong connector for a path in both directions; `table=` and `query=` argument
handling.

**Identifier safety.** Injection through `table=` refused before the driver on
SQLite, Snowflake and Databricks — the sentinel table survives in all three,
confirmed by re-reading `information_schema` afterwards. Snowflake case
folding, quoted mixed-case identifiers and reserved words; Databricks
three-part names and case preservation.

**Integer exactness.** Six of the routes into the SDK are exact and are pinned
by `qa/probe_connectors.py`: Parquet `Int64` with a null, Parquet `int64`,
DuckDB `BIGINT` with a null, SQLite `integer` with a null, CSV with no blank,
and a Python `int` list through `data=` **with no `None` in it**. The five that
are not exact are [1](001-decimal-columns-lose-precision.md),
[4](004-uint64-declared-as-int64.md),
[9](009-csv-reader-rounds-large-integers.md) (CSV and `data=` alike) and
[14](014-a-sum-over-integers-in-duckdb-is-rounded.md).

**Warehouses.** Snowflake connect, read, and a real business table scored end
to end; a semantic view described, queried through `SEMANTIC_VIEW()` and its
metric confirmed exact; `NUMBER(38,0)` past the JS-safe range travelling
exactly — Snowflake's driver returns it as `int64`, which is why it escapes
[1](001-decimal-columns-lose-precision.md). Databricks connect,
`samples.tpch` read and scored, `BIGINT` past the JS-safe range exact,
three-part names, metric views listed, refused without `MEASURE()` with the
server's own message intact, and read and scored **with** `MEASURE()`.
Databricks `DECIMAL` is *not* clean — that is
[1](001-decimal-columns-lose-precision.md).

**Tabular model.** Binary classification (Adult), multiclass (Wine, 6 classes,
probabilities summing to 1), regression, nulls and bools (Titanic), category
dtype, single row, single feature, one context row, output selection, an
unsupported output refused locally, `row_index` joining one-to-one, a shuffled
pandas index, sessions (reuse, after close, two at once), the predictive-query
path, an all-null target refused, a single-class target, an unknown column
named, a duplicate column name refused. A column that is all-null **in both
frames** is fine; all-null in the context with values to score is
[10](010-all-null-context-column-stringifies-real-values.md).

**Relational model.** Northwind schema inferred from frames alone — three
relationships, right way round, including the table whose name contains a space
(what is then *sent* is [15](015-the-schemas-relationships-are-additive-only.md));
classification and both temporal aggregations (`COUNT`, `SUM`);
`num_context_samples` honoured exactly at 5/40/300; `num_neighbors` reaching
the sampler; an anchor time inside the data; an explanation refused for more
than one entity; sessions; string primary keys correlating back; entity values
never travelling raw.

**Error contract.** Unreachable host, sub-millisecond timeout, wrong model for
the endpoint, a credential in the query string, a credential in the userinfo
over plaintext to a non-local host — all `StructuredDataError` with a code.
`float('inf')` refused before the wire (`Decimal('Infinity')` is not, see
[1](001-decimal-columns-lose-precision.md)). Tz-aware and naive timestamps
normalising to the same instant. A session given a mismatched schema is refused
by the endpoint as `422 SESSION_PREDICT_SCHEMA_MISMATCH`, naming the offending
column — a round trip rather than a local check, but a clean outcome, so it is
not filed.

## What a third pass would look at

Both passes exercised constructors and `materialize()`. The class that nearly
escaped both is **mutation after construction** — `schema['t'].primary_key = x`
is the documented way to correct a schema and runs no checks at all, not even
that the column exists. That is now [17](017-schema-checks-that-a-column-exists-not-that-it-fits.md),
found by asking what shape of defect the method itself could not see.

What is still unexamined here is release shape rather than behaviour: a clean
install from built wheels, the optional extras resolving on their own, the
native engine on a platform other than this one, dependency metadata, and drift
between this client's pinned `RuntimeContract` and whatever the endpoint serves
next. Those want release smoke tests, not more probes.

## Method

Two reviewers, working against each other. One executed live probes against
both NIMs and the real warehouses; the other read the code and predicted what
would break. Each finding here was predicted by one and reproduced by the
other, or found live and then traced to a line. Several early candidates died
that way — a Snowflake "injection" that turned out to be a `_` wildcard in the
probe's own `ILIKE`, a "userinfo accepted" that turned out to be the deliberate
localhost exemption, a "`num_neighbors` does nothing" that turned out to be the
probe counting the wrong dictionary. Those are not in this list, which is the
point of running it twice.
