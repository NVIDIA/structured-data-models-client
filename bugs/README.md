# Bugs

Findings from an end-to-end QA pass against both live NIMs, real public
datasets (UCI Adult, UCI Wine Quality, Titanic, Northwind) and real
warehouses (Snowflake, Databricks, S3).

Each file states the problem, where it is, why it matters, and a proposed fix.
Every one was reproduced before it was written down. Nothing here is fixed yet.

| # | Title | Severity |
|---|---|---|
| [1](001-decimal-columns-lose-precision.md) | A `Decimal` column loses precision, and walks past the guard that exists to stop it | High |
| [2](002-non-unique-primary-key-accepted.md) | A non-unique primary key is accepted, and the model silently sees one of the duplicates | High |
| [9](009-csv-reader-rounds-large-integers.md) | The CSV reader rounds large integers when the column has a blank | High |
| [14](014-a-sum-over-integers-in-duckdb-is-rounded.md) | A `SUM` over an integer column in DuckDB comes back rounded | High |
| [3](003-engine-exceptions-escape-the-error-surface.md) | Engine exceptions escape the SDK's error surface, one advertising an API this SDK does not have | Medium |
| [4](004-uint64-declared-as-int64.md) | A `uint64` column is declared `int64` while carrying values `int64` cannot hold | Medium |
| [5](005-empty-frames-are-not-refused.md) | An empty `predict` frame costs a round trip and returns a result with no columns | Medium |
| [6](006-s3-parquet-prefix-needs-a-trailing-slash.md) | A Parquet prefix on S3 is refused unless it ends in `/` | Medium |
| [10](010-all-null-context-column-stringifies-real-values.md) | A column empty in the context turns real values into strings in the rows being scored | Medium |
| [11](011-numpy-scalars-in-inference-raise-a-raw-typeerror.md) | A numpy scalar in `inference` raises a raw `TypeError` from the JSON encoder | Medium |
| [12](012-an-explicit-categorical-becomes-text.md) | An explicitly declared `Categorical` becomes `text` once it has 65 levels | Medium |
| [7](007-materialize-prints-to-stdout.md) | `materialize()` prints progress to stdout | Low |
| [8](008-endpoint-limits-are-not-declared-locally.md) | Two endpoint limits are discovered only by being rejected, and one does not name the column | Low |
| [13](013-a-malformed-response-is-coerced-rather-than-refused.md) | A malformed `200` response is coerced rather than refused | Low |

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

Two more are the same shape in the type system: the SDK works out something
careful — `UNTYPED` for a column with no values
([10](010-all-null-context-column-stringifies-real-values.md)), the caller's
own `Categorical` ([12](012-an-explicit-categorical-becomes-text.md)) — and
then discards it one function later.

And three are error-surface holes: engine exceptions
([3](003-engine-exceptions-escape-the-error-surface.md)), a JSON encoder
`TypeError` ([11](011-numpy-scalars-in-inference-raise-a-raw-typeerror.md)),
and a response coerced instead of refused
([13](013-a-malformed-response-is-coerced-rather-than-refused.md)).

## What was checked and is clean

Worth recording, because it is most of the surface.

**Connectors (20/20).** Local CSV, Parquet, gzip and Parquet directories; S3
CSV, Parquet and gzip; SQLite including a table name with a space and a quoted
identifier; DuckDB over Parquet and in-memory; missing file, missing key,
missing bucket, unknown suffix, unknown backend, unknown keyword, wrong
connector for a path in both directions; `table=` and `query=` argument
handling.

**Identifier safety.** Injection through `table=` refused before the driver on
SQLite, Snowflake and Databricks — the sentinel table survives in all three.
Snowflake case folding, quoted mixed-case identifiers and reserved words;
Databricks three-part names and case preservation.

**Warehouses.** Snowflake connect/read/round-trip-into-a-prediction, a semantic
view described and queried through `SEMANTIC_VIEW()` and fed to a model,
`NUMBER(38,0)` past the JS-safe range travelling exactly, `DECIMAL` scale
preserved. Databricks connect, `samples.tpch` read and scored, `BIGINT` past
the JS-safe range exact, three-part names, view listing.

**Tabular model.** Binary classification (Adult), multiclass (Wine, 6 classes,
probabilities summing to 1), regression, nulls and bools (Titanic), category
dtype, single row, single feature, one context row, an all-null column, output
selection, an unsupported output refused locally, `row_index` joining
one-to-one, a shuffled pandas index, sessions (reuse, after close, two at
once), the predictive-query path, an all-null target refused, a single-class
target, an unknown column named, a duplicate column name refused.

**Relational model.** Northwind schema inferred from frames alone — three
relationships, right way round, including the table whose name contains a
space; classification and both temporal aggregations (`COUNT`, `SUM`);
`num_context_samples` honoured exactly at 5/40/300; `num_neighbors` reaching
the sampler; an anchor time inside the data; an explanation refused for more
than one entity; sessions; string primary keys correlating back; entity values
never travelling raw.

**Error contract.** Unreachable host, sub-millisecond timeout, wrong model for
the endpoint, a credential in the query string, a credential in the userinfo
over plaintext to a non-local host — all `StructuredDataError` with a code.
`inf` refused before the wire. Tz-aware and naive timestamps normalising to the
same instant.
