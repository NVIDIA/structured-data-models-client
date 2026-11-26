# Your own source

A `Source` is where a schema's data is read.
The built-in sources (`DuckDBSource`, `SQLiteSource`, `PostgresSource`, `DatabricksSource`, `SnowflakeSource`, and in Java `DatabricksSparkSource`) read a database through your connection, with SQL the library writes.
They are sources like any other: the client reaches every source through the same `capabilities()` and `answer()`, and a built-in source asked directly answers exactly what an operation over it receives.
To read data through your own code instead, subclass `Source` in Python or implement it in Java.
Do that for an engine the client has no built-in source for, or to control exactly how each read runs.
Python and Java take the same source, with the same requests, in each language's own types.

The client never sends your source SQL.
It sends typed requests, such as "these columns of this table", and the source answers each with Arrow data.

## A complete example

This source serves two in-memory tables.
It answers the two requests every source answers, `describe` and `scan`, and lets the client answer every other request in memory from scans.

```python
import datetime

import pyarrow as pa
import pyarrow.compute as pc

from sdm_client import PredictiveQuery, Relationship, Schema, Source
from sdm_client.provider import (
    Capabilities,
    Describe,
    DescribePart,
    Fallback,
    KeyIn,
    NotNull,
    Scan,
    TimeRange,
)

START = datetime.datetime(2024, 1, 1)
TABLES = {
    'users': pa.table({'id': [1, 2, 3], 'age': [25, 40, 35]}),
    'orders': pa.table(
        {
            'order_id': list(range(60)),
            'user_id': [i % 3 + 1 for i in range(60)],
            'amount': [float(i) for i in range(60)],
            'placed': [START + datetime.timedelta(days=6 * i) for i in range(60)],
        }
    ),
}


class InMemorySource(Source):
    def capabilities(self):
        # The client answers every request not declared here in memory, from
        # scans, within these limits.
        return Capabilities(fallback=Fallback(max_rows=10_000_000, max_bytes=2**31))

    def answer(self, request, expected):
        if isinstance(request, Describe):
            return describe(request, expected)
        if isinstance(request, Scan):
            table = TABLES[request.table.table]
            for condition in request.filters:
                table = table.filter(holds(table, condition))
            table = table.select(list(request.columns))
            return table if request.limit is None else table.slice(0, request.limit)
        raise NotImplementedError(request.name)


def describe(request, expected):
    names = [relation.table for relation in request.tables] or list(TABLES)
    if request.part is DescribePart.TABLES:
        rows = [{'table': name} for name in names]
    elif request.part is DescribePart.COLUMNS:
        rows = [
            {'name': f.name, 'type': str(f.type), 'not_null': False, 'primary_key': False}
            for f in TABLES[names[0]].schema
        ]
    else:
        rows = []  # No keys, foreign keys or namespace: the schema names them.
    return pa.Table.from_pylist(rows, schema=expected)


def holds(table, condition):
    column = table[condition.columns[0] if isinstance(condition, KeyIn) else condition.column]
    if isinstance(condition, NotNull):
        return pc.is_valid(column)
    if isinstance(condition, KeyIn):
        keys = condition.keys.reader.read_all().column(0)
        return pc.is_in(column, value_set=keys.cast(column.type))
    if isinstance(condition, TimeRange):
        mask = pc.is_valid(column)
        for bound, inclusive_op, exclusive_op in (
            (condition.lower, pc.greater_equal, pc.greater),
            (condition.upper, pc.less_equal, pc.less),
        ):
            if bound is not None:
                op = inclusive_op if bound.inclusive else exclusive_op
                mask = pc.and_(mask, op(column, pa.scalar(bound.value.value, column.type)))
        return mask
    raise NotImplementedError(type(condition).__name__)


schema = Schema.from_source(
    InMemorySource(),
    primary_keys={'users': 'id', 'orders': 'order_id'},
    time_columns={'orders': 'placed'},
    relationships=[Relationship('orders', 'user_id', 'users', 'id')],
)
request = PredictiveQuery(
    'PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.id WHERE users.age > 30',
    schema,
).materialize(indices=[2, 3])
print(request.context_report)
```

The same source in Java implements `com.nvidia.sdm.Source`, whose `answer` returns an `ArrowReader`.
The complete, runnable version is [`InMemoryProviderExampleTest.java`](../../java/src/test/java/com/nvidia/sdm/InMemoryProviderExampleTest.java); its core is:

```java
class InMemorySource implements Source {
  public Capabilities capabilities() {
    return Capabilities.builder().fallback(10_000_000, 1L << 31).build();
  }

  public ArrowReader answer(
      DataRequest request, Optional<Schema> expected, BufferAllocator allocator) throws Exception {
    if (request instanceof DataRequest.Describe describe)
      return describe(describe, expected.orElseThrow(), allocator);
    if (request instanceof DataRequest.Scan scan) return scan(scan, allocator);
    throw new UnsupportedOperationException(request.name());
  }
}

Source source = new InMemorySource();
var schema =
    Schema.fromSource(
            source,
            FromSourceOptions.builder()
                .table("users")
                .table("orders")
                .primaryKey("users", "id")
                .primaryKey("orders", "order_id")
                .timeColumn("orders", "placed")
                .build())
        .withRelationships(List.of(new Relationship("orders", "user_id", "users", "id")));
```

## A source over a query engine

The Trino example is a complete source over a query engine the client has no built-in source for, in both languages: [`trino_source.py`](../../packages/structured-data-models-client/tests/trino_source.py), over Trino's HTTP protocol with the standard library, and [`TrinoSource.java`](../../java/src/test/java/com/nvidia/sdm/examples/TrinoSource.java), over Trino's JDBC driver.
Each uses only the public API, and turns every request it declares into one Trino query:

- `describe` reads the tables, columns and namespace from Trino's `information_schema`; Trino declares no keys, so the schema names them.
- `scan` and `preview` are a `SELECT` with the request's filters, every filter kind written in Trino's SQL.
- `count_rows` is a `count(*)`, which writes every filter, since a count is exact.
- `sample_entities` orders the rows by a hash of each key and the seed, so the same seed draws the same rows.
- `lookup` joins the seeds' key set to the table, with a key limit, so a large key set reaches it as several lookups.
- `failure` reads Trino's error codes into the `FailureFacts` they mean, such as a table that does not exist.

Every other request, `neighbors` among them, is answered by the client from scans within the declared fallback.
Each value is written as a literal of its `Value.type`, so a decimal keeps its precision and a timestamp its unit.
[`test_trino_source.py`](../../packages/structured-data-models-client/tests/test_trino_source.py) and [`TrinoProviderTest.java`](../../java/src/test/java/com/nvidia/sdm/TrinoProviderTest.java) run them against a Trino server named by `SDM_TRINO_TEST_URL`: they build a schema, materialize queries keeping exactly the examples the data holds, split a key set past the key limit, and fail an operation over a dropped table with `PROVIDER_ERROR`.

## What a source must answer

| Request | Who answers it |
|---|---|
| `describe` | Always the source |
| `scan` | Always the source |
| `preview`, `count_rows`, `key_counts`, `time_bounds`, `column_ranges`, `repeated_key`, `sample_entities`, `neighbors`, `lookup` | The source, where it declares the request in `Capabilities.optional`; otherwise the client, in memory, from scans, within `Capabilities.fallback` |

Without a fallback, every request the source does not declare fails with `UNSUPPORTED_REQUEST`, so a source answering only `describe` and `scan` declares one.
Declaring `sample_entities` lets the source draw samples itself, which matters for a large table: the fallback reads the whole table to sample it.

`expected` is the exact schema the answer must have, for the requests whose answer the client shapes: `describe` and `key_counts`.
For the others it is empty, and the answer holds the columns the request names, in order, typed as the source stores them.
Every source answers in these shapes, the built-in sources included.

A `time_bounds` answer has one row per column asked, in order: `table_index`, the column's position in the request (an `int32` or `int64`), then `min_date` and `max_date`, its smallest and largest value, null where no row holds one.
Each bound is of the column's own type: a timestamp of any unit or zone, a date, or text, as a source with no time type stores it.
The client reads a timestamp with a zone at its UTC instant, a date at its midnight, and text by the reading `Capabilities.text_times` declares.

The client reads `capabilities()` once at the start of each operation, before any request of it, so a source may keep what it learns within one operation, as the built-in sources keep the namespace they settle.

A `scan` or `preview` with a `limit` asks for at most that many rows, any of them.

## Describe answers

Each `describe` request asks for one part of the catalog, for the tables it names, or every table when it names none.

| Part | One row per | Columns |
|---|---|---|
| `tables` | table | `table` |
| `columns` | column of the one table named, in order | `name`, `type` (the source's own type name, shown in messages), `not_null`, `primary_key` |
| `keys` | key the table declares | `kind` (`primary`, `unique` or `foreign`), `columns` (a list) |
| `foreign_keys` | column pair of each foreign key | `key` (a number naming the key), `referenced_table`, `columns`, `referenced_columns` |
| `namespace` | part of the namespace a one-part table name is read in, outermost first | `name` |

An empty answer is valid for `keys`, `foreign_keys` and `namespace`.
A source that declares no primary or foreign keys, such as Trino, still works: name the keys, time columns and relationships when building the schema, as the example does.
Keys and time columns given to `Schema.from_source` override what the source declares, and relationships given replace them.

## Describing the source

Beyond the requests it answers, `Capabilities` describes the source to the client.
Each default is a plain source of tables, so a source declares only what differs.

| Python | Java | Declares |
|---|---|---|
| `namespace` | `namespace` | The namespace a table named by one part is read under, outermost first |
| `ask_namespace_first` | `askNamespaceFirst` | Ask the source its namespace (`describe` of `namespace`, answered in full) before reading a table named by one part |
| `retry_upper_case_names` | `retryUpperCaseNames` | Try a table the catalog does not find again upper-cased |
| `column_types_are_affinities` | `columnTypesAreAffinities` | A column's declared type is only an affinity; its type is read from its values |
| `reread_tables_every_run` | `rereadTablesEveryRun` | Each run reads the tables as they stand, not as a given schema holds them |
| `engine` | `engine` | The database engine, by the name messages use (`duckdb`, `sqlite`, `postgres`, `databricks`, `snowflake`); its semantic models are read in that engine's SQL |
| `semantic_model` | `semanticModel` | The semantic model the source is read through: its name, and whether its expressions are trusted |
| `derived_relations` | `derivedRelations` | The source reads a derived relation (`Relation.columns`, `Relation.filter`) |
| `text_times` | `textTimes` | How the client reads a time stored as text |
| `index_advice` | `indexAdvice` | Which indexes serve the sampler's reads, for its warning of missing ones |
| `sample_by_row_count` | `sampleByRowCount` | Size a sample by the table's rows, which the client counts first |
| `sample_by_key_type` | `sampleByKeyType` | Order a seeded sample by the key's data type, which the client names |
| `neighbor_lookback` | `neighborLookback` | Look back from each seed only as far as the table's rate of rows says |
| `keys_optional` | `keysOptional` | Read a failed read of declared keys as no declared keys |
| `key_enforcement` | `keyEnforcement` | The catalog answers which tables enforce their keys |
| `compares_stored_times` | `comparesStoredTimes` | Compare a window's bounds with the stored text of its column |
| `nan_is_missing` | `nanIsMissing` | Read a NaN as missing, and `-0.0` as `0.0` |
| `opaque_vendor` | `opaqueVendor` | The vendor whose opaque Arrow extension types the source hands over |
| `fallback_text_order` | `fallbackTextOrder` | How the source compares text, where the fallback compares it |

A built-in source's `capabilities()` returns what its engine declares, in the same type.

## Filters

A `scan` or `sample_entities` carries the conditions its rows must meet; all of them hold.

**Always applied.** `KeyIn` (the row's key is one of a key set's), `TimeRange` and `NotNull` reach every source, which applies them: the client relies on the rows it gets meeting them.

**Applied where declared.** Value filters reach a source only where it declares their kind in `Capabilities.filters`: comparisons (`Compare`), `In`, `IsNull`, `OR` (`AnyOf`), `AND` (`AllOf`), and the window check `HasRows`.
The client applies every value filter itself to the rows it gets, so a source may return more rows than a value filter holds for, but never fewer.
A source whose engine cannot write a value may leave that filter out, which only widens its read.
Declaring a kind the engine handles makes a selective query read fewer rows.

## Large key sets

With a key limit (`max_keys_per_request` in Python, `maxKeysPerRequest` in Java), a `neighbors` or `lookup` with more seeds reaches the source as several requests of at most that many seeds each.
Each seed's rows carry the seed's own batch position, so the client joins the answers, in order, into the request's one answer.

## Errors

A source's exception while answering a data request fails the operation with `PROVIDER_ERROR`, which carries that exception: `error.source_error` in Python and `error.sourceError()` in Java.
That holds whether `answer` raised it or the reader it returned raised it as its batches were read; where several requests of one step fail, the first failure is the one carried.
Read its message there, such as which input failed to parse.
The error's own message, `details` and stack trace name only the exception's class (`driver_class`), and the exception is not the error's cause, since a driver's message can quote data values or connection settings that should not reach logs.
The exception is not pickled (Python) or serialized (Java), and holding the error keeps it alive.
A source may also override `failure(error)`, returning a `ProviderFailure` with the SQLSTATE, vendor code, error class and message it reads from its error, and the `FailureFacts` it means, such as a table not found or access denied; by default, the client reads a database driver's `sqlstate` and `vendor_code` attributes (Java: a `SQLException`'s state and vendor code).
Python builds facts by keyword, `FailureFacts(not_found=True)`, and Java with `FailureFacts.builder().notFound().build()`.
A built-in source refuses a request its database's SQL cannot answer with the client's own error, which the operation fails with.
The `ProviderFailure` message is kept internal: the client never puts it in an error it raises.

## Values

Each value in a request is a `Value`: a plain value of each language, `value`, with the Arrow `type` the client holds it as, such as `int64` or `timestamp[us, tz=UTC]`.
The type is what lets a source write the value exactly in its own SQL: an integer's width and sign, a decimal's precision and scale, a timestamp's unit and zone, and the type of a null.

| Value | Python | Java |
|---|---|---|
| Nulls | `None` | `null` |
| Integers | `int` | `Byte`, `Short`, `Integer`, `Long` by width; an unsigned one in the next wider type, and an unsigned 64-bit one a `BigInteger` |
| Floats | `float` | `Double`, `Float` |
| Decimals | `Decimal` | `BigDecimal` |
| Text, bytes | `str`, `bytes` | `String`, `byte[]` |
| Dates | `datetime.date` | `LocalDate` |
| Timestamps without a zone, a wall clock | `datetime.datetime` | `LocalDateTime` |
| Timestamps with a zone, an instant | `datetime.datetime` in that zone | `Instant`, its zone in its `type` |
| Durations | `datetime.timedelta` | `Duration` |
| Times of day | `datetime.time` | `LocalTime` |

Every value is exact.
A nanosecond timestamp or duration is a `pandas.Timestamp` or `pandas.Timedelta`, which are `datetime.datetime` and `datetime.timedelta` keeping nanoseconds, and a time of day with nanoseconds is a `pandas.Timedelta` since midnight.
A date or timestamp outside the years `datetime` holds, 1 to 9999, is a `numpy.datetime64`.
So a time bound is never shifted, and a source never reads fewer rows than a filter holds for.

The library reads a source's `capabilities()` once as each operation starts, then calls `answer()` from the thread that runs the operation, one request at a time.
It never closes the source, and a source, or a schema over one, cannot be pickled or saved.
