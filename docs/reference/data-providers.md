# Your own source

A `Source` is where a schema's data is read.
The built-in sources (`DuckDBSource`, `SQLiteSource`, `PostgresSource`, `DatabricksSource`, `SnowflakeSource`, and in Java `DatabricksSparkSource`) read a database through your connection, with SQL the library writes.
To read data through your own code instead, subclass `Source`.
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
                mask = pc.and_(mask, op(column, pa.scalar(bound.value, column.type)))
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

The same source in Java extends `com.nvidia.sdm.Source`, whose `answer` returns an `ArrowReader`.
The complete, runnable version is [`InMemoryProviderExampleTest.java`](../../java/src/test/java/com/nvidia/sdm/InMemoryProviderExampleTest.java); its core is:

```java
class InMemorySource extends Source {
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

[`TrinoProviderTest.java`](../../java/src/test/java/com/nvidia/sdm/TrinoProviderTest.java) is a source over a query engine, applying every filter kind.

## What a source must answer

| Request | Who answers it |
|---|---|
| `describe` | Always the source |
| `scan` | Always the source |
| `preview`, `count_rows`, `key_counts`, `time_bounds`, `column_ranges`, `repeated_key`, `sample_entities`, `neighbors`, `lookup` | The source, where it declares the request in `Capabilities.optional`; otherwise the client, in memory, from scans, within `Capabilities.fallback` |

Without a fallback, every request the source does not declare fails with `UNSUPPORTED_REQUEST`, so a source answering only `describe` and `scan` declares one.
Declaring `sample_entities` lets the source draw samples itself, which matters for a large table: the fallback reads the whole table to sample it.

`expected` is the exact schema the answer must have, for the requests whose answer the client shapes: `describe`, `key_counts` and `time_bounds`.
For the others it is empty, and the answer holds the columns the request names, in order, typed as the source stores them.

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

A source's exception fails the operation with `PROVIDER_ERROR`.
A source may also override `failure(error)`, returning a `ProviderFailure` with the SQLSTATE, vendor code, error class and message it reads from its error; by default, the client reads a database driver's `sqlstate` and `vendor_code` attributes.
The message is kept internal and never shown to end users.

## Values

Values in requests are plain values of each language.

| Value | Python | Java |
|---|---|---|
| Integers | `int` | `Long`, or the integer type of its width |
| Floats | `float` | `Double`, `Float` |
| Decimals | `Decimal` | `BigDecimal` |
| Text, bytes | `str`, `bytes` | `String`, `byte[]` |
| Dates | `datetime.date` | `LocalDate` |
| Timestamps without a zone, a wall clock | `datetime.datetime` | `LocalDateTime` |
| Timestamps with a zone, an instant | `datetime.datetime` in that zone | `Instant` |
| Durations | `datetime.timedelta` | `Duration` |
| Times of day | `datetime.time` | `LocalTime` |

Every value is exact.
A nanosecond timestamp or duration is a `pandas.Timestamp` or `pandas.Timedelta`, which are `datetime.datetime` and `datetime.timedelta` keeping nanoseconds, and a time of day with nanoseconds is a `pandas.Timedelta` since midnight.
A date or timestamp outside the years `datetime` holds, 1 to 9999, is a `numpy.datetime64`.
So a time bound is never shifted, and a source never reads fewer rows than a filter holds for.

The library calls a source from the thread that runs the operation, one request at a time.
It never closes the source, and a source, or a schema over one, cannot be pickled or saved.
