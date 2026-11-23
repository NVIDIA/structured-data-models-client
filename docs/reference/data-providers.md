# Your own data provider

A data provider lets the client read data through your own code instead of a database connection.
Use one for an engine the client has no connector for, or to control exactly how each read runs.
Python and Java take the same provider, with the same requests, in each language's own types.

The core never sends a provider SQL.
It sends one of a fixed set of typed requests, and the provider answers each with Arrow rows.

## Writing one

| | Python | Java |
|---|---|---|
| The provider | `sdm_client.provider.DataProvider`, a protocol | `com.nvidia.sdm.provider.DataProvider`, an interface |
| What it answers | `capabilities() -> Capabilities` | `Capabilities capabilities()` |
| Each request | `answer(request, expected)`, returning any object exporting an Arrow stream: a pyarrow `Table` or `RecordBatchReader`, or a DuckDB or Polars result | `ArrowReader answer(request, expected, allocator)` |
| Its failures, optional | `failure(error) -> ProviderFailure` | `ProviderFailure failure(Exception)` |
| As a source | `sdm_client.ProviderSource(provider)` | `ProviderSource.of(provider)` |

```python
from sdm_client import ProviderSource, Schema
from sdm_client.provider import Capabilities, OptionalRequest, Scan

class MyProvider:
    def capabilities(self):
        return Capabilities(optional={OptionalRequest.SAMPLE_ENTITIES})

    def answer(self, request, expected):
        if isinstance(request, Scan):
            ...  # read request.table, request.columns, request.filters

schema = Schema.from_source(
    ProviderSource(MyProvider()),
    primary_keys={'users': 'id'},
)
```

```java
Source source = ProviderSource.of(new MyProvider());
Schema schema = Schema.fromSource(source, options);
```

`expected` is the exact schema the answer must have, for the requests whose answer the contract shapes: `describe`, `key_counts` and `time_bounds`.
For the others it is empty, and the answer holds the columns the request names, in order, typed as the source stores them.

## What a provider answers

Every provider answers `describe` and `scan`.
It declares any other request it answers in its capabilities' optional requests: `preview`, `count_rows`, `key_counts`, `time_bounds`, `column_ranges`, `repeated_key`, `sample_entities`, `neighbors` and `lookup`.
The core answers an undeclared one in memory, from the rows a `scan` reads, within the fallback limits the provider declares, or refuses it with `UNSUPPORTED_REQUEST` when none are declared.

A source that declares no primary or foreign keys, such as Trino, still works: name the keys and relationships when building the schema.

## Filters

A provider declares the kinds of value filter it applies to a `scan` or `sample_entities`: comparisons, `IN`, `IS NULL`, `OR` (`any_of`), `AND` (`all_of`), and the window check `has_rows`.
The core never sends a kind the provider does not declare.
It applies that condition to the rows the provider returns, so the provider reads more rows but every result is the same.

A provider may return more rows than a filter holds for, but never fewer: the core checks every row against the query's own rules.
So a provider whose engine cannot write a value may leave that filter out, which only widens its read.

## Large key sets

With a key limit (`max_keys_per_request` in Python, `maxKeysPerRequest` in Java), a `neighbors` or `lookup` with more seeds reaches the provider as several requests of at most that many seeds each.
Each seed's rows carry the seed's own batch position, so the core joins the answers, in order, into the request's one answer.

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
So a time bound is never shifted, and a provider never reads fewer rows than a filter holds for.

The library calls a provider from the thread that runs the operation, one request at a time.
It never closes the provider, and a `ProviderSource` cannot be pickled or saved, as no source can.
