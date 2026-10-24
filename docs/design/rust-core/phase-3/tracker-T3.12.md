# T3.12: Java S1 through JDBC

Lane J, stacked on T3.11 (landed at 9c843c07).
Design: the structure design's "Java: T3.11, T3.12, T3.13" (one `internal/jdbc/JdbcProvider` with a per-driver key-set binder and a per-driver column in the JDBC-to-Arrow table; the value formatter and `Advice` spellings; `Schema.toHeld()`), fixes A, B and D, and design.md's choice 10.

## What it delivers

| Output | Where | Held by |
|---|---|---|
| Java's S1 API, Python's in Java's words and plain data | `com.nvidia.sdm`: `Source` (`DuckDbSource`, `SqliteSource`, `PostgresSource`), `Schema` (`fromSource`, the edits as new schemas, `snapshot`, `checkConsistent`), `Table`, `Column`, `Relationship`, `FromSourceOptions`, `PredictiveQuery.materialize`, `MaterializeOptions`, `Request`, `SdmClient`, `ClientOptions`, `ModelCapabilities`, `PredictionResult`, `ModelEndpointException` | `T312SchemaTest`, `T312PredictTest`; `scope-java.yaml`'s `api` fields |
| One JDBC provider | `internal/jdbc/JdbcProvider`: the plan's checks once; per driver a key-set binder (`KeySets`: DuckDB registers the Arrow and keeps it as a temporary table, SQLite ingests it as ADBC does, Postgres binds JSON text) and a result's reading | the Java legs; `T312ConversionTableTest` |
| The JDBC-to-Arrow table, one column per driver | `internal/jdbc/ConversionTable`: DuckDB's own Arrow export; SQLite's ADBC type inference; Postgres's types by name, as Python's provider reads them | L5 for Java; `T312ConversionTableTest` |
| L5 for Java | `packages/structured-data-models-client/tests_test_ops/test_l5_java.py` | the same type-map table, through Java's and Python's drivers, identical Arrow, metadata included, the JVM's default zone not UTC; a seeded unit and scale change in the Postgres path fails |
| The model transport over `java.net.http` | `internal/HttpTransport`: the core's bytes sent unchanged, retries as Python's transport, the endpoint's refusal as `ModelEndpointException` | `T312TransportTest`, `T312PredictTest`; every predict case's Java leg (`run_java.py`'s `check_sent`) |
| The schema as plain data | `Schema.toHeld()` (fix A); every edit and materialize sends it | `T312SchemaTest` |
| Typed results | `internal/Results` (fix D) | every Java run |
| Words: the value formatter and the `Advice` spellings | `internal/Wording` (fix B): every conversion of the grammar, `:to()`, `:place`, `:backend`, `{host.<name>}`, the groups | `T312WordingTest` (every advice the core names is spelled) |
| The local zone rule in `sdm-host` | `sdm-host/src/local_zone.rs`; Python's `facts.py` and Java's `HostFacts` ask it (T3.11 review P2 #2) | `sdm-host/tests/t312_local_zone.rs`, `test_t312_host_facts.py`, `T312HostFactsTest` |
| The models' capabilities from the core | `sdm-request`'s spec table gains `sessions` and `explanation_formats`; `sdm_host::capabilities()` through both bindings | `test_t312_host_facts.py` (equal to Python's `describe`), `T312PredictTest` |
| Validation internal | `QuerySchema`, `Profile`, `Validation` and `QueryValidation` moved to `com.nvidia.sdm.internal` (T3.11 review P2 #1) | `PredictiveQueryTest` |
| Java's S1 legs | `contract/runners/run_java.py --database duckdb|sqlite|postgres`, `ContractRunner.java` | `contract/tests/test_t312_java.py`; CI `java (<jdk>, <platform>, duckdb|sqlite|postgres)` |
| Coverage | `scope-java.yaml`'s `tests:` name JUnit methods for the paths no case claims; `coverage.py` and `scope.schema.json` read them | `test_t312_every_s0_and_s1_java_path_is_covered` |
| CI | `java-db` (JDK 17 and 21 on x86_64, 21 on aarch64, each database, Postgres as a service; L5 for Java on the x86_64 Postgres leg), `java-test` with Postgres required, the JDBC drivers in the JAR's artifact | `tests/test_t312_java_ci.py` |

## Choices made here

- **Materialize asks the core once.** Java's `materialize` sends the query as written to the core's `materialize`, which resolves it, as python-core does where its runner does not force the S0 route; a case naming `validate_query` among its core operations is validated first by the runner, through the internal validation, as python-core's forced route does, so both hosts start the same operations.
- **DuckDB through its own Arrow.** The DuckDB column of the table is the driver's own export (`arrowExportStream`), the same C++ conversion as Python's `to_arrow_table`; a key set is registered as Arrow and kept as a temporary table, since DuckDB reads a registered stream once.
- **SQLite as ADBC reads it.** The reference's driver is ADBC SQLite: a column's type is settled by its first 1,024 values, with ADBC's own text of a real in a text column, and a later value of another kind fails the read; the table holds it, probed from ADBC 1.12.
- **Postgres as psycopg reads it.** A column's type by its OID, read as psycopg's registry reads it (a type it does not name, a user's enum or its array, as the server's text); every value read in text, as psycopg reads it (a statement is never prepared on the server, and a binary transfer the caller's connection forces is off for the read and restored after); a `float4` as the double of the server's text; a zoned value written as text in the zone psycopg resolves the session's to (UTC for one Python does not name), by the zone data Python reads, through the core (`sdm_host::session_zone_offset`), never the JVM's; dates, times and intervals psycopg refuses refused by name; a value psycopg reads as an object written as Python's `str` of it (a range, an element of a `timestamptz`, `timetz` or `interval` array); JSON written by the core (`sdm_host::json_value_text`), as Python's provider writes it; a `numeric` past Arrow's decimals as its exact text, marked.
- **A Postgres read takes a transaction as psycopg's does.** None under autocommit; outside a transaction one of its own, committed; inside the caller's a savepoint.
- **Parity with Python's providers, value by value.** `tests_test_ops/test_t312_driver_parity.py` runs each type the table names, nulls, edge values and the values the reference's drivers refuse through both hosts' providers, seven runs plain and bound on one connection.
- **The core's calls on a stack of their own.** Each call into the core runs on a JVM thread with a 64 MiB stack, so no caller's stack, and no input, overflows it; the core's calls back into the JVM are made from a JVM thread.
- **A Java path's test is one JUnit ran.** coverage.py credits a JUnit test a Java path names only where Surefire's reports of the Java test run record it run and not skipped (`--junit-reports`, in the java-test job); no Java source is read.
- **What an answer left out is the core's.** The result's schema names it (`absent`, `sdm_request::predict::left_out`); `PredictionResult.absent` reads it.
- **IPC buffers realigned.** Arrow read back from an IPC stream is copied to buffers of their own, since an IPC body's 8-byte offsets are no alignment a 128-bit decimal may be read at across the C Data interface (arrow-rs panics on one).
- **The schema is immutable.** An edit gives a new schema, so `snapshot()` is the schema itself.
- **Options as names and plain values.** `MaterializeOptions.option(name, value)` passes each as Python's keyword arguments do; the core refuses one it does not read.
- **The runner reuses the Python runner's capture.** The Java program tells what crossed the binding and what the provider ran; `run_java.py` replays it into `run_python._Capture` and `arrow_values`, so both hosts are recorded alike.
- **macOS.** The S1 legs run on Linux, where the client wheel the runner's capture reads with is built; the macOS Java jobs stay S0's.

## Measured differences from the reference

Every S1 case on Java's paths, on DuckDB (245 cases), SQLite (227) and Postgres (243), agrees with python-core on every core field (`compare.py`'s core rule, python-core against java: no difference on any database).
Against the reference, Java differs exactly where rust-host does, under owner-approved entries: 117 `case_outcomes` rows in 24 entries copy rust-host's to runner `java` (DT-12, P1B-7 to P1B-11, P1B-26, P1B-33, P1B-53, P1B-55, P1B-72, P2B-1, P2B-10 to P2B-13, P2C-3, P2C-4, P2C-10, P2E-1, P2E-3 to P2E-5, P2I-4), each entry's notes saying so, listed for the owner.
The endpoint's refusal (P2G-1's cases) is the reference's on Java, `NimRequestError`'s twin with no code, so it needs no row.
New: JV-1 (proposed): the two cases naming a key or time column by a number cannot be written in Java's types and have no Java leg.
JV-2, JV-3 and JV-4 (proposed, fix rounds 1 and 2): an anonymous Postgres record is refused by name; a JSON value nested past the core's 2,000 levels is handed over as the server's text, which the core refuses on both hosts; a zoned time written as text names the session's zone, which pgjdbc sets to the JVM's.
A row on a DuckDB line for `java` moves from `none`, as no Java baseline is committed.

## Rounds

| Round | SHA | Result |
|---|---|---|
