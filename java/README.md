<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Structured Data Models client for Java

The Java library on the client's Rust core: the same core the Python client runs, loaded from the JAR through JNI (design choice 10 of `docs/design/rust-core/phase-3/design.md`).
It describes the tables of a DuckDB, SQLite, Postgres or Databricks database through the caller's own JDBC connection, gathers a predictive query's request from them, and predicts with a model endpoint, each on the core, as the Python client does.
Databricks metric views are read through `source.semanticModel(name, trustExpressions)`, and `schema.conversionNotes()` explains their conversion in the core's words.
Nothing is published to a repository before Phase 5.

## Coordinates

| | |
|---|---|
| Group | `com.nvidia` |
| Artifact | `structured-data-models-client` |
| Version | the client's (`1.0.0`) |
| Java package | `com.nvidia.sdm` |
| Bytecode | Java 17, tested on JDK 17 and 21 |
| Natives in the JAR | Linux x86_64 and aarch64 (glibc 2.17 or later), macOS arm64 and x86_64 |

The JAR depends on Arrow Java (`arrow-c-data`, `arrow-vector`, `arrow-memory-core`, and `arrow-memory-unsafe` at run time) and on the SLF4J 2 API, which it declares and does not bind: the application chooses its logger.

## Runtime flags

Every JVM that loads the library runs with both of these flags, and every test of this repository does:

```
--enable-native-access=ALL-UNNAMED
--add-opens=java.base/java.nio=ALL-UNNAMED
```

The first allows the JAR's JNI library; JDK 17 and 21 do not ask for it, and JDK 24 and later warn without it and will later refuse.
The second is Arrow Java's: its C Data module reaches native memory through `java.nio`'s internals, and without it the first Arrow exchange with the core is refused as `INVALID_CONFIGURATION`, naming the flag; validation crosses no Arrow and runs without it.

## Use

```java
import com.nvidia.sdm.DuckDbSource;
import com.nvidia.sdm.MaterializeOptions;
import com.nvidia.sdm.PredictionResult;
import com.nvidia.sdm.PredictiveQuery;
import com.nvidia.sdm.Request;
import com.nvidia.sdm.Schema;
import com.nvidia.sdm.SdmClient;
import java.sql.Connection;
import java.sql.DriverManager;
import java.time.LocalDate;
import java.util.List;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;

try (Connection connection = DriverManager.getConnection("jdbc:duckdb:shop.duckdb");
    SdmClient client = SdmClient.connect("http://localhost:8000");
    RootAllocator allocator = new RootAllocator()) {
  Schema schema = Schema.fromSource(DuckDbSource.of(connection))
      .withStype("users", "age", "categorical");
  Request request = PredictiveQuery
      .parse("PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id")
      .materialize(schema, MaterializeOptions.builder()
          .indices(List.of(1L, 2L))
          .anchorTime(LocalDate.of(2024, 3, 1))
          .build());
  PredictionResult result = client.predict(request, SdmClient.KUMO_RELATIONAL);
  try (ArrowReader rows = result.read(allocator)) {
    while (rows.loadNextBatch()) {
      System.out.println(rows.getVectorSchemaRoot().contentToTSVString());
    }
  }
}
```

The API mirrors Python's with Java names and plain data:

| Python | Java |
|---|---|
| `DuckDBSource(connection)`, `SQLiteSource`, `PostgresSource(connection, schema=)`, `DatabricksSource(connection, catalog=, schema=)` | `DuckDbSource.of(connection)`, `SqliteSource.of`, `PostgresSource.of(connection[, schema])`, `DatabricksSource.of(connection[, catalog, schema])`, each over a `java.sql.Connection` the caller keeps open |
| `Schema.from_source(source, tables, primary_keys=, time_columns=, relationships=, check_keys=)` | `Schema.fromSource(source[, FromSourceOptions])` |
| `schema[t].primary_key = k`, `.time_column = c`, `schema[t][c].stype = s`, `del schema[t][c]`, `schema.relationships = [...]` | `withPrimaryKey`, `withoutPrimaryKey`, `withTimeColumn`, `withoutTimeColumn`, `withStype`, `withoutColumn`, `withRelationships`, `withRelationship`: each a new schema, checked and made by the core; a schema never changes, so `snapshot()` is the schema itself |
| `schema.tables`, `schema[t]`, `check_consistent()` | `tables()`, `table(name)`, `relationships()`, `checkConsistent()`; `Table` and `Column` |
| `PredictiveQuery(text, schema).materialize(indices=, anchor_time=, **options)` | `PredictiveQuery.parse(text).materialize(schema, MaterializeOptions)`; an option is a name and a plain value (`option("num_neighbors", List.of(8, 8))`), as a keyword argument |
| `Request.context_rows`, `predict_rows`, `payloads`, ... | `Request.contextRows()`, `predictRows()`, `payloads()`, `nothingToScore()`, `entityTable()`, ... |
| `SDMClient(url, api_key, timeout=, max_retries=)`, `capabilities(model)`, `predict(request, model=, outputs=)` | `SdmClient.connect(url[, ClientOptions])`, `capabilities(model)`, `predict(request, model[, outputs])` |
| `SDMClient.for_databricks_serving(endpoint, workspace_client=, timeout=)` | `SdmClient.forDatabricksServing(DatabricksServing.of(host, endpoint, tokenSupplier[, maxRequestBytes]))`, with `timeout(Duration)`; a caller-supplied token only |
| `PredictionResult.to_pandas()`, `absent()`, `explanation` | `PredictionResult.read(allocator)` (the core's rows as Arrow, ordered by `row_index`), `absent(requested)`, `explanation()` |

The JDBC drivers are the caller's: DuckDB's (`org.duckdb:duckdb_jdbc`), SQLite's (`org.xerial:sqlite-jdbc`), Postgres's (`org.postgresql:postgresql`) and Databricks's (`com.databricks:databricks-jdbc`); the JAR ships none.
What a driver gives reaches the core as the same Arrow Python's drivers give it, by the reviewed conversion table (`com.nvidia.sdm.internal.jdbc.ConversionTable`): DuckDB's own Arrow export; SQLite's columns typed as ADBC SQLite types them; Postgres's by type, as Python's provider types them through psycopg's registry, every value read in text as psycopg reads it; Databricks's by type, as the Databricks SQL connector gives Python's provider its Arrow.
The model transport sends the core's call as the bytes the core wrote, over `java.net.http`, retrying a transient failure (429, 502, 503, 504, a dropped connection) as Python's transport does.

Every error is a `StructuredDataException`, as every Python error is a `StructuredDataError`: `code()` is the contract's error code to branch on (`INVALID_REQUEST`, `UNSUPPORTED_REQUEST`, `INTERNAL_ERROR`, ...), and `details()` holds the structured fields the core gave.
A platform the JAR holds no native library for is refused as `UNSUPPORTED_REQUEST`, naming the platform.
The model endpoint's own refusal is a `ModelEndpointException`, with the endpoint's status, code, words and other fields, as Python's `NimRequestError`.
A Databricks serving endpoint's refusal is a `ModelEndpointException` too, under the endpoint's status and its own code, in the core's words, as Python's serving client raises it.
A panic inside the core is contained: the call raises `INTERNAL_ERROR` with `details().get("panic") == true`, and the JVM goes on.

The core's log events reach SLF4J under the loggers its crates name (`sdm_core`, `sdm_host`, `sdm_data` and their modules), and its warnings under `com.nvidia.sdm.warnings`, at `WARN`.

## Build

The natives are built by cargo and copied into `target/natives/<platform>/` before Maven packages the JAR (`<platform>` is `linux-x86_64`, `linux-aarch64`, `macos-aarch64` or `macos-x86_64`):

```
cargo build --release --manifest-path rust/Cargo.toml -p sdm-jni
mkdir -p java/target/natives/linux-x86_64
cp rust/target/release/libsdm_jni.so java/target/natives/linux-x86_64/
mvn -f java/pom.xml package
```

`mvn -Pall-natives package` refuses to build a JAR missing any platform's native; CI builds the JAR that way, from the natives each platform's job built.
`mvn package` also copies the runtime dependencies into `target/lib`, which the contract runner and the JAR-load test put on the class path.

The tests load a library built with the test-only operations, packaged as the JAR packages its natives (or named by `-Dsdm.native.library`); `scripts/rust_core/java_build.py verify` builds it, puts it there, and runs every Java test:

```
cargo build --manifest-path rust/Cargo.toml -p sdm-jni --features test-ops
cp rust/target/debug/libsdm_jni.so java/target/natives/linux-x86_64/
mvn -f java/pom.xml verify
```

The contract cases run on the Java host through `contract/runners/run_java.py` (`--database duckdb|sqlite|postgres` for the S1 cases), which hands `contract/runners/java/ContractRunner.java` the calls to make and writes the observations the comparator judges; `mvn package` copies the JDBC drivers it opens the databases with into `target/jdbc`.
