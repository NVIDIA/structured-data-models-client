<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Structured Data Models client for Java

The Java library on the client's Rust core: the same core the Python client runs, loaded from the JAR through JNI (design choice 10 of `docs/design/rust-core/phase-3/design.md`).
It answers query validation today (Java S0); sources, `materialize` and `predict` follow on the same core (T3.12, T3.13).
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
import com.nvidia.sdm.PredictiveQuery;
import com.nvidia.sdm.Profile;
import com.nvidia.sdm.QuerySchema;
import com.nvidia.sdm.StructuredDataException;
import com.nvidia.sdm.Validation;
import java.util.List;

QuerySchema schema = new QuerySchema(
    List.of(
        new QuerySchema.Table("users",
            List.of(new QuerySchema.Column("user_id", "int", "ID")),
            "user_id", null, null),
        new QuerySchema.Table("orders",
            List.of(
                new QuerySchema.Column("order_id", "int", "ID"),
                new QuerySchema.Column("user_id", "int", "ID"),
                new QuerySchema.Column("ts", "time", "timestamp")),
            "order_id", "ts", null)),
    List.of(List.of(
        new QuerySchema.ColumnRef("orders", "user_id"),
        new QuerySchema.ColumnRef("users", "user_id"))));

PredictiveQuery query = PredictiveQuery.parse(
    "PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id");
Validation validation = query.validate(schema, Profile.RELATIONAL);
if (!validation.accepted()) {
  System.out.println(validation.reasons());
}
```

Every error is a `StructuredDataException`, as every Python error is a `StructuredDataError`: `code()` is the contract's error code to branch on (`INVALID_REQUEST`, `UNSUPPORTED_REQUEST`, `INTERNAL_ERROR`, ...), and `details()` holds the structured fields the core gave.
A platform the JAR holds no native library for is refused as `UNSUPPORTED_REQUEST`, naming the platform.
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

The contract cases run on the Java host through `contract/runners/run_java.py`, which hands `contract/runners/java/ContractRunner.java` the calls to make and writes the observations the comparator judges.
