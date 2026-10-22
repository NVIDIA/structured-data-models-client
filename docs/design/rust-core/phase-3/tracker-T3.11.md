# T3.11: Java binding, API and Java S0

Lane J, stack 3, stacked on T3.2's candidate (the `java` runner name).
Design: [design.md](design.md), choice 10, the Packages row T3.11, locked check 14.

## What it delivers

| Output | Where | Held by |
|---|---|---|
| The binding, `sdm-jni`, on `jni` 0.22 over `sdm-host` and `sdm-contract` only | `rust/bindings/sdm-jni/` | `java_build.py verify` (the Java tests load it); `l0_deps.py` once T3.2 classifies it; `cargo deny` |
| Control messages as `byte[]` both ways, never `String` | `sdm-jni/src/lib.rs`, `com.nvidia.sdm.internal.Run` | `PredictiveQueryTest` (non-ASCII, outside the BMP, a NUL, a lone surrogate, a query past 64 KB); the test author's `T311BindingTest` |
| Arrow through Arrow Java's C Data module, no copy, both ways | `ArrowExchange`, `NativeArrow`, `exportArrow` | `ArrowTest` (round trip, empty stream, two million rows, every allocator closing with nothing held) |
| A panic contained at every entry | `sdm-host`'s `contain` for the core's, the `Raise` policy at each JNI entry for the binding's own | `ProcessTest` and the test author's `T311PanicTest`, each in a JVM of its own |
| One owner and an explicit release per native resource | `Run` (a read-write lock: calls share it, `close` waits for them), `NativeArrow`, each with a `Cleaner` behind | `RunTest` (released twice, used after release, busy from two threads, `close` waiting for a call in flight) |
| The Java API of choice 10 for S0 | `PredictiveQuery.parse(text).validate(schema, profile)`, `QuerySchema`, `Profile`, `Validation`, `StructuredDataException` | `PredictiveQueryTest`; `scope-java.yaml`'s `api` |
| `tracing` to the SLF4J 2 API, bound by the application | `sdm-jni/src/logging.rs`, `NativeLog` | `ProcessTest` (an event reaches SLF4J; one SLF4J drops never crosses) |
| One JAR from one `pom.xml`, Java 17 bytecode, the natives of four platforms | `java/pom.xml` (`-Pall-natives` refuses a JAR missing one) | CI `java-jar`; the test author's JAR tests |
| The Java runner | `contract/runners/run_java.py`, `contract/runners/java/ContractRunner.java` | CI `java (<jdk>, <platform>, s0)`; `test_t311_java.py` |
| The Java legs in the handoff | `handoff.py`'s reference step | `tests/test_t311_java_process.py` |
| CI | `java-native` (macOS), the Linux natives beside the wheels, `java-jar`, `java-test (17, 21)`, `java-load (<jdk>, <platform>)`, `java (<jdk>, <platform>, s0)` | `tests/test_ci_workflow.py`, `tests/test_t311_java_ci.py` |

## Choices made here

- **The S0 call.** Java's S0 leg is `PredictiveQuery.parse(text).validate(schema, profile)`, the core's `validate_query` through the public API, answered rather than thrown, as rust-host's leg is the core's answer as the route maps it.
  The Java host has no provider before T3.12, so `materialize` (which Python's S0 route goes through) arrives with T3.12 and validates first, as Python resolves in `materialize`.
  The observation leaves the stage and the exception class null, as rust-host's does.
- **`QuerySchema`, not `Schema`.** The schema S0 validates against is the contract's `QuerySchema` as plain records; `Schema`, the description `Schema.fromSource` returns, is T3.12's and gives a `QuerySchema`.
- **`parse` refuses what Python's constructor refuses.** A text blank once stripped as Python's `str.strip()` strips: the same characters on JDK 17 and 21 as `str.isspace()` on Python 3.10 and 3.12 (checked over every code point).
- **The native's loading.** From the JAR, at first use, by platform (`<os>-<arch>`); a platform the JAR holds none for is `UNSUPPORTED_REQUEST` naming it, and the same refusal on every later call.
  `-Dsdm.native.library` loads another file instead (the tests' build with the test-only operations); the JAR-load test refuses to run with it set.
- **The second runtime flag's error.** Without `--add-opens=java.base/java.nio=ALL-UNNAMED`, Arrow Java fails with its own `UnsupportedOperationException`; the host names it as `INVALID_CONFIGURATION` with the flag in `details`.
  Validation crosses no Arrow and runs without either flag on JDK 17 and 21; `--enable-native-access` is asked for from JDK 24.
- **The host facts** are read as the Python host reads them (`TZ`, `/etc/localtime`, Python's default `zoneinfo.TZPATH`, `sys.platform`'s names), so both hosts hand the core the same facts on one machine.
- **The Linux natives are built in the wheels' jobs** (`native-wheel`, `native-wheel-aarch64`, `native-test-ops`), in their manylinux2014 images, from the same build, as choice 10 says; macOS's in `java-native` on the engine wheels' runners.
  macOS is in because the design's locked check 14 names it; the core had not been built for macOS before.
- **The JAR-load jobs' containers.** Eclipse Temurin's JDK images from ECR Public, pinned by digest, the workflow's only job containers; `tests/test_ci_workflow.py` admits them by name.

## Probes

- JSON from outside (`Json.read`): 100,000 nested arrays and 5,000 nested objects refused by name, a 200,000-member object read (`JsonTest`).
- Java's 24 S0 answers against rust-host's on the same cases: the same operation inputs, acceptance and error on every case, the five diagnostic cases included.

## Measured differences from the reference

Java's S0 leg differs from the reference on the five diagnostic S0 cases exactly as rust-host's does, the core's answer being the same: `s0.boundary.date-long-field` and `date-packed` (DT-5), `escape-lone-surrogate` and `escape-surrogate-pair` (PQL-1), `depth-999` (PQL-2), all approved by the owner.
No new register entry: whether those entries' per-runner rows extend to the `java` runner is the ratchet's admission of Java's legs (T3.2).

## Rounds

| Round | SHA | Result |
|---|---|---|
