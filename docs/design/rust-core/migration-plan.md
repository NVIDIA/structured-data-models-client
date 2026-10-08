# Rust core migration: the new plan

**Status: approved by the owner; executed phase by phase, each with its own design doc and tracker.**
It replaces the first plan; the owner's decisions are in section 2.
The postmortem behind it is [learnings.md](learnings.md).

The goal is the same: one implementation of the client's logic shared by Python and Java, so the two can't drift.
What changes is where the line between the core and each language sits, how data access and PQL plug in, how the work is sliced and tested, and how the rules are enforced.
All of the work happens on an epic branch and stays experimental until the owner merges it into `main`.

![New architecture](images/architecture.png)

## 1. What changes from the original proposal

| Original proposal | New plan | Why |
|---|---|---|
| No API change, read as bit-for-bit answers | Behavior is defined by a public contract corpus and a register of kept Python quirks; anything else changes only by a named, approved entry | Bit-for-bit copying spread pandas' accidents through the core |
| Core receives query results as Arrow; frames described by the host | Core takes only a closed set of Arrow types; each language converts its own types through a reviewed table | The core grew six pandas column models and parsed dtype names |
| Core never calls into the host (stated) | Checked in CI: core crates can't depend on Python or Java bindings or name Python types, and a pure-Rust test host runs every contract case | The implementation called back into Python in 19 ways |
| SQL dialects inside the core | Data access is pluggable: the core emits typed data requests; SQL dialects are one built-in provider, and the layer can be swapped or a host can answer with its own provider (for example for Pyramid) | Agreed when the proposal was posted, then lost: the first attempt hard-coded the dialects |
| PQL: a hand-written port | One PQL implementation, the PQL team's C++ core, used through its published Rust crate | Several PQL copies already drift |
| Port the client logic | Port defined algorithms; use Rust libraries for general data work; keep presentation and object behavior in the host | 27-29% of the Rust re-created libraries by hand |
| Python first, Java much later | Python leads every slice; Java runs the same contract cases from the start, so Python assumptions in the core surface early, but never holds Python back | Python assumptions surfaced only at the end |
| Module-by-module phases, merged at the end | Vertical slices through the public API, each merged into an epic branch at its gate; `main` stays pure Python until the owner merges the epic | Modules passed alone and broke together; nothing merged |

Unchanged: Rust, no I/O in the core, the run loop that asks the host to fetch data or call the model, official database drivers per language, and per-language packaging.

## 2. Decisions

The owner made these before any code:

1. **Python comes first.** Python is the main interface and matters most: no regressions for Python users, and Python's needs set the order of the work.
   Java is a second binding on the same core, built so the core stays language-neutral, never at Python's expense.
2. **Scope:** Python supports every backend it supports today (Databricks, Snowflake, Lakebase, Postgres, SQLite, DuckDB and in-memory frames, including files read from Amazon S3).
   Java starts with DuckDB, SQLite, Postgres and Databricks.
   Platforms: the client becomes one CPython `abi3` wheel per platform (3.10 and later, no sdist) for Linux x86_64 and aarch64 (`manylinux2014`, and `musllinux_1_2` if the PQL C++ builds there), macOS (10.12 x86_64, 11.0 arm64) and Windows x86_64, with `pyarrow>=10.0.1` and `pandas>=1.5`.
   Dropped by the owner: 32-bit, Windows on Arm, other architectures, PyPy, GraalPy and free-threaded CPython; Java natives are Linux x86_64 and aarch64 plus macOS, and only tested environments are claimed (`phase-0/support-matrix.md`, decisions G1 to G14).
3. **Regressions:** refusing an input the old Python client accepted is a regression, and so is dropping a platform or version Python supports today, unless the owner decides it explicitly.
   Java on Snowflake is not prioritized, so it is not held to this yet.
4. **Python quirks:** the register (section 5) follows common sense, with a small bias towards today's Python behavior but without catering to its accidents.
5. **Reference environment:** the reference branch `rust-core-reference` is `main` plus the minimum determinism fixes, never contains Rust, and is pushed to GitHub so it isn't lost; it is the reference the tests compare against.
   The fixes are only those needed for repeatable answers (seeded draws, stable ordering, independence from the hash seed), taken from the first attempt's `rust-core/p1-determinism` work, with no refusals and no other behavior changes; `phase-0/determinism-fixes.md` lists the selection.
   Each kept fix is first compared with unpatched Python, and any changed outcome is registered before the reference adopts it.
   It is installed in its own worktree and environment at `~/code/sdmc-reference`, with its Python, pandas, NumPy, PyArrow and torch versions recorded.
   Reference comparisons (L3, the full L2, the live and job matrices) run locally on the developer's machine, not in CI.
   It tracks `main`, which is safe because `main` never contains the Rust core before the epic merges; at each slice gate it is updated to the latest `main` and the full comparison rerun, and a new difference must trace to a deliberate commit on `main`.
   The original baseline is kept as an immutable tag `reference-baseline-<sha>`, and a guard fails if the reference ever loads the Rust core.
   Every test run records the reference commit and the identity of the installed artifact it used.
6. **PQL:** the PQL team owns one C++ core and publishes it as a Rust crate and a Python package; this client's core uses the Rust crate, with the hand-written parser as a stopgap (section 3).
7. **Data layer:** the data layer must be switchable: a team given access to this repository should be able to replace it, for example to work with Pyramid, without touching the rest of the core.
   A Pyramid provider itself is not in scope.
8. **Java:** the first Java consumer is a team using Pyramid, which brings its own data layer.
   Java gets basic support for the offline backends and Databricks, proven working on Databricks, and loads the core as a native library; WebAssembly is out, because its size limits are too restrictive.
9. **No calendar budget and no time estimates:** the work is paced by its gates, not by dates, and dates must not change the quality of the work.
10. **Ownership:** this team owns the core long term; that does not change how it is built.
11. **Epic branch:** an epic branch (`rust-core`) based on `main` holds all migration work.
    Each phase's work happens on its own branch off `rust-core` (for example `rust-core-phase-0`), pushed to GitHub so nothing is lost, and merges locally into `rust-core` once verified (section 9); there are no pull requests.
    Slices merge into the epic branch, never into `main`; `main` stays pure Python and keeps moving, and the epic branch is updated from `main` at each slice gate.
    Merging the epic into `main` is the owner's decision at the end; until then the work is experimental.
12. **People:** one developer works with agents.
    The independent reviewer is a fresh agent that did not implement the change; the owner is the only human approver.
    The owner approves by merging a verified phase or slice branch locally, by decisions recorded in the phase's decisions file, and by register entries marked `approved_by: owner`.
    An agent never approves its own exception or marks anything `approved_by: owner`.
13. **The first attempt:** its best candidate (commit `ab85a056`, tag `pm/2026-10-07/all-public-cases-pass`) is archived on GitHub as `archive/rust-core-attempt-1` (commit `3887762`, internal catalog and workspace names redacted), a source to port from that is never merged.

Still to agree with the PQL team: one PQL dialect, the versioned validated-query format, the conformance suite, platform builds, and when the Rust crate ships.
Phase 0 sends these asks and tracks them in `phase-0/pql-asks.md`; agreement is not a Phase 0 exit.

## 3. Architecture

### The contract

- **Inputs:** Arrow record batches restricted to a closed type set (below).
  Each language converts its native types through a reviewed table, without losing values or the metadata that matters (precision, scale, units, missing values, order).
  Inputs the old client supported stay supported unless the register says otherwise; anything else is refused by name.
  The core never sees a pandas dtype or a Java class.
- **Operations:** a versioned list of typed requests and responses (serde types, published as JSON Schema), not functions looked up by string name.
- **Runs:** the existing `Run`/`Step` loop, executed by one state machine in `sdm-core`.
  A data step carries either SQL (for SQL providers) or a typed data request (for custom providers), with ordered typed parameters and the Arrow schema it expects back, or a marker that the schema is being discovered.
  The host returns Arrow or a structured error.
- **Scalars:** the contract defines exact encodings for large integers, decimals, non-finite floats and temporal values, including precision, scale, units, overflow and rounding.
  Control messages round-trip them through Rust, Python and Java without floating-point conversion.
  There is no general Python scalar or object model.
- **Errors and warnings:** `{kind, code, message, fields, cause}` with a closed list of kinds, one channel for each, and warnings kept in order.
  Each language maps kinds to its own exception and warning classes.
  Codes are a closed, listed enum with a `retryable` flag, and each language's mapping is tested exhaustively, never reading message text (Phase 1 decision H6, approved by owner 2026-10-08).
- **Host facts:** clock, time zone, the tzdata version and similar facts are passed in explicitly, so tests can pin them; the core reads no locale.
- **State and memory:** typed, versioned values the host holds; Arrow buffers and runs have defined owners and release rules; no handles into host objects.

### The type set

| Kind | Arrow types |
|---|---|
| Integers | int8, int16, int32, int64, uint8, uint16, uint32, uint64 (kept as they are: today's payloads distinguish them) |
| Floats | float32, float64 |
| Decimals | decimal128, decimal256 |
| Text and bytes | utf8, binary (hosts convert `large_*` without loss, or refuse by name) |
| Time | date32, timestamp(ns or us, with or without zone), duration(ns or us), time64 (hosts cast `timestamp[s]` and `timestamp[ms]` to `us` without loss) |
| Nested | list, struct, map, dictionary of utf8; child types come from this set (hosts convert `large_list` and `fixed_size_list` without loss, or refuse by name) |
| Other | bool, null (all-null columns) |

Every accepted type is preserved or refused with a named error, never silently turned into nulls.
Arrow-to-Arrow admission into this set is one lossless table in `sdm-host`, shared by every language; a value the admitted type cannot hold is refused by name (Phase 1 decisions B-D6 and E13, pending owner confirmation).
Phase 1b validates this set against its case inventory before it is final.

### What lives where

| Concern | Same in every language? | Where |
|---|---|---|
| PQL parsing, validation, error codes and spans | Yes | The PQL team's C++ core, through its Rust crate, behind one adapter in `sdm-core` |
| The run state machine, public operation dispatch, cancellation, model capability checks and profiles | Yes | `sdm-core` |
| Canonical schema, semantic inference over typed columns (keys, time column, links, semantic types), source-neutral view interpretation | Yes | `sdm-schema` |
| Catalog queries and view retrieval per source, typed data requests, SQL for each built-in dialect, provider capabilities, the in-memory fallback | Yes | `sdm-data` |
| Time rules, random streams, sampling plan, neighbor sampler, labels and aggregations | Yes | `sdm-engine` |
| Materialization, session state, decomposition, payload limits, payload encoding, response decoding, result layout | Yes | `sdm-request` |
| Control-message codecs, native resource lifetime, panic containment, threading | Yes | `sdm-host` |
| Native types to Arrow (pandas dtypes, JDBC types) | No | Host, as a reviewed table |
| Running SQL on the user's connection, or answering data requests another way | No | Host data providers |
| Object behavior, exceptions, presentation (`to_pandas`, Java records) | No | Host |
| HTTP and authentication | No | Host |

Dependencies flow one way: bindings → `sdm-host` → `sdm-core` → domain crates → `sdm-contract`.
SQL fragments and provider matches never appear in `sdm-schema`, `sdm-engine` or `sdm-request`.

### Data access

The core asks for data through a small, fixed vocabulary of typed data requests.
The minimum every provider supports is two operations: describe the catalog, and scan a table with a column list and key and time-range filters.
Optional operations make common steps faster, and each provider declares which it supports: rows by key, rows in a time window, counts and minimum or maximum times, the latest neighbors per key, and a seeded sample of entities.
Before a run, the core picks pushdown or an in-memory fallback from the provider's declared capabilities.
Core-side sampling uses the registered random streams over a defined ordering of the same data, so it gives the same rows on every provider; a provider's own seeded sampling is repeatable only for that provider, and changing a backend's sampling needs a register entry.
A declared capability that fails is reported as an error, never silently replaced.

- **SQL providers:** the core compiles requests to SQL for each built-in dialect (Databricks, Snowflake, Postgres, SQLite, DuckDB), and the host runs it with the vendor's driver.
- **Custom providers:** the host receives the typed request and answers it its own way, for example by compiling it to a Pyramid query.
- **The provider contract** is versioned and defines null and duplicate keys, time units and zones, interval bounds, ordering of ties, typed errors (with retryability and driver codes), and complete results: scans page through every matching row, nothing is silently truncated, and the in-memory fallback has declared memory and transfer limits that fail loudly when exceeded.
- **Testing:** a provider conformance kit (a fixture dataset, multi-page results, limits, and expected answers including hand-checked ones) that every provider passes, the built-in SQL ones included, ending with a full prediction compared with the reference provider.

The data layer is one crate behind one documented interface, with the SQL providers as its reference implementation and the conformance kit as its acceptance test, so a team with access to this repository can swap it, or implement a provider in Java, without touching the rest of the core.
S1 proves this: a minimal non-SQL test provider passes the conformance kit, and the same prediction runs with an alternate data-layer implementation swapped in at the composition root, with no edits to domain crates.
A Pyramid provider is not part of this plan.

### PQL

![One PQL implementation](images/pql.png)

Several separate PQL implementations exist today, and only two share tests.
The drift risk is less in the grammar than in the code around it (tree building, error handling, validators), so "defined once" means one implementation.

- **One implementation, owned by the PQL team:** a C++ core holding the grammar, parser, validator, error codes and spans, and the validated query as versioned JSON.
  The team publishes it as a Rust crate (wrapping the C++ core) and a Python package.
- **This client's core uses the Rust crate**, so this client's Python and Java get PQL through the core; Kumo's consumers use the Python package or the C++ core directly.
- **One conformance suite** of valid and invalid queries with expected codes and spans, seeded with this client's 417 cases, run by every consumer.
- **Until the Rust crate is published,** this client's Python and Java both use the first attempt's hand-written Rust parser as an interim parser, behind one adapter in `sdm-core` and held to the shared suite; today's ANTLR grammar stays as a CI reference.
- **As soon as the PQL team's artifact exists,** it is exercised behind the same adapter in the active slice: build, load, validated-query format, codes and spans, and coexistence with its Python package in one process.
  In Phase 5 this client switches to a pinned release of the crate and deletes the interim parser; deferring that is an explicit scope decision.
- **What to ask the PQL team for (the PQL asks):** a C interface or a Rust crate that never lets a C++ exception cross into Rust; static libraries for every platform this client ships (Linux x86_64 and aarch64 as `manylinux2014`, and `musllinux_1_2` if feasible; macOS x86_64 and arm64; Windows x86_64), with private C++ symbols so two copies can live in one process; matching crate and PyPI releases; no aborts or console output, and a recursion limit; a versioned validated-query format with a JSON Schema; and the conformance suite in their repository, run in their CI.
- **This client keeps** only the model capability checks on top of the validated query.

### Bindings

![Bindings](images/bindings.png)

The core is plain Rust and doesn't know which language loads it.
Each language uses its standard binding tool, over one shared Rust layer:

- **`sdm-host`**, plain Rust, written once: encodes control messages as versioned JSON, hands data over as Arrow, owns native resource lifetimes, contains panics, and sets the threading rules; it schedules no domain work.
  Its calls take and give control messages as bytes with the attachments each names (Phase 1 decision E13, pending owner confirmation).
- **Python:** a thin PyO3 binding built with maturin as `abi3` wheels; it converts types, maps errors to Python classes, releases the GIL while the core runs, and passes Arrow through the standard PyCapsule interface without copying.
  For pyarrow older than 14, Arrow crosses through `_export_to_c` instead of PyCapsule.
- **Java:** a thin `jni-rs` binding loaded from the JAR (JDK 17 and 21); it maps errors to Java classes and passes Arrow through Arrow Java's C Data module without copying.
- **Native safety:** first-party domain crates forbid `unsafe` Rust; only the binding and host adapters and external dependencies (Arrow, the PQL wrapper) contain native code, and they are tested explicitly.
  Semantic callbacks into the host are forbidden; Arrow release callbacks are allowed.
  C++ exceptions are translated inside the PQL wrapper.
  A Rust panic is contained at each exported entry and invalidates its run.
  Aborts and memory corruption are tested in subprocesses, not claimed recoverable.
  Memory has one owner and an explicit release, and the threading model is written down.
- **Java runtime flags:** users run with `--enable-native-access` (Java warns without it from JDK 24 and will later refuse) and the flag Arrow Java needs; the Java consumer's environment must allow both.
- **PQL's C++ is linked into both native libraries** with its symbols kept private, so a Python process that also loads the PQL team's own package gets no clashes.
- **Platforms and versions:** the owner decided the native coverage in Phase 0 (decision 2 and `phase-0/support-matrix.md`): Python wheels as above, and Java natives for Linux x86_64 and aarch64 plus macOS, to confirm with the Java consumer.
  pyarrow 8, as Databricks Runtime 14.3 ships, is upgraded by the `pyarrow>=10.0.1` floor (raised to 14.0.1 on the epic branch by Phase 1 decision E16).
  Dropping any further platform or version, or moving to `manylinux_2_28`, is an explicit owner decision, never a side effect.

Considered and not chosen: one C interface (`extern "C"`) with generic glue per language, because it means maintaining our own unsafe boundary where PyO3 and `jni-rs` already provide tested ones.
Other binding tools and running the core in a separate process are out of scope unless the owner changes it.

### Performance rules

Today a predict's wall time goes to warehouse round trips and the model call, and most client CPU goes to building the payload cell by cell in Python.
The core keeps the round trips as they are and should cut the CPU, as long as these rules hold:

- Data providers hand the driver's Arrow straight to the core, with no pandas round trip; drivers without Arrow build Arrow columns from rows directly.
- Tables cross zero-copy through the Arrow C Data Interface.
- JSON is only for small control messages, capped (for example at 64 KB); anything row-shaped travels as Arrow.
  Byte buffers and Arrow tables cross as attachments named by id (Phase 1 decision E2, pending owner confirmation), and so does every field whose size grows with user input, such as query text and schemas, so the cap never refuses an input (Phase 1 decision H5, approved by owner 2026-10-08).
- The host never parses or re-encodes the model request or response body; it passes the bytes on.
- Data requests are batched per hop, with no chatty per-row or per-entity steps.
- Object columns convert column-wise, never cell by cell.
- pandas 3's `large_string` and `timestamp[s]` are cast cheaply in the conversion table, not refused.

L7 (section 6) checks these.

### Crates

The names and boundaries below are the target ownership, not a reason to rename or rewrite everything before the first slice works.

| Crate | Holds | From today's tree |
|---|---|---|
| `sdm-contract` | Versioned types: the type set, operations, Step, request and response schemas, data requests and provider capabilities, errors and warnings, exact scalar encodings; no orchestration | New, folding in `protocol` |
| `pql` (from the PQL team) | Rust crate wrapping the C++ PQL core | External dependency, pinned by version and digest; replaces the interim parser |
| `sdm-data` | Catalog queries and view retrieval per source, typed data requests, SQL for each dialect, provider capabilities, the in-memory fallback | Keep the dialect backends behind a provider interface; drop the capture and render scaffolding |
| `sdm-schema` | Canonical schema, inference over Arrow, source-neutral metric and semantic view interpretation | Keep the engine; move catalog SQL to `sdm-data`; drop the pandas column models and dtype parsing; YAML through a library |
| `sdm-engine` | Modules for time (one calendar, the date parsing rules recorded in Phase 0, `DateOffset`, time zones), random streams, sampling (plan, neighbor sampler, graph store) and labels (kernels and windows) | Merges today's time, random, sampling and label code with one value model; time on a library, not a pandas date-inference clone; Arrow compute where answers allow |
| `sdm-request` | Modules for materialization, session state, decomposition, limits, payload encoding, response decoding and result layout | Keep the logic; drop the CPython sort and `json.dumps` size emulation |
| `sdm-core` | The single Run state machine, public operation dispatch, cancellation, model capability checks and profiles; the interim PQL parser (crate `sdm-pql-interim`, Phase 1 decision A3, pending owner confirmation) behind one adapter | Rewrite the host API; typed operations only; capability checks from today's `pql` crate |
| `sdm-testkit` | Contract runner, provider conformance kit, fixtures | Dev-only; nothing in the shipped crates |
| `sdm-host` | Control-message codecs, Arrow handoff, native resource lifetime, panic containment, threading; no domain scheduling | New, written once |
| `sdm-py`, `sdm-jni` | Thin PyO3 and `jni-rs` bindings over `sdm-host`: convert types, map errors, release the GIL | Rewrite `sdm-py` much slimmer; new `sdm-jni` |

Today's `wire` crate is generated from the model API; `sdm-request` uses it, and there is no second hand-written wire model.

### Dependencies

| Need | Choice |
|---|---|
| Columnar data and kernels | `arrow` crates, including `arrow-cast`, `arrow-ord`, `arrow-arith`, `arrow-select`, `arrow-row` and `arrow-ipc` |
| Decimals and big integers | Arrow's decimal types and `arrow-buffer`'s `i256`, under the contract's exact-encoding policy |
| PQL parsing | The PQL team's Rust crate over its C++ core (so every build also compiles and links C++) |
| Dates, times, zones | `jiff`, with the host's ordered zone lookup roots and a bundled fallback, the source and version recorded per zone; `chrono` only as Arrow's transitive dependency and `chrono-tz` banned (Phase 1 decision A7, pending owner confirmation): it met DT-1 and DT-2 on every T0.6 zone and `DateOffset` answer, where `chrono-tz` failed. Instants past `jiff`'s range use civil arithmetic and are never refused for that reason (Phase 1 decision E6, pending owner confirmation). Inside the core, one instant and one span type in nanoseconds, columns kept as Arrow arrays, conversions checked (Phase 1 decision F3, approved by owner 2026-10-08) |
| YAML (metric and semantic views) | A maintained YAML 1.2 crate such as `saphyr` or `yaml-rust2` (not the archived `serde_yaml`); behavior differences go through the register |
| JSON | `serde`, `serde_json` |
| Error types | `thiserror`; no `anyhow` in core crates |
| Logging | `tracing`, bridged to Python `logging` by `pyo3-log` with the GIL released during core calls and a deadlock regression test (Phase 1 decision H8, approved by owner 2026-10-08), and to SLF4J, decided in the Java phase |
| Output order | `indexmap` or `BTreeMap` for anything that affects output order; `HashMap` iteration in outputs is banned through clippy's `disallowed_types` |
| Encoding and hashing | `base64`, `regex` and `sha2` where needed |
| Bindings | `pyo3` with `maturin` (abi3 wheels); `jni` (`jni-rs`); the Arrow C Data Interface in `arrow`, `pyarrow` and Arrow Java |
| Dataframe engine | Decided at Phase 2 entry: compare plain Arrow kernels with DataFusion on one label aggregation and one sampling join; correct answers are required, then weigh hand-written code, maintenance, speed, binary size and JNI loading; record the results (Polars is out: it uses its own Arrow implementation, not arrow-rs) |

**Policy:** `Cargo.lock` is committed and the toolchain is pinned; the pinned toolchain is the minimum supported Rust version.
`cargo-deny` checks advisories, licenses, bans and sources in L0, starting from the first attempt's `deny.toml` and toolchain file.
The wheel and JAR ship third-party notices, including for statically linked C++.
Dependency updates are reviewed changes made only at slice gates, with `arrow`, `pyo3` and `jni` updated together (today Arrow 58, `pyo3` 0.28 and `duckdb` 1.10506.0: Phase 1 decision B-D2, pending owner confirmation).
The PQL crate and its C++ artifact are pinned by version and digest.

On the Python side, pandas and pyarrow stay as host dependencies, and the old engine is deleted on the epic branch in Phase 5.

## 4. Delivery: slices, not modules

![Delivery](images/delivery.png)

| Slice | Public path | Languages |
|---|---|---|
| S0 | PQL validation | Python; Java follows |
| S1 | `Schema.from_source` → `materialize` → `predict` on DuckDB, SQLite and Postgres, offline, including ranking and forecasting; a minimal non-SQL test provider passes the conformance kit, and an alternate data layer swapped in at the composition root runs the same prediction with no domain edits | Python; Java follows |
| S2 | The same live on Databricks (with a metric view), Snowflake (with a semantic view) and Lakebase, including ranking and forecasting; the Databricks job matrix across runtimes; Python and Java agree on the same Databricks data (for Java, the questions without ranking or forecasting) | Python on all three; Java follows on Databricks |
| S3 | In-memory frames (including files read from Amazon S3) with their ranking and forecasting, link prediction, sessions, tabular requests | Python; Java as the owner adds scope |

Amazon S3 is unrelated to slice S3: `sdm_connectors.read` reading CSV or Parquet from `s3://` into pandas stays in the host, and its frames are covered as S3's in-memory frames.

A path is one public operation with one input mode on one backend (for example `predict` of a regression query on Snowflake), and one routing boundary chooses the old implementation or the core for each path.
Python switches a path on the epic branch only after its slice passes all its gates, live ones included: S0 (PQL validation), S1 and S2 paths switch together after S2, in Phase 3, so no call mixes old and new code, and S3 paths (frames, link prediction, sessions, tabular requests) switch after S3.
Phase 0 maps every Databricks job in the matrix to its paths, so S2 includes what its jobs need.
Each phase's branch merges locally into the epic branch once verified, never into `main`; the epic branch is updated from `main` at each gate, and merging it into `main` is the owner's decision at the end.
The old engine is deleted on the epic branch in Phase 5; the reference (decision 5) stays available after that.
Python sets the pace: a slice ships for Python when its Python gates pass, and Java follows on the same contract cases without holding it back.

## 5. The quirk register

The register is a machine-readable file at `docs/design/rust-core/register.yaml`.
Entries follow common sense, with a small bias towards today's Python behavior but without catering to its accidents.
Every entry names the old and new behavior, the affected inputs and contract case IDs (or issue or task IDs until Phase 1a creates case IDs), and who approved it: the independent reviewer approves entries, and an entry that refuses an input Python accepted needs the owner's approval (`approved_by: owner`), since that is a regression.

| Keep (part of the promise) | Drop (by recorded decision) |
|---|---|
| The random streams of the reference (numpy `RandomState` and `default_rng` where it uses them, `std::mt19937` for neighbor draws), pinned in Phase 0; keeping `default_rng` means porting its generator too; their vectors are never re-recorded to pass a dependency bump (Phase 1 decision F4, approved by owner 2026-10-08) | CPython's sort order around NaN scores (NaN ranks last) |
| Label arithmetic as today: float32 Kahan sums in row order | The size limit counted as `json.dumps` bytes (count the bytes sent) |
| `DateOffset` month-end rules | Python object protocols inside the core |
| PQL codes, spans and grammar quirks, as agreed with the PQL team | Errors as Python class paths |
| PQL date-literal rules, written down | Silent nulls or internal errors for types the old code mishandled |
| Today's public exception and warning classes and codes | Time zone rules ignored after 2037 |
| Text-column timestamp detection, under the reference pandas major (pandas 3: Phase 1 decision H1, approved by owner 2026-10-08) | |
| Integer widths in payloads | |
| Payload equality after parsing; SQL per database as reviewed golden files | |

Each needs its own decision, not a blanket rule: answers that differ between pandas 1.5, 2 and 3, SQL text changes, and YAML 1.1 values such as `yes` and `on`.
A difference only one runtime shows is a register entry naming that runtime, never an exclusion (Phase 1 decision H1, approved by owner 2026-10-08).

## 6. Testing

| Layer | What it checks | When |
|---|---|---|
| L0 Boundary guards | First-party domain crates forbid `unsafe` and their production dependency graph has no `pyo3`, `jni` or `arrow-pyarrow` (only the binding crates may use them); the run steps are only data and model steps (no requests for language behavior such as truthiness, sorting or dtype inference); a pure-Rust test host passes the contract subset; the contract's JSON Schema, run steps and the PQL validated-query format don't change without an owner-approved version bump (within a phase, each bump is a decisions row and the phase lands as one bump per merge into `rust-core`: Phase 1 decision H4, approved by owner 2026-10-08); `cargo-deny` and clippy's disallowed types | CI, on every push to `rust-core` and phase branches |
| L1 Unit and property tests | Algorithms in each crate, with every past mismatch seed; the shared PQL conformance suite; fuzzing of the PQL, YAML and contract-JSON decoders | CI, on every push to `rust-core` and phase branches |
| L2 Contract corpus | One shared list of cases, run in each language: a public call and its inputs in; ordered data requests, SQL and typed parameters, payload, result, warnings and error kind, code, details and cause out; row order compared only where the operation defines it; SQL traces compared only for SQL providers, and data-request traces checked against their own contract (against the reference, statement count and each statement's result gate, with order only under a total `ORDER BY`, while typed parameters and raw SQL text are diagnostic; the core's SQL is pinned by goldens: Phase 1 decision H9, approved by owner 2026-10-08); pickles written by the reference load in the new client, and new state carries a version; a coverage matrix is generated from the scope manifest (per language) over the case inventory for the current slice, with representative nested shapes and one generated refusal case per unsupported type, and an empty cell fails | CI on every push (subset, for feedback); local, before each slice gate (all) |
| L3 Differential generator | Random public-API inputs compared with the reference (decision 5), installed in its own environment, before and after the old code is deleted | Local, before each slice gate |
| L4 SQL goldens and provider kit | Reviewed SQL per database, executed on DuckDB, SQLite and Postgres; the local providers pass the conformance kit | CI, on every push to `rust-core` and phase branches |
| L5 Conversion tables | Each language's native-to-Arrow table, on pandas 1.5, 2 and 3, in CI cells named by runtime from `phase-0/support-matrix.md` (Phase 1 decision H1, approved by owner 2026-10-08); the same tables loaded into DuckDB, SQLite and Postgres reach the core with identical Arrow schemas, values, null masks, order and required metadata from Python's drivers and Java's JDBC, checked against expected values (large integers, decimal scale, timestamp units, nested nulls, list widths, dictionaries) | CI, on every push to `rust-core` and phase branches |
| L6 Live matrix | The scenarios in the scope manifest (per language) over the case inventory, on Databricks, Snowflake and Lakebase (Databricks' managed Postgres), plus the Databricks job matrix (one-time jobs across Databricks runtimes, compared with the reference's results); old versus new, and Python versus Java on the same data | Local, before the S2 slice gate and the switch, epic merge and release gates |
| L7 Performance | Offline, on recorded runs: query counts equal; CPU per scenario at most today's (aim well below; the first attempt measured about half), no scenario above 1.1 times; boundary overhead per in-memory run within a small fixed bound; the control-message size cap; zero-copy and "no pandas in providers" checks; import time and native library size within set limits; live: push-down median over at least five interleaved runs, caches off, within 10% on Databricks and Snowflake | Local: offline before each slice gate; live at switch and release gates only |
| L8 Packaging and safety | The real wheel and JAR install and load in clean containers (JDK 17 and 21, every platform in the declared support matrix); exported symbols, private C++ symbols and minimum system library versions checked; a Python process loading both this client and the PQL team's package; `sdm-host` under sanitizers; panic, C++ error, crash and leak tests in each language; HTTP retry, timeout and redaction tests | Focused checks whenever bindings or native dependencies change; the full platform matrix before the epic merges |

- **Scope manifests:** Python and Java each have a scope manifest (per language) over the case inventory.
  Python gates require the Python and pure-Rust checks; Java-only checks (such as JDBC conversion equality in L5) block only Java shipping, but a shared-core defect they expose blocks the affected path.
- **Ratchet:** the checks compare every case's full observed outcome with the last-green baseline (CI for the L2 subset, locally for the full corpus), so a changed answer, a new refusal or a removed case fails unless a register entry states that case's exact old and new outcome.
  The baseline is updated as part of each merge into the epic branch.
  This replaces choosing "neighboring cases" by hand.
  Each gate run uses the harness, runners, comparator, ratchet, guards and expectations of the checkpoint, the merge base with the phase branch (the last reviewed landing), not the candidate's; a protected path changes only when the independent review names it, an outcome only with an approved register entry, and a relaxation only with the owner; a field a runner cannot observe never passes (Phase 1 decision H2, approved by owner 2026-10-08).
- **Runners:** the reference is the oracle; the old client on the working branch is a smoke runner and the comparison base on runtimes other than the reference's (Phase 1 decisions H1 and H10, approved by owner 2026-10-08).
  The reference stays recorded on the owner's machine, where comparisons with it run; CI gates the platform-independent checks (Phase 1 decision H1, approved by owner 2026-10-08).
  About 40 seeded cases unseen by the implementer run once per phase gate for information, with a no-go threshold (Phase 1 decision H11, approved by owner 2026-10-08; the threshold value pending owner confirmation).
- **Run time:** keep the CI checks fast and run slow ones locally before each gate; run time is something to optimize, never a reason to skip cases.
- **Test the tests:** before the L2 runners count for acceptance, show they fail on a corrupted expectation, a missing case, a skipped call and a missing native library.
- **Local first:** reproduce a failure from the installed package before spending a remote run on it; a failure that only happens remotely gets one bounded diagnostic run.
- **Retiring old tests:** the first attempt's internal-equivalence tests go once the L2 and L3 failure families they caught are covered, measured by seeding those failures and checking the new layers catch them.
  Algorithm unit tests stay.

## 7. Release gates

| Gate | Passes when |
|---|---|
| A change lands on a phase branch | L0, L1, L4, L5 and the L2 subset green in CI for Python and the pure-Rust host; the ratchet holds; a type or conversion fix comes with a public reproduction that failed first; focused L8 checks if bindings or native dependencies changed |
| Slice done (its branch merges locally into the epic branch) | The epic branch updated from `main` and the reference rerun on it locally; the full L2 for the slice green in Python (Java must be green for its own scope before it ships that slice); L3 on the agreed seed sets with no unexplained divergence; every difference has an approved register entry; for S2, L6 green for its scenarios; a fresh-agent review; the owner merges it locally |
| Python switch for a path (on the epic branch) | Its slice done; L6 green for it; the Databricks job matrix equal to the baseline; L7 within its limits, live timing included |
| Epic merges into `main` (the owner's decision) | Every slice done and switched; the full L8 platform matrix green; L6 and the job matrix green on the epic commit; the existing Python test suites green |
| Release | Every applicable layer green on the release commit, with the old engine deleted, including L6 and the job matrix; live test resources cleaned up |

**Release and versioning:**
- The native core ships inside the structured-data-models-client wheel, so users still install one package, unless a later decision says otherwise.
- The contract version is independent of the package version.
- Error codes and classes are promised; error message text is not.

## 8. Phases

| Phase | Work | Exit |
|---|---|---|
| Phase 0 | Create the epic branch `rust-core` and the phase branch `rust-core-phase-0`; extend `.github/workflows/ci.yml` to run the fast checks on pushes to `rust-core` and phase branches; create and push `rust-core-reference` (decision 5) with the fixes listed in `phase-0/determinism-fixes.md`, set up its worktree and environment at `~/code/sdmc-reference` with its guard, and tag the baseline `reference-baseline-<sha>`; compare each kept fix with unpatched Python and register changed outcomes; record today's platform and version support and the proposed wheel coverage; record the accepted text-date formats and inference rules; check `chrono` with `chrono-tz` against the date corpus; pin the random streams; map every Databricks job to its paths; choose the Phase 1b case inventory; send the PQL asks (section 3) and track them in `phase-0/pql-asks.md`; archive the first attempt (decision 13) | The reference branch, baseline tag and guard work; decisions recorded in `phase-0/decisions.md`; CI runs on `rust-core` and phase branches; the PQL asks drafted, with sending deferred by the owner (`phase-0/decisions.md` row 31; agreement is not a Phase 0 exit) |
| Phase 1a | `sdm-contract`, `sdm-testkit`, L0 guards, L2 runners for Python and the reference, the CI fast checks and the local gate checks; build and clean-install a real abi3 wheel in CI | S0 green in Python; the runners fail on seeded faults; the wheel installs and loads in a clean container |
| Phase 1b | Spike, fixed in scope: DuckDB `from_source` → `materialize` → `predict` → result through the new contract and the data provider interface, in Python and through the pure-Rust test host (which shows the contract needs nothing from Python), against a scripted model endpoint; the case inventory chosen in Phase 0 covers the hard seams: keys, decimal precision, nested values, empty and all-null data, and time inference; the text-date rules recorded in Phase 0 checked against the reference | Matches the reference on that inventory (inferred types, keys, time columns, data requests and SQL, payloads, results, warnings and refusals); the owner decides go or no-go (go: Phase 1a and Phase 1b merge into the epic branch and Java's S0 work starts; no-go: repair the first attempt in place, or stop) |
| Phase 2 | The dataframe-engine comparison at entry; S1, moving only the code S1 needs off the Python value model and onto the contract; the provider conformance kit, the minimal non-SQL test provider and the swapped data layer | Slice gate |
| Phase 3 | S2 and the Python switch for S0, S1 and S2 paths, on the epic branch | Switch gate |
| Phase 4 | S3, moving the rest of the code as its paths need it | Slice gate and switch gate |
| Phase 5 | Switch to the PQL crate release; delete the interim parser and the old engine on the epic branch; release candidate; the owner decides whether the epic merges into `main` | Epic merge gate, then release gate |

## 9. Process rules

| Rule | How it's enforced |
|---|---|
| The epic branch holds the only candidate; work on the next slice starts only after this slice's gate passes; independent tasks within a slice may run in parallel | CI fast checks on the epic and phase branches; the status page lists open branches |
| Each phase's work happens on its own branch off `rust-core` (for example `rust-core-phase-0`), pushed to GitHub so nothing is lost; a candidate that passes its checks is pushed before more work builds on it; phase branches never merge into `main`; the epic branch is updated from `main` at each slice gate | Once verified (CI fast checks, local gate checks, a fresh-agent review), the phase or slice branch is merged locally into `rust-core` and `rust-core` is pushed; there are no pull requests |
| Acceptance is what CI and the local gate checks run, plus one independent reviewer per change (a fresh agent that did not implement it), the owner's decisions, and one independent release audit of the existing evidence | The owner's approval is the local merge of a verified phase or slice branch, plus decisions in `phase-N/decisions.md` and register entries marked `approved_by: owner` |
| At most two review rounds per change; one independent reviewer checks code and behavior, not prose; a third round goes to the owner | Recorded in the phase's tracker |
| A phase's acceptance checks are locked before implementation; each change states what and why; a third change to a phase's locked acceptance checks goes to the owner | Locked files change only with the owner's approval |
| Approval by meaning: one decision approves all the files it explains | Register entries and owner decisions; no per-file hashes |
| Stop rules (below) are lifted only by the owner, never by delegation; on a stop, implementation and live runs pause and the evidence is reported; each phase names owner check-ins, with a five-line report, past which work waits for the owner's answer (Phase 1 decision H12, approved by owner 2026-10-08) | The owner's decision in `phase-N/decisions.md`; an agent never approves its own exception or marks anything `approved_by: owner` |
| One status page: candidate commit, gates passed, open failures, decisions needed | Generated from CI and git, not written by the executor |
| Anything that changes behavior or scope goes to the owner straight away | Register entries and the status page |
| New acceptance infrastructure beyond the agreed test runner needs the owner's approval | Listed on the status page |
| Each phase has `docs/design/rust-core/phase-N/design.md` and `tracker.md` on its phase branch | Created with the phase branch |

Where things live: the reference worktree at `~/code/sdmc-reference`; the baseline tag `reference-baseline-<sha>`; owner decisions in `phase-N/decisions.md`; the register at `docs/design/rust-core/register.yaml`, whose entries may cite issue or task IDs until contract case IDs exist in Phase 1a.

**Stop rules:**
- a phase that keeps finding new work instead of closing its exit checks stops for a re-plan with concrete cuts;
- a check failing twice the same way changes method, never adds a phase;
- after two fix rounds on the same divergence, the owner chooses between repairing it, deferring the slice with the old path unchanged, and approving a named behavior change; nothing becomes an accepted refusal automatically;
- three live failures for non-product reasons stop live work;
- if S1 can't go green for Python, the owner chooses the next step;
- while a blocking question waits for the owner, work narrows to the current slice; silence never authorizes going on.

## 10. Risks

| Risk | Mitigation |
|---|---|
| The Java consumer's environment can't load a native library | Check it with the consumer before Java packaging; if it can't, the owner decides Java's scope |
| The PQL Rust crate isn't ready when needed | The interim parser, held to the shared conformance suite |
| The C++ PQL core complicates every build and can crash across the boundary | See the PQL asks in section 3; exercise their artifact as soon as it exists; load and crash tests in L8 |
| A native library builds but doesn't load on some platform | Phase 1a installs a real wheel; L8 installs the real artifacts in clean containers whenever native code changes, and for every platform before the epic merges |
| Native wheels silently drop a platform or version Python supports today | Phase 0 records today's support; any drop is an explicit owner decision |
| The Java consumer's environment disallows the needed runtime flags | Confirm `--enable-native-access` and Arrow Java's flag with the consumer before Java packaging |
| A team swapping the data layer for Pyramid finds the interface insufficient | The interface is documented, a minimal non-SQL provider and a swapped data layer prove it in S1, and the conformance kit defines what a replacement must do |
| Host conversion tables drift between Python and Java | L5 checks both languages convert the same table to the same Arrow types |
| The new client is slower than today's | The performance rules in section 3, checked offline by L7 |
| Process grows faster than product again | Rules enforced by CI and the owner, not by the executor; new acceptance infrastructure needs approval |
