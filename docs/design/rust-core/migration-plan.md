# Rust core migration: the new plan

**Status: proposed.**
It replaces the first plan once the owner approves it; the owner's decisions are in section 2.
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
3. **Regressions:** refusing an input the old Python client accepted is a regression, and so is dropping a platform or version Python supports today, unless the owner decides it explicitly.
   Java on Snowflake is not prioritized, so it is not held to this yet.
4. **Python quirks:** the register (section 5) follows common sense, with a small bias towards today's Python behavior but without catering to its accidents.
5. **Reference environment:** a local reference branch off the latest `main`, with the first attempt's determinism fixes applied (so the old Python gives the same answer on every run), installed in its own worktree and environment with its Python, pandas, NumPy, PyArrow and torch versions recorded; it is the reference the tests compare against.
   It tracks `main`, which is safe because `main` never contains the Rust core before the epic merges; at each slice gate it is updated to the latest `main` and the full comparison rerun, and a new difference must trace to a deliberate commit on `main`.
   Each determinism fix is first compared with unpatched Python, and any changed outcome is recorded in the quirk register before the reference adopts it.
   An immutable copy of the original baseline is kept, and a guard fails if the reference ever loads the Rust core.
   Every test run records the reference commit and the identity of the installed artifact it used.
6. **PQL:** the PQL team owns one C++ core and publishes it as a Rust crate and a Python package; this client's core uses the Rust crate, with the hand-written parser as a stopgap (section 3).
7. **Data layer:** the data layer must be switchable: a team given access to this repository should be able to replace it, for example to work with Pyramid, without touching the rest of the core.
   A Pyramid provider itself is not in scope.
8. **Java:** the first Java consumer is a team using Pyramid, which brings its own data layer.
   Java gets basic support for the offline backends and Databricks, proven working on Databricks, and loads the core as a native library; WebAssembly is out, because its size limits are too restrictive.
9. **No calendar budget and no time estimates:** the work is paced by its gates, not by dates, and dates must not change the quality of the work.
10. **Ownership:** this team owns the core long term; that does not change how it is built.
11. **Epic branch:** an epic branch (`rust-core`) based on `main` holds all migration work.
    Slices merge into the epic branch, never into `main`; `main` stays pure Python and keeps moving, and the epic branch is updated from `main` at each slice gate.
    Merging the epic into `main` is the owner's decision at the end; until then the work is experimental.
12. **People:** one developer works with agents.
    The independent reviewer is a fresh agent that did not implement the change; the owner is the only human approver, and no second human or code-owner account is assumed.
    An agent never approves its own exception under the owner's account.

Still to agree with the PQL team: one PQL dialect, the versioned validated-query format, the conformance suite, platform builds, and when the Rust crate ships.

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
- **Host facts:** clock, time zone, locale, the tzdata version and similar facts are passed in explicitly, so tests can pin them.
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
R1b validates this set against its case inventory before it is final.

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
  Before R5 this client switches to a pinned release of the crate and deletes the interim parser; deferring that is an explicit scope decision.
- **What to ask the PQL team for:** a C interface or a Rust crate that never lets a C++ exception cross into Rust, builds for every platform this client ships with private C++ symbols, so two copies can live in one process, a versioned validated-query format with a JSON Schema, and the conformance suite in their repository.
- **This client keeps** only the model capability checks on top of the validated query.

### Bindings

![Bindings](images/bindings.png)

The core is plain Rust and doesn't know which language loads it.
Each language uses its standard binding tool, over one shared Rust layer:

- **`sdm-host`**, plain Rust, written once: encodes control messages as versioned JSON, hands data over as Arrow, owns native resource lifetimes, contains panics, and sets the threading rules; it schedules no domain work.
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
- **Platforms and versions:** R0 records today's support: the pure-Python client installs anywhere for Python 3.10 or later, with pandas 1.5 or later and any pyarrow, and the relational extra has its existing wheels.
  R0 also records the proposed native wheel coverage.
  Dropping any platform or version (for example Windows, musllinux, or an old pyarrow such as Databricks Runtime 14.3's pyarrow 8) is an explicit owner decision, never a side effect.

Considered and not chosen: one C interface (`extern "C"`) with generic glue per language, because it means maintaining our own unsafe boundary where PyO3 and `jni-rs` already provide tested ones.
Other binding tools and running the core in a separate process are out of scope unless the owner changes it.

### Performance rules

Today a predict's wall time goes to warehouse round trips and the model call, and most client CPU goes to building the payload cell by cell in Python.
The core keeps the round trips as they are and should cut the CPU, as long as these rules hold:

- Data providers hand the driver's Arrow straight to the core, with no pandas round trip; drivers without Arrow build Arrow columns from rows directly.
- Tables cross zero-copy through the Arrow C Data Interface.
- JSON is only for small control messages, capped (for example at 64 KB); anything row-shaped travels as Arrow.
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
| `sdm-engine` | Modules for time (one calendar, the date parsing rules recorded in R0, `DateOffset`, time zones), random streams, sampling (plan, neighbor sampler, graph store) and labels (kernels and windows) | Merges today's time, random, sampling and label code with one value model; time on a library, not a pandas date-inference clone; Arrow compute where answers allow |
| `sdm-request` | Modules for materialization, session state, decomposition, limits, payload encoding, response decoding and result layout | Keep the logic; drop the CPython sort and `json.dumps` size emulation |
| `sdm-core` | The single Run state machine, public operation dispatch, cancellation, model capability checks and profiles; the interim PQL parser behind one adapter | Rewrite the host API; typed operations only; capability checks from today's `pql` crate |
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
| Dates, times, zones | `chrono` with `chrono-tz` (`arrow-cast` already uses it), checked against the date corpus including zone rules after 2037; the tzdata version is recorded as a host fact; chosen before R1b, because the spike needs it |
| YAML (metric and semantic views) | A maintained YAML 1.2 crate such as `saphyr` or `yaml-rust2` (not the archived `serde_yaml`); behavior differences go through the register |
| JSON | `serde`, `serde_json` |
| Error types | `thiserror`; no `anyhow` in core crates |
| Logging | `tracing`, bridged to Python `logging` and SLF4J |
| Output order | `indexmap` or `BTreeMap` for anything that affects output order; `HashMap` iteration in outputs is banned through clippy's `disallowed_types` |
| Encoding and hashing | `base64`, `regex` and `sha2` where needed |
| Bindings | `pyo3` with `maturin` (abi3 wheels); `jni` (`jni-rs`); the Arrow C Data Interface in `arrow`, `pyarrow` and Arrow Java |
| Dataframe engine | Decided at R2 entry: compare plain Arrow kernels with DataFusion on one label aggregation and one sampling join; correct answers are required, then weigh hand-written code, maintenance, speed, binary size and JNI loading; record the results (Polars is out: it uses its own Arrow implementation, not arrow-rs) |

**Policy:** `Cargo.lock` is committed and the toolchain is pinned; the pinned toolchain is the minimum supported Rust version.
`cargo-deny` checks advisories, licenses, bans and sources in L0, starting from the first attempt's `deny.toml` and toolchain file.
The wheel and JAR ship third-party notices, including for statically linked C++.
Dependency updates are reviewed changes made only at slice gates, with `arrow`, `pyo3` and `jni` updated together.
The PQL crate and its C++ artifact are pinned by version and digest.

On the Python side, pandas and pyarrow stay as host dependencies, and the old engine goes away at the end.

## 4. Delivery: slices, not modules

![Delivery](images/delivery.png)

| Slice | Public path | Languages |
|---|---|---|
| S0 | PQL validation | Python; Java follows |
| S1 | `Schema.from_source` → `materialize` → `predict` on DuckDB, SQLite and Postgres, offline; a minimal non-SQL test provider passes the conformance kit, and an alternate data layer swapped in at the composition root runs the same prediction with no domain edits | Python; Java follows |
| S2 | The same live on Databricks (with a metric view), Snowflake (with a semantic view) and Lakebase; the Databricks job matrix across runtimes; Python and Java agree on the same Databricks data | Python on all three; Java follows on Databricks |
| S3 | In-memory frames (including files read from Amazon S3), ranking, forecasting, link prediction, sessions, tabular requests | Python; Java as the owner adds scope |

Amazon S3 is unrelated to slice S3: `sdm_connectors.read` reading CSV or Parquet from `s3://` into pandas stays in the host, and its frames are covered as S3's in-memory frames.

A path is one public operation with one input mode on one backend (for example `predict` of a regression query on Snowflake), and one routing boundary chooses the old implementation or the core for each path.
Python switches a path on the epic branch only after its slice passes all its gates, live ones included: S0 (PQL validation), S1 and S2 paths together after S2, so no call mixes old and new code, and S3 paths (ranking, forecasting, sessions, on any backend) after S3.
R0 maps every Databricks job in the matrix to its paths, so S2 includes what its jobs need.
Each slice merges into the epic branch at its gate, never into `main`; the epic branch is updated from `main` at each gate, and merging it into `main` is the owner's decision at the end.
The old engine is deleted when the epic merges into `main`; the reference (decision 5) stays available after that.
Python sets the pace: a slice ships for Python when its Python gates pass, and Java follows on the same contract cases without holding it back.

## 5. The quirk register

The register is a machine-readable file in the repository.
Entries follow common sense, with a small bias towards today's Python behavior but without catering to its accidents.
Every entry names the old and new behavior, the affected inputs and contract case IDs, and who approved it: the independent reviewer approves entries, and an entry that refuses an input Python accepted needs the owner's approval, since that is a regression.

| Keep (part of the promise) | Drop (by recorded decision) |
|---|---|
| The random streams of the reference (numpy `RandomState` and `default_rng` where it uses them, `std::mt19937` for neighbor draws), pinned in R0; keeping `default_rng` means porting its generator too | CPython's sort order around NaN scores (NaN ranks last) |
| Label arithmetic as today: float32 Kahan sums in row order | The size limit counted as `json.dumps` bytes (count the bytes sent) |
| `DateOffset` month-end rules | Python object protocols inside the core |
| PQL codes, spans and grammar quirks, as agreed with the PQL team | Errors as Python class paths |
| PQL date-literal rules, written down | Silent nulls or internal errors for types the old code mishandled |
| Today's public exception and warning classes and codes | Time zone rules ignored after 2037 |
| Text-column timestamp detection, under the reference pandas major | |
| Integer widths in payloads | |
| Payload equality after parsing; SQL per database as reviewed golden files | |

Each needs its own decision, not a blanket rule: answers that differ between pandas 2 and 3, SQL text changes, and YAML 1.1 values such as `yes` and `on`.

## 6. Testing

| Layer | What it checks | When |
|---|---|---|
| L0 Boundary guards | First-party domain crates forbid `unsafe` and their production dependency graph has no `pyo3`, `jni` or `arrow-pyarrow` (only the binding crates may use them); the run steps are only data and model steps (no requests for language behavior such as truthiness, sorting or dtype inference); a pure-Rust test host passes the contract subset; the contract's JSON Schema, run steps and the PQL validated-query format don't change without an owner-approved version bump; `cargo-deny` and clippy's disallowed types | Every PR |
| L1 Unit and property tests | Algorithms in each crate, with every past mismatch seed; the shared PQL conformance suite; fuzzing of the PQL, YAML and contract-JSON decoders | Every PR |
| L2 Contract corpus | One shared list of cases, run in each language: a public call and its inputs in; ordered data requests, SQL and typed parameters, payload, result, warnings and error kind, code, details and cause out; row order compared only where the operation defines it; SQL traces compared only for SQL providers, and data-request traces checked against their own contract; pickles written by the reference load in the new client, and new state carries a version; a coverage matrix is generated from a declared scope file for the current slice, with representative nested shapes and one generated refusal case per unsupported type, and an empty cell fails | Every PR (subset, for feedback), before each slice gate (all) |
| L3 Differential generator | Random public-API inputs compared with the reference (decision 5), installed in its own environment, before and after the old code is deleted | Nightly |
| L4 SQL goldens and provider kit | Reviewed SQL per database, executed on DuckDB, SQLite and Postgres; the local providers pass the conformance kit | Every PR |
| L5 Conversion tables | Each language's native-to-Arrow table, both pandas majors; the same tables loaded into DuckDB, SQLite and Postgres reach the core with identical Arrow schemas, values, null masks, order and required metadata from Python's drivers and Java's JDBC, checked against expected values (large integers, decimal scale, timestamp units, nested nulls, list widths, dictionaries) | Every PR |
| L6 Live matrix | The scope inventory's scenarios on Databricks, Snowflake and Lakebase (Databricks' managed Postgres), plus the Databricks job matrix (one-time jobs across Databricks runtimes, compared with the reference's results); old versus new, and Python versus Java on the same data | Weekly and before release |
| L7 Performance | Offline, on recorded runs: query counts equal; CPU per scenario at most today's (aim well below; the first attempt measured about half), no scenario above 1.1 times; boundary overhead per in-memory run within a small fixed bound; the control-message size cap; zero-copy and "no pandas in providers" checks; import time and native library size within set limits; live: push-down median over at least five interleaved runs, caches off, within 10% on Databricks and Snowflake | Offline nightly; live at switch and release gates only |
| L8 Packaging and safety | The real wheel and JAR install and load in clean containers (JDK 17 and 21, every platform in the declared support matrix); exported symbols, private C++ symbols and minimum system library versions checked; a Python process loading both this client and the PQL team's package; `sdm-host` under sanitizers; panic, C++ error, crash and leak tests in each language; HTTP retry, timeout and redaction tests | Focused checks whenever bindings or native dependencies change; the full platform matrix before the epic merges |

- **Scope manifests:** Python and Java have separate scope manifests over the shared cases.
  Python gates require the Python and pure-Rust checks; Java-only checks (such as JDBC conversion equality in L5) block only Java shipping, but a shared-core defect they expose blocks the affected path.
- **Ratchet:** CI compares every case's full observed outcome with the protected last-green baseline, so a changed answer, a new refusal or a removed case fails unless a register entry states that case's exact old and new outcome.
  The baseline updates automatically on merge into the epic branch.
  This replaces choosing "neighboring cases" by hand.
- **Run time:** keep the PR checks fast and move slow ones to nightly; run time is something to optimize, never a reason to skip cases.
- **Test the tests:** before the L2 runners count for acceptance, show they fail on a corrupted expectation, a missing case, a skipped call and a missing native library.
- **Local first:** reproduce a failure from the installed package before spending a remote run on it; a failure that only happens remotely gets one bounded diagnostic run.
- **Retiring old tests:** the first attempt's internal-equivalence tests go once the L2 and L3 failure families they caught are covered, measured by seeding those failures and checking the new layers catch them.
  Algorithm unit tests stay.

## 7. Release gates

| Gate | Passes when |
|---|---|
| Merge a change into the epic branch | L0, L1, L4, L5 and the L2 subset green for Python and the pure-Rust host; the ratchet holds; a type or conversion fix comes with a public reproduction that failed first; focused L8 checks if bindings or native dependencies changed |
| Slice done (merges into the epic branch) | The epic branch updated from `main` and the reference rerun on it; the full L2 for the slice green in Python (Java must be green for its own scope before it ships that slice); L3 on the agreed seed sets with no unexplained divergence; every difference has an approved register entry; for S2, L6 green for its scenarios |
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
| R0 | Create the epic branch; create the reference branch and environment (decision 5) in its own worktree, with its guard and an immutable copy of the original baseline; compare each determinism fix with unpatched Python and record changed outcomes in the register; record today's platform and version support and the proposed wheel coverage; record the accepted text-date formats and inference rules; choose the time library; agree the PQL crate's interface and platform builds with the PQL team; verify branch protection, CI and the owner's approval route; push the first attempt's best candidate as a named branch and run CI once, as a fallback; archive the rest | Reference branch and its guard work; decisions recorded; CI and approval route verified |
| R1a | `sdm-contract`, `sdm-testkit`, L0 guards, L2 runners for Python and the reference, CI tiers; build and clean-install a real abi3 wheel in CI | S0 green in Python; the runners fail on seeded faults; the wheel installs and loads in a clean container |
| R1b | Spike, fixed in scope: DuckDB `from_source` → `materialize` → `predict` → result through the new contract and the data provider interface, in Python and through the pure-Rust test host (which shows the contract needs nothing from Python), against a scripted model endpoint; a fixed case inventory, chosen before R1b, covers the hard seams: keys, decimal precision, nested values, empty and all-null data, and time inference; the text-date rules recorded in R0 checked against the reference | Matches the reference on that inventory (inferred types, keys, time columns, data requests and SQL, payloads, results, warnings and refusals); the owner decides go or no-go (go: R1a and R1b merge into the epic branch and Java's S0 work starts; no-go: repair the first attempt in place, or stop) |
| R2 | The dataframe-engine comparison at entry; S1, moving only the code S1 needs off the Python value model and onto the contract; the provider conformance kit, the minimal non-SQL test provider and the swapped data layer | Slice gate |
| R3 | S2 and the Python switch for warehouse paths, on the epic branch | Switch gate |
| R4 | S3, moving the rest of the code as its paths need it | Slice gate and switch gate |
| R5 | Switch to the PQL crate release; delete the interim parser and the old engine; release candidate; the owner decides whether the epic merges into `main` | Epic merge gate, then release gate |

## 9. Process rules

| Rule | How it's enforced |
|---|---|
| The epic branch holds the only candidate; work on the next slice starts only after this slice's gate passes; independent tasks within a slice may run in parallel | Required CI and review on the epic branch; the status page lists open branches |
| A candidate that passes its checks is pushed before more work builds on it; each slice merges into the epic branch at its gate, never into `main`; the epic branch is updated from `main` at each slice gate | CI on the epic branch; the slice gate is a pull request into the epic branch |
| Acceptance is what CI runs, plus one independent reviewer per change (a fresh agent that did not implement it), the owner's decisions, and one independent release audit of the existing evidence | Required checks on the pull request |
| At most two review rounds per change; one independent reviewer checks code and behavior, not prose; a third round goes to the owner | Recorded on the pull request |
| A phase's acceptance checks are locked before implementation; each change states what and why; a third change in one phase goes to the owner | Locked files change only with the owner's approval |
| Approval by meaning: one decision approves all the files it explains | Register entries and pull request approvals; no per-file hashes |
| Stop rules (below) are lifted only by the owner, never by delegation; on a stop, implementation and live runs pause and the evidence is reported | The owner's approval on the record; an agent never approves its own exception under the owner's account (R0 verifies the approval route, or the owner checks it by hand) |
| One status page: candidate commit, gates passed, open failures, decisions needed | Generated from CI and git, not written by the executor |
| Anything that changes behavior or scope goes to the owner straight away | Register entries and the status page |
| New acceptance infrastructure beyond the agreed test runner needs the owner's approval | Listed on the status page |

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
| The C++ PQL core complicates every build and can crash across the boundary | Ask the PQL team for a C interface that never lets exceptions escape, static libraries per platform built for manylinux with private symbols, matching crate and PyPI releases, no aborts or console output, a recursion limit, and the conformance suite in their CI; exercise their artifact as soon as it exists; load and crash tests in L8 |
| A native library builds but doesn't load on some platform | R1a installs a real wheel; L8 installs the real artifacts in clean containers whenever native code changes, and for every platform before the epic merges |
| Native wheels silently drop a platform or version Python supports today | R0 records today's support; any drop is an explicit owner decision |
| The Java consumer's environment disallows the needed runtime flags | Confirm `--enable-native-access` and Arrow Java's flag with the consumer before Java packaging |
| A team swapping the data layer for Pyramid finds the interface insufficient | The interface is documented, a minimal non-SQL provider and a swapped data layer prove it in S1, and the conformance kit defines what a replacement must do |
| Host conversion tables drift between Python and Java | L5 checks both languages convert the same table to the same Arrow types |
| The new client is slower than today's | The performance rules in section 3, checked offline by L7 |
| Process grows faster than product again | Rules enforced by CI and the owner, not by the executor; new acceptance infrastructure needs approval |
