# Phase 1: the 1a harness and the 1b spike

**Branch:** `rust-core-phase-1`, off `rust-core`, for both halves; it merges locally into `rust-core` only after the owner's go at the end of 1b.
**Topic branches:** `rust-core-phase-1-<task>`, so CI runs on them; at most one per task.
**Tracker:** [tracker.md](tracker.md); **decisions:** [decisions.md](decisions.md).

## Changes since lock

The design was locked at T1.1 (ec37923).
Every later decision that changes it is listed here and written into the section named; the decision row is the authority, and this file is updated in the same change as the row.

| Row | Change | Section |
|---|---|---|
| E1 | `deny.toml` ignores two pyo3 0.28 advisories; `sdm-py`'s clippy bans the affected calls | Dependencies |
| E2 | Every byte buffer and Arrow table crosses as an attachment named by `AttachmentId`, with numbering and release rules per direction | Contract |
| E3 | Zone names resolve in any case and are recorded under their IANA spelling | Core, host and binding |
| E4 | The encoder never writes what the decoder refuses; one limit rule for types and scalars; negative decimal scale kept | Contract |
| E5 | After `cancel`, calls return kind `cancelled`; id overflow ends the run; `facts.now` must be UTC | Core, host and binding |
| E6 | Instants past `jiff::Timestamp::MAX` use civil arithmetic and are never refused for that reason | Time |
| E7 | Contract version 2: PQL spans as rows and columns | Contract |
| E8 | The validated-query format is versioned apart from the contract; provisional until the PQL asks are answered | Interim parser |
| E9 | `escapes` joins `pyint` and `float_repr` on the emulation allowlist (three modules) | Interim parser |
| E10 | An interim date reader in T1.6a; D8 and D4's warning in T1.6b | Interim parser |
| E11 | Where each capability check runs | Interim parser |
| E12 | The PQL corpus and journey queries live with `sdm-core`'s adapter tests | Interim parser |
| E13 | `HostRun`'s calls, attachment checks, admission refusals ending the run, `VALUE_OUT_OF_RANGE` | Core, host and binding |
| E14 | The cap refused wide starts; superseded by contract v3 (H5), register HOST-1 rejected by the owner | Contract |
| Re-review R1 | Input-sized fields and error overflow as attachments; `ErrorCode` enum with `retryable`; seed of any size; description-free hash (H5 to H7) | Contract |
| Re-review R1 | Runtime-named CI cells; `python-old` demoted to smoke (H1, H10) | Runners; CI |
| Re-review R1 | The gate runs the checkpoint's harness against protected paths; "unavailable" cannot pass; one gate run per reviewed work package (H2, H3) | Reference observations and the gate |
| Re-review R1 | SQL gated on count, typed parameters and batches; text diagnostic (H9) | Observations and comparator |
| Re-review R1 | Logging through `tracing` and `pyo3-log`; threading model; diagnostics carry no input values; boundary schema match; lossless admission additions (H8, H16, H17) | Core, host and binding |
| Re-review R1 | A dependency-safe data port, `Neighbors` and `Lookup` requests, zones passed into the core (H14, H15, F1) | Data access |
| Re-review R1 | One internal `Instant`/`Span` type in nanoseconds (F3) | Time |
| Re-review R1 | New tasks T1.7b, T1.7c, T1.10c; T1.11 split into a to e; owner check-ins; stop-rule counters; exits corrected (H11, H12, H18 to H24) | Tasks; Owner check-ins; Gates |
| Re-review R1 | Locked checks 1 and 4 amended as the owner approved (H13) | Locked acceptance checks |

## Goal

Phase 1a builds the contract, the first core crates, the binding and wheel, the runners, the comparator, the ratchet and the CI checks, and proves them on S0 (PQL validation).
Phase 1b runs one hard journey end to end on the new architecture: DuckDB `Schema.from_source` to `materialize` to `predict` to the result, through the contract and the data provider interface, in Python and on the pure-Rust test host, over the 40 fixed cases of [phase-1b-inventory.yaml](../phase-0/phase-1b-inventory.yaml).
Work starts thin: a minimal contract and wheel, one S0 case through Python and Rust, then the first DuckDB regression journey, and only then the general tooling; the early journey reduces risk, and does not merge early or relax the 1a exit (decision C5).
The phase ends with an evidence pack and the owner's go or no-go: go merges 1a and 1b into `rust-core` and starts Java's S0 work; no-go means repairing the first attempt in place, or stopping.

## Scope

### Phase 1a

- The Rust workspace, toolchain, lints, `cargo-deny` and the L0 guards.
- Crates: `sdm-contract`, `sdm-core` (a skeleton with the single Run state machine), `sdm-pql-interim`, `sdm-host`, `sdm-py` and `sdm-testkit` (decision A1).
- The client wheel built by maturin with the native module `sdm_client._native`, installed in clean containers; the private routing seam, and S0 through the core route.
- The case, observation and inventory formats, the S0 corpus, the runners (three Python modes and the pure-Rust host), the comparator, recorded reference observations, the ratchet with register format v2, the S0 coverage gate and the seeded-fault suite.
- The CI fast tier, the local gate script and the status page.

### Phase 1b

- Crates: `sdm-data` (the start of the data layer, not throwaway), `sdm-schema`, `sdm-engine`, `sdm-wire` and `sdm-request`, each holding only what the 40 cases need.
- The DuckDB provider in the Python host and in the pure-Rust host, with key sets as Arrow attachments.
- `Schema.from_source`, `materialize` and `predict` core routes, and the scripted model endpoint in both languages.
- DuckDB SQL goldens (the L4 part) and the Arrow-to-pandas presentation on pandas 2 and 3 (the L5 part), only for what the inventory needs (decision A5).
- The DT-3 and ORD-1 findings, register entries per changed case, the type-set findings, the held-out run (H11) and the evidence pack.

### Explicitly out

- Java: `sdm-jni`, a JAR, JDBC conversion tables, Java S0 and any Java runner; the pure-Rust host is the second language in Phase 1; Java's logging bridge (SLF4J) is decided in the Java phase.
- Backends other than DuckDB in 1b (frames and S3 reads included), live work (L6, the Databricks job matrix, any remote run), and operations outside the inventory (ranking, forecasting, link prediction, `fit`, sessions, tabular requests, metric and semantic views, YAML).
- Phase 2 items: the provider conformance kit and paged results, the non-SQL test provider, the swapped data layer, the dataframe-engine comparison, the remaining S1 cells, the full L4 and L5 matrices, reference `Request` pickles (H23), capability-failure and retryability conformance, PD-1's NaN ranking, and `sdm-host` under ASan and LSan.
- L3 differential generation, full L7 measurement, the full L8 platform matrix, wheels other than manylinux2014 x86_64, and any claim about the PQL C++ build floor.
- Integrating the PQL team's crate, sending the PQL asks, deleting the interim parser, switching any public path to the core, deleting the old engine, and retiring old tests (seeding their failure families is Phase 5).
- Random streams the journey does not reach: RS-2 to RS-5 (RS-3's state across calls is a Phase 4 contract item), and the Postgres, Snowflake and Databricks sampling SQL.
- Frame text inference (DT-6, PK-7) and the in-memory sampler's semantics, which are Phase 4 (S3).
- Growing the 40-case inventory; a missing case goes to the owner as a scope decision.
- Updating the reference from `main` mid-phase, except at the go merge if the owner asks.

## Architecture choices

### Layout

```text
rust/
  Cargo.toml, Cargo.lock, rust-toolchain.toml, deny.toml, clippy.toml
  testdata/tzdata-2026c/  vendored TZif directory for zone tests
  crates/
    sdm-contract/      types only, including the data-layer port; schema/ holds the generated JSON Schema and the locks
    sdm-core/          the Run state machine, dispatch, capability checks, the PQL adapter
    sdm-pql-interim/   the archive's parser, reachable only through sdm-core's adapter
    sdm-host/          control codecs, the 64 KB cap, Arrow admission, panic containment, run ownership, zones
    sdm-testkit/       dev-only: case loader, pure-Rust host, duckdb-rs, scripted endpoint
    sdm-data/          1b: the data layer: SQL rendering, provider strategy, the DuckDB dialect
    sdm-schema/        1b: canonical schema and inference over Arrow
    sdm-engine/        1b: time, random streams, sampling, labels
    sdm-wire/          1b: generated wire types
    sdm-request/       1b: materialization, payload, response, result layout
  bindings/sdm-py/     PyO3 cdylib `_native` over sdm-host
contract/              schemas; bases.yaml; compare-policy.yaml; availability.yaml; protected-paths.txt; reference-env.txt; cases/; reference/; baselines/; runners/run_python.py
scripts/rust_core/     compare.py, ratchet.py, register.py, coverage.py, l0_*.py, status.py, gate.sh, selftest.py, emulation-allowlist.txt
packages/structured-data-models-client/src/sdm_client/
  _native.pyi          typed stub, so mypy needs no binary
  _routing.py          the routing seam
  _rust/               the Python host: run loop, providers, facts, error table, schema bridge
```

- `contract/` sits at the root because it belongs to neither language.
- The Python host package is `_rust`, not `_core`, so it cannot be confused with a name the reference guard probes.
- `dependencies.md` in this folder records one row per dependency and per hand-written substitution: use a library, port a defined algorithm, or keep in the host, with the measured reason.
- Bridge code (`sdm-pql-interim`, `_routing.py`, `SDM_CORE_ROUTES`, the S0 shadow check) carries a header naming the phase that deletes it, checked by `l0_deps.py` (F7).

### Workspace rules

- Edition 2024, `rust-version = "1.97"`, toolchain pinned to 1.97.0 as the minimum supported Rust version, `Cargo.lock` committed.
- `panic = "unwind"` stated explicitly, so panics can be contained.
- `unsafe_code = "forbid"` in workspace lints and at every domain crate root; only `sdm-py`, `sdm-testkit` and later `sdm-jni` relax it, locally, for Arrow FFI.
- Clippy runs with `-D warnings`, each `#[allow]` carries a one-line reason, and every `.rs` and `.toml` file carries the SPDX header the hygiene check enforces.
- No `#[ignore]`, `skip`, `xfail` or `continue-on-error` under `rust/`, `scripts/rust_core/` or the Rust and contract CI jobs; a CI step fails on one (T1.10c).

### Dependencies

- Arrow 58 across the workspace: `duckdb =1.10506.0` (DuckDB 1.5.6, the reference's version) uses Arrow 58, and `arrow-pyarrow` 58 needs `pyo3` 0.28; the archive's Arrow 60 and `pyo3` 0.29 are not used, and the three are bumped together, only at slice gates (decision B-D2).
- `cargo-deny` bans duplicate versions of `arrow-*` crates and `pyo3` only; two pyo3 0.28 advisories are ignored with `sdm-py`'s clippy banning the affected calls until the next bump (E1).
- `jiff` is the one calendar and zone library, with the host's ordered zone lookup roots passed in and a bundled fallback; `chrono` stays only as Arrow's transitive dependency and `chrono-tz` is banned (decision A7).
- `tracing` for logging in Rust, bridged to Python `logging` by `pyo3-log` in `sdm-py` (H8).
- `serde`, `serde_json`, `schemars` 1.x, `thiserror`, `indexmap`, `stacker` (the interim parser and `sdm-core`'s adapter); `duckdb` and `anyhow` only in `sdm-testkit`.
- Arrow's own kernels (`arrow-cast` and the rest) are used where their tested semantics fit; a hand-written cast or kernel needs a row in `dependencies.md` saying why.

### Contract

- `ColumnType` is the closed type set of plan section 3; `ColumnType::from_arrow` is the only way in, and anything else is refused with `kind: unsupported, code: UNSUPPORTED_TYPE` naming the column and the Arrow type.
- The scope manifests' cell vocabulary is generated from that enum.
- Operations are a versioned, serde-tagged enum; Phase 1 needs `validate_query`, `describe_source`, `materialize` and `predict`.
- `Step` has exactly three variants: `Fetch` (a batch of typed data requests, each with the Arrow schema it expects back or a discovery marker), `CallModel` (the body an attachment) and `Done`; no step asks the host for language behavior.
- Every byte buffer and Arrow table that crosses (fetch results, key sets, model bodies, request state, results) is an attachment named by `AttachmentId`; the sender numbers what it sends, core-to-host ids are unique within a run, and host-to-core ids are scoped to one message (E2).
- **The 64 KB control cap bounds the envelope only.** Any field whose size grows with user input (query text, `QuerySchema`, `SourceDescription` in a `start` or a `Done`, later view definitions) travels as a bytes attachment holding UTF-8 JSON of the same serde type; no inline field grows with column, table or row count (contract v3, H5; supersedes HOST-1).
- Errors are `{kind, code, message, fields, cause, retryable}` with closed kinds; `code` is a `#[non_exhaustive]` `ErrorCode` enum with SCREAMING_SNAKE names (PQL diagnostic codes stay strings inside `PqlDiagnostic`); `retryable` defaults to false and is carried from provider and transport errors; each host maps every `(kind, code)` through one reviewed table, tested exhaustively, never reading message text (H6).
- An error whose encoding passes the cap travels whole: the control message carries its kind and code and names an attachment holding the full error with its fields and cause; nothing is dropped (H7).
- Warnings travel on their own channel, in order; no contract string names a Python or Java class.
- Every scalar in a control message is a tagged object with its value as a JSON string (integers per width, `d128` and `d256` with precision and scale, floats with shortest round-trip text and non-finite values, temporal counts with unit and zone), so nothing passes through a double; the encoder never writes what the decoder refuses (E4).
- `materialize`'s seed is a non-negative integer of any size, as decimal text; per-backend ranges (RS-7) are applied in the core, not by the decoder (H7).
- `HostFacts` (the clock, a `ts` in UTC; the local zone; the zone lookup roots in order with their versions; the platform padding rule) is part of every operation; locale is not a host fact, and the core reads none.
- State such as `RequestState` is typed and carries its version; an unknown version is refused by name; runs and Arrow buffers have defined owners and release rules.
- The JSON Schema is generated by `schemars`, committed, and locked by hash in `contract.lock`; existing entries are never rewritten; entries added from contract v3 hash a description-free projection, so a doc comment needs no bump (H7).
- The contract version is independent of the package version; within a phase each bump is a decisions row (E7 for version 2, H5 for version 3), the lock compares against the merge base with `rust-core`, and a phase lands as one bump (H4).
- Version review covers behavior as well as shape: a change to validation, units, ownership, defaults or error meaning bumps the version even when the schema is unchanged; host and core ship together and refuse any other version before execution (H25).

### Key sets

- A key set crosses the core/host boundary as an Arrow attachment referenced by ID from a typed data request; control JSON carries only its descriptor, never the keys (decision Q1).
- The DuckDB provider owns the physical binding (for example, registering the attachment as a relation), and both hosts implement the same semantics.
- Tests prove a key set whose serialized contents exceed 64 KB without raising the control cap, and exact binding of uint64 above int64 max, `decimal128(38, 0)` and scaled decimal keys.
- The decimal tests use values that collide as float64 (for example `0.100000000000000001` and `0.100000000000000002` at scale 18): distinct keys stay distinct through binding, join and result association, and a forced decimal-to-float conversion in that path fails the test (H22).
- Exact identity is not exact arithmetic: where the reference converts a numeric target or sums labels in float32 (PK-2), that conversion is intentional and stated, not a key defect.

### Data access

- `sdm-contract::data` holds the request vocabulary, the capabilities and the port: `trait DataLayer { fn plan(&self, request: &DataRequest, capabilities: &ProviderCapabilities) -> Result<FetchPlan, CoreError> }` (H14).
- `sdm-data` implements it; SQL dialects are internal to `sdm-data`, and the wire names a source's backend only for `sdm-data` to interpret.
- `CompositionRoot` holds one `Arc<dyn DataLayer>`, built by `sdm_host::default_root()`; swapping the data layer replaces that one value (proved in Phase 2).
- Domain crates do no I/O and hold no `Asker`; only `sdm-core`'s operation bodies await, and `Asker::fetch` takes a `DataRequest`.
- Dependencies: `sdm-core` → `sdm-request` → `sdm-engine` → `sdm-schema` → `sdm-contract`, and `sdm-core` → `sdm-data` → `sdm-contract`; no crate below `sdm-core` depends on it, checked by `l0_deps.py` from T1.10c.
- A fetch result is complete: the host drains every driver batch of a request into one Arrow attachment; multiple batches and an empty result keeping its schema are tested on both hosts; paging is Phase 2.
- The Phase 1 vocabulary is `Describe`, `Scan`, `Preview`, `CountRows`, `KeyCounts`, `TimeBounds` and `SampleEntities`, plus, from T1.11a:
  - `Neighbors { table, key, columns, seeds, time, per_seed, order }`: `seeds` a key-set attachment of key, batch and optional anchor; `time` an optional `TimeRule { column, inclusive, nulls }`; `per_seed` an optional limit; `order` an explicit `TieBreak` (F1, H15);
  - `Lookup`, rows by key, returning the batch and the match count.
- A provider that does not declare `Neighbors` gets `Scan` with a key filter, and `sdm-engine` applies the window, limit and order in memory.
- Every field of a request is honored or the request is refused with a typed error; no flag is ignored.
- A declared capability that fails is an error, never silently replaced.
- `sdm_core::start` takes `Arc<dyn Zones>` (the trait in `sdm-engine::time`), implemented by `sdm-host`'s resolver and cached per lookup roots and versions; a future Java host passes the bundled database with its version recorded.

### Time

- Inside the core every instant is `Instant(i128)`, nanoseconds since the UTC epoch, and every span is `Span(i128)`, nanoseconds (F3).
- Columns stay Arrow arrays with their own unit, zero-copy; `Instant` and `Span` exist for scalars and transiently inside per-row kernels.
- They are built only from an Arrow temporal column with its unit or a contract `ts` or `dur` scalar, and leave only by checked conversion to a named unit, which returns `VALUE_OUT_OF_RANGE`, never saturating or panicking.
- No raw `i64` meaning time crosses a function boundary; `jiff::Zoned` exists only inside a zone-conversion call; no unit-generic type family.
- At or past `jiff::Timestamp::MAX` (9999-12-30T22:00Z), conversion uses civil arithmetic and offsets; an instant is never refused because `jiff` cannot hold it (E6).
- T1.25 adds vectors at both ends of the int64 microsecond range and at 9999-12-31 in every vendored zone.

### Core, host and binding

- `sdm-core` has one `Run` type: operation bodies are `async fn`s over an asker handle, polled once per step, with no executor and no threads.
- States are `Ready`, `AwaitingFetch`, `AwaitingModel`, `Finished`, `Cancelled` and `Poisoned`; an illegal call returns a `protocol` error and never panics; after `cancel`, every call returns kind `cancelled`, `RUN_CANCELLED`; id overflow ends the run with `INTERNAL_ERROR` (E5).
- `sdm-host` is plain Rust and the only API a binding calls: `HostRun::start`, `next`, `reply`, `cancel`, `take_attachment` and `take_warnings` take and give control messages as bytes with the attachments each names (E13).
- Each call is contained by `catch_unwind` (a panic poisons the run and returns `INTERNAL_ERROR` with `fields.panic = true`), and a concurrent call on the run gets `RUN_BUSY`.
- A host message must attach exactly the ids it names, as the kind it names, or it is refused with `INVALID_MESSAGE` before the run sees it.
- **Schema match (H16):** each batch must match its attachment's declared schema (column count, order, names, types and required metadata) before admission; each admitted fetch result must match the outstanding request's expected schema, stored with the request, before domain code sees it; a discovery request establishes the schema and later batches must match it; a mismatch is `PROVIDER_SCHEMA_MISMATCH`, never a dropped, relabelled or filled column.
- **Admission (B-D6, H16)** lives once in `sdm-host` and is lossless: `large_*` to small forms, `timestamp`, `duration` and `time32` at `s` or `ms` to `us`, `date64` to `date32` when every value is a whole day, `utf8_view` and `binary_view` to `utf8` and `binary`, `float16` to `float32`, `decimal32` and `decimal64` to `decimal128`, `large_list` and `fixed_size_list` to `list`, `dictionary<K, large_utf8>` to `dictionary<K, utf8>`; a batch whose offsets would pass int32 is split, not refused; a value the admitted type cannot hold is `VALUE_OUT_OF_RANGE` naming the column; a refusal ends the run (E13).
- Native-to-Arrow tables stay in each host.
- `sdm-host` resolves each zone, in any case, through the host's lookup roots in order, then the bundled copy, and records the source and version under the zone's IANA spelling (E3).
- **Threading:** the core starts no threads and holds no global state; a run is used from one thread at a time (`RUN_BUSY` otherwise); runs are independent; `sdm-py` releases the GIL for every core call and holds it only while touching Python objects; importing `_native` starts no thread and installs no logger, signal handler or global pool.
- **Logging (H8):** `tracing` in Rust; `sdm-py` bridges it to Python `logging` with `pyo3-log`; the Python host resets `pyo3-log`'s level cache at each run start, so a logging configuration change takes effect; a GIL-deadlock regression test guards the bridge.
- **Diagnostics (H17):** no input value or credential reaches a `CoreError`'s message, fields or cause: a contained panic keeps its flag and location, not its payload text; a codec error names the position and category, not the offending text; a provider error keeps SQLSTATE and vendor codes as fields and leaves the driver's message text out, while the Python host may chain the driver's own exception as `__cause__`, as the client does today.
- `sdm-py` is a thin PyO3 module over `sdm-host`: it passes Arrow through the PyCapsule interface from pyarrow 14 and `_export_to_c` below it, raises one `_native.CoreError`, and passes model bodies as bytes attachments; it makes no semantic callbacks into Python, while Arrow export and release callbacks are allowed and counted.
- The pure-Rust host calls `HostRun`, so its fetched data passes the same admission and schema match as Python's.

### Packaging

- The client's build backend becomes maturin on the epic branch (decision B-D3); `main` is untouched.
- `sdm_client._native` (decision A3) ships in the `structured-data-models-client` wheel, tagged `cp310-abi3`, built as `manylinux2014` x86_64, with third-party notices for every Rust dependency linked in.
- The wheel build records its effective Rust target flags and fails if `target-cpu` is set (F7).
- `pyarrow>=10.0.1` becomes a base dependency; `pandas>=1.5` stays.
- A version guard holds the Cargo version of `sdm-py` equal to the other four distributions; CI's `build` job builds four wheels as today and takes the client wheel from `native-wheel`, so "five wheels, one version" still holds.
- Importing `sdm_client` never imports `_native`; only `_rust/` does, lazily, on a core route; a test holds this.
- A missing `_native` or an unknown forced route on a core route raises an error naming the cause, never falls back, and counts as a runner failure.

### Routing seam

- `sdm_client/_routing.py` maps the `scope-python.yaml` path IDs to `old` or `core`; every default is `old`, and a test fails if one changes before the Phase 3 switch.
- `SDM_CORE_ROUTES` forces paths onto the core for tests; it is read at every call, is private, and is deleted in Phase 5 (decision B-D7).
- S0 is the public call that validates PQL today, `materialize()`; under the core route `_resolve` calls `validate_query` and maps a refusal to `StructuredDataError(code='INVALID_REQUEST')`; `UNSUPPORTED_REQUEST` from `validate_query` (E10) never maps to a PQL refusal.
- "S0 green in Python" compares validity and `StructuredDataError.code` with the reference, not spans or message text; parser codes and spans are checked in L1 by the 417-query corpus (decision A2).
- An accepted S0 query finishes through the old materializer, and is not evidence for core materialization.
- While S1 is not switched, an accepted query is also checked by the old validator, and a disagreement is `INTERNAL_ERROR`; this shadow check leaves with the old code.
- On the three core paths of the journey, an import hook raises if `sdm_relational_engine` is imported after route entry, and `_rust/` imports nothing from it.

### Interim parser

- `sdm-pql-interim` holds the archive's lexer, parser, visitor, AST, profiles, codes and the seven validators; the 417-query corpus (193 valid, 224 invalid) and the journey's queries live with `sdm-core`'s adapter tests and run through the adapter (E12); the 29 quirks and custom-profile tests come with T1.6b.
- Validity-affecting emulation stays, inside the crate, on the allowlist: `pyint`'s 4,300-digit limit, `float_repr` for AST constant text, and `escapes` for string literals (A4, E9); `float_repr` writes only message text in the ported path and is a candidate to remove, not a reason to add emulation.
- `pyrepr`'s other helpers, `printable.rs`, `py_repr` and `py_str` are not ported, since message text is not promised; a neutral `Literal` replaces the archive's Python value type, and integer literals keep their digits exactly.
- The 1,695-line `dates.rs` is replaced by a reader written from rules D1 to D9 over `jiff`, taking the clock and padding rule from host facts: T1.6a reads the plain shapes and answers D8 with `UNSUPPORTED_REQUEST` (E10); T1.6b completes D8 and D4's warning, and takes the current year from `facts.now` in `facts.local_zone`, as the reference's `dateutil` does.
- Capability checks: the pre-resolution temporal-target check runs on the parsed query in `sdm_core::profiles`; the two filter flags are consulted during resolution, as the reference does; the rest run on the validated query (E11).
- The validated-query format is versioned and locked apart from the contract (E8), and is provisional until the PQL asks are answered.
- Today's ANTLR grammar is the CI reference through the reference observations and the corpus recorded on it; Phase 1 adds no separate grammar check.
- Only `sdm-core` has a production dependency on the crate, and only its adapter uses it; the PQL team's crate later goes behind the same function.
- `scripts/rust_core/emulation-allowlist.txt` holds exactly those three modules; emulation is banned everywhere else, each addition needs an owner-approved decisions row, and the list expires with the crate (H21).

### Pure-Rust test host

- `sdm-testkit` holds the case loader, the pure-Rust host, the scripted endpoint and the runner binary `sdm-contract-run`; it is never a dependency of a shipped crate.
- It builds each case's data with `duckdb-rs`, checks `SELECT version()` against the recorded one, answers `Fetch` with the driver's Arrow and `CallModel` with the scripted endpoint, through `HostRun`, with no Python in the process; `REPEATABLE` sampling is compared only within one DuckDB version.

### Runners

- One Python runner, `contract/runners/run_python.py`, run with `python -I`, in three modes:
  - `python-ref`: the reference environment, the oracle;
  - `python-core`: `SDM_CORE_ROUTES` set for the case's paths;
  - `python-old`: defaults on the phase branch; a smoke runner that must complete every case without a crash, and the comparison base in the runtime cells other than the reference's (H10). It lacks the determinism fixes, so fields they affect (DF-1 to DF-5) are compared there by the named invariants of `random-streams.md`.
- `python-ref` loads the reference guard by path, calls `require_isolation()` before the first case and after the last, and records the reference commit, the environment hash and the guard result.
- The scripted endpoint is one written rule served over localhost HTTP, reached by Python runners through the client's real transport, implemented in Python and in Rust and held together by a shared vector file.
- **Driver capture (Q2):** a test-only capture in `run_python.py` wraps the connection the runner hands the source and records each statement, its typed parameters, the batches consumed and provider-created temporary-relation bindings, derived cursors included, changing no value, order or execution count, with no change to reference code and no rerun of logged queries.
  All three Python modes run under it; "setup" means statements before the connection reaches `Schema.from_source`; the proxy is checked first against the reference's dispatch on class module and `isinstance(cursor, type(connection))`.
  `rust-host` takes the same fields from the step trace.

#### Runtime cells (H1)

CI cells are named by runtime, from [support-matrix.md](../phase-0/support-matrix.md); every observation's identity records the platform and cell.

| Cell | Python | pandas | pyarrow | NumPy | Stands for | Compared |
|---|---|---|---|---|---|---|
| `ref-pandas3` | 3.12 | 3.0.6 | 25.0.1 | 2.5.3 | The reference environment (frozen as `contract/reference-env.txt`); DBR 16.4 LTS and serverless env 3 after install | `python-core` against the reference on all fields |
| `dbr17-pandas2` | 3.12 | 2.2.3 | 19.0.1 | 2.1.3 | DBR 17.3 LTS and serverless env 4 as shipped | `python-core` equal to `python-old` on `host` and `core` fields |
| `floor-pandas1.5` | 3.10 | 1.5.3 | 10.0.1 | below 2 | The A8 floor; DBR 14.3 LTS and serverless env 1 for `kumo-tabular` installs | `python-core` equal to `python-old`; S0 only, since the DuckDB extra needs pandas 2.0 |

- A difference that only one runtime shows is a register entry naming the cell, never an exclusion.
- A `host` field that differs between platforms in the same cell is an identity failure for the reviewer to classify, then a register entry.
- The newest-dependency wheel smoke (Python 3.14, newest pyarrow) is a separately identified smoke lane, not a semantic cell.

### Observations and comparator

- An observation (`sdm-observation/1`) is JSON in which every scalar uses the contract's tagged-string encoding; payloads keep their raw bytes and a parsed form with numbers as decimal text; result tables are Arrow IPC files referenced by SHA-256.
- Every compared field names its acquisition point: public output, HTTP request or response, driver-boundary capture, or core `Step`.
- `contract/availability.yaml` fixes, per runner and field, available or unavailable; a field is never fabricated or treated as equal; a field a locked check requires that is unavailable fails that check, and downgrading a field to unavailable is a relaxation (H2, H13).
- `core` fields: Arrow types, stypes, keys, time column, data requests, statements and the batches each returns, payloads, the pre-presentation Arrow result, error kind, code, fields and cause chain, and warnings by code in order.
- `host` fields: pandas dtypes, Python error and warning classes, `StructuredDataError.code` and the `to_pandas()` result, through which the reference's result is compared, never as an Arrow table rebuilt from pandas.
- `rust-host` and `python-core` are compared on `core` fields; Python runners are compared on `host` fields and on the `core` fields that one projection in `run_python.py` obtains, as the runtime cells say.
- An `identity` envelope (runner, cell, platform, commit, `_native` SHA-256, reference commit, corpus and input digests, contract version, DuckDB version, per-zone database source and version) is validated separately and excluded from digests.
- `scripts/rust_core/compare.py` prints the first difference per field with its path, under the rules in `contract/compare-policy.yaml`.
- Exact fields keep null apart from NaN, `-0.0` from `0.0`, decimal precision and scale, temporal unit and zone, nested validity, field metadata and duplicate rows; message text never gates, and the inventory's message fragments are reported for information.
- Payloads are compared parsed with lossless numbers, excluding only the approved UUID and nondeterministic field paths; results by Arrow schema and values, with row order only where the operation defines it; raw payload and IPC hashes verify artifact integrity only.
- **SQL (H9):** against the reference, the comparator gates on the statement count (setup excluded), the typed parameters and the batches each statement returns; the reference's raw SQL text and bindings are diagnostic evidence only; the core's SQL text is pinned by the DuckDB goldens (T1.30), and a golden update never replaces a register entry.
- Nondeterministic draws follow `random-streams.md`: membership, cardinality and named invariants instead of row equality, with exclusions only for named field paths.
- No Phase 1 case overrides the default seed, so its draws are deterministic; no fetch replay is built; common-input comparison is a Phase 2 prerequisite for PostgreSQL and a Phase 3 one for live warehouses, and until then those downstream comparisons are listed as unproven, not equal by invariant (H24).

### Reference observations and the gate

- Reference results are committed under `contract/reference/<reference-commit>.jsonl`, with the reference's frozen environment in `contract/reference-env.txt`, so CI compares without running the reference; `gate.sh` regenerates them locally and compares the semantic projection, not bytes or hashes, with identity checked separately (decision B-D8).
- `contract/baselines/<cell>/<runner>.jsonl` holds each case's projected observation and its digest.
- **The checkpoint** is the merge base with `rust-core` until the first green promotion, then the tag `rust-core-phase-1-green`; each gate run resolves it once to an immutable SHA (C1).
- **The checkpoint's harness judges (H2):** each gate run, in CI or `gate.sh`, checks the checkpoint out into a separate worktree and runs that worktree's `compare.py`, `ratchet.py`, L0 scripts, runner projection and expectations against the candidate's observations; the candidate's own harness result is reported only.
- **Protected paths:** `contract/protected-paths.txt`, read at the checkpoint, lists the cases, reference observations, `reference-env.txt`, register, comparison policy, availability, L1 expectation files (the PQL corpus and journey queries, zone, stream and hash vectors, vendored tzdata, layout vectors, goldens), the emulation allowlist, `deny.toml`, `clippy.toml`, `ci.yml` and the harness scripts; a candidate change to any of them fails unless a register entry or decisions row in the same change names the path.
- Stream and hash vectors are never re-recorded to pass a dependency bump (F4).
- Against the checkpoint, a changed digest, a new refusal, a removed case (moving a case to diagnostic counts as removal), or a changed case input fails unless an approved entry in the change names that case, cell, runner, and exact old and new outcome.
- A repair is judged against the effective expected outcome (the reference plus approved deviations); returning to raw reference behavior cannot revoke an approved deviation or regress a protected success without its own approved transition.
- **Deviations are approved when the changed case lands (H19):** each semantic or SQL deviation is recorded and independently approved in the change that first admits that case to acceptance; a refusal of an input the reference accepts waits for the owner; T1.33 reconciles completeness, and is not the first approval point.
- Adding or tightening a comparison rule passes; relaxing one, adding an exclusion or downgrading availability needs an owner-confirmed decisions row.
- **Approval:** the reviewer approves entries (`approved_by: reviewer`, never by the implementer); `approved_by: owner` records only the owner's own written answer; each landing names its review run in a `Reviewed-by:` trailer; `gate.sh` lists every `approved_by: owner` added since the last owner merge, and the owner's local merge is the only hard approval; there is no signing machinery.
- **One gate run per work package (H3):** a change is one reviewed work package, not every intermediate commit; its activated gate runs once on the package, and the reviewer inspects the same evidence plus targeted adversarial checks.
- `ratchet.py accept` writes the baselines in their own commit at a reviewed green landing; that commit is validated against the current checkpoint, its CI and local gate results are recorded, and only then does `gate.sh` move the tag to it, serialized, only on `rust-core-phase-1`, failing if the checkpoint moved since the run started.
- The first promotion (T1.14) comes after the four test-the-tests faults fail as they should, and baselines only cases equal to their reference projection on every compared runner; after it a missing baseline is an error, and at the go merge the checkpoint's baselines become `rust-core`'s.

### Register v2

- `register.yaml` moves to `sdm-register/2` (decision A6): each entry may carry `case_outcomes`, a list of `{case, cell, runner, old, new, summary}`, and `decision_ref` becomes `{file, row}`.
- `register.py validate` checks the schema, unique and never reused IDs, case IDs that resolve to corpus cases or an allowed legacy form, and the status rules.
- An entry counts only when approved by someone other than its implementer; an entry that refuses an input the reference accepts needs `approved_by: owner`.
- Changed contents under an approved ID count as a new change and need approval again; the ratchet compares entry contents with the checkpoint; no entry can accept a panic.

### Coverage generator

- `scripts/rust_core/coverage.py` implements the rules of [scope-format.md](../phase-0/scope-format.md), checking both manifests structurally but gating only the active language and slice: S0 requires its validation paths and no cells, and Java runtime coverage starts with Java.
- Rule 6 (with all 40 cases): a cell claim names the observed boundary and stage (received Arrow type, admitted type, declared role, inferred role if reached), and a refusal covers its rejected input without claiming a later stage.
- The manifests' `cases` lists are generated (`--write`) and checked (`--check`), never edited by hand.
- `--gate S0` runs in Phase 1; `--gate S1`, which also enforces rule 7, is a Phase 2 obligation.

### Harness scope

- The agreed acceptance capabilities are the runners, the comparator, the ratchet, the coverage checks and the guards and gate around them; a new capability needs the owner's approval (decision C6), and the status page lists every acceptance capability added since T1.1.
- File splits, renames and tests implementing those capabilities need normal review only.
- The status page reports size trends: non-test, non-generated lines of harness and product per crate (`tokei`), with the interim parser reported separately; unexplained growth triggers a review against the agreed capabilities, not an automatic stop.

### CI

| Job | Runs |
|---|---|
| `rust-lint` | `cargo fmt --check`, clippy with `-D warnings` on the workspace except `sdm-testkit` (so no DuckDB build), the L0 scripts, the skip and ignore ban, the contract schema regeneration and lock check |
| `rust-deny` | `cargo deny check` |
| `rust-test` | clippy on `sdm-testkit`, `cargo test --workspace` (L1), then `sdm-contract-run` over the L2 corpus with the checkpoint's comparator and ratchet; diagnostic cases reported apart |
| `native-wheel` | Pinned `maturin` in a manylinux2014 container, no new action: x86_64 abi3 wheel, effective target flags recorded, `auditwheel show`, upload |
| `wheel-smoke` | A clean `python:3.10-slim` with `pyarrow==10.0.1`, `pandas==1.5.3` and `numpy<2` (decision A8), and a clean `python:3.14-slim` with the newest pyarrow: install the wheel only, import, the admission round trip, one S0 validation, the missing-library control |
| `contract-python` | Per runtime cell (H1): `python-old` and `python-core` over the L2 corpus with the checkpoint's comparator and ratchet; register validation; coverage `--check` |
| Existing jobs | `lint`, `hygiene`, `typecheck`, `test`, `engine`, `build`, `installed-wheel`, `sbom`, now installing the client from the `native-wheel` artifact |

- Rust jobs use the pinned toolchain and a cache keyed on `Cargo.lock`; third-party actions are pinned by commit, as the organisation's allow list requires; the hygiene check's shipped suffixes cover `rust/`; `publish.yml` does not change; the fast tier's run time is optimized, never a reason to drop cases.
- The local gate (`scripts/rust_core/gate.sh`) runs `python-ref` with the guard, the full L2 on all runners and cells it can run, the checkpoint's comparator and ratchet, coverage `--gate S0`, the seeded-fault suite and the wheel smoke in local Docker; `gate.sh --locked 1a|1b` exits non-zero when a locked check fails, prints the check table, and writes `evidence/gate/<sha>.json`.
- `scripts/rust_core/status.py` builds one markdown status page from git, CI, decisions.md, the tracker and the register: candidate commit, gates passed, open failures, decisions needed, stop-rule counters, open branches, cold and warm run times of the fast tier, and the size trends (decision A9).

## Tasks

Tasks are listed in execution order; T1.6b runs after T1.11e, and the half column keeps the plan's 1a and 1b exits.
T1.12 to T1.19 may proceed on topic branches while T1.11a to T1.11e are open; a T1.11 sub-task failing its exit after two fix rounds goes to the owner, and 1a continues.
"Green" means in CI on `rust-core-phase-1` unless it says local; the last column names the checks a task activates, which every later landing runs.
A task is done when its exit holds in CI and this file and the tracker say what it changed (H18).
Each readiness item (sections 1.1 to 1.14 of the planning input `A-readiness-audit.md`) maps to a task below; the items it marks later are the Databricks percentage `repr` (Phase 3), scope rule 7 (Phase 2), column-wise object columns (S3) and RS-5 (tabular).

| ID | Half | Task | Output | Exit check | Activates |
|---|---|---|---|---|---|
| T1.1 | 1a | Phase documents and locked checks | This file, [tracker.md](tracker.md), [decisions.md](decisions.md) | The documents pass review; re-review round 1's findings each have a disposition and one review of the fixes agrees; the locked checks are locked as amended (H13) | |
| T1.2 | 1a | Workspace scaffold | `rust/` tree, toolchain, lints, `clippy.toml`, `deny.toml` from the archive's plus the bans above, empty 1a crates, the Arrow 58 pins, `dependencies.md`; root `uv.lock` resolved | `cargo build`, clippy and `cargo deny check` pass in CI; the hygiene check passes | `rust-lint`, `rust-deny`, `rust-test` (L1) |
| T1.3 | 1a | Zone and calendar vectors | The T0.6 corpus as an L1 vector file run against `jiff`; `rust/testdata/tzdata-2026c/` | All 2,808 zone answers and 208 offset answers agree against the vendored 2026c directory and against the bundled database, each with expectations tied to its version; ordered host lookup and the recorded per-zone source are tested on fixture directories, never the runner's system database; parsing differences from `chrono` are listed in `dependencies.md` | The zone vectors |
| T1.4 | 1a | `sdm-contract` types, schema and lock | The types S0 and the journey need, the generated schema, `contract.lock`, the lock check, a property-based decoder fuzz test | Round trips pass for `i64::MIN`, `u64::MAX`, uint64 above int64 max, `decimal128(38, 0)`, `decimal256` at precision 76, `-0.0`, NaN and both int64 timestamp ends; a Python test decodes them exactly with only `json` and `decimal`; the regeneration diff is empty; a schema change without a version bump fails, and so does a rewritten existing version entry; the fuzz test runs bounded with no panic | Schema lock check |
| T1.5 | 1a | `sdm-core` Run state machine | `Run`, dispatch, the composition root, capability checks and profiles | Every state and illegal transition is tested, including stale and duplicate reply IDs and cancellation while awaiting a fetch; a scripted operation runs to `Done` | |
| T1.6a | 1a | Interim parser port and adapter | The ported lexer, parser, visitor, AST, profiles, codes and seven validators behind the adapter, `emulation-allowlist.txt`, the validated-query schema and lock entry | The queries of T1.10 and T1.11 give their recorded outcomes; the 417-query corpus runs with its pass count reported | |
| T1.7 | 1a | `sdm-host` | Codecs, the 64 KB cap, the admission table, panic containment, busy handling, zone resolution | A passing test per rule and per refusal, including a contained panic (test-only operation), an overflowing cast that returns a named error, and a generated refusal for a bounded representative of every Arrow type family outside the set | |
| T1.7b | 1a | Contract v3 | Input-sized fields as attachments (H5); `ErrorCode` and `retryable` (H6); the seed of any size, the description-free hash for new lock entries, error overflow as an attachment (H7); `sdm_core::start` and `HostRun::start` take attachments; decisions row for the bump | Through `HostRun`: a `validate_query` start with a 5,000-column schema and query text past 64 KB is accepted, and boundary tests just below and above the cap pass for a start, a `SourceDescription` result and an error whose fields and cause arrive whole; a test lists `ErrorCode` and every code has a register or decisions row; seed `2**70` round trips exactly; a doc-comment-only change leaves the v3 hash unchanged; the lock shows one bump against the merge base; the owner check-in report is sent | Contract v3 lock |
| T1.7c | 1a | Batch 1 and 2 code fixes | Schema match at admission and against the outstanding request (H16); the admission zip fix; the lossless admission additions; `contain()` without payload text and codec errors without input text (H17); `KNOWN_DIFFERING`, a list of corpus IDs that may only shrink, in place of `AGREEING_AT_LEAST` | Required: a batch with an extra column, swapped same-typed columns, a wrong decimal scale, a wrong timestamp unit, incompatible metadata, or a different schema in a later batch is refused with `PROVIDER_SCHEMA_MISMATCH` through `HostRun`, never admitted; each lossless addition round-trips values and validity, and an int32 offset overflow splits the batch; canaries planted in a malformed control value and a test-only panic payload appear in no `CoreError` message, field or cause, and the panic hook's output is checked separately | Schema-match and canary tests |
| T1.8 | 1a | `sdm-py`, the client wheel and CI rewiring | `_native`, `_native.pyi`, the `pyo3-log` bridge, maturin backend, `pyarrow>=10.0.1`, version guard, third-party notices; Python jobs and `build` take the client from `native-wheel` | `native-wheel` builds a `cp310-abi3` wheel that `auditwheel show` reports as manylinux2014 or older, with no `target-cpu` flag; every existing job is green with the client from the wheel, and "five wheels, one version" passes; while a core call blocks, another Python thread progresses and logs under the `logging` lock without deadlock (a timeout and `faulthandler` catch a hang); a level change takes effect after the per-run cache reset; in a fresh subprocess, importing `_native` starts no thread, adds no logging handler or signal handler and reads no environment variable; a forced abort in a subprocess is reported, never "recovered"; T1.7c's schema-match tests pass through the binding | `native-wheel`; existing jobs on its wheel; the binding tests |
| T1.9 | 1a | Python host and routing seam | `_rust/`, `_routing.py`, `SDM_CORE_ROUTES`, the S0 core route in `_resolve`, `_rust/errors.py` | A refused and an accepted query give the expected outcome under the core route; "old paths never import `_native`", "all defaults are old" and "unknown route fails" pass; a test iterates `ErrorCode` and fails on any `(kind, code)` without one `StructuredDataError` subclass, and the mapping reads no message text; ordered warnings map to their public classes; a wide-schema, long-query `materialize()` the old client accepts passes validation through the installed wheel | The routing and error-mapping tests |
| T1.10 | 1a | First S0 case on every runner | `case.schema.json`, `observation.schema.json`, `inventory.schema.json`, `bases.yaml`, the inventory loader; `run_python.py` in three modes, with the reference guard's name list including `sdm_client._native` (updated on `rust-core-reference` first if needed); `sdm-testkit` and `sdm-contract-run`; `compare.py` with the outcome, kind and code rules; the case's reference observation, recorded through `run_python.py` in `python-ref` mode directly until T1.14 lands `gate.sh` | One accepted and one refused S0 case agree on `python-core` and `rust-host` with their reference observation, and `python-old` completes them; the inventory validates and the loader's lossless-mapping test passes; `python-ref` stops when the guard fails; `cargo tree -e normal` for the testkit shows no `pyo3` | The two S0 cases, in `rust-test` and the existing `test` job until T1.18 moves them to `contract-python` |
| T1.10c | 1a | L0 guards before the domain crates | `l0_deps.py` (the dependency graph of Data access with every workspace crate classified, the interim-parser rule, nothing shipped on `sdm-testkit`, no `pyo3`, `jni`, `arrow-pyarrow`, `numpy` or `duckdb` in domain crates, bridge-file headers); clippy bans in domain and shipped crates (`HashMap`, `HashSet`, `RandomState`, clocks, environment, file system, `jiff::tz::db`, `StdRng`, `ThreadRng`); the vocabulary scan with the emulation allowlist and the SQL-fragment scan of `sdm-schema`, `sdm-engine` and `sdm-request`; the skip and ignore ban | Each guard passes on the branch and fails on a hand-made breach recorded in the task's commit; seeded fixtures follow in T1.16 | L0 graph, bans and scans |
| T1.11a | 1b | Data port and driver capture | The empty 1b crates under the graph guard; `DataLayer`, `CompositionRoot` with one data layer, `Asker::fetch` over `DataRequest`, `Zones` into `sdm_core::start` (H14); `Neighbors` and `Lookup` (H15); the test-only driver capture in `run_python.py` | The full workspace builds under the graph guard with no SQL in domain crates; a fetch with several driver batches and an empty fetch with its schema give the same admitted attachment on both hosts; the capture proxy is checked against the reference's dispatch before other T1.11 code; `python-core`'s digest with capture on equals its digest with capture off on `base.regression`, and the captured statement count equals DuckDB's own query log; or Q2's infeasibility is recorded for the owner | |
| T1.11b | 1b | `describe_source` and the `Schema.from_source` route | Thin `sdm-schema` and `sdm-data` catalog requests; the DuckDB provider in both hosts; the schema bridge for `base.regression` | `base.regression`'s schema completes on `python-core` and `rust-host`; the import hook shows no `sdm_relational_engine` import after route entry; an AST scan of `_rust/providers/**` and a subprocess run with `sys.modules['pandas'] = None` show the provider uses no pandas | `base.regression` schema, diagnostic |
| T1.11c | 1b | `materialize` to the payload | Thin `sdm-engine` and `sdm-request`; key sets as attachments | `base.regression` reaches its payload on both core runners; the large key set and exact key-binding tests pass, including the float64-colliding decimals (H22); a seed of `2**70` crosses the `materialize` start and reaches the core's RS-7 check unchanged | |
| T1.11d | 1b | Scripted endpoint, `predict` and the result | The localhost endpoint in Python and Rust with its vector file; `sdm-wire`; `predict` with `RequestState` crossing runs | `base.regression` completes a model request and its result on both core runners; `RequestState` carries its version and an unknown one is refused by name; the canary test plants sentinels in data, a fake DSN credential and a test-only panic, drives every error path in the inventory, and finds none in a `CoreError` message, field or cause or in `str(exc)` | |
| T1.11e | 1b | Comparator payload and result rules | The comparator's payload and result rules | `base.regression`'s observations compare field by field on all runners, its differences diagnostic until T1.29; the owner check-in report is sent | `base.regression`, diagnostic |
| T1.12 | 1a | S0 case corpus | Accept cases per task kind and profile, a refusal per validator family, date-literal and capability refusals, and boundary cases: the 4,300-digit limit on both sides, escapes, a D8 packed date, the deepest valid query and a start past 64 KB | Every case file validates in CI | Case validation |
| T1.13 | 1a | S0 reference observations | `contract/reference/<commit>.jsonl` and `contract/reference-env.txt`, recorded after the guard update through `run_python.py` in `python-ref` mode directly until T1.14 lands `gate.sh` | The file carries the guard result and full identity, the observation schema validates it, and the comparator reads it | |
| T1.14 | 1a | Ratchet, register v2, gate and first promotion | `ratchet.py`, `register.schema.json` for `sdm-register/2` with today's entries migrated, `register.py validate`, `compare-policy.yaml`, `availability.yaml`, `protected-paths.txt`, a minimal `gate.sh` running the checkpoint's harness from its own worktree; the first promotion | First, a corrupted expectation, a missing case, a skipped call and a missing native library each fail; then seeded checks: a changed answer passes with an approved exact entry and fails without one; a new refusal fails without the owner; a removed case and a demotion to diagnostic fail; a relaxed rule or availability downgrade fails without a decisions row; a candidate edit to a protected path or the harness fails without a naming row; a branch-side baseline edit has no effect; an identity-only change passes | Ratchet; register validation; protected paths; reference immutability |
| T1.15 | 1a | Coverage generator for S0 | `coverage.py` with rules 1 to 5 and 8, `--check`, `--gate S0` | `--check` and `--gate S0` pass; an unresolved ID and a dropped path each fail | Coverage `--check`, `--gate S0` |
| T1.16 | 1a | L0 guard self-tests | `selftest.py` fixtures for every T1.10c guard; the `unsafe` check; the step-variant test | Each guard fails on its seeded fixture and passes on the branch | L0 self-tests |
| T1.17 | 1a | Clean-container wheel smoke | `wheel-smoke` job | Passes on 3.10 with pyarrow 10.0.1, pandas 1.5.3 and `numpy<2`, with the resolved environment recorded, and on 3.14 with the newest pyarrow; canonical inputs cross sharing their buffers and are released exactly; admitted types match the expected canonical schema with values and null masks kept; outputs stay valid after input collection, error and run destruction, over both the capsule and `_export_to_c` paths; busy and cancel behave through the binding | `wheel-smoke` |
| T1.18 | 1a | L2 in CI and the status page | `rust-test` and `contract-python` per runtime cell running the L2 corpus with the checkpoint's comparator and ratchet; `status.py` | Both jobs green in every cell; the status page regenerates with every item listed under CI | The full S0 L2 corpus in every cell |
| T1.19 | 1a | Local gate, seeded faults and comparator faults | The rest of `gate.sh` including `--locked`, the rest of `selftest.py`, comparator fault tests | Through the real `gate.sh` against a temporary copy of the repository, a corrupted expectation, a missing case, a skipped call, a missing native library, a guard breach and a ratchet bypass each fail, and the first, second, third, fourth and sixth also in CI; the comparator fails one seeded fault per exactness rule (null against NaN, `-0.0`, precision and scale, unit and zone, nested validity, metadata, duplicates, order, `int64` to `float64` with equal values, a last-digit decimal change, a swapped key and value) and a forced decimal-to-float key conversion (H22), and passes a regenerated observation whose request UUID changed; each locked check passes on a stub or the reference and fails on a seeded defect, or goes to the owner with a proposed narrowing; `gate.sh --locked 1a` passes | Seeded and comparator faults |
| T1.6b | 1a | Interim parser completion | The D1 to D9 reader with D4's warning and the local-zone year; the 29 quirks and custom-profile tests; a PQL fuzz test; a bounded deep-query subprocess test | All 417 corpus queries give their recorded codes and spans with `KNOWN_DIFFERING` empty and `without_readings` deleted, so date readings in the accepted trees equal the reference's; all 29 quirks hold; all 45 T0.6 date literals and the generated packed-date cases agree, with a vector at 2026-12-31T23:30 America/New_York; the deep-query test covers parsing, tree traversal and destruction; the fuzz test runs bounded with no panic | L1 parser corpus, quirks, fuzz |
| T1.20 | 1a | Phase 1a review | A completed independent review, findings and resolutions in the tracker; the held-out cases written (H11) | Every blocking finding is resolved; locked checks 1 to 3 hold through `gate.sh --locked 1a`; every Phase 1a check is active and green; the owner check-in report is sent | |
| T1.21 | 1b | DT-3 and ORD-1 findings | Answers in the register entries, before any baseline must match | DT-3 is answered from the reference alone with a bounded algorithm-level sentinel probe, the public in-memory proof deferred to Phase 4; ORD-1 says whether tie order reaches an inventory payload, with any proposed change | |
| T1.22 | 1b | Random streams | RS-1 (`SeedSequence`, PCG64, bounded draws, the shuffle) and the RS-7 refusals; other stream modules only if a reached call needs them | Algorithm vectors from numpy pass for every seed in the T0.6 model, including `2**70`; DuckDB seeds from `2**31` are refused with `INVALID_REQUEST` by the core; the vectors are protected paths | Stream vectors |
| T1.23 | 1b | `sdm-schema` and `describe_source` | Inference over Arrow; catalog requests in `sdm-data` | Every request kind renders and runs on the `shop` and `declared` bases through `sdm-testkit` with the expected Arrow schema; the same fetch gives the same admitted Arrow on both hosts for every base; `base.regression`'s schema observations match the reference; `describe_source` and a `materialize` start on a generated 5,000-column source succeed on both core runners | |
| T1.24 | 1b | `Schema.from_source` core route | The schema bridge building today's `Schema` objects | The public `Schema` under the core route equals the old one on `base.regression`; a `Schema` pickled by the reference loads, and a core-built one pickles and reloads | |
| T1.25 | 1b | Time module on `jiff` | `Instant` and `Span` (F3) first; DT-7 value-by-value reading on SQL sources, inference over the preview, offsets; the text-date shape reader as one module, `sdm-engine::time::text_shapes`, allowlisted by an owner row and written only from `date-rules.md` with the T0.6 vectors; session zone from host facts | The T0.6 column cases and the inventory's time cases infer the reference's time columns; the range-end vectors pass in every vendored zone; any shape outside `date-rules.md` is a register entry, not code | |
| T1.26 | 1b | Sampling | `sample_target` call order, default seed 42, SQL steps as batched data requests including `Neighbors`, RS-6 reservoir SQL; the comparator's nondeterminism rules | `base.regression`'s fetch traces match the reference's statement count, typed parameters and batches (H9); L1 sampler vectors on the DuckDB path (anchor equality, sub-second gaps, ties, NULL time, negative times, several seeds reaching one neighbor) pass, recording each reference sampler's own policy (register ORD-2, F2) | |
| T1.27 | 1b | Labels | Kernels with float32 Kahan sums in row order, `DateOffset` rules on `jiff` | Labels in `base.regression`'s payload match exactly | |
| T1.28 | 1b | Wire and request | `sdm-wire` from the pinned OpenAPI file; payload, response and result layout in `sdm-request`, without the `json.dumps` size count or CPython sort emulation | `base.regression`'s parsed payload matches; L1 layout vectors with distinct per-row answers and reordered responses pass; an L1 check at the payload limit counts the bytes sent (PD-2) | |
| T1.29 | 1b | Three baseline cases | The routes generalized past `base.regression` | `base.regression`, `base.binary` and `base.multiclass` meet their effective approved expectations on all applicable runners and cells, each deviation approved in this change (H19), before any other case is worked on | The three cases, as acceptance |
| T1.30 | 1b | DuckDB SQL goldens (L4 part) | Reviewed golden SQL for every statement the inventory emits, executed on DuckDB in CI | A seeded SQL change without a golden update and an entry fails | Goldens |
| T1.31 | 1b | Arrow-to-pandas presentation (L5 part) | The presentation table for the inventory's types | Its expected values, generated by the reference in `ref-pandas3` and by `python-old` in `dbr17-pandas2`, never written by the implementer, pass in CI | Presentation table |
| T1.32 | 1b | All 40 cases | Observations on all runners and cells; coverage rule 6 and `--write` | `python-core` and `rust-host` agree on `core` fields for all 40; every difference is in the report with its entry, approved as its case landed (H19); statement counts equal the reference's or are registered; a contradicted cell claim fails | All 40 cases; coverage rule 6 |
| T1.33 | 1b | Register reconciliation | Every changed case's entry complete; the owner's list of new refusals | P1B-1 is replaced case by case; RS-6, DT-7, DT-4 and DT-8 are reviewed; no proposed entry remains without its approver; the ratchet passes | |
| T1.34 | 1b | Type-set findings | Each arrow-rs `DataType` classified as accepted, admitted or refused by name; any proposed change to the set | Every type the inventory produced is classified, and any change is proposed to the owner | |
| T1.35 | 1b | Evidence pack and held-out run | `phase-1/evidence/`; the held-out cases run once (H11) | Every item below is present and generated from the branch's own runs | |
| T1.36 | 1b | Phase 1b review, decision and merge | A completed independent review; the owner's go or no-go in decisions.md | Every blocking finding is resolved; `gate.sh --locked 1a` and `--locked 1b` pass on the exact merge SHA; migration-plan.md carries every confirmed row; the owner's decision is recorded; on go, the local merge into `rust-core` | |

## Owner check-ins

The executor stops at each point below and sends the owner a five-line report (H12): tasks done since the last report; stop-rule counters at 1 or more; decisions awaiting the owner; locked-check status; the next cut the executor would make if asked to cut.
Work past a check-in waits for the owner's answer on the rows it lists; tasks already in progress finish.

| Point | After | Rows put to the owner |
|---|---|---|
| Contract v3 | T1.7b | E1 to E14, H5 to H7, and the orchestrator rows of this revision |
| Journey | T1.11e | Q2 if capture was infeasible, H14 and H15's shapes, T1.11 escalations |
| Phase 1a | T1.20 | Every open row; the held-out threshold (H11) |

## Phase 1b go or no-go evidence pack

Written to `phase-1/evidence/`, generated from the branch's own runs wherever it can be.

- Identity: phase branch commit, `rust-core` base, reference commit and checkpoint SHA, runtime cells and platforms, the DuckDB version on both hosts, per-zone database sources and versions, the wheel's SHA-256.
- The 40-row table printed by `gate.sh --locked 1b`: the reference outcome, `python-core` per cell, `python-old`, `rust-host`, and, best effort, the archive (commit `3887762`, where it already builds; otherwise "not measured"); each outcome is classed against the approved semantics (correct success, expected refusal, approved improvement or regression), with the register entry for each difference.
- Per hard seam, the deepest stage each case reached (driver Arrow, admission, inferred schema, payload, result, or refusal); stages not reached are listed as unproven.
- The comparator's field-level report for every difference; the register entries, with the cases that need the owner's call; the type-set findings; the DT-1, DT-2, DT-3 and ORD-1 results.
- The held-out run: per case, outcome against the reference, and the count against H11's threshold.
- Boundary evidence: the L0 report, `cargo tree` for the pure-Rust host, and the count of semantic callbacks into Python (zero).
- Size: Rust lines per crate (`tokei`) against the Python each replaces; the size trends; the dependency table; the emulation allowlist as it stands.
- Offline measurements, for information: statements per case against the reference, and wheel size; CPU and import time belong to L7.
- CI run links, the seeded-fault results, the review's findings and resolutions, and an explicit list of what was not done.
- The no-go criteria (C9) with each one's result, and the executor's recommendation, labelled as such; the decision is the owner's.

## Gates

### Merge rules

- Every candidate that passes its checks is pushed before more work builds on it.
- A work package lands on the phase branch when existing CI and every check activated so far in the task table are green; no activated check is disabled later, and every Phase 1a check is active at T1.20.
- Diagnostic cases run and report in CI but stay apart from the acceptance corpus: they count toward no coverage and set no baseline until matched or approved; a diagnostic case's fields are gating from the landing at which they match; `base.regression` is diagnostic until T1.29, and reaching its result is gating from T1.11e.
- A change to bindings or native dependencies also passes `wheel-smoke` once it is active.
- A type or conversion fix lands only with a public case that failed first.
- Phase 1a does not merge on its own; 1a and 1b merge into `rust-core` together, locally, after the owner's go.
- S0 has no separate slice-done gate in Phase 1: its L3 run and the update from `main` come before any S0 path switches in Phase 3.

### Locked acceptance checks

Locked at T1.1 and amended by the owner on 2026-10-08 (H13); any later change goes to the owner.

1. 1a exit: S0 green against the committed reference observations, `python-core` and `rust-host` matching them in `ref-pandas3` (validity and `StructuredDataError.code` in Python, kind and code on the core) and `python-core` equal to `python-old` in the other cells; the L1 corpus green on codes and spans; coverage `--gate S0` passing; no S0 case refuses an input the reference accepts without an owner-approved register entry.
2. 1a exit: the seeded faults and comparator fault tests of T1.19 and the ratchet's seeded checks of T1.14 each behave as stated.
3. 1a exit: the wheel installs and loads in both clean containers of T1.17, including the pyarrow 10.0.1 floor.
4. 1b exit: on all 40 inventory cases, `python-core` and `rust-host` agree on `core` fields; in `ref-pandas3`, `python-core` matches the reference, and in `dbr17-pandas2` it equals `python-old`, on inferred types, keys, time columns, data requests (statement count, typed parameters and consumed batches; raw SQL text is diagnostic), payloads, results, warnings and refusals, or differs only where an approved register entry states that case's cell and exact old and new outcome; a field marked unavailable fails this check; the core's SQL matches the DuckDB goldens.
5. 1b exit: no case refuses an input the reference accepted without an owner-approved entry, and no case panics.
6. Both halves: every activated check is green, and all Phase 1a checks are active from T1.20.

### Process rules

- At most two review rounds per change and per half; the reviewer is a fresh agent that checks code and behavior, not prose; a third round goes to the owner.
- Stop rules: two fix rounds on the same divergence go to the owner; a check failing twice the same way changes method; a phase that keeps finding new work stops for a re-plan; while a blocking question waits for the owner, work narrows to the current task.
- The tracker keeps the stop-rule counters; a counter reaching 2 is escalated to the owner in the same commit that records it (H20).
- An agent never sets `approved_by: owner` except to record the owner's own written answer, citing it, and never approves its own entry.
- Reproduce a failure from the installed package before any remote run; Python gates need the Python and pure-Rust checks only.

## Risks

| Risk | Guard | Learnings lesson |
|---|---|---|
| The harness grows before anything runs end to end | One S0 case and `base.regression` run before the tooling generalizes; new acceptance capabilities need the owner; size trends reviewed | 5, 9 |
| The Python value model returns, through the parser or through pandas dtype modelling | Three named emulation modules, confined to one crate, additions only by owner row; `Literal` replaces the Python value type; dtype names come from a host table; the vocabulary scan from T1.10c | 1, 3 |
| Bit-for-bit matching returns on SQL text and messages | Message text never gates; core SQL is gated by goldens and the reference by statement count, parameters and batches; each changed case gets its own entry | 2 |
| A fix breaks neighboring cases, or an expectation is edited to pass | The checkpoint's own harness judges; protected paths; the ratchet against a fixed checkpoint SHA; approved entries with exact outcomes | 7, 9 |
| A Python assumption hides in the contract | The pure-Rust host runs every case through `HostRun`; no callback step; key sets as Arrow; the L0 graph guard before the domain crates | 5, 11 |
| A boundary silently transforms data | Schema match against the declared and the expected schema; lossless admission only; float64-colliding decimal keys tested | 1, 6 |
| The reference is contaminated by the new code | `python -I`, `require_isolation()` before and after, identity per run, the guard-breach control; the driver capture changes no reference code | 6 |
| One T1.11 sub-task stalls the phase | T1.11 split in five with the escalation rule; 1a continues on topic branches | 9, 10 |
