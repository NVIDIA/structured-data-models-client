# Phase 1: the 1a harness and the 1b spike

**Branch:** `rust-core-phase-1`, off `rust-core`, for both halves; it merges locally into `rust-core` only after the owner's go at the end of 1b.
**Tracker:** [tracker.md](tracker.md).
**Decisions:** [decisions.md](decisions.md).

## Goal

Phase 1a builds the contract, the first core crates, the binding and wheel, the runners, the comparator, the ratchet and the CI checks, and proves them on S0 (PQL validation).
Phase 1b runs one hard journey end to end on the new architecture: DuckDB `Schema.from_source` to `materialize` to `predict` to the result, through the contract and the data provider interface, in Python and on the pure-Rust test host, over the 40 fixed cases of [phase-1b-inventory.yaml](../phase-0/phase-1b-inventory.yaml).
The phase ends with an evidence pack and the owner's go or no-go: go merges 1a and 1b into `rust-core` and starts Java's S0 work; no-go means repairing the first attempt in place, or stopping.

## Scope

### Phase 1a

- The Rust workspace, toolchain, lints, `cargo-deny` and the L0 guards.
- Crates: `sdm-contract`, `sdm-core` (a skeleton with the single Run state machine), `sdm-pql-interim`, `sdm-host`, `sdm-py` and `sdm-testkit` (decision A1).
- The client wheel built by maturin with the native module `sdm_client._native`, installed in a clean container.
- The private routing seam, and S0 through the core route.
- The L2 case, observation and inventory formats, the S0 corpus, the runners (three Python modes and the pure-Rust host), the comparator, recorded reference observations, the ratchet, register format v2, the coverage generator and the seeded-fault suite.
- The CI fast tier, the local gate script, and a minimal status page generator.

### Phase 1b

- Crates: `sdm-data` (the start of the data layer, not throwaway), `sdm-schema`, `sdm-engine`, `sdm-wire` and `sdm-request`, each holding only what the 40 cases need.
- The DuckDB provider in the Python host and in the pure-Rust host.
- `Schema.from_source`, `materialize` and `predict` core routes, and the scripted model endpoint in both languages.
- DuckDB SQL goldens (the L4 part) and the Python conversion table on pandas 2 and 3 (the L5 part), only for what the inventory needs (decision A5).
- The DT-3 and ORD-1 findings, register entries per changed case, the type-set findings and the evidence pack.

### Explicitly out

- Java: `sdm-jni`, a JAR, JDBC conversion tables, Java S0 and any Java runner; the pure-Rust host is the second language in Phase 1.
- Backends other than DuckDB in 1b: SQLite, Postgres, Lakebase, Databricks, Snowflake, frames and S3 reads.
- Live work: L6, the Databricks job matrix and any remote run.
- Operations outside the inventory: ranking, forecasting, link prediction, `fit`, sessions, tabular requests, metric and semantic views, YAML.
- Phase 2 items: the provider conformance kit, the non-SQL test provider, the swapped data layer, the dataframe-engine comparison, characterizing the remaining S1 cells, and the full L4 and L5 matrices (pandas 1.5 included).
- L3 differential generation, full L7 measurement, and the full L8 platform matrix; Phase 1 has one clean-container install and focused checks.
- Wheels for aarch64, musl, macOS and Windows, and any claim about the PQL C++ build floor.
- Integrating the PQL team's crate, sending or negotiating the PQL asks, and deleting the interim parser.
- Switching any public path to the core, deleting the old engine, and retiring old tests.
- Random streams the journey does not reach: RS-2, RS-3, RS-4, RS-5, and the Postgres, Snowflake and Databricks sampling SQL.
- Growing the 40-case inventory; a missing case goes to the owner as a scope decision.
- Updating the reference from `main` mid-phase, except at the go merge if the owner asks.

## Architecture choices

### Layout

```text
rust/
  Cargo.toml, Cargo.lock, rust-toolchain.toml, deny.toml, clippy.toml
  crates/
    sdm-contract/      types only; schema/ holds the generated JSON Schema and contract.lock
    sdm-core/          the Run state machine, dispatch, capability checks, the PQL adapter
    sdm-pql-interim/   the archive's parser, reachable only through sdm-core's adapter
    sdm-host/          control codecs, the 64 KB cap, Arrow admission, panic containment, run ownership
    sdm-testkit/       dev-only: case loader, pure-Rust host, duckdb-rs, scripted endpoint
    sdm-data/          1b: data requests, capabilities, SqlDialect, the DuckDB dialect
    sdm-schema/        1b: canonical schema and inference over Arrow
    sdm-engine/        1b: time, random streams, sampling, labels
    sdm-wire/          1b: generated wire types
    sdm-request/       1b: materialization, payload, response, result layout
  bindings/sdm-py/     PyO3 cdylib `_native` over sdm-host
contract/              case, observation and inventory schemas; bases.yaml; cases/; reference/; baselines/; runners/run_python.py
scripts/rust_core/     compare.py, ratchet.py, register.py, coverage.py, l0_*.py, status.py, gate.sh
packages/structured-data-models-client/src/sdm_client/
  _native.pyi          typed stub, so mypy needs no binary
  _routing.py          the routing seam
  _rust/               the Python host: run loop, providers, facts, error table, schema bridge
```

- `contract/` sits at the root because it belongs to neither language.
- The Python host package is `_rust`, not `_core`, so it cannot be confused with a name the reference guard probes.
- `dependencies.md` in this folder records one row per dependency: use a library, port a defined algorithm, or keep in the host, with the measured reason.

### Workspace rules

- Edition 2024, `rust-version = "1.97"`, toolchain pinned to 1.97.0 as the minimum supported Rust version, `Cargo.lock` committed.
- `panic = "unwind"` stated explicitly, so panics can be contained.
- `unsafe_code = "forbid"` in workspace lints and at every domain crate root; only `sdm-py` and `sdm-testkit` relax it, locally, for Arrow FFI.
- Clippy runs with `-D warnings`; each `#[allow]` carries a one-line reason.
- Every `.rs` and `.toml` file carries the SPDX header the hygiene check enforces.

### Dependencies

- One `arrow-*` major across the workspace: the one `duckdb-rs` uses at the reference's DuckDB version (1.5.6), with the matching `arrow-pyarrow` and `pyo3`; `arrow`, `duckdb` and `pyo3` are bumped together, only at slice gates (decision B-D2).
- `jiff` with the host's zone directory passed in explicitly and a bundled fallback is the one calendar and zone library; `chrono` stays only as Arrow's transitive dependency and `chrono-tz` is banned (decision A7).
- `serde`, `serde_json`, `schemars` 1.x, `thiserror`, `indexmap`, `stacker`; `duckdb` only in `sdm-testkit`; `anyhow` only in `sdm-testkit`.

### Contract

- `ColumnType` is the closed type set of plan section 3; `ColumnType::from_arrow` is the only way in, and anything else is refused with `kind: unsupported, code: UNSUPPORTED_TYPE` naming the column and the Arrow type.
- The scope manifests' cell vocabulary is generated from that enum.
- Operations are a versioned, serde-tagged enum; Phase 1 needs `validate_query`, `describe_source`, `materialize` and `predict`.
- `Step` has exactly three variants: `Fetch` (a batch of typed data requests, each with compiled SQL and typed parameters for SQL providers, and the expected Arrow schema or a discovery marker), `CallModel` (body as bytes) and `Done`.
- There is no step that asks the host for language behavior.
- Errors are `{kind, code, message, fields, cause}` with closed kinds; warnings travel on their own channel, in order.
- No contract string names a Python or Java class; each host maps kinds and codes through one reviewed table.
- Every scalar in a control message is a tagged object with its value as a JSON string (integers per width, `d128` and `d256` with precision and scale, floats with shortest round-trip text and non-finite values, temporal counts with unit and zone), so nothing passes through a double.
- `HostFacts` (clock, local zone, zone database source, path and version, platform padding rule) is part of every operation.
- State such as `RequestState` is typed and versioned; runs and Arrow buffers have defined owners and release rules.
- The JSON Schema is generated by `schemars`, committed, and locked by hash in `contract.lock`; the contract version is independent of the package version.

### Core, host and binding

- `sdm-core` has one `Run` type: operation bodies are `async fn`s over an asker handle, polled once per step, with no executor and no threads.
- States are `Ready`, `AwaitingFetch`, `AwaitingModel`, `Finished`, `Cancelled` and `Poisoned`; an illegal call returns a `protocol` error and never panics.
- The composition root picks a built-in SQL provider by dialect or a host provider with declared capabilities; nothing else in the core matches on a backend name.
- `sdm-host` is plain Rust: it owns runs, enforces the 64 KB control cap, wraps every entry in `catch_unwind` (a panic poisons the run and returns `INTERNAL_ERROR` with `fields.panic = true`), and answers a concurrent call with `RUN_BUSY`.
- Arrow-to-Arrow admission (`large_*` to small forms, `timestamp[s]` and `[ms]` to `us`, `large_list` and `fixed_size_list` to `list`) lives once in `sdm-host`, shared by every language; native-to-Arrow tables stay in each host (decision B-D6).
- `sdm-host` builds the zone database from the host fact: the directory the host names, else the bundled copy, and records which and its version.
- `sdm-py` is a thin PyO3 module over `sdm-host`; it releases the GIL, passes Arrow through the PyCapsule interface from pyarrow 14 and `_export_to_c` below it, raises one `_native.CoreError`, passes model bodies as bytes, and has no Python callbacks of any kind.

### Packaging

- The client's build backend becomes maturin on the epic branch (decision B-D3); `main` is untouched.
- The native module is `sdm_client._native` (decision A3), in the `structured-data-models-client` wheel, tagged `cp310-abi3`, built as `manylinux2014` x86_64 in Phase 1.
- `pyarrow>=10.0.1` becomes a base dependency; `pandas>=1.5` stays.
- The wheel ships third-party notices for every Rust dependency linked in.
- A version guard holds the Cargo version of `sdm-py` equal to the other four distributions.
- CI's `build` job asserts five pure wheels today; it changes to build four as today and take the client wheel from the `native-wheel` job, so the "five wheels, one version" check still holds.
- Importing `sdm_client` never imports `_native`; only `_rust/` does, lazily, on a core route; a test holds this.
- If `_native` is missing on a core route, the host raises `ImportError` naming the broken install, and runners count it as a failure.

### Routing seam

- `sdm_client/_routing.py` maps public path patterns (the `scope-python.yaml` path IDs) to `old` or `core`; every default is `old`, and a test fails if one changes before the Phase 3 switch.
- `SDM_CORE_ROUTES` forces paths onto the core for tests; it is read at every call, never at import, is private, is not public API, and is deleted in Phase 5 (decision B-D7).
- S0 is the public call that validates PQL today, `materialize()`, which validates before reading data; under the core route `_resolve` calls `validate_query` and maps a refusal to `StructuredDataError(code='INVALID_REQUEST')`.
- While S1 is not switched, an accepted query is also checked by the old validator, and a disagreement is `INTERNAL_ERROR`; this shadow check leaves with the old code.
- "S0 green in Python" means: under the forced core route, the public call gives the same outcome as the reference, compared on validity, error code and span, not message text (decision A2).

### Interim parser

- `sdm-pql-interim` holds the archive's lexer, parser, visitor, AST, profiles, codes and the seven validators, with their corpus (193 valid, 224 invalid), 29 quirks and custom-profile tests.
- `pyrepr.rs`, `pyint.rs`, `printable.rs` and the `py_repr` and `py_str` helpers are not ported, since message text is not promised.
- A neutral `Literal` replaces the archive's Python value type; integer literals keep their digits exactly.
- The 1,695-line `dates.rs` is replaced by a reader written from rules D1 to D9 over `jiff`, taking the clock and padding rule from host facts; D8 (packed dates) is ported only as far as the recorded examples need.
- `sdm_core::pql` is the only module that names the crate; the PQL team's crate later goes behind the same function.
- Python-emulation modules are allowed only in `sdm-pql-interim`, on an allowlist that expires when the PQL team's crate replaces it, and are banned everywhere else (decision A4).

### Pure-Rust test host

- `sdm-testkit` holds the case loader, the pure-Rust host, the scripted endpoint and the runner binary `sdm-contract-run`; it is a 1a deliverable (decision A1) and never a dependency of a shipped crate.
- It builds each case's data with `duckdb-rs` at the reference's DuckDB version, checks `SELECT version()` against the recorded one, answers `Fetch` with the driver's Arrow, and answers `CallModel` with the scripted endpoint.
- It writes the same observation format as the Python runner, with no Python in the process.

### Runners

- One Python runner, `contract/runners/run_python.py`, run with `python -I`, in three modes: `python-old` (defaults), `python-core` (`SDM_CORE_ROUTES` set for the case's paths) and `python-ref` (the reference environment).
- `python-ref` loads the reference guard by path and calls `require_isolation()` in the same interpreter before the first case and after the last, and records the reference commit, the environment hash and the guard result.
- SQL traces on the old path and the reference come from DuckDB's own `QueryLog`; core runners also record SQL and parameters from the step trace, and the two must agree.
- The scripted endpoint is one written rule, implemented in Python and in Rust and held together by a shared vector file.
- Every observation records the runner, the repository commit, the `_native` SHA-256, the reference commit, the contract version, the DuckDB version and the zone database source and version.

### Comparator

- One implementation, `scripts/rust_core/compare.py`, prints the first difference per field with its path.
- Exact: outcome, stage, error class, code, kind (between core runners), inferred schema, statement count, warnings in order by class and code.
- SQL text is exact after whitespace normalization and parameters exact where both sides record them; a difference passes only with a register entry or an approved golden; numeric text in SQL is never normalized.
- Payloads are compared after parsing, excluding only the named UUID fields; results by Arrow schema and values, with row order only where the operation defines it.
- Message text is never a gate; the inventory's message fragments are reported as information.
- Nondeterministic draws follow `random-streams.md`: membership, cardinality and named invariants instead of row equality, with downstream fields compared by replaying the same captured fetch results; exclusions only for named field paths.

### Reference observations and ratchet

- Reference results are committed as expected observations under `contract/reference/<reference-commit>.jsonl`, regenerated locally at each gate with the reference commit recorded, so CI compares without running the reference (decision B-D8).
- `contract/baselines/<runner>.jsonl` holds each case's normalized observation and its digest, excluding message text and named nondeterministic fields.
- CI reads the baseline from `origin/rust-core`, never from the branch under test.
- A changed digest, a new refusal or a removed case fails unless a register entry lists that case, runner, old and new outcome; a new case passes as an addition.
- `ratchet.py accept` rewrites the baselines on `rust-core` as part of each merge.

### Register v2

- `register.yaml` moves to `sdm-register/2` (decision A6): each entry may carry `case_outcomes`, a list of `{case, runner, old, new, summary}`, and `decision_ref` becomes `{file, row}` so it can cite any phase's decisions file.
- `register.py validate` checks the schema, that IDs are unique and never reused, that case IDs resolve to corpus cases or an allowed legacy form, and the status rules.
- `register.py diff origin/rust-core` lists changes of `status` and `approved_by`, for the status page and the owner's merge.
- Only the owner sets `approved_by: owner`; the reviewer may approve entries that refuse nothing.

### Coverage generator

- `scripts/rust_core/coverage.py` implements the eight rules of [scope-format.md](../phase-0/scope-format.md) over both manifests and the corpus.
- Rule 6 checks each cell claim against the case's own observation (that Arrow type in that role on its own backend and path).
- The manifests' `cases` lists are generated (`--write`) and checked (`--check`), never edited by hand.
- `--gate S0` runs in Phase 1; `--gate S1`, which also enforces rule 7, is a Phase 2 obligation.

### CI

| Job | Runs |
|---|---|
| `rust-lint` | `cargo fmt --check`, clippy with `-D warnings`, the L0 scripts, the contract schema regeneration and lock check |
| `rust-deny` | `cargo deny check` |
| `rust-test` | `cargo test --workspace` (L1), then `sdm-contract-run` over the L2 corpus against the reference observations and the ratchet |
| `native-wheel` | maturin manylinux2014 x86_64 abi3 wheel, `auditwheel show`, upload |
| `wheel-smoke` | A clean `python:3.10-slim` with `pyarrow==10.0.1` and `pandas==1.5.3` (the floor, decision A8) and a clean `python:3.14-slim` with the newest pyarrow: install the wheel only, import, an Arrow round trip over every admitted type, one S0 validation, the missing-library control |
| `contract-python` | `python-old` and `python-core` over the L2 corpus against the reference observations and the ratchet; register validation; coverage `--check` |
| Existing jobs | `lint`, `hygiene`, `typecheck`, `test`, `engine`, `build`, `installed-wheel`, `sbom`, now installing the client from the `native-wheel` artifact |

- Rust jobs use the pinned toolchain and a cache keyed on `Cargo.lock`.
- The hygiene check's shipped suffixes cover `rust/`.
- `publish.yml` does not change in Phase 1.
- The warm-cache run time of the fast tier is shown on the status page; it is optimized, never a reason to drop cases.
- The local gate (`scripts/rust_core/gate.sh`) runs `python-ref` with the guard, the full L2 on all four runners, comparator and ratchet, coverage `--gate S0`, the seeded-fault suite and the wheel smoke in local Docker.
- A minimal generator (`scripts/rust_core/status.py`) turns git and CI status into one markdown status page: candidate commit, gates passed, open failures, decisions needed, open branches (decision A9).

## Tasks

IDs follow the technical draft; tasks added here take the next free number.
The table is in execution order; independent tasks may run in parallel on short topic branches off `rust-core-phase-1`, at most one per task.
"Green" means in CI on `rust-core-phase-1` unless it says local.

### Phase 1a

| ID | Task | Output | Exit check |
|---|---|---|---|
| T1a.1 | Phase documents and locked checks | This file, [tracker.md](tracker.md), [decisions.md](decisions.md) | The owner has confirmed or changed every orchestrator decision in decisions.md; the acceptance checks under "Gates" are locked |
| T1a.2 | Workspace scaffold | `rust/` tree, toolchain, workspace lints, `clippy.toml`, `deny.toml` from the archive's plus the bans below, the 1a crates empty, Arrow major pinned, `dependencies.md`; root `uv.lock` ignored or removed | `cargo build`, clippy `-D warnings` and `cargo deny check` pass in `rust-lint` and `rust-deny`; the hygiene check passes with SPDX headers on every new file |
| T1a.3 | Zone and calendar vectors | The T0.6 corpus (parsing shapes, units, offsets, zones) as an L1 vector file run against `jiff` | All 2,808 zone answers and 208 offset answers agree with both the host directory and the bundled database; each parsing difference from the earlier `chrono` result is listed in `dependencies.md` |
| T1a.4 | `sdm-contract` types | Type set, operations, data requests, capabilities, steps, replies, errors, warnings, scalars, host facts, versioned state | Round-trip tests pass over every type and the edge values (`i64::MIN`, `u64::MAX`, uint64 above int64 max, `decimal128(38, 0)`, `decimal256` at precision 76, `-0.0`, NaN, both int64 timestamp ends); a Python test decodes the same values with only `json` and `decimal`, exactly |
| T1a.5 | Contract JSON Schema and version guard | Generated schema, `contract.lock`, the lock check, a property-based decoder fuzz test | Regeneration diff is empty; a seeded schema change without a version bump fails; the fuzz test runs bounded in `rust-test` with no panic |
| T1a.6 | `sdm-core` Run state machine | `Run`, dispatch, the composition root, capability checks and profiles | Every state and every illegal transition is tested; a scripted operation runs to `Done` |
| T1a.7 | `sdm-pql-interim` and the adapter | The ported parser behind `sdm_core::pql`, the D1 to D9 reader, the validated-query schema and lock entry, a property-based PQL fuzz test, the ANTLR grammar kept as CI reference | All 417 corpus queries give their recorded codes and spans, all 29 quirks hold, all 45 T0.6 date literals agree; the fuzz test runs bounded with no panic |
| T1a.8 | `sdm-host` | Codecs, the 64 KB cap, the admission table, panic containment, busy handling, zone database construction | A passing test per rule and per refusal, including a contained panic (test-only operation) and a generated refusal for every Arrow type outside the set |
| T1a.9 | `sdm-py` and the client wheel | `_native`, `_native.pyi`, maturin backend, `pyarrow>=10.0.1`, version guard, third-party notices | `native-wheel` builds a `cp310-abi3` manylinux2014 wheel and `auditwheel show` reports manylinux2014 or older |
| T1a.10 | CI rewiring | Python jobs install the client from the wheel; `build` job takes it from `native-wheel` | Every existing job is green with the client from the wheel, and the "five wheels, one version" check passes |
| T1a.11 | Clean-container wheel smoke | `wheel-smoke` job | Passes on 3.10 with pyarrow 10.0.1 and pandas 1.5.3 and on 3.14 with the newest pyarrow, including a table with every admitted type in and back out with identical schema, values and null masks |
| T1a.12 | Python host and routing seam | `_rust/`, `_routing.py`, `SDM_CORE_ROUTES`, the S0 core route in `_resolve` | A refused and an accepted query give the expected outcome under the core route; the "old paths never import `_native`" and "all defaults are old" tests pass |
| T1a.13 | Case, observation and inventory formats | `case.schema.json` (`sdm-case/1`), `observation.schema.json` (`sdm-observation/1`), `inventory.schema.json` for the inventory as written, `bases.yaml`, the inventory loader | Every case file and the inventory validate in CI; the loader's lossless-mapping test passes; inventory IDs are used unchanged as L2 case IDs |
| T1a.14 | S0 case corpus | Accept cases per task kind and profile, a refusal per validator family, date-literal and capability refusals | Coverage `--gate S0` passes on the case declarations |
| T1a.15 | Python runner, three modes | `run_python.py`, `QueryLog` traces; the reference guard's name list confirmed to include `sdm_client._native`, updated and pushed on `rust-core-reference` if not | All three modes write schema-valid observations for the S0 corpus; `python-ref` stops when the guard fails, and the guard's self-test passes in the reference session |
| T1a.16 | `sdm-testkit` and `sdm-contract-run` | Pure-Rust host, scripted endpoint, `duckdb-rs` | The S0 corpus runs with no Python in the process; `cargo tree -e normal` for the testkit shows no `pyo3` |
| T1a.17 | Comparator | `compare.py` with the rules above | Every rule has a passing test and a must-fail test on hand-built observation pairs |
| T1a.18 | S0 reference observations | `contract/reference/<commit>.jsonl` recorded locally | The file exists with the guard result and full identity, and the comparator reads it |
| T1a.19 | Register v2 tooling | `register.schema.json` for `sdm-register/2`, today's entries migrated, `register.py validate` and `diff` | `validate` passes on every entry in CI; `diff` lists a seeded status change |
| T1a.20 | Ratchet and bootstrap baseline | `ratchet.py`, the first baselines on `rust-core` | A seeded changed outcome fails, and passes once an entry with the matching case outcome exists; a branch-side baseline edit has no effect |
| T1a.21 | Coverage generator | `coverage.py`, rules 1 to 8, generated `cases` lists | `--check` and `--gate S0` pass; an unresolved ID, a dropped path and a contradicted cell claim each fail |
| T1a.22 | L0 guards | `l0_deps.py` (graph, one-way flow, only `sdm-core` names `sdm-pql-interim`, nothing shipped depends on `sdm-testkit`, no `pyo3`, `jni`, `arrow-pyarrow`, `numpy` or `duckdb` in domain crates), `unsafe` attribute check, vocabulary scan with the A4 allowlist, step-variant test, SQL-fragment scan of `sdm-schema`, `sdm-engine` and `sdm-request`, clippy bans (`HashMap` and `HashSet` in domain crates, clocks, environment, file system and `jiff::tz::db`) | Each guard fails on a seeded violation on a throwaway branch and passes on the phase branch |
| T1a.26 | Status page generator | `status.py` and its markdown output | The page regenerates from git and CI alone and lists commit, gates, open failures, decisions needed and open branches |
| T1a.23 | L2 in CI | `rust-test` and `contract-python` running the L2 corpus, comparator and ratchet | Both jobs are green, and the fast tier's warm-cache run time is on the status page |
| T1a.24 | Local gate and seeded faults | `gate.sh`, `selftest.py` | A corrupted expectation, a missing case, a skipped call, a missing native library, a guard breach and a ratchet bypass each fail locally, the first, second, third and sixth also in CI; `gate.sh` passes on the branch |
| T1a.25 | Phase 1a review | Review findings and resolutions in the tracker | S0 green on `python-old`, `python-core` and `rust-host`; the seeded faults fail; the wheel installs in both clean containers; a fresh-agent review closed in at most two rounds |

### Phase 1b

| ID | Task | Output | Exit check |
|---|---|---|---|
| T1b.1 | `sdm-data` | Data requests, capabilities, provider errors with SQLSTATE and vendor codes, `SqlDialect`, the DuckDB dialect ported without the archive's capture scaffolding; key sets as one JSON parameter (Q1) | Every request kind renders and runs on the `shop` and `declared` bases through `sdm-testkit` with the expected Arrow schema |
| T1b.2 | DuckDB provider in both hosts | `_rust/providers.py` and the testkit provider, both through admission | The same fetch gives the same admitted Arrow on both hosts for every base; an import tracker shows the Python provider imports no pandas |
| T1b.3 | Random streams | `RandomState` and MT19937 ported; `SeedSequence`, PCG64, bounded draws and the shuffle written new; RS-7 seed refusals | L1 vectors from numpy pass for every seed in the T0.6 model, including `2**70`; seeds from `2**31` are refused with `INVALID_REQUEST` |
| T1b.4 | `sdm-schema` and `describe_source` | Inference ported onto Arrow; catalog SQL moved to `sdm-data` | `schema` observations match the reference on the three baseline cases |
| T1b.5 | `Schema.from_source` core route | The schema bridge building today's `Schema` objects | The public `Schema` under the core route equals the old one on the baseline cases; a `Schema` pickled by the reference loads, and a core-built one pickles and reloads |
| T1b.6 | Time module on `jiff` | DT-7 value-by-value reading on SQL sources, inference over the preview, the accepted shapes of `date-rules.md`, offsets; session zone from host facts | The T0.6 column cases and the inventory's time cases infer the reference's time columns |
| T1b.7 | Sampling | `sample_target` call order, default seed 42, SQL hops as batched data requests, RS-6 reservoir SQL | Fetch traces of the baseline cases match the reference in count, SQL text and rows |
| T1b.8 | Labels | Kernels with float32 Kahan sums in row order, `DateOffset` rules on `jiff` | Labels in the baseline payloads match exactly |
| T1b.9 | Wire and request | `sdm-wire` generated from the pinned OpenAPI file; payload, response and result layout in `sdm-request`, without the `json.dumps` size count or CPython sort emulation | Parsed payloads of the baseline cases match the reference |
| T1b.10 | `materialize` and `predict` routes | Core routes, the scripted endpoint in both languages with its vector file, model bodies passed as bytes | `base.regression`, `base.binary` and `base.multiclass` match the reference on all four runners, before any other case is worked on |
| T1b.17 | DuckDB SQL goldens (L4 part) | Reviewed golden SQL for every statement the inventory emits, executed on DuckDB in CI | Goldens are committed and checked in CI; a seeded SQL change without a golden update or entry fails |
| T1b.18 | Python conversion table (L5 part) | The pandas-to-Arrow table for the inventory's types on pandas 2 and 3 | The table's expected values pass in CI on pandas 2 and on pandas 3 |
| T1b.11 | DT-3 and ORD-1 findings | Answers written into the register entries | DT-3 says whether the `Timestamp.min` sentinel survives the sampler's round trip; ORD-1 says whether tie order reaches an inventory payload, with any proposed change |
| T1b.12 | All 40 cases | Observations on all four runners | `python-core` and `rust-host` agree on all 40; every difference from the reference is in the comparator report; statement counts equal the reference's |
| T1b.13 | Register entries | One entry per changed case with its case outcomes; the owner's list of new refusals | P1B-1 is replaced case by case; RS-6, DT-7 and DT-4 are reviewed; the ratchet passes |
| T1b.14 | Type-set findings | Each received Arrow type accepted, admitted or refused by name; any proposed change to the set | Every type the inventory produced is classified, and any change is proposed to the owner |
| T1b.15 | Evidence pack | `phase-1/evidence/` | Every item of the evidence pack below is present and generated from the branch's own runs |
| T1b.16 | Phase 1b review and decision | Review findings; the owner's go or no-go in decisions.md | A fresh-agent review closed in at most two rounds; the owner's decision recorded |

### Where each readiness item lives

The readiness audit's checklist (sections 1.1 to 1.14) maps to tasks as follows; items marked "later" do not apply to Phase 1.

| Area | Item | Where |
|---|---|---|
| Phase | 1a deliverables and real wheel | T1a.2 to T1a.24 |
| Phase | 1a exit | T1a.25 |
| Phase | Case format and inventory schema first; inventory taken as written | T1a.13 |
| Phase | 1b journey in Python and the pure-Rust host, scripted endpoint | T1b.1 to T1b.10 |
| Phase | 40 cases over the hard seams; 1b exit | T1b.12, T1b.13 |
| Phase | Text-date rules: inference over the preview and DT-7; frame rules (PK-7, DT-6) wait for S3 | T1b.6 |
| Phase | Type set validated against the inventory | T1b.14 |
| Phase | Go or no-go; P1B-1 case by case | T1b.13, T1b.16 |
| Contract | Closed type set refused by name; never silent nulls | T1a.4, T1a.8, T1b.14 |
| Contract | Versioned operations as serde types and JSON Schema | T1a.4, T1a.5 |
| Contract | Data steps with SQL or typed requests, typed parameters, expected schema or discovery | T1a.4 |
| Contract | Exact scalar encodings, including the inventory's wide integers and decimals | T1a.4 |
| Contract | Errors and warnings | T1a.4 |
| Contract | Host facts, session zone, two-digit years | T1a.4, T1a.8, T1b.6 |
| Contract | Typed, versioned state; owners and release rules | T1a.4, T1a.8 |
| Contract | JSON control messages capped at 64 KB | T1a.8 |
| Contract | Version bump guard for contract, steps and validated-query format | T1a.5, T1a.7, T1a.22 |
| Contract | Contract version independent; codes promised, text not | T1a.4, T1a.17 |
| Contract | JVM-representable scalars and kinds | T1a.4 |
| L0 | No `unsafe` in domain crates | T1a.2, T1a.22 |
| L0 | No `pyo3`, `jni`, `arrow-pyarrow` in domain graphs | T1a.22 |
| L0 | No Python types in core crates, with the interim allowlist | T1a.22 |
| L0 | Only data and model steps | T1a.22 |
| L0 | Pure-Rust host passes the contract subset | T1a.16, T1a.23 |
| L0 | `cargo-deny`, `disallowed_types`, `thiserror` without `anyhow` | T1a.2, T1a.22 |
| L0 | One-way dependency flow; no SQL in schema, engine or request | T1a.22 |
| L0 | `Cargo.lock` and pinned toolchain; SPDX headers | T1a.2 |
| L2 | Shared case list; compared outputs | T1a.13, T1a.17 |
| L2 | Row order, SQL traces, data-request traces | T1a.17 |
| L2 | Payloads after parsing; SQL goldens | T1a.17, T1b.17 |
| L2 | Message text not compared | T1a.17 |
| L2 | Nondeterministic draws | T1a.17 |
| L2 | Databricks percentage as Python `repr` | Later (Phase 3); the comparator never normalizes numeric SQL text (T1a.17) |
| L2 | Reference pickles load; new state versioned | T1b.5, T1a.4 |
| L2 | Coverage matrix; generated refusal per unsupported type and nested shapes | T1a.21, T1a.8, T1b.12; the full generated set across backends is later (Phase 2) |
| L2 | Ratchet | T1a.20 |
| L2 | Test the tests | T1a.24 |
| L2 | Reference commit and artifact identity per run | T1a.13, T1a.15 |
| L2 | CI compares committed reference observations | T1a.18 |
| L2 | Local first | Gates |
| Guard | `require_isolation()` in-process; name list; self-test per session | T1a.15 |
| Scope | Rules 1 to 6 and 8; case ID pattern | T1a.21, T1a.13 |
| Scope | Rule 7 | Later (S1 gate, Phase 2) |
| Register | Validation in CI; v2 per-case outcomes; cases resolve to L2 IDs | T1a.19 |
| Register | Only the owner approves owner entries | Gates |
| Register | RS-6, DT-7, ORD-1, DT-4, P1B-1 | T1b.11, T1b.13; RS-5 is later (tabular) |
| PQL | Interim parser behind one adapter; ANTLR as CI reference | T1a.7 |
| PQL | 417-case conformance suite; PQL and contract-JSON fuzzing | T1a.7, T1a.5 |
| PQL | PK-4 codes, spans, quirks; PK-5 and DT-5 date literals | T1a.7 |
| PQL | S0 observed through `materialize()` | T1a.12, T1a.14 |
| PQL | Model capability checks only, on the validated query | T1a.6 |
| PQL | PQL asks unsent, not blocking | Out (decision 31) |
| Dates | Zone check for DT-1 and DT-2 (done, A7) and vectors | T1a.3 |
| Dates | tzdata version as a host fact, pinned in tests | T1a.4, T1a.8 |
| Dates | DT-3 | T1b.11 |
| Dates | DT-7 | T1b.6 |
| Streams | RS-1, RS-7 | T1b.3 |
| Streams | RS-6, default seed 42 | T1b.7 |
| Streams | ORD-1 | T1b.11 |
| Packaging | Native abi3 wheel, no fallback, no sdist; `pyarrow>=10.0.1`; manylinux2014 | T1a.9 |
| Packaging | GIL release, PyCapsule and `_export_to_c` | T1a.9, T1a.11 |
| Packaging | Panic containment; no semantic callbacks | T1a.8, T1a.9 |
| Packaging | Focused L8 checks when bindings change | T1a.11, Gates |
| Packaging | Third-party notices | T1a.9 |
| Packaging | CI `build` job and clean-container install | T1a.10, T1a.11 |
| Performance | Driver Arrow straight to the core; zero-copy crossing | T1b.2, T1a.9 |
| Performance | JSON only for small control messages; model body passed through | T1a.8, T1b.10 |
| Performance | Batched data requests | T1b.7 |
| Performance | Object columns column-wise | Later (S3 frames) |
| Performance | `large_string` and `timestamp[s]` cast, not refused | T1a.8, T1b.2 |
| Performance | Query counts equal | T1b.12 |
| Process | Phase documents and decisions file | T1a.1 |
| Process | Locked checks, two review rounds, owner approval for new infrastructure, public reproduction first, stop rules, push passing candidates, Python first | Gates |
| Process | Status page from CI and git | T1a.26 |
| Java | No Java code; `jni` banned in domain crates | Out; T1a.22 |

## Phase 1b go or no-go evidence pack

Written to `phase-1/evidence/` on the phase branch, generated from the branch's own runs wherever it can be.

- Identity: phase branch commit, `rust-core` base, reference commit and baseline tag, both environments' versions, the DuckDB version on both hosts, zone database sources and versions, the wheel's SHA-256.
- A 40-row table: the reference outcome, `python-core`, `python-old`, `rust-host`, match or difference, and the register entry for each difference.
- The comparator's field-level report for every difference.
- The register entries proposed or approved in Phase 1, and the cases that need the owner's call.
- The type-set findings.
- Boundary evidence: the L0 report, `cargo tree` for the pure-Rust host, and the count of callbacks into Python (zero).
- Size and shape: Rust lines per crate split into ported, new and glue, against the Python each replaces; the dependency table with each use, port or host decision; any emulation the vocabulary scan found, and the A4 allowlist as it stands.
- Offline measurements, for information: statements per case against the reference, CPU per case old against new, wheel size and import time.
- The DT-1 and DT-2 result, the DT-3 finding and the ORD-1 finding.
- CI run links and the seeded-fault results.
- The fresh-agent review's findings and how each was resolved, and an explicit list of what was not done.
- The executor's recommendation, labelled as such; the decision is the owner's.

## Gates

### Merge rules

- One branch, `rust-core-phase-1`, for both halves, with at most one short-lived topic branch per task; every candidate that passes its checks is pushed before more work builds on it.
- A change lands on the phase branch when L0, L1, the L2 subset (Python and the pure-Rust host) and the ratchet are green in CI, and, from T1b.17 and T1b.18 on, the DuckDB goldens and the pandas 2 and 3 conversion table (decision A5).
- A change to bindings or native dependencies also passes `wheel-smoke` (the focused L8 check).
- A type or conversion fix lands only with a public case that failed first.
- Phase 1a does not merge on its own; 1a and 1b merge into `rust-core` together, locally, after the owner's go, with the baselines accepted as part of the merge.
- S0 has no separate slice-done gate in Phase 1: its L3 run and the update from `main` happen when L3 exists, before any S0 path switches in Phase 3.

### Locked acceptance checks

These are locked when the owner confirms T1a.1; a third change to them goes to the owner.

1. 1a exit: S0 green in Python as defined in decision A2 and on the pure-Rust host, against the committed reference observations, with coverage `--gate S0` passing.
2. 1a exit: the six seeded faults of T1a.24 each fail.
3. 1a exit: the wheel installs and loads in both clean containers of T1a.11, including the pyarrow 10.0.1 floor.
4. 1b exit: on all 40 inventory cases, `python-core` and `rust-host` agree, and each matches the reference on inferred types, keys, time columns, data requests and SQL, payloads, results, warnings and refusals, or differs only where a register entry states that case's old and new outcome.
5. 1b exit: no case refuses an input the reference accepted without an owner-approved entry.
6. Both halves: the L0 guards, the ratchet and CI's fast tier are green, and a fresh-agent review closed in at most two rounds.

### Process rules

- At most two review rounds per change; the reviewer is a fresh agent that checks code and behavior, not prose; a third round goes to the owner.
- New acceptance infrastructure beyond the runner, comparator, ratchet, register validator, coverage generator, L0 scripts and status page needs the owner's approval.
- Reproduce a failure from the installed package before any remote run.
- Stop rules: two fix rounds on the same divergence go to the owner; a check failing twice the same way changes method; a phase that keeps finding new work stops for a re-plan; while a blocking question waits for the owner, work narrows to the current task.
- An agent never sets `approved_by: owner` or approves its own exception.
- Python gates need the Python and pure-Rust checks only.

## Risks

| Risk | Guard | Learnings lesson |
|---|---|---|
| `Schema.from_source` parity tempts the core into pandas dtype modelling (the reference reports pandas dtype names) | Display dtype names come from a host table keyed by Arrow type; the core infers over typed Arrow only; the vocabulary scan forbids dtype strings in core crates | 1 |
| The Python value model returns through the interim parser | `pyrepr`, `pyint` and `dates.rs` are not ported; `Literal` replaces the Python value type; the A4 allowlist is confined to one crate and expires | 1, 3 |
| Bit-for-bit matching returns on SQL text and messages | Message text never gates; codes and classes do; SQL differences need an entry or an approved golden; each changed case gets its own entry | 2 |
| Text time reading grows into a `to_datetime` clone | Date code bounded to the shapes in `date-rules.md` and the T0.6 corpus; anything outside goes to the register | 3 |
| Hand-writing what a library does (zones, casts, decimals, DuckDB access) | `jiff` measured against the T0.6 corpus; `arrow-cast` for admission casts; Arrow's decimal types; `dependencies.md` records each choice with its measurement | 4 |
| The harness grows before anything runs end to end | 1a's contract covers only S0 and what the 1b journey needs; 1b's first exit is the three baseline cases on all four runners; new acceptance infrastructure needs the owner | 5, 9 |
| A Python assumption hides in the contract | The pure-Rust host runs every case from 1a on; no callback step exists; the L0 graph guard | 5 |
| Tests measure internals | Runners use public calls only; observations keyed by case ID; SQL traces from DuckDB's own log; no private names in baselines | 6 |
| A fix breaks neighboring cases | The ratchet against `origin/rust-core`'s baseline; a public failing case first | 7 |
| Hand-judged coverage lists | Generated case lists; cell claims checked against observed Arrow types; the inventory is fixed | 8 |
| Process outgrows product | Locked checks; two review rounds; scope fixed at 40 cases; stop rules acted on by the owner | 9 |
| Work fragments across worktrees and commits | One phase branch, one topic branch per task, passing candidates pushed | 10 |
| The data-layer seam is hard-coded again | The core emits `DataRequest`s from the first fetch; SQL only behind `SqlDialect`; no backend match outside `sdm-data` | 11 |
| The interim parser drifts from the grammar | The 417-case corpus and 29 quirks in L1; ANTLR as CI reference; one adapter for the swap | 11 |
| Arrow, `pyo3` and `duckdb-rs` majors disagree | One pinned Arrow major (B-D2); `cargo-deny` forbids duplicates; the three bumped together at slice gates | 4 |
| The reference is contaminated by the new code | `python -I`, `require_isolation()` before and after, identity per run, the guard-breach control | 6 |
| The maturin switch breaks existing CI | Wheel artifact for every Python job; `_native.pyi` for mypy; the "old paths never import `_native`" test | 10 |
| The two DuckDB builds differ | Both pinned to the reference's version; the Rust host checks `version()`; `REPEATABLE` sampling compared only within one version | 6 |
| Tie order and draws produce false differences | The comparator's nondeterminism rule and replay; ORD-1 measured first, normalized only after an approved entry | 2 |
| Zone database drift between host and bundled copies | The version is a host fact recorded on every run; tests pin the bundled copy | 2 |
| Panics reach users as ordinary refusals | `catch_unwind` at every entry; `INTERNAL_ERROR` with `fields.panic`; the run is poisoned | 1 |

## Plan changes this phase makes

To be applied to [migration-plan.md](../migration-plan.md) once the owner confirms the decisions.

- Section 3, Dependencies: `jiff` (host zone directory passed in, bundled fallback) replaces `chrono` with `chrono-tz` for dates, times and zones; `chrono` remains only through Arrow and `chrono-tz` is banned (A7).
- Section 3, Dependencies policy: `arrow`, `duckdb` and `pyo3` are updated together, with the Arrow major set by `duckdb-rs` at the reference's DuckDB version (B-D2); `jni` joins when Java starts.
- Section 3, Crates: the interim parser is its own crate, `sdm-pql-interim`, reachable only through `sdm-core`'s adapter, instead of living inside `sdm-core` (A3).
- Section 3, Type set and What lives where: Arrow-to-Arrow normalization (`large_*`, `timestamp[s]` and `[ms]`, `large_list`, `fixed_size_list`) happens once in `sdm-host`, not in each host (B-D6).
- Section 3, Bindings: the native module is `sdm_client._native`, and the client's build backend is maturin on the epic branch (A3, B-D3).
- Section 4: the routing boundary is private and read per call, with an environment variable for tests, and is not public API (B-D7).
- Section 5 and the register: format `sdm-register/2`, with per-case old and new outcomes and decision references to any phase's decisions file (A6).
- Section 6, L0: Python-emulation modules are allowed only in `sdm-pql-interim`, on an allowlist that expires with it (A4).
- Section 6, L2 and decision 5: CI compares with committed reference observations, regenerated locally at each gate (B-D8).
- Section 6, L8: Phase 1 tests the pyarrow 10.0.1 floor in one CI job (A8).
- Section 7, first gate row: during Phase 1, L4 and L5 apply only to what the 1b inventory needs (DuckDB goldens; pandas 2 and 3), the rest from Phase 2 (A5).
- Section 8, Phase 1a row: the work adds `sdm-core` (skeleton), `sdm-pql-interim`, `sdm-host`, `sdm-py`, the pure-Rust test host and the status page generator, and "S0 green in Python" is defined as in A2 (A1, A2, A9).
- Section 9: the status page is generated by `scripts/rust_core/status.py` from Phase 1a on (A9).
