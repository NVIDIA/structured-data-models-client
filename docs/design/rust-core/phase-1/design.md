# Phase 1: the 1a harness and the 1b spike

**Branch:** `rust-core-phase-1`, off `rust-core`, for both halves; it merges locally into `rust-core` only after the owner's go at the end of 1b.
**Topic branches:** `rust-core-phase-1-<task>`, so CI runs on them; at most one per task.
**Tracker:** [tracker.md](tracker.md); **decisions:** [decisions.md](decisions.md).

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
- DuckDB SQL goldens (the L4 part) and the Python conversion table on pandas 2 and 3 (the L5 part), only for what the inventory needs (decision A5).
- The DT-3 and ORD-1 findings, register entries per changed case, the type-set findings and the evidence pack.

### Explicitly out

- Java: `sdm-jni`, a JAR, JDBC conversion tables, Java S0 and any Java runner; the pure-Rust host is the second language in Phase 1.
- Backends other than DuckDB in 1b (frames and S3 reads included), live work (L6, the Databricks job matrix, any remote run), and operations outside the inventory (ranking, forecasting, link prediction, `fit`, sessions, tabular requests, metric and semantic views, YAML).
- Phase 2 items: the provider conformance kit, the non-SQL test provider, the swapped data layer, the dataframe-engine comparison, the remaining S1 cells, and the full L4 and L5 matrices (pandas 1.5 included).
- L3 differential generation, full L7 measurement, the full L8 platform matrix, wheels other than manylinux2014 x86_64, and any claim about the PQL C++ build floor.
- Integrating the PQL team's crate, sending the PQL asks, deleting the interim parser, switching any public path to the core, deleting the old engine, and retiring old tests.
- Random streams the journey does not reach: RS-2 to RS-5, and the Postgres, Snowflake and Databricks sampling SQL.
- Growing the 40-case inventory; a missing case goes to the owner as a scope decision.
- Updating the reference from `main` mid-phase, except at the go merge if the owner asks.

## Architecture choices

### Layout

```text
rust/
  Cargo.toml, Cargo.lock, rust-toolchain.toml, deny.toml, clippy.toml
  testdata/tzdata-2026c/  vendored TZif directory for zone tests
  crates/
    sdm-contract/      types only; schema/ holds the generated JSON Schema and contract.lock
    sdm-core/          the Run state machine, dispatch, capability checks, the PQL adapter
    sdm-pql-interim/   the archive's parser, reachable only through sdm-core's adapter
    sdm-host/          control codecs, the 64 KB cap, Arrow admission, panic containment, run ownership
    sdm-testkit/       dev-only: case loader, pure-Rust host, duckdb-rs, scripted endpoint
    sdm-data/          1b: SQL compilation, provider strategy, the DuckDB dialect
    sdm-schema/        1b: canonical schema and inference over Arrow
    sdm-engine/        1b: time, random streams, sampling, labels
    sdm-wire/          1b: generated wire types
    sdm-request/       1b: materialization, payload, response, result layout
  bindings/sdm-py/     PyO3 cdylib `_native` over sdm-host
contract/              schemas; bases.yaml; compare-policy.yaml; cases/; reference/; baselines/; runners/run_python.py
scripts/rust_core/     compare.py, ratchet.py, register.py, coverage.py, l0_*.py, status.py, gate.sh, selftest.py, harness-paths.txt, emulation-allowlist.txt
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
- Clippy runs with `-D warnings`, each `#[allow]` carries a one-line reason, and every `.rs` and `.toml` file carries the SPDX header the hygiene check enforces.

### Dependencies

- Arrow 58 across the workspace: `duckdb =1.10506.0` (DuckDB 1.5.6, the reference's version) uses Arrow 58, and `arrow-pyarrow` 58 needs `pyo3` 0.28; the archive's Arrow 60 and `pyo3` 0.29 are not used, and the three are bumped together, only at slice gates (decision B-D2).
- `cargo-deny` bans duplicate versions of `arrow-*` crates and `pyo3` only.
- `jiff` is the one calendar and zone library, with the host's zone directory passed in and a bundled fallback; `chrono` stays only as Arrow's transitive dependency and `chrono-tz` is banned (decision A7).
- `serde`, `serde_json`, `schemars` 1.x, `thiserror`, `indexmap`, `stacker`; `duckdb` and `anyhow` only in `sdm-testkit`.

### Contract

- `ColumnType` is the closed type set of plan section 3; `ColumnType::from_arrow` is the only way in, and anything else is refused with `kind: unsupported, code: UNSUPPORTED_TYPE` naming the column and the Arrow type.
- The scope manifests' cell vocabulary is generated from that enum.
- Operations are a versioned, serde-tagged enum; Phase 1 needs `validate_query`, `describe_source`, `materialize` and `predict`.
- `Step` has exactly three variants: `Fetch` (a batch of typed data requests, each with compiled SQL and typed parameters for SQL providers, key sets referenced as Arrow attachments, and the expected Arrow schema or a discovery marker), `CallModel` (body as bytes) and `Done`; no step asks the host for language behavior.
- Data-request and capability types live in `sdm-contract`, the canonical schema in `sdm-schema`, and SQL compilation and provider strategy in `sdm-data`.
- Errors are `{kind, code, message, fields, cause}` with closed kinds; warnings travel on their own channel, in order; no contract string names a Python or Java class, and each host maps kinds and codes through one reviewed table.
- Every scalar in a control message is a tagged object with its value as a JSON string (integers per width, `d128` and `d256` with precision and scale, floats with shortest round-trip text and non-finite values, temporal counts with unit and zone), so nothing passes through a double.
- `HostFacts` (clock, local zone, zone database source, path and version, platform padding rule) is part of every operation.
- State such as `RequestState` is typed and versioned; runs and Arrow buffers have defined owners and release rules.
- The JSON Schema is generated by `schemars`, committed, and locked by hash in `contract.lock`; existing version entries are never rewritten, and the contract version is independent of the package version.

### Key sets

- A key set crosses the core/host boundary as an Arrow attachment referenced by ID from a typed data request; control JSON carries only its descriptor, never the keys (decision Q1).
- The DuckDB provider owns the physical binding (for example, registering the attachment as a relation), and both hosts implement the same semantics.
- Tests prove a key set whose serialized contents exceed 64 KB without raising the control cap, and exact binding of uint64 above int64 max, `decimal128(38, 0)` and scaled decimal keys.

### Core, host and binding

- `sdm-core` has one `Run` type: operation bodies are `async fn`s over an asker handle, polled once per step, with no executor and no threads.
- States are `Ready`, `AwaitingFetch`, `AwaitingModel`, `Finished`, `Cancelled` and `Poisoned`; an illegal call returns a `protocol` error and never panics.
- The composition root picks a built-in SQL provider by dialect or a host provider with declared capabilities; nothing else in the core matches on a backend name.
- `sdm-host` is plain Rust: it owns runs, enforces the 64 KB control cap, wraps every entry in `catch_unwind` (a panic poisons the run and returns `INTERNAL_ERROR` with `fields.panic = true`), and answers a concurrent call with `RUN_BUSY`.
- Arrow-to-Arrow admission (`large_*` to small forms, `timestamp[s]` and `[ms]` to `us`, `large_list` and `fixed_size_list` to `list`) lives once in `sdm-host`; native-to-Arrow tables stay in each host (decision B-D6).
- `sdm-host` builds the zone database from the host fact: the directory the host names, else the bundled copy, per zone, and records which and its version.
- `sdm-py` is a thin PyO3 module over `sdm-host`: it holds the GIL only while touching Python objects, passes Arrow through the PyCapsule interface from pyarrow 14 and `_export_to_c` below it, raises one `_native.CoreError`, and passes model bodies as bytes; it makes no semantic callbacks into Python, while Arrow export and release callbacks are allowed and counted.

### Packaging

- The client's build backend becomes maturin on the epic branch (decision B-D3); `main` is untouched.
- `sdm_client._native` (decision A3) ships in the `structured-data-models-client` wheel, tagged `cp310-abi3`, built as `manylinux2014` x86_64, with third-party notices for every Rust dependency linked in.
- `pyarrow>=10.0.1` becomes a base dependency; `pandas>=1.5` stays.
- A version guard holds the Cargo version of `sdm-py` equal to the other four distributions; CI's `build` job builds four wheels as today and takes the client wheel from `native-wheel`, so "five wheels, one version" still holds.
- Importing `sdm_client` never imports `_native`; only `_rust/` does, lazily, on a core route; a test holds this.
- A missing `_native` or an unknown forced route on a core route raises an error naming the cause, never falls back, and counts as a runner failure.

### Routing seam

- `sdm_client/_routing.py` maps the `scope-python.yaml` path IDs to `old` or `core`; every default is `old`, and a test fails if one changes before the Phase 3 switch.
- `SDM_CORE_ROUTES` forces paths onto the core for tests; it is read at every call, is private, and is deleted in Phase 5 (decision B-D7).
- S0 is the public call that validates PQL today, `materialize()`; under the core route `_resolve` calls `validate_query` and maps a refusal to `StructuredDataError(code='INVALID_REQUEST')`.
- "S0 green in Python" compares validity and `StructuredDataError.code` with the reference, not spans or message text; parser codes and spans are checked in L1 by the 417-query corpus (decision A2).
- An accepted S0 query finishes through the old materializer, and is not evidence for core materialization.
- While S1 is not switched, an accepted query is also checked by the old validator, and a disagreement is `INTERNAL_ERROR`; this shadow check leaves with the old code.

### Interim parser

- `sdm-pql-interim` holds the archive's lexer, parser, visitor, AST, profiles, codes and the seven validators, with their corpus (193 valid, 224 invalid), 29 quirks and custom-profile tests.
- Validity-affecting emulation stays, inside the crate: `pyint`'s 4,300-digit limit and `float_repr` for the text of AST constants (decision A4).
- `pyrepr`'s other helpers, `printable.rs`, `py_repr` and `py_str` are not ported, since message text is not promised; a neutral `Literal` replaces the archive's Python value type, and integer literals keep their digits exactly.
- The 1,695-line `dates.rs` is replaced by a reader written from rules D1 to D9 over `jiff`, taking the clock and padding rule from host facts; D8 (packed dates) is checked by the recorded examples plus bounded generated cases, and any deferred family is a decision.
- Only `sdm-core` has a production dependency on the crate, and only its adapter uses it; the PQL team's crate later goes behind the same function.
- `scripts/rust_core/emulation-allowlist.txt` starts with exactly those two modules; emulation is banned everywhere else, each addition is a decision, and the list expires with the crate.

### Pure-Rust test host

- `sdm-testkit` holds the case loader, the pure-Rust host, the scripted endpoint and the runner binary `sdm-contract-run`; it is never a dependency of a shipped crate.
- It builds each case's data with `duckdb-rs`, checks `SELECT version()` against the recorded one, answers `Fetch` with the driver's Arrow and `CallModel` with the scripted endpoint, with no Python in the process; `REPEATABLE` sampling is compared only within one DuckDB version.

### Runners

- One Python runner, `contract/runners/run_python.py`, run with `python -I`, in three modes: `python-old` (defaults), `python-core` (`SDM_CORE_ROUTES` set for the case's paths) and `python-ref` (the reference environment).
- `python-ref` loads the reference guard by path, calls `require_isolation()` before the first case and after the last, and records the reference commit, the environment hash and the guard result.
- The scripted endpoint is one written rule served over localhost HTTP, reached by Python runners through the client's real transport, implemented in Python and in Rust and held together by a shared vector file.
- DuckDB's `QueryLog` gives the statement text and order on the old path and the reference; core runners also record SQL and parameters from the step trace, and the two must agree.

### Observations and comparator

- An observation (`sdm-observation/1`) is JSON in which every scalar uses the contract's tagged-string encoding; payloads keep their raw bytes and a parsed form with numbers as decimal text; result tables are Arrow IPC files referenced by SHA-256.
- `core` fields: Arrow types, stypes, keys, time column, data requests, statement count and the rows each step returns, payloads, the result as Arrow, error kind, code, fields and cause chain, and warnings by code in order.
- `host` fields: pandas dtypes, Python error and warning classes, `StructuredDataError.code` and the `to_pandas()` result.
- Python runners fill `core` fields from public outputs through one projection in `run_python.py`, and `rust-host` fills them from the run; `rust-host` and `python-core` are compared on `core` fields, and `host` fields only between Python runners.
- An `identity` envelope (runner, commit, `_native` SHA-256, reference commit, contract version, DuckDB version, zone database source and version) is validated separately and excluded from digests.
- `scripts/rust_core/compare.py` prints the first difference per field with its path, under the rules in `contract/compare-policy.yaml`.
- Exact fields keep null apart from NaN, `-0.0` from `0.0`, decimal precision and scale, temporal unit and zone, nested validity, field metadata and duplicate rows; message text never gates, and the inventory's message fragments are reported for information.
- Payloads are compared parsed with lossless numbers, excluding only the named UUID fields; results by Arrow schema and values, with row order only where the operation defines it.
- The core's SQL is gated by the DuckDB goldens; against the reference, the comparator checks statement count, excluding setup and trace statements by execution boundary, and the rows each step returns; `QueryLog` carries no parameters, so reference parameters are marked unavailable, and equal rows are not evidence of equal parameters.
- SQL whitespace is normalized only outside literals, quoted identifiers and comments; numeric text is never normalized; a golden update never replaces a register entry.
- Nondeterministic draws follow `random-streams.md`: membership, cardinality and named invariants instead of row equality, with downstream fields compared by replaying the same captured fetch results; exclusions only for named field paths.
- Rules land with the first task that needs them: outcome, kind and code with the first S0 case, payloads and results with the first journey, SQL and nondeterminism with sampling.

### Reference observations and ratchet

- Reference results are committed under `contract/reference/<reference-commit>.jsonl`, regenerated locally by `gate.sh`, which diffs the result against the committed file, so CI compares without running the reference (decision B-D8).
- `contract/baselines/<runner>.jsonl` holds each case's projected observation and its digest.
- The tag `rust-core-phase-1-green` marks the previous green commit; only `gate.sh` moves it, after that commit's CI run is green and the local gate passes.
- CI reads baselines, reference observations, the comparison policy and register entries from that tag and from the merge base with `rust-core`, never from the branch under test (decision C1).
- A changed digest, a new refusal, a removed case, or a changed case input, reference observation or comparison policy fails unless an approved register entry names that case, runner, and exact old and new outcome.
- A case absent from the green baseline passes as an addition, and is protected from the next acceptance on; `ratchet.py accept` runs only in `gate.sh` on a green commit, writes the baselines in their own commit, and moves the tag.
- The first baseline is written once by `gate.sh` and checked by the reviewer; after that a missing baseline is an error, and at the go merge the green baselines become `rust-core`'s.

### Register v2

- `register.yaml` moves to `sdm-register/2` (decision A6): each entry may carry `case_outcomes`, a list of `{case, runner, old, new, summary}`, and `decision_ref` becomes `{file, row}`.
- `register.py validate` checks the schema, unique and never reused IDs, case IDs that resolve to corpus cases or an allowed legacy form, and the status rules.
- An entry counts only when approved by someone other than its implementer; an entry that refuses an input the reference accepts needs `approved_by: owner`, which only the owner sets.
- Changed contents under an approved ID count as a new change and need approval again; the ratchet compares entry contents with the green tag; no entry can accept a panic.

### Coverage generator

- `scripts/rust_core/coverage.py` implements the rules of [scope-format.md](../phase-0/scope-format.md), checking both manifests structurally but gating only the active language and slice: S0 requires its validation paths and no cells, and Java runtime coverage starts with Java.
- Rule 6 (with all 40 cases): a cell claim names the observed boundary and stage (received Arrow type, admitted type, declared role, inferred role if reached), and a refusal covers its rejected input without claiming a later stage.
- The manifests' `cases` lists are generated (`--write`) and checked (`--check`), never edited by hand.
- `--gate S0` runs in Phase 1; `--gate S1`, which also enforces rule 7, is a Phase 2 obligation.

### Harness bounds

- `scripts/rust_core/harness-paths.txt` lists every file allowed under `scripts/rust_core/`, `contract/runners/` and `rust/crates/sdm-testkit/`; each line cites this design or an owner-confirmed decisions row, and an L0 check fails CI on any other file there.
- The status page shows harness lines (those paths) against product lines (the other crates, `sdm-py`, `_rust/` and `_routing.py`) at each green commit.
- Stop rule: when harness lines exceed product lines, or harness grows faster than product over two green commits in a row, harness work stops until the owner reviews.

### CI

| Job | Runs |
|---|---|
| `rust-lint` | `cargo fmt --check`, clippy with `-D warnings`, the L0 scripts, the harness-path check, the contract schema regeneration and lock check |
| `rust-deny` | `cargo deny check` |
| `rust-test` | `cargo test --workspace` (L1), then `sdm-contract-run` over the L2 corpus against the reference observations and the ratchet |
| `native-wheel` | maturin manylinux2014 x86_64 abi3 wheel, `auditwheel show`, upload |
| `wheel-smoke` | A clean `python:3.10-slim` with `pyarrow==10.0.1`, `pandas==1.5.3` and `numpy<2` (decision A8), and a clean `python:3.14-slim` with the newest pyarrow: install the wheel only, import, the admission round trip, one S0 validation, the missing-library control |
| `contract-python` | `python-old` and `python-core` over the L2 corpus against the reference observations and the ratchet; register validation; coverage `--check` |
| Existing jobs | `lint`, `hygiene`, `typecheck`, `test`, `engine`, `build`, `installed-wheel`, `sbom`, now installing the client from the `native-wheel` artifact |

- Rust jobs use the pinned toolchain and a cache keyed on `Cargo.lock`; the hygiene check's shipped suffixes cover `rust/`; `publish.yml` does not change; the fast tier's run time is optimized, never a reason to drop cases.
- The local gate (`scripts/rust_core/gate.sh`) runs `python-ref` with the guard, the full L2 on all four runners, comparator and ratchet, coverage `--gate S0`, the seeded-fault suite and the wheel smoke in local Docker.
- `scripts/rust_core/status.py` builds one markdown status page from git, CI, decisions.md and the register: candidate commit, gates passed, open failures, decisions needed, open branches, the fast tier's warm-cache run time, and harness against product lines (decision A9).

## Tasks

Tasks are numbered in execution order; the half column keeps the plan's 1a and 1b exits, and independent tasks may run in parallel on topic branches.
"Green" means in CI on `rust-core-phase-1` unless it says local.
Each readiness item (sections 1.1 to 1.14 of the planning input `A-readiness-audit.md`) maps to a task below; the items it marks later are the Databricks percentage `repr` (Phase 3), scope rule 7 (Phase 2), column-wise object columns (S3) and RS-5 (tabular).

| ID | Half | Task | Output | Exit check |
|---|---|---|---|---|
| T1.1 | 1a | Phase documents and locked checks | This file, [tracker.md](tracker.md), [decisions.md](decisions.md) | The owner has confirmed or changed every orchestrator decision, including the no-go criteria (C9); the locked checks are locked |
| T1.2 | 1a | Workspace scaffold | `rust/` tree, toolchain, lints, `clippy.toml`, `deny.toml` from the archive's plus the bans above, empty 1a crates, the Arrow 58 pins, `dependencies.md`, `harness-paths.txt` and its check; root `uv.lock` resolved | `cargo build`, clippy and `cargo deny check` pass in CI; the hygiene check passes; an unlisted harness file fails its `selftest.py` fixture |
| T1.3 | 1a | Zone and calendar vectors | The T0.6 corpus as an L1 vector file run against `jiff`; `rust/testdata/tzdata-2026c/` | All 2,808 zone answers and 208 offset answers agree against the vendored 2026c directory and against the bundled database, each with expectations tied to its version; host-directory loading (named directory, per-zone fallback, version recorded) is tested on fixture directories, never the runner's system database; parsing differences from `chrono` are listed in `dependencies.md` |
| T1.4 | 1a | `sdm-contract` types, schema and lock | The types S0 and the journey need, the generated schema, `contract.lock`, the lock check, a property-based decoder fuzz test | Round trips pass for `i64::MIN`, `u64::MAX`, uint64 above int64 max, `decimal128(38, 0)`, `decimal256` at precision 76, `-0.0`, NaN and both int64 timestamp ends; a Python test decodes them exactly with only `json` and `decimal`; the regeneration diff is empty; a schema change without a version bump fails, and so does a rewritten existing version entry; the fuzz test runs bounded with no panic |
| T1.5 | 1a | `sdm-core` Run state machine | `Run`, dispatch, the composition root, capability checks and profiles | Every state and illegal transition is tested, including stale and duplicate reply IDs and cancellation while awaiting a fetch; a scripted operation runs to `Done` |
| T1.6 | 1a | `sdm-pql-interim` and the adapter | The ported parser behind the adapter, the D1 to D9 reader, `emulation-allowlist.txt`, the validated-query schema and lock entry, a PQL fuzz test, the ANTLR grammar as CI reference | All 417 corpus queries give their recorded codes and spans; all 29 quirks hold; all 45 T0.6 date literals and the generated packed-date cases agree; the fuzz test runs bounded with no panic |
| T1.7 | 1a | `sdm-host` | Codecs, the 64 KB cap, the admission table, panic containment, busy handling, zone database construction | A passing test per rule and per refusal, including a contained panic (test-only operation), an overflowing cast that returns a named error, and a generated refusal for a bounded representative of every Arrow type family outside the set |
| T1.8 | 1a | `sdm-py`, the client wheel and CI rewiring | `_native`, `_native.pyi`, maturin backend, `pyarrow>=10.0.1`, version guard, third-party notices; Python jobs and `build` take the client from `native-wheel` | `native-wheel` builds a `cp310-abi3` wheel that `auditwheel show` reports as manylinux2014 or older; every existing job is green with the client from the wheel, and "five wheels, one version" passes |
| T1.9 | 1a | Python host and routing seam | `_rust/`, `_routing.py`, `SDM_CORE_ROUTES`, the S0 core route in `_resolve` | A refused and an accepted query give the expected outcome under the core route; "old paths never import `_native`", "all defaults are old" and "unknown route fails" pass |
| T1.10 | 1a | First S0 case on every runner | `case.schema.json`, `observation.schema.json`, `inventory.schema.json`, `bases.yaml`, the inventory loader; `run_python.py` in three modes, with the reference guard's name list including `sdm_client._native` (updated on `rust-core-reference` first if needed); `sdm-testkit` and `sdm-contract-run`; `compare.py` with the outcome, kind and code rules; the case's reference observation | One accepted and one refused S0 case agree on `python-old`, `python-core` and `rust-host` with their reference observation; the inventory validates and the loader's lossless-mapping test passes; `python-ref` stops when the guard fails; `cargo tree -e normal` for the testkit shows no `pyo3` |
| T1.11 | 1b | First DuckDB journey: `base.regression` | Thin `sdm-data`, `sdm-schema`, `sdm-engine`, `sdm-wire` and `sdm-request`; the DuckDB dialect ported without the archive's capture scaffolding, provider errors with SQLSTATE and vendor codes, and the DuckDB provider in both hosts with key sets as attachments; `materialize` and `predict` routes; the localhost scripted endpoint in Python and Rust with its vector file; the comparator's payload and result rules | `base.regression` runs on all four runners and every difference is in the comparator report; the large key set and exact key-binding tests pass; an import tracker shows the Python provider imports no pandas |
| T1.12 | 1a | S0 case corpus | Accept cases per task kind and profile, a refusal per validator family, date-literal and capability refusals | Every case file validates in CI |
| T1.13 | 1a | S0 reference observations | `contract/reference/<commit>.jsonl`, recorded by `gate.sh` after the guard update, and its immutability check | The file carries the guard result and full identity, the comparator reads it, and an edited existing entry fails CI |
| T1.14 | 1a | Ratchet, register v2 and bootstrap baseline | `ratchet.py`, the green tag, `compare-policy.yaml`, `register.schema.json` for `sdm-register/2` with today's entries migrated, `register.py validate`, the first baseline | Seeded checks: a changed outcome fails and passes once an approved entry states its exact old and new outcome; a proposed, rejected or unrelated entry does not pass it; changed contents under an approved ID fail; an accepted case whose answer then changes fails; a relaxed comparison policy fails; a branch-side baseline edit has no effect; an identity-only change passes with no entries; `validate` passes in CI |
| T1.15 | 1a | Coverage generator for S0 | `coverage.py` with rules 1 to 5 and 8, `--check`, `--gate S0` | `--check` and `--gate S0` pass; an unresolved ID and a dropped path each fail |
| T1.16 | 1a | L0 guards | `l0_deps.py` (graph, one-way flow, the interim-parser dependency rule, nothing shipped on `sdm-testkit`, no `pyo3`, `jni`, `arrow-pyarrow`, `numpy` or `duckdb` in domain crates), the `unsafe` check, the vocabulary scan over product code with the emulation allowlist (fixtures, provenance and comments excluded), the step-variant test, clippy bans (`HashMap` and `HashSet` in domain crates, clocks, environment, file system, `jiff::tz::db`) | Each guard fails on its seeded fixture in `selftest.py` and passes on the branch |
| T1.17 | 1a | Clean-container wheel smoke | `wheel-smoke` job | Passes on 3.10 with pyarrow 10.0.1, pandas 1.5.3 and `numpy<2`, with the resolved environment recorded, and on 3.14 with the newest pyarrow; canonical types round-trip unchanged, admitted types match the expected canonical schema with values and null masks kept; outputs stay valid after input collection, error and run destruction, over both the capsule and `_export_to_c` paths |
| T1.18 | 1a | L2 in CI and the status page | `rust-test` and `contract-python` running the L2 corpus, comparator and ratchet; `status.py` | Both jobs are green; the status page regenerates with every item listed under CI |
| T1.19 | 1a | Local gate and seeded faults | `gate.sh`, the rest of `selftest.py` | A corrupted expectation, a missing case, a skipped call, a missing native library, a guard breach and a ratchet bypass each fail locally, the first, second, third and sixth also in CI; `gate.sh` passes |
| T1.20 | 1a | Phase 1a review | Findings and resolutions in the tracker | Locked checks 1 to 3 hold, and L0, the ratchet and the fast tier are green |
| T1.21 | 1b | DT-3 and ORD-1 findings | Answers in the register entries, from the journey's observations, before any baseline must match | DT-3 says whether the `Timestamp.min` sentinel survives the sampler's round trip; ORD-1 says whether tie order reaches an inventory payload, with any proposed change |
| T1.22 | 1b | Random streams | RS-1 (`SeedSequence`, PCG64, bounded draws, the shuffle) and the RS-7 refusals; other stream modules only if a reached call needs them | Algorithm vectors from numpy pass for every seed in the T0.6 model, including `2**70`; separately, DuckDB seeds from `2**31` are refused with `INVALID_REQUEST` |
| T1.23 | 1b | `sdm-schema` and `describe_source` | Inference over Arrow; catalog SQL in `sdm-data`; the SQL-fragment scan of `sdm-schema`, `sdm-engine` and `sdm-request` | Every request kind renders and runs on the `shop` and `declared` bases through `sdm-testkit` with the expected Arrow schema; the same fetch gives the same admitted Arrow on both hosts for every base; `schema` observations match the reference on the three baseline cases; the scan fails on its fixture |
| T1.24 | 1b | `Schema.from_source` core route | The schema bridge building today's `Schema` objects | The public `Schema` under the core route equals the old one on the baseline cases; a `Schema` pickled by the reference loads, and a core-built one pickles and reloads |
| T1.25 | 1b | Time module on `jiff` | DT-7 value-by-value reading on SQL sources, inference over the preview, the shapes of `date-rules.md`, offsets; session zone from host facts | The T0.6 column cases and the inventory's time cases infer the reference's time columns |
| T1.26 | 1b | Sampling | `sample_target` call order, default seed 42, SQL steps as batched data requests, RS-6 reservoir SQL; the comparator's SQL and nondeterminism rules | Fetch traces of the baseline cases match the reference in statement count and the rows each step returns; a whitespace change inside a SQL literal fails the comparator |
| T1.27 | 1b | Labels | Kernels with float32 Kahan sums in row order, `DateOffset` rules on `jiff` | Labels in the baseline payloads match exactly |
| T1.28 | 1b | Wire and request | `sdm-wire` from the pinned OpenAPI file; payload, response and result layout in `sdm-request`, without the `json.dumps` size count or CPython sort emulation | Parsed payloads of the baseline cases match; L1 layout vectors with distinct per-row answers and reordered responses pass |
| T1.29 | 1b | Three baseline cases | The routes generalized past `base.regression` | `base.regression`, `base.binary` and `base.multiclass` match the reference on all four runners, before any other case is worked on |
| T1.30 | 1b | DuckDB SQL goldens (L4 part) | Reviewed golden SQL for every statement the inventory emits, executed on DuckDB in CI | A seeded SQL change without a golden update and an entry fails |
| T1.31 | 1b | Python conversion table (L5 part) | The pandas-to-Arrow table for the inventory's types | The table's expected values pass in CI on pandas 2 and on pandas 3 |
| T1.32 | 1b | All 40 cases | Observations on all four runners; coverage rule 6 and `--write` | `python-core` and `rust-host` agree on `core` fields for all 40; every difference from the reference is in the report; statement counts equal the reference's or are registered; a contradicted cell claim fails |
| T1.33 | 1b | Register entries | One entry per changed case with its case outcomes; the owner's list of new refusals | P1B-1 is replaced case by case; RS-6, DT-7 and DT-4 are reviewed; the ratchet passes |
| T1.34 | 1b | Type-set findings | Each received Arrow type accepted, admitted or refused by name; any proposed change to the set | Every type the inventory produced is classified, and any change is proposed to the owner |
| T1.35 | 1b | Evidence pack | `phase-1/evidence/` | Every item below is present and generated from the branch's own runs |
| T1.36 | 1b | Phase 1b review, decision and merge | Review findings; the owner's go or no-go in decisions.md | Locked checks 4 to 6 hold; the owner's decision is recorded; on go, the local merge into `rust-core` |

## Phase 1b go or no-go evidence pack

Written to `phase-1/evidence/`, generated from the branch's own runs wherever it can be.

- Identity: phase branch commit, `rust-core` base, reference commit and baseline tag, environment versions, the DuckDB version on both hosts, zone database sources and versions, the wheel's SHA-256.
- A 40-row table: the reference outcome, `python-core`, `python-old`, `rust-host`, and the archive (commit `3887762`, run by the same runner where it builds), match or difference, and the register entry for each difference.
- Per hard seam, the deepest stage each case reached (driver Arrow, admission, inferred schema, payload, result, or refusal); stages not reached are marked unproven.
- The comparator's field-level report for every difference; the register entries proposed or approved, with the cases that need the owner's call; the type-set findings; the DT-1, DT-2, DT-3 and ORD-1 results.
- Boundary evidence: the L0 report, `cargo tree` for the pure-Rust host, and the count of semantic callbacks into Python (zero).
- Size: Rust lines per crate split into ported, new and glue, against the Python each replaces; harness lines against product lines; the dependency table; the emulation allowlist as it stands.
- Offline measurements, for information: statements per case against the reference, CPU per case old against new, wheel size and import time.
- CI run links, the seeded-fault results, the review's findings and resolutions, and an explicit list of what was not done.
- The no-go criteria (C9) with each one's result, and the executor's recommendation, labelled as such; the decision is the owner's.

## Gates

### Merge rules

- Every candidate that passes its checks is pushed before more work builds on it.
- A change lands on the phase branch when L0, L1, the L2 subset (Python and the pure-Rust host) and the ratchet are green in CI, and, from T1.30 and T1.31 on, the DuckDB goldens and the pandas 2 and 3 conversion table (decision A5).
- A change to bindings or native dependencies also passes `wheel-smoke`.
- A type or conversion fix lands only with a public case that failed first.
- Phase 1a does not merge on its own; 1a and 1b merge into `rust-core` together, locally, after the owner's go.
- S0 has no separate slice-done gate in Phase 1: its L3 run and the update from `main` come before any S0 path switches in Phase 3.

### Locked acceptance checks

These are locked when the owner confirms T1.1; a third change to them goes to the owner.

1. 1a exit: S0 green on `python-old`, `python-core` and `rust-host` against the committed reference observations (validity and `StructuredDataError.code` in Python, kind and code on the core), the L1 corpus green on codes and spans, and coverage `--gate S0` passing.
2. 1a exit: the six seeded faults of T1.19 and the ratchet's seeded checks of T1.14 each fail.
3. 1a exit: the wheel installs and loads in both clean containers of T1.17, including the pyarrow 10.0.1 floor.
4. 1b exit: on all 40 inventory cases, `python-core` and `rust-host` agree on `core` fields, and each Python runner matches the reference on inferred types, keys, time columns, data requests (statement count and rows per step), payloads, results, warnings and refusals, or differs only where an approved register entry states that case's exact old and new outcome; the core's SQL matches the goldens.
5. 1b exit: no case refuses an input the reference accepted without an owner-approved entry, and no case panics.
6. Both halves: the L0 guards, the ratchet and CI's fast tier are green.

### Process rules

- At most two review rounds per change and per half; the reviewer is a fresh agent that checks code and behavior, not prose; a third round goes to the owner.
- New harness files need the owner's approval, enforced by `harness-paths.txt`; the harness stop rule is acted on by the owner.
- Stop rules: two fix rounds on the same divergence go to the owner; a check failing twice the same way changes method; a phase that keeps finding new work stops for a re-plan; while a blocking question waits for the owner, work narrows to the current task.
- An agent never sets `approved_by: owner` or approves its own entry.
- Reproduce a failure from the installed package before any remote run; Python gates need the Python and pure-Rust checks only.

## Risks

| Risk | Guard | Learnings lesson |
|---|---|---|
| The harness grows before anything runs end to end | One S0 case and `base.regression` run before the tooling generalizes; the harness-path list; harness against product lines with a stop rule | 5, 9 |
| The Python value model returns, through the parser or through pandas dtype modelling | Two named emulation modules, confined to one crate; `Literal` replaces the Python value type; dtype names come from a host table; the vocabulary scan | 1, 3 |
| Bit-for-bit matching returns on SQL text and messages | Message text never gates; core SQL is gated by goldens and the reference by statement count and rows; each changed case gets its own entry | 2 |
| A fix breaks neighboring cases, or an expectation is edited to pass | The ratchet against the green tag and merge base; reference observations immutable; approved entries with exact outcomes | 7 |
| A Python assumption hides in the contract | The pure-Rust host runs every case; no callback step; key sets as Arrow; the L0 graph guard | 5, 11 |
| The reference is contaminated by the new code | `python -I`, `require_isolation()` before and after, identity per run, the guard-breach control | 6 |
