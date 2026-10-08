# Phase 4 acceptance contract

The acceptance checks in this directory test behavior through a small, fixed interface.
This file is that interface.
Everything it doesn't name is the implementer's choice.

## Workspace

- `rust/Cargo.toml` is the root of one Cargo workspace, and `rust/Cargo.lock` is committed.
- The checks find crates with `cargo metadata --no-deps`, run in `rust/`.
  The crate for `wire`, `pql`, `schema`, `dialects`, `sampling`, `labels`, `request` or `core` is the one workspace package named `X`, `sdm-X` or `sdm_X`, with a `lib` target.
  Since a library named `core` would shadow Rust's own `core`, `sdm-core` is the expected name for the run loop's crate.
- `pql`, `schema`, `dialects`, `sampling`, `labels` and `request` are empty: no dependencies, one `.rs` file, and in it exactly one `//!` line, besides `//` comments and blank lines.
- The binding is the workspace package whose manifest is `rust/bindings/python/Cargo.toml`.
  Its library's crate types include `cdylib`, and it depends on `pyo3`.
  `rust/bindings/python/pyproject.toml` builds it with maturin, as the distribution `sdm-core`.
- `rust/rust-toolchain.toml` sets `[toolchain] channel` to a release number, such as `1.97.0`, never `stable`, `beta` or `nightly`.
- `rust/deny.toml` allows only licenses in the Apache-2.0-compatible set named in `accept_deny.py`, in `[licenses] allow` and in any `exceptions`.

## The `wire` generator

- One generated Rust file, inside the `wire` crate, holds the types.
  Beside it, a `PROVENANCE.json` records it as `packages/sdm-api/src/sdm_api/generated/PROVENANCE.json` records `tfm_api.py`:
  - `generated_file`: the file's name, relative to the record's directory;
  - `generated_sha256`: the SHA-256 of its bytes;
  - `generator`: the repository-relative path of the generator script;
  - `generator_sha256`: the SHA-256 of the generator's bytes;
  - `source_path`, `source_repository`, `source_revision` and `source_sha256`: equal to `sdm-api`'s.
- The generator is run as `python GENERATOR --spec SPEC`, from the repository root:
  - with `--check`, it writes nothing, and exits 0 only if the committed Rust file and its record are what it would write;
  - without it, it writes the Rust file and its record.
  The checks take `SPEC` from a local clone of the contract repository at the pinned revision, so the generator can read the revision there as `generate_sdm_api.py` does.
  The clone is made from `~/code/structured-data-api`, or `$SDM_CONTRACT_REPO` when it's set.
- Every schema under `components.schemas` becomes a Rust item of the same name: `pub struct`, `pub enum` or `pub type`.
  The checks add an object schema, `AcceptanceProbe` with the property `probe_field`, and a `PredictionRequest` property `acceptance_probe` referring to it, to a copy of the contract; the generator must then emit `pub struct AcceptanceProbe`, and both property names.
- The `wire` crate's own tests (`cargo test -p <wire>`) parse and re-serialize every payload in `tests/golden/wire_payloads.json` and every payload in every `tests/parity/transcripts/**/transcript.json`, and fail when one doesn't come back parsed-equal.
  The checks prove it by adding an unexpected top-level key to the last golden payload, and separately to the last payload of the last transcript, in a copy of the repository, and requiring the crate's tests to fail.

## The Python binding

The wheel `sdm-core` holds the extension module `sdm_core._native`, built against the stable ABI (`abi3`) for Python 3.10 and later.

- `version()` returns the installed distribution's version, as `importlib.metadata.version('sdm-core')` gives it.
- `roundtrip_arrow(data)`:
  - takes a `pyarrow.Table` and returns a `pyarrow.Table`, or takes a `pyarrow.RecordBatchReader` and returns a `pyarrow.RecordBatchReader`;
  - the result has the same schema, metadata included, and the same batches (a table's chunks) in the same order;
  - every buffer is the same memory: each array's `buffers()` have the same addresses, and so do a dictionary's;
  - it works with pyarrow 8, which has no PyCapsule interface;
  - anything else raises an `Exception` (not a Rust panic).
- `roundtrip_payload(json_str)` parses `json_str` as a `PredictionRequest`, the body of `POST /v1/predictions`, through the `wire` types, and returns it serialized as a JSON `str`:
  - the result parses equal to the input, with every JSON value of the same JSON type: an integer stays an integer, a float a float, and an absent key absent;
  - an object the contract leaves open (`additionalProperties: true`) keeps its extra keys;
  - text that isn't a `PredictionRequest` raises `ValueError`: invalid JSON, a missing required field, a key the contract doesn't allow where it allows none, a value outside an `enum`, an unknown table `format`, or a value of the wrong JSON type.

## The run loop

### In Rust

The run loop's crate exports, at its root, `Run`, `Step`, `Query`, `Outcome` and `Diagnostic`, shaped as the brief says; the checks compile a crate that uses them so:

- `Step<T>` has the variants `RunQueries(Vec<Query>)` and `Done(T)`;
- `Query` has the fields `id`, `sql`, `params` and `columns`;
- `Outcome` has the variants `Data` and `Error { sqlstate, vendor_code, message }`;
- `Diagnostic` has the fields `code` and `message`.

Other variants and fields may be added.
The Rust replay host's tests have `replay` in their names: `cargo test -p <core> replay` runs at least one test, and they pass.

### In Python

`sdm_core._native` exposes the toy run, "count the rows of table T":

- `count_rows(table)` returns a run for the table named `table`.
- `run.next()` returns a step with a `kind` attribute:
  - `'run_queries'`: `step.queries` is a list of queries;
  - `'done'`: `step.value` is the run's result, the number of rows as an `int`.
- A query has the attributes:
  - `id`, which `submit` takes back;
  - `sql`, a `str`;
  - `params`, a list of the values its placeholders take, possibly empty;
  - `columns`, a list of the names, in order, of the columns the run reads from the query's result.
- `run.submit(id, outcome)` gives the run the outcome of the query `id`.
  `Outcome.data(table)` is a result, a `pyarrow.Table`, and `Outcome.error(sqlstate, vendor_code, message)` is a failure, with `sqlstate` a `str` or `None`, `vendor_code` an `int` or `None` and `message` a `str`; `Outcome` is in `sdm_core._native`.
- The toy run asks for exactly one query, and its result is read only from the outcome submitted.
- `submit` with an id the run didn't ask for raises an `Exception`.
- After an `Outcome.error`, `run.next()` raises an `Exception` whose text holds the submitted message.
- A result the run can't read, such as one with no rows or without its column, makes `run.submit` or the `run.next()` after it raise an `Exception`.
  No call ever ends in a Rust panic.

The Python host runs each query with `source._run(query.sql, query.params or None)` on a `sdm_connectors.SQLiteSource`, the Phase 3 seam (DV13).

## The smoke test

`smoke.py` is the smoke test every built wheel must pass: on macOS in the phase venv, in each `manylinux_2_28` build container, and on Python 3.10 with pyarrow 8.0.0.
It needs nothing but `pyarrow` and the installed wheel, exits non-zero on any failure, and on success prints one line starting `SDM_CORE_SMOKE_OK`, carrying the SHA-256 of its own bytes, the machine, and the Python and pyarrow versions.
The checks match that hash against `smoke.py`, so the line proves this smoke test ran.

## Linux wheels

- The checks run `cibuildwheel --platform linux rust/bindings/python --output-dir OUT` from the root of a copy of the repository, with `CIBW_ARCHS_LINUX="x86_64 aarch64"`, and the cibuildwheel configuration in `rust/bindings/python/pyproject.toml`.
- That configuration's `test-command` runs `smoke.py`, for example `python {project}/tests/rust_core/smoke.py`, with pyarrow in `test-requires`.
- The output is one `sdm_core` wheel per architecture, tagged `cp310-abi3`, whose platform tags are `manylinux` tags for glibc 2.28 or older, and no other wheel, such as a `musllinux` one.
- The checks keep the wheels of a clean commit in `$TMPDIR/sdm-rust-core-p4/<sha>/`, or `$SDM_RC_P4_CACHE`, for P4.linux and P4.pyarrow8 to share.

## Evidence for P4.databricks

`phases/phase-4/evidence/manual/databricks-matrix.md` in the harness, as `checks.tsv` describes.
