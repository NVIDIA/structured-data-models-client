# The Rust core

The Rust core of sdm-client, which nothing calls yet.
It is ported from the Python packages one part of the pipeline at a time, and every language will reach it through one of the bindings here.

- `crates/` holds one crate per part of the pipeline.
  `wire` has the API's wire types, `core` the run loop a binding drives, and `pql` the predictive query language.
  The other crates are empty until their logic is ported.
- `bindings/python` is the Python binding, the internal `sdm-core` wheel with its one module, `sdm_core._native`.

`rust-toolchain.toml` pins the compiler, and `deny.toml` the licences a dependency may have.
Run everything below from this directory unless it says otherwise.

## Checks

These are the commands the `rust-lint`, `rust-test` and `rust-deny` CI jobs run:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo deny check licenses bans
```

The binding builds only as an extension module, so `cargo test` has no tests for it.
Its tests are in `bindings/python/tests`, and run on the installed wheel.

## PQL

`crates/pql` parses and validates a predictive query as Python's `predictive_query.validate` does, and answers in the JSON the PQL corpus records (`tests/parity/pql/`).
`DATES.md` states how it reads date literals, as pandas reads them.
`UNREACHABLE_CODES.md` lists the diagnostic codes no corpus case can produce, and how the ones a custom profile or validator option reaches are tested instead.
Its acceptance tests compare it with today's Python on the corpus, on generated queries and on generated dates.
The Python ones need an environment with the repository's packages installed, as CI's test job has, and `maturin`; they build the `sdm-core` wheel from this checkout and install it there:

```sh
cargo test -p sdm-pql
cd .. && python -m pytest tests/rust_core/pql/accept_*.py
```

Three tables the crate's tests read are recorded from today's Python, and each recorder takes `--check` to confirm its table is still what Python gives:

- `crates/pql/tests/accept_quirks.json`, by `tests/rust_core/pql/quirks.py`;
- `crates/pql/tests/custom_profiles.json`, by `crates/pql/tests/custom_profiles.py`;
- `crates/pql/tests/padded_years.tsv`, by `crates/pql/tests/padded_years.py`, which also needs Docker.

## The wire types

`crates/wire/src/generated.rs` is generated from the API contract, `nim-sd.openapi.yaml` in the `structured-data-api` repository.
Don't edit it; regenerate it.
`PROVENANCE.json` beside it records the contract's revision and bytes, and the hashes of the generated file and of the generator.

The Rust types and sdm-api's Python bindings must come from the same contract revision.
When the contract moves, regenerate both from one checkout, from the repository root:

```sh
python scripts/generate_sdm_api.py --spec ../structured-data-api/nim-sd.openapi.yaml \
    --output packages/structured-data-models-api/src/sdm_api/generated/tfm_api.py
python scripts/generate_wire_rs.py --spec ../structured-data-api/nim-sd.openapi.yaml
```

The `rust-wire` CI job fetches the contract at the revision sdm-api records and runs the second command with `--check`, so it fails if the Rust types were left behind.
Changing the generator itself changes its hash in `PROVENANCE.json`, so rerun it after any edit, formatting included.

`crates/wire/tests/edge/` holds a contract made of every shape the generator models, and the types generated from it.
`edge.rs` tests how those types behave, and `test_generate_wire_rs.py` tests the generator.
After changing the generator, regenerate the fixture from the repository root:

```sh
python scripts/generate_wire_rs.py --spec rust/crates/wire/tests/edge/contract.json \
    --output rust/crates/wire/tests/edge/generated.rs --no-record
python -m pytest rust/crates/wire/tests/test_generate_wire_rs.py
```

## The Python wheel

A wheel for this machine, with [maturin](https://www.maturin.rs), from the repository root:

```sh
maturin build --release -m rust/bindings/python/Cargo.toml -o dist
pip install dist/sdm_core-*.whl
python tests/rust_core/smoke.py
python -m pytest rust/bindings/python/tests
```

The Linux wheels, `manylinux_2_28` for x86_64 and aarch64, are built with [cibuildwheel](https://cibuildwheel.pypa.io) in Docker, again from the repository root:

```sh
cibuildwheel --platform linux rust/bindings/python --output-dir dist
```

Its configuration is in `bindings/python/pyproject.toml`.
Each wheel passes the smoke test in the container that built it, and the binding's tests there with the newest pyarrow and with pyarrow 8.0.0.
On a machine of the other architecture, Docker needs QEMU to build the second wheel.
