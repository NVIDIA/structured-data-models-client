# Phase 1 dependencies

One row per dependency of the Rust workspace in `rust/`: whether the core uses a library, ports a defined algorithm, or keeps the concern in the host, and the measured reason.
Versions are pinned by `rust/Cargo.lock`; `rust/deny.toml` holds the licence, advisory, source and ban rules.
A row changes with the dependency, in the same commit.

| Dependency | Version | Crates | Choice | Reason |
|---|---|---|---|---|
| `arrow-*` (`arrow-schema`, `arrow-buffer`, `arrow-array`, `arrow-data`, `arrow-ipc`) | 58 | contract, host, later domain crates | Use a library | The closed type set is Arrow's (plan section 3); one major across the workspace, because `duckdb =1.10506.0` and `arrow-pyarrow` 58 both need 58 and two majors in one process need FFI glue between them (decision B-D2); `cargo-deny` refuses a second version of any `arrow-*` crate |
| `pyo3` | 0.28 (`abi3-py310`) | `sdm-py` only | Use a library | The binding tool the plan names; 0.28 because `arrow-pyarrow` 58 needs it (B-D2); RUSTSEC-2026-0176 and RUSTSEC-2026-0177 are ignored with reasons until the next Arrow bump brings 0.29 (decision E1) |
| `arrow-pyarrow` | 58 | `sdm-py` only | Use a library | PyCapsule from pyarrow 14 and `_export_to_c` below it, without our own FFI code |
| `duckdb` (duckdb-rs, `bundled`) | `=1.10506.0` (DuckDB 1.5.6) | `sdm-testkit` only | Use a library | The pure-Rust test host runs the core's DuckDB SQL at the reference's DuckDB version; `cargo-deny` refuses it anywhere else |
| `jiff` (`std`, `tzdb-zoneinfo`, `tzdb-bundle-always`; no `tz-system`) | 0.2.38 | host, later engine and the interim parser | Use a library | The one calendar and zone library (decision A7): it meets DT-1 and DT-2 on all 2,808 zone answers and all 208 `DateOffset` answers of the T0.6 corpus, with a host directory and with its bundled copy; `tz-system` stays off so nothing reads the environment |
| `jiff-tzdb` | 0.1.9 | host | Use a library | The bundled zone database behind `jiff`; its version (`jiff_tzdb::VERSION`) is recorded as the bundled source's version |
| `chrono` | 0.4 (transitive) | none directly | Not used | Only Arrow's own dependency; first-party code uses `jiff` (A7) |
| `chrono-tz` | none | none | Banned | Expands rules only to 2099 and embeds its database at build time, so it fails DT-1 and DT-2 (T0.6); `cargo-deny` refuses it |
| `serde`, `serde_json` (`float_roundtrip`, `preserve_order`) | 1 | contract, host, later all | Use a library | Control messages are versioned JSON (plan section 3) |
| `schemars` | 1 | contract | Use a library | Generates the committed JSON Schema from the serde types, so the schema cannot drift from the code |
| `thiserror` | 2 | first-party crates | Use a library | Typed errors; `anyhow` stays out of first-party crates other than `sdm-testkit` |
| `indexmap` | 2 | where output order matters | Use a library | Insertion order for anything that reaches an output; `HashMap` iteration is banned there |
| `stacker` | 0.1 | interim parser | Use a library | Deep queries, as the archive's parser handles them |
| `anyhow` | 1 | `sdm-testkit` only | Use a library | Test-host plumbing only; `cargo-deny` refuses it elsewhere |
