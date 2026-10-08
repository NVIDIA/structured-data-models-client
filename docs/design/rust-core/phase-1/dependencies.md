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
| `jiff-tzdb` | 0.1.9 (tzdata 2026e) | host | Use a library | The bundled zone database behind `jiff`, the DT-2 fallback; its version (`jiff_tzdb::VERSION`) is recorded as the bundled source's version, and the zone vectors are checked against it |
| `regex` | 1 | `sdm-host` tests only | Use a library | Repeats the T0.6 probe's reading of the accepted shapes in the zone and calendar vectors |
| `chrono` | 0.4 (transitive) | none directly | Not used | Only Arrow's own dependency; first-party code uses `jiff` (A7) |
| `chrono-tz` | none | none | Banned | Expands rules only to 2099 and embeds its database at build time, so it fails DT-1 and DT-2 (T0.6); `cargo-deny` refuses it |
| `serde`, `serde_json` (`float_roundtrip`, `preserve_order`) | 1 | contract, host, later all | Use a library | Control messages are versioned JSON (plan section 3) |
| `schemars` | 1 | contract | Use a library | Generates the committed JSON Schema from the serde types, so the schema cannot drift from the code |
| `thiserror` | 2 | first-party crates | Use a library | Typed errors; `anyhow` stays out of first-party crates other than `sdm-testkit` |
| `indexmap` | 2 | where output order matters | Use a library | Insertion order for anything that reaches an output; `HashMap` iteration is banned there |
| `stacker` | 0.1 | interim parser | Use a library | Deep queries, as the archive's parser handles them |
| `anyhow` | 1 | `sdm-testkit` only | Use a library | Test-host plumbing only; `cargo-deny` refuses it elsewhere |
| `base64` | 0.22 | contract | Use a library | Binary scalars travel as standard base64 text |
| `proptest` | 1 (`std` only) | contract tests | Use a library | The bounded, property-based fuzz test of the contract's decoders |
| `sha2` | 0.10 | contract tests | Use a library | Hashes the committed schema for the lock check |

## Zone and calendar vectors (T1.3)

`rust/testdata/t06-vectors.json` holds the T0.6 corpus ([chrono-check.md](../phase-0/chrono-check.md)) as an L1 vector file, run against `jiff` by `rust/crates/sdm-host/tests/t06_vectors.rs` in `rust-test`.
`rust/testdata/tzdata-2026c/` holds the TZif files of the corpus's 18 zones and `+VERSION`, copied from the macOS 26.7.1 `/usr/share/zoneinfo` (2026c) that the reference's answers were taken over.

| Vectors | Count | Compared with | Result on `jiff` 0.2.38 |
|---|---|---|---|
| Zone, from a UTC instant | 1,404 | The reference (pandas 3.0.6, `zoneinfo`, 2026c) | All agree, over the vendored 2026c directory and over the bundled database (`jiff-tzdb` 0.1.9, 2026e) |
| Zone, from a wall time, gaps and folds refused | 1,404 | The reference | All agree, over both databases |
| `DateOffset` plus, minus and times three | 208 | The reference | All agree |
| Nanosecond and millisecond conversions | 24 | The T0.6 probe's `chrono` answers | All agree |
| Single values of the accepted shapes | 2,068 | The T0.6 probe's `chrono` answers | All agree |
| PQL date literals (D1 to D9 without D8) | 45 | The T0.6 probe's `chrono` answers | All agree |

- Each zone run first checks its database's version: the vendored directory's `+VERSION` against the 2026c the expectations were taken over, and `jiff_tzdb::VERSION` against the bundled version the vector file names as checked (2026e); a new bundled version fails until the expectations are reviewed against it.
- Ordered lookup and the recorded per-zone source are tested on fixture directories built from the vendored files (`rust/crates/sdm-host/tests/zone_lookup.rs`): the first root holding a zone wins, missing and empty roots are skipped, a zone no root holds comes from the bundled copy, and an unknown zone is refused by name and not recorded.
- `jiff` answers `UTC` and `Etc/Unknown` from no database at all; the resolver records `UTC` as fixed and refuses `Etc/Unknown`, which is not an IANA zone and which Python's `zoneinfo` refuses.

### Parsing differences from `chrono`

The probe used `chrono` to build and check civil dates and times (`NaiveDate::from_ymd_opt`, `NaiveTime::from_hms_micro_opt`), to move between instants and civil time, and for month arithmetic; its reading of the accepted shapes is its own, and is repeated unchanged with `jiff` in the test.

- With `jiff`'s civil types (`Date::new`, `Time::new`, `DateTime` arithmetic and `duration_since` from the epoch) the answers equal `chrono`'s on every vector: there is no parsing difference on the corpus.
- One limit differs and changes how the core must count instants: `jiff::Timestamp` spans -9999-01-02T01:59:59Z to 9999-12-30T22:00:00.999999999Z, so that every instant has a civil time in every offset, while pandas and `chrono` read `9999-12-31` and the millisecond count `253402300799999`.
  Going through `Timestamp`, those three corpus answers are refused (the test `jiff_timestamps_stop_before_pandas_does` holds this); counting from the epoch with civil arithmetic gives them exactly.
  So the core keeps instants as Arrow's integer counts, converts between counts and UTC civil time with civil arithmetic, and uses `Timestamp` and `Zoned` only for zone conversions, which therefore refuse instants after 9999-12-30T22:00:00Z.
- `jiff`'s civil years run from -9999 to 9999, against `chrono`'s ±262,142; pandas never reads a year outside 1 to 9999, so the corpus cannot tell them apart.
