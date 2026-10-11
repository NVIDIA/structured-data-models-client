# Phase 1 dependencies

One row per dependency of the Rust workspace in `rust/`, and per algorithm the core ports or concern it leaves in the host: whether the core uses a library, ports a defined algorithm, or keeps the concern in the host, and the measured reason.
Rows marked "planned" name the task that adds them.
Versions are pinned by `rust/Cargo.lock`; `rust/deny.toml` holds the licence, advisory, source and ban rules.
A row changes with the dependency, in the same commit.

| Dependency | Version | Crates | Choice | Reason |
|---|---|---|---|---|
| `arrow-*` (`arrow-schema`, `arrow-buffer`, `arrow-array`, `arrow-data`, `arrow-ipc`, `arrow-select`) | 58 | contract, host, `sdm-core` (`arrow-select` from T1.11c), `sdm-schema` (from T1.11b), `sdm-engine` and `sdm-request` (from T1.11c), later domain crates; `sdm-py` (`arrow-array` and `arrow-schema`, with `ffi`) | Use a library | The closed type set is Arrow's (plan section 3); one major across the workspace, because `duckdb =1.10506.0` and `arrow-pyarrow` 58 both need 58 and two majors in one process need FFI glue between them (decision B-D2); `cargo-deny` refuses a second version of any `arrow-*` crate |
| `pyo3` | 0.28 (`abi3-py310`) | `sdm-py` only | Use a library | The binding tool the plan names; 0.28 because `arrow-pyarrow` 58 needs it (B-D2); RUSTSEC-2026-0176 and RUSTSEC-2026-0177 are ignored with reasons until the next Arrow bump brings 0.29 (decision E1) |
| `arrow-pyarrow` | 58 | `sdm-py` only | Use a library | PyCapsule only, which the `pyarrow>=14.0.1` floor guarantees (E16), without our own FFI code; as built at T1.8 it reads Arrow in, after the binding checks for `__arrow_c_stream__`, and Arrow out is the binding's own `ArrowStream` over `arrow-array`'s safe FFI types, since `arrow-pyarrow` exports through pyarrow's private `_import_from_c` |
| `duckdb` (duckdb-rs, `bundled`; `vtab-arrow` from T1.11c) | `=1.10506.0` (DuckDB 1.5.6) | `sdm-testkit` only | Use a library | The pure-Rust test host runs the core's DuckDB SQL at the reference's DuckDB version; `cargo-deny` refuses it anywhere else. `vtab-arrow` gives the Arrow table function the test host binds a key set through, its rows as Arrow, never as text (decision E42); it adds the `num` crates, MIT or Apache-2.0, to the test host only, and keeps each batch it is handed for the life of the process, which a test process allows. ICU, which the Python wheel links, lets a host set a `TimeZone` so `CAST(TIMESTAMPTZ AS TIMESTAMP)` matches the Python host (`time.timestamptz`): the `icu` feature planned for T1.11b cannot build from the published crate (it needs DuckDB's sources beside it), so the test host loads DuckDB's signed ICU extension for this version explicitly, autoloading and autoinstalling off, after checking the installed file's SHA-256 against the one recorded for its platform, and checks its version (decision E37); a signature-verified network bootstrap, cached in CI, not hermetic |
| `jiff` (`std`, `tzdb-zoneinfo`, `tzdb-bundle-always`; no `tz-system`) | 0.2.38 | host, `sdm-engine` (the `Zones` port, T1.11a), `sdm-core` (resolving through it), the PQL parser | Use a library | The one calendar and zone library (decision A7): it meets DT-1 and DT-2 on all 2,808 zone answers and all 208 `DateOffset` answers of the T0.6 corpus, with a host directory and with its bundled copy; `tz-system` stays off so nothing reads the environment; from T1.10c `rust/clippy.toml` bans the global database (`jiff::tz::db`, `TimeZone::get`, the `in_tz` methods, `TimeZone::system`, the parsers that look zones up in it: `DateTimeParser::parse_zoned`, `Zoned::strptime`, `BrokenDownTime::to_zoned`) and the clocks, and `l0_scan.py` refuses a `Zoned` parsed through `str::parse` or `FromStr`, and `TimeZoneDatabase::from_dir` everywhere but `sdm-host`'s zone module, which reads the host's lookup roots (E24) |
| `jiff-tzdb` | 0.1.9 (tzdata 2026e) | host | Use a library | The bundled zone database behind `jiff`, the DT-2 fallback; its version (`jiff_tzdb::VERSION`) is recorded as the bundled source's version, and the zone vectors are checked against it |
| `regex` | 1 | `sdm-host` tests only | Use a library | Repeats the T0.6 probe's reading of the accepted shapes in the zone and calendar vectors |
| `half` | 2 | `sdm-host` tests only | Use a library | Builds the `float16` values the widening tests admit; Arrow's own `f16` type |
| `chrono` | 0.4 (transitive) | none directly | Not used | Only Arrow's own dependency; first-party code uses `jiff` (A7) |
| `chrono-tz` | none | none | Banned | Expands rules only to 2099 and embeds its database at build time, so it fails DT-1 and DT-2 (T0.6); `cargo-deny` refuses it |
| `rand` | none | none | Banned | Random draws come only from the registered streams, ported from numpy (RS-1, T1.22); `rust/clippy.toml` bans `StdRng`, `ThreadRng`, `thread_rng`, `rng` and `random` in the domain and shipped crates (T1.10c) |
| `serde`, `serde_json` (`float_roundtrip`, `preserve_order`; `raw_value` in `sdm-testkit` and, from T1.11d, `sdm-wire`) | 1 | contract, host, `sdm-testkit` (cases and observations), `sdm-request` (a payload's numbers and text, from T1.11c), `sdm-wire` (the model API's JSON, from T1.11d), later all | Use a library | Control messages are versioned JSON (plan section 3); `raw_value` lets the case loader keep an entity key's integer literal of any size as its digits, never through a double, and only adds `RawValue`. From T1.11d `sdm-wire` reads the model API's JSON with it (`json::read`): the whole text is read once by `serde_json`, which refuses what JSON does not allow with the fault's position and category, then each value's own text is taken as a `RawValue`, so a number stays the token its sender wrote and a request state's payload, read and written again, is the same text; text nested past 128 levels (`serde_json`'s default) is refused, its nesting counted without recursion before any value is built, so neither building nor dropping one can overflow the stack; a `NaN` or `Infinity` token is refused as `not_finite` before `serde_json` reads the text; an object's repeated member names are found through an ordered index (`BTreeMap`), so a wide object reads in n log n; `arbitrary_precision` would do the same through `serde_json::Value`, but changes how every crate in the build reads numbers, untagged enums included, so it is not used |
| The prediction response reader (`sdm-wire::response`) | — | `sdm-wire` | Port a defined algorithm (thin, by hand) | The client's bindings (`sdm_api.generated.tfm_api`, generated from the pinned OpenAPI file) are the definition: the same fields required and types refused, in the same order, held to them by `contract/response-vectors.json`, which both read (`sdm-wire` `tests/response_vectors.rs`, `contract/tests/test_response_vectors.py`); generating it from the OpenAPI file is T1.28's (decision E45) |
| `schemars` | 1 | contract | Use a library | Generates the committed JSON Schema from the serde types, so the schema cannot drift from the code |
| `thiserror` | 2 | first-party crates | Use a library | Typed errors; `anyhow` stays out of first-party crates other than `sdm-testkit` |
| `indexmap` | 2 | where output order matters | Use a library | Insertion order for anything that reaches an output; `HashMap`, `HashSet` and `RandomState` are banned in the domain and shipped crates by `rust/clippy.toml` (T1.10c) |
| `stacker` | 0.1 | `sdm-core` | Use a library | Deep queries: `sdm-core`'s adapter grows one stack per call (`on_adapter_stack`) and parses, resolves, builds its answer and drops the tree on it; the interim parser grows none of its own since T1.7c; T1.6b tests traversal and destruction of a deep tree in a subprocess |
| `anyhow` | 1 | `sdm-testkit` only | Use a library | Test-host plumbing only; `cargo-deny` refuses it elsewhere |
| `base64` | 0.22 | contract | Use a library | Binary scalars travel as standard base64 text |
| `proptest` | 1 (`std` only) | contract tests | Use a library | The bounded, property-based fuzz test of the contract's decoders |
| `sha2` | 0.10 | contract tests; `sdm-testkit` | Use a library | Hashes the committed schema for the lock check, and each case file for an observation's identity |
| `tracing` (features `std`, `log`) | 0.1.44 | first-party crates (`sdm-core` from T1.8) | Use a library | Logging in Rust (decision H8); the `log` feature emits each event to the `log` facade when no subscriber is set, which is what `pyo3-log` reads |
| `serde_stacker` | 0.1.14 | contract | Use a library | Reads the validated-query format back at `MAX_DEPTH`, past `serde_json`'s recursion limit (H27), through `read_validated_query`, which refuses text nested past `MAX_JSON_DEPTH` (2,008) before reading; `sdm-contract` turns on `serde_json`'s `unbounded_depth` for it, which only adds the switch that reader uses: every other read keeps the 128-level limit |
| `pyo3-log` | 0.13 | `sdm-py` only | Use a library | Bridges Rust `log` records to Python `logging`, initialized at the first core call; its level cache is reset at each run start (H8) |

## Ported algorithms and host concerns

| Concern | Choice | Where | Reason |
|---|---|---|---|
| `SeedSequence`, PCG64, bounded draws and the shuffle (RS-1) | Port a defined algorithm | `sdm-engine`, from T1.11c; completed at T1.22: `sdm_engine::random` holds `SeedSequence`, PCG64, `random_interval` and `Generator.shuffle` | A promised stream (PK-1); no Rust crate reproduces numpy's draws; vectors from numpy pin it (F4): `sdm-engine`'s `tests/rs1.json`, recorded from numpy 2.5.3, for eleven seeds from 0 to past 2**200, and `tests/rs1-draws.json` (T1.22, `tests/rs1_record.py`), for every seed of T0.6's model and numpy's compiled `random_interval` to a maximum of 2**64 - 1 |
| float32 Kahan sums in row order (PK-2) | Port a defined algorithm | `sdm-engine`, T1.27; from T1.11c `sdm_engine::sample::sum_labels` for a windowed `SUM` (pandas' `group_sum` at float32) | Promised label arithmetic |
| `DateOffset` month rules (PK-3) | Port a defined algorithm over `jiff` | `sdm-engine`, T1.27 | Promised; `jiff`'s civil arithmetic gives the dates, the rules come from pandas' definition, checked by the 208 T0.6 answers |
| PQL date literals D1 to D9 | Port a defined algorithm over `jiff` | `sdm-pql-interim`, T1.6a and T1.6b | Written rules from `date-rules.md`, replacing the archive's 1,695-line clone |
| `pyint`'s digit limit, `float_repr`, `escapes` | Emulate, allowlisted | `sdm-pql-interim` | They decide validity (A4, E9); the allowlist is permanent and owner-gated (H21, E18) |
| Arrow-to-Arrow admission casts | Hand-written, with `arrow-select`'s `take` | `sdm-host`, T1.7 and T1.7c | `arrow-cast` casts silently where admission must refuse by name (`VALUE_OUT_OF_RANGE`), split a batch between rows at the offset capacity and check schemas first (H16); `take` decodes a dictionary and gathers a list view's values, where its semantics fit exactly |
| Text-date shapes (DT-7) | Port a defined algorithm | `sdm-engine::time::text_shapes`, T1.25 | Only the shapes of `date-rules.md`, with its T0.6 vectors (H21) |
| `DateOffset` month arithmetic in PQL date literals | Port a defined algorithm | `sdm-pql-interim`, `dor.rs` | pandas' definition, as the reference's parser applies it |
| `str.isspace` | Port a defined algorithm | `sdm-pql-interim`, `syntax.rs` | Python's whitespace set decides what the parser skips before a query's first word |
| The `query[:-1]` quirk | Port a defined algorithm | `sdm-pql-interim`, `visitor.rs` | With no space in the query, the reference reads the `EXPLAIN` or `EVALUATE` prefix from all but the last character; which flag a query gets depends on it |
| pandas and NumPy conversion and presentation | Keep in the host | `_rust/`, the reviewed table | The core never sees a dtype (plan section 3) |
| HTTP, authentication and the model call | Keep in the host | Client transport | The core passes model bodies as bytes |

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
  So the core keeps instants as Arrow's integer counts in columns and as `Instant(i128)` nanoseconds in scalars (decision F3), converts between counts and UTC civil time with civil arithmetic, and uses `Timestamp` and `Zoned` only inside zone conversions; past `Timestamp::MAX` a zone conversion is computed with civil arithmetic and the zone's offsets, never refused because `jiff` cannot hold the instant (decision E6).
  That path is not yet proven: T1.25 adds vectors at 9999-12-31 in every vendored zone, and any refusal it cannot avoid needs a case and a register entry first.
- `jiff`'s civil years run from -9999 to 9999, against `chrono`'s ±262,142; pandas never reads a year outside 1 to 9999, so the corpus cannot tell them apart.
