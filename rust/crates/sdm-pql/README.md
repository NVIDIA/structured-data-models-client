# sdm-pql

The PQL (Predictive Query Language) lexer, parser and validators.
Entry points: `parse` and `resolve` (`api.rs`), with `ParsedQuery`, `ValidatedQuery`, `SchemaDefinition` and `ValidationResponse`.
Depends on no crate of ours (only `jiff`, for dates), so it can be read, tested and reused on its own.
Only `sdm-core` depends on it, and only `sdm-core/src/pql.rs` calls it.
The model's capabilities are checked in `sdm-core`, not here, except the two filter capabilities resolution needs (`Filters`).
Copies of Python behavior are allowed only in `pyint.rs`, `float_repr.rs` and `escapes.rs` (see `scripts/rust_core/emulation-allowlist.txt`).
Must not hold: a dependency on `sdm-contract` or any other crate of ours, SQL, or any knowledge of data sources.
