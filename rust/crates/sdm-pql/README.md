# sdm-pql

The PQL (Predictive Query Language) lexer, parser and validators.
Entry points: `parse` and `resolve` (`api.rs`), with `ParsedQuery`, `ValidatedQuery`, `SchemaDefinition` and `ValidationResponse`.
It depends on no crate of ours (its one third-party dependency is `jiff`, for dates), so it can be read and tested on its own.
Only `sdm-core` depends on it: `sdm-core/src/pql.rs` calls it, and so does the test-only `sdm-core/src/test_ops.rs` (a fault injected into the parser, built only with the `test-ops` feature).
The model's capabilities are checked in `sdm-core`, not here, except the two filter capabilities resolution needs (`Filters`).
Copying a Python library's incidental behavior (a codec, a repr) is allowed only in `pyint.rs`, `float_repr.rs` and `escapes.rs` (see `scripts/rust_core/emulation-allowlist.txt`); ports of rules the Python code defines, such as date offsets in `dor.rs`, are not emulation.
Must not hold: a dependency on `sdm-contract` or any other crate of ours, SQL, or any knowledge of data sources.

```text
used by:     sdm-core
     |
sdm-pql
     |
depends on:  no crate of ours
```
