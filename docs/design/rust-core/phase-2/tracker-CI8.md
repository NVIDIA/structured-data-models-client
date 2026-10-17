# Phase 2 tracker: CI8

The design branch owns `tracker.md`; this file holds CI8's section until it folds in.

## CI8

On 2026-10-17 RustSec published three advisories against `zip` 6.0.0, all about extraction, and every branch's `rust-deny` job went red.
`zip` 6 reaches the graph only as a build-dependency of `libduckdb-sys` 1.10506.0 (`zip = "6"`), so `cargo update -p zip` cannot reach the fixed `zip` 9.
The owner chose a reviewed exception, removed at the next DuckDB upgrade (row KO2 in `phase-2/decisions.md`, the owner's).
CI8 changes a judging file, `rust/deny.toml` (a protected path), so it lands alone.

### What changed, by exit

1. **The exception.**
   `rust/deny.toml`'s `[advisories]` block ignores the three IDs `cargo deny --locked check` reports against `zip` 6.0.0:
   - RUSTSEC-2026-0367: SUID and SGID bits kept on extraction.
   - RUSTSEC-2026-0368: `ZipArchive::extract` creates symbolic links whose targets leave the destination.
   - RUSTSEC-2026-0369: `ZipStreamReader::extract` and `ZipArchive::extract` apply Unix permissions through symbolic links.

   The comment above them says why the code is not compiled and when they leave (the next DuckDB upgrade, or `libduckdb-sys` moving to `zip` >= 9), and each entry has a short `reason`.
   The comment above `ignore` now points at each phase's `decisions.md`, not only Phase 1's.
   Nothing else in `deny.toml` changes.
2. **It passes with the change and fails without it.**
   Locally with `cargo-deny` 0.20.2, the version CI's pinned `taiki-e/install-action` installed in the red run:
   - Without the change (`cargo deny --locked --config <deny.toml at 292f8d8a> check`): exit 1, `advisories FAILED, bans ok, licenses ok, sources ok`, naming RUSTSEC-2026-0367, RUSTSEC-2026-0368 and RUSTSEC-2026-0369, the same three CI's red `rust-deny` job named.
   - With the change (`cargo deny --locked check`): exit 0, `advisories ok, bans ok, licenses ok, sources ok`.
   - Both runs print the same warnings: the duplicate versions `multiple-versions = "warn"` reports, and that `zip` 6.0.0 is yanked (`yanked` is a warning by default; it leaves with the same DuckDB upgrade).
3. **The extract path is not compiled in our build.**
   - `cargo tree --locked -e build,normal -i zip`: `zip` 6.0.0 is a build-dependency of `libduckdb-sys` only, through `duckdb` and `sdm-testkit`, the dev-only test host.
   - `cargo tree --locked -e features -i libduckdb-sys`: the feature `bundled` is on (the workspace's `duckdb = { version = "=1.10506.0", features = ["bundled"] }`).
   - `libduckdb-sys` 1.10506.0's `build.rs` calls `zip::ZipArchive::new` and `extract` only in `extract_libduckdb`, inside `mod build_linked`, which is `#[cfg(not(feature = "bundled"))]`; `main` calls `build_linked::main` under the same cfg.
   - The built build-script binary (a target built from the same `Cargo.lock`, unchanged between `292f8d8a` and `rust-core`'s `3b81df85`) holds no symbol of the `zip` crate: `nm -C` finds no `ZipArchive` or `extract`, no `build_linked` symbol, and 27 `build_bundled` symbols.
     The one name containing `zip::` is `core::iter::adapters::zip`.
4. **A watch item.**
   `docs/design/rust-core/` has no list of dependency watch items.
   The place for it is the `duckdb` row of `phase-1/dependencies.md` (or its Phase 2 successor), as that file already records for pyo3 that RUSTSEC-2026-0176 and RUSTSEC-2026-0177 are ignored until the next Arrow bump.
   CI8 does not edit that Phase 1 file; the orchestrator adds the note, or the design branch carries it into Phase 2's dependencies.

### For the landing

`rust/deny.toml` is a protected path, so the landing names it in its Protected-paths trailer.
