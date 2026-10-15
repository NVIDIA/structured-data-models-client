# Phase 1 evidence snapshot (T1.35)

An evidence snapshot for the Phase 1b go or no-go: the ten items design.md
lists under "Phase 1b go or no-go evidence pack", each linked to what the
branch's own runs produced, with every sub-item not covered here marked
pending or not measured and, where it exists, pointed to elsewhere in the
repository. Read [NOTES.md](NOTES.md) first: it says which commit each run was
made on, what was rerun under the owner's items 49 and 50, and what is not
done.

**Generated, not to be edited:** identity.md, cases.md, seams.md,
comparator.md, heldout/ (compare, counts, differences, identity and report
files), boundary/, size/, gate/, dbr143/. **Hand-written:** this index,
NOTES.md, heldout.md's last section, heldout/rust-host-leg.md,
heldout/h11-classification.md, reviews.md and c9.md. One generated cell was reworded by hand after generation, its values unchanged: identity.md's CI-run row reads "(Moved-from; same tree as the phase commit)" where the identity block writes "(same tree)".

The gate's evidence JSON stays outside the tree (design.md:432); its SHA-256
is in [gate/evidence-json.sha256](gate/evidence-json.sha256). The held-out
cases are not in the pack: only their digests, counts and the comparator's
reports. Local paths in the copied logs are written `<worktree>`; the
Databricks Volume the DBR job read from is written `<the authorized volume>`.

| # | Design item | Here | Made by | Not covered here |
|---|---|---|---|---|
| 1 | Identity: commit, base, reference, checkpoint, cells and platforms, DuckDB on both hosts, zone database sources, wheel hashes | [identity.md](identity.md) | the T1.35 run script's identity block, which hashes the wheels' bytes itself ([tools/identity-block.txt](tools/identity-block.txt), the script's digest in [tools/run_all.sh.sha256](tools/run_all.sh.sha256)) | DuckDB in Rust (`duckdb =1.10506.0`): design.md's dependency row. Per-zone database provenance: not recorded; register DT-2 states the sources (the host lookup, the bundled fallback) |
| 2 | The 40+1 per-case table, every cell and runner, classed, with entries; the archive best effort | [cases.md](cases.md) | `scripts/rust_core/evidence.py` over the `gate.sh --locked 1b --out` observations and the per-cell runs (`dbr17-pandas2`, `floor-pandas2.2`) | The archive (3887762): not measured (C9) |
| 3 | Per hard seam, the deepest stage; unproven stages | [seams.md](seams.md) | `scripts/rust_core/evidence.py`, the same run | - |
| 4 | Comparator report; register entries and owner calls; type-set findings; DT-1, DT-2, DT-3, ORD-1 | [comparator.md](comparator.md) | `scripts/rust_core/evidence.py` from the `compare` step's log | Type-set findings: as-built.md, T1.33, and `contract/type-set-vectors.json`. DT-1, DT-2, DT-3 and ORD-1: register.yaml, each approved by the owner. Owner calls: the register holds no proposed entry |
| 5 | Held-out run: per case against the reference, the H11 count, the bundle's digests | [heldout.md](heldout.md), [heldout/](heldout/), [heldout/h11-classification.md](heldout/h11-classification.md) | the T1.35 held-out script (digests, run-start register digest, `compare.py`); the classification and verdict by both reviewers | - |
| 6 | Boundary: the L0 report, `cargo tree` for the pure-Rust host, semantic callbacks (zero) | [boundary/](boundary/) | `cargo tree --locked -e normal -p sdm-testkit`, and `-i pyo3` finding nothing | The L0 report: the gate's `l0` step ([gate/](gate/)); callbacks: [c9.md](c9.md) |
| 7 | Size: Rust lines per crate against the Python replaced; size trends; dependency table; emulation allowlist | [size/](size/) | `tokei` over `rust/crates` and the client's Python | Size trends: tracker.md's size-trend reviews, to checkpoint 4; later trends not measured. Dependencies: `phase-1/dependencies.md`. Allowlist: `scripts/rust_core/emulation-allowlist.txt` |
| 8 | Offline: statements per case against the reference; wheel size | [comparator.md](comparator.md) | the comparator's `sql.count` diagnostics | Wheel size: not measured |
| 9 | CI links, seeded faults, the review's findings and resolutions, what was not done | [reviews.md](reviews.md), [gate/](gate/) (the final code's locked gates in [gate/final-9779fcb/](gate/final-9779fcb/), NOTES.md item 9), [dbr143/](dbr143/) | hand-written from the runs and the reviews; the gate summaries; the DBR 14.3 LTS job's public result | - |
| 10 | C9: each criterion's result; the executor's recommendation | [c9.md](c9.md) | hand-written | The decision is the owner's (decisions.md, E154) |
