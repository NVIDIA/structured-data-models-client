# Phase 1 evidence pack (T1.35)

The evidence the design's go/no-go list asks for (design.md, "The evidence
pack", items 1 to 8), each generated from the branch's own runs. Read
[NOTES.md](NOTES.md) first: it says which commit each run was made on, what
was rerun under the owner's items 49 and 50, and what is not done.

Nothing here is written by hand except NOTES.md and this index. The gate's
evidence JSON stays outside the tree (design.md:432); its SHA-256 is in
[gate/evidence-json.sha256](gate/evidence-json.sha256). The held-out bundle's
cases are not in the pack: only their digests, counts and the comparator's
reports are.

Local paths in the copied logs are written `<worktree>`; the Databricks
Volume the DBR job read from is written `<the authorized volume>`.

| # | Item | File | Made by |
|---|---|---|---|
| 1 | Identity: phase commit, base, checkpoint, reference, CI and local wheels, cell versions, platform | [identity.md](identity.md) | `scripts/rust_core/evidence.py` from the gate run; wheel digests from each `<wheel>.json` that `native_wheel.py` writes |
| 2 | Per-case table, every cell and runner, each case classed, with its register entry | [cases.md](cases.md) | `scripts/rust_core/evidence.py` over the `gate.sh --locked 1b --out` observations and the per-cell runs (`dbr17-pandas2`, `floor-pandas2.2`) |
| 3 | Per hard seam: the deepest stage reached, and what is unproven | [seams.md](seams.md) | `scripts/rust_core/evidence.py`, the same run |
| 4 | Comparator report, by rule | [comparator.md](comparator.md) | `scripts/rust_core/evidence.py` from the `compare` step's log |
| 5 | Held-out run (H11): digests, the one-shot run and the reruns of the owner's items 49 and 50, counts against H11, the entries proposed at the run's start | [heldout.md](heldout.md), [heldout/](heldout/) | the T1.35 held-out script (as-built.md's T1.20 digests, `register-at-run-start` digest, `compare.py`); each rerun's `identity.txt` and `binaries.txt` name its commit and builds |
| 6 | Boundary: the pure-Rust host's dependency tree, no `pyo3` | [boundary/](boundary/) | `cargo tree --locked -e normal -p sdm-testkit`, and `-i pyo3` finding nothing; the gate's `l0` step |
| 7 | Size: Rust lines per crate and the Python they replace | [size/](size/) | `tokei` over `rust/crates` and the client's Python |
| 8 | Offline, for information: statements per public call against the reference | [comparator.md](comparator.md) | the comparator's `sql.count` diagnostics (statements per operation, against the reference). The wheel's size is not recorded here; its digest is in identity.md |
| - | The locked gates (1a, 1b) and the DBR 14.3 LTS install check | [gate/](gate/), [dbr143/](dbr143/) | `final_gate.sh` and the gate's `--locked 1b`; the one-time Databricks job's public result |
