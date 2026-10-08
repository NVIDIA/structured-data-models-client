# Owner decisions

Every decision the owner makes, newest last.
The plan's section 2 holds the standing decisions; this file records how and when they were made, and any made during Phase 0.

| # | Decision | Where it applies |
|---|---|---|
| 1 | Python is the main interface and comes first; Java follows and never holds Python back | Plan section 2, decision 1 |
| 2 | Python keeps every backend it supports today; Java starts with DuckDB, SQLite, Postgres and Databricks | Decision 2 |
| 3 | Refusing an input the old Python client accepted is a regression (Java on Snowflake not held to this yet) | Decision 3 |
| 4 | Quirks follow common sense with a small bias towards today's Python | Decision 4, section 5 |
| 5 | The reference is `rust-core-reference`: `main` plus only the minimum determinism fixes, pushed to GitHub, compared locally | Decision 5, T0.3 |
| 6 | PQL comes from the PQL team's C++ core through its Rust crate; the hand-written parser is a stopgap | Decision 6 |
| 7 | The data layer is swappable; a Pyramid provider is not in scope | Decision 7 |
| 8 | Java's first consumer brings its own data layer; Java loads the core as a native library; no WebAssembly | Decision 8 |
| 9 | No calendar budget and no time estimates | Decision 9 |
| 10 | This team owns the core long term | Decision 10 |
| 11 | Work lives on the epic branch `rust-core`; each phase on its own branch, pushed, then merged locally once verified; no pull requests | Decision 11 |
| 12 | The owner is the only human approver; a fresh agent is the independent reviewer | Decision 12 |
| 13 | The first attempt is archived on GitHub as `archive/rust-core-attempt-1`, as one squashed snapshot commit with no earlier history | Decision 13, T0.2 |
| 14 | Phases are named Phase 0 to Phase 5; each has a design doc and tracker in the branch | Plan section 9 |
| 15 | The old engine is deleted when the epic is ready to merge, with no switch-back release | Plan section 7 |
| 16 | The archive redacts the internal catalog and workspace names and has a reworded commit message; it is pushed as `archive/rust-core-attempt-1` (commit `3887762`); all other scan hits listed in archive.md accepted as listed | Decision 13, T0.2, [archive.md](archive.md) |
| 17 | J1: ranking and forecasting on the SQL backends move from S3 into S1 (offline backends) and S2 (live backends), because every Databricks job cell asks them; on in-memory frames they stay in S3 | Plan section 4, scope manifests, [databricks-jobs.md](databricks-jobs.md) |
| 18 | J2: Java's S2 agreement with Python covers 4 of the 6 base questions (no ranking or forecasting) for now | Plan section 4, `scope-java.yaml`, databricks-jobs.md |
| 19 | J3: the 20 job cells stay the minimum, plus one warehouse-route cell on Databricks Runtime 17.3 or later | databricks-jobs.md, L6 |
| 20 | G1, G12: the base client becomes a native wheel with no pure-Python fallback, and the client publishes no sdist | Plan sections 2 and 3, [support-matrix.md](support-matrix.md), Phase 1a packaging |
| 21 | G2 to G6: wheels for Linux x86_64 and aarch64 as `manylinux2014` (glibc 2.17; `manylinux_2_28` only if the PQL C++ forces it, as a new decision), `musllinux_1_2` if the PQL C++ builds there, macOS 10.12 x86_64 and 11.0 arm64, and Windows x86_64; 32-bit, Windows on Arm, Linux i686, armv7, ppc64le and s390x are dropped | Plan section 3, support-matrix.md, [pql-asks.md](pql-asks.md), L8, installation guide and changelog |
| 22 | G7, G8: CPython only through `abi3` wheels; PyPy, GraalPy and free-threaded CPython are dropped for now | Plan section 3, support-matrix.md, L8 |
| 23 | G9, G10: `pyarrow>=10.0.1` becomes a base dependency of the client; `pandas>=1.5` stays, with pandas 1.5 tested in L5 | Plan sections 3 and 6, support-matrix.md, client `pyproject.toml` |
| 24 | G11: the `relational` extra name stays; the engine distribution is not published after Phase 5 | Plan section 7, support-matrix.md, Phase 5 |
| 25 | G13: Java natives for Linux x86_64 and aarch64 plus macOS, to confirm with the Java consumer before Java packaging | Plan sections 3 and 10, support-matrix.md |
| 26 | G14: only environments the job matrix and L8 test are claimed as supported | support-matrix.md, installation guide, L6, L8 |
| 27 | D1: the RandomState shuffle (`508845fe`, with `f2f5b86c` and `c9695f3c`) is not adopted; the reference keeps `default_rng` | [determinism-fixes.md](determinism-fixes.md), T0.3 |
| 28 | D2: the new seeded SQLite draw is accepted; it becomes a register entry and a random stream the core must reproduce | determinism-fixes.md, `register.yaml` (T0.8), T0.6 |
| 29 | The plan's section 5 keep and drop entries (PK-1 to PK-9, PD-1 to PD-6) are approved register entries | register.yaml |

## Open decisions

Proposed entries in [`register.yaml`](../register.yaml) that wait for the owner; each has a recommendation in its notes.
The other proposed entries can be approved by the independent reviewer.

- RS-4: what an unseeded in-memory entity sample draws from (recommended: OS entropy).
- RS-7: whether in-memory and DuckDB keep refusing seeds from `2**31` on (recommended: keep the refusals).
- DT-1: daylight saving time after 2099, which `chrono-tz` does not apply (recommended: a zone library that applies the last rule, else register the cutoff).
- DT-2: a bundled, pinned zone database or the host's (recommended: bundled and pinned, version recorded).
- DT-3: dropping the int64 wrap at `Timestamp.min` (recommended: drop, after the Phase 1 check of the sentinel).
- DT-5: PQL packed dates, which go to the PQL asks (recommended: keep unless the agreed dialect drops them).
- DT-6: free-form values in per-row time columns, if the core refuses them instead of reading them (recommended: keep reading them).
- P1B-1: any Phase 1b case whose new outcome refuses an input Python accepted.
