# Phase 0: prepare

**Branch:** `rust-core-phase-0`, off `rust-core`; merged locally into `rust-core` once verified.
**Tracker:** [tracker.md](tracker.md).
**Decisions:** [decisions.md](decisions.md).

## Goal

Set up everything the later phases compare against or build on, before any Rust is written: a repeatable reference, the scope it must cover, CI on the migration branches, and the inputs Phase 1 needs.
No product code changes on `rust-core` in this phase.

## Tasks

### T0.1 CI on the migration branches

- Extend `.github/workflows/ci.yml` so it also runs on pushes to `rust-core` and `rust-core-phase-*` (not `archive/*` or `rust-core-reference`).
- Exit: a push to `rust-core-phase-0` runs CI, and it is green.

### T0.2 Archive the first attempt

- Build one squashed snapshot commit of the first attempt's best candidate (`ab85a056`, tag `pm/2026-10-07/all-public-cases-pass`, about 7,100 files and 217 MB, no file over 50 MB) on an orphan branch `archive/rust-core-attempt-1`, with no earlier history.
- Remove `.github/` and `.gitlab-ci.yml` from the snapshot, so it can't trigger workflows or publishing.
- Before pushing, scan it with the hygiene check's `internal` patterns, a credential scanner (gitleaks), and a deny-list of workspace, catalog, warehouse and endpoint names and email addresses; redact each hit or have the owner accept it (the repository is internal), and record what was changed in the commit message.
  The hygiene check's `retired` scan is expected to flag historical interface names; those are not leaks.
- The commit message records the source commit and remote, and says the branch is a read-only source to port from, never merged.
- Exit: the branch is on GitHub; a fresh clone of it contains the expected tree minus the recorded redactions; every scan hit is redacted or accepted by the owner.

### T0.3 Reference branch and environment

- Select the minimum determinism fixes from the first attempt's `rust-core/p1-determinism` (27 commits on the old GitLab layout): only those that make today's Python give the same answer on every run (seeded draws, stable ordering, independence from the hash seed).
  No refusals and no other behavior changes.
  Record each kept and dropped commit with a one-line reason in [determinism-fixes.md](determinism-fixes.md).
- Port the kept fixes onto the latest `main` as `rust-core-reference`; the histories are unrelated and the packages were renamed, so this means path-rewritten patches, not cherry-picks.
- For each kept fix, compare outcomes with unpatched `main`; any changed outcome gets a register entry before the fix is adopted.
  A fix that changes warehouse behavior (Snowflake, Postgres, Databricks sampling) is kept only if a local live comparison is done; otherwise it is dropped.
  A fix that changes answers users see, such as the shuffle moving from `default_rng` to `RandomState` (`508845fe`), is an owner decision.
- Tag the unpatched `main` commit as `reference-baseline-<sha>` (the immutable copy of the original baseline).
- Install the reference in its own worktree and environment at `~/code/sdmc-reference`, recording Python, pandas, NumPy, PyArrow and torch versions in `reference-environment.md`.
- Add a guard that fails if the reference environment ever loads the Rust core, and show it fails when a stub native module is put on the path; it runs at the start of every reference session.
- Protect the `reference-baseline-*` tags with a GitHub tag ruleset.
- Push `rust-core-reference` to GitHub; comparisons run locally.
- Exit: the ported determinism checks and `scripts/snapshot_wire_payloads.py` output are byte-identical under `PYTHONHASHSEED=0` and `=1`, with and without torch installed (live and network tests excluded); the guard passes and fails on the stub; the environment is recorded.

### T0.4 Support matrix

- Record what today's client supports, per distribution and extra, from the `pyproject.toml` files, the README and installation docs: Python versions, operating systems and architectures, pandas and pyarrow ranges (the client and the connectors differ), and the relational extra's native wheels; plus the Databricks runtime cells (for example, Databricks Runtime 14.3 ships pyarrow 8).
- Propose the native wheel coverage for the Rust core, marking every gap against today's support; each gap is a decision for the owner, recorded in [decisions.md](decisions.md).
- Exit: `support-matrix.md` exists and every gap has an owner decision.

### T0.5 Scope manifests and case inventory format

- Define the scope manifest (one per language) and the case inventory format: one YAML or JSON file listing public paths (operation × input mode × backend), the type × role cells in scope, and the case IDs that cover them (empty until Phase 1a).
- Write the first Python and Java manifests from the plan's decisions (Python: every backend today; Java: DuckDB, SQLite, Postgres, Databricks).
- Map every cell of the first attempt's 20-cell Databricks job manifest (`investigations/P95-ACCEPTANCE-PREPLAN.md` in the first attempt's harness), through today's notebooks in `examples/databricks/`, to the public paths it exercises, so Phase 3 includes what those jobs need; the old `matrix*.sh` scripts point at notebooks that no longer exist and are not used.
- Exit: the format is documented, both manifests exist, and the job-to-path map is complete.

### T0.6 Random streams and time library

- Pin the reference's random streams: list every generator the reference uses (numpy `RandomState`, `default_rng`, the C++ `std::mt19937`) and every seeded draw done in SQL per backend, where, and with which seeds, in `random-streams.md`; these become register entries the core must reproduce.
- Record the accepted text-date formats and inference rules in `date-rules.md`, from `sdm_client/core/dates.py` and the public paths that reach it.
- Confirm `chrono` with `chrono-tz` using a throwaway Rust probe kept outside the branch, over the first attempt's date corpus plus added time-zone and post-2037 cases; post-2037 differences are expected, since the register drops the reference's rule.
- Exit: the three documents exist; any `chrono` mismatch is listed with a decision.

### T0.7 Phase 1b case inventory

- Choose the fixed case inventory for the Phase 1b spike from the first attempt's public-API tests (`tests/rust_core/test_public_*_on_both_paths.py`) and the nine known type regressions from its probe: keys, decimal precision, nested values, empty and all-null data, time inference, on DuckDB.
- Exit: `phase-1b-inventory.yaml` lists the cases, each with its reason; it is small enough to run in minutes.

### T0.8 Register and decision record

- Create `docs/design/rust-core/register.yaml` with its schema (old behavior, new behavior, affected inputs, case or task IDs, approved_by) and the entries produced by T0.3 and T0.6.
- Keep [decisions.md](decisions.md) current with every owner decision, including those made before this phase.
- Exit: `register.yaml` validates against its JSON Schema, and `decisions.md` is current.

### T0.9 PQL asks

- Draft the asks to the PQL team (section 3 of the plan) in [pql-asks.md](pql-asks.md) for the owner to send, and track their answers there.
- Exit: the asks are sent; agreement is not a Phase 0 exit.

## Order

T0.1 first (so later pushes run CI).
Then T0.2, T0.3, T0.4, T0.5 and T0.9; T0.6 after T0.3 (the kept fixes decide the random streams); T0.7 after T0.5; T0.8 collects entries from T0.3 and T0.6.

## Phase exit

Every task's exit met, CI green on the phase branch, a fresh-agent review of the phase's changes and documents, and the owner's local merge into `rust-core`.

## Not in this phase

Any Rust code on the branch, the contract, test runners (Phase 1a), and Java work.

## Risks

| Risk | Mitigation |
|---|---|
| A determinism fix also changes answers users see | Compare each with unpatched `main`; register any change, or drop the fix |
| Porting the fixes across the package rename is larger than expected | Take the smallest set that makes runs repeatable; skip anything not needed for that |
| The archive leaks internal names or credentials | Scan and redact before pushing; deleting a branch doesn't remove it from GitHub (objects and forks persist), so publishing the repository later needs a GitHub object purge or a fresh repository |
| Phase 0 grows into a harness project | Documents and the reference only; test runners wait for Phase 1a |
