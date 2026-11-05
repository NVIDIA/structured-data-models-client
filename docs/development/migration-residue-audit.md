<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Remaining migration residue

**Phase 10 cleanup is incomplete.**
I removed large historical datasets and renamed one test directory, but did not finish a repository-wide audit of names, comments, helper machinery and provenance checks.
The earlier test results remain valid; they do not establish that migration baggage was removed.
**Implementation approved; in progress on `rust-core-phase-10`.**
Preserve useful correctness tests and product behavior.

## Scope

Scanned all **2,086 tracked files** at `230878c9a`, including hidden CI configuration, Python, Rust, Java, schemas, fixtures and filenames.
The scan flags **1,023 candidate files**: 831 contain numbered `T…` milestones, 169 have migration-style filenames, and 303 contain numbered decision references; these groups overlap.
There are 5,149 rule matches, not 5,149 independent problems.
Generic API migration guidance and runtime checkpoints are legitimate matches and must stay.
Executable-tool conclusions below also use source inspection and consumer tracing; a marker alone is not evidence that code is obsolete.

The complete file/line inventory and per-file proposed treatment are outside the repo, avoiding another committed migration archive:
`~/.local/share/sdm-orchestration/phase-10/residue-audit/` contains `files.tsv`, `inventory.json`, `scope.json` and the runner reproduction.
The inventory is exhaustive for the recorded search patterns, not a claim that every matched line should be deleted.

## Why `test_wheel_smoke.py` remains

[The tests](../../tests/test_wheel_smoke.py) and [runner](../../scripts/rust_core/wheel_smoke.py) are still used by CI to check installed wheels.
They check dependency compatibility, imports from the installed environment, the native library, missing-library failures, artifact identity and whether required tests actually ran.
Those are useful release checks.

But the pair is **2,750 lines**, including support for a frozen checkpoint's tests judging a different checkout's inputs and defenses against that other checkout rewriting test reports.
That was migration gate machinery; current CI calls the script from the same checkout and does not request a separate `--inputs` checkout.
**Keep a simpler same-revision installed-wheel smoke and its failure controls; remove the separate-checkout machinery and its dedicated tests.**
Keep commit/hash verification and isolation from the source tree.
Remove obsolete `T1.17`, `T1.20`, `E16` and checkpoint prose.

## Findings and proposed treatment

| Area | Verified finding | Treatment |
|---|---|---|
| Dead task administration | [`tests/t30-pending.json`](../../tests/t30-pending.json) describes an old agent/xfail process; no consumer remains. | Delete it. |
| Old comparison artifacts | Eighteen files under `tests/golden/t132g*`, including `python-old.jsonl` and `python-ref.jsonl`, total **1.87 MB**; no behavioral consumer found. | Delete these obsolete artifacts through an explicit removal; preserve current API/wire goldens and never hand-edit their expected values. |
| One-off rename check | [`t128w_rename.rs`](../../rust/crates/sdm-testkit/tests/t128w_rename.rs) recursively scans the repo for the parser's discarded interim name. | Delete this historical rename gate; current crate/dependency checks remain. |
| Runner leftovers | [`run_python.py`](../../contract/runners/run_python.py) rejects old modes, yet retains unreachable `python-ref` branches, `_held_observations`, legacy arguments and a `legs` CLI whose comparison set comes from old runners. | Remove those obsolete paths and dedicated tests; retain native execution, provider applicability and the current completeness checker. |
| Provenance-only assertions | `tests_test_ops/test_l5_conversion.py` and `test_t214d_postgres_values.py` still assert old runner identities and a commit in `contract/reference-env.txt`; the binding-vector test also preserves unrelated historical metadata. | Remove historical identity assertions and then the environment dump; retain actual type/value/driver expectations and fixture completeness checks. |
| Recorder names and stale instructions | Fifteen of seventeen `contract/tools/*.py` files have no CLI entry point; most are used helpers despite claiming they record the old client. | Move used scenario/value helpers into `contract/support`, name them by behavior, remove unused leftovers and false CLI instructions. Keep the two actual current tools: SQLite case loading and native binding-vector generation. |
| Architecture guards | `l0_deps.py`, `l0_scan.py` and `selftest.py` total **4,675 lines**; current CI runs them. They mix useful dependency/SQL/boundary enforcement with checkpoint rename machinery and a `PLANNED = {'sdm-jni'}` exemption, although Java exists. | Keep current architecture enforcement and focused negative tests; remove obsolete cross-checkout machinery and the planned-Java exemption. Do not delete all three files blindly. |
| Remaining milestone names | Root Python tests, package `tests_test_ops`, Rust integration tests, Java classes and vectors still use `T…`/`Phase…` names. | Rename by behavior, updating dynamic imports, Cargo/Maven/CI selections and capability-ledger references atomically. |
| Comments and fixture metadata | Hundreds of case files and production-source comments cite deleted plans, numbered decisions and old agents. | Explain the current rule directly; remove administrative metadata, preserving case IDs/behavior and regenerating generated files through their owner. |
| CI and developer docs | CI still names current jobs `python-ref`, exports unused `SDM_PUSH_BASE`, and describes removed checkpoint worktrees. `validation.md` still links to `tests/parity` and private phase evidence. The testkit README says it will go away with the migration. | Correct job labels and current-role documentation; remove unused environment settings and obsolete phase guidance. Preserve real validation limitations and useful benchmark results. |

The runner issue was reproduced: Python reports cases `a` and `b`, Rust reports only `a`, and the old `check_legs()` returns no problems because no old-client observations supplied its expected set.
Current CI calls `check_observations.py` with explicit case expectations instead; this is a demonstrated defect in leftover tooling, not evidence that current CI silently accepts missing cases.

## What must not be swept away

- Tests of current behavior, native ownership, errors, dependencies, serialization and Python/Java consistency.
- Saved old requests that verify supported deserialization, and compact independent expected values originating from earlier implementations.
- Real runtime checkpoints, model terminology, SQL words and general API migration guidance.
- Generated API/wire snapshots and exact-format fixtures: update their source/generator when appropriate, never rewrite expected answers merely to remove historical words.

## Implementation plan

| Pass | Work | Completion check |
|---|---|---|
| 1. Remove obsolete machinery | Delete the task ledger, unused comparison artifacts and interim-name test. Remove old runner branches/CLI and provenance assertions. Move still-used recorder helpers to behavior-named support modules. | Current runner rejects incomplete/failed evidence; fixture values and driver/type checks remain; no helper imports point at deleted files. |
| 2. Simplify current tooling | Make wheel smoke test the checkout's own revision; keep installed-artifact identity, native failure and required-test controls. Remove checkpoint parser-renaming support and the planned-Java exemption from architecture guards. | Focused positive/negative wheel and architecture tests pass; CI still invokes all required checks. |
| 3. Finish the whole-tree cleanup | Rename milestone files/classes/tests/vectors and update their CI, dynamic-import and capability references together. Rewrite source/docs comments as current behavior. Remove administrative fixture metadata, preserving independent expected values. | Full inventory has an explicit disposition for every remaining match; no stale executable reference; API/capability coverage unchanged. |

Keep these passes sequential so fixture/helper changes do not race global renames.
Before changing generated files, find and use their owning generator; do not edit them by hand.
Keep useful third-party/version/serialized-format provenance and real runtime checkpoint terminology.
Do not change public APIs, sampling, explanation behavior or supported saved-data formats.

Run focused tests after each structural pass; run maintained Python/contract/Rust/Java checks once after integration, using existing warm production/test-ops builds and resource guards.
Validate CI wiring, schema/API/capability checks, lint/type checks and hygiene.
Rerun affected model/connector journeys only if execution or input semantics change; unchanged tests need no new hosted warehouse campaign.
Report retained coverage, deliberate removals, remaining exceptions and size reductions.
Commit reviewable changes on Phase 10; do not merge, push or publish without a further request.
No new permanent scanner, framework or ledger: keep the existing enforcement and external audit inventory.
