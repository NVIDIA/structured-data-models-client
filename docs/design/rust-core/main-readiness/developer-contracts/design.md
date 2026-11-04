<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Developing without losing capabilities

**Implementation plan: extend the existing compatibility ledger and add one developer change guide.**

Phase 8 implements ownership and applicability on API groups, direct behavior links for changed public symbols, and disposable generated views.
The CI effort owns live-registry execution; fixture reduction preserves the type-contract inputs.
Final correctness and benchmarks run after integration, with valid baseline/candidate comparisons and reported numbers, without performance tuning.

## Existing protection and real gaps

After Phase 6, the ledger maps 240 Python symbols, 534 Java symbols, 34 API groups, 41 scenarios and 110 Arrow vectors.
The scenarios declare 792 language/model/connector cells; declaration is not passing evidence.

| Existing mechanism | Protection | Remaining gap |
|---|---|---|
| Python snapshots; Java inventory/japicmp | Names, signatures, mappings; Java source/binary compatibility | Equivalent behavior |
| Manifest/ledger | API groups have tests or language exceptions; scenario scope cannot silently narrow | Mapping a new method to existing tests does not prove they exercise it |
| API/JUnit outcome gates | Required tests run and pass; skips remain incomplete | Unlisted behavior |
| Type contracts | Arrow admission/inference, semantic-type edits, selected reader journeys | Full dtype × stype × connector × model coverage |
| Schema locks and Rust boundary scans | Host/core message shapes, crate dependencies, SQL/runtime ownership | Connection from user capability to engine operation |

The 16 `live_api_tests` records have no execution consumer or test-target validation.
The live workflow runs general scenarios and giant-value tests instead.
Guidance exists in `CONTRIBUTING.md`, `tests/parity/README.md` and `docs/reference/stability.md`, but not as one complete feature-change workflow.
`rust/README.md` still describes Java as planned and Python as the only host.

## Preserve the boundaries

```text
User capability
    → Python / Java API       names, containers, I/O, presentation
    → host/core messages      operations, Arrow ownership, errors
    → Rust engine             query, schema, sampling, SQL, results
    → behavior tests          declared local/live evidence
```

Rust is the engine, not a home for every shared utility.
The native protocol is internal and version-matched; unpublished Rust `pub` items are not independently stable user APIs.
Document operations and ownership without freezing every internal function.

## Options

| Option | Benefit | Limitation |
|---|---|---|
| A. Guide/checklist only | Clear developer instructions | Missing coverage still depends on review |
| **B. Extend the ledger; generate readable views** | One record connects APIs, engine ownership, scope and tests | Review must still judge assertion quality |
| C. Generate language APIs from one schema | Less signature drift | Does not establish semantic coverage; risks replacing idiomatic APIs/bindings |

**Recommend B plus A's short guide.**
Keep existing [japicmp](https://github.com/siom79/japicmp); consider [cargo-semver-checks](https://github.com/obi1kenobi/cargo-semver-checks) only if Rust crates become supported embedding APIs.
Neither substitutes for behavior tests.

## Concrete workflow and enforcement

1. Add `docs/development/change-guide.md`, linked from `CONTRIBUTING.md` and `AGENTS.md`.
   Cover code ownership, commands, generated files, tests, additions/removals and review; repair the existing Rust crate guide.
2. Extend API groups with engine owner/operation links and behavior-test links for changed symbols.
   Shared tests remain valid, but their covered symbols must be explicit.
   Validate operation names against the generated current contract, not a second hand-maintained inventory.
3. Keep applicable language, input route, model, task, dtype and stype scope in the existing manifest.
   Generate API-equivalence, engine-operation and execution/type views from it.
   Separate **supported / intentionally unsupported** from **local passed / live passed / untested / unavailable**, with artifact/model identity.
4. Retain merge-base change decisions; require reasons for removals, narrowed scope and language exceptions.
   CI checks inventories, mappings, schema and actual required outcomes, including executing the live-test registry in its designated lane.
   Replace migration-specific default baselines with the actual PR target or an explicitly supplied reference before adopting `main`.

Do not expand every theoretical combination: test shared engine rules in Rust, host conversions in both languages, and affected driver representations through connector journeys.
An omitted combination cannot become “never supported” without evidence.
Keep generated views as disposable artifacts; retain compact declarations and necessary fixtures.

## Example and acceptance

For gradients, the existing group links Python `GradientExplanations.to_pandas` and Java `GradientExplanations.toArrow` to public tests.
The improved workflow also names Rust decoding as owner, declares model/task scope and identifies assertions for each accessor.
Real-model success remains separate from universal warehouse/deployment support.
A removal requires a migration/deprecation decision and replacement coverage, not merely deletion of its mapping.

**Done when:** a changed API missing language applicability, ownership, behavior coverage or scope fails the gate, while private refactoring needs no invented capability.
A developer or agent can follow the guide and read the generated views without migration-era context.

Sources: `tests/parity/{manifest.json,ledger.py,api_contracts.py,check_junit.py}`, `scripts/{snapshot_public_api,snapshot_java_api,check_java_api}.py`, `scripts/rust_core/{contract_lock,l0_deps,l0_scan}.py`, and `docs/reference/stability.md`.
