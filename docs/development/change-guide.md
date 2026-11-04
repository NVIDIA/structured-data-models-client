<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changing a capability

Start with [CONTRIBUTING.md](../../CONTRIBUTING.md) for installation, maintained tests, formatting, typing, and signed commits.
The [compatibility manifest](../../tests/parity/manifest.json) connects public APIs to behavior tests, applicability, and engine ownership.
Use the actual pull request target as the comparison base.
Private refactoring does not create a new public API contract.

## Find the owner and scope

Python and Java own public containers, connections, I/O, conversions, and presentation.
The Rust engine owns schema and query semantics, sampling, SQL planning, request construction, and result decoding.
The native protocol is internal and version-matched; unpublished Rust `pub` functions are not stable embedding APIs.
Do not move a utility into Rust solely because both hosts use it.
Consult the [Rust crate guide](../../rust/README.md), then the API group's `ownership` and `scope` in the manifest.
Operation names are checked against the current generated locked schema, not a second operation inventory.
Host-only groups require an explicit reason.

Generate a readable, disposable view with:

```bash
.venv/bin/python tests/parity/developer_views.py --out /tmp/sdm-developer-contracts.md
```

This view reports declarations, existing group coverage, explicit per-symbol assertion links, language exceptions, scenario scope, and type boundaries.
It deliberately does not claim that a test has passed.
Use local and live runner receipts tied to the exact candidate, installed artifact, configuration, and model identity for execution evidence.
A missing or unavailable receipt remains incomplete, and an omitted combination is not automatically unsupported.

## Change behavior and protect it

For a bug fix, first reproduce it through the public journey closest to how users encounter it.
Test shared engine rules in Rust, conversion and ownership in both hosts, and affected driver representations through actual connector journeys.
Preserve independent expected outcomes when reducing fixtures.
Do not multiply every dtype, semantic type, task, route, model, and language into a theoretical Cartesian product.
The manifest's type contracts document admission boundaries, not universal model acceptance.

For an added or changed public symbol:

1. Update the public implementation and meaningful behavior tests.
2. Regenerate Python API/wire snapshots using the commands in CONTRIBUTING, and collect Java declarations with `scripts/snapshot_java_api.py` after compilation.
3. Update its `api_inventory` mapping and the group's ownership, applicability, test links, and language exceptions.
4. Add an `assertions` list to that symbol's inventory record.
   Each item contains `test`, an exact test already in the group's language-specific test list, and `behavior`, the observable assertion it protects.
   Shared tests are valid when each covered symbol names its relevant assertion explicitly.
5. Update affected scenario/capability requirements and type contracts without silently narrowing their scope.
6. Record each changed manifest record in `contract_changes`, with the actual merge-base commit and a specific reason.

For example, a new result accessor can name the existing test that checks its output columns and retained ownership.
Merely mapping it to a broad result group is insufficient.
The gate detects Python callable signature changes from snapshots and Java additions/signature changes from compiled declaration identities.
Unchanged historical symbols retain their existing group coverage without claiming that all have direct symbol assertions.
Review must still judge assertion quality, behavior-only changes, and compatibility implications that inventories cannot infer.

A removal needs a deprecation or migration decision and replacement coverage where appropriate.
Deleting its tests and inventory entry does not remove the merge-base decision requirement.
Language-specific APIs need a reasoned applicability exception, not an empty successful test run.

## Check the change

Run focused tests first, then the applicable maintained checks from CONTRIBUTING.
Generate Java inventory after compiling the API, and run the existing gate against the actual target:

```bash
.venv/bin/python scripts/snapshot_java_api.py --out java/target/public-api.json
.venv/bin/python tests/parity/ledger.py --base origin/main --java-inventory java/target/public-api.json
.venv/bin/python -m pytest tests/parity/test_ledger.py -q
```

Substitute the real pull request target for `origin/main` when necessary.
CI must have enough Git history to find the merge base.
The ledger checks declaration ownership and scope; existing API/JUnit gates require actual outcomes, and the designated live lane executes live requirements.
Skipped or missing required tests do not establish acceptance.

Never hand-edit generated API code, golden snapshots, attribution notices, or schema outputs.
Use their maintained generators and retain compact independent regressions, not migration archives or generated view artifacts.
Before review, inspect the diff, test results, compatibility decisions, and applicability changes together.
Final correctness and comparable cold/warm benchmarks run after integration; report timings and memory without performance tuning based on this campaign.
