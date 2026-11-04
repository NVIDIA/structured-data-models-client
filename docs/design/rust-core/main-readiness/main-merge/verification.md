<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Main merge verification

Status: implementation, independent review and final checks passed on Phase 7.
Integrated into `rust-core` in merge commit `711968587`.
Merge parents are Phase 7 `ff08b03d7` and main `847a1e9e3`.
The [file disposition](files.md) covers every incoming file, including clean merges.

## What changed

| Contract | Resolution |
|---|---|
| Manual explanation collections | Python accepts caller-owned entries; Rust validates typed identities and plans merging. Java-produced results use the same engine policy. |
| EXPLAIN admission | Empty outputs remain invalid. Unsupported candidate-ranking explanations fail before warehouse reads. |
| Local runtime | Unsupported tasks/outputs fail before loading model weights. Tensor conversion stays in the optional runtime adapter. |
| Generated keys | Explicit client-only provenance survives saved requests and sessions. Synthetic keys are hidden from explanation views; real source keys remain visible. Provenance never enters model requests or consumes their size allowance. |
| Result merging | Both hosts keep prediction rows and gradient keys aligned, preserve unique caller positions, and leave ordinary unexplained merges unaffected. |
| Legacy routing | Installed Python has one explanation module, no shadow package or retired dependency. Journey tests require actual native operations and reject restored Python planning/decoding helpers. |

## Evidence and limits

- Python actual pretrained GPU models: four successful journeys, Tabular and Relational, regression and classification, with nonempty gradient views.
- Java actual pretrained GPU models: the same four successful journeys through JNI and a local model bridge.
- Java final local checks: **9 passed** – seven main-merge contracts and two existing gradient contracts, including ordinary nested merging.
- Python final maintained client/parity/skills suite: **4,909 passed, 411 skipped**, zero failures.
- Optional Python runtime: **52 passed**. Declared Snowflake dependency verification: **48 passed**.
- Rust: **566 passed** across core, request, wire and contract crates; strict Clippy passed for these crates and both bindings.
- Ruff lint/format, all 112 Python source files under mypy, repository hygiene and Rust notices passed.
- Skipped optional connectors/models are not claimed as live coverage. This validates the main-merge changes, not the separate full-matrix readiness initiative.
- Public API snapshot and all nine wire snapshots match. State metadata is asserted separately from unchanged historical model-payload bytes; old captures were not rewritten.

The broad run exposed obsolete explanation limits in historical tests and isolated subprocesses selecting an older installed client.
Those tests now assert the current supported behavior and the installed candidate under test.
The saved-request fixture explicitly disables unsupported session probing; strict historical payload/result comparisons remain unchanged.
A public near-limit materialization also proved that counting internal provenance as model bytes was wrong; the corrected limit accepts the same model body and still rejects a genuinely oversized body.
The original contract schemas were retained and version 70 was generated with the maintained schema generators.

Final v3 artifacts match the current Python module ASTs and the reviewed native sources.
The eight actual-model journeys used v2; v3 only corrected ordinary merge handling, retained nonfinite identity behavior, added an explicit existing Rust type annotation and excluded client-only metadata from model byte limits.
All those final differences passed local v3 regression checks; model execution and gradient extraction were unchanged.

| Final artifact | SHA-256 |
|---|---|
| Python wheel | `94b0ccb0f9361324054d2595d2e233c09d0d158a500325a1739e50f68a7f3968` |
| Python native library | `15c695e98b1559eaee7c25ba04ef7568b88a50976364fce663c6b79748266db8` |
| JNI library | `ddbeca7db2a2b57ead716e0f9623343c882a60ac62eedc913001b362c7793341` |

Builds were produced during the uncommitted merge and are not represented as clean-tree release artifacts.
No package was published.
