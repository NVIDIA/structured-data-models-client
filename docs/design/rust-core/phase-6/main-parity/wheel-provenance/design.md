<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prove CI tests the wheels it built

**Gap:** some provenance assertions from main are missing, although explicit wheel-file installation is retained.
There is no evidence that current CI actually tested the wrong wheel.

## Applicability (source-audited)

| Dimension | Scope |
|---|---|
| Models/connectors | All capabilities shipped in the three Python distributions indirectly benefit from trustworthy artifact tests. The check itself runs no connector or model. |
| Language | Python distributions only, including the bundled Rust native module. Java JAR provenance is a separate existing artifact concern, not covered by Python `direct_url.json`. |
| Sampling | None: no in-memory or push-down behavior changes, and a passing provenance check does not establish either one's functional parity. |

Exercise each affected installation workflow, not a costly connector-by-model cross-product for an assertion about installed files.

## Purpose and history

A published package can have the same version as a candidate build.
Tests must exercise the candidate, not silently resolve the published version or import the checkout.
Main `09ed6af4b` added file-based installs, exact installed-artifact directory checks and distinct distribution checks.
This is documented in CI code and the commit, not a user-facing API or notebook.

## Small design

Keep this entirely in CI/build tooling, outside Rust and host product APIs.
After dependencies/extras are installed, check all three maintained distributions: client, API and connectors.
Require distinct expected distribution names, one agreed version, imports from the installed environment, and `direct_url.json` pointing to the exact resolved artifact path for that job.
Parse file URLs correctly, including escaping and platform path rules; a matching basename alone is insufficient.
Retain existing native wheel SHA/commit checks; do not restore the retired engine/PQL distributions or build a new provenance framework.

## Tests and acceptance

Use small fixtures to reject a missing distribution, duplicate name and same wheel filename from another directory.
Accept the intended artifact path, including spaces, and fail a source-checkout import.
Run the existing installed-wheel job against its actual artifacts and check all three packages after extras resolution.
No model endpoint or GPU is needed.

Evidence: main `09ed6af4b`; `.github/workflows/ci.yml` build/installed-wheel/runtime-install jobs and `scripts/rust_core/wheel_smoke.py`.
