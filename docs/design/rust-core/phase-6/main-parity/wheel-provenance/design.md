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

## Implemented and checked

CI now requires exactly one candidate for each of the three distributions, at one version, and checks exact installed artifact paths plus import resolution after extras installation.
The general installed-wheel, runtime-install and CI/release SBOM paths share the same small standard-library check.
Native wheel manifest/hash checks remain unchanged; no product API or engine code changed.

- 13 focused tests passed, including wrong-directory same-name wheels, missing/duplicate distributions, version mismatch, escaped paths and source-checkout shadowing.
- Real wheels installed with the installed-wheel job's extras passed; reinstalling the identical client wheel from another directory reproduced the old false pass and was rejected by the new check.
- All three packages and the native module imported from the isolated environment; Ruff and targeted mypy passed.
- Independent review found no material issue.
- The existing CI shell tests were updated for the shared checker while preserving swapped-wheel, dependency-resolution, NumPy and pandas fault controls; 67 combined CI/provenance tests and 24 related parity/release workflow tests passed.

The API/connectors wheels were built from package sources at `10f47a63c`; their SHA256 values are `861dc4d8109704f749c2fdc78d23d97d00b6a720c5bca66c32b4b3ba6eb80e7d` and `e2325ec9cde3957fbe2b5d07c1dcade4f39ed62b1236138e4facb249acf028dd`.
The initial negative-control client wheel is SHA256 `00d6f371cffc035871164c2044db2fd5c2df751601d4a6db95d7723f97e19dc5`; its existing manifest names `b6e238bb4d4a53674cfc50ac76aa8b1e36586991` and `tree_clean: false`.
That initial artifact was valid input for installation-path tests, not a clean-release build claim.
Final verification reinstalled the clean temporal-fix client from `e74b32178699dfd3e4580a18e95e35ac1bf90031`, SHA256 `9f35eee4b2e414ae7144b3a22497f64d1ada8694c642ebc48b47d321fae9dfda`, with manifest `tree_clean: true`.
The exact three-wheel provenance check and native import passed again in the extras-resolved environment; no native rebuild was performed for this task.
Local logs and artifact identities are under `/tmp/sdm-p6-wheel-provenance/`.
The full GitHub CI/release workflow and cross-platform runtime matrix were not rerun for this tooling-only change; release publishing was not invoked.
