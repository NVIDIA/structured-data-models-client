<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Effort 3: verify, build and publish

**Implementation in progress.**
Finish Phase 5 with a tested client ready for `main`, then publish Python and Java packages.
Keep the current architecture and test tools.

```text
Reuse valid results → test gaps → build all platforms → final CI → publish → verify installation
```

## 1. Finish verification

Update [verification.md](../verification.md), using the existing compatibility matrix in `tests/parity/manifest.json`.

| Work | What to do |
|---|---|
| Reuse previous tests | Check which Phase 3/4 and Effort 2 results still apply after cleanup; record the source version and scope, with two independent reviews for reused live results |
| Close correctness gaps | Run missing or affected API, connector and prediction cases; use the same data/model version across Python and Java, without repairing reader output in tests |
| Check performance | Measure missing or affected operations for latency and peak memory; compare with recorded baselines and preserve previously approved costs |
| Close original requirements | Map archived correctness, runtime and performance requirements to proof or an explicit owner decision; do not restore obsolete tests just to match the old client |
| Validate final source | Reconcile changes from `main`, then run required Python/Rust/Java CI, API coverage, lint and type checks on the final candidate |

Reuse existing runners and build artifacts; do not rerun every live test on every CPU.
Keep GPU/warehouse checks outside per-commit CI; local live validation is sufficient for its recorded scope.
Reproduce real bugs through the public API before fixing them; missing required results remain incomplete.

### Main reconciliation: inference option admission

Live checks found that invalid options return `VALIDATION_FAILED` remotely but `INVALID_REQUEST` locally.
Restore main's early rejection contract with one Rust helper for `use_prediction_time`, `run_mode` and `inference_config`.
Use it for direct/gathered predictions and fitted sessions in both languages; retain model-specific checks separately.
Remove the duplicate Python typing helper after ordinary options reach Rust on every path, including paths that currently replace prepared options in Python.
Keep NumPy-to-native conversion at Python admission; do not change sampling, SQL or add callbacks.
Verify invalid options cause no transport call, and accepted options work against the endpoint and local models.

### Cross-platform index ordering: owner decision pending

The current Python path recognizes specific Linux NumPy binaries to reproduce their sorting and comparison behavior.
That cannot support the promised macOS/Windows PQL paths without more platform-specific implementations.
Even a unique source index can acquire duplicate labels during sampling.

| Case | Current behavior | Recommended shared Rust rule |
|---|---|---|
| Sampled labels `[2, 1, 1]` | Match the installed NumPy sort; unknown binaries are refused | Sort by label, retaining sampled order between equal labels |
| Object labels containing NumPy numbers | Match that binary's numeric comparison rules | Compare supported numeric values with the same documented rules in both languages |
| Ordinary unique built-in labels | Already handled without a NumPy binary profile | Preserve their ordering |

```text
Python/Java values → Rust compares and orders labels → same sampled rows → model
```

Recommendation: approve deterministic shared ordering as a bounded improvement.
Ties can change context row order and therefore model output; validate those differences explicitly with live examples.
This changes neither sampled membership nor warehouse push-down SQL.
Custom comparison functions remain unsupported.
Until approved, retain existing ordering and audit the missing platform producers; do not claim full platform support.

## 2. Build the complete distribution set

“Configured” below means the workflow exists, not that this release has passed.

| Platform | Python today | Java today | Required change |
|---|---|---|---|
| Linux Intel/AMD x86-64 | Configured | Configured | Verify final builds |
| Linux ARM64 | Configured | Configured | Verify final builds |
| macOS Intel x86-64 | Configured | Configured | Verify final builds |
| macOS Apple ARM64 | Configured | Configured | Verify final builds |
| Windows Intel/AMD x86-64 | Configured | Configured | Verify final builds |
| Windows ARM64 | Deferred by owner | Deferred by owner | Wait for upstream Arrow dependencies |

**Files to change:** `.github/workflows/ci.yml`, `.github/workflows/publish.yml`, `scripts/rust_core/java_build.py`, Java native loading and `java/pom.xml`.
Extend existing artifact checks to reject releases missing any of the five targets for each language.

- **Python:** five client wheels using the stable Python ABI (`cp310-abi3`), usable on standard Python 3.10–3.14; one platform-independent wheel each for API and connectors, all at the same version.
- **Java:** one JAR containing all five native libraries, with its POM/dependency metadata; verify JDK 17 and 21 support.
- **Each platform:** install the built package in a clean environment, load Rust and run a small public-API test; check optional connector drivers and document any platform restrictions.
- **Artifact contents:** keep checksums, build/source identities, dependency/license records and checks excluding retired packages and test-only features.

Native Windows ARM64 Python is deferred by owner decision: PyArrow has no published wheel there, and Python 3.10 also lacks NumPy/pandas wheels.
Java Windows ARM64 is also deferred by owner decision: Arrow Java 19.0.0 has no ARM64 Windows DLL.
Verify the remaining targets’ minimum OS versions and dependencies.
Use current Linux glibc builds; 32-bit, Alpine/musl, PyPy and free-threaded Python are outside this release matrix.
Keep API/connector source archives; explicitly decide whether to add a client source archive, which the current release workflow does not produce.
Reuse the heavy-job guard and Cargo jobs=2; run large-memory checks sequentially.

## 3. Publish and verify what users receive

| Work | Concrete result |
|---|---|
| Fix release instructions | Update `RELEASING.md`: three Python distributions, Java, version/tag steps and recovery after partial publication; remove stale five-package/engine instructions |
| Publish Python | Extend the existing protected release and owner-run index publishing process to include all five wheels |
| Publish Java | Add JAR/POM publication, required repository metadata and access; current `publish.yml` publishes no Java package |
| Verify publication | Install from the actual Python index and Java repository in clean consumers; confirm versions, file hashes, dependencies and native loading |
| Record completion | Short `completion.md` with final commit, reused/new test results, performance findings, release links and any remaining limits |

**Resolve before release:** version, Java publishing destination/credentials and protected release approval.
Report merge readiness before merging/tagging; preserve existing release protections.
Independent review must confirm the final evidence and artifact set.

**Done:** required tests pass, all five Python and Java targets build and load, and Python/Java packages are published and installable.
Unproven requirements need an explicit owner decision; if publication is blocked, report “validated, publication pending”.
