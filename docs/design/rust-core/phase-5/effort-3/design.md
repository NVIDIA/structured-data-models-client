<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Effort 3: verify and prepare the 2.0 release candidate

**RC preparation complete; merge and publication remain on hold.**
Prepare a tested 2.0.0 release candidate for review.
**Owner instruction: do not publish anything or create a release yet.**
Prepare Python `2.0.0rc1` and corresponding Rust/Java RC versions; GitHub Releases is the initial Java destination only after separate publication authorization.
Keep the current architecture and test tools.

```text
Reuse valid results → test gaps → build all platforms → final CI → verify local installs → owner review
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

### Cross-platform index ordering: approved

The previous Python path recognized specific Linux NumPy binaries to reproduce their sorting and comparison behavior.
That cannot support the promised macOS/Windows PQL paths without more platform-specific implementations.
Sampling is without replacement: unique source labels remain unique.
The gap concerns existing duplicate labels and certain NumPy-valued object comparisons; ordinary unique primitive labels already work without a binary profile.
The old Python client also sorted reduced context by index after sampling; this orders at most 10,000 context rows, not the full source table or scoring rows.

| Case | Previous behavior | Approved shared Rust rule |
|---|---|---|
| Sampled labels `[2, 1, 1]` | Match the installed NumPy sort; unknown binaries are refused | Sort by label, retaining sampled order between equal labels |
| Object labels containing NumPy numbers | Match that binary's numeric comparison rules | Compare supported numeric values with the same documented rules in both languages |
| Ordinary unique built-in labels | Already handled without a NumPy binary profile | Preserve their ordering |

```text
Python/Java values → Rust compares and orders labels → same sampled rows → model
```

Owner approved deterministic shared ordering as a bounded improvement.
Ties can change context row order and therefore model output; validate those differences explicitly with live examples.
This changes neither sampled membership nor warehouse push-down SQL.
Custom comparison functions remain unsupported.
Implemented stable ties and exact represented numeric-value comparisons in Rust.
Validation: 407 Python index tests, targeted Rust tests and Clippy passed; duplicate-label and mixed-number examples also completed live Tabular predictions.
Independent review found no remaining blocker; final cross-platform RC validation passed.
Keep legacy contract profiles readable while removing their use from the Python admission path.

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
Keep API/connector source archives.
For this RC the native client remains wheel-only, matching the supported-platform release contract; a client source distribution is outside scope.
Reuse the heavy-job guard and Cargo jobs=2; run large-memory checks sequentially.

## 3. Prepare artifacts; publication is on hold

| Work | Concrete result |
|---|---|
| RC version | Align Python `2.0.0rc1`, Rust/Java prerelease versions, internal pins and artifact checks |
| Local distribution checks | Verify clean wheel/JAR consumers, source identities, hashes, dependencies and native loading |
| Release preparation | Keep protected publishing controls; prepare GitHub JAR/POM assets without uploading or tagging |
| Record readiness | Short `completion.md` with tested commit, reused/new evidence, performance and outstanding limits |

Maven Central is a follow-up option, not required for this RC.
It needs namespace access, signing, source/Javadoc JARs and publishing configuration.
Actual index/GitHub publication and published-consumer verification wait for explicit owner authorization.
Report merge readiness before merging/tagging; no unproven gate may be marked passed.

**Ready for review:** required tests pass, all five Python and Java targets build and load, and local RC artifacts install correctly.
This is not a claim that the product is released or that publication has been verified.
