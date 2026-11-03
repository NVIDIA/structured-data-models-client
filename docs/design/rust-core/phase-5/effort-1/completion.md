<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Effort 1 completion

The compatibility contract now has API ownership documentation, mappings for 158 recorded Python symbols and 307 compiled Java API identities, and explicit task/type coverage requirements.
It extends the existing manifest and runners.

Changes to APIs, applicability, required tests or fixture/type-vector content require a recorded explanation against the target merge base.
Missing mappings, missing required tests, skipped required results and duplicate results fail the checks.
Java source/binary compatibility uses japicmp; shared semantic-type assignment is checked in Rust.

| Local validation | Result | Time |
|---|---|---|
| Python API behavior | 430 passed, no skips | 17.94 s |
| Java API behavior, including local databases | 103 passed, no skips | 33.83 s |
| Python type-contract checks | 202 passed | 74.00 s |
| Rust provider/type/schema checks | 72 passed, no ignored tests | 2.32 s warm |
| Parity and JUnit checker regression tests | 75 passed | 12.97 s |
| Final focused enforcement checks | 18 passed | 2.66 s |
| Java API comparison against `rust-core` | Passed | 12.80 s warm |
| Workflow checks | 199 passed; one integration failure corrected and its test rerun successfully | 120.58 s + 0.19 s |

The installed Python sources match this checkout; Java tests used the preserved Phase 4 native artifact whose Rust sources match, apart from formatting and the new test.
CI uses the same commands with its own freshly built artifacts.
The CI workflow itself has not been run remotely for this change.

Repository hygiene, changed-file lint and Rust formatting checks pass.
Independent design and implementation reviews completed; the identified coverage gaps were corrected.
No new live endpoint evidence is claimed, and these gates do not prove every type combination.
[Remaining Phase 5 acceptance](../verification.md), legacy removal and eventual merge into `main` are separate work.
