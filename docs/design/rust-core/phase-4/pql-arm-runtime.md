<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Targeted native Arm PQL evidence

Dispatch existing `ci.yml` on the reviewed phase commit with `pql_runtime_only=true` and `fresh=false`.
The opt-in mode builds only the shipped Arm wheel and runs Python 3.10 floor and Python 3.14 newest PQL journeys on native `ubuntu-24.04-arm` runners.
Normal push, pull-request, scheduled and default dispatch behavior is unchanged.
The mode has a separate concurrency group and cannot replace a full CI or phase-end result.
Watchdog nightly selection and phase-end run selection explicitly exclude its display title, including caller-supplied phase-end run IDs.

Each lane installs the candidate wheel and records dependency versions and wheel provenance.
It invokes the canonical public recorder with the pinned frozen checkout before running candidate tests outside the source tree against those same-runtime expectations.
Missing producer bindings fail explicitly; they are not replaced with guessed scalar mappings or skipped tests.
Artifacts retain reference receipts, runtime and wheel identities, resolved requirements and candidate JUnit results even on failure.
The run may first supply evidence for a missing Arm producer binding; a later reviewed candidate must pass the lane before native Arm acceptance is complete.
These artifacts are PQL-specific evidence, not the full native binding or phase-end suite.
