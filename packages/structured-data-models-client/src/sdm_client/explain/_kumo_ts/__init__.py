# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Kumo-TS's forecast attribution and its report, as that package wrote them.

Adapted from NVIDIA/Kumo-TS at af74798bbf7d020d3298f545fb9be383560a18ce.
Kept together and kept as written: an attribution is a statement about what a
model did, and rewriting the arithmetic that makes it would make it a
statement about something else.

Only the imports are rewired, to reach each other here. These modules are
excluded from type checking for the same reason the generated parser is
excluded from linting: they are not this repository's to edit.
"""
