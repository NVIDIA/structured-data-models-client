# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Which implementation this process runs: the Rust core, or the old Python.

Internal and short-lived. The core is what every caller gets. Setting
``SDM_CLIENT_IMPLEMENTATION=python`` before ``sdm_client`` is first imported
runs the Python implementation instead, with the relational engine and
``predictive_query``, so the two can be compared side by side. It is read once,
for the whole process, and is not public API: it goes away with the old code.
"""

from __future__ import annotations

import os

#: The environment variable that chooses the implementation.
SELECTOR = 'SDM_CLIENT_IMPLEMENTATION'

#: Whether this process runs the old Python implementation.
OLD = os.environ.get(SELECTOR, '').strip().lower() == 'python'
