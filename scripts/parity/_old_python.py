# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What the recorders of today's Python share while the Rust core is what
``sdm_client`` runs.

The dialect and sampler captures record calls into the relational engine,
which ``sdm_client`` reaches only on the old path, chosen by
``SDM_CLIENT_IMPLEMENTATION=python`` before it is first imported. And the
scenarios a fix in the core re-recorded with the core's answer
(``tests/rust_core/shadow/fix_changes.json``) cannot replay there as stored:
today's Python stops where the fix starts. Their calls up to that point are
still today's, so the captures keep them.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The environment variable choosing the implementation (``sdm_client._impl``).
SELECTOR = 'SDM_CLIENT_IMPLEMENTATION'

#: The scenarios a fix in the core re-recorded, with why.
FIX_CHANGES = REPO / 'tests' / 'rust_core' / 'shadow' / 'fix_changes.json'


def use_old_python() -> None:
    r"""Make this process run today's Python, or fail if it already runs the
    core."""
    impl = sys.modules.get('sdm_client._impl')
    if impl is not None and not impl.OLD:
        raise SystemExit(
            f'sdm_client already runs the core here; run this with {SELECTOR}=python'
        )
    os.environ[SELECTOR] = 'python'


def fixed_scenarios() -> frozenset[str]:
    r"""The scenario names a fix in the core re-recorded."""
    if not FIX_CHANGES.is_file():
        return frozenset()
    return frozenset(json.loads(FIX_CHANGES.read_text(encoding='utf-8')))
