# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``handoff.py``'s previous-round Rust tests build in a target directory
of their own (T3.3): built into the caller's, the previous tree's
artifacts were newer than the candidate's sources, so the next build took
them as fresh and a module the candidate adds was not found
(``sdm_testkit::standin``, T3.3's fix round 1)."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/rust_core'))
_spec = importlib.util.spec_from_file_location(
    't33_handoff', ROOT / 'scripts/rust_core/handoff.py'
)
assert _spec is not None and _spec.loader is not None
handoff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(handoff)


def test_the_previous_round_builds_apart_from_the_candidate() -> None:
    root = Path('/repo')
    assert (
        handoff.previous_cargo_env({'CARGO_TARGET_DIR': '/t/x/'}, root)[
            'CARGO_TARGET_DIR'
        ]
        == '/t/x-previous'
    )
    assert handoff.previous_cargo_env({}, root)['CARGO_TARGET_DIR'] == (
        '/repo/rust/target-previous'
    )
    # Every cargo run on the previous tree takes it.
    source = inspect.getsource(handoff.step_fails_before)
    assert (
        source.count("'cargo', 'test'")
        == source.count('previous_cargo_env(os.environ, root)')
        == 2
    )
