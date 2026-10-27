# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A package lands on its handoff stamp and two approving reviews, with no
CI run (the owner's "2026-10-22 evening: CI off the landing path"; T3.2
fix 4): ``land.py`` with both reviews and a stamp, and no CI run at all,
lands and records the stamp in its ``Handoff-stamp`` trailer; the same
landing without the stamp is refused, naming ``handoff.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/rust_core'))


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Its dataclasses look their module up by name.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SELFTEST = _module(
    ROOT / 'scripts/rust_core/land_selftest.py', 't32_land_selftest'
)


def test_a_package_lands_on_its_stamp_and_two_reviews_with_no_ci_run(
    tmp_path: Path,
) -> None:
    (tmp_path / 'stamped').mkdir()
    landed = SELFTEST.Case(tmp_path / 'stamped', SELFTEST.Fake(runs={})).land()
    assert landed.code == 0, landed.text
    assert f'LANDED {SELFTEST.LAND}' in landed.text
    message = landed.fake.committed[0]
    assert f'Handoff-stamp: {SELFTEST.CAND} {SELFTEST.STAMP_TIME}' in message
    assert not [call for call in landed.calls if call[0] == 'gh']

    (tmp_path / 'unstamped').mkdir()
    fake = SELFTEST.Fake(runs={}, stamped=set())
    refused = SELFTEST.Case(tmp_path / 'unstamped', fake).land()
    assert refused.code == 2, refused.text
    assert 'handoff.py run' in refused.text.partition('REFUSED:')[2]
    assert refused.untouched()
