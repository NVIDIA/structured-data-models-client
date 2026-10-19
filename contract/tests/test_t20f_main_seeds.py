# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0f's own seeded faults for the ``main`` step (the implementer's,
beside the test author's ``contract/tests/test_t20f_gate.py``, whose fixtures they
reuse): ``contract/reference-env.txt`` naming no reference commit, and a
reference commit the clone does not hold, each fail the step by name, so
the rule fails on its evidence missing, not only on main moved. It runs a
real gate.sh, which needs the built native core, so it lives under
contract/tests, run in the core's contract-python job.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate = _module(
    ROOT / 'contract/tests/test_t20f_gate.py', 't20f_main_seeds_gate'
)
GM = gate.GM
origin = GM.origin
observed = GM.observed
stubbed = GM.stubbed
stubs = GM.stubs
mainline = gate.mainline


def _reference_env(edit: Any) -> Any:
    def change(tree: Path) -> None:
        path = tree / 'contract/reference-env.txt'
        path.write_text(edit(path.read_text()))

    return change


@pytest.mark.parametrize(
    ('seed', 'word'),
    [
        pytest.param(
            lambda text: re.sub(r'^reference: .*$', '', text, flags=re.M),
            'reference-env.txt',
            id='no-reference-line',
        ),
        pytest.param(
            lambda text: re.sub(
                r'^reference: [0-9a-f]{40}$',
                'reference: ' + 'f' * 40,
                text,
                flags=re.M,
            ),
            'not in this clone',
            id='reference-not-in-the-clone',
        ),
    ],
)
def test_the_main_step_fails_on_a_reference_it_cannot_read(
    stubbed: Path,
    tmp_path: Path,
    observed: Any,
    stubs: dict[str, Any],
    mainline: dict[str, str],
    seed: Any,
    word: str,
) -> None:
    r"""Seeded: with ``origin/main`` at R (the twin, which passes, is the
    test author's), the candidate's reference-env.txt names no commit, or
    one the clone lacks: the main step fails, naming it."""
    gate._refs(stubbed, origin_main=mainline['R'], local_main=None)
    GM.commit(stubbed, 'T2.0f: a seed', _reference_env(seed))
    done = gate.run_locked(stubbed, stubs, observed, tmp_path)
    assert GM.summary(done).get('main') == 'FAIL', done.stdout + done.stderr
    assert word in gate.step_log(done, 'main'), done.stdout
    assert GM.table(done, '2a')['7'] == 'FAIL'
