# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The real-data benchmark's verdict fails closed (T1.32e part 1, the
implementer's seeded faults, decision E49; the test author's seeds are
``test_t132e_benchmark_seeds.py``): a unit whose evidence is missing,
empty or retyped is never a pass. Offline: no data, no network.

    python -m pytest contract/tests/test_real_data_benchmark_faults.py -q
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _benchmark() -> ModuleType:
    name = 'real_data_benchmark'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'scripts/live/real_data_benchmark.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


TRUTH = [True] * 4 + [False] * 16


def scored(below: int) -> list[dict[str, Any]]:
    r"""Answers of AUROC ``4 * below / 64`` (the seeds' construction)."""
    answers = [{'prediction': True, 'score': 0.5} for _ in range(4)]
    for at in range(16):
        score = 0.01 * (at + 1) if at < below else 0.6 + 0.01 * at
        answers.append({'prediction': score > 0.5, 'score': score})
    return answers


GOOD = [scored(16), scored(16)]


def test_the_control_passes() -> None:
    assert _benchmark().judge('binary', TRUTH, GOOD, GOOD) == 'PASS'


@pytest.mark.parametrize(
    'core',
    [
        pytest.param([], id='no-seed'),
        pytest.param([scored(16)], id='a-seed-missing'),
        pytest.param([[], []], id='no-entity'),
        pytest.param(
            [[{**a, 'score': math.nan} for a in scored(16)]] * 2,
            id='a-score-not-a-number',
        ),
    ],
)
def test_a_core_without_evidence_fails(core: list[Any]) -> None:
    assert _benchmark().judge('binary', TRUTH, GOOD, core) == 'FAIL'


@pytest.mark.parametrize(
    'old',
    [
        pytest.param([], id='no-seed'),
        pytest.param([[], []], id='no-entity'),
        pytest.param([[{'prediction': 1.0}] * 20] * 2, id='no-score'),
    ],
)
def test_an_old_route_without_evidence_is_inconclusive_or_refused(
    old: list[Any],
) -> None:
    # Never a pass: inconclusive where its metric is undefined, an error
    # where its answers are not what the metric reads.
    try:
        found = _benchmark().judge('binary', TRUTH, old, GOOD)
    except (KeyError, TypeError, ValueError):
        return
    assert found == 'INCONCLUSIVE'


def test_a_retyped_regression_answer_is_refused() -> None:
    truth = [1.0, 3.0, 5.0, 7.0]
    good = [[{'prediction': t + 0.1} for t in truth]] * 2
    retyped = [[{'prediction': 'many'} for _ in truth]] * 2
    with pytest.raises(ValueError):
        _benchmark().judge('regression', truth, good, retyped)
    nan = [[{'prediction': math.nan} for _ in truth]] * 2
    assert _benchmark().judge('regression', truth, good, nan) == 'FAIL'
