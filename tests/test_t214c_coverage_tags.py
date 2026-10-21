# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T2.14c's exit on the case files (the package's test author's): the
``float64/target`` claim T2.13's tests dropped from eight cases (1e89eacc:
"coverage cannot place") is back on them, and ``coverage.py --check``
holds it, now that T2.16a's coverage.py reads the target of an EVALUATE,
an EXPLAIN and a named-entity query (``tests/test_t216a_coverage_target.py``
holds that reading, and the ninth case staying unclaimed).

The ``--check`` run is this repository's HEAD with each of the eight cases
claiming the target, written as a commit whose objects live in a temporary
object directory (nothing of the repository is written), so it holds the
claim whether or not the case files carry it yet.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _seeds() -> ModuleType:
    r"""T2.16a's coverage test module, for its seeding and its names."""
    path = ROOT / 'tests/test_t216a_coverage_target.py'
    name = 't214c_coverage_seeds'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_the_eight_cases_claim_the_target_again() -> None:
    seeds = _seeds()
    for case_id, form in seeds.RESTORED.items():
        case = seeds._case(case_id)
        assert form in case['input']['pql'], case_id  # the form it holds
        assert seeds.CLAIM in case['cells'], (
            f'{case_id}: {seeds.CLAIM} is not restored'
        )


def test_rule_6_holds_the_restored_claims(tmp_path: Path) -> None:
    seeds = _seeds()
    done = seeds.seeded(
        tmp_path,
        {
            case_id: seeds._claiming(seeds._case(case_id))
            for case_id in seeds.RESTORED
        },
    )
    failing = seeds._rule_6(done)
    assert not failing, '\n'.join(failing)
    assert done.returncode == 0, done.stdout + done.stderr
