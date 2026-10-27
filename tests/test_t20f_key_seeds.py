# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0f's own seeded faults for the ratchet's key rule (``check_keys``;
the implementer's, beside the test author's
``tests/test_t20f_case_keys.py``, whose fixtures they reuse): a baseline
keyed by an unknown database, a bare baseline of no case, and a reference
line whose key is no string each fail by name; the twin, the reference's
``X@sqlite`` and ``X@postgres`` lines, passes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[1]


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


keys = _module(
    ROOT / 'tests/test_t20f_case_keys.py', 't20f_key_seeds_case_keys'
)
origin = keys.origin
recorded = keys.recorded


def _baseline_keyed(runner: str, key: Any) -> Any:
    def edit(tree: Path) -> None:
        path = keys._baseline_file(tree, runner)
        lines = keys._jsonl(path)
        added = dict(next(x for x in lines if x['case'] == keys.X))
        added['case'] = key
        lines.append(added)
        keys._write_jsonl(path, lines)

    return edit


def _reference_keyed(key: Any) -> Any:
    def edit(tree: Path) -> None:
        path = keys._reference_file(tree)
        lines = keys._jsonl(path)
        added = dict(next(x for x in lines if x['case'] == keys.X))
        added['case'] = key
        lines.append(added)
        path.write_text(''.join(json.dumps(x) + '\n' for x in lines))

    return edit


@pytest.mark.parametrize(
    ('seed', 'word'),
    [
        pytest.param(
            _baseline_keyed('rust-host', f'{keys.X}@oracle'),
            f'{keys.X}@oracle',
            id='baseline-unknown-database',
        ),
        pytest.param(
            _baseline_keyed('rust-host', 'nocase.at-all'),
            'nocase.at-all',
            id='baseline-of-no-case',
        ),
        pytest.param(
            _reference_keyed(7),
            'a key of type int',
            id='reference-key-retyped',
        ),
    ],
)
def test_a_line_keyed_by_no_case_and_database_fails_by_name(
    recorded: Path, seed: Any, word: str
) -> None:
    r"""Seeded: the tree rule, with no observations. Twin: the reference's
    ``X@sqlite`` and ``X@postgres`` lines (the landing's) pass."""
    landed = keys.git(recorded, 'rev-parse', 'HEAD')
    keys.commit(recorded, 'T2.0f: the twin')
    assert keys.helpers.check(recorded, landed) == []
    keys.commit(recorded, 'T2.0f: a seed', seed)
    found = keys.helpers.check(recorded, landed)
    assert found, 'a line keyed by no case and database passed'
    assert all(word in p for p in found if p.startswith('cases:')), found
    assert any(word in p for p in found), found
