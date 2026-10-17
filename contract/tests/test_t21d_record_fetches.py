# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1d: python-core's ``--record-fetches`` names, for each fetch the
core asked, the statements the driver ran for it, in order: the driver
capture's first statements, one per statement of each fetch's plan, a
call left on the old route running after them; a capture holding fewer
than the plans is a runner failure, never a recording that misnames them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_python = _module('run_python', ROOT / 'contract/runners/run_python.py')


def _fetch(statements: int) -> dict[str, Any]:
    return {
        'request': {'request': 'scan'},
        'plan': {'plan': 'sql', 'statements': [{}] * statements},
        'data': '',
    }


def _driver(texts: list[str]) -> dict[str, Any]:
    return {'capture': {'statements': [{'sql': t} for t in texts]}}


def test_each_fetch_names_the_statements_run_for_it() -> None:
    capture = SimpleNamespace(answered=[_fetch(2), _fetch(1)])
    # A third statement after the core's, of a call on the old route.
    driver = _driver(['SELECT setseed(0.5)', 'SELECT a', 'SELECT b', 'old'])
    fetches = run_python._with_statements({'id': 'c'}, capture, driver)
    assert [f['statements'] for f in fetches] == [
        ['SELECT setseed(0.5)', 'SELECT a'],
        ['SELECT b'],
    ]
    with pytest.raises(run_python.RunnerFailure, match='2 statements for 3'):
        run_python._with_statements(
            {'id': 'c'}, capture, _driver(['SELECT a', 'SELECT b'])
        )
