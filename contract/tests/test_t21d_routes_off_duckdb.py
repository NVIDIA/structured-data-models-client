# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1d's runner change for the routes off DuckDB (T2.2's test
author's): python-core forces, off DuckDB, every route of each public call
a case makes through its ``until``, so no call of a core leg takes the old
route there; on DuckDB a case's routes are its own.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from sdm_client import _routing

ROOT = Path(__file__).resolve().parents[2]


def _module(path: Path, name: str) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


RUNNER = _module(ROOT / 'contract/runners/run_python.py', 't21d_run_python')


def _case(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def _on(paths: tuple[str, ...], database: str) -> set[str]:
    return {RUNNER.on_database(path, database) for path in paths}


def test_off_duckdb_every_call_a_case_makes_is_routed() -> None:
    r"""An S0 case (validation alone routed, ``until`` materialize) forces
    ``from_source`` and ``materialize`` on SQLite too, not ``predict``; a
    case DuckDB stops at ``from_source`` (``nested.map``, ``until``
    predict) forces all three; every forced path is one the client routes;
    and on DuckDB each case's routes are its own."""
    s0 = _case('contract/cases/s0/s0.accept.binary-classification.json')
    nested = _case('contract/cases/s1/nested.map.json')
    from_source = _on(_routing.FROM_SOURCE_DUCKDB, 'sqlite')
    materialize = _on(_routing.MATERIALIZE_DUCKDB, 'sqlite')
    predict = _on(_routing.PREDICT_DUCKDB, 'sqlite')
    forced = set(RUNNER.routes_on(s0, 'sqlite'))
    assert set(s0['routes']) <= forced
    assert from_source | materialize <= forced
    assert not predict & forced
    assert from_source | materialize | predict <= set(
        RUNNER.routes_on(nested, 'sqlite')
    )
    for case in (s0, nested):
        assert RUNNER.routes_on(case, 'duckdb') == case['routes']
        assert set(RUNNER.routes_on(case, 'sqlite')) <= set(_routing.PATHS)
