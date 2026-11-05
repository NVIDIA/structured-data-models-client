# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public query scenarios for unusual column names and refusal diagnostics."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
KINDS = ('task', 'typo', 'parts')
TYPO = 'PREDICT SUM(orders.pricee, 0, 30, days) FOR EACH users.user_id'


def _names() -> Any:
    name = 'column_names_record'
    if name not in sys.modules:
        path = ROOT / 'contract/support/column_name_vectors.py'
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


N = _names()


def scenario(kind: str, name: str) -> dict[str, Any]:
    r"""The setup, query and ``materialize`` keywords of ``kind`` over the
    aggregation role's data with its column named ``name``."""
    base = N.scenario('aggregation', name)
    if kind == 'task':
        return {**base, 'task': 'regression'}
    if kind == 'typo':
        return {**base, 'pql': TYPO}
    if kind == 'parts':
        pql = base['pql'].replace(
            ' FOR EACH ', ' FORECAST 2 TIMEFRAMES FOR EACH '
        )
        return {**base, 'pql': pql, 'most_parts': 1}
    raise ValueError(f'no scenario {kind!r}')


def vectors() -> list[dict[str, Any]]:
    r"""Every vector and twin, by database, then scenario, the twin first."""
    out = []
    for database in N.DATABASES:
        for kind in KINDS:
            out.append(
                {
                    'id': f'twin.{kind}@{database}',
                    'database': database,
                    'kind': kind,
                    'character': None,
                    'name': N.TWIN,
                    **scenario(kind, N.TWIN),
                }
            )
            for label, character in N.CHARACTERS.items():
                name = N.name_of(character)
                out.append(
                    {
                        'id': f'{label}.{kind}@{database}',
                        'database': database,
                        'kind': kind,
                        'character': label,
                        'name': name,
                        'twin': f'twin.{kind}@{database}',
                        **scenario(kind, name),
                    }
                )
    return out


def run(vector: dict[str, Any]) -> dict[str, Any]:
    r"""One vector through ``Schema.from_source`` and ``materialize`` on its
    database, on whatever route the environment gives: the outcome, with
    a refusal's class, code and message."""
    from sdm_client import PredictiveQuery, Schema

    runner = N._runner()
    case = {
        'id': vector['id'],
        'input': {'base': 'shop', 'setup': vector['setup']},
    }
    connection, cleanup, _ = runner._open(
        case, runner.load_bases(), vector['database']
    )
    stage = 'from_source'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                schema = Schema.from_source(
                    runner._source(vector['database'], connection)
                )
                stage = 'materialize'
                arguments = runner._materialize_arguments(vector['materialize'])
                if 'task' in vector:
                    arguments['task'] = vector['task']
                from sdm_client import decompose

                limit = decompose.MOST_PARTS
                decompose.MOST_PARTS = vector.get('most_parts', limit)
                try:
                    request = PredictiveQuery(
                        vector['pql'], schema
                    ).materialize(**arguments)
                finally:
                    decompose.MOST_PARTS = limit
            except Exception as error:  # noqa: BLE001 - the call's outcome
                return {
                    'result': 'refusal',
                    'stage': stage,
                    'error': type(error).__name__,
                    'code': getattr(error, 'code', None),
                    'message': str(error),
                }
        return {
            'result': 'success',
            'stage': stage,
            'parts': len(list(request.payloads)),
        }
    finally:
        cleanup()
