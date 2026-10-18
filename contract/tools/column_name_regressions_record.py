#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's answers to T2.18's round 1 regressions into
``contract/column-name-regression-vectors.json``.

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root, ``SDM_POSTGRES_TEST_DSN`` naming a Postgres 17 server
the recorder may create databases on::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/column_name_regressions_record.py

Written by T2.18's test author from the round 1 review (Claude's P2-2 and
P3-3). Each scenario is the ``aggregation`` role of
``column_names_record.py`` (orders holding a DOUBLE column named
``a<character>b``, or its twin ``a_b``), on DuckDB, SQLite and Postgres:

- ``task``: the role's own query, ``materialize`` called with
  ``task='regression'``, which every relational query refuses (P2-2);
- ``typo``: ``PREDICT SUM(orders.pricee, 0, 30, days) FOR EACH
  users.user_id``, a misspelt ordinary column over the same schema, whose
  diagnostic names the misspelt column (P3-3).

The answer is the refusal's class, ``StructuredDataError.code``, the call
it ended and its message: the message is the diagnostic these vectors
hold, so it is recorded here, as no case observation records one (H17).
:func:`run` is imported by ``contract/tests/test_t218_r1_regressions.py``,
so ``python-core`` is run by the same code.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/column-name-regression-vectors.json'
FORMAT = 'sdm-column-name-regression-vectors/1'
TYPO = 'PREDICT SUM(orders.pricee, 0, 30, days) FOR EACH users.user_id'


def _names() -> Any:
    name = 'column_names_record'
    if name not in sys.modules:
        path = ROOT / 'contract/tools/column_names_record.py'
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
    raise ValueError(f'no scenario {kind!r}')


def vectors() -> list[dict[str, Any]]:
    r"""Every vector and twin, by database, then scenario, the twin first."""
    out = []
    for database in N.DATABASES:
        for kind in ('task', 'typo'):
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
                PredictiveQuery(vector['pql'], schema).materialize(**arguments)
            except Exception as error:  # noqa: BLE001 - the call's outcome
                return {
                    'result': 'refusal',
                    'stage': stage,
                    'error': type(error).__name__,
                    'code': getattr(error, 'code', None),
                    'message': str(error),
                }
        return {'result': 'success', 'stage': stage}
    finally:
        cleanup()


def main() -> int:
    reference = Path.home() / 'code/sdmc-reference'
    commit = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    import sdm_client

    if not Path(sdm_client.__file__).resolve().is_relative_to(reference):
        print("sdm_client is not the reference checkout's", file=sys.stderr)
        return 1
    if not os.environ.get('SDM_POSTGRES_TEST_DSN'):
        print('SDM_POSTGRES_TEST_DSN names no server', file=sys.stderr)
        return 1
    recorded = []
    for vector in vectors():
        recorded.append({**vector, 'reference': run(vector)})
        print(vector['id'], recorded[-1]['reference']['code'], file=sys.stderr)
    document = {
        'format': FORMAT,
        'recorded_by': 'contract/tools/column_name_regressions_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'environment': {'python': platform.python_version()},
        'twin': N.TWIN,
        'vectors': recorded,
    }
    OUT.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    print(f'{len(recorded)} vectors written', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
