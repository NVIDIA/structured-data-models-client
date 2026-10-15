#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's request where sampling reaches a null key into
``contract/sampled-null-key-vectors.json`` (T1.32e part 1, from the review
of 18bd89d6: a single sampled null text key the core refused although the
reference succeeds).

Record with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/sampled_null_key_record.py

and observe a candidate's route, as ``contract/tests/test_t132e_sampled_null_keys.py``
does, with::

    python -I contract/tools/sampled_null_key_record.py --observe python-core --out FILE

Written by T1.32e's independent test author. Each vector is two generated
DuckDB tables: ``accounts`` (200 rows, a BIGINT key, ``amount``) and
``payments`` (200 rows, ``payment_id`` of the vector's type with one or
more nulls, ``account_id`` referring to an account, ``amount``), so that
``payments.payment_id`` is kept as the key (at most 1% of its rows
null) and the rows sampling reaches for account 3 hold the vector's mix
of null and non-null keys. The question is ``PREDICT accounts.amount FOR
EACH accounts.account_id``, materialized with ``indices=[3]``,
``num_context_samples=1`` and ``random_seed=0``, then predicted against
``contract/runners/scripted_endpoint.py`` in process. The case format
holds neither ``num_context_samples`` nor ``random_seed``, so these
vectors are observed through the public calls directly, not as cases,
and on the Python host only.

For each vector this script keeps the reference's answer:

- ``outcome``: success, or the refusal's class, ``StructuredDataError.code``
  and stage (``from_source``, ``materialize`` or ``predict``);
- ``keys``: each table's key as the public schema gives it;
- ``request``: the body the endpoint received, parsed with every number
  as its text, ``metadata.request_id`` left out; null when none was sent;
- ``warnings``: the warning classes, in order;

and, beside the answer and never compared, the warnings' text
(``reference_messages``; ``messages`` in an observation), read only to
find which warning names a table.

``register`` is derived as T1.32e's other vectors derive it (an unnamed
refusal or INTERNAL_ERROR); any difference passes only under an approved
register entry naming the vector with its transition
(``REGISTER_REF<id or glob> <from> -> <to>``).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/sampled-null-key-vectors.json'
FORMAT = 'sdm-sampled-null-key-vectors/1'
REGISTER_REF = 'contract/sampled-null-key-vectors.json#'
PQL = 'PREDICT accounts.amount FOR EACH accounts.account_id'
MATERIALIZE = {'indices': [3], 'num_context_samples': 1, 'random_seed': 0}

_ACCOUNTS = (
    'CREATE TABLE accounts AS SELECT i AS account_id, i::DOUBLE AS amount '
    'FROM range(200) t(i)'
)


def _payments(key: str, refers: str) -> str:
    return (
        f'CREATE TABLE payments AS SELECT {key} AS payment_id, '
        f'{refers} AS account_id, i::DOUBLE AS amount FROM range(200) t(i)'
    )


#: The vectors: the payments' key and which payments account 3 has.
VECTORS: list[dict[str, Any]] = [
    {
        # The review's reproduction, as given.
        'id': 'singleton-null-text',
        'setup': [
            _ACCOUNTS,
            _payments(
                "CASE WHEN i=0 THEN NULL ELSE 'k'||i::VARCHAR END",
                'CASE WHEN i=0 THEN 3 ELSE 150 END',
            ),
        ],
    },
    {
        # Account 3's payments: one null text key beside two others.
        'id': 'mixed-null-text',
        'setup': [
            _ACCOUNTS,
            _payments(
                "CASE WHEN i=0 THEN NULL ELSE 'k'||i::VARCHAR END",
                'CASE WHEN i<3 THEN 3 ELSE 150 END',
            ),
        ],
    },
    {
        # Account 3's payments: two non-null text keys, the null elsewhere.
        'id': 'null-text-not-sampled',
        'setup': [
            _ACCOUNTS,
            _payments(
                "CASE WHEN i=0 THEN NULL ELSE 'k'||i::VARCHAR END",
                'CASE WHEN i IN (1, 2) THEN 3 ELSE 150 END',
            ),
        ],
    },
    {
        'id': 'singleton-null-decimal',
        'setup': [
            _ACCOUNTS,
            _payments(
                'CASE WHEN i=0 THEN NULL ELSE i::DECIMAL(18, 0) END',
                'CASE WHEN i=0 THEN 3 ELSE 150 END',
            ),
        ],
    },
    {
        # Account 3's payments: two non-null decimal keys, the null
        # elsewhere (mixed-null-decimal's twin, as null-text-not-sampled
        # is mixed-null-text's).
        'id': 'null-decimal-not-sampled',
        'setup': [
            _ACCOUNTS,
            _payments(
                'CASE WHEN i=0 THEN NULL ELSE i::DECIMAL(18, 0) END',
                'CASE WHEN i IN (1, 2) THEN 3 ELSE 150 END',
            ),
        ],
    },
    {
        'id': 'mixed-null-decimal',
        'setup': [
            _ACCOUNTS,
            _payments(
                'CASE WHEN i=0 THEN NULL ELSE i::DECIMAL(18, 0) END',
                'CASE WHEN i<3 THEN 3 ELSE 150 END',
            ),
        ],
    },
    {
        # Two null text keys, both sampled: two of 200 null keeps the key.
        'id': 'two-nulls-text',
        'setup': [
            _ACCOUNTS,
            _payments(
                "CASE WHEN i<2 THEN NULL ELSE 'k'||i::VARCHAR END",
                'CASE WHEN i<2 THEN 3 ELSE 150 END',
            ),
        ],
    },
]


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _runner() -> ModuleType:
    return _module('run_python', ROOT / 'contract/runners/run_python.py')


def _request(body: bytes) -> Any:
    parsed = _runner()._payload(body)['parsed']
    parsed = copy.deepcopy(parsed)
    if isinstance(parsed, dict) and isinstance(parsed.get('metadata'), dict):
        parsed['metadata'].pop('request_id', None)
    return parsed


def observe(vector: dict[str, Any]) -> dict[str, Any]:
    r"""One vector through the public calls, on whatever route the
    environment's ``SDM_CORE_ROUTES`` names."""
    import duckdb
    from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient
    from sdm_connectors import DuckDBSource

    serve = _module(
        'scripted_endpoint', ROOT / 'contract/runners/scripted_endpoint.py'
    ).serve
    connection = duckdb.connect()
    for statement in vector['setup']:
        connection.execute(statement)
    stage = 'from_source'
    keys = None
    error: BaseException | None = None
    with (
        serve() as (url, received),
        warnings.catch_warnings(record=True) as shown,
    ):
        warnings.simplefilter('always')
        try:
            schema = Schema.from_source(DuckDBSource(connection))
            keys = {
                name: (
                    table.primary_key
                    if table.primary_key is None
                    or isinstance(table.primary_key, str)
                    else list(table.primary_key)
                )
                for name, table in sorted(schema.tables.items())
            }
            stage = 'materialize'
            request = PredictiveQuery(PQL, schema).materialize(**MATERIALIZE)
            stage = 'predict'
            with SDMClient(url) as client:
                client.predict(request, model=KumoRelational())
        except Exception as failed:  # noqa: BLE001 - the call's outcome
            error = failed
        finally:
            connection.close()
        bodies = [body for _, _, body in received.requests]
    code = getattr(error, 'code', None)
    return {
        'outcome': {
            'result': 'success' if error is None else 'refusal',
            'stage': stage,
            'error': None if error is None else type(error).__name__,
            'code': code if isinstance(code, str) else None,
        },
        'keys': keys,
        'request': _request(bodies[0]) if bodies else None,
        'warnings': [type(w.message).__name__ for w in shown],
        # Kept beside the answer, never compared as text (H17): read only
        # to find which warning names a table.
        'messages': [str(w.message) for w in shown],
    }


def observe_all(mode: str) -> dict[str, Any]:
    r"""Every vector on ``mode``'s route: ``python-ref`` and ``python-old``
    as the environment has them, ``python-core`` forced onto the core's
    DuckDB routes with each run's core operations recorded."""
    boundary = None
    routes: list[str] = []
    if mode == 'python-core':
        from sdm_client import _routing

        boundary = _runner()._Boundary()
        routes = sorted(
            {
                *_routing.FROM_SOURCE_DUCKDB,
                *_routing.MATERIALIZE_DUCKDB,
                *_routing.PREDICT_DUCKDB,
            }
        )
    observed = {}
    for vector in VECTORS:
        capture = boundary.reset() if boundary is not None else None
        if routes:
            os.environ['SDM_CORE_ROUTES'] = ','.join(routes)
        try:
            answer = observe(vector)
        finally:
            os.environ.pop('SDM_CORE_ROUTES', None)
        if capture is not None:
            answer['operations'] = list(capture.operations)
        observed[vector['id']] = answer
    return observed


def outcome_class(outcome: dict[str, Any]) -> str:
    if outcome['result'] == 'success':
        return 'success'
    return f'refusal {outcome.get("code") or "UNNAMED"}'


def register(answer: dict[str, Any]) -> list[str]:
    r"""As ``time_inference_record.register``: an unnamed refusal or
    INTERNAL_ERROR is a change to register (E66(b))."""
    outcome = answer['outcome']
    if outcome['result'] != 'refusal':
        return []
    if outcome['code'] is None:
        return [
            f'the reference fails with an unnamed {outcome["error"]} at '
            f'{outcome["stage"]} (E66(b): a named refusal or a success in '
            'its place is a change to register)'
        ]
    if outcome['code'] == 'INTERNAL_ERROR':
        return [
            f'the reference fails with INTERNAL_ERROR at {outcome["stage"]} '
            '(E66(b): a named refusal or a success in its place is a change '
            'to register)'
        ]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--observe', choices=('python-core', 'python-old'))
    parser.add_argument('--out', type=Path)
    arguments = parser.parse_args(argv)
    if arguments.observe is not None:
        if arguments.out is None:
            parser.error('--observe needs --out')
        observed = observe_all(arguments.observe)
        arguments.out.write_text(json.dumps(observed, indent=1), 'utf-8')
        return 0

    import duckdb
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import sdm_client

    reference = Path.home() / 'code/sdmc-reference'
    if not Path(sdm_client.__file__).resolve().is_relative_to(reference):
        print("sdm_client is not the reference checkout's", file=sys.stderr)
        return 1
    commit = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    observed = observe_all('python-ref')
    recorded = []
    for vector in VECTORS:
        answer = observed[vector['id']]
        messages = answer.pop('messages')
        why = register(answer)
        recorded.append(
            {
                **vector,
                'reference': answer,
                'reference_messages': messages,
                'register': {'required': bool(why), 'why': why},
            }
        )
        print(
            vector['id'],
            answer['outcome'],
            answer['keys'],
            answer['warnings'],
            'request' if answer['request'] else 'no request',
            '(register)' if why else '',
            file=sys.stderr,
        )
    document = {
        'format': FORMAT,
        'recorded_by': 'contract/tools/sampled_null_key_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'pyarrow': pa.__version__,
            'numpy': np.__version__,
            'duckdb': duckdb.__version__,
        },
        'pql': PQL,
        'materialize': MATERIALIZE,
        'register_ref': REGISTER_REF,
        'vectors': recorded,
    }
    OUT.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    print(
        f'{len(recorded)} vectors written to {OUT.relative_to(ROOT)}',
        file=sys.stderr,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
