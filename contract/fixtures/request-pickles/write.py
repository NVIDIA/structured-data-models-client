#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Writes the ``Request`` pickles the reference client makes (H23, T2.14b),
and a ``Schema`` pickle holding ``conversion_notes``.

    python -I contract/fixtures/request-pickles/write.py \
        --reference CHECKOUT [--write]

Run by the reference environment's interpreter (``contract/reference-env.txt``:
the reference commit's client, Python 3.12, pandas 3), it builds each
object of :func:`objects` with the reference checkout's client, pickles it
with protocol 4, and records beside the pickles, in ``expected.json``, the
reference's own reading of each as it loads back from the pickle
(:func:`projection`) and, for a request a predictive query gathered, what
predicting it sends and answers against the scripted endpoint
(:func:`predicted`). ``contract/tests/test_t214b_pickles.py`` loads each
pickle in the client under test and holds it to that.

A request gathered over a source holds the source, which no client pickles
(``Source.__reduce__`` refuses an open connection), so the requests a
reference user can pickle are those of a schema of frames: one asked in one
call (``request-one-call``), a forecast asked as its steps
(``request-forecast``), a ranking asked as one question per candidate
(``request-ranking``), and a request built from frames
(``request-frames``). The schema (``schema-notes``) is ``main``'s semantic
model read through the reference's own fake Databricks warehouse
(``packages/structured-data-models-client/tests/test_semantic_model_source.py``'s
``_Warehouse``, which answers a metric view's metadata and preview), so its
``conversion_notes`` are the reference's; it is pickled with the state a
schema pickled before it could be bound to a source holds: no source and no
engine reading of it (``Schema.__setstate__``'s defaults), as
``contract/tests/test_schema_pickles.py`` pickles a core-built schema.

The script refuses to run but from the reference checkout given, at the
reference commit ``contract/reference-env.txt`` names. Without ``--write``
it builds the pickles and compares them with the committed ones, and the
exit status says whether they are the same; CI never runs it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import pickle
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REFERENCE_ENV = ROOT / 'contract/reference-env.txt'
ENDPOINT = ROOT / 'contract/runners/scripted_endpoint.py'
#: The objects pickled, by file stem.
NAMES = (
    'request-one-call',
    'request-forecast',
    'request-ranking',
    'request-frames',
    'schema-notes',
)


def _reference_commit() -> str:
    for line in REFERENCE_ENV.read_text(encoding='utf-8').splitlines():
        if line.startswith('reference: '):
            return line.split(': ', 1)[1].strip()
    raise SystemExit('reference-env.txt names no reference commit')


def endpoint() -> Any:
    name = 'scripted_endpoint'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ENDPOINT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _frames() -> dict[str, Any]:
    import pandas as pd

    users = pd.DataFrame(
        {
            'user_id': list(range(8)),
            'age': [20 + i for i in range(8)],
        }
    )
    orders = pd.DataFrame(
        {
            'order_id': list(range(60)),
            'user_id': [i % 8 for i in range(60)],
            'price': [10 + i * 1.5 for i in range(60)],
            'ts': pd.date_range('2024-01-01', periods=60, freq='D'),
        }
    )
    items = pd.DataFrame({'item_id': [0, 1, 2, 3], 'kind': list('abcd')})
    buys = pd.DataFrame(
        {
            'buy_id': list(range(90)),
            'user_id': [i % 8 for i in range(90)],
            'item_id': [(i * 3) % 4 for i in range(90)],
            'ts': pd.date_range('2024-01-01', periods=90, freq='D'),
        }
    )
    return {'users': users, 'orders': orders, 'items': items, 'buys': buys}


def _semantic_schema() -> Any:
    from sdm_client import Schema

    reference = Path(sys.modules['sdm_client'].__file__).resolve().parents[2]
    path = reference / 'tests/test_semantic_model_source.py'
    spec = importlib.util.spec_from_file_location('semantic_model_source', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from sdm_connectors import DatabricksSource

    source = DatabricksSource(
        module._Warehouse(), catalog=module._CATALOG, schema=module._SCHEMA
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_source(source.semantic_model('sales_mv'))


def objects() -> dict[str, Any]:
    r"""Each object pickled, by name."""
    import pandas as pd
    from sdm_client import PredictiveQuery, Request, Schema

    anchor = pd.Timestamp('2024-03-01')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(_frames())

        def asked(pql: str, **options: Any) -> Any:
            return PredictiveQuery(pql, schema).materialize(
                indices=[0, 1], anchor_time=anchor, **options
            )

        made = {
            'request-one-call': asked(
                'PREDICT SUM(orders.price, 0, 7, days) FOR EACH users.user_id'
            ),
            'request-forecast': asked(
                'PREDICT SUM(orders.price, 0, 7, days) FORECAST 2 TIMEFRAMES '
                'FOR EACH users.user_id'
            ),
            'request-ranking': asked(
                'PREDICT LIST_DISTINCT(buys.item_id, 0, 30, days) RANK TOP 2 '
                'FOR EACH users.user_id',
                candidates=[2, 1, 0],
            ),
            'request-frames': Request(
                pd.DataFrame({'x': [1, 2, 3], 'y': [0.5, 1.5, 2.5]}),
                pd.DataFrame({'x': [4, 5]}),
                target='y',
                task='regression',
            ),
        }
    made['schema-notes'] = _semantic_schema()
    return made


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _frame(frame: Any) -> dict[str, Any]:
    return {
        'columns': [str(c) for c in frame.columns],
        'rows': len(frame),
        'digest': _digest(frame.astype(str).values.tolist()),
    }


def _request(request: Any) -> dict[str, Any]:
    task = request.task
    out: dict[str, Any] = {
        'kind': type(request).__name__,
        'task': None
        if task is None
        else {
            'kind': task.kind,
            'target': [
                task.target.column_name,
                task.target.dtype,
                task.target.table_name,
            ],
            'entity_table': task.entity_table,
        },
        'query': request.query,
        'entity_table': request.entity_table,
        'windowed': getattr(request, '_windowed', None),
        'indices': [str(i) for i in getattr(request, 'indices', None) or []],
        'payloads': None
        if request.payloads is None
        else [_digest(p) for p in request.payloads],
        'context': _frame(request.context.instance_table),
        'predict': _frame(request.predict.instance_table),
        'schema': sorted(request.schema.tables),
    }
    if request.parts is not None:
        out['parts'] = [_request(part) for part in request.parts]
        rewritten = request._decomposition
        out['decomposition'] = {
            'kind': rewritten.kind,
            'parts': [[query, str(about)] for query, about in rewritten.parts],
        }
    return out


def _schema(schema: Any) -> dict[str, Any]:
    tables = []
    for name, table in schema.tables.items():
        key = table.primary_key
        tables.append(
            {
                'name': name,
                'kind': type(table).__name__,
                'columns': [
                    [column, c.dtype, c.stype, c.stype_declared]
                    for column, c in table.columns.items()
                ],
                'primary_key': key
                if key is None or isinstance(key, str)
                else list(key),
                'time_column': table.time_column,
            }
        )
    return {
        'kind': type(schema).__name__,
        'tables': tables,
        'links': [
            [
                link.source_table,
                link.source_column,
                link.target_table,
                link.target_column,
            ]
            for link in schema.relationships
        ],
        'conversion_notes': list(schema.conversion_notes),
        'source': schema.source is not None,
    }


def projection(value: Any) -> dict[str, Any]:
    r"""What a loaded object says, as the reference and the candidate both
    read it: a request's task, query, entities, facts, payloads (each by
    its digest), halves and parts; a schema's tables, links and notes."""
    if type(value).__name__ == 'Schema':
        return _schema(value)
    return _request(value)


def predicted(request: Any) -> dict[str, Any]:
    r"""What predicting ``request`` sends and answers, against the scripted
    endpoint: each body by its digest (its request id left out), and the
    result through ``to_pandas()``, or the refusal's class and code."""
    from sdm_client import KumoRelational, SDMClient

    with (
        endpoint().serve(probabilities=True) as (url, received),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter('ignore')
        out: dict[str, Any] = {}
        try:
            with SDMClient(url) as client:
                result = client.predict(request, model=KumoRelational())
            out['result'] = _frame(result.to_pandas())
        except Exception as error:  # noqa: BLE001 - recorded
            out['refusal'] = [
                type(error).__name__,
                getattr(error, 'code', None),
            ]
        bodies = []
        for _, _, body in received.requests:
            parsed = json.loads(body)
            parsed.get('metadata', {}).pop('request_id', None)
            bodies.append(_digest(parsed))
        out['bodies'] = bodies
    return out


def _without_the_source(
    buffer: io.BytesIO, schema: Any, state: dict[str, Any]
) -> pickle.Pickler:
    r"""A pickler writing ``schema`` with ``state`` in place of its own."""

    class _WithoutTheSource(pickle.Pickler):
        def reducer_override(self, obj: Any) -> Any:
            if obj is schema:
                return object.__new__, (type(schema),), state
            return NotImplemented

    return _WithoutTheSource(buffer, protocol=4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    given = parser.parse_args(argv)
    reference = given.reference.resolve()
    head = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != _reference_commit():
        raise SystemExit(f'{reference} is at {head}, not the reference commit')
    import pandas as pd
    import sdm_client

    client = Path(sdm_client.__file__).resolve()
    if not client.is_relative_to(reference):
        raise SystemExit(f'sdm_client is {client}, not the reference checkout')
    expected: dict[str, Any] = {
        'reference': head,
        'python': '.'.join(map(str, sys.version_info[:3])),
        'pandas': pd.__version__,
        'protocol': 4,
        'objects': {},
    }
    written: dict[str, bytes] = {}
    made = objects()
    assert tuple(made) == NAMES
    for name, value in made.items():
        buffer = io.BytesIO()
        if name == 'schema-notes':
            # No source and no engine reading, as Schema.__setstate__
            # restores a schema pickled before it could have either.
            state = {
                key: item
                for key, item in vars(value).items()
                if key not in ('_source', '_engine_tables')
            }

            _without_the_source(buffer, value, state).dump(value)
        else:
            pickle.dump(value, buffer, protocol=4)
        written[f'{name}.pkl'] = buffer.getvalue()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loaded = pickle.loads(buffer.getvalue())
        entry: dict[str, Any] = {'projection': projection(loaded)}
        if name.startswith('request-') and name != 'request-frames':
            entry['predicted'] = predicted(loaded)
        expected['objects'][name] = entry
    written['expected.json'] = (
        json.dumps(expected, indent=2, sort_keys=True) + '\n'
    ).encode()
    if given.write:
        for file, data in written.items():
            (HERE / file).write_bytes(data)
        return 0
    same = all(
        (HERE / file).exists() and (HERE / file).read_bytes() == data
        for file, data in written.items()
    )
    print("the committed pickles are the reference's" if same else 'differ')
    return 0 if same else 1


if __name__ == '__main__':
    sys.exit(main())
