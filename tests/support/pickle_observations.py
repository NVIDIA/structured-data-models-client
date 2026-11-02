#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Read-only projections and endpoint observations for pickle compatibility tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = ROOT / 'tests/support/scripted_endpoint.py'


def endpoint() -> Any:
    name = 'pickle_scripted_endpoint'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ENDPOINT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


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
    result through ``to_pandas()``; unexpected failures retain their traceback."""
    from sdm_client import KumoRelational, SDMClient

    with (
        endpoint().serve(probabilities=True) as (url, received),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter('ignore')
        out: dict[str, Any] = {}
        with SDMClient(url) as client:
            result = client.predict(request, model=KumoRelational())
        out['result'] = _frame(result.to_pandas())
        bodies = []
        for _, _, body in received.requests:
            parsed = json.loads(body)
            parsed.get('metadata', {}).pop('request_id', None)
            bodies.append(_digest(parsed))
        out['bodies'] = bodies
    return out
