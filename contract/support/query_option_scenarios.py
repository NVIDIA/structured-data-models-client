#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query option scenarios: independent inputs and current-client test helpers."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/t213-option-vectors.json'
CASES = ROOT / 'contract/cases/s1'
FORMAT = 'sdm-t213-option-vectors/1'
#: The T2.13 cases, by the start of their ids.
PREFIXES = (
    'filter.',
    'ids.',
    'assuming.',
    'evaluate.',
    'explain.',
    'metadata.',
    'transforms.',
    'config-text.',
    'date.',
)
DATABASES = ('duckdb', 'sqlite', 'postgres')
#: How a warning T2.13's exits name opens: the entity filter's notice, an
#: explanation asked for and not given, a day-first date literal.
WARNINGS = (
    'the entity filter ',
    'kumo-relational answered without ',
    'Parsing dates in ',
)
#: The code a core warning's text is prefixed with on python-core (E41(i),
#: E59(d)), left out so a warning is held by its words.
CODE_PREFIX = re.compile(r'^\[[A-Z][A-Z0-9_]*\] ')


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('t213_run_python', ROOT / 'contract/runners/run_python.py')


def endpoint() -> ModuleType:
    return _module(
        'scripted_endpoint', ROOT / 'tests/support/scripted_endpoint.py'
    )


def case_ids() -> list[str]:
    r"""T2.13's case ids, in name order."""
    return sorted(
        path.name[: -len('.json')]
        for path in CASES.glob('*.json')
        if path.name.startswith(PREFIXES)
    )


def as_json(value: Any) -> Any:
    r"""``value`` as JSON writes it: a NumPy scalar as its Python value, a
    NaN as null, a mapping's keys as text, a sequence as a list, anything
    else as its text."""
    if hasattr(value, 'tolist'):
        value = value.tolist()
    elif hasattr(value, 'item'):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if value != value else value
    if isinstance(value, dict):
        return {str(key): as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(item) for item in value]
    return str(value)


def projection(result: Any) -> dict[str, Any] | None:
    r"""A ``PredictionResult`` through ``to_pandas()``: its columns, its
    row count and its rows."""
    if result is None or not hasattr(result, 'to_pandas'):
        return None
    frame = result.to_pandas()
    rows = [
        [as_json(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    return {
        'columns': [str(name) for name in frame.columns],
        'count': len(rows),
        'rows': rows,
    }


def item_metadata(result: Any) -> list[Any] | None:
    r"""Each item's ``row_index`` and ``metadata``, by ``row_index``."""
    if result is None or not hasattr(result, 'items'):
        return None
    kept = [
        [as_json(item.row_index), as_json(item.metadata)]
        for item in result.items
    ]
    return sorted(kept, key=lambda pair: (pair[0] is None, pair[0] or 0))


def said(shown: list[Any]) -> list[list[str]]:
    r"""The warnings :data:`WARNINGS` names, each as its class name and
    text (a core code's prefix left out), once each, in the order first
    given."""
    found: list[list[str]] = []
    for warning in shown:
        text = CODE_PREFIX.sub('', str(warning.message))
        if not text.startswith(WARNINGS):
            continue
        entry = [warning.category.__name__, text]
        if entry not in found:
            found.append(entry)
    return found


def _rows_sent(body: bytes) -> int | None:
    try:
        rows = json.loads(body)['predict']['instance_table']['rows']
    except (ValueError, KeyError, TypeError):
        return None
    return len(rows) if isinstance(rows, list) else None


def observe(case: dict[str, Any], database: str) -> dict[str, Any]:
    """Exercise current-client public calls against a scripted endpoint or source."""
    from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient

    run = runner()
    given = case['input']
    arguments = run._materialize_arguments(given['materialize'])
    from_source = run._from_source_arguments(given.get('from_source', {}))
    outputs = {'outputs': list(given['outputs'])} if 'outputs' in given else {}
    observed: dict[str, Any] = {
        'outcome': None,
        'calls': [],
        'result': None,
        'explanation': None,
        'item_metadata': None,
        'warnings': [],
        'message': None,
        'details': None,
    }
    connection, cleanup, _ = run._open(case, run.load_bases(), database)
    stage = 'from_source'
    try:
        with (
            endpoint().serve(
                fail_call=given.get('fail_call'),
                probabilities=given.get('probabilities', False),
                quantiles=given.get('quantiles', False),
                explanation=given.get('explanation', False),
                item_metadata=given.get('item_metadata', False),
            ) as (url, received),
            warnings.catch_warnings(record=True) as shown,
        ):
            warnings.simplefilter('always')
            try:
                schema = Schema.from_source(
                    run._source(database, connection), **from_source
                )
                run._edit(schema, given.get('edits', []))
                if case['until'] != 'from_source':
                    stage = 'materialize'
                    request = PredictiveQuery(given['pql'], schema).materialize(
                        **arguments
                    )
                if case['until'] == 'predict':
                    stage = 'predict'
                    with SDMClient(url) as client:
                        result = client.predict(
                            request, model=KumoRelational(), **outputs
                        )
                    observed['result'] = projection(result)
                    observed['explanation'] = as_json(result.explanation)
                    observed['item_metadata'] = item_metadata(result)
                observed['outcome'] = run._outcome('success', stage)
            except Exception as error:  # noqa: BLE001 - the outcome
                observed['outcome'] = run._outcome('refusal', stage, error)
                details = getattr(error, 'details', None)
                details = details if isinstance(details, dict) else {}
                observed['details'] = {
                    str(name): as_json(value)
                    for name, value in sorted(details.items())
                }
                observed['message'] = str(error)
            observed['warnings'] = said(shown)
            observed['calls'] = [
                _rows_sent(body) for _, _, body in received.requests
            ]
    finally:
        cleanup()
    return observed


def load() -> dict[str, Any]:
    r"""The vector file, as committed."""
    return json.loads(OUT.read_text(encoding='utf-8'))
