# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public query prefixes, normalized inference options and response fields.
DuckDB materialization and local HTTP prediction use explicit expectations,
including metadata, explanations and warning ordering.
"""

from __future__ import annotations

import json
import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

duckdb = pytest.importorskip('duckdb')

from sdm_connectors import DuckDBSource  # noqa: E402

from sdm_client import (  # noqa: E402
    KumoRelational,
    PredictiveQuery,
    Request,
    Schema,
    SDMClient,
    StructuredDataError,
)
from sdm_client.query import RELATIONAL  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
COUNT = 'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id'
SUM = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'


def _shop() -> list[str]:
    return json.loads((Path(__file__).parent / 'data/shop.json').read_text())[
        'statements'
    ]


def _gathered(query: str, **options: Any) -> Request | StructuredDataError:
    """Materialize the shop query for users 0 and 1, unless overridden."""
    connection = duckdb.connect()
    for statement in _shop():
        connection.execute(statement)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(connection))
            return PredictiveQuery(query, schema).materialize(
                **{
                    'indices': [0, 1],
                    'anchor_time': pd.Timestamp('2024-03-01'),
                    **options,
                }
            )
    except StructuredDataError as error:
        return error


def _payloads(request: Request | StructuredDataError) -> list[Any]:
    assert isinstance(request, Request), request
    return json.loads(json.dumps(request.payloads))


@pytest.mark.parametrize(
    'query',
    [
        'EVALUATE ' + COUNT,
        'EXPLAIN ' + COUNT,
        COUNT.replace('FOR EACH users.user_id', 'FOR users.user_id = 3'),
        COUNT.replace('FOR EACH users.user_id', 'FOR users.user_id IN (5, 6)'),
    ],
)
def test_a_prefix_or_the_entities_named_gather_the_plain_query_s_request(
    query: str,
) -> None:
    core = _payloads(_gathered(query))
    plain = _payloads(_gathered(COUNT))
    # The query as written is the only difference from the plain query's.
    assert [p['metadata'].pop('source_query') for p in core] == [query]
    for payload in plain:
        payload['metadata'].pop('source_query')
    assert core == plain


@pytest.mark.parametrize(
    ('query', 'config'),
    [
        (SUM, {'num_estimators': ' 3 ', 'column_shuffle': 'yes'}),
        (SUM, {'num_estimators': '2.00', 'hop_shuffle': 'OFF'}),
        (SUM, {'target_transforms': ['quantile', None]}),
        (COUNT, {'class_shuffle': 'On', 'category_shuffle': '0'}),
        (SUM, {'num_estimators': 'x'}),
        (SUM, {'column_shuffle': ' yes'}),
        (SUM, {'num_estimators': '5'}),
    ],
)
def test_inference_text_options_have_explicit_normalized_values(
    query: str, config: dict[str, Any]
) -> None:
    request = _gathered(query, inference_config=config)
    invalid = (
        config.get('num_estimators') in ('x', '5')
        or config.get('column_shuffle') == ' yes'
    )
    if invalid:
        assert isinstance(request, StructuredDataError), request
        assert request.code == 'INVALID_REQUEST'
        assert repr(next(iter(config))) in str(request)
        assert all(
            not isinstance(v, str) or v not in str(request)
            for v in config.values()
        )
        return
    payload = _payloads(request)[0]
    expected = {
        'num_estimators': {' 3 ': 3, '2.00': 2},
        'column_shuffle': {'yes': True},
        'hop_shuffle': {'OFF': False},
        'class_shuffle': {'On': True},
        'category_shuffle': {'0': False},
    }
    for key, value in config.items():
        assert payload['inference']['inference_config'][key] == (
            ['quantile', None]
            if key == 'target_transforms'
            else expected[key][value]
        )


@contextmanager
def _answering(body: bytes) -> Iterator[str]:
    r"""An endpoint answering every request with ``body``: its URL."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            length = int(self.headers.get('Content-Length') or 0)
            self.rfile.read(length)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _instance(request: Request) -> str:
    r"""The instance id the request's one scored row is asked under."""
    payload = (request.payloads or ())[0]
    return str(payload['predict']['instance_table']['rows'][0][0])


def _answer(*items: dict[str, Any]) -> bytes:
    return json.dumps(
        {
            'object': 'prediction.response',
            'id': 'r1',
            'model': 'kumo-relational',
            'metadata': {'served_by': 'a'},
            'predictions': list(items),
        }
    ).encode()


def _predicted(
    request: Request, body: bytes, outputs: list[str]
) -> tuple[Any, list[str]]:
    """Public HTTP prediction and the warning classes it emits."""
    with _answering(body) as url, warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        try:
            with SDMClient(url, max_retries=0) as client:
                outcome: Any = client.predict(
                    request, model=KumoRelational(), outputs=outputs
                )
        except StructuredDataError as error:
            outcome = error
    return outcome, [type(w.message).__name__ for w in shown]


@pytest.mark.parametrize(
    ('explanation', 'metadata'),
    [
        ({'top_features': [{'column': 'age', 'score': 0.5}]}, {'ms': 3}),
        ([['a', 1], [2, 'two'], [2.0, 'deux']], None),
        ('', [['k', 'v']]),
        ({}, {}),
        (None, None),
    ],
)
def test_response_item_metadata_and_explanation_are_preserved(
    explanation: Any, metadata: Any
) -> None:
    request = _gathered(COUNT, indices=[0])
    assert isinstance(request, Request), request
    item = {
        'id': _instance(request),
        'row_index': 0,
        'prediction': 'true',
        'probabilities': {'false': 0.25, 'true': 0.75},
        'explanation': explanation,
        'metadata': metadata,
    }
    expected_explanation = None if explanation is None else dict(explanation)
    expected_metadata = None if metadata is None else dict(metadata)
    for outputs in (['prediction', 'explanation'], ['prediction']):
        result, shown = _predicted(request, _answer(item), outputs)
        assert not isinstance(result, Exception), result
        assert len(result.items) == 1
        actual = result.items[0]
        assert actual.explanation == expected_explanation
        assert actual.metadata == expected_metadata
        assert actual.row_index == 0
        assert actual.probabilities == {'False': 0.25, 'True': 0.75}
        assert result.to_pandas()[
            ['probability_False', 'probability_True']
        ].values.tolist() == [[0.25, 0.75]]
        assert result.to_pandas()['id'].tolist() == [0]
        assert result.to_pandas()['prediction'].tolist() == [True]
        assert result.metadata == {'served_by': 'a'}
        assert result.explanation == (expected_explanation or None)
        missing = (
            ('explanation',)
            if 'explanation' in outputs and not expected_explanation
            else ()
        )
        assert result.absent(outputs) == missing
        assert shown == (['SDMWarning'] if missing else [])


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('explanation', 'ab'),
        ('explanation', [['a']]),
        ('metadata', 3),
        ('metadata', [[['a'], 1]]),
    ],
)
def test_non_mapping_response_item_field_is_refused(
    field: str, value: Any
) -> None:
    request = _gathered(COUNT, indices=[0])
    assert isinstance(request, Request), request
    item = {
        'id': _instance(request),
        'row_index': 0,
        'prediction': 'true',
        field: value,
    }
    core, _ = _predicted(request, _answer(item), ['prediction'])
    assert isinstance(core, StructuredDataError), core
    assert core.code == 'INVALID_RESPONSE'
    assert core.details['field'] == f'prediction.{field}'
    assert f"item's {field}" in str(core)


def test_explanation_of_two_entities_is_refused() -> None:
    request = _gathered(COUNT)
    assert isinstance(request, Request), request
    core, _ = _predicted(request, _answer(), ['prediction', 'explanation'])
    assert isinstance(core, StructuredDataError), core
    assert core.code == 'UNSUPPORTED_REQUEST'
    assert 'explanation' in str(core).lower()
    assert 'one' in str(core).lower() or 'single' in str(core).lower()


def _shown_on_the_core(text: str) -> tuple[list[str], str | None]:
    r"""What validating ``text`` on the core shows, and the code it ends
    with."""
    connection = duckdb.connect()
    for statement in _shop():
        connection.execute(statement)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(connection))
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        try:
            PredictiveQuery(text, schema)._resolve(RELATIONAL)
            code = None
        except StructuredDataError as error:
            code = error.code
    return [str(w.message) for w in shown], code


def test_a_day_first_literal_read_before_an_error_keeps_its_warning() -> None:
    # U1: the literal read before a packed one, which the core refuses by
    # name (decision E79), is warned of before the error.
    shown, code = _shown_on_the_core(
        'PREDICT COUNT(orders.* WHERE orders.ts > 13/01/2024 AND orders.ts '
        '< 20240101-01-01, 0, 30, days) FOR EACH users.user_id'
    )
    assert code == 'INVALID_REQUEST'
    assert len(shown) == 1
    assert shown[0].startswith('[DATE_LITERAL_DAY_FIRST] Parsing dates in')
