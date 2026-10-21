# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The query and option long tail on the core, against the old route (T2.13).

The implementer's own tests, beside the test author's cases in
``contract/cases``: on a DuckDB source forced onto the core, each holds the
core route's outcome to the old route's in the same process.

- ``EVALUATE`` and ``EXPLAIN``, and the entities ``FOR t.c = 1`` or
  ``FOR t.c IN (...)`` names, gather the request the same query without
  them gathers, but for the query text the payload carries.
- An inference option given as text is read as the engine's validation
  reads it, and a null target transform is written as null.
- An item's explanation and its own metadata are carried, and the result's
  explanation is the first item's that holds one.
- A day-first date literal read before the error a query ends with keeps
  its warning (carried row U1).
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

# The old route these tests compare against runs the relational engine,
# which the jobs without it do not install.
pytest.importorskip('sdm_relational_engine.relationallib')
duckdb = pytest.importorskip('duckdb')

from predictive_query import RELATIONAL  # noqa: E402
from sdm_connectors import DuckDBSource  # noqa: E402

from sdm_client import (  # noqa: E402
    KumoRelational,
    PredictiveQuery,
    Request,
    Schema,
    SDMClient,
    StructuredDataError,
    _routing,
)

ROOT = Path(__file__).resolve().parents[3]
MATERIALIZE = ','.join(
    _routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB
)
CORE = ','.join(
    _routing.FROM_SOURCE_DUCKDB
    + _routing.MATERIALIZE_DUCKDB
    + _routing.PREDICT_DUCKDB
)
COUNT = 'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id'
SUM = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'


@pytest.fixture(autouse=True)
def _old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)


def _shop() -> list[str]:
    lines = (
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8').splitlines()
    )
    start = lines.index('  shop:') + 1
    statements = []
    for line in lines[start:]:
        if not line.startswith('  - '):
            break
        statements.append(line[len('  - ') :])
    return statements


def _gathered(
    monkeypatch: pytest.MonkeyPatch,
    routes: str | None,
    query: str,
    **options: Any,
) -> Request | StructuredDataError:
    r"""``query`` materialized over the shop base on ``routes`` (the old
    route where none), for users 0 and 1 unless ``options`` name others,
    or the error it raised."""
    if routes is None:
        monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    else:
        monkeypatch.setenv(_routing.ENVIRONMENT, routes)
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
    finally:
        monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)


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
    query: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = _payloads(_gathered(monkeypatch, MATERIALIZE, query))
    old = _payloads(_gathered(monkeypatch, None, query))
    plain = _payloads(_gathered(monkeypatch, MATERIALIZE, COUNT))
    assert core == old
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
def test_an_inference_option_given_as_text_is_read_as_the_engine_reads_it(
    query: str, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    core = _gathered(monkeypatch, MATERIALIZE, query, inference_config=config)
    old = _gathered(monkeypatch, None, query, inference_config=config)
    if isinstance(old, StructuredDataError):
        assert isinstance(core, StructuredDataError), core
        assert (type(core), core.code) == (type(old), old.code)
        # Said with the option it is about, never the value given.
        assert repr(next(iter(config))) in str(core)
        assert all(
            not isinstance(v, str) or v not in str(core)
            for v in config.values()
        )
        return
    assert _payloads(core) == _payloads(old)


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
    monkeypatch: pytest.MonkeyPatch,
    request: Request,
    body: bytes,
    outputs: list[str],
) -> tuple[Any, Any]:
    r"""The predict call's outcome on the core and on the old route, each
    the result, or the error raised, with the warnings shown."""
    outcomes = []
    for routes in (CORE, None):
        if routes is None:
            monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
        else:
            monkeypatch.setenv(_routing.ENVIRONMENT, routes)
        with (
            _answering(body) as url,
            warnings.catch_warnings(record=True) as shown,
        ):
            warnings.simplefilter('always')
            try:
                with SDMClient(url, max_retries=0) as client:
                    outcome: Any = client.predict(
                        request, model=KumoRelational(), outputs=outputs
                    )
            except StructuredDataError as error:
                outcome = error
        outcomes.append((outcome, [type(w.message).__name__ for w in shown]))
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    return outcomes[0], outcomes[1]


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
def test_an_item_s_explanation_and_metadata_are_the_old_route_s(
    explanation: Any, metadata: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _gathered(monkeypatch, MATERIALIZE, COUNT, indices=[0])
    assert isinstance(request, Request), request
    item: dict[str, Any] = {
        'id': _instance(request),
        'row_index': 0,
        'prediction': 'true',
        'probabilities': {'false': 0.25, 'true': 0.75},
        'explanation': explanation,
        'metadata': metadata,
    }
    for outputs in (['prediction', 'explanation'], ['prediction']):
        (core, core_shown), (old, old_shown) = _predicted(
            monkeypatch, request, _answer(item), outputs
        )
        assert not isinstance(core, Exception), core
        assert core_shown == old_shown
        assert core.items == old.items
        assert core.explanation == old.explanation
        assert core.absent(outputs) == old.absent(outputs)
        pd.testing.assert_frame_equal(core.to_pandas(), old.to_pandas())


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('explanation', 'ab'),
        ('explanation', [['a']]),
        ('metadata', 3),
        ('metadata', [[['a'], 1]]),
    ],
)
def test_an_item_field_that_is_no_mapping_is_refused_as_on_the_old_route(
    field: str, value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _gathered(monkeypatch, MATERIALIZE, COUNT, indices=[0])
    assert isinstance(request, Request), request
    item = {
        'id': _instance(request),
        'row_index': 0,
        'prediction': 'true',
        field: value,
    }
    (core, _), (old, _) = _predicted(
        monkeypatch, request, _answer(item), ['prediction']
    )
    assert isinstance(core, StructuredDataError), core
    assert isinstance(old, StructuredDataError), old
    assert (type(core), core.code) == (type(old), old.code)
    assert core.code == 'INVALID_RESPONSE'
    assert core.details['field'] == f'prediction.{field}'
    assert f"item's {field}" in str(core)


def test_an_explanation_of_two_entities_is_refused_as_on_the_old_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _gathered(monkeypatch, MATERIALIZE, COUNT)
    assert isinstance(request, Request), request
    (core, _), (old, _) = _predicted(
        monkeypatch, request, _answer(), ['prediction', 'explanation']
    )
    assert isinstance(core, StructuredDataError), core
    assert isinstance(old, StructuredDataError), old
    assert (type(core), core.code) == (type(old), old.code)
    assert str(core).endswith(str(old).split('] ', 1)[-1]), (core, old)


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


def test_a_day_first_literal_read_before_an_error_keeps_its_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # U1: the literal read before a packed one, which the core refuses by
    # name (decision E79), is warned of before the error.
    monkeypatch.setenv(
        _routing.ENVIRONMENT,
        ','.join(
            _routing.FROM_SOURCE_DUCKDB
            + ('query.validate/accept/any', 'query.validate/refuse/any')
        ),
    )
    shown, code = _shown_on_the_core(
        'PREDICT COUNT(orders.* WHERE orders.ts > 13/01/2024 AND orders.ts '
        '< 20240101-01-01, 0, 30, days) FOR EACH users.user_id'
    )
    assert code == 'INVALID_REQUEST'
    assert len(shown) == 1
    assert shown[0].startswith('[DATE_LITERAL_DAY_FIRST] Parsing dates in')
