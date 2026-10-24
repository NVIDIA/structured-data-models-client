# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The canary test (T1.11d; decision H17, an owner-approved exit check).

Sentinels are planted in data (two values of a DuckDB table the journey
reads and sends), in a fake DSN credential (a connection string held as
data, and the userinfo of the model endpoint's URL), and in a test-only
panic's payload. Every error path the Python host reaches at T1.11d is
driven through it:

- codec: an operation the contract cannot read, an attachment that is not
  the JSON it names, and model answers the core cannot read (one not the
  contract's, one answering under another row's id);
- provider: a DuckDB statement failing on a planted value, which the
  driver's own message quotes;
- transport: the model endpoint unreachable at a URL carrying the fake
  credential;
- refusal: the endpoint refusing the call (HTTP 422 in the endpoint's own
  words, a sentinel of its own in its code, detail, title and other fields;
  HTTP 500, echoing nothing), behind a URL carrying the credential, raised
  as the old route raises it, the endpoint's own words shown (decision E47,
  owner decision 2026-10-10), none of ours;
- panic: the test-only operation, panicking in the run's body with the
  sentinel as its payload.

None of our sentinels may appear in the core's error (its message, every
field and every cause, as the control message carried it), in ``str`` or
``repr`` of the exception, in its formatted traceback, in any warning
shown, in any log record captured at ``DEBUG`` from every logger (the
core's included, through the logging bridge), or on stderr or stdout.

The check is not vacuous: each sentinel is first shown on the path that
fails (the driver's message, the URL the transport was given, the bytes the
codec refused, the panic's location written to stderr), and the same check
catches each when a deliberately leaky formatter, installed only inside
this test, writes it into the exception.

One exception, the owner's (decision E47, 2026-10-10): text the model
endpoint itself writes in a refusal is shown on the core route, as the old
route shows it. A sentinel only the endpoint writes is asserted to reach
``str`` and the details, so the exception works and cannot quietly regress
to hiding; and an endpoint that echoes our data has that echo shown
verbatim, as on the old route, which is the case the owner accepted. Our
own sentinels still reach nothing through our code: the credential in the
URL is never chained or added. As on the old route, an endpoint that echoes
our credential in its own text has it shown, and the refusal is the old
route's exactly (a scrub of such echoes was tried in fix rounds 3 and 4 and
removed, E47). The client adds only its own explanations, which name our
schema's tables and columns, never a value.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import logging
import socket
import sys
import threading
import traceback
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Schema,
    SDMClient,
    StructuredDataError,
    _routing,
)
from sdm_client._rust import codec, errors
from sdm_client._rust import run as run_module
from sdm_client._rust.providers.duckdb import DuckDBProvider
from sdm_client._rust.run import run
from sdm_client.errors import NimRequestError

ROOT = Path(__file__).resolve().parents[3]
DATA = 'SENTINEL-data-7a1f'
DSN = 'SENTINEL-dsn-c03e'
PANIC = 'SENTINEL-panic-9b52'
SENTINELS = (DATA, DSN, PANIC)
CREDENTIAL = f'postgresql://sdm:{DSN}@db.invalid:5432/sales'
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
CORE = ','.join(
    _routing.FROM_SOURCE_DUCKDB
    + _routing.MATERIALIZE_DUCKDB
    + _routing.PREDICT_DUCKDB
)


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


def _planted(*, vault: bool = False) -> Any:
    r"""The ``shop`` base with the sentinels in the text the journey reads
    and sends, two users' countries; with ``vault``, a table beside it
    holding the fake credential's whole connection string and the data
    sentinel, which only the provider scenario reads (a described source
    with it would be refused by name, since text holding ``:`` reads as
    several categories, register P1B-2)."""
    connection = duckdb.connect()
    for statement in _shop():
        connection.execute(statement)
    connection.execute(f"UPDATE users SET country = '{DATA}' WHERE user_id = 3")
    connection.execute(f"UPDATE users SET country = '{DSN}' WHERE user_id = 4")
    if vault:
        connection.execute('CREATE TABLE vault (kind VARCHAR, secret VARCHAR)')
        connection.execute(
            f"INSERT INTO vault VALUES ('dsn', '{CREDENTIAL}'), "
            f"('data', '{DATA}')"
        )
    return connection


class _Seen:
    r"""Everything one scenario wrote anywhere a person or a log could read
    it."""

    def __init__(self) -> None:
        self.texts: list[tuple[str, str]] = []

    def add(self, where: str, text: str) -> None:
        self.texts.append((where, text))

    def leaks(self) -> list[tuple[str, str]]:
        return [
            (where, sentinel)
            for where, text in self.texts
            for sentinel in SENTINELS
            if sentinel in text
        ]


def _core_error_texts(control: bytes, attachments: dict[int, Any]) -> str:
    r"""The core's error as it crossed: message, fields and causes."""
    error = codec.read(control, attachments)['error']
    parts = []
    while error is not None:
        parts += [error['message'], json.dumps(error.get('fields', {}))]
        parts.append(json.dumps(error))
        error = error.get('cause')
    return '\n'.join(parts)


@contextmanager
def _watching(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> Iterator[_Seen]:
    r"""Collects, around one scenario, the core's error, every warning and
    every log record at DEBUG; stderr and stdout are read at the end."""
    seen = _Seen()
    decoded = errors.error_from_core

    def recording(control: bytes, attachments: Any, **kwargs: Any) -> Any:
        seen.add('core error', _core_error_texts(control, attachments))
        return decoded(control, attachments, **kwargs)

    monkeypatch.setattr(errors, 'error_from_core', recording)
    records: list[logging.LogRecord] = []

    class Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    root = logging.getLogger()
    handler = Keep(logging.DEBUG)
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        with warnings.catch_warnings(record=True) as shown:
            warnings.simplefilter('always')
            yield seen
    finally:
        root.removeHandler(handler)
        root.setLevel(level)
    formatter = logging.Formatter()
    for record in records:
        seen.add(f'log {record.name}', formatter.format(record))
    for warning in shown:
        seen.add('warning', str(warning.message))
    captured = capfd.readouterr()
    seen.add('stderr', captured.err)
    seen.add('stdout', captured.out)
    seen.records = records  # type: ignore[attr-defined]
    seen.stderr = captured.err  # type: ignore[attr-defined]
    seen.warnings = [str(w.message) for w in shown]  # type: ignore[attr-defined]


def _raised(seen: _Seen, call: Callable[[], Any]) -> StructuredDataError:
    with pytest.raises(StructuredDataError) as raised:
        call()
    error = raised.value
    seen.add('str', str(error))
    seen.add('repr', repr(error))
    seen.add('details', repr(error.details))
    seen.add('traceback', ''.join(traceback.format_exception(error)))
    return error


def _endpoint() -> Any:
    name = 'scripted_endpoint'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'contract/runners/scripted_endpoint.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@contextmanager
def _answering(body: bytes, status: int = 200) -> Iterator[str]:
    r"""An endpoint answering every request ``status`` with ``body``; its
    URL carries the fake credential."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            self.rfile.read(int(self.headers.get('Content-Length') or 0))
            # An endpoint that echoes the credential it was sent: the
            # Authorization header the transport built of the URL's
            # userinfo, wherever the body says {AUTH}.
            sent = self.headers.get('Authorization', '').encode()
            answer = body.replace(b'{AUTH}', sent)
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(answer)))
            self.end_headers()
            self.wfile.write(answer)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://sdm:{DSN}@127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _request() -> Any:
    r"""base.regression's request over the planted data, on the core."""
    schema = Schema.from_source(DuckDBSource(_planted()))
    return PredictiveQuery(QUERY, schema).materialize(
        indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01')
    )


def _predict(url: str) -> Any:
    request = _request()
    with SDMClient(url, max_retries=0) as client:
        return client.predict(request, model=KumoRelational())


def _closed_port() -> int:
    with socket.socket() as closed:
        closed.bind(('127.0.0.1', 0))
        return int(closed.getsockname()[1])


#: A sentinel only the endpoint writes, in every field of its refusal it
#: may write: its code, detail, title, ``endpoint_code``, ``diagnosis`` and
#: another field. The owner chose to show the endpoint's own words on the
#: core route, as the old route does (decision E47, owner decision
#: 2026-10-10), so this one must reach ``str(exc)`` and the details.
ENDPOINT = 'SENTINEL-endpoint-5d1e'
_ENDPOINT_CODE = 'ENDPOINT_SENTINEL_5D1E'
_ENDPOINT_S_OWN = {
    'code': _ENDPOINT_CODE,
    'detail': f'this deployment does not serve that ({ENDPOINT})',
    'title': f'title {ENDPOINT}',
    'endpoint_code': f'{ENDPOINT}-code',
    'diagnosis': f'{ENDPOINT}-diagnosis',
    'hint': f'{ENDPOINT}-hint',
}


#: The ``Authorization`` header the transport builds of the URL's userinfo.
_BASIC_TOKEN = base64.b64encode(f'sdm:{DSN}'.encode()).decode()


def _shows_the_endpoint_s_own_words(error: NimRequestError) -> None:
    r"""The exception is a working one: the endpoint's own words are in its
    text and its details, as on the old route, and none of ours is."""
    assert error.status_code == 422
    assert error.code == _ENDPOINT_CODE
    assert ENDPOINT in str(error)
    assert _ENDPOINT_CODE in str(error)
    assert error.details['endpoint_code'] == f'{ENDPOINT}-code'
    assert error.details['diagnosis'] == f'{ENDPOINT}-diagnosis'
    assert error.details['hint'] == f'{ENDPOINT}-hint'


def _failing(kind: str) -> str:
    r"""A statement whose cast fails on the planted value of ``kind``,
    which the driver's message quotes."""
    return f"SELECT CAST(secret AS INTEGER) FROM vault WHERE kind = '{kind}'"


def _provider_error(kind: str) -> Callable[[], Any]:
    r"""The provider scenario: the test-only fetch, answered by the DuckDB
    provider on the planted source, with a plan of the failing statement."""
    fetch = {
        'request': {
            'request': 'scan',
            'table': {'source': ['vault']},
            'columns': [],
            'filters': [],
            'limit': None,
        },
        'plan': {
            'plan': 'sql',
            'statements': [
                {'dialect': 'duckdb', 'text': _failing(kind), 'params': []}
            ],
            'expect': {'expect': 'discover'},
        },
    }

    def call() -> Any:
        return run(
            {'op': 'test', 'test': 'fetch', 'fetch': 1},
            {1: codec.attachment(fetch)},
            provider=DuckDBProvider(_planted(vault=True)),
        )

    return call


def _scenarios() -> dict[str, Callable[[], Any]]:
    answer_under_another_id = json.dumps(
        {
            'id': 'r1',
            'model': 'kumo-relational',
            'metadata': {},
            'predictions': [
                {'id': DATA, 'row_index': 0, 'prediction': 1.0},
                {'id': DSN, 'row_index': 1, 'prediction': 1.0},
            ],
        }
    ).encode()
    answer_not_the_contract_s = json.dumps(
        {'id': [DATA], 'model': DSN, 'metadata': {}, 'predictions': []}
    ).encode()

    def answered(body: bytes, status: int = 200) -> Callable[[], Any]:
        def call() -> Any:
            with _answering(body, status) as url:
                return _predict(url)

        return call

    endpoint_s_refusal = json.dumps(_ENDPOINT_S_OWN).encode()

    def operation() -> Any:
        return run({'op': DATA, 'state': CREDENTIAL}, {})

    def attachment() -> Any:
        return run(
            {
                'op': 'validate_query',
                'query': 1,
                'schema': 2,
                'profile': {'name': 'RELATIONAL', 'model': None},
            },
            {
                1: f'"{DATA}'.encode(),
                2: json.dumps({'tables': CREDENTIAL, 'links': []}).encode(),
            },
        )

    def panic() -> Any:
        return run(
            {'op': 'test', 'test': 'panic', 'at': 'body', 'payload': PANIC}, {}
        )

    def transport() -> Any:
        return _predict(f'http://sdm:{DSN}@127.0.0.1:{_closed_port()}')

    return {
        'codec: an operation the contract cannot read': operation,
        'codec: an attachment that is not its JSON': attachment,
        'codec: an answer under another row s id': answered(
            answer_under_another_id
        ),
        'codec: an answer not the contract s': answered(
            answer_not_the_contract_s
        ),
        'provider: a statement failing on the credential': _provider_error(
            'dsn'
        ),
        'provider: a statement failing on a planted value': _provider_error(
            'data'
        ),
        'transport: an unreachable endpoint behind a credential': transport,
        'refusal: a 422 in the endpoint s own words behind a credential': (
            answered(endpoint_s_refusal, 422)
        ),
        'refusal: a 500 behind a credential': answered(
            b'{"code": "INTERNAL_ERROR", "detail": "internal failure"}', 500
        ),
        'panic: in the run s body': panic,
    }


@pytest.fixture(autouse=True)
def _core_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)


@pytest.mark.parametrize('name', list(_scenarios()))
def test_no_sentinel_reaches_anything_written(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    scenario = _scenarios()[name]
    with _watching(monkeypatch, capfd) as seen:
        error = _raised(seen, scenario)
    assert seen.leaks() == [], name
    path = name.split(':')[0]
    if path == 'refusal':
        # The endpoint's refusal, as the host raises it on the core route:
        # the old route's class, status, code, message and details, the
        # endpoint's own words (decision E47), nothing of ours, and no
        # cause and no context.
        assert isinstance(error, NimRequestError), name
        assert error.__cause__ is None and error.__context__ is None, name
        if 'own words' in name:
            _shows_the_endpoint_s_own_words(error)
    else:
        # The error is the core's, raised with no cause chained.
        assert error.__cause__ is None and error.__suppress_context__, name
        assert any(where == 'core error' for where, _ in seen.texts), name
    expected = {
        'codec': {'INVALID_MESSAGE', 'INVALID_RESPONSE'},
        'provider': {'PROVIDER_ERROR'},
        'transport': {'TRANSPORT_ERROR'},
        'refusal': {_ENDPOINT_CODE, 'INTERNAL_ERROR'},
        'panic': {'INTERNAL_ERROR'},
    }[path]
    assert error.code in expected, (name, error.code)


def test_each_sentinel_is_on_the_path_that_fails(
    capfd: pytest.CaptureFixture[str],
) -> None:
    # Data and the credential: the driver's own message quotes each.
    for kind, sentinel in (('data', DATA), ('dsn', DSN)):
        with pytest.raises(duckdb.Error) as driver:
            _planted(vault=True).execute(_failing(kind))
        assert sentinel in str(driver.value)
    # The credential: the URL the transport is given holds it.
    url = f'http://sdm:{DSN}@127.0.0.1:{_closed_port()}'
    with SDMClient(url, max_retries=0) as client:
        assert DSN in client._transport.url
    # The payloads: the planted values are sent to the model.
    with _endpoint().serve() as (served, received):
        _predict(served)
    [(_, _, body)] = received.requests
    assert DATA.encode() in body and DSN.encode() in body
    # The panic: it happened, its location written to stderr, its payload
    # not.
    capfd.readouterr()
    with pytest.raises(StructuredDataError) as panicked:
        run({'op': 'test', 'test': 'panic', 'at': 'body', 'payload': PANIC}, {})
    assert panicked.value.details['panic'] is True
    assert 'sdm core: a contained panic at ' in capfd.readouterr().err


def test_the_capture_sees_the_core_s_logs_and_the_journey_s_warnings(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    with _watching(monkeypatch, capfd) as seen:
        run({'op': 'test', 'test': 'log', 'hold_ms': 0}, {})
        _request()
    assert any(r.name.startswith('sdm_core') for r in seen.records)  # type: ignore[attr-defined]
    assert any('MISSING_INDICES' in w for w in seen.warnings)  # type: ignore[attr-defined]
    assert seen.leaks() == []


def _leaky_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""A provider mapping that writes the driver's text into a field."""
    mapped = run_module._driver

    def leaky(
        error: Exception, provider: Any = None
    ) -> tuple[dict[str, Any], str]:
        provided, kind = mapped(error, provider)
        return {**provided, 'sqlstate': str(error)[:200]}, kind

    monkeypatch.setattr(run_module, '_driver', leaky)


def _leaky_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""A sentence table that writes what the client was sent to into
    every error, as a formatter naming the endpoint would."""
    made = errors._instance

    def leaky(cls: Any, message: str, code: str, details: Any) -> Any:
        url = getattr(_leaky_transport, 'url', '')
        return made(cls, f'{message} at {url}', code, details)

    monkeypatch.setattr(errors, '_instance', leaky)


def _leaky_panic(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""A formatter that writes the operation it ran into the error."""
    made = errors._instance

    def leaky(cls: Any, message: str, code: str, details: Any) -> Any:
        return made(
            cls, f'{message} running panic payload {PANIC}', code, details
        )

    monkeypatch.setattr(errors, '_instance', leaky)


@pytest.mark.parametrize(
    ('name', 'leak', 'sentinel'),
    [
        (
            'provider: a statement failing on a planted value',
            _leaky_driver,
            DATA,
        ),
        (
            'transport: an unreachable endpoint behind a credential',
            _leaky_transport,
            DSN,
        ),
        ('panic: in the run s body', _leaky_panic, PANIC),
    ],
)
def test_a_leaky_formatter_is_caught(
    name: str,
    leak: Callable[[pytest.MonkeyPatch], None],
    sentinel: str,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    port = _closed_port()
    _leaky_transport.url = f'http://sdm:{DSN}@127.0.0.1:{port}'  # type: ignore[attr-defined]
    leak(monkeypatch)
    scenario = _scenarios()[name]
    if name.startswith('transport'):

        def scenario() -> Any:
            return _predict(_leaky_transport.url)  # type: ignore[attr-defined]

    with _watching(monkeypatch, capfd) as seen:
        _raised(seen, scenario)
    assert any(found == sentinel for _, found in seen.leaks()), seen.leaks()


def test_an_endpoint_echoing_our_data_is_shown_verbatim_as_on_the_old_route(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    r"""The owner-accepted exception (decision E47): an endpoint that echoes
    a value of our request in its refusal has that text shown, verbatim, as
    the old route shows it. It is the endpoint's text, not ours: nothing of
    our side (the credential in the URL) is added, and nothing is chained."""
    echoing = json.dumps(
        {
            'code': 'UNSUPPORTED_FEATURE',
            'detail': f'column country holds {DATA}',
            'invalid_params': [{'name': 'country', 'reason': DATA}],
        }
    ).encode()
    shown = {}
    with _answering(echoing, 422) as url:
        for route, routes in (
            ('core', CORE),
            (
                'old',
                ','.join(
                    _routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB
                ),
            ),
        ):
            monkeypatch.setenv(_routing.ENVIRONMENT, routes)
            with pytest.raises(NimRequestError) as raised:
                _predict(url)
            shown[route] = raised.value
    core, old = shown['core'], shown['old']
    assert DATA in str(core)
    assert (str(core), core.code, core.status_code, core.details) == (
        str(old),
        old.code,
        old.status_code,
        old.details,
    )
    assert core.__cause__ is None and core.__context__ is None
    formatted = ''.join(traceback.format_exception(core))
    assert DSN not in formatted + repr(core.details)


def test_the_client_s_own_explanation_names_schema_columns_never_values(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    r"""The boundary of what the client adds to the endpoint's words (E47,
    orchestrator reading): the adapter names the tables and columns of our
    schema a refusal is about, as on the old route, and never a value, a
    credential or a URL. Here the endpoint refuses a semantic type without
    naming a column, and the column it means holds the planted values."""
    request = _request()
    schema = request.payloads[0]['schema']
    schema['related_tables']['users']['columns']['country']['stype'] = 'text'
    refusal = json.dumps(
        {
            'code': 'UNSUPPORTED_FEATURE',
            'detail': "semantic type 'text' is unsupported",
        }
    ).encode()
    with _answering(refusal, 422) as url, _watching(monkeypatch, capfd) as seen:

        def call() -> Any:
            with SDMClient(url, max_retries=0) as client:
                return client.predict(request, model=KumoRelational())

        error = _raised(seen, call)
    assert isinstance(error, NimRequestError)
    assert 'users.country' in str(error)
    assert seen.leaks() == []
    assert error.__cause__ is None and error.__context__ is None


def test_an_endpoint_echoing_our_credential_is_shown_as_on_the_old_route(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    r"""As on the old route, an endpoint that echoes our credential (the
    Authorization header made of the URL's userinfo, and the password) in
    its own refusal has it shown: the core route's refusal is the old
    route's exactly (E47; a scrub of such echoes was tried in fix rounds 3
    and 4 and removed by change of method). Our own code adds nothing: the
    credential is nowhere in the exception but where the endpoint's text
    puts it, nothing is chained, and, on the core route, the canary's
    watchers (the core's error, every log record at DEBUG, every warning,
    stderr and stdout) find the credential, its token and the endpoint's
    sentinel written nowhere else."""
    echoing = json.dumps(
        {
            'code': 'UNAUTHORIZED',
            'detail': f'{{AUTH}} for sdm:{DSN} refused ({ENDPOINT})',
            'seen': {'header': '{AUTH}', 'password': DSN},
        }
    ).encode()
    shown = {}
    watched: dict[str, _Seen] = {}
    with _answering(echoing, 401) as url:
        for route, routes in (
            ('core', CORE),
            (
                'old',
                ','.join(
                    _routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB
                ),
            ),
        ):
            monkeypatch.setenv(_routing.ENVIRONMENT, routes)
            with _watching(monkeypatch, capfd) as seen:
                with pytest.raises(NimRequestError) as raised:
                    _predict(url)
            shown[route] = raised.value
            watched[route] = seen
    for route, seen in watched.items():
        # Not vacuous: the watchers saw the journey's own warning (the
        # missing indices) on both routes.
        assert any(where == 'warning' for where, _ in seen.texts), route
        written = [
            where
            for where, text in seen.texts
            for echoed in (DSN, _BASIC_TOKEN, ENDPOINT)
            if echoed in text
        ]
        assert written == [], route
    core, old = shown['core'], shown['old']
    assert (str(core), repr(core), core.code, core.status_code) == (
        str(old),
        repr(old),
        old.code,
        old.status_code,
    )
    assert core.details == old.details
    assert DSN in str(core) and _BASIC_TOKEN in str(core)
    assert ENDPOINT in str(core)
    assert core.__cause__ is None and core.__context__ is None
    # Beyond the endpoint's own text, the formatted exception holds none.
    formatted = ''.join(traceback.format_exception(core)).replace(str(core), '')
    assert DSN not in formatted and _BASIC_TOKEN not in formatted
