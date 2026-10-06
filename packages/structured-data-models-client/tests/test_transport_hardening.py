# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regressions for the transport, and for the two defects review caught.

The transport here is a graft: the POST and session surface came from one
client, the redirect hardening from another. Both defects below were introduced
by that graft and survived because none of the donors' transport tests came
across with the code. These are the ones that would have caught them.
"""

from __future__ import annotations

import pytest
import requests
import requests_mock

from sdm_client import SDMClient
from sdm_client.core.transport import Transport
from sdm_client.errors import NimRequestError, StructuredDataError

_REMOTE = 'nim.example'


# --- a credential must never leave over an unencrypted connection ----------


@pytest.mark.parametrize(
    'url',
    [
        f'http://user:secret@{_REMOTE}',
        f'http://user@{_REMOTE}',
        f'http://:secret@{_REMOTE}',
    ],
)
def test_userinfo_in_the_url_is_refused_over_plaintext_to_a_remote_host(url):
    """``requests`` turns URL userinfo into ``Authorization: Basic``.

    Checking only the ``api_key`` argument let this leave in the clear.
    """
    with pytest.raises(StructuredDataError) as error:
        Transport(url)
    assert error.value.code == 'INVALID_CONFIGURATION'
    assert 'secret' not in str(error.value)


def test_an_api_key_is_still_refused_over_plaintext_to_a_remote_host():
    with pytest.raises(StructuredDataError):
        Transport(f'http://{_REMOTE}', 'a-key')


@pytest.mark.parametrize(
    'url',
    [
        'http://user:secret@127.0.0.1:8000',
        'http://user:secret@localhost:8000',
        f'https://user:secret@{_REMOTE}',
    ],
)
def test_a_credential_is_allowed_where_it_cannot_be_read_off_the_wire(url):
    """Localhost never leaves the machine; https is encrypted."""
    transport = Transport(url)
    try:
        assert transport.url.startswith(('http://', 'https://'))
    finally:
        transport.close()


def test_a_credential_in_the_url_is_not_rendered_back_to_the_caller():
    transport = Transport('https://user:secret@nim.example')
    try:
        assert 'secret' not in repr(
            SDMClient('https://user:secret@nim.example')
        )
    finally:
        transport.close()


# --- the retry=False branch ------------------------------------------------


def test_a_readiness_check_without_retries_reaches_the_network():
    """It built its one-shot session without the TLS argument and raised."""
    transport = Transport('http://127.0.0.1:9', max_retries=0)
    try:
        with pytest.raises(StructuredDataError) as error:
            transport.health_ready(retry=False)
        assert error.value.code == 'TRANSPORT_ERROR'
    finally:
        transport.close()


def test_a_readiness_check_reports_a_200_as_ready():
    transport = Transport('https://nim.example')
    try:
        with requests_mock.Mocker() as mock:
            mock.get('https://nim.example/v1/health/ready', status_code=200)
            assert transport.health_ready() is True
            mock.get('https://nim.example/v1/health/ready', status_code=503)
            assert transport.health_ready() is False
    finally:
        transport.close()


# --- what the endpoint says --------------------------------------------


def test_list_models_reads_the_endpoint_rather_than_a_local_list():
    transport = Transport('https://nim.example')
    try:
        with requests_mock.Mocker() as mock:
            mock.get(
                'https://nim.example/v1/models',
                json={
                    'object': 'list',
                    'data': [
                        {'id': 'kumo-tabular'},
                        {'id': 'something-else'},
                    ],
                },
            )
            assert transport.list_models() == [
                'kumo-tabular',
                'something-else',
            ]
    finally:
        transport.close()


def test_an_endpoint_without_the_models_route_is_an_error_not_a_guess():
    """The route is the runtime's, not the contract's, so it may be absent."""
    transport = Transport('https://nim.example')
    try:
        with requests_mock.Mocker() as mock:
            mock.get('https://nim.example/v1/models', status_code=404)
            with pytest.raises((NimRequestError, StructuredDataError)):
                transport.list_models()
    finally:
        transport.close()


# --- session ids are put into a path -------------------------------------


def test_a_session_id_is_escaped_before_it_becomes_a_path_segment():
    transport = Transport('https://nim.example')
    try:
        with requests_mock.Mocker() as mock:
            mock.delete(requests_mock.ANY, status_code=204)
            transport.delete_session('a/../b')
            path = mock.request_history[-1].path
            assert '/../' not in path
    finally:
        transport.close()


# --- errors --------------------------------------------------------------


def test_an_error_status_becomes_a_nim_error_carrying_the_status():
    transport = Transport('https://nim.example', timeout=2.0, max_retries=0)
    try:
        with requests_mock.Mocker() as mock:
            mock.post(
                'https://nim.example/v1/predictions',
                status_code=422,
                json={
                    'title': 'Request validation failed.',
                    'code': 'VALIDATION_FAILED',
                },
            )
            with pytest.raises(NimRequestError) as error:
                transport.predict({'model': 'kumo-tabular'})
            assert error.value.status_code == 422
    finally:
        transport.close()


def test_a_transport_failure_is_reported_without_the_url_credential():
    url = 'https://user:secret@nim.example'
    transport = Transport(url, timeout=1.0, max_retries=0)
    try:
        with requests_mock.Mocker() as mock:
            mock.post(
                f'{url}/v1/predictions',
                exc=requests.ConnectTimeout('timed out'),
            )
            with pytest.raises(StructuredDataError) as error:
                transport.predict({'model': 'kumo-tabular'})
            assert 'secret' not in str(error.value)
    finally:
        transport.close()


def test_a_closed_transport_refuses_every_later_call():
    transport = Transport('https://nim.example')
    transport.close()
    with pytest.raises(StructuredDataError) as error:
        transport.predict({'model': 'kumo-tabular'})
    assert error.value.code == 'INVALID_CONFIGURATION'


@pytest.mark.parametrize(
    ('timeout', 'retries'),
    [(0, 3), (-1.0, 3), (60.0, -1), (60.0, 'three'), ('long', 3)],
)
def test_unusable_transport_limits_are_refused_at_construction(
    timeout, retries
):
    with pytest.raises(StructuredDataError):
        Transport('https://nim.example', timeout=timeout, max_retries=retries)


@pytest.mark.parametrize(
    'url',
    [
        'http://nim.example?api_key=query-secret',
        'https://nim.example?token=query-secret',
        'https://nim.example/api?access_token=query-secret',
        'https://nim.example#fragment',
    ],
)
def test_a_base_url_carrying_a_query_or_fragment_is_refused(url):
    """A route is appended to this string, so a query breaks the URL as well.

    ``http://host?api_key=x`` became ``http://host/?api_key=x/v1/predictions``:
    a malformed request line, with the secret in it, that redaction never saw
    because it only ever stripped userinfo.
    """
    with pytest.raises(StructuredDataError) as error:
        Transport(url)
    assert error.value.code == 'INVALID_CONFIGURATION'
    assert 'query-secret' not in str(error.value)


@pytest.mark.parametrize(
    'url',
    [
        'https://nim.example',
        'http://localhost:8000',
        'https://nim.example/api/v2',
    ],
)
def test_an_ordinary_base_url_with_a_path_prefix_is_still_accepted(url):
    transport = Transport(url)
    try:
        assert transport.url == url.rstrip('/')
    finally:
        transport.close()


def test_a_payload_nested_too_deeply_is_refused_not_raised_raw():
    """Checked before encoding, not by catching what the encoder raises.

    Whether `json.dumps` raises `RecursionError` is an implementation detail:
    CPython 3.12 raises on a 30,000 level mapping and 3.14 encodes the same
    value happily. A guard written around the exception protects one
    interpreter and not the next, and this package supports both.
    """
    from sdm_client import SDMClient
    from sdm_client.errors import StructuredDataError

    payload: dict = {}
    cursor = payload
    for _ in range(30_000):
        nxt: dict = {}
        cursor['x'] = nxt
        cursor = nxt

    with SDMClient(url='https://example.invalid') as client:
        with pytest.raises(StructuredDataError) as error:
            client._transport._post('/v1/predictions', payload)
    assert error.value.code == 'INVALID_REQUEST'
    assert 'nests more than' in str(error.value)


def test_an_adapter_that_adds_nothing_does_not_become_its_own_cause():
    """`raise error from error` makes an exception its own `__cause__`.

    Both adapters return the same object when they have nothing to explain,
    so this was the common case, and anything walking the cause chain for
    logging loops on it.
    """
    from sdm_client import SDMClient
    from sdm_client.errors import StructuredDataError

    class Unchanged:
        def explain_error(self, error, payload):
            return error

    class Explains:
        def explain_error(self, error, payload):
            return StructuredDataError('explained', code='INVALID_REQUEST')

    class Refusing:
        def predict(self, payload):
            raise StructuredDataError('boom', code='INVALID_REQUEST')

    client = SDMClient(url='https://example.invalid')
    client._transport = Refusing()

    with pytest.raises(StructuredDataError) as error:
        client._send(Unchanged(), {})
    assert error.value.__cause__ is not error.value

    with pytest.raises(StructuredDataError) as error:
        client._send(Explains(), {})
    assert error.value.__cause__ is not error.value
    assert isinstance(error.value.__cause__, StructuredDataError)


def test_a_payload_that_refers_back_to_itself_is_refused():
    from sdm_client import SDMClient
    from sdm_client.errors import StructuredDataError

    cyclic: dict = {}
    cyclic['self'] = cyclic
    with SDMClient(url='https://example.invalid') as client:
        with pytest.raises(StructuredDataError) as error:
            client._transport._post('/v1/predictions', cyclic)
    assert 'refers back to itself' in str(error.value)


def test_the_check_does_not_depend_on_the_encoder_raising():
    """The point of the guard: it must refuse the value that `json.dumps`
    on this interpreter happens to accept."""
    import json

    from sdm_client.core.transport import _check_encodable
    from sdm_client.errors import StructuredDataError

    deep: dict = {}
    cursor = deep
    for _ in range(30_000):
        nxt: dict = {}
        cursor['x'] = nxt
        cursor = nxt

    encoder_raised = False
    try:
        json.dumps(deep)
    except RecursionError:
        encoder_raised = True

    with pytest.raises(StructuredDataError):
        _check_encodable(deep)
    assert isinstance(encoder_raised, bool)


def test_an_ordinary_payload_and_a_repeated_subobject_are_accepted():
    """The same object referenced twice is a DAG, not a cycle."""
    from sdm_client.core.transport import _check_encodable

    shared = {'x': 1}
    _check_encodable({'a': [1, 2, {'b': 3}], 'p': shared, 'q': [shared]})


def test_a_non_ascii_body_is_encoded_here_not_by_requests():
    """A `str` body is handed to httplib, which encodes latin-1.

    That only worked because `json.dumps` defaults to `ensure_ascii=True`.
    Encoding to utf-8 here removes the dependency on a default nobody will
    remember when it changes.
    """
    import requests_mock

    from sdm_client import SDMClient

    with SDMClient(url='https://example.invalid') as client:
        with requests_mock.Mocker() as mock:
            mock.post(
                'https://example.invalid/v1/predictions',
                json={'id': 'x', 'model': 'kumo-relational', 'predictions': []},
            )
            client._transport._post('/v1/predictions', {'note': 'naïve café'})
            assert isinstance(mock.last_request.body, bytes)


def test_merge_raises_the_clients_own_error_type():
    """`except StructuredDataError` is the documented catch for this client.

    These three raises were bare `ValueError`, so a caller following the
    documentation missed them, on the same premise as the engine-error
    classification beside them.
    """
    from sdm_client.errors import StructuredDataError
    from sdm_client.result import PredictionResult

    with pytest.raises(StructuredDataError) as error:
        PredictionResult.merge([])
    assert error.value.code == 'INTERNAL_ERROR'


def test_a_data_field_that_is_not_a_list_is_refused_however_falsy():
    """`.get('data') or []` normalised every falsy value before the check, so
    an empty object, an empty string, `0` and `False` were all reported as
    "no models" rather than as the malformed body they are."""
    import json as _json

    from sdm_client.core.transport import Transport
    from sdm_client.errors import StructuredDataError

    class Response:
        status_code = 200

        def __init__(self, body):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def iter_content(self, size):
            yield self._body

    class Session:
        def __init__(self, body):
            self._body = body

        def get(self, *args, **kwargs):
            return Response(self._body)

        def close(self):
            return None

    def models_for(data):
        transport = Transport('http://x.invalid', max_retries=0)
        transport._session = Session(_json.dumps({'data': data}).encode())
        return transport.list_models()

    for data in ({}, '', 0, False, 'abc', 5):
        with pytest.raises(StructuredDataError) as raised:
            models_for(data)
        assert 'Expected a list of models' in str(raised.value)

    assert models_for([]) == []
    assert models_for(None) == []
