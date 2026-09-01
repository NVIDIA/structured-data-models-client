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

from sdm_client import StructuredDataClient
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
            StructuredDataClient('https://user:secret@nim.example')
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
