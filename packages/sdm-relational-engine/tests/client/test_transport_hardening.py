# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

r"""Guards on what the RFM transport accepts back from a NIM.

Both cases here are about a response the client does not control: an error body
large enough to be unreadable, and a session id chosen by the server.
"""

from __future__ import annotations

from typing import Any

import pytest
from sdm_api.endpoints import Endpoint
from sdm_relational_engine.client.client import _MAX_BODY_SNIPPET, capped_body
from sdm_relational_engine.client.rfm import RFMAPI, _path_segment
from sdm_relational_engine.client.utils import raise_on_error
from sdm_relational_engine.exceptions import (
    HTTPException,
    InvalidResponseError,
)


class _Response:
    r"""The four members ``raise_on_error`` reads."""

    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text
        self.ok = status_code < 400

    def json(self) -> Any:
        raise ValueError('not json')


class _RecordingClient:
    r"""Captures the path RFMAPI builds, without sending anything."""

    def __init__(self) -> None:
        self.paths: list[str] = []

    def _request(self, endpoint: Endpoint, **kwargs: Any) -> _Response:
        self.paths.append(endpoint.get_path())
        return _Response(200, '{}')


def test_error_body_is_capped() -> None:
    r"""A proxy can answer with a large HTML page; it must not become the
    whole exception message.
    """
    with pytest.raises(HTTPException) as info:
        raise_on_error(_Response(502, '<html>' + 'x' * 200_000 + '</html>'))

    detail = info.value.detail
    assert len(detail) < _MAX_BODY_SNIPPET + 100
    assert 'truncated' in detail
    assert '200013 chars total' in detail


def test_short_error_body_is_untouched() -> None:
    with pytest.raises(HTTPException) as info:
        raise_on_error(_Response(422, 'categorical cardinality 15001'))
    assert info.value.detail == 'categorical cardinality 15001'


def test_capped_body_collapses_whitespace() -> None:
    assert capped_body('a\n\n  b\tc') == 'a b c'
    assert capped_body(None) == ''
    assert capped_body('') == ''


@pytest.mark.parametrize(
    'session_id,expected',
    [
        ('sess-123', 'sess-123'),
        ('../v1/predictions', '..%2Fv1%2Fpredictions'),
        ('a/b', 'a%2Fb'),
        ('a?b=c', 'a%3Fb%3Dc'),
        ('a b', 'a%20b'),
    ],
)
def test_path_segment_escapes_server_chosen_ids(
    session_id: str, expected: str
) -> None:
    assert _path_segment(session_id) == expected


@pytest.mark.parametrize('session_id', ['.', '..', ''])
def test_a_session_id_of_only_dots_is_refused(session_id: str) -> None:
    r"""``quote`` leaves a dot alone, because it is unreserved, and a
    resolver then removes ``.`` and ``..``. An empty id contributes no
    segment at all. None of the three survives as the segment it was
    written as, so the call would reach a route other than the one named.
    """
    with pytest.raises(InvalidResponseError):
        _path_segment(session_id)


@pytest.mark.parametrize('session_id', ['...', '....', 'a.b', '.hidden'])
def test_a_segment_of_more_than_two_dots_is_ordinary(session_id: str) -> None:
    r"""A resolver removes ``.`` and ``..`` and no others, so refusing every
    run of dots would turn away ids a server may legitimately choose.
    """
    assert _path_segment(session_id) == session_id


def test_session_predict_escapes_the_session_id() -> None:
    r"""The id comes back from the NIM, so a slash in it would otherwise
    redirect the call to a different route.
    """
    client = _RecordingClient()
    api = RFMAPI(client)  # type: ignore[arg-type]
    try:
        api.session_predict(
            '../predictions', {'predict': {}}, entity_ids=[1], instance_ids=[0]
        )
    except Exception:  # the fake returns no usable body; the path is the point
        pass
    assert client.paths == ['/v1/sessions/..%2Fpredictions/predictions']


def test_delete_session_escapes_the_session_id() -> None:
    client = _RecordingClient()
    api = RFMAPI(client)  # type: ignore[arg-type]
    api.delete_session('../../etc')
    assert client.paths == ['/v1/sessions/..%2F..%2Fetc']


def test_every_redirect_hop_is_released_unread() -> None:
    r"""``_read_capped`` bounds the response this client parses, but it only
    runs after ``requests`` returns. Redirect hops are consumed before that, in
    full and uncapped, so each is released instead.
    """
    from unittest import mock

    import requests
    from sdm_relational_engine.client.client import _Session

    class _Raw:
        def __init__(self) -> None:
            self.closed = False
            self.reads = 0

        def close(self) -> None:
            self.closed = True

        def read(self, *args: object, **kwargs: object) -> bytes:
            self.reads += 1
            return b'x' * 1024 if self.reads == 1 else b''

    def _redirect(url: str, location: str) -> requests.Response:
        response = requests.Response()
        response.raw = _Raw()  # type: ignore[assignment]
        response.status_code = 307
        response.url = url
        response.headers['Location'] = location
        return response

    session = _Session()
    first = _redirect('https://a.test/v1/predictions', 'https://b.test/x')
    second = _redirect('https://b.test/x', 'https://c.test/y')
    final = requests.Response()
    final.status_code = 200
    final.url = 'https://c.test/y'
    final.raw = second.raw  # type: ignore[assignment]

    sent = [second, final]

    def _send(
        _self: object, request: object, **kwargs: object
    ) -> requests.Response:
        response = sent.pop(0)
        response.request = request  # type: ignore[assignment]
        return response

    prepared = session.prepare_request(
        requests.Request('GET', 'https://a.test/v1/predictions')
    )
    first.request = prepared

    with mock.patch.object(_Session, 'send', _send):
        list(session.resolve_redirects(first, prepared))

    assert first.raw.closed and second.raw.closed  # type: ignore[union-attr]
    assert first.raw.reads == 0  # type: ignore[union-attr]
    assert second.raw.reads == 0  # type: ignore[union-attr]
