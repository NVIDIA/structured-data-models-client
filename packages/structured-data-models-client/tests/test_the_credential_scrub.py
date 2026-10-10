# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The credential scrub of the core route's refusals (``_rust/scrub.py``;
decision E47, orchestrator reading (a), owner to confirm): which
credentials it collects, every spelling it builds of them, and how it
replaces them in a value of any JSON shape. The end-to-end check, through
a refusal, is ``test_the_predict_route.py``'s."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from sdm_client._rust import scrub
from sdm_client.core.transport import Transport

REDACTED = scrub.REDACTED


def test_the_forms_are_every_spelling_of_an_exact_echo() -> None:
    pair = 'sdm:SECRET/PW'
    raw = pair.encode()
    standard = base64.b64encode(raw).decode()
    urlsafe = base64.urlsafe_b64encode(raw).decode()
    assert set(scrub.forms(['SECRET%2FPW', 'sdm:SECRET%2FPW'])) == {
        # The value raw, decoded, and percent-encoded in upper and lower
        # hex, raw and decoded.
        'SECRET%2FPW',
        'SECRET/PW',
        'SECRET%252FPW',
        'SECRET%2fPW',
        # The pair, the same, and its Basic token: standard and URL-safe
        # base64, padded and unpadded, of the pair as written and decoded.
        'sdm:SECRET%2FPW',
        'sdm:SECRET/PW',
        'sdm%3ASECRET%252FPW',
        'sdm%3aSECRET%252FPW',
        'sdm%3ASECRET%2FPW',
        'sdm%3aSECRET%2fPW',
        standard,
        standard.rstrip('='),
        urlsafe,
        urlsafe.rstrip('='),
        base64.b64encode(b'sdm:SECRET%2FPW').decode(),
        base64.b64encode(b'sdm:SECRET%2FPW').decode().rstrip('='),
        base64.urlsafe_b64encode(b'sdm:SECRET%2FPW').decode(),
        base64.urlsafe_b64encode(b'sdm:SECRET%2FPW').decode().rstrip('='),
    }
    # The longest first, so one holding another is replaced whole.
    found = scrub.forms(['ab', 'abcdef'])
    assert found == sorted(found, key=len, reverse=True)


def test_the_transport_s_credentials_are_collected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    netrc = tmp_path / 'netrc'
    netrc.write_text('machine 127.0.0.1 login nuser password NETRC-SECRET-9\n')
    netrc.chmod(0o600)
    monkeypatch.setenv('NETRC', str(netrc))
    transport = Transport(
        'http://sdm:SECRET%2FPW@127.0.0.1:9', api_key='KEY-123'
    )
    try:
        found = scrub.credentials(transport)
    finally:
        transport.close()
    assert set(found) >= {
        'SECRET%2FPW',
        'sdm',
        'sdm:SECRET/PW',
        'sdm:SECRET%2FPW',
        'KEY-123',
    }
    # From .netrc, as requests reads it for the URL when it sends.
    transport = Transport('http://127.0.0.1:9')
    try:
        found = scrub.credentials(transport)
    finally:
        transport.close()
    assert {'NETRC-SECRET-9', 'nuser', 'nuser:NETRC-SECRET-9'} <= set(found)
    # A workspace's token, password and client secret.
    workspace = SimpleNamespace(
        config=SimpleNamespace(
            token='dapiTOKEN', password=None, client_secret='CS-1'
        )
    )
    found = scrub.credentials(SimpleNamespace(_workspace=workspace))
    assert {'dapiTOKEN', 'CS-1'} <= set(found)


def test_a_user_name_under_three_characters_is_not_a_credential_alone() -> None:
    transport = SimpleNamespace(url='http://ab:pw@127.0.0.1:9')
    found = scrub.credentials(transport)
    assert 'ab' not in found
    assert {'pw', 'ab:pw'} <= set(found)


def test_a_short_form_is_replaced_only_as_a_whole_token() -> None:
    found = scrub.pattern(scrub.forms(['pw']))
    assert scrub.scrubbed('refused pw; a pwd, a password, pw.', found) == (
        f'refused {REDACTED}; a pwd, a password, {REDACTED}.'
    )
    # Every non-empty credential is scrubbed, a one-character one too.
    found = scrub.pattern(scrub.forms(['x']))
    assert (
        scrub.scrubbed('x marks the box', found) == f'{REDACTED} marks the box'
    )


def test_every_json_shape_is_scrubbed_keys_numbers_and_depth_included() -> None:
    found = scrub.pattern(scrub.forms(['SECRETPW', 'u:SECRETPW', '4321']))
    deep: object = 'SECRETPW'
    for _ in range(5000):
        deep = [deep]
    value = {
        'code': [
            'SECRETPW',
            {'h': 'Basic ' + base64.b64encode(b'u:SECRETPW').decode()},
        ],
        'SECRETPW': 1,
        'number': 4321,
        'float': 4321.0,
        'flag': True,
        'none': None,
        'deep': deep,
    }
    out = scrub.scrubbed(value, found)
    assert out['code'] == [REDACTED, {'h': f'Basic {REDACTED}'}]
    assert out[REDACTED] == 1
    # A number whose spelling holds a credential, as an integer or a float.
    assert out['number'] == REDACTED and out['float'] == REDACTED
    assert out['flag'] is True and out['none'] is None
    node = out['deep']
    for _ in range(5000):
        node = node[0]
    assert node == REDACTED


def test_keys_that_become_one_keep_both_values() -> None:
    found = scrub.pattern(scrub.forms(['SECRETPW']))
    out = scrub.scrubbed(
        {'a SECRETPW': 1, 'a <redacted>': 2, 'a <redacted>#2': 3}, found
    )
    assert sorted(out.values()) == [1, 2, 3]
    assert len(out) == 3


def test_no_credential_scrubs_nothing() -> None:
    assert scrub.pattern(scrub.forms([])) is None
    assert scrub.scrubbed({'a': ['b']}, None) == {'a': ['b']}
