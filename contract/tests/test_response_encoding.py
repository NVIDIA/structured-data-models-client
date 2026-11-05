# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public model-response decoding over HTTP and Databricks serving.

UTF-8, UTF-16 and UTF-32 variants preserve result values and metadata.
Malformed byte forms fail predictably; lone surrogates become U+FFFD.
Retained observations provide independent expectations without an old runtime.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / 'contract/support/response_encoding_scenarios.py'


def scenarios() -> ModuleType:
    name = 'response_encoding_scenarios'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SCENARIOS)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


#: Each scenario's form, by the name the vectors give it: the HTTP forms,
#: then the serving client's.
FORMS = (
    *scenarios().FORMS,
    *(f'serving-{form}' for form in scenarios().SERVING),
)
#: The forms whose lone surrogates the core reads as U+FFFD (register
#: P1B-24).
REPLACED = frozenset(scenarios().REPLACED)
#: A lone surrogate: one not in a pair, as Python's ``str`` holds it.
LONE = re.compile('[\ud800-\udfff]')
NAMES = tuple(
    f'{query}.{form}' for query in scenarios().QUERY_NAMES for form in FORMS
)


def _replaced(value: Any) -> Any:
    r"""``value`` with each lone surrogate in its texts U+FFFD."""
    if isinstance(value, str):
        return LONE.sub('\ufffd', value)
    if isinstance(value, dict):
        return {_replaced(k): _replaced(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_replaced(v) for v in value]
    return value


def _named(refusal: dict[str, Any]) -> tuple[str, str]:
    return (refusal['error'], refusal['code'])


@functools.cache
def vectors() -> dict[str, Any]:
    return scenarios().load()


@functools.cache
def core() -> dict[str, Any]:
    r"""python-core's observation of every scenario, on the core routes."""
    return scenarios().observe()


def test_scenarios_cover_encodings_values_and_refusals() -> None:
    """Validate byte forms and expected results independently of the client."""
    document = vectors()
    assert set(document['answers']) == set(NAMES)
    want = {'windowed': [1.0, 1.0], 'condition': [False, False]}
    for name, observed in document['answers'].items():
        query, form = name.split('.', 1)
        assert observed['calls'] == 1, name
        if form in REPLACED:
            # The reference reads it and keeps each surrogate.
            assert observed['refusal'] is None, name
            assert observed['values']['prediction'] == want[query], name
            assert observed['request_id'] == scenarios().LONE_ID, name
            assert observed['metadata'] == {'note': scenarios().LONE_NOTE}, name
        elif form.startswith('malformed-'):
            assert observed['refusal']['code'] == 'TRANSPORT_ERROR', name
            assert observed['values'] is None, name
        else:
            assert observed['refusal'] is None, name
            assert observed['values']['prediction'] == want[query], name
    text = '{"object": "prediction.response", "id": "r1"}'
    detected = {
        form: json.detect_encoding(write(text))
        for form, write in scenarios().FORMS.items()
    }
    assert detected == {
        'utf-8': 'utf-8',
        'utf-8-sig': 'utf-8-sig',
        'utf-16-le-bom': 'utf-16',
        'utf-16-be-bom': 'utf-16',
        'utf-16-le': 'utf-16-le',
        'utf-16-be': 'utf-16-be',
        'utf-32-le-bom': 'utf-32',
        'utf-32-be-bom': 'utf-32',
        'utf-32-le': 'utf-32-le',
        'utf-32-be': 'utf-32-be',
        'malformed-utf-16-odd': 'utf-16-le',
        'malformed-utf-8-invalid': 'utf-8',
        'lone-surrogate-escaped': 'utf-8',
        'lone-surrogate-utf-8': 'utf-8',
        'lone-surrogate-utf-16': 'utf-16',
        'lone-surrogate-utf-32': 'utf-32',
    }
    for form, write in scenarios().FORMS.items():
        if form.startswith('malformed-'):
            with pytest.raises(UnicodeDecodeError):
                json.loads(write(text))
        elif not form.startswith('lone-'):
            assert json.loads(write(text)) == json.loads(text), form
    # Each lone-surrogate form holds its surrogates as it says, and
    # Python's reader keeps them.
    answer = '{"id": "r1", "metadata": {}}'
    kept = {
        'id': scenarios().LONE_ID,
        'metadata': {'note': scenarios().LONE_NOTE},
    }
    for form, write in scenarios().FORMS.items():
        if form.startswith('lone-'):
            assert json.loads(write(answer)) == kept, form
    raw = scenarios().SERVING['lone-surrogate'](answer)
    escaped = scenarios().SERVING['lone-surrogate-escaped'](answer)
    assert LONE.search(raw) and '\\u' not in raw
    assert not LONE.search(escaped) and '\\ud800' in escaped
    assert json.loads(raw) == json.loads(escaped) == kept
    assert b'\xed\xa0\x80' in scenarios().FORMS['lone-surrogate-utf-8'](answer)


@pytest.mark.parametrize(
    'name', [n for n in NAMES if n.split('.', 1)[1] not in REPLACED]
)
def test_an_answer_in_each_encoding_is_read_as_the_reference_reads_it(
    name: str,
) -> None:
    r"""Codex P2 (whole-phase round two): an answer the endpoint writes in
    an encoding the reference's ``json.loads`` takes gives the reference's
    predictions on the core route, and one no such encoding decodes is
    refused as the reference refuses it."""

    def check() -> None:
        want = vectors()['answers'][name]
        got = core()['answers'][name]
        if want['refusal'] is not None:
            # A refusal by its class and code: the core route's transport
            # failure is said in a fixed sentence that names no URL (H17),
            # as tests/test_t128b_predict_result.py holds the reference's
            # own not-JSON answers.
            want = {**want, 'refusal': _named(want['refusal'])}
            got = {
                **got,
                'refusal': got['refusal'] and _named(got['refusal']),
            }
        assert got == want, (
            f'{name}: {json.dumps(got)[:400]} where python-ref has '
            f'{json.dumps(want)[:400]}'
        )

    check()


@pytest.mark.parametrize(
    'name', [n for n in NAMES if n.split('.', 1)[1] in REPLACED]
)
def test_a_lone_surrogate_in_an_answer_is_read_as_u_fffd(name: str) -> None:
    r"""Register P1B-24 (the owner's rule of 2026-10-21): an answer holding
    lone surrogates, which the reference reads and keeps, is answered on the
    core route, each lone surrogate U+FFFD: the predictions, the request id
    and the metadata are the reference's with only that character
    differing, as a JSON escape and written as itself in UTF-8, UTF-16 and
    UTF-32, and from a Databricks serving endpoint (where the answer's text
    was refused, or escaped as a bare ``UnicodeEncodeError``)."""

    def check() -> None:
        want = vectors()['answers'][name]
        got = core()['answers'][name]
        assert LONE.search(json.dumps(want, ensure_ascii=False)), want
        replaced = _replaced(want)
        assert replaced['request_id'] == scenarios().LONE_ID.replace(
            '\ud800', '\ufffd'
        )
        assert got == replaced, (
            f'{name}: {ascii(got)[:400]} where python-ref, its lone '
            f'surrogates U+FFFD, has {ascii(replaced)[:400]}'
        )

    check()
