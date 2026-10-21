# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A model answer in each encoding the reference's reader takes, on
``python-core`` (T2.16's test author's, from the whole-phase review's round
two, Codex P2).

    python -m pytest contract/tests/test_t216_response_encoding.py -q

Since T2.14b the core reads the endpoint's answer from the bytes the
endpoint sent. The reference reads them with ``json.loads``, which takes
UTF-8 with or without its byte order mark, and UTF-16 and UTF-32 of either
byte order with or without theirs (``json.detect_encoding``); the core
refused all but UTF-8 as ``INVALID_RESPONSE`` (``field=response``,
``reason=not_json``).

Each scenario of ``contract/tools/t216_encoding_record.py`` runs on
python-core with the core routes forced, by the recorder's own
``observe``, against a real localhost endpoint through
``Schema.from_source``, ``PredictiveQuery.materialize`` and
``SDMClient.predict``, and is held to what the reference gave
(``contract/t216-encoding-vectors.json``): the result's columns and values
for each encoding, and the refusal for two byte forms no encoding decodes.

An answer holding a lone surrogate is a difference the core keeps
(register P1B-24, the owner's rule of 2026-10-21): the reference keeps the
surrogate, and the core reads it as U+FFFD and answers. The answer's id
(the result's ``request_id``) and its metadata each hold one, as a JSON
escape, and written as itself in UTF-8, UTF-16 and UTF-32, over HTTP and
from a Databricks serving endpoint (the recorder's ``REPLACED``). The
core's observation is held to the reference's with each lone surrogate
U+FFFD, and nothing else differing. The serving client's plain answer
(``serving-plain``) is held to the reference's as the HTTP forms are.

The core's own reader on the same byte forms is
``rust/crates/sdm-core/tests/t216_answer_encoding.rs``.

Gating: ``t216-pending.json``'s ``pending`` names, as ``py:<query>.<form>``,
the tests that failed when written. Each must still fail (strict); once
one passes it fails, naming itself, until it is taken out.
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
RECORDER = ROOT / 'contract/tools/t216_encoding_record.py'
PENDING = Path(__file__).with_name('t216-pending.json')
PREFIX = 'py:'


def recorder() -> ModuleType:
    name = 't216_encoding_record'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, RECORDER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


#: Each scenario's form, by the name the vectors give it: the HTTP forms,
#: then the serving client's.
FORMS = (
    *recorder().FORMS,
    *(f'serving-{form}' for form in recorder().SERVING),
)
#: The forms whose lone surrogates the core reads as U+FFFD (register
#: P1B-24).
REPLACED = frozenset(recorder().REPLACED)
#: A lone surrogate: one not in a pair, as Python's ``str`` holds it.
LONE = re.compile('[\ud800-\udfff]')
NAMES = tuple(
    f'{query}.{form}' for query in recorder().QUERY_NAMES for form in FORMS
)


def held(name: str, check: Any) -> None:
    r"""Runs ``check``: it must pass, or, while ``name`` is pending, fail."""
    key = f'{PREFIX}{name}'
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert key in gating['authored'], f'{key} is not authored'
    if key not in gating['pending']:
        check()
        return
    try:
        check()
    except Exception:  # noqa: BLE001 - pending: failing, as when written
        return
    raise AssertionError(
        f'{key} passes now: take it out of '
        'contract/tests/t216-pending.json pending'
    )


def test_the_pending_list_only_shrinks() -> None:
    r"""Control: ``pending`` names only keys the test author wrote; this
    file's authored keys are its scenarios."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert set(gating['pending']) <= set(gating['authored'])
    mine = {k for k in gating['authored'] if k.startswith(PREFIX)}
    assert mine == {f'{PREFIX}{name}' for name in NAMES}


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
    return recorder().load()


@functools.cache
def core() -> dict[str, Any]:
    r"""python-core's observation of every scenario, on the core routes."""
    return recorder().observe(core=True)


def test_the_vectors_are_the_reference_s_and_exercise_what_they_name() -> None:
    r"""Control: recorded from the reference commit the contract names, by
    the committed recorder; every encoding answered with the rule's
    predictions, each of the two malformed forms refused; and each form's
    bytes are what its name says, as ``json.detect_encoding`` finds them."""
    document = vectors()
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {document["reference"]}' in env
    assert document['recorded_by'] == 'contract/tools/t216_encoding_record.py'
    assert set(document['answers']) == set(NAMES)
    want = {'windowed': [1.0, 1.0], 'condition': [False, False]}
    for name, observed in document['answers'].items():
        query, form = name.split('.', 1)
        assert observed['calls'] == 1, name
        if form in REPLACED:
            # The reference reads it and keeps each surrogate.
            assert observed['refusal'] is None, name
            assert observed['values']['prediction'] == want[query], name
            assert observed['request_id'] == recorder().LONE_ID, name
            assert observed['metadata'] == {'note': recorder().LONE_NOTE}, name
        elif form.startswith('malformed-'):
            assert observed['refusal']['code'] == 'TRANSPORT_ERROR', name
            assert observed['values'] is None, name
        else:
            assert observed['refusal'] is None, name
            assert observed['values']['prediction'] == want[query], name
    text = '{"object": "prediction.response", "id": "r1"}'
    detected = {
        form: json.detect_encoding(write(text))
        for form, write in recorder().FORMS.items()
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
    for form, write in recorder().FORMS.items():
        if form.startswith('malformed-'):
            with pytest.raises(UnicodeDecodeError):
                json.loads(write(text))
        elif not form.startswith('lone-'):
            assert json.loads(write(text)) == json.loads(text), form
    # Each lone-surrogate form holds its surrogates as it says, and
    # Python's reader keeps them.
    answer = '{"id": "r1", "metadata": {}}'
    kept = {
        'id': recorder().LONE_ID,
        'metadata': {'note': recorder().LONE_NOTE},
    }
    for form, write in recorder().FORMS.items():
        if form.startswith('lone-'):
            assert json.loads(write(answer)) == kept, form
    raw = recorder().SERVING['lone-surrogate'](answer)
    escaped = recorder().SERVING['lone-surrogate-escaped'](answer)
    assert LONE.search(raw) and '\\u' not in raw
    assert not LONE.search(escaped) and '\\ud800' in escaped
    assert json.loads(raw) == json.loads(escaped) == kept
    assert b'\xed\xa0\x80' in recorder().FORMS['lone-surrogate-utf-8'](answer)


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

    held(name, check)


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
        assert replaced['request_id'] == recorder().LONE_ID.replace(
            '\ud800', '\ufffd'
        )
        assert got == replaced, (
            f'{name}: {ascii(got)[:400]} where python-ref, its lone '
            f'surrogates U+FFFD, has {ascii(replaced)[:400]}'
        )

    held(name, check)
