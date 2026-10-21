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

Two scenarios are differences the core keeps (register P1B-24, an answer
holding a lone surrogate, which the reference reads and keeps): a UTF-16
answer whose id is a lone surrogate (``lone-surrogate-utf-16``), and a
Databricks serving answer whose ``response_json`` holds one written as
itself (``serving-lone-surrogate``). The reference's success is recorded;
the core's refusal by name is pinned (the recorder's ``REFUSED`` and
``SERVING_REFUSED``). The serving client's plain answer
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
#: The forms the core refuses where the reference reads them (register
#: P1B-24): the code and details pinned.
REFUSED = {
    **recorder().REFUSED,
    **{f'serving-{k}': v for k, v in recorder().SERVING_REFUSED.items()},
}
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
        if form in REFUSED:
            # The reference reads it: the difference the core keeps.
            assert observed['refusal'] is None, name
            assert observed['values']['prediction'] == want[query], name
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
        'lone-surrogate-utf-16': 'utf-16',
    }
    for form, write in recorder().FORMS.items():
        if form.startswith('malformed-'):
            with pytest.raises(UnicodeDecodeError):
                json.loads(write(text))
        elif form not in REFUSED:
            assert json.loads(write(text)) == json.loads(text), form
    # The lone surrogate is in the bytes, and Python's reader keeps it.
    answer = '{"id": "r1"}'
    assert json.loads(recorder().FORMS['lone-surrogate-utf-16'](answer)) == {
        'id': 'x\ud800'
    }
    served = recorder().SERVING['lone-surrogate'](answer)
    assert '\ud800' in served and '\\ud800' not in served


@pytest.mark.parametrize(
    'name', [n for n in NAMES if n.split('.', 1)[1] not in REFUSED]
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
    'name', [n for n in NAMES if n.split('.', 1)[1] in REFUSED]
)
def test_an_answer_holding_a_lone_surrogate_is_refused_by_name(
    name: str,
) -> None:
    r"""Register P1B-24: an answer holding a lone surrogate, which the
    reference reads, is refused by the core as ``INVALID_RESPONSE``,
    ``field=response``, ``reason=not_json``, with the category the
    difference names: ``encoding`` for UTF-16 bytes no UTF-8 text holds,
    ``syntax`` for a Databricks ``response_json`` holding one written as
    itself (not a bare ``UnicodeEncodeError``)."""
    code, details = REFUSED[name.split('.', 1)[1]]

    def check() -> None:
        got = core()['answers'][name]
        refusal = got['refusal']
        assert refusal is not None, f'{name}: answered {json.dumps(got)}'
        assert (refusal['error'], refusal['code']) == (
            'StructuredDataError',
            code,
        ), refusal
        assert {k: refusal['details'].get(k) for k in details} == details, (
            refusal
        )
        assert got['calls'] == 1, got

    held(name, check)
