# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public model-response decoding over HTTP and Databricks serving.

UTF-8, UTF-16 and UTF-32 variants preserve result values and metadata.
Malformed byte forms fail predictably; lone surrogates become U+FFFD.
Expected values come from the independently scripted endpoint response.
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


@functools.cache
def core() -> dict[str, Any]:
    r"""python-core's observation of every scenario, on the core routes."""
    return scenarios().observe()


def test_scenarios_cover_encodings_values_and_refusals() -> None:
    """Validate byte forms and expected results independently of the client."""
    assert len(NAMES) == 38 and len(set(NAMES)) == 38
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


@pytest.mark.parametrize('name', NAMES)
def test_response_encoding_preserves_values_or_refuses_malformed_bytes(
    name: str,
) -> None:
    query, form = name.split('.', 1)
    got = core()['answers'][name]
    assert got['calls'] == 1
    if form.startswith('malformed-'):
        assert all(
            got[field] is None
            for field in ('columns', 'values', 'request_id', 'metadata')
        )
        assert got['refusal']['error'] == 'StructuredDataError'
        assert got['refusal']['code'] == 'TRANSPORT_ERROR'
        return
    assert got['refusal'] is None
    assert got['columns'] == ['row_index', 'id', 'prediction']
    assert set(got['values']) == {'row_index', 'id', 'prediction'}
    for col in ('row_index', 'id'):
        assert got['values'][col] == [0, 1]
        assert all(type(v) is int for v in got['values'][col])
    prediction = 1.0 if query == 'windowed' else False
    assert got['values']['prediction'] == [prediction, prediction]
    assert all(type(v) is type(prediction) for v in got['values']['prediction'])
    assert got['request_id'] == ('x\ufffdy' if form in REPLACED else 'r1')
    assert got['metadata'] == ({'note': 'a\ufffdb'} if form in REPLACED else {})
