# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.36, register P1B-21: ``AVG(...) = NULL`` and ``!= NULL`` as the reference.

Written by the package's independent test author, before the fix.

    python -m pytest contract/tests/test_t136_average_null.py -q

Until T1.36 the core refused a windowed ``AVG`` compared with ``NULL`` on
every source (``UNSUPPORTED_REQUEST``, rule ``average_null``), before
fetching anything. The reference does not fail on every source: it labels
the context wherever each of its training trials has a non-null value in
its window (``= NULL`` every label ``False``, ``!= NULL`` every label
``True``, a binary classification), and fails only where one has none
(``IndexError``, ``INTERNAL_ERROR``; with no value at all, a ``ValueError``
the client gives as ``INVALID_REQUEST`` in numpy's words).
``contract/tools/average_null_record.py`` says why, and which rows are the
trials: the first 10,000 of the training-trial list (the context size asked
narrows the context afterwards, not the trials; the predicted rows are not
labelled).

``contract/t136-average-null-vectors.json`` holds the reference's answers,
recorded by that script with the reference checkout's interpreter; each
source is asked with both operators. Here each runs forced onto the core
(``SDM_CORE_ROUTES``: the schema's and materialize's routes):

- where the reference labels (``expect: labels``), the core must build the
  payload, and its task, its context's and predicted instance rows (their
  labels among them), the instance ids each related table keeps and the
  counts must equal the reference's (locked check 5: an input the
  reference accepts is not refused);
- where the reference fails (``expect: fails``), the core must refuse it by
  name, ``UNSUPPORTED_REQUEST`` with rule ``average_null`` (E66(b): the
  reference's failure is an internal or an unnamed one), and neither
  ``INTERNAL_ERROR`` nor a panic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from sdm_client import _routing

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / 'contract/t136-average-null-vectors.json'
RECORDED = json.loads(VECTORS.read_text(encoding='utf-8'))
CORE = ','.join(_routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB)
#: What a payload's comparison holds equal to the reference's.
COMPARED = (
    'parts',
    'task',
    'counts',
    'context',
    'context_related',
    'predict',
    'predict_related',
)


def _recorder() -> ModuleType:
    name = 'average_null_record'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'contract/tools/average_null_record.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _recorder()
CASES = RECORDED['cases']
LABELLED = [c for c in CASES if c['expect'] == 'labels']
FAILED = [c for c in CASES if c['expect'] == 'fails']


def _labels(case: dict[str, Any]) -> set[Any]:
    rows = case['expected']['context']['rows']
    columns = case['expected']['context']['columns']
    return {row[columns.index('TARGET')] for row in rows}


def _on_the_core(
    spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    return R.observe(spec)


def test_the_recording_is_the_reference_s() -> None:
    # The vectors came from the reference by the committed recorder, its
    # cases in its order, each source asked with both operators; where the
    # reference labels, = NULL labels every row False and != NULL True, a
    # binary classification; where it fails, it fails INTERNAL_ERROR from
    # an IndexError, or with no value at all INVALID_REQUEST from a
    # ValueError (numpy's broadcast).
    assert RECORDED['recorded_by'] == 'contract/tools/average_null_record.py'
    assert len(RECORDED['reference']) == 40
    assert [c['name'] for c in CASES] == [c['name'] for c in R.CASES]
    for case, spec in zip(CASES, R.CASES, strict=True):
        assert {k: case[k] for k in spec} == spec, case['name']
    assert len(LABELLED) >= 10 and len(FAILED) >= 10
    for case in CASES:
        assert case['query'].count('NULL') == 1, case['name']
        equals = ' = NULL ' in case['query']
        expected = case['expected']
        if case['expect'] == 'labels':
            assert expected['outcome'] == 'payload', case['name']
            assert expected['task']['kind'] == 'binary_classification'
            assert expected['counts']['num_context_examples'] > 0
            assert _labels(case) == {not equals}, case['name']
        else:
            assert expected['outcome'] == 'refused', case['name']
            assert (expected['code'], expected['cause'].split(':')[0]) in {
                ('INTERNAL_ERROR', 'IndexError'),
                ('INVALID_REQUEST', 'ValueError'),
            }, case['name']
            if expected['code'] == 'INVALID_REQUEST':
                assert 'could not be broadcast' in expected['cause']


@pytest.mark.parametrize('spec', LABELLED, ids=[c['name'] for c in LABELLED])
def test_the_core_labels_what_the_reference_labels(
    spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    got = _on_the_core(spec, monkeypatch)
    assert got['outcome'] == 'payload', (
        f'refused ({got.get("code")}, rule {got.get("rule")!r}) where the '
        f'reference labels every row {" = NULL " not in spec["query"]}: '
        f'{got.get("message")}'
    )
    expected = spec['expected']
    for key in COMPARED:
        assert got[key] == expected[key], key


@pytest.mark.parametrize('spec', FAILED, ids=[c['name'] for c in FAILED])
def test_the_core_refuses_by_name_where_the_reference_fails(
    spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    got = _on_the_core(spec, monkeypatch)
    assert got['outcome'] == 'refused', (
        'labelled where the reference fails '
        f'({spec["expected"]["code"]}, {spec["expected"]["cause"]})'
    )
    assert (got['code'], got['rule']) == (
        'UNSUPPORTED_REQUEST',
        'average_null',
    ), got['message']
    assert 'StructuredDataError' in got['mro'], got['mro']
    assert got['class'] == 'UnsupportedRequestError', got['class']
