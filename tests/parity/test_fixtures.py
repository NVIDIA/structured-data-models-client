# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Every parity fixture case, run against today's Python.

Each JSON file under ``tests/parity/fixtures/`` holds ``cases``: an operation,
its input, and either the output it gives (``expected``) or the error it
raises (``expected_error``: its class, code, message and the keys of its
``details``), and the warnings it gives its caller (``expected_warnings``).
One test per case, named by the case's id. ``SDM_PARITY_FIXTURES_DIR`` reads
another directory.

The operations are in ``_fixture_ops.py``; the format is in ``FORMAT.md``.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fixture_ops import FixtureError, run, same, warned  # noqa: E402

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / 'rust_core/request')
)

from _fixture_expectations import (  # noqa: E402
    CASE_ID,
    expected_case,
    validate_catalog,
)

FIXTURES = Path(
    os.environ.get(
        'SDM_PARITY_FIXTURES_DIR', Path(__file__).resolve().parent / 'fixtures'
    )
)


def _cases() -> list[dict[str, Any]]:
    out = []
    for path in sorted(FIXTURES.rglob('*.json')):
        out += json.loads(path.read_text(encoding='utf-8'))['cases']
    return out


CASES = _cases()


@pytest.mark.parametrize('case', CASES, ids=[case['id'] for case in CASES])
def test_fixture(case: dict[str, Any]) -> None:
    pytest.importorskip('sdm_relational_engine')
    if case['id'] == CASE_ID:
        from sdm_client._impl import OLD

        validate_catalog([case], FIXTURES)
        case = expected_case(case, core=not OLD)
    refused = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            got = json.loads(json.dumps(run(case['operation'], case['input'])))
        except FixtureError as error:
            refused = True
            assert 'expected_error' in case, (
                f'raised {error.code}: {error.message}, expected '
                f'{case["expected"]!r}'
            )
            want = case['expected_error']
            assert error.code == want['code'], (error.code, error.message)
            for key, got_value in error.described().items():
                if key in want:
                    assert got_value == want[key], (key, got_value, want[key])
            got = None
    if case['id'] == CASE_ID:
        assert refused, 'the governed refusal must actually be raised'
    if 'expected_warnings' in case:
        assert warned(caught) == case['expected_warnings']
    if got is None and 'expected_error' in case:
        return
    assert 'expected' in case, (
        f'returned {got!r}, expected the error {case["expected_error"]!r}'
    )
    assert same(got, case['expected']), (got, case['expected'])
