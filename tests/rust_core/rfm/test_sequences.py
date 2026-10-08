# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A sequence column read back as text, by the core and by today's Python.

``Table._sanitize`` reads a list or ``sequence`` column held as text with
``np.fromstring(row.strip('[]'), sep=',')``, falls back to ``json.loads``
where numpy refuses a row, and keeps the text where that fails too
(``rust/crates/sampling/src/sequences.rs``). The cases of
``tests/parity/fixtures/sampler-sequences.json`` pin the edges; here many
generated rows, made of the pieces numpy and ``json`` read differently, go
through both ``sampling.to_list`` runners. The core must give what Python
gives, except where every row is JSON and one is not a list numpy reads:
there Python gives whatever ``json`` makes of the rows, and the core must
refuse with a ``TypeError`` instead.

Without the core or the engine installed, as in CI's test job, it skips.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip('sdm_core')
pytest.importorskip('sdm_relational_engine')

REPO = Path(__file__).resolve().parents[3]
if str(REPO / 'tests' / 'parity') not in sys.path:
    sys.path.insert(0, str(REPO / 'tests' / 'parity'))

from _fixture_ops import FixtureError, run, same  # noqa: E402

#: The pieces rows are made of.
PIECES = (
    '0',
    '1',
    '7',
    '42',
    '-',
    '+',
    '.',
    'e',
    'E',
    ',',
    ', ',
    ' ',
    '\t',
    '\n',
    '[',
    ']',
    'nan',
    'NaN',
    'inf',
    'Infinity',
    '-Infinity',
    'x',
    '0x1',
    '_',
    '"',
    '"a"',
    '{',
    '}',
    ':',
    'true',
    'null',
    '\\u0041',
    '\\ud800',
    '1e400',
    '9223372036854775808',
    '00',
    '1.5',
    '()',
    'é',
)
SEED = 20261003
ROWS = 4_000


def _rows() -> list[str]:
    rng = np.random.RandomState(SEED)
    out = []
    for _ in range(ROWS):
        n = int(rng.randint(0, 8))
        out.append(''.join(PIECES[i] for i in rng.randint(0, len(PIECES), n)))
    return out


def _core(input: dict[str, Any]) -> dict[str, Any]:
    native = importlib.import_module('sdm_core._native')
    return json.loads(native.sampling_op('sampling.to_list', json.dumps(input)))


def _python(input: dict[str, Any]) -> dict[str, Any]:
    try:
        return {
            'output': json.loads(json.dumps(run('sampling.to_list', input)))
        }
    except FixtureError as error:
        return {'error': {'code': error.code, 'message': error.message}}


def _made_by_json(output: dict[str, Any]) -> bool:
    return any(
        not isinstance(v, (str, dict)) and v is not None
        for v in output['values']
    )


@pytest.mark.parametrize(
    ('dtype', 'stype'),
    [('string', 'sequence'), ('intlist', None), ('stringlist', None)],
)
def test_generated_rows(dtype: str, stype: str | None) -> None:
    problems = []
    refused = read = kept = 0
    for row in _rows():
        input = {
            'series': 'arrow',
            'values': [row, '[1]'],
            'dtype': dtype,
            'stype': stype,
        }
        want, got = _python(input), _core(input)
        if 'output' in want and _made_by_json(want['output']):
            refused += 1
            if got.get('error', {}).get('code') != 'TypeError':
                problems.append(
                    f'{row!r}: json made {want}, the core gave {got}'
                )
            continue
        if want != got and not (
            'output' in want
            and 'output' in got
            and same(want['output'], got['output'])
        ):
            problems.append(f'{row!r}: Python gave {want}, the core {got}')
            continue
        if want['output']['dtype'] == 'object':
            read += 1
        else:
            kept += 1
    assert not problems, f'{len(problems)} rows differ:\n' + '\n'.join(
        problems[:20]
    )
    # Each way a row goes is taken many times over.
    assert min(read, kept) > 100 or dtype == 'stringlist', (read, kept, refused)
    assert refused > 20, (read, kept, refused)
