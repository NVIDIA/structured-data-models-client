# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P65.describer: the Python describer's content tags are right for every
``object`` column shape the fixtures use.

The describer hands the core Arrow arrays and, for an ``object`` column, a
content tag (DV19). Its tags are right when the core, given them, infers what
today's Python infers from the frame itself. So every ``object`` column of
every fixture under ``tests/parity/fixtures/``, and the shapes below, goes
through ``describe_frames`` and ``schema_from_data``, alone in a table and
beside an integer key, and the outcome must equal ``Schema.from_data``'s,
refusals included (a list or a bytes cell is refused by the core with
today's message).

The describer must also describe, not infer: it works with today's
inference unimportable, and gives the same tags every time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from _schema import (
    FIXTURES,
    HERE,
    INFERENCE_MODULES,
    REPO,
    describer,
    difference,
    loads,
    python_outcome,
    rust_from_data,
)

#: Shapes ``infer_tfm_dtype`` and ``require_scalar_cells`` tell apart that
#: the fixtures may not all hold, as fixture specs (``tests/parity/FORMAT.md``).
SHAPES: dict[str, list[Any]] = {
    'decimal-nan': [{'$decimal': 'NaN'}, {'$decimal': '1'}],
    'decimal-null': [{'$decimal': '2.5'}, None, {'$decimal': '1'}],
    'decimal-negative-wide': [{'$decimal': '-9223372036854775809'}],
    'decimal-exponent': [{'$decimal': '1E+3'}, {'$decimal': '2'}],
    'int-wider-than-int64': [{'$int': '1180591620717411303424'}, 1],
    'int-int64-min': [{'$int': '-9223372036854775808'}, 3],
    'int-just-below-int64-min': [{'$int': '-9223372036854775809'}, 3],
    'int-and-text': [1, 'a', 2],
    'int-and-float': [1, 2.5, 3],
    'int-and-bool': [1, True, 0],
    'int-and-decimal': [1, {'$decimal': '2'}],
    'numpy-ints': [
        {'$np': {'type': 'int64', 'value': 1}},
        {'$np': {'type': 'int64', 'value': 2}},
    ],
    'numpy-uint64-wide': [
        {'$np': {'type': 'uint64', 'value': {'$int': '18446744073709551615'}}},
    ],
    'text-with-none': ['a', None, 'b', 'a'],
    'text-many-distinct': [f'value-{i}' for i in range(60)],
    'text-few-distinct': ['x', 'y'] * 30,
    'text-date-like': ['2021-01-01', '2021-02-01', '2021-03-01'],
    'bool-with-none': [True, None, False],
    'float-with-nan': [1.5, {'$float': 'nan'}, 2.5],
    'float-whole': [1.0, 2.0, 3.0],
    'all-nan': [{'$float': 'nan'}, {'$float': 'nan'}],
    'na-and-ints': [{'$na': None}, 1, 2],
    'nat-and-timestamps': [
        {'$nat': None},
        {'$timestamp': '2024-01-01T00:00:00'},
    ],
    'timestamps': [
        {'$timestamp': '2024-01-01T00:00:00'},
        {'$timestamp': '2024-01-02T12:30:00'},
    ],
    'datetimes-naive': [
        {'$datetime': '2024-01-01T00:00:00'},
        None,
        {'$datetime': '2024-01-03T00:00:00'},
    ],
    'dates-and-text': [{'$date': '2024-01-01'}, 'b'],
    'uuids-and-none': [{'$uuid': '7470717b-4f3d-5c24-833d-a7fa0b50c286'}, None],
    'bytes-and-none': [None, {'$bytes': 'AAc='}],
    'mapping': [{'k': 1}, {'k': 2}],
    'tuple': [{'$tuple': [1, 2]}, {'$tuple': [3, 4]}],
    'list-after-scalars': [1, 2, [3]],
    'empty': [],
}


def _walk(value: Any, found: list[list[Any]]) -> None:
    if isinstance(value, dict):
        if value.get('dtype') == 'object' and isinstance(
            value.get('values'), list
        ):
            found.append(value['values'])
        for item in value.values():
            _walk(item, found)
    elif isinstance(value, list):
        for item in value:
            _walk(item, found)


def fixture_shapes() -> dict[str, list[Any]]:
    r"""Every ``object`` column of every fixture, by case id."""
    out: dict[str, list[Any]] = {}
    for path in sorted(FIXTURES.glob('*.json')):
        for case in json.loads(path.read_text())['cases']:
            found: list[list[Any]] = []
            _walk(case['input'], found)
            for i, values in enumerate(found):
                out[f'{case["id"]}#{i}'] = values
    return out


FIXTURE_SHAPES = fixture_shapes()
ALL_SHAPES = {**FIXTURE_SHAPES, **{f'extra/{k}': v for k, v in SHAPES.items()}}


def test_the_fixtures_hold_object_columns() -> None:
    assert len(FIXTURE_SHAPES) >= 50, (
        f'only {len(FIXTURE_SHAPES)} object columns found in the fixtures'
    )


def _frames(values: list[Any], keyed: bool) -> dict[str, Any]:
    import pandas as pd
    from _fixture_ops import build_series

    column = build_series({'dtype': 'object', 'values': values})
    if keyed:
        key = pd.Series(range(len(column)), dtype='int64')
        return {'t': pd.DataFrame({'id': key, 'c': column})}
    return {'t': pd.DataFrame({'c': column})}


def _python(frames: dict[str, Any]) -> dict[str, Any]:
    from sdm_client import Schema

    return python_outcome(lambda: Schema.from_data(frames))


@pytest.mark.parametrize('keyed', [False, True], ids=['alone', 'beside-a-key'])
@pytest.mark.parametrize('shape', list(ALL_SHAPES), ids=list(ALL_SHAPES))
def test_the_core_infers_what_python_infers(shape: str, keyed: bool) -> None:
    frames = _frames(ALL_SHAPES[shape], keyed)
    want = _python(frames)
    got = rust_from_data(_frames(ALL_SHAPES[shape], keyed))
    problem = difference(want, got)
    assert problem is None, f'{shape}: {problem}'


@pytest.mark.parametrize('shape', list(ALL_SHAPES), ids=list(ALL_SHAPES))
def test_every_object_column_has_a_tag(shape: str) -> None:
    module = describer()
    tables, tags = module.describe_frames(_frames(ALL_SHAPES[shape], True))
    import pyarrow as pa

    assert isinstance(tables['t'], pa.Table)
    assert tables['t'].num_rows == len(ALL_SHAPES[shape])
    found = loads(tags, 'describe_frames')
    assert isinstance(found, dict) and isinstance(found.get('t'), dict), found
    assert found['t'].get('c') is not None, (
        f'{shape}: the object column has no tag ({found})'
    )
    _, again = module.describe_frames(_frames(ALL_SHAPES[shape], True))
    assert loads(again, 'describe_frames') == found, 'the tags changed'


_ISOLATED = r"""
import json, sys
for name in sys.argv[2].split(','):
    sys.modules[name] = None
sys.path.insert(0, sys.argv[3])
from _fixture_ops import build_series
import pandas as pd
import sdm_core.describe as describe
shapes = json.loads(open(sys.argv[1]).read())
out = {}
for name, values in shapes.items():
    column = build_series({'dtype': 'object', 'values': values})
    frame = pd.DataFrame({'id': pd.Series(range(len(column)), dtype='int64'), 'c': column})
    tables, tags = describe.describe_frames({'t': frame})
    out[name] = json.loads(tags)
print(json.dumps(out))
"""


def test_the_describer_needs_no_inference(tmp_path: Path) -> None:
    r"""Run with ``sdm_client``, ``sdm_relational_engine`` and
    ``predictive_query`` unimportable, the describer gives the tags it gives
    with them."""
    module = describer()
    shapes = tmp_path / 'shapes.json'
    shapes.write_text(json.dumps(ALL_SHAPES))
    proc = subprocess.run(
        [
            sys.executable,
            '-c',
            _ISOLATED,
            str(shapes),
            ','.join(INFERENCE_MODULES),
            str(REPO / 'tests' / 'parity'),
        ],
        cwd=HERE,
        env={**os.environ, 'PYTHONHASHSEED': '0'},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-6000:]
    isolated = json.loads(proc.stdout)
    for name, values in ALL_SHAPES.items():
        _, tags = module.describe_frames(_frames(values, True))
        assert isolated[name] == loads(tags, 'describe_frames'), name
