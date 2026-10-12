# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records ``labels.json``, the reference's answers that ``labels.rs``
holds ``sdm_engine::labels`` and ``sdm_engine::time::offset`` to (T1.27,
decision E19; register PK-2 and PK-3).

Run once, by hand, under the reference's environment (Python 3.12, pandas
3, pyarrow 25, the reference checkout's ``predictive_query`` and
``sdm_relational_engine``); CI never runs it (F4):

    python rust/crates/sdm-engine/tests/labels_record.py OUT.json

Every answer is the reference's own code's, never re-derived here:

- ``offsets``: ``anchor + pd.DateOffset(**{unit: n})`` for anchors at month
  ends, on leap days and with nanoseconds, as the sampler's ``_by_time``
  seeds and the executor's window add them, and ``timeframe`` of a window
  of months (``DateOffsetRange.timeframe``), each in nanoseconds or
  ``null`` where pandas cannot hold the answer;
- ``aggregations``: ``PQueryPandasExecutor.execute_aggregation_type`` over
  seeded groups of float32, int64 and decimal values with nulls and NaN,
  the values each label takes and which examples have one, a float32
  answer as its bits;
- ``conditions``: ``execute_rel_op`` of a float32 or int64 series and a
  constant, as the query writes it;
- ``decimals``: a ``DECIMAL(38, 18)`` value read at float32, as pandas'
  ``astype('float32')`` reads it.
"""

from __future__ import annotations

import decimal
import json
import random
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
from predictive_query.ast import Constant
from predictive_query.ast.date_offset_range import DateOffsetRange
from predictive_query.schema.typing import AggregationType, RelOp
from sdm_relational_engine.rfm.query.pandas_executor import (
    PQueryPandasExecutor,
)

rng = random.Random(27)
EXECUTOR = PQueryPandasExecutor()


def ns(ts: pd.Timestamp) -> int:
    return int(ts.value)


def offsets() -> list[dict]:
    anchors = [
        '2024-01-31',
        '2024-02-29',
        '2024-03-31',
        '2023-02-28',
        '2023-03-31 23:59:59.999999999',
        '2024-02-29 12:34:56.123456789',
        '2000-02-29',
        '1900-02-28',
        '2100-02-28 06:00',
        '2262-03-31',
        '1677-10-31',
        '1970-01-01',
        '1969-12-31 23:59:59.5',
    ]
    for _ in range(20):
        anchors.append(
            str(pd.Timestamp(rng.randrange(-(2**62), 2**62), unit='ns'))
        )
    counts = [0, 1, -1, 2, -2, 11, 12, 13, -13, 25, -25, 30, 59, -59, 120]
    units = ['months', 'days', 'weeks', 'hours', 'minutes', 'seconds']
    out = []
    for text in anchors:
        anchor = pd.Timestamp(text)
        for unit in units:
            for n in counts:
                try:
                    answer = ns(anchor + pd.DateOffset(**{unit: n}))
                except (
                    OverflowError,
                    ValueError,
                    pd.errors.OutOfBoundsDatetime,
                ):
                    answer = None
                out.append(
                    {
                        'anchor': ns(anchor),
                        'unit': unit,
                        'count': n,
                        'at': answer,
                    }
                )
    return out


def timeframes() -> list[dict]:
    out = []
    for start in range(-14, 14):
        for end in range(start + 1, start + 15):
            for unit in ['months', 'days', 'weeks']:
                span = DateOffsetRange(start, end, unit).timeframe
                out.append(
                    {
                        'start': start,
                        'end': end,
                        'unit': unit,
                        'nanos': span.value,
                    }
                )
    return out


def f32_bits(value: float) -> str:
    return f'{np.float32(value).view(np.uint32):08x}'


def values_of(kind: str, size: int) -> tuple[pd.Series, list]:
    r"""A column of ``kind`` as the executor holds it after reading it, and
    its rows as written to the vectors: a float32's bits, an int's
    digits, a decimal's unscaled digits, ``null`` for a missing value."""
    rows: list = []
    if kind == 'float32':
        pool = [0.1, 1e8, 1.0, -2.5, 3.3, 1e-3, 16777217.0, -0.0, 7.0]
        values = []
        for _ in range(size):
            pick = rng.random()
            if pick < 0.1:
                values.append(np.nan)
            elif pick < 0.15:
                values.append(np.inf if rng.random() < 0.5 else -np.inf)
            else:
                values.append(
                    rng.choice(pool)
                    if rng.random() < 0.5
                    else rng.uniform(-1e6, 1e6)
                )
        series = pd.Series(np.array(values, dtype=np.float32))
        rows = [f32_bits(v) for v in series]
        return series, rows
    if kind == 'int64':
        values = [
            rng.choice(
                [0, 1, -1, 2**40, -(2**40), 2**62, rng.randrange(-1000, 1000)]
            )
            for _ in range(size)
        ]
        series = pd.Series(np.array(values, dtype=np.int64))
        return series, [str(v) for v in values]
    assert kind == 'decimal'
    unscaled = [rng.randrange(-(10**22), 10**22) for _ in range(size)]
    array = pa.array(
        [decimal.Decimal(v).scaleb(-18) for v in unscaled],
        type=pa.decimal128(38, 18),
    )
    series = pd.Series(array, dtype=pd.ArrowDtype(pa.decimal128(38, 18)))
    return series, [str(v) for v in unscaled]


def aggregations() -> list[dict]:
    out = []
    ops = {
        'sum': AggregationType.SUM,
        'count': AggregationType.COUNT,
        'avg': AggregationType.AVG,
        'min': AggregationType.MIN,
        'max': AggregationType.MAX,
    }
    for kind in ['float32', 'int64', 'decimal']:
        for name, op in ops.items():
            if kind == 'decimal' and name == 'avg':
                continue
            for _ in range(12):
                examples = rng.randrange(1, 6)
                size = rng.randrange(0, 25)
                feat, rows = values_of(kind, size)
                batch = np.array(
                    sorted(rng.randrange(examples) for _ in range(size)),
                    dtype=np.int64,
                )
                if rng.random() < 0.5:
                    order = list(range(size))
                    rng.shuffle(order)
                    feat = feat.iloc[order].reset_index(drop=True)
                    rows = [rows[i] for i in order]
                    batch = batch[order]
                # The executor's own reading of a column first drops what
                # is missing, as execute_column(filter_na=True) does.
                present = feat.notna().to_numpy()
                label, mask = EXECUTOR.execute_aggregation_type(
                    op,
                    feat=feat[present].reset_index(drop=True),
                    batch=batch[present],
                    batch_size=examples,
                    filter_na=True,
                )
                out.append(
                    {
                        'aggregation': name,
                        'kind': kind,
                        'examples': examples,
                        'batch': batch.tolist(),
                        'values': [
                            r if p else None
                            for r, p in zip(rows, present, strict=True)
                        ],
                        'labels': [f32_bits(v) for v in label],
                        'mask': mask.tolist(),
                    }
                )
    return out


def conditions() -> list[dict]:
    out = []
    constants = [
        ('int', '0'),
        ('int', '1'),
        ('int', '-3'),
        ('int', '16777217'),
        ('int', '9007199254740993'),
        ('float', '0.1'),
        ('float', '1.5'),
        ('float', '-2.5'),
        ('float', '16777216.5'),
        ('float', '1e8'),
    ]
    ops = {
        'eq': RelOp.EQ,
        'ne': RelOp.NEQ,
        'lt': RelOp.LT,
        'le': RelOp.LEQ,
        'gt': RelOp.GT,
        'ge': RelOp.GEQ,
    }
    for kind in ['float32', 'int64']:
        for _ in range(4):
            if kind == 'float32':
                pool = [
                    0.0,
                    0.1,
                    1.0,
                    1.5,
                    -2.5,
                    16777216.0,
                    16777218.0,
                    1e8,
                    -0.0,
                ]
                left = pd.Series(
                    np.array(
                        [rng.choice(pool) for _ in range(20)] + [np.nan],
                        dtype=np.float32,
                    )
                )
                rows = [f32_bits(v) for v in left]
            else:
                pool = [
                    0,
                    1,
                    -3,
                    2,
                    16777217,
                    9007199254740993,
                    9007199254740992,
                ]
                left = pd.Series(
                    np.array(
                        [rng.choice(pool) for _ in range(20)], dtype=np.int64
                    )
                )
                rows = [str(v) for v in left]
            for type_, text in constants:
                value = int(text) if type_ == 'int' else float(text)
                for name, op in ops.items():
                    answer = EXECUTOR.execute_rel_op(
                        left=left, op=op, right=Constant.from_value(value)
                    )
                    out.append(
                        {
                            'kind': kind,
                            'left': rows,
                            'op': name,
                            'constant': {'type': type_, 'text': text},
                            'answer': answer.tolist(),
                        }
                    )
    return out


def decimals() -> list[dict]:
    unscaled = [5, 1, 10**18, 97 * 10**18 + 2, 2**53, 2**53 + 1, -(2**53) - 3]
    for _ in range(800):
        digits = rng.choice([1, 5, 10, 15, 18, 19, 20, 25, 30, 37])
        unscaled.append(rng.randrange(-(10**digits), 10**digits))
    series = pd.Series(
        pa.array(
            [decimal.Decimal(v).scaleb(-18) for v in unscaled],
            type=pa.decimal128(38, 18),
        ),
        dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
    )
    read = series.astype('float32').to_numpy()
    return [
        {'unscaled': str(v), 'scale': 18, 'f32': f'{r.view(np.uint32):08x}'}
        for v, r in zip(unscaled, read, strict=True)
    ]


def main() -> None:
    document = {
        'format': 'sdm-labels-vectors/1',
        'about': (
            "The reference's answers sdm_engine::labels and "
            'sdm_engine::time::offset are held to, recorded by '
            'labels_record.py under the reference environment '
            f'(pandas {pd.__version__}, pyarrow {pa.__version__}, '
            f'numpy {np.__version__})'
        ),
        'offsets': offsets(),
        'timeframes': timeframes(),
        'aggregations': aggregations(),
        'conditions': conditions(),
        'decimals': decimals(),
    }
    # One vector a line, so a change to one reads as a change to its line.
    lines = ['{']
    keys = list(document)
    for k, key in enumerate(keys):
        value = document[key]
        end = ',' if k < len(keys) - 1 else ''
        if not isinstance(value, list):
            lines.append(f'{json.dumps(key)}: {json.dumps(value)}{end}')
            continue
        lines.append(f'{json.dumps(key)}: [')
        for i, row in enumerate(value):
            comma = ',' if i < len(value) - 1 else ''
            lines.append(json.dumps(row, separators=(',', ':')) + comma)
        lines.append(f']{end}')
    lines.append('}')
    with open(sys.argv[1], 'w', encoding='utf-8') as out:
        out.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
