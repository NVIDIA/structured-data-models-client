# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inert public-input builders shared with the independent reference recorder."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

NUMPY_WIDTHS = (
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'float16',
    'float32',
    'float64',
)
NUMPY_MIXTURES = (
    'signed_unsigned_63',
    'int64_float64_53',
    'float32_pythonfloat_24',
    'float64_huge_pythonint',
    'int64_huge_pythonint',
    'mixed_precision_triplet',
    'missing_numeric_owners',
)
NUMPY_PROMOTION_PAIRS = tuple(
    (integer, floating)
    for integer in NUMPY_WIDTHS[:8]
    for floating in NUMPY_WIDTHS[8:]
)

LABEL_FAMILIES = (
    (
        'complex64',
        'complex128',
        'python_complex_objects',
        'decimal_integer',
        'naive_datetime_objects',
        'date_objects',
        'fixed_offset_datetime_objects',
        'naive_aware_refusal',
        'date_datetime_mismatch',
        'decimal_text',
        'datetime_integer',
    )
    + tuple('numpy_' + name for name in NUMPY_WIDTHS)
    + tuple(
        'numpy_' + name + '_' + direction
        for name in NUMPY_MIXTURES
        for direction in ('forward', 'reverse')
    )
    + tuple(
        'numpy_pair_' + integer + '_' + floating
        for integer, floating in NUMPY_PROMOTION_PAIRS
    )
)


def label_frame(family: str) -> pd.DataFrame:
    fixed = timezone(timedelta(hours=2))
    patterns: dict[str, list[Any]] = {
        'complex64': [
            3 + 2j,
            1 + 4j,
            1 - 2j,
            complex(-0.0, 0),
            complex(0, -0.0),
            complex(float('nan'), 1),
        ],
        'complex128': [
            3 + 2j,
            1 + 4j,
            1 - 2j,
            complex(-0.0, 0),
            complex(0, -0.0),
            complex(1, float('nan')),
        ],
        'python_complex_objects': [complex(2, 1), complex(1, 2)],
        'decimal_integer': [
            Decimal('1.00'),
            Decimal('1.0'),
            2,
            Decimal('-3.5'),
            Decimal('1E+3'),
        ],
        'naive_datetime_objects': [
            datetime(2020, 1, 3),
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
        ],
        'date_objects': [date(2020, 1, 3), date(2020, 1, 1), date(2020, 1, 2)],
        'fixed_offset_datetime_objects': [
            datetime(2020, 1, 1, 1, tzinfo=fixed, fold=0),
            datetime(2020, 1, 1, 1, tzinfo=fixed, fold=1),
            datetime(2020, 1, 1, tzinfo=timezone.utc),
        ],
        'naive_aware_refusal': [
            datetime(2020, 1, 1),
            datetime(2020, 1, 1, tzinfo=fixed),
        ],
        'date_datetime_mismatch': [date(2020, 1, 1), datetime(2020, 1, 1)],
        'decimal_text': [Decimal('1.1'), 'x'],
        'datetime_integer': [datetime(2020, 1, 1), 1],
    }
    for name in NUMPY_WIDTHS:
        scalar = getattr(np, name)
        patterns['numpy_' + name] = [scalar(2), scalar(0), scalar(1), scalar(2)]
    mixtures = {
        'signed_unsigned_63': [np.int64(2**63 - 1), np.uint64(2**63)],
        'int64_float64_53': [np.int64(2**53 + 1), np.float64(2**53)],
        'float32_pythonfloat_24': [np.float32(2**24), float(2**24 + 1)],
        'float64_huge_pythonint': [np.float64(1), 10**400],
        'int64_huge_pythonint': [np.int64(1), 10**400],
        'mixed_precision_triplet': [
            np.float32(2**24),
            float(2**24 + 1),
            np.int64(2**24 + 1),
        ],
        'missing_numeric_owners': [
            np.float32(2),
            None,
            np.int16(-1),
            np.float64(float('nan')),
        ],
    }
    for name, pair in mixtures.items():
        patterns['numpy_' + name + '_forward'] = pair
        patterns['numpy_' + name + '_reverse'] = pair[::-1]
    for integer, floating in NUMPY_PROMOTION_PAIRS:
        integer_type = getattr(np, integer)
        float_type = getattr(np, floating)
        exponent = {'float16': 12, 'float32': 25, 'float64': 54}[floating]
        boundary = min(int(np.iinfo(integer_type).max) - 1, 2**exponent)
        patterns['numpy_pair_' + integer + '_' + floating] = [
            integer_type(boundary + 1),
            float_type(boundary),
            integer_type(0),
            float_type(0),
        ]
    pattern = patterns[family]
    labels = [pattern[i % len(pattern)] for i in range(81)]
    if family in ('complex64', 'complex128'):
        index = pd.Index(np.asarray(labels, dtype=family))
        assert index.dtype == np.dtype(family)
    else:
        values = np.empty(81, dtype=object)
        for i, label in enumerate(labels):
            values[i] = label
        index = pd.Index(values, dtype=object, copy=False)
        assert all(index._values[i] is labels[i] for i in range(81))
    data = pd.DataFrame(
        {
            'id': range(81),
            'feature': np.arange(81) * 0.5,
            'target': np.arange(81) + 100.0,
        },
        index=index,
    )
    assert data.index is index
    return data


INDEX_FAMILIES = (
    'range',
    'signed_duplicates',
    'unsigned',
    'floating_missing',
    'boolean',
    'datetime',
    'timezone',
    'timedelta',
    'period',
    'categorical',
    'multi',
    'string',
    'object_mixed',
    'nullable_integer',
    'arrow_integer',
    'arrow_string',
)


def indexed_frame(family: str) -> pd.DataFrame:
    n = 81
    data = pd.DataFrame(
        {
            'id': range(n),
            'feature': [i * 0.5 for i in range(n)],
            'target': [100.0 + i for i in range(n)],
        }
    )
    labels = [(i * 13) % 17 for i in range(n)]
    if family == 'range':
        index = pd.RangeIndex(n, 0, -1)
    elif family == 'signed_duplicates':
        index = pd.Index(labels, dtype='int64')
    elif family == 'unsigned':
        index = pd.Index([2**63 + value for value in labels], dtype='uint64')
    elif family == 'floating_missing':
        index = pd.Index(
            [
                float('nan') if i % 7 == 0 else float(value)
                for i, value in enumerate(labels)
            ]
        )
    elif family == 'boolean':
        index = pd.Index([bool(value % 2) for value in labels])
    elif family in ('datetime', 'timezone'):
        index = pd.DatetimeIndex(
            [
                pd.NaT
                if i % 7 == 0
                else pd.Timestamp('2020-01-01') + pd.Timedelta(days=value)
                for i, value in enumerate(labels)
            ]
        )
        if family == 'timezone':
            index = index.tz_localize('US/Pacific')
    elif family == 'timedelta':
        index = pd.TimedeltaIndex(
            [
                pd.NaT if i % 7 == 0 else pd.Timedelta(days=value)
                for i, value in enumerate(labels)
            ]
        )
    elif family == 'period':
        index = pd.PeriodIndex(
            [
                pd.NaT if i % 7 == 0 else pd.Period('2020-01', freq='M') + value
                for i, value in enumerate(labels)
            ],
            freq='M',
        )
    elif family == 'categorical':
        index = pd.CategoricalIndex(
            [
                None if i % 7 == 0 else str(value)
                for i, value in enumerate(labels)
            ],
            categories=[str(value) for value in range(16, -1, -1)],
            ordered=False,
        )
    elif family == 'multi':
        index = pd.MultiIndex.from_tuples(
            [
                (None if i % 7 == 0 else value % 3, str(value))
                for i, value in enumerate(labels)
            ],
            names=['a', 'b'],
        )
    elif family == 'string':
        index = pd.Index(
            [
                None if i % 7 == 0 else str(value)
                for i, value in enumerate(labels)
            ],
            dtype='str',
        )
    elif family == 'object_mixed':
        index = pd.Index(
            [
                str(value) if i % 3 == 0 else value
                for i, value in enumerate(labels)
            ],
            dtype=object,
        )
    elif family == 'nullable_integer':
        index = pd.Index(
            [None if i % 7 == 0 else value for i, value in enumerate(labels)],
            dtype='Int64',
        )
    elif family == 'arrow_integer':
        index = pd.Index(
            [None if i % 7 == 0 else value for i, value in enumerate(labels)],
            dtype='int64[pyarrow]',
        )
    elif family == 'arrow_string':
        index = pd.Index(
            [
                None if i % 7 == 0 else str(value)
                for i, value in enumerate(labels)
            ],
            dtype='string[pyarrow]',
        )
    else:
        raise AssertionError(family)
    data.index = index
    return data
