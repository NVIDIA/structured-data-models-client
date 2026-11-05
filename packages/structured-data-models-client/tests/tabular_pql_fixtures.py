# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inert public-input builders shared with the independent reference recorder."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

INDEX_SAMPLE_COUNTS = {
    'sampled_sorted_duplicates_missing': 320,
    'sampled_constant_missing': 320,
    'sampled_present_nan': 320,
}
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
    'missing_pandas_na',
    'missing_pandas_nat',
    'missing_nan_huge_integer',
    'only_missing_numpy_owner',
    'longlong_unsigned_63',
)
NUMPY_PROMOTION_PAIRS = tuple(
    (integer, floating)
    for integer in NUMPY_WIDTHS[:8]
    for floating in NUMPY_WIDTHS[8:]
)

TIMESTAMP_FAMILIES = (
    'naive_ns',
    'fixed',
    'fold',
    'mixed_datetime',
    'mixed_date',
    'same_zone_fold',
    'distinct_zone_same_fold',
    'distinct_zone_other_fold',
    'out_of_ns',
    'naive_aware',
    'wide_year',
    'wide_zoneinfo',
    'wide_datetime',
    'zero_year_datetime',
    'wide_date',
    'pytz_fold_timestamp',
    'pytz_mixed_datetime',
    'pytz_static',
    'pytz_utc',
    'named_default',
)


def timestamp_pattern(name: str) -> list[Any]:
    zone = ZoneInfo('America/New_York')
    other = ZoneInfo.no_cache('America/New_York')
    early = datetime(2020, 11, 1, 1, 30, tzinfo=zone, fold=0)
    late = early.replace(fold=1)
    if name.startswith('pytz_'):
        import pytz

        if name == 'pytz_static':
            zone_p = pytz.timezone('Etc/GMT-2')
            return [
                pd.Timestamp(datetime(2020, 1, 1, tzinfo=zone_p)),
                datetime(2020, 1, 2, tzinfo=zone_p),
            ]
        if name == 'pytz_utc':
            return [
                pd.Timestamp(datetime(2020, 1, 1, tzinfo=pytz.UTC)),
                datetime(2020, 1, 2, tzinfo=pytz.UTC),
            ]
        zone_p = pytz.timezone('America/New_York')
        first = zone_p.localize(datetime(2020, 11, 1, 1, 30), is_dst=True)
        second = zone_p.localize(datetime(2020, 11, 1, 1, 30), is_dst=False)
        return [
            pd.Timestamp(first),
            pd.Timestamp(second) if name == 'pytz_fold_timestamp' else second,
        ]

    def wide() -> pd.Timestamp:
        return pd.Timestamp(np.datetime64('12000-01-01', 's'))

    patterns = {
        'naive_ns': lambda: [
            pd.Timestamp('2020-01-01 00:00:00.000000001'),
            pd.Timestamp('2020-01-01 00:00:00.000000002'),
        ],
        'fixed': lambda: [
            pd.Timestamp(
                datetime(2020, 1, 1, tzinfo=timezone(timedelta(hours=2)))
            ),
            pd.Timestamp(datetime(2020, 1, 1, tzinfo=timezone.utc)),
        ],
        'fold': lambda: [pd.Timestamp(early), pd.Timestamp(late)],
        'named_default': lambda: [
            pd.Timestamp('2020-01-01', tz='America/New_York'),
            pd.Timestamp('2020-01-02', tz='America/New_York'),
        ],
        'mixed_datetime': lambda: [
            pd.Timestamp('2020-01-01 00:00:00.000000001'),
            datetime(2020, 1, 1),
        ],
        'mixed_date': lambda: [pd.Timestamp('2020-01-01'), date(2020, 1, 1)],
        'same_zone_fold': lambda: [pd.Timestamp(early), late],
        'distinct_zone_same_fold': lambda: [
            pd.Timestamp(early),
            early.replace(tzinfo=other),
        ],
        'distinct_zone_other_fold': lambda: [
            pd.Timestamp(early),
            late.replace(tzinfo=other),
        ],
        'out_of_ns': lambda: [
            pd.Timestamp(datetime(2500, 1, 1)),
            pd.Timestamp(datetime(2500, 1, 2)),
        ],
        'naive_aware': lambda: [
            pd.Timestamp('2020-01-01'),
            pd.Timestamp('2020-01-01', tz='UTC'),
        ],
        'wide_year': lambda: [
            wide(),
            pd.Timestamp(np.datetime64('12001-01-01', 's')),
        ],
        'wide_zoneinfo': lambda: [
            wide().tz_localize('UTC').tz_convert(ZoneInfo('UTC')),
            pd.Timestamp(np.datetime64('12001-01-01', 's'))
            .tz_localize('UTC')
            .tz_convert(ZoneInfo('UTC')),
        ],
        'wide_datetime': lambda: [wide(), datetime(2020, 1, 1)],
        'zero_year_datetime': lambda: [
            pd.Timestamp(np.datetime64('0000-01-01', 's')),
            datetime(2020, 1, 1),
        ],
        'wide_date': lambda: [wide(), date(2020, 1, 1)],
    }
    return patterns[name]()


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
    + tuple(
        'timestamp_' + name + '_' + direction
        for name in TIMESTAMP_FAMILIES
        for direction in ('forward', 'reverse')
    )
    + tuple('numpy_' + name for name in NUMPY_WIDTHS)
    + ('numpy_longlong', 'numpy_ulonglong')
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
    for name in (*NUMPY_WIDTHS, 'longlong', 'ulonglong'):
        scalar = getattr(np, name)
        patterns['numpy_' + name] = [scalar(2), scalar(0), scalar(1), scalar(2)]
    mixtures = {
        'signed_unsigned_63': [np.int64(2**63 - 1), np.uint64(2**63)],
        'longlong_unsigned_63': [np.longlong(2**63 - 1), np.ulonglong(2**63)],
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
        'missing_pandas_na': [np.float32(2), pd.NA, np.int16(-1)],
        'missing_pandas_nat': [np.float32(2), pd.NaT, np.int16(-1)],
        'missing_nan_huge_integer': [
            np.float64(float('nan')),
            10**400,
            np.float64(1),
            None,
        ],
        'only_missing_numpy_owner': [np.float64(float('nan')), 10**400],
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
    if family.startswith('timestamp_'):
        name, direction = family.removeprefix('timestamp_').rsplit('_', 1)
        pattern = timestamp_pattern(name)
        if direction == 'reverse':
            pattern = pattern[::-1]
    else:
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
    'multi_temporal_range_forward',
    'multi_temporal_range_reverse',
    'multi_temporal_fold',
    'multi_temporal_fold_alternating_forward',
    'multi_temporal_fold_alternating_reverse',
    'multi_temporal_unused_ordered',
    'multi_temporal_unused_normalized',
    'string',
    'object_mixed',
    'nullable_integer',
    'arrow_integer',
    'arrow_string',
) + tuple(INDEX_SAMPLE_COUNTS)


def indexed_frame(family: str) -> pd.DataFrame:
    n = (
        2 * INDEX_SAMPLE_COUNTS[family] + 1
        if family in INDEX_SAMPLE_COUNTS
        else 81
    )
    data = pd.DataFrame(
        {
            'id': range(n),
            'feature': [i * 0.5 for i in range(n)],
            'target': [100.0 + i for i in range(n)],
        }
    )
    labels = [(i * 13) % 17 for i in range(n)]
    if family in INDEX_SAMPLE_COUNTS:
        sampled = np.random.RandomState(42).choice(
            n - 1, size=INDEX_SAMPLE_COUNTS[family], replace=False
        )
        values = np.arange(n, dtype='float64') + 100000
        for order, position in enumerate(sampled):
            values[position] = (
                order // 3
                if family == 'sampled_sorted_duplicates_missing'
                else 1.0
            )
        if family == 'sampled_present_nan':
            values[sampled[::5]] = float('nan')
            index = pd.Index(
                pd.arrays.FloatingArray(values, np.zeros(n, dtype=bool))
            )
        else:
            values[sampled[0]] = float('nan')
            index = pd.Index(values)
    elif family == 'range':
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
    elif family.startswith('multi_temporal_fold'):
        owners = timestamp_pattern('distinct_zone_same_fold')
        if family.endswith('_reverse'):
            owners = owners[::-1]
        values = np.empty(2, dtype=object)
        values[:] = owners
        level = pd.Index(values, dtype=object, copy=False)
        first_codes = [i % 2 for i in range(n)]
        sampled = np.random.RandomState(42).choice(
            n - 1, size=40, replace=False
        )
        for at, position in enumerate(sampled):
            first_codes[position] = (
                at % 2 if '_alternating_' in family else int(at >= 20)
            )
        index = pd.MultiIndex(
            levels=[level, pd.Index([0])],
            codes=[first_codes, [0] * n],
            verify_integrity=False,
        )
    elif family.startswith('multi_temporal_unused_'):
        values = np.empty(2, dtype=object)
        values[:] = timestamp_pattern('wide_datetime')
        level = pd.Index(values, dtype=object, copy=False)
        # The unused datetime must not be sorted when the used codes already
        # permit the native early return. Only secondary codes vary here.
        second_codes = (
            [i % 2 for i in range(n)]
            if family.endswith('_normalized')
            else [0] * n
        )
        index = pd.MultiIndex(
            levels=[level, pd.Index([0, 1])],
            codes=[[0] * n, second_codes],
            verify_integrity=False,
        )
    elif family.startswith('multi_temporal_range_'):
        owners = timestamp_pattern('wide_datetime')
        if family.endswith('_reverse'):
            owners = owners[::-1]
        values = np.empty(2, dtype=object)
        values[:] = owners
        level = pd.Index(values, dtype=object, copy=False)
        index = pd.MultiIndex(
            levels=[level, pd.Index([0, 1])],
            codes=[[i % 2 for i in range(n)], [(i // 2) % 2 for i in range(n)]],
            verify_integrity=False,
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
