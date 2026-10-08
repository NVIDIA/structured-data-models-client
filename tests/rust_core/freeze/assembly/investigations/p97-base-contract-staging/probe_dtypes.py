# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public constructor/describer inventory only; no product inference or services."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import warnings
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from sdm_core import describe


def arrow_cases():
    primitives = {
        'null': (pa.null(), [None, None, None]),
        'bool': (pa.bool_(), [True, None, False]),
        'string': (pa.string(), ['one', None, 'two']),
        'large_string': (pa.large_string(), ['one', None, 'two']),
        'string_view': (pa.string_view(), ['one', None, 'two']),
        'binary': (pa.binary(), [b'one', None, b'two']),
        'large_binary': (pa.large_binary(), [b'one', None, b'two']),
        'binary_view': (pa.binary_view(), [b'one', None, b'two']),
        'fixed_binary_3': (pa.binary(3), [b'one', None, b'two']),
        'date32': (
            pa.date32(),
            [dt.date(2024, 1, 1), None, dt.date(2024, 1, 3)],
        ),
        'date64': (
            pa.date64(),
            [dt.date(2024, 1, 1), None, dt.date(2024, 1, 3)],
        ),
        'month_day_nano_interval': (
            pa.month_day_nano_interval(),
            [(1, 2, 3), None, (2, 3, 4)],
        ),
    }
    for width in (8, 16, 32, 64):
        for prefix in ('int', 'uint'):
            name = f'{prefix}{width}'
            primitives[name] = (getattr(pa, name)(), [1, None, 3])
    for width in (16, 32, 64):
        primitives[f'float{width}'] = (
            getattr(pa, f'float{width}')(),
            [1.5, None, 3.5],
        )
    for width, precision in ((32, 8), (64, 16), (128, 38), (256, 76)):
        for scale in (0, 2, -2):
            primitives[f'decimal{width}_{precision}_{scale}'] = (
                getattr(pa, f'decimal{width}')(precision, scale),
                [Decimal('100'), None, Decimal('300')],
            )
    for unit in ('s', 'ms', 'us', 'ns'):
        primitives[f'duration_{unit}'] = (pa.duration(unit), [1, None, 3])
        for zone in (None, 'UTC', 'America/New_York', '+05:30'):
            label = 'naive' if zone is None else zone.replace('/', '_')
            primitives[f'timestamp_{unit}_{label}'] = (
                pa.timestamp(unit, zone),
                [0, None, 1000],
            )
    for width, units in ((32, ('s', 'ms')), (64, ('us', 'ns'))):
        for unit in units:
            primitives[f'time{width}_{unit}'] = (
                getattr(pa, f'time{width}')(unit),
                [1, None, 3],
            )
    for name, kind in (
        ('list_int64', pa.list_(pa.int64())),
        ('large_list_int64', pa.large_list(pa.int64())),
        ('fixed_list_int64_2', pa.list_(pa.int64(), 2)),
        ('list_view_int64', pa.list_view(pa.int64())),
        ('large_list_view_int64', pa.large_list_view(pa.int64())),
    ):
        primitives[name] = (kind, [[1, 2], None, [3, 4]])
    primitives['list_list_int64'] = (
        pa.list_(pa.list_(pa.int64())),
        [[[1, 2]], None, [[3]]],
    )
    primitives['struct_int_string'] = (
        pa.struct([('number', pa.int64()), ('label', pa.string())]),
        [{'number': 1, 'label': 'one'}, None, {'number': 3, 'label': 'three'}],
    )
    for sorted_keys in (False, True):
        primitives[f'map_string_int64_sorted_{sorted_keys}'] = (
            pa.map_(pa.string(), pa.int64(), keys_sorted=sorted_keys),
            [[('one', 1)], None, [('two', 2)]],
        )
    for index in (8, 16, 32, 64):
        for ordered in (False, True):
            primitives[f'dictionary_int{index}_string_ordered_{ordered}'] = (
                pa.dictionary(
                    getattr(pa, f'int{index}')(), pa.string(), ordered=ordered
                ),
                ['one', None, 'two'],
            )
    for name, (kind, values) in primitives.items():
        yield name, lambda kind=kind, values=values: pa.array(values, type=kind)
    yield (
        'dense_union_int_string',
        lambda: pa.UnionArray.from_dense(
            pa.array([0, 1, 0], type=pa.int8()),
            pa.array([0, 0, 1], type=pa.int32()),
            [pa.array([1, 3]), pa.array(['two'])],
        ),
    )
    yield (
        'sparse_union_int_string',
        lambda: pa.UnionArray.from_sparse(
            pa.array([0, 1, 0], type=pa.int8()),
            [pa.array([1, None, 3]), pa.array([None, 'two', None])],
        ),
    )
    yield (
        'run_end_encoded_int16_string',
        lambda: pa.RunEndEncodedArray.from_arrays(
            pa.array([2, 3], type=pa.int16()), pa.array(['one', 'two'])
        ),
    )
    for name, kind, storage in (
        ('json', pa.json_(), pa.array(['{"value":1}', None, '{"value":3}'])),
        (
            'uuid',
            pa.uuid(),
            pa.array([b'0' * 16, None, b'1' * 16], type=pa.binary(16)),
        ),
        ('bool8', pa.bool8(), pa.array([1, None, 0], type=pa.int8())),
    ):
        yield (
            name,
            lambda kind=kind, storage=storage: pa.ExtensionArray.from_storage(
                kind, storage
            ),
        )


def cases():
    for name in (
        'bool',
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
        'longdouble',
        'complex64',
        'complex128',
        'clongdouble',
        'object',
        'S6',
        'U6',
    ):
        yield (
            f'numpy/{name}',
            lambda name=name: pd.Series(np.array([1, 2, 3], dtype=name)),
        )
    for name in (
        'Int8',
        'Int16',
        'Int32',
        'Int64',
        'UInt8',
        'UInt16',
        'UInt32',
        'UInt64',
        'Float32',
        'Float64',
    ):
        yield (
            f'pandas/{name}',
            lambda name=name: pd.Series([1, None, 3], dtype=name),
        )
    yield (
        'pandas/boolean',
        lambda: pd.Series([True, None, False], dtype='boolean'),
    )
    for storage in ('python', 'pyarrow'):
        yield (
            f'pandas/string_{storage}',
            lambda storage=storage: pd.Series(
                ['one', None, 'two'], dtype=pd.StringDtype(storage=storage)
            ),
        )
    for ordered in (False, True):
        yield (
            f'pandas/category_ordered_{ordered}',
            lambda ordered=ordered: pd.Series(
                pd.Categorical(['one', None, 'two'], ordered=ordered)
            ),
        )
    for unit in ('s', 'ms', 'us', 'ns'):
        for prefix in ('datetime64', 'timedelta64'):
            yield (
                f'numpy/{prefix}_{unit}',
                lambda prefix=prefix, unit=unit: pd.Series(
                    np.array([0, 1, 2], dtype=f'{prefix}[{unit}]')
                ),
            )
        for zone in ('UTC', 'America/New_York', '+05:30'):
            yield (
                f'pandas/datetime64_{unit}_{zone.replace("/", "_")}',
                lambda unit=unit, zone=zone: pd.Series(
                    pd.date_range('2024-01-01', periods=3, tz=zone)
                ).astype(pd.DatetimeTZDtype(unit=unit, tz=zone)),
            )
    yield (
        'pandas/period_month',
        lambda: pd.Series(pd.period_range('2024-01', periods=3, freq='M')),
    )
    yield (
        'pandas/interval_int64',
        lambda: pd.Series(pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3])),
    )
    yield (
        'pandas/sparse_int64',
        lambda: pd.Series(pd.arrays.SparseArray([0, 1, 2])),
    )
    for name, build in arrow_cases():
        yield (
            f'arrow/{name}',
            lambda build=build: pd.Series(
                pd.arrays.ArrowExtensionArray(build())
            ),
        )


def main():
    records = []
    for identity, build in cases():
        record = {'id': identity}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                series = build()
                record['pandas_dtype'] = str(series.dtype)
                table, tags = describe.describe_frames(
                    {'public_table': pd.DataFrame({'value': series})}
                )
                record.update(
                    status='described',
                    arrow_type=str(
                        table['public_table'].schema.field('value').type
                    ),
                    tag=json.loads(tags)['public_table']['value'],
                )
            except Exception as error:
                record.update(
                    status='constructor_or_describer_refused',
                    error_class=f'{type(error).__module__}.{type(error).__qualname__}',
                )
            record['warning_classes'] = [
                f'{item.category.__module__}.{item.category.__qualname__}'
                for item in caught
            ]
        records.append(record)
    print(
        json.dumps(
            {
                'purpose': 'Constructor/describer capability only; not public inference acceptance',
                'python': platform.python_version(),
                'pandas': pd.__version__,
                'numpy': np.__version__,
                'pyarrow': pa.__version__,
                'describer_sha256': hashlib.sha256(
                    Path(describe.__file__).read_bytes()
                ).hexdigest(),
                'records': records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    main()
