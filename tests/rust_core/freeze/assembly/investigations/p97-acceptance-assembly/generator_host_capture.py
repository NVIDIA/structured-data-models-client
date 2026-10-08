# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import inspect
import json
import sys
import time
from pathlib import Path


def collect_host_inputs(root, name, value):
    import pandas as pd
    from generator_inputs import typed
    from generator_oracles import restore_result, source_codes
    from sdm_client import _impl
    from sdm_client.result import PredictionResult
    from sdm_core import _native
    from sdm_core.answers import ResultValues
    from sdm_core.describe import timestamp_series_units

    if not _impl.OLD or name not in {'request.results', 'request.merge'}:
        raise ValueError('unsupported original host witness route')
    if sys.getprofile() is not None:
        raise ValueError('host witness profiler already active')
    expected_source = (
        root / 'packages/structured-data-models-client/src/sdm_client/result.py'
    )
    if inspect.getsourcefile(PredictionResult) != str(expected_source):
        raise ValueError('host witness original source mismatch')
    for function in (
        PredictionResult.__init__,
        PredictionResult.to_pandas,
        PredictionResult._empty_frame,
        PredictionResult.merge,
    ):
        if function.__code__ not in source_codes(
            expected_source.read_bytes(), function.__code__.co_filename
        ):
            raise ValueError('host witness original code mismatch')
    primitive_source = root / 'rust/bindings/python/python/sdm_core/answers.py'
    for function in (ResultValues.infer_column, ResultValues.column_order):
        loaded_source = inspect.getsourcefile(function)
        if (
            Path(loaded_source).read_bytes() != primitive_source.read_bytes()
            or function.__code__
            not in source_codes(
                primitive_source.read_bytes(), function.__code__.co_filename
            )
        ):
            raise ValueError('host witness primitive code mismatch')
    descriptor_source = (
        root / 'rust/bindings/python/python/sdm_core/describe.py'
    )
    if (
        Path(inspect.getsourcefile(timestamp_series_units)).read_bytes()
        != descriptor_source.read_bytes()
        or timestamp_series_units.__code__
        not in source_codes(
            descriptor_source.read_bytes(),
            timestamp_series_units.__code__.co_filename,
        )
    ):
        raise ValueError('host witness descriptor code mismatch')
    native_functions = tuple(
        function
        for function in vars(_native).values()
        if inspect.isbuiltin(function)
    )
    constructors = []
    native_calls = 0
    original_calls = 0
    caller_codes = {
        PredictionResult.to_pandas.__code__,
        PredictionResult._empty_frame.__code__,
    }

    def observed(frame, event, function):
        nonlocal native_calls, original_calls
        if event == 'c_call' and any(
            function is candidate for candidate in native_functions
        ):
            native_calls += 1
        elif (
            event == 'call'
            and frame.f_code is PredictionResult.to_pandas.__code__
        ):
            original_calls += 1
        elif (
            event == 'call'
            and frame.f_code is pd.DataFrame.__init__.__code__
            and frame.f_back.f_code in caller_codes
        ):
            constructors.append(frame.f_locals['data'])

    sys.setprofile(observed)
    try:
        result = (
            restore_result(value)
            if name == 'request.results'
            else PredictionResult.merge(
                [restore_result(part) for part in value['parts']]
            )
        )
        result.to_pandas()
    finally:
        sys.setprofile(None)
    if native_calls != 0 or original_calls != 1 or len(constructors) != 1:
        raise ValueError('host witness original call count mismatch')
    supplied = constructors[0]
    if isinstance(supplied, list) and all(
        type(row) is dict for row in supplied
    ):
        names = list(dict.fromkeys(key for row in supplied for key in row))
        columns = [
            (name, [row.get(name, float('nan')) for row in supplied])
            for name in names
        ]
    elif type(supplied) is dict and all(
        isinstance(column, pd.Series) and len(column) == 0
        for column in supplied.values()
    ):
        columns = [(name, []) for name in supplied]
    else:
        raise ValueError('unsupported original frame constructor')
    references = ResultValues()
    entries = []
    held = {}
    for column_name, cells in columns:
        held_cells = [
            references.hold(cell, none_is_value=True) for cell in cells
        ]
        reference, length = references.infer_column(json.dumps(held_cells))
        column = references.values[int(reference)]
        if (
            str(column.dtype)
            not in {
                'object',
                'int64',
                'float64',
                'bool',
                'string',
                'str',
                'datetime64[s]',
                'datetime64[ms]',
                'datetime64[us]',
                'datetime64[ns]',
            }
            or type(column.index).__name__ != 'RangeIndex'
        ):
            raise ValueError('unsupported inferred representation')
        entries.append(
            {
                'operation': 'infer_column',
                'arguments': typed(cells),
                'outcome': {
                    'status': 'success',
                    'dtype': str(column.dtype),
                    'index_class': 'RangeIndex',
                    'length': length,
                },
            }
        )
        held[column_name] = reference
    if 'row_index' in held:
        reference = held['row_index']
        column = references.values[int(reference)]
        arguments = {
            'column': typed(column),
            'stable': True,
            'ascending': True,
            'missing_last': True,
        }
        permutation = references.column_order(reference, True, True, True)
        if sorted(permutation) != list(range(len(column))):
            raise ValueError('invalid actual host permutation')
        entries.append(
            {
                'operation': 'column_order',
                'arguments': arguments,
                'outcome': {
                    'status': 'success',
                    'permutation': permutation,
                },
            }
        )
    names = [name for name, _ in columns]
    if not all(type(name) is str for name in names):
        raise ValueError('unsupported frame column names')
    row_count = len(supplied) if type(supplied) is list else 0
    index = (
        pd.Index(names, name=None)
        if names
        else pd.RangeIndex(0)
        if row_count == 0
        else pd.Index([], dtype=object, name=None)
    )
    entries.append(
        {
            'operation': 'index_constructor',
            'arguments': {
                'values': typed(names),
                'name': typed(None),
                'row_count': row_count,
            },
            'outcome': {
                'status': 'success',
                'dtype': str(index.dtype),
                'class': type(index).__name__,
                'length': len(index),
            },
        }
    )
    now = time.localtime()
    return {
        'host': {
            'year': now.tm_year,
            'today': [now.tm_year, now.tm_mon, now.tm_mday],
            'pads_year': datetime.datetime(1, 1, 1).strftime('%Y') == '0001',
            'tznames': list(time.tzname),
            'timestamp_series_units': timestamp_series_units()
            or ['s', 'ms', 'us', 'ns'],
        },
        'primitives': entries,
    }
