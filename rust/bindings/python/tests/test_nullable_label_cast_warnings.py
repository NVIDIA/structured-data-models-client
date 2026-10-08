# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import pandas as pd
import pytest
from test_label_cast_warnings import aggregation, execute


@pytest.mark.parametrize('operation', ['MIN', 'COUNT'])
@pytest.mark.parametrize('mode', ['warn', 'ignore', 'raise', 'warning-error'])
def test_nullable_float_cast_uses_masked_array_policy(operation, mode):
    given = aggregation(operation, [1e100, {'$na': None}, 1.0])
    given['tables']['fact']['columns']['value']['dtype'] = 'Float64'
    error = None
    with np.errstate(over='warn' if mode == 'warning-error' else mode):
        with warnings.catch_warnings(record=True) as baseline:
            warnings.simplefilter(
                'error' if mode == 'warning-error' else 'always', RuntimeWarning
            )
            try:
                pd.Series([1e100, None, 1.0], dtype='Float64').astype('float32')
            except Exception as raised:
                error = {'code': type(raised).__name__, 'message': str(raised)}
        with warnings.catch_warnings(record=True) as actual:
            warnings.simplefilter(
                'error' if mode == 'warning-error' else 'always', RuntimeWarning
            )
            result = execute(given)
    assert [(item.category, str(item.message)) for item in actual] == [
        (item.category, str(item.message)) for item in baseline
    ]
    if error is not None:
        assert result == {'error': error}
    else:
        assert result['output']['out']['dtype'] == 'float32'
        assert result['output']['out']['values'] == (
            [1.0] if operation == 'MIN' else [2.0]
        )
        assert result['output']['mask'] == [True]
