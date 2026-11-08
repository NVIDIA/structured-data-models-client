# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arrow row-label ordering and wide-date regression contracts."""

import pytest
from tabular_arrow_index_cases import CASES, digest, frame_for, observe_case


def expected_case(name):
    """Same-runtime pandas ordering, independent of client implementation."""
    family, entry, shape = name.split('/')
    # The predict route sends the frame's labelled rows whole: no sampling.
    n, wanted = (81, 40) if entry == 'indices' else (10002, 10001)
    frame = frame_for(family, n, shape)
    selected = (
        frame.iloc[:-1]
        if entry == 'predict'
        else frame.iloc[:-1].sample(n=wanted, random_state=42).sort_index()
    )
    ids = selected['id'].tolist()
    return {
        'context_count': wanted,
        'context_ids_sha256': digest(ids),
        'encoded_ids_sha256': digest(ids),
        'index_matches_source': True,
        'index_dtype': str(selected.index.dtype),
        'predictions': [{'row_index': 0, 'id': n - 1, 'prediction': 0.0}],
    }


@pytest.mark.parametrize('name', CASES)
def test_arrow_labels_preserve_pandas_order_and_prediction(name):
    assert observe_case(name) == expected_case(name)


@pytest.mark.parametrize('entry', ['indices', 'predict'])
@pytest.mark.parametrize('shape', ['forward', 'reverse'])
def test_wide_dates_avoid_python_date_conversion(entry, shape):
    # Adding a constant to every non-null physical day preserves its order.
    # Use ordinary dates as an independent ordering oracle, because Python
    # date objects cannot represent the wide physical date range.
    actual = observe_case(f'date32_wide/{entry}/{shape}')
    assert actual == expected_case(f'date32/{entry}/{shape}')
    assert 'error' not in actual
