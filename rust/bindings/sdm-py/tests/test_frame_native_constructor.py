# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fixed native constructor returns useful typed meaning without callbacks."""

import json

import pytest

from sdm_client import _native


def test_native_constructor_binds_name_source_type_and_completed_meaning():
    column = _native.FrameNativeConstructor(
        'history', 'part', b'{"storage":"text"}', b'null'
    )
    assert json.loads(column.dtype()) == 'string'
    assert column.timestamp_sample()
    with pytest.raises(_native.CoreError):
        column.meaning()
    column.timestamp_counts(0, 2)
    assert column.separated_sample()
    assert column.separator([ord(char) for char in 'x|y;x|y;']) == '|'
    column.separated_counts(4, 2)
    assert json.loads(column.meaning()) == {
        'name': 'part', 'dtype': 'string', 'stype': 'multicategorical'
    }
    with pytest.raises(_native.CoreError):
        column.distinct_count(1)


def test_fixed_native_declaration_cannot_be_replaced_by_exported_storage():
    with pytest.raises(_native.CoreError):
        _native.FrameNativeConstructor(
            'history', 'part', b'{"storage":"text"}',
            b'{"declaration":"physical","physical":{"type":"int64"}}',
        )
