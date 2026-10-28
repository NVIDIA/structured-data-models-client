# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The native pointwise conversion returns its actual authoritative values."""

from __future__ import annotations

import numpy as np
import pytest

from sdm_client._rust.frame_native import NativeFrameColumns


@pytest.mark.parametrize('empty', [False, True])
def test_pointwise_result_retains_actual_native_return_and_original_owner(
    empty,
):
    events = []
    result = np.empty(0 if empty else 2, dtype=object)
    if not empty:
        result[0], result[1] = [9], [3, 2]

    class Native:
        def _cast_pointwise_result(self, groups):
            events.append((self, groups))
            return result

    owner = Native()
    columns = NativeFrameColumns(
        {'items': {'id': object}}, {'items': {'id': owner}}
    )
    values = [] if empty else [[1, 2], []]
    before = [list(row) for row in values]
    actual = columns.label_result(
        {'source': {'source': ['items']}, 'column': 'id'}, values
    )
    assert actual is result
    assert len(events) == 1 and events[0][0] is owner
    assert events[0][1].dtype == object
    assert events[0][1].tolist() == before
    assert values == before


def test_failed_native_result_is_not_replaced_by_the_mathematical_groups():
    failure = ValueError('actual native completion failed')

    class Native:
        def _cast_pointwise_result(self, groups):
            raise failure

    columns = NativeFrameColumns(
        {'items': {'id': object}}, {'items': {'id': Native()}}
    )
    with pytest.raises(ValueError) as raised:
        columns.label_result(
            {'source': {'source': ['items']}, 'column': 'id'}, [[1]]
        )
    assert raised.value is failure
