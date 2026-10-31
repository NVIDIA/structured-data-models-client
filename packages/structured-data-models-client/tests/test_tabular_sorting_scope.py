# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public controls for the owner-approved ordinary sorting boundary."""

import numpy as np
import pandas as pd
import pytest
from test_tabular_pql_core import ROUTES, query

from sdm_client._routing import RouteError
from sdm_client._rust import tabular_index


class CustomSortFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return CustomSortFrame

    def sort_index(self, *args, **kwargs):
        raise AssertionError('custom sorting must not execute')


class CustomLabel:
    def __lt__(self, other):
        raise AssertionError('custom comparison must not execute')

    def __gt__(self, other):
        raise AssertionError('custom comparison must not execute')


def data(n):
    return pd.DataFrame(
        {
            'id': range(n),
            'feature': np.arange(n) * 0.5,
            'target': np.arange(n) + 100.0,
        }
    )


def materialize(frame, entry, wanted):
    arguments = (
        {'indices': [len(frame) - 1], 'num_context_samples': wanted}
        if entry == 'indices'
        else {'predict': frame.iloc[[-1]].drop(columns=['target'])}
    )
    return query(frame).materialize(**arguments, task='regression')


@pytest.mark.parametrize('entry', ['indices', 'predict'])
@pytest.mark.parametrize('owner', ['frame_sort', 'sampled_frame_sort', 'label'])
def test_reduced_custom_sorting_is_refused_without_execution(
    monkeypatch, entry, owner
):
    monkeypatch.setenv('SDM_CORE_ROUTES', ROUTES)
    n, wanted = (81, 40) if entry == 'indices' else (10002, 10000)
    frame = data(n)
    if owner == 'frame_sort':
        frame = CustomSortFrame(frame)
    elif owner == 'sampled_frame_sort':

        class ChangedAfterTake(pd.DataFrame):
            @property
            def _constructor(self):
                return ChangedAfterTake

            def take(self, indices, axis=0, **kwargs):
                result = super().take(indices, axis=axis, **kwargs)
                return (
                    CustomSortFrame(result)
                    if axis == 0 and len(indices) == wanted
                    else result
                )

        frame = ChangedAfterTake(frame)
    else:
        frame.index = pd.Index([CustomLabel() for _ in range(n)], dtype=object)
    message = (
        'object index comparison family'
        if owner == 'label'
        else 'custom Python sorting methods'
    )
    with pytest.raises(RouteError, match=message):
        materialize(frame, entry, wanted)


@pytest.mark.parametrize('entry', ['indices', 'predict'])
def test_unreduced_context_never_inspects_sorting(monkeypatch, entry):
    monkeypatch.setenv('SDM_CORE_ROUTES', ROUTES)

    def forbidden(*args):
        raise AssertionError('unused sorting must not be inspected')

    monkeypatch.setattr(tabular_index, 'has_custom_sorter', forbidden)
    request = materialize(CustomSortFrame(data(81)), entry, 80)
    assert request.context.instance_table['id'].tolist() == list(range(80))
