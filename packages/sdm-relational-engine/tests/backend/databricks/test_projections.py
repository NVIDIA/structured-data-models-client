# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""``_projections`` decides SELECT-list column order, and the model is
order-sensitive: driving it from the ``columns`` set left order at the mercy of
``PYTHONHASHSEED``, so one graph predicted anywhere between 117 and 237 for a
fixed ``random_seed``.
"""

from __future__ import annotations

from typing import Any

import pytest

try:
    from sdm_relational_engine.rfm.backend.databricks.sampler import (
        DatabricksSampler,
    )
except ImportError:
    pytest.skip("'databricks' extension not installed", allow_module_level=True)

_PROJ = {
    'users': {
        'user_id': '"user_id"',
        'ts': '"ts"',
        'age': '"age"',
    },
}


def _sampler() -> Any:
    r"""Only ``_table_column_proj_dict`` is read, and a real sampler needs a
    live warehouse connection to build.
    """
    sampler = object.__new__(DatabricksSampler)
    sampler._table_column_proj_dict = _PROJ
    return sampler


@pytest.mark.parametrize(
    'columns',
    [
        {'user_id', 'ts', 'age'},
        {'age', 'user_id', 'ts'},
        ['ts', 'age', 'user_id'],
    ],
)
def test_projection_order_follows_the_table_not_the_request(
    columns: Any,
) -> None:
    assert _sampler()._projections('users', columns) == [
        '"user_id"',
        '"ts"',
        '"age"',
    ]


def test_a_subset_keeps_the_table_order() -> None:
    assert _sampler()._projections('users', {'age', 'user_id'}) == [
        '"user_id"',
        '"age"',
    ]


def test_an_unknown_column_raises_instead_of_shortening_the_select() -> None:
    """Indexing per column used to raise; driving from the projection dict
    would silently return a short SELECT list instead.
    """
    with pytest.raises(KeyError, match='nope'):
        _sampler()._projections('users', {'user_id', 'nope'})
