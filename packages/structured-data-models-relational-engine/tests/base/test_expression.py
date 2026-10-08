# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from sdm_relational_engine.rfm.base import Expression, LocalExpression


def test_coerce() -> None:
    assert Expression.coerce(None) is None
    assert Expression.coerce(LocalExpression('A')) == LocalExpression('A')
    assert Expression.coerce('A') == LocalExpression('A')
    assert Expression.coerce(dict(value='A')) == LocalExpression('A')

    with pytest.raises(TypeError):
        Expression.coerce(dict(value='A', on='B'))
    with pytest.raises(TypeError):
        Expression.coerce(['A', 'B'])  # type: ignore
