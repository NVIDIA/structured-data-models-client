# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures shared across the client's test modules."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from sdm_client import Request


@pytest.fixture
def tabular_request() -> Callable[..., Request]:
    """Build a single-table request: 20 labelled context rows, ``rows`` to score.

    A classification target alternates 0/1; a regression target is a float.
    """

    def build(rows: int = 1, task: str = 'classification') -> Request:
        labels = [
            i % 2 if task == 'classification' else float(i) for i in range(20)
        ]
        context = pd.DataFrame(
            {
                'row_id': range(20),
                'age': [20.0 + i for i in range(20)],
                'churn': labels,
            }
        )
        predict = pd.DataFrame(
            {
                'row_id': range(90, 90 + rows),
                'age': [33.0 + i for i in range(rows)],
            }
        )
        return Request(
            context=context, predict=predict, target='churn', task=task
        )

    return build
