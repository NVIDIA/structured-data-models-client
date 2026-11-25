# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A core route over frames runs where the warehouse package
(``sdm_connectors``, an optional dependency of the client) is not installed
: the run loop imports nothing of it.

Both models are held to it. The relational route gathers through
``_rust.frame_materialize``, which the tabular route does not reach, so each
needs its own run to say the package stayed out of it."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

ABSENT = """
import importlib.abc
import sys

class Absent(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'sdm_connectors':
            raise ModuleNotFoundError(
                "No module named 'sdm_connectors'", name=fullname
            )

for name in [m for m in sys.modules if m.split('.')[0] == 'sdm_connectors']:
    del sys.modules[name]
sys.meta_path.insert(0, Absent())

import pandas as pd
"""

TABULAR = """
from sdm_client import KumoTabular, PredictiveQuery, Schema

schema = Schema.from_data(
    {'users': pd.DataFrame({'id': [1, 2, 3, 4], 'age': [10.5, 20.5, 30.5, 40.5]})}
)
request = PredictiveQuery(
    'PREDICT users.age FOR EACH users.id', schema
).materialize(model=KumoTabular(), indices=[1])
"""

# The relational gather reads the client's data providers to find the one the
# schema's source names. A schema built from frames names no source, so it
# must reach none of them, and none of the warehouse package they import.
RELATIONAL = """
from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Relationship,
    Schema,
)

users = pd.DataFrame({'user_id': range(12)})
orders = pd.DataFrame(
    {
        'order_id': range(120),
        'user_id': [i % 12 for i in range(120)],
        'ordered_at': pd.date_range('2026-01-01', periods=120, freq='D'),
    }
)
schema = Schema.from_data(
    {'users': users, 'orders': orders},
    primary_keys={'users': 'user_id', 'orders': 'order_id'},
    time_columns={'orders': 'ordered_at'},
    relationships=[Relationship('orders', 'user_id', 'users', 'user_id')],
)
assert schema.source is None
request = PredictiveQuery(
    'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id', schema
).materialize(model=KumoRelational(), indices=[1, 2, 3])
"""

REPORT = """
assert 'sdm_connectors' not in sys.modules
print(type(request).__name__)
"""


@pytest.mark.parametrize(
    'route', [TABULAR, RELATIONAL], ids=['tabular', 'relational']
)
def test_an_in_memory_core_query_runs_without_the_connectors_package(
    route: str,
) -> None:
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(ABSENT + route + REPORT)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'Request'
