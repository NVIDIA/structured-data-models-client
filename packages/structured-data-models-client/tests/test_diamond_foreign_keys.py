# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The original diamond's reached rows retain their foreign keys on either path."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema

FIXTURE = Path(__file__).with_name('data') / 'diamond.json'


@pytest.mark.parametrize(
    'names',
    [
        ('customers', 'orders', 'addresses'),
        ('orders', 'addresses', 'customers'),
    ],
)
def test_original_diamond_keeps_every_reached_customer_link(names):
    case = json.loads(FIXTURE.read_text())
    connection = duckdb.connect()
    try:
        for statement in case['statements']:
            connection.execute(statement)
        schema = Schema.from_source(
            DuckDBSource(connection), tables={name: name for name in names}
        )
        arguments = dict(case['materialize'])
        arguments['anchor_time'] = pd.Timestamp(arguments['anchor_time'])
        request = PredictiveQuery(case['pql'], schema).materialize(**arguments)
        assert len(request.payloads) == 1
        payload = request.payloads[0]
        reached = 0
        for half, count in [('context', 4), ('predict', 39)]:
            orders = payload[half]['related_tables']['orders']
            customers = payload[half]['related_tables']['customers']
            oi = {name: at for at, name in enumerate(orders['columns'])}
            ci = {name: at for at, name in enumerate(customers['columns'])}
            assert len(orders['rows']) == count
            for row in orders['rows']:
                matching = [
                    customer
                    for customer in customers['rows']
                    if customer[ci['instance_id']] == row[oi['instance_id']]
                    and int(customer[ci['tag']]) == int(row[oi['tag']]) % 10
                ]
                assert len(matching) == 1
                assert row[oi['cid']] == matching[0][ci['__node_id']]
                reached += 1
        assert reached == 43
    finally:
        connection.close()
