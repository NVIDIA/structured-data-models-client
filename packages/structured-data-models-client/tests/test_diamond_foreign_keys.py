# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The original diamond's reached rows retain their foreign keys on either path."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import yaml
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    'names',
    [
        ('customers', 'orders', 'addresses'),
        ('orders', 'addresses', 'customers'),
    ],
)
def test_original_diamond_keeps_every_reached_customer_link(names, monkeypatch):
    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)
    case = json.loads(
        (ROOT / 'contract/cases/s1/shape.diamond-from-source.json').read_text()
    )
    bases = yaml.safe_load((ROOT / 'contract/bases.yaml').read_text())['bases']
    connection = duckdb.connect()
    try:
        for statement in [
            *bases[case['input']['base']],
            *case['input']['setup'],
        ]:
            connection.execute(statement)
        schema = Schema.from_source(
            DuckDBSource(connection), tables={name: name for name in names}
        )
        arguments = dict(case['input']['materialize'])
        arguments['anchor_time'] = pd.Timestamp(arguments['anchor_time'])
        request = PredictiveQuery(case['input']['pql'], schema).materialize(
            **arguments
        )
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
