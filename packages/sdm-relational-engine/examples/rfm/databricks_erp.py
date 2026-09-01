# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time

from sdm_client import RelationalClient
from sdm_relational_engine import rfm
from sdm_relational_engine.testing.databricks import connect

nim_url = os.environ['KUMO_RELATIONAL_API_ENDPOINT']

connection = connect(
    catalog='kumo_test_catalogue',
    schema='nvidia_erp',
)

t = time.perf_counter()
graph = rfm.Graph.from_databricks(
    connection,
    tables=['customers', 'order_lines', 'products'],
)
print(f'Created graph in {time.perf_counter() - t:.2f} seconds')

with connection.cursor() as cursor:
    cursor.execute('SELECT customer_id FROM customers LIMIT 10')
    indices = [row[0] for row in cursor.fetchall()]

query = 'PREDICT COUNT(order_lines.*, 0, 30)=0 FOR EACH customers.customer_id'
# `order_lines` is large (~1.4M rows), so cap the per-hop neighbor fan-out to
# keep the in-context tables under the request-size limit:
with RelationalClient(url=nim_url) as client:
    pred = client.relational(graph).predict(
        query,
        indices=indices,
        run_mode='fast',
        num_neighbors=[4],
    )
print(pred)
