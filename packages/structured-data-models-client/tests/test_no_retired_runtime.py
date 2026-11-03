# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An installed-client journey refuses retired imports from process startup."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from test_source_schema_refresh import SHOP

QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'


def test_public_warehouse_journey_cannot_import_retired_packages(
    tmp_path: Path,
) -> None:
    script = f"""
    import importlib.util, sys, warnings

    OLD = ('predictive_query', 'sdm_relational_engine')

    class Refused:
        def find_spec(self, name, path=None, target=None):
            if name in OLD or name.startswith(tuple(f'{{o}}.' for o in OLD)):
                raise ImportError(f'{{name}} is not to be imported')

    sys.meta_path.insert(0, Refused())

    import duckdb
    import pandas as pd
    from sdm_client import (
        Column, KumoRelational, PredictiveQuery, Relationship, SDMClient,
        Schema, StructuredDataError,
    )
    from sdm_connectors import DuckDBSource

    import json, threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    received = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            received.append(payload)
            table = payload['predict']['instance_table']
            at = table['columns'].index('instance_id')
            body = json.dumps({{
                'model': 'kumo-relational', 'id': 'answer', 'metadata': {{}},
                'predictions': [{{'id': str(row[at]), 'row_index': i, 'prediction': 1.0}}
                                for i, row in enumerate(table['rows'])],
            }}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{{server.server_address[1]}}'

    connection = duckdb.connect()
    for statement in {list(SHOP)!r}:
        connection.execute(statement)
    connection.execute('ALTER TABLE users ADD COLUMN plan VARCHAR')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(
            DuckDBSource(connection), {{'users': 'users', 'orders': 'orders'}}
        )
        del schema['users']['plan']
        schema['users'].columns['plan'] = Column('plan', 'string', 'text')
        schema['users']['plan'].stype = 'categorical'
        schema['users']['age'].stype = 'categorical'
        schema['orders'].primary_key = 'order_id'
        schema.relationships = [
            Relationship('orders', 'user_id', 'users', 'user_id')
        ]
        try:
            try:
                PredictiveQuery(
                    'PREDICT SUM(orders.nope, 0, 30, days) FOR EACH users.user_id',
                    schema,
                ).materialize(indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01'))
            except StructuredDataError as error:
                assert error.code == 'INVALID_REQUEST', error
            else:
                raise AssertionError('An unknown target column was accepted')
            request = PredictiveQuery({QUERY!r}, schema).materialize(
                indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01')
            )
            with SDMClient(url, max_retries=0) as client:
                result = client.predict(request, model=KumoRelational())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    print(json.dumps({{'rows': result.to_pandas().to_dict('records'),
                      'calls': len(received),
                      'old_imports': sorted(n for n in sys.modules if n.startswith(OLD))}}))
    """
    done = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {
        'rows': [
            {'row_index': 0, 'id': 0, 'prediction': 1.0},
            {'row_index': 1, 'id': 1, 'prediction': 1.0},
        ],
        'calls': 1,
        'old_imports': [],
    }
