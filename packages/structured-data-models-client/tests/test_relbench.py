# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Both language loaders consume tests/fixtures/relbench unchanged."""

import hashlib
import json
from pathlib import Path

import pytest

from sdm_client import Schema
from sdm_client.errors import StructuredDataError

FIXTURE = Path(__file__).resolve().parents[3] / 'tests/fixtures/relbench'


@pytest.fixture
def catalog(monkeypatch):
    from sdm_client import _relbench

    registry = json.loads((FIXTURE / 'registry.json').read_text())
    overrides = json.loads((FIXTURE / 'overrides.json').read_text())
    monkeypatch.setattr(_relbench, '_catalog', lambda: (registry, overrides))
    return _relbench, registry


def seed(cache, registry):
    for dataset, entry in registry['datasets'].items():
        for table in entry['tables']:
            path = (
                cache
                / entry['repository']
                / registry['revisions'][entry['repository']]
                / dataset
                / 'db'
                / f'{table}.parquet'
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                (FIXTURE / dataset / 'db' / path.name).read_bytes()
            )


def test_offline_metadata_rows_and_no_http(catalog, tmp_path, monkeypatch):
    loader, registry = catalog
    seed(tmp_path, registry)
    monkeypatch.setattr(
        loader, 'urlopen', lambda *a, **kw: pytest.fail('offline HTTP')
    )
    schema = Schema.from_relbench('FIXTURE', cache_dir=tmp_path, offline=True)
    assert schema['customers'].primary_key == 'id'
    assert schema['orders'].primary_key is None
    assert schema['customers'].time_column is None
    assert schema['orders'].time_column == 'time'
    assert 'excluded' not in schema['customers'].columns
    assert schema['customers']['name'].stype == 'text'
    assert schema['orders'].data['amount'].tolist() == [2.5, 3.5]
    assert len(schema.relationships) == 1
    assert schema.relationships[0].target_column == 'id'


def test_corrupt_offline_and_unknown(catalog, tmp_path):
    _, registry = catalog
    seed(tmp_path, registry)
    next(tmp_path.rglob('orders.parquet')).write_bytes(b'corrupt')
    with pytest.raises(StructuredDataError) as error:
        Schema.from_relbench('fixture', cache_dir=tmp_path, offline=True)
    assert error.value.code == 'PROVIDER_ERROR'
    with pytest.raises(StructuredDataError) as error:
        Schema.from_relbench('unknown', cache_dir=tmp_path)
    assert error.value.code == 'INVALID_REQUEST'


def test_all_pins_have_valid_metadata():
    from sdm_client import _relbench

    registry, overrides = _relbench._catalog()
    assert len(registry['datasets']) == 29
    for dataset, entry in registry['datasets'].items():
        assert len(registry['revisions'][entry['repository']]) == 40
        for table, spec in entry['tables'].items():
            assert (
                len(bytes.fromhex(spec['sha256']))
                == hashlib.sha256().digest_size
            )
            for target in spec['foreign_keys'].values():
                assert entry['tables'][target]['primary_key'] is not None
            _relbench._relative(
                entry['repository'],
                registry['revisions'][entry['repository']],
                dataset,
                table,
            )
    assert set(overrides) <= set(registry['datasets'])


@pytest.mark.parametrize('failure', ['truncated', 'wrong-digest'])
def test_http_download_cache_and_failure_recovery(
    catalog, tmp_path, monkeypatch, failure
):
    import http.server
    import threading

    loader, _ = catalog
    requests = []
    broken = [False]

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            data = (
                FIXTURE / 'rel-fixture/db' / self.path.rsplit('/', 1)[-1]
            ).read_bytes()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            if broken[0]:
                data = data[:10] if failure == 'truncated' else bytes(len(data))
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(
        loader, '_HUB', f'http://127.0.0.1:{server.server_port}'
    )
    try:
        Schema.from_relbench('fixture', cache_dir=tmp_path)
        assert len(requests) == 2
        Schema.from_relbench('fixture', cache_dir=tmp_path)
        Schema.from_relbench('fixture', cache_dir=tmp_path, offline=True)
        assert len(requests) == 2
        customers = next(tmp_path.rglob('customers.parquet'))
        preserved = customers.read_bytes()
        orders = next(tmp_path.rglob('orders.parquet'))
        orders.write_bytes(b'invalid cached bytes')
        broken[0] = True
        with pytest.raises(StructuredDataError) as error:
            Schema.from_relbench('fixture', cache_dir=tmp_path)
        assert error.value.code == 'PROVIDER_ERROR'
        assert orders.read_bytes() == b'invalid cached bytes'
        assert customers.read_bytes() == preserved
        assert not list(tmp_path.rglob('.*parquet*'))
        broken[0] = False
        assert (
            len(
                Schema.from_relbench(
                    'fixture', cache_dir=tmp_path
                ).relationships
            )
            == 1
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_unsafe_manifest_path_rejected(catalog, tmp_path):
    _, registry = catalog
    entry = registry['datasets'].pop('rel-fixture')
    registry['datasets']['../escape'] = entry
    with pytest.raises(StructuredDataError) as error:
        Schema.from_relbench('../escape', cache_dir=tmp_path)
    assert error.value.code == 'INVALID_REQUEST'


def test_loaded_schema_composes_reports_exports_and_persisted_requests(
    catalog, tmp_path, monkeypatch
):
    import pickle
    import warnings

    import pandas as pd

    from sdm_client import KumoRelational, PredictiveQuery

    loader, registry = catalog
    seed(tmp_path, registry)
    monkeypatch.setattr(
        loader, 'urlopen', lambda *a, **kw: pytest.fail('offline HTTP')
    )
    schema = Schema.from_relbench('fixture', cache_dir=tmp_path, offline=True)
    metadata = schema.metadata_report()
    links = schema.link_report()
    assert [table.name for table in metadata.tables] == ['customers', 'orders']
    assert metadata.tables[0].primary_key == ('id',)
    assert metadata.tables[1].primary_key == ()
    assert metadata.tables[1].time_column == 'time'
    assert links.to_pandas().to_dict('records') == [
        dict(
            source_table='orders',
            source_column='customer_id',
            target_table='customers',
            target_column='id',
        )
    ]
    schema.visualize(tmp_path / 'loaded.html')
    schema.visualize(tmp_path / 'loaded.mmd')
    assert (tmp_path / 'loaded.html').read_text() == schema.to_html()
    assert (tmp_path / 'loaded.mmd').read_text().strip() == schema.to_mermaid()
    assert schema.to_mermaid().count(' -->|') == 1

    # Reports and cached bytes must not share mutable metadata with this owner.
    schema['customers']['name'].stype = 'categorical'
    second = Schema.from_relbench(
        'rel-fixture', cache_dir=tmp_path, offline=True
    )
    assert second['customers']['name'].stype == 'text'
    assert metadata.tables[0].columns[1].stype == 'text'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(
            'PREDICT SUM(orders.amount, 0, 1, days) FOR EACH customers.id',
            schema,
        ).materialize(
            indices=[1],
            anchor_time=pd.Timestamp('2024-01-02'),
            model=KumoRelational(),
            num_context_samples=1,
            random_seed=7,
        )
    assert request.predict_rows == 1
    report = request.sanitization_report
    assert report.status == 'available'
    assert [(r.table, r.input_rows, r.output_rows) for r in report.tables] == [
        ('customers', 2, 2),
        ('orders', 2, 2),
    ]
    assert all(
        r.null_primary_key_rows
        == r.duplicate_primary_key_rows
        == r.null_time_rows
        == 0
        for r in report.tables
    )
    restored = pickle.loads(pickle.dumps(request))
    assert restored.sanitization_report == report
    assert restored.payloads == request.payloads
    schema['orders'].data.loc[0, 'amount'] = -999
    assert second['orders'].data['amount'].tolist() == [2.5, 3.5]
    assert restored.sanitization_report == report
