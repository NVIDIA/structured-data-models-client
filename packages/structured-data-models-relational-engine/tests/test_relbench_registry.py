# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A RelBench table loads only if its bytes match the digest pinned for it.

RelBench used to publish a hash file beside its archives, and the loader read
it from upstream's ``main`` on every call. When RelBench moved its data to
Hugging Face that file was deleted, and every load failed with a 404 before a
byte was downloaded. The digests, and the commit they were taken at, now ship
with the package, so these tests check the pinned registry itself and serve
the files from a local server rather than the Hub.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pooch
import pytest
from pytest_httpserver import HTTPServer
from sdm_relational_engine.rfm import relbench

_REVISION = 'c0ffee' * 6 + 'c0ff'
_REPOSITORY = 'stanford-star/relbench-v1'


def _parquet(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


_USERS = _parquet(pd.DataFrame({'user_id': [1, 2, 3]}))
_ORDERS = _parquet(
    pd.DataFrame(
        {
            'order_id': [10, 11, 12],
            'user_id': [1, 1, 3],
            'placed_at': pd.to_datetime(
                ['2024-01-01', '2024-01-02', '2024-01-03']
            ),
        }
    )
)


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _registry(**digests: str) -> dict[str, Any]:
    return {
        'revisions': {_REPOSITORY: _REVISION},
        'datasets': {
            'rel-shop': {
                'repository': _REPOSITORY,
                'tables': {
                    'users': {
                        'sha256': digests.get('users', _sha256(_USERS)),
                        'primary_key': 'user_id',
                        'time_column': None,
                        'foreign_keys': {},
                    },
                    'orders': {
                        'sha256': digests.get('orders', _sha256(_ORDERS)),
                        'primary_key': 'order_id',
                        'time_column': 'placed_at',
                        'foreign_keys': {'user_id': 'users'},
                    },
                },
            }
        },
    }


@pytest.fixture
def hub(
    httpserver: HTTPServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> HTTPServer:
    r"""A local stand-in for the Hub, serving each table at the pinned
    commit only, with an empty cache.
    """
    prefix = f'/datasets/{_REPOSITORY}/resolve/{_REVISION}/rel-shop/db'
    httpserver.expect_request(f'{prefix}/users.parquet').respond_with_data(
        _USERS
    )
    httpserver.expect_request(f'{prefix}/orders.parquet').respond_with_data(
        _ORDERS
    )
    monkeypatch.setattr(relbench, 'HUB', httpserver.url_for('').rstrip('/'))
    monkeypatch.setattr(relbench, 'cache_dir', lambda: tmp_path)
    # A mismatch is retried before it is reported; the wait between is
    # pooch's, and only slows these tests down.
    monkeypatch.setattr(pooch.core.time, 'sleep', lambda seconds: None)
    return httpserver


def test_tables_whose_digests_match_load_into_a_graph(
    hub: HTTPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(relbench, 'get_registry', _registry)

    graph = relbench.from_relbench('shop', verbose=False)

    assert graph['users'].primary_key_columns == ('user_id',)
    assert graph['orders'].primary_key_columns == ('order_id',)
    assert getattr(graph['orders'].time_column, 'name', None) == 'placed_at'
    assert [tuple(edge) for edge in graph.edges] == [
        ('orders', 'user_id', 'users')
    ]


def test_a_table_whose_bytes_differ_from_its_digest_is_refused(
    hub: HTTPServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        relbench, 'get_registry', lambda: _registry(orders='0' * 64)
    )

    with pytest.raises(ValueError, match='hash'):
        relbench.from_relbench('shop', verbose=False)

    # Nothing that failed its check is left where a later load would read it.
    cached = tmp_path / _REPOSITORY / _REVISION / 'rel-shop' / 'db'
    assert not (cached / 'orders.parquet').exists()


def test_a_cached_table_that_no_longer_matches_is_fetched_again(
    hub: HTTPServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(relbench, 'get_registry', _registry)
    relbench.from_relbench('shop', verbose=False)
    cached = tmp_path / _REPOSITORY / _REVISION / 'rel-shop' / 'db'
    (cached / 'orders.parquet').write_bytes(b'truncated')

    relbench.from_relbench('shop', verbose=False)

    assert (cached / 'orders.parquet').read_bytes() == _ORDERS


_HEX40 = re.compile(r'[0-9a-f]{40}')
_HEX64 = re.compile(r'[0-9a-f]{64}')


def test_every_shipped_dataset_is_pinned_to_a_commit_and_a_digest() -> None:
    r"""A branch name or a missing digest would let what loads change under
    a caller without any change here.
    """
    registry = relbench.get_registry()
    revisions = registry['revisions']
    datasets = registry['datasets']

    assert all(_HEX40.fullmatch(revision) for revision in revisions.values())
    assert {'rel-f1', 'rel-salt', 'rel-amazon', 'rel-hm'} <= set(datasets)
    for name, dataset in datasets.items():
        assert dataset['repository'] in revisions, name
        assert dataset['tables'], name
        for table, spec in dataset['tables'].items():
            assert _HEX64.fullmatch(spec['sha256']), (name, table)
            # A link to a table the dataset does not have fails only once the
            # graph is built, long after the download it waited for.
            assert set(spec['foreign_keys'].values()) <= set(
                dataset['tables']
            ), (name, table)


def test_the_tables_the_loader_adjusts_are_in_the_registry() -> None:
    r"""``EXCLUDE`` and the rel-salt stype fixes name tables by hand."""
    datasets = relbench.get_registry()['datasets']

    for dataset, tables in relbench.EXCLUDE.items():
        assert set(tables) <= set(datasets[f'rel-{dataset}']['tables'])
    assert {'salesdocument', 'salesdocumentitem', 'address'} <= set(
        datasets['rel-salt']['tables']
    )


@pytest.mark.integration
def test_rel_f1_loads_from_the_hub_at_its_pinned_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""The pinned digests match what the Hub serves. Needs the network, so it
    runs only under ``--runintegration``.
    """
    monkeypatch.setattr(relbench, 'cache_dir', lambda: tmp_path)

    graph = relbench.from_relbench('f1', verbose=False)

    assert len(graph.tables) == 9
    assert getattr(graph['races'].time_column, 'name', None) == 'date'
