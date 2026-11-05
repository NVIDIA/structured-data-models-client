# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded live authentication/readiness probes, not model acceptance."""

import argparse
import contextlib
import json
import os
from pathlib import Path

from live_api import environment, model_identities, preflight
from python_runner import HERE, source_tables
from shop_fixtures import load_fixture

from contract import effective_case, manifest


def probe_endpoint(endpoint):
    from sdm_client import SDMClient

    databricks = endpoint['transport'] == 'databricks'
    factory = SDMClient.for_databricks_serving if databricks else SDMClient
    with factory(**endpoint['client']) as client:
        if client.health_ready() is not True:
            raise ValueError('Endpoint is not ready')
        listing = 'unsupported-by-transport'
        if not databricks:
            try:
                listing = (
                    'observed-nonempty' if client.models() else 'observed-empty'
                )
            except Exception:
                listing = 'unavailable'
    return {'ready': True, 'model_listing': listing}


def probe_source(connector, config, fixture):
    from sdm_connectors import quote_ident, read_table

    settings = config.get('sources', {}).get(connector, {})
    if connector.startswith('s3-') or settings.get('paths'):
        import fsspec

        paths = settings.get('paths', {})
        if not paths:
            raise ValueError('Missing file fixture paths')
        # Avoid loading whole objects just to check authentication. Actual reader
        # schema/value equivalence remains mandatory in each matrix cell.
        for path in paths.values():
            with fsspec.open(path, 'rb', block_size=1) as stream:
                if not stream.read(1):
                    raise ValueError('Empty fixture resource')
        return
    with contextlib.ExitStack() as stack:
        _, tables, connection = source_tables(stack, connector, config, fixture)
        if connection is None:
            return  # Generated local fixtures are bounded by the checked-in data.
        mappings = settings.get('tables', {name: name for name in tables})
        for actual in mappings.values():
            if connector == 'databricks-spark':
                namespace = settings.get('source', {})
                qualified = '.'.join(
                    '`' + part.replace('`', '``') + '`'
                    for part in (
                        namespace.get('catalog'),
                        namespace.get('schema'),
                        actual,
                    )
                    if part
                )
                rows = connection.table(qualified).limit(1).toPandas()
            else:
                namespace = settings.get('source', {})
                parts = (
                    namespace.get('catalog', namespace.get('database')),
                    namespace.get('schema'),
                    actual,
                )
                quoted = '.'.join(
                    quote_ident(
                        part, '`' if connector.startswith('databricks') else '"'
                    )
                    for part in parts
                    if part
                )
                query = 'SELECT * FROM ' + quoted + ' LIMIT 1'
                rows = read_table(connection, query=query)
            if len(rows) != 1:
                raise ValueError('Fixture table is empty')


def check(config_path, *, giant_endpoint=None):
    config = json.loads(config_path.read_text())
    if giant_endpoint:
        # This is the same transport constructor the two giant tests use.
        return {
            'scope': 'giant-route-readiness-only',
            'endpoint': probe_endpoint(
                {
                    'transport': 'databricks',
                    'client': {'endpoint': giant_endpoint},
                }
            ),
        }
    missing = preflight(environment(config_path, config, os.environ))
    if missing:
        raise ValueError('Registered live resources are incomplete')
    identities = model_identities(config)
    data = manifest()
    checked = []
    # Effective variants may select different physical fixture namespaces.
    pairs = dict.fromkeys(
        (connector, effective_case(case, connector)['fixture'])
        for case in data['scenarios']
        for connector in case['connectors']
    )
    for connector, fixture_name in pairs:
        fixture = load_fixture(HERE / fixture_name)
        scoped = {
            **config,
            'sources': {
                **config.get('sources', {}),
                **config.get('fixture_sources', {}).get(fixture_name, {}),
            },
        }
        probe_source(connector, scoped, fixture)
        checked.append({'connector': connector, 'fixture': fixture_name})
    endpoints = {
        name: probe_endpoint(endpoint)
        for name, endpoint in config['endpoints'].items()
    }
    return {
        'scope': 'authentication-bounded-read-and-route-readiness-only',
        'sources': checked,
        'endpoints': endpoints,
        'configured_models': identities,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--giant-endpoint')
    args = parser.parse_args()
    try:
        result = {
            'status': 'passed',
            **check(args.config, giant_endpoint=args.giant_endpoint),
        }
    except Exception as error:
        # Never serialize exception text, connection arguments, URLs or tokens.
        result = {'status': 'failed', 'error_type': type(error).__name__}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    if result['status'] != 'passed':
        raise SystemExit(
            'Live resource preflight failed; no acceptance certified'
        )


if __name__ == '__main__':
    main()
