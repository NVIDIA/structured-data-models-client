#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demo the KumoRelational client against a local Universal TFM NIM.

This example intentionally does not hand-build the Universal TFM prediction
payload. The user-facing prediction flow is:

1. Initialize the client.
2. Build a Graph from local tables.
3. Call KumoRelational.predict(...) with a predictive query.

Run on the GPU host where the container is reachable:

    python examples/universal_tfm_endpoint_demo.py

Override the target if needed:

    python examples/universal_tfm_endpoint_demo.py --base-url http://host:8001
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd
import sdm_relational_engine.rfm as rfm
from predictive_query.schema.typing import Stype
from sdm_relational_engine.client import (
    NimClient,
)
from sdm_relational_engine.client.endpoints import Endpoint, HTTPMethod

DEFAULT_BASE_URL = os.getenv('TFM_BASE_URL', 'http://127.0.0.1:8001')


def build_demo_graph() -> rfm.Graph:
    tables = {
        'USERS': pd.DataFrame(
            {
                'USER_ID': [0, 1, 2, 3, 4],
                'AGE': [20.0, 30.0, 40.0, 50.0, 35.0],
                'GENDER': ['male', 'female', 'female', 'male', 'female'],
                'STATUS': ['A', 'B', 'A', 'C', None],
            }
        ),
        'ORDERS': pd.DataFrame(
            {
                'ORDER_ID': list(range(13)),
                'USER_ID': [0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3],
                'STORE_ID': [0, 1, 0, 1, 2, 2, 0, 1, 2, 0, 1, 1, 2],
                'AMOUNT': [
                    10.0,
                    15.0,
                    float('nan'),
                    20.0,
                    25.0,
                    30.0,
                    10.0,
                    25.0,
                    20.0,
                    10.0,
                    15.0,
                    15.0,
                    20.0,
                ],
                'CAT': [10, 15, None, 20, 25, 30, 11, 26, 27, 28, 29, 32, 33],
                'TIME': pd.to_datetime(
                    [
                        '2025-01-01',
                        '2024-12-20',
                        '2025-01-03',
                        '2025-01-02',
                        '2025-01-03',
                        '2025-01-04',
                        '2025-01-09',
                        '2025-01-02',
                        '2025-01-02',
                        '2025-01-01',
                        '2025-01-02',
                        '2025-01-03',
                        '2025-01-04',
                    ]
                ),
            }
        ),
        'STORES': pd.DataFrame(
            {
                'STORE_ID': [0, 1, 2],
                'CAT': ['burger', 'pizza', 'fries'],
            }
        ),
    }

    graph = rfm.Graph.from_data(tables, verbose=False)
    graph['USERS']['AGE'].stype = Stype.numerical
    graph['ORDERS']['AMOUNT'].stype = Stype.numerical
    return graph


def run_high_level_prediction(base_url: str, api_key: str | None) -> None:
    print('\n=== client.relational(graph).predict(...) ===')
    from sdm_client import RelationalClient

    graph = build_demo_graph()
    with RelationalClient(url=base_url, api_key=api_key) as client:
        result = client.relational(graph).predict(
            'PREDICT USERS.STATUS = "A" FOR USERS.USER_ID = 4',
            run_mode='best',
            anchor_time=pd.Timestamp('2025-01-10'),
            num_neighbors=[],
            inference_config={'num_estimators': 1},
        )

    print(result.to_string(index=False))


def run_sdk_route_descriptors(
    base_url: str,
    api_key: str | None,
    timeout: float,
    max_body_chars: int,
) -> None:
    """Show low-level client route descriptors for non-predict NIM routes."""
    print('\n=== client route descriptors for NIM metadata ===')
    client = NimClient(base_url, api_key=api_key)

    endpoint_examples = (
        ('health live', Endpoint('/v1/health/live', HTTPMethod.GET)),
        ('health ready', Endpoint('/v1/health/ready', HTTPMethod.GET)),
        ('version', Endpoint('/v1/version', HTTPMethod.GET)),
        ('models', Endpoint('/v1/models', HTTPMethod.GET)),
        ('metadata', Endpoint('/v1/metadata', HTTPMethod.GET)),
        ('manifest', Endpoint('/v1/manifest', HTTPMethod.GET)),
        ('license', Endpoint('/v1/license', HTTPMethod.GET)),
        ('metrics', Endpoint('/v1/metrics', HTTPMethod.GET)),
    )

    for name, endpoint in endpoint_examples:
        response = client._request(endpoint, timeout=timeout)
        print(f'\n{name}: {endpoint.method.value} {endpoint.get_path()}')
        print(f'status: {response.status_code}')
        if response.content:
            print(_format_response(response, max_body_chars))


def _format_response(response: Any, max_chars: int) -> str:
    content_type = response.headers.get('content-type', '')
    if 'json' in content_type:
        try:
            return _truncate(
                json.dumps(response.json(), indent=2, sort_keys=True),
                max_chars,
            )
        except ValueError:
            pass
    return _truncate(response.text, max_chars)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f'{text[:max_chars]}... <truncated {len(text) - max_chars} chars>'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Demo the KumoRelational client against a Universal TFM NIM.',
    )
    parser.add_argument(
        '--base-url',
        default=DEFAULT_BASE_URL,
        help=f'Base URL for the NIM. Default: {DEFAULT_BASE_URL}',
    )
    parser.add_argument(
        '--api-key',
        default=os.getenv('KUMO_RELATIONAL_API_KEY'),
        help='Optional API key, only needed if the NIM is behind an '
        'authenticating gateway. Default: KUMO_RELATIONAL_API_KEY or none.',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=120.0,
        help='Per-request timeout in seconds. Default: 120.',
    )
    parser.add_argument(
        '--max-body-chars',
        type=int,
        default=1000,
        help='Maximum metadata response characters to print. Default: 1000.',
    )
    parser.add_argument(
        '--skip-metadata',
        action='store_true',
        help='Only run the high-level prediction demo.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip('/')
    run_high_level_prediction(base_url, args.api_key)
    if not args.skip_metadata:
        run_sdk_route_descriptors(
            base_url,
            args.api_key,
            args.timeout,
            args.max_body_chars,
        )


if __name__ == '__main__':
    main()
