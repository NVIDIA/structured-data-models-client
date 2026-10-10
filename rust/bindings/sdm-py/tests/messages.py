# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Control messages the binding's tests send, written as the contract's JSON.

The Python host of T1.9 writes these for real; here they are spelled out,
so a test reads as the messages it exchanges.
"""

from __future__ import annotations

import json
from typing import Any

#: The contract version these messages are written for.
CONTRACT = 6

FACTS: dict[str, Any] = {
    'now': {'t': 'ts', 'unit': 'us', 'zone': 'UTC', 'v': '1767225600000000'},
    'local_zone': None,
    'zone_roots': [],
    'platform': {'os': 'linux', 'strftime_pads_year': False},
}


def encoded(value: object) -> bytes:
    return json.dumps(value).encode()


def start(operation: dict[str, Any], contract: int = CONTRACT) -> bytes:
    return encoded(
        {
            'message': 'start',
            'contract': contract,
            'facts': FACTS,
            'operation': operation,
        }
    )


def validate(query: str) -> tuple[bytes, dict[int, bytes]]:
    """A ``validate_query`` start over one table, with its attachments."""
    schema = {
        'tables': [
            {
                'name': 'users',
                'columns': [
                    {'name': 'id', 'stype': 'ID', 'dtype': 'int'},
                    {'name': 'age', 'stype': 'numerical', 'dtype': 'float'},
                ],
                'primary_key': 'id',
                'time_column': None,
                'end_time_column': None,
            }
        ],
        'links': [],
    }
    operation = {
        'op': 'validate_query',
        'query': 1,
        'schema': 2,
        'profile': {'name': 'RELATIONAL', 'model': None},
    }
    return start(operation), {1: encoded(query), 2: encoded(schema)}


def operation_for_tests(test: str, **fields: object) -> bytes:
    return start({'op': 'test', 'test': test, **fields})


def fetch(
    expect: dict[str, Any], echo: bool = False
) -> tuple[bytes, dict[int, bytes]]:
    """A start of the test-only fetch: one scan, expecting ``expect``."""
    request: dict[str, Any] = {
        'request': {
            'request': 'scan',
            'table': 't',
            'columns': [],
            'filters': [],
            'limit': None,
        },
        'plan': {'plan': 'host', 'expect': expect},
        'echo': echo,
    }
    return operation_for_tests('fetch', fetch=1), {1: encoded(request)}


def fetch_reply(data: int = 1) -> bytes:
    return encoded(
        {
            'message': 'fetch_reply',
            'id': 1,
            'outcome': {'outcome': 'data', 'data': data},
        }
    )


def int64_field(
    name: str, nullable: bool = True, metadata: dict[str, str] | None = None
) -> dict[str, Any]:
    """A contract field of type ``int64``."""
    field: dict[str, Any] = {
        'name': name,
        'type': {'type': 'int64'},
        'nullable': nullable,
    }
    if metadata:
        field['metadata'] = metadata
    return field


def read(control: bytes) -> dict[str, Any]:
    message: dict[str, Any] = json.loads(control)
    return message
