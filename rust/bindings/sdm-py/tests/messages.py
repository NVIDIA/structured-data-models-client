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
CONTRACT = 28

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
            'table': {'source': ['t']},
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


#: A sample of every operation's result, as a ``done`` step carries it
#: (``OperationResult``), each field set: what a host decodes into its own
#: typed structures (T3.16, the structure design's fix D). Java reads the
#: same samples.
RESULTS: dict[str, dict[str, Any]] = {
    'validate_query': {
        'op': 'validate_query',
        'accepted': True,
        'diagnostics': [
            {
                'code': 'W1',
                'message': 'a note',
                'severity': 'warning',
                'span': {
                    'start_row': 1,
                    'start_col': 2,
                    'end_row': 1,
                    'end_col': 9,
                },
            }
        ],
        'query': {
            'entity_table': 'users',
            'problem_type': 'forecast',
            'windowed': True,
            'num_forecasts': 1,
            'text': 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
        },
    },
    'describe_source': {'op': 'describe_source', 'description': 3},
    'edit_schema': {
        'op': 'edit_schema',
        'schema': 4,
        'declared': {'table': 'users', 'column': 'age'},
    },
    'materialize': {
        'op': 'materialize',
        'state': {
            'version': 2,
            'payload': 5,
            'query': 6,
            'forecast': [
                {'step': 2, 'payload': 7, 'query': 8, 'request_id': None}
            ],
            'batches': [{'payload': 9, 'rows': 40, 'request_id': None}],
            'ranking': {
                'top_k': 3,
                'candidate': 0,
                'value': '"a"',
                'parts': [
                    {
                        'candidate': 1,
                        'value': '"b"',
                        'payload': 10,
                        'query': 11,
                        'request_id': None,
                    }
                ],
            },
        },
        'query': {
            'entity_table': 'users',
            'problem_type': None,
            'windowed': False,
            'num_forecasts': 2,
            'text': None,
        },
    },
    'predict': {
        'op': 'predict',
        'result': 12,
        'model': 'kumo-relational',
        'request_id': 'sdm-1',
        'metadata': 13,
        'outputs': ['prediction'],
        'unscored_candidates': [2, 4],
    },
}

#: A column's source facts as ``HeldColumn.source`` carries them (T3.16).
SOURCE_FACTS: dict[str, Any] = {
    'relation': {'source': ['users']},
    'dtype': 'string',
    'received': {'type': 'utf8'},
    'nullable': False,
    'unique': True,
    'declared_type': 'NUMERIC',
    'exact_number': 'declared_key',
}
