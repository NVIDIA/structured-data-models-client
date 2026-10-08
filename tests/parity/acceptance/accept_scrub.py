# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P2.scrub: planted secrets are all removed, nothing else is, and ``scan``
finds no secret in any stored transcript.

Every secret here is assembled at run time, so no secret-shaped literal sits
in this file for a scanner to trip on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from _support import (
    REPO,
    SCHEMA,
    SCRUB,
    TRANSCRIPTS,
    arrow_refs,
    deep,
    execute_steps,
    load,
    output,
    read_arrow,
    recorded,
    run,
    write_arrow,
)

_HEX = '0123456789abcdef'

# Known from the recording environment: only the variable says they're secret.
ENV_SECRETS = {
    'DATABRICKS_SERVER_HOSTNAME': 'dbc-0a1b2c3d-'
    + 'pl4nted.cloud.databricks.com',
    'DATABRICKS_TOKEN': 'pLaNtEdOaUtH' + 'TokenValue' + '0123456789xyz',
    'DATABRICKS_HTTP_PATH': '/sql/1.0/warehouses/' + 'fedcba' + '9876543210',
    'DATABRICKS_WAREHOUSE_ID': 'fedcba' + '9876543210',
    'SNOWFLAKE_ACCOUNT': 'plantedorg-' + 'plantedacct',
    'SNOWFLAKE_USER': 'PLANTED_' + 'LOADER_USER',
    'SNOWFLAKE_PASSWORD': 'Sn0w-' + 'Planted-Secret!',
    'PGHOST': 'instance-0f1e2d3c-4b5a-6978-8a9b-0c1d2e3f4a5b'
    + '.database.cloud.databricks.com',
    'PGUSER': 'planted.user' + '@example.com',
    'PGPASSWORD': 'Lakebase-' + 'Planted-Pass',
}

# Recognizable by their shape alone.
PAT = 'dapi' + _HEX[::-1] + _HEX
JWT = '.'.join(
    [
        'eyJhbGciOiJSUzI1NiIsImtpZCI6InBsYW50ZWQifQ',
        'eyJzdWIiOiJwbGFudGVkQGV4YW1wbGUuY29tIn0',
        'c2lnbmF0dXJlcGxhbnRlZHNpZ25hdHVyZQ',
    ]
)
BEARER = 'Zm9vYmFyYmF6' + 'cGxhbnRlZHRva2Vu'
AZURE_HOST = 'adb-' + '9876543210987654' + '.3.azuredatabricks.net'
WORKSPACE_IN_URL = '1234123412341234'
WORKSPACE_KEY = '7777888899990000'
SNOWFLAKE_HOST = 'xy98765' + '.eu-central-1.snowflakecomputing.com'
DSN_PASSWORD = 'Pl4nted-' + 'Pa55word'
PASSWORD_KEY = 'Hunter2-' + 'Planted-Value'
TOKEN_KEY = 'opaque-planted-' + 'token-value-42'
AWS_KEY_ID = 'AK' + 'IA' + 'PLANTED0EXAMPLE1'
AWS_SECRET = 'wJalrXUtnFEMI/K7MDENG/' + 'bPxRfiCYPLANTEDKEY'
ACCOUNT_KEY = 'acme-planted-' + 'account'

PATTERN_SECRETS = [
    PAT,
    JWT,
    BEARER,
    AZURE_HOST,
    WORKSPACE_IN_URL,
    WORKSPACE_KEY,
    SNOWFLAKE_HOST,
    DSN_PASSWORD,
    PASSWORD_KEY,
    TOKEN_KEY,
    AWS_KEY_ID,
    AWS_SECRET,
    ACCOUNT_KEY,
]
ALL_SECRETS = [*ENV_SECRETS.values(), *PATTERN_SECRETS]

# Must survive the scrub unchanged.
KEPT_SQL = (
    'SELECT "user_id", "ts" FROM "main"."shop"."orders" WHERE "user_id" = ?'
)
KEPT_PQL = 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'
KEPT_ROWS = [[1, 2.5, 'north'], [2, 3.5, 'south']]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(ENV_SECRETS)
    return env


def _hand_made(directory: Path) -> Path:
    r"""A live-style Databricks transcript with secrets in every corner."""
    import pyarrow as pa

    directory.mkdir(parents=True)
    host = ENV_SECRETS['DATABRICKS_SERVER_HOSTNAME']
    dsn = (
        f'postgresql://{ENV_SECRETS["PGUSER"]}:{DSN_PASSWORD}'
        f'@{ENV_SECRETS["PGHOST"]}:5432/databricks_postgres'
    )
    transcript = {
        'format_version': 1,
        'name': 'live/databricks-count',
        'backend': 'databricks',
        'task_kind': 'count',
        'data': {
            'catalog': 'main',
            'schema': 'shop',
            'connection': {
                'server_hostname': host,
                'http_path': ENV_SECRETS['DATABRICKS_HTTP_PATH'],
                'access_token': ENV_SECRETS['DATABRICKS_TOKEN'],
                'lakebase': dsn,
            },
        },
        'calls': [
            {
                'call': 'Schema.from_source',
                'args': {
                    'tables': ['users', 'orders'],
                    'url': f'https://{host}',
                },
            },
            {'call': 'PredictiveQuery', 'args': {'query': KEPT_PQL}},
        ],
        'steps': [
            {
                'op': 'execute',
                'sql': f'SELECT current_user() /* {JWT} */',
                'params': [
                    f'https://{AZURE_HOST}/?o={WORKSPACE_IN_URL}',
                    ENV_SECRETS['DATABRICKS_WAREHOUSE_ID'],
                    7,
                ],
                'result': 'step-000.arrow',
                'error': None,
            },
            {
                'op': 'execute',
                'sql': KEPT_SQL,
                'params': [3],
                'result': None,
                'error': {
                    'type': 'ServerOperationError',
                    'message': (
                        f'Authorization: Bearer {BEARER} was refused by '
                        f'{SNOWFLAKE_HOST} for '
                        f'{ENV_SECRETS["SNOWFLAKE_ACCOUNT"]} as '
                        f'{ENV_SECRETS["SNOWFLAKE_USER"]} with '
                        f'{ENV_SECRETS["SNOWFLAKE_PASSWORD"]} and '
                        f'{ENV_SECRETS["PGPASSWORD"]}; token {PAT}'
                    ),
                },
            },
            {'op': 'commit'},
        ],
        'payloads': [
            {
                'metadata': {
                    'request_id': '<request-id>',
                    'source_query': KEPT_PQL,
                },
                'context': {
                    'instance_table': {
                        'columns': ['user_id', 'amount', 'region'],
                        'format': 'arrays',
                        'rows': KEPT_ROWS,
                    },
                    'credentials': {
                        'workspace_id': WORKSPACE_KEY,
                        'password': PASSWORD_KEY,
                        'token': TOKEN_KEY,
                        'aws_access_key_id': AWS_KEY_ID,
                        'aws_secret_access_key': AWS_SECRET,
                        'account': ACCOUNT_KEY,
                    },
                },
            }
        ],
        'result': 'result.arrow',
    }
    path = directory / 'live-databricks-count.json'
    path.write_text(json.dumps(transcript, indent=1))
    write_arrow(
        directory / 'step-000.arrow',
        pa.table({'who': [host, 'north'], 'n': [1, 2]}),
        'file',
    )
    write_arrow(
        directory / 'result.arrow',
        pa.table({'user_id': [1, 2], 'prediction': [0.25, 0.75]}),
        'file',
    )
    return path


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            *value.keys(),
            *(s for v in value.values() for s in _strings(v)),
        ]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _leaks(root: Path) -> list[tuple[str, str]]:
    found = []
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        raw = path.read_bytes()
        texts = [raw.decode('utf-8', 'replace')]
        if path.suffix == '.json':
            texts += _strings(json.loads(raw))
        elif path.suffix == '.arrow':
            table, _ = read_arrow(path)
            texts += [str(v) for col in table.to_pydict().values() for v in col]
        for secret in ALL_SECRETS:
            if any(secret in text for text in texts):
                found.append((path.name, secret))
    return found


def _scrub(source: Path, target: Path) -> None:
    proc = run(SCRUB, 'scrub', source, target, env=_env(), timeout=300)
    assert proc.returncode == 0, output(proc)


def test_planted_secrets_are_all_removed(tmp_path: Path) -> None:
    source = _hand_made(tmp_path / 'in').parent
    planted = {secret for _, secret in _leaks(source)}
    assert planted == set(ALL_SECRETS), 'not every secret was planted'
    target = tmp_path / 'out'
    _scrub(source, target)
    assert not _leaks(target), _leaks(target)


def test_scrubbing_keeps_what_is_not_secret(tmp_path: Path) -> None:
    source = _hand_made(tmp_path / 'in')
    target = tmp_path / 'out'
    _scrub(source.parent, target)
    scrubbed = load(target / source.name)
    assert scrubbed['name'] == 'live/databricks-count'
    assert scrubbed['backend'] == 'databricks'
    assert scrubbed['format_version'] == 1
    assert len(scrubbed['steps']) == 3
    assert scrubbed['steps'][1]['sql'] == KEPT_SQL
    assert scrubbed['steps'][1]['params'] == [3]
    assert scrubbed['steps'][2] == {'op': 'commit'}
    payload = scrubbed['payloads'][0]
    assert payload['metadata']['source_query'] == KEPT_PQL
    assert payload['context']['instance_table']['rows'] == KEPT_ROWS
    for ref in arrow_refs(scrubbed):
        assert (target / ref).is_file(), ref
    result, _ = read_arrow(target / scrubbed['result'])
    assert result.to_pydict() == {'user_id': [1, 2], 'prediction': [0.25, 0.75]}
    step, _ = read_arrow(target / scrubbed['steps'][0]['result'])
    assert step.column('who').to_pylist()[1] == 'north'


def test_a_scrubbed_recording_still_validates(tmp_path: Path) -> None:
    r"""Plant secrets in a real recording's SQL; scrub; it still validates."""
    import jsonschema

    entry = next(e for e in recorded() if e['backend'] == 'postgres')
    stored = REPO / entry['transcript']
    source = tmp_path / 'in'
    source.mkdir()
    transcript = deep(load(stored))
    for ref in arrow_refs(transcript):
        (source / ref).parent.mkdir(parents=True, exist_ok=True)
        (source / ref).write_bytes((stored.parent / ref).read_bytes())
    originals = {}
    for i in execute_steps(transcript):
        originals[i] = transcript['steps'][i]['sql']
        transcript['steps'][i]['sql'] += (
            f' /* {ENV_SECRETS["PGHOST"]} {ENV_SECRETS["PGPASSWORD"]} {JWT} */'
        )
    (source / stored.name).write_text(json.dumps(transcript, indent=1))
    target = tmp_path / 'out'
    _scrub(source, target)
    assert not _leaks(target), _leaks(target)
    scrubbed = load(target / stored.name)
    jsonschema.validate(scrubbed, load(SCHEMA))
    for i, sql in originals.items():
        assert scrubbed['steps'][i]['sql'].startswith(sql), (
            f'step {i}: the scrub changed SQL that held no secret'
        )


def test_scan_finds_planted_secrets_and_passes_scrubbed_ones(
    tmp_path: Path,
) -> None:
    source = _hand_made(tmp_path / 'in').parent
    proc = run(SCRUB, 'scan', source, env=_env(), timeout=300)
    assert proc.returncode == 1, output(proc)
    target = tmp_path / 'out'
    _scrub(source, target)
    proc = run(SCRUB, 'scan', target, env=_env(), timeout=300)
    assert proc.returncode == 0, output(proc)


@pytest.mark.parametrize(
    'secret', [PAT, JWT, AZURE_HOST, SNOWFLAKE_HOST, f'Bearer {BEARER}']
)
def test_scan_finds_a_secret_by_its_shape_alone(
    tmp_path: Path, secret: str
) -> None:
    root = tmp_path / 'corpus'
    root.mkdir()
    (root / 'one.json').write_text(
        json.dumps({'steps': [{'op': 'execute', 'sql': f'SELECT {secret!r}'}]})
    )
    env = {k: v for k, v in os.environ.items() if k not in ENV_SECRETS}
    proc = run(SCRUB, 'scan', root, env=env, timeout=300)
    assert proc.returncode == 1, output(proc)


def test_scan_reads_arrow_files(tmp_path: Path) -> None:
    import pyarrow as pa

    root = tmp_path / 'corpus'
    root.mkdir()
    (root / 'one.json').write_text(json.dumps({'steps': []}))
    write_arrow(root / 'one.arrow', pa.table({'host': [AZURE_HOST]}), 'stream')
    env = {k: v for k, v in os.environ.items() if k not in ENV_SECRETS}
    proc = run(SCRUB, 'scan', root, env=env, timeout=300)
    assert proc.returncode == 1, output(proc)


def test_scan_finds_no_secret_in_the_stored_corpus() -> None:
    assert TRANSCRIPTS.is_dir()
    proc = run(SCRUB, 'scan', timeout=300)
    assert proc.returncode == 0, output(proc)
