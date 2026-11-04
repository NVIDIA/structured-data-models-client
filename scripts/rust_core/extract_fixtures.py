#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract maintained facts from a checksummed migration archive.

The archive is an explicit input, never needed by ordinary tests. No engine
code is called: expected values are copied from independent old recordings.
Run before deleting originals; --verify compares every compact expansion to
its source and validates every extracted observation and Arrow byte stream.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any

from fixture_tables import dumps, pack, unpack

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEST = Path('contract/fixtures/maintained')
VECTORS = (
    'view-expr-vectors.json',
    'view-step-vectors.json',
    'view-yaml-vectors.json',
    'sqlite-key-pair-vectors.json',
    'view-captures.json',
    'pinned/base.regression.core.json',
    'ranking-vectors.json',
    'view-metric-vectors.json',
)
REFERENCE = 'contract/reference/43e55f4f58924384c3cb52f67c8ed2e73ba45fa1.jsonl'
# Executable declarations supply the case inventory; no tests or engine run.
CONSUMERS = {
    'test_t28_ranking': 'CASE_KEYS',
    'test_t213_options': 'KEYS_ALL',
    'test_t28c_multicall': 'KEYS_ALL',
    'test_t214b_decimal_stype': 'KEYS',
    'test_t214b_host': 'CASE_KEYS',
    'test_t29_forecasting': None,
    'test_t214b_wording': None,
}


def sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def reference_keys() -> dict[str, list[str]]:
    consumers: dict[str, list[str]] = {}
    for name, attribute in CONSUMERS.items():
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'contract/tests' / f'{name}.py'
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if attribute:
            keys = getattr(module, attribute)
        elif name == 'test_t29_forecasting':
            keys = [
                module._runner().case_key(c, d)
                for c in module.FAMILY
                for d in module.DATABASES
            ]
        else:
            keys = [
                key
                for case in module.recorder().OWN
                for key in module.keys_of(case)
            ]
        consumers[name] = sorted(keys)
    consumers['rust/crates/sdm-testkit/tests/labels.rs'] = [
        'base.regression',
        'base.binary',
        'base.multiclass',
        'decimal.target-38-18',
    ]
    consumers[
        'packages/structured-data-models-client/tests_test_ops/test_the_labels.py'
    ] = consumers['rust/crates/sdm-testkit/tests/labels.rs']
    consumers['rust/crates/sdm-testkit/tests/duckdb_provider.rs'] = [
        'base.regression'
    ]
    consumers['tests/test_t21a_runner_rule.py'] = [
        's0.accept.sum-window',
        's0.accept.binary-classification',
    ]
    return consumers


def extract(archive: Path, verify: bool) -> None:
    provenance: dict[str, Any] = {
        'format': 'sdm-fixture-extraction/1',
        'archive_sha256': file_sha(archive),
        'extractor_sha256': file_sha(Path(__file__)),
        'table_codec_sha256': file_sha(
            Path(__file__).with_name('fixture_tables.py')
        ),
        'source_commit': '4e3f4a86fb37adcbcf093e39f7cabaf68a7bdd9f',
        'extractor': 'scripts/rust_core/extract_fixtures.py',
        'sources': {},
        'consumers': {},
    }
    with tarfile.open(archive, 'r:*') as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}

        def read(path: str) -> bytes:
            stream = tar.extractfile(members[path])
            assert stream is not None
            body = stream.read()
            provenance['sources'][path] = sha(body)
            return body

        def write(path: Path, body: bytes) -> None:
            target = ROOT / path
            if verify:
                assert target.read_bytes() == body, (
                    f'{path}: differs from extraction'
                )
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)

        for name in VECTORS:
            path = Path('contract') / name
            original = json.loads(read(str(path)))
            compact = pack(original)
            assert unpack(compact) == original
            write(path, dumps(compact).encode())

        decimal = json.loads(read('contract/decimal-vectors.json'))
        decimal_inputs = {
            key: decimal[key] for key in ('question', 'indices', 'anchor_time')
        }
        decimal_inputs['vectors'] = [
            {
                key: vector[key]
                for key in ('id', 'setup', 'indices')
                if key in vector
            }
            for vector in decimal['vectors']
            if vector['id'] == 'key.both.d9-0'
            or vector['id'].startswith('key.pair')
        ]
        write(
            DEST / 'decimal-key-inputs.json',
            json.dumps(decimal_inputs, indent=2).encode() + b'\n',
        )
        provenance['consumers']['decimal-key-inputs.json'] = [
            'contract/tests/test_t132b_decimal_vectors.py'
        ]

        consumers = reference_keys()
        keys = {key for cases in consumers.values() for key in cases}
        observations = []
        inventory = []
        sql_counts: dict[tuple[str, str], int] = {}
        lookups = []
        for raw in read(REFERENCE).splitlines():
            row = json.loads(raw)
            statements = ((row.get('driver') or {}).get('capture') or {}).get(
                'statements'
            ) or []
            inventory.append(
                {
                    'case': row['case'],
                    'identity': {
                        key: row['identity'][key]
                        for key in ('runner', 'reference')
                    },
                    'statement_count': len(statements),
                }
            )
            if row['case'].endswith('@postgres'):
                for statement in statements:
                    key = (row['case'], statement['sql'])
                    sql_counts[key] = sql_counts.get(key, 0) + 1
                    if 'WHERE __KUMO_ROW__ = 1' in statement['sql']:
                        for result in statement['results']:
                            values = result['values']
                            lookups.append(
                                [
                                    statement['sql'],
                                    values['columns'],
                                    values['types'],
                                ]
                            )
            if row['case'] in keys:
                # SQL-result and observation-schema tests need complete observations.
                # Product parity checks consume outcome, public and identity;
                # driver SQL/results are never their oracle.
                if row['case'] not in ('base.regression',):
                    row = {k: v for k, v in row.items() if k != 'driver'}
                observations.append(row)
        assert {row['case'] for row in observations} == keys
        write(
            DEST / 'reference-inventory.json',
            dumps(pack({'cases': inventory})).encode(),
        )
        sql_facts = {
            'statements': [
                {'case': case, 'sql': sql, 'count': count}
                for (case, sql), count in sorted(sql_counts.items())
            ],
            'lookups': lookups,
        }
        write(
            DEST / 'postgres-statements.json', dumps(pack(sql_facts)).encode()
        )
        provenance['consumers']['reference-inventory.json'] = [
            'tests/test_t21a_reference_lines.py'
        ]
        provenance['consumers']['postgres-statements.json'] = [
            'tests/test_t33_write_guard.py',
            'packages/structured-data-models-client/tests_test_ops/test_the_postgres_provider.py',
        ]
        write(
            DEST / 'reference-observations.jsonl',
            ''.join(
                json.dumps(row, ensure_ascii=True) + '\n'
                for row in observations
            ).encode(),
        )
        provenance['consumers']['reference-observations.jsonl'] = consumers

        rust = (
            ROOT / 'rust/crates/sdm-testkit/tests/t36_snowflake.rs'
        ).read_text()
        literals = set(re.findall(r'"([^"\n]+)"', rust))
        python_consumer = 'packages/structured-data-models-client/tests/test_snowflake_from_source.py'
        python_literals = {
            node.value
            for node in ast.walk(
                ast.parse((ROOT / python_consumer).read_text())
            )
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        literals |= python_literals
        cases = {}
        public = {}
        for prefix in (
            'contract/captures/snowflake/',
            'contract/captures/s2/snowflake/',
        ):
            for path in sorted(members):
                if path.startswith(prefix) and path.endswith('/capture.json'):
                    case = path[len(prefix) :].split('/')[0]
                    if case not in literals or case in cases:
                        continue
                    capture = json.loads(read(path))
                    statements = []
                    for statement in capture['statements']:
                        kept = {
                            k: statement[k]
                            for k in ('text', 'parameters', 'result')
                            if k in statement
                        }
                        if isinstance(kept.get('result'), str):
                            source = str(Path(path).parent / kept['result'])
                            body = read(source)
                            blob = sha(body) + '.arrow'
                            write(DEST / 'snowflake' / blob, body)
                            kept['result'] = blob
                        statements.append(kept)
                    cases[case] = {'statements': statements}
                    if case in python_literals:
                        observed = capture['reference']['public']
                        public[case] = {
                            key: observed[key]
                            for key in ('schema', 'payloads')
                            if key in observed
                        }
        assert cases and public, (
            'Snowflake fixture consumers must name archived cases'
        )
        write(
            DEST / 'snowflake.json',
            json.dumps(cases, indent=2).encode() + b'\n',
        )
        write(
            DEST / 'snowflake-public.json',
            dumps(
                pack(
                    [{'case': case, **facts} for case, facts in public.items()]
                )
            ).encode(),
        )
        provenance['consumers']['snowflake.json'] = [
            'rust/crates/sdm-testkit/tests/t36_snowflake.rs',
            python_consumer,
        ]
        provenance['consumers']['snowflake-public.json'] = [python_consumer]
        provenance['consumers']['vectors'] = list(VECTORS)
        write(
            DEST / 'PROVENANCE.json',
            json.dumps(provenance, indent=2, sort_keys=True).encode() + b'\n',
        )
    print(
        'verified' if verify else 'extracted',
        len(observations),
        'observations and',
        len(cases),
        'Snowflake cases',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    extract(args.archive, args.verify)


if __name__ == '__main__':
    main()
