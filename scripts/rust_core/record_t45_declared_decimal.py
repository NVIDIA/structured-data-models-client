# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write exact public Decimal materialize observations from frozen receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REFERENCE = '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'
INPUTS = {
    'decimal-meanings-r1-310.json': '42438fe84e4b0e98f2015ff064bdbe8e53b7d3f30a7fdb66b6487d28d157c6da',
    'decimal-meanings-r1-312.json': 'ab925eda539252a52583646d833061bb3a4b0b008b3648b5477866139ac0f11d',
    'decimal-meanings-r2-310.json': '92e8790c283a77cbd76a595c20ce2bdfadbfd134fcab04e37ea1c43709db4dfb',
    'decimal-meanings-r2-312.json': '253db9c79a3bff78e6b847c61b6acb65d46a329b82f21483eb99cc8750d7b240',
}
NATIVE = {'float', 'int', 'str', 'repr', 'effect'}


def payload_digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def write(receipts: Path, output: Path):
    documents = {}
    for name, expected in INPUTS.items():
        source = receipts / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f'{name}: measured receipt changed')
        document = json.loads(source.read_text())
        if document['reference'] != REFERENCE:
            raise ValueError(f'{name}: wrong frozen reference')
        documents[name] = document
    for cell in ('310', '312'):
        original = documents[f'decimal-meanings-r1-{cell}.json']
        rows = original['records'] + [
            r
            for r in documents[f'decimal-meanings-r2-{cell}.json']['records']
            if r['observed']
        ]
        cases = []
        for row in rows:
            if 'error' in row or len(row['calls']) != 2:
                raise ValueError('setup error or missing public call')
            calls = []
            for call in row['calls']:
                observed = {
                    'outcome': call['outcome'],
                    'native_calls': [
                        event['event']
                        for event in call['trace']
                        if event['event'] in NATIVE
                    ],
                }
                if call['outcome'] == 'success':
                    observed.update(
                        payload_sha256=payload_digest(call['payloads']),
                        context_rows=call['context_rows'],
                        predict_rows=call['predict_rows'],
                    )
                else:
                    observed['error'] = {
                        name: call['error'][name]
                        for name in ('type', 'code', 'message')
                    }
                calls.append(observed)
            cases.append(
                {
                    **{
                        name: row[name]
                        for name in ('storage', 'shape', 'declared', 'mutation')
                    },
                    'expected': {
                        'calls': calls,
                        'warnings': row['warnings'],
                        'effect_reached': row['effect_reached'],
                    },
                }
            )
        fixture = {
            'reference': REFERENCE,
            'source_sha256': INPUTS,
            'pandas': original['pandas'],
            'pyarrow': original['pyarrow'],
            'cases': cases,
        }
        pandas = int(original['pandas'].split('.')[0])
        path = output / f't45_declared_decimal_pandas{pandas}_reference.json'
        path.write_text(json.dumps(fixture, indent=2, ensure_ascii=True) + '\n')
        print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument(
        '--out',
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / 'packages/structured-data-models-client/tests',
    )
    args = parser.parse_args()
    write(args.receipts, args.out)


if __name__ == '__main__':
    main()
