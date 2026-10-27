# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write public LINK fixtures from immutable, executed frozen receipts.

This tool only reads JSON and producer bytes.
It never imports the SDK, runs a case, or changes a measured result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REFERENCE = '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'
PRODUCERS = {
    'link_probe_r1.py': '32e0b6f8f97ef637ec25141c57804d6224f9b5eae515214cc0ae86029f2c8ef8',
    'link_probe_r2.py': '7a2687bbc9f7808985b4f233e6d999971029342ece068e7ae7a7e98a647364c3',
}
RECEIPTS = {
    'frozen-link-r1-310.json': '72d77a30f61be17ee42d3cdf23161ee31d20a9dfa83d20fd54091976016e19ee',
    'frozen-link-r1-312.json': '672acb7e8f275c11df89e5e3efa597365354070de52bcca6ec1342ad3c463eb3',
    'frozen-link-r2-310.json': '2e391153d7ebbbd8fd8797cd5ee8d81c39de90f87e8450cad6abe67c5c8f12e1',
    'frozen-link-r2-312.json': '02094391d36ead61902d5e9e5377423699f8b595e0cdc273197328dcc299048e',
}
ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(receipts: Path, output: Path) -> None:
    sources = {}
    for name, expected in PRODUCERS.items():
        actual = digest(receipts / name)
        if actual != expected:
            raise ValueError(f'{name}: producer changed')
        sources[name] = actual
    for cell in ('310', '312'):
        documents = []
        for revision in ('r1', 'r2'):
            name = f'frozen-link-{revision}-{cell}.json'
            actual = digest(receipts / name)
            if actual != RECEIPTS[name]:
                raise ValueError(f'{name}: measured receipt changed')
            sources[name] = actual
            document = json.loads((receipts / name).read_text())
            if document['reference'] != REFERENCE or document['cell'] != cell:
                raise ValueError(f'{name}: wrong reference or cell')
            documents.append(document)
        cases = []
        for document in documents:
            for record in document['records']:
                expected = {
                    key: record[key]
                    for key in (
                        'outcome',
                        'stage',
                        'payloads',
                        'inspect',
                        'result',
                        'merged',
                        'warnings',
                        'sent',
                    )
                    if key in record
                }
                if 'error' in record:
                    expected['error'] = {
                        key: record['error'][key]
                        for key in ('type', 'module', 'code', 'message')
                    }
                cases.append(
                    {
                        'kind': record.get('kind', 'int64'),
                        'shape': record['shape'],
                        'expected': expected,
                    }
                )
        if len(cases) != 32:
            raise ValueError('a frozen case is missing')
        original = documents[0]
        pandas = int(original['pandas'].split('.')[0])
        fixture = {
            'reference': REFERENCE,
            'numpy': original['numpy'],
            'pandas': original['pandas'],
            'pyarrow': original['pyarrow'],
            'source_sha256': dict(sources),
            'explicit_shapes': [
                record['shape'] for record in documents[1]['records']
            ],
            'cases': cases,
        }
        path = output / f't45_frames_pandas{pandas}_reference.json'
        path.write_text(json.dumps(fixture, indent=2, ensure_ascii=True) + '\n')
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', required=True, type=Path)
    parser.add_argument(
        '--out',
        type=Path,
        default=ROOT / 'packages/structured-data-models-client/tests',
    )
    arguments = parser.parse_args()
    write(arguments.receipts, arguments.out)


if __name__ == '__main__':
    main()
