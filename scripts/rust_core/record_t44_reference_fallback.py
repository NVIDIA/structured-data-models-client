# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the actual frozen public native-reference fallback receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RECEIPTS = {
    '310': {
        'r1': '20987b676a5685c201cc8b8e6ca511320425f0bddd712ca577412b82c9d720a5',
        'r2': 'c6be9f68d058d3e6fc5519930dc1396bc55e8f601dca9a83cfabf3fa7bde79fa',
    },
    '312': {
        'r1': 'feb9285a1841bd6f5e5d30625535137e14aef5173eb0b203bbfa1e7cdac45dd8',
        'r2': '32b24b0290bfdb7232e23c81e9d64bb6e0b6613e58141382c5256742f41d2b0d',
    },
}
CHUNK_RECEIPTS = {
    '310': '2a5fc688bf9dafa49643bc6a00dc9aad505d3608e8748a3924924319e6c8a90d',
    '312': 'f1afe4c53b3e9de8b2cb690bfec976285e62d4dcca7d786b91b489c6d2ddb686',
}
REFERENCE = '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'


def record(receipts: Path) -> dict:
    result = {'reference': REFERENCE, 'receipts': RECEIPTS, 'cells': {}}
    for cell, versions in RECEIPTS.items():
        records = []
        for version, expected in versions.items():
            raw = (receipts / f'fallback-{version}-{cell}.json').read_bytes()
            assert hashlib.sha256(raw).hexdigest() == expected
            observed = json.loads(raw)
            assert observed['reference'] == REFERENCE
            assert observed['cell'] == cell
            assert observed['network_attempts'] == []
            for item in observed['records']:
                if version == 'r1' and item['case'] not in (
                    'whole-type-once',
                    'whole-overflow-once',
                    'batch-type-once',
                    'batch-type-both',
                ):
                    continue
                outcome = item['outcome']
                if outcome['kind'] == 'error':
                    outcome = {
                        'kind': 'error',
                        'chain': outcome['error']['chain'],
                    }
                records.append(
                    {
                        'storage': item['storage'],
                        'case': item['case'].replace('batch-', 'label-', 1)
                        if version == 'r1' and item['case'].startswith('batch-')
                        else item['case'],
                        'outcome': outcome,
                        'warnings': item['warnings'],
                    }
                )
        result['cells']['pandas2' if cell == '310' else 'pandas3'] = records
    result['chunk_receipts'] = CHUNK_RECEIPTS
    result['chunk_cells'] = {}
    for cell, expected in CHUNK_RECEIPTS.items():
        raw = (receipts / f'fallback-r4-{cell}.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        observed = json.loads(raw)
        assert observed['reference'] == REFERENCE and observed['cell'] == cell
        assert observed['network_attempts'] == []
        records = []
        for item in observed['records']:
            outcome = item['outcome']
            if outcome['kind'] == 'error':
                outcome = {'kind': 'error', 'chain': outcome['error']['chain']}
            records.append(
                {
                    'shape': item['shape'],
                    'second_label_fault': item['second_label_fault'],
                    'outcome': outcome,
                    'warnings': item['warnings'],
                    'label_cast_rows': [
                        step['rows']
                        for step in item['trace']
                        if step['site'] == 'labels'
                        and step['target'] == "<class 'str'>"
                    ],
                    'native_faults': len(item['faults']),
                }
            )
        result['chunk_cells']['pandas2' if cell == '310' else 'pandas3'] = (
            records
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipts', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(record(args.receipts), indent=2, ensure_ascii=True) + '\n'
    )


if __name__ == '__main__':
    main()
