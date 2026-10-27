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
                        'case': item['case'],
                        'outcome': outcome,
                        'warnings': item['warnings'],
                    }
                )
        result['cells']['pandas2' if cell == '310' else 'pandas3'] = records
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
