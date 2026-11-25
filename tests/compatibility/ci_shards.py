# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require every connector artifact before the existing full-matrix comparator."""

import argparse
import json
from pathlib import Path

from contract import manifest


def combine(root, connectors, commit, run_id, out):
    expected = {'live-shard-' + connector for connector in connectors}
    if {path.name for path in root.iterdir()} != expected:
        raise ValueError('Missing or unexpected connector artifacts')
    records = []
    identity = None
    for connector in connectors:
        directory = root / ('live-shard-' + connector)
        receipt = json.loads((directory / 'artifacts.json').read_text())
        if (
            receipt['commit'] != commit
            or receipt['ci_run_id'] != int(run_id)
            or receipt['build'] != 'production'
        ):
            raise ValueError('Shard did not use the required production CI run')
        if identity is not None and receipt != identity:
            raise ValueError('Shards did not use identical artifacts')
        identity = receipt
        for language in ('python', 'java'):
            path = directory / (language + '.jsonl')
            lines = path.read_text().splitlines()
            if not lines:
                raise ValueError('Missing language observations')
            for line in lines:
                row = json.loads(line)
                if (
                    not receipt.get(language)
                    or row.get('artifact') != receipt[language]
                ):
                    raise ValueError(
                        'Observation artifact differs from verified CI artifact'
                    )
                if row['connector'] != connector or row['language'] != language:
                    raise ValueError(
                        'Observation is outside its declared shard'
                    )
            records.extend(lines)
    out.write_text('\n'.join(records) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--commit')
    parser.add_argument('--run-id')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--matrix', action='store_true')
    args = parser.parse_args()
    connectors = manifest()['connectors']
    if args.matrix:
        print(json.dumps({'connector': connectors}))
    else:
        combine(args.root, connectors, args.commit, args.run_id, args.out)


if __name__ == '__main__':
    main()
