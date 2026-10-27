# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy actual public index Arrow facts and selected/expected ordinals for tests.

This canonical authoring tool imports no candidate or native Python module.
Expected orders are copied only from the pinned actual public receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.receipt.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if (
        digest
        != '7dc2b05a227f49ce5176c0acf2f01f1fea9a4157f81082a3f25edf45eb9fa655'
    ):
        raise ValueError(
            'The actual reference receipt does not match its root-inspected pin'
        )
    receipt = json.loads(raw)
    if (
        receipt['versions']['numpy'] != '2.5.3'
        or not receipt['numpy_cpu_features']['X86_V3']
        or receipt['numpy_cpu_features']['X86_V4']
    ):
        raise ValueError(
            'The pinned reference process is not the established NumPy253 X86_V3 producer'
        )
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for case in receipt['results']:
        name = case['name']
        included = (
            name.startswith(
                (
                    'primitive-int',
                    'primitive-uint',
                    'nullable-int',
                    'nullable-uint',
                )
            )
            or name
            in {
                'primitive-float32',
                'primitive-float64',
                'nullable-float32',
                'nullable-float64',
                'primitive-bool',
                'nullable-bool',
                'range',
                'datetime',
                'datetime-zone',
                'timedelta',
                'period',
                'all-equal-monotonic-shortcut',
                'all-missing-float',
            }
            or name.startswith('categorical-')
        )
        if not included:
            continue
        facts = case['input_facts']
        copied = []
        for kind, item in [
            ('values', facts['attachment']),
            *(
                [('dictionary', facts['categories']['attachment'])]
                if name.startswith('categorical-')
                else []
            ),
        ]:
            path = args.receipt.parent / item['path']
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError(
                    f'Actual Arrow attachment digest differs: {path}'
                )
            destination = args.output / path.name
            shutil.copyfile(path, destination)
            copied.append(
                {
                    'role': kind,
                    'path': destination.name,
                    'sha256': item['sha256'],
                }
            )
        call = case['calls'][0]
        if (
            call['outcome'] != 'captured_wire'
            or call['wire_ids'] != call['context_ids']
        ):
            raise ValueError(
                f'Case has no exact successful public wire observation: {name}'
            )
        cases.append(
            {
                'name': name,
                'source_rows': case['n'],
                'layout': 'categorical'
                if name.startswith('categorical-')
                else 'range'
                if name == 'range'
                else 'temporal'
                if name in {'datetime', 'datetime-zone', 'timedelta', 'period'}
                else 'numpy',
                'missing': 'validity'
                if name.startswith('nullable-')
                else 'nan_or_validity',
                'dictionary_rows': facts.get('categories', {}).get('length'),
                'ordered': facts.get('ordered'),
                'code_storage': facts.get('code_storage'),
                'attachments': copied,
                'selected_original_rows': call['sample_ids_before_sort'],
                'expected_context_original_rows': call['context_ids'],
            }
        )
    result = {
        '_generated': 'scripts/rust_core/record_index_context_fixtures.py; do not edit',
        'reference_sha': receipt['reference_sha'],
        'receipt_sha256': digest,
        'runtime': receipt['versions'],
        'source_profile': 'numpy_253_x86_v3_gnu14',
        'cases': cases,
    }
    (args.output / 'cases.json').write_text(json.dumps(result, indent=2) + '\n')
    print(
        f'Authored {len(cases)} actual public index fixtures with immutable typed Arrow inputs'
    )


if __name__ == '__main__':
    main()
