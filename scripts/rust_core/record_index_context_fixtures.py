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


PROFILE_PINS = {
    'cp310': {
        'receipt_sha256': '81d4ab496e483771561cfd1194d10dc06ec2e672f9c1b840be46f64ee6e8fe02',
        'versions': {'pandas': '2.3.3', 'numpy': '2.2.6', 'pyarrow': '25.0.1'},
        'native_sha256': 'c634eaf05367af02b815cf3d2020fbeee9b47426af1fe26cf3189e4798932dc5',
        'source_profile': 'numpy_226_avx2_intel5_entry_shared_gnu',
    },
    'cp312': {
        'receipt_sha256': 'bdad19b7bfd823f8be640e61ad49a108095b7a450db6ef0c4086f94a75576efc',
        'versions': {'pandas': '3.0.6', 'numpy': '2.5.3', 'pyarrow': '25.0.1'},
        'native_sha256': 'ceec5ba13ab88318632307fdd2df74f317fa293ce77be559c4298ddd1f4306b0',
        'source_profile': 'numpy_253_x86_v3_gnu14',
    },
}


def write_profile_cases(receipt_paths: list[Path], output: Path):
    cases, producers = [], []
    for receipt_path in receipt_paths:
        raw = receipt_path.read_bytes()
        receipt = json.loads(raw)
        cell = receipt['cell']
        pin = PROFILE_PINS[cell]
        if (
            hashlib.sha256(raw).hexdigest() != pin['receipt_sha256']
            or receipt['versions'] != pin['versions']
        ):
            raise ValueError(
                f'Public profile receipt differs from its actual root-inspected pin: {cell}'
            )
        features = receipt['numpy_cpu_features']
        if (
            not features['AVX2']
            or features['AVX512F']
            or features['AVX512_SKX']
        ):
            raise ValueError(
                f'Actual process does not establish the required four-lane native target: {cell}'
            )
        native = [
            item
            for item in receipt['expected_source_identity']['files']
            if '/numpy/_core/_multiarray_umath.' in item['path']
        ]
        if len(native) != 1 or native[0]['sha256'] != pin['native_sha256']:
            raise ValueError(f'Actual source/native producer differs: {cell}')
        producers.append(
            {
                'cell': cell,
                'receipt_sha256': pin['receipt_sha256'],
                'versions': receipt['versions'],
                'native_sha256': pin['native_sha256'],
                'source_profile': pin['source_profile'],
            }
        )
        for case in receipt['results']:
            name = case['name']
            if name not in {
                'profile-sorted-nonmissing294-float64',
                'profile-sorted-nonmissing294-category32',
            }:
                raise ValueError(f'Unexpected public profile case: {name}')
            facts, call = case['input_facts'], case['calls'][0]
            if (
                not call['fixture_control_holds']
                or call['outcome'] != 'captured_wire'
                or call['context_ids'] != call['wire_ids']
            ):
                raise ValueError(
                    f'No exact successful public profile observation: {cell}/{name}'
                )
            categorical = name.endswith('category32')
            descriptors = [('values', facts['attachment'])]
            if categorical:
                descriptors.append(
                    ('dictionary', facts['categories']['attachment'])
                )
            copied = []
            for role, item in descriptors:
                path = receipt_path.parent / item['path']
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != item['sha256']:
                    raise ValueError(
                        f'Original profile Arrow attachment differs: {path}'
                    )
                destination = output / f'{cell}-{path.name}'
                if destination.exists():
                    raise ValueError(
                        f'Refusing to overwrite an existing generated profile attachment: {destination}'
                    )
                destination.write_bytes(data)
                copied.append(
                    {
                        'role': role,
                        'path': destination.name,
                        'sha256': item['sha256'],
                    }
                )
            counts = call['nonmissing_kernel_control']
            cases.append(
                {
                    'name': f'{cell}/{name}',
                    'cell': cell,
                    'source_rows': case['n'],
                    'source_profile': pin['source_profile'],
                    'layout': 'categorical' if categorical else 'numpy',
                    'ordered': facts.get('ordered'),
                    'code_storage': facts.get('code_storage_diagnostic_only'),
                    'attachments': copied,
                    'missing': 'nan_or_validity',
                    'candidate_count': counts['candidate_count'],
                    'missing_count': counts['missing_count'],
                    'nonmissing_count': counts['nonmissing_count'],
                    'selected_original_rows': call['sample_ids_before_sort'],
                    'expected_context_original_rows': call['context_ids'],
                }
            )
    if (
        len(producers) != 2
        or {item['cell'] for item in producers} != {'cp310', 'cp312'}
        or len(cases) != 4
    ):
        raise ValueError(
            'Both pinned source profiles and all four actual public cases are required'
        )
    result = {
        '_generated': 'scripts/rust_core/record_index_context_fixtures.py --profile-receipt; do not edit',
        'writer_sha256': hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        'producers': producers,
        'cases': cases,
    }
    destination = output / 'profile_cases.json'
    if destination.exists():
        raise ValueError(
            'Profile cases already exist; preserve the previous canonical generation'
        )
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(
        'Authored four actual source-profile cases with original400-row Arrow inputs and actual300-row orders'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--receipt', type=Path)
    mode.add_argument('--profile-receipt', type=Path, action='append')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.profile_receipt:
        args.output.mkdir(parents=True, exist_ok=True)
        write_profile_cases(args.profile_receipt, args.output)
        return
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
