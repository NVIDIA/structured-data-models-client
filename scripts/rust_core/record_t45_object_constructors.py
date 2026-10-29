# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export actual public native object constructor receipts without SDK imports."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

REFERENCE = '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'
PRODUCERS = {
    'probe_r1.py': '09bf7c2d949ff083ce5b9b2fd903e94f1e93d90420e12b81f45aff432086477c',
    'probe_r2.py': '08d0c3a03f32732a73829a8db48e2e654ba3ab39f5b43121174807e67b77bb3d',
}
DECIMAL_RECEIPTS = {
    '310': {
        'frozen-object-decimal-310.json': '022766e60c16a1fbe6daf2684ece5815746aeba494f9d0ea3c14c4943c675159',
        'frozen-object-decimal-r2-310.json': 'eff036bcd8c66eed3350fb4938e1920763a94205a4cf7275bc0e3f4f29b6c02a',
    },
    '312': {
        'frozen-object-decimal-312.json': 'f80641b5d96105f488ca07228d864ea385c0cbdab78d69ca57eae89ec6f1ed4a',
        'frozen-object-decimal-r2-312.json': '7adc54e5214bb87cd93d60533f5e05cd6e4ef92c9d8a6612011818815278b2bf',
    },
}
GENERIC_PRODUCER = (
    'a6aa5dbdd33d4390c1a8cb65f824a0bc731da03d89aa7378ced7c9f8a6e24225'
)
GENERIC_RECEIPTS = {
    '310': 'b4e5925b92000e4c3ca610f28c21a7d3c50d4beb107cc9cdea05636b280943ab',
    '312': '7de410ee5348253f8e9726e2474826b0f5bd8f86d1c8c48c1b73964bf47d1305',
}


def verified(path: Path, expected: str) -> bytes:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'{path.name}: executed source or receipt changed')
    return raw


def inputs(source: bytes) -> dict:
    tree = ast.parse(source)
    (assignment,) = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == 'CASES'
            for target in node.targets
        )
    ]
    return {
        name: {'values': values, 'mode': mode, 'key_shaped': key}
        for name, values, mode, key in ast.literal_eval(assignment.value)
    }


def trace(record: dict) -> list[dict]:
    # Sites and stack frames are passive observer implementation labels.
    # Every public callback method and its ordinal remain exact.
    return [
        {field: step[field] for field in ('method', 'call')}
        for step in record['trace']
    ]


def record(decimal: Path, generic: Path) -> dict:
    sources = {
        name: verified(decimal / name, expected)
        for name, expected in PRODUCERS.items()
    }
    specifications = inputs(sources['probe_r1.py'])
    repeated = inputs(sources['probe_r2.py'])
    assert all(
        specifications[name] == value for name, value in repeated.items()
    )
    verified(generic / 'probe_r1.py', GENERIC_PRODUCER)
    result = {
        'reference': REFERENCE,
        'producers': {**PRODUCERS, 'generic-probe_r1.py': GENERIC_PRODUCER},
        'decimal_receipts': DECIMAL_RECEIPTS,
        'generic_receipts': GENERIC_RECEIPTS,
        'cells': {},
    }
    for cell, receipts in DECIMAL_RECEIPTS.items():
        documents = [
            json.loads(verified(decimal / name, expected))
            for name, expected in receipts.items()
        ]
        original, continuation = documents
        assert (
            original['bootstrap']['commit']
            == continuation['bootstrap']['commit']
            == REFERENCE
        )
        assert (
            len(original['records']) == 18 and len(continuation['records']) == 4
        )
        assert original['producer_sha256'] == PRODUCERS['probe_r1.py']
        assert continuation['producer_sha256'] == PRODUCERS['probe_r2.py']
        for observed in continuation['records']:
            (first,) = [
                item
                for item in original['records']
                if item['case'] == observed['case']
            ]
            assert trace(first) == trace(observed)
            assert first.get('table') == observed.get('table')
            assert first['warnings'] == observed['warnings']
        cases = []
        for observed in original['records']:
            expected = {
                key: observed[key]
                for key in ('table', 'warnings')
                if key in observed
            }
            if 'error' in observed:
                error = observed['error']
                # All recorded native failures are direct and unchained.
                assert (
                    'The above exception was the direct cause'
                    not in error['traceback']
                )
                assert (
                    'During handling of the above exception'
                    not in error['traceback']
                )
                expected['error'] = {
                    key: error[key] for key in ('type', 'code', 'message')
                }
            expected['trace'] = trace(observed)
            cases.append(
                {
                    'case': observed['case'],
                    **specifications[observed['case']],
                    'expected': expected,
                }
            )
        name = f'constructor-public-r1-{cell}.json'
        native = json.loads(verified(generic / name, GENERIC_RECEIPTS[cell]))
        assert native['reference'] == REFERENCE and native['cell'] == cell
        assert native['producer_sha256'] == GENERIC_PRODUCER
        assert native['network_attempts'] == []
        assert {item['kind'] for item in native['records']} == {
            'object',
            'category',
        }
        assert all('error' not in item for item in native['records'])
        result['cells']['pandas2' if cell == '310' else 'pandas3'] = {
            'runtime': original['runtime'],
            'decimals': cases,
            'strings': native['records'],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decimals', type=Path, required=True)
    parser.add_argument('--generic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(
            record(args.decimals, args.generic), indent=2, ensure_ascii=True
        )
        + '\n'
    )


if __name__ == '__main__':
    main()
