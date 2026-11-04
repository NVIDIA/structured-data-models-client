# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print disposable developer views of declarations, never inferred pass evidence."""

import argparse
import json
from pathlib import Path

from ledger import validate_developer_contracts

from contract import HERE, manifest


def render(data):
    lines = [
        '# Developer contract views',
        '',
        'These are declarations, not execution evidence.',
        'Execution status is untested in this view; use exact-artifact local/live runner receipts for outcomes.',
        'A test link is not proof of an assertion, and omitted combinations are not implicitly unsupported.',
        '',
        '## API equivalence and engine ownership',
        '',
        '| Capability | Python symbols | Java symbols | Owner | Operations / host boundary |',
        '|---|---:|---:|---|---|',
    ]
    for group in data['api_contracts']:
        counts = [
            sum(
                record['capability'] == group['id']
                for record in data['api_inventory'][language].values()
            )
            for language in ('python', 'java')
        ]
        owner = group['ownership']
        operations = ', '.join(owner['operations']) or owner['host_only_reason']
        lines.append(
            f'| {group["id"]} | {counts[0]} | {counts[1]} | {", ".join(owner["paths"])} | {operations} |'
        )
    lines += ['', '## Explicit language exceptions', '']
    for exception in data['api_exceptions']:
        lines.append(
            f'- {exception["capability"]} / {exception["language"]}: {exception["reason"]}'
        )
    lines += ['', '## Applicability and assertion links', '']
    for group in data['api_contracts']:
        lines += [f'### {group["id"]}', '']
        lines += [
            f'- {axis}: {scope}' for axis, scope in group['scope'].items()
        ]
        for language, inventory in data['api_inventory'].items():
            for symbol, record in inventory.items():
                if record['capability'] != group['id']:
                    continue
                assertions = record.get('assertions', [])
                details = '; '.join(
                    f'{item["test"]}: {item["behavior"]}' for item in assertions
                )
                lines.append(
                    f'- {language} `{symbol}`: {details or "existing group coverage; no direct symbol assertion declared"}'
                )
        lines.append('')
    lines += [
        '## Scenario execution scope',
        '',
        '| Scenario | Models | Connectors | Evidence |',
        '|---|---|---|---|',
    ]
    for case in data['scenarios']:
        lines.append(
            f'| {case["id"]} | {", ".join(case["models"])} | {", ".join(case["connectors"])} | untested in this view |'
        )
    lines += ['', '## Type boundaries', '']
    for record in data['type_contracts']:
        lines += [
            f'### {record["id"]}',
            '',
            record['boundary'],
            '',
            '```json',
            json.dumps(record, indent=2),
            '```',
            '',
        ]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out', type=Path, help='Disposable Markdown artifact; default stdout'
    )
    args = parser.parse_args()
    data = manifest()
    validate_developer_contracts(data, HERE.parents[1])
    output = render(data)
    if args.out:
        args.out.write_text(output)
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
