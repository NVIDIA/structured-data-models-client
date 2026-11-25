# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print disposable developer views of declarations, never inferred pass evidence."""

import argparse
import json
import re
from pathlib import Path

from ledger import validate_developer_contracts

from contract import HERE, cells, effective_case, expectation, manifest


def artifact_identity(artifact, commit):
    source = artifact.get('execution') == 'source-equivalent'
    revision = artifact.get('source_commit' if source else 'commit')
    package = artifact.get(
        'verified_reference_jar_hash' if source else 'package_hash'
    )
    hashes = (package, artifact.get('native_hash'))
    if (
        revision != commit
        or type(artifact.get('contract')) is not int
        or any(
            not isinstance(value, str)
            or not re.fullmatch(r'[0-9a-f]{64}', value)
            for value in hashes
        )
        or (source and artifact.get('tree_clean') is not True)
    ):
        raise ValueError('receipt artifact identity missing or mismatched')
    return (
        'source-equivalent' if source else 'recorded-artifact',
        revision,
        *hashes,
        artifact['contract'],
    )


def model_identities(models):
    if (
        not isinstance(models, dict)
        or not {'tabular', 'relational', 'pretrained', 'gradients'}
        <= models.keys()
    ):
        raise ValueError('receipt model identity missing')
    safe = {}
    for name, model in models.items():
        if name == 'case_routes':
            continue
        if (
            not isinstance(model, dict)
            or not isinstance(model.get('version'), str)
            or not model['version'].strip()
            or model.get('transport')
            not in {'pretrained': ('in-process',), 'gradients': ('http',)}.get(
                name, ('nim', 'databricks', 'private-bridge')
            )
        ):
            raise ValueError('receipt model identity invalid')
        if model['transport'] == 'private-bridge' and (
            model.get('identity_basis') != 'configured-route'
            or not isinstance(model.get('route_sha256'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', model['route_sha256'])
        ):
            raise ValueError('receipt model route identity invalid')
        safe[name] = {
            key: model[key]
            for key in (
                'version',
                'transport',
                'identity_basis',
                'route_sha256',
            )
            if key in model
        }
    return safe


def execution_evidence(data, commit, observations=(), api_receipts=()):
    """Reuse the maintained comparator; declarations never become pass evidence."""
    from compare import assess

    from contract import records

    if not re.fullmatch(r'[0-9a-f]{40}', commit or ''):
        raise ValueError(
            'execution evidence requires an exact candidate commit'
        )
    result = {'commit': commit, 'scenarios': [], 'api_tests': []}
    identities = {}
    seen = set()
    for path, live in observations:
        observed = records(path)
        for cell, record in observed.items():
            if (cell, live) in seen:
                raise ValueError(
                    'duplicate execution observation in one evidence mode'
                )
            seen.add((cell, live))
            artifact = record.get('artifact', {})
            identity = artifact_identity(artifact, commit)
            previous = identities.setdefault(cell[3], identity[1:])
            if previous != identity[1:]:
                raise ValueError(
                    'execution view mixes artifacts for one language'
                )
        report = assess(
            data,
            observed,
            live=live,
            selected=set(observed),
            candidate_commit=commit,
        )
        for row in report['cells']:
            key = tuple(row['cell'])
            if key not in observed:
                continue
            record = observed[key]
            status = (
                ('live passed' if live else 'local passed')
                if row['status'] == 'passed'
                else row['status']
            )
            if row['status'] == 'incomplete':
                status = 'unavailable'
            endpoint = record.get('endpoint') or {}
            result['scenarios'].append(
                {
                    'cell': key,
                    'status': status,
                    'reason': row.get('reason', ''),
                    'artifact': record['artifact'],
                    'model': {
                        key: endpoint[key]
                        for key in ('model', 'version', 'transport')
                        if key in endpoint
                    },
                }
            )
    required = {
        (entry['api_contract'], entry['test'])
        for entry in data.get('live_api_tests', [])
    }
    for path in api_receipts:
        receipt = json.loads(path.read_text())
        if receipt.get('candidate_commit') != commit:
            raise ValueError('live API receipt candidate differs')
        if {
            (entry['api_contract'], entry['test'])
            for entry in receipt.get('requirements', [])
        } != required:
            raise ValueError(
                'live API receipt requirements differ from manifest'
            )
        for group, test in sorted(required):
            language = 'java' if '#' in test else 'python'
            evidence = receipt.get('results', {}).get(language, {})
            artifact = receipt.get('artifacts', {}).get(language, {})
            models = receipt.get('models', {})
            safe_models = {}
            if evidence.get('status') == 'passed':
                identity = artifact_identity(artifact, commit)
                safe_models = model_identities(models)
                previous = identities.setdefault(language, identity[1:])
                if previous != identity[1:]:
                    raise ValueError(
                        'live API receipt artifact differs from observations'
                    )
            completed = evidence.get('completed_tests', [])
            observed = any(
                node == test or node.startswith(test + '[')
                for node in completed
            )
            passed = (
                evidence.get('status') == 'passed'
                and evidence.get('exit_code') == 0
                and test in evidence.get('required_methods', [])
                and observed
            )
            result['api_tests'].append(
                {
                    'group': group,
                    'test': test,
                    'status': 'live passed' if passed else 'unavailable',
                    'reason': ''
                    if passed
                    else 'No matching completed test with verified identity; aggregate pass is insufficient.',
                    'artifact': artifact,
                    'models': safe_models,
                }
            )
    return result


def identity_text(artifact):
    return ', '.join(
        f'{key}={artifact[key]}'
        for key in (
            'execution',
            'source_commit',
            'commit',
            'package_hash',
            'verified_reference_jar_hash',
            'native_hash',
            'contract',
        )
        if key in artifact
    )


def render(data, evidence=None):
    lines = [
        '# Developer contract views',
        '',
        'These are declarations, not execution evidence.',
        'Execution status uses only validated supplied receipts; absent evidence is untested.',
        'Recorded model identities retain their receipt meaning; a configured route is not independent server attestation.',
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
        scope = group['scope']
        cases = [
            case
            for case in data['scenarios']
            if case['id'] in scope['scenarios']['refs']
        ]
        if cases:
            for axis in ('models', 'connectors', 'task'):
                values = sorted(
                    {
                        value
                        for case in cases
                        for value in (
                            [case[axis]]
                            if isinstance(case[axis], str)
                            else case[axis]
                        )
                    }
                )
                lines.append(
                    f'- {axis}: {", ".join(values)} (per referenced scenarios, not their Cartesian product)'
                )
        for axis, entry in scope.items():
            lines.append(
                f'- {axis}: {", ".join(entry["refs"]) or entry["reason"]}'
            )
        for name in scope['input_paths']['refs']:
            lines.append(
                f'  - {name}: {json.dumps(data["input_paths"][name], sort_keys=True)}'
            )
        for record in data['type_contracts']:
            if record['id'] in scope['type_contracts']['refs']:
                types = record.get(
                    'semantic_types',
                    record.get('required_vectors', record.get('families', [])),
                )
                lines.append(
                    f'  - {record["id"]}: {len(types)} declared types/families; {record["boundary"]}'
                )
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
        '| Scenario / connector / model / language | Declared support | Execution evidence | Artifact / model identity |',
        '|---|---|---|---|',
    ]
    indexed = {}
    for row in (evidence or {}).get('scenarios', []):
        indexed.setdefault(tuple(row['cell']), []).append(row)
    scenarios = {case['id']: case for case in data['scenarios']}
    for cell in cells(data):
        case, connector, model, language = cell
        declared = expectation(
            effective_case(scenarios[case], connector), model, language
        )['status']
        rows = indexed.get(tuple(cell), [])
        status = '; '.join(row['status'] for row in rows) or 'untested'
        identity = '; '.join(
            identity_text(row['artifact'])
            + ' / '
            + json.dumps(row['model'], sort_keys=True)
            for row in rows
        )
        lines.append(
            f'| {" / ".join(cell)} | {declared} | {status} | {identity} |'
        )
    if evidence:
        lines += [
            '',
            f'Execution receipts identify commit `{evidence["commit"]}`; they do not relabel a different source/artifact revision.',
            '',
            '## Registered live API outcomes',
            '',
        ]
        for row in evidence['api_tests']:
            lines.append(
                f'- {row["group"]} / `{row["test"]}`: {row["status"]}. {row["reason"]}'
            )
            if row['artifact']:
                lines.append('  - ' + identity_text(row['artifact']))
            if row['models']:
                lines.append(
                    '  - Models: ' + json.dumps(row['models'], sort_keys=True)
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
    parser.add_argument('--candidate-commit')
    parser.add_argument(
        '--local-observations', action='append', type=Path, default=[]
    )
    parser.add_argument(
        '--live-observations', action='append', type=Path, default=[]
    )
    parser.add_argument(
        '--live-api-receipt', action='append', type=Path, default=[]
    )
    args = parser.parse_args()
    from api_contracts import validate

    data = manifest()
    validate(data)
    validate_developer_contracts(data, HERE.parents[1])
    supplied = [(path, False) for path in args.local_observations] + [
        (path, True) for path in args.live_observations
    ]
    evidence = (
        execution_evidence(
            data, args.candidate_commit, supplied, args.live_api_receipt
        )
        if supplied or args.live_api_receipt
        else None
    )
    output = render(data, evidence)
    if args.out:
        args.out.write_text(output)
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
