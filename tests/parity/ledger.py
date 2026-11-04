# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inventory and change checks for the existing compatibility manifest."""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path


def python_symbols(snapshot):
    """Include exported objects and their public members, not only __all__."""
    symbols = set()
    for module, surface in snapshot.items():
        for name in surface['__all__']:
            qualified = f'{module}.{name}'
            symbols.add(qualified)
            member = surface['members'][name]
            for attribute in member.get('public_attributes', []):
                symbols.add(f'{qualified}.{attribute}')
            for method in member.get('public_methods', {}):
                symbols.add(f'{qualified}.{method}')
    return symbols


def python_declarations(snapshot):
    """Compare callable signatures without treating a new member as a class edit."""
    declarations = {}
    for module, surface in snapshot.items():
        for name in surface['__all__']:
            symbol = f'{module}.{name}'
            member = surface['members'][name]
            declarations[symbol] = {
                key: value
                for key, value in member.items()
                if key not in {'public_attributes', 'public_methods'}
            }
            for attribute in member.get('public_attributes', []):
                declarations[f'{symbol}.{attribute}'] = 'attribute'
            for method, signature in member.get('public_methods', {}).items():
                declarations[f'{symbol}.{method}'] = signature
    return declarations


def current_operations(root):
    """The generated locked schema is the operation inventory."""
    directory = root / 'rust/crates/sdm-contract/schema'
    entries = [
        line.split()
        for line in (directory / 'contract.lock').read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]
    current = max(entries, key=lambda entry: int(entry[0]))
    schema = json.loads((directory / current[2]).read_text())
    return {
        variant['properties']['op']['const']
        for variant in schema['$defs']['Operation']['oneOf']
    }


def covered_test(node, required):
    """A selected file/class includes its explicitly named child tests."""
    return any(
        node == parent
        or node.startswith(parent + '::')
        or (parent.endswith('.java') and node.startswith(parent + '#'))
        for parent in required
    )


def validate_assertion_node(node, required, root):
    if not covered_test(node, required):
        raise ValueError('behavior assertion needs a group test: ' + node)
    if '::' in node:
        path, *parts = node.split('::')
        scope = ast.parse((root / path).read_text()).body
        for part in parts:
            found = [
                item for item in scope if getattr(item, 'name', None) == part
            ]
            if len(found) != 1:
                raise ValueError(
                    'behavior assertion names a missing test: ' + node
                )
            scope = getattr(found[0], 'body', [])
        if not parts[-1].startswith('test_') or not isinstance(
            found[0], (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            raise ValueError(
                'behavior assertion must name a collected test: ' + node
            )
        if any(not part.startswith('Test') for part in parts[:-1]):
            raise ValueError(
                'behavior assertion must name a collected test class: ' + node
            )


def validate_developer_contracts(data, root):
    """Ownership is about engine boundaries, not stability of internal Rust items."""
    operations = current_operations(root)
    for group in data['api_contracts']:
        name = group['id']
        ownership = group.get('ownership', {})
        if not ownership.get('paths') or any(
            not (root / path).exists() for path in ownership['paths']
        ):
            raise ValueError(f'API ownership missing or stale: {name}')
        linked = ownership.get('operations', [])
        if not set(linked) <= operations:
            raise ValueError(f'unknown native operation: {name}')
        if not linked and not ownership.get('host_only_reason', '').strip():
            raise ValueError(
                f'API needs operations or host-only reason: {name}'
            )
        scope = group.get('scope', {})
        allowed_scenarios = {
            required['scenario']
            for capability in data['capabilities']
            if name in capability['api_contracts']
            for required in capability['requirements']
        }
        scenarios = {case['id']: case for case in data['scenarios']}
        for axis, declarations in (
            ('scenarios', scenarios),
            ('input_paths', data['input_paths']),
            (
                'type_contracts',
                {item['id']: item for item in data['type_contracts']},
            ),
        ):
            entry = scope.get(axis, {})
            refs = entry.get('refs')
            if not isinstance(refs, list) or len(refs) != len(set(refs)):
                raise ValueError(
                    f'API applicability missing or duplicate: {name}/{axis}'
                )
            if not set(refs) <= declarations.keys():
                raise ValueError(f'stale scope reference: {name}/{axis}')
            if not refs and not entry.get('reason', '').strip():
                raise ValueError(f'empty scope needs a reason: {name}/{axis}')
        if set(scope['scenarios']['refs']) != allowed_scenarios:
            raise ValueError(f'scope scenario linkage differs: {name}')
        for record in data['type_contracts']:
            if (
                record['id'] in scope['type_contracts']['refs']
                and record.get('scenarios')
                and not set(record['scenarios']) & allowed_scenarios
            ):
                raise ValueError(
                    f'type journey not linked to group scenarios: {name}'
                )
        for route in scope['input_paths']['refs']:
            declaration = data['input_paths'][route]
            if not any(
                set(declaration.get(language, []))
                & (
                    set(group['apis'].get(language, []))
                    | {
                        api
                        for case in scope['scenarios']['refs']
                        for api in scenarios[case]['apis'].get(language, [])
                    }
                )
                for language in data['languages']
            ):
                raise ValueError(
                    f'scope input route not linked to APIs: {name}/{route}'
                )
        for language in data['languages']:
            if language not in group.get('apis', {}):
                raise ValueError(
                    f'API language applicability missing: {name}/{language}'
                )
    groups = {group['id']: group for group in data['api_contracts']}
    for language, inventory in data['api_inventory'].items():
        for symbol, record in inventory.items():
            for assertion in record.get('assertions', []):
                group = groups[record['capability']]
                validate_assertion_node(
                    assertion.get('test', ''),
                    group['tests'].get(language, []),
                    root,
                )
                if not assertion.get('behavior', '').strip():
                    raise ValueError(
                        f'behavior assertion needs a description: {symbol}'
                    )


def validate_changed_symbols(
    before, after, old_python, new_python, *, introduced_java=False
):
    """Existing group coverage is not retroactively called direct symbol coverage."""
    old = python_declarations(old_python)
    new = python_declarations(new_python)
    changed = {
        'python': {symbol for symbol in new if old.get(symbol) != new[symbol]},
        'java': set()
        if introduced_java
        else set(after['api_inventory']['java'])
        - set(before['api_inventory']['java']),
    }
    for language, symbols in changed.items():
        for symbol in symbols:
            record = after['api_inventory'][language].get(symbol, {})
            if not record.get('assertions'):
                raise ValueError(
                    f'changed API needs explicit behavior assertions: {language}/{symbol}'
                )


def validate_introduction(
    data, old_python, new_python, base_commit, *, baseline_has_java
):
    """Compare actual baseline declarations without inventing a historical ledger."""
    if baseline_has_java:
        raise ValueError(
            'first adoption needs a reviewed Java baseline inventory; baseline Java is not an introduction'
        )
    decisions = {
        item['record']: item
        for item in data.get('contract_changes', [])
        if item.get('base_commit') == base_commit
        and item.get('reason', '').strip()
    }
    if 'api_inventory/java' not in decisions:
        raise ValueError(
            'Java introduction needs an explicit baseline rationale'
        )
    removed = python_symbols(old_python) - python_symbols(new_python)
    for symbol in removed:
        if 'api_inventory/python/' + symbol not in decisions:
            raise ValueError(
                'baseline declaration removal needs a reviewed reason: '
                + symbol
            )
    for symbol in removed:
        decision = decisions['api_inventory/python/' + symbol]
        if decision.get('retained_test'):
            validate_assertion_node(
                decision['retained_test'],
                [
                    test
                    for group in data['api_contracts']
                    for test in group['tests'].get('python', [])
                ],
                Path(__file__).resolve().parents[2],
            )
    validate_changed_symbols(
        {'api_inventory': {'java': {}}},
        data,
        old_python,
        new_python,
        introduced_java=True,
    )


def validate_api_inventory(actual, mappings):
    """Require explicit per-symbol mappings, without wildcard escape hatches."""
    declared = set(mappings)
    missing = set(actual) - declared
    stale = declared - set(actual)
    if missing or stale:
        raise ValueError(
            f'API mappings differ: missing={sorted(missing)}, stale={sorted(stale)}'
        )
    for symbol, record in mappings.items():
        if not record.get('capability'):
            raise ValueError(f'API has no capability: {symbol}')


def validate_coverage(data, root):
    """Promises survive deletion or narrowing of executable scenarios."""
    scenarios = {case['id']: case for case in data['scenarios']}
    covered = set()
    identifiers = set()
    apis = {group['id'] for group in data['api_contracts']}
    for capability in data['capabilities']:
        identifier = capability['id']
        if identifier in identifiers:
            raise ValueError(f'duplicate capability: {identifier}')
        identifiers.add(identifier)
        links = capability.get('api_contracts', [])
        if not links or not set(links) <= apis:
            raise ValueError(f'capability needs valid API links: {identifier}')
        for required in capability['requirements']:
            name = required['scenario']
            if name not in scenarios:
                raise ValueError(f'missing required scenario: {name}')
            case = scenarios[name]
            covered.add(name)
            fixtures = {case['fixture']} | {
                variant['fixture']
                for variant in case.get('variants', {}).values()
                if 'fixture' in variant
            }
            hashes = {
                path: hashlib.sha256(
                    (root / 'tests/parity' / path).read_bytes()
                ).hexdigest()
                for path in fixtures
            }
            if required.get('fixture_hashes') != hashes:
                raise ValueError(f'coverage fixture changed: {name}')
            for dimension in ('languages', 'models', 'connectors'):
                actual = (
                    data['languages']
                    if dimension == 'languages'
                    else case[dimension]
                )
                if set(required[dimension]) != set(actual):
                    raise ValueError(
                        f'coverage scope differs: {name}/{dimension}'
                    )
            for field in (
                'expected',
                'expected_by_model',
                'expected_by_language',
            ):
                if required.get(field, {}) != case.get(field, {}):
                    raise ValueError(
                        f'coverage expectation differs: {name}/{field}'
                    )
    if covered != scenarios.keys():
        raise ValueError(
            f'scenarios need capability mapping: {sorted(scenarios.keys() - covered)}'
        )
    for record in data['type_contracts']:
        if 'vectors' in record:
            if (
                hashlib.sha256(
                    (root / record['vectors']).read_bytes()
                ).hexdigest()
                != record['vectors_sha256']
            ):
                raise ValueError(
                    f'type vector contract changed: {record["id"]}'
                )
            vectors = json.loads((root / record['vectors']).read_text())[
                'vectors'
            ]
            if set(record['required_vectors']) != {
                vector['id'] for vector in vectors
            }:
                raise ValueError(
                    f'type vector inventory differs: {record["id"]}'
                )
        for name in record.get('scenarios', []):
            if name not in covered:
                raise ValueError(f'type coverage missing scenario: {name}')
        for tests in record.get('tests', {}).values():
            for test in tests:
                if not (root / test.split('::')[0]).is_file():
                    raise ValueError(f'type test missing: {test}')


def changed_records(before, after):
    """Diff promises, independently of which scenarios remain executable."""
    changes = set()
    for section in (
        'api_contracts',
        'scenarios',
        'capabilities',
        'type_contracts',
    ):
        previous = {record['id']: record for record in before.get(section, [])}
        current = {record['id']: record for record in after.get(section, [])}
        for identifier in previous.keys() | current.keys():
            if previous.get(identifier) != current.get(identifier):
                changes.add(f'{section}/{identifier}')
    for section in (
        'languages',
        'models',
        'connectors',
        'deferrals',
        'input_paths',
        'api_inventory',
        'api_exceptions',
        'live_api_tests',
        'public_api_snapshot_sha256',
    ):
        if before.get(section) != after.get(section):
            changes.add(section)
    return changes


def validate_changes(before, after, base_commit):
    """Decisions are tied to the reviewed merge base, not reusable blanket waivers."""
    decisions = {}
    for decision in after.get('contract_changes', []):
        if decision.get('base_commit') != base_commit:
            continue
        key = decision['record']
        if key in decisions:
            raise ValueError(f'duplicate contract decision: {key}')
        if not decision.get('reason', '').strip():
            raise ValueError(f'contract decision needs a reason: {key}')
        decisions[key] = decision
    missing = changed_records(before, after) - decisions.keys()
    if missing:
        raise ValueError(f'contract changes need decisions: {sorted(missing)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--base',
        required=True,
        help='Target Git ref; compare against its merge base',
    )
    parser.add_argument('--java-inventory', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]

    def git(*arguments):
        return subprocess.run(
            ['git', *arguments],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

    from api_contracts import validate

    from contract import manifest

    data = manifest()
    validate(data)
    validate_coverage(data, root)
    validate_developer_contracts(data, root)
    validate_api_inventory(
        json.loads(args.java_inventory.read_text()),
        data['api_inventory']['java'],
    )
    base = git('merge-base', args.base, 'HEAD')
    baseline_present = bool(
        git('ls-tree', '--name-only', base, 'tests/parity/manifest.json')
    )
    old_python = json.loads(git('show', f'{base}:tests/golden/public_api.json'))
    new_python = json.loads((root / 'tests/golden/public_api.json').read_text())
    if baseline_present:
        before = json.loads(git('show', f'{base}:tests/parity/manifest.json'))
        validate_changes(before, data, base)
        validate_changed_symbols(
            before,
            data,
            old_python,
            new_python,
        )
    else:
        validate_introduction(
            data,
            old_python,
            new_python,
            base,
            baseline_has_java=bool(
                git('ls-tree', '--name-only', base, 'java/pom.xml')
            ),
        )
    print(
        json.dumps(
            {
                'base_commit': base,
                'baseline_manifest': 'present'
                if baseline_present
                else 'introduced',
                'python_symbols': len(data['api_inventory']['python']),
                'java_symbols': len(data['api_inventory']['java']),
                'status': 'passed',
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
