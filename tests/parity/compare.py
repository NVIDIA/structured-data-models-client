# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare frozen Python to each installed language and render evidence, fail closed."""

import argparse
import json
import math
import operator
import re
from pathlib import Path

from identity import IdentityError, canonical, canonical_scoring_payload

from contract import (
    cells,
    deferred,
    effective_case,
    evidence_problem,
    manifest,
    records,
)


def difference(left, right, tolerance, path='$'):
    if isinstance(left, bool) or isinstance(right, bool):
        return None if type(left) is type(right) and left == right else path
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if left == right:
            return None
        if isinstance(left, int) and isinstance(right, int):
            return path
        if tolerance['absolute'] == 0 and tolerance['relative'] == 0:
            return path
        return (
            None
            if math.isclose(
                left,
                right,
                abs_tol=tolerance['absolute'],
                rel_tol=tolerance['relative'],
            )
            else path
        )
    if type(left) is not type(right):
        return path
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return path + '.keys'
        for name in left:
            found = difference(
                left[name], right[name], tolerance, path + '.' + name
            )
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return path + '.length'
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            found = difference(a, b, tolerance, f'{path}[{index}]')
            if found:
                return found
        return None
    return None if left == right else path


def equivalent_reader_time_units(old, new):
    """Normalize only exact timestamp values with compatible declared resolutions."""
    exact = {'absolute': 0, 'relative': 0}
    if difference(old['data'], new['data'], exact) is not None:
        return
    if len(old['schema']) != len(new['schema']):
        return
    for before, after in zip(old['schema'], new['schema'], strict=True):
        if before[0] != after[0] or before[1] == after[1]:
            continue
        a = re.fullmatch(r'timestamp\[(s|ms|us|ns)(, tz=[^]]+)?\]', before[1])
        b = re.fullmatch(r'timestamp\[(s|ms|us|ns)(, tz=[^]]+)?\]', after[1])
        if not a or not b or a[2] != b[2]:
            continue
        digits = min(
            {'s': 0, 'ms': 3, 'us': 6, 'ns': 9}[a[1]],
            {'s': 0, 'ms': 3, 'us': 6, 'ns': 9}[b[1]],
        )
        at = old['data']['columns'].index(before[0])
        for row in old['data']['rows']:
            value = row[at]
            if value is None:
                continue
            match = (
                re.fullmatch(
                    r'[+-]?\d{4,}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.(\d{1,9}))?(?:Z|[+-]\d{2}:\d{2})?',
                    value,
                )
                if isinstance(value, str)
                else None
            )
            if not match or any(c != '0' for c in (match[1] or '')[digits:]):
                break
        else:
            # Caller compares all other schema fields and every original value.
            after[1] = before[1]


def equivalent_derived_entity_refs(old_payloads, new_payloads, fixture):
    """Accept exact nullable integer storage only for declared temporal-link references."""
    if (
        not isinstance(old_payloads, list)
        or not isinstance(new_payloads, list)
        or len(old_payloads) != len(new_payloads)
    ):
        return False
    improved = False
    for old, new in zip(old_payloads, new_payloads, strict=True):
        task = old.get('task', {})
        if (
            task != new.get('task')
            or task.get('kind') != 'temporal_link_prediction'
        ):
            continue
        entities = task.get('entity_table_names', [])
        old_schema, new_schema = old.get('schema', {}), new.get('schema', {})
        links = old_schema.get('relationships', [])
        if links != new_schema.get('relationships', []):
            continue
        a = old_schema.get('instance_table', {}).get('columns', {})
        b = new_schema.get('instance_table', {}).get('columns', {})
        for index, target in enumerate(entities):
            name = f'__kumo_entity_ref_{index}'
            before, after = a.get(name, {}), b.get(name, {})
            if (
                before.get('dtype') != 'float64'
                or after.get('dtype') != 'int64'
            ):
                continue
            if (
                before.get('stype') != 'ID'
                or {**before, 'dtype': 'int64'} != after
            ):
                continue
            matching = [
                link
                for link in links
                if not link.get('source_table')
                and link.get('source_columns') == [name]
            ]
            if (
                len(matching) != 1
                or matching[0].get('target_table') != target
                or matching[0].get('target_columns')
                != [fixture['primary_keys'].get(target)]
            ):
                continue
            valid = True
            observed = False
            for half in ('context', 'predict'):
                left = old.get(half, {}).get('instance_table', {})
                right = new.get(half, {}).get('instance_table', {})
                if (
                    left.get('columns') != right.get('columns')
                    or len(left.get('rows', [])) != len(right.get('rows', []))
                    or name not in left.get('columns', [])
                ):
                    valid = False
                    break
                at = left['columns'].index(name)
                for row, other in zip(left['rows'], right['rows'], strict=True):
                    x, y = row[at], other[at]
                    if x is None and y is None:
                        continue
                    observed = True
                    if (
                        type(x) is not float
                        or type(y) is not int
                        or not math.isfinite(x)
                        or not x.is_integer()
                        or int(x) != y
                    ):
                        valid = False
                        break
            if valid and observed:
                before['dtype'] = 'int64'
                improved = True
    return improved


def assess(
    data, baseline, candidate, *, live=True, selected=None, candidate_commit
):
    scenarios = {case['id']: case for case in data['scenarios']}
    required = set(cells(data))
    unknown = set(candidate) - required
    if unknown:
        raise ValueError(f'unknown candidate cells: {sorted(unknown)}')
    result = []
    artifacts = {}
    for cell, record in candidate.items():
        identity = record.get('artifact', {})
        if not identity.get('package_hash'):
            continue
        pinned = (
            identity.get('commit'),
            identity.get('package_hash'),
            identity.get('native_hash'),
            identity.get('contract'),
        )
        previous = artifacts.setdefault(cell[3], pinned)
        if previous != pinned:
            raise ValueError('candidate records mix artifacts for ' + cell[3])
        if (
            candidate_commit is not None
            and identity.get('commit') != candidate_commit
        ):
            raise ValueError(
                'candidate observation is not the requested commit'
            )
    for cell in sorted(required):
        row = {
            'cell': list(cell),
            'status': 'incomplete',
            'reason': 'missing observation',
        }
        result.append(row)
        if deferred(data, cell):
            row.update(
                status='deferred',
                reason='manifest-approved Java Snowflake deferral',
            )
            continue
        if selected is not None and cell not in selected:
            row['reason'] = (
                'outside requested scope; full goal remains incomplete'
            )
            continue
        case = effective_case(scenarios[cell[0]], cell[1])
        old = baseline.get((*cell[:3], 'python'))
        new = candidate.get(cell)
        if old is None or new is None:
            continue
        problem = evidence_problem(
            old, case, 'baseline', live
        ) or evidence_problem(new, case, 'candidate', live)
        if problem:
            row['reason'] = problem
            continue
        if old['source_snapshot'] != new['source_snapshot']:
            row['reason'] = 'different source snapshots'
            continue
        if live and old.get('endpoint') != new.get('endpoint'):
            row['reason'] = 'different model versions or transports'
            continue
        expected = case.get('expected_by_language', {}).get(
            cell[3], case['expected']
        )
        baseline_expected = case['expected']
        wanted = expected.get(
            'outcome',
            'refused' if expected['status'] == 'refused' else 'passed',
        )
        old_wanted = baseline_expected.get(
            'outcome',
            'refused' if baseline_expected['status'] == 'refused' else 'passed',
        )
        if new.get('status') != wanted or old.get('status') != old_wanted:
            row.update(
                status='failed',
                reason='unexpected outcome or missing baseline behavior',
            )
            continue
        if expected['status'] == 'refused':
            if any(
                record.get('error', {}).get(name) != spec[name]
                for record, spec in ((old, baseline_expected), (new, expected))
                for name in ('stage', 'code', 'rule')
            ):
                row.update(
                    status='failed',
                    reason='unexpected baseline/candidate refusal stage/code/rule',
                )
                continue

        # Differences must have executable case-specific policies before they
        # can waive anything. A prose difference ID alone never grants a pass.
        import copy

        old_semantic, new_semantic = (
            copy.deepcopy(old.get('semantic')),
            copy.deepcopy(new.get('semantic')),
        )
        if case.get('supplement') and wanted == 'passed':
            from supplements import normalize_response_ids

            for semantic in (old_semantic, new_semantic):
                normalize_response_ids(semantic, case)
        improved_entity_ref = False
        if cell[0] == 'relational.temporal-link' and wanted == 'passed':
            from contract import HERE

            fixture = json.loads((HERE / case['fixture']).read_text())
            improved_entity_ref = equivalent_derived_entity_refs(
                old_semantic['request'], new_semantic['request'], fixture
            )
        if (
            cell[2] == 'relational'
            and case.get('comparison') == 'fixture-source-identity'
            and wanted == 'passed'
            and difference(
                old_semantic['request'],
                new_semantic['request'],
                {'absolute': 0, 'relative': 0},
            )
            is not None
        ):
            from contract import HERE

            fixture = json.loads((HERE / case['fixture']).read_text())
            try:
                for semantic in (old_semantic, new_semantic):
                    semantic['request'] = canonical(
                        semantic['request'], fixture
                    )
            except (IdentityError, KeyError, TypeError) as error:
                row.update(
                    status='failed',
                    reason='identity proof failed: ' + str(error),
                )
                continue
        if (
            case.get('supplement')
            and cell[2] == 'relational'
            and wanted == 'passed'
        ):
            from contract import HERE

            fixture = json.loads((HERE / case['fixture']).read_text())
            try:
                for before, after in zip(
                    old_semantic['transport'],
                    new_semantic['transport'],
                    strict=True,
                ):
                    if (
                        before['operation'] in {'fit', 'predict'}
                        and difference(
                            before['payload'],
                            after['payload'],
                            {'absolute': 0, 'relative': 0},
                        )
                        is not None
                    ):
                        before['payload'] = canonical(
                            [before['payload']], fixture
                        )[0]
                        after['payload'] = canonical(
                            [after['payload']], fixture
                        )[0]
                    elif (
                        before['operation'] == 'score'
                        and difference(
                            before['payload'],
                            after['payload'],
                            {'absolute': 0, 'relative': 0},
                        )
                        is not None
                    ):
                        before['payload'] = canonical_scoring_payload(
                            before['payload'],
                            old['semantic']['transport'][0]['payload'][
                                'schema'
                            ],
                            fixture,
                        )
                        after['payload'] = canonical_scoring_payload(
                            after['payload'],
                            new['semantic']['transport'][0]['payload'][
                                'schema'
                            ],
                            fixture,
                        )
            except (IdentityError, KeyError, TypeError) as error:
                row.update(
                    status='failed',
                    reason='transport identity proof failed: ' + str(error),
                )
                continue
        exact = {'absolute': 0, 'relative': 0}
        old_results = (
            old_semantic.pop('results', None)
            if isinstance(old_semantic, dict)
            else None
        )
        new_results = (
            new_semantic.pop('results', None)
            if isinstance(new_semantic, dict)
            else None
        )
        changed = difference(old_semantic, new_semantic, exact)
        changed = changed or difference(
            old_results, new_results, case['tolerance'], '$.results'
        )
        from contract import HERE

        fixture = json.loads((HERE / case['fixture']).read_text())

        def readers(record, fixture=fixture):
            values = {}
            for name, reading in record.get('reader', {}).items():
                data = copy.deepcopy(reading['data'])
                at = data['columns'].index(fixture['primary_keys'][name])
                data['rows'].sort(key=operator.itemgetter(at))
                values[name] = {
                    'schema': copy.deepcopy(reading['schema']),
                    'data': data,
                }
            return values

        old_readers, new_readers = readers(old), readers(new)
        for name in old_readers.keys() & new_readers.keys():
            equivalent_reader_time_units(old_readers[name], new_readers[name])
        changed = changed or difference(
            old_readers, new_readers, exact, '$.reader'
        )

        row.update(
            status='failed' if changed else 'passed',
            verdict=(
                'Improvement (private bridge)'
                if (new.get('endpoint') or {}).get('transport')
                == 'private-bridge'
                else 'Improvement'
            )
            if improved_entity_ref and not changed
            else expected.get('verdict', 'Parity never supported')
            if expected['status'] == 'refused'
            else 'Parity (private bridge)'
            if (new.get('endpoint') or {}).get('transport') == 'private-bridge'
            else 'Parity',
            reason=f'difference at {changed}'
            if changed
            else 'Exact nullable integer storage for declared temporal entity reference'
            if improved_entity_ref
            else 'matching evidence',
        )
    scoped = [
        row
        for row in result
        if selected is None or tuple(row['cell']) in selected
    ]

    def complete(rows):
        return bool(rows) and all(
            row['status'] in ('passed', 'deferred') for row in rows
        )

    return {
        'version': 1,
        'scope_complete': complete(scoped),
        'goal_complete': live and complete(result),
        'coverage_mode': 'live' if live else 'local',
        'candidate_commit': candidate_commit,
        'cells': result,
    }


def label(row, mode):
    if row.get('status') == 'passed':
        return (
            row.get('verdict', 'Parity')
            if mode == 'live'
            else 'Local check only'
        )
    return row.get('status', 'not applicable')


def matrix(report):
    indexed = {tuple(row['cell']): row for row in report['cells']}
    lines = [
        '# Prediction parity evidence',
        '',
        '<!-- Generated by tests/parity/compare.py; do not edit. -->',
        '',
        '| Case / connector | Python Relational | Python Tabular | Java Relational | Java Tabular |',
        '|---|---|---|---|---|',
    ]
    for case, connector in sorted({key[:2] for key in indexed}):
        values = [
            label(
                indexed.get((case, connector, model, language), {}),
                report['coverage_mode'],
            )
            for language in ('python', 'java')
            for model in ('relational', 'tabular')
        ]
        lines.append(
            '| ' + ' | '.join([case + ' / ' + connector, *values]) + ' |'
        )
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path(__file__).with_name('manifest.json'),
    )
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate-commit', required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--matrix', type=Path)
    parser.add_argument('--local', action='store_true')
    for axis in ('case', 'connector', 'model', 'language'):
        parser.add_argument('--' + axis, action='append')
    args = parser.parse_args()
    data = manifest(args.manifest)
    selected = {
        cell
        for cell in cells(data)
        if all(
            not getattr(args, axis) or cell[i] in getattr(args, axis)
            for i, axis in enumerate(('case', 'connector', 'model', 'language'))
        )
    }
    report = assess(
        data,
        records(args.baseline),
        records(args.candidate),
        live=not args.local,
        selected=selected,
        candidate_commit=args.candidate_commit,
    )
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    if args.matrix:
        args.matrix.write_text(matrix(report))
    return 0 if report['scope_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
