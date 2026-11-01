# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare frozen Python to each installed language and render evidence, fail closed."""

import argparse
import json
import math
import operator
from pathlib import Path

from identity import IdentityError, canonical

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
        if (
            cell[2] == 'relational'
            and case.get('comparison') == 'fixture-source-identity'
            and wanted == 'passed'
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
                values[name] = {'schema': reading['schema'], 'data': data}
            return values

        changed = changed or difference(
            readers(old), readers(new), exact, '$.reader'
        )

        row.update(
            status='failed' if changed else 'passed',
            verdict=expected.get('verdict', 'Parity never supported')
            if expected['status'] == 'refused'
            else 'Parity (private bridge)'
            if (new.get('endpoint') or {}).get('transport') == 'private-bridge'
            else 'Parity',
            reason=f'difference at {changed}'
            if changed
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
