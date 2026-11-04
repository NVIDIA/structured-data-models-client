# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the maintained Java API ledger against fresh Surefire results."""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

_TESTS = {
    'java': re.compile(
        r'(?P<file>java/src/test/java/[A-Za-z0-9_./-]+\.java)#(?P<name>[A-Za-z0-9_]+)'
    )
}
_FORM = {'java': 'java/src/test/java/<file>.java#<method>'}


class Report:
    def __init__(self):
        self.problems = []

    def fail(self, scope, message):
        self.problems.append(f'{scope}: {message}')


def junit_ran(reports: Path) -> set[tuple[str, str]]:
    r"""Each test JUnit ran and did not skip, as Surefire's reports in
    ``reports`` (``TEST-*.xml``) record it: its class and its method's name
    (a parameterized test's invocations, ``name(String)[1]``, by the name
    before its parameters). Only :func:`junit_run` reads them, from a
    directory its own Maven run filled."""
    import xml.etree.ElementTree as ElementTree

    ran: set[tuple[str, str]] = set()
    for path in sorted(reports.glob('TEST-*.xml')):
        for case in ElementTree.parse(path).getroot().iter('testcase'):
            if case.find('skipped') is not None:
                continue
            name = re.split(r'[(\[]', case.get('name', ''), maxsplit=1)[0]
            ran.add((case.get('classname', ''), name))
    return ran


def junit_problem(test: str, ran: set[tuple[str, str]]) -> str | None:
    r"""Why the Java test ``test`` (``java/src/test/java/<file>.java#<method>``)
    is no test JUnit ran, by :func:`junit_ran`, or None: its file's class,
    or a class nested in it, ran the method, and did not skip it."""
    found = _TESTS['java'].fullmatch(test)
    if found is None:
        return f'not {_FORM["java"]}'
    name = (
        found['file'].removeprefix('java/src/test/java/').removesuffix('.java')
    )
    owner = name.replace('/', '.')
    if any(
        method == found['name']
        and (cls == owner or cls.startswith(owner + '$'))
        for cls, method in ran
    ):
        return None
    return 'not a test JUnit ran (target/surefire-reports)'


def parity_junit_check(
    repo: Path, reports: Path, report: Report, gate='local'
) -> None:
    """Hold the maintained API ledger to this same fresh Surefire run."""
    import xml.etree.ElementTree as ElementTree

    ledger = repo / 'tests/parity/manifest.json'
    if not ledger.is_file():
        report.fail('parity API', f'missing ledger: {ledger}')
        return
    data = json.loads(ledger.read_text())
    from api_contracts import groups

    required_tests = {
        node
        for group in groups(data, gate)
        for node in group['tests'].get('java', [])
    }
    required = {
        node.split('#')[0]
        .removeprefix('java/src/test/java/')
        .removesuffix('.java')
        .replace('/', '.')
        for node in required_tests
    }
    expected_counts = (
        {
            record['test']: record.get('invocations', 1)
            for record in data.get('live_api_tests', [])
            if '#' in record['test']
        }
        if gate == 'live'
        else {}
    )
    invocation_counts = dict.fromkeys(expected_counts, 0)
    seen = set()
    reached = set()
    for path in sorted(reports.glob('TEST-*.xml')):
        suite = ElementTree.parse(path).getroot()
        for case in suite.iter('testcase'):
            owner, name = case.get('classname', ''), case.get('name', '')
            matches = {
                item
                for item in required
                if owner == item or owner.startswith(item + '$')
            }
            if not matches:
                continue
            method = re.split(r'[(\[]', name, maxsplit=1)[0]
            if not any(
                test.endswith('#' + method)
                and junit_problem(test, {(owner, method)}) is None
                for test in required_tests
            ):
                continue
            for test in invocation_counts:
                if junit_problem(test, {(owner, method)}) is None:
                    invocation_counts[test] += 1
            identity = (owner, name)
            if identity in seen:
                report.fail('parity API', f'duplicate JUnit result: {identity}')
            seen.add(identity)
            reached.update(matches)
            if any(
                case.find(tag) is not None
                for tag in ('skipped', 'failure', 'error')
            ):
                report.fail(
                    'parity API',
                    f'required JUnit test did not pass: {identity}',
                )
    for owner in sorted(required - reached):
        report.fail('parity API', f'missing required JUnit class: {owner}')
    for test, expected in expected_counts.items():
        if invocation_counts[test] != expected:
            report.fail(
                'parity API',
                f'expected {expected} invocations, got {invocation_counts[test]}: {test}',
            )
    ran = junit_ran(reports)
    for test in sorted(required_tests):
        problem = junit_problem(test, ran)
        if problem:
            report.fail('parity API', f'{test}: {problem}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--repo', type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        '--run',
        action='store_true',
        help='Run Maven verify with fresh reports before checking.',
    )
    parser.add_argument('--maven', default='mvn')
    parser.add_argument('--gate', choices=['local', 'live'], default='local')
    parser.add_argument('--maven-arg', action='append', default=[])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    reports = repo / 'java/target/surefire-reports'
    if args.run:
        if reports.exists():
            shutil.rmtree(reports)
        selection = []
        if args.gate == 'live':
            from api_contracts import groups, validate_live_targets

            data = json.loads((repo / 'tests/parity/manifest.json').read_text())
            validate_live_targets(
                data, {item['id'] for item in data['api_contracts']}
            )
            nodes = {
                node
                for group in groups(data, 'live')
                for node in group['tests'].get('java', [])
            }
            if not nodes:
                parser.error('no live Java tests selected')
            selection = [
                '-Dtest='
                + ','.join(
                    sorted(
                        Path(node.split('#')[0]).stem + '#' + node.split('#')[1]
                        for node in nodes
                    )
                )
            ]
        completed = subprocess.run(
            [
                args.maven,
                '-B',
                '-f',
                'java/pom.xml',
                *selection,
                *args.maven_arg,
                'verify',
            ],
            cwd=repo,
            check=False,
        )
    else:
        completed = None
    report = Report()
    parity_junit_check(repo, reports, report, args.gate)
    for problem in report.problems:
        print(problem)
    failed = bool(report.problems) or (
        completed is not None and completed.returncode != 0
    )
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    'gate': args.gate,
                    'status': 'failed' if failed else 'passed',
                    'problems': report.problems,
                },
                indent=2,
            )
            + '\n'
        )
    return (
        completed.returncode
        if completed is not None and completed.returncode
        else int(failed)
    )


if __name__ == '__main__':
    raise SystemExit(main())
