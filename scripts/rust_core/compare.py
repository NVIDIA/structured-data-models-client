#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Compares runners' observations of the contract cases.

    scripts/rust_core/compare.py --cases DIR [--reference FILE] OBSERVATIONS...

Each OBSERVATIONS file is one runner's output (run_python.py or
sdm-contract-run); FILE is the committed reference observations
(``contract/reference/<reference-commit>.jsonl``). Every observation is
validated against ``contract/observation.schema.json``; a file with none,
or a case a given runner did not observe, is a runner failure. Each
observation must be of the case and bases files compared (their SHA-256 in
its identity), and leaves null only what its runner cannot observe: a
Python runner always has a stage, and a class for a refusal. Then, for
each acceptance case, the rules of Phase 1a so far (T1.10):

- outcome: the call succeeded or refused, as expected and as the reference
  did, at the same stage and with the same exception class wherever both
  sides observe them;
- code: ``StructuredDataError.code``, or the code the S0 route maps the
  core's answer to, as expected and as the reference;
- kind and code on the core: each core runner's outcome is the core's
  answer as the S0 route maps it (accepted, a success; refused,
  ``INVALID_REQUEST``; an error, a refusal with its code); ``python-core``
  and ``rust-host`` started the same operations with the same inputs, and
  the core answered both the same: the same acceptance, or an error of the
  same kind and code;
- route proof: ``python-core`` ran the case's core operations with
  ``sdm_client._native`` loaded; ``python-old`` and ``python-ref`` never
  loaded it; ``python-ref``'s guard passed; every runner ran DuckDB at the
  reference's version.

The fields ``rust-host`` cannot observe, a stage and an exception class,
are null there and not compared, never treated as equal. Diagnostic cases are
reported, not gated. The first difference per case, runner and field is
printed with its path; message text is never compared, and input values are
never printed. Exit status 1 on any difference or runner failure.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / 'contract/runners/run_python.py'
#: The outcome fields, compared wherever both sides observe them.
OUTCOME = ('result', 'stage', 'error', 'code')
#: The outcome fields the pure-Rust host cannot observe: it makes no public
#: call, so it has no stage and no exception class. Every other runner
#: observes both, and a null there is a defect, never a pass.
UNOBSERVED_BY_RUST_HOST = ('stage', 'error')
PYTHON = ('python-ref', 'python-core', 'python-old')


def _runner() -> ModuleType:
    r"""``run_python.py``, whose case loader and schemas this shares."""
    spec = importlib.util.spec_from_file_location('run_python', RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def fail(self, text: str) -> None:
        self.failures.append(text)

    def note(self, text: str) -> None:
        self.notes.append(text)


def _observations(
    path: Path, runner: ModuleType, report: Report
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    r"""One runner's observations by case, each validated."""
    validate = runner.validator('observation')
    by_case: dict[str, dict[str, Any]] = {}
    runners: set[str] = set()
    if not path.is_file():
        report.fail(f'{path.name}: runner failure: no observations written')
        return None, by_case
    for number, line in enumerate(
        path.read_text(encoding='utf-8').splitlines(), 1
    ):
        try:
            observation = json.loads(line)
        except json.JSONDecodeError:
            report.fail(f'{path.name}:{number}: runner failure: not JSON')
            continue
        problem = runner._invalid(validate, observation)
        if problem is not None:
            report.fail(
                f'{path.name}:{number}: runner failure: not a valid '
                f'observation: {problem}'
            )
            continue
        case = observation['case']
        if case in by_case:
            report.fail(f'{path.name}: {case} observed twice')
        by_case[case] = observation
        runners.add(observation['identity']['runner'])
    if len(runners) > 1:
        report.fail(f'{path.name}: observations of more than one runner')
    return (runners.pop() if len(runners) == 1 else None), by_case


def _differ(
    report: Report,
    where: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    show: bool = True,
) -> None:
    if show:
        report.fail(
            f'{where}: {field}: expected {expected!r}, observed {observed!r}'
        )
    else:
        report.fail(f'{where}: {field}: differs')


def _expectation(
    report: Report, case: dict[str, Any], who: str, outcome: dict[str, Any]
) -> None:
    r"""The outcome and code rules against the case's expectation."""
    expect = case['expect']
    where = f'{case["id"]}: {who}'
    if outcome['result'] != expect['outcome']:
        _differ(
            report,
            where,
            'outcome.result',
            expect['outcome'],
            outcome['result'],
        )
        return
    wanted = {
        'stage': expect['stage'],
        'error': expect.get('error'),
        'code': expect.get('code'),
    }
    for field, value in wanted.items():
        seen = outcome[field]
        if who == 'rust-host' and field in UNOBSERVED_BY_RUST_HOST:
            continue
        if seen != value:
            _differ(report, where, f'outcome.{field}', value, seen)


def _against_reference(
    report: Report,
    case: dict[str, Any],
    who: str,
    outcome: dict[str, Any],
    reference: dict[str, Any],
) -> None:
    for field in OUTCOME:
        mine, theirs = outcome[field], reference['outcome'][field]
        if who == 'rust-host' and field in UNOBSERVED_BY_RUST_HOST:
            continue
        if mine != theirs:
            _differ(
                report,
                f'{case["id"]}: {who} against python-ref',
                f'outcome.{field}',
                theirs,
                mine,
            )


def _route(
    report: Report, case: dict[str, Any], who: str, observation: dict[str, Any]
) -> None:
    identity = observation['identity']
    where = f'{case["id"]}: {who}'
    wanted = sorted(case.get('core', {}))
    if who == 'python-core':
        if identity['native_loaded'] is not True or identity['native'] is None:
            report.fail(f'{where}: route: sdm_client._native was not loaded')
        missing = [op for op in wanted if op not in identity['operations']]
        if missing or observation['core'] is None:
            report.fail(f'{where}: route: {missing} did not run on the core')
    elif who in ('python-old', 'python-ref'):
        if identity['native_loaded'] is not False or identity['operations']:
            report.fail(f'{where}: route: the old route loaded the core')
        if who == 'python-ref' and identity['reference'] is None:
            report.fail(f'{where}: route: no reference guard result')
    elif who == 'rust-host':
        missing = [op for op in wanted if op not in identity['operations']]
        if missing or observation['core'] is None:
            report.fail(f'{where}: route: {missing} did not run on the host')


def _shape(
    report: Report, case: dict[str, Any], who: str, observation: dict[str, Any]
) -> None:
    r"""Which fields may be null, by runner and outcome."""
    where = f'{case["id"]}: {who}'
    outcome = observation['outcome']
    if who in PYTHON:
        if outcome['stage'] is None:
            report.fail(f'{where}: outcome.stage: null on a Python runner')
        if outcome['result'] == 'refusal' and outcome['error'] is None:
            report.fail(f'{where}: outcome.error: a refusal without a class')
        if outcome['result'] == 'success' and (
            outcome['error'] is not None or outcome['code'] is not None
        ):
            report.fail(f'{where}: outcome: a success with an error')
        if observation['host'] is None:
            report.fail(f'{where}: host: null on a Python runner')
        if observation['identity']['inputs']['bases'] is None:
            report.fail(f'{where}: identity.inputs.bases: null')
        if (observation['core'] is None) != (who != 'python-core'):
            report.fail(f'{where}: core: present only on python-core')
    else:
        for field in UNOBSERVED_BY_RUST_HOST:
            if outcome[field] is not None:
                report.fail(f'{where}: outcome.{field}: not observable here')
        if observation['host'] is not None or observation['core'] is None:
            report.fail(f'{where}: a host section, or no core section')


def _inputs(
    report: Report,
    case: dict[str, Any],
    who: str,
    observation: dict[str, Any],
    digests: dict[str, str],
) -> None:
    r"""The observation is of the case and bases files compared."""
    inputs = observation['identity']['inputs']
    where = f'{case["id"]}: {who}'
    if inputs['case'] != digests[case['id']]:
        report.fail(f'{where}: identity.inputs.case: not this case file')
    if inputs['bases'] is not None and inputs['bases'] != digests['']:
        report.fail(f'{where}: identity.inputs.bases: not this bases file')


def _core_consistent(
    report: Report, case: dict[str, Any], who: str, observation: dict[str, Any]
) -> None:
    r"""The core's answer is the outcome, as the S0 route maps it: an
    accepted query succeeds, a refused one is ``INVALID_REQUEST``, and a
    core error is a refusal with its own code."""
    core = observation['core']
    if core is None or 'validate_query' not in case.get('core', {}):
        return
    outcome = observation['outcome']
    if core['error'] is not None:
        wanted = ('refusal', core['error']['code'])
        if core['accepted'] is not None:
            report.fail(f'{case["id"]}: {who}: core: an answer and an error')
    elif core['accepted'] is None:
        report.fail(f'{case["id"]}: {who}: core: no answer')
        return
    else:
        wanted = (
            ('success', None)
            if core['accepted']
            else ('refusal', 'INVALID_REQUEST')
        )
    seen = (outcome['result'], outcome['code'])
    if seen != wanted:
        _differ(
            report, f'{case["id"]}: {who}', 'outcome against core', wanted, seen
        )


def _core_agree(
    report: Report,
    case: dict[str, Any],
    python: dict[str, Any],
    rust: dict[str, Any],
) -> None:
    r"""The kind and code rule: the two core runners on ``core`` fields."""
    where = f'{case["id"]}: python-core against rust-host'
    if python['identity']['operations'] != rust['identity']['operations']:
        _differ(
            report,
            where,
            'identity.operations',
            python['identity']['operations'],
            rust['identity']['operations'],
        )
    mine, theirs = python['core'], rust['core']
    if mine is None or theirs is None:
        return
    if mine['inputs'] != theirs['inputs']:
        _differ(report, where, 'core.inputs', None, None, show=False)
    for field in ('accepted', 'error'):
        if mine[field] != theirs[field]:
            _differ(report, where, f'core.{field}', mine[field], theirs[field])
    for field in ('contract',):
        if python['identity'][field] != rust['identity'][field]:
            _differ(
                report,
                where,
                f'identity.{field}',
                python['identity'][field],
                rust['identity'][field],
            )


def _case_inputs(
    report: Report, case: dict[str, Any], who: str, observation: dict[str, Any]
) -> None:
    r"""A core runner's recorded inputs are the case's ``core`` inputs."""
    core = observation['core']
    if core is None or 'validate_query' not in case.get('core', {}):
        return
    spec = case['core']['validate_query']
    wanted = {
        'op': 'validate_query',
        'query': case['input']['pql'],
        'profile': spec['profile'],
        'schema': spec['schema'],
    }
    if wanted not in core['inputs']:
        _differ(
            report,
            f'{case["id"]}: {who}',
            'core.inputs',
            None,
            None,
            show=False,
        )


def compare(
    cases_dir: list[Path], reference_file: Path | None, files: list[Path]
) -> Report:
    report = Report()
    runner = _runner()
    try:
        loaded = runner.load_cases(cases_dir)
    except runner.RunnerFailure as failure:
        report.fail(f'cases: {failure}')
        return report
    cases = [case for case, _ in loaded]
    digests = {case['id']: digest for case, digest in loaded}
    digests[''] = runner.digest(runner.BASES.read_bytes())
    reference: dict[str, dict[str, Any]] = {}
    if reference_file is not None:
        who, reference = _observations(reference_file, runner, report)
        if who != 'python-ref':
            report.fail(f'{reference_file.name}: not python-ref observations')
    observed: dict[str, dict[str, dict[str, Any]]] = {}
    for path in files:
        who, by_case = _observations(path, runner, report)
        if who is None:
            report.fail(f'{path.name}: runner failure: no valid observations')
            continue
        if who in observed:
            report.fail(f'{path.name}: a second file of {who}')
        observed[who] = by_case
    known = {case['id'] for case in cases}
    for who, by_case in [('python-ref', reference), *observed.items()]:
        for name in sorted(set(by_case) - known):
            report.fail(f'{name}: {who} observed a case that does not exist')
    for case in cases:
        gating = case['status'] == 'acceptance'
        before = len(report.failures)
        ref = reference.get(case['id'])
        if reference_file is not None and ref is None:
            report.fail(f'{case["id"]}: no reference observation')
        if ref is not None:
            _shape(report, case, 'python-ref', ref)
            _inputs(report, case, 'python-ref', ref, digests)
            _expectation(report, case, 'python-ref', ref['outcome'])
            _route(report, case, 'python-ref', ref)
        for who, by_case in observed.items():
            observation = by_case.get(case['id'])
            if observation is None:
                report.fail(
                    f'{case["id"]}: {who}: runner failure: no observation'
                )
                continue
            _shape(report, case, who, observation)
            _inputs(report, case, who, observation, digests)
            _core_consistent(report, case, who, observation)
            _expectation(report, case, who, observation['outcome'])
            if ref is not None:
                _against_reference(
                    report, case, who, observation['outcome'], ref
                )
            _route(report, case, who, observation)
            _case_inputs(report, case, who, observation)
            if ref is not None:
                theirs = ref['identity']['versions']['duckdb']
                mine = observation['identity']['versions']['duckdb']
                if mine != theirs:
                    _differ(
                        report,
                        f'{case["id"]}: {who}',
                        'identity.versions.duckdb',
                        theirs,
                        mine,
                    )
        python = observed.get('python-core', {}).get(case['id'])
        rust = observed.get('rust-host', {}).get(case['id'])
        if python is not None and rust is not None:
            _core_agree(report, case, python, rust)
        if not gating:
            moved = report.failures[before:]
            del report.failures[before:]
            for text in moved:
                report.note(f'diagnostic: {text}')
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--cases', type=Path, action='append', required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('observations', type=Path, nargs='+')
    arguments = parser.parse_args(argv)
    report = compare(
        arguments.cases, arguments.reference, arguments.observations
    )
    for text in report.notes:
        print(text)
    for text in report.failures:
        print(f'DIFFERS {text}')
    if report.failures:
        print(f'compare: {len(report.failures)} differences')
        return 1
    print('compare: no differences')
    return 0


if __name__ == '__main__':
    sys.exit(main())
