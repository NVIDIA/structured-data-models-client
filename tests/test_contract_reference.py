# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The committed reference observations and their environment (T1.13).

``contract/reference/<commit>.jsonl`` holds one ``python-ref`` observation of
every case in ``contract/cases``, recorded through ``run_python.py`` in
``python-ref`` mode from a clean tree, the reference guard passing before the
first case and after the last; ``contract/reference-env.txt`` records the
environment they were recorded in. Every line validates against the
observation schema and carries the guard's result and the full identity of
that recording: the reference commit (the file's name), the environment's
digest (the distributions ``reference-env.txt`` lists), the platform, cell
and versions, a clean checkout, the old route, and the case and bases files
as they are now, so a case changed without its observation re-recorded
fails. Each rule fails on missing, empty or retyped evidence, seeded below.

``tests/test_contract_cases.py`` already validates each line against the
schema and runs ``compare.py`` over the file, which reads it as the
reference; CI's ``rust-test`` job compares ``rust-host`` with it. Reference
immutability (a line may be added but not changed or removed) is T1.14's
check, with the ratchet and the protected paths.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / 'contract/runners/run_python.py'
CASES = ROOT / 'contract/cases'
BASES = ROOT / 'contract/bases.yaml'
REFERENCE = ROOT / 'contract/reference'
ENVIRONMENT = ROOT / 'contract/reference-env.txt'

#: The header keys of ``reference-env.txt``, each with its form.
HEADER = {
    'reference': re.compile(r'[0-9a-f]{40}'),
    'platform': re.compile(r'[a-z0-9_]+-[a-z0-9_]+'),
    'python': re.compile(r'[0-9]+\.[0-9]+\.[0-9]+'),
    'environment': re.compile(r'[0-9a-f]{64}'),
}
DISTRIBUTION = re.compile(r'([a-z0-9][a-z0-9._-]*)==(\S+)')
#: The versions an observation's identity records, by distribution; Python's
#: is the header's.
VERSIONED = ('pandas', 'pyarrow', 'numpy', 'duckdb')
SHA = re.compile(r'[0-9a-f]{40}')


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location('run_python', RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_python = _runner()


def read_environment(
    path: Path,
) -> tuple[dict[str, str], list[str], list[str]]:
    r"""``reference-env.txt``'s header, its distribution lines and every
    problem with them."""
    problems: list[str] = []
    if not path.is_file():
        return {}, [], [f'{path.name}: missing']
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return {}, [], [f'{path.name}: not UTF-8']
    lines = [line for line in text.splitlines() if not line.startswith('#')]
    while lines and lines[0] == '':
        del lines[0]
    if '' not in lines:
        return {}, [], [f'{path.name}: no blank line before the distributions']
    blank = lines.index('')
    header: dict[str, str] = {}
    for line in lines[:blank]:
        key, colon, value = line.partition(': ')
        if not colon or key not in HEADER or key in header:
            problems.append(f'{path.name}: header line {line[:40]!r}')
            continue
        if not HEADER[key].fullmatch(value):
            problems.append(f'{path.name}: {key}: not of its form')
        header[key] = value
    for key in sorted(set(HEADER) - set(header)):
        problems.append(f'{path.name}: no {key}')
    distributions = lines[blank + 1 :]
    if not distributions:
        problems.append(f'{path.name}: no distributions')
    for line in distributions:
        if not DISTRIBUTION.fullmatch(line):
            problems.append(f'{path.name}: distribution line {line[:40]!r}')
    if distributions != sorted(distributions):
        problems.append(f'{path.name}: the distributions are not sorted')
    found = hashlib.sha256('\n'.join(distributions).encode()).hexdigest()
    if header.get('environment') != found:
        problems.append(
            f'{path.name}: environment: not the digest of its distributions'
        )
    return header, distributions, problems


def _at(document: Any, *path: str) -> Any:
    r"""The value at ``path``, or ``_MISSING``."""
    for key in path:
        if not isinstance(document, dict) or key not in document:
            return _MISSING
        document = document[key]
    return document


_MISSING = object()


def _identity(
    observation: Any,
    header: dict[str, str],
    versions: dict[str, str],
    digests: dict[str, str],
) -> list[str]:
    r"""Each way an observation's identity is not the recording's."""
    wanted: list[tuple[tuple[str, ...], Callable[[Any], bool]]] = [
        (('runner',), lambda v: v == 'python-ref'),
        (('cell',), lambda v: v == 'ref-pandas3'),
        (('platform',), lambda v: v == header.get('platform')),
        (
            ('commit', 'sha'),
            lambda v: isinstance(v, str) and SHA.fullmatch(v) is not None,
        ),
        (('commit', 'clean'), lambda v: v is True),
        (
            ('reference', 'commit', 'sha'),
            lambda v: v == header.get('reference'),
        ),
        (('reference', 'commit', 'clean'), lambda v: v is True),
        (
            ('reference', 'environment'),
            lambda v: v == header.get('environment'),
        ),
        (
            ('reference', 'guard'),
            lambda v: v == {'before': 'passed', 'after': 'passed'},
        ),
        (('native',), lambda v: v is None),
        (('native_loaded',), lambda v: v is False),
        (('operations',), lambda v: v == []),
        (('contract',), lambda v: v is None),
        (('versions', 'python'), lambda v: v == header.get('python')),
        *(
            (('versions', name), lambda v, name=name: v == versions.get(name))
            for name in VERSIONED
        ),
        (('inputs', 'bases'), lambda v: v == digests['']),
    ]
    identity = _at(observation, 'identity')
    problems = [
        f"identity.{'.'.join(path)}: not the recording's"
        for path, holds in wanted
        if (value := _at(identity, *path)) is _MISSING or not holds(value)
    ]
    case = _at(observation, 'case')
    found = _at(identity, 'inputs', 'case')
    if not isinstance(case, str) or found != digests.get(case):
        problems.append('identity.inputs.case: not the case file as it is')
    return problems


def reference_problems(
    reference: Path = REFERENCE,
    environment: Path = ENVIRONMENT,
    cases: Path = CASES,
) -> list[str]:
    r"""Every problem with the committed reference observations."""
    header, distributions, problems = read_environment(environment)
    versions = {}
    for line in distributions:
        match = DISTRIBUTION.fullmatch(line)
        if match is not None:
            versions[match[1]] = match[2]
    files = sorted(reference.iterdir()) if reference.is_dir() else []
    name = f'{header.get("reference")}.jsonl'
    if [path.name for path in files] != [name]:
        problems.append(f'{reference.name}: not exactly one file, {name}')
        return problems
    loaded = run_python.load_cases([cases])
    digests = {case['id']: digest for case, digest in loaded}
    digests[''] = run_python.digest(BASES.read_bytes())
    validate = run_python.validator('observation')
    try:
        lines = files[0].read_text(encoding='utf-8').splitlines()
    except UnicodeDecodeError:
        return [*problems, f'{name}: not UTF-8']
    if not lines:
        problems.append(f'{name}: no observations')
    seen: dict[str, int] = {}
    for number, line in enumerate(lines, 1):
        where = f'{name}:{number}'
        try:
            observation = json.loads(line)
        except (ValueError, RecursionError):
            problems.append(f'{where}: not JSON it reads')
            continue
        invalid = run_python._invalid(validate, observation)
        if invalid is not None:
            problems.append(f'{where}: not a valid observation: {invalid}')
        problems.extend(
            f'{where}: {problem}'
            for problem in _identity(observation, header, versions, digests)
        )
        case = _at(observation, 'case')
        if isinstance(case, str):
            seen[case] = seen.get(case, 0) + 1
    for case, _ in loaded:
        count = seen.pop(case['id'], 0)
        if count != 1:
            problems.append(f'{case["id"]}: {count} reference observations')
    problems.extend(f'{case}: not a case' for case in sorted(seen))
    return problems


def test_the_reference_observations_carry_the_recording_s_identity() -> None:
    assert reference_problems() == []
    header, distributions, _ = read_environment(ENVIRONMENT)
    # The reference T0.3 recorded (phase-0/reference-environment.md), and
    # the cell's versions among the distributions.
    assert header['reference'].startswith('b413e9a')
    for line in ('pandas==3.0.6', 'pyarrow==25.0.1', 'numpy==2.5.3'):
        assert line in distributions


# --- seeded faults ---------------------------------------------------------------


def _copy(tmp_path: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    reference = tmp_path / 'reference'
    shutil.copytree(REFERENCE, reference)
    environment = tmp_path / 'reference-env.txt'
    shutil.copy(ENVIRONMENT, environment)
    (path,) = reference.iterdir()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    return reference, environment, lines


def _write(reference: Path, lines: list[Any]) -> None:
    (path,) = reference.iterdir()
    path.write_text(''.join(json.dumps(line) + '\n' for line in lines))


def _set(path: str, value: Any) -> Callable[[list[dict[str, Any]]], None]:
    r"""Sets the field at the dotted ``path`` of the first line; a value of
    ``_MISSING`` removes it."""
    *parents, last = path.split('.')

    def change(lines: list[dict[str, Any]]) -> None:
        at = lines[0]
        for key in parents:
            at = at[key]
        if value is _MISSING:
            del at[last]
        else:
            at[last] = copy.deepcopy(value)

    return change


LINE_FAULTS: dict[str, tuple[Callable[[list[dict[str, Any]]], None], str]] = {
    'another runner': (_set('identity.runner', 'python-old'), 'runner'),
    'no cell': (_set('identity.cell', None), 'identity.cell'),
    'another platform': (
        _set('identity.platform', 'linux-x86_64'),
        'identity.platform',
    ),
    'an unclean checkout': (
        _set('identity.commit.clean', False),
        'identity.commit.clean',
    ),
    'a commit retyped': (_set('identity.commit.sha', 7), 'identity.commit.sha'),
    'no reference': (_set('identity.reference', None), 'identity.reference'),
    'another reference commit': (
        _set('identity.reference.commit.sha', 'f' * 40),
        'identity.reference.commit.sha',
    ),
    'an unclean reference': (
        _set('identity.reference.commit.clean', False),
        'identity.reference.commit.clean',
    ),
    'another environment': (
        _set('identity.reference.environment', '0' * 64),
        'identity.reference.environment',
    ),
    'no environment': (
        _set('identity.reference.environment', _MISSING),
        'identity.reference.environment',
    ),
    'no guard result': (
        _set('identity.reference.guard', _MISSING),
        'identity.reference.guard',
    ),
    'a failed guard': (
        _set('identity.reference.guard.after', 'failed'),
        'identity.reference.guard',
    ),
    'a guard retyped': (
        _set('identity.reference.guard', 'passed'),
        'identity.reference.guard',
    ),
    'an empty guard': (
        _set('identity.reference.guard', {}),
        'identity.reference.guard',
    ),
    'the core loaded': (
        _set('identity.native_loaded', True),
        'identity.native_loaded',
    ),
    'native_loaded unrecorded': (
        _set('identity.native_loaded', None),
        'identity.native_loaded',
    ),
    'a core operation': (
        _set('identity.operations', ['validate_query']),
        'identity.operations',
    ),
    'a contract version': (_set('identity.contract', 9), 'identity.contract'),
    'another pandas': (
        _set('identity.versions.pandas', '2.2.3'),
        'identity.versions.pandas',
    ),
    'no Python version': (
        _set('identity.versions.python', _MISSING),
        'identity.versions.python',
    ),
    'no duckdb version': (
        _set('identity.versions.duckdb', _MISSING),
        'identity.versions.duckdb',
    ),
    'a stale case digest': (
        _set('identity.inputs.case', '0' * 64),
        'identity.inputs.case',
    ),
    'a stale bases digest': (
        _set('identity.inputs.bases', '0' * 64),
        'identity.inputs.bases',
    ),
    'no bases digest': (
        _set('identity.inputs.bases', None),
        'identity.inputs.bases',
    ),
    'no identity': (_set('identity', _MISSING), 'identity.runner'),
    'a line missing': (lambda lines: lines.pop(), '0 reference observations'),
    'a line twice': (
        lambda lines: lines.append(copy.deepcopy(lines[0])),
        '2 reference observations',
    ),
    'a case that does not exist': (
        _set('case', 's0.accept.no-such-case'),
        's0.accept.no-such-case: not a case',
    ),
    'a key the schema does not know': (
        _set('extra', 1),
        "fails 'additionalProperties'",
    ),
    'a very wide object': (
        _set('wide', {f'k{n}': n for n in range(100_000)}),
        "fails 'additionalProperties'",
    ),
}


@pytest.mark.parametrize('fault', sorted(LINE_FAULTS))
def test_a_seeded_fault_in_a_line_fails(tmp_path: Path, fault: str) -> None:
    reference, environment, lines = _copy(tmp_path)
    assert reference_problems(reference, environment) == []
    change, wanted = LINE_FAULTS[fault]
    change(lines)
    _write(reference, lines)
    problems = reference_problems(reference, environment)
    assert any(wanted in problem for problem in problems), problems


def test_a_line_nested_past_the_parser_s_depth_is_refused_by_name(
    tmp_path: Path,
) -> None:
    reference, environment, lines = _copy(tmp_path)
    (path,) = reference.iterdir()
    deep = 100_000
    text = path.read_text().splitlines()
    text[0] = '[' * deep + ']' * deep
    text[1] = '{"case": ' + '{"a": ' * deep + '0' + '}' * deep + '}'
    text[2] = '{"format": '
    path.write_text('\n'.join(text) + '\n')
    problems = reference_problems(reference, environment)
    for number in (1, 2, 3):
        assert any(
            problem.endswith(f'.jsonl:{number}: not JSON it reads')
            for problem in problems
        ), problems


def _file_fault(
    change: Callable[[Path, Path], None], wanted: str, tmp_path: Path
) -> None:
    reference, environment, _ = _copy(tmp_path)
    assert reference_problems(reference, environment) == []
    change(reference, environment)
    problems = reference_problems(reference, environment)
    assert any(wanted in problem for problem in problems), problems


def _environment_lines(
    edit: Callable[[list[str]], None],
) -> Callable[[Path, Path], None]:
    def change(_: Path, environment: Path) -> None:
        lines = environment.read_text().splitlines()
        edit(lines)
        environment.write_text('\n'.join(lines) + '\n')

    return change


def _header(key: str, value: str | None) -> Callable[[list[str]], None]:
    def edit(lines: list[str]) -> None:
        at = next(n for n, line in enumerate(lines) if line.startswith(key))
        if value is None:
            del lines[at]
        else:
            lines[at] = f'{key}: {value}'

    return edit


def _distribution(lines: list[str]) -> None:
    at = next(n for n, line in enumerate(lines) if line.startswith('pandas=='))
    lines[at] = 'pandas==3.0.7'


def _no_distributions(lines: list[str]) -> None:
    at = next(
        n for n, line in enumerate(lines) if line.startswith('environment')
    )
    del lines[at + 2 :]


def _unsorted(lines: list[str]) -> None:
    lines[-1], lines[-2] = lines[-2], lines[-1]


FILE_FAULTS: dict[str, tuple[Callable[[Path, Path], None], str]] = {
    'no environment file': (
        lambda _, environment: environment.unlink(),
        'reference-env.txt: missing',
    ),
    'an empty environment file': (
        lambda _, environment: environment.write_text(''),
        'no blank line',
    ),
    'no environment digest': (
        _environment_lines(_header('environment', None)),
        'no environment',
    ),
    'an environment digest retyped': (
        _environment_lines(_header('environment', 'sha256')),
        'environment: not of its form',
    ),
    'no reference commit': (
        _environment_lines(_header('reference', None)),
        'no reference',
    ),
    'no Python version': (
        _environment_lines(_header('python', None)),
        'no python',
    ),
    'a distribution changed': (
        _environment_lines(_distribution),
        'not the digest of its distributions',
    ),
    'no distributions': (
        _environment_lines(_no_distributions),
        'no distributions',
    ),
    'the distributions unsorted': (
        _environment_lines(_unsorted),
        'not sorted',
    ),
    'no reference file': (
        lambda reference, _: next(reference.iterdir()).unlink(),
        'not exactly one file',
    ),
    'a second reference file': (
        lambda reference, _: (reference / ('0' * 40 + '.jsonl')).write_text(''),
        'not exactly one file',
    ),
    'the file renamed': (
        lambda reference, _: next(reference.iterdir()).rename(
            reference / ('0' * 40 + '.jsonl')
        ),
        'not exactly one file',
    ),
    'an empty reference file': (
        lambda reference, _: next(reference.iterdir()).write_text(''),
        'no observations',
    ),
}


@pytest.mark.parametrize('fault', sorted(FILE_FAULTS))
def test_a_seeded_fault_in_the_files_fails(tmp_path: Path, fault: str) -> None:
    change, wanted = FILE_FAULTS[fault]
    _file_fault(change, wanted, tmp_path)


def test_every_line_is_checked_not_only_the_first(tmp_path: Path) -> None:
    reference, environment, lines = _copy(tmp_path)
    lines[-1]['identity']['reference']['guard']['before'] = 'failed'
    _write(reference, lines)
    problems = reference_problems(reference, environment)
    assert any(
        f'.jsonl:{len(lines)}: identity.reference.guard' in problem
        for problem in problems
    ), problems
