# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P85.fixtures, beside ``accept_request_fixtures.rs``: the exclusions keep
their rules, and the binding's ``request_fixture_op`` gives every case what
the Rust runner gives (``CONTRACT.md``, "Fixtures")."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

from _fixture_expectations import CASE_ID, expected_case, validate_catalog
from _request import (
    EXCLUSIONS,
    FIXTURE_TEST,
    FIXTURES,
    REPO,
    exclusions,
    recorder,
    request_fixture_cases,
    short,
)

AT_LEAST = 500
ERROR_KEYS = ('class', 'code', 'message', 'details_keys')


def test_the_runner_is_there_and_the_cases_are() -> None:
    assert FIXTURE_TEST.is_file(), FIXTURE_TEST
    cases = request_fixture_cases()
    assert len(cases) >= AT_LEAST, f'only {len(cases)} request fixture cases'
    ids = [str(c.get('id')) for c in cases]
    assert len(ids) == len(set(ids)), 'request fixture case ids repeat'


def test_the_exclusions_keep_their_rules() -> None:
    if not EXCLUSIONS.is_file():
        return
    pairs: list[tuple[str, Any]] = []
    json.loads(
        EXCLUSIONS.read_text(encoding='utf-8'), object_pairs_hook=pairs.extend
    )
    ids = [key for key, _ in pairs]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    assert not repeated, f'listed more than once: {repeated}'
    known = {str(c.get('id')) for c in request_fixture_cases()}
    problems = []
    for case_id, reason in exclusions().items():
        if case_id not in known:
            problems.append(f'{case_id}: not a request fixture case')
        if not isinstance(reason, str) or len(reason.split()) < 3:
            problems.append(
                f'{case_id}: the reason {reason!r} is under three words'
            )
    assert not problems, '\n'.join(problems)


def difference(case: dict[str, Any], got: Any) -> str | None:
    r"""As ``accept_request_fixtures.rs``'s ``difference``."""
    if not isinstance(got, dict):
        return f'gave {short(got)}, not an object'
    given = got.get('warnings')
    if not isinstance(given, list):
        return f'gave {short(got)}, without a warnings list'
    from _request import json_diff

    if 'expected' in case:
        if 'error' in got or 'output' not in got:
            return f'gave {short(got)}, where it records an output'
        found = json_diff(case['expected'], got['output'], '$output')
        if found:
            return found
    if 'expected_error' in case:
        if 'output' in got or not isinstance(got.get('error'), dict):
            return f'gave {short(got)}, where it records an error'
        for key in ERROR_KEYS:
            if key in case['expected_error']:
                found = json_diff(
                    case['expected_error'][key], got['error'].get(key), key
                )
                if found:
                    return f'error {found}'
    if 'expected_warnings' in case:
        found = json_diff(case['expected_warnings'], given, '$warnings')
        if found:
            return found
    return None


def test_the_binding_gives_every_case_what_it_records() -> None:
    module = recorder().module
    skipped = exclusions()
    problems = []
    cases = request_fixture_cases()
    validate_catalog(cases, FIXTURES, skipped)
    for case in cases:
        if case['id'] in skipped:
            continue
        try:
            text = module.request_fixture_op(
                case['operation'], json.dumps(case.get('input'))
            )
            found = difference(expected_case(case, core=True), json.loads(text))
        except Exception as error:  # noqa: BLE001 - reported
            found = f'raised {type(error).__name__}: {short(str(error))}'
        if found:
            problems.append(f'{case["id"]}: {found}')
    assert not problems, f'{len(problems)} case(s) differ:\n' + '\n'.join(
        problems[:50]
    )


def test_the_corrected_case_preserves_the_actual_old_public_oracle() -> None:
    child = subprocess.run(
        [
            sys.executable,
            '-B',
            '-c',
            'import sys, pytest, sdm_relational_engine; '
            'from sdm_client._impl import OLD; '
            'assert OLD; raise SystemExit(pytest.main(sys.argv[1:]))',
            f'tests/parity/test_fixtures.py::test_fixture[{CASE_ID}]',
            '-q',
            '--tb=no',
            '-p',
            'no:cacheprovider',
        ],
        cwd=REPO,
        env={
            **os.environ,
            'SDM_CLIENT_IMPLEMENTATION': 'python',
            'PYTEST_ADDOPTS': '',
            'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert child.returncode == 0, 'the actual old public oracle no longer holds'


def test_actual_corrected_refusal_rejects_a_seeded_wrong_message() -> None:
    case = next(
        case for case in request_fixture_cases() if case['id'] == CASE_ID
    )
    expected = expected_case(case, core=True)
    actual = json.loads(
        recorder().module.request_fixture_op(
            case['operation'], json.dumps(case['input'])
        )
    )
    assert difference(expected, actual) is None
    actual['error']['message'] = 'seeded wrong refusal'
    assert difference(expected, actual) is not None
