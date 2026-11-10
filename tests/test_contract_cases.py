# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Maintained case input validation, parser boundaries and timezone-safe fixtures."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent


RUNNER = ROOT / 'contract/runners/run_python.py'


CASES = ROOT / 'contract/cases'


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location('run_python', RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_python = _runner()


_ZONE = re.compile(r'time\s*_?\s*zone', re.IGNORECASE)


_GLOBAL_ZONE = re.compile(
    r"^\s*SET\s+GLOBAL\s+TimeZone\s*(=|TO)\s*'[^']+'\s*;?\s*$",
    re.IGNORECASE,
)


def _zone_problems(where: str, statements: list[str]) -> list[str]:
    r"""Each statement of ``statements`` that names the time zone other
    than as ``SET GLOBAL TimeZone = '...'``."""
    return [
        f'{where}: {statement!r}'
        for statement in statements
        if _ZONE.search(statement) and not _GLOBAL_ZONE.match(statement)
    ]


_ZONED = re.compile(
    r'timestamptz|timestamp\s+with\s+time\s+zone', re.IGNORECASE
)


def _unzoned(where: str, statements: list[str]) -> list[str]:
    r"""``where``, when its statements hold a zoned timestamp and none sets
    the zone with ``SET GLOBAL TimeZone`` (C18)."""
    zoned = any(_ZONED.search(each) for each in statements)
    if zoned and not any(_GLOBAL_ZONE.match(each) for each in statements):
        return [f'{where}: zoned timestamps with no zone set']
    return []


def test_a_case_sets_its_time_zone_for_every_cursor() -> None:
    # The source reads through cursors, which take the process's local zone
    # unless the database's is set: a session SET applies to the setup's
    # connection alone, so the reference recorded the recorder's machine
    # zone (time.timestamptz, found by the bundle's landing CI). Every case,
    # inventory entry and refusal-map probe setup that names the zone sets
    # it with SET GLOBAL, so no answer depends on the recorder's TZ.
    problems = []
    named = 0
    for case, _ in run_python.load_cases([CASES]):
        setup = case['input'].get('setup', [])
        named += sum(bool(_ZONE.search(each)) for each in setup)
        problems += _zone_problems(case['id'], setup)
    # C18: a base setting the zone sets it for every cursor too, and
    # a case holding zoned timestamps sets the zone, so its answers never
    # depend on the recorder's.
    bases = run_python.load_bases()
    for name, statements in bases.items():
        problems += _zone_problems(f'base {name}', statements)
    for case, _ in run_python.load_cases([CASES]):
        problems += _unzoned(
            case['id'],
            [*bases[case['input']['base']], *case['input'].get('setup', [])],
        )
    assert problems == []
    assert named, 'no case names the zone: the check reads nothing'
    assert any(
        _ZONED.search(each)
        for case, _ in run_python.load_cases([CASES])
        for each in case['input'].get('setup', [])
    ), 'no case holds zoned timestamps: the C18 rule reads nothing'
    assert _unzoned('seed', ['ALTER TABLE orders ALTER ts TYPE TIMESTAMPTZ'])
    assert _unzoned('seed', ["SELECT TIMESTAMP WITH TIME ZONE '2024-01-01'"])
    assert not _unzoned(
        'seed',
        [
            "SET GLOBAL TimeZone = 'UTC'",
            'ALTER TABLE orders ALTER ts TYPE TIMESTAMPTZ',
        ],
    )
    # Seeded: every other way to name it fails, and the global form passes.
    for statement in (
        "SET TimeZone = 'America/New_York'",
        "SET SESSION TimeZone = 'UTC'",
        "set timezone to 'UTC'",
        'RESET TimeZone',
        "SET LOCAL TimeZone = 'UTC'",
        "SELECT set_config('TimeZone', 'UTC')",
        "SET GLOBAL TimeZone = 'UTC'; SET TimeZone = 'Asia/Tokyo'",
    ):
        assert _zone_problems('seed', [statement]), statement
    for statement in (
        "SET GLOBAL TimeZone = 'America/New_York'",
        "set global timezone to 'UTC'",
        'ALTER TABLE orders ALTER ts TYPE TIMESTAMPTZ',
    ):
        assert not _zone_problems('seed', [statement]), statement


_VECTOR_SETUP_KEYS = ('setup', 'witness')


_VECTOR_ZONE_PENDING: set[str] = set()


_SESSION_ZONE = re.compile(
    r"^\s*SET\s+(SESSION\s+)?TimeZone\s*(=|TO)\s*'[^']+'\s*;?\s*$",
    re.IGNORECASE,
)


def _vector_zone_problems(where: str, statements: list[str]) -> list[str]:
    r""":func:`_zone_problems`, but a plain session ``SET TimeZone`` is
    allowed in a setup that also sets the database's zone with ``SET
    GLOBAL``: every cursor then takes the global zone, and the session
    statement only sets the setup's own connection, as a vector may do on
    purpose to show that the source does not read in it
    (session and verbatim timestamp scenarios). Any
    other form (RESET, LOCAL, set_config, a compound statement) is still a
    problem, global zone or not."""
    pinned = any(_GLOBAL_ZONE.match(each) for each in statements)
    return _zone_problems(
        where,
        [
            each
            for each in statements
            if not (pinned and _SESSION_ZONE.match(each))
        ],
    )


def _vector_setups(node: Any, where: str) -> list[tuple[str, list[str]]]:
    r"""Each setup ``node`` holds under:data:`_VECTOR_SETUP_KEYS`, with
    the id of the nearest enclosing object that has one."""
    found: list[tuple[str, list[str]]] = []
    if isinstance(node, dict):
        where = str(node.get('id', where))
        for key, value in node.items():
            if key in _VECTOR_SETUP_KEYS and isinstance(value, (str, list)):
                statements = [value] if isinstance(value, str) else value
                found.append(
                    (where, [s for s in statements if isinstance(s, str)])
                )
            found += _vector_setups(value, where)
    elif isinstance(node, list):
        for each in node:
            found += _vector_setups(each, where)
    return found


def test_a_vector_sets_its_time_zone_for_every_cursor() -> None:
    # Setup and source reads use separate cursors. Set the database-wide
    # timezone so results do not depend on each cursor's process-local zone.
    problems = set()
    named = 0
    for path in sorted((ROOT / 'contract').glob('*-vectors.json')):
        document = json.loads(path.read_text(encoding='utf-8'))
        for vector, setup in _vector_setups(document, '?'):
            named += sum(bool(_ZONE.search(each)) for each in setup)
            problems.update(
                _vector_zone_problems(f'{path.name} {vector}', setup)
            )
    assert problems == _VECTOR_ZONE_PENDING
    assert named > len(_VECTOR_ZONE_PENDING), 'no vector sets its zone'
    # Seeded: a session SET passes only beside a global one, in the same
    # setup; nothing else does.
    pinned = "SET GLOBAL TimeZone = 'UTC'"
    for setup in (
        ["SET TimeZone = 'America/New_York'"],
        ["SET SESSION TimeZone = 'UTC'"],
        [pinned, 'RESET TimeZone'],
        [pinned, "SET LOCAL TimeZone = 'UTC'"],
        [pinned, "SELECT set_config('TimeZone', 'UTC')"],
        [pinned, "SET GLOBAL TimeZone = 'UTC'; SET TimeZone = 'Asia/Tokyo'"],
    ):
        assert _vector_zone_problems('seed', setup), setup
    for setup in (
        [pinned, "SET TimeZone = 'America/New_York'"],
        ["SET TimeZone='America/New_York'", pinned],
        [pinned, "set session timezone to 'UTC'"],
        [pinned],
    ):
        assert not _vector_zone_problems('seed', setup), setup


def test_every_committed_case_validates_with_a_unique_id() -> None:
    cases = run_python.load_cases([CASES])
    ids = [case['id'] for case, _ in cases]
    assert len(ids) == len(set(ids)) >= 2
    s0 = [case for case, _ in cases if case['slice'] == 'S0']
    assert {case['expect']['outcome'] for case in s0} == {'success', 'refusal'}


def _holds_packed_date(query: str) -> bool:
    r"""Whether ``query`` holds a date literal with a run of five digits or
    more (D8): a ``DATE`` token, three runs of digits joined by one of
    ``/``, ``-`` or ``.``, with such a run, or followed by a ``TIME`` token,
    three runs joined by ``:``, with one."""
    words = re.split(r'[\s(),]', query)

    def runs(word: str, separators: str) -> list[int] | None:
        sep = next((c for c in word if c in separators), None)
        if sep is None:
            return None
        parts = word.split(sep)
        if len(parts) != 3 or not all(
            part.isascii() and part.isdigit() for part in parts
        ):
            return None
        return [len(part) for part in parts]

    for i, word in enumerate(words):
        date = runs(word, '/-.')
        if date is None:
            continue
        time = runs(words[i + 1], ':') if i + 1 < len(words) else None
        if any(n >= 5 for n in date + (time or [])):
            return True
    return False


def test_the_s0_boundary_cases_sit_on_their_boundaries() -> None:
    # boundary cases, held to the shapes their `why` states.
    cases = {
        case['id']: case for case, _ in run_python.load_cases([CASES / 's0'])
    }

    def pql(name: str) -> str:
        text: str = cases[f's0.boundary.{name}']['input']['pql']
        return text

    # Python's 4,300-digit limit, counting leading zeros, on both sides.
    for name, digits in (('digits-4300', 4300), ('digits-4301', 4301)):
        literal = pql(name).split('orders.price > ')[1].split(',')[0]
        assert literal == '0' * (digits - 1) + '1'
    # Past the 64 KB control cap.
    assert len(pql('start-past-64kb').encode()) > 64 * 1024
    # Below the depth the reference reaches before RecursionError (about
    # 461 in the runner on Python 3.12) and the core's MAX_DEPTH (999).
    deepest = pql('deepest')
    assert deepest.count('(' * 400) == 1 and '(' * 401 not in deepest
    # The two date cases with a run of five digits or more (D8's packed
    # date and DT-9's long field) stay diagnostic (a changed case file
    # needs an approved entry). the core refuses such a literal
    # by name where the reference reads it, so no acceptance case may
    # expect it to succeed: each S0 case holding one is diagnostic, under a
    # register entry, or expects the refusal.
    packed = {
        case['id']: case
        for case in cases.values()
        if _holds_packed_date(case['input']['pql'])
    }
    assert {
        's0.boundary.date-packed',
        's0.boundary.date-long-field',
    } <= set(packed)
    for case in packed.values():
        if case['status'] == 'diagnostic':
            assert case.get('register'), case['id']
        else:
            assert case['expect']['outcome'] == 'refusal', case['id']
            assert case['expect'].get('code') == 'INVALID_REQUEST', case['id']
    # Escapes as written in the query text, a backslash and its letters: a
    # \\u escape and no surrogate in the accepted case; an escaped pair and
    # a lone escaped surrogate in the two cases register PQL-1 holds.
    escapes = pql('escapes')
    assert '\\u00e9' in escapes and '\\ud' not in escapes
    assert escapes.isascii()
    assert '\\ud83d\\ude00' in pql('escape-surrogate-pair')
    lone = pql('escape-lone-surrogate')
    assert '\\ud83d' in lone and '\\ude' not in lone
    # 999 levels exactly: one more is the core's first refusal.
    depth_999 = pql('depth-999')
    assert depth_999.count('(' * 999) == 1 and '(' * 1000 not in depth_999


def test_both_loaders_refuse_an_unknown_key_inside_the_call_arguments(
    tmp_path: Path,
) -> None:
    # sdm-testkit's tests/s0_cases.rs refuses the same key.
    case = json.loads((CASES / 's0/s0.accept.sum-window.json').read_text())
    case['input']['materialize']['ignored_typo'] = True
    path = tmp_path / 'case.json'
    path.write_text(json.dumps(case))
    with pytest.raises(run_python.RunnerFailure, match='ignored_typo|fails'):
        run_python.load_cases([path])


def _changed(
    case: dict[str, Any], path: list[Any], value: Any
) -> dict[str, Any]:
    changed = copy.deepcopy(case)
    at = changed
    for key in path[:-1]:
        at = at[key]
    at[path[-1]] = copy.deepcopy(value)
    return changed


MALFORMED = ROOT / 'contract/tests/malformed-cases.json'


def test_a_case_nested_past_the_parser_s_depth_is_refused_by_name(
    tmp_path: Path,
) -> None:
    # sdm-testkit's loader refuses it by serde_json's depth limit; the Python
    # loader refused nothing and raised RecursionError (probe).
    case = json.loads(
        (ROOT / 'contract/cases/s1/base.adventureworks.json').read_text()
    )
    case['expect'] = {
        'outcome': 'refusal',
        'stage': 'predict',
        'error': 'E',
        'code': None,
    }
    deep = 100_000
    path = tmp_path / 'case.json'
    path.write_text(
        json.dumps(case).replace(
            '"code": null', '"code": ' + '[' * deep + ']' * deep
        )
    )
    with pytest.raises(run_python.RunnerFailure, match='not a case'):
        run_python.load_cases([path])
    path.write_text('{"format": ')
    with pytest.raises(run_python.RunnerFailure, match='not a case'):
        run_python.load_cases([path])


def test_both_loaders_refuse_the_same_malformed_edits_and_relationships(
    tmp_path: Path,
) -> None:
    # sdm-testkit's tests/s0_cases.rs loads the same changes with the Rust
    # loader, which must refuse and take the same ones.
    listed = json.loads(MALFORMED.read_text())
    case = json.loads((ROOT / listed['case']).read_text())
    assert len(listed['refused']) >= 54
    path = tmp_path / 'case.json'
    path.write_text(json.dumps(case))
    assert run_python.load_cases([path])
    for change in listed['refused']:
        path.write_text(
            json.dumps(_changed(case, change['path'], change['value']))
        )
        with pytest.raises(run_python.RunnerFailure, match='not a case'):
            run_python.load_cases([path])
    for change in listed['accepted']:
        path.write_text(
            json.dumps(_changed(case, change['path'], change['value']))
        )
        assert run_python.load_cases([path]), change['why']
    assert listed['omitted']
    for change in listed['omitted']:
        omitted = copy.deepcopy(case)
        at = omitted
        for key in change['path'][:-1]:
            at = at[key]
        del at[change['path'][-1]]
        path.write_text(json.dumps(omitted))
        assert run_python.load_cases([path]), change['why']


def test_an_index_is_an_integer_literal_of_any_size_or_text(
    tmp_path: Path,
) -> None:
    # The same literals sdm-testkit's tests/s0_cases.rs reads.
    literals = json.loads(
        (ROOT / 'contract/tests/index_literals.json').read_text()
    )
    case = json.loads((CASES / 's0/s0.accept.sum-window.json').read_text())
    case['input']['materialize']['indices'] = ['@index@']
    template = json.dumps(case)

    def load(literal: str) -> Any:
        path = tmp_path / 'case.json'
        path.write_text(template.replace('"@index@"', literal))
        [(loaded, _)] = run_python.load_cases([path])
        return loaded

    for literal in literals['accepted']:
        loaded = load(literal)
        [index] = loaded['input']['materialize']['indices']
        assert index == json.loads(literal), literal
        assert type(index) is type(json.loads(literal))
        # As a decimal index, every digit is kept.
        loaded['input']['materialize']['indices_as'] = 'decimal'
        if isinstance(index, int):
            [decimal] = run_python._materialize_arguments(
                loaded['input']['materialize']
            )['indices']
            assert decimal == Decimal(literal) and str(decimal) == literal
    for literal in literals['refused']:
        with pytest.raises(run_python.RunnerFailure):
            load(literal)
