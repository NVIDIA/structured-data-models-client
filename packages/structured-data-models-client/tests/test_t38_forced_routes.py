# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public schema behavior across connectors, held to preserved snapshots.

Warning-code prefixes and repeated-key example selection are normalized;
all other recorded outcomes remain strict.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
import schema_scenarios as R

HERE = Path(__file__).resolve().parent
RECORDED = json.loads(
    (HERE / 't38-forced-routes.json').read_text(encoding='utf-8')
)


KEYS = R.keys()


def _require_database(database):
    if database == 'postgres' and not os.environ.get('SDM_POSTGRES_TEST_DSN'):
        if os.environ.get('SDM_REQUIRE_POSTGRES'):
            pytest.fail('SDM_REQUIRE_POSTGRES requires SDM_POSTGRES_TEST_DSN')
        pytest.skip('requires SDM_POSTGRES_TEST_DSN')


def _examples_aside(value: Any, examples: list[str] | None) -> Any:
    r"""``value`` with each ``e.g. <repeated value>`` written alike: which
    of the repeated values a scan meets first is either route's own."""
    if not examples:
        return value
    if isinstance(value, str):
        for example in sorted(examples, key=len, reverse=True):
            value = value.replace(f'e.g. {example}', 'e.g. <repeated>')
        return value
    if isinstance(value, list):
        return [_examples_aside(item, examples) for item in value]
    if isinstance(value, dict):
        return {k: _examples_aside(v, examples) for k, v in value.items()}
    return value


def _differences(expected: Any, actual: Any, at: str = '') -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        found = []
        for name in sorted(set(expected) | set(actual)):
            if name not in actual:
                found.append(f'{at}/{name}: missing on the core route')
            elif name not in expected:
                found.append(f'{at}/{name}: not in the reference')
            else:
                found += _differences(
                    expected[name], actual[name], f'{at}/{name}'
                )
        return found
    if expected != actual:
        return [f'{at}: the reference {expected!r}\n  the core {actual!r}']
    return []


def _compare(key: str, actual: Any) -> None:
    recorded = RECORDED['scenarios'][key]
    examples = recorded['examples']
    expected = _examples_aside(recorded['expected'], examples)
    actual = _examples_aside(actual, examples)
    differences = _differences(expected, actual)
    assert not differences, f'{key} ({recorded["covers"]}):\n' + '\n'.join(
        differences
    )


@pytest.mark.parametrize('key', KEYS)
def test_the_core_answers_as_the_reference(key: str, tmp_path: Path) -> None:
    """Each connector case retains its recorded public behavior."""
    name, database = key.rsplit('@', 1)
    _require_database(database)
    found = R.SCENARIOS[name]
    _compare(key, R.observe(found, database, tmp_path))


# --- controls ---------------------------------------------------------------


def test_every_scenario_is_recorded_on_each_of_its_databases() -> None:
    r"""Control: the recording holds every scenario on each database it
    names, and nothing else."""
    assert sorted(RECORDED['scenarios']) == sorted(KEYS)
    assert RECORDED['families'] == R.FAMILIES
    for key, recorded in RECORDED['scenarios'].items():
        found = R.SCENARIOS[key.rsplit('@', 1)[0]]
        assert recorded['family'] == found.family, key
        assert recorded['covers'] == found.covers, key


def test_every_family_runs_on_three_databases() -> None:
    r"""Exit 2: a case per family on DuckDB, SQLite and Postgres."""
    for family in R.FAMILIES:
        databases = {
            key.rsplit('@', 1)[1]
            for key, recorded in RECORDED['scenarios'].items()
            if recorded['family'] == family
        }
        assert databases == set(R.DATABASES), family


def _answers(expected: Any) -> list[dict[str, Any]]:
    r"""The answers a scenario recorded, one or a step's each."""
    if 'outcome' in expected:
        return [expected]
    return list(expected.values())


def _said(answer: dict[str, Any]) -> str:
    return '\n'.join(text for _, text in answer.get('notices', []))


def _exercises(name: str, expected: Any) -> None:
    r"""The reference's answer is the case the scenario's name says: no
    seed passes for a reason other than its own."""
    first = _answers(expected)[0]
    if name.startswith('time.'):
        assert first['outcome'] == 'schema'
        said = _said(first)
        assert said.startswith('Some time columns were inferred'), said
        override = re.match(r'time\.the-override-for-(\w+?)-', name)
        if override:
            assert f'- {override.group(1)}.' not in said
    elif name.startswith('keys.score.'):
        ids = name.rsplit('.', 1)[1].split('-')
        if 'u007' in ids:
            assert first['outcome'] == 'refusal'
            assert "Cannot score 'u007'" in first['message']
        else:
            assert first['outcome'] == 'payload'
            assert len(first['predict']['rows']) == len(ids)
    elif name.startswith('keys.'):
        users = first.get('tables', {}).get('users', {})
        if name.endswith('.unchecked'):
            assert users['duplicate_keys'] is None
            assert first['notices'] == []
        elif '.holds' in name or name.endswith('in-the-catalog'):
            assert first['outcome'] == 'schema'
            assert users['duplicate_keys'] == 0
            assert first['notices'] == []
        elif 'missing-a-value' in name:
            assert 'no value' in (first.get('message', '') + _said(first))
        else:
            assert 'repeat' in name
            assert (
                'not unique enough' in first.get('message', '')
                or 'already been used' in _said(first)
                or 'repeat a value' in _said(first)
            ), first
    elif name.startswith('refusals.'):
        assert first['outcome'] == 'refusal'
        assert first['code'] == 'INVALID_REQUEST'
        assert 'the request is not one the core accepts' not in first['message']
    elif name.startswith('described.'):
        if name.endswith(('.temporal', '.static')):
            assert first['outcome'] == 'payload'
            assert len(first['predict']['rows']) == 3
        else:
            assert first['outcome'] == 'refusal'
    else:
        assert name.startswith('postgres.')
        assert first['outcome'] == 'schema'
        assert first['tables']


@pytest.mark.parametrize('key', KEYS)
def test_each_scenario_exercises_its_case(key: str) -> None:
    r"""Control (test-author rule on vacuous seeds): the reference's
    recorded answer is the case the scenario names."""
    _exercises(key.rsplit('@', 1)[0], RECORDED['scenarios'][key]['expected'])


def test_a_shadowing_table_differs_from_the_one_to_read() -> None:
    r"""Control: ``public``'s tables of the same names hold other columns,
    so reading them in place of the source's schema shows."""
    for name in (
        'postgres.a-table-public-also-holds',
        'postgres.every-table-when-public-holds-others',
    ):
        recorded = RECORDED['scenarios'][f'{name}@postgres']['expected']
        for table in recorded['tables'].values():
            assert 'other_id' not in table['columns']
            assert table['primary_key'] is not None


def test_the_core_s_warning_prefix_is_set_aside_and_nothing_else() -> None:
    r"""Control: :func:`bare` takes off one ``[CODE] `` and keeps the rest,
    so a notice the core words otherwise still differs."""
    assert R.bare('[KEY_REPEATS] users.user_id identifies') == (
        'users.user_id identifies'
    )
    assert R.bare('[INVALID_REQUEST] x') == 'x'
    assert R.bare('Some time columns [X] y') == 'Some time columns [X] y'
    assert R.bare('[not a code] y') == '[not a code] y'
