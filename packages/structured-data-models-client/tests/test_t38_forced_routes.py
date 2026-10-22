# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""T3.8: with every core route forced, the client answers as the old client.

Written by the package's independent test author from the brief, the
design's T3.8 row, the carried rows P2-T2.16-9, -10 and -11 and the
reference, before the implementation. Phase 2's flip dry run
(``docs/design/rust-core/phase-2/evidence/flip/``) failed 118 client tests
with every core route forced; each scenario of
``t38_forced_routes_record.py`` is one input of one of their families, as a
user meets it through the public API, on DuckDB, SQLite and Postgres, and
``t38-forced-routes.json`` holds what the reference answered for it,
recorded through the same :func:`observe` this test calls. Here each runs
with every path of ``_routing.CORE_ROUTED`` forced, and its answer must
equal the reference's:

- ``time`` (P2-T2.16-9, exit 2's time-column family): the time columns and
  the "Some time columns..." notice, in the reference's words;
- ``keys`` (P2-T2.16-10, exit 2's key family): keys, duplicate counts, the
  key notices and the refusals at the 1% and 99% thresholds, a missing
  value, ``check_keys=False``, keys past 2^53, and "Cannot score" for an id
  two rows hold;
- ``refusals`` (P2-T2.16-10 and -11): a refused source call in the
  reference's words, never "the request is not one the core accepts";
- ``postgres`` (P2-T2.16-11, exit 2's Postgres family): a
  ``PostgresSource`` read in the schema it names, also when ``public`` holds
  a table of the same name;
- ``described`` (P2-T2.16-11's ``RouteError``): ``materialize`` of a schema
  built by hand, written out column by column, or described on the old
  route, as the reference materializes any ``Schema``.

What a scenario compares (``observe``): a schema's tables (source name,
key, time column, duplicate count, columns) and relationships; the
family's notices, class and words; a refusal's class, code and words; a
payload's task, scored rows and their history, and the context's tables
and columns (a context row is the draw's). Two readings, both the
decisions' and not the package's to change:

- every warning the core gives starts ``[CODE] `` (decisions E41(i) and
  E59(d)); the words after it are compared (:func:`bare`);
- a repeated key's example (``e.g. ...``) is whichever repeated value a
  scan meets first on either route, so any of the scenario's repeated
  values is the reference's.

Gating: ``t38-pending.json``'s ``pending`` names, as ``py:<scenario>@<db>``,
the keys whose comparison failed when written. Each must still fail
(strict); once one passes it fails, naming itself, until it is taken out.
``pending`` stays within ``authored``, the test author's. Without
``SDM_POSTGRES_TEST_DSN`` the Postgres keys skip, and with
``SDM_REQUIRE_POSTGRES`` set they fail instead.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from sdm_client import _routing

pytest.importorskip('duckdb')
pytest.importorskip('adbc_driver_sqlite.dbapi')

HERE = Path(__file__).resolve().parent
PENDING = HERE / 't38-pending.json'
RECORDED = json.loads(
    (HERE / 't38-forced-routes.json').read_text(encoding='utf-8')
)
#: Every path this build can run on the core: the flip's forcing.
CORE = ','.join(sorted(_routing.CORE_ROUTED))
ENGINE = importlib.util.find_spec('sdm_relational_engine') is not None


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _module('t38_forced_routes_record', HERE / 't38_forced_routes_record.py')
KEYS = R.keys()


def _gating() -> dict[str, Any]:
    return json.loads(PENDING.read_text(encoding='utf-8'))


def held(key: str, check: Any) -> None:
    r"""Runs ``check``: it must pass, or, while ``py:<key>`` is pending,
    fail."""
    if f'py:{key}' not in _gating()['pending']:
        check()
        return
    try:
        check()
    except Exception:  # noqa: BLE001 - pending: failing, as when written
        return
    raise AssertionError(
        f'{key} passes now: take py:{key} out of '
        'packages/structured-data-models-client/tests/t38-pending.json '
        'pending'
    )


def _needs(database: str) -> None:
    if database == 'postgres' and not os.environ.get('SDM_POSTGRES_TEST_DSN'):
        if os.environ.get('SDM_REQUIRE_POSTGRES'):
            pytest.fail(
                'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
            )
        pytest.skip('SDM_POSTGRES_TEST_DSN is not set')


def _needs_the_engine(name: str) -> None:
    if not ENGINE and name.startswith('described.old-route.'):
        pytest.skip('the old route is not installed')


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
def test_the_core_answers_as_the_reference(
    key: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    r"""Exits 1 and 2: with every core route forced, each scenario's answer
    is the reference's (the family and input its ``covers`` line names)."""
    name, database = key.rsplit('@', 1)
    _needs(database)
    _needs_the_engine(name)
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    found = R.SCENARIOS[name]
    held(key, lambda: _compare(key, R.observe(found, database, tmp_path)))


@pytest.mark.skipif(not ENGINE, reason='the old route is not installed')
@pytest.mark.parametrize('key', KEYS)
def test_the_old_route_answers_as_recorded(
    key: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    r"""Control (exit 1's unforced run): the old route in this tree gives
    the recorded answer, so a difference above is the core's."""
    name, database = key.rsplit('@', 1)
    _needs(database)
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    _compare(key, R.observe(R.SCENARIOS[name], database, tmp_path))


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


def test_the_pending_list_only_shrinks() -> None:
    r"""Control: ``pending`` names only keys the test author wrote, and
    ``authored`` is every scenario on each of its databases."""
    gating = _gating()
    assert set(gating['pending']) <= set(gating['authored'])
    assert set(gating['authored']) == {f'py:{key}' for key in KEYS}


def test_the_core_s_warning_prefix_is_set_aside_and_nothing_else() -> None:
    r"""Control: :func:`bare` takes off one ``[CODE] `` and keeps the rest,
    so a notice the core words otherwise still differs."""
    assert R.bare('[KEY_REPEATS] users.user_id identifies') == (
        'users.user_id identifies'
    )
    assert R.bare('[INVALID_REQUEST] x') == 'x'
    assert R.bare('Some time columns [X] y') == 'Some time columns [X] y'
    assert R.bare('[not a code] y') == '[not a code] y'
