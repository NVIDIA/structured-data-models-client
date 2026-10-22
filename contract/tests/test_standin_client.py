# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The forwarding provider's own tests (``contract/runners/standin_client.py``,
T3.3): ``open`` serves a case's capture under ``captures/<leg>/<case>/``
through ``sdm-standin``, its ``provider(step, keys)`` answering as the
leg's provider would, a step judged live only stopping the run with its
code in the verdict, and a case with no capture named pending live. The
binary is ``SDM_STANDIN``, else Cargo's build; where neither exists the
tests have nothing to hold and return. The acceptance tests are the test
author's.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load(name: str, path: Path) -> ModuleType:
    key = f'standin_client_tests_{name}'
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


client = _load('client', ROOT / 'contract/runners/standin_client.py')
captures = _load('captures', ROOT / 'contract/runners/captures.py')
fixtures = _load('fixtures', HERE / 't33_fixtures.py')


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path | None:
    try:
        client.binary_path()
    except client.StandInFailure:
        return None
    for name in captures.PRIVATE:
        monkeypatch.delenv(name, raising=False)
    found = fixtures.capture()
    captures.write(tmp_path / 'databricks-sql' / found['case'], found)
    return tmp_path


def test_the_session_answers_as_the_leg_s_provider(root: Path | None) -> None:
    if root is None:
        return
    case = {'id': fixtures.capture()['case']}
    session = client.open('databricks-sql', case, None, root)
    try:
        assert session.names['catalog'] == '{catalog}'
        table = session.provider(fixtures.scan_users())
        assert sorted(table.column('user_id').to_pylist()) == list(range(8))
        with pytest.raises(client.StandInFailure, match='seed'):
            session.provider(fixtures.sample_users(seed=fixtures.DRAW_SEED + 1))
        describe = json.loads(json.dumps(fixtures.scan_users(step_id=3)))
        describe['request'] = {
            'request': 'describe',
            'tables': [{'source': ['users']}],
            'part': 'columns',
        }
        with pytest.raises(client.LiveOnly) as stopped:
            session.provider(describe)
        assert stopped.value.code == 'B1'
        assert session.verdict() == {'live_only': 'B1', 'flagged': []}
    finally:
        session.close()


def test_a_case_with_no_capture_is_pending_live(root: Path | None) -> None:
    if root is None:
        return
    with pytest.raises(client.StandInFailure, match='pending live'):
        client.open('databricks-sql', {'id': 'no.such.case'}, None, root)
    with pytest.raises(client.StandInFailure, match='not served'):
        client.open('duckdb', {'id': 'x'}, None, root)


def test_lakebase_is_served_on_the_local_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Review r2 P2: Lakebase's offline engine is the local Postgres, its
    # tables the case's, loaded under the capture's {schema}.
    import os

    if not os.environ.get('SDM_POSTGRES_TEST_DSN'):
        assert not os.environ.get('SDM_REQUIRE_POSTGRES')
        return
    case = json.loads(
        (ROOT / 'contract/cases/s1/base.regression.json').read_text()
    )
    session = client.open('lakebase', case, None, ROOT / 'contract/captures')
    try:
        assert session.names['schema'] == '{schema}'
        step = {
            'id': 1,
            'request': {
                'request': 'scan',
                'table': {'source': ['{schema}', 'users']},
                'columns': ['user_id'],
                'filters': [],
                'limit': None,
            },
            'plan': {
                'plan': 'sql',
                'statements': [
                    {
                        'dialect': 'postgres',
                        'text': 'SELECT "user_id" FROM "{schema}"."users"',
                        'params': [],
                    }
                ],
                'expect': {'expect': 'discover'},
            },
        }
        table = session.provider(step, {})
        assert sorted(table.column('user_id').to_pylist()) == list(range(8))
        assert session.verdict() == {'live_only': None, 'flagged': []}
    finally:
        session.close()
