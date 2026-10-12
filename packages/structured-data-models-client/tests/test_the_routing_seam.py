# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The private routing seam, and S0 on the core route (design, "Routing seam").

Every core path of the Python scope manifest is routed, and every default is
the old code until the Phase 3 switch. ``SDM_CORE_ROUTES`` forces paths onto
the core for tests, read at every call; a path it names that is unknown or
has no core route fails the call, and nothing falls back. Under the core
route, validation in ``materialize()`` is the core's ``validate_query``, and
a query is accepted or refused as on the old route, with the same code.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from predictive_query import RELATIONAL, TABULAR

from sdm_client import (
    PredictiveQuery,
    Request,
    Schema,
    StructuredDataError,
    _routing,
)
from sdm_client._rust import s0

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / 'docs/design/rust-core/phase-0/scope-python.yaml'
CORE = ','.join(_routing.S0)


@pytest.fixture
def core(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """S0 forced onto the core; the list records each query it validated."""
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    validated: list[str] = []
    real = s0.validate

    def spy(text: str, definition: Any, profile: Any) -> Any:
        validated.append(text)
        return real(text, definition, profile)

    monkeypatch.setattr(s0, 'validate', spy)
    return validated


@pytest.fixture(autouse=True)
def _old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)


def _users() -> pd.DataFrame:
    return pd.DataFrame({'id': [1, 2, 3, 4], 'age': [10.5, 20.5, 30.5, 40.5]})


ACCEPTED = 'PREDICT users.age FOR EACH users.id'
REFUSED = 'PREDICT users.nope FOR EACH users.id'


def test_every_core_path_of_the_manifest_is_routed_and_every_default_is_old() -> (
    None
):
    # Read without a YAML library, which the installed-wheel job lacks:
    # each operation's `- id:` line is followed by its `layer:`, and a
    # path's ID is operation/mode/backend.
    text = MANIFEST.read_text(encoding='utf-8')
    operations, rest = text.split('\npaths:\n')
    paths = rest.split('\ncells:\n')[0]
    layer = dict(
        re.findall(
            r'^- id: (\S+)\n(?:  .*\n)*?  layer: (\w+)$', operations, re.M
        )
    )
    assert len(layer) == 26
    ids = re.findall(r'^- id: (\S+)$', paths, re.M)
    assert len(ids) == 281
    core = [path for path in ids if layer[path.split('/')[0]] == 'core']
    assert list(_routing.PATHS) == core
    assert set(_routing.ROUTES) == set(core)
    # Every default is the old code until the Phase 3 switch.
    assert set(_routing.ROUTES.values()) == {'old'}
    # S0 and, from T1.11b, a DuckDB source's Schema.from_source with no
    # tables named (with tables named from T1.24), from T1.11c, materialize
    # with indices over a DuckDB source's schema, and from T1.11d, predict
    # of a request materialized over one, have core routes.
    assert _routing.CORE_ROUTED == set(
        _routing.S0
        + _routing.FROM_SOURCE_DUCKDB
        + _routing.FROM_SOURCE_DUCKDB_TABLES
        + _routing.MATERIALIZE_DUCKDB
        + _routing.PREDICT_DUCKDB
    )
    assert set(_routing.CORE_ROUTED) <= set(core)
    assert _routing.route(_routing.S0) == 'old'
    assert _routing.route(_routing.FROM_SOURCE_DUCKDB) == 'old'
    assert _routing.route(_routing.MATERIALIZE_DUCKDB) == 'old'
    assert _routing.route(_routing.PREDICT_DUCKDB) == 'old'


def test_a_refused_and_an_accepted_query_give_the_old_outcome_on_the_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = {}
    for route in ('old', 'core'):
        if route == 'core':
            monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            accepted = PredictiveQuery(ACCEPTED, {'users': _users()})
            request = accepted.materialize(indices=[1, 2])
            with pytest.raises(StructuredDataError) as refused:
                PredictiveQuery(REFUSED, {'users': _users()}).materialize(
                    indices=[1]
                )
        outcomes[route] = (
            type(request),
            type(refused.value),
            refused.value.code,
        )
    assert outcomes['old'] == outcomes['core']
    assert outcomes['core'] == (Request, StructuredDataError, 'INVALID_REQUEST')


def test_the_core_route_runs_validation_on_the_core(core: list[str]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        PredictiveQuery(ACCEPTED, {'users': _users()}).materialize(indices=[1])
        with pytest.raises(StructuredDataError):
            PredictiveQuery(REFUSED, {'users': _users()}).materialize(
                indices=[1]
            )
    assert core == [ACCEPTED, REFUSED]
    assert 'sdm_client._native' in sys.modules


def test_the_route_is_read_at_every_call(
    core: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    query = PredictiveQuery(ACCEPTED, {'users': _users()})
    query._resolve(TABULAR)
    monkeypatch.delenv(_routing.ENVIRONMENT)
    query._resolve(TABULAR)
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    query._resolve(TABULAR)
    assert core == [ACCEPTED, ACCEPTED]


@pytest.mark.parametrize(
    ('setting', 'named'),
    [
        ('query.validate/accept/nowhere', 'query.validate/accept/nowhere'),
        (f'{CORE},nope', 'nope'),
        # A host path is never routed.
        ('source.open/explicit/duckdb', 'source.open/explicit/duckdb'),
        # A core path with no core route in this build.
        (
            'client.predict/regression/sqlite',
            'client.predict/regression/sqlite',
        ),
    ],
)
def test_an_unknown_or_unrouted_forced_path_fails_the_call(
    monkeypatch: pytest.MonkeyPatch, setting: str, named: str
) -> None:
    if named.startswith('client.predict'):
        assert named in _routing.ROUTES
    monkeypatch.setenv(_routing.ENVIRONMENT, setting)
    query = PredictiveQuery(ACCEPTED, {'users': _users()})
    with pytest.raises(_routing.RouteError, match=named):
        query.materialize(indices=[1])
    assert not isinstance(_routing.RouteError('x'), StructuredDataError)


def test_a_call_whose_paths_are_forced_apart_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, _routing.S0[0])
    with pytest.raises(_routing.RouteError, match='one route'):
        PredictiveQuery(ACCEPTED, {'users': _users()})._resolve(TABULAR)


def _subprocess(script: str, **env: str) -> subprocess.CompletedProcess[str]:
    import os

    environment = {
        key: value
        for key, value in os.environ.items()
        if key != _routing.ENVIRONMENT
    }
    return subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**environment, **env},
    )


#: Fails any import of the native module from the moment it is installed.
_BLOCK_NATIVE = """
    import sys

    class Block:
        def find_spec(self, name, path=None, target=None):
            if name == 'sdm_client._native':
                raise ImportError('the native module was imported')
            return None

    sys.meta_path.insert(0, Block())
"""


def test_old_paths_never_import_the_native_module() -> None:
    script = (
        _BLOCK_NATIVE
        + """
    import warnings
    import pandas as pd
    from sdm_client import PredictiveQuery, StructuredDataError

    warnings.simplefilter('ignore')
    users = pd.DataFrame({'id': [1, 2, 3, 4], 'age': [10.5, 20.5, 30.5, 40.5]})
    PredictiveQuery('PREDICT users.age FOR EACH users.id', {'users': users}
        ).materialize(indices=[1, 2])
    try:
        PredictiveQuery('PREDICT users.nope FOR EACH users.id', {'users': users}
            ).materialize(indices=[1])
    except StructuredDataError as refused:
        assert refused.code == 'INVALID_REQUEST'
    else:
        raise AssertionError('accepted')
    print('sdm_client._native' in sys.modules, 'sdm_client._rust' in sys.modules)
    """
    )
    result = _subprocess(script)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ['False', 'False']


def test_a_core_route_without_the_native_module_fails_naming_it() -> None:
    script = (
        _BLOCK_NATIVE
        + """
    import warnings
    import pandas as pd
    from sdm_client import PredictiveQuery
    from sdm_client._routing import RouteError

    warnings.simplefilter('ignore')
    users = pd.DataFrame({'id': [1, 2, 3, 4], 'age': [10.5, 20.5, 30.5, 40.5]})
    try:
        PredictiveQuery('PREDICT users.age FOR EACH users.id', {'users': users}
            ).materialize(indices=[1, 2])
    except RouteError as missing:
        print(missing)
    """
    )
    result = _subprocess(script, **{_routing.ENVIRONMENT: CORE})
    assert result.returncode == 0, result.stderr
    assert 'sdm_client._native' in result.stdout


def _orders_schema() -> Any:
    users = pd.DataFrame({'user_id': [1, 2, 3]})
    orders = pd.DataFrame(
        {
            'order_id': [1, 2, 3],
            'user_id': [1, 2, 2],
            'ts': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data({'users': users, 'orders': orders})


def test_the_core_s_own_refusal_is_not_a_refusal_of_the_query(
    core: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A local zone the run cannot read is the core's refusal of the host's
    # facts, raised as its code maps, never as a refusal of the query; it is
    # read only for a two-digit year (D5), so a query without one is
    # answered.
    from sdm_client._rust import facts

    monkeypatch.setattr(
        facts, 'local_zone', lambda *_: {'iana': 'Mars/Olympus'}
    )
    schema = _orders_schema()
    PredictiveQuery(
        'PREDICT COUNT(orders.* WHERE orders.ts > 2024-01-13, 0, 30, days) '
        'FOR EACH users.user_id',
        schema,
    )._resolve(RELATIONAL)
    text = (
        'PREDICT COUNT(orders.* WHERE orders.ts > 1/2/76, 0, 30, days) '
        'FOR EACH users.user_id'
    )
    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery(text, schema)._resolve(RELATIONAL)
    assert raised.value.code == 'INVALID_REQUEST'
    assert raised.value.details['field'] == 'facts.local_zone'
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


def test_a_date_dt9_held_back_is_accepted_on_the_core(core: list[str]) -> None:
    # Register DT-9 (rejected at T1.6b): a packed date and a long field the
    # interim reader answered with UNSUPPORTED_REQUEST are read by D8, and
    # the old validator, the shadow check, agrees.
    for literal in ('20240101-01-01', '2024-01-01 00000:00:00'):
        text = (
            f'PREDICT COUNT(orders.* WHERE orders.ts > {literal}, 0, 30, '
            'days) FOR EACH users.user_id'
        )
        PredictiveQuery(text, _orders_schema())._resolve(RELATIONAL)
        assert core[-1] == text


def _shown(text: str, schema: Any) -> tuple[list[str], str | None]:
    r"""The warnings ``_resolve`` shows, and the code it refuses with."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            PredictiveQuery(text, schema)._resolve(RELATIONAL)
            code = None
        except StructuredDataError as refused:
            code = refused.code
    return [f'{type(w.message).__name__}: {w.message}' for w in caught], code


def test_a_day_first_literal_warns_on_the_core_as_on_the_old_route(
    core: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # D4. Accepted: the old validator, still run beside the core here (the
    # shadow check), gives the reference's own warning and the core's is
    # held back (s0.SHADOWED), so the caller sees what the old route shows.
    # Refused, after the literal was read (an unknown table, then a literal
    # that names no date): no shadow check runs, and the core's warning is
    # shown, once per literal, in the reference's words (register PQL-3).
    accepted = (
        'PREDICT COUNT(orders.* WHERE orders.ts > 13/01/2024, 0, 30, days) '
        'FOR EACH users.user_id'
    )
    refused = [
        'PREDICT COUNT(orders.* WHERE missing.ts > 13/01/2024, 0, 30, days) '
        'FOR EACH users.user_id',
        'PREDICT COUNT(orders.* WHERE orders.ts > 13/01/2024 AND orders.ts < '
        '2023-02-29, 0, 30, days) FOR EACH users.user_id',
    ]
    schema = _orders_schema()
    on_core = {text: _shown(text, schema) for text in [accepted, *refused]}
    assert core == [accepted, *refused]
    monkeypatch.delenv(_routing.ENVIRONMENT)
    on_old = {text: _shown(text, schema) for text in [accepted, *refused]}
    said = (
        'UserWarning: Parsing dates in %d/%m/%Y format when dayfirst=False '
        '(the default) was specified. Pass `dayfirst=True` or specify a '
        'format to silence this warning.'
    )
    assert on_core[accepted] == on_old[accepted]
    assert on_core[accepted][1] is None and said in on_core[accepted][0]
    for text in refused:
        (core_shown, core_code), (old_shown, old_code) = (
            on_core[text],
            on_old[text],
        )
        assert core_code == old_code == 'INVALID_REQUEST', text
        assert old_shown == [said], text
        assert core_shown == [
            said.replace(
                'UserWarning: ', 'UserWarning: [DATE_LITERAL_DAY_FIRST] '
            )
        ], text


def test_a_disagreement_with_the_old_validator_is_internal(
    core: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import predictive_query

    def refuses(*_: object) -> Any:
        class Response:
            ok = False
            errors: list[Any] = []

        return None, Response()

    monkeypatch.setattr(predictive_query, 'validate', refuses)
    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery(ACCEPTED, {'users': _users()})._resolve(TABULAR)
    assert raised.value.code == 'INTERNAL_ERROR'
    assert core == [ACCEPTED]


def test_a_wide_schema_long_query_the_old_client_accepts_passes_on_the_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 5,000 columns and a query past the 64 KB control cap: the query text
    # and the schema travel as attachments (decision H5).
    long = 'x' * 70_000
    columns = {f'c{i}': [0.5, 1.5, 2.5, 3.5] for i in range(5_000)}
    frame = pd.DataFrame({'id': [1, 2, 3, 4], **columns, long: [1.5, 2, 3, 4]})
    text = f'PREDICT users.{long} FOR EACH users.id'
    assert len(text.encode()) > 64 * 1024
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'users': frame})
        old = PredictiveQuery(text, schema).materialize(indices=[1, 2])
        validated: list[bool] = []
        real = s0.validate

        def spy(*args: Any) -> Any:
            answer = real(*args)
            validated.append(answer[0])
            return answer

        monkeypatch.setattr(s0, 'validate', spy)
        monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
        new = PredictiveQuery(text, schema).materialize(indices=[1, 2])
    assert validated == [True]
    assert type(old) is type(new) is Request
