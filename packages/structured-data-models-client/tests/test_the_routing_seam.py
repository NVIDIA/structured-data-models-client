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
    MissingExtraError,
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
    # of a request materialized over one, have core routes; from T2.2 the
    # same calls over a SQLite source, and from T2.3 over a Postgres one.
    assert _routing.CORE_ROUTED == set(
        _routing.S0
        + _routing.FROM_SOURCE_DUCKDB
        + _routing.FROM_SOURCE_DUCKDB_TABLES
        + _routing.MATERIALIZE_DUCKDB
        + _routing.PREDICT_DUCKDB
        + _routing.FROM_SOURCE_SQLITE
        + _routing.FROM_SOURCE_SQLITE_TABLES
        + _routing.MATERIALIZE_SQLITE
        + _routing.PREDICT_SQLITE
        + _routing.FROM_SOURCE_POSTGRES
        + _routing.FROM_SOURCE_POSTGRES_TABLES
        + _routing.MATERIALIZE_POSTGRES
        + _routing.PREDICT_POSTGRES
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
        # A core path with no core route in this build (SQLite's has one
        # from T2.2, Postgres's from T2.3).
        (
            'client.predict/regression/snowflake',
            'client.predict/regression/snowflake',
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


#: The old validator holds a date as a pandas Timestamp. pandas 3 holds any
#: year; pandas 2 (on Python 3.10, which pandas 3 does not support) only
#: 1677 to 2262, so there the old route refuses a date in year 0 or 1 as a
#: query (OutOfBoundsDatetime), and the core, told pandas 2's range by the
#: host, refuses it alike (T1.20 fix 1).
PANDAS_3 = int(pd.__version__.split('.')[0]) >= 3

#: Date literals with a run of five digits or more (D8), which the
#: reference reads through dateutil or refuses: a packed first field of
#: six, eight, twelve and fourteen digits, a long field in the time of day
#: (register DT-9's two queries are the second and the sixth), a long
#: zero-padded field read as year 1 (on pandas 3 only), and two the
#: reference refuses.
PACKED_READ = (
    '230203-06-09',
    '20240101-01-01',
    '202401011230-1-1',
    '20240101123456-1-1',
    '20240101/00/123456',
    '2024-01-01 00000:00:00',
)
PACKED_READ_ON_PANDAS_3 = ('00001/1/2',)
PACKED_REFUSED = ('10000-01-01', '12345-1-1')


def _filtered(condition: str) -> str:
    return (
        f'PREDICT COUNT(orders.* WHERE {condition}, 0, 30, days) '
        'FOR EACH users.user_id'
    )


def _assert_refused_as_packed(
    refused: StructuredDataError, literal: str, query: str
) -> None:
    r"""T1.6c's refusal by name: ``INVALID_REQUEST`` naming ``operation``
    ``validate_query`` and ``rule`` ``date_literal_packed``, with the
    literal's place. From KO2-86 (an error may name a value of the caller's
    own call) it names the query given (``query``) and its packed literal
    (``literal``), exactly, and its message names the literal; no other
    detail holds the literal's text."""
    assert type(refused) is StructuredDataError
    assert refused.code == 'INVALID_REQUEST'
    assert refused.details['rule'] == 'date_literal_packed', refused.details
    assert refused.details['operation'] == 'validate_query'
    assert refused.details['field'] == 'query', refused.details
    for name in ('start_row', 'start_col', 'end_row', 'end_col'):
        assert isinstance(refused.details[name], int), (name, refused.details)
    assert refused.details['literal'] == literal, refused.details
    assert refused.details['query'] == query, refused.details
    assert literal in str(refused), str(refused)
    rest = {
        name: value
        for name, value in refused.details.items()
        if name not in ('literal', 'query')
    }
    assert literal not in repr(rest)


@pytest.mark.parametrize(
    'literal', PACKED_READ + PACKED_READ_ON_PANDAS_3 + PACKED_REFUSED
)
def test_a_packed_date_literal_is_refused_by_name_on_the_core(
    core: list[str], literal: str
) -> None:
    # T1.6c brief, Tests: a D8 literal is refused by name on python-core,
    # through materialize and the validation it runs (register DT-9's two
    # queries included: no longer read, as T1.6b read them), where the
    # reference reads it or refuses it as a query.
    text = _filtered(f'orders.ts > {literal}')
    schema = _orders_schema()
    with pytest.raises(StructuredDataError) as validated:
        PredictiveQuery(text, schema)._resolve(RELATIONAL)
    _assert_refused_as_packed(validated.value, literal, text)
    assert validated.value.details['start_row'] == 1
    assert validated.value.details['start_col'] == 41
    with pytest.raises(StructuredDataError) as materialized:
        PredictiveQuery(text, schema).materialize(indices=[1])
    _assert_refused_as_packed(materialized.value, literal, text)
    assert core == [text, text]


def test_a_packed_date_in_an_array_is_refused_by_name_on_the_core(
    core: list[str],
) -> None:
    text = _filtered('orders.ts IN (2024-01-01, 20240101-01-01)')
    with pytest.raises(StructuredDataError) as refused:
        PredictiveQuery(text, _orders_schema()).materialize(indices=[1])
    _assert_refused_as_packed(refused.value, '20240101-01-01', text)


def test_the_old_route_still_reads_a_packed_date_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # T1.6c: python-old is unchanged. Its validator reads every D8 literal
    # the reference reads and refuses the others as a query, never by the
    # core's rule (a year-1 literal too, on pandas 2, which cannot hold
    # it); two of them it materializes. Gathering them takes the
    # relational engine, an extra the test and installed-wheel jobs lack:
    # there, materialize must get past validation and reach the gathering,
    # which then names the missing extra; with the engine (the engine job)
    # it must return the request.
    from sdm_client.materializers.relational import RelationalMaterializer

    gathered: list[str] = []
    real = RelationalMaterializer._gather

    def spy(self: Any, query: Any, *args: Any) -> Any:
        gathered.append(query.to_string())
        return real(self, query, *args)

    monkeypatch.setattr(RelationalMaterializer, '_gather', spy)
    try:
        # The client's own test for the extra (_get_model): an import.
        import sdm_relational_engine.rfm  # noqa: F401
    except ImportError:
        engine = False
    else:
        engine = True
    schema = _orders_schema()
    for literal in PACKED_READ:
        PredictiveQuery(_filtered(f'orders.ts > {literal}'), schema)._resolve(
            RELATIONAL
        )
    refused_as_a_query: tuple[str, ...] = PACKED_REFUSED
    if PANDAS_3:
        for literal in PACKED_READ_ON_PANDAS_3:
            PredictiveQuery(
                _filtered(f'orders.ts > {literal}'), schema
            )._resolve(RELATIONAL)
    else:
        refused_as_a_query += PACKED_READ_ON_PANDAS_3
    for literal in refused_as_a_query:
        with pytest.raises(StructuredDataError) as refused:
            PredictiveQuery(
                _filtered(f'orders.ts > {literal}'), schema
            )._resolve(RELATIONAL)
        assert refused.value.code == 'INVALID_REQUEST'
        assert 'rule' not in refused.value.details
    materialized = ('20240101/00/123456', '2024-01-01 00000:00:00')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for literal in materialized:
            query = PredictiveQuery(_filtered(f'orders.ts > {literal}'), schema)
            if engine:
                assert type(query.materialize(indices=[1])) is Request
            else:
                with pytest.raises(MissingExtraError) as missing:
                    query.materialize(indices=[1])
                assert missing.value.details['extra'] == 'relational'
    assert len(gathered) == len(materialized)
    for literal, text in zip(materialized, gathered, strict=True):
        assert literal in text, text


@pytest.mark.parametrize(
    ('literal', 'accepted'),
    [
        ('2024-01-31', True),
        ('2024-01-31 23:59:59', True),
        ('1/2/24', True),
        ('0000-01-01', PANDAS_3),
        ('0001/1/2', PANDAS_3),
        ('2024-01-01 0000:00:00', True),
        ('13/01/2024', True),
        ('2023-02-29', False),
        ('123-4-567', False),
    ],
)
def test_a_literal_with_no_run_of_five_digits_answers_as_on_the_old_route(
    core: list[str],
    monkeypatch: pytest.MonkeyPatch,
    literal: str,
    accepted: bool,
) -> None:
    # T1.6c brief: D1 to D7 and D9 unchanged. Each answers on the core as
    # on the old route: read and accepted, or (D9) refused as a query,
    # INVALID_REQUEST with no rule named. On pandas 2 the old route refuses
    # a date in year 0 or 1 as a query, and so does the core: the host
    # declares pandas 2's range (date_literal_range) and the core refuses a
    # literal outside it as unreadable (T1.20 fix 1), never INTERNAL_ERROR.
    text = _filtered(f'orders.ts > {literal}')
    schema = _orders_schema()
    answers = []
    for route in ('core', 'old'):
        if route == 'old':
            monkeypatch.delenv(_routing.ENVIRONMENT)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                PredictiveQuery(text, schema)._resolve(RELATIONAL)
                answers.append(None)
            except StructuredDataError as refused:
                assert 'rule' not in refused.details, refused.details
                answers.append(refused.code)
    assert core == [text]
    want = [None, None] if accepted else ['INVALID_REQUEST', 'INVALID_REQUEST']
    assert answers == want


def test_an_earlier_refusal_wins_over_a_packed_date_on_the_core(
    core: list[str],
) -> None:
    # T1.6c: node order decides. A D9 literal built first refuses the query
    # as on the old route; a D8 literal built first ends the parse, by name.
    schema = _orders_schema()
    with pytest.raises(StructuredDataError) as refused:
        PredictiveQuery(
            _filtered('orders.ts > 2023-02-29 AND orders.ts < 20240101-01-01'),
            schema,
        )._resolve(RELATIONAL)
    assert refused.value.code == 'INVALID_REQUEST'
    assert 'rule' not in refused.value.details
    text = _filtered('orders.ts > 20240101-01-01 AND orders.ts < 2023-02-29')
    with pytest.raises(StructuredDataError) as packed:
        PredictiveQuery(text, schema)._resolve(RELATIONAL)
    _assert_refused_as_packed(packed.value, '20240101-01-01', text)


def test_a_packed_date_needs_no_local_zone_on_the_core(
    core: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # T1.6c: a D8 literal is refused before it is read, so a local zone the
    # run cannot read is never asked for, even for a six-digit packed date,
    # whose two-digit year the reference places.
    from sdm_client._rust import facts

    monkeypatch.setattr(
        facts, 'local_zone', lambda *_: {'iana': 'Mars/Olympus'}
    )
    text = _filtered('orders.ts > 760101-01-01')
    with pytest.raises(StructuredDataError) as refused:
        PredictiveQuery(text, _orders_schema())._resolve(RELATIONAL)
    # The query's own refusal (field query), not the local zone's.
    _assert_refused_as_packed(refused.value, '760101-01-01', text)


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
