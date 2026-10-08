# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Branches of the request path that public calls reach and the parity
corpus cannot, held to today's Python through a host.

The fixture flows and the transcripts make the calls their vocabularies name
(``tests/parity/_fixture_ops.py``'s ``Flow``, ``scripts/parity/_parity``'s
runner), and those are locked. Some branches are reached only by calls
neither has, and ``scripts/parity/request_coverage_gaps.json`` lists them as
such. Each is run here twice, by today's Python and by the core through
``scripts/parity/request_host.py``, and the two must agree on what was sent,
what came back, and what was refused:

- ``Request.materialized(payloads=...)``, the public way to build a request
  from envelopes, given envelopes the engine does not write: several of
  them, one without a context, one without an inference section or its
  config, one whose inference names no run mode, and one with no rows to
  score
  (``adapters/relational.py`` ``session_plan``, ``prediction_plan``,
  ``_asks_for_the_larger_context``; ``client.py`` ``_names_its_rows``);
  given without ``requested_inference``
  (``adapters/relational.py`` ``_keys_the_caller_set``); fitted from an
  envelope whose schema describes no instance table (``client.py``
  ``_pinned_dtypes``); and refused by the endpoint for a column's
  cardinality, its envelope holding what is not a table, a list of columns
  or a whole row (``rfm/inference/payload.py``
  ``_payload_tables_with_names``, ``high_cardinality_columns``); edited
  as a caller holding them can, then predicted, fitted, or fitted holding
  rows out, with or without ``entity_table``
  (``adapters/relational.py`` ``_how_to_name_the_target``,
  ``_distinct_labels``, ``_columns_with_stype``, ``_named_entities``,
  ``_key_dtype``; ``wire/keys.py`` ``rows_in``, ``without_the_keys``,
  ``_drop_instances``; ``request.py`` ``_entity_table_in``).
- ``request.entity_table = None`` on a request built from frames, with rows
  to score and without (``request.py``
  ``_warn_of_rows_scored_from_their_own_answers``; ``client.py``
  ``_names_its_rows``); its key column deleted from its schema
  (``wire/keys.py`` ``key_dtypes``); and one asked of a session pinned
  without key dtypes (``wire/keys.py`` ``scored_rows``).
- ``result.items = ()`` on a forecast's result (``result.py``
  ``_empty_frame``).
- A send that fails without an answer, its text carrying the endpoint's
  wording for a refusal, which the relational adapter rewords as the
  refusal it reads as (``adapters/relational.py`` ``_reworded``).
- ``Schema(tables, source=...)``, a schema over a source built without
  ``Schema.from_source``, which reads its tables at the first gather
  (``materializers/relational.py`` ``_build_graph``), and a table another
  schema over the source described put into ``schema.tables``, which the
  gather reads then (``materializers/sources.py`` ``graph``).
- A schema that says more than its frames hold: a column put into
  ``schema[table].columns``, by itself, under a relationship appended over
  it, or as the target of the tabular route, and a ``Schema`` built by hand
  of ``Table``\ s naming a column their frame lacks
  (``materializers/relational.py`` ``_build_graph``, ``_as_typed``;
  ``materializers/tabular.py`` ``table``); a column of decimals put in
  under a key that is not text (``_as_typed``, ``_engine_errors``); and a
  table deleted from ``schema.tables`` (``_joins_nothing``). The host says
  the same of its schema (``request_host.put_column``, ``put_relationship``,
  ``put_table``, ``drop_table``, ``schema_of_tables``).
- A PostgreSQL table emptied between ``Schema.from_source`` and the gather
  (``backend/postgres/table.py`` ``_get_num_rows``, and the engine failure
  ``materializers/relational.py`` ``_engine_errors`` reports for it).
- A frame changed in place after ``Schema.from_data`` read it: before the
  query's first gather, which reads it as it stands against what the schema
  said of it (``materializers/relational.py`` ``_build_graph``,
  ``_as_typed``, ``_joins_nothing``), a name the wire reserves among the
  columns added; between two gathers of one query, whose model keeps the
  graph it built, even when its gather was refused after building it; and
  after a gather, whose request is settled. Relabelled with a
  ``MultiIndex``, repeated or empty names, or given a column labelled by an
  int, it is refused as the engine's ``LocalTable`` refuses it. The tabular route reads the one
  frame it sends as it stands at each gather
  (``materializers/tabular.py`` ``indices``, ``table``).

The core builds the request from envelopes as ``Request.materialized`` does
(``request_host.request_materialized``): the envelopes given, the first
one's task and inference section as the request's, and the entity table,
where none is named, off the first envelope that names one.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip('sdm_relational_engine')
pytest.importorskip('sdm_core._native')
duckdb = pytest.importorskip('duckdb')

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / 'scripts' / 'parity'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Today's Python, which these hold the core to, runs public calls here.
if str(Path(__file__).resolve().parents[0]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from _old_path import python_path  # noqa: E402,F401 - the fixture

pytestmark = pytest.mark.usefixtures('python_path')

from _fixture_messages import COUNT, people, shop  # noqa: E402
from _parity.nim import SyntheticSession, normalized  # noqa: E402

REL = 'kumo-relational'
TAB = 'kumo-tabular'
GATHER = {'indices': [1, 2], 'num_context_samples': 6, 'random_seed': 7}


def _host() -> Any:
    if 'request_host' in sys.modules:
        return sys.modules['request_host']
    spec = importlib.util.spec_from_file_location(
        'request_host', SCRIPTS / 'request_host.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['request_host'] = module
    spec.loader.exec_module(module)
    return module


def _client(session: Any) -> Any:
    from sdm_client import SDMClient

    client = SDMClient(url='http://localhost:8000')
    client._transport._session.close()
    client._transport._session = session
    return client


def _outcome(make: Callable[[], Any]) -> tuple[str, Any]:
    r"""What ``make`` gave, or what it raised as class, code, message and
    details, with the warnings given on the way."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            got: Any = ('value', make())
        except Exception as error:  # noqa: BLE001 - compared
            cls = type(error)
            got = (
                'error',
                (
                    f'{cls.__module__}.{cls.__qualname__}',
                    getattr(error, 'code', None),
                    getattr(error, 'message', str(error)),
                    dict(getattr(error, 'details', None) or {}),
                ),
            )
    said = [
        (f'{w.category.__module__}.{w.category.__qualname__}', str(w.message))
        for w in caught
    ]
    return got, said


def _sent(session: SyntheticSession) -> list[Any]:
    r"""Every request the session was sent, its body with the request id
    normalized."""
    return [
        {
            key: normalized(copy.deepcopy(value)) if key == 'request' else value
            for key, value in entry.items()
        }
        for entry in session.responses
    ] + [normalized(copy.deepcopy(body)) for body in session.payloads]


# --- Requests built from envelopes ----------------------------------------------


def _two_batches(payloads: list[Any]) -> list[Any]:
    return [payloads[0], copy.deepcopy(payloads[0])]


def _no_context(payloads: list[Any]) -> list[Any]:
    return [{k: v for k, v in payloads[0].items() if k != 'context'}]


def _no_inference(payloads: list[Any]) -> list[Any]:
    return [{k: v for k, v in payloads[0].items() if k != 'inference'}]


def _no_inference_config(payloads: list[Any]) -> list[Any]:
    payload = copy.deepcopy(payloads[0])
    del payload['inference']['inference_config']
    return [payload]


def _no_run_mode(payloads: list[Any]) -> list[Any]:
    payload = copy.deepcopy(payloads[0])
    del payload['inference']['run_mode']
    return [payload]


def _nothing_to_score(payloads: list[Any]) -> list[Any]:
    payload = copy.deepcopy(payloads[0])
    payload['predict']['instance_table']['rows'] = []
    return [payload]


def _today(
    edit: Callable[[list[Any]], list[Any]],
    gather: Any,
    asked: bool = True,
    named: bool = True,
) -> Any:
    r"""``Request.materialized`` of the envelopes a gather wrote, edited;
    without ``requested_inference`` unless ``asked``, and without
    ``entity_table`` unless ``named``."""
    from sdm_client import PredictiveQuery, Request, Schema

    gathered = PredictiveQuery(COUNT, Schema.from_data(shop())).materialize(
        **gather
    )
    payloads = edit(json.loads(json.dumps(list(gathered.payloads))))
    told: dict[str, Any] = {}
    if asked:
        told['requested_inference'] = gathered.requested_inference
    if named:
        told['entity_table'] = gathered.entity_table
    return Request.materialized(
        payloads=tuple(payloads),
        schema=gathered.schema,
        query=gathered.query,
        indices=list(gathered.indices),
        windowed=gathered.target_moves_with_time,
        **told,
    )


def _core(
    edit: Callable[[list[Any]], list[Any]],
    gather: Any,
    asked: bool = True,
    named: bool = True,
) -> Any:
    r"""``_today``'s request, built by the core: the envelopes its gather
    wrote, edited, through the core's ``Request.materialized``."""
    from sdm_core.answers import decode

    host = _host()
    schema = host.schema_from_data(shop())
    gathered = host.materialize(host.predictive_query(COUNT, schema), **gather)
    state = gathered.state
    return host.request_materialized(
        payloads=edit(copy.deepcopy(state['payloads'])),
        schema=schema,
        query=state['query'],
        indices=[decode(index) for index in state['indices']],
        windowed=state['windowed'],
        entity_table=state['entity_table'] if named else None,
        requested_inference=decode(state['requested_inference'])
        if asked
        else None,
    )


@pytest.mark.parametrize(
    ('edit', 'how'),
    [
        pytest.param(_two_batches, 'fit', id='a-fit-of-two-batches'),
        pytest.param(
            _no_context, 'fit', id='a-fit-of-an-envelope-without-context'
        ),
        pytest.param(
            _no_inference, 'predict', id='an-envelope-without-inference'
        ),
        pytest.param(
            _no_inference_config,
            'predict',
            id='an-envelope-without-an-inference-config',
        ),
        pytest.param(
            _no_run_mode,
            'explain',
            id='an-explanation-asked-of-an-envelope-without-a-run-mode',
        ),
        pytest.param(
            _nothing_to_score, 'predict', id='an-envelope-with-nothing-to-score'
        ),
    ],
)
def test_a_request_built_from_envelopes(
    edit: Callable[[list[Any]], list[Any]], how: str
) -> None:
    import pandas as pd

    host = _host()
    outputs = ['prediction', 'explanation'] if how == 'explain' else None
    # An explanation covers one entity.
    gather = {**GATHER, 'indices': [1]} if how == 'explain' else GATHER

    def today(session: SyntheticSession) -> Any:
        client = _client(session)
        request = _today(edit, gather)
        if how == 'fit':
            fitted = client.fit(request, model=REL)
            return fitted._session_id
        return client.predict(request, model=REL, outputs=outputs).to_pandas()

    def core(session: SyntheticSession) -> Any:
        client = host.Client(_client(session)._transport)
        request = _core(edit, gather)
        if how == 'fit':
            return client.fit(request, model=REL).state['session_id']
        answer = client.predict(request, model=REL, outputs=outputs)
        return host.frame(host.flatten(answer))

    sessions = SyntheticSession(), SyntheticSession()
    want, said = _outcome(lambda: today(sessions[0]))
    got, given = _outcome(lambda: core(sessions[1]))
    assert given == said
    assert _sent(sessions[1]) == _sent(sessions[0])
    assert got[0] == want[0], (got, want)
    if isinstance(want[1], pd.DataFrame):
        pd.testing.assert_frame_equal(got[1], want[1])
    else:
        assert got[1] == want[1]


def _as_given(payloads: list[Any]) -> list[Any]:
    return payloads


def test_a_request_built_from_envelopes_without_what_was_asked() -> None:
    r"""``Request.materialized`` without ``requested_inference``: nothing
    was asked of the inference config, so nothing in it is refused
    (``adapters/relational.py`` ``_keys_the_caller_set``)."""
    import pandas as pd

    host = _host()
    sessions = SyntheticSession(), SyntheticSession()
    want, said = _outcome(
        lambda: (
            _client(sessions[0])
            .predict(_today(_as_given, GATHER, asked=False), model=REL)
            .to_pandas()
        )
    )
    got, given = _outcome(
        lambda: host.frame(
            host.flatten(
                host.Client(_client(sessions[1])._transport).predict(
                    _core(_as_given, GATHER, asked=False), model=REL
                )
            )
        )
    )
    assert want[0] == got[0] == 'value', (want, got)
    assert given == said
    assert _sent(sessions[1]) == _sent(sessions[0])
    pd.testing.assert_frame_equal(got[1], want[1])


def _no_instance_table_described(payloads: list[Any]) -> list[Any]:
    payload = copy.deepcopy(payloads[0])
    payload['schema'] = {
        k: v for k, v in payload['schema'].items() if k != 'instance_table'
    }
    return [payload]


def test_a_fitted_model_asked_of_an_envelope_describing_no_instance_table() -> (
    None
):
    r"""A session pinned from an envelope whose schema describes no
    instance table pins no key dtypes, and ``FittedModel.predict`` reads
    its rows' keys as they are (``client.py`` ``_pinned_dtypes``)."""
    import pandas as pd

    host = _host()

    def today(session: SyntheticSession) -> Any:
        request = _today(_no_instance_table_described, GATHER)
        fitted = _client(session).fit(request, model=REL)
        return fitted.predict(request).to_pandas()

    def core(session: SyntheticSession) -> Any:
        request = _core(_no_instance_table_described, GATHER)
        client = host.Client(_client(session)._transport)
        fitted = client.fit(request, model=REL)
        return host.frame(host.flatten(fitted.predict(request)))

    sessions = SyntheticSession(), SyntheticSession()
    want, said = _outcome(lambda: today(sessions[0]))
    got, given = _outcome(lambda: core(sessions[1]))
    assert want[0] == got[0] == 'value', (want, got)
    assert given == said
    assert _sent(sessions[1]) == _sent(sessions[0])
    pd.testing.assert_frame_equal(got[1], want[1])


#: The endpoint refusing a request for a column's cardinality.
CARDINALITY = {
    'fail': [
        {
            'at': 0,
            'status': 400,
            'body': {
                'error': {
                    'code': 'INVALID_REQUEST',
                    'message': 'categorical cardinality 99 exceeds limit 1',
                }
            },
        }
    ]
}


def _edited(change: Callable[[Any], None]) -> Callable[[list[Any]], list[Any]]:
    def edit(payloads: list[Any]) -> list[Any]:
        payload = copy.deepcopy(payloads[0])
        change(payload)
        return [payload]

    return edit


def _cut_short(payload: Any) -> None:
    rows = payload['context']['instance_table']['rows']
    rows[0] = rows[0][:1]


@pytest.mark.parametrize(
    'change',
    [
        pytest.param(
            lambda p: p['predict'].__setitem__('instance_table', None),
            id='a-predict-half-without-an-instance-table',
        ),
        pytest.param(
            lambda p: p['predict'].__setitem__('related_tables', None),
            id='a-predict-half-without-related-tables',
        ),
        pytest.param(
            lambda p: p['predict']['related_tables'].__setitem__(
                'orders', None
            ),
            id='a-related-table-that-is-not-a-table',
        ),
        pytest.param(
            lambda p: p['predict']['related_tables']['orders'].__setitem__(
                'columns', None
            ),
            id='a-related-table-without-columns',
        ),
        pytest.param(_cut_short, id='a-context-row-cut-short'),
    ],
)
def test_a_cardinality_refusal_of_an_edited_envelope(
    change: Callable[[Any], None],
) -> None:
    r"""An envelope the engine does not write, refused by the endpoint for a
    column's cardinality: the refusal names the columns it counts, reading
    past what is not a table, a list of columns or a whole row
    (``rfm/inference/payload.py`` ``_payload_tables_with_names``,
    ``high_cardinality_columns``)."""
    host = _host()
    edit = _edited(change)
    sessions = SyntheticSession(CARDINALITY), SyntheticSession(CARDINALITY)
    want, said = _outcome(
        lambda: _client(sessions[0]).predict(_today(edit, GATHER), model=REL)
    )
    got, given = _outcome(
        lambda: host.Client(_client(sessions[1])._transport).predict(
            _core(edit, GATHER), model=REL
        )
    )
    assert want[0] == 'error' and 'cardinality' in want[1][2], want
    assert (got, given) == (want, said)
    assert _sent(sessions[1]) == _sent(sessions[0])


# --- Envelopes edited before they are sent ---------------------------------------
#
# Each of these edits the COUNT envelope a gather of ``shop()`` wrote, the
# way a caller holding ``request.payloads`` can, and builds the request of it
# with ``Request.materialized``. The engine writes none of them.


def _renamed(table: Any, old: str, new: str) -> None:
    table['columns'] = [
        new if name == old else name for name in table['columns']
    ]


def _dropped(table: Any, name: str) -> None:
    at = table['columns'].index(name)
    del table['columns'][at]
    for row in table['rows']:
        del row[at]


def _reference(table: Any) -> str:
    return next(
        name
        for name in table['columns']
        if str(name).startswith('__kumo_entity_ref')
    )


def _a_constant_target_renamed(payload: Any) -> None:
    payload['task']['target']['column_name'] = 'n_orders'
    context = payload['context']['instance_table']
    _renamed(context, 'TARGET', 'n_orders')
    at = context['columns'].index('n_orders')
    for row in context['rows']:
        row[at] = 1.0


def _a_classification_without_its_target(payload: Any) -> None:
    payload['task']['kind'] = 'binary_classification'
    _renamed(payload['context']['instance_table'], 'TARGET', 'T2')


def _a_forecast(payload: Any) -> None:
    payload['task']['kind'] = 'forecasting'


def _related_orders_not_a_table(payload: Any) -> None:
    payload['schema']['related_tables']['orders'] = None


def _no_predict_reference(payload: Any) -> None:
    table = payload['predict']['instance_table']
    _renamed(table, _reference(table), 'ref0')


def _no_predict_instance_ids(payload: Any) -> None:
    _renamed(payload['predict']['instance_table'], 'instance_id', 'iid')


def _reference_from_another_column(payload: Any) -> None:
    payload['schema']['relationships'][0]['source_columns'] = ['other']


def _reference_to_two_columns(payload: Any) -> None:
    payload['schema']['relationships'][0]['target_columns'] = [
        'user_id',
        'age',
    ]


def _no_context_reference(payload: Any) -> None:
    table = payload['context']['instance_table']
    _dropped(table, _reference(table))
    payload['schema'].setdefault('instance_table', {})['primary_key'] = (
        'nonexistent'
    )


def _no_anchor_time(payload: Any) -> None:
    _dropped(payload['context']['instance_table'], '__kumo_anchor_time')


def _without_context(payload: Any) -> None:
    del payload['context']


def _context_without_columns(payload: Any) -> None:
    payload['context']['instance_table']['columns'] = None


def _related_without_rows(payload: Any) -> None:
    for table in payload['context']['related_tables'].values():
        table['rows'] = None


def _no_context_instance_ids(payload: Any) -> None:
    _dropped(payload['context']['instance_table'], 'instance_id')


def _naming_no_entity_table(payload: Any) -> None:
    payload['task']['entity_table_names'] = []


def _answered(change: Callable[[Any], Any]) -> Callable[[Any], Any]:
    r"""The endpoint's answer to the unedited request, each item changed by
    ``change``, given in place of the stand-in's own."""

    def quirks(body: Any) -> Any:
        answer = copy.deepcopy(body)
        answer['predictions'] = [
            change(dict(item)) for item in answer['predictions']
        ]
        return {'fail': [{'at': 0, 'status': 200, 'body': answer}]}

    return quirks


def _refused(message: str) -> Callable[[Any], Any]:
    r"""The endpoint refusing the request, saying ``message``."""

    def quirks(body: Any) -> Any:
        error = {'code': 'INVALID_REQUEST', 'message': message}
        return {'fail': [{'at': 0, 'status': 400, 'body': {'error': error}}]}

    return quirks


class _Watched(SyntheticSession):
    r"""The stand-in endpoint, keeping what it failed with, if it failed to
    read what it was sent."""

    failed: Exception | None = None

    def post(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return super().post(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - kept, then raised
            self.failed = error
            raise


@pytest.mark.parametrize('named', [True, False])
@pytest.mark.parametrize('field', ['task', 'target', 'options', 'inference'])
@pytest.mark.parametrize('value', [None, 0, True, 1.5, 'x', [0]])
def test_malformed_envelope_constructor_stops_before_transport(
    field: str, value: Any, named: bool
) -> None:
    host = _host()

    def edit(payloads: list[Any]) -> list[Any]:
        parent = (
            payloads[0]['task']
            if field in {'target', 'options'}
            else payloads[0]
        )
        parent[field] = value
        return payloads

    sessions = _Watched(), _Watched()
    want, said = _outcome(
        lambda: _client(sessions[0]).predict(
            _today(edit, GATHER, named=named), model=REL
        )
    )
    got, given = _outcome(
        lambda: host.Client(_client(sessions[1])._transport).predict(
            _core(edit, GATHER, named=named), model=REL
        )
    )
    assert want[0] == 'error'
    assert want[1][0] in {
        'builtins.AttributeError',
        'builtins.TypeError',
        'builtins.ValueError',
    }
    assert got == want
    assert given == said
    assert all(session.failed is None for session in sessions)
    assert all(not session.payloads for session in sessions)


@pytest.mark.parametrize('field', ['options', 'inference'])
@pytest.mark.parametrize(
    'value',
    [
        [],
        '',
        {},
        [['alpha', 1], ['alpha', 2]],
        [[1, 'one'], ['1', 'text']],
        [[True, 'bool'], [1, 'int']],
        [[1, 'int'], [True, 'bool']],
        [[None, 'none']],
        [[1.0, 'float'], [1, 'int']],
        [[False, 'bool'], [0, 'int']],
        {'alpha': 1, 'beta': {'nested': [None, True]}},
        ['ab'],
        [{'alpha': 1, 'beta': 2}],
    ],
)
def test_envelope_constructor_coerces_dictionary_sequences(
    field: str, value: Any
) -> None:
    def edit(payloads: list[Any]) -> list[Any]:
        parent = payloads[0]['task'] if field == 'options' else payloads[0]
        parent[field] = value
        return payloads

    def today() -> Any:
        request = _today(edit, GATHER)
        return request.task.options if field == 'options' else request.inference

    def core() -> Any:
        state = _core(edit, GATHER).state
        return (
            state['task']['options']
            if field == 'options'
            else state['inference']
        )

    want, said = _outcome(today)
    got, given = _outcome(core)
    assert want == ('value', dict(value))
    assert got == want
    assert given == said
    assert repr(got[1]) == repr(want[1])


@pytest.mark.parametrize('field', ['kind', 'column_name', 'dtype'])
@pytest.mark.parametrize('value', [None, False, 0, [], {}, ['x'], ''])
def test_envelope_constructor_retains_explicit_metadata(
    field: str, value: Any
) -> None:
    def edit(payloads: list[Any]) -> list[Any]:
        task = payloads[0]['task']
        parent = task if field == 'kind' else task['target']
        parent[field] = value
        return payloads

    def today() -> Any:
        task = _today(edit, GATHER).task
        return getattr(task if field == 'kind' else task.target, field)

    def core() -> Any:
        task = _core(edit, GATHER).state['task']
        return (task if field == 'kind' else task['target'])[field]

    want, said = _outcome(today)
    got, given = _outcome(core)
    assert want == ('value', value)
    assert got == want
    assert type(got[1]) is type(want[1])
    assert given == said


@pytest.mark.parametrize('outputs', [None, ['prediction'], []])
@pytest.mark.parametrize('value', [None, False, 0, [], {}, ['x'], ''])
def test_envelope_constructor_invalid_kind_stops_before_transport(
    value: Any, outputs: Any
) -> None:
    host = _host()
    baseline = _Watched()
    _client(baseline).predict(_today(_as_given, GATHER), model=REL)
    response = {
        'fail': [
            {'at': 0, 'status': 200, 'body': baseline.responses[-1]['body']}
        ]
    }

    def edit(payloads: list[Any]) -> list[Any]:
        payloads[0]['task']['kind'] = value
        return payloads

    sessions = (
        _Watched(copy.deepcopy(response)),
        _Watched(copy.deepcopy(response)),
    )
    want, said = _outcome(
        lambda: _client(sessions[0]).predict(
            _today(edit, GATHER), model=REL, outputs=outputs
        )
    )
    got, given = _outcome(
        lambda: host.Client(_client(sessions[1])._transport).predict(
            _core(edit, GATHER), model=REL, outputs=outputs
        )
    )
    assert want[0] == 'error'
    assert got == want
    assert given == said
    assert all(session.failed is None for session in sessions)
    assert all(not session.payloads for session in sessions)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        (field, value)
        for field in ['column_name', 'dtype']
        for value in [None, False, 0, [], {}, ['x'], '']
    ]
    + [
        (field, value)
        for field in ['options', 'inference']
        for value in [
            [[1, 'one'], ['1', 'text']],
            [[True, 'bool'], [1, 'int']],
            [[1, 'int'], [True, 'bool']],
            [[None, 'none']],
            [[1.0, 'float'], [1, 'int']],
            [['alpha', 1], ['alpha', 2]],
            {'alpha': 1, 'beta': {'nested': [None, True]}},
        ]
    ]
    + [('unedited', None), ('missing_kind', None)],
)
def test_envelope_constructor_preserved_values_reach_transport(
    field: str, value: Any
) -> None:
    import pandas as pd

    host = _host()
    baseline = _Watched()
    _client(baseline).predict(_today(_as_given, GATHER), model=REL)
    response = {
        'fail': [
            {'at': 0, 'status': 200, 'body': baseline.responses[-1]['body']}
        ]
    }

    def edit(payloads: list[Any]) -> list[Any]:
        task = payloads[0]['task']
        if field == 'missing_kind':
            del task['kind']
        elif field != 'unedited':
            parent = (
                task['target']
                if field in {'column_name', 'dtype'}
                else task
                if field == 'options'
                else payloads[0]
            )
            parent[field] = value
        return payloads

    sessions = (
        _Watched(copy.deepcopy(response)),
        _Watched(copy.deepcopy(response)),
    )
    want, said = _outcome(
        lambda: (
            _client(sessions[0])
            .predict(_today(edit, GATHER), model=REL)
            .to_pandas()
        )
    )
    got, given = _outcome(
        lambda: host.frame(
            host.flatten(
                host.Client(_client(sessions[1])._transport).predict(
                    _core(edit, GATHER), model=REL
                )
            )
        )
    )
    assert want[0] == got[0] == 'value'
    pd.testing.assert_frame_equal(got[1], want[1])
    assert given == said
    assert all(session.failed is None for session in sessions)
    assert all(len(session.payloads) == 1 for session in sessions)
    assert _sent(sessions[1]) == _sent(sessions[0])


@pytest.mark.parametrize('field', ['column_name', 'dtype'])
@pytest.mark.parametrize(
    ('pinned', 'asked'),
    [
        (False, 0),
        (1, 1.0),
        ([False], [0]),
        ({'x': False}, {'x': 0}),
        ('TARGET', 'TARGET'),
        (False, False),
        (1, 2),
        (0, False),
        (1.0, True),
        (-0.0, False),
        ({'x': [False, {'y': 1}]}, {'x': [0, {'y': 1.0}]}),
        ({'x': 1, 'y': False}, {'y': 0, 'x': True}),
        (2**53 + 1, float(2**53)),
        (2**53, float(2**53)),
        (2**63 - 1, float(2**63)),
        (-(2**63), float(-(2**63))),
        (2**64 - 1, float(2**64)),
        (2**64 - 2048, float(2**64 - 2048)),
        ([False], [1]),
        ([False], [0, 0]),
        ({'x': False}, {'y': 0}),
        ({'x': False}, {'x': 0, 'y': 0}),
        ('1', 1),
        (None, False),
        ([], {}),
    ],
)
def test_fitted_session_compares_metadata_with_python_equality(
    field: str, pinned: Any, asked: Any
) -> None:
    _compare_fitted_metadata({field: pinned}, {field: asked})


@pytest.mark.parametrize('field', ['column_name', 'dtype'])
@pytest.mark.parametrize(
    ('pinned', 'asked'),
    [
        ((False,), [0]),
        ([0], (False,)),
        ({'x': (False,)}, {'x': [0]}),
        ({1: 'a'}, {'1': 'a'}),
        ({False: 'a'}, {0: 'a'}),
        ((False,), (0,)),
        ((False,), (1,)),
        (2**64, 2**64 + 1),
        (2**64 + 1, float(2**64)),
        (-(2**63) - 1, -(2**63)),
        (2**64, float(2**64)),
        (-(2**64), float(-(2**64))),
        (2**130, float(2**130)),
        (float(2**130), 2**130),
        (2**130 + 1, float(2**130)),
        (float(2**130), 2**130 + 1),
        (int(sys.float_info.max), sys.float_info.max),
        (int(sys.float_info.max) + 1, sys.float_info.max),
        (-int(sys.float_info.max), -sys.float_info.max),
        ({'x': [(False, {1: 2**130})]}, {'x': [(0, {True: 2**130})]}),
        ({'$tuple': [False]}, {'$tuple': [0]}),
        ({'$tuple': [False]}, (0,)),
        ({'$int': str(2**64)}, 2**64),
        ({'$map': [[False, 'a']]}, {'$map': [[0, 'a']]}),
        ({'$dict': {'$tuple': [False]}}, {'$dict': {'$tuple': [0]}}),
        ({None: 'a'}, {'null': 'a'}),
        ({1.5: 'a'}, {'1.5': 'a'}),
    ],
)
def test_fitted_session_preserves_typed_metadata(
    field: str, pinned: Any, asked: Any
) -> None:
    _compare_fitted_metadata({field: pinned}, {field: asked})


@pytest.mark.parametrize(
    ('pinned', 'asked'),
    [
        (
            {'column_name': (False,), 'dtype': 'float32'},
            {'column_name': (0,), 'dtype': 'float64'},
        ),
        (
            {'column_name': {False: 'a'}, 'dtype': 'float32'},
            {'column_name': {0: 'a'}, 'dtype': 'float64'},
        ),
        (
            {'column_name': 2**130, 'dtype': (False,)},
            {'column_name': float(2**130), 'dtype': [0]},
        ),
    ],
)
def test_fitted_session_typed_metadata_diagnostic_priority(
    pinned: dict[str, Any], asked: dict[str, Any]
) -> None:
    _compare_fitted_metadata(pinned, asked)


@pytest.mark.parametrize('field', ['column_name', 'dtype'])
@pytest.mark.parametrize('action', ['predict', 'fit'])
@pytest.mark.parametrize(
    'value', [{(1,): 'a'}, b'a', float('inf'), float('nan')]
)
def test_typed_metadata_transport_refusal_follows_construction(
    field: str, action: str, value: Any
) -> None:
    host = _host()

    def edit(payloads: list[Any]) -> list[Any]:
        payloads[0]['task']['target'][field] = value
        return payloads

    request = _today(edit, GATHER)
    core_request = _core(edit, GATHER)
    assert repr(getattr(request.task.target, field)) == repr(value)
    assert repr(core_request.state['task']['target'][field]) == repr(value)
    sessions = (_Watched(), _Watched())
    want, said = _outcome(
        lambda: getattr(_client(sessions[0]), action)(request, model=REL)
    )
    got, given = _outcome(
        lambda: getattr(host.Client(_client(sessions[1])._transport), action)(
            core_request, model=REL
        )
    )
    assert got == want
    assert want[0] == 'error'
    assert want[1][1] == 'INVALID_REQUEST'
    assert given == said == []
    assert all(session.failed is None for session in sessions)
    assert all(session.payloads == [] for session in sessions)
    assert _sent(sessions[1]) == _sent(sessions[0])


@pytest.mark.parametrize(
    ('pinned', 'asked'),
    [
        (
            {'column_name': False, 'dtype': 1},
            {'column_name': 0, 'dtype': 2},
        ),
        (
            {'column_name': {'x': [False]}, 'dtype': 'float32'},
            {'column_name': {'x': [0]}, 'dtype': 'float64'},
        ),
        (
            {'column_name': 1, 'dtype': False},
            {'column_name': 2, 'dtype': 0},
        ),
    ],
)
def test_fitted_session_reports_first_python_unequal_metadata(
    pinned: dict[str, Any], asked: dict[str, Any]
) -> None:
    _compare_fitted_metadata(pinned, asked)


@pytest.mark.parametrize(
    'field',
    [
        'kind',
        'column_name',
        'dtype',
        'options',
        'inference',
        'extra',
        'task_extra',
        'target_extra',
        'schema_extra',
        'context_extra',
        'predict_extra',
        'metadata_extra',
        'entity_table_names',
        'output_statistic',
        'run_mode',
        'inference_config',
        'inference_config_value',
        'anchor_time_column',
        'context_cell',
        'predict_cell',
    ],
)
@pytest.mark.parametrize(
    'value',
    [
        (1,),
        2**80 + 1,
        {1: 'one', '1': 'text'},
        {False: 'bool'},
        {'x': (1,)},
        {'x': 2**80 + 1},
        [[(1,), 'one']],
        {'$tuple': [1]},
        {'$map': [[1, 'one']]},
        {'$dict': {'$int': '9'}},
        {(1,): 'one'},
        {'x': b'bytes'},
        None,
        [],
        'ordinary',
    ],
)
def test_materialized_constructor_semantic_boundary(
    field: str, value: Any
) -> None:
    import pandas as pd

    host = _host()
    baseline = _Watched()
    _client(baseline).predict(_today(_as_given, GATHER), model=REL)
    response = {
        'fail': [
            {'at': 0, 'status': 200, 'body': baseline.responses[-1]['body']}
        ]
    }

    def edit(payloads: list[Any]) -> list[Any]:
        task = payloads[0]['task']
        if field in {
            'output_statistic',
            'run_mode',
            'inference_config',
            'inference_config_value',
            'anchor_time_column',
        }:
            if field == 'output_statistic':
                parent, key = (
                    payloads[0].setdefault('output', {}),
                    'prediction_statistic',
                )
            elif field == 'run_mode':
                parent, key = payloads[0]['inference'], 'run_mode'
            elif field == 'inference_config':
                parent, key = payloads[0]['inference'], 'inference_config'
            elif field == 'inference_config_value':
                parent, key = (
                    payloads[0]['inference']['inference_config'],
                    'num_estimators',
                )
            else:
                parent, key = (
                    task.setdefault('options', {}),
                    'anchor_time_column',
                )
            parent[key] = copy.deepcopy(value)
        elif field in {'context_cell', 'predict_cell'}:
            table = payloads[0][field.removesuffix('_cell')]['instance_table']
            table['rows'][0][-1] = copy.deepcopy(value)
        elif field.endswith('_extra'):
            container = field.removesuffix('_extra')
            parent = (
                task['target']
                if container == 'target'
                else payloads[0].setdefault(container, {})
            )
            parent['extra'] = copy.deepcopy(value)
        else:
            parent = (
                task['target']
                if field in {'column_name', 'dtype'}
                else task
                if field in {'kind', 'options', 'entity_table_names'}
                else payloads[0]
            )
            parent[field] = copy.deepcopy(value)
        return payloads

    def describe(request: Any, core: bool) -> str:
        if core:
            state = request.state
            task = state['task']
            return repr(
                (
                    task['kind'],
                    task['target']['column_name'],
                    task['target']['dtype'],
                    task['options'],
                    state['inference'],
                    state['entity_table'],
                    state['payloads'],
                )
            )
        return repr(
            (
                request.task.kind,
                request.task.target.column_name,
                request.task.target.dtype,
                request.task.options,
                request.inference,
                request.entity_table,
                list(request.payloads),
            )
        )

    want, said = _outcome(
        lambda: describe(
            _today(edit, GATHER, named=field != 'entity_table_names'), False
        )
    )
    got, given = _outcome(
        lambda: describe(
            _core(edit, GATHER, named=field != 'entity_table_names'), True
        )
    )
    assert got == want
    assert given == said
    sessions = (
        _Watched(copy.deepcopy(response)),
        _Watched(copy.deepcopy(response)),
    )
    want, said = _outcome(
        lambda: (
            _client(sessions[0])
            .predict(_today(edit, GATHER), model=REL)
            .to_pandas()
        )
    )
    got, given = _outcome(
        lambda: host.frame(
            host.flatten(
                host.Client(_client(sessions[1])._transport).predict(
                    _core(edit, GATHER), model=REL
                )
            )
        )
    )
    if want[0] == 'value':
        assert got[0] == 'value'
        pd.testing.assert_frame_equal(got[1], want[1])
    else:
        assert got == want
    assert given == said
    assert _sent(sessions[1]) == _sent(sessions[0])
    assert [entry['method'] for entry in sessions[1].responses] == [
        entry['method'] for entry in sessions[0].responses
    ]
    assert all(session.failed is None for session in sessions)


@pytest.mark.parametrize('how', ['predict', 'fit'])
@pytest.mark.parametrize(
    'field',
    ['query', 'entity_table', 'windowed', 'indices', 'requested_inference'],
)
@pytest.mark.parametrize(
    'value',
    [
        None,
        False,
        0,
        '',
        [],
        (1,),
        2**80 + 1,
        {1: 'one', '1': 'text'},
        {'x': (1,)},
        [[(1,), 'one']],
        {'$tuple': [1]},
    ],
)
def test_materialized_constructor_argument_boundary(
    field: str, value: Any, how: str
) -> None:
    import pandas as pd
    from sdm_client import Request

    host = _host()
    baseline = _today(_as_given, GATHER)
    schema = host.schema_from_data(shop())
    args = dict(
        payloads=list(baseline.payloads),
        query=baseline.query,
        entity_table=baseline.entity_table,
        windowed=False,
        indices=list(baseline.indices),
        requested_inference={},
    )
    args[field] = copy.deepcopy(value)

    def construct(core: bool) -> Any:
        return (
            host.request_materialized(schema=schema, **args)
            if core
            else Request.materialized(schema=baseline.schema, **args)
        )

    def describe(core: bool) -> Any:
        request = construct(core)
        return repr(
            request.state[field]
            if core
            else request.target_moves_with_time
            if field == 'windowed'
            else getattr(request, field)
        )

    want, said = _outcome(lambda: describe(False))
    got, given = _outcome(lambda: describe(True))
    assert got == want
    assert given == said

    reference = _Watched()
    _client(reference).predict(baseline, model=REL)
    response = {
        'fail': [
            {'at': 0, 'status': 200, 'body': reference.responses[-1]['body']}
        ]
    }
    sessions = (
        _Watched(copy.deepcopy(response) if how == 'predict' else None),
        _Watched(copy.deepcopy(response) if how == 'predict' else None),
    )

    def ask(core: bool) -> Any:
        client = _client(sessions[int(core)])
        if core:
            client = host.Client(client._transport)
        answer = getattr(client, how)(construct(core), model=REL)
        if how == 'fit':
            return 'fitted'
        return host.frame(host.flatten(answer)) if core else answer.to_pandas()

    want, said = _outcome(lambda: ask(False))
    got, given = _outcome(lambda: ask(True))
    if how == 'predict' and want[0] == 'value':
        assert got[0] == 'value'
        pd.testing.assert_frame_equal(got[1], want[1])
    else:
        assert got == want
    assert given == said
    assert _sent(sessions[1]) == _sent(sessions[0])
    assert [entry['method'] for entry in sessions[1].responses] == [
        entry['method'] for entry in sessions[0].responses
    ]
    assert all(session.failed is None for session in sessions)


@pytest.mark.parametrize(
    'case',
    [
        'list-cycle',
        'dict-cycle',
        'ancestor-cycle',
        'cycle-invalid-kind',
        'literal-cycle-tag',
        'literal-reference-tag',
        'list-160',
        'tuple-160',
        'list-220',
        'tuple-220',
        'native-excessive',
    ],
)
def test_materialized_constructor_terminates_at_boundaries(case: str) -> None:
    import subprocess

    script = f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r})['_materialized_termination_probe']({case!r})"
    completed = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize('depth', [511, 512, 513])
@pytest.mark.parametrize('shape', ['array', 'mixed', 'tagged'])
@pytest.mark.parametrize('threading', ['main', 'worker'])
def test_materialized_constructor_native_depth_ceiling(
    depth: int, shape: str, threading: str
) -> None:
    test_materialized_constructor_terminates_at_boundaries(
        f'ceiling-{depth}-{shape}-{threading}'
    )


@pytest.mark.parametrize('seed', [0, 1, 17, 41])
@pytest.mark.parametrize(
    'case', ['ordinary', 'mixed', 'requested', 'allowed', 'validation-first']
)
@pytest.mark.parametrize('how', ['predict', 'fit'])
def test_materialized_constructor_ordering_protocol(
    seed: int, case: str, how: str
) -> None:
    import os
    import subprocess

    script = f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r})['_materialized_ordering_probe']({case!r}, {how!r})"
    completed = subprocess.run(
        [sys.executable, '-c', script],
        env={**os.environ, 'PYTHONHASHSEED': str(seed)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _materialized_ordering_probe(case: str, how: str) -> None:
    import pandas as pd
    from sdm_client import Request
    from sdm_core import answers

    host = _host()
    baseline = _today(_as_given, GATHER)
    reference = _Watched()
    _client(reference).predict(baseline, model=REL)
    response = {
        'fail': [
            {'at': 0, 'status': 200, 'body': reference.responses[-1]['body']}
        ]
    }
    payloads = copy.deepcopy(list(baseline.payloads))
    config = payloads[0]['inference']['inference_config']
    if case in {'mixed', 'validation-first'}:
        config.update({1: 'number', '1': 'text', (2,): 'tuple', None: 'none'})
    elif case != 'ordinary':
        payloads[0]['inference']['inference_config'] = {
            3: 'unsupported-three',
            2: 'unsupported',
            1: 'unsupported-one',
            'num_estimators': 4,
        }
    if case == 'validation-first':
        payloads[0]['task']['kind'] = 'unknown'
    arguments = dict(
        payloads=payloads,
        query=baseline.query,
        indices=list(baseline.indices),
        entity_table=baseline.entity_table,
        windowed=False,
        requested_inference={'inference_config': {2: None}}
        if case == 'requested'
        else {},
    )
    calls: list[str] = []
    primitive = answers._host_protocol

    def observed(operation: str, operands: str) -> str:
        calls.append(operation)
        return primitive(operation, operands)

    answers._host_protocol = observed
    original = Request.materialized(schema=baseline.schema, **arguments)
    rebuilt = host.request_materialized(
        schema=host.schema_from_data(shop()), **arguments
    )
    assert calls == []
    assert repr(rebuilt.state['payloads']) == repr(list(original.payloads))
    sessions = (
        _Watched(copy.deepcopy(response) if how == 'predict' else None),
        _Watched(copy.deepcopy(response) if how == 'predict' else None),
    )

    def ask(core: bool) -> Any:
        client = _client(sessions[int(core)])
        if core:
            client = host.Client(client._transport)
        result = getattr(client, how)(rebuilt if core else original, model=REL)
        if how == 'fit':
            return 'fitted'
        return host.frame(host.flatten(result)) if core else result.to_pandas()

    want, said = _outcome(lambda: ask(False))
    got, given = _outcome(lambda: ask(True))
    if how == 'predict' and want[0] == 'value':
        assert got[0] == 'value'
        pd.testing.assert_frame_equal(got[1], want[1])
    else:
        assert got == want
    assert given == said
    assert _sent(sessions[1]) == _sent(sessions[0])
    assert all(session.failed is None for session in sessions)
    assert all(
        len(session.responses) == (1 if case in {'allowed', 'ordinary'} else 0)
        for session in sessions
    )
    assert calls == (
        []
        if case in {'validation-first', 'ordinary'}
        else ['sorted_set_difference']
        if case == 'mixed'
        else ['sorted_set_difference', 'sorted_set_intersection']
    )


def _materialized_termination_probe(case: str) -> None:
    host = _host()
    if case.startswith('ceiling-'):
        from concurrent.futures import ThreadPoolExecutor

        _, depth_text, shape, threading = case.split('-')
        depth = int(depth_text)
        schema = host.schema_from_data(shop())
        original = _today(_as_given, GATHER)
        payloads = list(original.payloads)
        payloads[0]['extra'] = '__DEPTH_SENTINEL__'
        state = dict(
            payloads=payloads,
            schema=schema.state,
            query=None,
            indices=[],
            entity_table='users',
            windowed=False,
            requested_inference={},
            tags=schema.frames.tags,
        )
        nested = '0'
        remaining = depth - 3
        while remaining:
            if shape == 'tagged' and remaining >= 2:
                nested = '{"$tuple":[' + nested + ']}'
                remaining -= 2
            else:
                nested = (
                    '{"child":' + nested + '}'
                    if shape == 'mixed' and remaining % 2
                    else '[' + nested + ']'
                )
                remaining -= 1
        encoded = json.dumps(state).replace('"__DEPTH_SENTINEL__"', nested)

        def exercise() -> None:
            outcome, said = _outcome(
                lambda: host._call(
                    host._native.request_materialized,
                    encoded,
                    schema.frames.tables,
                )
            )
            assert said == []
            if depth > 512:
                assert outcome[0] == 'error'
                assert 'nests more than 512 levels deep' in outcome[1][2]
                return
            assert outcome[0] == 'value', outcome
            answer, tables = outcome[1]
            built = json.loads(answer)
            asking = json.dumps(
                {
                    'request': built['request'],
                    'model': REL,
                    'outputs': None,
                    'tags': built['tags'],
                }
            )
            started, warned = _outcome(
                lambda: host._call(host._native.predict_run, asking, tables)
            )
            assert warned == []
            if started[0] == 'error':
                assert (
                    depth == 512
                    and 'nests more than 512 levels deep' in started[1][2]
                )
            else:
                step = started[1].next()
                assert step.kind == 'call_model'
                assert step.endpoint == 'predict'
                assert json.loads(step.body)['model'] == REL

        if threading == 'worker':
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(exercise).result(timeout=20)
        else:
            exercise()
        return
    if case == 'native-excessive':
        outcome, said = _outcome(
            lambda: host._call(
                host._native.request_materialized,
                '[' * 600 + '0' + ']' * 600,
                {},
            )
        )
        assert outcome[0] == 'error'
        assert 'nests more than 512 levels deep' in outcome[1][2]
        assert said == []
        return
    value: Any = []
    if case == 'list-cycle':
        value.append(value)
    elif case == 'dict-cycle':
        value = {}
        value['self'] = value
    elif case == 'ancestor-cycle':
        child: list[Any] = [value]
        value.append(child)
    elif case == 'cycle-invalid-kind':
        value.append(value)
    elif case == 'literal-cycle-tag':
        value = {'$refers_back': 'list'}
    elif case == 'literal-reference-tag':
        value = {'$refers_back': {'type': 'list', 'reference': 0}}
    else:
        shape, depth = case.split('-')
        for _unused in range(int(depth)):
            value = (value,) if shape == 'tuple' else [value]

    def edit(payloads: list[Any]) -> list[Any]:
        payloads[0]['extra'] = value
        if case == 'cycle-invalid-kind':
            payloads[0]['task']['kind'] = 'unknown'
        return payloads

    original = _today(edit, GATHER)
    rebuilt = _core(edit, GATHER)
    assert repr(rebuilt.state['payloads']) == repr(list(original.payloads))
    if case in {
        'list-cycle',
        'dict-cycle',
        'ancestor-cycle',
        'cycle-invalid-kind',
    }:
        assert rebuilt.state['payloads'][0]['extra'] is value
    sessions = _Watched(), _Watched()

    def ask(core: bool) -> None:
        client = _client(sessions[int(core)])
        if core:
            host.Client(client._transport).predict(rebuilt, model=REL)
        else:
            client.predict(original, model=REL)

    want, said = _outcome(lambda: ask(False))
    got, given = _outcome(lambda: ask(True))
    assert got == want
    assert given == said
    assert _sent(sessions[1]) == _sent(sessions[0])
    refused = case in {
        'list-cycle',
        'dict-cycle',
        'ancestor-cycle',
        'cycle-invalid-kind',
        'list-220',
        'tuple-220',
    }
    assert want[0] == ('error' if refused else 'value')
    assert all(
        len(session.responses) == (0 if refused else 1) for session in sessions
    )
    assert all(session.failed is None for session in sessions)


@pytest.mark.parametrize(
    'field',
    [
        'envelope',
        'task',
        'target',
        'schema',
        'context',
        'predict',
        'metadata',
        'inference',
        'inference_config',
        'output',
        'context_table',
        'predict_table',
        'schema_instance',
    ],
)
@pytest.mark.parametrize(
    'extra',
    [{1: 'one', '1': 'text'}, {False: (2**80 + 1,)}, {(1,): 'tuple-key'}],
)
@pytest.mark.parametrize('how', ['predict', 'fit', 'holdout'])
def test_materialized_constructor_structural_maps(
    field: str, extra: Any, how: str
) -> None:
    import pandas as pd

    host = _host()
    reference = _Watched()
    _client(reference).predict(_today(_as_given, GATHER), model=REL)
    response = {
        'fail': [
            {'at': 0, 'status': 200, 'body': reference.responses[-1]['body']}
        ]
    }

    def edit(payloads: list[Any]) -> list[Any]:
        if extra is None:
            half, component = field.split('_')
            table = payloads[0][half]['instance_table']
            if component == 'row':
                table['rows'][0] = tuple(table['rows'][0])
            else:
                table[component] = tuple(table[component])
            return payloads
        if field == 'envelope':
            parent = payloads[0]
        elif field == 'target':
            parent = payloads[0]['task']['target']
        elif field == 'inference_config':
            parent = payloads[0]['inference']['inference_config']
        elif field in {'context_table', 'predict_table'}:
            parent = payloads[0][field.removesuffix('_table')]['instance_table']
        elif field == 'schema_instance':
            parent = payloads[0]['schema']['instance_table']
        else:
            parent = payloads[0].setdefault(field, {})
        parent.update(extra)
        return payloads

    original = _today(edit, GATHER)
    rebuilt = _core(edit, GATHER)
    assert repr(rebuilt.state['payloads']) == repr(list(original.payloads))
    sessions = (
        _Watched(copy.deepcopy(response) if how == 'predict' else None),
        _Watched(copy.deepcopy(response) if how == 'predict' else None),
    )

    def ask(core: bool) -> Any:
        client = _client(sessions[int(core)])
        if core:
            client = host.Client(client._transport)
        result = getattr(client, 'fit' if how == 'holdout' else how)(
            rebuilt if core else original,
            model=REL,
            **({'holdout': HELD_OUT} if how == 'holdout' else {}),
        )
        if how in {'fit', 'holdout'}:
            return 'fitted'
        return host.frame(host.flatten(result)) if core else result.to_pandas()

    want, said = _outcome(lambda: ask(False))
    got, given = _outcome(lambda: ask(True))
    if how == 'predict' and want[0] == 'value':
        assert got[0] == 'value'
        pd.testing.assert_frame_equal(got[1], want[1])
    else:
        assert got == want
    assert given == said
    assert _sent(sessions[1]) == _sent(sessions[0])
    assert all(session.failed is None for session in sessions)


@pytest.mark.parametrize(
    'field',
    [
        'context_columns',
        'context_rows',
        'context_row',
        'predict_columns',
        'predict_rows',
        'predict_row',
    ],
)
@pytest.mark.parametrize('how', ['predict', 'fit'])
def test_materialized_constructor_sequence_containers(
    field: str, how: str
) -> None:
    test_materialized_constructor_structural_maps(field, None, how)


def _compare_fitted_metadata(
    pinned: dict[str, Any], asked: dict[str, Any]
) -> None:
    import pandas as pd

    host = _host()
    baseline = _Watched()
    fitted = _client(baseline).fit(_today(_as_given, GATHER), model=REL)
    fitted.predict(_today(_as_given, GATHER))
    response = {
        'fail': [
            {'at': 1, 'status': 200, 'body': baseline.responses[-1]['body']}
        ]
    }

    def edit(metadata: dict[str, Any]) -> Callable[[list[Any]], list[Any]]:
        def change(payloads: list[Any]) -> list[Any]:
            payloads[0]['task']['target'].update(copy.deepcopy(metadata))
            return payloads

        return change

    sessions = (
        _Watched(copy.deepcopy(response)),
        _Watched(copy.deepcopy(response)),
    )

    def today() -> Any:
        fitted = _client(sessions[0]).fit(
            _today(edit(pinned), GATHER), model=REL
        )
        return fitted.predict(_today(edit(asked), GATHER)).to_pandas()

    def core() -> Any:
        client = host.Client(_client(sessions[1])._transport)
        pinned_request = _core(edit(pinned), GATHER)
        asked_request = _core(edit(asked), GATHER)
        for metadata, request in (
            (pinned, pinned_request),
            (asked, asked_request),
        ):
            for field, value in metadata.items():
                assert repr(request.state['task']['target'][field]) == repr(
                    value
                )
                assert repr(
                    request.state['payloads'][0]['task']['target'][field]
                ) == repr(value)
        fitted = client.fit(pinned_request, model=REL)
        return host.frame(host.flatten(fitted.predict(asked_request)))

    want, said = _outcome(today)
    got, given = _outcome(core)
    if pinned == asked:
        assert want[0] == got[0] == 'value'
        pd.testing.assert_frame_equal(got[1], want[1])
    else:
        assert got == want
        assert want[0] == 'error'
        assert want[1][1] == 'INVALID_REQUEST'
    assert given == said == []
    assert all(session.failed is None for session in sessions)
    expected_posts = 2 if pinned == asked else 1
    assert all(
        sum(entry['method'] == 'POST' for entry in session.responses)
        == expected_posts
        for session in sessions
    )
    assert _sent(sessions[1]) == _sent(sessions[0])


@pytest.mark.parametrize('field', ['task', 'target'])
@pytest.mark.parametrize('missing', [False, True])
def test_envelope_constructor_distinguishes_missing_mapping_from_empty_list(
    field: str, missing: bool
) -> None:
    def edit(payloads: list[Any]) -> list[Any]:
        parent = payloads[0]['task'] if field == 'target' else payloads[0]
        if missing:
            del parent[field]
        else:
            parent[field] = []
        return payloads

    def today() -> Any:
        request = _today(edit, GATHER, named=False)
        return (
            request.task.kind,
            request.task.target.column_name,
            request.task.target.dtype,
        )

    def core() -> Any:
        task = _core(edit, GATHER, named=False).state['task']
        return (
            task['kind'],
            task['target']['column_name'],
            task['target']['dtype'],
        )

    want, said = _outcome(today)
    got, given = _outcome(core)
    assert want[0] == ('value' if missing else 'error')
    assert got == want
    assert given == said


@pytest.mark.parametrize('field', ['options', 'inference'])
@pytest.mark.parametrize(
    'value', [[['alpha']], [['alpha', 1], 0], [[[], 1]], [[{}, 1]]]
)
def test_envelope_constructor_preserves_dictionary_sequence_errors(
    field: str, value: Any
) -> None:
    host = _host()

    def edit(payloads: list[Any]) -> list[Any]:
        parent = payloads[0]['task'] if field == 'options' else payloads[0]
        parent[field] = value
        return payloads

    sessions = _Watched(), _Watched()
    want, said = _outcome(
        lambda: _client(sessions[0]).predict(_today(edit, GATHER), model=REL)
    )
    got, given = _outcome(
        lambda: host.Client(_client(sessions[1])._transport).predict(
            _core(edit, GATHER), model=REL
        )
    )
    assert want[0] == 'error'
    assert got == want
    assert given == said
    assert all(session.failed is None for session in sessions)
    assert all(not session.payloads for session in sessions)


@pytest.mark.parametrize('missing_inference', [False, True])
def test_valid_envelope_constructor_reaches_transport(
    missing_inference: bool,
) -> None:
    import pandas as pd

    host = _host()
    edit = _no_inference if missing_inference else _as_given
    sessions = _Watched(), _Watched()
    want, said = _outcome(
        lambda: (
            _client(sessions[0])
            .predict(_today(edit, GATHER), model=REL)
            .to_pandas()
        )
    )
    got, given = _outcome(
        lambda: host.frame(
            host.flatten(
                host.Client(_client(sessions[1])._transport).predict(
                    _core(edit, GATHER), model=REL
                )
            )
        )
    )
    assert want[0] == got[0] == 'value'
    pd.testing.assert_frame_equal(got[1], want[1])
    assert given == said
    assert all(session.failed is None for session in sessions)
    assert all(len(session.payloads) == 1 for session in sessions)
    assert _sent(sessions[1]) == _sent(sessions[0])


@pytest.mark.parametrize('payloads', [[], [None], [0], [[]], ['x']])
def test_envelope_constructor_requires_a_first_mapping(
    payloads: list[Any],
) -> None:
    host = _host()

    def edit(original: list[Any]) -> list[Any]:
        return payloads

    sessions = _Watched(), _Watched()
    want, said = _outcome(
        lambda: _client(sessions[0]).predict(_today(edit, GATHER), model=REL)
    )
    got, given = _outcome(
        lambda: host.Client(_client(sessions[1])._transport).predict(
            _core(edit, GATHER), model=REL
        )
    )
    assert want[0] == 'error'
    assert got == want
    assert given == said
    assert all(session.failed is None for session in sessions)
    assert all(not session.payloads for session in sessions)


@pytest.mark.parametrize('named', [False, True])
@pytest.mark.parametrize('has_entity', [False, True])
def test_envelope_constructor_stops_entity_lookup_when_named(
    named: bool, has_entity: bool
) -> None:
    def edit(payloads: list[Any]) -> list[Any]:
        if not has_entity:
            payloads[0]['task'].pop('entity_table_names', None)
        return [payloads[0], {'task': ['invalid']}]

    want, said = _outcome(
        lambda: _today(edit, GATHER, named=named).entity_table
    )
    got, given = _outcome(
        lambda: _core(edit, GATHER, named=named).state['entity_table']
    )
    assert want[0] == ('value' if named or has_entity else 'error')
    assert got == want
    assert given == said


def _context_null(payload: Any) -> None:
    payload['context'] = None


#: Rows of the context to hold out of a fit: every entity its rows name.
HELD_OUT = [1, 2, 3]


@pytest.mark.parametrize(
    ('change', 'how', 'quirks', 'named'),
    [
        pytest.param(
            _a_constant_target_renamed,
            'predict',
            None,
            True,
            id='a-constant-target-not-named-TARGET',
        ),
        pytest.param(
            _a_classification_without_its_target,
            'predict',
            _refused('the context holds no target'),
            True,
            id='a-classification-whose-context-lacks-its-target',
        ),
        pytest.param(
            _related_orders_not_a_table,
            'predict',
            _refused("semantic type 'sequence' is unsupported"),
            True,
            id='a-related-schema-not-a-table-refused-for-a-stype',
        ),
        pytest.param(
            _no_predict_reference,
            'predict',
            None,
            True,
            id='a-predict-half-without-an-entity-reference',
        ),
        pytest.param(
            _a_forecast,
            'predict',
            _answered(lambda item: {**item, 'forecast_step': None}),
            True,
            id='a-forecast-answered-without-a-step',
        ),
        pytest.param(
            _a_forecast,
            'predict',
            _answered(lambda item: {**item, 'forecast_step': 1}),
            True,
            id='a-forecast-answered-with-a-step',
        ),
        pytest.param(
            _no_predict_instance_ids,
            'predict',
            None,
            True,
            id='a-predict-half-without-instance-ids',
        ),
        pytest.param(
            _reference_from_another_column,
            'predict',
            None,
            True,
            id='a-reference-relationship-from-another-column',
        ),
        pytest.param(
            _reference_to_two_columns,
            'predict',
            None,
            True,
            id='a-reference-relationship-to-two-columns',
        ),
        pytest.param(
            _no_context_reference,
            'fit',
            None,
            True,
            id='a-fit-whose-context-lacks-the-entity-reference',
        ),
        pytest.param(
            _no_anchor_time,
            'fit',
            None,
            True,
            id='a-windowed-fit-whose-context-lacks-the-anchor-time',
        ),
        pytest.param(
            _without_context,
            'holdout',
            None,
            True,
            id='a-fit-holding-out-of-an-envelope-without-context',
        ),
        pytest.param(
            _context_without_columns,
            'holdout',
            None,
            True,
            id='a-fit-holding-out-of-a-context-without-columns',
        ),
        pytest.param(
            _cut_short,
            'holdout',
            None,
            True,
            id='a-fit-holding-out-of-a-context-row-cut-short',
        ),
        pytest.param(
            _related_without_rows,
            'holdout',
            None,
            True,
            id='a-fit-holding-out-of-related-tables-without-rows',
        ),
        pytest.param(
            _no_context_instance_ids,
            'holdout',
            None,
            True,
            id='a-fit-holding-out-of-a-context-without-instance-ids',
        ),
        pytest.param(
            _context_null,
            'predict',
            None,
            True,
            id='an-envelope-whose-context-is-null',
        ),
        pytest.param(
            _context_null,
            'holdout',
            None,
            True,
            id='a-fit-holding-out-of-an-envelope-whose-context-is-null',
        ),
        pytest.param(
            lambda payload: None,
            'predict',
            None,
            False,
            id='envelopes-asked-without-their-entity-table',
        ),
        pytest.param(
            _naming_no_entity_table,
            'predict',
            None,
            False,
            id='envelopes-naming-no-entity-table-asked-without-one',
        ),
    ],
)
def test_an_edited_envelope(
    change: Callable[[Any], None],
    how: str,
    quirks: Callable[[Any], Any] | None,
    named: bool,
) -> None:
    r"""An envelope a gather wrote, edited as a caller holding it can, then
    built into a request by ``Request.materialized`` (without its
    ``entity_table`` unless ``named``), and predicted, fitted, or fitted
    holding rows out: the adapter, the wire's key readers and the leak
    checks read what the engine never writes (``adapters/relational.py``
    ``_how_to_name_the_target``, ``_distinct_labels``,
    ``_columns_with_stype``, ``_named_entities``, ``_key_dtype``;
    ``wire/keys.py`` ``rows_in``, ``without_the_keys``, ``_drop_instances``;
    ``request.py`` ``_entity_table_in``). An endpoint's answer, where the
    route needs one, is the stand-in's answer to the unedited request,
    changed."""
    import pandas as pd

    host = _host()
    edit = _edited(change)
    given = None
    if quirks is not None:
        plain = SyntheticSession()
        _client(plain).predict(_today(_as_given, GATHER), model=REL)
        given = quirks(plain.responses[-1]['body'])

    def today(session: SyntheticSession) -> Any:
        client = _client(session)
        request = _today(edit, GATHER, named=named)
        if how == 'predict':
            return client.predict(request, model=REL).to_pandas()
        holdout = HELD_OUT if how == 'holdout' else None
        return client.fit(request, model=REL, holdout=holdout)._session_id

    def core(session: SyntheticSession) -> Any:
        client = host.Client(_client(session)._transport)
        request = _core(edit, GATHER, named=named)
        if how == 'predict':
            return host.frame(host.flatten(client.predict(request, model=REL)))
        holdout = HELD_OUT if how == 'holdout' else None
        fitted = client.fit(request, model=REL, holdout=holdout)
        return fitted.state['session_id']

    sessions = _Watched(copy.deepcopy(given)), _Watched(copy.deepcopy(given))
    want, said = _outcome(lambda: today(sessions[0]))
    got, given_back = _outcome(lambda: core(sessions[1]))
    # What the client says, not the stand-in endpoint failing to read it.
    assert sessions[0].failed is None, sessions[0].failed
    assert given_back == said
    assert _sent(sessions[1]) == _sent(sessions[0])
    assert got[0] == want[0], (got, want)
    if isinstance(want[1], pd.DataFrame):
        pd.testing.assert_frame_equal(got[1], want[1])
    else:
        assert got[1] == want[1]


# --- A request built from frames, its entity table unset ----------------------------


def test_a_request_from_frames_whose_entity_table_was_unset() -> None:
    r"""``request.entity_table = None`` on a request built from frames: the
    leak check has no table to read and says nothing
    (``request.py`` ``_warn_of_rows_scored_from_their_own_answers``), and the
    send refuses the table it cannot find."""
    host = _host()
    frame = people()

    def args() -> dict[str, Any]:
        return {
            'context': frame.iloc[4:],
            'predict': frame.iloc[:2].drop(columns=['churn']),
            'target': 'churn',
            'task': 'classification',
        }

    def today(session: SyntheticSession) -> Any:
        from sdm_client import Request

        request = Request(**args())
        request.entity_table = None
        return _client(session).predict(request, model=TAB).to_pandas()

    def core(session: SyntheticSession) -> Any:
        request = host.request(**args())
        request.state['entity_table'] = None
        client = host.Client(_client(session)._transport)
        return host.frame(host.flatten(client.predict(request, model=TAB)))

    sessions = SyntheticSession(), SyntheticSession()
    want, said = _outcome(lambda: today(sessions[0]))
    got, given = _outcome(lambda: core(sessions[1]))
    assert want[0] == 'error', want
    assert (got, given) == (want, said)
    assert _sent(sessions[1]) == _sent(sessions[0])


def test_a_request_from_frames_with_nothing_to_score_whose_entity_table_was_unset() -> (
    None
):
    r"""``request.entity_table = None`` on a request built from frames with
    no rows to score: nothing is sent, and the empty answer names no rows,
    there being no table to name them by (``client.py`` ``_empty_answer``)."""
    import pandas as pd

    host = _host()
    frame = people()

    def args() -> dict[str, Any]:
        return {
            'context': frame.iloc[4:],
            'predict': frame.iloc[:0].drop(columns=['churn']),
            'target': 'churn',
            'task': 'classification',
        }

    def today(session: SyntheticSession) -> Any:
        from sdm_client import Request

        request = Request(**args())
        request.entity_table = None
        return _client(session).predict(request, model=TAB).to_pandas()

    def core(session: SyntheticSession) -> Any:
        request = host.request(**args())
        request.state['entity_table'] = None
        client = host.Client(_client(session)._transport)
        return host.frame(host.flatten(client.predict(request, model=TAB)))

    sessions = SyntheticSession(), SyntheticSession()
    want, said = _outcome(lambda: today(sessions[0]))
    got, given = _outcome(lambda: core(sessions[1]))
    assert want[0] == got[0] == 'value', (want, got)
    assert want[1].empty and 'id' not in want[1].columns, want
    assert given == said
    assert _sent(sessions[1]) == _sent(sessions[0]) == []
    pd.testing.assert_frame_equal(got[1], want[1])


# --- A request from frames whose schema was edited ------------------------------------


def test_a_request_from_frames_whose_key_column_left_its_schema() -> None:
    r"""``del request.schema['instance'].columns['user_id']``, the key
    column, on a request built from frames: the leak check reads no key
    dtypes off the schema and passes none on (``wire/keys.py``
    ``key_dtypes``), and the send refuses the column the schema no longer
    describes."""
    host = _host()
    frame = people()

    def args() -> dict[str, Any]:
        return {
            'context': frame.iloc[:4],
            'predict': frame.iloc[:2].drop(columns=['churn']),
            'target': 'churn',
            'task': 'classification',
        }

    def today(session: SyntheticSession) -> Any:
        from sdm_client import Request

        request = Request(**args())
        del request.schema['instance'].columns['user_id']
        return _client(session).predict(request, model=TAB).to_pandas()

    def core(session: SyntheticSession) -> Any:
        request = host.request(**args())
        for table in request.state['schema']['schema']['tables']:
            if table['name'] == 'instance':
                table['columns'] = [
                    c for c in table['columns'] if c['name'] != 'user_id'
                ]
        client = host.Client(_client(session)._transport)
        return host.frame(host.flatten(client.predict(request, model=TAB)))

    sessions = SyntheticSession(), SyntheticSession()
    want, said = _outcome(lambda: today(sessions[0]))
    got, given = _outcome(lambda: core(sessions[1]))
    assert want[0] == 'error' and "['user_id']" in want[1][2], want
    assert (got, given) == (want, said)
    assert _sent(sessions[1]) == _sent(sessions[0])


def test_a_fitted_model_without_key_dtypes_asked_of_a_request_from_frames() -> (
    None
):
    r"""A session pinned from an envelope describing no instance table, so
    with no key dtypes, asked by ``FittedModel.predict`` of a request built
    from frames and given the fitted request's ``task`` and ``query`` and
    no context: the leak check settles the frame's key dtypes itself
    (``wire/keys.py`` ``scored_rows``), and the relational model then
    refuses a request that was not gathered.

    The core compares contexts by a request's ``context_fingerprint``, which
    ``request.context = None`` clears, as ``_what_the_context_holds(None)``
    gives none."""
    host = _host()
    users = shop()['users']

    def args() -> dict[str, Any]:
        return {
            'context': users.iloc[3:],
            'predict': users.iloc[:2].drop(columns=['age']),
            'target': 'age',
            'task': 'regression',
            'entity_table': 'users',
        }

    def today(session: SyntheticSession) -> Any:
        from sdm_client import Request

        gathered = _today(_no_instance_table_described, GATHER)
        fitted = _client(session).fit(gathered, model=REL)
        request = Request(**args())
        request.task = gathered.task
        request.query = gathered.query
        request.context = None
        return fitted.predict(request).to_pandas()

    def core(session: SyntheticSession) -> Any:
        gathered = _core(_no_instance_table_described, GATHER)
        client = host.Client(_client(session)._transport)
        fitted = client.fit(gathered, model=REL)
        request = host.request(**args())
        request.state['task'] = copy.deepcopy(gathered.state['task'])
        request.state['query'] = gathered.state['query']
        request.state['context_fingerprint'] = None
        return host.frame(host.flatten(fitted.predict(request)))

    sessions = SyntheticSession(), SyntheticSession()
    want, said = _outcome(lambda: today(sessions[0]))
    got, given = _outcome(lambda: core(sessions[1]))
    assert want[0] == 'error' and 'gathered first' in want[1][2], want
    assert (got, given) == (want, said)
    assert _sent(sessions[1]) == _sent(sessions[0])


# --- A result whose items were reassigned ---------------------------------------------


def test_a_forecast_result_whose_items_were_reassigned() -> None:
    r"""``result.items = ()`` on a forecast's result, then ``to_pandas()``:
    the empty frame keeps the step column the forecast names
    (``result.py`` ``_empty_frame``)."""
    import pandas as pd
    from _fixture_messages import FORECAST
    from sdm_client import PredictiveQuery, Schema
    from sdm_core.answers import Answer

    gather = {
        'indices': [1],
        'anchor_time': pd.Timestamp('2024-09-01'),
        'num_context_samples': 6,
    }
    host = _host()

    def today(session: SyntheticSession) -> Any:
        request = PredictiveQuery(
            FORECAST, Schema.from_data(shop())
        ).materialize(**gather)
        result = _client(session).predict(request, model=REL)
        result.items = ()
        return result.to_pandas()

    def core(session: SyntheticSession) -> Any:
        query = host.predictive_query(FORECAST, host.schema_from_data(shop()))
        client = host.Client(_client(session)._transport)
        answer = client.predict(host.materialize(query, **gather), model=REL)
        plain = json.loads(answer.plain)
        plain['items'] = []
        return host.frame(host.flatten(Answer(json.dumps(plain))))

    sessions = SyntheticSession(), SyntheticSession()
    want, said = _outcome(lambda: today(sessions[0]))
    got, given = _outcome(lambda: core(sessions[1]))
    assert want[0] == got[0] == 'value', (want, got)
    assert want[1].empty and 'forecast_step' in want[1].columns, want
    assert given == said
    assert _sent(sessions[1]) == _sent(sessions[0])
    pd.testing.assert_frame_equal(got[1], want[1])


# --- A send that fails with the endpoint's wording -----------------------------------


class _Dropped:
    r"""A transport session whose every post fails without an answer, the
    failure saying what an endpoint says of a column it will not read."""

    said = "semantic type 'text' is unsupported"

    def post(self, url: str, data: bytes, **kwargs: Any) -> Any:
        import requests

        raise requests.ConnectionError(self.said)

    def close(self) -> None:
        return None


def test_a_failed_send_that_reads_as_a_refusal() -> None:
    from sdm_client import PredictiveQuery, Schema

    crowd = shop()
    users = crowd['users'].iloc[[0] * 70].reset_index(drop=True)
    crowd['users'] = users.assign(
        user_id=range(1, 71), label=[f'label {i:02d}' for i in range(70)]
    )
    host = _host()

    def today() -> Any:
        request = PredictiveQuery(COUNT, Schema.from_data(crowd)).materialize(
            **GATHER
        )
        return _client(_Dropped()).predict(request, model=REL)

    def core() -> Any:
        query = host.predictive_query(COUNT, host.schema_from_data(crowd))
        request = host.materialize(query, **GATHER)
        return host.Client(_client(_Dropped())._transport).predict(
            request, model=REL
        )

    want, said = _outcome(today)
    got, given = _outcome(core)
    assert want[0] == 'error' and 'users.label' in want[1][2], want
    assert (got, given) == (want, said)


# --- A schema over a source built without from_source -------------------------------


@pytest.fixture
def source() -> Iterator[Any]:
    from sdm_connectors import DuckDBSource

    connection = duckdb.connect()
    connection.execute('SET threads TO 1')
    for name, frame in shop().items():
        connection.register(f'{name}_frame', frame)
        connection.execute(f'CREATE TABLE {name} AS SELECT * FROM {name}_frame')
        connection.unregister(f'{name}_frame')
    yield DuckDBSource(connection)
    connection.close()


def test_a_source_schema_built_by_its_constructor(source: Any) -> None:
    import pandas as pd
    from sdm_client import PredictiveQuery, Schema

    tables = {'users': 'users', 'orders': 'orders'}
    read = Schema.from_source(source, tables=tables)
    built = Schema(dict(read.tables), list(read.relationships), source=source)
    assert built._engine_tables is None
    today = SyntheticSession()
    want, said = _outcome(
        lambda: (
            _client(today)
            .predict(
                PredictiveQuery(COUNT, built).materialize(**GATHER), model=REL
            )
            .to_pandas()
        )
    )

    host = _host()
    held = host.schema_from_source(source, tables=tables)
    held.state['engine'] = []
    core = SyntheticSession()
    got, given = _outcome(
        lambda: host.frame(
            host.flatten(
                host.Client(_client(core)._transport).predict(
                    host.materialize(
                        host.predictive_query(COUNT, held), **GATHER
                    ),
                    model=REL,
                )
            )
        )
    )

    assert want[0] == got[0] == 'value', (want, got)
    assert given == said
    assert _sent(core) == _sent(today)
    pd.testing.assert_frame_equal(got[1], want[1])


def test_source_constructor_warning_comparison_rejects_one_sided_warning(
    source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = _host()
    original = host.Client.predict

    def changed(client: Any, *args: Any, **kwargs: Any) -> Any:
        warnings.warn('one-sided source warning', UserWarning, stacklevel=2)
        return original(client, *args, **kwargs)

    monkeypatch.setattr(host.Client, 'predict', changed)
    with pytest.raises(AssertionError):
        test_a_source_schema_built_by_its_constructor(source)


def test_a_table_another_source_schema_described(source: Any) -> None:
    r"""``schema.tables[name] = other.tables[name]``, a table of the same
    source that this schema has not read: the gather reads it then
    (``materializers/sources.py`` ``graph``)."""
    import pandas as pd
    from sdm_client import PredictiveQuery, Schema

    tables = {'users': 'users', 'orders': 'orders'}
    read = Schema.from_source(source, tables=tables)
    read.tables['products'] = Schema.from_source(source).tables['products']
    today = SyntheticSession()
    want, said = _outcome(
        lambda: (
            _client(today)
            .predict(
                PredictiveQuery(COUNT, read).materialize(**GATHER), model=REL
            )
            .to_pandas()
        )
    )

    host = _host()
    held = host.schema_from_source(source, tables=tables)
    host.put_table(held, 'products', host.schema_from_source(source))
    core = SyntheticSession()
    got, given = _outcome(
        lambda: host.frame(
            host.flatten(
                host.Client(_client(core)._transport).predict(
                    host.materialize(
                        host.predictive_query(COUNT, held), **GATHER
                    ),
                    model=REL,
                )
            )
        )
    )
    assert want[0] == got[0] == 'value', (want, got)
    assert given == said
    assert _sent(core) == _sent(today)
    pd.testing.assert_frame_equal(got[1], want[1])


# --- A schema that says more than its frames hold -----------------------------------


def _ghost() -> Any:
    from sdm_client import Column

    return Column('ghost', dtype='float64', stype='numerical')


def _hand_built(columns: Callable[[str, dict[str, Any]], None]) -> Any:
    r"""The tables of ``shop()``'s schema, built by hand as ``Table``\ s over
    its frames, each table's columns changed by ``columns``, and its
    relationships."""
    from sdm_client import Schema, Table

    described = Schema.from_data(shop())
    frames = shop()
    tables = {}
    for name, table in described.tables.items():
        held = dict(table.columns)
        columns(name, held)
        tables[name] = Table(
            name,
            frames[name],
            held,
            primary_key=table.primary_key,
            time_column=table.time_column,
        )
    return tables, list(described.relationships)


def _asked_of(
    make: Callable[[Any], Any],
    session: SyntheticSession,
    *,
    core: bool,
    query: str = COUNT,
    gather: Any = None,
    model: str = REL,
) -> Any:
    r"""``query`` asked of the schema ``make`` builds and sent, through
    today's Python or the core."""
    gather = GATHER if gather is None else gather
    if not core:
        from sdm_client import PredictiveQuery

        request = PredictiveQuery(query, make(None)).materialize(**gather)
        return _client(session).predict(request, model=model).to_pandas()
    host = _host()
    request = host.materialize(
        host.predictive_query(query, make(host)), **gather
    )
    client = host.Client(_client(session)._transport)
    return host.frame(host.flatten(client.predict(request, model=model)))


def _agree(make: Callable[[Any], Any], **how: Any) -> tuple[Any, Any]:
    import pandas as pd

    sessions = SyntheticSession(), SyntheticSession()
    want, said = _outcome(
        lambda: _asked_of(make, sessions[0], core=False, **how)
    )
    got, given = _outcome(
        lambda: _asked_of(make, sessions[1], core=True, **how)
    )
    assert given == said
    assert _sent(sessions[1]) == _sent(sessions[0])
    assert got[0] == want[0], (got, want)
    if isinstance(want[1], pd.DataFrame):
        pd.testing.assert_frame_equal(got[1], want[1])
    else:
        assert got[1] == want[1]
    return want, said


def test_a_column_put_into_a_schema_its_frame_lacks() -> None:
    r"""``schema[table].columns[name] = Column(...)`` for a column the frame
    does not hold: the graph is built without it
    (``materializers/relational.py`` ``_build_graph``, ``_as_typed``)."""

    def make(host: Any) -> Any:
        if host is None:
            from sdm_client import Schema

            schema = Schema.from_data(shop())
            schema['users'].columns['ghost'] = _ghost()
            return schema
        schema = host.schema_from_data(shop())
        host.put_column(schema, 'users', _ghost())
        return schema

    want, _ = _agree(make)
    assert want[0] == 'value', want


def test_a_hand_built_schema_naming_a_column_its_frame_lacks() -> None:
    r"""``Schema(tables, relationships)`` of hand-built ``Table``\ s, one of
    which names a column its frame does not hold
    (``materializers/relational.py`` ``_build_graph``, ``_as_typed``)."""

    def ghostly(name: str, columns: dict[str, Any]) -> None:
        if name == 'users':
            columns['ghost'] = _ghost()

    def make(host: Any) -> Any:
        tables, relationships = _hand_built(ghostly)
        if host is None:
            from sdm_client import Schema

            return Schema(tables, relationships)
        return host.schema_of_tables(tables, relationships)

    want, _ = _agree(make)
    assert want[0] == 'value', want


def test_a_relationship_over_a_column_its_frame_lacks() -> None:
    r"""A column put into a schema and a relationship appended over it: the
    frame does not hold the column the relationship joins on
    (``materializers/relational.py`` ``_build_graph``)."""
    from sdm_client import Column, Relationship

    favourite = Relationship('users', 'fav', 'products', 'product_id')

    def make(host: Any) -> Any:
        column = Column('fav', dtype='int64', stype='ID')
        if host is None:
            from sdm_client import Schema

            schema = Schema.from_data(shop())
            schema['users'].columns['fav'] = column
            schema.relationships.append(favourite)
            return schema
        schema = host.schema_from_data(shop())
        host.put_column(schema, 'users', column)
        host.put_relationship(schema, favourite)
        return schema

    want, _ = _agree(make)
    assert want[0] == 'error' and "'fav'" in want[1][2], want


def test_a_decimal_column_put_into_a_schema_under_a_key_not_text() -> None:
    r"""A column of decimals added to a frame under the label ``5`` after
    ``Schema.from_data`` read it, and put into its table's columns under the
    same key: casting it to its dtype makes the engine's frame with a
    keyword that is not text, which today's Python refuses as a call this
    client made wrongly (``materializers/relational.py`` ``_as_typed``,
    ``_engine_errors``, ``_is_call_signature_error``)."""
    from decimal import Decimal

    from sdm_client import Column

    def make(host: Any) -> Any:
        frames = shop()
        column = Column('5', dtype='float64', stype='numerical')
        if host is None:
            from sdm_client import Schema

            schema = Schema.from_data(frames)
            frames['users'][5] = [Decimal('1.5')] * len(frames['users'])
            schema['users'].columns[5] = column
            return schema
        schema = host.schema_from_data(frames)
        frames['users'][5] = [Decimal('1.5')] * len(frames['users'])
        host.put_column(schema, 'users', column, key=5)
        return schema

    want, _ = _agree(make)
    assert want[0] == 'error' and 'called incorrectly' in want[1][2], want


def test_a_table_deleted_from_a_schemas_tables() -> None:
    r"""``del schema.tables['products']``, a table a relationship points at:
    the schema hands out its own dict, so the relationship stays, the check
    for links that join nothing reads past the table it lacks
    (``materializers/relational.py`` ``_joins_nothing``), and the gather
    refuses the table it cannot find."""

    def make(host: Any) -> Any:
        if host is None:
            from sdm_client import Schema

            schema = Schema.from_data(shop())
            del schema.tables['products']
            return schema
        schema = host.schema_from_data(shop())
        host.drop_table(schema, 'products')
        return schema

    want, _ = _agree(make)
    assert want[0] == 'error' and "no table 'products'" in want[1][2], want


def test_a_tabular_target_its_frame_lacks() -> None:
    r"""A target put into a schema's columns that its frame does not hold,
    asked on the tabular route with rows to score: today raises pandas'
    ``KeyError`` (``materializers/tabular.py`` ``table``)."""
    from sdm_client import Column

    def make(host: Any) -> Any:
        column = Column('score', dtype='float64', stype='numerical')
        if host is None:
            from sdm_client import Schema

            schema = Schema.from_data({'people': people()})
            schema['people'].columns['score'] = column
            return schema
        schema = host.schema_from_data({'people': people()})
        host.put_column(schema, 'people', column)
        return schema

    gather = {
        'predict': people().drop(columns=['churn']).head(2),
        'model': TAB,
    }
    want, _ = _agree(
        make,
        query='PREDICT people.score FOR EACH people.user_id',
        gather=gather,
        model=TAB,
    )
    assert want[0] == 'error' and 'score' in want[1][2], want


# --- A table emptied after its schema was read ---------------------------------------


def test_a_postgres_table_emptied_after_its_schema_was_read() -> None:
    r"""``Schema.from_source`` reads a sample of each table and refuses an
    empty one there, so the row count the gather takes
    (``PostgresTable._get_num_rows``) finds one empty only when its rows go
    in between; the engine's ``RuntimeError`` then comes back as an internal
    error."""
    import os

    from _parity.runner import _source, real_connection
    from sdm_client import PredictiveQuery, Schema

    if not os.environ.get('SDM_POSTGRES_TEST_DSN'):
        if os.environ.get('SDM_REQUIRE_POSTGRES') == '1':
            pytest.fail('SDM_REQUIRE_POSTGRES is set and no DSN is given')
        pytest.skip('needs SDM_POSTGRES_TEST_DSN')
    transcript = {
        'backend': 'postgres',
        'data': {
            'generator': 'marketplace',
            'seed': 0,
            'namespace': {'schema': 'request_emptied'},
        },
    }
    names = ['customers', 'items', 'reviews', 'stores', 'purchases']
    read = {
        'tables': {name: name for name in names},
        'primary_keys': {'stores': 'store_code'},
    }
    query = (
        'PREDICT COUNT(purchases.*, 0, 30, days) FOR EACH customers.customer_id'
    )
    gather = {'indices': [1, 2], 'num_context_samples': 30, 'random_seed': 7}
    host = _host()
    with real_connection(transcript) as connection:
        source = _source(transcript, connection, None)
        schema = Schema.from_source(source, **read)
        held = host.schema_from_source(source, **read)
        connection.execute('DELETE FROM request_emptied.reviews')
        want = _outcome(
            lambda: PredictiveQuery(query, schema).materialize(**gather)
        )
        got = _outcome(
            lambda: host.materialize(
                host.predictive_query(query, held), **gather
            )
        )
    assert want[0][0] == 'error' and 'is empty' in want[0][1][2], want
    assert got == want


# --- Frames changed after the schema read them --------------------------------------

#: What a caller can do to a frame of ``shop()`` in place, each a statement
#: between calls that neither corpus vocabulary makes.
CHANGES: dict[str, Callable[[dict[str, Any]], None]] = {
    'a-column-dropped': lambda f: f['users'].drop(
        columns=['age'], inplace=True
    ),
    'a-column-renamed': lambda f: f['users'].rename(
        columns={'age': 'years'}, inplace=True
    ),
    'a-column-added': lambda f: f['users'].insert(
        1, 'height', [float(i) for i in range(len(f['users']))]
    ),
    'values-changed': lambda f: f['users'].loc.__setitem__(
        (slice(None), 'age'), 99
    ),
    'a-column-retyped': lambda f: f['users'].__setitem__(
        'age', f['users']['age'].astype(str)
    ),
    'a-row-dropped': lambda f: f['users'].drop(
        index=f['users'].index[:1], inplace=True
    ),
    'a-row-added': lambda f: f['users'].loc.__setitem__(
        len(f['users']) + 100, [99, 50, 'FR']
    ),
    'a-key-repeated': lambda f: f['users'].__setitem__('user_id', 1),
    'the-time-column-dropped': lambda f: f['orders'].drop(
        columns=['ordered_at'], inplace=True
    ),
    'a-foreign-key-dropped': lambda f: f['orders'].drop(
        columns=['user_id'], inplace=True
    ),
    'a-table-emptied': lambda f: f['users'].drop(
        index=f['users'].index, inplace=True
    ),
    # A name the wire needs, which today refuses only where the schema
    # described it: added after, the payload's own column collides with it.
    'a-transport-key-added': lambda f: f['users'].insert(
        1, 'instance_id', range(len(f['users']))
    ),
    # Added after, the anchor time is a column like any other.
    'an-anchor-time-added': lambda f: f['users'].insert(
        1, '__kumo_anchor_time', _timestamp('2024-01-01')
    ),
    # Labelled by an int, the column is one the engine's LocalTable cannot
    # name (`ColumnSpec.coerce`).
    'a-column-labelled-by-an-int': lambda f: f['users'].insert(
        1, 5, [1.5] * len(f['users'])
    ),
    # Relabelled, the frame is one the engine's LocalTable refuses.
    'a-multi-index-given': lambda f: setattr(
        f['users'],
        'columns',
        _multi_index([('user_id', ''), ('age', ''), ('country', '')]),
    ),
    'repeated-names-given': lambda f: setattr(
        f['users'], 'columns', ['user_id', 'extra', 'extra']
    ),
    'an-empty-name-given': lambda f: setattr(
        f['users'], 'columns', ['user_id', 'age', '']
    ),
}


def _multi_index(labels: list[tuple[str, str]]) -> Any:
    import pandas as pd

    return pd.MultiIndex.from_tuples(labels)


def _timestamp(text: str) -> Any:
    import pandas as pd

    return pd.Timestamp(text)


class _Today:
    r"""The calls, made through today's Python."""

    def __init__(self, session: SyntheticSession) -> None:
        self.client = _client(session)

    def schema(self, frames: dict[str, Any]) -> Any:
        from sdm_client import Schema

        return Schema.from_data(frames)

    def query(self, schema: Any) -> Any:
        from sdm_client import PredictiveQuery

        return PredictiveQuery(COUNT, schema)

    def materialize(self, query: Any) -> Any:
        return query.materialize(**GATHER)

    def predict(self, request: Any) -> Any:
        return self.client.predict(request, model=REL).to_pandas()


class _Core(_Today):
    r"""The calls, made through the core by the host."""

    def __init__(self, session: SyntheticSession) -> None:
        self.host = _host()
        self.client = self.host.Client(_client(session)._transport)

    def schema(self, frames: dict[str, Any]) -> Any:
        return self.host.schema_from_data(frames)

    def query(self, schema: Any) -> Any:
        return self.host.predictive_query(COUNT, schema)

    def materialize(self, query: Any) -> Any:
        return self.host.materialize(query, **GATHER)

    def predict(self, request: Any) -> Any:
        return self.host.frame(
            self.host.flatten(self.client.predict(request, model=REL))
        )


def _asked(
    calls: type[_Today], when: str, change: Callable[[dict[str, Any]], None]
) -> tuple[Any, list[Any]]:
    r"""What a question over ``shop()`` gives, ``change`` made to its frames
    ``when``: after the schema read them, between two gathers of one query
    (and before a new query's), or between a gather and its send."""
    session = SyntheticSession()
    api = calls(session)

    def ask() -> list[Any]:
        frames = shop()
        schema = api.schema(frames)
        query = api.query(schema)
        if when == 'after-the-schema':
            change(frames)
            return [api.predict(api.materialize(query))]
        if when == 'between-gathers':
            first = api.predict(api.materialize(query))
            change(frames)
            again = api.predict(api.materialize(query))
            fresh = _outcome(
                lambda: api.predict(api.materialize(api.query(schema)))
            )
            return [first, again, fresh]
        request = api.materialize(query)
        change(frames)
        return [api.predict(request)]

    got = _outcome(ask)
    return got, _sent(session)


def _same(got: Any, want: Any) -> None:
    import pandas as pd

    if isinstance(want, pd.DataFrame):
        pd.testing.assert_frame_equal(got, want)
    elif isinstance(want, (list, tuple)):
        assert type(got) is type(want) and len(got) == len(want), (got, want)
        for one, other in zip(got, want, strict=True):
            _same(one, other)
    else:
        assert got == want


#: Changes a Phase 9 task 7b fix refuses by name where today's code did not
#: (``CHANGELOG.md``), and the column each refusal names; once refused, a
#: gather sends nothing more.
FIXED_IN_THE_CORE = {
    'a-transport-key-added': 'users.instance_id',
    'an-anchor-time-added': 'users.__kumo_anchor_time',
    'the-target-dropped': 'people.churn',
    'the-entity-dropped': 'people.user_id',
}


def _refused_by_the_fix(change: str, got: Any) -> None:
    column = FIXED_IN_THE_CORE[change]
    if change in ('a-transport-key-added', 'an-anchor-time-added'):
        renamed = {
            'a-transport-key-added': 'instance_id_from_the_data',
            'an-anchor-time-added': 'kumo_anchor_time_from_the_data',
        }[change]
        code = 'INVALID_SCHEMA'
        message = (
            f"'{column}' uses a name this client needs for the request it "
            'builds, so it cannot also be a column of the data. Rename it, '
            f"for instance to '{renamed}'."
        )
    else:
        code = 'INVALID_REQUEST'
        message = (
            f"'{column}' is no longer in the frame of 'people' that "
            'Schema.from_data read, and the question reads it. Put it back, '
            'or build the schema again from the frame as it is now'
        )
    assert got == (
        'error',
        ('sdm_client.errors.StructuredDataError', code, message, {}),
    )


@pytest.mark.parametrize(
    'when', ['after-the-schema', 'between-gathers', 'after-a-gather']
)
@pytest.mark.parametrize('change', list(CHANGES))
def test_a_frame_changed_after_its_schema_read_it(
    change: str, when: str
) -> None:
    (want, said), sent = _asked(_Today, when, CHANGES[change])
    (got, given), posted = _asked(_Core, when, CHANGES[change])
    if change in FIXED_IN_THE_CORE and when != 'after-a-gather':
        assert given == said == []
        expected_count = {
            ('a-transport-key-added', 'after-the-schema'): 0,
            ('a-transport-key-added', 'between-gathers'): 4,
            ('an-anchor-time-added', 'after-the-schema'): 2,
            ('an-anchor-time-added', 'between-gathers'): 6,
        }[(change, when)]
        assert len(sent) == expected_count
        assert posted == (
            [sent[0], sent[len(sent) // 2]] if when == 'between-gathers' else []
        )
        old_error = (
            'error',
            (
                'sdm_client.errors.StructuredDataError',
                'INVALID_REQUEST',
                'cannot insert instance_id, already exists',
                {},
            ),
        )
        if change == 'a-transport-key-added':
            if when == 'after-the-schema':
                assert want == old_error
            else:
                assert want[0] == 'value' and len(want[1]) == 3
                _same(want[1][0], want[1][1])
                assert want[1][2] == (old_error, [])
        else:
            assert want[0] == 'value'
            assert len(want[1]) == (3 if when == 'between-gathers' else 1)
            if when == 'between-gathers':
                _same(want[1][0], want[1][1])
                assert want[1][2][0][0] == 'value'
                _same(want[1][0], want[1][2][0][1])
                assert want[1][2][1] == []
        _refused_by_the_fix(change, got)
        return
    assert given == said
    assert posted == sent
    assert got[0] == want[0], (got, want)
    _same(got[1], want[1])


def test_a_frame_changed_before_a_query_gathers_is_read_as_it_stands() -> None:
    r"""The schema says what the frame held when it was read; the gather
    reads the frame as it is now, so a dropped column is not sent."""
    dropped = CHANGES['a-column-dropped']
    (want, _), sent = _asked(_Today, 'after-the-schema', dropped)
    (got, _), posted = _asked(_Core, 'after-the-schema', dropped)
    assert want[0] == got[0] == 'value'
    users = [
        body[half]['related_tables']['users']
        for body in sent
        if 'predict' in body
        for half in ('context', 'predict')
    ]
    assert users and all('age' not in table['columns'] for table in users)
    assert posted == sent


@pytest.mark.parametrize(
    'change',
    ['a-column-dropped', 'values-changed', 'a-row-dropped', 'a-column-retyped'],
)
def test_a_frame_changed_after_a_gather_refused_late(change: str) -> None:
    r"""A gather refused after its model was built leaves that model with
    the query, so the next gather goes on with the graph it built, not the
    frames as they now stand."""

    def asked(calls: type[_Today]) -> tuple[Any, list[Any]]:
        session = SyntheticSession()
        api = calls(session)

        def ask() -> Any:
            frames = shop()
            query = api.query(api.schema(frames))
            late = {**GATHER, 'indices': [1, 2, 999]}
            refused = _outcome(
                lambda: (
                    query.materialize(**late)
                    if calls is _Today
                    else api.host.materialize(query, **late)
                )
            )
            CHANGES[change](frames)
            return refused, api.predict(api.materialize(query))

        return _outcome(ask), _sent(session)

    (want, said), sent = asked(_Today)
    (got, given), posted = asked(_Core)
    assert want[0] == 'value', want
    refused, _ = want[1][0]
    assert refused[0] == 'error', want
    assert given == said
    assert posted == sent
    assert got[0] == want[0], (got, want)
    assert got[1][0] == want[1][0]
    _same(got[1][1], want[1][1])


# --- Frames changed on the tabular route ---------------------------------------------

CHURN = 'PREDICT people.churn FOR EACH people.user_id'

#: What a caller can do in place to the one frame the tabular model reads.
TABULAR_CHANGES: dict[str, Callable[[dict[str, Any]], None]] = {
    'a-column-dropped': lambda f: f['people'].drop(
        columns=['age'], inplace=True
    ),
    'a-column-added': lambda f: f['people'].insert(1, 'height', 1.5),
    'values-changed': lambda f: f['people'].loc.__setitem__(
        (slice(None), 'age'), 5.0
    ),
    'rows-dropped': lambda f: f['people'].drop(
        index=f['people'].index[:2], inplace=True
    ),
    'every-row-dropped': lambda f: f['people'].drop(
        index=f['people'].index, inplace=True
    ),
    # `drop(columns=[target])` and `entity_table[entity]` raise pandas' own
    # `KeyError`s, each with its own wording.
    'the-target-dropped': lambda f: f['people'].drop(
        columns=['churn'], inplace=True
    ),
    'the-entity-dropped': lambda f: f['people'].drop(
        columns=['user_id'], inplace=True
    ),
    'a-column-labelled-by-an-int': lambda f: f['people'].insert(1, 5, 1.5),
    'the-target-made-text': lambda f: f['people'].__setitem__(
        'churn', f['people']['churn'].astype(str)
    ),
}


def _tabular(
    calls: type[_Today],
    route: str,
    when: str,
    change: Callable[[dict[str, Any]], None],
) -> tuple[Any, list[Any]]:
    r"""What a churn question over ``people()`` gives through ``route``
    (``indices=`` or ``predict=``), ``change`` made to its frame ``when``:
    before the gather, between the gather and its send, or between two
    gathers of one query."""
    session = SyntheticSession()
    api = calls(session)
    host = api.host if isinstance(api, _Core) else None

    def gather(query: Any) -> Any:
        args: dict[str, Any] = {'model': TAB}
        if route == 'indices':
            args['indices'] = [1, 2]
        else:
            args['predict'] = people().iloc[:3].drop(columns=['churn'])
        if host is not None:
            return host.materialize(query, **args)
        return query.materialize(**args)

    def send(request: Any) -> Any:
        if host is not None:
            return host.frame(
                host.flatten(api.client.predict(request, model=TAB))
            )
        return api.client.predict(request, model=TAB).to_pandas()

    def ask() -> list[Any]:
        frames = {'people': people()}
        schema = api.schema(frames)
        query = (
            host.predictive_query(CHURN, schema)
            if host is not None
            else _predictive_query(CHURN, schema)
        )
        if when == 'before-the-gather':
            change(frames)
            return [send(gather(query))]
        request = gather(query)
        if when == 'after-a-gather':
            change(frames)
            return [send(request)]
        first = send(request)
        change(frames)
        return [first, _outcome(lambda: send(gather(query)))]

    return _outcome(ask), _sent(session)


def _predictive_query(text: str, schema: Any) -> Any:
    from sdm_client import PredictiveQuery

    return PredictiveQuery(text, schema)


@pytest.mark.parametrize(
    'when', ['before-the-gather', 'after-a-gather', 'between-gathers']
)
@pytest.mark.parametrize('route', ['indices', 'predict'])
@pytest.mark.parametrize('change', list(TABULAR_CHANGES))
def test_a_frame_changed_on_the_tabular_route(
    change: str, route: str, when: str
) -> None:
    (want, said), sent = _tabular(_Today, route, when, TABULAR_CHANGES[change])
    (got, given), posted = _tabular(_Core, route, when, TABULAR_CHANGES[change])
    if change in FIXED_IN_THE_CORE and when != 'after-a-gather':
        binary_warning = [
            (
                'sdm_client.core.warnings.SDMWarning',
                'people.churn holds two whole numbers, which is how a binary '
                'label is usually written, so this is being asked as a '
                'classification. If they are quantities rather than labels, '
                'pass task="regression" to ask for one; pass '
                'task="classification" to say this was meant and silence this.',
            )
        ]
        old_warns = change == 'the-entity-dropped' and route == 'predict'
        assert given == (binary_warning if when == 'between-gathers' else [])
        assert said == (
            binary_warning if when == 'between-gathers' or old_warns else []
        )
        assert len(sent) == (2 if when == 'between-gathers' else 0)
        assert posted == sent
        if when == 'between-gathers':
            assert got[0] == want[0] == 'value'
            assert len(got[1]) == len(want[1]) == 2
            _same(got[1][0], want[1][0])
            (got, native_warnings) = got[1][1]
            (want, old_warnings) = want[1][1]
            assert native_warnings == []
            assert old_warnings == (binary_warning if old_warns else [])
        if old_warns:
            expected_old = (
                'sdm_client.errors.StructuredDataError',
                'INVALID_REQUEST',
                "predict has column(s) ['user_id'] that context does not; "
                'the model is shown context first, so it cannot score a '
                'feature it has never been given',
                {},
            )
        else:
            message = (
                (
                    '"[\'churn\'] not found in axis"'
                    if route == 'indices'
                    else "'churn'"
                )
                if change == 'the-target-dropped'
                else "'user_id'"
            )
            expected_old = ('builtins.KeyError', None, message, {})
        assert want == ('error', expected_old)
        _refused_by_the_fix(change, got)
        return
    assert given == said
    assert posted == sent
    assert got[0] == want[0], (got, want)
    _same(got[1], want[1])
