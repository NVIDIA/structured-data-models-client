# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three ways a request was answered about rows it should not have been.

A session pins the context the `fit()` request gathered, which excluded the rows
*that* request named. Every later call names different rows, and nothing
excluded those, so they sit in the pinned context with their target values and
the model is asked about rows it has been shown the answers for. Measured on a
target that is pure noise: a session scored 0.595 where the same rows scored
one-shot came back 0.455, against a chance rate of 0.5.

A key the caller set to `None` was let through and then discarded, so the
engine's own guess -- made from a column *name*, without checking uniqueness --
survived, and linking that column was refused for being a primary key the
caller had explicitly removed.

And the engine's node lookup cast keys to the Python type of its index's first
element, which for an object index is a class rather than a dtype.
"""

from __future__ import annotations

import decimal
import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Relationship,
    Schema,
)
from sdm_client.errors import StructuredDataError
from sdm_client.wire.keys import _named_by, keys_in, scored_keys

D = decimal.Decimal


# --- the keys a pinned context holds ---------------------------------------


def envelope(half: str, columns, rows, primary_key=None) -> dict:
    payload = {half: {'instance_table': {'columns': columns, 'rows': rows}}}
    if primary_key is not None:
        payload['schema'] = {'instance_table': {'primary_key': primary_key}}
    return payload


def test_a_tabular_context_is_read_by_its_declared_key():
    payload = envelope(
        'context',
        ['rid', 'x', 'label'],
        [[1, 0.5, 'a'], [2, 0.6, 'b']],
        primary_key=['rid'],
    )
    assert keys_in(payload, 'context') == {_named_by([1]), _named_by([2])}


def test_a_relational_context_is_read_by_its_entity_reference():
    """The two models name the key differently, and the reference wins because
    the caller's own key never travels on that path."""
    payload = envelope(
        'context',
        ['instance_id', '__kumo_entity_ref_0', 'TARGET'],
        [[0, 7, 1.0], [1, 11, 2.0]],
        primary_key=['instance_id'],
    )
    assert keys_in(payload, 'context') == {_named_by([7]), _named_by([11])}


@pytest.mark.parametrize(
    'payload',
    [
        None,
        {},
        {'context': {}},
        {'context': {'instance_table': {}}},
        envelope('context', ['a'], [[1]]),
        envelope('context', ['a'], [[1]], primary_key=['b']),
        envelope('context', 'not a list', [[1]], primary_key=['a']),
    ],
    ids=[
        'none',
        'empty',
        'no instance table',
        'no rows',
        'no key named',
        'key not a column',
        'columns not a list',
    ],
)
def test_a_context_this_cannot_read_yields_no_keys(payload):
    """Reporting a key it did not find would be worse than reporting none:
    the warning it feeds is about rows, and a wrong row set is a wrong claim."""
    assert keys_in(payload, 'context') == frozenset()


def test_a_row_too_short_to_hold_the_key_is_skipped():
    payload = envelope(
        'context', ['x', 'rid'], [[0.5, 1], [0.6]], primary_key=['rid']
    )
    assert keys_in(payload, 'context') == {_named_by([1])}


# --- scoring rows the context already answers ------------------------------


def frame(rows: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(31)
    return pd.DataFrame(
        {
            'rid': np.arange(rows),
            'x1': rng.normal(0, 1, rows),
            'label': rng.integers(0, 2, rows).astype(str),
        }
    )


def query(data: pd.DataFrame) -> PredictiveQuery:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': data}, primary_keys={'t': 'rid'})
    return PredictiveQuery('PREDICT t.label FOR EACH t.rid', schema)


class Transport:
    def delete_session(self, session_id: str) -> None:
        pass


class Client:
    def __init__(self) -> None:
        self._transport = Transport()


def handle(pinned_keys) -> object:
    from sdm_client.client import FittedModel

    class Adapter:
        class spec:  # noqa: N801 - a stand-in for a ModelSpec
            model_id = 'kumo-tabular'

        def session_prediction_plan(self, request, outputs):
            return ()

    return FittedModel(
        client=Client(),
        adapter=Adapter(),
        session_id='s1',
        outputs=('prediction',),
        pinned_keys=frozenset(pinned_keys),
    )


def warnings_from(call) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            call()
        except StructuredDataError:
            pass
    return [str(w.message) for w in caught]


def test_scoring_rows_the_context_holds_is_reported():
    data = frame()
    request = query(data).materialize(indices=list(range(200, 260)))
    said = warnings_from(
        lambda: handle(_named_by([k]) for k in range(600)).predict(request)
    )
    complaint = [s for s in said if 'pinned context' in s]
    assert complaint, said
    assert '60 of the 60 rows' in complaint[0]


def test_the_report_names_both_things_that_work():
    data = frame()
    request = query(data).materialize(indices=list(range(200, 260)))
    said = warnings_from(
        lambda: handle(_named_by([k]) for k in range(600)).predict(request)
    )
    complaint = next(s for s in said if 'pinned context' in s)
    assert 'before calling fit()' in complaint
    assert 'use predict()' in complaint


def test_rows_the_context_excluded_are_not_reported():
    """The rows `fit()` named were excluded from its own context, which is the
    half of this that was always right."""
    data = frame()
    request = query(data).materialize(indices=list(range(20)))
    said = warnings_from(lambda: handle(range(20, 600)).predict(request))
    assert not any('pinned context' in s for s in said)


def test_a_partial_overlap_counts_only_what_overlaps():
    data = frame()
    request = query(data).materialize(indices=list(range(10, 30)))
    said = warnings_from(
        lambda: handle(_named_by([k]) for k in range(20)).predict(request)
    )
    complaint = next(s for s in said if 'pinned context' in s)
    assert '10 of the 20 rows' in complaint


def test_a_session_that_pinned_nothing_says_nothing():
    """No keys means they could not be read, not that none overlap."""
    data = frame()
    request = query(data).materialize(indices=list(range(200, 260)))
    said = warnings_from(lambda: handle([]).predict(request))
    assert not any('pinned context' in s for s in said)


def test_the_rows_being_scored_are_read_from_the_request():
    data = frame()
    request = query(data).materialize(indices=[1, 2, 3])
    assert scored_keys(request) == {_named_by([k]) for k in (1, 2, 3)}


# --- a key set to None is taken back from the engine -----------------------


def test_a_child_keyed_id_can_be_linked():
    """The engine infers a key from a column *name* without checking
    uniqueness, so `ID` holding [1, 1, 2, 2, 3] became that table's primary
    key. The client let the caller set it to None and never passed that on."""
    pytest.importorskip('sdm_relational_engine')

    severity = pd.DataFrame({'ID': [1, 1, 2, 2, 3], 'sev': list('abcde')})
    location = pd.DataFrame({'ID': [1, 2, 3], 'loc': list('xyz')})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'location': location, 'severity_type': severity},
            primary_keys={'location': 'ID'},
            relationships=[
                Relationship('severity_type', 'ID', 'location', 'ID')
            ],
        )
    schema['severity_type'].primary_key = None

    request = PredictiveQuery(
        'PREDICT location.loc FOR EACH location.ID', schema
    ).materialize(indices=[1, 2], model=KumoRelational())

    assert request.predict_rows == 2
    assert 'severity_type' in request.payloads[0]['schema']['related_tables']


def test_a_declared_key_is_still_passed_to_the_engine():
    """Read off the engine graph, because the wire renames a related table's
    key to the transport's own columns and would hide either answer."""
    pytest.importorskip('sdm_relational_engine')

    from sdm_client.materializers.relational import RelationalMaterializer

    severity = pd.DataFrame({'sid': range(5), 'ID': [1, 1, 2, 2, 3]})
    location = pd.DataFrame({'ID': [1, 2, 3], 'loc': list('xyz')})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'location': location, 'severity_type': severity},
            primary_keys={'location': 'ID', 'severity_type': 'sid'},
            relationships=[
                Relationship('severity_type', 'ID', 'location', 'ID')
            ],
        )
    graph = RelationalMaterializer(schema)._build_graph()

    assert graph['severity_type'].primary_key.name == 'sid'
    assert graph['location'].primary_key.name == 'ID'


def test_a_key_set_to_none_reaches_the_engine_as_none():
    """The engine had already guessed `ID`, and the client only ever assigned
    a key it had, so the guess survived."""
    pytest.importorskip('sdm_relational_engine')

    from sdm_client.materializers.relational import RelationalMaterializer

    severity = pd.DataFrame({'ID': [1, 1, 2, 2, 3], 'sev': list('abcde')})
    location = pd.DataFrame({'ID': [1, 2, 3], 'loc': list('xyz')})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'location': location, 'severity_type': severity},
            primary_keys={'location': 'ID'},
            relationships=[
                Relationship('severity_type', 'ID', 'location', 'ID')
            ],
        )
    schema['severity_type'].primary_key = None

    graph = RelationalMaterializer(schema)._build_graph()

    assert graph['severity_type'].primary_key is None


# --- the engine casts to a dtype, not to a class ---------------------------


@pytest.mark.parametrize(
    'values',
    [[1, 2, 3], ['a', 'b', 'c'], [D(1), D(2)], [1.5, 2.5]],
    ids=['int', 'str', 'decimal', 'float'],
)
def test_keys_are_cast_to_the_index_dtype(values):
    """`type(index[0])` is `np.int64` for a numpy index, which pandas accepts,
    and `decimal.Decimal` for an object one, which it does not."""
    index = pd.Index(values)
    cast = pd.Series(values).astype(index.dtype)
    assert list(cast) == values


def test_the_old_cast_target_is_the_one_that_fails():
    """The control: what it used to do, on the data it used to fail on."""
    index = pd.Index([D(1), D(2)])
    with pytest.raises(TypeError):
        pd.Series([D(1)]).astype(type(index[0]))


def test_the_engine_looks_up_a_key_on_an_object_index():
    """Through the engine, not through pandas. `type(index[0])` on an object
    index of `Decimal` is a class rather than a dtype, and the guard below the
    cast caught `ValueError` while the cast raised `TypeError`, so the message
    written for exactly this never ran."""
    pytest.importorskip('sdm_relational_engine')

    from sdm_relational_engine.rfm import Graph
    from sdm_relational_engine.rfm.backend.local.graph_store import (
        LocalGraphStore,
    )

    people = pd.DataFrame({'pid': [D(1), D(2), D(3)], 'v': [1.0, 2.0, 3.0]})
    graph = Graph.from_data({'people': people}, edges=[], verbose=False)
    graph['people'].primary_key = 'pid'
    store = LocalGraphStore(graph, verbose=False)

    found = store.get_node_id('people', pd.Series([D(2), D(1)]))

    assert len(found) == 2


def test_a_key_the_engine_does_not_hold_is_still_reported_clearly():
    pytest.importorskip('sdm_relational_engine')

    from sdm_relational_engine.rfm import Graph
    from sdm_relational_engine.rfm.backend.local.graph_store import (
        LocalGraphStore,
    )

    people = pd.DataFrame({'pid': [1, 2, 3], 'v': [1.0, 2.0, 3.0]})
    graph = Graph.from_data({'people': people}, edges=[], verbose=False)
    graph['people'].primary_key = 'pid'
    store = LocalGraphStore(graph, verbose=False)

    with pytest.raises(Exception) as raised:
        store.get_node_id('people', pd.Series(['not a key']))
    assert 'pid' in str(raised.value) or 'not a key' in str(raised.value)


def test_a_session_remembers_the_keys_it_pinned():
    """Through `fit()`, because the wiring from the pinned payload to the
    handle is the part a hand-built handle cannot exercise."""
    from sdm_client import SDMClient

    data = frame(300)
    request = query(data).materialize(indices=[1, 2], num_context_samples=5000)

    client = SDMClient('http://127.0.0.1:1', 'none', timeout=1.0, max_retries=0)

    class Fake:
        def create_session(self, payload):
            self.payload = payload
            return {'session_id': 's1'}

        def delete_session(self, session_id):
            pass

    client._transport = Fake()
    fitted = client.fit(request, model='kumo-tabular')

    assert len(fitted._pinned_keys) == 298
    assert _named_by([1]) not in fitted._pinned_keys
    assert _named_by([5]) in fitted._pinned_keys


def test_a_key_that_cannot_be_cast_gets_the_engines_own_message():
    """The guard below the cast catches `ValueError`, and a cast from a string
    raises that -- but from a list or a mapping it raises `TypeError`, which
    escaped as a raw pandas complaint past the message written for it."""
    pytest.importorskip('sdm_relational_engine')

    from sdm_relational_engine.rfm import Graph
    from sdm_relational_engine.rfm.backend.local.graph_store import (
        LocalGraphStore,
    )

    people = pd.DataFrame({'pid': [1, 2, 3], 'v': [1.0, 2.0, 3.0]})
    graph = Graph.from_data({'people': people}, edges=[], verbose=False)
    graph['people'].primary_key = 'pid'
    store = LocalGraphStore(graph, verbose=False)

    for unusable in ([[1]], [{'a': 1}]):
        with pytest.raises(ValueError) as raised:
            store.get_node_id('people', pd.Series(unusable, dtype=object))
        assert 'Could not cast primary keys' in str(raised.value)


# --- a key written the same way on both sides ------------------------------


def test_a_wide_key_matches_across_the_wire():
    """Past 2**53 a key travels as a base-10 string by contract, so the pinned
    key is text and the scored key is an int. Compared as they arrive they
    never meet, and the warning goes quiet on exactly the warehouse keys it
    matters most for."""
    wide = 9007199254740993
    assert _named_by([str(wide)]) == _named_by([wide])


def test_a_composite_key_is_a_tuple_rather_than_nothing():
    """Reading only single-column keys meant a session over a composite-keyed
    entity could score pinned rows in silence."""
    payload = envelope(
        'context',
        ['a', 'b', 'v'],
        [[1, 'x', 0.5], [2, 'y', 0.6]],
        primary_key=['a', 'b'],
    )
    assert keys_in(payload, 'context') == {
        _named_by([1, 'x']),
        _named_by([2, 'y']),
    }


def test_a_composite_key_distinguishes_its_parts():
    assert _named_by([1, 2]) != _named_by([2, 1])
    assert _named_by([1, 2]) != _named_by(['12'])


def test_a_composite_overlap_is_reported():
    """Built by hand, because a predictive query cannot name a composite key
    as its entity -- `FOR EACH` takes one column. The frame path can still
    carry one, and reading only single-column keys meant such a request scored
    pinned rows in silence."""
    from sdm_client.request import Request

    rng = np.random.default_rng(3)
    data = pd.DataFrame(
        {
            'a': np.repeat(np.arange(50), 4),
            'b': np.tile(np.arange(4), 50),
            'v': rng.normal(0, 1, 200),
            'label': rng.integers(0, 2, 200).astype(str),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': data}, primary_keys={'t': ['a', 'b']})
    request = Request(
        context=data,
        predict=data.head(6).drop(columns=['label']),
        target='label',
        task='classification',
        schema=schema,
        entity_table='t',
    )

    scored = scored_keys(request)
    assert scored == {
        _named_by([a, b])
        for a, b in zip(data['a'].head(6), data['b'].head(6), strict=True)
    }

    said = warnings_from(lambda: handle(scored).predict(request))
    assert any('6 of the 6 rows' in s for s in said), said


def test_a_composite_key_with_no_overlap_says_nothing():
    from sdm_client.request import Request

    rng = np.random.default_rng(3)
    data = pd.DataFrame(
        {
            'a': np.repeat(np.arange(50), 4),
            'b': np.tile(np.arange(4), 50),
            'v': rng.normal(0, 1, 200),
            'label': rng.integers(0, 2, 200).astype(str),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': data}, primary_keys={'t': ['a', 'b']})
    request = Request(
        context=data,
        predict=data.head(6).drop(columns=['label']),
        target='label',
        task='classification',
        schema=schema,
        entity_table='t',
    )
    said = warnings_from(
        lambda: handle({_named_by([999, 999])}).predict(request)
    )
    assert not any('pinned context' in s for s in said)


@pytest.mark.parametrize(
    'held, asked',
    [
        ([1, 2, 3], ['2']),
        ([1, 2, 3], [2]),
        ([D(1), D(2)], [D(2)]),
        (['a', 'b'], ['b']),
    ],
    ids=['object ints asked as text', 'object ints', 'decimals', 'strings'],
)
def test_an_object_index_still_coerces_what_it_used_to(held, asked):
    """The element type is tried first because that is what coerced a key
    arriving as text against an index of ints; the index's own dtype is the
    fallback for the case where the element type is a class pandas refuses."""
    pytest.importorskip('sdm_relational_engine')

    from sdm_relational_engine.rfm import Graph
    from sdm_relational_engine.rfm.backend.local.graph_store import (
        LocalGraphStore,
    )

    people = pd.DataFrame(
        {
            'pid': pd.Series(held, dtype=object),
            'v': [float(i) for i in range(len(held))],
        }
    )
    graph = Graph.from_data({'people': people}, edges=[], verbose=False)
    graph['people'].primary_key = 'pid'
    store = LocalGraphStore(graph, verbose=False)

    assert len(store.get_node_id('people', pd.Series(asked))) == 1
