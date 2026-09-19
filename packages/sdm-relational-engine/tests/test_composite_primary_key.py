# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from predictive_query.schema.typing import Dtype
from sdm_relational_engine.rfm.base.composite_key import (
    SEPARATOR,
    decode_value,
    encode_frame,
    encode_values,
    refuse_unfoldable_dtypes,
)


@pytest.fixture()
def superstore():
    r"""A customer dimension partitioned by region, as a warehouse holds it.

    Neither column identifies a row on its own; the pair does.
    """
    people = pd.DataFrame(
        {
            'Customer ID': ['AB-10015', 'AB-10015', 'ZZ-00001', 'ZZ-00001'],
            'Region': ['Central', 'East', 'Central', 'West'],
            'Segment': ['Consumer', 'Consumer', 'Corporate', 'Corporate'],
        }
    )
    rng = np.random.default_rng(0)
    size = 40
    orders = pd.DataFrame(
        {
            'Order ID': np.arange(size),
            'Customer ID': rng.choice(['AB-10015', 'ZZ-00001'], size),
            'Region': rng.choice(['Central', 'East', 'West'], size),
            'Order Date': (
                pd.to_datetime('2024-01-01')
                + pd.to_timedelta(rng.integers(0, 300, size), unit='D')
            ),
            'Sales': rng.uniform(5, 400, size).round(2),
        }
    )
    return people, orders


def build_graph(people, orders):
    import sdm_relational_engine.rfm as rfm

    graph = rfm.Graph.from_data(
        {'PEOPLE': people, 'ORDERS': orders},
        infer_metadata=False,
        verbose=False,
    )
    graph['PEOPLE'].primary_key = ('Customer ID', 'Region')
    graph['ORDERS'].primary_key = 'Order ID'
    graph['ORDERS'].time_column = 'Order Date'
    return graph


def test_folding_is_injective_across_a_separator_in_the_data() -> None:
    assert encode_values(['a', f'b{SEPARATOR}c']) != encode_values(
        [f'a{SEPARATOR}b', 'c']
    )


@pytest.mark.parametrize(
    'parts',
    [
        ('AB-10015', 'Central'),
        ('a', 'b', 'c'),
        (f'holds{SEPARATOR}separator', 'y'),
        ('holds\\backslash', 'z'),
        ('', 'empty-first-part'),
    ],
)
def test_a_folded_key_round_trips(parts: tuple) -> None:
    assert decode_value(encode_values(list(parts)), len(parts)) == parts


def test_frame_folding_matches_row_folding(superstore) -> None:
    people, _ = superstore
    columns = ['Customer ID', 'Region']
    per_row = [
        encode_values(list(row))
        for row in people[columns].itertuples(index=False)
    ]
    assert list(encode_frame(people, columns)) == per_row


def test_folded_identity_is_unique_where_neither_column_is(superstore) -> None:
    people, _ = superstore
    assert not people['Customer ID'].is_unique
    assert not people['Region'].is_unique
    assert encode_frame(people, ['Customer ID', 'Region']).is_unique


def test_a_null_part_is_refused() -> None:
    df = pd.DataFrame({'a': ['x', None], 'b': ['1', '2']})
    with pytest.raises(ValueError, match='null'):
        encode_frame(df, ['a', 'b'])


def test_a_missing_part_is_refused(superstore) -> None:
    people, _ = superstore
    with pytest.raises(ValueError, match='not present'):
        encode_frame(people, ['Customer ID', 'Nope'])


def test_an_approximate_part_is_refused() -> None:
    with pytest.raises(ValueError, match='floating-point'):
        refuse_unfoldable_dtypes([('score', Dtype.float64)])


def test_a_table_keys_on_the_declared_columns(superstore) -> None:
    import sdm_relational_engine.rfm as rfm

    people, _ = superstore
    table = rfm.LocalTable(people, name='PEOPLE')
    table.primary_key = ('Customer ID', 'Region')

    assert table.has_composite_primary_key
    assert table.primary_key_columns == ('Customer ID', 'Region')


def test_declaring_a_composite_key_leaves_the_callers_frame_alone(
    superstore,
) -> None:
    import sdm_relational_engine.rfm as rfm

    people, _ = superstore
    before = list(people.columns)
    rfm.LocalTable(people, name='PEOPLE').primary_key = (
        'Customer ID',
        'Region',
    )
    assert list(people.columns) == before


def test_a_single_column_key_is_untouched() -> None:
    import sdm_relational_engine.rfm as rfm

    table = rfm.LocalTable(pd.DataFrame({'id': [1, 2], 'x': [1, 2]}), name='T')
    table.primary_key = 'id'

    assert not table.has_composite_primary_key
    assert table.primary_key.name == 'id'
    assert table.primary_key_columns == ('id',)


def test_a_one_element_sequence_is_an_ordinary_key() -> None:
    import sdm_relational_engine.rfm as rfm

    table = rfm.LocalTable(pd.DataFrame({'id': [1, 2], 'x': [1, 2]}), name='T')
    table.primary_key = ('id',)

    assert not table.has_composite_primary_key
    assert table.primary_key.name == 'id'


def test_a_key_column_that_does_not_exist_is_refused(superstore) -> None:
    import sdm_relational_engine.rfm as rfm

    people, _ = superstore
    table = rfm.LocalTable(people, name='PEOPLE')
    with pytest.raises(ValueError, match='not present'):
        table.primary_key = ('Customer ID', 'Nope')


def duckdb_graph(people, orders=None):
    duckdb = pytest.importorskip('duckdb')
    import os
    import tempfile

    import sdm_relational_engine.rfm as rfm

    path = os.path.join(tempfile.mkdtemp(), 'g.duckdb')
    connection = duckdb.connect(path)
    connection.execute('CREATE TABLE PEOPLE AS SELECT * FROM people')
    if orders is not None:
        connection.execute('CREATE TABLE ORDERS AS SELECT * FROM orders')
    connection.close()
    return rfm.Graph.from_duckdb(path, verbose=False, infer_metadata=False)


def test_a_warehouse_folds_an_identity_exactly_as_pandas_does(
    superstore,
) -> None:
    people, _ = superstore
    graph = duckdb_graph(people)
    graph['PEOPLE'].primary_key = ('Customer ID', 'Region')

    folded = graph['PEOPLE']._sample_values(graph['PEOPLE'].primary_key.name)
    assert sorted(folded) == sorted(
        encode_frame(people, ['Customer ID', 'Region'])
    )


def test_a_warehouse_that_folds_differently_is_refused(superstore) -> None:
    r"""The failure this check exists for.

    Seeds are folded in pandas and rows in the warehouse. If the two disagree
    the join matches nothing and no error is raised anywhere, so the
    disagreement is caught where the key is declared.
    """
    from sdm_relational_engine.rfm.base import composite_key as module

    people, _ = superstore
    graph = duckdb_graph(people)

    original = module.sql_expression
    module.sql_expression = lambda columns, **kwargs: original(
        columns, **kwargs
    ).replace('CHR(31)', "'|'")
    try:
        with pytest.raises(ValueError, match='differently in the warehouse'):
            graph['PEOPLE'].primary_key = ('Customer ID', 'Region')
    finally:
        module.sql_expression = original


def test_a_warehouse_link_folds_on_both_sides(superstore) -> None:
    people, orders = superstore
    orders = orders.assign(**{'Order Date': orders['Order Date'].astype(str)})
    graph = duckdb_graph(people, orders)
    graph['PEOPLE'].primary_key = ('Customer ID', 'Region')
    graph['ORDERS'].primary_key = 'Order ID'
    graph['ORDERS'].time_column = 'Order Date'
    graph.link(
        src_table='ORDERS', fkey=('Customer ID', 'Region'), dst_table='PEOPLE'
    )

    referencing = graph['ORDERS']._sample_values(graph.edges[0].fkey)
    identity = graph['PEOPLE']._sample_values(graph['PEOPLE'].primary_key.name)
    assert len(set(referencing) & set(identity)) > 0


def test_a_fact_table_links_on_the_same_columns(superstore) -> None:
    people, orders = superstore
    graph = build_graph(people, orders)
    graph.link(
        src_table='ORDERS', fkey=('Customer ID', 'Region'), dst_table='PEOPLE'
    )

    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert (edge.src_table, edge.dst_table) == ('ORDERS', 'PEOPLE')
    assert edge.fkey == graph['PEOPLE'].primary_key.name


def test_a_link_naming_the_columns_in_the_wrong_order_is_refused(
    superstore,
) -> None:
    r"""The failure this guard exists for.

    Folding ``(Region, Customer ID)`` produces values that meet no row of
    PEOPLE. Left alone the graph builds, the edge carries nothing, and the
    model quietly learns from a relationship that is not there.
    """
    people, orders = superstore
    graph = build_graph(people, orders)
    with pytest.raises(ValueError, match='matches no rows'):
        graph.link(
            src_table='ORDERS',
            fkey=('Region', 'Customer ID'),
            dst_table='PEOPLE',
        )


def test_a_link_of_the_wrong_arity_is_refused(superstore) -> None:
    people, orders = superstore
    graph = build_graph(people, orders)
    with pytest.raises(ValueError, match='has to name 2 column'):
        graph.link(
            src_table='ORDERS',
            fkey=('Customer ID', 'Region', 'Order ID'),
            dst_table='PEOPLE',
        )


@pytest.mark.parametrize('reference', ['Customer ID', ('Customer ID',)])
def test_one_column_cannot_name_a_row_of_a_composite_identity(
    superstore,
    reference,
) -> None:
    r"""Otherwise the edge joins on values no row of PEOPLE carries.

    The graph builds, ``validate()`` passes, and the relationship is empty.
    """
    people, orders = superstore
    graph = build_graph(people, orders)
    with pytest.raises(ValueError, match='every part of the identity'):
        graph.link(src_table='ORDERS', fkey=reference, dst_table='PEOPLE')


def test_a_link_missing_a_key_column_is_refused(superstore) -> None:
    people, orders = superstore
    graph = build_graph(people, orders)
    with pytest.raises(ValueError, match='not present'):
        graph.link(
            src_table='ORDERS', fkey=('Customer ID', 'Nope'), dst_table='PEOPLE'
        )


def test_tuple_seeds_are_folded_into_identities() -> None:
    from sdm_relational_engine.rfm.rfm import _encode_composite_indices

    encoded = _encode_composite_indices(
        [('AB-10015', 'Central')], ('Customer ID', 'Region')
    )
    assert encoded == [encode_values(['AB-10015', 'Central'])]


def test_a_seed_of_the_wrong_arity_is_refused() -> None:
    from sdm_relational_engine.rfm.rfm import _encode_composite_indices

    with pytest.raises(ValueError, match='2 value'):
        _encode_composite_indices([('AB-10015',)], ('Customer ID', 'Region'))


def test_a_prediction_is_reported_against_the_declared_columns() -> None:
    from sdm_relational_engine.rfm.rfm import _decode_composite_entities

    frame = pd.DataFrame(
        {
            'ENTITY': [encode_values(['AB-10015', 'Central'])],
            'PREDICTION': [1.5],
        }
    )
    decoded = _decode_composite_entities(frame, ('Customer ID', 'Region'))

    assert 'ENTITY' not in decoded.columns
    assert list(decoded.columns) == ['Customer ID', 'Region', 'PREDICTION']
    assert decoded.iloc[0]['Customer ID'] == 'AB-10015'
    assert decoded.iloc[0]['Region'] == 'Central'


@pytest.mark.parametrize('replacement', ['id', None])
def test_replacing_a_composite_key_forgets_the_derived_column(
    replacement: str | None,
) -> None:
    r"""A derived identity left behind would reach the model as a feature."""
    import sdm_relational_engine.rfm as rfm

    df = pd.DataFrame({'a': ['x', 'y'], 'b': ['1', '2'], 'id': [1, 2]})
    table = rfm.LocalTable(df, name='T')
    table.primary_key = ('a', 'b')
    derived = table.primary_key.name

    table.primary_key = replacement

    assert derived not in table._data.columns
    assert derived not in [column.name for column in table.columns]
    assert not table.has_composite_primary_key


def test_replacing_one_composite_key_with_another_keeps_only_the_new_one() -> (
    None
):
    import sdm_relational_engine.rfm as rfm

    df = pd.DataFrame({'a': ['x', 'y'], 'b': ['1', '2'], 'c': ['p', 'q']})
    table = rfm.LocalTable(df, name='T')
    table.primary_key = ('a', 'b')
    first = table.primary_key.name

    table.primary_key = ('a', 'c')

    assert first not in table._data.columns
    assert table.primary_key_columns == ('a', 'c')
    assert table.primary_key.name in table._data.columns


def test_a_three_column_identity_is_supported() -> None:
    import sdm_relational_engine.rfm as rfm

    df = pd.DataFrame(
        {
            'a': ['x', 'x'],
            'b': ['1', '1'],
            'c': ['p', 'q'],
            'v': [1, 2],
        }
    )
    table = rfm.LocalTable(df, name='T')
    table.primary_key = ('a', 'b', 'c')

    assert table.primary_key_columns == ('a', 'b', 'c')
    assert table._data[table.primary_key.name].is_unique


def test_an_identity_may_mix_numbers_and_text() -> None:
    import sdm_relational_engine.rfm as rfm

    df = pd.DataFrame(
        {
            'cid': [1, 1, 2],
            'Region': ['C', 'E', 'C'],
            'v': [1, 2, 3],
        }
    )
    table = rfm.LocalTable(df, name='T')
    table.primary_key = ('cid', 'Region')

    assert table._data[table.primary_key.name].is_unique
    assert decode_value(table._data[table.primary_key.name].iloc[0], 2) == (
        '1',
        'C',
    )


def test_a_rejected_reassignment_keeps_the_existing_key(superstore) -> None:
    r"""A refusal must not leave the table keyless with an orphaned column.

    That column carries a row's identity, so it would otherwise reach the
    model as an ordinary high-cardinality feature.
    """
    import sdm_relational_engine.rfm as rfm

    people, _ = superstore
    people = people.assign(
        When=pd.to_datetime(
            ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
        )
    )
    table = rfm.LocalTable(people, name='PEOPLE')
    table.time_column = 'When'
    table.primary_key = ('Customer ID', 'Region')
    derived = table.primary_key.name

    with pytest.raises(ValueError, match='time column'):
        table.primary_key = 'When'

    assert table.primary_key.name == derived
    assert table.has_composite_primary_key
    definition = table._to_api_table_definition()
    assert definition.pkey == derived


def test_a_float_column_cannot_be_part_of_an_identity() -> None:
    import sdm_relational_engine.rfm as rfm

    table = rfm.LocalTable(
        pd.DataFrame({'f': [1.0, 2.0], 'b': ['1', '2']}), name='T'
    )
    with pytest.raises(ValueError, match='floating-point'):
        table.primary_key = ('f', 'b')


def test_a_float_foreign_key_is_refused(superstore) -> None:
    people, orders = superstore
    graph = build_graph(people, orders)
    with pytest.raises(ValueError, match='floating-point'):
        graph.link(
            src_table='ORDERS', fkey=('Sales', 'Region'), dst_table='PEOPLE'
        )


def test_declaring_the_same_identity_twice_reuses_one_column() -> None:
    import sdm_relational_engine.rfm as rfm

    table = rfm.LocalTable(
        pd.DataFrame({'a': ['x', 'y'], 'b': ['1', '2']}), name='T'
    )
    names = []
    for _ in range(4):
        table.primary_key = ('a', 'b')
        names.append(table.primary_key.name)

    assert len(set(names)) == 1
    derived = [
        name for name in table._data.columns if name.startswith('__kumo_key_')
    ]
    assert len(derived) == 1


def test_linking_the_same_composite_reference_twice_is_refused(
    superstore,
) -> None:
    people, orders = superstore
    graph = build_graph(people, orders)
    graph.link(
        src_table='ORDERS', fkey=('Customer ID', 'Region'), dst_table='PEOPLE'
    )
    with pytest.raises(ValueError, match='already exists'):
        graph.link(
            src_table='ORDERS',
            fkey=('Customer ID', 'Region'),
            dst_table='PEOPLE',
        )
    assert len(graph.edges) == 1


def test_a_rejected_link_leaves_no_column_behind(superstore) -> None:
    people, orders = superstore
    graph = build_graph(people, orders)
    with pytest.raises(ValueError, match='matches no rows'):
        graph.link(
            src_table='ORDERS',
            fkey=('Region', 'Customer ID'),
            dst_table='PEOPLE',
        )

    stray = [
        column.name
        for column in graph['ORDERS'].columns
        if column.name.startswith('__kumo_key_')
    ]
    assert stray == []


def test_a_linked_composite_reference_actually_meets_the_identity(
    superstore,
) -> None:
    r"""Names matching is not enough; the folded values have to intersect."""
    people, orders = superstore
    graph = build_graph(people, orders)
    graph.link(
        src_table='ORDERS', fkey=('Customer ID', 'Region'), dst_table='PEOPLE'
    )

    referencing = graph['ORDERS']._source_sample_df[graph.edges[0].fkey]
    identity = graph['PEOPLE']._source_sample_df[
        graph['PEOPLE'].primary_key.name
    ]
    assert len(set(referencing) & set(identity)) > 0


def test_a_one_column_reference_takes_the_ordinary_path(superstore) -> None:
    people, orders = superstore
    graph = build_graph(people, orders)
    graph['PEOPLE'].primary_key = 'Customer ID'
    graph.link(src_table='ORDERS', fkey=('Customer ID',), dst_table='PEOPLE')

    assert graph.edges[0].fkey == 'Customer ID'


def test_a_composite_link_can_be_removed_the_way_it_was_made(
    superstore,
) -> None:
    people, orders = superstore
    graph = build_graph(people, orders)
    graph.link(
        src_table='ORDERS', fkey=('Customer ID', 'Region'), dst_table='PEOPLE'
    )
    graph.unlink(
        src_table='ORDERS', fkey=('Customer ID', 'Region'), dst_table='PEOPLE'
    )

    assert len(graph.edges) == 0


def test_a_seed_naming_one_value_for_a_two_column_identity_is_refused() -> None:
    from sdm_relational_engine.rfm.rfm import _encode_composite_indices

    with pytest.raises(ValueError, match='on its own'):
        _encode_composite_indices(['AB-10015'], ('Customer ID', 'Region'))


def test_a_key_column_clashing_with_a_result_column_is_reported() -> None:
    from sdm_relational_engine.rfm.rfm import _decode_composite_entities

    frame = pd.DataFrame(
        {
            'ENTITY': [encode_values(['A', 'C'])],
            'PREDICTION': [1.5],
        }
    )
    with pytest.raises(ValueError, match='already carries'):
        _decode_composite_entities(frame, ('PREDICTION', 'Region'))


@pytest.mark.parametrize('value', [float('nan'), np.nan, pd.NaT, None])
def test_a_missing_key_part_is_refused(value: object) -> None:
    with pytest.raises(ValueError, match='null'):
        encode_values([value, 'x'])


def test_a_numpy_boolean_folds_like_a_python_one() -> None:
    r"""A frame yields Python bools while a seed often yields numpy ones."""
    frame = pd.DataFrame({'a': [True, False], 'b': ['x', 'y']})
    assert list(encode_frame(frame, ['a', 'b'])) == [
        encode_values([np.bool_(True), 'x']),
        encode_values([np.bool_(False), 'y']),
    ]


def test_two_identities_that_join_the_same_way_get_different_columns() -> None:
    r"""``('a_b', 'c')`` and ``('a', 'b_c')`` must not share a column.

    Joining names with a separator made them indistinguishable, so a second
    reference from one table landed on the first one's folded values and
    joined against the wrong rows without any error.
    """
    import sdm_relational_engine.rfm as rfm

    table = rfm.LocalTable(
        pd.DataFrame(
            {
                'a_b': ['x'],
                'c': ['1'],
                'a': ['p'],
                'b_c': ['q'],
            }
        ),
        name='T',
    )

    assert table._derived_key_column_name(
        ('a_b', 'c')
    ) != table._derived_key_column_name(('a', 'b_c'))


def test_the_derived_name_is_stable_and_ordered() -> None:
    import sdm_relational_engine.rfm as rfm

    table = rfm.LocalTable(pd.DataFrame({'a': ['x'], 'c': ['1']}), name='T')

    assert table._derived_key_column_name(
        ('a', 'c')
    ) == table._derived_key_column_name(('a', 'c'))
    assert table._derived_key_column_name(
        ('a', 'c')
    ) != table._derived_key_column_name(('c', 'a'))


def test_a_query_names_the_identity_by_any_of_its_columns(superstore) -> None:
    r"""A caller should not have to spell a column this client invented."""
    from sdm_relational_engine.rfm.query.parser import parse_query_locally

    people, orders = superstore
    graph = build_graph(people, orders)
    graph.link(
        src_table='ORDERS', fkey=('Customer ID', 'Region'), dst_table='PEOPLE'
    )
    definition = graph._to_api_graph_definition()
    derived = graph['PEOPLE'].primary_key.name

    for named in ('`Customer ID`', 'Region', f'`{derived}`'):
        query = (
            f'PREDICT COUNT(ORDERS.*, 0, 30, days) > 0 FOR EACH PEOPLE.{named}'
        )
        validated = parse_query_locally(query, definition)
        assert validated.entity_column == f'PEOPLE.{derived}'


def test_a_column_outside_the_identity_still_cannot_be_the_entity(
    superstore,
) -> None:
    from sdm_relational_engine.rfm.query.parser import parse_query_locally

    people, orders = superstore
    graph = build_graph(people, orders)
    graph.link(
        src_table='ORDERS', fkey=('Customer ID', 'Region'), dst_table='PEOPLE'
    )

    with pytest.raises(ValueError, match='primary key'):
        parse_query_locally(
            'PREDICT COUNT(ORDERS.*, 0, 30, days) > 0 FOR EACH PEOPLE.Segment',
            graph._to_api_graph_definition(),
        )


def test_a_caller_holding_only_the_graph_definition_gets_the_same_rule(
    superstore,
) -> None:
    r"""The service and a local pre-check have to agree.

    A caller validating before it sends holds the graph definition and nothing
    else. If the identity rule needed anything more, a query the service
    accepts would be refused locally and never sent.
    """
    from sdm_relational_engine.rfm.query.parser import parse_query_locally

    people, orders = superstore
    graph = build_graph(people, orders)
    graph.link(
        src_table='ORDERS', fkey=('Customer ID', 'Region'), dst_table='PEOPLE'
    )
    definition = graph._to_api_graph_definition()

    validated = parse_query_locally(
        'PREDICT COUNT(ORDERS.*, 0, 30, days) > 0 '
        'FOR EACH PEOPLE.`Customer ID`',
        definition,
    )

    assert (
        validated.entity_column == f'PEOPLE.{graph["PEOPLE"].primary_key.name}'
    )


def test_an_identity_is_recoverable_from_the_derived_column_alone() -> None:
    from sdm_relational_engine.rfm.base.composite_key import (
        DERIVED_PREFIX,
        decode_identity,
        encode_identity,
    )

    for columns in [
        ('Customer ID', 'Region'),
        ('a_b', 'c'),
        ('a', 'b_c'),
        ('a', 'b', 'c'),
        ('holds:colon', 'y'),
        ('Cust.ID', 'R'),
        ('back`tick', 'R'),
        ('ünïcode', 'x'),
    ]:
        derived = DERIVED_PREFIX + encode_identity(columns)
        assert decode_identity(derived) == columns
    assert decode_identity('Customer ID') is None
    assert decode_identity(DERIVED_PREFIX) is None
    assert decode_identity(DERIVED_PREFIX + 'nothex') is None


@pytest.mark.parametrize(
    'columns',
    [
        ('Customer ID', 'Region'),
        ('Cust.ID', 'Region'),
        ('back`tick', 'Region'),
        ('has space', 'Region'),
    ],
)
def test_a_derived_name_is_written_without_quoting(columns) -> None:
    r"""The name reaches places a column name generally cannot.

    It is written into a predictive query, split back out of a fully
    qualified name, and created as a warehouse column. Carrying the member
    names as themselves let a dot break the split, a backtick break the
    quoting, and a space force every reference to be quoted.
    """
    import re

    from sdm_relational_engine.rfm.base.composite_key import (
        DERIVED_PREFIX,
        encode_identity,
    )

    derived = DERIVED_PREFIX + encode_identity(columns)

    assert re.fullmatch(r'[_A-Za-z0-9]+', derived)


def test_an_identity_too_long_to_name_is_refused() -> None:
    from sdm_relational_engine.rfm.base.composite_key import encode_identity

    with pytest.raises(ValueError, match='longer than'):
        encode_identity(
            tuple(f'a_very_long_warehouse_column_name_{i}' for i in range(8))
        )


def test_the_derived_column_can_be_named_without_backticks(superstore) -> None:
    r"""How anyone on 2.27.0 would have written it, since it was bare then."""
    from sdm_relational_engine.rfm.query.parser import parse_query_locally

    people, orders = superstore
    graph = build_graph(people, orders)
    graph.link(
        src_table='ORDERS', fkey=('Customer ID', 'Region'), dst_table='PEOPLE'
    )
    definition = graph._to_api_graph_definition()
    derived = graph['PEOPLE'].primary_key.name

    for named in (derived, f'`{derived}`'):
        validated = parse_query_locally(
            f'PREDICT COUNT(ORDERS.*, 0, 30, days) > 0 FOR EACH PEOPLE.{named}',
            definition,
        )
        assert validated.entity_column == f'PEOPLE.{derived}'


def test_a_dotted_key_column_does_not_break_the_derived_name() -> None:
    r"""A dot in a member name used to make the entity unsplittable."""
    import numpy as np
    import sdm_relational_engine.rfm as rfm
    from sdm_relational_engine.rfm.query.parser import parse_query_locally

    rng = np.random.default_rng(0)
    size = 40
    people = pd.DataFrame(
        {
            'Cust.ID': ['A', 'A', 'B'],
            'Region': ['C', 'E', 'C'],
            'Segment': ['x', 'y', 'z'],
        }
    )
    orders = pd.DataFrame(
        {
            'Order ID': np.arange(size),
            'Cust.ID': rng.choice(['A', 'B'], size),
            'Region': rng.choice(['C', 'E'], size),
            'Order Date': (
                pd.to_datetime('2024-01-01')
                + pd.to_timedelta(rng.integers(0, 300, size), unit='D')
            ),
            'Sales': rng.uniform(5, 400, size).round(2),
        }
    )
    graph = rfm.Graph.from_data(
        {'PEOPLE': people, 'ORDERS': orders},
        infer_metadata=False,
        verbose=False,
    )
    graph['PEOPLE'].primary_key = ('Cust.ID', 'Region')
    graph['ORDERS'].primary_key = 'Order ID'
    graph['ORDERS'].time_column = 'Order Date'
    graph.link(
        src_table='ORDERS', fkey=('Cust.ID', 'Region'), dst_table='PEOPLE'
    )
    derived = graph['PEOPLE'].primary_key.name

    assert '.' not in derived
    validated = parse_query_locally(
        f'PREDICT COUNT(ORDERS.*, 0, 30, days) > 0 FOR EACH PEOPLE.{derived}',
        graph._to_api_graph_definition(),
    )
    assert validated.entity_column == f'PEOPLE.{derived}'
