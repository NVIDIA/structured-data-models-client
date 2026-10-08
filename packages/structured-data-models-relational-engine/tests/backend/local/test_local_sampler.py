# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
import sdm_relational_engine.relationallib as relationallib
from predictive_query.ast import (
    Aggregation,
    Column,
    Condition,
    Constant,
    DateOffsetRange,
    Filter,
    LogicalOperation,
)
from predictive_query.query import ValidatedPredictiveQuery
from predictive_query.schema.typing import BoolOp, Dtype
from sdm_relational_engine.rfm import Graph
from sdm_relational_engine.rfm.backend.local import LocalSampler
from sdm_relational_engine.rfm.base.utils import Timestamp, random_state


def test_relationallib_sampler_validation() -> None:
    node_types = ['a', 'b']
    edge_types = [('a', 'to', 'b'), ('b', 'rev_to', 'a')]
    colptr = {
        'a__to__b': np.array([0, 1, 2, 0, 1, 2]),
        'b__rev_to__a': np.array([0, 1, 2, 0, 1, 2]),
    }
    row = {
        'a__to__b': np.array([0, 1, 1, 0]),
        'b__rev_to__a': np.array([0, 1, 1, 0]),
    }
    node_time = {
        'a': np.array([1, 2]),
        'b': np.array([1, 2]),
    }
    num_neighbors = {
        'a__to__b': np.array([3, 3]),
        'b__rev_to__a': np.array([3, 3]),
    }
    seed_node = np.array([0, 1])
    seed_time = np.array([0, 1])
    with pytest.raises(RuntimeError, match='is not 1D'):
        sampler = relationallib.NeighborSampler(
            node_types,
            edge_types,
            {'a__to__b': 'invalid', 'b__to__a': None},
            row,
            node_time,
            123,
        )

    with pytest.raises(RuntimeError, match='invalid'):
        sampler = relationallib.NeighborSampler(
            node_types, edge_types, colptr, row, node_time, 123
        )
        sampler.sample({'invalid': 1}, {}, 'a', seed_node, seed_time)

    with pytest.raises(RuntimeError, match='Invalid entity table'):
        sampler = relationallib.NeighborSampler(
            node_types, edge_types, colptr, row, node_time, 123
        )
        sampler.sample(num_neighbors, {}, 'k', seed_node, seed_time)

    with pytest.raises(RuntimeError, match='size mismatch'):
        sampler = relationallib.NeighborSampler(
            node_types, edge_types, colptr, row, node_time, 123
        )
        sampler.sample(num_neighbors, {}, 'a', seed_node, np.array([0]))

    with pytest.raises(RuntimeError, match='same edge type'):
        sampler = relationallib.NeighborSampler(
            node_types, edge_types, colptr, row, node_time, 123
        )
        sampler.sample(
            num_neighbors,
            {'a__to__b': np.array([[0, 5], [None, 0]])},
            'a',
            seed_node,
            seed_time,
        )


def test_relationallib_sampler_temporal() -> None:
    node_types = ['A', 'B']
    edge_types = [('A', 'to', 'B'), ('B', 'to', 'A')]
    colptr = {
        'A__to__B': np.array([0, 2, 4]),
        'B__to__A': np.array([0, 1, 2, 3]),
    }
    row = {
        'A__to__B': np.array([0, 2, 0, 2]),
        'B__to__A': np.array([0, 1, 1]),
    }
    node_time = {
        'A': np.array([1, 0, 3]),
        'B': np.array([2, 1]),
    }

    sampler = relationallib.NeighborSampler(
        node_types, edge_types, colptr, row, node_time
    )

    num_neighbors = {
        'A__to__B': [1, 1],
        'B__to__A': [2, 1],
    }
    entity_table = 'A'
    seed_node = np.array([0, 1, 2])
    seed_time = np.array([3, 1, 3])
    (
        row,
        col,
        node_id,
        batch,
        num_sampled_nodes,
        num_sampled_edges,
    ) = sampler.sample(num_neighbors, {}, entity_table, seed_node, seed_time)
    assert row['A__to__B'].shape[0] == sum(num_sampled_edges['A__to__B'])
    assert row['B__to__A'].shape[0] == sum(num_sampled_edges['B__to__A'])
    assert col['A__to__B'].shape[0] == sum(num_sampled_edges['A__to__B'])
    assert col['B__to__A'].shape[0] == sum(num_sampled_edges['B__to__A'])
    assert node_id['A'].shape[0] == sum(num_sampled_nodes['A'])
    assert node_id['B'].shape[0] == sum(num_sampled_nodes['B'])
    assert row['A__to__B'].tolist() == [3, 4, 2]
    assert row['B__to__A'].tolist() == [0, 1, 2]
    assert col['A__to__B'].tolist() == [0, 1, 2]
    assert col['B__to__A'].tolist() == [0, 1, 2]
    assert node_id['A'].tolist() == [0, 1, 2, 2, 0]
    assert node_id['B'].tolist() == [0, 1, 1]
    assert batch['A'].tolist() == [0, 1, 2, 0, 1]
    assert batch['B'].tolist() == [0, 1, 2]
    assert num_sampled_nodes['A'].tolist() == [3, 0, 2]
    assert num_sampled_nodes['B'].tolist() == [0, 3, 0]
    assert num_sampled_edges['A__to__B'].tolist() == [0, 3]
    assert num_sampled_edges['B__to__A'].tolist() == [3, 0]


def test_relationallib_sampler_static() -> None:
    node_types = ['A', 'B']
    edge_types = [('A', 'to', 'B'), ('B', 'to', 'A')]
    colptr = {
        'A__to__B': np.array([0, 2, 4]),
        'B__to__A': np.array([0, 1, 2, 3]),
    }
    row = {
        'A__to__B': np.array([0, 2, 0, 2]),
        'B__to__A': np.array([0, 1, 1]),
    }

    sampler = relationallib.NeighborSampler(
        node_types, edge_types, colptr, row, []
    )

    num_neighbors = {
        'A__to__B': np.array([1, -1]),
        'B__to__A': np.array([1, 1]),
    }
    seed_node = np.array([0, 1, 2])
    entity_table = 'A'
    (
        row,
        col,
        node_id,
        batch,
        num_sampled_nodes,
        num_sampled_edges,
    ) = sampler.sample(num_neighbors, {}, entity_table, seed_node, np.array([]))
    # Outputs are non-deterministic but this specific
    # config is deterministic for this graph
    assert row['A__to__B'].tolist() == [0, 3, 4, 5, 6, 2]
    assert row['B__to__A'].tolist() == [0, 1, 2]
    assert col['A__to__B'].tolist() == [0, 0, 1, 1, 2, 2]
    assert col['B__to__A'].tolist() == [0, 1, 2]
    assert node_id['A'].tolist() == [0, 1, 2, 2, 0, 2, 0]
    assert node_id['B'].tolist() == [0, 1, 1]
    assert batch['A'].tolist() == [0, 1, 2, 0, 1, 1, 2]
    assert batch['B'].tolist() == [0, 1, 2]
    assert num_sampled_nodes['A'].tolist() == [3, 0, 4]
    assert num_sampled_nodes['B'].tolist() == [0, 3, 0]
    assert num_sampled_edges['A__to__B'].tolist() == [0, 6]
    assert num_sampled_edges['B__to__A'].tolist() == [3, 0]


def _floyd(state: np.random.RandomState, population: int, count: int) -> list:
    r"""The offsets the sampler picks from a neighborhood, from the raw
    32-bit draws of ``state`` (Floyd's algorithm, as ``_sample`` runs it).
    """
    picked: list[int] = []
    for i in range(population - count, population):
        rnd = int(state.randint(0, 2**32, dtype=np.uint32)) % (i + 1)
        picked.append(i if rnd in picked else rnd)
    return picked


@pytest.mark.parametrize(
    'seed', [0, 123, 2**31, 2**32 - 1, 2**32, 2**63, 2**64 + 3, 10**30]
)
@pytest.mark.parametrize('where', ['constructor', 'seed'])
def test_relationallib_seeds_as_random_state(seed: int, where: str) -> None:
    # Any seed the public API takes starts the generator from the state
    # random_state(seed) holds; a C int used to stop at 2**31 - 1.
    sampler = relationallib.NeighborSampler(
        ['A', 'B'],
        [('B', 'to', 'A')],
        {'B__to__A': np.array([0, 10])},
        {'B__to__A': np.arange(10)},
        {},
        seed=seed if where == 'constructor' else 1,
    )
    if where == 'seed':
        sampler.seed(seed)
    node_id = sampler.sample(
        {'B__to__A': np.array([3])}, {}, 'A', np.array([0]), np.array([])
    )[2]
    assert node_id['B'].tolist() == _floyd(random_state(seed), 10, 3)


def test_relationallib_refuses_seeds_below_a_c_int() -> None:
    with pytest.raises(TypeError):
        relationallib.NeighborSampler(['A'], [], {}, {}, {}, seed=-(2**31) - 1)
    with pytest.raises(TypeError):
        relationallib.NeighborSampler(['A'], [], {}, {}, {}, seed=1.5)


def test_relationallib_sampler_temporal_range() -> None:
    node_types = ['A', 'B']
    edge_types = [('A', 'to', 'B'), ('B', 'to', 'A')]
    colptr = {
        'A__to__B': np.array([0, 2, 4]),
        'B__to__A': np.array([0, 1, 2, 3]),
    }
    row = {
        'A__to__B': np.array([0, 2, 0, 2]),
        'B__to__A': np.array([0, 1, 1]),
    }
    node_time = {
        'A': np.array([1, 0, 3]),
        'B': np.array([2, 1]),
    }

    sampler = relationallib.NeighborSampler(
        node_types, edge_types, colptr, row, node_time
    )

    sampling_time_offsets = {
        'A__to__B': np.array([[0, 1], [-1, 2]]),
        'B__to__A': [[None, 0], [0, 0]],
    }
    entity_table = 'A'
    seed_node = np.array([0, 1, 2])
    seed_time = np.array([3, 1, 3])
    (
        row,
        col,
        node_id,
        batch,
        num_sampled_nodes,
        num_sampled_edges,
    ) = sampler.sample(
        {}, sampling_time_offsets, entity_table, seed_node, seed_time
    )
    assert row['A__to__B'].shape[0] == sum(num_sampled_edges['A__to__B'])
    assert row['B__to__A'].shape[0] == sum(num_sampled_edges['B__to__A'])
    assert col['A__to__B'].shape[0] == sum(num_sampled_edges['A__to__B'])
    assert col['B__to__A'].shape[0] == sum(num_sampled_edges['B__to__A'])
    assert node_id['A'].shape[0] == sum(num_sampled_nodes['A'])
    assert node_id['B'].shape[0] == sum(num_sampled_nodes['B'])
    assert row['A__to__B'].tolist() == [3, 4, 5, 2]
    assert row['B__to__A'].tolist() == [0, 1, 2]
    assert col['A__to__B'].tolist() == [0, 1, 1, 2]
    assert col['B__to__A'].tolist() == [0, 1, 2]
    assert node_id['A'].tolist() == [0, 1, 2, 2, 0, 2]
    assert node_id['B'].tolist() == [0, 1, 1]
    assert batch['A'].tolist() == [0, 1, 2, 0, 1, 1]
    assert batch['B'].tolist() == [0, 1, 2]
    assert num_sampled_nodes['A'].tolist() == [3, 0, 3]
    assert num_sampled_nodes['B'].tolist() == [0, 3, 0]
    assert num_sampled_edges['A__to__B'].tolist() == [0, 4]
    assert num_sampled_edges['B__to__A'].tolist() == [3, 0]


def test_sample_entity_table_with_int_entity_ids() -> None:
    df_dict = {
        'USERS': pd.DataFrame(
            {
                'USER_ID': [1, 2, 3, 4, 5],
                'AGE': [20, 30, 40, 50, 60],
            }
        ),
        'ORDERS': pd.DataFrame(
            {
                'ORDER_ID': range(10),
                'USER_ID': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                'AMOUNT': range(10),
                'TIME': pd.date_range('2025-01-01', periods=10, freq='D'),
            }
        ),
    }
    graph = Graph.from_data(df_dict, verbose=False)
    sampler = LocalSampler(graph, verbose=False)
    df = sampler._sample_entity_table(
        table_name='USERS',
        columns={'USER_ID', 'AGE'},
        num_rows=100,
        entity_ids=[2, 4],
    )
    assert set(df['USER_ID'].tolist()) == {2, 4}
    assert len(df) == 2


def test_sample_entity_table_with_string_entity_ids() -> None:
    df_dict = {
        'USERS': pd.DataFrame(
            {
                'USER_ID': ['alice', 'bob', 'charlie'],
                'AGE': [20, 30, 40],
            }
        ),
        'ORDERS': pd.DataFrame(
            {
                'ORDER_ID': range(3),
                'USER_ID': ['alice', 'bob', 'charlie'],
                'AMOUNT': [10, 20, 30],
            }
        ),
    }
    graph = Graph.from_data(df_dict, verbose=False)
    sampler = LocalSampler(graph, verbose=False)

    df = sampler._sample_entity_table(
        table_name='USERS',
        columns={'USER_ID', 'AGE'},
        num_rows=100,
        entity_ids=['bob'],
    )
    assert df['USER_ID'].tolist() == ['bob']
    assert len(df) == 1


def test_sample_target() -> None:
    df_dict: dict[str, pd.DataFrame] = {}
    df_dict['USERS'] = pd.DataFrame(
        {
            'USER_ID': [10, 10, 0],
            'AGE': [30, 30, 20],
        }
    )
    df_dict['ORDERS'] = pd.DataFrame(
        {
            'USER_ID': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'AMOUNT': range(18),
            'TIME': pd.date_range('2025-01-01', periods=18, unit='ns'),
        }
    )
    df_dict['VIEWS'] = pd.DataFrame(
        {
            'USER_ID': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'AMOUNT': range(18),
            'TIME': pd.date_range('2025-01-01', periods=18, unit='ns'),
        }
    )

    graph = Graph.from_data(df_dict, verbose=False)
    sampler = LocalSampler(graph, verbose=False)

    query = ValidatedPredictiveQuery(
        target_ast=LogicalOperation(
            left=Condition(
                target=Aggregation(
                    aggr='SUM',
                    target=Column(fqn='ORDERS.AMOUNT'),
                    aggr_time_range=DateOffsetRange(2, 7),
                ),
                op='>',
                value=Constant(value='5', dtype_maybe=Dtype.int),
            ),
            bool_op=BoolOp.AND,
            right=Condition(
                target=Column(fqn='USERS.USER_ID'),
                op='=',
                value=Constant(value='0', dtype_maybe=Dtype.int),
            ),
        ),
        entity_ast=Filter(
            target=Column(fqn='USERS.USER_ID'),
            condition=LogicalOperation(
                left=Condition(
                    target=Aggregation(
                        aggr='SUM',
                        target=Column(fqn='ORDERS.AMOUNT'),
                        aggr_time_range=DateOffsetRange(None, -2),
                    ),
                    op='>',
                    value=Constant(value='5', dtype_maybe=Dtype.int),
                ),
                bool_op=BoolOp.OR,
                right=Condition(
                    target=Aggregation(
                        aggr='SUM',
                        target=Column(fqn='VIEWS.AMOUNT'),
                        aggr_time_range=DateOffsetRange(-7, 0),
                    ),
                    op='<=',
                    value=Constant(value='5', dtype_maybe=Dtype.int),
                ),
            ),
        ),
    )

    train, test = sampler.sample_target(
        query,
        num_train_examples=2,
        train_anchor_time=Timestamp('2025-01-08'),
        num_train_trials=20,
        num_test_examples=0,
        test_anchor_time=Timestamp('2025-01-06'),
        num_test_trials=20,
    )

    perm = train.entity_pkey.argsort()
    pd.testing.assert_series_equal(
        train.entity_pkey[perm].reset_index(drop=True),
        pd.Series([0, 10], name='USER_ID'),
    )
    pd.testing.assert_series_equal(
        train.anchor_time,
        pd.to_datetime(pd.Series(['2025-01-08'] * 2)).astype('datetime64[ns]'),
    )
    pd.testing.assert_series_equal(
        train.target[perm].reset_index(drop=True),
        pd.Series([True, False]),
    )
    assert len(test.entity_pkey) == 0
    assert len(test.anchor_time) == 0
    assert len(test.target) == 0


def test_sample_target_empty_entity_table_names_the_table() -> None:
    # Regression: this message used to print the literal
    # '{query.entity_table}'.
    df_dict = {
        'USERS': pd.DataFrame(
            {
                'USER_ID': [1, 2, 3],
                'AGE': [20, 30, 40],
            }
        ),
        'ORDERS': pd.DataFrame(
            {
                'ORDER_ID': range(6),
                'USER_ID': [1, 1, 2, 2, 3, 3],
                'AMOUNT': range(6),
                'TIME': pd.date_range('2025-01-01', periods=6, freq='D'),
            }
        ),
    }
    graph = Graph.from_data(df_dict, verbose=False)
    sampler = LocalSampler(graph, verbose=False)
    query = ValidatedPredictiveQuery(
        target_ast=Condition(
            target=Aggregation(
                aggr='SUM',
                target=Column(fqn='ORDERS.AMOUNT'),
                aggr_time_range=DateOffsetRange(0, 7),
            ),
            op='>',
            value=Constant(value='5', dtype_maybe=Dtype.int),
        ),
        entity_ast=Column(fqn='USERS.USER_ID'),
    )
    sampler._sample_entity_table = lambda **kwargs: pd.DataFrame()

    with pytest.raises(ValueError, match="entity table 'USERS'"):
        sampler.sample_target(
            query,
            num_train_examples=1,
            train_anchor_time=Timestamp('2025-01-05'),
            num_train_trials=5,
            num_test_examples=0,
            test_anchor_time=Timestamp('2025-01-04'),
            num_test_trials=5,
        )


def test_a_context_shortfall_is_reported() -> None:
    r"""Ask for a context the data cannot fill and the shortfall is reported.

    The two coverage warnings were guarded on ``not num_train_examples > 0``,
    which also makes ``num_train_examples // 2`` non-positive, so the
    ``len(train_y) <`` comparison could never hold and neither warning could
    ever be emitted. Nobody was told to raise `max_pq_iterations`.
    """
    import warnings

    import sdm_relational_engine.rfm.base.sampler as sampler_module
    from predictive_query.schema.typing import Stype
    from sdm_relational_engine.rfm.backend.local import LocalSampler

    rng = np.random.default_rng(3)
    num_users = 12_000
    score = np.full(num_users, np.nan)
    score[:5] = rng.uniform(0, 1, 5)
    users = pd.DataFrame({'USER_ID': np.arange(num_users), 'SCORE': score})
    orders = pd.DataFrame(
        {
            'ORDER_ID': np.arange(4_000),
            'USER_ID': rng.integers(0, num_users, 4_000),
            'AMOUNT': rng.uniform(1, 9, 4_000),
            'TIME': pd.Timestamp('2024-01-01')
            + pd.to_timedelta(rng.integers(0, 300, 4_000), unit='D'),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        graph = Graph.from_data(
            {'USERS': users, 'ORDERS': orders}, verbose=False
        )
        graph.infer_metadata(verbose=False)
        graph.infer_links(verbose=False)
        local_sampler = LocalSampler(graph, verbose=False)

    query = ValidatedPredictiveQuery(
        target_ast=Column(fqn='USERS.SCORE', stype_maybe=Stype.numerical),
        entity_ast=Column(fqn='USERS.USER_ID'),
    )

    sampler_module._coverage_warned = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        local_sampler.sample_target(
            query=query,
            num_train_examples=1_000,
            train_anchor_time=pd.Timestamp('2024-06-01'),
            num_train_trials=10_000,
            num_test_examples=0,
            test_anchor_time=pd.Timestamp('2024-06-01'),
            num_test_trials=0,
            random_seed=7,
        )
    sampler_module._coverage_warned = False

    shortfall = [
        str(w.message) for w in caught if 'max_pq_iterations' in str(w.message)
    ]
    assert len(shortfall) == 1
    assert '1,000 context examples' in shortfall[0]
    assert '10,000 candidates' in shortfall[0]


_ORDINAL_ORIGIN = pd.Timestamp('2020-01-01')


@pytest.fixture(scope='module')
def daily_ordinal_sampler() -> LocalSampler:
    r"""One fact row per calendar day, carrying its own day ordinal.

    ``MIN`` and ``MAX`` over a window then read back the window's bounds
    exactly, which is what makes an off-by-a-few-days boundary visible.
    """
    from sdm_relational_engine.rfm.backend.local import LocalTable

    days = pd.date_range(_ORDINAL_ORIGIN, '2023-12-31', freq='D')
    entities = LocalTable(
        df=pd.DataFrame({'EID': [0, 1, 2]}),
        name='ENT',
        primary_key='EID',
    )
    facts = LocalTable(
        df=pd.DataFrame(
            {
                'FID': range(3 * len(days)),
                'EID': [eid for eid in range(3) for _ in days],
                'ORD': [
                    (day - _ORDINAL_ORIGIN).days
                    for _ in range(3)
                    for day in days
                ],
                'TIME': [day for _ in range(3) for day in days],
            }
        ),
        name='FACT',
        primary_key='FID',
        time_column='TIME',
    )
    graph = Graph([entities, facts])
    graph.link('FACT', 'EID', 'ENT')
    return LocalSampler(graph, verbose=False)


def _window_bound(
    sampler: LocalSampler,
    aggr: str,
    start: int,
    end: int,
    unit: str,
    anchor: pd.Timestamp,
) -> int:
    query = ValidatedPredictiveQuery(
        target_ast=Aggregation(
            aggr=aggr,
            target=Column(fqn='FACT.ORD'),
            aggr_time_range=DateOffsetRange(start, end, unit),
        ),
        entity_ast=Column(fqn='ENT.EID'),
    )
    train, _ = sampler.sample_target(
        query,
        num_train_examples=3,
        train_anchor_time=Timestamp(anchor),
        num_train_trials=5,
        num_test_examples=0,
        test_anchor_time=Timestamp(anchor),
        num_test_trials=5,
    )
    bounds = set(train.target.tolist())
    assert len(bounds) == 1
    return int(bounds.pop())


def _ordinal_after(anchor: pd.Timestamp, offset: pd.DateOffset) -> int:
    return (anchor + offset - _ORDINAL_ORIGIN).days + 1


@pytest.mark.parametrize(
    ('start', 'anchor'),
    [
        (1, '2022-11-01'),
        (2, '2022-10-01'),
        (3, '2022-09-01'),
        (3, '2023-01-01'),
    ],
)
def test_a_month_window_starts_on_calendar_arithmetic(
    daily_ordinal_sampler, start, anchor
) -> None:
    r"""The window's start used 31 days a month while its end used the real
    calendar, so a window was short at the front by up to three days and the
    rows in that gap never reached label computation.
    """
    anchor = pd.Timestamp(anchor)
    first = _window_bound(
        daily_ordinal_sampler, 'MIN', start, start + 1, 'months', anchor
    )

    assert first == _ordinal_after(anchor, pd.DateOffset(months=start))


@pytest.mark.parametrize('start', [0, 1, 2])
def test_consecutive_month_windows_tile_without_a_gap(
    daily_ordinal_sampler, start
) -> None:
    r"""`FORECAST n TIMEFRAMES` walks a window forward one step at a time, so
    a step that starts late leaves days belonging to no step at all.
    """
    anchor = pd.Timestamp('2023-01-01')

    last = _window_bound(
        daily_ordinal_sampler, 'MAX', start, start + 1, 'months', anchor
    )
    first_of_next = _window_bound(
        daily_ordinal_sampler, 'MIN', start + 1, start + 2, 'months', anchor
    )

    assert first_of_next == last + 1


@pytest.mark.parametrize(('start', 'unit'), [(30, 'days'), (4, 'weeks')])
def test_a_fixed_length_window_starts_where_it_always_did(
    daily_ordinal_sampler, start, unit
) -> None:
    anchor = pd.Timestamp('2023-01-01')
    first = _window_bound(
        daily_ordinal_sampler, 'MIN', start, start + 1, unit, anchor
    )

    assert first == _ordinal_after(anchor, pd.DateOffset(**{unit: start}))


@pytest.mark.parametrize(
    ('offset', 'earliest_days', 'latest_days'),
    [
        (pd.DateOffset(months=3), 3 * 28, 3 * 31),
        (pd.DateOffset(months=-3), -3 * 31, -3 * 28),
        (pd.DateOffset(years=1), 365, 366),
        (pd.DateOffset(years=-1), -366, -365),
        (pd.DateOffset(days=5), 5, 5),
    ],
)
def test_a_seconds_bound_brackets_the_calendar_offset(
    offset, earliest_days, latest_days
) -> None:
    r"""A negative offset reaches further back the longer its unit is, so the
    sign flips which of the two month lengths is the earlier instant.
    """
    from sdm_relational_engine.rfm.base.utils import (
        _SECONDS_IN_DAY,
        date_offset_to_seconds,
    )

    assert (
        date_offset_to_seconds(offset, 'earliest')
        == earliest_days * _SECONDS_IN_DAY
    )
    assert (
        date_offset_to_seconds(offset, 'latest')
        == latest_days * _SECONDS_IN_DAY
    )
