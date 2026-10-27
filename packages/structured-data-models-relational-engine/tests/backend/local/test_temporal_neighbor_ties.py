# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The public local sampler retains DF-1 source order on temporal ties."""

import numpy as np
import pandas as pd
from sdm_relational_engine.rfm import Graph
from sdm_relational_engine.rfm.backend.local import LocalSampler, LocalTable
from sdm_relational_engine.rfm.inference.context import SampledEdgeLayout


def test_sanitized_temporal_neighbors_keep_source_order_at_equal_seconds():
    # This is the frozen public first-duplicate-without-time journey.
    # Half-second features tie with whole-second features in the sampler.
    counts = np.tile(
        np.array(
            [1200, -1800, 500, -1200, -1500, -700, 0, -500],
            dtype='int64',
        )
        * 1_000_000,
        8,
    )
    events = pd.DataFrame(
        {
            'event_id': np.arange(64),
            'user_id': np.repeat(np.arange(1, 9), 8),
            'marker': np.arange(64),
            'at': pd.Series(counts.view('datetime64[ns]')),
        }
    )
    events.loc[0, 'at'] = pd.NaT
    events.loc[1, 'event_id'] = 0
    events.loc[2, 'at'] = pd.NaT
    graph = Graph(
        [
            LocalTable(
                df=pd.DataFrame({'user_id': np.arange(1, 9)}),
                name='users',
                primary_key='user_id',
            ),
            LocalTable(
                df=events,
                name='events',
                primary_key='event_id',
                time_column='at',
            ),
        ]
    )
    graph.link('events', 'user_id', 'users')
    sampler = LocalSampler(graph, verbose=False)
    subgraph = sampler.sample_subgraph(
        entity_table_names=('users',),
        entity_pkey=pd.Series([7, 8]),
        anchor_time=pd.Series(
            [pd.Timestamp('1970-01-01T00:00:00.100000000')] * 2
        ),
        num_neighbors=[2],
        random_seed=7,
    )

    sampled = subgraph.tables['events']
    pd.testing.assert_frame_equal(
        sampled.df,
        pd.DataFrame(
            {
                'marker': [50, 54, 58, 62],
                'at': pd.Series(
                    np.array([500_000_000, 0, 500_000_000, 0]).view(
                        'datetime64[ns]'
                    )
                ),
            }
        ),
    )
    assert sampled.batch.tolist() == [0, 0, 1, 1]
    assert sampled.row is None
    assert sampled.num_sampled_nodes == [0, 4]
    assert subgraph.tables['users'].df['user_id'].tolist() == [7, 8]
    assert subgraph.anchor_time.tolist() == [0, 0]
    link = subgraph.links[('events', 'user_id', 'users')]
    assert link.layout == SampledEdgeLayout.CSC
    assert link.row is None
    assert link.col is not None
    assert link.col.tolist() == [0, 2, 4]
    assert link.num_sampled_edges == [4]
    dropped = sampler.sanitization_report.tables['events']
    assert dropped.output_rows == 61
    assert dropped.duplicate_primary_key_rows == 1
    assert dropped.null_time_rows == 2
