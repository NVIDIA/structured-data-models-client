# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed shared item projection preserves Python's public value ownership."""

import gc
import json

import numpy as np
import pyarrow as pa
import pytest

from sdm_client._rust import codec, predict, result_items, results


@pytest.fixture(autouse=True)
def default_routes(monkeypatch):
    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)

    def legacy(*args, **kwargs):
        pytest.fail('item projection called a legacy Python decoder')

    for name in ('_rows', '_id', '_as_answered'):
        monkeypatch.setattr(predict, name, legacy)


def result(table):
    return predict._result(
        table,
        model='kumo-relational',
        request_id='one',
        metadata={},
        outputs=('prediction', 'rankings', 'scores', 'embeddings'),
        by_step=False,
    )


def test_projection_preserves_tuple_fields_mappings_and_endpoint_order():
    wide = 2**80 + 1
    table = pa.table(
        {
            'row_index': [0, 1],
            'answer_index': [1, 0],
            'id': [100, 101],
            'prediction': [1.5, 2.5],
            'scores': [[0.1, 0.2], None],
            'embeddings': [[3.0], None],
            'rankings': [[json.dumps({'candidate': wide, 'score': 0.5})], None],
            'probabilities': pa.array(
                [[('yes', 0.8)], None], type=pa.map_(pa.string(), pa.float64())
            ),
            'quantiles': [json.dumps({'0.5': 1.5}), None],
            'metadata': [
                json.dumps({'wide': wide, 'nested': {'ok': True}}),
                None,
            ],
            'explanation': [json.dumps({'feature': ['a', 2]}), None],
        }
    ).replace_schema_metadata(
        {b'sdm.answer_type': b'Float', b'sdm.text_ids': b'false'}
    )
    observed = result(table)
    del table
    gc.collect()
    assert [item.row_index for item in observed.items] == [1, 0]
    empty, item = observed.items
    assert (
        empty.scores is None
        and empty.rankings is None
        and empty.embeddings is None
    )
    assert item.scores == (0.1, 0.2)
    assert item.embeddings == (3.0,)
    assert type(item.rankings) is tuple
    assert item.rankings[0]['candidate'] == wide
    assert item.metadata == {'wide': wide, 'nested': {'ok': True}}
    assert item.explanation == {'feature': ['a', 2]}
    assert item.quantiles == {'0.5': 1.5}
    assert item.probabilities == {'yes': 0.8}
    assert observed.to_pandas()['row_index'].tolist() == [0, 1]


def ranking_table():
    return pa.table(
        {
            'row_index': [0],
            'answer_index': [0],
            'id': [100],
            'rankings': [['"first"', '"second"']],
            'scores': [[0.75, 0.25]],
            'candidates': [[1, 0]],
        }
    ).replace_schema_metadata({b'sdm.text_ids': b'false'})


def test_ranking_projection_reuses_original_candidate_owners_and_requested_outputs():
    first, second = np.int64(2**53 + 1), np.int64(7)
    given = {0: first, 1: second}
    answer = results.Predicted(
        0, 'kumo-relational', 'one', 1, ('prediction', 'rankings'), ()
    )
    observed = predict._ranked(answer, ranking_table(), {}, given)
    item = observed.items[0]
    assert type(item.prediction) is list
    assert item.prediction[0] is second and item.prediction[1] is first
    assert type(item.rankings) is tuple
    assert item.rankings[0]['candidate'] is second
    assert item.rankings[1]['candidate'] is first
    assert [entry['score'] for entry in item.rankings] == [0.75, 0.25]
    only_prediction = results.Predicted(
        0, answer.model, 'two', 1, ('prediction',), ()
    )
    assert (
        predict._ranked(only_prediction, ranking_table(), {}, given)
        .items[0]
        .rankings
        is None
    )


def test_candidate_position_without_original_owner_is_rejected():
    with pytest.raises(
        codec.CodecError, match='candidate position has no owner'
    ):
        result_items.read(
            ranking_table(),
            model='kumo-relational',
            request_id='one',
            metadata={},
            outputs=('prediction', 'rankings'),
            candidates={0: object()},
        )


def test_mixed_regular_ranking_segments_preserve_global_answer_order():
    owner = np.int64(2**53 + 1)
    # Physical row_index order differs from endpoint answer order. The middle
    # answer is a ranking; the rows on either side are ordinary regression.
    table = pa.table(
        {
            'row_index': [0, 1, 2],
            'answer_index': [1, 2, 0],
            'id': [100, 101, 102],
            'prediction': [None, 3.5, 2.5],
            'rankings': [['"candidate"'], None, None],
            'scores': [[0.75], None, None],
            'candidates': [[0], None, None],
        }
    ).replace_schema_metadata(
        {b'sdm.answer_type': b'Float', b'sdm.text_ids': b'false'}
    )
    items = result_items.read(
        table,
        model='kumo-relational',
        request_id='mixed',
        metadata={},
        outputs=('prediction', 'rankings'),
        candidates={0: owner},
    )
    assert [item['row_index'] for item in items] == [2, 0, 1]
    assert items[0]['prediction'] == 2.5
    assert items[1]['prediction'][0] is owner
    assert items[1]['rankings'][0]['candidate'] is owner
    assert type(items[1]['rankings']) is tuple
    assert items[2]['prediction'] == 3.5
