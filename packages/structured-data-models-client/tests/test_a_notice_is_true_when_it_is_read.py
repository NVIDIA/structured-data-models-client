# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the client says about a request has to be true when it is said.

Two notices are easy to get subtly wrong. The one about context rows that
carry the answer being asked for is issued before those rows are removed, so
that a caller running with warnings as errors gets the raise with the
payloads still whole, which means it must describe the rule rather than an
edit already made. And the one about a column that could be a key has to
agree with the graph it is describing, for one candidate and for several, and
for one table it is shared with and for several.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from sdm_client import Schema


def test_the_suggestion_reads_as_a_sentence_for_several_candidates(recwarn):
    """Plural candidates took a singular tail, so the message fell apart on
    exactly the graphs with most to say."""
    Schema.from_data(
        {
            'line_items': pd.DataFrame(
                {'ORDER_NO': ['O1', 'O2'], 'SKU': ['s1', 's2'], 'qty': [1, 2]}
            ),
            'shipments': pd.DataFrame(
                {'SHIPMENT_ID': [1, 2], 'ORDER_NO': ['O1', 'O2']}
            ),
            'stock': pd.DataFrame({'STOCK_ID': [1, 2], 'SKU': ['s1', 's2']}),
        },
        primary_keys={'shipments': 'SHIPMENT_ID', 'stock': 'STOCK_ID'},
    )

    said = ' '.join(str(w.message) for w in recwarn)
    assert "'ORDER_NO' and 'SKU' identify every row" in said
    assert 'are named by other tables too' in said
    assert 'each of them either a key or a foreign key' in said
    assert 'If one of them identifies' in said
    assert 'if they instead name' in said
    assert 'the tables they are shared with' in said
    assert 'those tables are the ones that need the key' in said
    assert 'it identifies' not in said


def test_a_unique_column_is_offered_as_a_key_and_as_a_foreign_key(recwarn):
    """A column unique in the rows read is either this table's key or a
    foreign key that happens not to repeat there. Naming only the first
    reading sends a caller to declare a key that invents a parent."""
    Schema.from_data(
        {
            'line_items': pd.DataFrame(
                {'ORDER_NO': ['O1', 'O2'], 'qty': [1, 2]}
            ),
            'shipments': pd.DataFrame(
                {'SHIPMENT_ID': [1, 2, 3], 'ORDER_NO': ['O1', 'O1', 'O2']}
            ),
        },
        primary_keys={'shipments': 'SHIPMENT_ID'},
    )

    said = ' '.join(str(w.message) for w in recwarn)
    assert 'nothing can link to a table without one' in said
    assert "'ORDER_NO' identifies every row" in said
    assert 'is named by another table too' in said
    assert 'the table it is shared with' in said
    assert 'they are shared with' not in said
    assert 'makes it either a key or a foreign key' in said
    assert 'If it identifies' in said
    assert 'that table is the one that needs the key' in said


def one_unkeyed_table(candidates: int, others: int) -> dict:
    """`candidates` unique columns of an unkeyed table, spread over `others`
    tables that each declare a key of their own."""
    columns = [f'K{i}' for i in range(candidates)]
    frames = {
        'li': pd.DataFrame(
            {c: [f'{c}a', f'{c}b'] for c in columns} | {'q': [1, 2]}
        )
    }
    keys = {}
    for i in range(others):
        held = columns[i::others] or columns
        frames[f't{i}'] = pd.DataFrame(
            {f'ID{i}': [1, 2]} | {c: [f'{c}a', f'{c}b'] for c in held}
        )
        keys[f't{i}'] = f'ID{i}'
    return frames, keys


@pytest.mark.parametrize(
    ('candidates', 'others', 'expected'),
    [
        (1, 1, 'the table it is shared with, that table is the one'),
        (2, 1, 'the table they are shared with, that table is the one'),
        (1, 2, 'the tables it is shared with, those tables are the ones'),
        (2, 2, 'the tables they are shared with, those tables are the ones'),
    ],
)
def test_the_sentence_agrees_for_every_count_of_column_and_table(
    candidates, others, expected, recwarn
):
    """The two counts move independently: several columns can be shared with
    one table, and one column with several. Agreeing either by the other put
    a sentence in front of the caller that disagrees with their own graph."""
    frames, keys = one_unkeyed_table(candidates, others)
    Schema.from_data(frames, primary_keys=keys)

    said = ' '.join(str(w.message) for w in recwarn)
    assert 'nothing can link to a table without one' in said
    assert expected in said


def test_context_warning_as_error_preserves_input_data():
    from test_context_that_answers_the_question import STATIC, graph

    from sdm_client import PredictiveQuery

    schema = graph()
    before = {
        name: table.data.copy(deep=True)
        for name, table in schema.tables.items()
    }
    query = PredictiveQuery(STATIC, schema)
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message=r'.*CONTEXT_ANSWERS.*')
        with pytest.raises(UserWarning, match='are not sent as context'):
            query.materialize(indices=[1, 2], num_context_samples=300)
    for name, expected in before.items():
        pd.testing.assert_frame_equal(schema[name].data, expected)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        request = query.materialize(indices=[1, 2], num_context_samples=300)
    context = request.payloads[0]['context']['instance_table']
    reference = context['columns'].index('__kumo_entity_ref_0')
    assert len(context['rows']) == 298
    assert {row[reference] for row in context['rows']}.isdisjoint({1, 2})
    assert any('are not sent as context' in str(w.message) for w in caught)


def test_each_batch_excludes_its_own_scored_entities():
    from test_context_that_answers_the_question import STATIC, graph

    from sdm_client import PredictiveQuery

    query = PredictiveQuery(STATIC, graph())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        request = query.materialize(
            indices=[1, 2, 3],
            rows_per_call=1,
            num_context_samples=300,
        )
    assert len(request.payloads) == 3
    assert any('are not sent as context' in str(w.message) for w in caught)
    for payload in request.payloads:
        context = payload['context']['instance_table']
        predict = payload['predict']['instance_table']
        context_ref = context['columns'].index('__kumo_entity_ref_0')
        predict_ref = predict['columns'].index('__kumo_entity_ref_0')
        assert {r[context_ref] for r in context['rows']}.isdisjoint(
            {r[predict_ref] for r in predict['rows']}
        )
