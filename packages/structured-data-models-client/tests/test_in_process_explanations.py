# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A model run in this process explains its answers as a NIM does.

sdm's gradient tensors are turned into the record a NIM sends per row, the
same bytes the NIM's converter writes, so the client's ``GradientDecoder``
reads an in-process explanation unchanged. Small random-weight models stand
in for the published ones.
"""

from __future__ import annotations

import copy
import json
import math
import warnings
from types import SimpleNamespace
from typing import Any

import pytest
from sample_tables import graph, people
from test_relational_explanations import ROWS, TABLES, gathered
from test_tabular_explanations import CHURN, tabular_asked

from sdm_client import (
    KumoRelational,
    KumoTabular,
    PredictiveQuery,
    Request,
    SDMClient,
)
from sdm_client.errors import StructuredDataError
from sdm_client.explanations import GradientConverter, GradientDecoder

pytestmark = pytest.mark.filterwarnings('ignore')

CLASSES = [
    {'kind': 'probability', 'class': 'false'},
    {'kind': 'probability', 'class': 'true'},
]


def attribution(sdm, gradients: list, columns: dict[str, list[Any]]) -> Any:
    """An explainer's table: ``gradients[output][row][column]`` over the
    numerical columns of a table holding ``columns``."""
    import torch

    stypes = {
        name: 'numerical' if isinstance(cells[0], float) else 'categorical'
        for name, cells in columns.items()
    }
    table = sdm.TableTensor.from_columns(columns, stypes=stypes)
    return torch.stack(
        [
            table.replace_blocks(
                numerical=torch.tensor(by_row, dtype=torch.float32).reshape(
                    len(by_row), -1
                )
            )
            for by_row in gradients
        ]
    )


def related_tables(sdm, tables: dict[str, Any]) -> Any:
    return sdm.RelatedTables(tables=tables, relationships=[], task_links=[])


def decoded(explained: list[dict], payload: Any = None) -> Any:
    """``explained``, read back as the client reads a NIM's records."""
    items = [
        SimpleNamespace(id=None, row_index=index, explanation=record)
        for index, record in enumerate(explained)
    ]
    return GradientDecoder(payload).decode(items)


def test_classes_are_named_as_they_are_sent():
    """Each class is one probability output, named by its label's text.

    Assumptions:
    - The labels are what the caller calls each class, in encoded order.
    """
    assert GradientConverter.class_outputs(['false', 'true']) == CLASSES
    assert GradientConverter.class_outputs([0, 1]) == [
        {'kind': 'probability', 'class': '0'},
        {'kind': 'probability', 'class': '1'},
    ]


def test_a_regression_explains_its_statistic_then_each_level():
    """The point statistic comes first, then one output per quantile level.

    Assumptions:
    - No levels means quantiles were not requested.
    """
    assert GradientConverter.regression_outputs('median', [0.1, 0.9]) == [
        {'kind': 'prediction', 'statistic': 'median'},
        {'kind': 'quantile', 'level': 0.1},
        {'kind': 'quantile', 'level': 0.9},
    ]
    assert GradientConverter.regression_outputs('mean', []) == [
        {'kind': 'prediction', 'statistic': 'mean'}
    ]


def test_a_tabular_row_gets_the_nim_record(sdm):
    """Row ``i``'s record carries its own numerical gradients, per output.

    Assumptions:
    - Two outputs, three rows, numerical columns ``a`` and ``b`` beside a
      categorical one, which has no gradient.
    """
    instance = attribution(
        sdm,
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]],
        ],
        {'a': [0.0, 0.0, 0.0], 'k': ['x', 'y', 'z'], 'b': [0.0, 0.0, 0.0]},
    )

    explained = GradientConverter.convert(CLASSES, instance, rows=3)

    assert len(explained) == 3
    assert explained[1] == {
        'format': 'sdm_gradient_v1',
        'details': {
            'input_space': 'preprocessed_numerical',
            'outputs': CLASSES,
            'instance_table': {
                'row_indices': [1],
                'columns': ['a', 'b'],
                'gradients': [[[3.0, 4.0]], [[-3.0, -4.0]]],
            },
            'related_tables': {},
        },
    }
    gradients = explained[1]['details']['instance_table']['gradients']
    assert all(type(value) is float for value in gradients[0][0])
    assert json.loads(json.dumps(explained)) == explained


def test_singleton_batch_dimensions_are_dropped(sdm):
    """Gradients ``[output, 1, row, column]`` give the same records.

    Assumptions:
    - sdm may keep a batch dimension of one between output and row axes.
    """
    instance = attribution(sdm, [[[1.0], [2.0]]], {'a': [0.0, 0.0]})

    batched = instance.unsqueeze(1)

    assert GradientConverter.convert(
        CLASSES[:1], batched, rows=2
    ) == GradientConverter.convert(CLASSES[:1], instance, rows=2)


def test_related_rows_go_to_the_row_that_owns_them(sdm):
    """Each record holds the related rows its predict row owns, in order;
    a row owned by none is in no record.

    Assumptions:
    - Two predict rows; ``orders`` row 2 is owned by none (owner -1).
    """
    instance = attribution(sdm, [[[0.5], [0.6]]], {'a': [0.0, 0.0]})
    orders = attribution(
        sdm, [[[1.0], [2.0], [3.0], [4.0]]], {'amount': [0.0] * 4}
    )

    first, second = GradientConverter.convert(
        CLASSES[:1],
        instance,
        related_tables(sdm, {'orders': orders}),
        rows=2,
        related_rows={'orders': 4},
        owners={'orders': [1, 0, -1, 1]},
    )

    assert first['details']['related_tables'] == {
        'orders': {
            'row_indices': [1],
            'columns': ['amount'],
            'gradients': [[[2.0]]],
        }
    }
    assert second['details']['related_tables'] == {
        'orders': {
            'row_indices': [0, 3],
            'columns': ['amount'],
            'gradients': [[[1.0], [4.0]]],
        }
    }


def test_a_related_table_without_numerical_columns_has_empty_gradients(sdm):
    """A table with nothing numerical names no columns, one empty list per
    owned row and output.

    Assumptions:
    - ``tags`` holds one categorical column, two rows owned by row 0.
    """
    instance = attribution(sdm, [[[0.5]], [[0.7]]], {'a': [0.0]})
    tags = attribution(sdm, [[[], []], [[], []]], {'tag': ['x', 'y']})

    (record,) = GradientConverter.convert(
        CLASSES,
        instance,
        related_tables(sdm, {'tags': tags}),
        rows=1,
        related_rows={'tags': 2},
        owners={'tags': [0, 0]},
    )

    assert record['details']['related_tables']['tags'] == {
        'row_indices': [0, 1],
        'columns': [],
        'gradients': [[[], []], [[], []]],
    }


@pytest.mark.parametrize(
    ('outputs', 'rows', 'related_rows'),
    [
        (CLASSES, 3, {'orders': 1}),
        (CLASSES[:1], 2, {'orders': 1}),
        (CLASSES, 2, {'orders': 2}),
        (CLASSES, 2, {}),
    ],
    ids=['rows', 'outputs', 'related rows', 'related tables'],
)
def test_gradients_of_another_shape_are_refused(
    sdm, outputs, rows, related_rows
):
    """Gradients that disagree with the outputs, the rows or the tables sent
    raise rather than explain the wrong thing.

    Assumptions:
    - The tensors hold two outputs, two predict rows, one ``orders`` row.
    """
    instance = attribution(
        sdm, [[[1.0], [2.0]], [[3.0], [4.0]]], {'a': [0.0, 0.0]}
    )
    orders = attribution(sdm, [[[1.0]], [[2.0]]], {'amount': [0.0]})

    with pytest.raises(StructuredDataError) as caught:
        GradientConverter.convert(
            outputs,
            instance,
            related_tables(sdm, {'orders': orders}),
            rows=rows,
            related_rows=related_rows,
            owners={'orders': [0]},
        )
    assert caught.value.code == 'INTERNAL_ERROR'


@pytest.mark.parametrize('value', [math.nan, math.inf])
def test_a_non_finite_gradient_is_refused(sdm, value):
    """A NaN or infinite gradient raises.

    Assumptions:
    - Tabular: no related tables.
    """
    instance = attribution(sdm, [[[1.0], [value]]], {'a': [0.0, 0.0]})

    with pytest.raises(StructuredDataError) as caught:
        GradientConverter.convert(CLASSES[:1], instance, rows=2)
    assert caught.value.code == 'INTERNAL_ERROR'


def test_a_tabular_record_reads_back_as_its_entry(sdm):
    """The decoder reads each record as the documented tabular entry.

    Assumptions:
    - No payload, as for kumo-tabular; rows have no id.
    """
    instance = attribution(
        sdm,
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        {'a': [0.0, 0.0], 'b': [0.0, 0.0]},
    )

    explanations = decoded(GradientConverter.convert(CLASSES, instance, rows=2))

    assert dict(explanations) == {
        0: {
            'outputs': CLASSES,
            'instance': {
                'columns': ['a', 'b'],
                'gradients': [[1.0, 2.0], [5.0, 6.0]],
            },
        },
        1: {
            'outputs': CLASSES,
            'instance': {
                'columns': ['a', 'b'],
                'gradients': [[3.0, 4.0], [7.0, 8.0]],
            },
        },
    }


def test_a_relational_record_reads_back_as_the_entitys_rows(sdm):
    """Each entity's related rows decode as its own rows, with the gradient
    the converter put on each.

    Assumptions:
    - Users 5 and 7 gathered in one call, related rows interleaved; each row
      is owned by the entity its ``instance_id`` names and its gradient is
      its position in its table.
    """
    payload = gathered(5, 7)
    instances = [row[0] for row in payload['predict']['instance_table']['rows']]
    sent = payload['predict']['related_tables']
    related = related_tables(
        sdm,
        {
            name: attribution(
                sdm,
                [[[float(at)] for at in range(len(sent[name]['rows']))]],
                {column: [0.0] * len(sent[name]['rows'])},
            )
            for name, (_, column) in TABLES.items()
        },
    )
    instance = attribution(sdm, [[[0.0], [0.0]]], {'anchor': [0.0, 0.0]})

    explained = GradientConverter.convert(
        CLASSES[:1],
        instance,
        related,
        rows=2,
        related_rows={name: len(table['rows']) for name, table in sent.items()},
        owners={
            name: [instances.index(row[0]) for row in table['rows']]
            for name, table in sent.items()
        },
    )
    explanations = decoded(explained, payload)

    for index, user in enumerate((5, 7)):
        for name, (columns, column) in TABLES.items():
            block = explanations[index]['related'][name]
            visible = [
                c for c in columns if c not in ('instance_id', '__node_id')
            ]
            assert block['rows'] == [
                dict(zip(visible, row, strict=True)) for row in ROWS[user][name]
            ]
            assert block['columns'] == [column]
            positions = [
                at
                for at, row in enumerate(sent[name]['rows'])
                if row[0] == instances[index]
            ]
            assert block['gradients'] == [[[float(at)] for at in positions]]


def test_owners_are_the_predict_rows_sdm_joins_each_related_row_to(sdm):
    """Each related row's owner is the predict row sdm's task graph assigns
    it; a row joining none is -1.

    Assumptions:
    - Two predict rows link to users 0 and 1; order 2 names user 9, absent.
    """
    x = sdm.TableTensor.from_columns(
        {'ENTITY': [1, 0]}, stypes={'ENTITY': 'id'}
    )
    related = sdm.RelatedTables(
        tables={
            'users': sdm.TableTensor.from_columns(
                {'user_id': [0, 1]}, stypes={'user_id': 'id'}
            ),
            'orders': sdm.TableTensor.from_columns(
                {'user_id': [1, 0, 9]}, stypes={'user_id': 'id'}
            ),
        },
        relationships=[
            {
                'left_table': 'orders',
                'left_column': 'user_id',
                'right_table': 'users',
                'right_column': 'user_id',
            }
        ],
        task_links=[
            {
                'task_column': 'ENTITY',
                'table': 'users',
                'table_column': 'user_id',
            }
        ],
    )

    assert GradientConverter.entity_rows(x, related) == {
        'users': [1, 0],
        'orders': [0, 1, -1],
    }


def test_a_fitted_tabular_model_is_explained_per_row(
    sdm, tabular, tabular_request
):
    """A real kumo-tabular explanation converts and decodes to one entry per
    row, in the payload's row order.

    Assumptions:
    - A small random-weight model fitted with one estimator, as in process.
    - The query is read from the payload as the in-process backend reads it.
    """
    import torch
    from sdm.explain import GradientExplainer

    from sdm_client.adapters.tabular import TabularAdapter
    from sdm_client.in_process.tabular import Tabular

    (payload,) = TabularAdapter().prediction_plan(
        tabular_request(rows=3), ('prediction', 'explanation')
    )
    served = Tabular()
    context = served.context(payload)
    x, related = served.query(context, payload, payload['schema'])
    fitted = copy.copy(tabular)
    fitted.fit(
        context.x,
        context.y,
        num_estimators=1,
        generator=torch.Generator().manual_seed(42),
    )

    instance, nothing = GradientExplainer().explain(fitted, x, related)
    alone, _ = GradientExplainer().explain(fitted, x[[2]], related)
    sent = len(payload['predict']['instance_table']['rows'])
    explanations = decoded(
        GradientConverter.convert(
            GradientConverter.class_outputs(context.labels), instance, rows=sent
        )
    )

    assert nothing is None
    assert instance.numerical.shape[-2] == sent
    assert torch.allclose(instance.numerical[:, 2], alone.numerical[:, 0])
    assert sorted(explanations) == [0, 1, 2]
    for entry in explanations.values():
        assert entry['outputs'] == GradientConverter.class_outputs(
            context.labels
        )
        assert entry['instance']['columns'] == ['age']
        assert [
            len(by_column) for by_column in entry['instance']['gradients']
        ] == [1, 1]


def relational_payload() -> dict[str, Any]:
    """A gathered kumo-relational call: six labelled users, two to score."""

    def related(users: list[int], first: int) -> dict[str, Any]:
        orders: list[list[Any]] = []
        for at, user in enumerate(users):
            for n in range(2):
                orders.append(
                    [first + at, user * 10 + n, user, float(user + n)]
                )
        return {
            'users': {
                'format': 'arrays',
                'columns': ['instance_id', 'user_id', 'tenure'],
                'rows': [
                    [first + at, user, float(user)]
                    for at, user in enumerate(users)
                ],
            },
            'orders': {
                'format': 'arrays',
                'columns': ['instance_id', 'order_id', 'user_id', 'amount'],
                'rows': orders,
            },
        }

    labelled = list(range(6))
    return {
        'model': 'kumo-relational',
        'task': {
            'kind': 'classification',
            'target': {'column_name': 'TARGET', 'dtype': 'string'},
            'entity_table_names': ['users'],
        },
        'schema': {
            'instance_table': {
                'columns': {
                    'instance_id': {'dtype': 'int64'},
                    '__kumo_entity_ref_0': {'dtype': 'int64'},
                    'TARGET': {'dtype': 'string'},
                },
                'primary_key': 'instance_id',
            },
            'related_tables': {
                'users': {
                    'columns': {
                        'instance_id': {'dtype': 'int64'},
                        'user_id': {'dtype': 'int64'},
                        'tenure': {'dtype': 'float64'},
                    },
                    'primary_key': ['instance_id', 'user_id'],
                },
                'orders': {
                    'columns': {
                        'instance_id': {'dtype': 'int64'},
                        'order_id': {'dtype': 'int64'},
                        'user_id': {'dtype': 'int64'},
                        'amount': {'dtype': 'float64'},
                    },
                    'primary_key': ['instance_id', 'order_id'],
                },
            },
            'relationships': [
                {
                    'source_columns': ['__kumo_entity_ref_0'],
                    'target_table': 'users',
                    'target_columns': ['user_id'],
                },
                {
                    'source_table': 'orders',
                    'source_columns': ['user_id'],
                    'target_table': 'users',
                    'target_columns': ['user_id'],
                },
            ],
        },
        'context': {
            'instance_table': {
                'format': 'arrays',
                'columns': ['instance_id', '__kumo_entity_ref_0', 'TARGET'],
                'rows': [
                    [at, user, 'yes' if user % 2 else 'no']
                    for at, user in enumerate(labelled)
                ],
            },
            'related_tables': related(labelled, 0),
        },
        'predict': {
            'instance_table': {
                'format': 'arrays',
                'columns': ['instance_id', '__kumo_entity_ref_0'],
                'rows': [[100, 7], [101, 8]],
            },
            'related_tables': related([7, 8], 100),
        },
    }


def test_a_fitted_relational_model_is_explained_per_entity(sdm, relational):
    """A real kumo-relational explanation converts and decodes; sdm's tensors
    hold the payload's related rows in the payload's order.

    Assumptions:
    - A small random-weight model fitted with one estimator, as in process.
    - Each related row's owner per sdm is the entity its ``instance_id`` names.
    """
    import torch
    from sdm.explain import GradientExplainer

    from sdm_client.in_process.relational import Relational

    payload = relational_payload()
    served = Relational()
    context = served.context(payload)
    x, related = served.query(context, payload, payload['schema'])
    fitted = copy.copy(relational)
    fitted.fit(
        context.x,
        context.y,
        context.related,
        num_estimators=1,
        generator=torch.Generator().manual_seed(42),
    )

    instance, attributed = GradientExplainer().explain(fitted, x, related)
    sent = payload['predict']['related_tables']
    instances = [row[0] for row in payload['predict']['instance_table']['rows']]
    owned = GradientConverter.entity_rows(x, related)
    explanations = decoded(
        GradientConverter.convert(
            GradientConverter.class_outputs(context.labels),
            instance,
            attributed,
            rows=len(instances),
            related_rows={name: len(t['rows']) for name, t in sent.items()},
            owners=owned,
        ),
        payload,
    )

    for name, table in sent.items():
        assert attributed.tables[name].numerical.shape[-2] == len(table['rows'])
        assert owned[name] == [instances.index(row[0]) for row in table['rows']]
    assert sorted(explanations) == [0, 1]
    for index, user in enumerate((7, 8)):
        entry = explanations[index]
        assert entry['outputs'] == GradientConverter.class_outputs(
            context.labels
        )
        assert [
            row['user_id'] for row in entry['related']['orders']['rows']
        ] == [user, user]
        assert entry['related']['orders']['columns'] == ['amount']


BINARY = 'PREDICT COUNT(orders.*, 0, 30, days) > 2 FOR EACH users.user_id'
EXPLAINED = ['prediction', 'probabilities', 'explanation']


def split(target: str) -> tuple[Any, Any]:
    """``people``'s 45 labelled rows, and the rest to score."""
    frame = people()[['person_id', 'age', 'visits', target]]
    return frame.iloc[:45], frame.iloc[45:].drop(columns=[target])


def asked(target: str, task: str) -> Request:
    context, scored = split(target)
    return Request(context=context, predict=scored, target=target, task=task)


def outputs_of(result: Any) -> list[list[dict[str, Any]]]:
    """Each row's explained outputs, in row order."""
    return [item.explanation['details']['outputs'] for item in result]


@pytest.fixture(scope='module')
def responsive(tabular) -> Any:
    """``tabular`` with its weights nudged, so its answer moves with its
    input: the random-weight model answers every row alike."""
    import torch

    model = copy.deepcopy(tabular)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(
                0.1 * torch.randn(parameter.shape, generator=generator)
            )
    return model


@pytest.fixture
def explains(sdm, monkeypatch) -> list[Any]:
    """Every model ``GradientExplainer`` is asked to explain, as called."""
    from sdm.explain import GradientExplainer

    calls: list[Any] = []
    explain = GradientExplainer.explain

    def watched(self, model, *args, **kwargs):
        calls.append(model)
        return explain(self, model, *args, **kwargs)

    monkeypatch.setattr(GradientExplainer, 'explain', watched)
    return calls


@pytest.mark.parametrize('target', ['churn', 'plan'])
def test_a_tabular_answer_carries_a_nim_record_per_row(tabular, target):
    """Every row carries an ``sdm_gradient_v1`` record over the classes
    its probabilities report, in their order, and reads back as an
    explanation of the columns it was scored on.

    Assumptions:
    - ``churn`` has two classes and ``plan`` several.
    """
    with SDMClient.in_process(tabular) as client:
        result = client.predict(
            asked(target, 'classification'),
            model=KumoTabular(),
            outputs=EXPLAINED,
        )
    frame = result.to_pandas()

    assert len(result) == len(frame) > 0
    assert {item.explanation['format'] for item in result} == {
        'sdm_gradient_v1'
    }
    for item, outputs in zip(result, outputs_of(result), strict=True):
        assert outputs == [
            {'kind': 'probability', 'class': name}
            for name in item.probabilities
        ]
    assert len(result.explanations) == len(frame)
    for entry in result.explanations.values():
        assert set(entry) >= {'outputs', 'instance'}
        assert sorted(entry['instance']['columns']) == ['age', 'visits']


def test_a_tabular_explanation_is_the_gradient_of_each_reported_class(
    sdm, responsive
):
    """Output ``k`` of a row's record is sdm's gradient of the class it
    names, whatever order the model put its columns in.

    Assumptions:
    - A model whose answer moves with its input, fitted as in process: one
      estimator, seed 42.
    - sdm's explainer, run on the bare model, explains every probability
      column in the model's own order.
    """
    import torch
    from sdm.explain import GradientExplainer

    with SDMClient.in_process(responsive) as client:
        result = client.predict(
            asked('plan', 'classification'),
            model=KumoTabular(),
            outputs=EXPLAINED,
        )

    context, rows = split('plan')
    features = {'age': 'numerical', 'visits': 'numerical'}
    fitted = copy.copy(responsive)
    fitted.fit(
        sdm.TableTensor.from_pandas(context, features),
        sdm.TableTensor.from_pandas(context, {'plan': 'categorical'}),
        num_estimators=1,
        generator=torch.Generator().manual_seed(42),
    )
    scored = sdm.TableTensor.from_pandas(rows, features)
    with torch.enable_grad():
        bare, _ = GradientExplainer().explain(fitted, scored)
    columns = list(fitted.predict(scored).columns[sdm.Stype.numerical])
    expected = bare.numerical.detach().reshape(len(columns), len(result), -1)

    assert expected.abs().max() > 1e-3

    for row, item in enumerate(result):
        details = item.explanation['details']
        for output, gradients in zip(
            details['outputs'],
            details['instance_table']['gradients'],
            strict=True,
        ):
            at = columns.index(output['class'])
            assert torch.allclose(
                torch.tensor(gradients[0]), expected[at, row], atol=1e-6
            )


def test_nothing_is_explained_unless_asked(tabular, explains):
    """An answer without ``'explanation'`` never runs the explainer and
    carries no explanation.

    Assumptions:
    - ``GradientExplainer.explain`` is watched, not replaced.
    """
    with SDMClient.in_process(tabular) as client:
        result = client.predict(
            asked('churn', 'classification'), model=KumoTabular()
        )

    assert explains == []
    assert all(item.explanation is None for item in result)
    assert len(result.explanations) == 0


def regression_answer(tabular, fields, levels=None) -> Any:
    """A tabular regression answered in process for ``fields``, with
    ``levels`` as the quantile levels the payload asks for."""
    from sdm_client.adapters.tabular import TabularAdapter

    adapter = TabularAdapter()
    (payload,) = adapter.prediction_plan(asked('spend', 'regression'), fields)
    if levels is not None:
        payload['output']['quantile_levels'] = levels
    with SDMClient.in_process(tabular) as client:
        return client._backend.predict(adapter, payload, fields)


def test_a_regression_explains_its_point_and_the_levels_asked_for(tabular):
    """A regression explains its point prediction, then each quantile level
    the request asked for, not the 999 the model predicts.

    Assumptions:
    - The payload asks for levels 0.25 and 0.75, mean as the statistic.
    """
    result = regression_answer(
        tabular, ('prediction', 'quantiles', 'explanation'), [0.25, 0.75]
    )

    expected = [
        {'kind': 'prediction', 'statistic': 'mean'},
        {'kind': 'quantile', 'level': 0.25},
        {'kind': 'quantile', 'level': 0.75},
    ]
    assert outputs_of(result) == [expected] * len(result)
    for item in result:
        gradients = item.explanation['details']['instance_table']['gradients']
        assert len(gradients) == 3


def test_a_regression_without_quantiles_explains_its_point_alone(tabular):
    """Without ``'quantiles'`` asked for, only the point prediction is
    explained.

    Assumptions:
    - The default statistic, mean.
    """
    result = regression_answer(tabular, ('prediction', 'explanation'))

    assert outputs_of(result) == [
        [{'kind': 'prediction', 'statistic': 'mean'}]
    ] * len(result)


def test_a_session_explains_when_asked(tabular, explains):
    """A session fitted for ``'explanation'`` explains each scoring call
    with the model it fitted.

    Assumptions:
    - ``GradientExplainer.explain`` is watched, not replaced.
    """
    request = asked('churn', 'classification')
    with SDMClient.in_process(tabular) as client:
        with client.fit(
            request, model=KumoTabular(), outputs=EXPLAINED
        ) as fitted:
            result = fitted.predict(request)

    assert len(explains) == 1
    assert len(result.explanations) == len(result) > 0
    assert {item.explanation['format'] for item in result} == {
        'sdm_gradient_v1'
    }


def test_a_relational_answer_explains_each_entity_and_its_rows(relational):
    """Every entity carries a record, and its explanation names the related
    rows it was scored with.

    Assumptions:
    - A binary question over two entities of the sample graph.
    """
    request = PredictiveQuery(BINARY, graph()).materialize(indices=[0, 1])
    with SDMClient.in_process(relational) as client:
        result = client.predict(
            request, model=KumoRelational(), outputs=EXPLAINED
        )

    assert len(result) == 2
    assert {item.explanation['format'] for item in result} == {
        'sdm_gradient_v1'
    }
    assert len(result.explanations) == 2
    for entry in result.explanations.values():
        assert set(entry) == {'outputs', 'related'}
        assert 'users' in entry['related']
        assert [output['kind'] for output in entry['outputs']] == [
            'probability',
            'probability',
        ]


def test_in_process_tabular_explains(tabular):
    """An `EXPLAIN` query answered in process carries an explanation per row.

    Assumptions:
    - A small random-weight tabular model.
    """
    with SDMClient.in_process(tabular) as client:
        result = client.predict(
            tabular_asked(f'EXPLAIN {CHURN}'), model=KumoTabular()
        )

    assert len(result.explanations) == len(result) == 2
    assert {item.explanation['format'] for item in result} == {
        'sdm_gradient_v1'
    }


def test_in_process_relational_explains(relational):
    """An `EXPLAIN` query gathered by the engine and answered in process
    carries an explanation per entity.

    Assumptions:
    - A small random-weight relational model; a binary question over two
      entities of the sample graph.
    """
    request = PredictiveQuery(
        'EXPLAIN PREDICT COUNT(orders.*, 0, 30, days) > 2 '
        'FOR EACH users.user_id',
        graph(),
    ).materialize(indices=[0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with SDMClient.in_process(relational) as client:
            result = client.predict(request, model=KumoRelational())

    assert len(result.explanations) == len(result) == 2
