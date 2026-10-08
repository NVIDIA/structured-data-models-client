# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P7.labels-gen: labels from the Rust core are pandas', bit for bit.

- At least 100,000 generated groups (seed 20261015) through
  ``labels.aggregation_type``: ``float32`` and ``float64`` features with
  NaN, infinities, signed zeros, subnormals and the largest ``float32``;
  ``int64`` near its bounds, where sums wrap; nullable ``Int64`` and
  ``Float64`` with NA; timestamps; every aggregation; groups of thousands of
  rows, where ``float32`` Kahan summation differs from naive summation; and
  empty groups.
- 3,000 windows through ``labels.aggregation``: a target, its times on and
  around the window's edges, month ends and leap days, in nanoseconds and
  microseconds, and windows of every unit.
- Every relational question of the PQL corpus that resolves, and 1,500
  generated ones, each over two generated batches, through
  ``labels_execute``: the labels and the mask, a ``float32`` label compared
  by its bits (NaN as one NaN).

Outputs compare as the fixtures do (by value and JSON type, every NaN
written ``{"$float": "nan"}``), errors by class name and text; where today's
executor raises, ``labels_execute`` must raise.
"""

from __future__ import annotations

import collections
import json
import random
import warnings

from _labels import (
    LABELS_SEED,
    NUM_GROUPS,
    aggregation_inputs,
    aggregation_type_inputs,
    corpus_cases,
    execute_problem,
    generated_questions,
    groups_of,
    python_execute,
    python_op,
    resolve,
    rust_execute,
    rust_op,
    same_answer,
    sample_data,
    shop_schema,
)
from _sampling import report, short


def test_the_groups_reach_every_kind() -> None:
    inputs = aggregation_type_inputs()
    assert groups_of(inputs) >= NUM_GROUPS
    kinds = collections.Counter(i['feat']['dtype'] for i in inputs)
    for dtype in (
        'float32',
        'float64',
        'int64',
        'Int64',
        'Float64',
        'datetime64[ns]',
    ):
        assert kinds[dtype] >= 100, f'{kinds}'
    ops = collections.Counter(i['op'] for i in inputs)
    assert all(
        ops[op] >= 100
        for op in ('SUM', 'AVG', 'MIN', 'MAX', 'COUNT', 'LIST_DISTINCT')
    )
    assert (
        sum(
            len(i['batch']) >= 1000
            for i in inputs
            if i['feat']['dtype'] == 'float32'
        )
        >= 30
    )


def _compare(operation: str, inputs: list[dict]) -> None:
    failures = []
    with warnings.catch_warnings():
        # pandas warns of the overflowing casts the generated values make.
        warnings.simplefilter('ignore')
        for index, given in enumerate(inputs):
            want = python_op(operation, given)
            got = rust_op(operation, given)
            problem = same_answer(want, got)
            if problem:
                failures.append(
                    f'{operation} #{index} {short(given, 200)}: {problem}'
                )
    report(failures, len(inputs), f'{operation} inputs')


def test_aggregation_types_match_pandas() -> None:
    _compare('labels.aggregation_type', aggregation_type_inputs())


def test_aggregations_over_windows_match_pandas() -> None:
    _compare('labels.aggregation', aggregation_inputs())


def _questions() -> list[tuple[str, dict, object]]:
    shop = shop_schema()
    asked = [(c['query'], c['schema']) for c in corpus_cases()]
    asked += [(q, shop) for q in generated_questions(1_500)]
    out = []
    for query, schema in asked:
        resolved = resolve(query, schema)
        if resolved is not None:
            out.append((query, schema, resolved))
    return out


def test_questions_label_as_today() -> None:
    from predictive_query.schema import json_serde

    questions = _questions()
    assert len(questions) >= 1_200, f'only {len(questions)} questions resolve'
    rng = random.Random(LABELS_SEED + 3)
    failures = []
    labelled = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for query, schema, resolved in questions:
            tree = json.dumps(json_serde.to_json_dict(resolved))
            for _ in range(2):
                data = sample_data(rng, schema, resolved.entity_table)
                want = python_execute(resolved, data)
                got = rust_execute(tree, data)
                labelled += 'error' not in want
                problem = execute_problem(want, got)
                if problem:
                    failures.append(f'{query}: {problem}')
    assert labelled >= 2_000, f'only {labelled} batches were labelled today'
    report(failures, 2 * len(questions), 'labelled batches')
