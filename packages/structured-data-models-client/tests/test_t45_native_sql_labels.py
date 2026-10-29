# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete same-runtime SQL label twins of already reproduced public inputs.

The immutable SQL bootstrap remains the source of these tables.
Each route may complete or encounter its actual native conversion failure;
the complete observation must match, with only existing approved changes.
No endpoint, capture campaign or runtime-version expectation is involved.
"""

from __future__ import annotations

import copy
import dataclasses
import warnings
from unittest.mock import patch

import pandas as pd
import pytest
import yaml
from sdm_connectors import DuckDBSource
from test_t42_arrow_decimal_frames import _decimal_error_graph
from test_t45_sql_link import QUERY, ROOT, connection, route

from sdm_client import PredictiveQuery, Schema
from sdm_client.core.transport import Transport
from sdm_client.materializers.relational import RelationalMaterializer


def _input(case):
    if case == 'ordinary':
        return [], QUERY
    if case == 'string-rhs':
        expression = "'key-' || CAST(item_id AS VARCHAR)"
        return [
            f'ALTER TABLE {table} ALTER COLUMN item_id TYPE VARCHAR '
            f'USING {expression}'
            for table in ('items', 'buys')
        ], QUERY
    assert case in ('distinct', 'distinct-null')
    setup = [
        "INSERT INTO buys VALUES (999000, 0, 0, TIMESTAMP '2023-01-30'), "
        "(999001, 0, 1, TIMESTAMP '2023-02-01'), "
        "(999002, 0, 1, TIMESTAMP '2023-02-02'), "
        "(999004, 0, 3, TIMESTAMP '2023-03-01'), "
        "(999005, 0, 2, TIMESTAMP '2023-03-01 00:00:00.000001')",
    ]
    if case == 'distinct-null':
        setup.append(
            "INSERT INTO buys VALUES (999003, 0, NULL, TIMESTAMP '2023-02-03')"
        )
    return setup, (
        'PREDICT LIST_DISTINCT(buys.item_id WHERE buys.buy_id >= 999000, '
        '0, 30, days) FOR EACH users.user_id'
    )


def _source_rows(db):
    return {
        'items': db.execute('SELECT * FROM items ORDER BY item_id').fetchall(),
        'buys': db.execute('SELECT * FROM buys ORDER BY buy_id').fetchall(),
    }


def _observe(monkeypatch, case, core):
    setup, text = _input(case)
    stage = 'schema'
    warning_stages = []
    transports = []
    engine_calls = []
    observed = {}

    def no_transport(*args, **kwargs):
        transports.append(stage)
        raise AssertionError('materialization called the model transport')

    def no_engine(*args, **kwargs):
        engine_calls.append(stage)
        raise AssertionError('core materialization called the old engine')

    with monkeypatch.context() as selected, connection(setup) as db:
        route(selected, core)
        if core:
            selected.setattr(RelationalMaterializer, '_get_model', no_engine)
        before = _source_rows(db)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            original_showwarning = warnings.showwarning

            def shown(*args, **kwargs):
                warning_stages.append(stage)
                return original_showwarning(*args, **kwargs)

            with (
                patch.object(warnings, 'showwarning', shown),
                patch.object(Transport, 'predict', no_transport),
                patch.object(Transport, 'predict_body', no_transport),
            ):
                try:
                    schema = Schema.from_source(DuckDBSource(db))
                    stage = 'materialize'
                    request = PredictiveQuery(text, schema).materialize(
                        anchor_time=pd.Timestamp('2023-03-01'), indices=[0, 1]
                    )
                    observed.update(
                        outcome='success',
                        request={
                            'payloads': copy.deepcopy(request.payloads),
                            'task': dataclasses.asdict(request.task),
                            'query': request.query,
                            'indices': copy.deepcopy(request.indices),
                            'inference': copy.deepcopy(request.inference),
                            'requested_inference': copy.deepcopy(
                                request.requested_inference
                            ),
                            'context_rows': request.context_rows,
                            'predict_rows': request.predict_rows,
                            'nothing_to_score': request.nothing_to_score,
                            'is_relational': request.is_relational,
                            'target_moves_with_time': request.target_moves_with_time,
                        },
                    )
                except Exception as error:
                    observed.update(
                        outcome='error',
                        stage=stage,
                        error=_decimal_error_graph(error),
                        details=copy.deepcopy(getattr(error, 'details', None)),
                    )
            assert len(notices) == len(warning_stages)
            observed['warnings'] = [
                {
                    'type': type(notice.message).__name__,
                    'category': (
                        notice.category.__module__
                        + '.'
                        + notice.category.__qualname__
                    ),
                    'message': str(notice.message),
                    'stage': position,
                }
                for notice, position in zip(
                    notices, warning_stages, strict=True
                )
            ]
        # A native failure must leave the caller's actual connection usable
        # and its source rows intact. These reads happen after the outcome.
        observed['source_after'] = _source_rows(db)
        assert observed['source_after'] == before
    assert transports == []
    assert engine_calls == []
    return observed


def _approved_old(old, case):
    register = yaml.safe_load(
        (ROOT / 'docs/design/rust-core/register.yaml').read_text()
    )
    entries = {entry['id']: entry for entry in register['entries']}
    for name in ('P3E-2', 'P1B-13'):
        assert (entries[name]['status'], entries[name]['approved_by']) == (
            'approved',
            'owner',
        )
    expected = copy.deepcopy(old)
    # The finite mappings keep native pandas warnings, their categories,
    # wording, order and reached stage unchanged.
    codes = {
        'Missing 7 indices in 4 tables for optimal database querying. For improving runtime, we strongly suggest to create indices for primary and foreign keys, e.g., automatically by instantiating Kumo Relational via `KumoRelational(graph, optimize=True)`.': 'MISSING_INDICES',
        "Context anchor timestamp is too early or aggregation time range is too large. To form proper input data, we would need data back to '2022-12-31 00:00:00', however, your data only contains data back to '2023-01-01 00:00:00'.": 'CONTEXT_BEFORE_DATA',
    }
    native_codes = {}
    for condition in ('users.age > 22', 'users.created_at IS NOT NULL'):
        native_codes[
            f"the entity filter '{condition}' selects the labelled examples the model learns from, "
            'and does not narrow the 2 rows named for scoring. Those are scored as given, whether '
            'or not they satisfy it. Pass only indices that satisfy the condition if that is what you meant.'
        ] = 'FILTER_CHOSE_EXAMPLES'
    for instant in ('2023-01-01 00:00:00', '2023-01-01 00:00:00+00:00'):
        native_codes[
            'Some time columns were inferred, or refused, on a guess about what a column holds:\n'
            '- users.created_at was not taken as the time column because every row holds '
            + instant
            + ". If it does say when a row became true, name it: time_columns={'users': 'created_at'}."
        ] = 'TIME_COLUMNS_GUESSED'
    for notice in expected['warnings']:
        native_code = native_codes.get(notice['message'])
        if native_code is not None:
            assert notice['category'] == 'sdm_client.core.warnings.SDMWarning'
            assert notice['type'] == 'SDMWarning'
            notice['message'] = f'[{native_code}] {notice["message"]}'
        code = codes.get(notice['message'])
        if code is not None:
            assert notice['category'] == 'builtins.UserWarning'
            notice['category'] = 'sdm_client._rust.errors.UserWarning'
            notice['message'] = f'[{code}] {notice["message"]}'
    if case == 'distinct-null' and expected['outcome'] == 'success':
        # Exactly the already asserted P1B-13 delta in test_t45_sql_link:
        # two shifted links and one identical phantom row, no other cells.
        [payload] = expected['request']['payloads']
        related = payload['predict']['related_tables']
        buys = related['buys']
        user_at = buys['columns'].index('user_id')
        item_at = buys['columns'].index('item_id')
        shifted = [row for row in buys['rows'] if row[user_at] == 1]
        assert len(shifted) == 2
        assert all(row[item_at] == 1 for row in shifted)
        for row in shifted:
            row[item_at] = 3
        items = related['items']
        key_at = items['columns'].index('item_id')
        phantom = [
            at
            for at, row in enumerate(items['rows'])
            if row[0] == 1 and row[key_at] == 1
        ]
        assert len(phantom) == 2
        assert items['rows'][phantom[0]] == items['rows'][phantom[1]]
        items['rows'].pop(phantom[1])
    return expected


@pytest.mark.parametrize(
    'case', ['ordinary', 'string-rhs', 'distinct', 'distinct-null']
)
def test_native_sql_labels_keep_the_complete_same_runtime_public_outcome(
    monkeypatch,
    case,
):
    old = _observe(monkeypatch, case, False)
    current = _observe(monkeypatch, case, True)
    if old['outcome'] == 'error':
        # A paired unrelated setup/validation error is not native parity.
        # These are the reached native failures already reproduced from
        # exactly these inputs; other profiles may complete the same call.
        assert old['stage'] == 'materialize'
        error = old['error']
        assert error['type'] == 'sdm_client.errors.StructuredDataError'
        assert error['code'] == 'INTERNAL_ERROR'
        assert error['cause']['type'] == 'pyarrow.lib.ArrowNotImplementedError'
        element = 'string' if case == 'string-rhs' else 'int64'
        target = 'utf8' if case == 'string-rhs' else 'int64'
        cast = 'cast_string' if case == 'string-rhs' else 'cast_int64'
        assert error['cause']['text'] == (
            f'Unsupported cast from list<item: {element}> to {target} '
            f'using function {cast}'
        )
    else:
        request = old['request']
        assert request['task']['kind'] == 'temporal_link_prediction'
        assert request['context_rows'] > 0
        assert request['predict_rows'] == 2
        assert request['nothing_to_score'] is False
    assert current == _approved_old(old, case)
