# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete public twins of the frozen native composite-history boundary.

These compare complete original/core observations, including payloads and
causes. The existing immutable full LINK fixture remains an independent oracle.
"""

from __future__ import annotations

import copy
import json
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from t38_forced_routes_record import bare
from t45_frame_routes import FRAME_ROUTES
from test_t45_frames import Scripted, configure

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Relationship,
    Schema,
    SDMClient,
)
from sdm_client.materializers.relational import RelationalMaterializer

CASES = (
    'ordinary',
    'reversed',
    'three',
    'escaped',
    'empty-text',
    'unicode',
    'surrogate',
    'signed',
    'uint64',
    'duplicate',
    'duplicate-null-time',
    'all-null-time',
    'empty',
    'null-first',
    'null-later',
    'float',
    'float-logical-edit',
    'long-name',
    'prefix',
    'collision-int',
    'collision-int-id',
    'collision-text',
    'collision-text-id',
    'collision-float',
    'key-moved',
    'later-table',
    'same-time',
    'same-time-context',
    'same-time-iterations',
    'relationships',
    'render-error',
    'constructor-error',
    'constructor-later-table',
    'constructor-reset',
    'constructor-columns-drop',
    'constructor-columns-add',
    'native-time-before-stype',
    'native-key-before-time',
    'constructor-sample-mutation',
    'fold-copy',
    'fold-copy-error',
    'fold-copy-mutation',
    'initial-copy-drop-feature',
    'initial-copy-drop-key',
    'initial-copy-add-feature',
    'native-link-dtype',
    'native-feature-dtype',
    'native-integer-key-logical-edit',
    'category-numeric-component',
    'category-numeric-feature-text',
    'replace',
    'cached',
    'warning-retry',
    'overlap-caller-dtype',
    'overlap-caller-marks',
    'overlap-caller-cached',
    'overlap-caller-reentry',
)


def _observe(label):
    trace = []
    active = []
    fold_started = False
    constructor_resets = {}
    fold_copies = 0
    native_sample = None
    column_mutation = False
    initial_copy_changed = False
    mutations = []
    schema = None

    class Rendered(str):
        def replace(self, old, new, *args):
            if active:
                trace.append(('replace', old, new))
            return Rendered(str.replace(self, old, new, *args))

    class Key(str):
        def __str__(self):
            trace.append(('native-str', bool(active)))
            if not active and stage == 'materialize':
                if label == 'constructor-error':
                    raise TypeError('native physical constructor failed')
                if label == 'constructor-later-table' and not mutations:
                    schema['users']['age'].stype = 'categorical'
                    mutations.append('constructor-users-age')
                if label == 'constructor-sample-mutation' and not mutations:
                    assert native_sample is not None
                    native_sample['later'] = ['changed'] * len(native_sample)
                    mutations.append('constructor-sample-later')
            if active:
                trace.append(('render', str.__str__(self)))
                if label == 'render-error':
                    raise TypeError('native composite render failed')
                if not mutations:
                    if label == 'key-moved':
                        schema['buys'].primary_key = 'buy_id'
                        mutations.append('key-moved')
                    elif label == 'later-table':
                        schema['users']['age'].stype = 'categorical'
                        mutations.append('users-age')
                    elif label.startswith('same-time'):
                        schema['buys'].time_column = None
                        mutations.append('no-time')
                    elif label == 'relationships':
                        schema.relationships.clear()
                        mutations.append('no-links')
            value = str.__str__(self)
            return Rendered(value) if label == 'replace' else value

    data, primary, times, links, query = configure('int64', 'composite-history')
    if label.startswith('overlap-caller-'):
        data['users']['age'] = data['users']['age'].astype(str)
        data['users']['unused_fk'] = np.arange(len(data['users'])) + 10000
        links.append(Relationship('users', 'unused_fk', 'items', 'item_id'))
    if label == 'native-time-before-stype':
        data['buys'].insert(1, 'early', np.arange(len(data['buys'])))
    if label == 'native-key-before-time':
        data['users']['ts'] = pd.date_range(
            '2020-01-01', periods=len(data['users'])
        )
        times['users'] = 'ts'
    if label == 'native-link-dtype':
        data['items']['item_id'] = data['items']['item_id'].astype(str)
        data['buys']['item_id'] = data['buys']['item_id'].astype(str)
    if label == 'native-feature-dtype':
        data['users']['age'] = data['users']['age'].astype(str)
    if label == 'constructor-sample-mutation':
        data['buys']['later'] = np.arange(120)
    if label == 'category-numeric-feature-text':
        data['users']['age'] = pd.Categorical(
            data['users']['age'].astype(float)
        )
    if label == 'category-numeric-component':
        data['buys']['part'] = pd.Categorical([0.25, 0.5] * 60)
    if label == 'reversed':
        primary['buys'].reverse()
    if label == 'three':
        data['buys']['other'] = ['x', 'y'] * 60
        primary['buys'].append('other')
    variants = {
        'escaped': ['a\x1fb', 'a\\u', '\\', '\x1f'] * 30,
        'empty-text': [''] * 120,
        'unicode': ['é', '東京', '😺'] * 40,
        'signed': [-(2**63), 2**63 - 1] * 60,
    }
    if label in variants:
        data['buys']['part'] = variants[label]
    if label == 'surrogate':
        data['buys']['part'] = pd.Series(
            ['\ud800', '\ud83d\ude3a', '😺'] * 40,
            dtype=pd.StringDtype(storage='python'),
        )
    if label == 'uint64':
        data['buys']['part'] = pd.Series(
            [2**63, 2**64 - 1] * 60, dtype='uint64'
        )
    if label in ('float', 'float-logical-edit'):
        data['buys']['part'] = 0.25
    if label == 'long-name':
        data['buys'] = data['buys'].rename(columns={'part': 'p' * 110})
        primary['buys'][1] = 'p' * 110
    derived = '__kumo_key_' + '_'.join(
        name.encode().hex() for name in primary['buys']
    )
    if label.startswith('collision-'):
        data['buys'][derived] = (
            ['key-' + str(i) for i in range(120)]
            if 'text' in label
            else np.arange(
                120, dtype='float64' if label.endswith('float') else 'int64'
            )
            + 9000
        )
    if label == 'prefix':
        data['buys']['__kumo_existing'] = np.arange(120)
    if label in (
        'key-moved',
        'later-table',
        'same-time',
        'same-time-context',
        'same-time-iterations',
        'relationships',
        'render-error',
        'constructor-error',
        'constructor-later-table',
        'constructor-sample-mutation',
        'native-time-before-stype',
        'fold-copy',
        'fold-copy-error',
        'fold-copy-mutation',
        'replace',
    ):
        data['buys']['part'] = pd.Series(
            [Key('part') for _ in range(120)], dtype=object
        )
    if label in ('later-table', 'constructor-later-table'):
        data = {name: data[name] for name in ('buys', 'users', 'items')}
    original_map = pd.Series.map
    original_reset = pd.DataFrame.reset_index
    original_copy = pd.DataFrame.copy

    def copied(frame, *args, **kwargs):
        nonlocal fold_copies, initial_copy_changed
        if (
            label.startswith('initial-copy-')
            and stage == 'materialize'
            and not initial_copy_changed
            and 'user_id' in frame
            and 'age' in frame
            and 'buy_id' not in frame
        ):
            initial_copy_changed = True
            result = original_copy(frame, *args, **kwargs)
            trace.append(('initial-copy', tuple(frame.columns)))
            if label.endswith('-drop-feature'):
                result.drop(columns=['age'], inplace=True)
            elif label.endswith('-drop-key'):
                result.drop(columns=['user_id'], inplace=True)
            else:
                result['copy_added'] = 7
            return result
        reached = (
            label.startswith('fold-copy')
            and stage == 'materialize'
            and not fold_started
            and 'buy_id' in frame
            and 'part' in frame
        )
        if reached:
            fold_copies += 1
            trace.append(('native-copy', fold_copies))
            if label == 'fold-copy-error' and fold_copies == 3:
                raise RuntimeError('native fold shallow copy failed')
        result = original_copy(frame, *args, **kwargs)
        if reached and label == 'fold-copy-mutation' and fold_copies == 3:
            result.loc[0, 'part'] = 'copy-changed'
        return result

    def reset(frame, *args, **kwargs):
        nonlocal native_sample, column_mutation
        if (
            label.startswith('constructor-columns-')
            and stage == 'materialize'
            and not column_mutation
            and 'user_id' in frame
            and 'age' in frame
            and 'buy_id' not in frame
        ):
            column_mutation = True
            trace.append(('constructor-columns', tuple(frame.columns)))
            if label.endswith('-drop'):
                frame.drop(columns=['age'], inplace=True)
            else:
                frame['reset_added'] = 7
        if (
            label == 'constructor-reset'
            and stage == 'materialize'
            and not fold_started
        ):
            columns = tuple(frame.columns)
            trace.append(('constructor-reset', columns))
            constructor_resets[columns] = constructor_resets.get(columns, 0) + 1
            if constructor_resets[columns] > 1:
                raise ValueError(
                    'a native constructor reset the same sample twice'
                )
        result = original_reset(frame, *args, **kwargs)
        if (
            label == 'constructor-sample-mutation'
            and stage == 'materialize'
            and not fold_started
            and 'buy_id' in frame
            and 'part' in frame
        ):
            native_sample = result
        return result

    def mapped(values, *args, **kwargs):
        nonlocal fold_started
        if values.name == 'buy_id' and stage == 'materialize':
            fold_started = True
        reached = values.name == 'part' and stage == 'materialize'
        if reached:
            active.append(True)
        try:
            return original_map(values, *args, **kwargs)
        finally:
            if reached:
                active.pop()

    original_showwarning = warnings.showwarning
    overlap_notices = 0

    def shown(message, *args, **kwargs):
        nonlocal overlap_notices
        if label.startswith(
            'overlap-caller-'
        ) and 'joins none of the rows given' in str(message):
            overlap_notices += 1
            reached_notice = overlap_notices
            if label == 'overlap-caller-reentry' and reached_notice == 1:
                gather()
            if (
                label == 'overlap-caller-reentry'
                and reached_notice == 1
                or label == 'overlap-caller-cached'
                and reached_notice == 2
                or label
                not in ('overlap-caller-cached', 'overlap-caller-reentry')
            ):
                schema['users']['age'].dtype = 'int64'
                if label == 'overlap-caller-marks':
                    schema.relationships.clear()
                mutations.append('overlap-caller-edit')
        return original_showwarning(message, *args, **kwargs)

    observed = {}
    script = Scripted()
    stage = 'schema'
    with (
        warnings.catch_warnings(record=True) as notices,
        pd.option_context('future.infer_string', False),
        patch.object(pd.Series, 'map', mapped),
        patch.object(pd.DataFrame, 'reset_index', reset),
        patch.object(pd.DataFrame, 'copy', copied),
        patch.object(warnings, 'showwarning', shown),
    ):
        warnings.simplefilter('always')
        try:
            schema = Schema.from_data(
                data,
                primary_keys=primary,
                time_columns=times,
                relationships=links,
            )
            if label.startswith('overlap-caller-'):
                schema['users']['age'].stype = 'text'
            if label == 'native-time-before-stype':
                data['buys']['early'] = ['changed'] * len(data['buys'])
                data['buys']['ts'] = np.arange(len(data['buys']))
            if label == 'native-key-before-time':
                data['users']['user_id'] = pd.date_range(
                    '2020-01-01', periods=len(data['users'])
                )
                data['users']['ts'] = np.arange(len(data['users']))
            if label == 'category-numeric-feature-text':
                schema['users']['age'].stype = 'text'
            if label == 'native-link-dtype':
                data['buys']['item_id'] = pd.to_numeric(
                    data['buys']['item_id']
                ).astype('int64')
            if label == 'native-feature-dtype':
                schema['users']['age'].dtype = 'int64'
                schema['users']['age'].stype = 'numerical'
            if label == 'native-integer-key-logical-edit':
                schema['buys']['part'].dtype = 'string'
            if label == 'float-logical-edit':
                schema['buys']['part'].dtype = 'string'
            if label.endswith('-id'):
                schema['buys'][derived].stype = 'ID'
            stage = 'query'
            question = PredictiveQuery(query, schema)
            if label in ('duplicate', 'duplicate-null-time', 'warning-retry'):
                data['buys'].loc[1, ['buy_id', 'part']] = (
                    data['buys'].loc[0, ['buy_id', 'part']].tolist()
                )
            if label == 'duplicate-null-time':
                data['buys'].loc[0, 'ts'] = pd.NaT
            if label == 'all-null-time':
                data['buys']['ts'] = pd.NaT
            if label == 'empty':
                data['buys'].drop(index=data['buys'].index, inplace=True)
            if label in ('null-first', 'null-later'):
                name = 'buy_id' if label == 'null-first' else 'part'
                data['buys'][name] = data['buys'][name].astype('Int64')
                data['buys'].loc[0, name] = pd.NA

            def gather():
                return question.materialize(
                    indices=[0, 1],
                    anchor_time=pd.Timestamp('2023-03-01'),
                    random_seed=7,
                    num_context_samples=3,
                    num_hops=1,
                    num_neighbors=[3, 3],
                    **(
                        {'context_anchor_time': 'entity'}
                        if label.endswith('-context')
                        else {'max_pq_iterations': 0}
                        if label.endswith('-iterations')
                        else {}
                    ),
                )

            stage = 'materialize'
            if label == 'warning-retry':
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            'error', message='the graph was built without'
                        )
                        gather()
                except Warning as error:
                    observed['warning_error'] = bare(str(error))
                data['buys']['part'] = pd.NA
            request = gather()
            observed['payloads'] = copy.deepcopy(request.payloads)
            observed['rows'] = (request.context_rows, request.predict_rows)
            if label == 'overlap-caller-cached':
                observed['again'] = copy.deepcopy(gather().payloads)
            if label == 'cached':
                data['buys']['part'] = pd.NA
                schema['users']['age'].stype = 'categorical'
                observed['again'] = copy.deepcopy(gather().payloads)
            stage = 'predict'
            with patch(
                'sdm_client.adapters.relational._request_id',
                return_value='composite-' + label,
            ):
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = script
                    answer = client.predict(
                        request, model=KumoRelational()
                    ).to_pandas()
            observed['result'] = json.loads(answer.to_json(orient='split'))
            observed['outcome'] = 'success'
        except Exception as error:
            chain = []
            while error is not None:
                message = bare(str(error))
                if label == 'uint64':
                    message = message.replace(
                        'value 9223372036854775808', 'a value'
                    )
                chain.append(
                    (
                        type(error).__name__,
                        getattr(error, 'code', None),
                        message,
                    )
                )
                error = error.__cause__ or (
                    None if error.__suppress_context__ else error.__context__
                )
            observed.update(outcome='error', stage=stage, chain=chain)
        observed['warnings'] = [
            (type(n.message).__name__, bare(str(n.message))) for n in notices
        ]
        if label.startswith('overlap-caller-') and schema is not None:
            observed['caller_marks'] = [
                (table_name, name, column.stype, column._stype_before_key)
                for table_name, table in schema.tables.items()
                for name, column in table.columns.items()
            ]
        observed['sent'] = script.calls
    observed['trace'] = trace
    observed['mutations'] = mutations
    return json.loads(json.dumps(observed))


@pytest.mark.parametrize('label', CASES)
def test_complete_native_composite_history_matches_original(monkeypatch, label):
    monkeypatch.setenv('SDM_CORE_ROUTES', '')
    expected = _observe(label)
    if label in (
        'ordinary',
        'collision-int-id',
        'collision-text-id',
        'replace',
        'constructor-reset',
        'constructor-columns-add',
        'fold-copy',
        'fold-copy-mutation',
        'initial-copy-drop-feature',
        'initial-copy-add-feature',
        'native-integer-key-logical-edit',
        'category-numeric-component',
        'cached',
        'warning-retry',
        'overlap-caller-cached',
        'overlap-caller-reentry',
    ):
        assert expected['outcome'] == 'success'
    if label == 'category-numeric-feature-text':
        if expected['outcome'] == 'error':
            message = (
                'Cannot setitem on a Categorical with a new category (), '
                'set the categories first'
            )
            assert expected['stage'] == 'materialize'
            assert expected['chain'] == [
                ['StructuredDataError', 'INVALID_REQUEST', message],
                ['TypeError', None, message],
            ]
            assert expected['sent'] == []
        else:
            assert expected['outcome'] == 'success'
    monkeypatch.setenv('SDM_CORE_ROUTES', FRAME_ROUTES)

    def old_engine_was_called(*args, **kwargs):
        raise AssertionError(
            'core composite materialization called the old engine'
        )

    monkeypatch.setattr(
        RelationalMaterializer, '_get_model', old_engine_was_called
    )
    assert _observe(label) == expected


def test_late_uint64_range_retains_the_active_caller_exception(monkeypatch):
    def observed_while_handling():
        try:
            raise LookupError('active caller context')
        except LookupError:
            return _observe('uint64')

    monkeypatch.setenv('SDM_CORE_ROUTES', '')
    expected = observed_while_handling()
    assert [item[0] for item in expected['chain']] == [
        'StructuredDataError',
        'ValueError',
        'ValueError',
        'LookupError',
    ]
    monkeypatch.setenv('SDM_CORE_ROUTES', FRAME_ROUTES)

    def old_engine_was_called(*args, **kwargs):
        raise AssertionError('core range conversion called the old engine')

    monkeypatch.setattr(
        RelationalMaterializer, '_get_model', old_engine_was_called
    )
    assert observed_while_handling() == expected
