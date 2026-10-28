# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public warning sites retain presentation and their native error scope."""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pandas as pd
import pytest
from test_t44_native_text_profiles import (
    HOPS,
    expected_record,
    reference,
    send,
    warning_values,
)
from test_t44_text_frames import Scripted, question, shop, text
from test_t44_text_frames import core as core

from sdm_client import StructuredDataError
from sdm_client._rust import codec, errors


def test_relational_preflight_keeps_the_frozen_context_warning_presentation(
    core,
):
    recorded = reference('profile')['python-target-python']
    expected = expected_record(recorded, core)
    frames = shop('python')
    frames['users'].loc[0, 'note'] = 'ΟΣ ΟΣΑ AΣ́ AΣ1 AΣ-A'
    frames['users'].loc[6, 'note'] = 'ΟΣ İ AΣ́ ΟΣΑ'
    frames['users'].loc[7, 'note'] = 'é' * 298 + 'İΣx'
    scripted = Scripted('age')
    ids = [body['metadata']['request_id'] for body in expected['sent']]
    options = recorded['requested_options']
    with (
        pd.option_context(
            'mode.string_storage',
            options['storage'],
            'future.infer_string',
            options['infer_string'],
        ),
        warnings.catch_warnings(record=True) as caught,
        patch('sdm_client.adapters.relational._request_id', side_effect=ids),
    ):
        warnings.simplefilter('always')
        asked = question(frames, prose=True)
        for wanted in expected['requests']:
            request = asked.materialize(
                indices=[7, 8],
                num_context_samples=3,
                num_neighbors=[3],
                num_hops=1,
                random_seed=7,
            )
            send(request, scripted, wanted, core)
    assert scripted.calls == expected['sent']
    assert warning_values(caught) == expected['warnings']
    context = [
        item
        for item in warning_values(caught)
        if 'every context row' in item['message']
    ]
    assert context
    assert all(
        item['message'].startswith('[CONTEXT_CONSTANT] ') == core
        for item in context
    )


@pytest.mark.parametrize(
    'kind',
    [
        UserWarning,
        ValueError,
        TypeError,
        RuntimeError,
        StructuredDataError,
        KeyboardInterrupt,
    ],
)
def test_hops_handler_errors_keep_native_identity_chain_and_cached_graph(
    core, kind
):
    frames = shop('python')
    prior = KeyError('native prior context')
    cause = LookupError('native explicit cause')
    native = (
        kind('native warning handler', code='INVALID_CONFIGURATION')
        if kind is StructuredDataError
        else kind('native warning handler')
    )
    native.__context__ = prior
    native.__cause__ = cause
    original_show = warnings.showwarning
    reached = []

    def on_warning(message, category, filename, lineno, file=None, line=None):
        if 'Received custom' in str(message):
            reached.append(message)
            raise native
        original_show(message, category, filename, lineno, file=file, line=line)

    with (
        pd.option_context(
            'future.infer_string', False, 'mode.string_storage', 'python'
        ),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter('always')
        asked = question(frames, prose=True)
        with patch.object(warnings, 'showwarning', on_warning):
            expected_kind = (
                kind
                if kind in (StructuredDataError, KeyboardInterrupt)
                else StructuredDataError
            )
            with pytest.raises(expected_kind) as raised:
                asked.materialize(
                    indices=[7, 8],
                    num_context_samples=3,
                    num_neighbors=[3],
                    num_hops=1,
                    random_seed=7,
                )
        assert len(reached) == 1
        if kind in (StructuredDataError, KeyboardInterrupt):
            assert raised.value is native
        else:
            assert raised.value.__cause__ is native
            assert raised.value.code == (
                'INVALID_REQUEST'
                if kind in (ValueError, TypeError)
                else 'INTERNAL_ERROR'
            )
        assert native.__context__ is prior
        assert native.__cause__ is cause
        deepest = native.__traceback__
        while deepest.tb_next is not None:
            deepest = deepest.tb_next
        assert deepest.tb_frame.f_code.co_name == 'on_warning'
        owner = None
        if core:
            owner = asked._relational_materializer._core_sampler_state['held'][
                'frame_checkpoint'
            ]['owner']
        # The warning happens after graph creation but before sampling.
        # Retry must retain that graph rather than recapture the changed source.
        frames['orders']['note'] = text(['changed-source'] * 64, 'python')
        request = asked.materialize(
            indices=[7, 8],
            num_context_samples=3,
            num_neighbors=[3],
            num_hops=1,
            random_seed=7,
        )
        assert 'changed-source' not in repr(request.payloads)
        if core:
            after = asked._relational_materializer._core_sampler_state['held'][
                'frame_checkpoint'
            ]['owner']
            assert owner.same_owner(after)


def test_hops_warning_filter_error_keeps_its_native_cause(core):
    with pd.option_context(
        'future.infer_string', False, 'mode.string_storage', 'python'
    ):
        asked = question(shop('python'), prose=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'error',
                message=r'(?:\[HOPS_IGNORED\] )?Received custom',
                category=UserWarning,
            )
            with pytest.raises(StructuredDataError) as raised:
                asked.materialize(
                    indices=[7, 8],
                    num_context_samples=3,
                    num_neighbors=[3],
                    num_hops=1,
                    random_seed=7,
                )
    native = raised.value.__cause__
    assert isinstance(native, UserWarning)
    assert str(native) == (f'[HOPS_IGNORED] {HOPS}' if core else HOPS)
    assert raised.value.code == 'INTERNAL_ERROR'
    assert raised.value.message.endswith(f'UserWarning: {native}')


def test_hops_handler_uses_the_reached_graph_summary_and_actual_batch_size(
    core,
):
    native = ValueError(
        'size exceeds the 30MB limit; the number of entities predicted at once. '
        'Please reduce either setting.'
    )
    frames = shop('python')
    original_show = warnings.showwarning

    def fail(message, category, filename, lineno, file=None, line=None):
        if 'Received custom' in str(message):
            raise native
        original_show(message, category, filename, lineno, file=file, line=line)

    with (
        pd.option_context(
            'future.infer_string', False, 'mode.string_storage', 'python'
        ),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter('always')
        asked = question(frames, prose=True)
        frames['orders'].loc[1, 'order_id'] = 0
        with patch.object(warnings, 'showwarning', fail):
            with pytest.raises(StructuredDataError) as raised:
                asked.materialize(
                    indices=[7, 8],
                    num_context_samples=3,
                    num_neighbors=[3],
                    num_hops=1,
                    rows_per_call=99,
                    random_seed=7,
                )
    assert raised.value.__cause__ is native
    assert raised.value.code == 'INVALID_REQUEST'
    assert raised.value.details['batch_too_large_to_send'] is True
    assert 'below the 2 this one holds' in raised.value.message
    assert raised.value.message.endswith(
        ' [orders: 1 of 64 rows dropped (1 with a repeated primary key)]'
    )


@pytest.mark.parametrize('presentation', ['coded', 'frame_graph'])
def test_context_cap_and_sql_hops_warning_handlers_remain_unguarded(
    presentation,
):
    # CONTEXT_CAPPED is outside the engine guard even in the same task batch
    # as HOPS_IGNORED. SQL HOPS likewise retains its separate existing scope.
    for sentence, code, fields in (
        (
            'context_capped',
            'CONTEXT_CAPPED',
            {
                'cap': codec.integer('u64', 100),
                'most': codec.integer('u64', 1000),
            },
        ),
        ('hops_ignored', 'HOPS_IGNORED', {'num_hops': {'t': 'utf8', 'v': '1'}}),
    ):
        native = ValueError('unguarded native warning handler')

        def fail(*args, _native=native, **kwargs):
            raise _native

        control = codec.encode(
            {
                'message': 'warnings',
                'warnings': [
                    {
                        'code': code,
                        'sentence': sentence,
                        'message': 'source notice',
                        'fields': fields,
                    }
                ],
            }
        )
        with (
            warnings.catch_warnings(),
            patch.object(warnings, 'showwarning', fail),
        ):
            warnings.simplefilter('always')
            with pytest.raises(ValueError) as raised:
                errors.warn_from_core(control, {}, presentation=presentation)
        assert raised.value is native
