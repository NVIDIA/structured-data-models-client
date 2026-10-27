# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public native Text profiles and their reached warning boundaries."""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from sdm_client import KumoRelational, SDMClient, StructuredDataError

from test_t44_text_frames import (
    Scripted,
    core as core,
    dt3_no_time_body,
    p3e2_warning_codes,
    question,
    shop,
    text,
)

HOPS = "Received custom 'num_neighbors' option; ignoring custom 'num_hops=1' option"


def reference(family):
    cell = 'pandas2' if int(pd.__version__.split('.')[0]) == 2 else 'pandas3'
    return json.loads(
        Path(__file__)
        .with_name(f't44_native_text_{family}_{cell}_reference.json')
        .read_text()
    )['records']


def expected_record(record, selected):
    expected = copy.deepcopy(record)
    if selected:
        for request in expected.get('requests', []):
            for body in request['payloads']:
                dt3_no_time_body(body)
        for body in expected.get('sent', []):
            dt3_no_time_body(body)
        expected['warnings'] = p3e2_warning_codes(expected['warnings'])
    return expected


def warning_values(caught):
    return [
        {'type': type(item.message).__name__, 'message': str(item.message)}
        for item in caught
    ]


def send(request, scripted, expected, selected):
    assert type(request.payloads) is tuple
    assert copy.deepcopy(list(request.payloads)) == expected['payloads']
    if selected:
        assert request._core is not None
    with SDMClient('http://localhost:1', max_retries=0) as client:
        client._transport.close()
        client._transport = scripted
        result = client.predict(request, model=KumoRelational()).to_pandas()
    assert json.loads(result.to_json(orient='split')) == expected['result']


PROFILE_CASES = [
    f'{source}-target-{target}'
    for source in ('python', 'pyarrow', 'utf8')
    for target in ('python', 'pyarrow')
] + ['inference-control', 'late-hops', 'late-between', 'late-context']


@pytest.mark.parametrize('case', PROFILE_CASES)
def test_effective_target_profile_keeps_full_public_bodies(core, case):
    records = reference('profile')
    major = int(pd.__version__.split('.')[0])
    if case == 'inference-control':
        case = 'future-infer-true' if major == 2 else 'future-infer-false'
    observed = records[case]
    expected = expected_record(observed, core)
    options = observed['requested_options']
    source = case.split('-target-')[0] if '-target-' in case else 'python'
    frames = shop(source)
    frames['users'].loc[0, 'note'] = 'ΟΣ ΟΣΑ AΣ́ AΣ1 AΣ-A'
    frames['users'].loc[6, 'note'] = 'ΟΣ İ AΣ́ ΟΣΑ'
    frames['users'].loc[7, 'note'] = 'é' * 298 + 'İΣx'
    scripted = Scripted('age')
    ids = [body['metadata']['request_id'] for body in expected['sent']]
    mutation = options['mutation']
    switched = []
    with (
        warnings.catch_warnings(record=True) as caught,
        pd.option_context(
            'mode.string_storage',
            options['storage'],
            'future.infer_string',
            options['infer_string'],
        ),
        patch('sdm_client.adapters.relational._request_id', side_effect=ids),
    ):
        warnings.simplefilter('always')
        original_show = warnings.showwarning

        def on_warning(
            message, category, filename, lineno, file=None, line=None
        ):
            value = str(message)
            hops = value == (f'[HOPS_IGNORED] {HOPS}' if core else HOPS)
            context = (
                'rows sampled as context are the rows being scored' in value
            )
            if not switched and (
                (mutation == 'hops' and hops)
                or (mutation == 'context' and context)
            ):
                pd.options.mode.string_storage = 'python'
                switched.append(value)
            original_show(
                message, category, filename, lineno, file=file, line=line
            )

        with patch.object(warnings, 'showwarning', on_warning):
            asked = question(frames, prose=True)
            for number, wanted in enumerate(expected['requests']):
                if number and mutation == 'between':
                    pd.options.mode.string_storage = 'python'
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
    if mutation in ('hops', 'context'):
        assert len(switched) == 1


@pytest.mark.parametrize(
    'case',
    [
        'graph-warning-indices',
        'hops-count-config',
        'hops-warning-error-retry',
        'batched',
        'sampling-error-before-normalize',
    ],
)
def test_native_warning_and_sampling_stages_keep_arguments_and_cached_rows(
    core, case
):
    expected = expected_record(reference('boundaries')[case], core)
    frames = shop('python')
    frames['users'].loc[6, 'note'] = 'ΟΣ İ'
    frames['users'].loc[7, 'note'] = 'AΣ́'
    indices, neighbors, config = [7, 8], [3], {'num_estimators': 1}
    if case == 'sampling-error-before-normalize':
        neighbors = [3.5]
    scripted = Scripted('age')
    ids = [body['metadata']['request_id'] for body in expected['sent']]
    switched = []
    with (
        warnings.catch_warnings(record=True) as caught,
        pd.option_context(
            'mode.string_storage',
            'pyarrow',
            'future.infer_string',
            int(pd.__version__.split('.')[0]) >= 3,
        ),
        patch('sdm_client.adapters.relational._request_id', side_effect=ids),
    ):
        warnings.simplefilter('always')
        original_show = warnings.showwarning

        def on_warning(
            message, category, filename, lineno, file=None, line=None
        ):
            value = str(message)
            if not switched:
                if case == 'graph-warning-indices' and value.startswith(
                    'the graph was built without'
                ):
                    indices[:] = [5, 6]
                    pd.options.mode.string_storage = 'python'
                    switched.append(value)
                elif case == 'hops-count-config' and value == (
                    f'[HOPS_IGNORED] {HOPS}' if core else HOPS
                ):
                    neighbors[:] = [1]
                    config['num_estimators'] = 2
                    pd.options.mode.string_storage = 'python'
                    switched.append(value)
            original_show(
                message, category, filename, lineno, file=file, line=line
            )

        with patch.object(warnings, 'showwarning', on_warning):
            asked = question(frames, prose=True)
            if case == 'graph-warning-indices':
                frames['orders'].loc[1, 'order_id'] = 0

            def gather():
                return asked.materialize(
                    indices=indices,
                    num_context_samples=3,
                    num_neighbors=neighbors,
                    num_hops=1,
                    random_seed=7,
                    inference_config=config,
                    **({'rows_per_call': 1} if case == 'batched' else {}),
                )

            if case == 'hops-warning-error-retry':
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'error',
                        message=r'(?:\[HOPS_IGNORED\] )?Received custom',
                        category=UserWarning,
                    )
                    with pytest.raises(StructuredDataError) as failure:
                        gather()
                wanted = expected['failures'][0]
                message = wanted['message']
                if core:
                    # P3E-2 adds only the registered core notice prefix,
                    # including when that notice is raised inside the
                    # native engine-error presentation context.
                    suffix = f'UserWarning: {HOPS}'
                    assert message.endswith(suffix)
                    message = message[: -len(suffix)] + (
                        f'UserWarning: [HOPS_IGNORED] {HOPS}'
                    )
                assert type(failure.value).__name__ == wanted['type']
                assert failure.value.code == wanted['code']
                assert str(failure.value) == message
                frames['orders']['note'] = text(
                    ['changed-source'] * 64, 'python'
                )
                pd.options.mode.string_storage = 'python'
            if 'error' in expected:
                with pytest.raises(StructuredDataError) as failure:
                    gather()
                assert {
                    'type': type(failure.value).__name__,
                    'code': failure.value.code,
                    'message': str(failure.value),
                } == expected['error']
            else:
                request = gather()
                send(request, scripted, expected['requests'][0], core)
    assert scripted.calls == expected['sent']
    assert warning_values(caught) == expected['warnings']
    assert {
        'indices': indices,
        'neighbors': neighbors,
        'config': config,
    } == expected['final_arguments']
