# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time
from dataclasses import FrozenInstanceError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import pytest

from sdm_client import (
    ExplanationSummarizer,
    ExplanationView,
    InferenceConfig,
    KumoRelational,
    PredictiveQuery,
    RelationalTask,
    Schema,
    StructuredDataError,
)


@pytest.fixture
def summary_server():
    state = {
        'calls': [],
        'status': 200,
        'body': b'{"choices":[{"message":{"content":" A summary. "}}]}',
        'delay': 0,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state['calls'].append(
                (
                    self.path,
                    dict(self.headers),
                    json.loads(
                        self.rfile.read(int(self.headers['Content-Length']))
                    ),
                )
            )
            time.sleep(state['delay'])
            self.send_response(state['status'])
            self.end_headers()
            try:
                self.wfile.write(state['body'])
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}/v1/chat/completions', state
    server.shutdown()
    server.server_close()
    thread.join()


def test_summary_request_is_explicit_preserves_prose_and_freezes_data(
    summary_server, monkeypatch
):
    url, state = summary_server
    monkeypatch.setenv('OPENAI_API_KEY', 'ambient-not-used')
    raw = {
        'format': 'kumo_rfm_v2_1',
        'details': {'cohorts': [{'count': 3}], 'subgraphs': []},
    }
    view = ExplanationView(raw)
    summarizer = ExplanationSummarizer(
        url, 'synthetic-model', api_key='explicit-key'
    )
    assert not state['calls']
    assert view.summary is None
    assert (
        summarizer.summarize(
            ExplanationView({'details': {'summary': 'Original'}}), '', None
        )
        == 'Original'
    )
    assert not state['calls']
    assert (
        summarizer.summarize(view, 'PREDICT x', {'prediction': 2})
        == 'A summary.'
    )
    path, headers, body = state['calls'][0]
    assert path == '/v1/chat/completions'
    assert headers['Authorization'] == 'Bearer explicit-key'
    assert body['model'] == 'synthetic-model'
    assert 'do not claim causality' in body['messages'][0]['content']
    assert json.loads(body['messages'][1]['content']) == {
        'query': 'PREDICT x',
        'prediction': {'prediction': 2},
        'explanation': raw,
    }
    assert view.summary is None
    assert (
        ExplanationSummarizer(url, 'synthetic-model').summarize(view, '', None)
        == 'A summary.'
    )
    assert 'Authorization' not in state['calls'][1][1]


@pytest.mark.parametrize(
    'body',
    [
        b'not json',
        b'{}',
        b'{"choices":[]}',
        b'{"choices":[{"message":{"content":12}}]}',
        b'{"choices":[{"message":{"content":" "}}]}',
        b'null',
    ],
)
def test_malformed_summary_is_error(summary_server, body):
    url, state = summary_server
    state['body'] = body
    with pytest.raises(StructuredDataError) as error:
        ExplanationSummarizer(url, 'model').summarize(
            ExplanationView({}), '', None
        )
    assert error.value.code == 'TRANSPORT_ERROR'
    assert len(state['calls']) == 1


def test_summary_status_timeout_and_no_secret_leak(summary_server):
    url, state = summary_server
    summarizer = ExplanationSummarizer(
        url, 'model', api_key='secret', timeout=0.02
    )
    for status, delay in [(500, 0), (302, 0), (200, 0.1)]:
        state.update(status=status, delay=delay)
        with pytest.raises(StructuredDataError) as error:
            summarizer.summarize(ExplanationView({}), '', None)
        assert error.value.code == 'TRANSPORT_ERROR'
        assert 'secret' not in str(error.value)
    assert len(state['calls']) == 3


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(endpoint='ftp://example.com', model='x'),
        dict(endpoint='http://user:secret@example.com', model='x'),
        dict(endpoint='http://example.com', model=''),
        dict(endpoint='http://example.com', model='x', timeout=0),
        dict(endpoint='http://example.com', model='x', api_key='a\nb'),
    ],
)
def test_summary_bad_configuration(kwargs):
    with pytest.raises(StructuredDataError, match='INVALID_REQUEST'):
        ExplanationSummarizer(**kwargs)


def fixture_schema():
    schema = Schema.from_data(
        {
            'users': pd.DataFrame(
                {'id': list(range(8)), 'score': [1.5 + i for i in range(8)]}
            )
        }
    )
    schema['users'].primary_key = 'id'
    schema['users']['score'].stype = 'numerical'
    return schema


@pytest.mark.parametrize(
    'config',
    [
        InferenceConfig(),
        InferenceConfig(
            num_estimators=2,
            column_shuffle=True,
            category_shuffle=True,
            hop_shuffle=True,
            target_transforms=('clip', None),
            output_type='mean',
        ),
    ],
)
def test_typed_mapping_equivalence_pql_and_examples(config):
    schema = fixture_schema()
    query = PredictiveQuery('PREDICT users.score FOR EACH users.id', schema)
    typed = query.materialize(
        indices=[6], model=KumoRelational(), inference_config=config
    )
    mapping = query.materialize(
        indices=[6], model=KumoRelational(), inference_config=config.to_dict()
    )
    assert typed.payloads == mapping.payloads
    context = pd.DataFrame({'ENTITY': [0, 1, 2], 'TARGET': [1.5, 2.5, 3.5]})
    predict = pd.DataFrame({'ENTITY': [6]})
    task = RelationalTask('regression', 'users')
    typed = schema.materialize_examples(
        task, context, predict, inference=config
    )
    mapping = schema.materialize_examples(
        task, context, predict, inference=config.to_dict()
    )
    assert typed.payloads == mapping.payloads


@pytest.mark.parametrize(
    'config',
    [
        InferenceConfig(num_estimators=0),
        InferenceConfig(class_shuffle=True),
        InferenceConfig(output_type='invalid'),
        InferenceConfig(target_transforms=('invalid',)),
    ],
)
def test_typed_errors_are_rust_mapping_errors(config):
    query = PredictiveQuery(
        'PREDICT users.score FOR EACH users.id', fixture_schema()
    )
    errors = []
    for options in [config, config.to_dict()]:
        with pytest.raises(StructuredDataError) as error:
            query.materialize(
                indices=[6], model=KumoRelational(), inference_config=options
            )
        errors.append((error.value.code, error.value.details))
    assert errors[0] == errors[1]


def test_config_is_immutable_and_omits_defaults():
    assert InferenceConfig().to_dict() == {}
    transforms = ['clip', None]
    config = InferenceConfig(target_transforms=transforms)
    transforms.clear()
    mapped = config.to_dict()
    mapped['target_transforms'].clear()
    assert config.target_transforms == ('clip', None)
    with pytest.raises(FrozenInstanceError):
        config.output_type = 'mean'


def test_typed_classification_uses_existing_rust_validation():
    schema = fixture_schema()
    config = InferenceConfig(class_shuffle=True)
    task = RelationalTask('binary_classification', 'users')
    context = pd.DataFrame({'ENTITY': [0, 1, 2], 'TARGET': [True, False, True]})
    predict = pd.DataFrame({'ENTITY': [6]})
    typed = schema.materialize_examples(
        task, context, predict, inference=config
    )
    mapped = schema.materialize_examples(
        task, context, predict, inference=config.to_dict()
    )
    assert typed.payloads == mapped.payloads
    assert (
        typed.payloads[0]['inference']['inference_config']['class_shuffle']
        is True
    )


def test_cyclic_summary_input_is_refused_before_http(summary_server):
    url, state = summary_server
    prediction = []
    prediction.append(prediction)
    with pytest.raises(StructuredDataError) as error:
        ExplanationSummarizer(url, 'model').summarize(
            ExplanationView({}), '', prediction
        )
    assert error.value.code == 'INVALID_REQUEST'
    assert state['calls'] == []


def test_each_typed_setting_preserves_its_explicit_value():
    config = InferenceConfig(
        num_estimators=3,
        column_shuffle=False,
        category_shuffle=True,
        hop_shuffle=False,
        class_shuffle=True,
        target_transforms=('clip', None),
        output_type='mean',
    )
    expected = {
        'num_estimators': 3,
        'column_shuffle': False,
        'category_shuffle': True,
        'hop_shuffle': False,
        'class_shuffle': True,
        'target_transforms': ['clip', None],
        'output_type': 'mean',
    }
    assert config.to_dict() == expected
    for field, value in expected.items():
        assert getattr(config, field) == (
            tuple(value) if field == 'target_transforms' else value
        )
