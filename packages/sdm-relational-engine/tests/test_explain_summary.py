# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
from sdm_relational_engine.rfm.explain import summary as explain_summary
from sdm_relational_engine.rfm.explain.summary import (
    SUMMARY_ERROR_MESSAGE,
    SUMMARY_NEEDS_EXTRA_MESSAGE,
    SUMMARY_NEEDS_MODEL_MESSAGE,
    SUMMARY_UNAVAILABLE_MESSAGE,
    SYSTEM_PROMPT,
    generate_summary,
)

_ENV_VARS = (
    'OPENAI_API_KEY',
    'KUMO_RELATIONAL_EXPLAIN_LLM_API_KEY',
    'KUMO_RELATIONAL_EXPLAIN_LLM_BASE_URL',
    'KUMO_RELATIONAL_EXPLAIN_LLM_MODEL',
    'KUMO_RELATIONAL_EXPLAIN_LLM_TIMEOUT',
)

try:
    import openai  # noqa: F401

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

requires_openai = pytest.mark.skipif(
    not _HAS_OPENAI,
    reason='openai (sdm_relational_engine[explain]) is not installed',
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: Any) -> None:
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class _FakeCompletions:
    def __init__(
        self,
        content: str | None = 'the summary',
        record: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._content = content
        self._record = record
        self._error = error

    def create(self, *, model: str, messages: list[dict[str, str]]) -> Any:
        if self._record is not None:
            self._record.update(model=model, messages=messages)
        if self._error is not None:
            raise self._error
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeClient:
    def __init__(self, **kwargs: Any) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(**kwargs))


def _system(record: dict[str, Any]) -> str:
    return record['messages'][0]['content']


def _user(record: dict[str, Any]) -> str:
    return record['messages'][1]['content']


def test_system_prompt_includes_both_sources() -> None:
    assert 'Explanation summary module' in SYSTEM_PROMPT
    assert 'Understanding the Global View: Cohorts' in SYSTEM_PROMPT
    assert 'Understanding the Local View: Subgraph' in SYSTEM_PROMPT


def test_generate_summary_passes_context_and_returns_text() -> None:
    record: dict[str, Any] = {}
    client = _FakeClient(content='Order frequency fell.', record=record)
    out = generate_summary(
        query='PREDICT ... FOR users.user_id=1',
        prediction=[{'ENTITY': 1, 'prediction': 0.5}],
        cohorts=[{'column_name': 'COUNT(*)'}],
        subgraphs=['SG0', 'SG1'],
        client=client,
        model='test-model',
    )
    assert out == 'Order frequency fell.'
    assert record['model'] == 'test-model'
    assert _system(record) == SYSTEM_PROMPT
    assert 'USER QUERY: PREDICT ... FOR users.user_id=1' in _user(record)
    assert "MODEL PREDICTION: [{'ENTITY': 1, 'prediction': 0.5}]" in _user(
        record
    )
    assert "COLUMN ANALYSIS: [{'column_name': 'COUNT(*)'}]" in _user(record)
    assert 'SUBGRAPH EXPLANATION: SG0' in _user(record)
    assert 'SG1' not in _user(record)


def test_missing_extra_message_when_openai_absent(monkeypatch: Any) -> None:
    monkeypatch.setitem(sys.modules, 'openai', None)
    monkeypatch.setenv('KUMO_RELATIONAL_EXPLAIN_LLM_API_KEY', 'k')
    assert (
        generate_summary('q', 'p', ['c'], ['s']) == SUMMARY_NEEDS_EXTRA_MESSAGE
    )


@requires_openai
def test_needs_key_message_when_no_key() -> None:
    assert (
        generate_summary('q', 'p', ['c'], ['s']) == SUMMARY_UNAVAILABLE_MESSAGE
    )


@requires_openai
def test_custom_endpoint_without_model_message() -> None:
    out = generate_summary(
        'q', 'p', ['c'], ['s'], base_url='https://ep.example/v1', api_key='k'
    )
    assert out == SUMMARY_NEEDS_MODEL_MESSAGE


def test_generate_summary_handles_api_error() -> None:
    client = _FakeClient(error=RuntimeError('boom'))
    out = generate_summary('q', 'p', ['c'], ['s'], client=client)
    assert out.startswith(SUMMARY_ERROR_MESSAGE)
    assert 'RuntimeError' in out


def test_generate_summary_timeout_tells_user_to_raise_timeout() -> None:
    class APITimeoutError(Exception):
        pass

    client = _FakeClient(error=APITimeoutError('timed out'))
    out = generate_summary('q', 'p', ['c'], ['s'], client=client, timeout=7)
    assert 'timed out after 7s' in out
    assert 'KUMO_RELATIONAL_EXPLAIN_LLM_TIMEOUT' in out
    assert '.cohorts' in out


def test_generate_summary_empty_content_is_error() -> None:
    client = _FakeClient(content=None)
    out = generate_summary('q', 'p', ['c'], ['s'], client=client)
    assert out == SUMMARY_ERROR_MESSAGE


def test_generate_summary_uses_env_model(monkeypatch: Any) -> None:
    monkeypatch.setenv('KUMO_RELATIONAL_EXPLAIN_LLM_MODEL', 'env-model')
    record: dict[str, Any] = {}
    client = _FakeClient(record=record)
    generate_summary('q', 'p', ['c'], ['s'], client=client)
    assert record['model'] == 'env-model'


def test_generate_summary_explicit_model_overrides_env(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv('KUMO_RELATIONAL_EXPLAIN_LLM_MODEL', 'env-model')
    record: dict[str, Any] = {}
    client = _FakeClient(record=record)
    generate_summary('q', 'p', ['c'], ['s'], client=client, model='explicit')
    assert record['model'] == 'explicit'


def test_generate_summary_empty_subgraphs() -> None:
    record: dict[str, Any] = {}
    client = _FakeClient(record=record)
    generate_summary('q', 'p', ['c'], [], client=client)
    assert 'SUBGRAPH EXPLANATION: ' in _user(record)


def test_make_client_none_without_key() -> None:
    assert explain_summary._make_client(None, None, 5.0) is None
    assert explain_summary._make_client('https://ep/v1', None, 5.0) is None


@requires_openai
def test_make_client_targets_custom_endpoint() -> None:
    client = explain_summary._make_client('https://ep.example/v1', 'k', 5.0)
    assert client is not None
    assert str(client.base_url).rstrip('/') == 'https://ep.example/v1'


def _capture_make_client(monkeypatch: Any) -> dict[str, Any]:
    call: dict[str, Any] = {}

    def fake_make_client(
        base_url: str | None, api_key: str | None, timeout: float
    ) -> Any:
        call.update(base_url=base_url, api_key=api_key, timeout=timeout)
        return _FakeClient(record=call)

    monkeypatch.setattr(explain_summary, '_make_client', fake_make_client)
    return call


@requires_openai
def test_uses_openai_env(monkeypatch: Any) -> None:
    monkeypatch.setenv(
        'KUMO_RELATIONAL_EXPLAIN_LLM_BASE_URL', 'https://openai.example/v1'
    )
    monkeypatch.setenv('KUMO_RELATIONAL_EXPLAIN_LLM_API_KEY', 'explain-key')
    monkeypatch.setenv('KUMO_RELATIONAL_EXPLAIN_LLM_MODEL', 'm')
    call = _capture_make_client(monkeypatch)
    generate_summary('q', 'p', ['c'], ['s'])
    assert call['base_url'] == 'https://openai.example/v1'
    assert call['api_key'] == 'explain-key'
    assert call['model'] == 'm'


@requires_openai
def test_an_ambient_openai_key_is_never_used(monkeypatch: Any) -> None:
    """A key exported for another tool must not send rows to a third party."""
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-key')
    monkeypatch.setenv('KUMO_RELATIONAL_EXPLAIN_LLM_API_KEY', 'explain-key')
    monkeypatch.setenv('KUMO_RELATIONAL_EXPLAIN_LLM_MODEL', 'm')
    call = _capture_make_client(monkeypatch)
    generate_summary('q', 'p', ['c'], ['s'])
    assert call['api_key'] == 'explain-key'


@requires_openai
def test_does_not_fall_back_to_the_openai_key(monkeypatch: Any) -> None:
    """With only OPENAI_API_KEY set, no request is made at all."""
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-key')
    call = _capture_make_client(monkeypatch)
    result = generate_summary('q', 'p', ['c'], ['s'])
    assert call == {}
    assert result == SUMMARY_UNAVAILABLE_MESSAGE


# The egress has to be discoverable from the docstrings a caller reads, and
# only the dedicated variable may enable it: an ambient OPENAI_API_KEY set for
# some other tool must not start sending rows to a third party.


def test_explain_config_docstring_discloses_the_egress() -> None:
    from sdm_relational_engine.rfm import ExplainConfig

    doc = ExplainConfig.__doc__ or ''
    assert 'api.openai.com' in doc
    assert 'cell values' in doc
    assert 'KUMO_RELATIONAL_EXPLAIN_LLM_API_KEY' in doc
    assert 'skip_summary=True' in doc


def test_generate_summary_docstring_discloses_the_egress() -> None:
    doc = generate_summary.__doc__ or ''
    assert 'api.openai.com' in doc
    assert 'cell values' in doc
    assert 'skip_summary=True' in doc


@requires_openai
def test_explicit_args_override_env(monkeypatch: Any) -> None:
    monkeypatch.setenv(
        'KUMO_RELATIONAL_EXPLAIN_LLM_BASE_URL', 'https://env.example/v1'
    )
    monkeypatch.setenv('KUMO_RELATIONAL_EXPLAIN_LLM_API_KEY', 'env-key')
    call = _capture_make_client(monkeypatch)
    generate_summary(
        'q',
        'p',
        ['c'],
        ['s'],
        base_url='https://explicit/v1',
        api_key='ekey',
        model='em',
        timeout=12.5,
    )
    assert call['base_url'] == 'https://explicit/v1'
    assert call['api_key'] == 'ekey'
    assert call['model'] == 'em'
    assert call['timeout'] == 12.5


@requires_openai
def test_timeout_defaults_to_20_and_reads_env(monkeypatch: Any) -> None:
    monkeypatch.setenv('KUMO_RELATIONAL_EXPLAIN_LLM_API_KEY', 'k')
    call = _capture_make_client(monkeypatch)
    generate_summary('q', 'p', ['c'], ['s'])
    assert call['timeout'] == 20.0

    monkeypatch.setenv('KUMO_RELATIONAL_EXPLAIN_LLM_TIMEOUT', '5')
    call2 = _capture_make_client(monkeypatch)
    generate_summary('q', 'p', ['c'], ['s'])
    assert call2['timeout'] == 5.0


def test_env_float_invalid_falls_back(monkeypatch: Any) -> None:
    monkeypatch.setenv('KUMO_RELATIONAL_EXPLAIN_LLM_TIMEOUT', 'not-a-number')
    assert (
        explain_summary._env_float('KUMO_RELATIONAL_EXPLAIN_LLM_TIMEOUT', 20.0)
        == 20.0
    )


def test_explanation_accessors_ignore_malformed_details() -> None:
    import pandas as pd
    from sdm_relational_engine.rfm import Explanation

    empty = pd.DataFrame()
    for details in (
        None,
        {'format': 'natural_language_summary', 'summary': 'x'},
        {'format': 'kumo_rfm_v2_1', 'details': 'oops'},
        {
            'format': 'kumo_rfm_v2_1',
            'details': {'cohorts': 'x', 'subgraphs': {'a': 1}},
        },
    ):
        e = Explanation(prediction=empty, summary='', details=details)
        assert e.cohorts == []
        assert e.subgraphs == []

    good = Explanation(
        prediction=empty,
        summary='',
        details={
            'format': 'kumo_rfm_v2_1',
            'details': {'cohorts': [{'c': 1}], 'subgraphs': [{'s': 2}]},
        },
    )
    assert good.cohorts == [{'c': 1}]
    assert good.subgraphs == [{'s': 2}]


def _explanation(subgraphs: Any) -> Any:
    import pandas as pd
    from sdm_relational_engine.rfm import Explanation

    return Explanation(
        prediction=pd.DataFrame(),
        summary='',
        details={
            'format': 'kumo_rfm_v2_1',
            'details': {'subgraphs': subgraphs},
        },
    )


def _subgraph(tables: dict[str, Any], **extra: Any) -> dict[str, Any]:
    subgraph = {
        'seed_id': 0,
        'seed_table': 'users',
        'seed_time': '2025-04-30T00:00:00',
        'tables': tables,
        'context_examples': [],
    }
    subgraph.update(extra)
    return subgraph


def test_feature_importance_aggregates_scores_by_table_column() -> None:
    from sdm_relational_engine.rfm.explain.gradient import GraphGradientScore

    exp = _explanation(
        [
            _subgraph(
                {
                    'users': {
                        '0': {
                            'cells': {
                                'status': {'value': 'ACTIVE', 'score': 1.0},
                                'age': {'value': None, 'score': 0.089},
                            }
                        }
                    },
                    'orders': {
                        '1': {
                            'cells': {'amount': {'value': 5.0, 'score': 0.4}}
                        },
                        '2': {
                            'cells': {'amount': {'value': 2.0, 'score': 0.1}}
                        },
                    },
                }
            )
        ]
    )
    fi = exp.feature_importance
    assert isinstance(fi, GraphGradientScore)
    assert set(fi.tables) == {'users', 'orders'}
    assert fi['users']['status'] == 1.0
    assert fi['users']['age'] == pytest.approx(0.089)  # null value still scores
    assert fi['orders']['amount'] == pytest.approx(0.5)  # summed across nodes
    assert fi['users'].total_score == pytest.approx(1.089)

    frame = fi.to_pandas()
    assert list(frame.columns) == ['table', 'column', 'score']
    assert frame.iloc[0]['score'] == 1.0  # sorted descending


def test_feature_importance_sums_across_multiple_subgraphs() -> None:
    exp = _explanation(
        [
            _subgraph(
                {
                    'users': {
                        '0': {'cells': {'age': {'value': 30, 'score': 0.2}}}
                    }
                }
            ),
            _subgraph(
                {
                    'users': {
                        '0': {'cells': {'age': {'value': 40, 'score': 0.5}}}
                    }
                }
            ),
        ]
    )
    assert exp.feature_importance['users']['age'] == pytest.approx(0.7)


def test_feature_importance_ignores_context_examples_and_non_numeric() -> None:
    exp = _explanation(
        [
            _subgraph(
                {
                    'orders': {
                        '1': {
                            'cells': {
                                'amount': {'value': 5.0, 'score': 0.3},
                                'note': {'value': 'x', 'score': None},
                            }
                        }
                    }
                },
                context_examples=[
                    {'entity_id': 1, 'score': 0.9, 'label': True}
                ],
            ),
            {'seed_id': 1},  # a subgraph without a 'tables' map is ignored
        ]
    )
    fi = exp.feature_importance
    assert set(fi.tables) == {'orders'}
    assert fi['orders']['amount'] == pytest.approx(0.3)
    assert 'note' not in fi['orders'].columns  # non-numeric score skipped


def test_feature_importance_empty_when_no_subgraphs() -> None:
    import pandas as pd
    from sdm_relational_engine.rfm import Explanation
    from sdm_relational_engine.rfm.explain.gradient import GraphGradientScore

    exp = Explanation(prediction=pd.DataFrame(), summary='', details=None)
    fi = exp.feature_importance
    assert isinstance(fi, GraphGradientScore)
    assert fi.tables == {}
    assert fi.to_pandas().empty


def test_nim_failure_error_explain_reports_gpu_pressure() -> None:
    from sdm_relational_engine.exceptions import HTTPException
    from sdm_relational_engine.rfm.rfm import _nim_failure_error

    msg = str(
        _nim_failure_error(
            HTTPException(503, 'Service Unavailable'), explain=True
        )
    )
    assert 'temporarily unavailable' in msg
    assert 'GPU memory pressure' in msg
    assert 'this explanation' in msg
    assert 'one at a time' in msg


def test_nim_failure_error_prediction_has_no_pacing_hint() -> None:
    from sdm_relational_engine.exceptions import HTTPException
    from sdm_relational_engine.rfm.rfm import _nim_failure_error

    msg = str(_nim_failure_error(HTTPException(500, 'boom'), explain=False))
    assert 'this prediction' in msg
    assert 'GPU memory pressure' in msg
    assert 'one at a time' not in msg


def test_nim_failure_error_connection_drop_is_unavailable() -> None:
    import requests
    from sdm_relational_engine.rfm.rfm import _nim_failure_error

    msg = str(
        _nim_failure_error(
            requests.exceptions.ConnectionError('connection reset'),
            explain=True,
        )
    )
    assert 'temporarily unavailable' in msg


def test_nim_failure_error_timeout_points_at_the_timeout_setting() -> None:
    import requests
    from sdm_relational_engine.rfm.rfm import _nim_failure_error

    msg = str(
        _nim_failure_error(
            requests.exceptions.ReadTimeout('read timed out'), explain=False
        )
    )
    assert 'timeout' in msg
    assert 'SDMClient(url, timeout=...)' in msg
    assert 'GPU memory pressure' not in msg


def test_nim_failure_error_client_error_is_not_an_sdk_bug_report() -> None:
    """A 4xx is about the request the caller sent, so it is reported as a rejected
    request rather than routed to the client's issue tracker.
    """
    from sdm_relational_engine.exceptions import HTTPException
    from sdm_relational_engine.rfm.rfm import _nim_failure_error

    error = _nim_failure_error(
        HTTPException(400, '{"detail": "bad predictive query"}'), explain=True
    )
    msg = str(error)
    assert 'bad predictive query' in msg
    assert 'create an issue' not in msg
    assert 'GPU memory pressure' not in msg
    assert error.status_code == 400
    assert error.transient is False


def test_nim_failure_error_surfaces_invalid_params() -> None:
    """The NIM names the exact table, row and column it rejected; the top-level
    ``detail`` is often only "Request validation failed."
    """
    import json

    from sdm_relational_engine.exceptions import HTTPException
    from sdm_relational_engine.rfm.rfm import _nim_failure_error

    body = json.dumps(
        {
            'type': '/problems/validation-failed',
            'status': 422,
            'detail': 'Request validation failed.',
            'invalid_params': [
                {
                    'name': 'context.related_tables.users.rows[0][big_feature]',
                    'reason': 'int64 value 4611686018427387905 exceeds the JSON safe '
                    'integer range and must be encoded as a base-10 string.',
                }
            ],
        }
    )
    error = _nim_failure_error(HTTPException(422, body), explain=False)
    msg = str(error)
    assert 'context.related_tables.users.rows[0][big_feature]' in msg
    assert 'base-10 string' in msg
    assert 'create an issue' not in msg
    assert error.status_code == 422
    assert len(error.invalid_params) == 1


def test_nim_failure_error_caps_invalid_params() -> None:
    import json

    from sdm_relational_engine.exceptions import HTTPException
    from sdm_relational_engine.rfm.rfm import (
        _MAX_INVALID_PARAMS,
        _nim_failure_error,
    )

    body = json.dumps(
        {
            'detail': 'Request validation failed.',
            'invalid_params': [
                {'name': f'col{index}', 'reason': 'bad'}
                for index in range(_MAX_INVALID_PARAMS + 3)
            ],
        }
    )
    msg = str(_nim_failure_error(HTTPException(422, body), explain=False))
    assert 'col0: bad' in msg
    assert f'col{_MAX_INVALID_PARAMS}' not in msg
    assert 'and 3 more' in msg


def test_nim_failure_error_unclassifiable_still_invites_an_issue() -> None:
    """The invitation is reserved."""
    from sdm_relational_engine.rfm.rfm import _nim_failure_error

    msg = str(_nim_failure_error(RuntimeError('something odd'), explain=False))
    assert 'create an issue' in msg


def test_nim_failure_error_is_a_runtime_error() -> None:
    """Existing callers keep working."""
    from sdm_relational_engine.exceptions import HTTPException, NimFailureError
    from sdm_relational_engine.rfm.rfm import _nim_failure_error

    error = _nim_failure_error(HTTPException(503, 'busy'), explain=False)
    assert isinstance(error, NimFailureError)
    assert isinstance(error, RuntimeError)
    assert error.transient is True
    assert error.status_code == 503


def _cardinality_error(status: int = 422) -> Any:
    from sdm_relational_engine.exceptions import HTTPException

    return HTTPException(
        status,
        '{"detail": "categorical cardinality 15001 exceeds limit 10000."}',
    )


def _cardinality_payload() -> dict[str, Any]:
    return {
        'task': {'entity_table_names': ['USERS']},
        'context': {
            'instance_table': {
                'columns': ['USER_ID', 'EMAIL'],
                'rows': [
                    [index, f'u{index}@example.com'] for index in range(15001)
                ],
            },
            'related_tables': {
                'ORDERS': {
                    'columns': ['SKU'],
                    'rows': [[f'sku-{index}'] for index in range(12000)],
                }
            },
        },
        'predict': {
            'instance_table': {'columns': ['USER_ID'], 'rows': [[1]]},
            'related_tables': {},
        },
    }


def test_cardinality_rejection_names_column_and_remedy() -> None:
    from sdm_relational_engine.rfm.rfm import _nim_failure_error

    msg = str(
        _nim_failure_error(
            _cardinality_error(), explain=False, payload=_cardinality_payload()
        )
    )
    assert 'categorical cardinality 15001 exceeds limit 10000' in msg
    assert "'USERS.EMAIL' holds 15,001" in msg
    assert "'ORDERS.SKU' holds 12,000" in msg
    assert "graph['USERS']['EMAIL'].stype = Stype.ID" in msg
    assert 'create an issue' not in msg
    assert 'instance_table' not in msg


def test_cardinality_rejection_without_payload_still_states_the_fix() -> None:
    from sdm_relational_engine.rfm.rfm import _nim_failure_error

    msg = str(_nim_failure_error(_cardinality_error(), explain=False))
    assert 'Stype.ID' in msg
    assert 'does not lift the limit' in msg
    assert 'create an issue' not in msg


def test_non_cardinality_rejection_has_no_cardinality_hint() -> None:
    from sdm_relational_engine.exceptions import HTTPException
    from sdm_relational_engine.rfm.rfm import _nim_failure_error

    msg = str(
        _nim_failure_error(
            HTTPException(422, '{"detail": "N_cols 7 exceeds limit 5."}'),
            explain=False,
            payload=_cardinality_payload(),
        )
    )
    assert 'N_cols 7 exceeds limit 5' in msg
    assert 'stype' not in msg


@pytest.mark.parametrize('status', [408, 429])
def test_come_back_later_statuses_are_transient(status: int) -> None:
    from sdm_relational_engine.exceptions import HTTPException
    from sdm_relational_engine.rfm.rfm import _nim_failure_error

    error = _nim_failure_error(HTTPException(status, 'later'), explain=False)
    assert error.transient is True


@pytest.mark.parametrize('status', [400, 404, 413, 422])
def test_rejections_are_not_transient(status: int) -> None:
    from sdm_relational_engine.exceptions import HTTPException
    from sdm_relational_engine.rfm.rfm import _nim_failure_error

    error = _nim_failure_error(HTTPException(status, 'nope'), explain=False)
    assert error.transient is False


def test_high_cardinality_columns_reports_worst_first_and_dedupes() -> None:
    from sdm_relational_engine.rfm.inference.payload import (
        high_cardinality_columns,
    )

    payload = {
        'task': {'entity_table_names': ['USERS']},
        'context': {
            'instance_table': {
                'columns': ['EMAIL'],
                'rows': [[f'{i}'] for i in range(30)],
            },
            'related_tables': {
                'ORDERS': {
                    'columns': ['SKU', 'QTY'],
                    'rows': [[f's{i}', i] for i in range(25)],
                }
            },
        },
        'predict': {
            'instance_table': {
                'columns': ['EMAIL'],
                'rows': [[f'{i}'] for i in range(12)],
            },
            'related_tables': {},
        },
    }
    found = high_cardinality_columns(payload, limit=10)
    assert found == [('USERS', 'EMAIL', 30), ('ORDERS', 'SKU', 25)]


def test_high_cardinality_columns_counts_tokenized_text_cells() -> None:
    r"""A 'text' column travels as one-element token lists, and the NIM counts
    those tokens, so the column has to be named there too.
    """
    from sdm_relational_engine.rfm.inference.payload import (
        high_cardinality_columns,
    )

    payload = {
        'task': {'entity_table_names': ['USERS']},
        'context': {
            'instance_table': {'columns': ['ID'], 'rows': [[1]]},
            'related_tables': {
                'ORDERS': {
                    'columns': ['SKU'],
                    'rows': [[[f'sku-{i}']] for i in range(25)],
                }
            },
        },
        'predict': {
            'instance_table': {'columns': ['ID'], 'rows': [[1]]},
            'related_tables': {},
        },
    }
    assert high_cardinality_columns(payload, limit=10) == [
        ('ORDERS', 'SKU', 25)
    ]


def test_high_cardinality_columns_ignores_non_string_and_within_limit() -> None:
    from sdm_relational_engine.rfm.inference.payload import (
        high_cardinality_columns,
    )

    payload = {
        'task': {'entity_table_names': ['USERS']},
        'context': {
            'instance_table': {
                'columns': ['ID', 'TAG'],
                'rows': [[i, 'same'] for i in range(50)],
            },
            'related_tables': {},
        },
        'predict': {
            'instance_table': {'columns': ['ID'], 'rows': [[1]]},
            'related_tables': {},
        },
    }
    assert high_cardinality_columns(payload, limit=10) == []
