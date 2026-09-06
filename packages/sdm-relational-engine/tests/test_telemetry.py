# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What a prediction says about itself, and what it must not say."""

import json
import logging

import pandas as pd
import pytest
import sdm_relational_engine.rfm as rfm
from sdm_relational_engine.rfm.rfm import KumoRelational
from sdm_relational_engine.rfm.telemetry import (
    SCHEMA_VERSION,
    PredictionRecord,
    describe_error,
    emit,
    recorded,
)
from sdm_relational_engine.runmode import RunMode
from sdm_relational_engine.utils import ProgressLogger

QUERY = 'PREDICT COUNT(orders.*, 0, 30) FOR customers.customer_id IN (1, 2, 3)'


def _a_real_task(model: KumoRelational) -> object:
    r"""The TaskTable predict() builds, so predict_task can be driven directly."""
    query = model._parse_query(QUERY)
    indices = query.get_rfm_entity_id_list()
    from dataclasses import replace as _replace

    query = _replace(query, for_each='FOR EACH', rfm_entity_ids=None)
    return model._get_task_table(
        query=query,
        indices=indices,
        anchor_time=None,
        context_anchor_time=None,
        run_mode=RunMode.FAST,
        lag_timesteps=0,
        max_pq_iterations=10,
        random_seed=42,
        logger=ProgressLogger.default(msg='', verbose=False),
    )


def _stub_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Replace the network call, leaving predict_task's own recording in place.

    Patched below predict_task rather than over it: predict_task is where the
    record is made, so replacing it would remove the very thing under test.
    """
    monkeypatch.setattr(
        KumoRelational,
        '_predict_batches',
        lambda self, requests, **kw: (
            [
                pd.DataFrame(
                    {'ENTITY': [1, 2, 3], 'TARGET_PRED': [1.0, 2.0, 3.0]}
                )
            ],
            None,
            None,
            None,
        ),
    )


@pytest.fixture
def model() -> KumoRelational:
    customers = pd.DataFrame(
        {
            'customer_id': range(60),
            'seg': ['gold', 'silver', 'bronze'] * 20,
        }
    )
    orders = pd.DataFrame(
        {
            'order_id': range(600),
            'customer_id': [i % 60 for i in range(600)],
            'amt': [10.0 + (i % 37) for i in range(600)],
            'ts': pd.to_datetime(
                [
                    f'2025-{1 + (i // 50) % 12:02d}-{1 + i % 28:02d}'
                    for i in range(600)
                ]
            ),
        }
    )
    graph = rfm.Graph.from_data(
        {'customers': customers, 'orders': orders},
        infer_metadata=True,
        verbose=False,
    )
    graph['customers'].primary_key = 'customer_id'
    graph['orders'].primary_key = 'order_id'
    if not graph.edges:
        graph.link('orders', 'customer_id', 'customers')
    return KumoRelational(graph, verbose=False)


def _written(caplog: pytest.LogCaptureFixture) -> list[dict]:
    return [
        json.loads(record.getMessage().removeprefix('rfm.predict '))
        for record in caplog.records
        if record.getMessage().startswith('rfm.predict ')
    ]


def test_a_prediction_names_the_graph_it_ran_against(
    model: KumoRelational,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _stub_batches(monkeypatch)

    with caplog.at_level(logging.INFO, logger='sdm_relational_engine'):
        model.predict(QUERY, verbose=False)

    written = _written(caplog)[0]
    assert len(written['graph_fingerprint']) == 64
    assert written['entities'] == 3
    assert written['outcome'] == 'returned'
    assert written['query_type'] == 'temporal'
    assert written['seconds'] >= 0


def test_a_prediction_that_raised_is_still_recorded(
    model: KumoRelational,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    r"""The run worth explaining afterwards is the one that failed."""

    def boom(self: KumoRelational, *args: object, **kw: object) -> None:
        raise RuntimeError('NIM returned 504')

    monkeypatch.setattr(KumoRelational, '_predict_batches', boom)

    with caplog.at_level(logging.INFO, logger='sdm_relational_engine'):
        with pytest.raises(RuntimeError, match='504'):
            model.predict(QUERY, verbose=False)

    written = _written(caplog)[0]
    assert written['outcome'] == 'raised'
    assert written['error'] == 'RuntimeError: NIM returned 504'
    # The same fields the happy path asserts. A regression that only filled
    # these in on success would leave the failure record hollow and still pass.
    assert len(written['graph_fingerprint']) == 64
    assert written['query_type'] == 'temporal'
    assert written['entities'] == 3
    assert written['seconds'] >= 0


def test_the_batch_size_in_force_is_the_one_recorded(
    model: KumoRelational,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _stub_batches(monkeypatch)

    with caplog.at_level(logging.INFO, logger='sdm_relational_engine'):
        model.predict(QUERY, verbose=False)
        with model.batch_mode(100):
            model.predict(QUERY, verbose=False)

    assert [w['batch_size'] for w in _written(caplog)] == ['None', '100']


def test_re_typing_a_column_afterwards_does_not_change_the_record(
    model: KumoRelational,
) -> None:
    r"""The model does not see the change, so the record must not report it.

    Re-typing a column on the caller's graph after the model was built leaves
    the parsed definition and the sampler untouched, so the prediction runs
    with the type the column had at build time. A fingerprint that moved here
    would name a graph the prediction did not use.
    """
    before = model._graph_fingerprint()

    model._graph['customers']['seg'].stype = 'text'

    assert model._graph_fingerprint() == before


def test_a_model_built_after_the_re_type_is_a_different_graph(
    model: KumoRelational,
) -> None:
    r"""The type is still part of a graph's identity; when it is read is what
    changed.
    """
    before = model._graph_fingerprint()
    model._graph['customers']['seg'].stype = 'text'

    rebuilt = KumoRelational(model._graph, verbose=False)

    assert rebuilt._graph_fingerprint() != before


def test_a_record_that_cannot_be_written_does_not_fail_the_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    r"""Telemetry describes a prediction; it must never be what breaks one."""

    class Unserializable:
        def __repr__(self) -> str:
            raise RuntimeError('no')

    record = PredictionRecord()
    record.entities = Unserializable()  # type: ignore[assignment]

    with caplog.at_level(logging.INFO, logger='sdm_relational_engine'):
        emit(record)

    assert not _written(caplog)


def test_a_cancelled_call_is_recorded_too(
    caplog: pytest.LogCaptureFixture,
) -> None:
    r"""A prediction killed by a timeout raises through, not from, ``except``."""
    with caplog.at_level(logging.INFO, logger='sdm_relational_engine'):
        with pytest.raises(KeyboardInterrupt):
            with recorded(PredictionRecord(entities=3)):
                raise KeyboardInterrupt

    written = _written(caplog)[0]
    assert written['outcome'] == 'raised'
    assert written['error'] == 'KeyboardInterrupt: '


def test_predict_task_called_directly_is_still_recorded(
    model: KumoRelational,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    r"""The client's adapter calls predict_task directly for a task request.

    Recording only in predict() left that path, which is the one a service
    actually takes for custom tasks, emitting nothing at all.
    """
    _stub_batches(monkeypatch)
    task = _a_real_task(model)

    with caplog.at_level(logging.INFO, logger='sdm_relational_engine'):
        model.predict_task(task, verbose=False)

    written = _written(caplog)
    assert len(written) == 1
    assert len(written[0]['graph_fingerprint']) == 64
    assert written[0]['outcome'] == 'returned'


def test_one_prediction_writes_exactly_one_record(
    model: KumoRelational,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    r"""predict() delegates to predict_task(), so both recording would double it."""
    _stub_batches(monkeypatch)

    with caplog.at_level(logging.INFO, logger='sdm_relational_engine'):
        model.predict(QUERY, verbose=False)

    assert len(_written(caplog)) == 1


def test_the_query_details_survive_the_delegation(
    model: KumoRelational,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    r"""predict() knows the query and entities; predict_task, which records, does not."""
    _stub_batches(monkeypatch)

    with caplog.at_level(logging.INFO, logger='sdm_relational_engine'):
        model.predict(QUERY, verbose=False)

    written = _written(caplog)[0]
    assert written['query_type'] == 'temporal'
    assert written['entities'] == 3


def test_a_direct_task_call_borrows_nothing_from_an_earlier_one(
    model: KumoRelational,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    r"""The lent detail must not outlive the call that lent it."""
    _stub_batches(monkeypatch)
    task = _a_real_task(model)

    with caplog.at_level(logging.INFO, logger='sdm_relational_engine'):
        model.predict(QUERY, verbose=False)
        model.predict_task(task, verbose=False)

    first, second = _written(caplog)
    assert first['entities'] == 3
    assert second['entities'] == 0, 'detail leaked out of the call that lent it'


def test_an_entity_named_in_an_error_is_not_written_to_the_log() -> None:
    r"""The sampler names the entity it could not find; this record must not."""
    described = describe_error(
        ValueError("Entity 'alice@acme.com' not found in table customers")
    )

    assert 'acme.com' not in described
    assert described.startswith('ValueError:')
    assert 'not found in table customers' in described


def test_an_echoed_request_body_is_not_written_to_the_log() -> None:
    described = describe_error(
        RuntimeError('NIM rejected request: {"indices": [8823, 9910, 4471]}')
    )

    assert '8823' not in described
    assert 'RuntimeError' in described


def test_the_error_type_is_always_kept() -> None:
    r"""The type is what a reader branches on, so it survives redaction."""
    assert describe_error(KeyError('customer_id')).startswith('KeyError:')
    assert describe_error(TimeoutError()).startswith('TimeoutError:')


def test_a_failure_records_the_redacted_message(
    model: KumoRelational,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def boom(self: KumoRelational, *args: object, **kw: object) -> None:
        raise ValueError("Entity 'bob@acme.com' is not in the graph")

    monkeypatch.setattr(KumoRelational, '_predict_batches', boom)

    with caplog.at_level(logging.INFO, logger='sdm_relational_engine'):
        with pytest.raises(ValueError):
            model.predict(QUERY, verbose=False)

    written = _written(caplog)[0]
    assert 'acme.com' not in written['error']
    assert written['error'].startswith('ValueError:')


def test_a_record_can_be_tied_back_to_one_prediction(
    model: KumoRelational,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    r"""Graph shape, query type, entity count and batch size can all match
    between two different calls, so those alone name no particular one.
    """
    _stub_batches(monkeypatch)

    with caplog.at_level(logging.INFO, logger='sdm_relational_engine'):
        model.predict(QUERY, verbose=False)
        model.predict(QUERY, verbose=False)

    first, second = _written(caplog)
    assert first['prediction_id'] and second['prediction_id']
    assert first['prediction_id'] != second['prediction_id']


def test_a_record_names_what_the_prediction_was_anchored_to(
    model: KumoRelational,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    r"""Two runs of one query at different anchors are different answers."""
    _stub_batches(monkeypatch)

    with caplog.at_level(logging.INFO, logger='sdm_relational_engine'):
        model.predict(QUERY, verbose=False)

    written = _written(caplog)[0]
    assert written['anchor_time']
    assert written['task_type']
    assert written['entity_table'] == 'customers'


def test_the_record_carries_its_schema_version(
    model: KumoRelational,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _stub_batches(monkeypatch)

    with caplog.at_level(logging.INFO, logger='sdm_relational_engine'):
        model.predict(QUERY, verbose=False)

    assert _written(caplog)[0]['schema_version'] == SCHEMA_VERSION


def test_a_quote_inside_a_value_does_not_end_the_redaction() -> None:
    """A value carrying an escaped quote must not survive in the message.

    Matching up to the next quote stops inside the value, so the remainder of
    it lands in the log: exactly the caller data this function exists to keep
    out. A backslash escape counts as one unit of the value.
    """
    described = describe_error(
        RuntimeError('NIM rejected request: {"entity":"alice\\"@acme.com"}')
    )
    assert 'acme.com' not in described
    assert 'alice' not in described


def test_a_quote_inside_a_single_quoted_value_is_redacted_too() -> None:
    described = describe_error(
        RuntimeError("entity 'o\\'brien@acme.com' not found")
    )
    assert 'acme.com' not in described
    assert 'brien' not in described


def test_two_quoted_values_stay_two_redactions() -> None:
    """The escape handling must not let one match swallow the text between
    two separate values, which would hide the shape of the message.
    """
    described = describe_error(
        RuntimeError("'a@x.com' and 'b@y.com' are unknown")
    )
    assert 'x.com' not in described
    assert 'y.com' not in described
    assert 'and' in described
    assert 'are unknown' in described


def test_a_newline_inside_a_value_is_redacted() -> None:
    described = describe_error(RuntimeError('entity "a\n@acme.com" not found'))
    assert 'acme.com' not in described


def test_the_fingerprint_names_the_graph_the_model_was_built_on(
    model: KumoRelational,
) -> None:
    """Changing the caller's graph afterwards must not change the record.

    The query is parsed against a definition taken when the model was built,
    and the rows come from a sampler built at the same moment. A fingerprint
    that followed the caller's later edits would name a graph the prediction
    never used, which is the one thing the record exists to get right.
    """
    before = model._graph_fingerprint()
    assert before

    model._graph.unlink('orders', 'customer_id', 'customers')

    assert model._graph_fingerprint() == before


def test_a_model_built_on_the_changed_graph_fingerprints_differently(
    model: KumoRelational,
) -> None:
    """Pinning the fingerprint records when the graph was read, not that two
    different graphs may share a name.
    """
    before = model._graph_fingerprint()
    model._graph.unlink('orders', 'customer_id', 'customers')

    rebuilt = KumoRelational(model._graph, verbose=False)

    assert rebuilt._graph_fingerprint() != before
