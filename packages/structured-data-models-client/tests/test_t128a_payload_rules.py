# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T1.28a's payload rules on the core route equal the reference's.

Written by the package's independent test author from the brief, the design
row and the reference, before the implementation. Each case in
``t128a-payload-rules.json`` is a DuckDB source, a query and the arguments
of ``PredictiveQuery.materialize``; ``t128a_payload_rules_record.py``
recorded what the reference gives for it, through the same ``observe`` this
test calls. Here the call runs forced onto the core (``SDM_CORE_ROUTES``:
the schema's, materialize's and, for a case that sends a call, predict's
routes) and what the case's exit compares must equal the reference's:

- exit "classification payload" (cases covering ``CLASSIFICATION``):
  ``base.binary``, ``base.multiclass`` and seeded binary, multiclass,
  boolean-static and static-condition sources: the task (kind, classes,
  their order, the positive class), the context's and predicted rows, each
  related table's columns and the instance ids it keeps (the context that
  answers the question taken out, a static target's column left out), the
  metadata's counts and the number the leakage notice gives. A BOOLEAN
  static target's classes are the strings 'False'/'True';
- exit "regression payload's parsed fields": the output (the statistic),
  the inference's run mode, the metadata's counts (the entity count), and
  the reference's refusals of an unknown statistic or one on a
  classification;
- exit "payload limits and num_context_samples' bound": the reference's
  refusals at each limit (num_context_samples 0 and 10,001, 1,001 entities,
  a 16-table graph) and its acceptance at the limit (10,000 context
  samples, 1,000 entities, 15 tables); and PD-2: at the byte limit, the
  longest payload the reference takes is taken and the bytes the core
  produced are the count the reference made, and one byte-step past it is
  refused as the reference refuses it;
- exit "empty and reserved-name refusals": the class and code;
- exit "inference_config written into the call": the payload's inference
  section, and the call's model, output and inference as the endpoint
  received them (or the reference's refusal of a key the model does not
  read, before sending);
- exit "a DECIMAL(38,0) feature's wire dtype": the payload's schema section
  and the column's values as the context sends them;
- carry-forward "AVG(...) = NULL": the reference fails with INTERNAL_ERROR;
  the core must refuse it by name, with a code other than INTERNAL_ERROR,
  and not by one of the rules T1.28a lifts.

Before T1.28a the core refuses these by the rules the package lifts
(``classification_task``, ``target_leakage``, ``num_context_samples``,
``inference_config``, ``wire_dtype``), and the test says so by name.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from sdm_client import _routing
from sdm_client._rust import run as _rust_run

HERE = Path(__file__).resolve().parent
RECORDED = json.loads(
    (HERE / 't128a-payload-rules.json').read_text(encoding='utf-8')
)
CORE = ','.join(
    _routing.FROM_SOURCE_DUCKDB
    + _routing.MATERIALIZE_DUCKDB
    + _routing.PREDICT_DUCKDB
)

#: The rules T1.28a lifts: a refusal by one of them is the package not
#: done yet, never an outcome a case may end with.
T128A_RULES = {
    'classification_task',
    'target_leakage',
    'num_context_samples',
    'inference_config',
    'wire_dtype',
}


def _recorder() -> Any:
    name = 't128a_payload_rules_record'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, HERE / 't128a_payload_rules_record.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _recorder()
LEAKED = re.compile(r'([\d,]+) of the rows sampled as context are the rows')


def _leaked(notices: list[str]) -> list[int]:
    r"""The counts the context-leakage notices give."""
    return sorted(
        int(m.group(1).replace(',', ''))
        for m in (LEAKED.search(n) for n in notices)
        if m
    )


def _counts(observed: dict[str, Any]) -> dict[str, Any]:
    metadata = observed['metadata']
    return {
        'num_context_examples': metadata.get('num_context_examples'),
        'num_prediction_examples': metadata.get('num_prediction_examples'),
        'parts': observed['parts'],
    }


def _on_the_core(
    spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], list[bytes]]:
    r"""The case observed on the core route, and the payload attachments'
    bytes as the core's runs gave them."""
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    taken: list[bytes] = []
    take = _rust_run.Finished.take

    def recording(self: Any, attachment: int) -> Any:
        got = take(self, attachment)
        state = (
            self.result.get('state') if isinstance(self.result, dict) else None
        )
        if isinstance(state, dict) and attachment == state.get('payload'):
            taken.append(bytes(got))
        return got

    monkeypatch.setattr(_rust_run.Finished, 'take', recording)
    return R.observe(spec), taken


def _not_lifted(got: dict[str, Any]) -> str:
    return (
        f"the core refused it by T1.28a's rule {got.get('rule')!r} "
        f'({got.get("code")}): not lifted yet'
    )


def _not_held(got: dict[str, Any]) -> bool:
    r"""Whether a refusal is the core's "not one the core supports yet",
    naming a rule it does not hold (``sdm_engine::not_yet``)."""
    return got.get('rule') is not None and 'supports yet' in (
        got.get('message') or ''
    )


def test_the_recording_is_the_reference_s() -> None:
    # The vectors came from the reference by the committed recorder, and
    # each exit has cases, accepted and refused where the exit names both.
    assert RECORDED['recorded_by'] == 't128a_payload_rules_record.py'
    assert len(RECORDED['reference']) == 40
    assert [c['name'] for c in RECORDED['cases']] == [
        c['name'] for c in R.CASES
    ] + ['size.at-the-limit', 'size.one-past-the-limit']
    by_exit: dict[str, set[str]] = {}
    for case in RECORDED['cases']:
        covers = case['covers']
        if covers.startswith(R.LIMITS):
            covers = R.LIMITS
        by_exit.setdefault(covers, set()).add(case['expected']['outcome'])
    assert by_exit[R.CLASSIFICATION] == {'payload'}
    assert by_exit[R.REGRESSION_FIELDS] == {'payload', 'refused'}
    assert by_exit[R.LIMITS] == {'payload', 'refused'}
    assert by_exit[R.REFUSALS] == {'refused'}
    assert by_exit[R.INFERENCE] == {'payload'}
    assert by_exit[R.DECIMAL] == {'payload'}
    assert by_exit[R.AVG_NULL] == {'refused'}
    kinds = {
        c['expected']['task']['kind']
        for c in RECORDED['cases']
        if c['covers'] == R.CLASSIFICATION
    }
    assert kinds == {'binary_classification', 'multiclass_classification'}
    # A boolean static target's labels are the strings 'False'/'True'.
    booleans = [
        c['expected']['task']['target']['classes']
        for c in RECORDED['cases']
        if c['name'].endswith('boolean-static')
    ]
    assert booleans and all(set(b) == {'False', 'True'} for b in booleans)
    # The leakage notice is among what is compared.
    assert any(
        _leaked(c['expected']['notices'])
        for c in RECORDED['cases']
        if c['covers'] == R.CLASSIFICATION
    )


CASES = RECORDED['cases']


@pytest.mark.parametrize('spec', CASES, ids=[c['name'] for c in CASES])
def test_the_core_s_payload_rules_are_the_reference_s(
    spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = spec['expected']
    got, attachments = _on_the_core(spec, monkeypatch)
    covers = spec['covers']

    if covers == R.AVG_NULL:
        # The reference crashes (INTERNAL_ERROR); the core refuses it by
        # name (E66(b)) and does not accept it.
        assert expected['code'] == 'INTERNAL_ERROR'
        assert got['outcome'] == 'refused', 'AVG(...) = NULL was accepted'
        assert got['rule'] not in T128A_RULES, _not_lifted(got)
        assert got['code'] not in (None, 'INTERNAL_ERROR'), got
        assert got['message'], got
        return

    if got['outcome'] == 'refused' and got.get('rule') in T128A_RULES:
        pytest.fail(_not_lifted(got))

    if expected['outcome'] == 'refused':
        # The reference's refusal: its code, raised as (a subclass of) its
        # class, and the core's own rule, not a rule it does not hold yet.
        assert got['outcome'] == 'refused', (
            'accepted where the reference refuses'
        )
        assert not _not_held(got), (
            f'refused by a rule the core does not hold yet ({got["rule"]!r}), '
            f'where the reference refuses it by its own'
        )
        assert got['code'] == expected['code'], got.get('message')
        assert expected['class'] in got['mro'], got['mro']
        return
    assert got['outcome'] == 'payload', (
        f'refused ({got.get("code")}, rule {got.get("rule")!r}) where the '
        f'reference builds the payload: {got.get("message")}'
    )

    if covers == R.CLASSIFICATION:
        fields = (
            'task',
            'context',
            'context_related',
            'predict',
            'predict_related',
        )
        assert {k: got[k] for k in fields} == {k: expected[k] for k in fields}
        assert _counts(got) == _counts(expected)
        assert _leaked(got['notices']) == _leaked(expected['notices'])
    elif covers == R.REGRESSION_FIELDS:
        assert got['task'] == expected['task']
        assert got['output'] == expected['output']
        assert got['inference']['run_mode'] == expected['inference']['run_mode']
        assert _counts(got) == _counts(expected)
        assert (got['context'], got['predict']) == (
            expected['context'],
            expected['predict'],
        )
    elif covers.startswith(R.LIMITS):
        assert _counts(got) == _counts(expected)
        assert got['context_related'] == expected['context_related']
        if spec.get('measure'):
            # PD-2: the count is of the bytes the core produced and sends;
            # here, ASCII and compact, it is the reference's count too.
            assert len(attachments) == 1, attachments
            assert len(attachments[0]) == expected['bytes']
            assert expected['bytes'] <= R.PAYLOAD_LIMIT
    elif covers == R.INFERENCE:
        assert got['inference'] == expected['inference']
        assert got.get('call') == expected['call']
        assert got.get('call_refusal') == expected['call_refusal']
    elif covers == R.DECIMAL:
        assert got['schema'] == expected['schema']
        assert got['decimal_values'] == expected['decimal_values']
    else:  # pragma: no cover - every case names an exit above
        pytest.fail(f'no exit named: {covers}')


def test_a_payload_past_the_byte_limit_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # PD-2, past the limit: the core's refusal is named (a rule or a code
    # the reference gives), never a crash, and the case before it is taken.
    past = next(c for c in CASES if c['name'] == 'size.one-past-the-limit')
    got, attachments = _on_the_core(past, monkeypatch)
    assert got['outcome'] == 'refused'
    assert not _not_held(got), got
    assert got['code'] == past['expected']['code']
    assert not attachments
