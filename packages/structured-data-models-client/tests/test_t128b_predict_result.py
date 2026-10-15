# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T1.28b's predict-side rules and result on the core route equal the reference's.

Written by the package's independent test author from the brief, the design
row and the reference, before the implementation. Each case in
``t128b-predict-result.json`` is a DuckDB source, a query, the arguments of
``materialize`` and ``predict``, and a local endpoint's script;
``t128b_predict_result_record.py`` recorded what the reference gives for it,
through the same ``observe`` this test calls, and the bytes each call was
answered with. Here the call runs forced onto the core (``SDM_CORE_ROUTES``:
the schema's, materialize's and predict's routes) and its public outcome
must equal the reference's. Each case names the exit it covers:

- ``LAYOUT``, exit "L1 layout vectors pass, with distinct per-row answers
  and reordered responses": every row answered with a different value, the
  answers given in order, reversed, rotated and interleaved;
- ``CLASSIFICATION``, exit "on both core hosts, over base.binary and
  base.multiclass, the boolean and text results match the reference": the
  prediction as a bool or as text, and the probabilities' columns;
- ``CONTEXT``, exit "python-core and rust-host public outcomes match for
  empty and constant context and multi-part results" and the ledger rows
  (the answer to a request with no rows; the parts merged into one result
  and the notice of an output the answer left out);
- ``PREPARE``, the ledger rows "the adapter's _preflight" and "the call's
  model and output checks (SDMClient._prepare)": the reference's refusal,
  class and code, or its result where it accepts;
- ``BOUNDARY``, exit "a bounded boundary-vector matrix ... deep and wide
  JSON, finite overflow and non-finite numbers, the endpoint's against a
  local error's provenance, and missing or retyped evidence": the vectors
  are ``contract/response-vectors.json``'s ``boundaries``, each answer
  replayed as recorded. A vector the reference reads and the core refuses
  may end so (INVALID_RESPONSE, a local error) only when its ``divergence``
  names a register entry the owner approved (P1B-22, an answer nested past
  the core's reader; P1B-23, a regression's prediction not a number); one
  naming a proposed entry (P1B-4) is held to the reference;
- ``CARRY``, the brief's carry-forward: ``np.longdouble`` options and a
  NumPy array of ``target_transforms`` end as on the reference;
- across cells: each case is held to the reference's outcome at this
  interpreter and pandas, recorded at each pin (``by_python``,
  ``dtypes_pandas2``); a probability in Kawi digits is the orchestrator's
  Unicode policy, refused INVALID_RESPONSE on the core on every
  interpreter (``core_outcome``), registered where the reference reads it;
- the Databricks serving route (``SDMClient.for_databricks_serving`` with a
  fake workspace client, no endpoint): a ``response_json`` nested 5,000
  levels, in the metadata or an item, refused on the core on every
  interpreter as P1B-22 says (INVALID_RESPONSE, field ``response``, reason
  ``too_deep``, ``core_outcome``); a ``response_json`` that is not JSON,
  cut short or empty, and an ordinary answer, as the reference's;
- ``INTEGERS``, ``ABSENT_VALUES`` and ``NUMERIC_TEXT`` (the review of
  part 2): a regression's integer predictions kept exact past 2**53 in an
  int64 column, a column mixing integers and floats as float64; a missing
  and a null prediction ``None`` in ``.items``, a NaN a NaN; probabilities
  written as text accepted where ``float()`` reads them (digit separators,
  Unicode digits, whitespace) and refused where the reference refuses them
  (``"1e999"``, ``"abc"``, ``"0__5"`` ...); ``.items`` compared as well as
  ``to_pandas()``;
- ``SIZE``, the carry-forward "size refusals on the core route ... make the
  message explain it": the refusal's code as the reference's, and its
  message names the limit.

Before T1.28b the core refuses a classification's answer by its rule
``problem_type``, option values by ``INVALID_REQUEST``, and every
non-finite number as ``INVALID_RESPONSE``; those cases fail here by name.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RECORDED = json.loads(
    (HERE / 't128b-predict-result.json').read_text(encoding='utf-8')
)
VECTORS = json.loads(
    (ROOT / 'contract/response-vectors.json').read_text(encoding='utf-8')
)
CORE = ','.join(
    _routing.FROM_SOURCE_DUCKDB
    + _routing.MATERIALIZE_DUCKDB
    + _routing.PREDICT_DUCKDB
)


def _recorder() -> Any:
    name = 't128b_predict_result_record'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, HERE / 't128b_predict_result_record.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _recorder()
CASES = RECORDED['cases']
BOUNDARIES = VECTORS['boundaries']
#: The prefix python-core puts on a core warning's text (E41(i), E59(d)).
CODE_PREFIX = re.compile(r'^\[[A-Z_]+\] ')


def _owner_approved(entry: str | None) -> bool:
    r"""Whether register entry ``entry`` is approved by the owner, read from
    its own block, anchored: its ``- id:`` line is exactly ``entry``, its
    ``status`` exactly ``approved`` and its ``approved_by`` exactly
    ``owner`` (``register.py``'s owner approval). A proposed entry, a
    reviewer's approval, or an entry not there approves nothing."""
    if not entry:
        return False
    text = (ROOT / 'docs/design/rust-core/register.yaml').read_text(
        encoding='utf-8'
    )
    for block in re.split(r'\n- id: ', text)[1:]:
        lines = block.split('\n')
        if lines[0].strip() != entry:
            continue
        fields = {
            key: value.strip()
            for key, _, value in (
                line[2:].partition(':')
                for line in lines[1:]
                if line.startswith('  ') and not line.startswith('   ')
            )
        }
        return (
            fields.get('status') == 'approved'
            and fields.get('approved_by') == 'owner'
        )
    return False


def _here(expected: dict[str, Any]) -> dict[str, Any]:
    r"""The reference's outcome at this interpreter and pandas, as the
    recorder recorded it at each pin (``--cell``): the outcome this Python
    gave where it differs (``by_python``), and pandas 2's dtypes where
    pandas 3's text column (``str``) is pandas 2's ``object``
    (``dtypes_pandas2``)."""
    import pandas as pd

    python = f'{sys.version_info.major}.{sys.version_info.minor}'
    here = dict(expected.get('by_python', {}).get(python) or expected)
    if 'calls' in expected:
        here.setdefault('calls', expected['calls'])
    if int(pd.__version__.split('.')[0]) < 3 and 'dtypes_pandas2' in expected:
        here['dtypes'] = expected['dtypes_pandas2']
    here.pop('by_python', None)
    here.pop('dtypes_pandas2', None)
    return here


def _registered(name: str) -> bool:
    r"""Whether the register names ``name`` (in an entry's notes, where
    vectors are named): a difference from the reference that is known."""
    text = (ROOT / 'docs/design/rust-core/register.yaml').read_text(
        encoding='utf-8'
    )
    return (
        re.search(rf'(?<![\w.-]){re.escape(name)}(?![\w-])', text) is not None
    )


def _on_the_core(
    spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    return R.observe(spec)


def _notices(notices: list[str]) -> list[str]:
    return sorted(CODE_PREFIX.sub('', n) for n in notices)


def _held(got: dict[str, Any], expected: dict[str, Any], name: str) -> None:
    r"""``got`` is the reference's outcome ``expected``, as each exit
    compares it."""
    assert got['stage'] == expected['stage'], (
        f'{name}: ended at {got["stage"]} ({got.get("class")} '
        f'{got.get("code")}, rule {got.get("rule")!r}: {got.get("message")}),'
        f' where the reference '
        f'ended at {expected["stage"]}'
    )
    assert got['outcome'] == expected['outcome'], (
        f'{name}: {got.get("class")} {got.get("code")} (rule '
        f'{got.get("rule")!r}): {got.get("message")}'
        if got['outcome'] == 'refused'
        else f'{name}: answered where the reference refused '
        f'({expected.get("code")})'
    )
    # What the endpoint was asked, call by call (none when refused before).
    assert [c['output'] for c in got['calls']] == [
        c['output'] for c in expected['calls']
    ], name
    if expected['outcome'] == 'refused':
        assert got['code'] == expected['code'], got['message']
        assert expected['class'] in got['mro'], got['mro']
        assert got['status_code'] == expected['status_code'], name
        if 'endpoint' in expected:
            # The endpoint's own refusal, shown as the old route shows it
            # (E47): its status, code, message and details.
            assert got.get('endpoint') == expected['endpoint'], name
        else:
            assert 'endpoint' not in got, (
                f"{name}: a local error raised as the endpoint's refusal"
            )
        return
    for key in ('model', 'outputs', 'metadata', 'columns', 'dtypes', 'rows'):
        assert got.get(key) == expected.get(key), f'{name}: {key}'
    assert got.get('frame_error') == expected.get('frame_error'), name
    assert _notices(got['notices']) == _notices(expected['notices']), name
    if 'items' in expected:
        # .items as the reference's: each prediction's value and Python type
        # (an int kept exact, None apart from NaN) and its probabilities.
        assert got.get('items') == expected['items'], f'{name}: items'


def test_the_recording_is_the_reference_s() -> None:
    # The vectors came from the reference by the committed recorder, and
    # each exit has cases, both answered and refused where it names both.
    assert RECORDED['recorded_by'] == 't128b_predict_result_record.py'
    assert len(RECORDED['reference']) == 40
    assert RECORDED['reference'] in VECTORS['boundaries_recorded_by']
    names = [c['name'] for c in CASES] + [b['name'] for b in BOUNDARIES]
    assert sorted(names) == sorted(
        [c['name'] for c in R.CASES] + ['size.one-past-the-limit']
    )
    outcomes: dict[str, set[str]] = {}
    for case in CASES:
        outcomes.setdefault(case['covers'], set()).add(
            case['expected']['outcome']
        )
    for vector in BOUNDARIES:
        outcomes.setdefault(R.BOUNDARY, set()).add(
            vector['reference']['outcome']
        )
    assert outcomes[R.LAYOUT] == {'result'}
    assert outcomes[R.CLASSIFICATION] == {'result', 'refused'}
    assert outcomes[R.CONTEXT] == {'result', 'refused'}
    assert outcomes[R.PREPARE] == {'result', 'refused'}
    assert outcomes[R.BOUNDARY] == {'result', 'refused'}
    # Not vacuous: answers distinct per row, a bool and a text prediction,
    # the absent output's and the constant context's notices, the
    # endpoint's own refusal and a local one, a non-finite number read.
    by_name = {c['name']: c['expected'] for c in CASES}
    six = by_name['layout.regression-six.reversed']['rows']
    assert len({json.dumps(r[2]) for r in six}) == 6
    assert 'bool' in by_name['base.binary.reversed']['dtypes']
    assert by_name['base.multiclass.reversed']['rows'][0][2].get('str')
    assert any(
        'answered without' in n
        for n in by_name['absent.probabilities-named']['notices']
    )
    assert any(
        'every context row has the same' in n
        for n in by_name['context.constant-regression']['notices']
    )
    vectors = {v['name']: v for v in BOUNDARIES}
    assert (
        'endpoint'
        in vectors['boundary.provenance.endpoint-422-detail']['reference']
    )
    assert (
        vectors['boundary.provenance.local-id-other-row']['reference']['code']
        == 'INVALID_RESPONSE'
    )
    assert (
        vectors['boundary.nonfinite.prediction-NaN']['reference']['outcome']
        == 'result'
    )
    assert any(len(v['body']) > 100_000 for v in BOUNDARIES), 'no wide answer'
    assert any(v['body'].count('{') > 4000 for v in BOUNDARIES), 'no deep'
    # The integer, absent-value and numeric-text exits: an int64 column
    # exact past 2**53, a mixed column as float64, None kept apart from
    # NaN in .items, and numeric text both accepted and refused.
    assert outcomes[R.INTEGERS] == {'result'}
    assert outcomes[R.ABSENT_VALUES] == {'result'}
    assert outcomes[R.NUMERIC_TEXT] == {'result', 'refused'}
    exact = by_name['integers.all-past-2-53']
    assert exact['dtypes'][2] == 'int64'
    assert exact['rows'][0][2] == {'int': '9007199254740993'}
    assert by_name['integers.mixed-int-float']['dtypes'][2] == 'float64'
    missing = by_name['absent-values.missing-one']['items'][0]
    assert missing['prediction'] is None, missing
    assert by_name['absent-values.nan-one']['items'][0]['prediction'] == {
        'f64': 'nan'
    }
    assert by_name['numeric-text.digit-separator']['outcome'] == 'result'
    # Across cells, from the reference at each pin: Kawi digits read on
    # 3.12, refused on 3.10 and 3.11 (Unicode 13 and 14); Arabic-Indic digits
    # read everywhere; a text column object on pandas 2 and str on pandas 3;
    # an answer nested 5,000 deep unreadable before 3.12.
    kawi = by_name['numeric-text.kawi-digits']
    assert kawi['outcome'] == 'result'
    assert {
        python: (o['outcome'], o['code'])
        for python, o in kawi['by_python'].items()
    } == {
        '3.10': ('refused', 'INVALID_RESPONSE'),
        '3.11': ('refused', 'INVALID_RESPONSE'),
    }
    arabic = by_name['numeric-text.arabic-indic-digits']
    assert arabic['outcome'] == 'result' and 'by_python' not in arabic
    multiclass = by_name['base.multiclass.given']
    assert (multiclass['dtypes'][2], multiclass['dtypes_pandas2'][2]) == (
        'str',
        'object',
    )
    # The Databricks serving route: the deep answers read on 3.12 and a
    # transport failure before, and a response_json that is not JSON the
    # transport's failure everywhere.
    for name in ('databricks.deep.metadata-5000', 'databricks.deep.item-5000'):
        assert by_name[name]['outcome'] == 'result'
        assert {
            p: o['code'] for p, o in by_name[name]['by_python'].items()
        } == {'3.10': 'TRANSPORT_ERROR', '3.11': 'TRANSPORT_ERROR'}
    for name in ('not-json', 'truncated', 'empty'):
        control = by_name[f'databricks.malformed.{name}']
        assert (
            control['code'] == 'TRANSPORT_ERROR' and 'by_python' not in control
        )
    assert by_name['databricks.layout.reversed']['outcome'] == 'result'
    deep = vectors['boundary.deep.metadata-5000']['reference']
    assert {p: o['code'] for p, o in deep['by_python'].items()} == {
        '3.10': 'TRANSPORT_ERROR',
        '3.11': 'TRANSPORT_ERROR',
    }
    assert by_name['numeric-text.double-separator']['outcome'] == 'refused'


@pytest.mark.parametrize('spec', CASES, ids=[c['name'] for c in CASES])
def test_the_core_s_predict_is_the_reference_s(
    spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _here(spec['expected'])
    got = _on_the_core(spec, monkeypatch)
    if 'core_outcome' in spec:
        # A policy, not the reference's outcome, on every interpreter: Kawi
        # digits refused INVALID_RESPONSE (the Unicode policy across cells),
        # and on the Databricks serving route an answer nested 5,000 levels
        # refused as P1B-22 says, field response and reason too_deep, as
        # over HTTP; each a local error. Where this interpreter's reference
        # gives another outcome, the difference is the approved entry the
        # case names, or one the register names it in.
        policy = spec['core_outcome']
        assert (got['outcome'], got.get('code')) == (
            policy['outcome'],
            policy['code'],
        ), f'{spec["name"]}: {got.get("code")}: {got.get("message")}'
        assert 'endpoint' not in got and got['status_code'] is None, got
        for key in ('field', 'reason'):
            if key in policy:
                assert got['fields'][key] == policy[key], (
                    f'{spec["name"]}: {key} {got["fields"][key]!r}, where '
                    f'the policy names {policy[key]!r}'
                )
        if (expected['outcome'], expected.get('code')) != (
            policy['outcome'],
            policy['code'],
        ):
            if spec.get('divergence'):
                assert _owner_approved(spec['divergence']), spec['divergence']
            else:
                assert _registered(spec['name']), (
                    f'{spec["name"]}: the reference here gives '
                    f'{expected["outcome"]}, and no register entry names it'
                )
        return
    if spec['covers'] == R.SIZE:
        # The reference's code, and a message that explains the limit as
        # the old route's does (it names 30MB).
        assert got['outcome'] == 'refused', 'past the limit, taken'
        assert got['code'] == expected['code'], got['message']
        assert '30MB' in expected['message']
        assert re.search(r'30 ?MB|30 ?MiB|31,?457,?280', got['message']), (
            f'the size refusal does not explain the limit: {got["message"]!r}'
        )
        return
    _held(got, expected, spec['name'])


#: The call base.regression's case makes, as the reference made it, which
#: every boundary vector answers.
RECORDED_CALLS = {
    'base.regression': next(
        c['expected']['calls']
        for c in CASES
        if c['name'] == 'layout.regression.given'
    )
}


@pytest.mark.parametrize(
    'vector', BOUNDARIES, ids=[v['name'] for v in BOUNDARIES]
)
def test_the_core_reads_each_boundary_answer_as_the_reference(
    vector: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = next(c for c in R.CASES if c['name'] == vector['name'])
    # The answer replayed exactly as recorded, whatever the call.
    monkeypatch.setattr(
        R, 'answer', lambda script, body: (vector['status'], vector['body'])
    )
    got = _on_the_core(spec, monkeypatch)
    expected = {
        **_here(vector['reference']),
        'calls': RECORDED_CALLS[vector['answers']],
    }
    if (
        _owner_approved(vector['divergence'])
        and got['outcome'] == 'refused'
        and got['code'] == 'INVALID_RESPONSE'
    ):
        # A divergence the owner approved (the vector names the register
        # entry): the core refuses it as a local error, by name.
        assert 'endpoint' not in got and got['status_code'] is None
        return
    _held(got, expected, vector['name'])


def test_a_boundary_answer_never_crashes_the_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # E49: the deepest and widest answers are read or refused by name, never
    # a RecursionError or a crash, on the core.
    deepest = max(BOUNDARIES, key=lambda v: v['body'].count('{'))
    widest = max(BOUNDARIES, key=lambda v: len(v['body']))
    for vector in (deepest, widest):
        spec = next(c for c in R.CASES if c['name'] == vector['name'])
        monkeypatch.setattr(
            R,
            'answer',
            lambda script, body, v=vector: (v['status'], v['body']),
        )
        got = _on_the_core(spec, monkeypatch)
        assert got.get('class') not in ('RecursionError', 'RouteError'), got
        if got['outcome'] == 'refused':
            assert got['code'], got
