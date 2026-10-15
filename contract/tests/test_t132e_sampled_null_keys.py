# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32e part 1: a null key reached by sampling, on ``python-core``, held
to the reference's request.

    python -m pytest contract/tests/test_t132e_sampled_null_keys.py -q

Written by the package's independent test author from the review of
18bd89d6 (a single sampled null text key the core refused, rule
``null_key``, although the reference succeeds: ``np.unique`` sorts a
one-element array) and the reference.
``contract/sampled-null-key-vectors.json`` is recorded from the reference
by ``contract/tools/sampled_null_key_record.py``; this test runs the same
script's ``--observe python-core`` in this interpreter, so both routes are
observed and projected by the same code.

- The review's reproduction (``singleton-null-text``: ``payments`` keyed by
  text, 199 of 200 distinct, the one null row the only row sampled for
  account 3, ``PREDICT accounts.amount FOR EACH accounts.account_id`` with
  ``indices=[3]``, ``num_context_samples=1``, ``random_seed=0``): the
  reference succeeds, so the core must, with the same request.
- Its neighbours, recorded whatever the reference does: a null text key
  sampled beside non-null ones (``mixed-null-text``), the null not sampled
  (``null-text-not-sampled``), two null text keys sampled alone
  (``two-nulls-text``), and a null DECIMAL(18, 0) key sampled alone and
  mixed (``singleton-null-decimal``, ``mixed-null-decimal``).
- Each vector's answer must be the reference's: the outcome (result,
  stage, class, code), each table's key, the request the endpoint
  received (``metadata.request_id`` aside) and the warning classes; and
  the core must have been reached. A difference passes only under an
  approved register entry naming the vector with its transition
  (``contract/sampled-null-key-vectors.json#<id or glob> <from> -> <to>``),
  as the other vector tests read it: a ref covers exactly the transition
  it declares, a proposed entry excuses nothing, and ``success ->
  success`` is a transition like any other, so a request that differs on
  a success passes only under an approved entry declaring that for the
  vector. ``two-nulls-text`` is the case in point: two null keys reached
  for one example, where P1B-13 (approved, re-approved by the owner on
  2026-10-13: the reference corrupts null-key requests) makes a null in
  the walk no link, so the core's request differs from the reference's;
  it passes once P1B-13 carries
  ``contract/sampled-null-key-vectors.json#two-nulls-text success ->
  success``. ``register.required`` (derived: ``mixed-null-decimal``,
  INTERNAL_ERROR at materialize) excuses nothing by itself.
- The singletons (:data:`EXACT`: one null key sampled alone, text and
  decimal, the review's reproduction among them) are held to the
  reference's exact answer and request: no entry excuses them, whatever
  it declares, since a single null in the walk is the reference's clean
  case (np.unique sorts it) and P1B-13 changes nothing there.
- P1B-40, option (b) (owner, 2026-10-13, relayed): where sampling reaches
  a key mixing nulls with real text or decimal values (``mixed-null-text``,
  refused INVALID_REQUEST by the reference; ``mixed-null-decimal``,
  INTERNAL_ERROR), the core leaves the null-key rows out and succeeds:
  :func:`test_option_b_leaves_the_null_key_rows_out_with_one_warning`
  holds its request to the twin's (the same tables with the null row not
  reached, recorded from the reference) with no null-key row, and its
  warnings to the twin's and one more naming the table and the count. The
  per-vector test then passes only under the approved entry's binding
  refs (``refusal INVALID_REQUEST -> success``, ``refusal INTERNAL_ERROR
  -> success``).
- The case format holds neither ``num_context_samples`` nor
  ``random_seed``, so these vectors are not cases and ``rust-host`` does
  not run them; the core's answer is held through the Python host.

Gating (``contract/tests/t132e_gate.py``): ``t132e-pending.json``'s
``pending["sampled-python-core"]`` lists the vectors that failed here when
this was written, at 18bd89d6; shrink-only, as the other groups.
"""

from __future__ import annotations

import copy
import fnmatch
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / 'contract/sampled-null-key-vectors.json'
RECORDER = ROOT / 'contract/tools/sampled_null_key_record.py'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _module('sampled_null_key_record', RECORDER)
G = _module('t132e_gate', ROOT / 'contract/tests/t132e_gate.py')
RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}
TRANSITION = re.compile(
    re.escape(R.REGISTER_REF)
    + r'(?P<glob>\S+) (?P<old>success|refusal [A-Z][A-Z_]*)'
    + r' -> (?P<new>success|refusal [A-Z][A-Z_]*)'
)


def transitions() -> tuple[list[tuple[str, str, str, str, str]], list[str]]:
    document = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))
    out: list[tuple[str, str, str, str, str]] = []
    malformed: list[str] = []
    for entry in document['entries']:
        status = entry.get('status')
        if status not in ('proposed', 'approved'):
            continue
        for ref in entry.get('refs') or []:
            if not isinstance(ref, str) or not ref.startswith(R.REGISTER_REF):
                continue
            found = TRANSITION.fullmatch(ref)
            if found is None:
                malformed.append(f'{entry["id"]}: {ref}')
                continue
            out.append(
                (
                    str(entry['id']),
                    str(status),
                    found['glob'],
                    found['old'],
                    found['new'],
                )
            )
    return out, malformed


#: The vectors held to the reference's exact answer, request included,
#: which no register entry excuses: one null key sampled alone.
EXACT = frozenset({'singleton-null-text', 'singleton-null-decimal'})


def problem_of(vector: dict[str, Any], got: dict[str, Any]) -> str | None:
    r"""Why ``got`` fails the vector: None when it is the reference's
    answer, or, outside :data:`EXACT`, a difference an approved entry
    declares exactly (``success -> success`` included)."""
    want = vector['reference']
    if got == want:
        return None
    old = R.outcome_class(want['outcome'])
    new = R.outcome_class(got['outcome'])
    if vector['id'] in EXACT:
        return (
            f'{vector["id"]}: got {got}, the reference {want} ({old} -> '
            f'{new}); held exactly, no entry excuses it'
        )
    for _, status, glob, frm, to in transitions()[0]:
        if (
            status == 'approved'
            and (frm, to) == (old, new)
            and fnmatch.fnmatchcase(vector['id'], glob)
        ):
            return None
    return f'{vector["id"]}: got {got}, the reference {want} ({old} -> {new})'


@pytest.fixture(scope='module')
def core_answers(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    r"""Every vector through ``python-core``, in a child of this
    interpreter (the core's route forced by ``SDM_CORE_ROUTES``)."""
    out = tmp_path_factory.mktemp('t132e-sampled') / 'python-core.json'
    done = subprocess.run(
        [
            sys.executable,
            '-I',
            str(RECORDER),
            '--observe',
            'python-core',
            '--out',
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
        timeout=900,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    loaded: dict[str, Any] = json.loads(out.read_text(encoding='utf-8'))
    return loaded


@pytest.mark.parametrize(
    'vector_id',
    [pytest.param(v['id'], id=v['id']) for v in RECORDED['vectors']],
)
def test_python_core_samples_a_null_key_as_the_reference(
    vector_id: str, core_answers: dict[str, Any]
) -> None:
    vector = BY_ID[vector_id]

    def check() -> None:
        got = copy.deepcopy(core_answers[vector_id])
        operations = got.pop('operations')
        got.pop('messages')
        assert 'materialize' in operations, (
            f'the core was not reached: {operations}'
        )
        problem = problem_of(vector, got)
        assert problem is None, problem

    G.held('sampled-python-core', vector_id, check)


def test_the_vectors_are_the_reference_s() -> None:
    assert RECORDED['format'] == R.FORMAT
    assert (
        RECORDED['recorded_by'] == 'contract/tools/sampled_null_key_record.py'
    )
    assert RECORDED['cell'] == 'ref-pandas3'
    assert RECORDED['register_ref'] == R.REGISTER_REF
    assert RECORDED['pql'] == R.PQL
    assert (
        RECORDED['materialize']
        == R.MATERIALIZE
        == {
            'indices': [3],
            'num_context_samples': 1,
            'random_seed': 0,
        }
    )
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    assert [v['id'] for v in RECORDED['vectors']] == [
        v['id'] for v in R.VECTORS
    ]
    for vector in R.VECTORS:
        assert BY_ID[vector['id']]['setup'] == vector['setup']
    for vector in RECORDED['vectors']:
        why = R.register(vector['reference'])
        assert vector['register'] == {'required': bool(why), 'why': why}


def test_the_review_s_reproduction_is_a_success_with_a_request() -> None:
    # The review's case, as given, and what the reference answers it.
    singleton = BY_ID['singleton-null-text']
    assert singleton['setup'] == [
        'CREATE TABLE accounts AS SELECT i AS account_id, i::DOUBLE AS amount '
        'FROM range(200) t(i)',
        'CREATE TABLE payments AS SELECT CASE WHEN i=0 THEN NULL ELSE '
        "'k'||i::VARCHAR END AS payment_id, CASE WHEN i=0 THEN 3 ELSE 150 END "
        'AS account_id, i::DOUBLE AS amount FROM range(200) t(i)',
    ]
    reference = singleton['reference']
    assert reference['outcome']['result'] == 'success'
    assert reference['outcome']['stage'] == 'predict'
    assert reference['keys'] == {
        'accounts': 'account_id',
        'payments': 'payment_id',
    }
    assert reference['request'] is not None
    assert 'request_id' not in reference['request']['metadata']
    # Its neighbours hold each answer the reference gives.
    outcomes = {
        v['id']: R.outcome_class(v['reference']['outcome'])
        for v in RECORDED['vectors']
    }
    assert outcomes['mixed-null-text'] == 'refusal INVALID_REQUEST'
    assert outcomes['mixed-null-decimal'] == 'refusal INTERNAL_ERROR'
    assert outcomes['singleton-null-decimal'] == 'success'
    assert BY_ID['mixed-null-decimal']['register']['required']


def test_the_comparison_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    module = sys.modules[__name__]
    entries: list[tuple[str, str, str, str, str]] = []
    monkeypatch.setattr(module, 'transitions', lambda: (entries, []))
    plain = BY_ID['null-text-not-sampled']
    want = plain['reference']
    assert plain['id'] not in EXACT
    assert problem_of(plain, copy.deepcopy(want)) is None
    refused = copy.deepcopy(want)
    refused['outcome'] = {
        'result': 'refusal',
        'stage': 'materialize',
        'error': 'StructuredDataError',
        'code': 'INVALID_REQUEST',
    }
    refused['request'] = None
    assert problem_of(plain, refused) is not None
    entries[:] = [
        ('P-1', 'proposed', 'null-text-*', 'success', 'refusal INVALID_REQUEST')
    ]
    assert problem_of(plain, refused) is not None
    entries[:] = [
        ('A-1', 'approved', 'null-text-*', 'success', 'refusal INVALID_REQUEST')
    ]
    assert problem_of(plain, refused) is None
    # The same approved transition, for the singleton: held exactly.
    entries[:] = [
        ('A-1', 'approved', 'singleton-*', 'success', 'refusal INVALID_REQUEST')
    ]
    singleton = BY_ID['singleton-null-text']
    refused_singleton = copy.deepcopy(refused)
    refused_singleton['keys'] = singleton['reference']['keys']
    refused_singleton['warnings'] = singleton['reference']['warnings']
    assert problem_of(singleton, refused_singleton) is not None
    entries[:] = []
    # Another request is a difference, however small.
    for vector in (plain, singleton):
        other = copy.deepcopy(vector['reference'])
        other['request']['metadata']['num_context_examples'] = {'number': '2'}
        assert problem_of(vector, other) is not None
        other = copy.deepcopy(vector['reference'])
        other['keys']['payments'] = None
        assert problem_of(vector, other) is not None


def test_an_approved_success_to_success_ref_excuses_a_request_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""``two-nulls-text``: another request on a success passes only
    under an approved ``success -> success`` ref naming it; a proposed
    one, another vector's, or another transition excuses nothing; and no
    ref excuses a singleton."""
    module = sys.modules[__name__]
    entries: list[tuple[str, str, str, str, str]] = []
    monkeypatch.setattr(module, 'transitions', lambda: (entries, []))
    two = BY_ID['two-nulls-text']
    assert R.outcome_class(two['reference']['outcome']) == 'success'
    other = copy.deepcopy(two['reference'])
    other['request']['metadata']['num_context_examples'] = {'number': '9'}
    assert problem_of(two, other) is not None
    entries[:] = [
        ('P1B-13', 'proposed', 'two-nulls-text', 'success', 'success')
    ]
    assert problem_of(two, other) is not None
    entries[:] = [
        ('P1B-13', 'approved', 'mixed-null-text', 'success', 'success'),
        ('P1B-13', 'approved', 'two-nulls-text', 'success', 'refusal X'),
    ]
    assert problem_of(two, other) is not None
    entries[:] = [
        ('P1B-13', 'approved', 'two-nulls-text', 'success', 'success')
    ]
    assert problem_of(two, other) is None
    # The same ref excuses no refusal in its place.
    refused = copy.deepcopy(two['reference'])
    refused['outcome'] = dict(
        refused['outcome'],
        result='refusal',
        stage='materialize',
        error='StructuredDataError',
        code='INVALID_REQUEST',
    )
    refused['request'] = None
    assert problem_of(two, refused) is not None
    # The singletons are held exactly, under any entry.
    entries[:] = [
        ('A-1', 'approved', '*', 'success', 'success'),
        ('A-2', 'approved', '*', 'success', 'refusal INVALID_REQUEST'),
    ]
    for vector_id in sorted(EXACT):
        singleton = BY_ID[vector_id]
        assert (
            problem_of(singleton, copy.deepcopy(singleton['reference'])) is None
        )
        changed = copy.deepcopy(singleton['reference'])
        changed['request']['metadata']['num_context_examples'] = {'number': '9'}
        assert 'held exactly' in (problem_of(singleton, changed) or '')
        refused = copy.deepcopy(singleton['reference'])
        refused['outcome'] = dict(
            refused['outcome'],
            result='refusal',
            stage='materialize',
            error='StructuredDataError',
            code='INVALID_REQUEST',
        )
        refused['request'] = None
        assert 'held exactly' in (problem_of(singleton, refused) or '')
    # Only singletons are exact: both sampled alone, one null each.
    assert {
        v['id'] for v in RECORDED['vectors'] if v['id'].startswith('singleton-')
    } == EXACT


#: P1B-40, option (b) (owner, 2026-10-13, relayed): where sampling
#: reaches a table whose key mixes nulls with real text or decimal values
#: (the reference refuses, INVALID_REQUEST, or fails, INTERNAL_ERROR),
#: the core leaves the null-key rows out of the walk and succeeds, with one
#: warning per table saying how many rows it left out. Each such vector,
#: its twin (the same tables with the null row not reached, which the
#: reference answers), and the rows left out.
OPTION_B = {
    'mixed-null-text': ('null-text-not-sampled', 1),
    'mixed-null-decimal': ('null-decimal-not-sampled', 1),
}


def _amounts(request: Any, table: str) -> list[float]:
    r"""Every ``amount`` the request sends for ``table``'s rows, in the
    context and the rows to predict (``payments.amount`` is the row's
    number, so the null-key row is amount 0)."""
    found = []
    for part in ('context', 'predict'):
        rows = ((request.get(part) or {}).get('related_tables') or {}).get(
            table
        )
        if not rows:
            continue
        at = rows['columns'].index('amount')
        found.extend(float(row[at]['number']) for row in rows['rows'])
    return found


@pytest.mark.parametrize('vector_id', [pytest.param(i, id=i) for i in OPTION_B])
def test_option_b_leaves_the_null_key_rows_out_with_one_warning(
    vector_id: str, core_answers: dict[str, Any]
) -> None:
    r"""The core succeeds where the reference refuses (the approved
    divergence: ``refusal INVALID_REQUEST -> success``, ``refusal
    INTERNAL_ERROR -> success``); its request holds no row of the null key
    (the payments reached for account 3 are rows 1 and 2, never row 0) and
    is the twin's, the request the reference sends when the null row is
    not reached at all; its warnings are the twin's and exactly one more,
    whose text names the table and the rows left out."""
    twin_id, left_out = OPTION_B[vector_id]
    twin = BY_ID[twin_id]
    reference = BY_ID[vector_id]['reference']

    def check() -> None:
        got = copy.deepcopy(core_answers[vector_id])
        operations = got.pop('operations')
        messages = got.pop('messages')
        assert 'materialize' in operations, operations
        assert got['outcome'] == {
            'result': 'success',
            'stage': 'predict',
            'error': None,
            'code': None,
        }, got['outcome']
        assert got['keys'] == reference['keys']
        amounts = _amounts(got['request'], 'payments')
        assert 0.0 not in amounts, amounts
        assert sorted(amounts) == [1.0, 2.0], amounts
        assert got['request'] == twin['reference']['request']
        extra = Counter(got['warnings']) - Counter(
            twin['reference']['warnings']
        )
        missing = Counter(twin['reference']['warnings']) - Counter(
            got['warnings']
        )
        assert sum(extra.values()) == 1 and not missing, got['warnings']
        naming = [m for m in messages if 'payments' in m]
        before = [m for m in twin['reference_messages'] if 'payments' in m]
        assert len(naming) == len(before) + 1, messages
        assert any(
            re.search(rf'(?<![\w.]){left_out}(?![\w.])', m) for m in naming
        ), naming

    G.held(
        'exits',
        f'py:test_option_b_leaves_the_null_key_rows_out_with_one_warning[{vector_id}]',
        check,
    )


def test_option_b_s_vectors_are_the_reference_s_refusals() -> None:
    # The divergence is from a refusal, and each twin is a success the
    # reference sends a request for, with the null row's key elsewhere.
    for vector_id, (twin_id, _) in OPTION_B.items():
        outcome = BY_ID[vector_id]['reference']['outcome']
        assert outcome['result'] == 'refusal'
        assert outcome['stage'] == 'materialize'
        assert outcome['code'] in ('INVALID_REQUEST', 'INTERNAL_ERROR')
        twin = BY_ID[twin_id]['reference']
        assert twin['outcome']['result'] == 'success'
        assert sorted(_amounts(twin['request'], 'payments')) == [1.0, 2.0]
    for vector in RECORDED['vectors']:
        assert len(vector['reference_messages']) == len(
            vector['reference']['warnings']
        )


def test_every_vector_ref_names_its_transition() -> None:
    assert transitions()[1] == []


def test_the_pending_list_only_shrinks() -> None:
    assert G.only_shrinks() == []
    assert G.pending('sampled-python-core') <= set(BY_ID)
