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
  (``contract/sampled-null-key-vectors.json#<id or glob> <from> -> <to>``);
  ``register.required`` (derived: ``mixed-null-decimal``, INTERNAL_ERROR at
  materialize) excuses nothing by itself.
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


def problem_of(vector: dict[str, Any], got: dict[str, Any]) -> str | None:
    r"""Why ``got`` fails the vector: None when it is the reference's
    answer, or a difference an approved entry declares exactly."""
    want = vector['reference']
    if got == want:
        return None
    old = R.outcome_class(want['outcome'])
    new = R.outcome_class(got['outcome'])
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
    singleton = BY_ID['singleton-null-text']
    want = singleton['reference']
    assert problem_of(singleton, copy.deepcopy(want)) is None
    refused = copy.deepcopy(want)
    refused['outcome'] = {
        'result': 'refusal',
        'stage': 'materialize',
        'error': 'StructuredDataError',
        'code': 'INVALID_REQUEST',
    }
    refused['request'] = None
    assert problem_of(singleton, refused) is not None
    entries[:] = [
        ('P-1', 'proposed', 'singleton-*', 'success', 'refusal INVALID_REQUEST')
    ]
    assert problem_of(singleton, refused) is not None
    entries[:] = [
        ('A-1', 'approved', 'singleton-*', 'success', 'refusal INVALID_REQUEST')
    ]
    assert problem_of(singleton, refused) is None
    entries[:] = []
    # Another request is a difference, however small.
    other = copy.deepcopy(want)
    other['request']['metadata']['num_context_examples'] = {'number': '2'}
    assert problem_of(singleton, other) is not None
    other = copy.deepcopy(want)
    other['keys']['payments'] = None
    assert problem_of(singleton, other) is not None


def test_every_vector_ref_names_its_transition() -> None:
    assert transitions()[1] == []


def test_the_pending_list_only_shrinks() -> None:
    assert G.only_shrinks() == []
    assert G.pending('sampled-python-core') <= set(BY_ID)
