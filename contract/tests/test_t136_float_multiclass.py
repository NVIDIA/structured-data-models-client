# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.36 fix group 1 (final review P2-1a): a float column inferred
categorical, asked as a static target, materializes as the reference's.

Written by the package's independent test author, before the fix.

    python -m pytest contract/tests/test_t136_float_multiclass.py -q

``PREDICT users.f FOR EACH users.user_id`` over a DuckDB ``DOUBLE`` or
``FLOAT`` column holding few distinct values (1.0 to 5.0, say) is a
``multiclass_classification`` in the reference; the core refuses it
``UNSUPPORTED_REQUEST``, rule ``target_type``
(``sdm-request/src/payload.rs``, ``(Kind::Multiclass, Labels::Float(_))``).
The owner's rule (locked check 5) is to match the reference where it
succeeds.

``contract/float-multiclass-vectors.json`` is recorded from the reference
by ``contract/tools/float_multiclass_record.py``, whose ``run`` and
``project`` this test calls, so every route is observed and projected by
the same code. The reference sends the label column ``TARGET`` declared
``float32`` / ``categorical``, each label the float32's double as a JSON
number, and ``task.target.classes`` as the text ``str`` gives that double
(``1.0``, ``0.10000000149011612``, ``1.0000000200408773e+20``,
``9.999999747378752e-05``, ``-0.0``): the context's distinct labels, in
the order ``pd.unique`` meets them in its training labels (not the order
the context's rows are sent in, and the asked-for entities' own labels
among them, though their examples are not sent), a negative and a
positive zero one class, spelled as first met there; a null or NaN
label's example leaves the context.

- ``test_python_core_answers_as_the_reference``: each vector on
  ``python-core``, forced onto every DuckDB route of the three public
  calls: its outcome, public schema, warning classes, each request (its
  digest as the ``payload`` rule compares it, with its tables' rows and
  null cells), the result, each request's ``task`` (kind, target dtype,
  classes in order) and ``labels`` (the label column as sent) are the
  reference's. The vectors the reference refuses (``refuse.*``: an
  infinite label, a double past float32's range, one class, none, 40) are
  the negative controls: the core must refuse them as the reference does,
  not accept them. One warning class is left out of this comparison, the
  ``RuntimeWarning`` pandas raises casting a double past float32's range
  (``RUNTIME_WARNINGS``). Fails today on every vector but the regression
  control (``large.double-sixty-one-values``), each refused
  ``UNSUPPORTED_REQUEST`` at materialize by the rule ``target_type``.
- ``test_python_old_answers_as_the_reference``: the same on
  ``python-old`` (the client's default route, the reference's code), for
  every vector. Passes today; it holds the vectors to the route they
  describe.
- ``test_preview_vectors_are_refused_by_e92_or_answered_as_the_reference``:
  a NaN, or both signed zeros, in the first 1,000 rows is refused by
  ``Schema.from_source`` on the core by name (``float_nan``,
  ``float_signed_zero``), the owner's E92 Phase 2 class: such a vector
  passes either refused there by exactly that rule, or answered as the
  reference. The same values past the first 1,000 rows (``large.*``) reach
  the labels and are held to the reference by the first test.
- The vector file: recorded at the reference of
  ``contract/reference-env.txt``, its inputs the recorder's, DuckDB holding
  what each vector's name says, the reference's classes what the
  paragraph above says of them (read from the recorded cells), and not
  vacuous.

A vector may differ from the reference otherwise only under an approved
register entry naming it, as ``REGISTER_REF`` + ``<id or glob> <from> ->
<to>`` (T1.32a's binding-transition form); none does today. The
implementer changes nothing of this file, the vectors or the recorder.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import re
import struct
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / 'contract/float-multiclass-vectors.json'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _module(
    'float_multiclass_record',
    ROOT / 'contract/tools/float_multiclass_record.py',
)
RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}
INPUTS = ('rows', 'setup')
#: The parts of an answer, as the recorder projects it.
PARTS = (
    'outcome',
    'schema',
    'warnings',
    'payload',
    'result',
    'task',
    'labels',
)
#: Warning classes the reference raises from the runtime, not the client,
#: left out of the core's comparison, by vector: casting a double past
#: float32's range, pandas raises a ``RuntimeWarning`` (from
#: ``pandas/core/arrays/arrow/array.py``, measured with ``python-old``);
#: the core does not, on a regression target either (measured: the same
#: column named ``score``, refused ``INVALID_REQUEST`` at materialize by
#: both routes, the old one with the ``RuntimeWarning`` beside its
#: ``UserWarning``, the core with the ``UserWarning`` alone). The
#: ``python-old`` comparison keeps it. T2.14b (FX-1) gives it on the core
#: too: the exclusion holds only while T2.14b's key for it
#: (``contract/tests/test_t214b_wording.py``, the same column refused at
#: materialize) is pending; after, the core's warnings are held whole.
FX1_KEY = 'wording:warned:runtime-warning:wording.float32-classification'
RUNTIME_WARNINGS = (
    {'refuse.double-past-float32': ['RuntimeWarning']}
    if FX1_KEY
    in json.loads(
        (ROOT / 'contract/tests/t214b-pending.json').read_text(encoding='utf-8')
    )['pending']
    else {}
)
HELD = [v['id'] for v in RECORDED['vectors'] if v['id'] not in R.PREVIEW_RULES]

TRANSITION = re.compile(
    re.escape(R.REGISTER_REF)
    + r'(?P<glob>\S+) (?P<old>success|refusal [A-Z][A-Z_]*)'
    + r' -> (?P<new>success|refusal [A-Z][A-Z_]*)'
)


def approved_transition(vector_id: str, old: str, new: str) -> str | None:
    r"""The approved register entry whose ref names the vector with exactly
    the transition ``old -> new``, if any (a proposed one excuses
    nothing)."""
    document = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))
    for entry in document['entries']:
        if entry.get('status') != 'approved':
            continue
        for ref in entry.get('refs') or []:
            found = TRANSITION.fullmatch(ref) if isinstance(ref, str) else None
            if (
                found is not None
                and (found['old'], found['new']) == (old, new)
                and fnmatch.fnmatchcase(vector_id, found['glob'])
            ):
                return str(entry['id'])
    return None


def problem_of(vector: dict[str, Any], got: dict[str, Any]) -> str | None:
    r"""Why ``got`` fails the vector, or None when it is the reference's
    answer, or a difference an approved entry declares exactly."""
    want = dict(vector['reference'])
    runtime = RUNTIME_WARNINGS.get(vector['id'], [])
    want['warnings'] = [w for w in want['warnings'] if w not in runtime]
    parts = [part for part in PARTS if got.get(part) != want.get(part)]
    if not parts:
        return None
    old = R.outcome_class(want['outcome'])
    new = R.outcome_class(got['outcome'])
    if approved_transition(vector['id'], old, new) is not None:
        return None
    lines = [
        f'{vector["id"]}: {parts} differ from the reference; got '
        f'{got["outcome"]}, the reference {want["outcome"]}'
    ]
    if 'task' in parts:
        lines.append(f'  task: got {got["task"]}, the reference {want["task"]}')
    if 'labels' in parts:
        lines.append(
            f'  labels: got {_short(got["labels"])}, the reference '
            f'{_short(want["labels"])}'
        )
    if 'warnings' in parts:
        lines.append(
            f'  warnings: got {got["warnings"]}, the reference '
            f'{want["warnings"]}'
        )
    return '\n'.join(lines)


def _short(value: Any) -> str:
    text = json.dumps(value)
    return text if len(text) <= 1500 else text[:1500] + '...'


def rule_of(error: BaseException) -> str | None:
    details = getattr(error, 'details', None)
    rule = details.get('rule') if isinstance(details, dict) else None
    return rule if isinstance(rule, str) else None


def described_rules(vectors: list[dict[str, Any]]) -> dict[str, Any]:
    r"""Each vector's ``Schema.from_source`` on the core's route, in this
    process: ``(code, rule)`` of its refusal, or None when it describes."""
    import os

    import duckdb
    from sdm_client import Schema, StructuredDataError, _routing
    from sdm_connectors import DuckDBSource

    out: dict[str, Any] = {}
    previous = os.environ.get(_routing.ENVIRONMENT)
    os.environ[_routing.ENVIRONMENT] = ','.join(_routing.FROM_SOURCE_DUCKDB)
    try:
        for vector in vectors:
            connection = duckdb.connect()
            try:
                for statement in R.EN.shop() + vector['setup']:
                    connection.execute(statement)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        Schema.from_source(DuckDBSource(connection))
                    out[vector['id']] = None
                except StructuredDataError as error:
                    out[vector['id']] = (error.code, rule_of(error))
            finally:
                connection.close()
    finally:
        if previous is None:
            os.environ.pop(_routing.ENVIRONMENT, None)
        else:
            os.environ[_routing.ENVIRONMENT] = previous
    return out


@pytest.fixture(scope='module')
def core_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    r"""Every vector through ``python-core``, forced onto the core: one run."""
    tmp = tmp_path_factory.mktemp('t136g1-core')
    return R.run('python-core', R.VECTORS, tmp, core=True)


@pytest.fixture(scope='module')
def old_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    r"""Every vector through ``python-old``, the default routes: one run."""
    tmp = tmp_path_factory.mktemp('t136g1-old')
    return R.run('python-old', R.VECTORS, tmp)


@pytest.mark.parametrize('vector_id', [pytest.param(v, id=v) for v in HELD])
def test_python_core_answers_as_the_reference(
    vector_id: str, core_runs: dict[str, Any]
) -> None:
    vector = BY_ID[vector_id]
    observation = core_runs[vector_id]
    assert observation['core'] is not None, 'the core was not reached'
    problem = problem_of(vector, R.project(observation, vector))
    assert problem is None, problem


@pytest.mark.parametrize(
    'vector_id',
    [pytest.param(v['id'], id=v['id']) for v in RECORDED['vectors']],
)
def test_python_old_answers_as_the_reference(
    vector_id: str, old_runs: dict[str, Any]
) -> None:
    vector = BY_ID[vector_id]
    got = R.project(old_runs[vector_id], vector)
    differ = [p for p in PARTS if got.get(p) != vector['reference'].get(p)]
    assert differ == [], (
        f'{vector_id}: python-old differs from the reference in {differ}: '
        f'got {got["outcome"]}, the reference {vector["reference"]["outcome"]}'
    )


@pytest.fixture(scope='module')
def preview_rules() -> dict[str, Any]:
    return described_rules([BY_ID[v] for v in R.PREVIEW_RULES])


@pytest.mark.parametrize(
    'vector_id', [pytest.param(v, id=v) for v in R.PREVIEW_RULES]
)
def test_preview_vectors_are_refused_by_e92_or_answered_as_the_reference(
    vector_id: str,
    core_runs: dict[str, Any],
    preview_rules: dict[str, Any],
) -> None:
    vector = BY_ID[vector_id]
    got = R.project(core_runs[vector_id], vector)
    if problem_of(vector, got) is None:
        return
    assert got['outcome'] == {
        'result': 'refusal',
        'stage': 'from_source',
        'error': 'UnsupportedRequestError',
        'code': 'UNSUPPORTED_REQUEST',
    }, (vector_id, got['outcome'])
    assert preview_rules[vector_id] == (
        'UNSUPPORTED_REQUEST',
        R.PREVIEW_RULES[vector_id],
    ), (vector_id, preview_rules[vector_id])


# --- the vectors ---------------------------------------------------------------


def test_the_vectors_are_the_reference_s() -> None:
    assert RECORDED['format'] == R.FORMAT
    assert (
        RECORDED['recorded_by'] == 'contract/tools/float_multiclass_record.py'
    )
    assert RECORDED['cell'] == 'ref-pandas3'
    assert RECORDED['register_ref'] == R.REGISTER_REF
    assert RECORDED['template'] == 'contract/cases/s1/base.multiclass.json'
    assert RECORDED['question'] == R.QUESTION
    assert RECORDED['indices'] == R.INDICES
    assert RECORDED['preview_rules'] == R.PREVIEW_RULES
    assert RECORDED['payload_exclusions'] == [['metadata', 'request_id']]
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    assert [v['id'] for v in RECORDED['vectors']] == [
        v['id'] for v in R.VECTORS
    ]
    for kind in R.VECTORS:
        vector = BY_ID[kind['id']]
        for key in INPUTS:
            assert vector[key] == kind[key], (kind['id'], key)
    for vector in RECORDED['vectors']:
        why = R.register(vector['reference'])
        assert vector['register'] == {'required': bool(why), 'why': why}
        assert why == [], vector['id']


def test_each_vector_is_a_case_the_runner_takes() -> None:
    import tempfile

    runner = _module('run_python', ROOT / 'contract/runners/run_python.py')
    with tempfile.TemporaryDirectory() as tmp:
        for kind in R.VECTORS:
            for core in (False, True):
                case = R.case_for(kind, core=core)
                assert case['input']['setup'] == kind['setup']
                assert case['input']['pql'] == R.QUESTION
                assert case['input']['materialize'] == {'indices': R.INDICES}
                assert bool(case['routes']) == core
                assert case['until'] == 'predict'
            (Path(tmp) / f'{case["id"]}.json').write_text(
                json.dumps(case), encoding='utf-8'
            )
        loaded = runner.load_cases([Path(tmp)])
    assert len(loaded) == len(R.VECTORS)


def test_duckdb_holds_what_each_vector_names() -> None:
    for vector in RECORDED['vectors']:
        vid = vector['id']
        held = vector['duckdb']
        assert held['rows'] == vector['rows'], vid
        if 'float-' in vid or vid.startswith('float.'):
            assert held['f_type'] == 'FLOAT', vid
        else:
            assert held['f_type'] == 'DOUBLE', vid
        preview, past = held['preview'], held['past_preview']
        nans = 'nan' in vid
        zeros = 'zero' in vid
        if vid in R.PREVIEW_RULES:
            # In the first 1,000 rows: what the core's description refuses.
            if nans:
                assert preview['nans'] > 0, vid
            else:
                assert preview['positive_zeros'] > 0, vid
                assert preview['negative_zeros'] > 0, vid
        elif vid.endswith('-past-preview'):
            # Only past them: the description takes the column.
            assert preview['nans'] == 0, vid
            assert (
                min(preview['positive_zeros'], preview['negative_zeros']) == 0
            )
            if nans:
                assert past['nans'] > 0, vid
            else:
                assert past['positive_zeros'] > 0, vid
                assert past['negative_zeros'] > 0, vid
                assert preview['positive_zeros'] + preview['negative_zeros'] > 0
        else:
            assert preview['nans'] == past['nans'] == 0, vid
            if zeros:
                assert preview['negative_zeros'] > 0, vid
                assert preview['positive_zeros'] == 0, vid
        if 'null' in vid:
            assert preview['nulls'] > 0, vid
        if 'inf' in vid:
            assert preview['infinities'] > 0, vid
    values = {v['id']: v['duckdb'].get('values') for v in RECORDED['vectors']}
    # The predicted users (0 and 1) hold the null, NaN or infinity of the
    # ``-asked`` vectors; the others' are in the context.
    assert values['double.null-asked'][0] is None
    assert values['preview.double-nan-asked'][0] == 'nan'
    assert values['refuse.double-inf-asked'][0] == 'inf'
    assert values['double.null-in-context'][3] is None
    assert values['refuse.double-inf-in-context'][3] == 'inf'
    distinct = {
        v['id']: v['duckdb']['distinct_in_preview'] for v in RECORDED['vectors']
    }
    assert distinct['large.double-ten-classes'] == 10
    assert distinct['refuse.double-forty-classes'] == 40
    assert distinct['large.double-sixty-one-values'] == 61


def _float32_text(text: str) -> str:
    r"""``str`` of the float32 nearest the number ``text`` names, as a
    double: the spelling the reference's ``_class_label`` gives a class."""
    value = struct.unpack('<f', struct.pack('<f', float(text)))[0]
    return repr(value)


def test_the_reference_s_classes_are_its_float32_labels() -> None:
    r"""What the docstring says of the reference, read from its own
    answers: every multiclass vector sends one request whose label column
    is ``float32`` / ``categorical``; each label is spelled as ``str``
    spells its float32's double; the classes are the context's labels,
    distinct (the two zeros one), spelled as sent, the asked-for entities'
    own labels among them. Their order is ``pd.unique``'s over the
    training labels, which the sent rows do not show: it is held by the
    recorded answers themselves."""
    classes: dict[str, list[str]] = {}
    for vector in RECORDED['vectors']:
        vid = vector['id']
        answer = vector['reference']
        if vid.startswith('refuse.'):
            assert answer['outcome']['result'] == 'refusal', vid
            continue
        assert answer['outcome'] == {
            'result': 'success',
            'stage': 'predict',
            'error': None,
            'code': None,
        }, vid
        [task] = answer['task']
        if vid == 'large.double-sixty-one-values':
            assert task['kind'] == 'regression', vid
            continue
        assert task['kind'] == 'multiclass_classification', vid
        assert task['target']['dtype'] == 'float32', vid
        assert task['target']['column_name'] == R.LABEL, vid
        [labels] = answer['labels']
        [(path, column)] = labels.items()
        assert path == 'context.instance_table', vid
        assert column['declared'] == {
            'dtype': 'float32',
            'stype': 'categorical',
        }, vid
        assert column['nulls'] == 0, vid
        for cell in column['listed']:
            assert _float32_text(cell['number']) == cell['number'], (vid, cell)
        # The classes are distinct values (the zeros one), each spelled as
        # a float32's double, and hold every label sent; they may hold more:
        # the training labels of the entities asked for, whose examples
        # the sent context leaves out.
        found = task['target']['classes']
        for text in found:
            assert _float32_text(text) == text, (vid, text)
        assert len({float(c) for c in found}) == len(found), vid
        sent = {float(cell['number']) for cell in column['listed']}
        assert sent <= {float(c) for c in found}, vid
        classes[vid] = found
    # Not vacuous: whole values, float32 rounding, exponents both ways, a
    # negative zero kept as first met, one class from two doubles, a NaN
    # past the preview left out.
    assert classes['double.one-to-five'] == ['3.0', '5.0', '1.0', '2.0', '4.0']
    assert classes['float.one-to-five'] == classes['double.one-to-five']
    assert '0.10000000149011612' in classes['double.tenths']
    assert classes['float.tenths'] == classes['double.tenths']
    assert '1.0000000200408773e+20' in classes['double.large-spellings']
    assert '123456792.0' in classes['double.large-spellings']
    assert '9.999999747378752e-05' in classes['double.small-spellings']
    assert '-0.0' in classes['double.negative-zero']
    assert '0.0' in classes['large.double-negative-zero-past-preview']
    assert '-0.0' not in classes['large.double-negative-zero-past-preview']
    assert '-0.0' in classes['large.double-positive-zero-past-preview']
    assert len(classes['double.one-float32']) == 2
    assert classes['large.double-nan-past-preview'] == ['3.0', '2.0', '1.0']
    assert len(classes['large.double-ten-classes']) == 10
    # The labels of users 0 and 1 (asked for, 7.0 and 8.0) are classes,
    # though the sent context holds neither.
    small = BY_ID['double.small-spellings']['reference']['labels'][0][
        'context.instance_table'
    ]
    assert {'7.0', '8.0'} <= set(classes['double.small-spellings'])
    assert {'number': '7.0'} not in small['cells']
    assert {'number': '8.0'} not in small['cells']
    # The negative zero's other sign is still sent as a label.
    listed = BY_ID['large.double-negative-zero-past-preview']['reference'][
        'labels'
    ][0]['context.instance_table']['listed']
    assert {'number': '-0.0'} in listed and {'number': '0.0'} in listed
    # And the refusals the controls hold.
    refused = {
        v['id']: (
            v['reference']['outcome']['stage'],
            v['reference']['outcome']['code'],
        )
        for v in RECORDED['vectors']
        if v['id'].startswith('refuse.')
    }
    assert refused == {
        'refuse.double-inf-in-context': ('materialize', 'INVALID_REQUEST'),
        'refuse.float-negative-inf-in-context': (
            'materialize',
            'INVALID_REQUEST',
        ),
        'refuse.double-inf-asked': ('materialize', 'INVALID_REQUEST'),
        'refuse.double-past-float32': ('materialize', 'INVALID_REQUEST'),
        'refuse.double-forty-classes': ('predict', 'UNSUPPORTED_REQUEST'),
        'refuse.double-one-class': ('predict', 'UNSUPPORTED_REQUEST'),
        'refuse.double-no-class': ('predict', 'UNSUPPORTED_REQUEST'),
    }
    assert all(BY_ID[v]['reference']['payload'] is None for v in refused), (
        'a refused vector sent a request'
    )
    # The runtime's warnings left out of the core's comparison are the
    # reference's, and only where named.
    for vector in RECORDED['vectors']:
        runtime = RUNTIME_WARNINGS.get(vector['id'], [])
        warned = vector['reference']['warnings']
        assert all(w in warned for w in runtime), vector['id']
        assert ('RuntimeWarning' in warned) == bool(runtime), vector['id']
