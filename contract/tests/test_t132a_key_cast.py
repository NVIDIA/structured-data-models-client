# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32a's cast of the entities to the key's type on ``python-core``, held
to the reference's rule-table vectors.

    python -m pytest contract/tests/test_t132a_key_cast.py -q

Written by the package's independent test author from the brief, the
design's ledger row ("The entities cast to the key's type") and the
reference, before the implementation, and widened in round 2 from the
reviews of part 1. ``contract/key-cast-vectors.json`` is recorded from the
reference by ``contract/tools/key_cast_record.py``, whose ``observe`` this
test calls, so both routes are observed by the same code;
``sdm-testkit``'s ``tests/t132a_key_cast.rs`` holds ``rust-host`` to the
same vectors, so the two hosts agree through them.

- Exit "the entities' cast runs in the core, both hosts giving the same
  answers and error codes over the rule table's vectors": each vector's
  key type described and materialized forced onto the core
  (``SDM_CORE_ROUTES``: ``Schema.from_source``'s and ``materialize``'s
  DuckDB routes) with the vector's entities as a caller passes them. The
  answer must be the reference's: a request whose predicted rows carry the
  reference's entities, in order, or a refusal of the same class, code and
  stage (INVALID_REQUEST against INTERNAL_ERROR is part of it, E41(l)).
- A vector carrying ``expected`` is held to it first: derived by the
  recorder from the exact value, never through a double. A text or
  decimal entity of a float key is the nearest float of the key's width,
  ties to even (float tables holding both neighbours, both signs, and the
  float32 overflow boundary); an exponent past any key, or a wide decimal
  no key holds, is refused by name (INVALID_REQUEST naming ``indices``
  and why); a decimal 38 digits wide is held exactly.
- A vector marked ``register.required`` may differ from the reference only
  under an entry of ``docs/design/rust-core/register.yaml``, proposed or
  approved, whose ``refs`` name the vector and the transition it makes:
  ``contract/key-cast-vectors.json#<id or glob> <from> -> <to>``, each of
  ``success`` or ``refusal <CODE>``, from the reference's answer to the
  host's. An entry covers only the transition it declares. Every other
  vector is held to the reference exactly, and no vector may end in
  ``UNSUPPORTED_REQUEST`` (a rule the core does not take yet).
- A mix of kinds the host hands over as text (``disposition: as_text``):
  its answer is the host's own answer to that text given as text, and,
  where the reference answers that text otherwise than INTERNAL_ERROR, the
  reference's answer to it. That text is the recorder's ``host_text``, and
  a float's is the core's text of a float entity (T1.32g, T1.32a's
  carried P3): ``1e16`` is ``10000000000000000``, not Python's ``1e+16``.
  The ``utf8-float-text`` key holds those texts as keys, so a host writing
  a float otherwise reaches no key.
- A float entity alone, of a text key (T1.36, the final review's P2-2):
  the reference writes it as pyarrow casts a float to text (``1e16`` is
  ``1e+16``, ``1.5e-07`` is ``1.5e-7``, ``1e-06`` is ``0.000001``), and the
  owner's decision is to match it. ``utf8-float-pyarrow-text`` holds those
  texts as keys, ``utf8-numeric-ids`` holds 17-digit IDs (``1e16`` reaches
  none), ``utf8-numeric-ids-and-exponent`` the same and ``1e+16``; the
  floats alone on ``utf8-float-text`` must miss the core's old text. No
  register entry may excuse one of these vectors.
- The vectors themselves: recorded at the reference of
  ``contract/reference-env.txt``; every key type and input of the
  recorder present; each input read back exactly; the host's column, the
  derived expectations and the register marks re-derived (so none is
  edited by hand); decimals written exactly, whatever the caller's decimal
  context; and not vacuous.

Gating (``contract/tests/t132a_gate.py``): ``t132a-pending.json``'s
``pending["python-core"]`` lists the vectors that failed here when this was
written. Each still runs and must still fail; one that passes fails, naming
itself, until it is taken out of the list, which may only shrink (within
``authored["python-core"]``). The implementer removes ids as they pass and
changes nothing else of this file, the vectors or the recorder.
"""

from __future__ import annotations

import copy
import decimal
import fnmatch
import importlib.util
import json
import math
import re
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / 'contract/key-cast-vectors.json'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'


def _recorder() -> ModuleType:
    name = 'key_cast_record'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'contract/tools/key_cast_record.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _gate() -> ModuleType:
    name = 't132a_gate'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'contract/tests/t132a_gate.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _recorder()
G = _gate()
RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
KEYS = {key['id']: key for key in RECORDED['keys']}


#: A ref naming vectors and the transition it covers.
TRANSITION = re.compile(
    re.escape('contract/key-cast-vectors.json#')
    + r'(?P<glob>\S+) (?P<old>success|refusal [A-Z][A-Z_]*)'
    + r' -> (?P<new>success|refusal [A-Z][A-Z_]*)'
)


def transitions() -> tuple[list[tuple[str, str, str, str]], list[str]]:
    r"""Each vector ref of the register's proposed or approved entries, as
    (entry, glob, from, to), and the refs naming vectors that declare no
    transition (malformed)."""
    document = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))
    out: list[tuple[str, str, str, str]] = []
    malformed: list[str] = []
    for entry in document['entries']:
        if entry.get('status') not in ('proposed', 'approved'):
            continue
        for ref in entry.get('refs') or []:
            if not isinstance(ref, str) or not ref.startswith(R.REGISTER_REF):
                continue
            found = TRANSITION.fullmatch(ref)
            if found is None:
                malformed.append(f'{entry["id"]}: {ref}')
                continue
            out.append(
                (str(entry['id']), found['glob'], found['old'], found['new'])
            )
    return out, malformed


def covered(vector_id: str, old: str, new: str) -> str | None:
    r"""The entry whose ref names the vector with exactly this transition."""
    for entry, glob, frm, to in transitions()[0]:
        if (frm, to) == (old, new) and fnmatch.fnmatchcase(vector_id, glob):
            return entry
    return None


def problem_of(
    vector: dict[str, Any],
    got: dict[str, Any],
    own_text: dict[str, Any] | None = None,
) -> str | None:
    r"""Why ``got`` fails the vector, or None: its derived expectation and
    disposition held, then the reference's answer, or a marked vector's
    difference under an entry declaring that transition. ``own_text`` is
    the host's answer to the vector's text column given as text."""
    vid = vector['id']
    reference = vector['reference']
    public = reference['public']
    if got.get('code') == 'UNSUPPORTED_REQUEST':
        return f'{vid}: refused by a rule the core does not take yet: {got}'
    expected = vector.get('expected')
    if expected is not None and not R.matches(expected, got):
        return f'{vid}: got {got}, the derived expectation {expected}'
    if vector['register'].get('disposition') == 'as_text':
        if own_text is not None and not R.same(got, own_text):
            return (
                f'{vid}: got {got}, but {own_text} for the same text given '
                'as text'
            )
        text = reference['as_text']
        if R.outcome_class(text) != 'refusal INTERNAL_ERROR' and not R.same(
            got, text
        ):
            return (
                f"{vid}: got {got}, the reference's answer to its text {text}"
            )
    if R.same(got, public):
        return None
    if not vector['register']['required']:
        return f'{vid}: got {got}, the reference {public} (held exactly)'
    old, new = R.outcome_class(public), R.outcome_class(got)
    if covered(vid, old, new) is None:
        return (
            f'{vid}: got {got}, the reference {public}: no register entry '
            f'names it with the transition {old} -> {new}'
        )
    return None


def _cases() -> list[Any]:
    return [
        pytest.param(vector, id=vector['id']) for vector in RECORDED['vectors']
    ]


@pytest.fixture
def core_route(monkeypatch: pytest.MonkeyPatch) -> None:
    from sdm_client import _routing

    monkeypatch.setenv(
        _routing.ENVIRONMENT,
        ','.join(_routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB),
    )


@pytest.mark.parametrize('vector', _cases())
def test_python_core_casts_the_entities_as_the_reference(
    vector: dict[str, Any], core_route: None
) -> None:
    def check() -> None:
        key = KEYS[vector['key']]
        values = [R.decode(tagged) for tagged in vector['input']]
        expected = vector.get('expected') or {}
        fields = ('field', 'reason') if expected.get('named') else ()
        got = R.observe(key, values, fields)
        own_text = None
        if vector['register'].get('disposition') == 'as_text':
            own_text = R.observe(key, R.as_text(values))
        problem = problem_of(vector, got, own_text)
        assert problem is None, problem

    G.held('python-core', vector['id'], check)


# --- the vectors ---------------------------------------------------------------


def test_the_vectors_are_the_reference_s() -> None:
    assert RECORDED['format'] == R.FORMAT
    assert RECORDED['recorded_by'] == 'contract/tools/key_cast_record.py'
    assert RECORDED['cell'] == 'ref-pandas3'
    assert RECORDED['register_ref'] == R.REGISTER_REF
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    # Every key type and every input the recorder names, each once.
    keys = R.all_keys()
    assert [key['id'] for key in RECORDED['keys']] == [k['id'] for k in keys]
    for key in keys:
        assert KEYS[key['id']]['setup'] == key['setup']
    wanted = [f'{key["id"]}.{name}' for key in keys for name in R.inputs(key)]
    assert [v['id'] for v in RECORDED['vectors']] == wanted


def test_each_input_reads_back_exactly() -> None:
    for vector in RECORDED['vectors']:
        values = [R.decode(tagged) for tagged in vector['input']]
        assert [R.encode(v) for v in values] == vector['input'], vector['id']


def test_the_column_expectations_and_marks_are_the_recorder_s() -> None:
    # Derived, never edited: the host's natural column (pyarrow at the
    # reference cell's pin), each derived expectation, and the register
    # marks from the recorded answers.
    expectations = {
        f'{key["id"]}.{name}': expected
        for key in R.all_keys()
        for name, (_, expected) in R.derived(key).items()
    }
    vectors = copy.deepcopy(RECORDED['vectors'])
    for vector in vectors:
        values = [R.decode(tagged) for tagged in vector['input']]
        assert R.host_arrow(values) == vector['host_arrow'], vector['id']
        assert vector.get('expected') == expectations.get(vector['id'])
        del vector['register']
    R.derive_marks(vectors)
    for derived, recorded in zip(vectors, RECORDED['vectors'], strict=True):
        assert derived['register'] == recorded['register'], derived['id']


#: Decimals 38 and 76 digits wide, at several scales, signed: the exact
#: unscaled digits each Arrow column must carry, written here from the
#: text by hand rules (digits shifted by the scale), never by arithmetic.
WIDE = [
    '12345678901234567890123456789012345678',
    '-99999999999999999999999999999999999999',
    '1234567890123456789012345678.0123456789',
    '-1.2345678901234567890123456789012345678',
    '0.00000000000000000000000000000000000001',
    '1' * 76,
    '-' + '9' * 40 + '.' + '1' * 36,
]


def _digits(text: str, scale: int) -> str:
    sign = '-' if text.startswith('-') else ''
    whole, _, fraction = text.lstrip('-').partition('.')
    digits = (whole + fraction.ljust(scale, '0')).lstrip('0') or '0'
    return sign + digits if digits != '0' else '0'


def _hostile() -> Any:
    context = decimal.Context(prec=3, Emax=5, Emin=-5)
    context.traps[decimal.Inexact] = True
    context.traps[decimal.Rounded] = True
    return decimal.localcontext(context)


def _wide(write: Any) -> list[str]:
    r"""What ``write`` (a value to its scalar) gets wrong of ``WIDE`` in a
    hostile decimal context: precision 3, rounding trapped."""
    import pyarrow as pa

    wrong = []
    for text in WIDE:
        value = Decimal(text)
        kind = pa.array([value]).type
        want = {
            't': 'd128' if pa.types.is_decimal128(kind) else 'd256',
            'p': kind.precision,
            's': kind.scale,
            'v': _digits(text, kind.scale),
        }
        try:
            with _hostile():
                got = write(value)
        except Exception as error:  # noqa: BLE001 - reported
            got = f'raised {type(error).__name__}'
        if got != want:
            wrong.append(f'{text}: {got}, not {want}')
    return wrong


def test_the_recorder_writes_wide_decimals_exactly() -> None:
    assert _wide(lambda value: R.host_arrow([value])['values'][0]) == []


def test_the_runner_writes_wide_decimals_exactly(
    request: pytest.FixtureRequest,
) -> None:
    # The Python runner's record of the entities the host sent
    # (run_python.py's _scalars), held to the same exact digits.
    import pyarrow as pa

    run_python = _module('run_python', ROOT / 'contract/runners/run_python.py')

    def write(value: Decimal) -> Any:
        return run_python._scalars(pa.table({'indices': pa.array([value])}))[0]

    def check() -> None:
        assert _wide(write) == []

    G.held('exits', f'py:{request.node.name}', check)


def test_the_recorded_decimal_columns_hold_in_a_hostile_context() -> None:
    with _hostile():
        for vector in RECORDED['vectors']:
            if 'Decimal' not in json.dumps(vector['input']):
                continue
            values = [R.decode(tagged) for tagged in vector['input']]
            assert R.host_arrow(values) == vector['host_arrow'], vector['id']


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_the_derived_float_rule_is_exact() -> None:
    # The derivation itself: nearest at the width, ties to even, by
    # integer arithmetic, against values worked by hand.
    from fractions import Fraction

    one, ulp = Fraction(1), Fraction(1, 2**23)
    tiny = Fraction(1, 10**30)
    assert R.nearest(one + ulp / 2 + tiny, 24) == one + ulp
    assert R.nearest(one + ulp / 2 - tiny, 24) == one
    assert R.nearest(one + ulp / 2, 24) == one  # tie: 1.0 is even
    assert R.nearest(Fraction(2**24 + 3), 24) == 2**24 + 4  # tie: up to even
    assert R.nearest(Fraction(2**53 + 1), 53) == 2**53
    assert R.nearest(-(one + ulp / 2 + tiny), 24) == -(one + ulp)
    assert R.nearest(R.F32_MAX + 2**103 - 1, 24) == R.F32_MAX
    assert R.nearest(R.F32_MAX + 2**103, 24) is None  # tie past the largest
    # Codex's repro: the text below gives 1.00000011920928955078125.
    exact = Fraction(Decimal('1.000000059604644775390625000001'))
    assert R.float_bits(R.nearest(exact, 24), 24) == '3f800001'


def test_the_host_text_of_a_float_is_the_core_s() -> None:
    # The recorder's text of a float in a mix, against values worked by
    # hand from Rust's ``{}`` of an f64 (keys.rs's Entity::Float): the
    # shortest digits, no exponent, a whole one with no point, a tie of
    # the shortest digits away from zero; in a hostile decimal context.
    worked = [
        (1e16, '10000000000000000'),
        (1.0, '1'),
        (-0.0, '-0'),
        (0.0, '0'),
        (2.5, '2.5'),
        (1.5e-07, '0.00000015'),
        (1e22, '10000000000000000000000'),
        (-1234.5, '-1234.5'),
        (1059438285926254.2, '1059438285926254.3'),
        (-1059438285926254.2, '-1059438285926254.3'),
        (1059438285926254.8, '1059438285926254.8'),
    ]
    with _hostile():
        for value, text in worked:
            assert R.float_text(value) == text, repr(value)
            assert R.host_text(value) == text, repr(value)
    assert R.host_text('1e16') == '1e16'
    assert R.host_text(7) == '7'
    assert R.as_text([1e16, None, '2']) == ['10000000000000000', None, '2']


#: T1.36 (the final review's P2-2): the key types and vectors where a float
#: entity's text decides the answer. The owner's decision is to write it as
#: the reference does, so no register entry may excuse one.
SPELLING_KEYS = (
    'utf8-float-pyarrow-text',
    'utf8-numeric-ids',
    'utf8-numeric-ids-and-exponent',
)
SPELLING_ON_THE_CORE_S_TEXT = (
    'utf8-float-text.float.*',
    'utf8-float-text.numpy.float32-*',
)


def _spelling(vector_id: str) -> bool:
    return vector_id.split('.')[0] in SPELLING_KEYS or any(
        fnmatch.fnmatchcase(vector_id, glob)
        for glob in SPELLING_ON_THE_CORE_S_TEXT
    )


def _floats_only(values: list[Any]) -> bool:
    import numpy as np

    present = [v for v in values if not R._missing(v)]
    return bool(present) and all(
        isinstance(v, (float, np.floating)) for v in present
    )


def test_the_reference_writes_a_float_entity_as_pyarrow_casts_it() -> None:
    # The rule a float entity of a text key is held to: the reference's key
    # is an ArrowDtype of text, and pandas casts floats to it with pyarrow
    # (pa.array, then cast to text), so each recorded cast of float
    # entities to a text key is pyarrow's cast of the same column, a NaN
    # missing; and pyarrow writes these, worked by hand: the shortest
    # digits of the float's own width, plain from 1e-6 to below 1e10,
    # otherwise with an exponent of its sign and unpadded digits.
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    def text(values: list[Any]) -> list[str | None]:
        return pc.cast(
            pa.array(values, from_pandas=True), pa.string()
        ).to_pylist()

    worked = [
        (1e16, '1e+16'),
        (1.2e15, '1.2e+15'),
        (1e15, '1e+15'),
        (1e21, '1e+21'),
        (123456789012345680.0, '1.2345678901234568e+17'),
        (1e10, '1e+10'),
        (1e9, '1000000000'),
        (1.5e-07, '1.5e-7'),
        (1e-06, '0.000001'),
        (1e-07, '1e-7'),
        (1e-05, '0.00001'),
        (0.0001234, '0.0001234'),
        (2.5, '2.5'),
        (3.0, '3'),
        (-0.0, '-0'),
        (5e-324, '5e-324'),
        (1.7976931348623157e308, '1.7976931348623157e+308'),
        (1059438285926254.2, '1.0594382859262542e+15'),
        (math.inf, 'inf'),
        (-math.inf, '-inf'),
        (float(np.float32(0.1)), '0.10000000149011612'),
        (np.float32(0.1), '0.1'),
        (np.float32(1e16), '1e+16'),
        (np.float32(2**24 + 1), '16777216'),
        (np.float32(1.5e-07), '1.5e-7'),
    ]
    for value, want in worked:
        assert text([value]) == [want], repr(value)
    assert text([1e16, float('nan')]) == ['1e+16', None]
    checked = 0
    for vector in RECORDED['vectors']:
        key = vector['key']
        if not (key == 'utf8' or key.startswith('utf8-')):
            continue
        values = [R.decode(tagged) for tagged in vector['input']]
        cast = vector['reference']['cast']
        if not _floats_only(values) or not cast or 'values' not in cast:
            continue
        want_cells = [None if t is None else ['s', t] for t in text(values)]
        assert cast['values'] == want_cells, vector['id']
        checked += 1
    assert checked >= 40, checked


def test_no_entry_excuses_a_float_entity_s_text() -> None:
    # The vectors exist, the reference both refuses and takes a float by
    # its text among them, and no register entry, proposed or approved,
    # names one: each is held to the reference exactly.
    spelling = [v for v in RECORDED['vectors'] if _spelling(v['id'])]
    assert len(spelling) >= 50, len(spelling)
    outcomes = {v['reference']['public']['outcome'] for v in spelling}
    assert outcomes == {'success', 'refusal'}
    by_id = {v['id']: v for v in spelling}
    assert (
        by_id['utf8-numeric-ids.float.1e16']['reference']['public']['outcome']
        == 'refusal'
    )
    assert by_id['utf8-numeric-ids-and-exponent.float.1e16']['reference'][
        'public'
    ] == {'outcome': 'success', 'entities': [['s', '1e+16']]}

    def excused(found: list[tuple[str, str, str, str]]) -> list[Any]:
        return [
            (entry, vector['id'])
            for entry, glob, _, _ in found
            for vector in spelling
            if fnmatch.fnmatchcase(vector['id'], glob)
        ]

    assert excused(transitions()[0]) == []
    # Fails closed on a seeded entry naming them.
    seeded = [('SEEDED-1', 'utf8-numeric-ids.*', 'refusal', 'success')]
    assert len(excused(seeded)) == 7


def test_the_vectors_are_not_vacuous() -> None:
    answers = [v['reference']['public'] for v in RECORDED['vectors']]
    outcomes = [(a['outcome'], a.get('code')) for a in answers]
    assert outcomes.count(('success', None)) > 100
    assert outcomes.count(('refusal', 'INVALID_REQUEST')) > 100
    assert outcomes.count(('refusal', 'INTERNAL_ERROR')) > 100
    assert set(outcomes) == {
        ('success', None),
        ('refusal', 'INVALID_REQUEST'),
        ('refusal', 'INTERNAL_ERROR'),
    }
    marked = [v for v in RECORDED['vectors'] if v['register']['required']]
    assert 0 < len(marked) < len(RECORDED['vectors'])
    for vector in RECORDED['vectors']:
        assert vector['register']['required'] == bool(vector['register']['why'])
    # The reference's own cast was observed wherever its engine cast.
    assert all(v['reference']['cast'] for v in RECORDED['vectors'])
    # Round 2: derived expectations of each kind, and the four mixed
    # float-and-text vectors whose text the reference takes, marked with
    # their disposition.
    expected = [v['expected'] for v in RECORDED['vectors'] if 'expected' in v]
    kinds = {e['entities'][0][0] for e in expected if e['outcome'] == 'success'}
    assert kinds == {'f32', 'f64', 'dec'}
    assert sum(1 for e in expected if e['outcome'] == 'refusal') > 20
    by_id = {v['id']: v for v in RECORDED['vectors']}
    for key in ('float32', 'float64', 'decimal128-18-0', 'decimal128-38-0'):
        mark = by_id[f'{key}.mixed.float-text']['register']
        assert mark['required'] and mark['disposition'] == 'as_text', key
    # T1.32g: each float whose text Python writes otherwise than the core
    # reaches a key of utf8-float-text only by the core's text.
    for name in ('1e16', 'tiny', 'tie'):
        vector = by_id[f'utf8-float-text.mixed.float-{name}-text']
        assert vector['register']['disposition'] == 'as_text', name
        assert vector['reference']['as_text']['outcome'] == 'success', name
    # Each key type has successes and refusals of both codes, or fails
    # everywhere with INTERNAL_ERROR (the reference's decimal and
    # large_string keys).
    for key in KEYS:
        if any(key == k['id'] for k in R.ROUND_2):
            continue
        mine = {
            (
                v['reference']['public']['outcome'],
                v['reference']['public'].get('code'),
            )
            for v in RECORDED['vectors']
            if v['key'] == key
        }
        assert ('success', None) in mine or mine == {
            ('refusal', 'INTERNAL_ERROR'),
            ('refusal', 'INVALID_REQUEST'),
        }, key


# --- the gating ----------------------------------------------------------------


def test_the_pending_lists_only_shrink() -> None:
    assert G.only_shrinks() == []
    ids = {v['id'] for v in RECORDED['vectors']}
    for host in ('python-core', 'rust-host'):
        assert G.pending(host) <= ids, f'{host}: ids that are no vector'


def test_the_gate_holds_a_pending_test_to_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A listed test passes only while its check fails, and an unlisted one
    # only while its check passes.
    def passing() -> None:
        pass

    def failing() -> None:
        raise AssertionError('seeded')

    monkeypatch.setattr(G, 'pending', lambda group: {'seeded.listed'})
    G.held('python-core', 'seeded.listed', failing)
    with pytest.raises(AssertionError, match='passes now'):
        G.held('python-core', 'seeded.listed', passing)
    G.held('python-core', 'seeded.unlisted', passing)
    with pytest.raises(AssertionError, match='seeded'):
        G.held('python-core', 'seeded.unlisted', failing)


def test_a_difference_is_held_unless_marked_and_its_transition_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Fail closed: the comparison itself, on seeded differences.
    vectors = RECORDED['vectors']
    plain = next(
        v
        for v in vectors
        if not v['register']['required'] and 'expected' not in v
    )
    marked = next(
        v
        for v in vectors
        if v['register']['required']
        and v['register']['disposition'] is None
        and v['reference']['public']['outcome'] == 'success'
    )
    other = {
        'outcome': 'refusal',
        'stage': 'materialize',
        'error': 'StructuredDataError',
        'code': 'NOT_THE_REFERENCE_S',
    }
    me = sys.modules[__name__]
    assert problem_of(plain, plain['reference']['public']) is None
    assert problem_of(plain, other) is not None
    unported = dict(other, code='UNSUPPORTED_REQUEST')
    assert problem_of(marked, unported) is not None
    monkeypatch.setattr(me, 'transitions', lambda: ([], []))
    assert problem_of(marked, other) is not None
    # An entry covers only the transition it declares.
    wrong = [('SEEDED-1', marked['id'], 'success', 'refusal INVALID_REQUEST')]
    monkeypatch.setattr(me, 'transitions', lambda: (wrong, []))
    assert problem_of(marked, other) is not None
    right = [
        ('SEEDED-1', marked['id'], 'success', 'refusal NOT_THE_REFERENCE_S')
    ]
    monkeypatch.setattr(me, 'transitions', lambda: (right, []))
    assert problem_of(marked, other) is None
    # An unmarked vector is never excused by an entry.
    anything = [('SEEDED-2', '*', 'success', 'refusal NOT_THE_REFERENCE_S')]
    monkeypatch.setattr(me, 'transitions', lambda: (anything, []))
    assert problem_of(plain, other) is not None
    # A derived expectation binds before any entry.
    derived = next(v for v in vectors if 'expected' in v)
    assert problem_of(derived, other) is not None
    # An as_text disposition binds to the host's own answer to its text.
    mixed = next(
        v for v in vectors if v['register']['disposition'] == 'as_text'
    )
    text = mixed['reference']['as_text']
    assert problem_of(mixed, text, own_text=other) is not None


def test_the_transition_syntax_reads_and_refuses() -> None:
    good = 'contract/key-cast-vectors.json#*.time.* refusal INTERNAL_ERROR -> refusal INVALID_REQUEST'
    found = TRANSITION.fullmatch(good)
    assert found is not None and found['glob'] == '*.time.*'
    for bad in (
        'contract/key-cast-vectors.json#*.time.*',
        'contract/key-cast-vectors.json#*.time.* refusal -> success',
        'contract/key-cast-vectors.json#*.time.* success->success',
    ):
        assert TRANSITION.fullmatch(bad) is None, bad


def test_every_vector_ref_names_its_transition(
    request: pytest.FixtureRequest,
) -> None:
    def check() -> None:
        found, malformed = transitions()
        assert malformed == [], malformed
        # Not vacuous: each ref names a vector the reference answers as its
        # transition starts.
        for entry, glob, old, _ in found:
            assert any(
                fnmatch.fnmatchcase(v['id'], glob)
                and R.outcome_class(v['reference']['public']) == old
                for v in RECORDED['vectors']
            ), (entry, glob, old)

    G.held('exits', f'py:{request.node.name}', check)
