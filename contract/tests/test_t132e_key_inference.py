# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32e part 1's key rules on ``python-core``, held to the reference's
vectors.

    python -m pytest contract/tests/test_t132e_key_inference.py -q

Written by the package's independent test author, before the
implementation, from the owner's ruling of 2026-10-13 (port the
reference's key inference rules now: never refuse a whole table because of
one null in a key) and the reference. ``contract/key-inference-vectors.json``
is recorded from the reference by ``contract/tools/key_inference_record.py``,
whose ``VECTORS``, ``run`` and ``project`` this test calls, so both routes
are observed and projected by the same code; ``sdm-testkit``'s
``tests/t132e_key_inference.rs`` holds ``rust-host`` to the same vectors.

- The ruling, over generated ``accounts`` and ``payments`` tables of 200,
  1,000 and 5,000 rows (at 5,000 every repeat and null after the first
  1,000 rows, so the rule holds over the whole table, not a preview): an
  inferred key's distinct share at 100%, 99.5%, exactly 99%, 98.9% and
  50%; 1, 2 and 10 nulls and every row null; BIGINT, VARCHAR and
  DECIMAL(18, 0) keys; keys named in ``primary_keys`` clean, with a null,
  with repeats within and past 1%, and text and decimal ones holding a
  null; a foreign key pointing at a table whose key the reference
  declined; and a keyless fact table read by foreign key at
  ``materialize`` (the core's ``keyless_neighbors``). Each vector runs
  through ``run_python.py --mode python-core``'s ``Schema.from_source``
  and ``materialize``, forced onto the core's DuckDB routes. Its public
  answer must be the reference's: the outcome (result, stage, class,
  code), both tables as the schema describes them (each column's dtype
  and stype, the key chosen or None, the time column, the duplicate-key
  count) and the links, and the warning classes.
- The ruling's exit: on every vector whose varied key holds a null and
  whose ``Schema.from_source`` the reference answered, the core answers it
  too, with the reference's key for that table (kept with its nulls, or
  declined): no table is refused for a null in its key
  (:func:`test_a_key_holding_nulls_refuses_no_table`).
- Any difference only under an approved register entry naming the vector
  with its transition (T1.32a's form,
  ``contract/key-inference-vectors.json#<id or glob> <from> -> <to>``); a
  proposed entry excuses nothing. ``register.required`` is derived (the
  reference fails unnamed or with INTERNAL_ERROR) and excuses nothing by
  itself. At b413e9ad it marked ``nulls-1-decimal.200``, a decimal key
  with a null sampled at ``materialize`` (INTERNAL_ERROR); at 6298913c
  (main's eb6c68eb reads the decimal key as an integer) the reference
  materializes it as it does its integer twin, so no vector is marked.
  ``nulls-1-text.200``, the same with a text key, the reference refuses
  with INVALID_REQUEST at ``materialize``: a named code, so not marked.

P1B-40, option (b) (owner, 2026-10-13, relayed):
``nulls-1-text.200`` (INVALID_REQUEST: a mixed text key reached by
sampling) must materialize on the core, describing the tables as the
reference did, with the twin's warnings (``nulls-1.200``) and exactly one
more (:func:`test_option_b_materializes_with_one_more_warning`).
From T2.0c (exact decimals) a null DECIMAL key beside values
(``nulls-1-decimal.200``) is kept as the reference and its integer twin
keep it, with no extra warning
(:func:`test_a_null_decimal_key_is_kept_as_its_integer_twin_s`).

Gating (``contract/tests/t132e_gate.py``): ``t132e-pending.json``'s
``pending["key-python-core"]`` and the exit's ``py:`` ids in
``pending["exits"]`` list the tests that failed here when this was
written; each still runs and must still fail until it is taken out, and
the lists only shrink. The implementer changes nothing else of this file,
the vectors or the recorder.
"""

from __future__ import annotations

import copy
import fnmatch
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / 'contract/key-inference-vectors.json'
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
    'key_inference_record', ROOT / 'contract/tools/key_inference_record.py'
)
G = _module('t132e_gate', ROOT / 'contract/tests/t132e_gate.py')
RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}


def key_of(answer: dict[str, Any], table: str) -> Any:
    r"""The key the answer's schema gives ``table``."""
    for entry in answer['schema']['tables']:
        if entry['name'] == table:
            return entry['primary_key']
    raise AssertionError(f'{table} is not described')


#: The exit's vectors: the varied key holds a null, and the reference
#: answered ``Schema.from_source`` (derived from the vectors).
NULL_KEYS = [
    vector['id']
    for vector in RECORDED['vectors']
    if vector['key']['nulls'] > 0 and vector['reference']['schema'] is not None
]


#: A vector ref and the transition it declares.
TRANSITION = re.compile(
    re.escape(R.REGISTER_REF)
    + r'(?P<glob>\S+) (?P<old>success|refusal [A-Z][A-Z_]*)'
    + r' -> (?P<new>success|refusal [A-Z][A-Z_]*)'
)


def transitions(
    document: dict[str, Any] | None = None,
) -> tuple[list[tuple[str, str, str, str, str]], list[str]]:
    r"""Each vector ref of the register's proposed or approved entries, as
    (entry, status, glob, from, to), and the refs naming vectors that
    declare no transition in the form (malformed)."""
    if document is None:
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


def covered(
    vector_id: str, old: str, new: str
) -> tuple[str | None, str | None]:
    r"""The approved entry whose ref names the vector with exactly the
    transition ``old -> new``, and a proposed one that would, if any."""
    approved = proposed = None
    for entry, status, glob, frm, to in transitions()[0]:
        if (frm, to) != (old, new) or not fnmatch.fnmatchcase(vector_id, glob):
            continue
        if status == 'approved':
            approved = approved or entry
        else:
            proposed = proposed or entry
    return approved, proposed


def problem_of(vector: dict[str, Any], got: dict[str, Any]) -> str | None:
    r"""Why ``got`` fails the vector, or None when it is the reference's
    answer, or a difference an approved entry declares exactly."""
    want = vector['reference']
    if got == want:
        return None
    old = R.outcome_class(want['outcome'])
    new = R.outcome_class(got['outcome'])
    approved, proposed = covered(vector['id'], old, new)
    if approved is not None:
        return None
    why = (
        f'{proposed} declares it but is only proposed'
        if proposed is not None
        else 'no approved entry declares it'
    )
    return (
        f'{vector["id"]}: got {got}, the reference {want}; the transition '
        f'{old} -> {new}: {why}'
    )


@pytest.fixture(scope='module')
def core_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    r"""Every vector through ``python-core``, forced onto the core: one run."""
    from sdm_client import _routing

    routes = sorted(
        {
            *_routing.FROM_SOURCE_DUCKDB,
            *_routing.FROM_SOURCE_DUCKDB_TABLES,
            *_routing.MATERIALIZE_DUCKDB,
        }
    )
    tmp = tmp_path_factory.mktemp('t132e-key-vectors')
    return R.run('python-core', R.VECTORS, tmp, routes=routes)


@pytest.mark.parametrize(
    'vector_id',
    [pytest.param(v['id'], id=v['id']) for v in RECORDED['vectors']],
)
def test_python_core_keys_the_tables_as_the_reference(
    vector_id: str, core_runs: dict[str, Any]
) -> None:
    vector = BY_ID[vector_id]

    def check() -> None:
        observation = core_runs[vector_id]
        assert observation['core'] is not None, 'the core was not reached'
        got = R.project(observation)
        problem = problem_of(vector, got)
        assert problem is None, problem

    G.held('key-python-core', vector_id, check)


@pytest.mark.parametrize(
    'vector_id', [pytest.param(i, id=i) for i in NULL_KEYS]
)
def test_a_key_holding_nulls_refuses_no_table(
    vector_id: str, core_runs: dict[str, Any]
) -> None:
    r"""The ruling: a null in a key refuses no table; the key is kept with
    its nulls or declined, as the reference keeps or declines it."""
    vector = BY_ID[vector_id]
    table = vector['table']

    def check() -> None:
        observation = core_runs[vector_id]
        assert observation['core'] is not None, 'the core was not reached'
        got = R.project(observation)
        assert got['schema'] is not None, got['outcome']
        assert key_of(got, table) == key_of(vector['reference'], table)

    G.held(
        'exits',
        f'py:test_a_key_holding_nulls_refuses_no_table[{vector_id}]',
        check,
    )


#: P1B-40, option (b) (owner, 2026-10-13, relayed): sampling reaches a
#: table whose key mixes a null with real text values, which the reference
#: refuses (INVALID_REQUEST); the core leaves the null-key rows out and
#: succeeds, with one warning per table. Each vector and its twin, the same
#: layout with an integer key, which the reference answers. Option (b)
#: stays for text keys only (T2.0c's exact decimals).
OPTION_B = {
    'nulls-1-text.200': 'nulls-1.200',
}
#: A null DECIMAL key beside values: at 6298913c (main's eb6c68eb) the
#: reference keeps it exactly as its integer twin does, the null-key row's
#: warning included and no other, and so does the core with exact decimals
#: (T2.0c), with no option (b) warning.
DECIMAL_KEPT = {
    'nulls-1-decimal.200': 'nulls-1.200',
}


@pytest.mark.parametrize('vector_id', [pytest.param(i, id=i) for i in OPTION_B])
def test_option_b_materializes_with_one_more_warning(
    vector_id: str, core_runs: dict[str, Any]
) -> None:
    r"""The core succeeds at ``materialize`` where the reference refuses
    (the approved divergence), describes the tables as the reference did
    before it refused, and gives the twin's warnings and exactly one more:
    the rows left out."""
    vector = BY_ID[vector_id]
    twin = BY_ID[OPTION_B[vector_id]]['reference']

    def check() -> None:
        observation = core_runs[vector_id]
        assert observation['core'] is not None, 'the core was not reached'
        got = R.project(observation)
        assert got['outcome'] == {
            'result': 'success',
            'stage': 'materialize',
            'error': None,
            'code': None,
        }, got['outcome']
        assert got['schema'] == vector['reference']['schema']
        extra = Counter(got['warnings']) - Counter(twin['warnings'])
        missing = Counter(twin['warnings']) - Counter(got['warnings'])
        assert sum(extra.values()) == 1 and not missing, got['warnings']

    G.held(
        'exits',
        f'py:test_option_b_materializes_with_one_more_warning[{vector_id}]',
        check,
    )


@pytest.mark.parametrize(
    'vector_id', [pytest.param(i, id=i) for i in DECIMAL_KEPT]
)
def test_a_null_decimal_key_is_kept_as_its_integer_twin_s(
    vector_id: str, core_runs: dict[str, Any]
) -> None:
    r"""The core keeps a null DECIMAL key beside values as the reference
    and the integer twin do: success at ``materialize``, the reference's
    description, and exactly the twin's warnings, no option (b) warning."""
    vector = BY_ID[vector_id]
    twin = BY_ID[DECIMAL_KEPT[vector_id]]['reference']
    observation = core_runs[vector_id]
    assert observation['core'] is not None, 'the core was not reached'
    got = R.project(observation)
    assert got['outcome'] == twin['outcome'], got['outcome']
    assert got['schema'] == vector['reference']['schema']
    assert Counter(got['warnings']) == Counter(twin['warnings']), got[
        'warnings'
    ]


def test_option_b_s_vectors_are_the_reference_s_refusals() -> None:
    for vector_id in OPTION_B:
        outcome = BY_ID[vector_id]['reference']['outcome']
        assert outcome['result'] == 'refusal'
        assert outcome['stage'] == 'materialize'
        assert outcome['code'] == 'INVALID_REQUEST'
        assert BY_ID[vector_id]['key']['kind'] == 'text'
    # The decimal one the reference answers exactly as its integer twin.
    for vector_id, twin_id in DECIMAL_KEPT.items():
        assert BY_ID[vector_id]['reference'] == BY_ID[twin_id]['reference']
        assert BY_ID[vector_id]['key']['kind'] == 'decimal'
    for vector_id, twin_id in {**OPTION_B, **DECIMAL_KEPT}.items():
        twin = BY_ID[twin_id]
        assert twin['reference']['outcome']['result'] == 'success'
        assert twin['key'] == {
            **BY_ID[vector_id]['key'],
            'kind': 'int',
        }


# --- the vectors ---------------------------------------------------------------


def test_the_vectors_are_the_reference_s() -> None:
    assert RECORDED['format'] == R.FORMAT
    assert RECORDED['recorded_by'] == 'contract/tools/key_inference_record.py'
    assert RECORDED['cell'] == 'ref-pandas3'
    assert RECORDED['register_ref'] == R.REGISTER_REF
    assert RECORDED['pql'] == R.PQL
    assert RECORDED['preview_rows'] == R.PREVIEW_ROWS == 1_000
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    assert [v['id'] for v in RECORDED['vectors']] == [
        v['id'] for v in R.VECTORS
    ]
    for vector in R.VECTORS:
        recorded = BY_ID[vector['id']]
        for key in (
            'table',
            'rows',
            'key',
            'setup',
            'primary_keys',
            'account_kind',
        ):
            assert recorded[key] == vector[key], (vector['id'], key)
        assert recorded['until'] == R.case_for(vector)['until']


def test_the_marks_are_the_recorder_s() -> None:
    for vector in RECORDED['vectors']:
        why = R.register(vector['reference'])
        assert vector['register'] == {'required': bool(why), 'why': why}, (
            vector['id']
        )


def test_each_vector_is_a_case_the_runner_takes() -> None:
    import tempfile

    runner = _module('run_python', ROOT / 'contract/runners/run_python.py')
    with tempfile.TemporaryDirectory() as tmp:
        for vector in R.VECTORS:
            case = R.case_for(vector)
            assert case['input']['setup'] == vector['setup']
            named = vector['primary_keys']
            source = case['core']['describe_source']['source']
            assert source['primary_keys'] == {
                t: {'value': [c]} for t, c in (named or {}).items()
            }
            if named is None:
                assert 'from_source' not in case['input']
            else:
                assert case['input']['from_source'] == {'primary_keys': named}
            if case['until'] == 'materialize':
                assert len(case['core']['materialize']['indices']) == 2
            (Path(tmp) / f'{case["id"]}.json').write_text(
                json.dumps(case), encoding='utf-8'
            )
        loaded = runner.load_cases([Path(tmp)])
    assert len(loaded) == len(R.VECTORS)


def test_the_vectors_are_not_vacuous() -> None:
    vectors = RECORDED['vectors']
    # Every vector reached materialize or was refused before it, and the
    # answers hold every kind of outcome the rules give.
    by_outcome = {}
    for vector in vectors:
        outcome = vector['reference']['outcome']
        by_outcome.setdefault(
            (outcome['result'], outcome['stage'], outcome['code']), []
        ).append(vector['id'])
    assert ('success', 'materialize', None) in by_outcome
    assert ('refusal', 'from_source', 'INVALID_REQUEST') in by_outcome
    assert ('refusal', 'materialize', 'INVALID_REQUEST') in by_outcome
    # At 6298913c the reference fails no vector unnamed or with
    # INTERNAL_ERROR, so none is marked; the rule that would mark one is
    # held by test_a_difference_passes_only_under_an_approved_transition's
    # seeded vector.
    marked = [v['id'] for v in vectors if v['register']['required']]
    assert marked == []
    assert not any(
        v['reference']['outcome']['code'] == 'INTERNAL_ERROR'
        or (
            v['reference']['outcome']['result'] == 'refusal'
            and v['reference']['outcome']['code'] is None
        )
        for v in vectors
    )
    # The threshold, as the reference draws it: kept at exactly 99%,
    # declined at 98.9%, over the whole table at 5,000 rows.
    for rows in (1_000, 5_000):
        keys = {
            share: key_of(
                BY_ID[f'share-{share}.{rows}']['reference'], 'payments'
            )
            for share in ('100', '99.5', '99', '98.9', '50')
        }
        assert keys == {
            '100': 'payment_id',
            '99.5': 'payment_id',
            '99': 'payment_id',
            '98.9': None,
            '50': None,
        }, rows
    # The controls, under a preview, keyed.
    assert key_of(BY_ID['share-100.200']['reference'], 'payments') == (
        'payment_id'
    )
    assert BY_ID['named-clean.200']['reference']['outcome']['result'] == (
        'success'
    )
    # Nulls: kept with 1 and 2 of 200, declined with 10; at 5,000 kept.
    assert key_of(BY_ID['nulls-2.200']['reference'], 'payments') == 'payment_id'
    assert key_of(BY_ID['nulls-10.200']['reference'], 'payments') is None
    assert (
        key_of(BY_ID['nulls-10.5000']['reference'], 'payments') == 'payment_id'
    )
    assert key_of(BY_ID['all-null.200']['reference'], 'payments') is None
    # A named key with a null is refused at from_source.
    assert BY_ID['named-null-1.5000']['reference']['outcome']['stage'] == (
        'from_source'
    )
    # A keyless fact table read by foreign key, materialized.
    keyless = BY_ID['share-50.5000']['reference']
    assert keyless['outcome']['result'] == 'success'
    assert keyless['outcome']['stage'] == 'materialize'
    assert ['payments', 'account_id', 'accounts', 'account_id'] in keyless[
        'schema'
    ]['links']
    # DuckDB's own counts: at 5,000 rows the first 1,000 are unique and
    # hold a value, the whole table not.
    for vector in vectors:
        facts = vector['duckdb']
        assert facts['whole']['rows'] == vector['rows'], vector['id']
        key = vector['key']
        assert facts['whole']['held'] == vector['rows'] - key['nulls']
        if key['nulls'] < vector['rows']:
            assert facts['whole']['distinct'] == (
                vector['rows'] - key['nulls'] - key['repeats']
            ), vector['id']
        if vector['rows'] == 5_000 and key['nulls'] < vector['rows']:
            assert facts['preview']['distinct'] == facts['preview']['held']
            assert facts['preview']['held'] == 1_000
    types = {v['duckdb']['type'] for v in vectors}
    assert {'BIGINT', 'VARCHAR', 'DECIMAL(18,0)'} <= types, types
    assert {v['rows'] for v in vectors} == {200, 1_000, 5_000}
    assert len(NULL_KEYS) >= 10
    assert {BY_ID[i]['table'] for i in NULL_KEYS} == {'accounts', 'payments'}


# --- the gating ----------------------------------------------------------------


def test_the_pending_lists_only_shrink() -> None:
    assert G.only_shrinks() == []
    ids = set(BY_ID)
    for host in ('key-python-core', 'key-rust-host'):
        assert G.pending(host) <= ids, f'{host}: ids that are no vector'


def test_every_vector_ref_names_its_transition() -> None:
    assert transitions()[1] == []


def test_a_difference_passes_only_under_an_approved_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = sys.modules[__name__]
    # nulls-1-decimal.200 as the reference answered it at b413e9ad
    # (INTERNAL_ERROR at materialize), seeded: no recorded vector fails so
    # at 6298913c. Its mark is the recorder's own, derived from it.
    marked = copy.deepcopy(BY_ID['nulls-1-decimal.200'])
    marked['reference']['outcome'] = {
        'result': 'refusal',
        'stage': 'materialize',
        'error': 'StructuredDataError',
        'code': 'INTERNAL_ERROR',
    }
    why = R.register(marked['reference'])
    marked['register'] = {'required': bool(why), 'why': why}
    assert R.outcome_class(marked['reference']['outcome']) == (
        'refusal INTERNAL_ERROR'
    )
    assert marked['register']['required']
    named = copy.deepcopy(marked['reference'])
    named['outcome'] = dict(
        named['outcome'],
        error='UnsupportedRequestError',
        code='UNSUPPORTED_REQUEST',
    )
    entries: list[tuple[str, str, str, str, str]] = []
    monkeypatch.setattr(module, 'transitions', lambda: (entries, []))
    assert problem_of(marked, marked['reference']) is None
    # Marked, but no entry: held exactly.
    assert problem_of(marked, named) is not None
    entries[:] = [
        (
            'P-1',
            'proposed',
            'nulls-1-decimal.*',
            'refusal INTERNAL_ERROR',
            'refusal UNSUPPORTED_REQUEST',
        )
    ]
    assert 'only proposed' in (problem_of(marked, named) or '')
    entries[:] = [
        (
            'A-1',
            'approved',
            'nulls-1-decimal.*',
            'refusal INTERNAL_ERROR',
            'refusal UNSUPPORTED_REQUEST',
        )
    ]
    assert problem_of(marked, named) is None
    assert problem_of(BY_ID['nulls-1-text.200'], named) is not None
    # Today's refusal of a key with a null, on an unmarked vector: fails.
    entries[:] = []
    kept = BY_ID['nulls-1.5000']
    refused = {
        'outcome': {
            'result': 'refusal',
            'stage': 'from_source',
            'error': 'UnsupportedRequestError',
            'code': 'UNSUPPORTED_REQUEST',
        },
        'schema': None,
        'warnings': [],
    }
    assert problem_of(kept, refused) is not None
    # Each part of a success counts: the key chosen, the duplicate count,
    # the links, a warning.
    declined = BY_ID['share-98.9.5000']
    for change in (
        lambda a: a['schema']['tables'][1].__setitem__(
            'primary_key', 'payment_id'
        ),
        lambda a: a['schema']['tables'][1].__setitem__('duplicate_keys', 55),
        lambda a: a['schema'].__setitem__('links', []),
        lambda a: a['warnings'].append('SDMWarning'),
    ):
        got = copy.deepcopy(declined['reference'])
        change(got)
        assert problem_of(declined, got) is not None


def test_the_exit_is_derived() -> None:
    for vector_id in NULL_KEYS:
        vector = BY_ID[vector_id]
        assert vector['key']['nulls'] > 0
        assert vector['reference']['schema'] is not None
    assert 'named-null-1.200' not in NULL_KEYS
    for wanted in (
        'nulls-1.200',
        'nulls-1.5000',
        'nulls-10.200',
        'all-null.5000',
        'entity-nulls-1.5000',
        'nulls-1-text.200',
        'nulls-1-decimal.200',
    ):
        assert wanted in NULL_KEYS, wanted
