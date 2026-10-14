# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.33's type set: ``contract/type-set-vectors.json`` held to its own
shape and to the register.

Written by the package's independent test author, before the change.

    python -m pytest contract/tests/test_t133_type_set.py -q

``rust/crates/sdm-host/tests/t133_type_set.rs`` holds the core to each
vector (admission's class, the admitted type, the received-type inference
vector). This file holds what the core cannot show:

- exit "each arrow-rs ``DataType`` classified as accepted, admitted or
  refused by name, with its received-type inference vector": every vector
  has one class, its inference (or none for a type no array can be built
  of), its answer's class (``core``) and the reference's (``reference``);
  the arrow-schema version is the one ``rust/Cargo.lock`` pins; each of
  the 41 variants has a vector and no type has two;
- "any change to the set proposed to the owner", "every changed case's
  entry complete": where the reference's answer (through DuckDB 1.5.6) and
  the core's differ, an approved register entry names the vector with
  exactly that transition in its ``refs`` (T1.32a's binding-transition
  form, ``<file>#<id or glob> <from> -> <to>``): the vector's
  ``covered_by`` vector where it names one, an earlier package's vector of
  the same DuckDB type, else this file's. A proposed entry excuses nothing;
  a type DuckDB hands no such column of (``not reachable``) needs none;
- "the owner's list of new refusals": a refusal of a type the reference
  reads is an entry the owner approved;
- exit "each type in HOST-2 marked reference-accepted or not": each type
  family HOST-2 names (in backticks) is followed in its entry by
  ``reference-accepted`` or ``not reference-accepted``.

Gating (``contract/tests/t133p2_gate.py``): a test named in
``t133p2-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``)
failed when written. It still runs and must still fail; once it passes it
fails, naming itself, until it is taken out of the list.
"""

from __future__ import annotations

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
VECTORS = ROOT / 'contract/type-set-vectors.json'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'
LOCK = ROOT / 'rust/Cargo.lock'
REF = 'contract/type-set-vectors.json#'
CLASSES = ('accepted', 'admitted', 'refused')
OUTCOME = re.compile(r'success|refusal [A-Z][A-Z_]*')


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _module('t133p2_gate', ROOT / 'contract/tests/t133p2_gate.py')
FILE: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
ALL: list[dict[str, Any]] = FILE['vectors']
#: The forms DuckDB hands over only under a session setting, or as an
#: extension type, that no vector is: each through real DuckDB alone.
REACHED: list[dict[str, Any]] = FILE['reached']
NESTED = ROOT / 'contract/nested-type-vectors.json'


def _register() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(
        REGISTER.read_text(encoding='utf-8')
    )
    return loaded


def _entry(entry_id: str) -> dict[str, Any]:
    for entry in _register()['entries']:
        if entry['id'] == entry_id:
            return dict(entry)
    raise AssertionError(f'register: no entry {entry_id}')


def _held(name: str, check: Any) -> None:
    G.held('exits', f'py:{name}', check)


# --- the file's shape --------------------------------------------------------


def test_the_pending_list_only_shrinks() -> None:
    assert G.only_shrinks() == []
    ids = {vector['id'] for vector in ALL}
    loaded = G.gating()
    assert set(loaded['pending']['rust-host']) <= ids


def test_the_vectors_are_of_the_arrow_schema_cargo_lock_pins() -> None:
    lock = LOCK.read_text(encoding='utf-8')
    found = re.search(r'name = "arrow-schema"\nversion = "([^"]+)"', lock)
    assert found is not None
    assert FILE['format'] == 'sdm-type-set-vectors/1'
    assert FILE['arrow_schema'] == found[1]


def test_each_variant_has_a_vector_and_no_type_has_two() -> None:
    variants = FILE['variants']
    assert len(variants) == len(set(variants)) == 41
    assert {vector['variant'] for vector in ALL} == set(variants)
    for key in ('id', 'arrow'):
        repeated = [k for k, n in Counter(v[key] for v in ALL).items() if n > 1]
        assert repeated == [], f'{key} repeated: {repeated}'


@pytest.mark.parametrize('vector', ALL, ids=[v['id'] for v in ALL])
def test_each_vector_has_one_class_and_its_inference(
    vector: dict[str, Any],
) -> None:
    assert vector['class'] in CLASSES
    assert (vector['admitted_as'] is not None) == (
        vector['class'] == 'admitted'
    )
    inference = vector['inference']
    if inference is None:
        # No array of the type can be built: held at admission alone.
        assert vector['buildable'] is False
        assert vector['class'] != 'refused'
        assert vector['core'] is None
        return
    assert len(inference) == 1 and next(iter(inference)) in (
        'dtype',
        'left_out',
        'refusal',
    )
    refusal = inference.get('refusal')
    if vector['class'] == 'refused':
        # Refused by name, the column and the Arrow type received there.
        assert refusal is not None
        assert refusal['code'] == 'UNSUPPORTED_TYPE'
        assert refusal['column'].startswith('f')
        assert refusal['arrow_type']
    elif refusal is not None:
        assert refusal == {
            'code': 'UNSUPPORTED_REQUEST',
            'rule': refusal['rule'],
        }
        assert refusal['rule']
    want = 'success' if refusal is None else f'refusal {refusal["code"]}'
    assert vector['core'] == want
    assert vector['reference'] == 'not reachable' or OUTCOME.fullmatch(
        vector['reference']
    ), vector['reference']
    assert vector['evidence'] or vector['reference'] == 'not reachable'
    witness = vector.get('witness')
    if witness is not None:
        # The DuckDB statements that hand the type over, after `shop`: a
        # reachable type's, ending in the one that adds `f` to `users`.
        assert vector['reference'] != 'not reachable'
        assert witness and all(isinstance(s, str) and s for s in witness)
        assert witness[-1].startswith('CREATE OR REPLACE TABLE users AS ')
        assert ' AS f FROM users' in witness[-1]


#: The vectors a review found reachable after they were marked not
#: reachable (T1.33 part 2, Codex's review): each with its witness, and
#: the reference's answer through it.
REVIEWED_REACHABLE = {
    'timestamp.us.utc': 'success',
    'dictionary.uint16-utf8': 'refusal UNNAMED',
    'dictionary.uint32-utf8': 'refusal UNNAMED',
    'struct.large-utf8': 'refusal UNNAMED',
    'map.large-utf8-int64': 'refusal UNNAMED',
}


def test_the_types_a_review_found_reachable_are_reached() -> None:
    by_id = {v['id']: v for v in ALL}
    for vector_id, reference in REVIEWED_REACHABLE.items():
        vector = by_id[vector_id]
        assert vector['reference'] == reference, vector_id
        assert vector.get('witness'), vector_id
    # Each difference among them is one the register check holds.
    differing = {v['id'] for v in DIFFERING}
    assert differing >= {
        i for i, r in REVIEWED_REACHABLE.items() if r != by_id[i]['core']
    }
    assert 'timestamp.us.utc' not in differing


# --- through real DuckDB: witnesses, recordings and settings ---------------------


def _from_source_class(answer: dict[str, Any]) -> str:
    r"""The reference's answer where ``describe_source`` answers."""
    outcome = answer['outcome']
    if outcome['result'] == 'refusal' and outcome['stage'] == 'from_source':
        return f'refusal {outcome["code"] or "UNNAMED"}'
    return 'success'


def test_ids_are_unique_across_vectors_and_reached_rows() -> None:
    ids = [row['id'] for row in ALL + REACHED]
    assert len(ids) == len(set(ids))


def test_every_reachable_row_has_its_witness_and_its_recording() -> None:
    assert FILE['recorded']['by'] == 'contract/tools/type_set_record.py'
    rows = [
        v
        for v in ALL
        if v['reference'] != 'not reachable' and not v.get('covered_by')
    ]
    for row in rows + REACHED:
        witness = row.get('witness')
        assert witness, f'{row["id"]}: no witness'
        assert row['duckdb_arrow'], row['id']
        answer = row['reference_answer']
        assert answer['outcome']['result'] in ('success', 'refusal'), row['id']
        assert row['reference'] == _from_source_class(answer), row['id']
    for row in ALL:
        if 'probe' in row['evidence'] and row['reference'] != 'not reachable':
            assert row.get('witness'), f'{row["id"]}: a probe with no setup'


@pytest.mark.parametrize('row', REACHED, ids=[r['id'] for r in REACHED])
def test_each_reached_row_is_a_form_admission_sees(row: dict[str, Any]) -> None:
    assert row['witness'][-1].startswith('CREATE OR REPLACE TABLE users AS ')
    assert row['extension'] is None or row['extension'].startswith('arrow.')
    if row['extension']:
        assert f'extension<{row["extension"]}' in row['duckdb_arrow']
    inference = row['inference']
    assert len(inference) == 1 and next(iter(inference)) in (
        'dtype',
        'left_out',
        'refusal',
    )
    refusal = inference.get('refusal')
    want = 'success' if refusal is None else f'refusal {refusal["code"]}'
    assert row['core'] == want


def test_every_form_a_session_setting_gives_has_a_row() -> None:
    nested = json.loads(NESTED.read_text(encoding='utf-8'))['vectors']
    rows = {r['duckdb_arrow'] for r in ALL + REACHED if r.get('duckdb_arrow')}
    rows |= {v['duckdb_arrow'] for v in nested}
    measured = FILE['duckdb_arrow_by_setting']
    assert 'arrow_lossless_conversion' in measured['BOOLEAN']
    missing = sorted(
        f'{name} under {setting}: {form}'
        for name, by in measured.items()
        for setting, form in by.items()
        if form not in rows
    )
    assert missing == [], missing


def test_varint_is_a_difference_to_register() -> None:
    by_id = {r['id']: r for r in REACHED}
    for row_id in ('varint', 'varint.large', 'varint.view'):
        row = by_id[row_id]
        assert row['extension'] == 'arrow.opaque'
        assert 'type_name=bignum' in row['duckdb_arrow']
        assert row['reference'] == 'refusal UNNAMED'
        assert row['core'] == 'success'
        assert row in DIFFERING


def test_a_lossless_connection_is_described() -> None:
    by_id = {r['id']: r for r in REACHED}
    plain = by_id['lossless.integer']
    assert plain['witness'][0] == 'SET arrow_lossless_conversion = true'
    assert plain['reference'] == plain['core'] == 'success'
    assert _reference_described(plain) == _core_described(plain) == 'int64'
    assert by_id['lossless.boolean']['inference'] == {'dtype': 'bool'}
    uuid = {v['id']: v for v in ALL}['fixed-size-binary.16']
    assert uuid['witness'][0] == 'SET arrow_lossless_conversion = true'
    assert uuid['inference'] == {'left_out': 'binary'}


# --- the register --------------------------------------------------------------

#: A vector ref and the transition it declares, in any vectors file.
TRANSITION = re.compile(
    r'(?P<file>contract/[a-z0-9-]+\.json)#(?P<glob>\S+) '
    r'(?P<old>success|refusal [A-Z][A-Z_]*) -> '
    r'(?P<new>success|refusal [A-Z][A-Z_]*)'
)


def _approved_transitions() -> list[tuple[str, str, str, str, str]]:
    r"""Each vector ref of an approved entry: (entry, file, glob, from, to)."""
    found = []
    for entry in _register()['entries']:
        if entry.get('status') != 'approved':
            continue
        for ref in entry.get('refs') or []:
            matched = TRANSITION.fullmatch(str(ref))
            if matched is not None:
                found.append(
                    (
                        str(entry['id']),
                        matched['file'],
                        matched['glob'],
                        matched['old'],
                        matched['new'],
                    )
                )
    return found


def _covering(vector: dict[str, Any]) -> list[str]:
    r"""The approved entries naming the vector's difference exactly."""
    target = vector.get('covered_by') or REF + vector['id']
    file, vector_id = target.split('#', 1)
    old, new = vector['reference'], vector['core']
    return [
        entry
        for entry, f, glob, frm, to in _approved_transitions()
        if f == file
        and (frm, to) == (old, new)
        and fnmatch.fnmatchcase(vector_id, glob)
    ]


def _wire(dtype: str) -> str:
    r"""A data type as the wire names it, as the reference describes one."""
    return {
        'int': 'int64',
        'float': 'float64',
        'date': 'timestamp[us]',
        'time': 'timestamp[us]',
    }.get(dtype, dtype)


def _core_described(row: dict[str, Any]) -> str | None:
    r"""What the row expects the core to describe ``f`` as."""
    inference = row['inference'] or {}
    if 'dtype' in inference:
        return _wire(inference['dtype'])
    if 'left_out' in inference:
        return 'left out'
    return None


def _reference_described(row: dict[str, Any]) -> str | None:
    r"""What the reference described ``f`` as, where it was recorded."""
    described = (row.get('reference_answer') or {}).get('described')
    if isinstance(described, dict):
        return str(described['dtype'])
    return described if isinstance(described, str) else None


def _differs(row: dict[str, Any]) -> bool:
    if row['core'] is None or row['reference'] in ('not reachable', None):
        return False
    if row['reference'] != row['core']:
        return True
    # Both describe f: a difference in what they describe it as is one too.
    theirs, ours = _reference_described(row), _core_described(row)
    return theirs is not None and ours is not None and theirs != ours


DIFFERING = [row for row in ALL + REACHED if _differs(row)]


@pytest.mark.parametrize('vector', DIFFERING, ids=[v['id'] for v in DIFFERING])
def test_a_difference_from_the_reference_is_registered(
    vector: dict[str, Any],
) -> None:
    def check() -> None:
        assert _covering(vector), (
            f'{vector["id"]}: the reference {vector["reference"]}, the core '
            f'{vector["core"]}: no approved entry names '
            f'{vector.get("covered_by") or REF + vector["id"]} with that '
            'transition'
        )

    _held(
        f'test_a_difference_from_the_reference_is_registered[{vector["id"]}]',
        check,
    )


def test_a_refusal_of_a_type_the_reference_reads_is_the_owner_s() -> None:
    new_refusals = [
        v
        for v in ALL + REACHED
        if v['reference'] == 'success'
        and (v['core'] or '').startswith('refusal')
    ]
    for vector in new_refusals:
        owners = [
            e for e in _covering(vector) if _entry(e)['approved_by'] == 'owner'
        ]
        assert owners, f"{vector['id']}: a new refusal, not the owner's"


def test_a_new_success_is_the_owner_s() -> None:
    # E66(b): the core reading what the reference refuses is kept for the
    # owner's individual decision; an entry a reviewer approved does not
    # cover it.
    for row in DIFFERING:
        if row['reference'].startswith('refusal') and row['core'] == 'success':
            for entry in _covering(row):
                assert _entry(entry)['approved_by'] == 'owner', (
                    f"{row['id']}: a new success under {entry}, not the owner's"
                )


def test_every_vector_ref_names_a_vector_and_a_transition() -> None:
    ids = {v['id'] for v in ALL + REACHED}
    for entry in _register()['entries']:
        for ref in entry.get('refs') or []:
            if not str(ref).startswith(REF):
                continue
            matched = TRANSITION.fullmatch(str(ref))
            assert matched is not None, f'{entry["id"]}: {ref}'
            assert any(fnmatch.fnmatchcase(i, matched['glob']) for i in ids), (
                f'{entry["id"]}: {ref} names no vector'
            )


# --- HOST-2 ----------------------------------------------------------------------


def _host_2_text() -> str:
    entry = _entry('HOST-2')
    return ' '.join(
        str(entry.get(k) or '')
        for k in ('title', 'old_behavior', 'new_behavior', 'notes')
    )


def test_host_2_marks_each_type_reference_accepted_or_not() -> None:
    def check() -> None:
        text = ' '.join(_host_2_text().split())
        families = sorted(set(re.findall(r'`([a-z_]+)`', text)))
        assert families, 'HOST-2 names no type family'
        for family in families:
            mark = re.search(
                rf'`{family}`[^`]*?\b(not reference-accepted|reference-accepted)\b',
                text,
            )
            assert mark is not None, f'HOST-2: `{family}` is not marked'

    _held('test_host_2_marks_each_type_reference_accepted_or_not', check)


def test_each_family_host_2_names_has_its_reference_answer() -> None:
    text = ' '.join(_host_2_text().split())
    families = set(re.findall(r'`([a-z_]+)`', text))
    variants = {
        'interval': 'Interval',
        'union': 'Union',
    }
    for family in families & set(variants):
        answered = [
            v
            for v in ALL
            if v['variant'] == variants[family]
            and v['reference'] != 'not reachable'
        ]
        assert answered, f'{family}: no vector records the reference'
