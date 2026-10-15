# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The umbrella entries P1B-1 to P1B-4 after T1.33: what they approve and
what they excuse.

    python -m pytest contract/tests/test_t133_umbrellas.py -q

Written by the package's independent test author from the orchestrator's
decision (T1.33 part 2), before the change:

- P1B-1 to P1B-4 stay approved by the owner (E92: the Phase 2 class of
  named refusals), with their ``cases`` emptied: a case is excused by an
  entry of its own, never by an umbrella.
- ``contract/tools/differential.py`` classes a refusal under an umbrella
  only for a rule of E92's Phase 2 class (``parent_table`` is
  ``approved:P1B-3``; ``problem_type`` ``approved:P1B-4``). Every other rule
  an umbrella names (``condition_value``, ``target_stype``, ...) never
  reads as an umbrella's ``approved``.
- ``text_time``: from T1.33 part 3 (the owner's item 24, "text"), no
  longer the owner's open question; a remaining text_time refusal reads
  under DT-15, approved, never under an umbrella (gated by
  ``t133p3_gate.py``).
- An umbrella that does list a case excuses nothing per case: every
  ``registered_case`` of the family tests returns None for it, while an
  approved entry of its own still excuses its case. The Rust families'
  ``registered_case`` skip the four umbrellas alike.

E92's class is read from ``phase-1/not-yet-refusals.md``: the 40 rules it
lists, less ``key_missing``, ``key_repeats`` and ``keyless_neighbors``,
which the owner moved into T1.32e (E84), so 37.

Gating (``contract/tests/t133p2_gate.py``): a test named in
``t133p2-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``)
failed when written. It still runs and must still fail; once it passes it
fails, naming itself, until it is taken out of the list.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'
SURVEY = ROOT / 'docs/design/rust-core/phase-1/not-yet-refusals.md'
UMBRELLAS = ('P1B-1', 'P1B-2', 'P1B-3', 'P1B-4')
#: Out of E92's class since T1.32e part 1 (E84).
MOVED_OUT = ('key_missing', 'key_repeats', 'keyless_neighbors')
REFUSED = 'UnsupportedRequestError'
#: The family tests whose ``registered_case`` excuses a case.
FAMILIES = (
    'test_t132a_keys_family.py',
    'test_t132b_decimal_family.py',
    'test_t132c_nested_family.py',
    'test_t132d_empty_null_family.py',
    'test_t132e_time_feature_family.py',
    'test_t133_adventureworks_cases.py',
)
RUST_FAMILIES = (
    't132a_keys_sql.rs',
    't132b_decimal_sql.rs',
    't132c_nested_sql.rs',
    't132d_empty_null_sql.rs',
    't132e_time_feature_sql.rs',
)


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _module('t133p2_gate', ROOT / 'contract/tests/t133p2_gate.py')
differential = _module(
    't133_umbrella_differential', ROOT / 'contract/tools/differential.py'
)


def held(request: pytest.FixtureRequest, check: Callable[[], None]) -> None:
    G.held('exits', f'py:{request.node.name}', check)


def committed() -> dict[str, dict[str, Any]]:
    register: dict[str, dict[str, Any]] = differential._register(REGISTER)
    return register


def e92_class() -> set[str]:
    r"""E92's class as the survey lists it, less the rules moved out."""
    text = ' '.join(SURVEY.read_text(encoding='utf-8').split())
    found = re.search(
        r'The class, the 40 rules the survey gives Phase 2 but those two: '
        r'(.*?)\.(?: |$)',
        text,
    )
    assert found is not None, 'the survey lists the class'
    listed = re.findall(r'`([a-z_]+)`', found[1])
    assert len(listed) == len(set(listed)) == 40, listed
    return set(listed) - set(MOVED_OUT)


#: Every rule the differential maps to an umbrella, with the path it is
#: classed on: ``materialize`` (the reference takes it, the core refuses
#: it) or ``predict`` (the same request, predict refused).
def umbrella_rules() -> dict[str, tuple[str, str]]:
    out = {}
    for rule, entry in differential.RULES.items():
        if entry in UMBRELLAS:
            out[rule] = (entry, 'materialize')
    for rule, entry in differential.PREDICT_RULES.items():
        if entry in UMBRELLAS:
            out[rule] = (entry, 'predict')
    return out


def _payload() -> dict[str, Any]:
    return {
        'context': {
            'instance_table': {
                'columns': [
                    'instance_id',
                    '__kumo_entity_ref_0',
                    '__kumo_anchor_time',
                    'TARGET',
                ],
                'rows': [
                    [
                        {'number': '0'},
                        {'number': '2'},
                        '2024-01-31T00:00:00.000000Z',
                        True,
                    ],
                    [
                        {'number': '1'},
                        {'number': '7'},
                        '2024-01-31T00:00:00.000000Z',
                        False,
                    ],
                ],
            }
        },
        'metadata': {'num_context_examples': {'number': '3'}},
    }


def _old() -> dict[str, Any]:
    return {
        'id': 's00',
        'from_source': ['ok'],
        'edits': [],
        'schema': {'tables': [['e', 'id', None, []]], 'links': []},
        'materialize': ['ok'],
        'payload': _payload(),
        'warnings': [],
        'seeds': [[2, 1, 0]],
        'predict': ['ok'],
        'result': ['digest'],
    }


def refused_by(rule: str, path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""The reference's success, and the core's refusal by ``rule``."""
    old = _old()
    if path == 'predict':
        core = {**_old(), 'predict': [REFUSED, 'UNSUPPORTED_REQUEST', rule]}
        del core['result']
        return old, core
    core = {
        key: value
        for key, value in _old().items()
        if key not in ('payload', 'predict', 'result')
    }
    core['materialize'] = [REFUSED, 'UNSUPPORTED_REQUEST', rule]
    # The labels the reference's context holds, so a rule refused after
    # them is read past them.
    # (The third example is the one predicted for, entity 0, which the
    # reference's static context leaves out.)
    core['labels'] = [
        [2, '2024-01-31T00:00:00.000000Z', True],
        [7, '2024-01-31T00:00:00.000000Z', False],
        [0, '2024-01-31T00:00:00.000000Z', True],
    ]
    return old, core


#: An aggregation's query, so a rule whose entry the query's shape decides
#: (target_type) reads as its umbrella's.
QUERY = 'PREDICT COUNT(orders.*, 0, 7, days) > 0 FOR EACH users.user_id'


def classed(rule: str, path: str) -> tuple[str, str]:
    old, core = refused_by(rule, path)
    kind, why = differential.classify(old, core, [0, 1], committed(), 0, QUERY)
    return str(kind), str(why)


# --- the register --------------------------------------------------------------


@pytest.mark.parametrize('entry_id', UMBRELLAS)
def test_the_umbrella_is_owner_approved_with_no_cases(
    entry_id: str, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        entry = committed()[entry_id]
        assert entry['status'] == 'approved', entry['status']
        assert entry.get('approved_by') == 'owner'
        assert not entry.get('cases'), f'{entry_id} lists {entry["cases"]}'
        assert not entry.get('case_outcomes')

    held(request, check)


def test_the_class_is_e92_s_thirty_seven() -> None:
    rules = e92_class()
    assert len(rules) == 37
    assert {'parent_table', 'problem_type', 'condition'} <= rules
    assert not {'condition_value', 'target_stype', 'text_time'} & rules


# --- the differential ----------------------------------------------------------


def _in_class() -> list[Any]:
    rules = e92_class()
    return [
        pytest.param(rule, id=rule)
        for rule in sorted(umbrella_rules())
        if rule in rules
    ]


def _not_in_class() -> list[Any]:
    rules = e92_class()
    return [
        pytest.param(rule, id=rule)
        for rule in sorted(umbrella_rules())
        if rule not in rules
    ]


@pytest.mark.parametrize('rule', _in_class())
def test_a_phase_2_rule_reads_as_its_umbrella_approved(
    rule: str, request: pytest.FixtureRequest
) -> None:
    entry, path = umbrella_rules()[rule]

    def check() -> None:
        kind, why = classed(rule, path)
        assert kind == f'approved:{entry}', why

    held(request, check)


def test_parent_table_and_problem_type_are_in_the_class() -> None:
    # The decision's own examples are among the rules read above.
    names = {p.values[0] for p in _in_class()}
    assert {'parent_table', 'problem_type'} <= names


@pytest.mark.parametrize('rule', _not_in_class())
def test_any_other_rule_never_reads_as_an_umbrella_approved(
    rule: str, request: pytest.FixtureRequest
) -> None:
    _, path = umbrella_rules()[rule]

    def check() -> None:
        kind, why = classed(rule, path)
        assert kind not in {f'approved:{u}' for u in UMBRELLAS}, (kind, why)

    held(request, check)


def test_condition_value_and_target_stype_are_read() -> None:
    names = {p.values[0] for p in _not_in_class()}
    assert {'condition_value', 'target_stype'} <= names


def _text_time() -> tuple[dict[str, Any], dict[str, Any]]:
    old = {'id': 's00', 'from_source': ['ok']}
    core = {
        'id': 's00',
        'from_source': [REFUSED, 'UNSUPPORTED_REQUEST', 'text_time'],
    }
    return old, core


def _p3() -> ModuleType:
    return _module('t133p3_gate', ROOT / 'contract/tests/t133p3_gate.py')


@pytest.mark.parametrize('path', ['from_source', 'materialize'])
def test_a_remaining_text_time_refusal_reads_under_dt_15(
    path: str, request: pytest.FixtureRequest
) -> None:
    # T1.33 part 3 (the owner's item 24): text_time no longer refuses by a
    # column's name, and the owner's answer is "text", so nothing of it is
    # pending the owner. What text_time refusals remain (a chosen time
    # column's values at materialize) are DT-15's, approved; never an
    # umbrella's.
    if path == 'from_source':
        old, core = _text_time()
        query = None
    else:
        old, core = refused_by('text_time', 'materialize')
        query = QUERY
    p2_only = {
        'P1B-2': {
            'id': 'P1B-2',
            'status': 'approved',
            'approved_by': 'owner',
            'new_behavior': 'text_time',
        }
    }

    def check() -> None:
        assert ('P1B-2', 'text_time') not in differential.PENDING_OWNER
        kind, why = differential.classify(
            old, core, [0, 1], committed(), 0, query
        )
        assert kind == 'approved:DT-15', (kind, why)
        kind, why = differential.classify(old, core, [0, 1], p2_only, 0, query)
        assert kind == 'UNCLASSIFIED', (kind, why)

    _p3().held('exits', f'py:test_t133_umbrellas::{request.node.name}', check)


# --- the families' registered_case ---------------------------------------------


def _family(name: str) -> ModuleType:
    return _module(
        f't133_umbrella_{Path(name).stem}', ROOT / 'contract/tests' / name
    )


def _crafted(tmp: Path) -> Path:
    r"""A register in which each umbrella, approved, lists a case, and an
    entry of its own, approved, lists another."""
    entries: list[dict[str, Any]] = [
        {
            'id': umbrella,
            'status': 'approved',
            'approved_by': 'owner',
            'cases': [f'umbrella.case-{umbrella.lower()}'],
        }
        for umbrella in UMBRELLAS
    ]
    entries.append(
        {
            'id': 'P1B-999',
            'status': 'approved',
            'approved_by': 'owner',
            'cases': ['own.case'],
        }
    )
    root = tmp / 'root'
    path = root / 'docs/design/rust-core/register.yaml'
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({'entries': entries}), encoding='utf-8')
    return root


@pytest.mark.parametrize('family', FAMILIES)
def test_an_umbrella_listing_a_case_excuses_nothing(
    family: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    module = _family(family)
    monkeypatch.setattr(module, 'ROOT', _crafted(tmp_path))

    def check() -> None:
        # The control: an entry of its own excuses its case.
        assert module.registered_case('own.case') == 'P1B-999'
        for umbrella in UMBRELLAS:
            case = f'umbrella.case-{umbrella.lower()}'
            assert module.registered_case(case) is None, (family, umbrella)

    held(request, check)


@pytest.mark.parametrize('family', RUST_FAMILIES)
def test_the_rust_families_skip_every_umbrella(
    family: str, request: pytest.FixtureRequest
) -> None:
    source = (ROOT / 'rust/crates/sdm-testkit/tests' / family).read_text(
        encoding='utf-8'
    )
    found = re.search(r'fn registered_case\(.*?\n}\n', source, flags=re.DOTALL)
    assert found is not None, f'{family} has no registered_case'
    body = found[0]
    # The umbrellas skipped, named in the function or by a constant it
    # reads that names them.
    constants = dict(re.findall(r'const (\w+): [^=]+= (\[[^\]]*\]);', source))

    def check() -> None:
        named = body + ''.join(
            value for key, value in constants.items() if key in body
        )
        missing = [u for u in UMBRELLAS if f'"{u}"' not in named]
        assert missing == [], f'{family} excuses by {missing}'

    held(request, check)
