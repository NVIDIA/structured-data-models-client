# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The old-against-core differential's classification (T1.27,
``contract/tools/differential.py``), on records written here: the tool is
a local pre-push probe, not a gate, and this holds only that it fails
closed. A difference no register entry covers is UNCLASSIFIED; so is
evidence missing, empty or retyped (labels absent where the core refuses
after them, a label of another type, an entry that does not name the rule,
probes of other sources), each seeded below.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    'differential', ROOT / 'contract/tools/differential.py'
)
assert _spec is not None and _spec.loader is not None
differential = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(differential)

REGISTER = {
    'P1B-3': {
        'id': 'P1B-3',
        'status': 'proposed',
        'new_behavior': 'classification_task target_type',
    },
    'P1B-19': {
        'id': 'P1B-19',
        'status': 'proposed',
        'new_behavior': 'wire_dtype',
    },
    'P1B-18': {
        'id': 'P1B-18',
        'status': 'approved',
        'new_behavior': 'target_type',
    },
}


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
        'id': 'g00',
        'from_source': ['ok'],
        'edits': [['ok'], ['StructuredDataError', 'INVALID_REQUEST', None]],
        'schema': {'tables': [['e', 'id', None, []]], 'links': []},
        'materialize': ['ok'],
        'payload': _payload(),
        'warnings': [],
        'seeds': [[2, 1, 0]],
        'predict': ['ok'],
        'result': ['digest'],
    }


def _core() -> dict[str, Any]:
    return {
        'id': 'g00',
        'from_source': ['ok'],
        'edits': [['ok'], ['StructuredDataError', 'INVALID_REQUEST', None]],
        'schema': {'tables': [['e', 'id', None, []]], 'links': []},
        'materialize': [
            'UnsupportedRequestError',
            'UNSUPPORTED_REQUEST',
            'classification_task',
        ],
        'warnings': [],
        'seeds': [[2, 1, 0]],
        # The third example is one predicted for, which the reference's
        # static context leaves out.
        'labels': [
            [2, '2024-01-31T00:00:00.000000Z', True],
            [7, '2024-01-31T00:00:00.000000Z', False],
            [0, '2024-01-31T00:00:00.000000Z', True],
        ],
    }


def test_a_refusal_after_equal_labels_is_its_entry_s() -> None:
    kind, _ = differential.classify(_old(), _core(), [0, 1], REGISTER, 2)
    assert kind == 'proposed:P1B-3'


def test_the_edits_are_compared_before_anything_else() -> None:
    old, core = _old(), _core()
    # An edit's outcome that differs between the routes, a code that
    # differs, an edit one route did not make (or made and recorded
    # nothing of), and the schema after them.
    differing = copy.deepcopy(core)
    differing['edits'][1] = ['ok']
    recoded = copy.deepcopy(core)
    recoded['edits'][1][1] = 'UNSUPPORTED_REQUEST'
    ignored = copy.deepcopy(core)
    del ignored['edits'][1]
    schema = copy.deepcopy(core)
    schema['schema']['tables'][0][2] = 'born'
    unrecorded = copy.deepcopy(core)
    del unrecorded['edits']
    for changed in (differing, recoded, ignored, schema, unrecorded):
        assert differential.classify(old, changed, [0, 1], REGISTER, 2)[0] == (
            'UNCLASSIFIED'
        )
    # An edit the spec makes and neither route recorded.
    assert (
        differential.classify(old, core, [0, 1], REGISTER, 3)[0]
        == 'UNCLASSIFIED'
    )
    # The probe makes each edit through its public setter.
    made: list[Any] = []

    class Table:
        def __setattr__(self, name: str, value: Any) -> None:
            made.append((name, value))

    class Schema(dict):
        relationships: list[Any] = []

    schema_obj = Schema(e=Table())
    differential._apply(
        schema_obj, {'op': 'time', 'table': 'e', 'value': 'ts'}, tuple
    )
    differential._apply(
        schema_obj, {'op': 'key', 'table': 'e', 'value': None}, tuple
    )
    assert made == [('time_column', 'ts'), ('primary_key', None)]


def test_equal_requests_agree_and_any_difference_is_unclassified() -> None:
    old = _old()
    same = copy.deepcopy(old)
    assert differential.classify(old, same, [0, 1], REGISTER)[0] == 'agree'
    for field, value in [
        ('payload', {}),
        ('seeds', []),
        ('warnings', ['Missing']),
        ('result', ['other']),
    ]:
        changed = {**copy.deepcopy(old), field: value}
        assert (
            differential.classify(old, changed, [0, 1], REGISTER)[0]
            == 'UNCLASSIFIED'
        )


def test_the_classification_fails_closed() -> None:
    old = _old()
    cases = []
    flipped = _core()
    flipped['labels'][0][2] = False
    cases.append(flipped)
    retyped = _core()
    retyped['labels'][1][2] = 0
    cases.append(retyped)
    missing = _core()
    del missing['labels']
    cases.append(missing)
    empty = _core()
    empty['labels'] = []
    cases.append(empty)
    failed = _core()
    failed['labels'] = ['StructuredDataError', 'INTERNAL_ERROR', None]
    cases.append(failed)
    not_predicted = _core()
    not_predicted['labels'][2][0] = 9
    cases.append(not_predicted)
    unknown_rule = _core()
    unknown_rule['materialize'][2] = 'a_rule_no_entry_names'
    cases.append(unknown_rule)
    # The windowed reads after which the core refuses: their seeds are
    # evidence, never switched off by an empty list.
    no_seeds = _core()
    no_seeds['seeds'] = []
    cases.append(no_seeds)
    other_seeds = _core()
    other_seeds['seeds'] = [[2, 1, -1]]
    cases.append(other_seeds)
    for core in cases:
        assert (
            differential.classify(old, core, [0, 1], REGISTER)[0]
            == 'UNCLASSIFIED'
        ), core
    # An entry that does not name the rule, or is not in the register.
    unnamed = {**REGISTER, 'P1B-3': {'id': 'P1B-3', 'status': 'proposed'}}
    assert (
        differential.classify(old, _core(), [0, 1], unnamed)[0]
        == 'UNCLASSIFIED'
    )
    gone = {k: v for k, v in REGISTER.items() if k != 'P1B-3'}
    assert (
        differential.classify(old, _core(), [0, 1], gone)[0] == 'UNCLASSIFIED'
    )
    # The core taking what the reference refuses.
    refused = {
        **old,
        'materialize': ['StructuredDataError', 'INVALID_REQUEST', None],
    }
    taken = {**_core(), 'materialize': ['ok']}
    assert (
        differential.classify(refused, taken, [0, 1], REGISTER)[0]
        == 'UNCLASSIFIED'
    )


def test_an_internal_error_named_by_the_core_is_e66_b_s_class() -> None:
    old = {
        **_old(),
        'materialize': ['StructuredDataError', 'INTERNAL_ERROR', None],
    }
    core = {
        **_core(),
        'materialize': [
            'UnsupportedRequestError',
            'UNSUPPORTED_REQUEST',
            'target_type',
        ],
    }
    static = 'PREDICT users.tier FOR EACH users.user_id'
    assert (
        differential.classify(old, core, [0, 1], REGISTER, 2, static)[0]
        == 'approved:P1B-18'
    )
    # Which entry is the query's shape's: unknown, it names none.
    assert differential.classify(old, core, [0, 1], REGISTER, 2)[0] == (
        'UNCLASSIFIED'
    )
    other = {
        **core,
        'materialize': ['StructuredDataError', 'INVALID_REQUEST', None],
    }
    assert (
        differential.classify(old, other, [0, 1], REGISTER)[0] == 'UNCLASSIFIED'
    )


def test_a_refusal_s_entry_is_its_query_shape_s() -> None:
    # A decimal's average refused after its fetch is P1B-3's, never the
    # static decimal target's P1B-18.
    core = _core()
    core['materialize'] = [
        'UnsupportedRequestError',
        'UNSUPPORTED_REQUEST',
        'target_type',
    ]
    average = 'PREDICT AVG(a.d, 0, 30, days) FOR EACH e.id'
    assert differential.classify(_old(), core, [0, 1], REGISTER, 2, average)[
        0
    ] == ('proposed:P1B-3')
    # The reference's INVALID_REQUEST where the core refuses wire_dtype is
    # P1B-19's; under another code it is not.
    old = {
        **_old(),
        'materialize': ['StructuredDataError', 'INVALID_REQUEST', None],
    }
    wire = {
        **_core(),
        'materialize': [
            'UnsupportedRequestError',
            'UNSUPPORTED_REQUEST',
            'wire_dtype',
        ],
    }
    assert differential.classify(old, wire, [0, 1], REGISTER, 2, average)[
        0
    ] == ('proposed:P1B-19')
    other = {
        **old,
        'materialize': ['StructuredDataError', 'PROVIDER_ERROR', None],
    }
    assert differential.classify(other, wire, [0, 1], REGISTER, 2, average)[
        0
    ] == ('UNCLASSIFIED')


def test_probes_of_other_sources_fail(tmp_path: Path) -> None:
    old, core = tmp_path / 'old.json', tmp_path / 'core.json'
    old.write_text(json.dumps([_old()]))
    core.write_text(json.dumps([{**_core(), 'id': 'g01'}]))
    register = tmp_path / 'register.yaml'
    register.write_text('entries: []\n')
    assert differential.compare(old, core, None, register) == 1
    core.write_text('[]')
    old.write_text('[]')
    assert differential.compare(old, core, None, register) == 1


def test_incomplete_or_doubled_evidence_fails(tmp_path: Path) -> None:
    old, core = tmp_path / 'old.json', tmp_path / 'core.json'
    spec = tmp_path / 'spec.json'
    register = tmp_path / 'register.yaml'
    register.write_text('entries: []\n')
    agreeing = _old()
    source = {'id': 'g00', 'indices': [0, 1], 'edits': [{}, {}]}
    old.write_text(json.dumps([agreeing]))
    core.write_text(json.dumps([agreeing]))
    spec.write_text(json.dumps({'sources': [source]}))
    # The control: the one source the spec names, agreeing.
    assert differential.compare(old, core, spec, register) == 0
    # The spec names a source neither probe holds.
    spec.write_text(json.dumps({'sources': [source, {**source, 'id': 'g01'}]}))
    assert differential.compare(old, core, spec, register) == 1
    # A source twice, in a probe or in the spec.
    spec.write_text(json.dumps({'sources': [source]}))
    old.write_text(json.dumps([agreeing, agreeing]))
    core.write_text(json.dumps([agreeing, agreeing]))
    assert differential.compare(old, core, None, register) == 1
    old.write_text(json.dumps([agreeing]))
    core.write_text(json.dumps([agreeing]))
    spec.write_text(json.dumps({'sources': [source, source]}))
    assert differential.compare(old, core, spec, register) == 1


#: The entries T1.28a's classifications name, as the register holds them.
T128A = {
    **REGISTER,
    'P1B-4': {
        'id': 'P1B-4',
        'status': 'proposed',
        'new_behavior': 'problem_type outputs',
    },
    'P1B-11': {
        'id': 'P1B-11',
        'status': 'approved',
        'new_behavior': 'empty_table',
    },
    'DT-18': {
        'id': 'DT-18',
        'status': 'proposed',
        'new_behavior': 'PROVIDER_ERROR',
    },
}


def _same_request(core_predict: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    old = _old()
    core = {**_old(), 'predict': core_predict}
    core.pop('result')
    return old, core


def test_a_predict_refusal_of_the_same_request_is_its_entry_s() -> None:
    # T1.28a: both routes build the same request; the core refuses its
    # answer by a predict rule (a classification's layout, T1.28b's).
    old, core = _same_request(
        ['UnsupportedRequestError', 'UNSUPPORTED_REQUEST', 'problem_type']
    )
    assert differential.classify(old, core, [0, 1], T128A, 2)[0] == (
        'proposed:P1B-4'
    )
    # Fail-closed: another rule, another code, no rule, a differing request
    # beside it, an entry that does not name the rule, an entry missing.
    for predicted in (
        ['UnsupportedRequestError', 'UNSUPPORTED_REQUEST', 'not_a_rule'],
        ['StructuredDataError', 'INVALID_REQUEST', 'problem_type'],
        ['UnsupportedRequestError', 'UNSUPPORTED_REQUEST', None],
        ['UnsupportedRequestError', 'UNSUPPORTED_REQUEST'],
        None,
    ):
        old, core = _same_request(predicted)
        assert differential.classify(old, core, [0, 1], T128A, 2)[0] == (
            'UNCLASSIFIED'
        ), predicted
    old, core = _same_request(
        ['UnsupportedRequestError', 'UNSUPPORTED_REQUEST', 'problem_type']
    )
    core['warnings'] = ['typed text']
    assert differential.classify(old, core, [0, 1], T128A, 2)[0] == (
        'UNCLASSIFIED'
    )
    unnamed = {**T128A, 'P1B-4': {**T128A['P1B-4'], 'new_behavior': 'x'}}
    old, core = _same_request(
        ['UnsupportedRequestError', 'UNSUPPORTED_REQUEST', 'problem_type']
    )
    assert differential.classify(old, core, [0, 1], unnamed, 2)[0] == (
        'UNCLASSIFIED'
    )
    missing = {k: v for k, v in T128A.items() if k != 'P1B-4'}
    assert differential.classify(old, core, [0, 1], missing, 2)[0] == (
        'UNCLASSIFIED'
    )


def test_an_empty_table_refused_by_name_is_e66b_s_entry() -> None:
    old = {'id': 'g00', 'from_source': ['RuntimeError', None, None]}
    core = {
        'id': 'g00',
        'from_source': [
            'StructuredDataError',
            'INVALID_REQUEST',
            'empty_table',
        ],
    }
    assert differential.classify(old, core, [0, 1], T128A, 0)[0] == (
        'approved:P1B-11'
    )
    for refused in (
        # The code before the owner's switch of P1B-11 (T1.32g).
        ['UnsupportedRequestError', 'UNSUPPORTED_REQUEST', 'empty_table'],
        ['StructuredDataError', 'INVALID_REQUEST', 'other'],
        ['StructuredDataError', 'INVALID_REQUEST', None],
        ['ok'],
    ):
        assert differential.classify(
            old, {**core, 'from_source': refused}, [0, 1], T128A, 0
        )[0] == ('UNCLASSIFIED'), refused
    # The reference refusing by name under another code is not E66(b)'s
    # class; under the same code both refuse alike.
    named = {
        **old,
        'from_source': ['StructuredDataError', 'UNSUPPORTED_TYPE', None],
    }
    assert differential.classify(named, core, [0, 1], T128A, 0)[0] == (
        'UNCLASSIFIED'
    )
    same = {
        **old,
        'from_source': ['StructuredDataError', 'INVALID_REQUEST', None],
    }
    assert differential.classify(same, core, [0, 1], T128A, 0)[0] == 'agree'


def test_the_source_s_failure_under_the_provider_s_code_is_dt_18_s() -> None:
    # Fix round 1 (Codex P2-4): DT-18 only where the spec makes a time of
    # day a table's time column, which the source casts to TIMESTAMP.
    old = {
        **_old(),
        'materialize': ['StructuredDataError', 'INTERNAL_ERROR', None],
    }
    core = {
        **_core(),
        'materialize': ['StructuredDataError', 'PROVIDER_ERROR', None],
    }
    assert differential.classify(old, core, [0, 1], T128A, 2, time_of_day=True)[
        0
    ] == ('proposed:DT-18')
    # Without the evidence, the same pair is unclassified.
    assert differential.classify(old, core, [0, 1], T128A, 2)[0] == (
        'UNCLASSIFIED'
    )
    named = {
        **old,
        'materialize': ['StructuredDataError', 'INVALID_REQUEST', None],
    }
    assert differential.classify(
        named, core, [0, 1], T128A, 2, time_of_day=True
    )[0] == ('UNCLASSIFIED')
    missing = {k: v for k, v in T128A.items() if k != 'DT-18'}
    assert differential.classify(
        old, core, [0, 1], missing, 2, time_of_day=True
    )[0] == ('UNCLASSIFIED')


def _tod_source(
    *edits: dict[str, Any], times: dict[str, str] | None = None
) -> dict[str, Any]:
    return {
        'id': 'g00',
        'setup': [
            'CREATE TABLE e AS SELECT i::BIGINT AS id, '
            'make_time((i % 24)::BIGINT, 30, 0) AS tod, '
            "'12:00'::TIME AS noon, "
            "TIMESTAMP '2024-01-01' AS born FROM range(4) t(i)",
            "CREATE TABLE a AS SELECT i::BIGINT AS aid, 'x' AS tod "
            'FROM range(4) t(i)',
        ],
        'from_source': {
            'primary_keys': {'e': 'id', 'a': 'aid'},
            'time_columns': {} if times is None else times,
        },
        'edits': list(edits),
    }


def test_the_time_of_day_evidence_is_a_time_column_of_type_time() -> None:
    evidence = differential.time_of_day_is_a_time_column
    # Positive: the time of day made the time column, by a setter or as
    # Schema.from_source is told it; a cast to TIME as make_time.
    assert evidence(_tod_source({'op': 'time', 'table': 'e', 'value': 'tod'}))
    assert evidence(
        _tod_source({'op': 'set_time_column', 'table': 'e', 'column': 'noon'})
    )
    assert evidence(_tod_source(times={'e': 'tod'}))
    # Negative: no time column set, a timestamp, a name the table lacks, a
    # time of day's name on a table where it is text, the time column
    # cleared after it was set.
    assert not evidence(_tod_source())
    assert not evidence(_tod_source(times={'e': 'born'}))
    assert not evidence(
        _tod_source({'op': 'time', 'table': 'e', 'value': 'nope'})
    )
    assert not evidence(
        _tod_source({'op': 'time', 'table': 'a', 'value': 'tod'})
    )
    assert not evidence(
        _tod_source(
            {'op': 'time', 'table': 'e', 'value': 'tod'},
            {'op': 'time', 'table': 'e', 'value': None},
        )
    )
    assert not evidence(
        _tod_source(
            {'op': 'set_time_column', 'table': 'e', 'column': 'tod'},
            {'op': 'clear_time_column', 'table': 'e'},
        )
    )


def test_compare_names_dt_18_only_with_the_evidence(tmp_path: Path) -> None:
    old = {
        **_old(),
        'id': 'g00',
        'materialize': ['StructuredDataError', 'INTERNAL_ERROR', None],
    }
    core = {
        **_core(),
        'id': 'g00',
        'materialize': ['StructuredDataError', 'PROVIDER_ERROR', None],
    }
    register = tmp_path / 'register.yaml'
    register.write_text(json.dumps({'entries': list(T128A.values())}))
    paths = {n: tmp_path / f'{n}.json' for n in ('old', 'core', 'spec')}
    paths['old'].write_text(json.dumps([old]))
    paths['core'].write_text(json.dumps([core]))
    # Two edits each, as the records hold: the last time column set is the
    # one read.
    born = {'op': 'time', 'table': 'e', 'value': 'born'}
    tod = {'op': 'time', 'table': 'e', 'value': 'tod'}
    for source, code in (
        (_tod_source(born, tod), 0),
        (_tod_source(tod, born), 1),
    ):
        paths['spec'].write_text(json.dumps({'sources': [source]}))
        assert (
            differential.compare(
                paths['old'], paths['core'], paths['spec'], register
            )
            == code
        )
