# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32h (register entry P1B-12): the keys and time columns
``describe_source`` is given, as the contract's ``Given``, on
``python-core``, held to ``contract/given-vectors.json``.

    python -m pytest contract/tests/test_t132h_given_vectors.py -q

Written by the package's independent test author from the design of
2026-10-13 (revised after Codex's review), section 4's table and section
6's plan, before the implementation. ``contract/given-vectors.json`` is
written by ``contract/tools/given_record.py``, the reference's answer
recorded beside each vector a caller can give; ``sdm-testkit``'s
``tests/t132h_given.rs`` holds ``rust-host`` to the same vectors, so the
two hosts agree through them.

- The wire, at the binding's boundary (section 6A): each vector's spec,
  its bytes as written, is the attachment of a ``describe_source`` started
  through ``sdm_client._native`` over the ``shop`` base, each fetch
  answered by the client's DuckDB provider. The answer is the vector's
  ``core``: the outcome; where it ends (``start``: refused before the run
  exists, nothing fetched; ``run``: after at least one fetch, so the source
  is read before any key or time column is judged); the code; the error's
  fields exactly (``"*"`` any value; ``kind`` and ``retryable``, which the
  client adds, aside); for a success each table's key and time column.
- The cases (section 6C): each ``case`` vector, a case of
  ``contract/cases``' format, runs through ``run_python.py --mode
  python-core`` (``Schema.from_source`` forced onto the core): the host
  writes the vector's spec (the comparator's ``inputs`` rule), and answers
  with its outcome, stage ``from_source`` and code.
- The register (section 4, "Register transitions"): where a case vector's
  core answer differs in class (``success``, ``refusal <CODE>``) from the
  reference's recorded one, the vector names what covers it, and it does:
  an approved entry whose ``refs`` declare exactly
  ``contract/given-vectors.json#<id> <from> -> <to>``, or a rule of
  decision E92's Phase 2 class, the core's refusal naming it. A vector
  whose class is the reference's names nothing. A success is the
  reference's key and time column of each table.
- ``edit_schema`` beside ``describe_source`` (section 3): each ``edit``
  vector's value given to the setter of a schema the core described, on
  the core route: the edit the host writes, and the answer; beside its
  pair's refusal by ``Schema.from_source``, the same code and ``rule``,
  the field the operation's own (``primary_key`` against
  ``primary_keys``); a ``null`` keeps each operation's meaning.
- A start speaking contract 22 is refused before a run exists,
  ``INVALID_MESSAGE`` naming the version (section 3, "The version exit").
- Seeds: the checks fail on a case whose core spec stays in contract 22's
  shape (seed 12, the ``inputs`` rule), on a written spec, an answer or a
  register difference seeded wrong.

Gating (``contract/tests/t132h_gate.py``): ``t132h-pending.json``'s
``pending["python-core"]`` lists the vectors that failed here when this was
written (a vector run as a case ``case:<id>``), and ``pending["exits"]`` the
other tests (``py:<test id>``); a vector of a decision still the owner's
(``owner``) is gated in that decision's group, as ``python-core:<id>``. Each
still runs and must still fail; one that passes fails, naming itself, until
it is taken out of the list, which only shrinks. The implementer removes ids
as they pass and changes nothing else of this file, the vectors or the
recorder.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / 'contract/given-vectors.json'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'
NOT_YET = ROOT / 'docs/design/rust-core/phase-1/not-yet-refusals.md'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _module('given_record', ROOT / 'contract/tools/given_record.py')
G = _module('t132h_gate', ROOT / 'contract/tests/t132h_gate.py')
RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}
#: What the client adds to the core's fields in ``.details``.
ADDED = ('kind', 'retryable')


def gate(
    vector: dict[str, Any], group: str, prefix: str = ''
) -> tuple[str, str]:
    r"""The group and name a vector's check is gated under: for a decision
    still the owner's, that decision's group, the name prefixed with the
    group it would otherwise be in (``python-core:case:<id>``)."""
    if vector.get('owner'):
        return vector['owner'], f'{group}:{prefix}{vector["id"]}'
    return group, f'{prefix}{vector["id"]}'


def _ids(kind: str | None = None, spec: bool = False) -> list[Any]:
    return [
        pytest.param(v['id'], id=v['id'])
        for v in RECORDED['vectors']
        if (kind is None or v['kind'] == kind) and (not spec or 'spec' in v)
    ]


def shop() -> list[str]:
    lines = (
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8').splitlines()
    )
    start = lines.index('  shop:') + 1
    statements = []
    for line in lines[start:]:
        if not line.startswith('  - '):
            break
        statements.append(line[len('  - ') :])
    return statements


def connection() -> Any:
    import duckdb

    made = duckdb.connect()
    for statement in shop():
        made.execute(statement)
    return made


def _key(value: Any) -> list[str] | None:
    if value is None or value == [] or value == ():
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def tables_of(described: dict[str, Any]) -> dict[str, Any]:
    return {
        table['name']: {
            'primary_key': _key(table['primary_key']),
            'time_column': table['time_column'],
        }
        for table in described['tables']
    }


def tables_of_schema(schema: Any) -> dict[str, Any]:
    return {
        name: {
            'primary_key': _key(table.primary_key),
            'time_column': table.time_column,
        }
        for name, table in schema.tables.items()
    }


class Starts:
    r"""A spy on ``sdm_client._native.Run.start``: each start's operation
    and attachments, and whether the core refused it, changing nothing
    that crosses."""

    def __init__(self, real: Any) -> None:
        self.real = real
        self.starts: list[dict[str, Any]] = []

    def start(self, control: bytes, attachments: Any) -> Any:
        record = {
            'operation': json.loads(control)['operation'],
            'attachments': dict(attachments),
            'refused': False,
        }
        self.starts.append(record)
        try:
            return self.real.start(control, attachments)
        except BaseException:
            record['refused'] = True
            raise


@pytest.fixture
def starts(monkeypatch: pytest.MonkeyPatch) -> Starts:
    from sdm_client._rust.run import native

    core = native()
    spy = Starts(core.Run)
    monkeypatch.setattr(core, 'Run', spy)
    return spy


def answer_of(
    call: Callable[[], Any], starts: Starts, fetched: list[int]
) -> dict[str, Any]:
    r"""What ``call`` answered, in the vectors' terms: a success's tables,
    or a refusal's code and fields and where it ended."""
    from sdm_client import StructuredDataError

    try:
        tables = call()
    except StructuredDataError as error:
        refused_at_start = bool(starts.starts) and starts.starts[-1]['refused']
        return {
            'outcome': 'refusal',
            'at': 'start' if refused_at_start else 'run',
            'code': error.code,
            'fields': {
                k: v for k, v in dict(error.details).items() if k not in ADDED
            },
            'fetches': fetched[0],
        }
    return {
        'outcome': 'success',
        'at': 'run',
        'tables': tables,
        'fetches': fetched[0],
    }


def differs(
    core: dict[str, Any], got: dict[str, Any], edit: bool = False
) -> list[str]:
    r"""Where ``got`` is not a vector's ``core``: empty when it is."""
    problems: list[str] = []
    if got['outcome'] != core['outcome']:
        return [f'outcome {got["outcome"]} (wanted {core["outcome"]}): {got}']
    if got['at'] != core['at']:
        problems.append(f'at {got["at"]} (wanted {core["at"]})')
    if got['at'] == 'start' and got['fetches']:
        problems.append(
            f'{got["fetches"]} fetches before a refusal at the start'
        )
    if got['at'] == 'run' and not edit and not got['fetches']:
        problems.append('judged before the source was read (no fetch)')
    if got['outcome'] == 'success':
        for table, wanted in core['tables'].items():
            if got['tables'].get(table) != wanted:
                problems.append(
                    f'{table}: {got["tables"].get(table)} (wanted {wanted})'
                )
        return problems
    if got['code'] != core['code']:
        problems.append(f'code {got["code"]} (wanted {core["code"]})')
    fields = got['fields']
    if set(fields) != set(core['fields']):
        problems.append(
            f'fields {sorted(fields)} (wanted {sorted(core["fields"])})'
        )
    for name, value in core['fields'].items():
        if value != '*' and fields.get(name) != value:
            problems.append(f'{name}: {fields.get(name)!r} (wanted {value!r})')
    return problems


def describe_raw(spec: bytes, starts: Starts) -> dict[str, Any]:
    r"""``describe_source`` started with ``spec``'s bytes as they stand,
    through ``sdm_client._native``, over a fresh ``shop``."""
    from sdm_client._rust.providers.duckdb import DuckDBProvider
    from sdm_client._rust.run import run

    provider = DuckDBProvider(connection())
    fetched = [0]

    def counted(request: Any, key_sets: Any) -> Any:
        fetched[0] += 1
        return provider(request, key_sets)

    def call() -> Any:
        operation = {
            'op': 'describe_source',
            'provider': RECORDED['provider'],
            'source': 1,
            'check_keys': True,
        }
        finished = run(operation, {1: spec}, provider=counted)
        return tables_of(
            json.loads(finished.take(finished.result['description']))
        )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return answer_of(call, starts, fetched)


# --- the wire, at the binding's boundary ---------------------------------------


@pytest.mark.parametrize('vector_id', _ids(spec=True))
def test_python_core_answers_each_vector_as_its_spec_is_written(
    vector_id: str, starts: Starts
) -> None:
    vector = BY_ID[vector_id]

    def check() -> None:
        spec = json.dumps(vector['spec']).encode()
        got = describe_raw(spec, starts)
        problems = differs(vector['core'], got)
        assert not problems, problems

    G.held(*gate(vector, 'python-core'), check)


# --- the cases -------------------------------------------------------------------


@pytest.fixture(scope='module')
def core_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    r"""Every case vector through ``python-core``, forced onto the core."""
    tmp = tmp_path_factory.mktemp('t132h-vectors')
    return R.run('python-core', R.VECTORS, tmp, core=True)


def case_problems(
    vector: dict[str, Any], observation: dict[str, Any]
) -> list[str]:
    r"""Where a case's ``python-core`` run is not the vector: what the host
    wrote (the ``inputs`` rule), and its outcome, stage and code."""
    problems = []
    core = observation.get('core')
    if core is None:
        return ['the core was not reached']
    case = R.case_for(vector, core=True)
    wanted = {'op': 'describe_source', **case['core']['describe_source']}
    if wanted not in core['inputs']:
        written = [
            i for i in core['inputs'] if i.get('op') == 'describe_source'
        ]
        problems.append(f'inputs: the host wrote {written}, the case {wanted}')
    got = R.project(observation)
    want = vector['core']
    if got['stage'] != 'from_source':
        problems.append(f'stage {got["stage"]}')
    if got['result'] != want['outcome']:
        problems.append(f'outcome {got["result"]} {got["error"]} {got["code"]}')
    elif want['outcome'] == 'refusal' and got['code'] != want['code']:
        problems.append(f'code {got["code"]} (wanted {want["code"]})')
    elif want['outcome'] == 'success':
        for table, wanted_table in want['tables'].items():
            if (got['tables'] or {}).get(table) != wanted_table:
                problems.append(f'{table}: {(got["tables"] or {}).get(table)}')
    return problems


@pytest.mark.parametrize('vector_id', _ids('case'))
def test_python_core_writes_and_answers_each_case_vector(
    vector_id: str, core_runs: dict[str, Any]
) -> None:
    vector = BY_ID[vector_id]

    def check() -> None:
        problems = case_problems(vector, core_runs[vector_id])
        assert not problems, problems

    G.held(*gate(vector, 'python-core', 'case:'), check)


# --- the register ----------------------------------------------------------------

TRANSITION = re.compile(
    re.escape(R.REGISTER_REF)
    + r'(?P<id>\S+) (?P<old>success|refusal [A-Z][A-Z_]*)'
    + r' -> (?P<new>success|refusal [A-Z][A-Z_]*)'
)


def declared(
    document: dict[str, Any],
) -> tuple[dict[tuple[str, str], tuple[str, str]], list[str]]:
    r"""Each vector ref of the register's entries, ``(entry, status)`` by
    ``(vector id, transition)``, and the refs naming the vectors in no
    transition of the form (malformed)."""
    out: dict[tuple[str, str], tuple[str, str]] = {}
    malformed: list[str] = []
    for entry in document['entries']:
        for ref in entry.get('refs') or []:
            if not isinstance(ref, str) or not ref.startswith(R.REGISTER_REF):
                continue
            found = TRANSITION.fullmatch(ref)
            if found is None or found['id'] not in BY_ID:
                malformed.append(f'{entry["id"]}: {ref}')
                continue
            out[(found['id'], f'{found["old"]} -> {found["new"]}')] = (
                str(entry['id']),
                str(entry.get('status')),
            )
    return out, malformed


def phase2_rules() -> set[str]:
    r"""The rules of decision E92's class, as ``not-yet-refusals.md``'s
    first paragraph lists them ("The class, ...")."""
    text = NOT_YET.read_text(encoding='utf-8')
    found = re.search(r'The class, [^:]*: (.*?)\.\n', text)
    assert found is not None, 'not-yet-refusals.md names no class'
    return set(re.findall(r'`([a-z_0-9]+)`', found[1]))


def register_problems(
    vector: dict[str, Any],
    document: dict[str, Any],
    phase2: set[str],
) -> list[str]:
    r"""Why the vector's difference from the reference is not covered as
    it says, or why it names a cover it does not need."""
    old = R.reference_class(vector['reference'])
    new = R.core_class(vector['core'])
    cover = vector.get('register')
    if old == new:
        problems = [] if cover is None else [f'names {cover} for no difference']
        if new == 'success':
            got = vector['core']['tables']
            want = vector['reference']['tables']
            if got != want:
                problems.append(f'tables {got}, the reference {want}')
        return problems
    if cover is None:
        return [f'{old} -> {new} and nothing covers it']
    if 'phase2' in cover:
        rule = cover['phase2']
        problems = []
        if rule not in phase2:
            problems.append(f'{rule} is not in decision E92 class')
        if (
            new != 'refusal UNSUPPORTED_REQUEST'
            or vector['core']['fields'].get('rule') != rule
        ):
            problems.append(f'the core does not refuse it by {rule}')
        return problems
    transition = f'{old} -> {new}'
    problems = []
    if cover.get('transition') != transition:
        problems.append(
            f'it names {cover.get("transition")}, the difference is {transition}'
        )
    entry, status = declared(document)[0].get(
        (vector['id'], transition), (None, None)
    )
    if entry != cover.get('entry'):
        problems.append(
            f'{cover.get("entry")} declares no ref {R.REGISTER_REF}{vector["id"]} {transition}'
        )
    elif status != 'approved':
        problems.append(f'{entry} declares it, {status}')
    return problems


@pytest.mark.parametrize('vector_id', _ids('case'))
def test_each_difference_from_the_reference_is_covered(vector_id: str) -> None:
    vector = BY_ID[vector_id]
    document = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))

    def check() -> None:
        problems = register_problems(vector, document, phase2_rules())
        assert not problems, problems

    G.held(*gate(vector, 'exits', 'py-register:'), lambda: check())


def test_every_ref_naming_the_vectors_declares_a_transition() -> None:
    document = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))
    assert declared(document)[1] == []


# --- edit_schema beside describe_source --------------------------------------------


def _core_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    from sdm_client import _routing

    monkeypatch.setenv(
        _routing.ENVIRONMENT,
        ','.join(
            dict.fromkeys(
                _routing.FROM_SOURCE_DUCKDB + _routing.FROM_SOURCE_DUCKDB_TABLES
            )
        ),
    )


def _fetch_counter(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    from sdm_client._rust.providers.duckdb import DuckDBProvider

    fetched = [0]
    real = DuckDBProvider.__call__

    def counted(self: Any, request: Any, key_sets: Any) -> Any:
        fetched[0] += 1
        return real(self, request, key_sets)

    monkeypatch.setattr(DuckDBProvider, '__call__', counted)
    return fetched


def from_source(arguments: dict[str, Any]) -> Any:
    from sdm_client import Schema
    from sdm_connectors import DuckDBSource

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_source(DuckDBSource(connection()), **arguments)


def edit_problems(
    vector: dict[str, Any],
    got: dict[str, Any],
    written: Any,
    paired: dict[str, Any],
) -> list[str]:
    r"""Where an edit's answer is not its vector's, or not its pair's
    refusal with the field the operation's own."""
    problems = differs(vector['core'], got, edit=True)
    if written != vector['edit']:
        problems.append(f'the host wrote {written}, wanted {vector["edit"]}')
    pair = BY_ID[vector['pair']]
    problems.extend(f'its pair: {p}' for p in differs(pair['core'], paired))
    if got['outcome'] == paired['outcome'] == 'refusal':
        if got['code'] != paired['code']:
            problems.append(f'code {got["code"]} beside {paired["code"]}')
        rule = got['fields'].get('rule')
        if rule is None or rule != paired['fields'].get('rule'):
            problems.append(
                f'rule {rule} beside {paired["fields"].get("rule")}'
            )
        if f'{got["fields"].get("field")}s' != paired['fields'].get('field'):
            problems.append(
                f'field {got["fields"].get("field")} beside '
                f'{paired["fields"].get("field")}'
            )
    return problems


@pytest.mark.parametrize('vector_id', _ids('edit'))
def test_the_setter_shares_from_source_s_refusal_and_keeps_its_null(
    vector_id: str, monkeypatch: pytest.MonkeyPatch, starts: Starts
) -> None:
    vector = BY_ID[vector_id]
    pair = BY_ID[vector['pair']]
    _core_routes(monkeypatch)
    fetched = _fetch_counter(monkeypatch)

    def check() -> None:
        fetched[0] = 0
        paired = answer_of(
            lambda: tables_of_schema(from_source(pair['from_source'])),
            starts,
            fetched,
        )
        schema = from_source({})
        starts.starts.clear()
        fetched[0] = 0
        made = vector['set']

        def edit() -> Any:
            setattr(schema[made['table']], made['attribute'], made['value'])
            return tables_of_schema(schema)

        got = answer_of(edit, starts, fetched)
        edits = [
            s for s in starts.starts if s['operation']['op'] == 'edit_schema'
        ]
        written = (
            json.loads(edits[-1]['attachments'][edits[-1]['operation']['edit']])
            if edits
            else None
        )
        problems = edit_problems(vector, got, written, paired)
        assert not problems, problems

    G.held(*gate(vector, 'exits', 'py-edit:'), check)


# --- the version exit ---------------------------------------------------------------


def test_a_start_speaking_contract_22_is_refused_before_a_run_exists(
    request: pytest.FixtureRequest,
) -> None:
    from sdm_client._rust import codec, facts
    from sdm_client._rust.run import native

    def check() -> None:
        core = native()
        start = codec.encode(
            {
                'message': 'start',
                'contract': 22,
                'facts': facts.host_facts(core),
                'operation': {
                    'op': 'describe_source',
                    'provider': RECORDED['provider'],
                    'source': 1,
                    'check_keys': True,
                },
            }
        )
        spec = json.dumps(R.spec()).encode()
        try:
            core.Run.start(start, {1: spec})
        except core.CoreError as refused:
            message = codec.read(refused.control, refused.attachments)
            error = message['error']
            assert error['code'] == 'INVALID_MESSAGE', error
            assert error['fields'] == {'contract': {'t': 'u32', 'v': '22'}}, (
                error
            )
            return
        raise AssertionError('a start speaking contract 22 began a run')

    G.held('exits', f'py:{request.node.name}', check)


# --- the vectors, the gate and the seeds -------------------------------------------


def test_the_vectors_are_the_recorder_s() -> None:
    assert RECORDED['format'] == R.FORMAT
    assert RECORDED['recorded_by'] == 'contract/tools/given_record.py'
    assert RECORDED['cell'] == 'ref-pandas3'
    assert RECORDED['register_ref'] == R.REGISTER_REF
    assert RECORDED['question'] == R.QUESTION
    assert RECORDED['provider'] == R.PROVIDER
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    assert [v['id'] for v in RECORDED['vectors']] == [
        v['id'] for v in R.VECTORS
    ]
    for written in R.VECTORS:
        vector = copy.deepcopy(BY_ID[written['id']])
        vector.pop('reference')
        vector.pop('reference_why', None)
        written = copy.deepcopy(written)
        written.pop('reference_why', None)
        assert vector == written, written['id']
    for vector in RECORDED['vectors']:
        if vector['kind'] == 'case':
            assert vector['reference']['stage'] == 'from_source', vector['id']
        else:
            assert vector['reference'] is None and vector['reference_why']


def test_the_pending_lists_only_shrink() -> None:
    assert G.only_shrinks() == []
    gating = G.gating()
    vector_ids = set(BY_ID)
    for group in G.groups():
        for name in gating['authored'][group]:
            bare = name.split(':', 1)[-1] if ':' in name else name
            if group in ('python-core', 'rust-host'):
                assert bare in vector_ids, (group, name)


def test_each_seeded_fault_fails_a_check() -> None:
    # Seed 12: a case's core spec left in contract 22's shape is not what
    # the host writes (the inputs rule).
    vector = BY_ID['key.name']
    old = copy.deepcopy(vector)
    old['spec'] = R.spec({'orders': ['order_id']})
    written = {
        'core': {
            'inputs': [
                {
                    'op': 'describe_source',
                    **R.case_for(vector)['core']['describe_source'],
                }
            ]
        },
        'outcome': {'result': 'success', 'stage': 'from_source'},
        'public': {
            'schema': {
                'tables': [
                    {
                        'name': 'orders',
                        'primary_key': 'order_id',
                        'time_column': 'ts',
                    },
                    {
                        'name': 'users',
                        'primary_key': 'user_id',
                        'time_column': None,
                    },
                ]
            }
        },
    }
    assert case_problems(vector, written) == []
    assert case_problems(old, written) != []
    # A refusal judged before the source is read, or with its rule
    # dropped, or at the start, fails the vector.
    good = {
        'outcome': 'refusal',
        'at': 'run',
        'code': 'INVALID_REQUEST',
        'fields': {
            'field': 'primary_keys',
            'table': 'orders',
            'rule': 'key_not_names',
            'given': '3',
            'variant': 'given',
        },
        'fetches': 2,
    }
    core = BY_ID['key.int']['core']
    assert differs(core, good) == []
    assert differs(core, {**good, 'fetches': 0}) != []
    assert (
        differs(
            core,
            {**good, 'fields': {'field': 'primary_keys', 'table': 'orders'}},
        )
        != []
    )
    # KO2-86: the value named, not another.
    assert (
        differs(core, {**good, 'fields': {**good['fields'], 'given': '4'}})
        != []
    )
    assert differs(core, {**good, 'at': 'start', 'fetches': 0}) != []
    assert (
        differs(
            core,
            {'outcome': 'success', 'at': 'run', 'tables': {}, 'fetches': 2},
        )
        != []
    )
    # A bare list taken (seed 8), an empty key as INVALID_MESSAGE (seed 9).
    assert (
        differs(
            BY_ID['wire.key-bare-list']['core'],
            {'outcome': 'success', 'at': 'run', 'tables': {}, 'fetches': 2},
        )
        != []
    )
    malformed = {
        'outcome': 'refusal',
        'at': 'start',
        'code': 'INVALID_MESSAGE',
        'fields': {'attachment': 1, 'category': 'data', 'line': 1, 'column': 9},
        'fetches': 0,
    }
    assert differs(BY_ID['wire.key-bare-list']['core'], malformed) == []
    assert differs(BY_ID['key.empty']['core'], malformed) != []
    # The edit's field made plural (seed 11), its rule dropped (seed 10).
    edit = BY_ID['edit.key-not-names']
    paired = {**good}
    mine = {
        **good,
        'fields': {**good['fields'], 'field': 'primary_key'},
        'fetches': 0,
    }
    assert edit_problems(edit, mine, edit['edit'], paired) == []
    plural = {**mine, 'fields': {**mine['fields'], 'field': 'primary_keys'}}
    assert edit_problems(edit, plural, edit['edit'], paired) != []
    ruleless = {**mine, 'fields': {'field': 'primary_key', 'table': 'orders'}}
    assert edit_problems(edit, ruleless, edit['edit'], paired) != []
    # A difference from the reference that nothing covers, or a proposed
    # entry's, or one named for no difference.
    document = {'entries': []}
    phase2 = phase2_rules()
    assert 'composite_key' in phase2
    assert register_problems(BY_ID['time.list'], document, phase2) != []
    proposed = {
        'entries': [
            {
                'id': 'P1B-12',
                'status': 'proposed',
                'refs': [
                    f'{R.REGISTER_REF}time.list refusal UNNAMED -> refusal INVALID_REQUEST'
                ],
            }
        ]
    }
    assert register_problems(BY_ID['time.list'], proposed, phase2) != []
    approved = copy.deepcopy(proposed)
    approved['entries'][0]['status'] = 'approved'
    assert register_problems(BY_ID['time.list'], approved, phase2) == []
    named = copy.deepcopy(BY_ID['key.int'])
    named['register'] = {'entry': 'P1B-12', 'transition': 'x'}
    assert register_problems(named, approved, phase2) != []
    uncovered = copy.deepcopy(BY_ID['key.twice-before-composite'])
    uncovered['register'] = None
    assert register_problems(uncovered, approved, phase2) != []
    # A refusal by a rule of E92's class is covered by naming it (T2.12
    # lifted composite_key, so no vector refuses by one now; a seed does).
    refused = copy.deepcopy(BY_ID['key.composite'])
    refused['core'] = {
        'outcome': 'refusal',
        'at': 'run',
        'code': 'UNSUPPORTED_REQUEST',
        'fields': {'operation': 'describe_source', 'rule': 'composite_key'},
    }
    refused['register'] = {'phase2': 'composite_key'}
    assert register_problems(refused, document, phase2) == []
    refused['register'] = {'phase2': 'not_a_rule'}
    assert register_problems(refused, document, phase2) != []
