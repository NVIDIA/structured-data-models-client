# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A case's refusal map, its probes on ``python-core``, and the check that
holds a run to the map (T1.21b).

A refusal map (``contract/refusal-maps/<case>.json``, format
``sdm-refusal-map/1``) lists each refusal the core route hits on a case's
journey, with the rule and operation it names and the task that owns
removing it, and the probes that reach each one. A refusal masks every
later one, so a probe is the case with the features refused before it
removed, one step at a time: the map is the first refusal of each probe,
which is the first refusal per operation once the earlier ones are gone.

Each probe has an input per runner: ``python-core`` makes the public calls
in process, with ``SDM_CORE_ROUTES`` set to the probe's routes, as
``run_python.py`` does; ``rust-host`` starts the core operations the probe
names (``sdm-testkit``'s ``tests/refusal_map.rs``). What a runner observes
of a probe is its first refusal, as the fields it can see, or its success:

- ``python-core``: the stage (the public call that refused), the
  exception's class, ``StructuredDataError.code``, and the ``operation``
  and ``rule`` of its ``details``, each a string or null; a ``RouteError``
  (no core route) is observed as itself, though ``run_python.py`` counts it
  as a runner failure;
- ``rust-host``: the error's code and its fields ``operation`` and
  ``rule``.

Every probe is bound to the case (:func:`bound`): the first probe of each
runner is the case itself, and every other names, in ``derive``, the
earlier probe it starts from and the removals it makes, which must give
its input exactly, so a change to the case alone fails the map.

:func:`check` holds the observations to the map: every probe the map gives
the runner is observed, as the entry it names or as success; a refusal the
map does not list fails, and so does an entry no probe of a runner it
lists reaches.

A map is of one case (``case``, T1.21b's AdventureWorks map) or of a family
of the Phase 1b inventory (``family``, the inventory's seam, and ``cases``,
every case of it, T1.21e), whose probes each name their ``case``. An entry
whose refusal is the reference's own (``reference: same``: the same stage,
class and code as the case's reference observation) is no departure and
names no register entry (``register: none``).

:func:`write` makes a map from its skeleton: what a person writes (the
entries' rule, owner, register entry and citation; each probe's id, case,
derivation and routes) stays, and what a run decides is recomputed: each
probe's input from its derivation, what each runner observed of it and the
entry it hits. Run as a script it rewrites a map in place::

    python -I contract/runners/refusal_map.py write MAP [--rust-observed FILE]

where FILE is what ``sdm-testkit``'s ``tests/refusal_map.rs`` writes of
every map's rust-host probes when run as::

    SDM_REFUSAL_MAP_OBSERVED=FILE cargo test -p sdm-testkit \\
        --test refusal_map the_family_maps_are_held_on_the_rust_host

Without FILE the map's rust-host observations are kept; with it, every
rust-host probe must be observed there, and nothing else. A run that
leaves anything to report leaves MAP as it was and writes the map it would
make beside it, as MAP.draft, for a person to finish.

A map whose every probe succeeds on every runner it names has no entries:
its probes stay, as the evidence that nothing is refused.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

FORMAT = 'sdm-refusal-map/1'
RUNNERS = ('python-core', 'rust-host')
#: What each runner observes of a refusal.
FIELDS = {
    'python-core': ('stage', 'error', 'code', 'operation', 'rule'),
    'rust-host': ('code', 'operation', 'rule'),
}
STAGES = ('from_source', 'materialize', 'predict')
#: The tasks a refusal of a map may name as its owner: T1.21c and T1.21d
#: (E50, added at T1.10e), T1.22 to T1.27, T1.28a and T1.28b (T1.28's
#: halves from T1.12b, which leave no T1.28) and the inventory's families,
#: T1.32a to T1.32e (T1.21e).
OWNER = re.compile(r'^T1\.(?:21[cd]|2[2-7]|28[ab]|32[a-e])$')
#: The family maps' families: the inventory's seams.
FAMILIES = (
    'keys',
    'decimal',
    'nested',
    'empty-null',
    'time',
    'refusal',
    'baseline',
)


class MapFailure(Exception):
    r"""The map is not one; its text never quotes a value."""


def load(path: Path) -> dict[str, Any]:
    r"""The refusal map at ``path``, its shape checked.

    Raises:
        MapFailure: The file is not a refusal map: not JSON, nested past
            what the reader takes, or of another shape.
    """
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except RecursionError:
        raise MapFailure(f'{path.name}: nested too deeply') from None
    except ValueError:
        raise MapFailure(f'{path.name}: not JSON') from None
    problem = shape(document)
    if problem is not None:
        raise MapFailure(f'{path.name}: {problem}')
    return document


def _strings(value: Any) -> bool:
    return isinstance(value, str) and value != ''


def shape(document: Any) -> str | None:
    r"""What is wrong with ``document`` as a refusal map, or None."""
    if not isinstance(document, dict):
        return 'not an object'
    if document.get('format') != FORMAT:
        return f'not {FORMAT}'
    if set(document) - {
        'format',
        'case',
        'family',
        'cases',
        'reading',
        'gaps',
        'entries',
        'probes',
    }:
        return 'keys a refusal map does not have'
    family = 'family' in document or 'cases' in document
    if family == ('case' in document):
        return 'neither of one case nor of a family'
    if family:
        cases = document.get('cases')
        if document.get('family') not in FAMILIES:
            return 'not a family of the inventory'
        if (
            not isinstance(cases, list)
            or not cases
            or not all(_strings(case) for case in cases)
            or len(set(cases)) != len(cases)
        ):
            return 'cases are not each a case, once'
    gaps = document.get('gaps', [])
    if not isinstance(gaps, list) or not all(
        isinstance(gap, dict)
        and all(
            _strings(gap.get(key)) for key in ('input', 'contract', 'owner')
        )
        for gap in gaps
    ):
        return 'gaps are not each an input, its contract and an owner'
    for at, gap in enumerate(gaps):
        if not OWNER.fullmatch(gap['owner']):
            return f'gaps/{at}/owner is not a task a map may name'
    if not family and not _strings(document.get('case')):
        return 'no case'
    entries = document.get('entries')
    probes = document.get('probes')
    # No entries is a map all of whose refusals are lifted: its probes,
    # every case's on each runner, then all expect success (T1.21e).
    if not isinstance(entries, list):
        return 'no entries'
    if not isinstance(probes, list) or not probes:
        return 'no probes'
    ids: set[str] = set()
    for at, entry in enumerate(entries):
        if not isinstance(entry, dict):
            return f'entries/{at} is not an object'
        for key in ('id', 'operation', 'rule', 'owner', 'register', 'where'):
            if not _strings(entry.get(key)):
                return f'entries/{at}/{key} is not text'
        if not OWNER.fullmatch(entry['owner']):
            return f'entries/{at}/owner is not a task a map may name'
        if entry['id'] in ids:
            return f'entries/{at}/id repeats'
        ids.add(entry['id'])
        observed = entry.get('observed')
        if not isinstance(observed, dict) or not observed:
            return f'entries/{at}/observed is not an object'
        for runner, seen in observed.items():
            if runner not in RUNNERS:
                return f'entries/{at}/observed names an unknown runner'
            problem = _observation(seen, runner)
            if problem is not None:
                return f'entries/{at}/observed/{runner}: {problem}'
    names: set[str] = set()
    for at, probe in enumerate(probes):
        if not isinstance(probe, dict) or not _strings(probe.get('id')):
            return f'probes/{at} has no id'
        if probe['id'] in names:
            return f'probes/{at}/id repeats'
        names.add(probe['id'])
        given = [runner for runner in RUNNERS if runner in probe]
        if not given or set(probe) - {'id', 'case', 'removes', *RUNNERS}:
            return f'probes/{at} has no runner, or keys it should not'
        if family and probe.get('case') not in document['cases']:
            return f'probes/{at} names no case of the family'
        if (
            not family
            and probe.get('case', document['case']) != (document['case'])
        ):
            return f"probes/{at} names another case than the map's"
        for runner in given:
            spec = probe[runner]
            if not isinstance(spec, dict):
                return f'probes/{at}/{runner} is not an object'
            expect = spec.get('expect')
            if expect != 'success' and expect not in ids:
                return f'probes/{at}/{runner}/expect names no entry'
            if expect != 'success' and runner not in next(
                entry['observed'] for entry in entries if entry['id'] == expect
            ):
                return f'probes/{at}/{runner}/expect is not seen by {runner}'
            if spec.get('until') not in STAGES:
                return f'probes/{at}/{runner}/until is not a stage'
    return None


def _observation(seen: Any, runner: str) -> str | None:
    r"""What is wrong with ``seen`` as a refusal ``runner`` observes."""
    if not isinstance(seen, dict) or set(seen) != set(FIELDS[runner]):
        return 'not the fields the runner observes'
    for key, value in seen.items():
        if value is not None and not _strings(value):
            return f'{key} is neither text nor null'
    return None


def success(until: str) -> dict[str, str]:
    r"""The observation of a probe that reached ``until``'s result."""
    return {'result': 'success', 'stage': until}


def check(
    document: dict[str, Any], runner: str, observed: dict[str, Any]
) -> list[str]:
    r"""Each way ``observed``, the runner's observation of each probe by
    its id, departs from the map; empty when the run is the map's.

    Evidence that is missing, empty or retyped fails, never passes: a probe
    the map gives the runner and the run did not observe, an observation of
    a probe the map does not give it, and an observation that is neither a
    success nor the fields the runner observes, each text or null.
    """
    failures: list[str] = []
    if runner not in RUNNERS:
        return [f'{runner}: not a runner a map names']
    probes = [probe for probe in document['probes'] if runner in probe]
    if not probes:
        return [f'{runner}: the map gives this runner no probe']
    if not isinstance(observed, dict) or not observed:
        return [f'{runner}: nothing was observed']
    entries = {entry['id']: entry for entry in document['entries']}
    given = {probe['id'] for probe in probes}
    for name in sorted(set(observed) - given):
        failures.append(f'{name}: observed, and not a probe of {runner}')
    hit: set[str] = set()
    for probe in probes:
        spec = probe[runner]
        name = probe['id']
        if name not in observed:
            failures.append(f'{name}: not observed')
            continue
        seen = observed[name]
        if seen != success(spec['until']):
            problem = _observation(seen, runner)
            if problem is not None:
                failures.append(f'{name}: {problem}')
                continue
        expect = spec['expect']
        wanted = (
            success(spec['until'])
            if expect == 'success'
            else entries.get(expect, {}).get('observed', {}).get(runner)
        )
        if seen == wanted:
            if expect != 'success':
                hit.add(expect)
            continue
        listed = [
            entry['id']
            for entry in document['entries']
            if entry['observed'].get(runner) == seen
        ]
        if seen == success(spec['until']):
            failures.append(f'{name}: succeeded, where the map has {expect}')
        elif listed:
            failures.append(
                f'{name}: hit {listed[0]}, where the map has {expect}'
            )
        else:
            failures.append(
                f'{name}: a refusal the map does not list, {_said(seen)}'
            )
    for entry in document['entries']:
        if runner in entry['observed'] and entry['id'] not in hit:
            failures.append(f'{entry["id"]}: listed for {runner} and not hit')
    return failures


def _said(seen: dict[str, Any]) -> str:
    return ', '.join(f'{key} {seen[key]!r}' for key in sorted(seen))


# --- the hosts' descriptions agree -------------------------------------------------

#: What each probe both core hosts reach ``describe_source`` on records of
#: their core operations (``contract/pinned/base.adventureworks.probes.json``):
#: the description, the schema ``materialize`` samples, and, from T1.23, the
#: core's warnings, each its code and fields, which each host words itself.
AGREE = ('described', 'materialize_schema', 'warnings')
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


def digest(value: Any) -> str:
    r"""The SHA-256 of ``value`` as canonical JSON: keys sorted, no
    whitespace, UTF-8; ``sdm-testkit``'s ``tests/refusal_map.rs`` writes
    the same."""
    text = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    )
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


#: What a description's zoned column says of its zone, in the family pin:
#: DuckDB hands a ``TIMESTAMPTZ`` over in the database's zone, the
#: machine's, so the zone is the host machine's, not the case's.
DATABASE_ZONE = '<the database zone>'


def zone_free(described: Any) -> Any:
    r"""``described`` with each zoned column's received zone written as
    :data:`DATABASE_ZONE`, so its digest is the same on every machine
    (``contract/pinned/families.probes.json``, T1.23)."""
    described = copy.deepcopy(described)
    for table in described.get('tables', []):
        for column in table.get('columns', []):
            received = column.get('received')
            if isinstance(received, dict) and received.get('zone') is not None:
                received['zone'] = DATABASE_ZONE
    return described


def agree(pin: Any, observed: dict[str, dict[str, str]]) -> list[str]:
    r"""Each way ``observed`` (a runner's digests, by probe, of what
    :data:`AGREE` names) departs from ``pin``'s; empty when they are the
    pin's. Missing, empty or retyped evidence fails: a pin that is not one,
    a probe the pin lists and the run did not reach, one the run reached
    and the pin does not list, and a digest that is not one."""
    probes = pin.get('probes') if isinstance(pin, dict) else None
    if not isinstance(probes, dict) or not probes:
        return ['the pin lists no probe']
    failures = []
    for name in sorted(set(probes) | set(observed)):
        wanted, seen = probes.get(name), observed.get(name)
        if not isinstance(wanted, dict) or not wanted:
            failures.append(f'{name}: reached describe_source, not pinned')
            continue
        if (
            any(
                key not in AGREE
                or not isinstance(v, str)
                or not _SHA256.match(v)
                for key, v in wanted.items()
            )
            or 'described' not in wanted
        ):
            failures.append(f'{name}: the pin is not digests')
            continue
        if seen is None:
            failures.append(f'{name}: pinned, and describe_source not reached')
        elif seen != wanted:
            failures.append(
                f'{name}: '
                + ', '.join(
                    key for key in AGREE if seen.get(key) != wanted.get(key)
                )
                + " not the pin's"
            )
    return failures


# --- python-core ------------------------------------------------------------------


def _edit(schema: Any, edits: list[dict[str, Any]]) -> None:
    r"""The probe's edits to the schema, in order: the case format's two,
    and a probe's own ``set_primary_key`` and ``set_relationships``."""
    from sdm_client import Relationship

    for edit in edits:
        op = edit['op']
        if op == 'clear_time_column':
            schema[edit['table']].time_column = None
        elif op == 'delete_column':
            del schema[edit['table']][edit['column']]
        elif op == 'set_primary_key':
            schema[edit['table']].primary_key = edit['column']
        elif op == 'set_relationships':
            schema.relationships = [
                Relationship(**link) for link in edit['relationships']
            ]
        else:
            raise MapFailure('a probe names an edit this driver does not make')


def observe_python_core(
    runner: Any, spec: dict[str, Any], bases: dict[str, list[str]]
) -> tuple[dict[str, Any], Any]:
    r"""One probe on ``python-core``: what it observed, and what crossed
    ``sdm_client._native`` (``runner``'s capture of the core runs).

    ``runner`` is ``run_python.py`` loaded as a module; ``spec`` the
    probe's ``python-core`` input: its routes, its input in the case
    format (``edits`` may also set a key or the relationships), and how far
    it goes.
    """
    from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient
    from sdm_client._rust import run as run_module
    from sdm_connectors import DuckDBSource

    given = spec['input']
    case = {'id': 'probe', 'input': given}
    arguments = runner._materialize_arguments(given['materialize'])
    from_source = runner._from_source_arguments(given.get('from_source', {}))
    connection = runner._connection(case, bases)
    serve = runner._sibling('scripted_endpoint').serve
    imported = run_module.native
    boundary = runner._Boundary()
    capture = boundary.reset()
    previous = os.environ.get('SDM_CORE_ROUTES')
    os.environ['SDM_CORE_ROUTES'] = ','.join(spec['routes'])
    stage = 'from_source'
    try:
        with (
            serve() as (url, _received),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter('ignore')
            try:
                schema = Schema.from_source(
                    DuckDBSource(connection), **from_source
                )
                # Built from the core's description, the object is that
                # description: its own key marking changes nothing (T1.21d).
                if runner.presented(schema):
                    raise MapFailure(
                        "the Schema built from the core's description "
                        'departs from it'
                    )
                _edit(schema, given.get('edits', []))
                if spec['until'] != 'from_source':
                    stage = 'materialize'
                    request = PredictiveQuery(given['pql'], schema).materialize(
                        **arguments
                    )
                if spec['until'] == 'predict':
                    stage = 'predict'
                    with SDMClient(url) as client:
                        result = client.predict(request, model=KumoRelational())
                    if len(result.to_pandas()) != len(arguments['indices']):
                        raise MapFailure('a result without a row per entity')
                return success(spec['until']), capture
            except MapFailure:
                raise
            except Exception as error:  # noqa: BLE001 - the probe's outcome
                return _refusal(runner, stage, error), capture
    finally:
        run_module.native = imported  # type: ignore[assignment]
        if previous is None:
            os.environ.pop('SDM_CORE_ROUTES', None)
        else:
            os.environ['SDM_CORE_ROUTES'] = previous
        connection.close()


def _refusal(runner: Any, stage: str, error: BaseException) -> dict[str, Any]:
    details = getattr(error, 'details', None)
    details = details if isinstance(details, dict) else {}
    code = getattr(error, 'code', None)
    return {
        'stage': stage,
        'error': 'RouteError'
        if runner._route_error(error)
        else type(error).__name__,
        'code': code if isinstance(code, str) else None,
        'operation': _text(details.get('operation')),
        'rule': _text(details.get('rule')),
    }


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def probe_input(case: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    r"""A probe's input in the case format, for validating against the case
    schema: its own edits left out, which the case format does not have."""
    given = copy.deepcopy(spec['input'])
    kept = [
        edit
        for edit in given.get('edits', [])
        if edit.get('op') in ('clear_time_column', 'delete_column')
    ]
    if kept:
        given['edits'] = kept
    else:
        given.pop('edits', None)
    return {**copy.deepcopy(case), 'input': given}


# --- the probes, bound to the case ---------------------------------------------------


def _rename(case: dict[str, Any]) -> Any:
    r"""Each table's name in the case, to the source table it names."""
    names = case['input'].get('from_source', {}).get('tables', {})
    names = names if isinstance(names, dict) else {}
    return lambda table: names.get(table, table)


def _renamed_link(link: dict[str, str], rename: Any) -> dict[str, str]:
    return {
        **link,
        'source_table': rename(link['source_table']),
        'target_table': rename(link['target_table']),
    }


def _python_step(
    given: dict[str, Any], step: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    r"""``given`` with one removal of a python-core probe's ``derive``."""
    given = copy.deepcopy(given)
    rename = _rename(case)
    [(op, value)] = step.items()
    if op == 'rename_tables':
        source = given.get('from_source', {})
        names = source.pop('tables')
        local = names.get
        for key in ('primary_keys', 'time_columns'):
            if key in source:
                source[key] = {local(t, t): v for t, v in source[key].items()}
        if 'relationships' in source:
            source['relationships'] = [
                _renamed_link(link, lambda t: local(t, t))
                for link in source['relationships']
            ]
        for edit in given.get('edits', []):
            edit['table'] = local(edit['table'], edit['table'])
        for alias, table in names.items():
            given['pql'] = given['pql'].replace(f'{alias}.', f'{table}.')
    elif op == 'remove':
        *parents, last = value.split('.')
        at = given
        for key in parents:
            at = at[key]
        del at[last]
        if 'from_source' in given and not given['from_source']:
            del given['from_source']
    elif op == 'drop_columns':
        given['setup'] = [
            *given.get('setup', []),
            *(
                f'ALTER TABLE {table} DROP COLUMN "{column}"'
                for table, column in value
            ),
        ]
    elif op == 'edit_in':
        # The case's own keys and relationships, set by editing the schema
        # rather than passed to Schema.from_source.
        source = case['input'].get('from_source', {})
        edits: list[dict[str, Any]] = []
        if 'primary_keys' in value:
            edits += [
                {'op': 'set_primary_key', 'table': rename(t), 'column': k}
                for t, k in source.get('primary_keys', {}).items()
            ]
        if 'relationships' in value:
            edits.append(
                {
                    'op': 'set_relationships',
                    'relationships': [
                        _renamed_link(link, rename)
                        for link in source.get('relationships', [])
                    ],
                }
            )
        given['edits'] = [*edits, *given.get('edits', [])]
    elif op == 'drop_relationship':
        for edit in given['edits']:
            if edit['op'] == 'set_relationships':
                edit['relationships'] = [
                    link for link in edit['relationships'] if link != value
                ]
    elif op == 'delete_column':
        table, column = value
        given['edits'] = [
            *given.get('edits', []),
            {'op': 'delete_column', 'table': table, 'column': column},
        ]
    else:
        raise MapFailure('a probe names a step this deriver does not take')
    return given


def _rust_step(core: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    r"""``core`` with one removal of a rust-host probe's ``derive``, on
    describe_source's source: a field emptied (``clear``), the
    relationships left out (``remove``), or the tables' names given up
    (``rename_tables``), each table then named as the source names it, in
    the keys, time columns and relationships too, as the python-core step
    of the same name renames them (T1.23, whose ``SourceSpec`` names
    tables and carries relationships)."""
    core = copy.deepcopy(core)
    [(op, value)] = step.items()
    source = core['describe_source']['source']
    if op == 'clear':
        source[value] = [] if isinstance(source[value], list) else {}
    elif op == 'remove' and value == 'relationships':
        del source['relationships']
    elif op == 'rename_tables' and value is True:
        local = {table['name']: table['source'] for table in source['tables']}
        source['tables'] = []
        for key in ('primary_keys', 'time_columns'):
            source[key] = {local.get(t, t): v for t, v in source[key].items()}
        if 'relationships' in source:
            source['relationships'] = [
                [
                    {**end, 'table': local.get(end['table'], end['table'])}
                    for end in link
                ]
                for link in source['relationships']
            ]
    else:
        raise MapFailure('a probe names a step this deriver does not take')
    return core


def _cases(given: dict[str, Any]) -> dict[str, dict[str, Any]]:
    r"""``given`` as the cases by id: one case, or a mapping of them."""
    if given.get('format') == 'sdm-case/1':
        return {given['id']: given}
    return given


def _probe_case(document: dict[str, Any], probe: dict[str, Any]) -> Any:
    return probe.get('case', document.get('case'))


def derive(
    given: dict[str, Any], document: dict[str, Any]
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    r"""Each probe's input as its ``derive`` makes it of its case, by
    runner and probe id (python-core: its input; rust-host: its core
    operations), and each way a probe's derivation does not apply.
    ``given`` is the map's case, or a family map's cases by id."""
    cases = _cases(given)
    failures: list[str] = []
    derived: dict[str, dict[str, Any]] = {runner: {} for runner in RUNNERS}
    owner: dict[str, Any] = {}
    for probe in document['probes']:
        name = probe['id']
        case = cases.get(_probe_case(document, probe))
        if case is None:
            failures.append(f'{name}: names no case given')
            continue
        for runner in RUNNERS:
            spec = probe.get(runner)
            if spec is None:
                continue
            rule = spec.get('derive')
            if not isinstance(rule, dict) or not isinstance(
                rule.get('steps'), list
            ):
                failures.append(f'{name}/{runner}: no derive')
                continue
            start = rule.get('from')
            if start == 'case':
                if rule['steps']:
                    failures.append(f'{name}/{runner}: the case with steps')
                    continue
                base = (
                    case['input'] if runner == 'python-core' else case['core']
                )
            elif start in derived[runner] and owner[start] == case['id']:
                base = derived[runner][start]
            else:
                failures.append(f'{name}/{runner}: derives from no probe')
                continue
            made = copy.deepcopy(base)
            try:
                for step in rule['steps']:
                    made = (
                        _python_step(made, step, case)
                        if runner == 'python-core'
                        else _rust_step(made, step)
                    )
            except (KeyError, ValueError, TypeError, MapFailure):
                failures.append(f'{name}/{runner}: its steps do not apply')
                continue
            derived[runner][name] = made
            owner[name] = case['id']
    return failures, derived


def bound(given: dict[str, Any], document: dict[str, Any]) -> list[str]:
    r"""Each way the map's probes are not their case's journey with the
    removals each names: empty when every probe's input is what its
    ``derive`` makes of its case. ``given`` is the map's case, or a family
    map's cases by id, every case of the family among them.

    The first probe of each case and runner derives from the case with no
    removal, so it is the case itself (python-core: its input and the stage
    it stops at; rust-host: its core operations); every other probe derives
    from an earlier one of the same case by the removals it names. A
    rust-host probe's setup is its python-core twin's, and so are the edits
    it makes to its own description before materialize samples it, the
    probe naming no schema (T1.21d); where its case names materialize and
    predict, they are the case's, and otherwise, as on the AdventureWorks
    case, which names describe_source alone, its other inputs, which
    python-core captures, are held to that twin's at run time.
    """
    cases = _cases(given)
    failures, derived = derive(cases, document)
    if 'cases' in document and set(document['cases']) != set(cases):
        failures.append("the family's cases are not the cases given")
    started: set[tuple[str, str]] = set()
    for probe in document['probes']:
        name = probe['id']
        case = cases.get(_probe_case(document, probe))
        if case is None:
            continue
        for runner in RUNNERS:
            spec = probe.get(runner)
            if spec is None or name not in derived[runner]:
                continue
            made = derived[runner][name]
            if spec['derive'].get('from') == 'case':
                started.add((case['id'], runner))
                if runner == 'python-core' and spec['until'] != case['until']:
                    failures.append(f"{name}/{runner}: not the case's until")
            if runner == 'python-core':
                if spec['input'] != made:
                    failures.append(f'{name}/{runner}: not its derived input')
                continue
            held = spec['core']
            if held != made and not (
                set(made) == {'describe_source'}
                and held.get('describe_source') == made['describe_source']
            ):
                failures.append(f'{name}/{runner}: not its derived core')
            twin = probe.get('python-core')
            if twin is None or spec.get('setup') != twin['input'].get('setup'):
                failures.append(f"{name}/{runner}: not its twin's setup")
            # materialize samples rust-host's own description with its
            # twin's edits made; the probe names no schema (T1.21d).
            if 'materialize' in held and (
                twin is None
                or spec.get('edits', []) != twin['input'].get('edits', [])
            ):
                failures.append(f"{name}/{runner}: not its twin's edits")
            if 'schema' in held.get('materialize', {}):
                failures.append(
                    f'{name}/{runner}: names the schema materialize samples'
                )
    # Every case of a family is probed from itself, on each runner the
    # family's map runs.
    if 'cases' in document:
        for runner in RUNNERS:
            if not any(runner in probe for probe in document['probes']):
                continue
            for case_id in document['cases']:
                if (case_id, runner) not in started:
                    failures.append(f'{case_id}/{runner}: no probe of the case')
    return failures


# --- the writer -----------------------------------------------------------------------

#: The stage whose result each core operation gives.
STAGE_OF = {
    'describe_source': 'from_source',
    'materialize': 'materialize',
    'predict': 'predict',
}
#: The fields a person writes in an entry; :func:`write` keeps them.
WRITTEN = ('id', 'operation', 'rule', 'owner', 'register', 'where')
UNSET = 'UNSET'


def _key(seen: dict[str, Any]) -> tuple[Any, ...] | None:
    r"""What both runners observe of a refusal the core names by rule."""
    if seen.get('operation') is None or seen.get('rule') is None:
        return None
    return (seen.get('code'), seen['operation'], seen['rule'])


def write(
    document: dict[str, Any],
    given: dict[str, Any],
    observe: Any,
    rust_observed: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    r"""The map ``document`` describes, its run recomputed, and what a
    person must still write.

    Kept as written: the map's own fields, each entry's written fields and
    any others but ``observed``, and each probe's id, case, ``removes``,
    derivation, python-core routes and stopping stage. Recomputed: each
    probe's input from its derivation (rust-host's setup and edits its
    twin's, its stopping stage its last operation's), each runner's
    observation (python-core's by ``observe(spec)``, rust-host's from
    ``rust_observed``, by probe id; without it the map's rust-host
    observations stay), the entry each probe hits, and each entry's
    ``observed``. A refusal no entry lists becomes an entry whose written
    fields read ``UNSET``, and an entry no probe hits any more is dropped;
    both are reported.
    """
    document = copy.deepcopy(document)
    failures, derived = derive(given, document)
    if failures:
        return document, failures
    report: list[str] = []
    entries = document['entries']
    for entry in entries:
        entry.setdefault('observed', {})
    # What each entry was observed as before this run, which a refusal is
    # matched against; its observations are this run's.
    before = {e['id']: copy.deepcopy(e['observed']) for e in entries}
    runners = ['python-core'] + (
        ['rust-host'] if rust_observed is not None else []
    )
    if rust_observed is not None:
        # Supplied rust-host evidence is the whole of it: a probe of
        # rust-host it does not observe, or an observation of no such
        # probe, is reported, never filled from the map.
        given = {
            probe['id'] for probe in document['probes'] if 'rust-host' in probe
        }
        if not isinstance(rust_observed, dict):
            return document, ['rust-host: the observations are not a map']
        report += [
            f'{name}/rust-host: observed, and not a probe of rust-host'
            for name in sorted(set(rust_observed) - given)
        ]
    for entry in entries:
        for runner in runners:
            entry['observed'].pop(runner, None)
    hit: set[str] = set()
    this_probe: dict[str, dict[str, Any]] = {}
    for probe in document['probes']:
        name = probe['id']
        twin = probe.get('python-core')
        rust = probe.get('rust-host')
        if twin is not None:
            twin['input'] = derived['python-core'][name]
        if rust is not None:
            core = derived['rust-host'][name]
            if set(core) == {'describe_source'}:
                # A case naming describe_source alone (the AdventureWorks
                # case): the probe's later inputs are held to its twin's
                # at run time, not derived, so they stay as written.
                core = {**rust.get('core', {}), **core}
            rust['setup'] = twin['input'].get('setup') if twin else None
            rust['core'] = core
            edits = twin['input'].get('edits') if twin else None
            if 'materialize' in core and edits:
                rust['edits'] = edits
            else:
                rust.pop('edits', None)
            rust['until'] = STAGE_OF[list(core)[-1]]
        for runner in RUNNERS:
            spec = probe.get(runner)
            if spec is None:
                continue
            if runner == 'rust-host' and rust_observed is None:
                if spec.get('expect', 'success') != 'success':
                    hit.add(spec['expect'])
                continue
            seen = (
                observe(spec)
                if runner == 'python-core'
                else (rust_observed or {}).get(name)
            )
            if seen is None:
                report.append(f'{name}/{runner}: not observed')
                continue
            if seen == success(spec['until']):
                spec['expect'] = 'success'
                continue
            problem = _observation(seen, runner)
            if problem is not None:
                report.append(f'{name}/{runner}: {problem}')
                continue
            entry = next(
                (
                    e
                    for e in entries
                    if e['observed'].get(runner) == seen
                    or before.get(e['id'], {}).get(runner) == seen
                ),
                None,
            )
            twin_entry = this_probe.get(name)
            if (
                entry is None
                and twin_entry is not None
                and runner not in twin_entry['observed']
                and all(
                    other.get(field) == seen.get(field)
                    for other in twin_entry['observed'].values()
                    for field in ('code', 'operation', 'rule')
                )
            ):
                # The same probe's refusal as the other runner saw it: one
                # refusal, the same code, rule and operation or neither.
                entry = twin_entry
            if entry is None and _key(seen) is not None:
                # The other runner's entry for the same rule.
                entry = next(
                    (
                        e
                        for e in entries
                        if runner not in before.get(e['id'], {})
                        and runner not in e['observed']
                        and any(
                            _key(other) == _key(seen)
                            for other in [
                                *e['observed'].values(),
                                *before.get(e['id'], {}).values(),
                            ]
                        )
                    ),
                    None,
                )
            if entry is None and _key(seen) is not None:
                # An entry written for a rule, not yet observed.
                entry = next(
                    (
                        e
                        for e in entries
                        if not e['observed']
                        and not before.get(e['id'])
                        and (e['operation'], e['rule'])
                        == (seen['operation'], seen['rule'])
                    ),
                    None,
                )
            if entry is None:
                entry = {
                    'id': '/'.join(
                        str(seen.get(k) or seen.get('code'))
                        for k in ('operation', 'rule')
                    ),
                    **{key: UNSET for key in WRITTEN[1:]},
                    'observed': {},
                }
                entries.append(entry)
                report.append(f'{name}/{runner}: a refusal no entry lists')
            entry['observed'][runner] = seen
            this_probe[name] = entry
            spec['expect'] = entry['id']
            hit.add(entry['id'])
    kept = []
    for entry in entries:
        if entry['id'] in hit or entry['observed']:
            kept.append(entry)
        else:
            report.append(f'{entry["id"]}: no probe hits it; dropped')
    document['entries'] = kept
    report += [
        f'{entry["id"]}: {key} is not written'
        for entry in kept
        for key in WRITTEN
        if entry.get(key) == UNSET
    ]
    return document, report


def _read(path: Path) -> Any:
    r"""The JSON at ``path``, refused by name when it is not JSON or is
    nested past what the reader takes."""
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except RecursionError:
        raise MapFailure(f'{path.name}: nested too deeply') from None
    except ValueError:
        raise MapFailure(f'{path.name}: not JSON') from None


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import importlib.util
    import sys

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('action', choices=['write'])
    parser.add_argument('map', type=Path)
    parser.add_argument('--rust-observed', type=Path)
    arguments = parser.parse_args(argv)
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        'run_python', here / 'run_python.py'
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules['run_python'] = runner
    spec.loader.exec_module(runner)
    document = _read(arguments.map)
    if (
        not isinstance(document, dict)
        or document.get('format') != FORMAT
        or not isinstance(document.get('probes'), list)
        or not isinstance(document.get('entries'), list)
    ):
        raise MapFailure(f'{arguments.map.name}: not a map or its skeleton')
    every = {
        case['id']: case
        for case, _ in runner.load_cases([here.parent / 'cases'])
    }
    wanted = document.get('cases', [document.get('case')])
    given = {case_id: every[case_id] for case_id in wanted}
    rust = None
    if arguments.rust_observed is not None:
        rust = _read(arguments.rust_observed)
        rust = rust.get(arguments.map.name) if isinstance(rust, dict) else None
        if not isinstance(rust, dict):
            raise MapFailure(
                f'{arguments.rust_observed.name}: no observations of '
                f'{arguments.map.name}'
            )
    bases = runner.load_bases()
    try:
        written, report = write(
            document,
            given,
            lambda spec: observe_python_core(runner, spec, bases)[0],
            rust,
        )
    except RecursionError:
        raise MapFailure(f'{arguments.map.name}: nested too deeply') from None
    except (KeyError, TypeError, AttributeError, ValueError):
        raise MapFailure(
            f'{arguments.map.name}: not a map or its skeleton'
        ) from None
    for line in report:
        print(line)
    problem = shape(written)
    if problem is not None:
        print(f'not yet a map: {problem}')
    # The map is rewritten only when the run is the whole of its evidence
    # and nothing is left to write by hand; otherwise the file is left as
    # it was, and the map the run would make is beside it, for a person to
    # finish.
    text = json.dumps(written, indent=2, ensure_ascii=False) + '\n'
    draft = arguments.map.with_name(arguments.map.name + '.draft')
    if report or problem:
        draft.write_text(text, encoding='utf-8')
        print(f'{arguments.map.name} left as it was; the draft is {draft.name}')
        return 1
    arguments.map.write_text(text, encoding='utf-8')
    # A draft an earlier run left is stale once the map is written.
    draft.unlink(missing_ok=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(_main())
    except MapFailure as failure:
        raise SystemExit(f'not a map: {failure}') from None
