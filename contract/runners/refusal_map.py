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

:func:`check` holds the observations to the map: every probe the map gives
the runner is observed, as the entry it names or as success; a refusal the
map does not list fails, and so does an entry no probe of a runner it
lists reaches.
"""

from __future__ import annotations

import copy
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
#: The tasks a refusal of T1.21b's map may name as its owner.
OWNER = re.compile(r'^T1\.2[2-8]$')


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
        'reading',
        'gaps',
        'entries',
        'probes',
    }:
        return 'keys a refusal map does not have'
    gaps = document.get('gaps', [])
    if not isinstance(gaps, list) or not all(
        isinstance(gap, dict)
        and all(
            _strings(gap.get(key)) for key in ('input', 'contract', 'owner')
        )
        for gap in gaps
    ):
        return 'gaps are not each an input, its contract and an owner'
    if not _strings(document.get('case')):
        return 'no case'
    entries = document.get('entries')
    probes = document.get('probes')
    if not isinstance(entries, list) or not entries:
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
        if not given or set(probe) - {'id', 'removes', *RUNNERS}:
            return f'probes/{at} has no runner, or keys it should not'
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
