#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Python runner of the contract cases, in three modes.

Run with ``python -I``, from the interpreter of the environment under test:

    python -I contract/runners/run_python.py --mode MODE [--reference DIR]
        [--database DATABASE] [--no-capture] [--inventory ID]... --out FILE
        [CASE_OR_DIRECTORY...]
    python -I contract/runners/run_python.py legs --database DATABASE
        --providers FILE OBSERVATIONS...

``legs`` holds a database job's observations to the staging rule
(:func:`check_legs`), each core leg's providers given as JSON in FILE.

- ``python-ref``: the reference, the oracle. ``--reference`` names the
  reference checkout, whose environment must be the interpreter's. Its guard
  (``scripts/reference_guard.py``) is loaded by path, and its
  ``require_isolation()`` runs before the first case and after the last;
  either failing stops the run, which writes nothing. ``sdm_client`` must be
  the reference checkout's.
- ``python-core``: the candidate, with ``SDM_CORE_ROUTES`` set for each case
  to its ``routes``. The core operations the case starts, with their inputs,
  and the core's answer are recorded at ``sdm_client._native``'s boundary,
  changing nothing that crosses it.
- ``python-old``: the candidate on its default routes, a smoke runner that
  must complete every case (decision H10).

Each case builds its DuckDB base on a fresh in-memory connection; with
``--database sqlite`` or ``postgres`` (T2.1a), the contract loader
(``databases.py``) loads its tables into a new SQLite file or a new
Postgres database's public schema, and the source is given a driver connection to that, its
observation keyed ``<case>@<database>``; a case whose ``databases`` leave
the database out is not run. Then it makes the public calls: ``Schema.from_source``, then ``materialize`` (the S0 call),
then ``SDMClient.predict`` against the scripted endpoint
(``scripted_endpoint.py``), served on localhost and reached through the
client's real transport, as far as the case's ``until``.

The source is handed the connection through the test-only driver capture
(design, "Driver capture"; decision Q2), which records every statement the
driver runs, its typed parameters, each result read and every relation
bound by name, derived cursors included, and changes nothing that crosses
it; ``--no-capture`` hands it the connection itself, the conservation
control. Setup, the statements before the connection reaches the source,
is outside it. DuckDB's own query log is read after each case either way,
so ``capture_check.py`` can hold the captured counts to it and the two runs
to the same public projection (``public``: the schema, each request the
scripted endpoint received, the result). Cases of the Phase 1b inventory are
named with ``--inventory ID``. An exception a public call raises is the
case's outcome, recorded by class, ``StructuredDataError.code`` and stage, a
``RuntimeError`` included; the comparator decides whether it is the expected
one. A runner failure is anything else: the guard failing, a missing
``sdm_client._native`` or a refused forced route (``RouteError``), an error
in the runner itself, an observation that does not validate, or the process
dying. It exits non-zero and leaves no output file, an earlier run's
included; observations are written only when every case was recorded. No message text is recorded (decision H17).

This file also holds the case loader and the Phase 1b inventory loader,
which maps each inventory case into the case format and back without loss.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import struct
import subprocess
import sys
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CONTRACT = HERE.parent
REPO = CONTRACT.parent
BASES = CONTRACT / 'bases.yaml'
INVENTORY = REPO / 'docs/design/rust-core/phase-0/phase-1b-inventory.yaml'
SCHEMAS = {
    'case': CONTRACT / 'case.schema.json',
    'inventory': CONTRACT / 'inventory.schema.json',
    'observation': CONTRACT / 'observation.schema.json',
}
MODES = ('python-ref', 'python-core', 'python-old')

#: A cell's earlier pins, still that cell of ``CELLS``: the floor's before
#: T2.0c moved it to pandas 2.2.2 and NumPy 1.23.5 (KO2-37). The phase
#: branch's ci.yml installs these until that landing, so the floor job
#: records its cell on either side of it.
EARLIER_PINS = {
    ('3.10', '2.2.0', '14.0.1', '1.26.4'): 'floor-pandas2.2',
}

#: The databases every case runs on unless its file says otherwise (design
#: choice 2, KO2-19), in this order. DuckDB's observations keep the bare
#: case ID; the others are keyed ``<case>@<database>``.
DATABASES = ('duckdb', 'sqlite', 'postgres')
#: The warehouse legs (Phase 3 design choice 1), in this order, keyed
#: ``<case>@<leg>`` as the databases are: an S2 case runs on the ones its
#: ``warehouses`` names and only there; an S1 case without ``databases`` on
#: every one as well (:func:`case_legs`).
WAREHOUSES = ('databricks-sql', 'databricks-spark', 'snowflake', 'lakebase')
#: Every leg a case may run on.
LEGS = DATABASES + WAREHOUSES
#: Each ``input.connect`` path and the one leg it opens (design choice 1).
CONNECT_LEGS = {
    'warehouse': 'databricks-sql',
    'spark': 'databricks-spark',
    'snowpark': 'snowflake',
    'lakebase': 'lakebase',
}
#: Each ``input.view`` kind and the legs that keep it: a metric view is
#: Databricks', a semantic view Snowflake's.
VIEW_LEGS = {
    'metric_view': ('databricks-sql', 'databricks-spark'),
    'semantic_view': ('snowflake',),
}
#: The S1 cases left off the warehouse legs: the two 100,001-row batching
#: cases, which add nothing there to what they test (KO3-45).
OFF_WAREHOUSES = ('multicall.entities.100000', 'multicall.entities.100001')

#: The runtime cells (design, "Runtime cells"): Python, pandas, pyarrow and
#: NumPy versions. An environment with exactly one's versions is that cell.
CELLS = {
    'ref-pandas3': ('3.12', '3.0.6', '25.0.1', '2.5.3'),
    'dbr17-pandas2': ('3.12', '2.2.3', '19.0.1', '2.1.3'),
    'floor-pandas2.2': ('3.10', '2.2.2', '14.0.1', '1.23.5'),
}


class RunnerFailure(Exception):
    r"""The runner, not the case, failed; its text never quotes an input."""


# --- schemas and cases ---------------------------------------------------------


def validator(name: str) -> Any:
    r"""A JSON Schema validator for ``name`` (``case``, ``inventory`` or
    ``observation``), or for a ``$ref`` into one, resolving the references
    between the three."""
    import jsonschema
    from referencing import Registry, Resource

    schemas = {
        key: json.loads(path.read_text(encoding='utf-8'))
        for key, path in SCHEMAS.items()
    }
    registry = Registry().with_resources(
        (schema['$id'], Resource.from_contents(schema))
        for schema in schemas.values()
    )
    schema = schemas[name] if name in schemas else {'$ref': name}
    cls = jsonschema.validators.validator_for(schemas['case'])
    cls.check_schema(schema)
    _long_lists_held(schema)
    return cls(schema, registry=registry)


def _invalid(validate: Any, document: Any) -> str | None:
    r"""The first schema error's location and rule, or None; never the value.

    jsonschema walks every item of a list: a case or an observation of
    100,000 entities takes about a second, and every tool that loads them
    pays it on each load. So a long list's items past its leading ones are
    checked in plain Python by their list's rule (:data:`_LONG_LISTS`), and
    jsonschema sees the document with the list cut. A document taken that
    way is one the whole-document validation takes; any other is
    validated whole, so a refusal names what it names (T2.1l).
    """
    lists = _LONG_LISTS.get(_schema_id(validate.schema), ())
    cut = _cut(document, lists) if lists else None
    if cut is not None and next(iter(validate.iter_errors(cut)), None) is None:
        return None
    error = next(iter(validate.iter_errors(document)), None)
    if error is None:
        return None
    where = '/'.join(str(part) for part in error.absolute_path) or '(root)'
    return f'{where} fails {error.validator!r}'


#: Each item of a list, in a path into a document.
_EACH = '*'
#: Where a long list may be, by the schema's ``$id``: its path in the
#: document, its path in the schema, the schema it has there (beside its
#: description), and its items' rule as Python reads them, the rule
#: :func:`_indices` holds an entity key to (a JSON integer or a string) or
#: an object. A case's call lists the entities it asks about, and its core
#: and its observation's core the Arrow column the host passed.
_LONG_LISTS: dict[
    str,
    tuple[
        tuple[
            tuple[str, ...],
            tuple[str | int, ...],
            dict[str, Any],
            Callable[[Any], bool],
        ],
        ...,
    ],
] = {
    'urn:sdm:case:1': (
        (
            ('input', 'materialize', 'indices'),
            ('$defs', 'materialize', 'properties', 'indices'),
            {'type': 'array', 'items': {'type': ['integer', 'string']}},
            lambda item: type(item) in (int, str),
        ),
        (
            ('input', 'materialize', 'candidates'),
            ('$defs', 'materialize', 'properties', 'candidates'),
            {'type': 'array', 'items': {'type': ['integer', 'string']}},
            lambda item: type(item) in (int, str),
        ),
        (
            ('core', 'materialize', 'indices'),
            (
                'properties',
                'core',
                'properties',
                'materialize',
                'properties',
                'indices',
            ),
            {'type': 'array', 'items': {'type': 'object'}},
            lambda item: type(item) is dict,
        ),
    ),
    'urn:sdm:observation:1': (
        (
            ('core', 'inputs', _EACH, 'indices'),
            (
                'properties',
                'core',
                'properties',
                'inputs',
                'items',
                'oneOf',
                2,
                'properties',
                'indices',
            ),
            {'type': ['array', 'null'], 'items': {'type': 'object'}},
            lambda item: type(item) is dict,
        ),
    ),
}
#: The leading items of a long list that jsonschema still sees.
_KEPT = 8


def _schema_id(schema: Any) -> str:
    return str(schema.get('$id', '')) if isinstance(schema, dict) else ''


def _at(document: Any, where: tuple[str | int, ...]) -> Any:
    r"""The value at ``where`` in ``document``, or None where there is
    none."""
    for key in where:
        if isinstance(document, dict) and key in document:
            document = document[key]
        elif isinstance(document, list) and isinstance(key, int):
            document = document[key] if key < len(document) else None
        else:
            return None
    return document


def _long_lists_held(schema: Any) -> None:
    r"""Refuse a schema whose long lists hold more than their items' one
    rule, or whose list a ``oneOf`` beside it could read too: the items
    past a list's leading ones are checked by that rule alone (T2.1l)."""
    for _, where, wanted, _ in _LONG_LISTS.get(_schema_id(schema), ()):
        node = _at(schema, where)
        held = isinstance(node, dict) and wanted == {
            key: value for key, value in node.items() if key != 'description'
        }
        if 'oneOf' in where:
            # Each other branch takes no key of the list's name.
            at = where.index('oneOf')
            branches = _at(schema, where[: at + 1])
            held = (
                held
                and isinstance(branches, list)
                and all(
                    isinstance(branch, dict)
                    and branch.get('additionalProperties') is False
                    and 'patternProperties' not in branch
                    and where[-1] not in branch.get('properties', {})
                    for number, branch in enumerate(branches)
                    if number != where[at + 1]
                )
            )
        if not held:
            raise RunnerFailure(
                f'{_schema_id(schema)}: {"/".join(map(str, where))} is not '
                'the list the validator checks item by item (T2.1l)'
            )


def _lists(
    document: Any, path: tuple[str, ...]
) -> Iterator[tuple[tuple[str | int, ...], Any]]:
    r"""Each value at ``path`` in ``document``, with its own path."""
    if not path:
        yield (), document
        return
    head, rest = path[0], path[1:]
    if head == _EACH:
        if isinstance(document, list):
            for number, item in enumerate(document):
                for where, found in _lists(item, rest):
                    yield (number, *where), found
    elif isinstance(document, dict) and head in document:
        for where, found in _lists(document[head], rest):
            yield (head, *where), found


def _replaced(document: Any, where: tuple[str | int, ...], value: Any) -> Any:
    r"""``document`` with ``value`` at ``where``, copying only the objects
    and arrays on the way, never changing ``document``."""
    if not where:
        return value
    head, rest = where[0], where[1:]
    copied: Any = (
        list(document) if isinstance(document, list) else dict(document)
    )
    copied[head] = _replaced(document[head], rest, value)
    return copied


def _cut(document: Any, lists: Any) -> Any:
    r"""``document`` with each long list cut to its :data:`_KEPT` leading
    items, or None where an item past them breaks its list's rule."""
    cut = document
    for path, _, _, holds in lists:
        for where, items in list(_lists(document, path)):
            if not isinstance(items, list) or len(items) <= _KEPT:
                continue
            if not all(map(holds, items[_KEPT:])):
                return None
            cut = _replaced(cut, where, items[:_KEPT])
    return cut


def _indices(given: dict[str, Any]) -> str | None:
    r"""An entity key is a JSON integer literal of any size, or a string,
    in ``indices`` and ``candidates`` alike; ``top_k`` and ``fail_call``
    are integers (``rows_per_call`` and ``lag_timesteps`` are passed as
    written, whatever they are).

    JSON Schema counts ``1.0`` as an integer; ``json`` reads it as a float,
    and a bool is no index either, so the type read is checked too, as
    sdm-testkit's loader refuses any number with a fraction or exponent.
    """
    spec = given['materialize']
    for key in ('indices', 'candidates'):
        for position, value in enumerate(spec.get(key, ())):
            if type(value) not in (int, NegativeZero, str):
                return f'input/materialize/{key}/{position} is not an integer'
    if 'top_k' in spec and type(spec['top_k']) not in (int, NegativeZero):
        return 'input/materialize/top_k is not an integer'
    # A whole number as written: -0 is refused, as the Rust loader refuses
    # it (P2-T2.1i-1).
    if 'fail_call' in given and type(given['fail_call']) is not int:
        return 'input/fail_call is not a whole number written as one'
    return None


def on_database(path: str, database: str) -> str:
    r"""A case's DuckDB route or path as the same operation and mode over
    ``database``'s source; any other is itself."""
    head, _, backend = path.rpartition('/')
    return f'{head}/{database}' if backend == 'duckdb' else path


def routes_on(case: dict[str, Any], database: str) -> list[str]:
    r"""The routes python-core forces for ``case`` on ``database``: on DuckDB
    the case's own; elsewhere the case's own over ``database``
    (:func:`on_database`) and every route of each public call the case
    makes, through its ``until`` (T2.1d). Off DuckDB the old route is not
    the reference's (python-old runs on DuckDB only, T2.1a fix 4), so no
    call of a core leg takes it: a case that routes only its DuckDB stop
    (S0's validation, a source refused at ``from_source``) still runs on
    the core where the database lets the call go further."""
    routes = [on_database(route, database) for route in case['routes']]
    if database == 'duckdb':
        return routes
    from sdm_client import _routing

    named = bool((case['input'].get('from_source') or {}).get('tables'))
    calls = list(
        _routing.FROM_SOURCE_DUCKDB_TABLES
        if named
        else _routing.FROM_SOURCE_DUCKDB
    )
    if case['until'] != 'from_source':
        calls += _routing.MATERIALIZE_DUCKDB
    if case['until'] == 'predict':
        calls += _routing.PREDICT_DUCKDB
    for path in calls:
        if on_database(path, database) not in routes:
            routes.append(on_database(path, database))
    return routes


def case_key(case_id: str, database: str) -> str:
    r"""An observation's key on ``database``: DuckDB's the bare case ID,
    another's ``<case>@<database>``."""
    return case_id if database == 'duckdb' else f'{case_id}@{database}'


def check_legs(
    database: str,
    providers: Mapping[str, Iterable[str]],
    observations: Iterable[Mapping[str, Any]],
) -> list[str]:
    r"""The staging rule (design choice 2, "CI"; T2.1a), one definition for
    ``gate.sh`` and CI: a database job runs cases, and each core leg runs
    on ``database`` exactly when it reports a provider for it.

    ``providers`` maps each core leg (``python-core``, ``rust-host``) to
    the databases it reports (``sdm_client._native.PROVIDERS``,
    ``sdm-contract-run --providers``); ``observations`` are the job's, each
    keyed as :func:`case_key` keys it. The problems, each naming what
    fails: no case observed on ``database``; a leg reporting a provider for
    it with no observation there; a leg without a case the reference side
    observed there (``python-ref``'s committed lines, or on DuckDB
    ``python-old``'s), named by its key. A leg reporting no
    provider for ``database`` is not required there."""
    on = [
        line
        for line in observations
        if isinstance(line.get('case'), str)
        and (
            '@' not in line['case']
            if database == 'duckdb'
            else line['case'].endswith(f'@{database}')
        )
    ]
    if not on:
        return [f'{database}: the job observed no case on {database}']
    observed: dict[str, set[str]] = {}
    for line in on:
        runner = (line.get('identity') or {}).get('runner')
        observed.setdefault(str(runner), set()).add(line['case'])
    reference = observed.get('python-ref', set()) | observed.get(
        'python-old', set()
    )
    problems = []
    for leg in sorted(providers):
        if database not in tuple(providers[leg]):
            continue
        made = observed.get(leg, set())
        if not made:
            problems.append(
                f'{database}: {leg} reports a {database} provider and made '
                f'no observation on {database} (missing)'
            )
            continue
        problems.extend(
            f'{database}: {leg} reports a {database} provider and skipped {key}'
            for key in sorted(reference - made)
        )
    return problems


def case_databases(case: dict[str, Any]) -> tuple[str, ...]:
    r"""The databases a loaded case runs on, in :data:`DATABASES` order:
    all three unless its ``databases`` (with its exclusion) narrows them;
    for a case naming ``warehouses``, those legs, in :data:`WAREHOUSES`
    order, and only those (design choice 1)."""
    if 'warehouses' in case:
        return tuple(leg for leg in WAREHOUSES if leg in case['warehouses'])
    named = case.get('databases', DATABASES)
    return tuple(database for database in DATABASES if database in named)


def case_warehouses(case: dict[str, Any]) -> tuple[str, ...]:
    r"""The warehouse legs a loaded case runs on, in :data:`WAREHOUSES`
    order (design choice 1): an S2 case's ``warehouses``; an S1 case with
    no ``databases`` every leg, but the batching cases of
    :data:`OFF_WAREHOUSES` (KO3-45); any other none. A leg whose engine
    cannot store the case's data is the loader's to refuse by name
    (``databases.Unloadable``), never left out here."""
    if 'warehouses' in case:
        named = case['warehouses']
    elif (
        case.get('slice') == 'S1'
        and 'databases' not in case
        and case.get('id') not in OFF_WAREHOUSES
    ):
        named = WAREHOUSES
    else:
        named = ()
    return tuple(leg for leg in WAREHOUSES if leg in named)


def case_legs(case: dict[str, Any]) -> tuple[str, ...]:
    r"""Every leg a loaded case runs on, in :data:`LEGS` order."""
    if 'warehouses' in case:
        return case_warehouses(case)
    return case_databases(case) + case_warehouses(case)


def warehouse_problem(case: dict[str, Any]) -> str | None:
    r"""What the case schema cannot say of a case's warehouse fields, or
    None (design choice 1): ``input.view`` and ``input.connect`` only on a
    case naming ``warehouses``; a view's kind kept by every leg named
    (:data:`VIEW_LEGS`); a connect path's case naming exactly its leg
    (:data:`CONNECT_LEGS`). sdm-testkit's loader holds the same rules."""
    given = case['input']
    legs = case.get('warehouses')
    if ('view' in given or 'connect' in given) and legs is None:
        return 'input/view and input/connect need warehouses'
    view = given.get('view')
    if view is not None:
        kept = VIEW_LEGS[view['kind']]
        if any(leg not in kept for leg in legs):
            return (
                f'input/view/kind {view["kind"]} is not kept on every leg named'
            )
    connect = given.get('connect')
    if connect is not None and legs != [CONNECT_LEGS[connect]]:
        return f'input/connect {connect} runs on {CONNECT_LEGS[connect]} alone'
    return None


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def case_files(paths: Iterable[Path]) -> list[Path]:
    r"""The case files named, a directory standing for its ``*.json``."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.rglob('*.json')))
        else:
            found.append(path)
    return found


def load_cases(paths: Iterable[Path]) -> list[tuple[dict[str, Any], str]]:
    r"""Each case with the SHA-256 of its file, validated, ids unique."""
    validate = validator('case')
    cases: list[tuple[dict[str, Any], str]] = []
    seen: set[str] = set()
    for path in case_files(paths):
        data = path.read_bytes()
        try:
            # NaN and Infinity are not JSON: refused, as the Rust loader
            # refuses them (P2-T2.1i-1).
            case = json.loads(
                data, parse_constant=_not_json, parse_int=_json_int
            )
        except (ValueError, RecursionError):
            # Not JSON, or nested past the parser's depth: refused by name,
            # as the Rust loader refuses it (T1.21c).
            raise RunnerFailure(
                f'{path.name}: not a case: not JSON it reads'
            ) from None
        problem = (
            _invalid(validate, case)
            or _indices(case['input'])
            or warehouse_problem(case)
        )
        if problem is not None:
            raise RunnerFailure(f'{path.name}: not a case: {problem}')
        if case['id'] in seen:
            raise RunnerFailure(f'{path.name}: case id {case["id"]} repeats')
        seen.add(case['id'])
        cases.append((case, digest(data)))
    return cases


class NegativeZero(int):
    r"""A JSON integer written ``-0``: zero, which the case format takes
    where it takes an integer of any form (an index, ``top_k``), and
    refuses where it takes a whole number as written (``fail_call``), as
    sdm-testkit's loader does (P2-T2.1i-1)."""


def _json_int(text: str) -> int:
    return NegativeZero(0) if text == '-0' else int(text)


def _not_json(constant: str) -> Any:
    r"""``json``'s reading of ``NaN``, ``Infinity`` or ``-Infinity``, which
    are not JSON: refused."""
    raise ValueError(f'{constant} is not JSON')


def load_bases(path: Path = BASES) -> dict[str, list[str]]:
    r"""The named DuckDB bases of ``contract/bases.yaml``."""
    import yaml

    document = yaml.safe_load(path.read_text(encoding='utf-8'))
    if (
        not isinstance(document, dict)
        or set(document) != {'format', 'bases'}
        or document['format'] != 'sdm-bases/1'
    ):
        raise RunnerFailure(f'{path.name}: not sdm-bases/1')
    problem = _invalid(
        validator('urn:sdm:case:1#/$defs/bases'), document['bases']
    )
    if problem is not None:
        raise RunnerFailure(f'{path.name}: not sdm-bases/1: {problem}')
    bases: dict[str, list[str]] = document['bases']
    return bases


# --- the Phase 1b inventory ------------------------------------------------------

#: What every case taken from the inventory has, which the inventory does
#: not say: diagnostic until its task admits it, S1, no paths or routes yet
#: (coverage and the routes arrive with the cases' tasks), run to predict.
_FROM_INVENTORY = {
    'format': 'sdm-case/1',
    'status': 'diagnostic',
    'slice': 'S1',
    'paths': [],
    'routes': [],
    'until': 'predict',
}
_INVENTORY_INPUT = ('base', 'setup', 'from_source')
_INVENTORY_KEPT = ('id', 'seam', 'cells', 'expect', 'register', 'why', 'source')


def load_inventory(path: Path = INVENTORY) -> dict[str, Any]:
    r"""The Phase 1b inventory, validated, with unique case ids."""
    import yaml

    inventory = yaml.safe_load(path.read_text(encoding='utf-8'))
    problem = _invalid(validator('inventory'), inventory)
    if problem is not None:
        raise RunnerFailure(f'{path.name}: not the inventory: {problem}')
    # The case-level rules beyond the schema hold for the defaults a case
    # inherits as for what it states; the indices rule is the only one.
    problem = _indices(inventory['defaults'])
    if problem is not None:
        raise RunnerFailure(f'{path.name}: defaults: {problem}')
    for entry in inventory['cases']:
        problem = _indices(entry) if 'materialize' in entry else None
        if problem is not None:
            raise RunnerFailure(f'{path.name}: {entry["id"]}: {problem}')
    ids = [entry['id'] for entry in inventory['cases']]
    if len(ids) != len(set(ids)):
        raise RunnerFailure(f'{path.name}: a case id repeats')
    unknown = {entry['base'] for entry in inventory['cases']} - set(
        inventory['bases']
    )
    if unknown:
        raise RunnerFailure(f'{path.name}: a case names an unknown base')
    loaded: dict[str, Any] = inventory
    return loaded


def inventory_case(
    entry: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:
    r"""An inventory case in the case format; :func:`inventory_entry` is
    its inverse. The inventory's ``pql`` and ``materialize`` default from
    its ``defaults``."""
    entry = copy.deepcopy(entry)
    case: dict[str, Any] = copy.deepcopy(_FROM_INVENTORY)
    case['id'] = entry['id']
    case['input'] = {
        key: entry[key] for key in _INVENTORY_INPUT if key in entry
    }
    case['input']['pql'] = entry.get('pql', defaults['pql'])
    case['input']['materialize'] = entry.get(
        'materialize', copy.deepcopy(defaults['materialize'])
    )
    for key in _INVENTORY_KEPT:
        if key in entry:
            case[key] = entry[key]
    return case


def inventory_entry(
    case: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:
    r"""A case taken from the inventory, as the inventory wrote it."""
    case = copy.deepcopy(case)
    for key, value in _FROM_INVENTORY.items():
        if case.pop(key) != value:
            raise RunnerFailure(f'{case["id"]}: not a case of the inventory')
    entry = {key: case.pop(key) for key in _INVENTORY_KEPT if key in case}
    given = case.pop('input')
    for key in _INVENTORY_INPUT:
        if key in given:
            entry[key] = given.pop(key)
    for key in ('pql', 'materialize'):
        value = given.pop(key)
        if value != defaults[key]:
            entry[key] = value
    if given or case:
        raise RunnerFailure(
            f'{entry["id"]}: keys the inventory has no place for'
        )
    return entry


def inventory_cases(ids: Iterable[str]) -> list[tuple[dict[str, Any], str]]:
    r"""The inventory cases named, in the case format, each with the
    SHA-256 of its canonical JSON."""
    wanted = list(ids)
    if not wanted:
        return []
    inventory = load_inventory()
    entries = {entry['id']: entry for entry in inventory['cases']}
    validate = validator('case')
    found = []
    for case_id in wanted:
        if case_id not in entries:
            raise RunnerFailure(f'{case_id}: not an inventory case')
        case = inventory_case(entries[case_id], inventory['defaults'])
        problem = _invalid(validate, case)
        if problem is not None:
            raise RunnerFailure(f'{case_id}: not a case: {problem}')
        text = json.dumps(case, sort_keys=True).encode()
        found.append((case, digest(text)))
    return found


# --- identity --------------------------------------------------------------------


def _git(directory: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        done = subprocess.run(
            ['git', '-C', str(directory), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0:
            raise RunnerFailure(f'git {arguments[0]} failed in {directory}')
        return done.stdout

    return {
        'sha': run('rev-parse', 'HEAD').strip(),
        'clean': run('status', '--porcelain').strip() == '',
    }


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _versions() -> dict[str, str]:
    found = {
        'python': platform.python_version(),
        'pandas': _version('pandas'),
        'pyarrow': _version('pyarrow'),
        'numpy': _version('numpy'),
        'duckdb': _version('duckdb'),
    }
    return {key: value for key, value in found.items() if value is not None}


def _cell(versions: dict[str, str]) -> str | None:
    have = (
        '.'.join(versions['python'].split('.')[:2]),
        versions.get('pandas'),
        versions.get('pyarrow'),
        versions.get('numpy'),
    )
    return next(
        (name for name, want in CELLS.items() if want == have),
        EARLIER_PINS.get(have),
    )


def _environment() -> str:
    r"""The SHA-256 of the environment's installed distributions, each
    counted once: a package installed editable is found twice, by the
    metadata installed in the environment and by the ``*.egg-info`` in its
    source tree, and the digest must not depend on whether that is there."""
    names = sorted(
        {
            f'{(dist.metadata["Name"] or "").lower()}=={dist.version}'
            for dist in importlib.metadata.distributions()
        }
    )
    return digest('\n'.join(names).encode())


# --- the reference guard -----------------------------------------------------------


class _Guard:
    r"""The reference checkout's guard, loaded by path."""

    def __init__(self, reference: Path) -> None:
        path = reference / 'scripts/reference_guard.py'
        spec = importlib.util.spec_from_file_location('reference_guard', path)
        if spec is None or spec.loader is None or not path.is_file():
            raise RunnerFailure('the reference checkout has no guard')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module = module

    def check(self, when: str) -> None:
        try:
            self._module.require_isolation()
        except Exception:  # noqa: BLE001 - any failure of the guard stops
            raise RunnerFailure(
                f'the reference guard failed {when}; the reference '
                'environment can load the Rust core'
            ) from None


# --- the core's boundary, for python-core --------------------------------------------


class _Capture:
    r"""What crossed ``sdm_client._native`` during one case: each operation
    started with its inputs, each fetch the core asked for (its request and
    plan, the step trace), and the core's answers: ``validate_query``'s
    acceptance, ``describe_source``'s description, ``materialize``'s
    request state (its version, the query and the payload, each number as
    the text the core wrote), ``predict``'s call (its body, each number as
    its text) and answer (the endpoint's model, call id and metadata, and
    the result's rows), or an error.

    Attachment ids are a run's own, so what this remembers of one run's ids
    is forgotten when the next run starts. The request id the host starts
    ``predict`` with is recorded where it reaches the endpoint, in the
    call's ``metadata.request_id``, and a call that does not carry it is a
    runner failure; a forecast's later calls each carry the id the host gave
    its step (T1.36 fix 24), and a request asked in several calls
    (``RequestState.batches``) each the id the host gave that call. Where
    predict makes more than one call, every call's body is kept in order,
    ``core.predicted.calls``, beside ``body``, the last's (T2.1g); a request
    of one call records ``body`` alone, as before."""

    def __init__(self) -> None:
        self.operations: list[str] = []
        self.inputs: list[dict[str, Any]] = []
        #: What the host passed ``edit_schema`` and ``materialize``'s held
        #: schema, in order (T1.24b).
        self.edits: list[dict[str, Any]] = []
        self.fetches: list[dict[str, Any]] = []
        #: Each fetch with the answer the host replied with, in the order
        #: replied, for ``--record-fetches`` (the replay's recording, T2.3).
        self.answered: list[dict[str, Any]] = []
        self.recording = False
        self._asked: dict[int, dict[str, Any]] = {}
        self.accepted: bool | None = None
        self.described: dict[str, Any] | None = None
        self.materialized: dict[str, Any] | None = None
        self.predicted: dict[str, Any] | None = None
        self.error: dict[str, str] | None = None
        #: The core's warnings, each its code and fields as the
        #: ``warnings`` control message carries them, in the order given
        #: (T1.23).
        self.warnings: list[dict[str, Any]] = []
        self._description: int | None = None
        self._state: dict[str, Any] | None = None
        self._request_id: str | None = None
        self._request_ids: list[str] = []
        self._skippable: set[str] = set()
        self._call: int | None = None
        self._body: dict[str, Any] | None = None
        self._bodies: list[dict[str, Any]] = []
        self._answer: dict[str, Any] | None = None

    def warned(self, message: bytes) -> None:
        for warning in json.loads(message)['warnings']:
            self.warnings.append(
                {'code': warning['code'], 'fields': warning.get('fields', {})}
            )

    def started(self, message: bytes, attachments: Any) -> None:
        operation = json.loads(message)['operation']
        self.operations.append(operation['op'])
        self._description = None
        self._state = None
        self._call = None
        self._answer = None
        self._bodies = []
        if operation['op'] == 'predict':
            state = operation['state']
            self._request_id = operation['request_id']
            # The id each call carries: one call per step with rows to
            # score, in step order, each with its own step's id (T1.36 fixes
            # 24 and 27); a step with none makes no call, as in the core.
            steps = (
                [(operation['request_id'], state['payload'])]
                + [
                    (step['request_id'], step['payload'])
                    for step in state.get('forecast') or ()
                ]
                + [
                    (batch['request_id'], batch['payload'])
                    for batch in state.get('batches') or ()
                ]
                # A ranking's later questions (T2.8), each the id the host
                # gave it, in order.
                + [
                    (part['request_id'], part['payload'])
                    for part in (state.get('ranking') or {}).get('parts', ())
                ]
            )
            self._request_ids = [
                request_id
                for request_id, payload in steps
                if not _nothing_to_score(json.loads(attachments[payload]))
            ]
            # A ranking's question whose context holds one answer is asked
            # no call (T2.8): its id may be passed over, in order.
            self._skippable = (
                {request_id for request_id, _ in steps}
                if state.get('ranking') is not None
                else set()
            )
            self.inputs.append(
                {
                    'op': 'predict',
                    'model': operation['model'],
                    'state': {
                        'version': state['version'],
                        'query': json.loads(attachments[state['query']]),
                        'payload': json.loads(
                            attachments[state['payload']],
                            parse_int=_number,
                            parse_float=_number,
                        ),
                    },
                }
            )
        if operation['op'] == 'validate_query':
            self.inputs.append(
                {
                    'op': 'validate_query',
                    'query': json.loads(attachments[operation['query']]),
                    'profile': operation['profile'],
                    'schema': json.loads(attachments[operation['schema']]),
                }
            )
        elif operation['op'] == 'describe_source':
            self.inputs.append(
                {
                    'op': 'describe_source',
                    'provider': operation['provider'],
                    'source': json.loads(attachments[operation['source']]),
                    'check_keys': operation['check_keys'],
                }
            )
        elif operation['op'] == 'edit_schema':
            # The schema and the edit a setter passed (T1.24b): the
            # description the core last answered, the schema as the host
            # holds it, and the edit; kept apart from ``inputs``, whose
            # shape a gate's checkpoint fixes.
            self.edits.append(
                {
                    'op': 'edit_schema',
                    'schema': json.loads(attachments[operation['schema']]),
                    'held': json.loads(attachments[operation['held']]),
                    'edit': json.loads(attachments[operation['edit']]),
                }
            )
        elif operation['op'] == 'materialize':
            indices = operation['indices']
            if operation.get('held') is not None:
                # The schema as the host holds it beside the description
                # (T1.24b), apart from ``inputs`` as an edit is.
                self.edits.append(
                    {
                        'op': 'materialize',
                        'held': json.loads(attachments[operation['held']]),
                    }
                )
            self.inputs.append(
                {
                    'op': 'materialize',
                    'provider': operation['provider'],
                    'schema': json.loads(attachments[operation['schema']]),
                    'query': json.loads(attachments[operation['query']]),
                    'indices': (
                        None
                        if indices is None
                        else _scalars(attachments[indices])
                    ),
                    'anchor_time': operation['anchor_time'],
                    'seed': operation['seed'],
                }
            )

    def stepped(self, step: dict[str, Any]) -> None:
        if step.get('step') == 'fetch':
            self.fetches.extend(
                {'request': asked['request'], 'plan': asked['plan']}
                for asked in step['requests']
            )
            for asked in step['requests']:
                self._asked[asked['id']] = {
                    'request': asked['request'],
                    'plan': asked['plan'],
                }
        if step.get('step') == 'call_model':
            self._call = step['body']

    def replied(self, message: bytes, attachments: Any) -> None:
        r"""A fetch's answer as the host replies with it, kept beside the
        fetch: its rows as an Arrow IPC stream in base64, or the provider's
        error, as the replay's recording holds it (``sdm-testkit``'s
        ``replay``)."""
        if not self.recording:
            return
        reply = json.loads(message)
        if reply.get('message') != 'fetch_reply':
            return
        asked = self._asked.pop(reply['id'], None)
        if asked is None:
            raise RunnerFailure('a fetch replied to that was never asked')
        outcome = reply['outcome']
        if outcome['outcome'] == 'error':
            self.answered.append({**asked, 'error': outcome['error']})
            return
        import base64

        import pyarrow as pa

        table = pa.RecordBatchReader._import_from_c_capsule(
            attachments[outcome['data']].__arrow_c_stream__()
        ).read_all()
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        self.answered.append(
            {
                **asked,
                'data': base64.b64encode(sink.getvalue().to_pybytes()).decode(),
            }
        )

    def failed(self, error: Any) -> None:
        self.error = {'kind': str(error.kind), 'code': str(error.code)}

    def done(self, result: dict[str, Any]) -> None:
        if 'accepted' in result:
            self.accepted = bool(result['accepted'])
        if result.get('op') == 'describe_source':
            self._description = result['description']
        if result.get('op') == 'materialize':
            self._state = result['state']
            self.materialized = {'version': result['state']['version']}
        if result.get('op') == 'predict':
            self._passed_over()
            if self._request_ids:
                raise RunnerFailure(
                    'predict answered without calling the model for a step '
                    'with rows to score'
                )
            self._answer = result
            self.predicted = {
                'body': self._body,
                **({'calls': self._bodies} if len(self._bodies) > 1 else {}),
                'model': result['model'],
                'request_id': result['request_id'],
                'metadata': None,
                'result': None,
            }

    def taken(self, attachment: int, value: Any) -> None:
        if attachment == self._call and isinstance(value, bytes):
            self._body = json.loads(
                value, parse_int=_number, parse_float=_number
            )
            self._bodies.append(self._body)
            # The next step with rows to score's id, and no call past them.
            carried = self._body.get('metadata', {}).get('request_id')
            self._passed_over(carried)
            if not self._request_ids or carried != self._request_ids.pop(0):
                raise RunnerFailure(
                    'the model call does not carry the request id predict '
                    'was started with'
                )
        answer = self._answer
        if answer is not None and self.predicted is not None:
            if attachment == answer['metadata'] and isinstance(value, bytes):
                self.predicted['metadata'] = json.loads(value)
            if attachment == answer['result'] and not isinstance(value, bytes):
                self.predicted['result'] = _result_rows(value)
        if attachment == self._description and isinstance(value, bytes):
            self.described = json.loads(value)
        state = self._state
        if state is None or self.materialized is None:
            return
        if attachment == state['payload'] and isinstance(value, bytes):
            self.materialized['payload'] = json.loads(
                value, parse_int=_number, parse_float=_number
            )
        if attachment == state['query'] and isinstance(value, bytes):
            self.materialized['query'] = json.loads(value)

    def _passed_over(self, carried: str | None = None) -> None:
        r"""Drops the ids of a ranking's questions asked no call before the
        call carrying ``carried``, or, with none, all that are left."""
        while (
            self._request_ids
            and self._request_ids[0] != carried
            and self._request_ids[0] in self._skippable
        ):
            self._request_ids.pop(0)

    def core(self) -> dict[str, Any]:
        core: dict[str, Any] = {
            'inputs': self.inputs,
            'fetches': self.fetches,
            'accepted': self.accepted,
            'described': self.described,
            'materialized': self.materialized,
            'predicted': self.predicted,
            'error': self.error,
        }
        if self.edits:
            core['edits'] = self.edits
        return core


def _result_rows(stream: Any) -> dict[str, Any]:
    r"""An Arrow result the core gave, as the pure-Rust host records one:
    the result's explanation where the core gave one (its schema's
    ``explanation``, T2.16a), its columns' names and types, and its rows in
    order, each cell an
    integer's digits (``i64``), a float's IEEE-754 bits (``f64_bits``, 16
    hex digits, so no text form of a float is compared) or text
    (``utf8``); from T1.28b a boolean (``bool``) and a class's
    probabilities (``map``: each class and its probability's bits, in
    order); from T2.8 a list (``list``: a ranking's scores and rankings, an
    item's own, each member a cell as above, a null ``None``), as the
    pure-Rust host records one."""
    import pyarrow as pa

    table = pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    ).read_all()
    kinds = {
        pa.int64(): 'int64',
        pa.float64(): 'float64',
        pa.utf8(): 'utf8',
        pa.bool_(): 'bool',
    }

    def kind_of(field: Any) -> str:
        if pa.types.is_map(field.type):
            return 'map'
        if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
            return 'list'
        return kinds.get(field.type, 'other')

    columns = [
        {'name': field.name, 'type': kind_of(field)} for field in table.schema
    ]
    # Each list column's members' kind, as a cell of their type records.
    members = [
        kinds.get(field.type.value_type, 'other')
        if kind_of(field) == 'list'
        else None
        for field in table.schema
    ]

    def cell(kind: str, value: Any, member: str | None = None) -> Any:
        if value is None:
            return None
        if kind == 'int64':
            return {'i64': str(value)}
        if kind == 'float64':
            return {'f64_bits': struct.pack('>d', value).hex()}
        if kind == 'utf8':
            return {'utf8': value}
        if kind == 'bool':
            return {'bool': value}
        if kind == 'map':
            return {
                'map': [[key, struct.pack('>d', p).hex()] for key, p in value]
            }
        if kind == 'list' and member is not None:
            return {'list': [cell(member, item) for item in value]}
        raise RunnerFailure(
            'a result column of a type this runner does not record'
        )

    rows = [
        [
            cell(column['type'], value, member)
            for column, value, member in zip(columns, row, members, strict=True)
        ]
        for row in zip(
            *(table.column(i).to_pylist() for i in range(table.num_columns)),
            strict=True,
        )
    ]
    recorded: dict[str, Any] = {'columns': columns, 'rows': rows}
    # The result's explanation (T2.13), on the schema's metadata as the JSON
    # text the core wrote, recorded as text so the hosts are held to the
    # same bytes (T2.16a); an item's explanation and metadata are the
    # result's own columns, among the rows.
    explanation = (table.schema.metadata or {}).get(b'explanation')
    if explanation is not None:
        recorded['explanation'] = explanation.decode('utf-8')
    return recorded


#: A pyarrow integer type's contract scalar tag.
_INTEGER_TAGS = {
    'int8': 'i8',
    'int16': 'i16',
    'int32': 'i32',
    'int64': 'i64',
    'uint8': 'u8',
    'uint16': 'u16',
    'uint32': 'u32',
    'uint64': 'u64',
}


def _nothing_to_score(payload: Any) -> bool:
    r"""Whether a payload has no row to score, so ``predict`` makes no call
    for it: the core's rule (``sdm_request::predict::nothing_to_score``),
    its ``predict`` half's ``instance_table`` with a list of text columns
    and an empty list of rows; any other shape has rows to score."""
    half = payload.get('predict') if isinstance(payload, dict) else None
    table = half.get('instance_table') if isinstance(half, dict) else None
    if not isinstance(table, dict):
        return False
    columns, rows = table.get('columns'), table.get('rows')
    return (
        isinstance(columns, list)
        and all(isinstance(column, str) for column in columns)
        and rows == []
    )


def _scalars(table: Any) -> list[dict[str, Any]]:
    r"""The one column of an Arrow table the host sent, each value as the
    contract's tagged scalar, of the column's own type (from T1.32a the
    host sends the entities as pyarrow infers them, and the core casts
    them): an integer by its width, text as ``utf8``, a boolean, a float
    as the shortest text that reads back as it, a decimal as its unscaled
    digits, a time as its count of the column's unit, bytes as base64, and
    a null as ``null``."""
    import base64
    import datetime

    import pyarrow as pa
    from sdm_client._rust import codec

    column = table.column(0)
    kind = column.type
    values = column.to_pylist()
    if str(kind) in _INTEGER_TAGS:
        return [
            {'t': 'null'}
            if v is None
            else codec.integer(_INTEGER_TAGS[str(kind)], v)
            for v in values
        ]

    def one(value: Any) -> dict[str, Any]:
        if value is None:
            return {'t': 'null'}
        if pa.types.is_string(kind) or pa.types.is_large_string(kind):
            return {'t': 'utf8', 'v': value}
        if pa.types.is_boolean(kind):
            return {'t': 'bool', 'v': value}
        if pa.types.is_float32(kind) or pa.types.is_float64(kind):
            return {
                't': 'f32' if pa.types.is_float32(kind) else 'f64',
                'v': _float_text(value),
            }
        if pa.types.is_decimal(kind):
            return {
                't': 'd128' if pa.types.is_decimal128(kind) else 'd256',
                'p': kind.precision,
                's': kind.scale,
                'v': str(_unscaled(value, kind.scale)),
            }
        if pa.types.is_date32(kind):
            return {
                't': 'date32',
                'v': str((value - datetime.date(1970, 1, 1)).days),
            }
        if pa.types.is_binary(kind) or pa.types.is_large_binary(kind):
            return {'t': 'binary', 'v': base64.b64encode(value).decode()}
        raise RunnerFailure(
            f'the entities sent are of a type not recorded: {kind}'
        )

    if pa.types.is_null(kind):
        return [{'t': 'null'} for _ in values]
    if pa.types.is_timestamp(kind) and kind.unit in ('us', 'ns'):
        counts = column.cast(pa.int64()).to_pylist()
        return [
            {'t': 'null'}
            if v is None
            else codec.timestamp(v, kind.unit, kind.tz)
            for v in counts
        ]
    return [one(v) for v in values]


def _unscaled(value: Any, scale: int) -> int:
    r"""A decimal's unscaled integer at ``scale``, from its own digits and
    exponent: no decimal context, whose precision would round a value of
    more digits than it holds (T1.32a fix round 1)."""
    sign, digits, exponent = value.as_tuple()
    number = int(''.join(map(str, digits)) or '0')
    shift = exponent + scale
    if shift >= 0:
        number *= 10**shift
    else:
        number, rest = divmod(number, 10 ** (-shift))
        if rest:
            raise RunnerFailure(
                'a decimal the host sent has more places than its type'
            )
    return -number if sign else number


def _float_text(value: float) -> str:
    r"""A float as the contract writes it: the shortest text that reads back
    as it, an exponent with no plus sign or leading zeros, ``NaN``, ``inf``
    or ``-inf``."""
    import math

    if math.isnan(value):
        return 'NaN'
    if math.isinf(value):
        return 'inf' if value > 0 else '-inf'
    text = repr(float(value))
    if 'e' in text:
        mantissa, exponent = text.split('e')
        sign = '-' if exponent.startswith('-') else ''
        text = f'{mantissa}e{sign}{exponent.lstrip("+-").lstrip("0") or "0"}'
    return text


class _Handle:
    def __init__(self, handle: Any, native: Any, capture: _Capture) -> None:
        self._handle = handle
        self._native = native
        self._capture = capture

    def next(self) -> Any:
        from sdm_client._rust import codec

        try:
            control, sent = self._handle.next()
        except self._native.CoreError as error:
            self._capture.failed(error)
            raise
        message = codec.read(control, sent)
        if message.get('message') == 'step':
            self._capture.stepped(message)
            if message.get('step') == 'done':
                self._capture.done(message['result'])
        return control, sent

    def take_attachment(self, attachment: int) -> Any:
        value = self._handle.take_attachment(attachment)
        self._capture.taken(attachment, value)
        return value

    def take_warnings(self) -> Any:
        taken = self._handle.take_warnings()
        if taken is not None:
            self._capture.warned(taken[0])
        return taken

    def reply(self, message: bytes, attachments: Any) -> None:
        self._capture.replied(message, attachments)
        try:
            self._handle.reply(message, attachments)
        except self._native.CoreError as error:
            self._capture.failed(error)
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


class _Run:
    def __init__(self, native: Any, capture: _Capture) -> None:
        self._native = native
        self._capture = capture

    def start(self, message: bytes, attachments: Any) -> _Handle:
        self._capture.started(message, attachments)
        try:
            handle = self._native.Run.start(message, attachments)
        except self._native.CoreError as error:
            self._capture.failed(error)
            raise
        return _Handle(handle, self._native, self._capture)


class _Native:
    r"""``sdm_client._native`` as the host sees it, its runs recorded."""

    def __init__(self, native: Any, capture: _Capture) -> None:
        self._native = native
        self.Run = _Run(native, capture)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native, name)


class _Boundary:
    r"""Records each case's core runs, by wrapping the one function through
    which the Python host imports ``sdm_client._native``."""

    def __init__(self) -> None:
        from sdm_client._rust import run

        self.capture = _Capture()
        imported = run.native

        def native() -> Any:
            return _Native(imported(), self.capture)

        run.native = native  # type: ignore[assignment]

    def reset(self) -> _Capture:
        self.capture = _Capture()
        return self.capture


def _sibling(name: str) -> Any:
    r"""A module beside this file, loaded by path: ``python -I`` puts no
    script directory on ``sys.path``."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    if spec is None or spec.loader is None:
        raise RunnerFailure(f'{name}.py is missing beside the runner')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --- the driver capture (Q2) ---------------------------------------------------------

#: The methods that read a statement's result, by what they give back.
_ROWS = ('fetchall', 'fetchmany')
_ROW = ('fetchone',)
_ARROW = ('to_arrow_table', 'fetch_arrow_table')
#: Results the capture cannot record without consuming them or changing
#: what the caller gets; reading one makes the capture incomplete, which
#: fails the case rather than drop a result.
_UNRECORDED = (
    # A connection's arrow() gives a RecordBatchReader, which the capture
    # cannot read without consuming what the caller reads.
    'arrow',
    'fetch_record_batch',
    'to_arrow_reader',
    'fetchdf',
    'fetch_df',
    'df',
    'fetch_df_chunk',
    'fetchnumpy',
    'pl',
    'torch',
    'tf',
)
#: Methods that start a statement the capture does not see as one (a lazy
#: relation runs when it is read, outside the capture).
_UNCAPTURED = (
    'executemany',
    'sql',
    'query',
    'from_query',
    'table',
    'view',
    'values',
    'read_csv',
    'read_parquet',
    'read_json',
    'from_df',
    'from_arrow',
    'append',
    'load_extension',
    'install_extension',
)


def _typed(value: Any) -> dict[str, Any]:
    r"""A parameter or a fetched value as its type and its text: the
    capture's own record, which no value passes through to change."""
    kind = type(value)
    return {
        'type': f'{kind.__module__}.{kind.__qualname__}',
        'value': repr(value),
    }


def _parameters(parameters: Any) -> list[dict[str, Any]] | None:
    r"""A statement's parameters as recorded: in order, each typed, and a
    named one with its name."""
    if parameters is None:
        return None
    if isinstance(parameters, Mapping):
        return [
            {'name': str(name), **_typed(value)}
            for name, value in parameters.items()
        ]
    return [_typed(value) for value in parameters]


def _rows_digest(rows: list[Any]) -> str:
    text = json.dumps(
        [[_typed(value) for value in row] for row in rows], sort_keys=True
    )
    return digest(text.encode())


def _arrow_digest(table: Any) -> str:
    r"""The SHA-256 of an Arrow table's contents, its batch boundaries left
    out (decision H9): the IPC stream of one combined copy."""
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    combined = table.combine_chunks()
    with pa.ipc.new_stream(sink, combined.schema) as writer:
        writer.write_table(combined)
    return digest(sink.getvalue().to_pybytes())


_EPOCH_DAYS = 719163  # date(1970, 1, 1).toordinal()


def _nanos(days: int, seconds: int, micros: int, nanos: int = 0) -> str:
    return str(((days * 86400 + seconds) * 1_000_000 + micros) * 1000 + nanos)


def _month_day_nano(value: Any) -> bool:
    r"""Whether ``value`` is pyarrow's month-day-nano interval."""
    if not isinstance(value, tuple):
        return False
    import pyarrow as pa

    return isinstance(value, pa.MonthDayNano)


def canonical_cell(value: Any, odd: list[str]) -> Any:
    r"""One value of a result in its canonical form (decision H9; design,
    "Observations and comparator"): the value itself, never its Python
    spelling, so a row read with ``fetchall`` and the same row read as
    Arrow give the same cell, and the pure-Rust host writes the same cell
    from Arrow (``sdm-testkit``'s ``values``; both held to
    ``contract/result-vectors.json``). A null is ``null`` and a boolean
    itself; anything else is a list naming its kind: ``["i", digits]``,
    ``["f", the IEEE-754 bits of the value as a double, 16 hex digits]``,
    ``["s", text]``, ``["b", hex]``, ``["d", unscaled digits, scale]``,
    ``["date", days since 1970-01-01]``, ``["ts", nanoseconds since
    1970-01-01 on the wall clock]``, ``["tstz", nanoseconds since the UTC
    epoch]``, ``["time", nanoseconds since midnight]``, ``["dur",
    nanoseconds]``, ``["list", cells]`` and ``["struct", [[name, cell],
    ...]]``; read from Arrow (:func:`_arrow_cells`), a map is ``["map",
    [[key, value], ...]]`` in Arrow's entry order, a union ``["union",
    type code, member name, cell]`` and a month-day-nano interval
    ``["interval", months, days, nanoseconds as digits]``, as is pyarrow's
    ``MonthDayNano`` read as a row's value. A value of any
    other kind is ``["?",
    its type's name]`` and is listed as unencodable, which the comparator
    never takes as equal. This
    reads Python values, which hold a time only to the microsecond: an
    Arrow result is read from Arrow instead (:func:`arrow_values`), and a
    row read of a column a driver reads with less than its precision is
    unencodable (:func:`result_values`)."""
    import datetime as dt
    from decimal import Decimal

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return ['i', str(value)]
    if isinstance(value, float):
        return ['f', struct.pack('>d', value).hex()]
    if isinstance(value, str):
        return ['s', value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ['b', bytes(value).hex()]
    if isinstance(value, Decimal):
        sign, digits, exponent = value.as_tuple()
        if isinstance(exponent, int):
            unscaled = int(''.join(map(str, digits)) or '0')
            unscaled = -unscaled if sign else unscaled
            if exponent > 0:
                return ['d', str(unscaled * 10**exponent), 0]
            return ['d', str(unscaled), -exponent]
    elif isinstance(value, dt.datetime):
        # A pandas Timestamp (pyarrow's reading of a nanosecond timestamp)
        # carries its nanoseconds as ``value``, on the wall clock when it
        # has no zone and in UTC when it has one.
        exact = getattr(value, 'value', None)
        if value.tzinfo is None:
            if isinstance(exact, int):
                return ['ts', str(exact)]
            delta = value - dt.datetime(1970, 1, 1)
            return [
                'ts',
                _nanos(delta.days, delta.seconds, delta.microseconds),
            ]
        if isinstance(exact, int):
            return ['tstz', str(exact)]
        delta = value - dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
        return ['tstz', _nanos(delta.days, delta.seconds, delta.microseconds)]
    elif isinstance(value, dt.date):
        return ['date', value.toordinal() - _EPOCH_DAYS]
    elif isinstance(value, dt.time) and value.tzinfo is None:
        seconds = (value.hour * 60 + value.minute) * 60 + value.second
        return ['time', _nanos(0, seconds, value.microsecond)]
    elif isinstance(value, dt.timedelta):
        exact = getattr(value, 'value', None)
        if isinstance(exact, int):
            return ['dur', str(exact)]
        return [
            'dur',
            _nanos(value.days, value.seconds, value.microseconds),
        ]
    elif _month_day_nano(value):
        # pyarrow's month-day-nano interval, a named tuple, which a driver
        # hands over for an interval read whole (T2.1d): the cell Arrow
        # gives the same interval, never a list.
        return ['interval', value.months, value.days, str(value.nanoseconds)]
    elif isinstance(value, (list, tuple)):
        return ['list', [canonical_cell(item, odd) for item in value]]
    elif isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return [
            'struct',
            [[k, canonical_cell(v, odd)] for k, v in value.items()],
        ]
    name = f'{type(value).__module__}.{type(value).__qualname__}'
    odd.append(name)
    return ['?', name]


def _canonical_text(value: Any) -> str:
    r"""The one text of a canonical row: compact JSON, ASCII only, so its
    order as text is its order as bytes in any language."""
    return json.dumps(
        value, ensure_ascii=True, separators=(',', ':'), allow_nan=False
    )


#: DuckDB types a Python driver reads into Python values with less than
#: their precision: ``TIME_NS`` and ``TIMESTAMP_NS`` to microseconds, an
#: ``INTERVAL`` to a ``timedelta`` that folds its months into days, a
#: ``UNION`` to its member's value without its tag, and a ``MAP`` to a
#: dictionary the canonical form would take for a struct (Arrow's reading
#: of each keeps what these lose). A value of a column of
#: such a type, or of one holding such a type at any depth, read as rows
#: (``fetchall``) is unencodable, never compared as if exact.
_LOSSY_AS_ROWS = frozenset(
    ('TIME_NS', 'TIMESTAMP_NS', 'INTERVAL', 'UNION', 'MAP')
)


class _TypeText:
    r"""A DuckDB type as its description spells it, read into nodes: a
    name and the types it holds (a list's or array's element, a struct's
    or union's members, a map's key and value), field names read and set
    aside, never taken for types."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.at = 0

    def _peek(self) -> str:
        return self.text[self.at] if self.at < len(self.text) else ''

    def _skip_spaces(self) -> None:
        while self._peek() == ' ':
            self.at += 1

    def _name(self) -> str:
        r"""A field name: double-quoted (``""`` an escaped quote) or bare,
        up to the space before its type."""
        self._skip_spaces()
        if self._peek() == '"':
            self.at += 1
            while True:
                end = self.text.index('"', self.at)
                if self.text[end + 1 : end + 2] == '"':
                    self.at = end + 2
                    continue
                self.at = end + 1
                return ''
        while self._peek() not in (' ', ''):
            self.at += 1
        return ''

    def _balanced(self) -> None:
        r"""Skips a parenthesised argument list that holds no types (a
        decimal's precision, an enum's quoted members)."""
        depth = 0
        while self.at < len(self.text):
            char = self.text[self.at]
            if char == "'":
                end = self.at + 1
                while True:
                    end = self.text.index("'", end)
                    if self.text[end + 1 : end + 2] == "'":
                        end += 2
                        continue
                    break
                self.at = end + 1
                continue
            self.at += 1
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    return
        raise ValueError('unbalanced')

    def node(self) -> tuple[str, list[Any]]:
        self._skip_spaces()
        start = self.at
        while self._peek() not in ('(', '[', ',', ')', ''):
            self.at += 1
        name = self.text[start : self.at].strip().upper()
        if not name:
            raise ValueError('no type')
        children: list[Any] = []
        if self._peek() == '(':
            if name in ('STRUCT', 'UNION'):
                self.at += 1
                while True:
                    self._name()
                    children.append(self.node())
                    self._skip_spaces()
                    if self._peek() == ',':
                        self.at += 1
                        continue
                    if self._peek() != ')':
                        raise ValueError('a member')
                    self.at += 1
                    break
            elif name == 'MAP':
                self.at += 1
                children.append(self.node())
                if self._peek() != ',':
                    raise ValueError('a map key')
                self.at += 1
                children.append(self.node())
                self._skip_spaces()
                if self._peek() != ')':
                    raise ValueError('a map value')
                self.at += 1
            else:
                self._balanced()
        while self._peek() == '[':
            end = self.text.index(']', self.at)
            self.at = end + 1
            name, children = 'LIST', [(name, children)]
        return name, children


def _lossy_as_rows(duckdb_type: str) -> bool:
    r"""Whether a Python driver reads a column of ``duckdb_type`` with less
    than its precision: the type, read into nodes, is or holds one of
    :data:`_LOSSY_AS_ROWS`; a type spelling this cannot read counts as
    lossy, so it fails closed."""

    def lossy(node: tuple[str, list[Any]]) -> bool:
        name, children = node
        return name in _LOSSY_AS_ROWS or any(lossy(c) for c in children)

    reader = _TypeText(duckdb_type)
    try:
        node = reader.node()
    except ValueError:
        return True
    reader._skip_spaces()
    return reader.at != len(duckdb_type) or lossy(node)


def _texts(rows: Iterable[list[Any]]) -> list[str]:
    return [_canonical_text(row) for row in rows]


def _values(
    columns: list[str],
    types: list[str] | None,
    texts: list[str],
    odd: list[str],
) -> dict[str, Any]:
    return {
        'columns': list(columns),
        'types': types,
        'ordered': digest(('[' + ','.join(texts) + ']').encode()),
        'unordered': digest(('[' + ','.join(sorted(texts)) + ']').encode()),
        'unencodable': sorted(set(odd)),
    }


def result_values(
    columns: list[str],
    types: list[str] | None,
    rows: Iterable[Any],
    duckdb_types: bool = True,
) -> dict[str, Any]:
    r"""A result read as Python rows (``fetchall``), its columns and rows in
    canonical form (:func:`canonical_cell`): the SHA-256 of the rows in the
    order read (``ordered``) and of the same rows sorted by their canonical
    text (``unordered``, so two results holding the same rows in another
    order agree, duplicates counted); ``types`` are DuckDB's own names of
    the columns' types where the driver gives them (``description``), and
    ``unencodable`` names every value of a kind the canonical form does not
    hold, and every value of a column whose type the driver reads with less
    than its precision (:data:`_LOSSY_AS_ROWS`; DuckDB's types alone,
    ``duckdb_types``: another driver's types are its own)."""
    odd: list[str] = []
    lossy = (
        [_lossy_as_rows(kind) for kind in types or []] if duckdb_types else []
    )
    cells = []
    for row in rows:
        cells.append(
            [
                None
                if value is None
                else (
                    _unencodable(f'duckdb:{types[i]} read as rows', odd)
                    if types is not None and i < len(lossy) and lossy[i]
                    else canonical_cell(value, odd)
                )
                for i, value in enumerate(row)
            ]
        )
    return _values(list(columns), types, _texts(cells), odd)


def _unencodable(name: str, odd: list[str]) -> list[str]:
    odd.append(name)
    return ['?', name]


#: Nanoseconds per unit of an Arrow temporal type.
_UNIT_NANOS = {'s': 10**9, 'ms': 10**6, 'us': 10**3, 'ns': 1}


def _json_cell(text: str | None, odd: list[str]) -> Any:
    r"""A JSON text as the value it holds, the canonical cell of a driver's
    reading of it; a text that is not JSON is unencodable."""
    if text is None:
        return None
    try:
        value = json.loads(text)
    except ValueError:
        odd.append('arrow:json')
        return ['?', 'arrow:json']
    return canonical_cell(value, odd)


def _arrow_cells(array: Any, odd: list[str]) -> list[Any]:
    r"""The canonical cells of one Arrow array, read from Arrow itself, as
    ``sdm-testkit``'s ``values`` reads them: temporal values from their
    integer storage at their unit (so a ``time64[ns]`` keeps its
    nanoseconds), lists, structs and maps element by element (a map's
    entries in Arrow's order, duplicate keys kept), a union as its chosen
    member's cell under its type code and name (:func:`_union_cells`), a
    month-day-nano interval as ``["interval", months, days, nanoseconds
    as digits]``, a dictionary as its values, text Arrow marks as JSON
    (``arrow.json``) as the value it holds, as a driver reads JSON
    (``json.loads``, then :func:`canonical_cell`); a type outside that set (a
    ``date64``, Arrow's year-month and day-time intervals) is unencodable,
    its non-null values ``["?", "arrow:<type>"]``."""
    import pyarrow as pa
    import pyarrow.types as is_

    kind = array.type
    valid = [not null for null in array.is_null().to_pylist()]
    if (
        isinstance(kind, pa.BaseExtensionType)
        and kind.extension_name == 'arrow.json'
        and is_.is_string(kind.storage_type)
    ):
        return [_json_cell(text, odd) for text in array.storage.to_pylist()]
    if is_.is_dictionary(kind):
        return _arrow_cells(array.dictionary_decode(), odd)
    if is_.is_time(kind) or is_.is_timestamp(kind) or is_.is_duration(kind):
        width = pa.int32() if is_.is_time32(kind) else pa.int64()
        factor = _UNIT_NANOS[kind.unit]
        if is_.is_time(kind):
            tag = 'time'
        elif is_.is_duration(kind):
            tag = 'dur'
        else:
            tag = 'tstz' if kind.tz else 'ts'
        counts = array.view(width).to_pylist()
        return [
            [tag, str(count * factor)] if ok else None
            for count, ok in zip(counts, valid, strict=True)
        ]
    if is_.is_date32(kind):
        days = array.view(pa.int32()).to_pylist()
        return [
            ['date', day] if ok else None
            for day, ok in zip(days, valid, strict=True)
        ]
    if (
        is_.is_list(kind)
        or is_.is_large_list(kind)
        or is_.is_fixed_size_list(kind)
    ):
        return [
            ['list', _arrow_cells(array[i].values, odd)] if ok else None
            for i, ok in enumerate(valid)
        ]
    if is_.is_struct(kind):
        names = [kind.field(j).name for j in range(kind.num_fields)]
        # Each child as stored, at the struct's offset: flatten() would merge
        # the struct's nulls into each child, which aborts the process for a
        # union child (a union has no validity bitmap). A child's slot under
        # a null struct row holds whatever the producer left there, so the
        # children are read with their names set aside, and only the cells
        # of the rows the struct shows count as unencodable, as with a
        # union's members (:func:`_union_cells`). A null struct row is null.
        scratch: list[str] = []
        children = [
            _arrow_cells(array.field(j), scratch)
            for j in range(kind.num_fields)
        ]
        rows: list[Any] = []
        for i, ok in enumerate(valid):
            if not ok:
                rows.append(None)
                continue
            fields = [[name, children[j][i]] for j, name in enumerate(names)]
            for _, cell in fields:
                _chosen_odd(cell, odd)
            rows.append(['struct', fields])
        return rows
    if is_.is_map(kind):
        cells: list[Any] = []
        for i, ok in enumerate(valid):
            if not ok:
                cells.append(None)
                continue
            entries = array[i].values
            keys = _arrow_cells(entries.field(0), odd)
            items = _arrow_cells(entries.field(1), odd)
            cells.append(
                ['map', [[k, v] for k, v in zip(keys, items, strict=True)]]
            )
        return cells
    if is_.is_union(kind):
        return _union_cells(array, odd)
    if is_.is_interval(kind) and str(kind) == 'month_day_nano_interval':
        return [
            ['interval', value.months, value.days, str(value.nanoseconds)]
            if value is not None
            else None
            for value in array.to_pylist()
        ]
    if is_.is_float16(kind):
        array = array.cast(pa.float32())
        kind = array.type
    string_view = getattr(is_, 'is_string_view', lambda _: False)
    binary_view = getattr(is_, 'is_binary_view', lambda _: False)
    if (
        is_.is_null(kind)
        or is_.is_boolean(kind)
        or is_.is_integer(kind)
        or is_.is_floating(kind)
        or is_.is_string(kind)
        or is_.is_large_string(kind)
        or string_view(kind)
        or is_.is_binary(kind)
        or is_.is_large_binary(kind)
        or is_.is_fixed_size_binary(kind)
        or binary_view(kind)
        or is_.is_decimal(kind)
    ):
        return [canonical_cell(value, odd) for value in array.to_pylist()]
    name = f'arrow:{kind}'
    return [_unencodable(name, odd) if ok else None for ok in valid]


def _union_cells(array: Any, odd: list[str]) -> list[Any]:
    r"""A union array's cells, ``["union", type code, member name, the
    member's cell]``, never null themselves: an Arrow union has no validity
    of its own, so a null member, or a union DuckDB writes as null (a null
    under its first member), is its tag with a null cell. The type codes
    (and a dense union's offsets) are read from the array's buffers at its
    offset, as pyarrow's ``type_codes`` and scalars do not follow a slice
    (a union inside a list is one). Only a chosen member's values count as
    unencodable: an interval member no row chooses names nothing."""
    import pyarrow as pa

    kind = array.type
    length, offset = len(array), array.offset
    buffers = array.buffers()
    codes = pa.Array.from_buffers(
        pa.int8(), length + offset, [None, buffers[1]]
    ).to_pylist()[offset:]
    at = (
        pa.Array.from_buffers(
            pa.int32(), length + offset, [None, buffers[2]]
        ).to_pylist()[offset:]
        if kind.mode == 'dense'
        else list(range(length))
    )
    position = {code: j for j, code in enumerate(kind.type_codes)}
    # Each member read whole, its names set aside, since a sparse member
    # holds a slot for every row, chosen or not; pyarrow's ``field`` gives
    # a sparse member at the array's offset and a dense one whole.
    scratch: list[str] = []
    members = [
        _arrow_cells(array.field(j), scratch) for j in range(kind.num_fields)
    ]
    cells: list[Any] = []
    for code, index in zip(codes, at, strict=True):
        j = position[code]
        member = members[j][index]
        _chosen_odd(member, odd)
        cells.append(['union', code, kind.field(j).name, member])
    return cells


def _chosen_odd(cell: Any, odd: list[str]) -> None:
    r"""The unencodable names a cell holds at any depth, added to ``odd``."""
    if not isinstance(cell, list):
        return
    tag = cell[0]
    if tag == '?':
        odd.append(cell[1])
    elif tag == 'list':
        for item in cell[1]:
            _chosen_odd(item, odd)
    elif tag == 'struct':
        for _, item in cell[1]:
            _chosen_odd(item, odd)
    elif tag == 'map':
        for key, item in cell[1]:
            _chosen_odd(key, odd)
            _chosen_odd(item, odd)
    elif tag == 'union':
        _chosen_odd(cell[3], odd)


#: The field metadata key, and its value, marking a month-day-nano interval
#: column built from a driver's ``timedelta``s (the Postgres provider's,
#: psycopg's reading of an ``interval``); ``sdm-testkit``'s ``INTERVAL_TAG``.
_INTERVAL_TAG = (b'sdm:interval', b'timedelta')


def _is_json_field(field: Any) -> bool:
    r"""Whether ``field`` holds JSON text, Arrow's ``arrow.json`` over a
    string: as the extension type a newer pyarrow builds, or as a string
    field carrying the extension's metadata, as pyarrow before the type
    (the 14 floor) keeps it."""
    import pyarrow as pa
    import pyarrow.types as is_

    kind = field.type
    if isinstance(kind, pa.BaseExtensionType):
        return kind.extension_name == 'arrow.json' and is_.is_string(
            kind.storage_type
        )
    return (
        is_.is_string(kind)
        and (field.metadata or {}).get(b'ARROW:extension:name') == b'arrow.json'
    )


def _json_texts(array: Any, odd: list[str]) -> list[Any]:
    r"""The cells of a JSON column (:func:`_is_json_field`), each the value
    its text holds (:func:`_json_cell`)."""
    import pyarrow as pa

    if isinstance(array.type, pa.BaseExtensionType):
        array = array.storage
    return [_json_cell(text, odd) for text in array.to_pylist()]


def _is_folded_interval(field: Any) -> bool:
    r"""Whether ``field`` is a month-day-nano interval column built from
    ``timedelta``s (:data:`_INTERVAL_TAG`)."""
    import pyarrow as pa

    key, value = _INTERVAL_TAG
    return (
        field.type == pa.month_day_nano_interval()
        and (field.metadata or {}).get(key) == value
    )


def _folded_intervals(array: Any, odd: list[str]) -> list[Any]:
    r"""The cells of an interval column built from ``timedelta``s, each the
    ``timedelta``'s own, ``["dur", nanoseconds]``. A ``timedelta`` holds no
    months (psycopg folds a month into 30 days and a year into 365, which
    months alone cannot tell apart), so an interval holding some is
    unencodable."""
    cells: list[Any] = []
    for value in array.to_pylist():
        if value is None:
            cells.append(None)
        elif value.months:
            cells.append(_unencodable('arrow:interval months', odd))
        else:
            cells.append(
                ['dur', str(value.days * 86_400 * 10**9 + value.nanoseconds)]
            )
    return cells


def arrow_values(table: Any, types: list[str] | None = None) -> dict[str, Any]:
    r"""An Arrow table's columns and rows in canonical form, read from
    Arrow (:func:`_arrow_cells`) column by column, so two columns of one
    name stay two; digests as :func:`result_values`'."""
    import pyarrow as pa

    odd: list[str] = []
    columns = [
        (
            _json_texts
            if _is_json_field(field)
            else _folded_intervals
            if _is_folded_interval(field)
            else _arrow_cells
        )(
            column.combine_chunks()
            if column.num_chunks
            else pa.array([], type=column.type),
            odd,
        )
        for field, column in zip(table.schema, table.columns, strict=True)
    ]
    rows = (
        [list(row) for row in zip(*columns, strict=True)]
        if columns
        else [[] for _ in range(table.num_rows)]
    )
    return _values(table.schema.names, types, _texts(rows), odd)


def _description(
    driver: Any, names: Mapping[int, str] | None = None
) -> tuple[list[str], list[str]] | None:
    r"""The columns of the result a driver holds, as its ``description``
    names them and their types (DuckDB's names; off DuckDB the driver's
    own: ADBC's Arrow type, psycopg's display of the Postgres type), or
    None when it gives none. A type the driver knows only by the number
    the server gave it (a composite type the loader created) is named by
    ``names``, never by that number, which differs from run to run."""
    try:
        described = driver.description
    except Exception:  # noqa: BLE001 - the capture's own reading
        return None
    if not described:
        return None
    types = []
    for column in described:
        code = column[1]
        named = (names or {}).get(code) if isinstance(code, int) else None
        types.append(
            str(named or getattr(column, 'type_display', None) or code)
        )
    return [str(c[0]) for c in described], types


class DriverLog:
    r"""What the capture recorded of one case: each statement, in the order
    the driver ran them, with its handle (0 the connection the source was
    given, then each derived cursor in the order made), the public call it
    ran in (``stage``: ``from_source``, ``materialize`` or ``predict``),
    typed parameters and every result read from it; each relation bound by
    name; and what it could not record."""

    def __init__(self) -> None:
        self.statements: list[dict[str, Any]] = []
        self.bindings: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.incomplete: list[str] = []
        self.passed: list[str] = []
        self.handles = 0
        #: The public call the runner is making, set by it before each.
        self.stage: str | None = None
        #: The names of the types the server numbers per run (Postgres's
        #: composite types), by number, for the results' type names.
        self.type_names: dict[int, str] = {}
        #: The recording mode's keeper of every result whole (Phase 3,
        #: choice 2), or None outside it.
        self.recorder: Recorder | None = None

    def handle(self) -> int:
        number = self.handles
        self.handles += 1
        return number

    def record(self) -> dict[str, Any]:
        return {
            'statements': self.statements,
            'bindings': self.bindings,
            'events': self.events,
            'incomplete': sorted(set(self.incomplete)),
            'passed_through': sorted(set(self.passed)),
            'cursors': self.handles - 1,
        }


class _Captured:
    r"""A DuckDB connection or derived cursor, as the capture hands it to
    the source: every call goes to the driver unchanged, with the same
    arguments, once, and its answer comes back unchanged; what was run and
    read is recorded on the side.

    It passes the reference's dispatch as the connection does (design,
    "Driver capture"): its class's module is under ``duckdb``, which
    ``sdm_connectors.source._backend_of`` reads over the MRO; a derived
    cursor is the same class, so ``reader._query_handle``'s
    ``isinstance(cursor, type(connection))`` holds as it does for the
    driver; and ``__class__`` answers the driver's class, so
    ``Graph.from_duckdb``'s ``isinstance(connection, DuckDBPyConnection)``
    holds too.
    """

    __slots__ = ('_driver', '_log', '_handle', '_statement')
    #: Whether the driver's ``description`` names DuckDB's types, whose
    #: reads as rows may lose precision (:data:`_LOSSY_AS_ROWS`).
    _DUCKDB_TYPES = True

    def __init__(self, driver: Any, log: DriverLog) -> None:
        object.__setattr__(self, '_driver', driver)
        object.__setattr__(self, '_log', log)
        object.__setattr__(self, '_handle', log.handle())
        object.__setattr__(self, '_statement', None)

    @property  # type: ignore[misc]
    def __class__(self) -> type:
        return type(self._driver)

    def _same(self, answer: Any) -> Any:
        # The driver gives itself back from execute and register; the
        # caller gets this proxy in its place, so what it reads next is
        # recorded too.
        return self if answer is self._driver else answer

    def cursor(self) -> Any:
        derived = self._driver.cursor()
        self._log.events.append({'event': 'cursor', 'handle': self._handle})
        return type(self)(derived, self._log)

    def execute(self, *arguments: Any, **keywords: Any) -> Any:
        # Bound as the driver binds them: execute(query, parameters=None).
        query = keywords.get(
            'query',
            keywords.get(
                'operation',
                keywords.get('command', arguments[0] if arguments else None),
            ),
        )
        # Bound by each driver's own name: DuckDB's and Databricks'
        # parameters, Snowflake's params (qmark), Spark's args.
        parameters = next(
            (
                keywords[name]
                for name in ('parameters', 'params', 'args')
                if name in keywords
            ),
            arguments[1] if len(arguments) > 1 else None,
        )
        statement: dict[str, Any] = {
            'handle': self._handle,
            'stage': self._log.stage,
            'sql': query if isinstance(query, str) else None,
            'parameters': _parameters(parameters),
            'results': [],
            'error': None,
        }
        if not isinstance(query, str):
            self._log.incomplete.append('execute:not_text')
        self._log.statements.append(statement)
        object.__setattr__(self, '_statement', statement)
        try:
            answer = self._driver.execute(*arguments, **keywords)
        except BaseException as error:
            statement['error'] = type(error).__name__
            raise
        if (
            answer is not None
            and answer is not self._driver
            and hasattr(answer, 'fetchall')
        ):
            # A connection's execute that gives a new cursor (psycopg's):
            # that cursor is derived, and reads the statement's result.
            self._log.events.append({'event': 'cursor', 'handle': self._handle})
            derived = type(self)(answer, self._log)
            object.__setattr__(derived, '_statement', statement)
            return derived
        return self._same(answer)

    def _read(self, method: str, *arguments: Any, **keywords: Any) -> Any:
        answer = getattr(self._driver, method)(*arguments, **keywords)
        statement = self._statement
        if statement is None:
            self._log.incomplete.append(f'{method}:no_statement')
            return answer
        # The result's columns and DuckDB's types for them, read from the
        # driver's description, which runs nothing.
        described = _description(self._driver, self._log.type_names)
        names, types = described if described else ([], None)
        if method in _ROWS or method in _ROW:
            if method in _ROWS:
                rows = list(answer)
            else:
                rows = [] if answer is None else [answer]
            result = {'method': method, 'rows': len(rows)}
            result['digest'] = _rows_digest(rows)
            if described is None:
                self._log.incomplete.append(f'{method}:no_description')
            result['values'] = result_values(
                names, types, rows, duckdb_types=self._DUCKDB_TYPES
            )
            kept: Any = rows
        else:
            # The DuckDB types are recorded for an Arrow read too, and the
            # comparator requires them from every Python driver's read.
            if described is None:
                self._log.incomplete.append(f'{method}:no_description')
            result = {
                'method': method,
                'rows': answer.num_rows,
                'schema': [
                    {'name': field.name, 'type': str(field.type)}
                    for field in answer.schema
                ],
                'digest': _arrow_digest(answer),
                'values': arrow_values(answer, types),
            }
            kept = answer
        statement['results'].append(result)
        _keep(self._log, statement, kept, names)
        return answer

    def fetchall(self, *arguments: Any, **keywords: Any) -> Any:
        return self._read('fetchall', *arguments, **keywords)

    def fetchmany(self, *arguments: Any, **keywords: Any) -> Any:
        return self._read('fetchmany', *arguments, **keywords)

    def fetchone(self, *arguments: Any, **keywords: Any) -> Any:
        return self._read('fetchone', *arguments, **keywords)

    def to_arrow_table(self, *arguments: Any, **keywords: Any) -> Any:
        return self._read('to_arrow_table', *arguments, **keywords)

    def fetch_arrow_table(self, *arguments: Any, **keywords: Any) -> Any:
        return self._read('fetch_arrow_table', *arguments, **keywords)

    def register(self, name: Any, value: Any) -> Any:
        binding: dict[str, Any] = {
            'handle': self._handle,
            'stage': self._log.stage,
            'name': name,
            'type': f'{type(value).__module__}.{type(value).__qualname__}',
        }
        if type(value).__module__.startswith('pyarrow') and hasattr(
            value, 'combine_chunks'
        ):
            binding['rows'] = value.num_rows
            binding['schema'] = [
                {'name': field.name, 'type': str(field.type)}
                for field in value.schema
            ]
            binding['digest'] = _arrow_digest(value)
            binding['values'] = arrow_values(value)
        else:
            self._log.incomplete.append('register:not_arrow')
        self._log.bindings.append(binding)
        return self._same(self._driver.register(name, value))

    def adbc_ingest(
        self, name: Any, data: Any, *arguments: Any, **keywords: Any
    ) -> Any:
        r"""ADBC's bulk load of an Arrow table under a name, recorded as a
        relation bound by name, as DuckDB's ``register`` is."""
        binding: dict[str, Any] = {
            'handle': self._handle,
            'stage': self._log.stage,
            'name': name,
            'type': f'{type(data).__module__}.{type(data).__qualname__}',
        }
        if type(data).__module__.startswith('pyarrow') and hasattr(
            data, 'combine_chunks'
        ):
            binding['rows'] = data.num_rows
            binding['schema'] = [
                {'name': field.name, 'type': str(field.type)}
                for field in data.schema
            ]
            binding['digest'] = _arrow_digest(data)
            binding['values'] = arrow_values(data)
        else:
            self._log.incomplete.append('adbc_ingest:not_arrow')
        self._log.bindings.append(binding)
        return self._driver.adbc_ingest(name, data, *arguments, **keywords)

    def unregister(self, name: Any) -> Any:
        self._log.events.append(
            {'event': 'unregister', 'handle': self._handle, 'name': name}
        )
        return self._same(self._driver.unregister(name))

    def _event(self, method: str, *arguments: Any, **keywords: Any) -> Any:
        self._log.events.append({'event': method, 'handle': self._handle})
        return self._same(getattr(self._driver, method)(*arguments, **keywords))

    def close(self) -> Any:
        return self._event('close')

    def commit(self) -> Any:
        return self._event('commit')

    def rollback(self) -> Any:
        return self._event('rollback')

    def begin(self) -> Any:
        return self._event('begin')

    def __enter__(self) -> Any:
        self._driver.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        self._log.events.append({'event': 'exit', 'handle': self._handle})
        return self._driver.__exit__(*exc)

    def __repr__(self) -> str:
        return repr(self._driver)

    def __getattr__(self, name: str) -> Any:
        if name in _UNRECORDED or name in _UNCAPTURED:
            self._log.incomplete.append(name)
        else:
            self._log.passed.append(name)
        return getattr(self._driver, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._driver, name, value)


# Under the driver's own package, which the reference's backend dispatch
# reads (sdm_connectors.source._DRIVER_MODULES); a test-only name no import
# resolves.
_Captured.__module__ = 'duckdb.sdm_capture'


class _CapturedSQLite(_Captured):
    r"""The capture of an ADBC SQLite connection (T2.1a), under that
    driver's package, which the reference's dispatch reads."""

    __slots__ = ()
    _DUCKDB_TYPES = False


class _CapturedPostgres(_Captured):
    r"""The capture of a psycopg connection (T2.1a), under that driver's
    package, which the reference's dispatch reads."""

    __slots__ = ()
    _DUCKDB_TYPES = False


_CapturedSQLite.__module__ = 'adbc_driver_sqlite.sdm_capture'
_CapturedPostgres.__module__ = 'psycopg.sdm_capture'
#: The capture's class for each database's driver.
_CAPTURES = {
    'duckdb': _Captured,
    'sqlite': _CapturedSQLite,
    'postgres': _CapturedPostgres,
}


def capture(connection: Any, log: DriverLog, database: str = 'duckdb') -> Any:
    r"""``connection``, a driver connection to ``database``, as the capture
    hands it to a source."""
    return _CAPTURES[database](connection, log)


def _logged_texts(
    log: DriverLog | None, logged: list[str]
) -> tuple[int, list[int]]:
    r"""Each captured statement's text against the text DuckDB logged for
    it, in order: how many were compared, and the positions that differ.
    A ``PRAGMA`` is not compared, since DuckDB logs it as the table
    function it runs; binding names are the same in both, being one run's.
    With no capture, or counts that differ (which completeness reports),
    nothing is compared."""
    ran = [text for text in logged if text]
    if log is None or len(ran) != len(log.statements):
        return 0, []
    compared, differing = 0, []
    for position, (statement, text) in enumerate(
        zip(log.statements, ran, strict=True)
    ):
        sql = statement['sql']
        if sql is None or sql.lstrip().upper().startswith('PRAGMA'):
            continue
        compared += 1
        if sql != text:
            differing.append(position)
    return compared, differing


def enable_query_log(connection: Any) -> None:
    r"""Turns on DuckDB's own query log, which every connection and
    derived cursor of the database writes to, after the case's setup."""
    connection.execute("CALL enable_logging('QueryLog')")


def query_log(connection: Any) -> list[str]:
    r"""Each query DuckDB logged since :func:`enable_query_log`, in order:
    a statement's text, or ``''`` for a relation bound through the API
    (``register``)."""
    rows = connection.execute(
        'SELECT message FROM duckdb_logs '
        "WHERE type = 'QueryLog' ORDER BY context_id"
    ).fetchall()
    return [row[0] for row in rows]


class StatementLog:
    r"""The query log of a run off DuckDB (T2.1a fix 4), DuckDB's own log's
    twin: every statement the driver itself runs, in order, with ``''``
    for a relation bound by name (ADBC's ``adbc_ingest``), whatever object
    called it, the capture's proxy or not. It wraps the driver's own
    classes for one case, restored after, and is entered only as a
    context (``with``), so a case that fails leaves the driver as it was
    (P2-T2.1a-2); test-only, as the capture is. :func:`_logged_texts`
    holds the capture's statements to it, as to DuckDB's.

    What it wraps, per leg: ADBC SQLite's cursor; psycopg's cursor on
    Postgres and Lakebase, ``copy`` and ``stream`` with ``execute`` and
    ``executemany`` (P2-T2.1a-2); the Databricks SQL connector's and
    Snowflake connector's cursors (Phase 3); and a Spark session's ``sql``,
    the classic session's and Spark Connect's, each that is importable."""

    #: Each leg's driver classes (module, class) and the methods logged,
    #: each method with whether it binds a relation rather than runs one.
    TARGETS: dict[
        str, tuple[tuple[str, str, tuple[tuple[str, bool], ...]], ...]
    ] = {
        'sqlite': (
            (
                'adbc_driver_manager.dbapi',
                'Cursor',
                (
                    ('execute', False),
                    ('executemany', False),
                    ('adbc_ingest', True),
                ),
            ),
        ),
        'postgres': (
            (
                'psycopg',
                'Cursor',
                (
                    ('execute', False),
                    ('executemany', False),
                    ('copy', False),
                    ('stream', False),
                ),
            ),
        ),
        'lakebase': (
            (
                'psycopg',
                'Cursor',
                (
                    ('execute', False),
                    ('executemany', False),
                    ('copy', False),
                    ('stream', False),
                ),
            ),
        ),
        'databricks-sql': (
            (
                'databricks.sql.client',
                'Cursor',
                (('execute', False), ('executemany', False)),
            ),
        ),
        'snowflake': (
            (
                'snowflake.connector.cursor',
                'SnowflakeCursor',
                (('execute', False), ('executemany', False)),
            ),
        ),
        'databricks-spark': (
            ('pyspark.sql.session', 'SparkSession', (('sql', False),)),
            ('pyspark.sql.connect.session', 'SparkSession', (('sql', False),)),
        ),
    }
    #: The names a driver's method takes its statement by.
    _QUERY = ('query', 'operation', 'command', 'statement', 'sqlQuery')

    def __init__(self, database: str) -> None:
        self.database = database
        self.texts: list[str] = []
        self._saved: list[tuple[Any, str, Any, bool]] = []

    def __enter__(self) -> StatementLog:
        import importlib

        log = self.texts
        found = 0

        def text(query: Any) -> str:
            # A statement composed by the driver's own SQL builder logs its
            # kind; the capture records no text for it either.
            return (
                query if isinstance(query, str) else f'<{type(query).__name__}>'
            )

        def wrap(cls: Any, method: str, binding: bool) -> None:
            if not hasattr(cls, method):
                return
            own = method in cls.__dict__
            original = cls.__dict__[method] if own else getattr(cls, method)

            def logged(self: Any, *arguments: Any, **keywords: Any) -> Any:
                if binding:
                    log.append('')
                else:
                    query = next(
                        (
                            keywords[k]
                            for k in StatementLog._QUERY
                            if k in keywords
                        ),
                        arguments[0] if arguments else None,
                    )
                    log.append(text(query))
                return original(self, *arguments, **keywords)

            self._saved.append((cls, method, original, own))
            setattr(cls, method, logged)

        try:
            seen: set[int] = set()
            for module, name, methods in self.TARGETS[self.database]:
                try:
                    cls = getattr(importlib.import_module(module), name)
                except (ImportError, AttributeError):
                    # Spark Connect or the classic session alone: each one
                    # importable is logged.
                    continue
                if id(cls) in seen:
                    continue
                seen.add(id(cls))
                found += 1
                for method, binding in methods:
                    wrap(cls, method, binding)
            if not found:
                raise RunnerFailure(
                    f'the {self.database} driver is not installed: its '
                    'statements cannot be logged'
                )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        while self._saved:
            cls, method, original, own = self._saved.pop()
            if own:
                setattr(cls, method, original)
            else:
                delattr(cls, method)


# --- warehouse legs and the recording mode (Phase 3, choice 2) -----------------------

#: A placeholder a case writes for a private name: ``{catalog}`` and
#: ``{schema}`` (``{catalog}`` standing for Snowflake's database), or an
#: existing view's (``{shop_metric_view}``). On a warehouse leg the run
#: resolves each from the live configuration's ``names`` (:func:`resolved`),
#: and the recording mode writes each back before anything is written
#: (:func:`placeheld`).
PLACEHOLDER = re.compile(r'\{([a-z][a-z0-9_]*)\}')
#: The parts of a case's ``input`` the placeholders are resolved in: what
#: names the warehouse's objects. ``base`` and ``setup`` build the data in a
#: local DuckDB, and ``pql`` and ``materialize`` name no warehouse object.
_RESOLVED_IN = ('from_source', 'edits', 'view')
#: The environment variable naming the live configuration file, a JSON file
#: outside the repository that only the orchestrator writes (C-1, E-0, E-1;
#: :func:`live_configuration`).
LIVE_CONFIGURATION = 'SDM_LIVE_CONFIG'
#: The live configuration's format.
LIVE_FORMAT = 'sdm-live/1'
#: The environment variables whose values are private names wherever they
#: are set (hosts, accounts, users): a capture holding one is not written.
PRIVATE_ENVIRONMENT = (
    'DATABRICKS_HOST',
    'DATABRICKS_CLIENT_ID',
    'DATABRICKS_WAREHOUSE_ID',
    'SNOWFLAKE_ACCOUNT',
    'SNOWFLAKE_USER',
    'SNOWFLAKE_HOST',
    'PGHOST',
    'PGUSER',
    'USER',
    'LOGNAME',
)
#: The shortest value counted as a private name: a shorter one ("db",
#: "1") would match ordinary words and refuse every capture.
_PRIVATE_MIN = 4
#: The recording mode's own format (design choice 2): one line per case
#: and leg, the input T3.3's ``captures.py import`` turns into the capture
#: (``sdm-capture/1``, the stand-in's), which ``capture_check.py
#: recording`` holds complete.
RECORD_FORMAT = 'sdm-record/1'
#: A statement's random draw: Databricks' ``TABLESAMPLE (p PERCENT)
#: [REPEATABLE (seed)]`` with the percentage as the old client printed it
#: (RE/backend/databricks/sampler.py:255-262), and Snowflake's unseeded
#: ``SAMPLE ROW (n ROWS)`` (RE/backend/snow/sampler.py:132).
DRAWS = (
    (
        'tablesample',
        re.compile(
            r'\bTABLESAMPLE\s*\(\s*(?P<fraction>[^\s()]+)\s+PERCENT\s*\)'
            r'(?:\s*REPEATABLE\s*\(\s*(?P<seed>[^\s()]+)\s*\))?',
            re.IGNORECASE,
        ),
    ),
    (
        'sample_rows',
        re.compile(
            r'\bSAMPLE\s+ROW\s*\(\s*(?P<fraction>[^\s()]+)\s+ROWS\s*\)',
            re.IGNORECASE,
        ),
    ),
)
_FROM = re.compile(r'\bFROM\s+', re.IGNORECASE)
#: A read of a relation derived from a view (the old client's
#: ``_quoted_source_name``, RE/backend/databricks/table.py:170-195, pasted
#: into ``FROM``), ending in the ``LIMIT`` its samples take: re-run without
#: the limit, it is the relation as the warehouse evaluates it.
_DERIVED = re.compile(r'\bFROM\s+\(\s*SELECT\b', re.IGNORECASE)
_LIMIT = re.compile(r'\s+LIMIT\s+[0-9]+\s*$', re.IGNORECASE)


def draws_of(sql: str | None) -> list[dict[str, Any]]:
    r"""The random draws one statement's text makes, in order: each draw's
    kind, the table it draws from as written (the text between the last
    ``FROM`` before it and it), and its fraction and seed as written
    (``None`` for an unseeded draw). ``capture_check.py`` reads the same
    draws from a capture's statements, so a draw recorded and a draw made
    are the one rule."""
    if not sql:
        return []
    found = []
    for kind, pattern in DRAWS:
        for match in pattern.finditer(sql):
            before = [m.end() for m in _FROM.finditer(sql, 0, match.start())]
            table = sql[before[-1] : match.start()].strip() if before else ''
            found.append(
                {
                    'kind': kind,
                    'at': match.start(),
                    'table': table,
                    'fraction': match.group('fraction'),
                    'seed': match.groupdict().get('seed'),
                }
            )
    found.sort(key=lambda draw: draw['at'])
    for draw in found:
        del draw['at']
    return found


def input_hash(
    case: dict[str, Any], bases: dict[str, list[str]] | None = None
) -> str:
    r"""The hash of what the old client's answer to a case depends on
    (design choice 2): its ``input``, its ``until`` and its base's
    statements (``bases`` as :func:`load_bases` gives them, this
    checkout's by default), as canonical JSON. A capture records it, so
    ``capture_check.py`` finds a case changed after the capture
    mechanically; its ``why``, paths, routes, status, expectation and
    ``core`` change nothing the old client is asked, so a case given
    ``core`` later keeps its capture (T3.4's brief)."""
    bases = load_bases() if bases is None else bases
    asked = {
        'input': case['input'],
        'until': case['until'],
        'base': bases.get(case['input']['base']),
    }
    return digest(
        json.dumps(asked, sort_keys=True, ensure_ascii=False).encode('utf-8')
    )


#: The environment variable each placeholder's private name is read from:
#: ``{catalog}`` from ``SDM_CAPTURE_CATALOG``, ``{shop_metric_view}`` from
#: ``SDM_CAPTURE_SHOP_METRIC_VIEW``.
CAPTURE_NAME = 'SDM_CAPTURE_'


def placeholder_names(environ: Mapping[str, str]) -> dict[str, str]:
    r"""Each placeholder's private name the environment sets
    (:data:`CAPTURE_NAME`), by placeholder."""
    return {
        key[len(CAPTURE_NAME) :].lower(): value
        for key, value in environ.items()
        if key.startswith(CAPTURE_NAME) and value
    }


def resolve_placeholders(
    case: dict[str, Any], environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    r"""``case`` with each ``{name}`` in its ``input`` replaced by
    ``environ['SDM_CAPTURE_' + NAME]`` (the S2 test authors' placeholders:
    ``{catalog}``, ``{schema}`` and the existing views'), for a live run; a
    placeholder with nothing set fails the run, named by its placeholder,
    never by a value. The case given is not changed."""
    environ = os.environ if environ is None else environ
    return resolved(case, placeholder_names(environ), everywhere=True)


def resolved(
    case: dict[str, Any], names: Mapping[str, str], everywhere: bool = False
) -> dict[str, Any]:
    r"""``case`` with every placeholder in the parts of its ``input`` that
    name warehouse objects (:data:`_RESOLVED_IN`) replaced by its private
    name from ``names``; a placeholder ``names`` lacks fails the run, named
    by its placeholder, never by a value."""

    def resolve(value: Any) -> Any:
        if isinstance(value, str):

            def name(match: re.Match[str]) -> str:
                if match.group(1) not in names:
                    raise RunnerFailure(
                        f'{case["id"]}: the live configuration names no '
                        f'{{{match.group(1)}}}'
                    )
                return names[match.group(1)]

            return PLACEHOLDER.sub(name, value)
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        return value

    out = copy.deepcopy(case)
    if everywhere:
        out['input'] = resolve(out['input'])
        return out
    for key in _RESOLVED_IN:
        if key in out['input']:
            out['input'][key] = resolve(out['input'][key])
    return out


def _resolve_environment(value: Any, private: list[str]) -> Any:
    r"""A configuration value with each ``{"env": NAME}`` read from the
    environment; every text it holds is added to ``private``."""
    if isinstance(value, dict) and set(value) == {'env'}:
        read = os.environ.get(str(value['env']))
        if read is None:
            raise RunnerFailure(
                f'the live configuration reads {value["env"]}, which is unset'
            )
        private.append(read)
        return read
    if isinstance(value, dict):
        return {
            key: _resolve_environment(item, private)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_environment(item, private) for item in value]
    if isinstance(value, str):
        private.append(value)
    return value


def live_configuration(
    leg: str, environ: Mapping[str, str] | None = None
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    r"""The live configuration of ``leg``: how its connection is opened, the
    private name of each placeholder (``names``), and every private value
    a capture must not hold.

    It is the JSON file ``SDM_LIVE_CONFIG`` names, outside the repository::

        {"format": "sdm-live/1",
         "private": ["<a host>", ...],
         "legs": {"<leg>": {"driver": {...}, "warehouse": {...},
                            "lakebase": {...}}}}

    The placeholders' private names come from the environment
    (``SDM_CAPTURE_CATALOG``, ``SDM_CAPTURE_SCHEMA``, an existing view's
    ``SDM_CAPTURE_SHOP_METRIC_VIEW``; :func:`resolve_placeholders`).

    ``driver`` holds the keyword arguments of the leg's own driver connect
    (``databricks.sql.connect``, ``snowflake.connector.connect``);
    ``warehouse`` those of ``sdm_connectors.databricks.warehouse`` (with
    ``profile``, the workspace client's), which the ``warehouse`` connect
    path and, without ``driver``, every databricks-sql case opens through;
    ``lakebase`` those of ``sdm_connectors.databricks.lakebase`` (with
    ``profile``), which every lakebase case opens through. Any value may be
    ``{"env": "NAME"}``, read from the environment (Snowflake's
    credentials, KO3-24). Every text the file holds, every placeholder's name, connect
    arguments alike, every private value read from the environment, and
    the values of :data:`PRIVATE_ENVIRONMENT` set at least
    :data:`_PRIVATE_MIN` long are private."""
    environ = os.environ if environ is None else environ
    where = environ.get(LIVE_CONFIGURATION)
    if not where:
        raise RunnerFailure(
            f'a {leg} run needs {LIVE_CONFIGURATION}, the live configuration'
        )
    try:
        document = json.loads(Path(where).read_text(encoding='utf-8'))
    except (OSError, ValueError, RecursionError) as error:
        raise RunnerFailure(
            f'the live configuration cannot be read ({type(error).__name__})'
        ) from None
    if not isinstance(document, dict) or document.get('format') != LIVE_FORMAT:
        raise RunnerFailure(f'the live configuration is not {LIVE_FORMAT}')
    spec = (document.get('legs') or {}).get(leg)
    if not isinstance(spec, dict):
        raise RunnerFailure(f'the live configuration has no {leg} leg')
    private: list[str] = []
    spec = _resolve_environment(spec, private)
    # Each placeholder's private name, from the environment
    # (resolve_placeholders), private wherever it appears.
    names = placeholder_names(environ)
    private.extend(names.values())
    _resolve_environment(document.get('private') or [], private)
    private.extend(
        environ[name] for name in PRIVATE_ENVIRONMENT if environ.get(name)
    )
    kept = sorted(
        {value for value in private if len(value) >= _PRIVATE_MIN},
        key=lambda value: (-len(value), value),
    )
    return spec, dict(names), kept


def _name_pattern(value: str) -> re.Pattern[str]:
    r"""``value`` as a whole name, in any case: not inside a longer name."""
    return re.compile(
        r'(?<![A-Za-z0-9_])' + re.escape(value) + r'(?![A-Za-z0-9_])',
        re.IGNORECASE,
    )


class Placeheld:
    r"""Writes each private name of a leg back as its placeholder, and finds
    any private value left (design choice 2: "Names of catalogs, schemas,
    hosts and users are replaced by placeholders before anything is
    written, and the write fails on any private name from the
    environment")."""

    def __init__(
        self, names: Mapping[str, str], private: Iterable[str]
    ) -> None:
        # The longest name first, so a name inside another is not split.
        self._names = [
            (_name_pattern(value), '{' + key + '}')
            for key, value in sorted(
                names.items(), key=lambda item: (-len(item[1]), item[0])
            )
        ]
        self._private = [value.lower() for value in private]

    def text(self, value: str) -> str:
        for pattern, placeholder in self._names:
            value = pattern.sub(placeholder, value)
        return value

    def value(self, value: Any) -> Any:
        r"""A JSON-like value with every text placeheld, keys included."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.value(item) for item in value)
        if isinstance(value, dict):
            return {
                self.value(key): self.value(item) for key, item in value.items()
            }
        return value

    def table(self, table: Any) -> Any:
        r"""An Arrow table with every text value and field name placeheld,
        column by column (a result may name a column twice); a table
        holding none comes back as it is."""
        import pyarrow as pa

        columns = [column.to_pylist() for column in table.columns]
        names = [field.name for field in table.schema]
        held_columns = self.value(columns)
        held_names = [self.text(name) for name in names]
        if held_columns == columns and held_names == names:
            return table
        fields = [
            field.with_name(name)
            for field, name in zip(table.schema, held_names, strict=True)
        ]
        return pa.Table.from_arrays(
            [
                pa.array(values, type=field.type)
                for values, field in zip(held_columns, fields, strict=True)
            ],
            schema=pa.schema(fields, metadata=table.schema.metadata),
        )

    def leaked(self, text: str) -> bool:
        r"""Whether ``text`` holds a private value, in any case."""
        lowered = text.lower()
        return any(value in lowered for value in self._private)


def arrow_ipc(table: Any) -> str:
    r"""An Arrow table as the base64 text of one IPC stream, its batches
    combined (batch boundaries are no part of a result, H9), compressed
    with zstd where the build has it."""
    import base64

    import pyarrow as pa

    options = pa.ipc.IpcWriteOptions(
        compression='zstd' if pa.Codec.is_available('zstd') else None
    )
    sink = pa.BufferOutputStream()
    combined = table.combine_chunks()
    with pa.ipc.new_stream(sink, combined.schema, options=options) as writer:
        writer.write_table(combined)
    return base64.b64encode(sink.getvalue().to_pybytes()).decode('ascii')


def read_arrow_ipc(text: str) -> Any:
    r"""The Arrow table :func:`arrow_ipc` wrote."""
    import base64

    import pyarrow as pa

    return pa.ipc.open_stream(base64.b64decode(text)).read_all()


class Recorder:
    r"""What the recording mode keeps beside the driver capture: every
    result as the driver gave it, by statement and read (an Arrow read as
    its table, with the warehouse's own column types; a read as rows as
    the column names and the rows), which the capture writes whole."""

    def __init__(self) -> None:
        self.kept: dict[tuple[int, int], Any] = {}


def _keep(
    log: DriverLog,
    statement: dict[str, Any],
    answer: Any,
    names: Iterable[str] | None = None,
) -> None:
    r"""Keeps the result just recorded on ``statement`` for the recording
    mode (``log.recorder``); nothing without one."""
    recorder = log.recorder
    if recorder is None:
        return
    key = (log.statements.index(statement), len(statement['results']) - 1)
    if hasattr(answer, 'schema') and hasattr(answer, 'num_rows'):
        recorder.kept[key] = answer
    else:
        recorder.kept[key] = (list(names or []), [tuple(row) for row in answer])


def rows_table(names: list[str], rows: list[tuple[Any, ...]]) -> Any:
    r"""Rows read from a driver as an Arrow table, each column's type the
    one Arrow infers from its values (null when it holds none)."""
    import pyarrow as pa

    columns = list(zip(*rows, strict=True)) if rows else [() for _ in names]
    return pa.Table.from_arrays(
        [pa.array(list(column)) for column in columns], names=list(names)
    )


class _CapturedWarehouse(_Captured):
    r"""The capture of a warehouse driver's connection or cursor (Phase 3):
    the DB-API calls of :class:`_Captured`, ``executemany``, and each
    Arrow read the driver has (``fetchall_arrow``, ``fetchmany_arrow``,
    Databricks'; ``fetch_arrow_all``, ``fetch_pandas_all``, Snowflake's),
    each kept whole in the recording mode."""

    __slots__ = ()
    _DUCKDB_TYPES = False
    #: The driver's Arrow reads, recorded where the driver has them.
    _READS = ('fetchall_arrow', 'fetchmany_arrow')

    def executemany(self, *arguments: Any, **keywords: Any) -> Any:
        # Bound as DB-API binds them: executemany(operation, seq_of_params).
        query = keywords.get(
            'operation',
            keywords.get('query', arguments[0] if arguments else None),
        )
        rows = keywords.get(
            'seq_of_parameters',
            keywords.get(
                'params_seq', arguments[1] if len(arguments) > 1 else None
            ),
        )
        rows = list(rows) if rows is not None else None
        statement: dict[str, Any] = {
            'handle': self._handle,
            'stage': self._log.stage,
            'sql': query if isinstance(query, str) else None,
            'parameters': (
                None
                if rows is None
                else [
                    {'name': str(position), **_typed(row)}
                    for position, row in enumerate(rows)
                ]
            ),
            'results': [],
            'error': None,
        }
        if not isinstance(query, str):
            self._log.incomplete.append('executemany:not_text')
        self._log.statements.append(statement)
        object.__setattr__(self, '_statement', statement)
        if rows is not None:
            if 'seq_of_parameters' in keywords:
                keywords['seq_of_parameters'] = rows
            elif 'params_seq' in keywords:
                keywords['params_seq'] = rows
            elif len(arguments) > 1:
                arguments = (arguments[0], rows, *arguments[2:])
        try:
            answer = self._driver.executemany(*arguments, **keywords)
        except BaseException as error:
            statement['error'] = type(error).__name__
            raise
        return self._same(answer)

    def __getattr__(self, name: str) -> Any:
        # A read is offered only where the driver has it: the reference's
        # reader takes the first Arrow read the cursor has (reader.py's
        # ``hasattr`` order), so a capture offering one the driver lacks
        # would change what it calls.
        if name in self._READS:
            if not hasattr(self._driver, name):
                raise AttributeError(name)
            return functools.partial(self._recorded_read, name)
        return super().__getattr__(name)

    def _recorded_read(
        self, name: str, *arguments: Any, **keywords: Any
    ) -> Any:
        return self._read(name, *arguments, **keywords)


class _CapturedDatabricks(_CapturedWarehouse):
    r"""The capture of a Databricks SQL connection or cursor, under the
    driver's package, which the reference's dispatch reads."""

    __slots__ = ()


class _CapturedSnowflake(_CapturedWarehouse):
    r"""The capture of a Snowflake connector connection or cursor, under the
    driver's package. ``fetch_arrow_all`` and ``fetch_pandas_all`` are its
    reads (RE/backend/snow/table.py:200,319); a pandas read is kept as the
    Arrow it converts from."""

    __slots__ = ()

    _READS = ('fetch_arrow_all', 'fetch_pandas_all')

    def _recorded_read(
        self, name: str, *arguments: Any, **keywords: Any
    ) -> Any:
        answer = getattr(self._driver, name)(*arguments, **keywords)
        if name == 'fetch_pandas_all':
            import pyarrow as pa

            self._recorded_arrow(
                name, pa.Table.from_pandas(answer, preserve_index=False)
            )
            return answer
        return self._recorded_arrow(name, answer)

    def _recorded_arrow(self, method: str, answer: Any) -> Any:
        statement = self._statement
        if statement is None:
            self._log.incomplete.append(f'{method}:no_statement')
            return answer
        if answer is None:
            # The connector gives None for a result with no rows unless
            # force_return_table: recorded as no rows read.
            statement['results'].append(
                {
                    'method': method,
                    'rows': 0,
                    'values': result_values([], None, [], duckdb_types=False),
                }
            )
            _keep(self._log, statement, [], [])
            return answer
        statement['results'].append(
            {
                'method': method,
                'rows': answer.num_rows,
                'schema': [
                    {'name': field.name, 'type': str(field.type)}
                    for field in answer.schema
                ],
                'digest': _arrow_digest(answer),
                'values': arrow_values(answer),
            }
        )
        _keep(self._log, statement, answer)
        return answer


class _CapturedPsycopg(_CapturedPostgres):
    r"""The capture of a psycopg connection (Postgres and Lakebase): the
    calls of :class:`_CapturedPostgres`, and ``executemany``, ``copy`` and
    ``stream`` (P2-T2.1a-2), each recorded as a statement whose result is
    every row or block the caller read."""

    __slots__ = ()

    def _statement_of(
        self, method: str, query: Any, parameters: Any
    ) -> dict[str, Any]:
        statement: dict[str, Any] = {
            'handle': self._handle,
            'stage': self._log.stage,
            'sql': query if isinstance(query, str) else None,
            'parameters': _parameters(parameters),
            'results': [],
            'error': None,
        }
        if not isinstance(query, str):
            self._log.incomplete.append(f'{method}:not_text')
        self._log.statements.append(statement)
        return statement

    def executemany(
        self, query: Any, params_seq: Any, *rest: Any, **keywords: Any
    ) -> Any:
        rows = list(params_seq)
        statement = self._statement_of('executemany', query, None)
        statement['parameters'] = [
            {'name': str(position), **_typed(row)}
            for position, row in enumerate(rows)
        ]
        object.__setattr__(self, '_statement', statement)
        try:
            return self._same(
                self._driver.executemany(query, rows, *rest, **keywords)
            )
        except BaseException as error:
            statement['error'] = type(error).__name__
            raise

    def stream(
        self, query: Any, params: Any = None, *rest: Any, **keywords: Any
    ) -> Any:
        statement = self._statement_of('stream', query, params)
        log = self._log

        def rows() -> Iterator[Any]:
            read: list[Any] = []
            try:
                for row in self._driver.stream(
                    query, params, *rest, **keywords
                ):
                    read.append(row)
                    yield row
            except BaseException as error:
                statement['error'] = type(error).__name__
                raise
            finally:
                described = _description(self._driver)
                names = described[0] if described else []
                statement['results'].append(
                    {
                        'method': 'stream',
                        'rows': len(read),
                        'digest': _rows_digest(read),
                        'values': result_values(
                            names, None, read, duckdb_types=False
                        ),
                    }
                )
                _keep(log, statement, read, names)

        return rows()

    def copy(
        self,
        statement_text: Any,
        params: Any = None,
        *rest: Any,
        **keywords: Any,
    ) -> Any:
        statement = self._statement_of('copy', statement_text, params)
        manager = self._driver.copy(statement_text, params, *rest, **keywords)
        return _CapturedCopy(manager, statement, self._log)


class _CapturedCopy:
    r"""A psycopg ``Copy`` as the capture hands it back: every block or row
    read or written goes through unchanged and is recorded on the
    statement."""

    def __init__(
        self, manager: Any, statement: dict[str, Any], log: DriverLog
    ) -> None:
        self._manager = manager
        self._copy: Any = None
        self._statement = statement
        self._log = log
        self._read: list[Any] = []

    def __enter__(self) -> _CapturedCopy:
        self._copy = self._manager.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        if exc[0] is not None:
            self._statement['error'] = exc[0].__name__
        blocks = [(item,) for item in self._read]
        self._statement['results'].append(
            {
                'method': 'copy',
                'rows': len(blocks),
                'digest': _rows_digest(blocks),
                'values': result_values(
                    ['block'], None, blocks, duckdb_types=False
                ),
            }
        )
        _keep(self._log, self._statement, blocks, ['block'])
        return self._manager.__exit__(*exc)

    def _through(self, method: str, *arguments: Any) -> Any:
        answer = getattr(self._copy, method)(*arguments)
        if answer:
            self._read.append(answer)
        return answer

    def read(self) -> Any:
        return self._through('read')

    def read_row(self) -> Any:
        return self._through('read_row')

    def rows(self) -> Iterator[Any]:
        for row in self._copy.rows():
            self._read.append(row)
            yield row

    def __iter__(self) -> Iterator[Any]:
        for block in self._copy:
            self._read.append(block)
            yield block

    def write(self, buffer: Any) -> Any:
        self._read.append(buffer)
        return self._copy.write(buffer)

    def write_row(self, row: Any) -> Any:
        self._read.append(row)
        return self._copy.write_row(row)

    def __getattr__(self, name: str) -> Any:
        self._log.incomplete.append(f'copy.{name}')
        return getattr(self._copy, name)


class _CapturedFrame:
    r"""A Spark DataFrame ``spark.sql`` gave, as the capture hands it back:
    reading it (``toArrow``, ``collect``, ``first``, ``toPandas``) runs the
    statement, and each read is recorded on it, the Arrow kept whole."""

    __slots__ = ('_driver', '_log', '_statement')

    def __init__(
        self, driver: Any, log: DriverLog, statement: dict[str, Any]
    ) -> None:
        object.__setattr__(self, '_driver', driver)
        object.__setattr__(self, '_log', log)
        object.__setattr__(self, '_statement', statement)

    @property  # type: ignore[misc]
    def __class__(self) -> type:
        return type(self._driver)

    def _ran(
        self,
        method: str,
        answer: Any,
        table: Any = None,
        rows: list[Any] | None = None,
    ) -> Any:
        statement = self._statement
        if table is not None:
            statement['results'].append(
                {
                    'method': method,
                    'rows': table.num_rows,
                    'schema': [
                        {'name': field.name, 'type': str(field.type)}
                        for field in table.schema
                    ],
                    'digest': _arrow_digest(table),
                    'values': arrow_values(table),
                }
            )
            _keep(self._log, statement, table)
        else:
            read = rows or []
            names = list(getattr(self._driver, 'columns', []) or [])
            statement['results'].append(
                {
                    'method': method,
                    'rows': len(read),
                    'digest': _rows_digest(read),
                    'values': result_values(
                        names, None, read, duckdb_types=False
                    ),
                }
            )
            _keep(self._log, statement, read, names)
        return answer

    def _run(self, method: str, *arguments: Any, **keywords: Any) -> Any:
        try:
            return getattr(self._driver, method)(*arguments, **keywords)
        except BaseException as error:
            self._statement['error'] = type(error).__name__
            raise

    def toArrow(self) -> Any:  # noqa: N802 - Spark's name
        answer = self._run('toArrow')
        return self._ran('toArrow', answer, table=answer)

    def collect(self) -> Any:
        answer = self._run('collect')
        return self._ran('collect', answer, rows=[tuple(row) for row in answer])

    def first(self) -> Any:
        answer = self._run('first')
        return self._ran(
            'first', answer, rows=[] if answer is None else [tuple(answer)]
        )

    def toPandas(self) -> Any:  # noqa: N802 - Spark's name
        import pyarrow as pa

        answer = self._run('toPandas')
        return self._ran(
            'toPandas',
            answer,
            table=pa.Table.from_pandas(answer, preserve_index=False),
        )

    def __getattr__(self, name: str) -> Any:
        # Any other method (a transformation, or a read the capture does
        # not record) leaves the capture incomplete rather than unseen.
        self._log.incomplete.append(f'frame.{name}')
        return getattr(self._driver, name)

    def __repr__(self) -> str:
        return repr(self._driver)


class _CapturedSpark:
    r"""A Spark session as the capture hands it to a source (the reference's
    ``SparkCursor`` runs SQL through ``spark.sql`` and reads
    ``DataFrame.toArrow()``, CN/backends/spark.py:40,84-91): each ``sql``
    call is a statement, its positional ``args`` its parameters, and the
    frame it gives is captured. Named ``SparkSession`` under a ``pyspark``
    package, which the reference's dispatch reads over the type's MRO."""

    __slots__ = ('_driver', '_log', '_handle')

    def __init__(self, driver: Any, log: DriverLog) -> None:
        object.__setattr__(self, '_driver', driver)
        object.__setattr__(self, '_log', log)
        object.__setattr__(self, '_handle', log.handle())

    def sql(self, query: Any, args: Any = None, **keywords: Any) -> Any:
        statement: dict[str, Any] = {
            'handle': self._handle,
            'stage': self._log.stage,
            'sql': query if isinstance(query, str) else None,
            'parameters': _parameters(args),
            'results': [],
            'error': None,
        }
        if not isinstance(query, str):
            self._log.incomplete.append('sql:not_text')
        self._log.statements.append(statement)
        try:
            if args is None:
                frame = self._driver.sql(query, **keywords)
            else:
                frame = self._driver.sql(query, args=args, **keywords)
        except BaseException as error:
            statement['error'] = type(error).__name__
            raise
        return _CapturedFrame(frame, self._log, statement)

    def __getattr__(self, name: str) -> Any:
        if name in ('table', 'read', 'createDataFrame', 'range'):
            self._log.incomplete.append(f'spark.{name}')
        else:
            self._log.passed.append(f'spark.{name}')
        return getattr(self._driver, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._driver, name, value)

    def __repr__(self) -> str:
        return repr(self._driver)


_CapturedSpark.__name__ = 'SparkSession'
_CapturedSpark.__qualname__ = 'SparkSession'
_CapturedSpark.__module__ = 'pyspark.sql.sdm_capture'


class _CapturedSnowpark:
    r"""A Snowpark session as the capture hands it to a source: the reference
    reaches it through its own connector connection
    (``SnowflakeSource``), which the capture wraps. Named ``Session`` under
    ``snowflake.snowpark``, which the reference's dispatch reads."""

    __slots__ = ('_driver', '_log', '_connection')

    def __init__(self, driver: Any, log: DriverLog) -> None:
        object.__setattr__(self, '_driver', driver)
        object.__setattr__(self, '_log', log)
        object.__setattr__(self, '_connection', None)

    @property
    def connection(self) -> Any:
        if self._connection is None:
            object.__setattr__(
                self,
                '_connection',
                _CapturedSnowflake(self._driver.connection, self._log),
            )
        return self._connection

    def __getattr__(self, name: str) -> Any:
        if name in ('sql', 'table', 'create_dataframe'):
            self._log.incomplete.append(f'snowpark.{name}')
        else:
            self._log.passed.append(f'snowpark.{name}')
        return getattr(self._driver, name)

    def __repr__(self) -> str:
        return repr(self._driver)


_CapturedSnowpark.__name__ = 'Session'
_CapturedSnowpark.__qualname__ = 'Session'
_CapturedSnowpark.__module__ = 'snowflake.snowpark.sdm_capture'
_CapturedDatabricks.__module__ = 'databricks.sql.sdm_capture'
_CapturedSnowflake.__module__ = 'snowflake.connector.sdm_capture'
_CapturedPsycopg.__module__ = 'psycopg.sdm_capture'
#: The capture of each warehouse leg's connection, by leg; a connect case's
#: session (Spark's, Snowpark's) by its path.
_WAREHOUSE_CAPTURES: dict[str, Any] = {
    'databricks-sql': _CapturedDatabricks,
    'databricks-spark': _CapturedSpark,
    'snowflake': _CapturedSnowflake,
    'lakebase': _CapturedPsycopg,
    'snowpark': _CapturedSnowpark,
}


class WarehouseTarget:
    r"""What the contract loader (``databases.load(duck, leg, target)``,
    T3.3's warehouse part) loads a case's tables into on a warehouse leg:
    the leg's open connection (a Spark session on ``databricks-spark``),
    the private names of the case's placeholders (``catalog`` and
    ``schema``: the one dedicated test schema, KO3-51), and the case's
    resolved ``input.view``, created from its YAML when it has one."""

    def __init__(
        self,
        connection: Any,
        names: Mapping[str, str],
        view: Mapping[str, Any] | None,
    ) -> None:
        self.connection = connection
        self.names = dict(names)
        self.view = dict(view) if view is not None else None


def _quoted(leg: str, names: Mapping[str, str], table: str) -> str:
    r"""``table`` in the test schema, qualified and quoted as ``leg``'s
    engine reads it, its name exactly as the case's DuckDB data has it."""
    if leg in ('databricks-sql', 'databricks-spark'):
        parts = [names.get('catalog', ''), names.get('schema', ''), table]
        return '.'.join('`' + part.replace('`', '``') + '`' for part in parts)
    if leg == 'snowflake':
        parts = [names.get('catalog', ''), names.get('schema', ''), table]
    else:
        parts = [names.get('schema', ''), table]
    return '.'.join('"' + part.replace('"', '""') + '"' for part in parts)


def raw_arrow(leg: str, connection: Any, sql: str) -> Any:
    r"""What ``sql`` answers on ``leg``, read through the leg's driver
    outside the capture, as Arrow with the warehouse's own column types
    (psycopg's rows as the Arrow :func:`rows_table` infers)."""
    if leg == 'databricks-spark':
        return connection.sql(sql).toArrow()
    if leg == 'lakebase':
        with connection.cursor() as cursor:
            cursor.execute(sql)
            rows = [tuple(row) for row in cursor.fetchall()]
            names = [column[0] for column in cursor.description or ()]
        return rows_table(names, rows)
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        if leg == 'snowflake':
            return cursor.fetch_arrow_all(force_return_table=True)
        return cursor.fetchall_arrow()
    finally:
        cursor.close()


#: The statement each leg's session time zone is read with, outside the
#: capture (choice 2: the stand-in sets DuckDB's session to it).
_ZONE = {
    'databricks-sql': 'SELECT current_timezone() AS zone',
    'databricks-spark': 'SELECT current_timezone() AS zone',
    'snowflake': "SELECT CURRENT_SETTING('TIMEZONE') AS ZONE",
    'lakebase': "SELECT current_setting('TimeZone') AS zone",
}
#: Each leg's driver distributions, whose versions a capture records.
_DRIVERS = {
    'databricks-sql': ('databricks-sql-connector', 'databricks-sdk'),
    'databricks-spark': ('pyspark', 'databricks-connect'),
    'snowflake': ('snowflake-connector-python', 'snowflake-snowpark-python'),
    'lakebase': ('psycopg', 'databricks-sdk'),
}


def _workspace(options: dict[str, Any]) -> Any:
    r"""The workspace client a connect function is given: the profile the
    configuration names, else the ambient one."""
    from databricks.sdk import WorkspaceClient

    profile = options.pop('profile', None)
    return WorkspaceClient(profile=profile) if profile else WorkspaceClient()


def connect_leg(
    leg: str,
    spec: Mapping[str, Any],
    names: Mapping[str, str],
    path: str | None,
) -> tuple[Any, Any]:
    r"""The leg's connection, opened as the live configuration says
    (:func:`live_configuration`), and what closes it: through the case's
    connect ``path`` when it has one, its namespace the test schema so a
    connection handed over as it is resolves the case's tables there."""
    catalog, schema = names.get('catalog'), names.get('schema')
    if leg == 'databricks-sql':
        if path == 'warehouse' or 'driver' not in spec:
            from sdm_connectors.databricks import warehouse

            options = dict(spec.get('warehouse') or {})
            warehouse_id = options.pop('warehouse_id', None)
            if not warehouse_id:
                raise RunnerFailure('the live configuration names no warehouse')
            workspace = _workspace(options)
            driver = dict(options.pop('driver', {}) or {})
            if path == 'warehouse':
                driver.update(catalog=catalog, schema=schema)
            connection = warehouse(warehouse_id, workspace=workspace, **driver)
        else:
            import databricks.sql

            connection = databricks.sql.connect(**spec['driver'])
        return connection, connection.close
    if leg == 'databricks-spark':
        from pyspark.sql import SparkSession

        session = (
            SparkSession.getActiveSession()
            or SparkSession.builder.getOrCreate()
        )
        if path == 'spark':
            session.sql(f'USE CATALOG `{catalog}`')
            session.sql(f'USE SCHEMA `{schema}`')
        # The notebook owns its session: closing the case's stops nothing.
        return session, lambda: None
    if leg == 'snowflake':
        if path == 'snowpark':
            from snowflake.snowpark import Session

            session = Session.builder.configs(
                {**spec['driver'], 'database': catalog, 'schema': schema}
            ).create()
            return session, session.close
        import snowflake.connector

        connection = snowflake.connector.connect(**spec['driver'])
        return connection, connection.close
    if leg == 'lakebase':
        from sdm_connectors.databricks import lakebase

        options = dict(spec.get('lakebase') or {})
        endpoint = options.pop('endpoint', None)
        database = options.pop('database', None)
        if not endpoint or not database:
            raise RunnerFailure(
                'the live configuration names no Lakebase endpoint'
            )
        workspace = _workspace(options)
        driver = dict(options.pop('driver', {}) or {})
        if path == 'lakebase':
            driver['options'] = f'-c search_path={schema}'
        connection = lakebase(
            endpoint, database=database, workspace=workspace, **driver
        )
        return connection, connection.close
    raise RunnerFailure(f'{leg} is not a warehouse leg')


def warehouse_source(
    leg: str, captured: Any, names: Mapping[str, str], given: Mapping[str, Any]
) -> Any:
    r"""The public source a user builds on ``leg`` over the captured
    connection: the raw connection itself for a connect case (found by
    detection, CN/source.py:97-107); the leg's source class in the test
    schema otherwise, and a view case's semantic model of it."""
    import sdm_connectors

    if 'connect' in given:
        return captured
    catalog, schema = names.get('catalog'), names.get('schema')
    if leg in ('databricks-sql', 'databricks-spark'):
        source = sdm_connectors.DatabricksSource(
            captured, catalog=catalog, schema=schema
        )
    elif leg == 'snowflake':
        source = sdm_connectors.SnowflakeSource(
            captured, database=catalog, schema=schema
        )
    else:
        return sdm_connectors.PostgresSource(captured, schema=schema)
    view = given.get('view')
    if view is None:
        return source
    return source.semantic_model(
        view['name'], trust_expressions=view.get('trust_expressions', False)
    )


def _duck_tables(duck: Any) -> list[str]:
    r"""The tables of the case's DuckDB data, in name order: what the loader
    loads, and the capture reads back through the leg's driver."""
    return [
        str(row[0])
        for row in duck.execute(
            'SELECT table_name FROM information_schema.tables '
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
    ]


class Live:
    r"""One case's open warehouse leg: the connection, the private names,
    the case resolved, the tables as the warehouse holds them, and what
    closes it all."""

    def __init__(
        self, case: dict[str, Any], bases: dict[str, list[str]], leg: str
    ) -> None:
        self.leg = leg
        spec, self.names, self.private = live_configuration(leg)
        self.case = resolve_placeholders(case)
        given = self.case['input']
        duck = _connection(case, bases)
        self._undo: list[Any] = []
        try:
            self.connection, close = connect_leg(
                leg, spec, self.names, given.get('connect')
            )
            self._undo.append(close)
            # The connect path's session is not the driver connection the
            # loader writes through: Snowpark's is.
            writer = (
                self.connection.connection
                if given.get('connect') == 'snowpark'
                else self.connection
            )
            self.raw = writer
            loader = _sibling('databases')
            self.table_names = _duck_tables(duck)
            loader.load(
                duck,
                leg,
                WarehouseTarget(writer, self.names, given.get('view')),
            )
            self.tables = [
                (
                    name,
                    raw_arrow(
                        leg,
                        writer,
                        f'SELECT * FROM {_quoted(leg, self.names, name)}',
                    ),
                )
                for name in self.table_names
            ]
            zone = raw_arrow(leg, writer, _ZONE[leg]).to_pylist()
            self.zone = str(next(iter(zone[0].values()))) if zone else None
        except RunnerFailure:
            self.close()
            raise
        except Exception as error:  # noqa: BLE001 - opening is not the call
            self.close()
            raise RunnerFailure(
                f'{case["id"]}: {leg} could not be opened or loaded '
                f'({type(error).__name__})'
            ) from None
        finally:
            duck.close()

    def captured(self, log: DriverLog) -> Any:
        path = self.case['input'].get('connect')
        kind = 'snowpark' if path == 'snowpark' else self.leg
        return _WAREHOUSE_CAPTURES[kind](self.connection, log)

    def relations(self, log: DriverLog) -> list[dict[str, Any]]:
        r"""Each relation derived from a view that a statement read, as the
        warehouse evaluates it: the statement's text without its sample's
        ``LIMIT``, run outside the capture (a statement the warehouse
        refused gives none; its column is served by exact replay alone,
        B2)."""
        found = []
        seen: set[str] = set()
        for position, statement in enumerate(log.statements):
            sql = statement['sql']
            if (
                not sql
                or statement['error'] is not None
                or statement['parameters']
                or not _DERIVED.search(sql)
                or not _LIMIT.search(sql)
            ):
                continue
            whole = _LIMIT.sub('', sql)
            if whole in seen:
                continue
            seen.add(whole)
            try:
                table = raw_arrow(self.leg, self.raw, whole)
            except Exception as error:  # noqa: BLE001 - recorded, B2
                found.append(
                    {
                        'statement': position,
                        'sql': whole,
                        'error': type(error).__name__,
                    }
                )
                continue
            found.append({'statement': position, 'sql': whole, 'table': table})
        return found

    def close(self) -> None:
        while self._undo:
            try:
                self._undo.pop()()
            except Exception:  # noqa: BLE001, S110 - closing after the case
                pass


#: The core's SQL dialect each warehouse leg's provider renders (Phase 3
#: design choice 2: the stand-in receives each request with the plan the
#: leg's own dialect rendered for it).
LEG_DIALECTS = {
    'databricks-sql': 'databricks',
    'databricks-spark': 'databricks',
    'snowflake': 'snowflake',
    'lakebase': 'postgres',
}


class _Offline:
    r"""What a warehouse leg's source holds offline in place of a driver
    connection: nothing runs through it (the stand-in's forwarding provider
    answers every fetch), it only names the leg's driver to the source's
    dispatch, as the real one would."""

    version = '4.0.0'

    def cursor(self) -> Any:
        raise RunnerFailure('a stand-in leg runs nothing through its driver')

    def sql(self, *arguments: Any, **keywords: Any) -> Any:
        raise RunnerFailure('a stand-in leg runs nothing through its driver')

    def close(self) -> None:
        pass


#: Each leg's offline connection class, under the driver's package the
#: source dispatch reads (a Spark session by its class name).
_OFFLINE = {
    'databricks-sql': type(
        'Connection', (_Offline,), {'__module__': 'databricks.sql.sdm_standin'}
    ),
    'databricks-spark': type(
        'SparkSession', (_Offline,), {'__module__': 'pyspark.sql.sdm_standin'}
    ),
    'snowflake': type(
        'SnowflakeConnection',
        (_Offline,),
        {'__module__': 'snowflake.connector.sdm_standin'},
    ),
    'lakebase': type(
        'Connection', (_Offline,), {'__module__': 'psycopg.sdm_standin'}
    ),
}


class Standin:
    r"""A warehouse leg served offline by the stand-in (Phase 3 design
    choice 2; T3.3's ``sdm-standin``) for python-core, through
    ``standin_client.py``'s forwarding provider over the pipe protocol: the
    stand-in needs each whole fetch step (the request and the plan the
    leg's dialect rendered), so the session's provider is plugged in as the
    leg's provider (:meth:`plugged`), answering every fetch the core asks,
    and the source holds an offline connection that only names the leg's
    driver (:class:`_Offline`). The run keeps the stand-in's verdict for
    the observation's ``standin`` field.

    ``standin_client.py`` meets this: ``open(leg, case, bases, captures)``
    gives a session with ``provider(step, keys)`` (the host's provider
    interface, ``sdm_client._rust.run.Provider``), ``names`` (each
    placeholder's name offline), ``verdict()`` (``{"live_only": code or
    None, "flagged": [codes]}``) and ``close()``; the captures are
    ``contract/captures/``."""

    def __init__(
        self, case: dict[str, Any], bases: dict[str, list[str]], leg: str
    ) -> None:
        self.leg = leg
        if (
            'standin_client' not in sys.modules
            and not (HERE / 'standin_client.py').is_file()
        ):
            raise RunnerFailure(
                f'{case["id"]}: {leg} runs offline on the stand-in, and '
                'standin_client.py is not beside the runner'
            )
        client = _sibling('standin_client')
        self._session = client.open(leg, case, bases, CONTRACT / 'captures')
        self.provider = self._session.provider
        self.connection = _OFFLINE[leg]()
        self.names = dict(self._session.names)
        self.case = resolved(case, self.names)

    def captured(self, log: DriverLog) -> Any:
        r"""The offline connection: nothing runs through it, so there is
        no driver call to capture; the stand-in's statements are its own,
        and only reported (KO3-45)."""
        return self.connection

    @contextlib.contextmanager
    def plugged(self) -> Iterator[None]:
        r"""The session's forwarding provider as the provider of every run
        over this leg's offline connection, for the case's calls, restored
        after: the Python host's ``_provider`` of ``describe_source`` and
        ``materialize`` (test-only, as the capture is)."""
        from sdm_client._rust import materialize, schema

        choice = {'provider': 'sql', 'dialect': LEG_DIALECTS[self.leg]}
        saved = [(module, module._provider) for module in (schema, materialize)]

        def forwarding(original: Any) -> Any:
            def chosen(source: Any) -> tuple[dict[str, Any], Any]:
                held = getattr(source, 'connection', None)
                # A Spark source wraps the session it was given.
                if self.connection in (held, getattr(held, '_session', None)):
                    return dict(choice), self.provider
                return original(source)

            return chosen

        try:
            for module, original in saved:
                module._provider = forwarding(original)
            yield
        finally:
            for module, original in saved:
                module._provider = original

    def verdict(self) -> dict[str, Any]:
        found = self._session.verdict()
        return {
            'live_only': found.get('live_only'),
            'flagged': list(found.get('flagged') or []),
        }

    def close(self) -> None:
        self._session.close()


def capture_line(
    live: Live,
    case: dict[str, Any],
    bases: dict[str, list[str]],
    log: DriverLog,
    observation: dict[str, Any],
    relations: list[dict[str, Any]],
) -> dict[str, Any]:
    r"""One case's capture on its leg (design choice 2), before placeholders:
    the reference line, every statement with its parameters and each
    result whole, every table of the case as the warehouse returns it, each
    view-derived relation as the warehouse evaluates it, every random draw
    with the rows it picked, the session zone, the versions, and the
    case's input hash."""
    recorder = log.recorder
    statements = []
    draws = []
    for position, statement in enumerate(log.statements):
        results = []
        for read, result in enumerate(statement['results']):
            kept = recorder.kept.get((position, read)) if recorder else None
            if kept is None:
                table = None
            elif isinstance(kept, tuple):
                table = rows_table(*kept)
            else:
                table = kept
            results.append({**result, 'arrow': table})
        statements.append({**statement, 'results': results})
        for draw in draws_of(statement['sql']):
            draws.append({'statement': position, **draw})
    versions = dict(observation['identity']['versions'])
    for name in _DRIVERS[live.leg]:
        found = _version(name)
        if found is not None:
            versions[name] = found
    runtime = os.environ.get('DATABRICKS_RUNTIME_VERSION')
    if runtime:
        versions['runtime'] = runtime
    reference = observation['identity'].get('reference') or {}
    return {
        'format': RECORD_FORMAT,
        'case': observation['case'],
        'leg': live.leg,
        'input': input_hash(case, bases),
        'recording': {
            'commit': observation['identity']['commit']['sha'],
            'reference': (reference.get('commit') or {}).get('sha'),
        },
        'versions': versions,
        'zone': live.zone,
        'observation': observation,
        'statements': statements,
        'tables': [
            {'name': name, 'arrow': table} for name, table in live.tables
        ],
        'relations': relations,
        'draws': draws,
        'incomplete': sorted(set(log.incomplete)),
    }


def written(line: dict[str, Any], held: Placeheld) -> str:
    r"""A capture line as the text the recording mode writes: every private
    name placeheld, every Arrow table as IPC text, and the write refused
    (:class:`RunnerFailure`, naming the case, never the value) when any
    private value is left anywhere in it, a table's values and names
    included (each checked before it is encoded, so an encoding's bytes
    never pass for a name)."""
    tables: list[Any] = []

    def hold(value: Any) -> Any:
        if hasattr(value, 'schema') and hasattr(value, 'num_rows'):
            try:
                table = held.table(value)
            except Exception as error:  # noqa: BLE001 - named, never written
                raise RunnerFailure(
                    f'{line["case"]}: a table of its capture cannot be '
                    f'placeheld ({type(error).__name__}); nothing is written'
                ) from None
            if held.leaked(
                json.dumps(
                    [table.schema.names, table.to_pylist()],
                    default=str,
                    ensure_ascii=False,
                )
            ):
                raise RunnerFailure(
                    f'{line["case"]}: a table of its capture holds a private '
                    'name; nothing is written'
                )
            tables.append(table)
            return {'\0arrow': len(tables) - 1}
        if isinstance(value, dict):
            return {
                held.text(str(key)): hold(item) for key, item in value.items()
            }
        if isinstance(value, list | tuple):
            return [hold(item) for item in value]
        if isinstance(value, str):
            return held.text(value)
        return value

    held_line = hold(line)
    if held.leaked(json.dumps(held_line, sort_keys=True, ensure_ascii=False)):
        raise RunnerFailure(
            f'{line["case"]}: its capture holds a private name; nothing is '
            'written'
        )

    def encode(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {'\0arrow'}:
                return arrow_ipc(tables[value['\0arrow']])
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [encode(item) for item in value]
        return value

    return json.dumps(encode(held_line), sort_keys=True, ensure_ascii=False)


# The Postgres leg's capture is psycopg's whole: copy, stream and
# executemany recorded, never passed through unseen (P2-T2.1a-2).
_CAPTURES['postgres'] = _CapturedPsycopg
# Each warehouse leg's capture (Phase 3, choice 2), as capture() hands it.
_CAPTURES.update(
    {
        'databricks-sql': _CapturedDatabricks,
        'databricks-spark': _CapturedSpark,
        'snowflake': _CapturedSnowflake,
        'lakebase': _CapturedPsycopg,
    }
)


# --- one case --------------------------------------------------------------------


def _connection(case: dict[str, Any], bases: dict[str, list[str]]) -> Any:
    import duckdb

    connection = duckdb.connect()
    given = case['input']
    if given['base'] not in bases:
        raise RunnerFailure(f'{case["id"]}: unknown base')
    for statement in [*bases[given['base']], *given.get('setup', [])]:
        try:
            connection.execute(statement)
        except Exception:  # noqa: BLE001 - building the input is not the call
            raise RunnerFailure(
                f'{case["id"]}: its data could not be built'
            ) from None
    return connection


def _open(
    case: dict[str, Any], bases: dict[str, list[str]], database: str
) -> tuple[Any, Any, dict[int, str]]:
    r"""The case's data on ``database``, and what removes it after: on
    DuckDB the in-memory connection its base and setup build; elsewhere
    that connection's tables loaded by the contract loader
    (``databases.py``) into a new SQLite file or the public schema of a new
    database on ``SDM_POSTGRES_TEST_DSN``'s server, and a driver connection
    to it opened
    as a user opens one (``adbc_driver_sqlite.dbapi.connect``,
    ``psycopg.connect``). Setup is everything before it reaches the source.
    The third item names each type the server numbered (Postgres's
    composite types and their arrays), read on a connection of the loader's
    own, outside the capture."""
    duck = _connection(case, bases)
    types: dict[int, str] = {}
    if database == 'duckdb':
        return duck, duck.close, types
    loader = _sibling('databases')
    undo: list[Any] = []

    def cleanup() -> None:
        while undo:
            undo.pop()()

    try:
        if database == 'sqlite':
            import tempfile

            import adbc_driver_sqlite.dbapi

            directory = tempfile.TemporaryDirectory(prefix='sdm-case-')
            undo.append(directory.cleanup)
            path = Path(directory.name) / 'case.sqlite'
            loader.load(duck, 'sqlite', path)
            connection = adbc_driver_sqlite.dbapi.connect(str(path))
        else:
            import uuid

            import psycopg
            from psycopg.conninfo import make_conninfo

            dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
            if not dsn:
                raise RunnerFailure(
                    'a Postgres run needs SDM_POSTGRES_TEST_DSN, its server'
                )
            # A database of its own, so the tables are in its public schema
            # as a user's are, and what a source reads of its catalog
            # (current_schema()) is the same on every run.
            name = f'sdm_case_{uuid.uuid4().hex[:16]}'
            with psycopg.connect(dsn, autocommit=True) as admin:
                admin.execute(f'CREATE DATABASE {name}')

            def drop() -> None:
                with psycopg.connect(dsn, autocommit=True) as admin:
                    admin.execute(f'DROP DATABASE {name} WITH (FORCE)')

            undo.append(drop)
            scoped = make_conninfo(dsn, dbname=name)
            loader.load(duck, 'postgres', scoped)
            with psycopg.connect(scoped) as side:
                types = {
                    int(oid): str(typname)
                    for oid, typname in side.execute(
                        'SELECT t.oid, t.typname FROM pg_type t '
                        'JOIN pg_namespace n ON n.oid = t.typnamespace '
                        "WHERE n.nspname = 'public'"
                    ).fetchall()
                }
            connection = psycopg.connect(scoped)
        undo.append(connection.close)
    except RunnerFailure:
        cleanup()
        raise
    except Exception as error:  # noqa: BLE001 - loading is not the call
        cleanup()
        raise RunnerFailure(
            f'{case["id"]}: its data could not be loaded on {database} '
            f'({type(error).__name__}'
            + (f': {error}' if isinstance(error, loader.Unloadable) else '')
            + ')'
        ) from None
    finally:
        duck.close()
    return connection, cleanup, types


def _source(database: str, connection: Any) -> Any:
    r"""The public source a user builds over ``connection``."""
    import sdm_connectors

    return {
        'duckdb': sdm_connectors.DuckDBSource,
        'sqlite': sdm_connectors.SQLiteSource,
        'postgres': sdm_connectors.PostgresSource,
    }[database](connection)


def _materialize_arguments(spec: dict[str, Any]) -> dict[str, Any]:
    from decimal import Decimal

    import pandas as pd

    indices = spec['indices']
    if spec.get('indices_as') == 'decimal':
        indices = [Decimal(value) for value in indices]
    arguments: dict[str, Any] = {'indices': list(indices)}
    # An anchor time is a timestamp, or the literal 'entity' (each entity
    # at its own time; T2.11's cases), as the public call takes it.
    for key in ('anchor_time', 'context_anchor_time'):
        if key in spec:
            arguments[key] = (
                'entity' if spec[key] == 'entity' else pd.Timestamp(spec[key])
            )
    if 'inference_config' in spec:
        arguments['inference_config'] = copy.deepcopy(spec['inference_config'])
    if 'candidates' in spec:
        arguments['candidates'] = list(spec['candidates'])
    if 'top_k' in spec:
        arguments['top_k'] = spec['top_k']
    for key in (
        'rows_per_call',
        'lag_timesteps',
        # The quickstart's context size (Phase 3's S2 cases).
        'num_context_samples',
        *SAMPLING_OPTIONS,
    ):
        if key in spec:
            # As written (T2.8c's, T2.9's and T2.11's cases): a value the
            # reference refuses is the case's input, and a null is None
            # (random_seed=None, an unseeded sample).
            arguments[key] = copy.deepcopy(spec[key])
    return arguments


#: The sampling options a case passes to materialize as written, any JSON
#: value, null as None (T2.11).
SAMPLING_OPTIONS = (
    'random_seed',
    'num_hops',
    'num_neighbors',
    'max_pq_iterations',
    'use_prediction_time',
    'return_embeddings',
    'exclude_cols_dict',
)


def _from_source_arguments(spec: dict[str, Any]) -> dict[str, Any]:
    r"""``Schema.from_source``'s keyword arguments, each relationship as a
    ``Relationship``."""
    from sdm_client import Relationship

    arguments = copy.deepcopy(spec)
    if 'relationships' in arguments:
        arguments['relationships'] = [
            Relationship(**link) for link in arguments['relationships']
        ]
    return arguments


def _edit(schema: Any, edits: list[dict[str, Any]]) -> None:
    r"""The case's edits to the schema, in order (T1.21b; from T1.24b a
    key, a time column, a meaning and the relationships set)."""
    from sdm_client import Relationship

    for edit in edits:
        op = edit['op']
        if op == 'clear_time_column':
            schema[edit['table']].time_column = None
        elif op == 'delete_column':
            del schema[edit['table']][edit['column']]
        elif op == 'set_primary_key':
            schema[edit['table']].primary_key = edit['column']
        elif op == 'set_time_column':
            schema[edit['table']].time_column = edit['column']
        elif op == 'set_stype':
            schema[edit['table']][edit['column']].stype = edit['stype']
        else:
            schema.relationships = [
                Relationship(**link) for link in edit['relationships']
            ]


def presented(schema: Any) -> list[str]:
    r"""Each way a schema the core described departs from the description
    it was built from (T1.21d): a table's key or time column, a column's
    meaning or what it meant before the schema joined on it, or the links,
    as the Python object holds them once built. Building it re-runs the
    object's own key marking and link checks, which must change nothing of
    the core's answer; empty for a schema not described on the core."""
    described = getattr(schema, '_core_description', None)
    if described is None:
        return []
    found = []
    held = schema.tables
    if [t['name'] for t in described['tables']] != list(held):
        return ["the tables are not the description's"]
    for table in described['tables']:
        shaped = held[table['name']]
        key = shaped.primary_key
        keys = (
            [] if key is None else [key] if isinstance(key, str) else list(key)
        )
        if keys != table['primary_key']:
            found.append(f'{table["name"]}: the key')
        if shaped.time_column != table['time_column']:
            found.append(f'{table["name"]}: the time column')
        if [c['name'] for c in table['columns']] != list(shaped.columns):
            found.append(f'{table["name"]}: the columns')
            continue
        for column in table['columns']:
            built = shaped.columns[column['name']]
            if built.stype != column['stype'] or (
                built._stype_before_key != column['stype_before_key']
            ):
                found.append(f'{table["name"]}.{column["name"]}: its meaning')
    links = [
        [
            {'table': link.source_table, 'column': link.source_column},
            {'table': link.target_table, 'column': link.target_column},
        ]
        for link in schema.relationships
    ]
    if links != described['links']:
        found.append('the links')
    return found


def _require_presented(schema: Any) -> None:
    found = presented(schema)
    if found:
        raise RunnerFailure(
            "the Schema built from the core's description departs from it: "
            + '; '.join(found)
        )


def _route_error(error: BaseException) -> bool:
    kind = type(error)
    return kind.__module__ == 'sdm_client._routing' and kind.__name__ == (
        'RouteError'
    )


def _outcome(
    result: str, stage: str, error: BaseException | None = None
) -> dict[str, Any]:
    code = getattr(error, 'code', None)
    return {
        'result': result,
        'stage': stage,
        'error': None if error is None else type(error).__name__,
        'code': code if isinstance(code, str) else None,
    }


def _number(token: str) -> dict[str, str]:
    return {'number': token}


def _payload(body: bytes) -> dict[str, Any]:
    r"""A request body the endpoint received: its SHA-256, and its parsed
    form with every number as the decimal text sent, so no value passes
    through a float."""
    try:
        parsed = json.loads(body, parse_int=_number, parse_float=_number)
    except ValueError:
        parsed = None
    return {'sha256': digest(body), 'parsed': parsed}


def _schema_projection(schema: Any) -> dict[str, Any]:
    r"""The public ``Schema``: each table's columns with their dtype and
    stype, its key, time column and duplicate-key count, and its
    relationships, each as source table and column, then target table and
    column."""
    tables = []
    for name, table in schema.tables.items():
        key = table.primary_key
        tables.append(
            {
                'name': name,
                'columns': [
                    {'name': column, 'dtype': c.dtype, 'stype': c.stype}
                    for column, c in table.columns.items()
                ],
                'primary_key': (
                    key if key is None or isinstance(key, str) else list(key)
                ),
                'time_column': table.time_column,
                'duplicate_keys': table.duplicate_keys,
            }
        )
    links = [
        [
            link.source_table,
            link.source_column,
            link.target_table,
            link.target_column,
        ]
        for link in schema.relationships
    ]
    return {'tables': tables, 'links': links}


def _result_projection(result: Any) -> dict[str, Any]:
    r"""The public result, through ``to_pandas()``: its columns with their
    dtypes, and its rows as typed values with their digest."""
    frame = result.to_pandas()
    rows = [list(row) for row in frame.itertuples(index=False, name=None)]
    return {
        'columns': [
            {'name': str(name), 'dtype': str(dtype)}
            for name, dtype in frame.dtypes.items()
        ],
        'rows': len(rows),
        'digest': _rows_digest(rows),
        # Each value as its text (P2-T2.21-1): a result whose digest an
        # entry excuses for a value's type is still held to its values.
        'text_digest': digest(
            json.dumps([[str(value) for value in row] for row in rows]).encode()
        ),
    }


def _call(
    case: dict[str, Any],
    bases: dict[str, list[str]],
    driver_capture: bool = True,
    database: str = 'duckdb',
    keep: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str], dict[str, Any], dict[str, Any]]:
    r"""Makes the case's public calls on ``database`` (a database or a
    warehouse leg), with the driver capture on or off: its outcome, warning
    classes, driver record and public projection. On a warehouse leg the
    case's placeholders are resolved from the live configuration
    (:class:`Live`); with ``keep``, the recording mode's, every result is
    kept whole and ``keep`` gets the leg, the capture's log and the
    view-derived relations, read before the leg closes."""
    offline = database in WAREHOUSES and keep is not None and 'standin' in keep
    live: Live | Standin | None = None
    if offline:
        live = Standin(case, bases, database)
    elif database in WAREHOUSES:
        live = Live(case, bases, database)
    given = (live.case if live is not None else case)['input']
    # The call's arguments are the harness's to build: an input it cannot
    # build is a runner failure, never the case's outcome.
    try:
        arguments = _materialize_arguments(given['materialize'])
        from_source = _from_source_arguments(given.get('from_source', {}))
    except Exception:  # noqa: BLE001 - building the input is not the call
        if live is not None:
            live.close()
        raise RunnerFailure(
            f'{case["id"]}: its materialize arguments could not be built'
        ) from None
    if live is not None:
        connection, cleanup, types = live.connection, live.close, {}
    else:
        connection, cleanup, types = _open(case, bases, database)
    # Setup is everything run before the connection reaches the source; the
    # query log (DuckDB's own, or off DuckDB the driver's statement log,
    # entered only in a with, so it is always left, P2-T2.1a-2; the
    # stand-in's leg keeps none, its statements being the stand-in's) and
    # the capture start after it.
    logs = database != 'duckdb' and not isinstance(live, Standin)
    try:
        if database == 'duckdb':
            enable_query_log(connection)
        log = DriverLog() if driver_capture else None
        if log is not None:
            log.type_names = types
            if keep is not None:
                log.recorder = Recorder()
        if live is not None:
            given_to_source = (
                live.captured(log) if log is not None else connection
            )
        else:
            given_to_source = (
                capture(connection, log, database)
                if log is not None
                else connection
            )
    except BaseException:
        cleanup()
        raise
    public: dict[str, Any] = {'schema': None, 'payloads': [], 'result': None}
    if log is not None:
        log.stage = 'from_source'

    def calls(statements: StatementLog | None) -> tuple[Any, ...]:
        return _calls(
            case,
            database,
            given,
            arguments,
            from_source,
            given_to_source,
            live,
            log,
            keep,
            connection,
            cleanup,
            public,
            statements,
        )

    if logs:
        with StatementLog(database) as statements:
            outcome, logged, shown = calls(statements)
    else:
        outcome, logged, shown = calls(None)
    compared, differing = _logged_texts(log, logged)
    driver = {
        'capture': log.record() if log is not None else None,
        # DuckDB's own query log, or off DuckDB the driver's statement log.
        'query_log': {
            'statements': sum(1 for text in logged if text),
            'bindings': sum(1 for text in logged if not text),
            'compared': compared,
            'differing': differing,
        },
    }
    return (
        outcome,
        [type(warning.message).__name__ for warning in shown],
        driver,
        public,
        [
            {
                'class': type(warning.message).__name__,
                'module': _origin(warning.filename),
            }
            for warning in shown
        ],
    )


def _calls(
    case: dict[str, Any],
    database: str,
    given: dict[str, Any],
    arguments: dict[str, Any],
    from_source: dict[str, Any],
    given_to_source: Any,
    live: Any,
    log: DriverLog | None,
    keep: dict[str, Any] | None,
    connection: Any,
    cleanup: Any,
    public: dict[str, Any],
    statements: StatementLog | None,
) -> tuple[Any, ...]:
    r"""The case's public calls (:func:`_call`'s), under the statement log
    when there is one: the outcome, the statements logged and the
    warnings shown."""
    from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient

    serve = _sibling('scripted_endpoint').serve
    stage = 'from_source'
    # On a stand-in leg, the stand-in's forwarding provider answers every
    # fetch of the case's calls (Standin.plugged).
    plugged = (
        live.plugged()
        if isinstance(live, Standin)
        else contextlib.nullcontext()
    )
    with (
        plugged,
        # The scripted endpoint, failing the case's chosen call and
        # answering a classification with its probabilities (T2.8's ranking
        # cases), a regression with its quantiles (T2.9's forecast cases),
        # a call asking for an explanation with one and each item with its
        # metadata (T2.13's cases), and a call asking for embeddings with
        # one on each item (T2.11's cases) where the case says so.
        serve(
            fail_call=given.get('fail_call'),
            probabilities=given.get('probabilities', False),
            quantiles=given.get('quantiles', False),
            explanation=given.get('explanation', False),
            item_metadata=given.get('item_metadata', False),
            embeddings=given.get('embeddings', False),
        ) as (url, received),
        warnings.catch_warnings(record=True) as shown,
    ):
        warnings.simplefilter('always')
        try:
            source = (
                warehouse_source(database, given_to_source, live.names, given)
                if live is not None
                else _source(database, given_to_source)
            )
            schema = Schema.from_source(source, **from_source)
            _require_presented(schema)
            public['schema'] = _schema_projection(schema)
            # The user's edits finish describing the schema, so a refusal
            # of one is the from_source stage's.
            _edit(schema, given.get('edits', []))
            if case['until'] != 'from_source':
                stage = 'materialize'
                if log is not None:
                    log.stage = stage
                request = PredictiveQuery(given['pql'], schema).materialize(
                    **arguments
                )
            if case['until'] == 'predict':
                stage = 'predict'
                if log is not None:
                    log.stage = stage
                # The output fields the case names, as written (T2.9's
                # quantile cases); none named, the model's defaults.
                outputs = (
                    {'outputs': list(given['outputs'])}
                    if 'outputs' in given
                    else {}
                )
                with SDMClient(url) as client:
                    result = client.predict(
                        request, model=KumoRelational(), **outputs
                    )
                public['result'] = _result_projection(result)
            outcome = _outcome('success', stage)
        except RunnerFailure:
            raise
        except Exception as error:  # noqa: BLE001 - the call's outcome
            if _route_error(error):
                raise RunnerFailure(
                    f'{case["id"]}: the route was refused '
                    f'({type(error).__name__})'
                ) from None
            outcome = _outcome('refusal', stage, error)
        finally:
            try:
                if statements is not None:
                    logged = list(statements.texts)
                elif live is not None:
                    logged = []
                else:
                    logged = query_log(connection)
                if isinstance(live, Standin):
                    keep['standin'] = live.verdict()
                elif keep is not None and live is not None and log is not None:
                    keep.update(
                        live=live, log=log, relations=live.relations(log)
                    )
            finally:
                cleanup()
        public['payloads'] = [
            {'method': method, 'path': path, **_payload(body)}
            for method, path, body in received.requests
        ]
    return outcome, logged, shown


def _origin(filename: str) -> str | None:
    r"""The top-level module whose files hold ``filename``, the file a
    warning names as raising it: the imported package or module whose
    directory (or file) is the longest prefix of it, or None when none is.
    Read from the modules loaded, never from the warning's text, so a
    comparison can tell the runtime's own warnings (pandas', pyarrow's)
    from the client's (decision E81, register HOST-5)."""
    try:
        target = os.path.realpath(filename)
    except (TypeError, ValueError):
        return None
    found, length = None, -1
    for name, module in list(sys.modules.items()):
        if '.' in name or module is None:
            continue
        where = getattr(module, '__file__', None)
        if not isinstance(where, str):
            continue
        where = os.path.realpath(where)
        if os.path.basename(where).startswith('__init__.'):
            root = os.path.dirname(where) + os.sep
            hit = target.startswith(root)
        else:
            root = where
            hit = target == root
        if hit and len(root) > length:
            found, length = name, len(root)
    return found


# --- the run ---------------------------------------------------------------------


#: Where the checkout declares its contract version.
CONTRACT_SOURCE = REPO / 'rust/crates/sdm-contract/src/lib.rs'


def contract_refusal(installed: Any, root: Path = REPO) -> str | None:
    r"""Why the installed client may not run the cases of the checkout at
    ``root`` (P2-MID2-7): its contract version is not the one
    ``rust/crates/sdm-contract`` declares there, naming both; or None."""
    return stale_client(installed, root / 'rust/crates/sdm-contract/src/lib.rs')


def stale_client(installed: Any, source: Path = CONTRACT_SOURCE) -> str | None:
    r"""Why the installed client may not run this checkout's cases
    (P2-MID2-7): its core's contract version is not the one the checkout
    declares, so it is a wheel built from another tree; or None. A
    checkout that declares none (a tree without the core) is not held."""
    try:
        text = source.read_text(encoding='utf-8')
    except OSError:
        return None
    declared = re.findall(
        r'^pub const CONTRACT_VERSION: u32 = (\d+);$', text, re.M
    )
    if len(declared) != 1:
        return (
            f'{source.name} declares {len(declared)} contract versions, not one'
        )
    if installed != int(declared[0]):
        return (
            f"the installed client's contract version {installed} is not the "
            f"checkout's {declared[0]}: a stale wheel; rebuild the lane's "
            'environment (scripts/rust_core/cells.sh)'
        )
    return None


def _native_sha256() -> str | None:
    module = sys.modules.get('sdm_client._native')
    location = getattr(module, '__file__', None)
    if module is None or location is None:
        return None
    return digest(Path(location).read_bytes())


def _require_reference_client(reference: Path) -> None:
    r"""``sdm_client`` is the reference checkout's own."""
    import sdm_client

    location = Path(sdm_client.__file__).resolve()
    if not location.is_relative_to(reference.resolve()):
        raise RunnerFailure("sdm_client is not the reference checkout's")


def run(
    mode: str,
    cases: list[tuple[dict[str, Any], str]],
    reference: Path | None,
    driver_capture: bool = True,
    database: str = 'duckdb',
    record_fetches: Path | None = None,
    record: Path | None = None,
) -> list[dict[str, Any]]:
    r"""Every case's observation on ``database``, keyed by :func:`case_key`,
    or :class:`RunnerFailure`. A case that does not run on ``database``
    (its legs, :func:`case_legs`) is left out; python-core runs on SQLite or
    Postgres only when the core reports a provider for it
    (``sdm_client._native.PROVIDERS``), as it does for DuckDB from the
    start.

    On a warehouse leg (Phase 3, choice 2) python-ref runs the old client
    live, on the leg the live configuration opens, and every observation
    is written with each private name as its placeholder, the run failing
    on any private value left (:class:`Placeheld`). ``record``, the
    recording mode, writes each case's capture there as well, one
    ``sdm-capture/1`` line per case (:func:`capture_line`), all or
    nothing."""
    if record is not None and (
        mode != 'python-ref' or database not in WAREHOUSES or not driver_capture
    ):
        raise RunnerFailure(
            'the recording mode runs python-ref on a warehouse leg, with the '
            'driver capture'
        )
    if database in WAREHOUSES and mode == 'python-old':
        raise RunnerFailure(
            f'python-old runs on DuckDB only; the {database} reference side '
            "is the capture's reference line"
        )
    if mode == 'python-old' and database != 'duckdb':
        # The orchestrator's decision (T2.1a fix 4): off DuckDB the
        # reference side is python-ref's committed lines until the core's
        # providers land; the old route, deleted in Phase 5, is not run
        # there.
        raise RunnerFailure(
            f'python-old runs on DuckDB only; the {database} reference side '
            'is python-ref, its lines committed'
        )
    cases = [pair for pair in cases if database in case_legs(pair[0])]
    if not cases:
        raise RunnerFailure(f'no case runs on {database}')
    os.environ.pop('SDM_CORE_ROUTES', None)
    guard = _Guard(reference) if reference is not None else None
    if guard is not None:
        guard.check('before the first case')
        _require_reference_client(reference)  # type: ignore[arg-type]
    bases_path = BASES
    bases = load_bases(bases_path)
    bases_digest = digest(bases_path.read_bytes())
    boundary = _Boundary() if mode == 'python-core' else None
    versions = _versions()
    if 'duckdb' not in versions:
        raise RunnerFailure('duckdb is not installed')
    if mode == 'python-core':
        from sdm_client import _native

        refused = stale_client(getattr(_native, 'CONTRACT_VERSION', None))
        if refused is not None:
            raise RunnerFailure(refused)
    if mode == 'python-core' and database != 'duckdb':
        from sdm_client import _native

        if database not in _native.PROVIDERS:
            raise RunnerFailure(
                f'the core reports no {database} provider: the {database} '
                'job runs the reference side only'
            )
    recorded = []
    kept: list[dict[str, Any]] = []
    standins: list[dict[str, Any] | None] = []
    for case, case_digest in cases:
        capture = boundary.reset() if boundary is not None else None
        if capture is not None:
            capture.recording = record_fetches is not None
        if mode == 'python-core':
            os.environ['SDM_CORE_ROUTES'] = ','.join(routes_on(case, database))
        keep: dict[str, Any] | None = {} if record is not None else None
        if mode == 'python-core' and database in WAREHOUSES:
            # Offline on the stand-in (choice 2): its verdict is kept.
            keep = {'standin': None}
        try:
            outcome, shown, driver, public, origins = _call(
                case, bases, driver_capture, database, keep
            )
        finally:
            os.environ.pop('SDM_CORE_ROUTES', None)
        if keep is not None and record is not None:
            kept.append(keep)
        standins.append(None if keep is None else keep.get('standin'))
        recorded.append(
            (
                case,
                case_digest,
                outcome,
                shown,
                capture,
                'sdm_client._native' in sys.modules,
                driver,
                public,
                origins,
            )
        )
    reference_identity = None
    if guard is not None and reference is not None:
        guard.check('after the last case')
        reference_identity = {
            'commit': _git(reference),
            'environment': _environment(),
            'guard': {'before': 'passed', 'after': 'passed'},
        }
    native_sha256 = _native_sha256() if mode == 'python-core' else None
    contract = None
    if mode == 'python-core' and 'sdm_client._native' in sys.modules:
        contract = sys.modules['sdm_client._native'].CONTRACT_VERSION
    commit = _git(REPO)
    validate = validator('observation')
    observations = []
    for (
        case,
        case_digest,
        outcome,
        shown,
        capture,
        loaded,
        driver,
        public,
        origins,
    ), standin in zip(recorded, standins, strict=True):
        observation = {
            'format': 'sdm-observation/1',
            'case': case_key(case['id'], database),
            'identity': {
                'runner': mode,
                'cell': _cell(versions),
                'platform': f'{sys.platform}-{platform.machine()}',
                'commit': commit,
                'reference': reference_identity,
                'native': (
                    {'sha256': native_sha256} if native_sha256 else None
                ),
                'native_loaded': loaded,
                'operations': capture.operations if capture else [],
                'contract': contract,
                'versions': versions,
                'inputs': {'case': case_digest, 'bases': bases_digest},
            },
            'outcome': outcome,
            # Each warning's origin on the candidate's runners, which a
            # cell other than the reference's compares (HOST-5); never on
            # python-ref, whose committed lines it would change.
            'host': (
                {'warnings': shown}
                if mode == 'python-ref'
                else {'warnings': shown, 'origins': origins}
            ),
            'core': capture.core() if capture else None,
            'driver': driver,
            'public': public,
        }
        if standin is not None:
            # The stand-in's verdict on a warehouse leg judged offline.
            observation['standin'] = standin
        problem = _invalid(validate, observation)
        if problem is not None:
            raise RunnerFailure(
                f'{case["id"]}: its observation is not valid: {problem}'
            )
        observations.append(observation)
    if database in WAREHOUSES and mode == 'python-ref':
        observations = _held_observations(
            observations, cases, bases, kept, record
        )
    if record_fetches is not None:
        # The replay's recording (T2.3, KO2-20): each case's fetches, with
        # the answers the Postgres provider replied with, in order, and the
        # statements the driver ran for each, so the replay's record of what
        # ran is python-core's.
        record_fetches.write_text(
            ''.join(
                json.dumps(
                    {
                        'format': 'sdm-replay/1',
                        'case': case_key(case['id'], database),
                        'fetches': _with_statements(case, capture, driver),
                    },
                    sort_keys=True,
                )
                + '\n'
                for case, _, _, _, capture, _, driver, *_ in recorded
                if capture is not None and capture.answered
            ),
            encoding='utf-8',
        )
    return observations


def _held_observations(
    observations: list[dict[str, Any]],
    cases: list[tuple[dict[str, Any], str]],
    bases: dict[str, list[str]],
    kept: list[dict[str, Any]],
    record: Path | None,
) -> list[dict[str, Any]]:
    r"""A warehouse leg's observations with every private name as its
    placeholder, and with ``record`` each case's capture written there,
    one line each, all or nothing; a private value left in either fails
    the run before anything is written (design choice 2)."""
    spec_names: dict[str, str] = {}
    private: list[str] = []
    if kept:
        spec_names, private = kept[0]['live'].names, kept[0]['live'].private
    else:
        _, spec_names, private = live_configuration(
            observations[0]['case'].rpartition('@')[2]
        )
    held = Placeheld(spec_names, private)
    out = []
    for observation in observations:
        placed = held.value(observation)
        if held.leaked(json.dumps(placed, sort_keys=True, ensure_ascii=False)):
            raise RunnerFailure(
                f'{observation["case"]}: its observation holds a private name; '
                'nothing is written'
            )
        out.append(placed)
    if record is not None:
        lines = [
            written(
                capture_line(
                    keep['live'],
                    case,
                    bases,
                    keep['log'],
                    observation,
                    keep['relations'],
                ),
                held,
            )
            + '\n'
            for (case, _), keep, observation in zip(
                cases, kept, observations, strict=True
            )
        ]
        partial = record.with_name(record.name + '.partial')
        partial.write_text(''.join(lines), encoding='utf-8')
        partial.replace(record)
    return out


def _with_statements(
    case: dict[str, Any], capture: Any, driver: dict[str, Any] | None
) -> list[dict[str, Any]]:
    r"""Each fetch the core asked ``case``, with its answer and the text of
    each statement the driver ran for it (``statements``), in order: the
    driver capture's first statements, one per statement of each fetch's
    plan (the core's calls come first; a call the case leaves on the old
    route runs after them); a capture holding fewer is a runner failure."""
    ran = [
        statement['sql']
        for statement in ((driver or {}).get('capture') or {}).get(
            'statements', []
        )
    ]
    planned = sum(
        len(fetch['plan'].get('statements', [])) for fetch in capture.answered
    )
    if len(ran) < planned:
        raise RunnerFailure(
            f'{case["id"]}: the driver ran {len(ran)} statements for '
            f'{planned} planned, so the recording cannot name them'
        )
    fetches = []
    for fetch in capture.answered:
        count = len(fetch['plan'].get('statements', []))
        fetches.append({**fetch, 'statements': ran[:count]})
        ran = ran[count:]
    return fetches


def legs_main(argv: list[str]) -> int:
    r"""``run_python.py legs --database D --providers FILE OBSERVATIONS...``:
    :func:`check_legs` over the observations files, ``FILE`` the JSON
    object of each core leg's providers; exit 1 naming each problem."""
    parser = argparse.ArgumentParser(prog='run_python.py legs')
    parser.add_argument('--database', choices=LEGS, required=True)
    parser.add_argument('--providers', type=Path, required=True)
    parser.add_argument('observations', type=Path, nargs='*')
    arguments = parser.parse_args(argv)
    try:
        providers = json.loads(arguments.providers.read_text(encoding='utf-8'))
        if not isinstance(providers, dict) or not all(
            isinstance(listed, list)
            and all(isinstance(name, str) for name in listed)
            for listed in providers.values()
        ):
            raise ValueError('not an object of lists of names')
        lines = [
            json.loads(text)
            for path in arguments.observations
            for text in path.read_text(encoding='utf-8').splitlines()
        ]
    except (OSError, ValueError, RecursionError) as error:
        print(f'legs: unreadable: {type(error).__name__}: {error}')
        return 1
    problems = check_legs(arguments.database, providers, lines)
    for problem in problems:
        print(f'FAILS legs: {problem}')
    print(f'legs: {"FAIL" if problems else "ok"} on {arguments.database}')
    return 1 if problems else 0


def environment_refusal(
    environ: Mapping[str, str], prefix: str = sys.prefix
) -> str | None:
    r"""Why this interpreter may not run the client's routes, or None (T3.0).

    On the orchestration machine (``SDM_ORCH_STATE`` set, outside CI), a
    checkout's own ``.venv`` is shared and holds whatever client was last
    installed there: Phase 2 read a stale one as an environment fault for
    twelve hours. A runner under it refuses, naming the lane's environment,
    which ``scripts/rust_core/cells.sh`` builds for the lane's tree. With
    ``SDM_LANE_ENV`` set (that environment's directory), any interpreter
    outside it refuses. CI builds its environment per run and is not held."""
    if environ.get('GITHUB_ACTIONS') == 'true':
        return None
    here = Path(prefix).resolve()
    lane = environ.get('SDM_LANE_ENV')
    right = (
        f'{lane}/current/cell-<cell>/bin/python (cells.sh prints each cell)'
        if lane
        else "your lane's environment: build it with scripts/rust_core/cells.sh "
        '$SDM_CELLS_ROOT/<lane>, set SDM_LANE_ENV to that directory, and run '
        'with the cell python it prints'
    )
    if lane:
        root = Path(lane).resolve()
        if here == root or root in here.parents:
            return None
        return (
            f'this runner runs under {prefix}, outside the lane environment '
            f'{lane} (SDM_LANE_ENV): run it with {right}'
        )
    if (
        environ.get('SDM_ORCH_STATE')
        and here.name == '.venv'
        and (here.parent / '.git').exists()
    ):
        return (
            f"this runner runs under {prefix}, a checkout's shared .venv, "
            'whose client is whatever was last installed there, not this '
            f"tree's: run it with {right}"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    given = sys.argv[1:] if argv is None else argv
    if given[:1] == ['legs']:
        return legs_main(given[1:])
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument(
        '--inventory',
        action='append',
        default=[],
        metavar='ID',
        help='a case of the Phase 1b inventory, by id, in the case format',
    )
    parser.add_argument(
        '--no-capture',
        action='store_true',
        help='hand the source the driver connection itself (the capture '
        "conservation control); DuckDB's query log is still read",
    )
    parser.add_argument(
        '--database',
        choices=LEGS,
        default='duckdb',
        help='the database or warehouse leg each case runs on (default '
        "duckdb): the others are loaded from the case's DuckDB tables "
        '(databases.py), Postgres in a new database on '
        "SDM_POSTGRES_TEST_DSN's server, a warehouse leg in the test schema "
        'the live configuration (SDM_LIVE_CONFIG) names',
    )
    parser.add_argument(
        '--record',
        type=Path,
        metavar='FILE',
        help='the recording mode (Phase 3, choice 2): python-ref on a '
        "warehouse leg writes each case's capture (sdm-capture/1) to FILE, "
        'every private name as its placeholder',
    )
    parser.add_argument(
        '--record-fetches',
        type=Path,
        metavar='FILE',
        help="python-core only: write each case's fetches with the answers "
        "the provider gave, the recording rust-host's Postgres replay reads "
        '(T2.3, KO2-20)',
    )
    parser.add_argument('cases', type=Path, nargs='*')
    arguments = parser.parse_args(argv)
    if arguments.mode != 'python-ref':
        refused = environment_refusal(os.environ)
        if refused is not None:
            print(f'runner failure: {refused}', file=sys.stderr)
            return 1
    if arguments.record_fetches is not None and arguments.mode != 'python-core':
        parser.error('--record-fetches is given in python-core mode only')
    if (arguments.mode == 'python-ref') != (arguments.reference is not None):
        parser.error('--reference is given in python-ref mode, and only there')
    # A failed run leaves no output, not an earlier run's.
    partial = arguments.out.with_name(arguments.out.name + '.partial')
    arguments.out.unlink(missing_ok=True)
    partial.unlink(missing_ok=True)
    if arguments.record is not None:
        arguments.record.unlink(missing_ok=True)
    try:
        cases = load_cases(arguments.cases) if arguments.cases else []
        cases.extend(inventory_cases(arguments.inventory))
        if not cases:
            raise RunnerFailure('no cases')
        observations = run(
            arguments.mode,
            cases,
            arguments.reference,
            driver_capture=not arguments.no_capture,
            database=arguments.database,
            record_fetches=arguments.record_fetches,
            record=arguments.record,
        )
    except RunnerFailure as failure:
        print(f'runner failure: {failure}', file=sys.stderr)
        return 1
    partial.write_text(
        ''.join(
            json.dumps(item, sort_keys=True) + '\n' for item in observations
        ),
        encoding='utf-8',
    )
    partial.replace(arguments.out)
    print(f'{arguments.mode}: {len(observations)} cases recorded')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
