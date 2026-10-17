#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The refusal census (Phase 2 design, choice 10; T2.0d's row, folded into
T2.1a): the core's refusal sites, generated from the code, each with a
probe, and each probe run on both routes.

    python -I contract/tools/refusal_census.py sites --root DIR
    python -I contract/tools/refusal_census.py check --root DIR
    python contract/tools/refusal_census.py run --reference DIR
        --reference-python PYTHON [--database duckdb] --out FILE

``sites`` prints the sites of DIR's code as JSON, never written by hand:

- every named rule, each ``not_yet("<rule>")`` in ``rust/crates/*/src``
  outside a file's ``#[cfg(test)]`` part, one site per rule (``rule:<rule>``;
  ``where`` its first call, ``also`` the others);
- every call of a rule-less refusal constructor: ``unsupported_type``
  (``sdm-contract/src/types.rs``, public, so a call in any crate),
  admission's ``unsupported`` (``sdm-host/src/admission.rs``) and predict's
  ``unsupported`` (``sdm-request/src/predict.rs``), each call its own site,
  named by its file, enclosing function, constructor and ordinal there
  (``<path>::<fn>::<constructor>#<n>``), its ``where`` the line;
- the Python host's own refusals: each ``raise StructuredDataError(...)``
  with a code other than ``INTERNAL_ERROR`` in the client's core bridge
  (``sdm_client/_rust``), named ``<path>::<fn>#<n>``.

Each site has ``id``, ``rule`` (or null), ``code`` (the error code it
fixes, or null) and ``where`` (``<path>:<line>``, relative to DIR).

``check`` fails, naming each, on a site of DIR's code with no probe in
DIR's probe list (:data:`PROBES`), and on a probe naming no site there: the
census step's site check, which the gate runs on the candidate.

``run`` runs every probe on both routes on ``--database`` (the first run
is DuckDB's; SQLite and Postgres are reported unmeasured until the core
runs there, never as zero hits): python-core in this interpreter, every
core route forced (``SDM_CORE_ROUTES``), and the reference with
``--reference-python``, whose ``sdm_client`` must be ``--reference``'s.
Each outcome is the public call's: the stage, the exception's class,
``StructuredDataError.code`` and its ``details``' ``rule``, or success.
Each probe's record names the site it was written for (``site``, the
intended site); the census claims nothing about which site produced an
outcome, since an outcome alone cannot prove it: reachability stays
unmeasured (``reachability``) until a real-data run or instrumentation
needs it (the orchestrator's decision at T2.1a fix 2). A probe *departs*
when python-core refuses where the reference answers. The real-data part (the public calls
over the phase's real datasets, each hit keyed by rule or by site and code
with its count of real sources) runs at each lane's first landing; until
it has, ``hits`` is empty and ``real_data`` says so.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
#: The probe list, relative to a tree's root.
PROBES = Path('contract/refusal-census.json')
CRATES = 'rust/crates/*/src/**/*.rs'
HOST = Path('packages/structured-data-models-client/src/sdm_client/_rust')
#: The rule-less refusal constructors: the file defining each, and the code
#: its error carries. ``unsupported_type`` is public, so its calls in any
#: crate are sites; the two private ones are called in their own file only.
RULELESS = {
    'unsupported_type': (
        'rust/crates/sdm-contract/src/types.rs',
        'UNSUPPORTED_TYPE',
    ),
    'admission::unsupported': (
        'rust/crates/sdm-host/src/admission.rs',
        'UNSUPPORTED_TYPE',
    ),
    'predict::unsupported': (
        'rust/crates/sdm-request/src/predict.rs',
        'UNSUPPORTED_REQUEST',
    ),
}
NOT_YET = re.compile(r'\bnot_yet\(\s*"([a-z0-9_]+)"')
FN = re.compile(r'\bfn\s+([A-Za-z_][A-Za-z0-9_]*)')
DATABASES = ('duckdb', 'sqlite', 'postgres')
#: The probes' case template: what the case format requires beside a
#: probe's input, its routes every core route the corpus forces.
TEMPLATE = {
    'format': 'sdm-case/1',
    'status': 'diagnostic',
    'slice': 'S1',
    'cells': [],
    'why': 'a refusal census probe',
    'source': 'contract/refusal-census.json',
}


class CensusError(Exception):
    r"""The census cannot read what it needs, named."""


# --- sites -----------------------------------------------------------------------


def _production(text: str) -> str:
    r"""A Rust source's text before its ``#[cfg(test)]`` part."""
    return text.split('#[cfg(test)]', 1)[0]


def _line(text: str, offset: int) -> int:
    return text.count('\n', 0, offset) + 1


def _function(lines: list[str], number: int) -> str:
    r"""The function a line of Rust sits in: the nearest ``fn`` at or
    before it."""
    for at in range(number - 1, -1, -1):
        found = FN.search(lines[at])
        if found:
            return found.group(1)
    return '(top)'


def _rust_sources(root: Path) -> list[Path]:
    return sorted(root.glob(CRATES))


def _named(root: Path) -> list[dict[str, Any]]:
    found: dict[str, list[str]] = {}
    for path in _rust_sources(root):
        relative = str(path.relative_to(root))
        text = _production(path.read_text(encoding='utf-8'))
        for match in NOT_YET.finditer(text):
            found.setdefault(match.group(1), []).append(
                f'{relative}:{_line(text, match.start())}'
            )
    return [
        {
            'id': f'rule:{rule}',
            'rule': rule,
            'code': 'UNSUPPORTED_REQUEST',
            'where': wheres[0],
            'also': wheres[1:],
        }
        for rule, wheres in sorted(found.items())
    ]


def _calls(root: Path) -> list[dict[str, Any]]:
    sites = []
    for path in _rust_sources(root):
        relative = str(path.relative_to(root))
        lines = _production(path.read_text(encoding='utf-8')).splitlines()
        counts: dict[tuple[str, str], int] = {}
        for number, line in enumerate(lines, 1):
            for name, (home, code) in RULELESS.items():
                short = name.rpartition('::')[2]
                if name != 'unsupported_type' and relative != home:
                    continue
                if not re.search(rf'\b{short}\(', line) or re.search(
                    rf'\bfn\s+{short}\(', line
                ):
                    continue
                function = _function(lines, number)
                counts[(function, name)] = counts.get((function, name), 0) + 1
                sites.append(
                    {
                        'id': f'{relative}::{function}::{name}'
                        f'#{counts[(function, name)]}',
                        'rule': None,
                        'code': code,
                        'where': f'{relative}:{number}',
                    }
                )
    return sites


def _raised_code(node: ast.Raise) -> str | None:
    r"""The code of a ``raise StructuredDataError(..., code='X')``, or None
    for any other raise."""
    call = node.exc
    if not isinstance(call, ast.Call):
        return None
    name = call.func
    named = (
        name.id if isinstance(name, ast.Name) else getattr(name, 'attr', None)
    )
    if named != 'StructuredDataError':
        return None
    for keyword in call.keywords:
        if keyword.arg == 'code' and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    return ''


def _host(root: Path) -> list[dict[str, Any]]:
    sites = []
    for path in sorted((root / HOST).rglob('*.py')):
        relative = str(path.relative_to(root))
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=relative)
        counts: dict[str, int] = {}

        def visit(
            node: ast.AST, function: str
        ) -> Iterator[tuple[str, ast.Raise]]:
            for child in ast.iter_child_nodes(node):
                inner = (
                    child.name
                    if isinstance(
                        child, (ast.FunctionDef, ast.AsyncFunctionDef)
                    )
                    else function
                )
                if isinstance(child, ast.Raise):
                    yield function, child
                yield from visit(child, inner)

        for function, node in visit(tree, '(module)'):
            code = _raised_code(node)
            if code is None or code == 'INTERNAL_ERROR':
                continue
            counts[function] = counts.get(function, 0) + 1
            sites.append(
                {
                    'id': f'{relative}::{function}#{counts[function]}',
                    'rule': None,
                    'code': code or None,
                    'where': f'{relative}:{node.lineno}',
                }
            )
    return sites


def sites(root: Path) -> list[dict[str, Any]]:
    r"""Every refusal site of ``root``'s code, ids unique."""
    found = [*_named(root), *_calls(root), *_host(root)]
    ids = [site['id'] for site in found]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise CensusError(f'site ids repeat: {repeated}')
    return found


# --- probes ------------------------------------------------------------------------


def probes(root: Path) -> list[dict[str, Any]]:
    r"""``root``'s probe list, each probe's input a case's ``input``."""
    path = root / PROBES
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, RecursionError) as error:
        raise CensusError(
            f'{PROBES}: not a probe list it reads ({type(error).__name__})'
        ) from None
    if (
        not isinstance(document, dict)
        or document.get('format') != 'sdm-refusal-census/1'
        or not isinstance(document.get('probes'), list)
    ):
        raise CensusError(f'{PROBES}: not sdm-refusal-census/1')
    listed: list[dict[str, Any]] = document['probes']
    for at, probe in enumerate(listed):
        if (
            not isinstance(probe, dict)
            or not isinstance(probe.get('site'), str)
            or not isinstance(probe.get('input'), dict)
            or probe.get('until')
            not in ('from_source', 'materialize', 'predict')
            or not isinstance(probe.get('options', {}), dict)
            or probe.get('model', 'KumoRelational')
            not in ('KumoRelational', 'KumoTabular')
            or not set(probe.get('databases', DATABASES)) <= set(DATABASES)
        ):
            raise CensusError(f'{PROBES}: probe {at} is not a probe')
    return listed


def check(root: Path) -> list[str]:
    r"""Each site of ``root``'s code with no probe, and each probe naming
    no site."""
    found = sites(root)
    listed = probes(root)
    probed = {probe['site'] for probe in listed}
    known = {site['id'] for site in found}
    return [
        f'{site["id"]} ({site["where"]}): a refusal site with no probe'
        for site in found
        if site['id'] not in probed
    ] + [
        f'{name}: a probe of no refusal site in the code'
        for name in sorted(probed - known)
    ]


# --- running probes ------------------------------------------------------------------


def _runner() -> ModuleType:
    name = 'run_python'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO / 'contract/runners/run_python.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _case(
    probe: dict[str, Any], number: int, routes: list[str]
) -> dict[str, Any]:
    return {
        **TEMPLATE,
        'id': f'census.{number}',
        'paths': routes[:1],
        'routes': routes,
        'input': probe['input'],
        'until': probe['until'],
        'expect': {'outcome': 'success', 'stage': probe['until']},
    }


def _routes() -> list[str]:
    runner = _runner()
    found: set[str] = set()
    for case, _ in runner.load_cases([REPO / 'contract/cases']):
        found.update(case['routes'])
    return sorted(found)


def _outcome(
    case: dict[str, Any],
    database: str,
    options: dict[str, Any] | None = None,
    model: str = 'KumoRelational',
) -> dict[str, Any]:
    r"""The public calls of one probe in this interpreter: its outcome,
    and a refusal's ``details`` rule. ``options`` are materialize's keyword
    arguments the case format does not carry (``num_hops``,
    ``anchor_time='entity'``), passed as written; ``model`` the class of
    the model predict is asked of."""
    import warnings

    import sdm_client
    from sdm_client import PredictiveQuery, Schema, SDMClient

    runner = _runner()
    serve = runner._sibling('scripted_endpoint').serve
    bases = runner.load_bases()
    given = case['input']
    arguments = runner._materialize_arguments(given['materialize'])
    arguments.update(options or {})
    from_source = runner._from_source_arguments(given.get('from_source', {}))
    connection, cleanup = runner._open(case, bases, database)
    stage = 'from_source'
    try:
        with serve() as (url, _), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(
                runner._source(database, connection), **from_source
            )
            runner._edit(schema, given.get('edits', []))
            if case['until'] != 'from_source':
                stage = 'materialize'
                request = PredictiveQuery(given['pql'], schema).materialize(
                    **arguments
                )
            if case['until'] == 'predict':
                stage = 'predict'
                with SDMClient(url) as client:
                    client.predict(request, model=getattr(sdm_client, model)())
        return {'result': 'success', 'stage': stage}
    except Exception as error:  # noqa: BLE001 - the probe's outcome
        details = getattr(error, 'details', None)
        details = details if isinstance(details, dict) else {}
        code = getattr(error, 'code', None)
        found = {
            'result': 'route_error'
            if runner._route_error(error)
            else 'refusal',
            'stage': stage,
            'error': type(error).__name__,
            'code': code if isinstance(code, str) else None,
        }
        # The structured fields that name what refused, and the names of
        # every field it carries, never a message or a value.
        for name in ('rule', 'field', 'operation'):
            value = details.get(name)
            found[name] = value if isinstance(value, str) else None
        found['detail_keys'] = sorted(str(key) for key in details)
        return found
    finally:
        cleanup()


def route(cases: Path, out: Path, database: str, core: bool) -> None:
    r"""Every probe case of ``cases`` on this interpreter's client, each
    outcome a line of ``out``; with ``core``, every core route forced."""
    lines = []
    for path in sorted(cases.glob('*.json'), key=lambda p: int(p.stem)):
        probe = json.loads(path.read_text(encoding='utf-8'))
        case = probe['case']
        if core:
            os.environ['SDM_CORE_ROUTES'] = ','.join(case['routes'])
        else:
            os.environ.pop('SDM_CORE_ROUTES', None)
        try:
            lines.append(
                _outcome(case, database, probe['options'], probe['model'])
            )
        finally:
            os.environ.pop('SDM_CORE_ROUTES', None)
    out.write_text(json.dumps(lines), encoding='utf-8')


def record(
    probe: dict[str, Any], mine: dict[str, Any], ref: dict[str, Any]
) -> dict[str, Any]:
    r"""One probe's evidence: the site it was written for, both routes'
    outcomes, and whether python-core refuses where the reference answers;
    never a claim that the outcome came from that site."""
    return {
        'site': probe['site'],
        'outcomes': {'python-core': mine, 'reference': ref},
        'departs': mine.get('result') != 'success'
        and ref.get('result') == 'success',
    }


def run(
    reference: Path, reference_python: str, database: str
) -> dict[str, Any]:
    r"""The census's evidence on ``database``: every probe on both routes."""
    runner = _runner()
    import sdm_client

    if Path(sdm_client.__file__).resolve().is_relative_to(reference.resolve()):
        raise CensusError("this interpreter's client is the reference's")
    found = {site['id']: site for site in sites(REPO)}
    listed = probes(REPO)
    routes = _routes()
    chosen = [p for p in listed if database in p.get('databases', DATABASES)]
    with tempfile.TemporaryDirectory(prefix='sdm-census-') as scratch:
        directory = Path(scratch) / 'cases'
        directory.mkdir()
        for number, probe in enumerate(chosen):
            case = _case(probe, number, routes)
            problem = runner._invalid(runner.validator('case'), case)
            if problem is not None:
                raise CensusError(f'probe of {probe["site"]}: {problem}')
            (directory / f'{number}.json').write_text(
                json.dumps(
                    {
                        'case': case,
                        'options': probe.get('options', {}),
                        'model': probe.get('model', 'KumoRelational'),
                    }
                )
            )
        core_out = Path(scratch) / 'python-core.json'
        route(directory, core_out, database, core=True)
        reference_out = Path(scratch) / 'reference.json'
        done = subprocess.run(
            [
                reference_python,
                '-I',
                str(Path(__file__).resolve()),
                'route',
                '--reference',
                str(reference),
                '--database',
                database,
                '--cases',
                str(directory),
                '--out',
                str(reference_out),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=3600,
            # The reference's interpreter, isolated: nothing of this one's
            # path reaches it.
            env={k: v for k, v in os.environ.items() if k != 'PYTHONPATH'},
            check=False,
        )
        if done.returncode != 0:
            raise CensusError(
                f'the reference route failed: {done.stderr[-2000:]}'
            )
        core = json.loads(core_out.read_text(encoding='utf-8'))
        theirs = json.loads(reference_out.read_text(encoding='utf-8'))
    results = [
        record(probe, mine, ref)
        for probe, mine, ref in zip(chosen, core, theirs, strict=True)
    ]
    return {
        'format': 'sdm-refusal-census-run/1',
        'database': database,
        'commit': runner._git(REPO),
        'reference': runner._git(reference),
        'sites': len(found),
        'probes': results,
        'reachability': 'unmeasured: a probe names the site it was written '
        'for, and an outcome does not show which site produced it',
        'departs': sorted({r['site'] for r in results if r['departs']}),
        'hits': [],
        'real_data': 'not run: the public calls over the real datasets run at '
        "each lane's first landing (design, choice 10)",
        'unmeasured': sorted(set(DATABASES) - {database}),
    }


# --- the command line --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('sites', 'check'):
        command = commands.add_parser(name)
        command.add_argument('--root', type=Path, default=REPO)
    running = commands.add_parser('run')
    running.add_argument('--reference', type=Path, required=True)
    running.add_argument('--reference-python', required=True)
    running.add_argument('--database', choices=DATABASES, default='duckdb')
    running.add_argument('--out', type=Path, required=True)
    routed = commands.add_parser('route')
    routed.add_argument('--reference', type=Path, required=True)
    routed.add_argument('--database', choices=DATABASES, default='duckdb')
    routed.add_argument('--cases', type=Path, required=True)
    routed.add_argument('--out', type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == 'sites':
            print(json.dumps(sites(arguments.root), indent=2))
            return 0
        if arguments.command == 'check':
            problems = check(arguments.root)
            for problem in problems:
                print(f'FAILS census: {problem}')
            print(f'census: {"FAIL" if problems else "ok"}')
            return 1 if problems else 0
        if arguments.command == 'route':
            import sdm_client

            client = Path(sdm_client.__file__).resolve()
            if not client.is_relative_to(arguments.reference.resolve()):
                raise CensusError("sdm_client is not the reference checkout's")
            route(
                arguments.cases, arguments.out, arguments.database, core=False
            )
            return 0
        evidence = run(
            arguments.reference, arguments.reference_python, arguments.database
        )
        arguments.out.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        print(
            f'census: {len(evidence["probes"])} probes on {arguments.database}, '
            f'{len(evidence["departs"])} sites where the core departs'
        )
        return 0
    except CensusError as error:
        print(f'census: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
