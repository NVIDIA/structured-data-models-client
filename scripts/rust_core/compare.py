#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Compares runners' observations of the contract cases.

    scripts/rust_core/compare.py --cases DIR [--reference FILE]
        [--policy FILE] [--bases FILE] [--expect FILE] [--register FILE]
        [--database DATABASE] [OBSERVATIONS...]

With no OBSERVATIONS, ``--reference`` is held to the cases alone (a
SQLite or Postgres job before the core's providers land, T2.1a fix 4).
Each OBSERVATIONS file is one runner's output (run_python.py or
sdm-contract-run) on one database, ``--database`` (``duckdb`` by default;
T2.1a): the cases that run there (their ``databases``) are compared, each
by its line's key there (the bare ID on DuckDB, ``<case>@<database>``
elsewhere), and lines of another database are noted, not compared. Off
DuckDB the case file's ``expect`` is DuckDB's, so the reference's line
there is the expectation, and a capture is complete by its own record
(DuckDB's query log is DuckDB's alone). FILE is the committed reference observations
(``contract/reference/<reference-commit>.jsonl``), and ``--bases`` the
bases file the cases build from (this checkout's ``contract/bases.yaml`` by
default; ``gate.sh`` gives the candidate's, which the checkpoint's ratchet
lets change only by additions), and ``--expect`` each case's effective
expected outcome by runner, which ``ratchet.py check --effective-out``
writes from the checkpoint's baselines and the transitions approved since
(T1.14). Every observation is
validated against ``contract/observation.schema.json``; a file with none,
or a case a given runner did not observe, is a runner failure. Each
observation must be of the case and bases files compared (their SHA-256 in
its identity), and leaves null only what its runner cannot observe: a
Python runner always has a stage, and a class for a refusal. Then, for
each case, the rules ``contract/compare-policy.yaml`` names, each a gate or
diagnostic there (the policy must name exactly the rules this runs). A
gate's difference fails an acceptance case; in a diagnostic case every
difference is reported, not gated, but for ``reach`` on the cases the
policy names under ``reach``. A diagnostic rule only reports. The rules:

- ``outcome`` (T1.10): the call succeeded or refused, as expected and as
  the reference did, at the same stage and with the same exception class
  and ``StructuredDataError.code`` wherever both sides observe them, or, for
  a case and runner ``--expect`` names, as its effective expectation says
  (the reference plus approved deviations, design "Reference observations
  and the gate"), the reference itself still held to the case; each
  core runner's outcome is the core's answer as the route maps it
  (accepted, a success; refused, ``INVALID_REQUEST``; an error, a refusal
  with its code; a description or request state, the call going on);
- ``route``: ``python-core`` ran the case's core operations with
  ``sdm_client._native`` loaded; ``python-old`` and ``python-ref`` never
  loaded it; ``python-ref``'s guard passed; every runner ran DuckDB at the
  reference's version;
- ``inputs``: each observation is of the case and bases files compared,
  and the core runners started the case's core operations with its inputs;
- ``reach`` (T1.11e): a case named under ``reach`` reached its result on
  every runner: a success at that stage, and from ``predict`` a result with
  rows (the public result on a Python runner, the core's on a core runner);
- ``schema``, ``warnings``, ``result`` (T1.11e): each Python runner's public
  ``Schema``, warning classes in order, and result through ``to_pandas()``
  (columns and dtypes, rows, and their digest in order) are the
  reference's; in a runtime cell other than the reference's
  (``identity.cell`` named and not ``ref-pandas3``; H1, T1.18) they are
  ``python-old``'s there, both runners required, and the reference's are
  reported, not gated, since a pandas or pyarrow of that cell may differ
  from the reference's (a difference only one cell shows is a register
  entry naming the cell, never an exclusion);
- ``payload`` (T1.11e): each request the scripted endpoint received is the
  reference's, parsed, every number compared as its kind (an integer or a
  float token) and its exact decimal value, the policy's exclusions left
  out (the value of ``metadata.request_id``, decision E35, approved by the
  owner on 2026-10-10; presence and type still compared); on
  ``rust-host``, the call the core wrote is the reference's request, and
  on ``python-core`` it is the request the endpoint received;
- ``core``: ``python-core`` and ``rust-host`` started the same operations
  with the same inputs, the core asked both for the same fetches with the
  same plans (the step trace), and answered both the same: the same
  acceptance, description or request state, payload included, the same
  call and result of ``predict`` (the call under the exclusions), or an
  error of the same kind and code, under one contract version;
- ``sql.count`` and ``sql.result`` (decision H9, T1.11e): against the
  reference, per public call, the number of statements run (setup
  excluded), and each statement's result: its columns, DuckDB's type names
  where both runners read them from a Python driver, its Arrow schema
  where both read Arrow, and its rows in canonical form, in order only for
  a statement the policy's ``ordered`` names, otherwise as a multiset;
  between the core runners, each statement's result likewise. A runner's
  record of what ran (the driver capture on a Python runner, the
  provider's own record on ``rust-host``) must be complete, its counts
  equal to DuckDB's own query log; ``rust-host`` is held to the public
  calls whose core operations the case names, since it makes no other;
- ``sql.text``, ``sql.parameters``, ``sql.bindings``: each statement's raw
  text and typed parameters, and the relations bound, against the
  reference's: reported (H9).

More exact than the reference (T2.0g, approved by the owner on
2026-10-17): a register entry approved by the owner may list, under
``more_exact``, a case, the core runners and the exact fields where the
core answers more exactly than the reference (the nearest double where the
reference's cast is one unit in the last place off, an exact text id where
it sends a float). Each field is a path as this prints a difference
(``public.payloads/0/2/context/...``, ``core.predicted.body/0/...``), ``*``
standing for any one key or index. For that case and runner, the
``schema``, ``result`` and ``payload`` rules leave out the value at each
such path, on both sides, and compare everything else; a field present on
one side only, a list of another length, or any other field still differs.
A value is left out only where both sides hold one value there (a
string, a number or a boolean), never a dict, a list or a null. The
register is the candidate's (``--register``; ``gate.sh`` gives it), but an
entry counts only when its ``decision_ref`` names a row of this harness's
(the checkpoint's) decisions approved by the owner, as a policy relaxation
does; an entry not approved so, or a field that is not such a path,
excuses nothing.

Past the reference's stop (E112): for a case and runner whose effective
expectation is an approved entry's move (``--expect``'s ``@moved``) of the
reference's refusal to a later stage, or to a success, the rules against
the reference hold only the evidence it produced: the statements and
bindings of each public call before its stop as above, at its stop the
reference's own, each against the runner's at its position (the runner's
further ones there came after the reference stopped), none after; the
``schema`` where the reference got past ``from_source``; the ``result``
and ``payload`` (the core's call on ``rust-host`` too) where it reached
``predict``, the result where it recorded one; its ``warnings`` the first
of the runner's, in order. The rules between the core runners are
unchanged past the stop. And a row read's DuckDB types name the Arrow
types DuckDB reads them as with ``arrow_large_buffer_size`` where the
case's own setup sets it (``large_string``, ``large_binary``,
``large_list``).

The fields ``rust-host`` cannot observe, a stage and an exception class,
are null there and not compared, never treated as equal; it has no public
call, so the host rules hold the Python runners only. The first difference
per case, runner and field is printed with its path; message text is never
compared, and input values are never printed. Each runner's statement count
per public call is printed for information. Exit status 1 on any gating
difference or runner failure.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import re
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / 'contract/runners/run_python.py'
CAPTURE_CHECK = REPO / 'contract/runners/capture_check.py'
#: The outcome fields, compared wherever both sides observe them.
OUTCOME = ('result', 'stage', 'error', 'code')
#: The outcome fields the pure-Rust host cannot observe: it makes no public
#: call, so it has no stage and no exception class. Every other runner
#: observes both, and a null there is a defect, never a pass.
UNOBSERVED_BY_RUST_HOST = ('stage', 'error')
PYTHON = ('python-ref', 'python-core', 'python-old')
#: The runtime cell the reference is recorded in (design, "Runtime cells",
#: H1). In another named cell the host rules compare ``python-core`` with
#: ``python-old`` there and only report the reference's (T1.18).
REFERENCE_CELL = 'ref-pandas3'
#: The rules this comparator runs, which the policy must name exactly.
RULES = (
    'outcome',
    'route',
    'inputs',
    'reach',
    'schema',
    'warnings',
    'result',
    'payload',
    'core',
    'sql.count',
    'sql.result',
    'sql.text',
    'sql.parameters',
    'sql.bindings',
)
#: The public call each core operation that reads data runs in.
STAGE_OF = {
    'describe_source': 'from_source',
    'materialize': 'materialize',
    'predict': 'predict',
}
STAGES = ('from_source', 'materialize', 'predict')
_INTEGER = re.compile(r'-?(0|[1-9][0-9]*)')
#: The register this comparator's own tree holds, read for HOST-5's
#: approval: the checkpoint's, when ``gate.sh`` runs the checkpoint's copy,
#: so a candidate never approves its own narrowing.
REGISTER = REPO / 'docs/design/rust-core/register.yaml'
#: The entry that, approved by the owner, narrows the ``warnings`` rule in
#: a cell other than the reference's (decision E81): a ``DeprecationWarning``
#: python-old shows and python-core does not, raised from one of
#: ``RUNTIME_LIBRARIES``' own files by the warning's recorded origin, is not
#: counted. Any other difference still fails.
RUNTIME_DEPRECATIONS = 'HOST-5'
RUNTIME_LIBRARIES = ('pandas', 'pyarrow')
#: The runners a ``more_exact`` item may name: the core's, since it is the
#: core that is more exact than the reference.
MORE_EXACT_RUNNERS = ('python-core', 'rust-host')
#: A ``more_exact`` field: a path under one of the roots the host and
#: payload rules compare, at least one key or index below it.
_MORE_EXACT_FIELD = re.compile(
    r'(public\.schema|public\.result|public\.payloads|core\.predicted\.body)'
    r'(/[^/]+)+'
)
#: What stands for a value ``more_exact`` leaves out, on both sides.
_MORE_EXACT = 'left out: more exact than the reference'


def _runner() -> ModuleType:
    r"""``run_python.py``, whose case loader and schemas this shares."""
    spec = importlib.util.spec_from_file_location('run_python', RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.cache
def _capture_check() -> ModuleType:
    r"""``capture_check.py``, whose E35 mask of a prediction payload this
    shares, so the rule is written once."""
    spec = importlib.util.spec_from_file_location(
        'capture_check', CAPTURE_CHECK
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predicted(core: dict[str, Any], paths: tuple[Any, ...]) -> Any:
    r"""``core.predicted``, its call's request id compared by presence and
    type only (decision E35, approved by the owner on 2026-10-10)."""
    predicted = core.get('predicted')
    if not isinstance(predicted, dict) or 'body' not in predicted:
        return predicted
    return {
        **predicted,
        'body': _capture_check().masked(predicted['body'], paths),
    }


class Report:
    r"""Differences by rule: a gate's is a failure, a diagnostic rule's a
    note; a difference with no rule (the files, a runner failure) always
    fails. :meth:`demote` turns a diagnostic case's failures into notes,
    but for the rules it keeps."""

    def __init__(
        self,
        rules: dict[str, str] | None = None,
        paths: tuple[Any, ...] = (),
    ) -> None:
        self.failures: list[tuple[str | None, str]] = []
        self.notes: list[str] = []
        self.info: list[str] = []
        self.rules = rules or {}
        #: The payload exclusions of the policy compared under, checked.
        self.paths = paths
        #: Whether HOST-5 is approved by the owner, so the ``warnings``
        #: rule leaves out the runtime's DeprecationWarnings python-old
        #: alone shows in another cell (decision E81).
        self.narrowed = False
        #: The fields left out as more exact than the reference, by case
        #: and runner (:func:`more_exact`).
        self.more_exact: dict[tuple[str, str], tuple[str, ...]] = {}

    def fail(self, text: str, rule: str | None = None) -> None:
        if rule is not None and self.rules.get(rule) == 'diagnostic':
            self.notes.append(f'diagnostic rule {rule}: {text}')
        else:
            self.failures.append((rule, text))

    def note(self, text: str) -> None:
        self.notes.append(text)

    def reported(self, since: int, why: str) -> None:
        r"""The failures since ``since`` reported as notes, not gated,
        each saying ``why``."""
        for rule, text in self.failures[since:]:
            self.notes.append(
                f'reported ({why}): {text if rule is None else f"{rule}: {text}"}'
            )
        del self.failures[since:]

    def demote(self, since: int, keep: tuple[str, ...] = ()) -> None:
        kept = []
        for rule, text in self.failures[since:]:
            if rule in keep:
                kept.append((rule, text))
            else:
                self.notes.append(
                    f'diagnostic: {text}'
                    if rule is None
                    else f'diagnostic: {rule}: {text}'
                )
        del self.failures[since:]
        self.failures.extend(kept)

    @property
    def texts(self) -> list[str]:
        return [
            text if rule is None else f'{rule}: {text}'
            for rule, text in self.failures
        ]


def _observations(
    path: Path, runner: ModuleType, report: Report
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    r"""One runner's observations by case, each validated."""
    validate = runner.validator('observation')
    by_case: dict[str, dict[str, Any]] = {}
    runners: set[str] = set()
    if not path.is_file():
        report.fail(f'{path.name}: runner failure: no observations written')
        return None, by_case
    for number, line in enumerate(
        path.read_text(encoding='utf-8').splitlines(), 1
    ):
        try:
            observation = json.loads(line)
        except (json.JSONDecodeError, RecursionError):
            # Nested past the parser is refused by name, never a traceback.
            report.fail(f'{path.name}:{number}: runner failure: not JSON')
            continue
        try:
            problem = runner._invalid(validate, observation)
        except RecursionError:
            # Python 3.14's parser reads what earlier ones refused, and the
            # validator then recurses past the limit: refused by name too.
            problem = 'nested past what the validator reads'
        if problem is not None:
            report.fail(
                f'{path.name}:{number}: runner failure: not a valid '
                f'observation: {problem}'
            )
            continue
        case = observation['case']
        if case in by_case:
            report.fail(f'{path.name}: {case} observed twice')
        by_case[case] = observation
        runners.add(observation['identity']['runner'])
    if len(runners) > 1:
        report.fail(f'{path.name}: observations of more than one runner')
    return (runners.pop() if len(runners) == 1 else None), by_case


def _differ(
    report: Report,
    where: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    show: bool = True,
    rule: str = 'outcome',
) -> None:
    if show:
        report.fail(
            f'{where}: {field}: expected {expected!r}, observed {observed!r}',
            rule,
        )
    else:
        report.fail(f'{where}: {field}: differs', rule)


def _expectation(
    report: Report, case: dict[str, Any], who: str, outcome: dict[str, Any]
) -> None:
    r"""The outcome and code rules against the case's expectation."""
    expect = case['expect']
    where = f'{case["id"]}: {who}'
    if outcome['result'] != expect['outcome']:
        _differ(
            report,
            where,
            'outcome.result',
            expect['outcome'],
            outcome['result'],
        )
        return
    wanted = {
        'stage': expect['stage'],
        'error': expect.get('error'),
        'code': expect.get('code'),
    }
    for field, value in wanted.items():
        seen = outcome[field]
        if who == 'rust-host' and field in UNOBSERVED_BY_RUST_HOST:
            continue
        if seen != value:
            _differ(report, where, f'outcome.{field}', value, seen)


def _against_reference(
    report: Report,
    case: dict[str, Any],
    who: str,
    outcome: dict[str, Any],
    reference: dict[str, Any],
) -> None:
    for field in OUTCOME:
        mine, theirs = outcome[field], reference['outcome'][field]
        if who == 'rust-host' and field in UNOBSERVED_BY_RUST_HOST:
            continue
        if mine != theirs:
            _differ(
                report,
                f'{case["id"]}: {who} against python-ref',
                f'outcome.{field}',
                theirs,
                mine,
            )


def _effective(
    report: Report,
    case: dict[str, Any],
    who: str,
    outcome: dict[str, Any],
    wanted: dict[str, Any],
    reference: dict[str, Any] | None,
) -> None:
    r"""The outcome rule against the case's effective expectation for this
    runner, in place of its expectation and the reference's outcome: each
    field the expectation names, the rest the reference's (or, with none,
    the case's)."""
    expect = case['expect']
    base = (
        reference['outcome']
        if reference is not None
        else {
            'result': expect['outcome'],
            'stage': expect['stage'],
            'error': expect.get('error'),
            'code': expect.get('code'),
        }
    )
    where = f'{case["id"]}: {who} against its effective expectation'
    for field in OUTCOME:
        if who == 'rust-host' and field in UNOBSERVED_BY_RUST_HOST:
            continue
        value = wanted.get(field, base[field])
        if outcome[field] != value:
            _differ(report, where, f'outcome.{field}', value, outcome[field])


#: ``--expect``'s key naming, by case, the runners whose expectation an
#: approved entry moved (``ratchet.MOVED``); no case id starts with ``@``.
MOVED = '@moved'


def expectations(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    r"""The effective expectations ``ratchet.py`` wrote: by case and runner,
    outcome fields, and under :data:`MOVED` the runners, by case, whose
    expectation is an approved entry's move; a file of another shape is
    refused by name."""
    try:
        found = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, RecursionError, OSError):
        raise ValueError(f'{path.name}: not JSON it reads') from None
    if not isinstance(found, dict):
        raise ValueError(f'{path.name}: not effective expectations')
    moved = found.get(MOVED, {})
    if not isinstance(moved, dict) or not all(
        isinstance(runners, list)
        and all(isinstance(each, str) for each in runners)
        for runners in moved.values()
    ):
        raise ValueError(f'{path.name}: not effective expectations')
    if not all(
        isinstance(by_runner, dict)
        and all(
            isinstance(wanted, dict) and set(wanted) <= set(OUTCOME)
            for wanted in by_runner.values()
        )
        for case, by_runner in found.items()
        if case != MOVED
    ):
        raise ValueError(f'{path.name}: not effective expectations')
    return found


def past_stop(
    who: str,
    wanted: dict[str, Any] | None,
    moved: bool,
    reference: dict[str, Any] | None,
) -> str | None:
    r"""The stage the reference stopped at, where this runner's effective
    expectation is an approved entry's move (``moved``) of the reference's
    refusal to a later stage, or to a success (a core runner's answer past
    every stage; ``rust-host`` observes no stage); None otherwise, and
    the rules against the reference are as ever (E112)."""
    if wanted is None or not moved or reference is None:
        return None
    theirs = reference['outcome']
    stop = theirs['stage']
    if theirs['result'] != 'refusal' or stop not in STAGES:
        return None
    if wanted.get('result', theirs['result']) == 'success':
        return str(stop)
    stage = wanted.get('stage', stop)
    if stage in STAGES and STAGES.index(stage) > STAGES.index(stop):
        return str(stop)
    return None


def _route(
    report: Report, case: dict[str, Any], who: str, observation: dict[str, Any]
) -> None:
    identity = observation['identity']
    where = f'{case["id"]}: {who}'
    wanted = sorted(case.get('core', {}))
    if who == 'python-core':
        if identity['native_loaded'] is not True or identity['native'] is None:
            report.fail(
                f'{where}: route: sdm_client._native was not loaded', 'route'
            )
        missing = [op for op in wanted if op not in identity['operations']]
        if missing or observation['core'] is None:
            report.fail(
                f'{where}: route: {missing} did not run on the core', 'route'
            )
    elif who in ('python-old', 'python-ref'):
        if identity['native_loaded'] is not False or identity['operations']:
            report.fail(
                f'{where}: route: the old route loaded the core', 'route'
            )
        if who == 'python-ref' and identity['reference'] is None:
            report.fail(f'{where}: route: no reference guard result', 'route')
    elif who == 'rust-host':
        missing = [op for op in wanted if op not in identity['operations']]
        if missing or observation['core'] is None:
            report.fail(
                f'{where}: route: {missing} did not run on the host', 'route'
            )


def _shape(
    report: Report, case: dict[str, Any], who: str, observation: dict[str, Any]
) -> None:
    r"""Which fields may be null, by runner and outcome."""
    where = f'{case["id"]}: {who}'
    outcome = observation['outcome']
    if who in PYTHON:
        if outcome['stage'] is None:
            report.fail(
                f'{where}: outcome.stage: null on a Python runner', 'outcome'
            )
        if outcome['result'] == 'refusal' and outcome['error'] is None:
            report.fail(
                f'{where}: outcome.error: a refusal without a class', 'outcome'
            )
        if outcome['result'] == 'success' and (
            outcome['error'] is not None or outcome['code'] is not None
        ):
            report.fail(f'{where}: outcome: a success with an error', 'outcome')
        if observation['host'] is None:
            report.fail(f'{where}: host: null on a Python runner', 'outcome')
        if observation['identity']['inputs']['bases'] is None:
            report.fail(f'{where}: identity.inputs.bases: null', 'outcome')
        if (observation['core'] is None) != (who != 'python-core'):
            report.fail(
                f'{where}: core: present only on python-core', 'outcome'
            )
    else:
        for field in UNOBSERVED_BY_RUST_HOST:
            if outcome[field] is not None:
                report.fail(
                    f'{where}: outcome.{field}: not observable here', 'outcome'
                )
        if observation['host'] is not None or observation['core'] is None:
            report.fail(
                f'{where}: a host section, or no core section', 'outcome'
            )


def _inputs(
    report: Report,
    case: dict[str, Any],
    who: str,
    observation: dict[str, Any],
    digests: dict[str, str],
) -> None:
    r"""The observation is of the case and bases files compared."""
    inputs = observation['identity']['inputs']
    where = f'{case["id"]}: {who}'
    if inputs['case'] != digests[case['id']]:
        report.fail(
            f'{where}: identity.inputs.case: not this case file', 'inputs'
        )
    if inputs['bases'] is not None and inputs['bases'] != digests['']:
        report.fail(
            f'{where}: identity.inputs.bases: not this bases file', 'inputs'
        )


def _core_consistent(
    report: Report, case: dict[str, Any], who: str, observation: dict[str, Any]
) -> None:
    r"""The core's answer is the outcome, as the route maps it: an accepted
    query succeeds, a refused one is ``INVALID_REQUEST``, and a core error
    is a refusal with its own code. ``describe_source`` answers with a
    description or an error, never both, and its error ends the call there,
    as a refusal with its code; a description lets the call go on, so the
    outcome is the later calls', the user's edits first (``edit_schema``,
    from T1.24b), whose refusal ends it beside the description."""
    core = observation['core']
    if core is None:
        return
    if (
        core.get('described') is not None
        or core.get('materialized') is not None
    ) and not (isinstance(core.get('fetches'), list) and core['fetches']):
        # A description or request state is built from fetched rows, so its
        # step trace holds at least one fetch: an empty or missing trace is
        # missing evidence, failed on its own runner, never taken as the
        # two hosts agreeing (T1.29; the test author's seed of a trace
        # emptied on both hosts).
        report.fail(
            f'{case["id"]}: {who}: core.fetches: a description or request '
            'state with no fetch recorded',
            'core',
        )
    if 'predict' in case.get('core', {}) and core['materialized'] is not None:
        # Materialized, the call goes on to predict, which answers its call
        # and result or an error, never both (T1.11d); its error ends the
        # call there, as a refusal with its code.
        where = f'{case["id"]}: {who}'
        if (core['predicted'] is None) == (core['error'] is None):
            report.fail(
                f'{where}: core: not one prediction or one error', 'outcome'
            )
        elif core['error'] is not None:
            seen = (
                observation['outcome']['result'],
                observation['outcome']['code'],
            )
            wanted = ('refusal', core['error']['code'])
            if seen != wanted:
                _differ(report, where, 'outcome against core', wanted, seen)
        return
    if 'materialize' in case.get('core', {}) and core['described'] is not None:
        # Described, the call goes on to materialize, which answers its
        # request state or an error, never both (T1.11c); its error ends
        # the call there, as a refusal with its code.
        where = f'{case["id"]}: {who}'
        if (core['materialized'] is None) == (core['error'] is None):
            report.fail(
                f'{where}: core: not one request state or one error', 'outcome'
            )
        elif core['error'] is not None:
            seen = (
                observation['outcome']['result'],
                observation['outcome']['code'],
            )
            wanted = ('refusal', core['error']['code'])
            if seen != wanted:
                _differ(report, where, 'outcome against core', wanted, seen)
        return
    if 'describe_source' in case.get('core', {}):
        where = f'{case["id"]}: {who}'
        # From T1.24b the user's edits follow the description, each an
        # edit_schema, and a refused one ends the call with its error
        # beside the description (decision E67).
        edits = core.get('edits') or []
        edited = bool(edits) and edits[-1].get('op') == 'edit_schema'
        if (core['described'] is None) == (core['error'] is None) and not (
            edited and core['described'] is not None
        ):
            report.fail(
                f'{where}: core: not one description or one error', 'outcome'
            )
        elif core['error'] is not None:
            seen = (
                observation['outcome']['result'],
                observation['outcome']['code'],
            )
            wanted = ('refusal', core['error']['code'])
            if seen != wanted:
                _differ(report, where, 'outcome against core', wanted, seen)
        return
    if 'validate_query' not in case.get('core', {}):
        return
    outcome = observation['outcome']
    if core['error'] is not None:
        wanted = ('refusal', core['error']['code'])
        if core['accepted'] is not None:
            report.fail(
                f'{case["id"]}: {who}: core: an answer and an error', 'outcome'
            )
    elif core['accepted'] is None:
        report.fail(f'{case["id"]}: {who}: core: no answer', 'outcome')
        return
    else:
        wanted = (
            ('success', None)
            if core['accepted']
            else ('refusal', 'INVALID_REQUEST')
        )
    seen = (outcome['result'], outcome['code'])
    if seen != wanted:
        _differ(
            report, f'{case["id"]}: {who}', 'outcome against core', wanted, seen
        )


def _core_agree(
    report: Report,
    case: dict[str, Any],
    python: dict[str, Any],
    rust: dict[str, Any],
) -> None:
    r"""The kind and code rule: the two core runners on ``core`` fields."""
    where = f'{case["id"]}: python-core against rust-host'
    if python['identity']['operations'] != rust['identity']['operations']:
        _differ(
            report,
            where,
            'identity.operations',
            python['identity']['operations'],
            rust['identity']['operations'],
            rule='core',
        )
    mine, theirs = python['core'], rust['core']
    if mine is None or theirs is None:
        return
    for field in ('inputs', 'fetches', 'described', 'materialized'):
        if mine[field] != theirs[field]:
            _differ(
                report,
                where,
                f'core.{field}',
                None,
                None,
                show=False,
                rule='core',
            )
    if _predicted(mine, report.paths) != _predicted(theirs, report.paths):
        _differ(
            report, where, 'core.predicted', None, None, show=False, rule='core'
        )
    for field in ('accepted', 'error'):
        if mine[field] != theirs[field]:
            _differ(
                report,
                where,
                f'core.{field}',
                mine[field],
                theirs[field],
                rule='core',
            )
    for field in ('contract',):
        if python['identity'][field] != rust['identity'][field]:
            _differ(
                report,
                where,
                f'identity.{field}',
                python['identity'][field],
                rust['identity'][field],
                rule='core',
            )


def _case_inputs(
    report: Report, case: dict[str, Any], who: str, observation: dict[str, Any]
) -> None:
    r"""A core runner's recorded inputs are the case's ``core`` inputs, and
    the schema ``materialize`` sampled is the runner's own: the description
    its ``describe_source`` answered, where the case makes no edit to it
    (T1.21d; the case names no schema, and with edits the hosts' agreement
    is the ``core`` rule's)."""
    core = observation['core']
    if core is None:
        return
    wanted = []
    if 'validate_query' in case.get('core', {}):
        spec = case['core']['validate_query']
        wanted.append(
            {
                'op': 'validate_query',
                'query': case['input']['pql'],
                'profile': spec['profile'],
                'schema': spec['schema'],
            }
        )
    if 'describe_source' in case.get('core', {}):
        wanted.append(
            {'op': 'describe_source', **case['core']['describe_source']}
        )
    missing = any(each not in core['inputs'] for each in wanted)
    if 'materialize' in case.get('core', {}):
        # The schema it samples is the host's own; the case names the rest.
        wanted_materialize = {
            'op': 'materialize',
            'query': case['input']['pql'],
            **case['core']['materialize'],
        }
        started = [
            each for each in core['inputs'] if each.get('op') == 'materialize'
        ]
        missing = missing or not any(
            all(
                each.get(key) == value
                for key, value in wanted_materialize.items()
            )
            for each in started
        )
        if not case['input'].get('edits') and any(
            each.get('schema') != core['described'] for each in started
        ):
            _differ(
                report,
                f'{case["id"]}: {who}',
                'core.inputs.materialize.schema',
                None,
                None,
                show=False,
                rule='inputs',
            )
    if 'predict' in case.get('core', {}):
        # predict's state is materialize's answer and its request id the
        # host's to draw, so the case names only its model.
        wanted_predict = {'op': 'predict', **case['core']['predict']}
        missing = missing or not any(
            all(each.get(key) == value for key, value in wanted_predict.items())
            for each in core['inputs']
            if each.get('op') == 'predict'
        )
    if missing:
        _differ(
            report,
            f'{case["id"]}: {who}',
            'core.inputs',
            None,
            None,
            show=False,
            rule='inputs',
        )


def _path_of(
    mine: Any, theirs: Any, path: str, excused: tuple[str, ...] = ()
) -> str | None:
    r"""Where two values first differ, as a path, never the values; the
    value at each ``excused`` path (:func:`more_exact`) left out on both
    sides where both hold one value there (:func:`_left_out`)."""
    if excused:
        mine, theirs = _left_out(mine, theirs, path, excused)
    found: str | None = _capture_check()._first_difference(mine, theirs, path)
    return found


def _left_out(
    mine: Any, theirs: Any, path: str, excused: tuple[str, ...]
) -> tuple[Any, Any]:
    r"""Both sides, found at ``path``, with what is at each ``excused`` path
    replaced by one marker where both hold one value there (:func:`_one`);
    a dict, a list or a null there, on either side, is kept, so it still
    differs, and so do keys and lengths."""
    if any(_matches(path, field) for field in excused):
        if _one(mine) and _one(theirs):
            return _MORE_EXACT, _MORE_EXACT
        return mine, theirs
    if isinstance(mine, dict) and isinstance(theirs, dict):
        pairs = {
            key: _left_out(mine[key], theirs[key], f'{path}/{key}', excused)
            for key in mine.keys() & theirs.keys()
        }
        return (
            {
                key: pairs[key][0] if key in pairs else v
                for key, v in mine.items()
            },
            {
                key: pairs[key][1] if key in pairs else v
                for key, v in theirs.items()
            },
        )
    if (
        isinstance(mine, (list, tuple))
        and isinstance(theirs, (list, tuple))
        and len(mine) == len(theirs)
    ):
        pairs_of = [
            _left_out(a, b, f'{path}/{index}', excused)
            for index, (a, b) in enumerate(zip(mine, theirs, strict=True))
        ]
        return [a for a, _ in pairs_of], [b for _, b in pairs_of]
    return mine, theirs


def _one(value: Any) -> bool:
    r"""Whether ``value`` is one value: a string, a number or a boolean, a
    payload number as :func:`_lossless` reads it, or that number's exact
    value (a ``Decimal``, the path compare prints for a changed number);
    never a null, a dict or a list."""
    if isinstance(value, list):
        # Only _lossless's own form: its kind, its exact value (a Decimal,
        # which no JSON value is) and its sign.
        return (
            len(value) == 4
            and value[0] == 'number'
            and value[1] in ('integer', 'float')
            and isinstance(value[2], Decimal)
            and isinstance(value[3], bool)
        )
    return isinstance(value, (str, int, float, Decimal))


def _matches(path: str, field: str) -> bool:
    mine, theirs = path.split('/'), field.split('/')
    return len(mine) == len(theirs) and all(
        part == want or want == '*'
        for part, want in zip(mine, theirs, strict=True)
    )


def _lossless(value: Any) -> Any:
    r"""A parsed payload with each number (``{"number": text}``) as its kind
    (an integer token or a float token) and its exact decimal value, so
    ``1e-05`` and ``1e-5`` agree, and ``1`` and ``1.0`` do not, nor ``-0.0``
    and ``0.0`` (its sign kept); nothing passes through a double."""
    if isinstance(value, dict):
        if set(value) == {'number'} and isinstance(value['number'], str):
            text = value['number']
            kind = 'integer' if _INTEGER.fullmatch(text) else 'float'
            value = Decimal(text)
            # The sign too, which Decimal equality ignores for zero, so
            # -0.0 and 0.0 differ while 1e-05 and 1e-5 agree.
            return ['number', kind, value, value.is_signed()]
        return {key: _lossless(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_lossless(item) for item in value]
    return value


def _payload(parsed: Any, paths: tuple[Any, ...]) -> Any:
    r"""A payload as the ``payload`` rule compares it: under the exclusions
    of the policy compared under (``paths``, ``capture_check.masked``),
    with lossless numbers."""
    return _lossless(_capture_check().masked(parsed, paths))


def _reach(
    report: Report,
    case: dict[str, Any],
    who: str,
    observation: dict[str, Any],
    stage: str,
) -> None:
    r"""The case reached its result at ``stage`` on this runner."""
    where = f'{case["id"]}: {who}'
    outcome = observation['outcome']
    if outcome['result'] != 'success' or (
        who != 'rust-host' and outcome['stage'] != stage
    ):
        report.fail(f'{where}: did not reach its result at {stage}', 'reach')
        return
    if stage != 'predict':
        return
    if who in PYTHON:
        result = (observation.get('public') or {}).get('result')
        if not result or not result.get('rows'):
            report.fail(f'{where}: no public result with rows', 'reach')
    if who in ('python-core', 'rust-host'):
        predicted = (observation.get('core') or {}).get('predicted') or {}
        if not (predicted.get('result') or {}).get('rows'):
            report.fail(f'{where}: no result with rows from the core', 'reach')


def _host_fields(
    report: Report,
    case: dict[str, Any],
    who: str,
    observation: dict[str, Any],
    reference: dict[str, Any],
    against: str = 'python-ref',
    stop: str | None = None,
) -> None:
    r"""A Python runner's public fields against the reference's (or, in a
    cell other than the reference's, ``python-old``'s, ``against``): the
    ``schema``, ``warnings``, ``result`` and ``payload`` rules; past the
    reference's ``stop`` (:func:`past_stop`), only what it produced."""
    where = f'{case["id"]}: {who} against {against}'
    excused = report.more_exact.get((case['id'], who), ())
    mine, theirs = observation.get('public'), reference.get('public')
    if mine is None or theirs is None:
        report.fail(f'{where}: public: not recorded', 'schema')
        return
    found = (
        _path_of(mine['schema'], theirs['schema'], 'public.schema', excused)
        if stop is None or stop != 'from_source'
        else None
    )
    if found:
        report.fail(f'{where}: {found}', 'schema')
    shown = observation['host']['warnings']
    if stop is not None:
        # The reference's warnings, raised before it stopped, are the
        # first of the runner's, in order.
        shown = shown[: len(reference['host']['warnings'])]
    if shown != reference['host']['warnings']:
        if against == 'python-old' and _runtime_only(
            observation['host'],
            reference['host'],
            report.narrowed,
            held=stop is not None,
        ):
            report.note(
                f'{where}: host.warnings: python-old alone shows the '
                f"runtime's DeprecationWarnings ({RUNTIME_DEPRECATIONS})"
            )
        else:
            report.fail(f'{where}: host.warnings: differ', 'warnings')
    if stop is not None and stop != 'predict':
        return
    found = (
        _path_of(mine['result'], theirs['result'], 'public.result', excused)
        if stop is None or theirs['result'] is not None
        else None
    )
    if found:
        report.fail(f'{where}: {found}', 'result')
    sent = [
        (each['method'], each['path'], _payload(each['parsed'], report.paths))
        for each in mine['payloads']
    ]
    wanted = [
        (each['method'], each['path'], _payload(each['parsed'], report.paths))
        for each in theirs['payloads']
    ]
    found = _path_of(sent, wanted, 'public.payloads', excused)
    if found:
        report.fail(f'{where}: {found}', 'payload')


def _runtime_only(
    core: Any, old: Any, narrowed: bool, held: bool = False
) -> bool:
    r"""Whether python-core's warnings are python-old's with only the
    runtime's DeprecationWarnings left out (HOST-5): each removed one a
    ``DeprecationWarning`` whose recorded origin is one of
    ``RUNTIME_LIBRARIES`` (:func:`_runtime`), the rest the same classes in
    the same order, as the rule compares them (the client's own warnings
    name the caller's frame, which the two routes reach from different
    modules, so a kept warning's origin is not compared). A runtime warning
    is matched only by a runtime warning and any other only by another
    (fix round 1, Codex P2): a warning python-core alone shows can never
    stand in for one of the runtime's that python-old shows. False unless
    ``narrowed`` (the entry the owner approved) and both runners recorded
    every warning's origin, its class the one the warning list names; a
    missing, short or retyped record never narrows anything.

    ``held``: python-core went past where python-old stopped (``stop`` in
    :func:`_host_fields`), so python-old's warnings, its runtime ones left
    out, are the first of python-core's, and the rest of python-core's,
    raised past the stop, are not compared, as :func:`_host_fields` holds a
    runner past the reference's stop. Before T1.32g fix 12, python-core's
    warnings were cut to python-old's count first, which held python-core's
    own warnings past the stop against the runtime's DeprecationWarnings
    python-old raised before it (pandas 2.2.0 under pyarrow 14, the
    floor-pandas2.2 cell)."""
    if not narrowed:
        return False
    mine, theirs = _origins(core), _origins(old)
    if mine is None or theirs is None:
        return False
    at = 0
    for each in theirs:
        if (
            at < len(mine)
            and each[0] == mine[at][0]
            and _runtime(each) == _runtime(mine[at])
        ):
            at += 1
        elif not _runtime(each):
            return False
    return held or at == len(mine)


def _runtime(warning: tuple[str, str | None]) -> bool:
    r"""Whether a warning is one HOST-5 may leave out: a
    ``DeprecationWarning`` raised from a runtime library's own files."""
    return (
        warning[0] == 'DeprecationWarning' and warning[1] in RUNTIME_LIBRARIES
    )


def _origins(host: Any) -> list[tuple[str, str | None]] | None:
    r"""A host section's warnings as (class, module) pairs, or None where
    its record of origins is missing or does not match its warnings."""
    if not isinstance(host, dict):
        return None
    shown, origins = host.get('warnings'), host.get('origins')
    if not isinstance(shown, list) or not isinstance(origins, list):
        return None
    if len(shown) != len(origins):
        return None
    found = []
    for name, each in zip(shown, origins, strict=True):
        if (
            not isinstance(each, dict)
            or set(each) != {'class', 'module'}
            or each['class'] != name
            or not isinstance(name, str)
            or not isinstance(each['module'], (str, type(None)))
        ):
            return None
        found.append((name, each['module']))
    return found


def owner_approved(entry: str, register: Path = REGISTER) -> bool:
    r"""Whether ``register`` holds ``entry`` approved by the owner; any
    file it cannot read, or an entry of another status, says no."""
    import yaml

    try:
        loaded = yaml.safe_load(register.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, yaml.YAMLError, RecursionError):
        return False
    entries = loaded.get('entries') if isinstance(loaded, dict) else None
    if not isinstance(entries, list):
        return False
    return any(
        isinstance(each, dict)
        and each.get('id') == entry
        and each.get('status') == 'approved'
        and each.get('approved_by') == 'owner'
        for each in entries
    )


@functools.cache
def _ratchet() -> ModuleType:
    r"""``ratchet.py``, whose reading of the owner's approval in a decision
    row's status this shares."""
    spec = importlib.util.spec_from_file_location(
        'compare_ratchet', Path(__file__).resolve().parent / 'ratchet.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _owner_rows() -> set[str]:
    r"""The IDs of this harness's (the checkpoint's) decision rows the owner
    approved, every phase's; none when they cannot be read."""
    check = _capture_check()
    try:
        rows = check.decision_rows(check.repository_decisions())
    except check.DecisionsError:
        return set()
    return {
        name
        for name, (_, status) in rows.items()
        if _ratchet().owner_approved(status)
    }


def more_exact(register: Path) -> dict[tuple[str, str], tuple[str, ...]]:
    r"""The fields each entry of ``register`` lists under ``more_exact``, by
    case and runner, from an entry approved by the owner whose
    ``decision_ref`` names a row of this harness's (the checkpoint's)
    decisions the owner approved, so a candidate cannot invent an approval
    (that the row is the right one is review's):
    only a core runner's, only a path under a root the host and payload
    rules compare; anything else, and a file it cannot read, excuses
    nothing."""
    import yaml

    try:
        loaded = yaml.safe_load(register.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, yaml.YAMLError, RecursionError):
        return {}
    approved = _owner_rows()
    entries = loaded.get('entries') if isinstance(loaded, dict) else None
    found: dict[tuple[str, str], tuple[str, ...]] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not (
            isinstance(entry, dict)
            and entry.get('status') == 'approved'
            and entry.get('approved_by') == 'owner'
            and isinstance(entry.get('decision_ref'), dict)
            and isinstance(entry['decision_ref'].get('row'), str)
            and entry['decision_ref']['row'] in approved
            and isinstance(entry.get('more_exact'), list)
        ):
            continue
        for item in entry['more_exact']:
            if not isinstance(item, dict):
                continue
            case, runners = item.get('case'), item.get('runners')
            fields = item.get('fields')
            if not (
                isinstance(case, str)
                and isinstance(runners, list)
                and isinstance(fields, list)
            ):
                continue
            kept = tuple(
                field
                for field in fields
                if isinstance(field, str) and _MORE_EXACT_FIELD.fullmatch(field)
            )
            for who in runners:
                if who in MORE_EXACT_RUNNERS:
                    found[case, who] = found.get((case, who), ()) + kept
    return found


def _core_payload(
    report: Report,
    case: dict[str, Any],
    who: str,
    observation: dict[str, Any],
    reference: dict[str, Any] | None,
    stop: str | None = None,
) -> None:
    r"""The call the core wrote (``core.predicted.body``) is the request
    the reference sent, on ``rust-host``, and the request the endpoint
    received, on ``python-core``: the ``payload`` rule on the core; on
    ``rust-host``, held to the reference only where it reached ``predict``
    when past its ``stop`` (:func:`past_stop`)."""
    predicted = (observation.get('core') or {}).get('predicted')
    if not predicted or predicted.get('body') is None:
        return
    body = _payload(predicted['body'], report.paths)
    if who == 'python-core':
        sent = [
            _payload(each['parsed'], report.paths)
            for each in (observation.get('public') or {}).get('payloads', [])
        ]
        found = _path_of([body], sent, 'core.predicted.body')
        if found:
            report.fail(
                f'{case["id"]}: python-core: the call against the request '
                f'received: {found}',
                'payload',
            )
    if (
        who == 'rust-host'
        and reference is not None
        and stop in (None, 'predict')
    ):
        wanted = [
            _payload(each['parsed'], report.paths)
            for each in reference['public']['payloads']
        ]
        found = _path_of(
            [body],
            wanted,
            'core.predicted.body',
            report.more_exact.get((case['id'], who), ()),
        )
        if found:
            report.fail(
                f'{case["id"]}: rust-host against python-ref: {found}',
                'payload',
            )


def _by_stage(
    observation: dict[str, Any], key: str = 'statements'
) -> dict[str, list[dict[str, Any]]] | None:
    r"""A runner's statements (or bindings) by the public call they ran
    in, in order; None when it recorded none (no capture)."""
    capture = (observation.get('driver') or {}).get('capture')
    if capture is None:
        return None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for each in capture[key]:
        grouped.setdefault(each['stage'], []).append(each)
    return grouped


def _stages(case: dict[str, Any], who: str) -> tuple[str, ...]:
    r"""The public calls whose statements this runner is held to: every
    one on a Python runner, which makes the case's public calls through
    the capture; on ``rust-host``, those whose core operations the case
    names, since it makes no other."""
    if who in PYTHON:
        return STAGES
    named = {STAGE_OF[op] for op in case.get('core', {}) if op in STAGE_OF}
    return tuple(stage for stage in STAGES if stage in named)


def counts(observation: dict[str, Any]) -> dict[str, int] | None:
    r"""A runner's statement count per public call, setup excluded."""
    grouped = _by_stage(observation)
    if grouped is None:
        return None
    return {stage: len(grouped.get(stage, [])) for stage in STAGES}


#: The methods a runner reads a result with, by what each gives: Arrow,
#: whose schema the read records, or Python rows, which have none.
ARROW_READS = ('to_arrow_table', 'fetch_arrow_table', 'query_arrow')
ROW_READS = ('fetchall', 'fetchmany', 'fetchone')
#: DuckDB's names of its types, as a Python driver's description gives
#: them, and the Arrow type (pyarrow's spelling) DuckDB reads each as; a
#: list is ``X[]`` and ``list<l: X>``, a decimal ``DECIMAL(p,s)`` and
#: ``decimal128(p, s)``. A name outside this map has no Arrow name here,
#: and is never taken as equal to one.
DUCKDB_ARROW = {
    'BOOLEAN': 'bool',
    'TINYINT': 'int8',
    'SMALLINT': 'int16',
    'INTEGER': 'int32',
    'BIGINT': 'int64',
    'UTINYINT': 'uint8',
    'USMALLINT': 'uint16',
    'UINTEGER': 'uint32',
    'UBIGINT': 'uint64',
    'FLOAT': 'float',
    'DOUBLE': 'double',
    'VARCHAR': 'string',
    'BLOB': 'binary',
    'DATE': 'date32[day]',
    'TIME': 'time64[us]',
    'TIME_NS': 'time64[ns]',
    'TIMESTAMP': 'timestamp[us]',
    'TIMESTAMP_S': 'timestamp[s]',
    'TIMESTAMP_MS': 'timestamp[ms]',
    'TIMESTAMP_NS': 'timestamp[ns]',
    'INTERVAL': 'month_day_nano_interval',
}
_DECIMAL = re.compile(r'DECIMAL\((\d+),(\d+)\)')


#: What DuckDB reads as Arrow's large types under ``SET
#: arrow_large_buffer_size = true``: text and blobs, and a list as
#: ``large_list`` (E112).
DUCKDB_ARROW_LARGE = {'VARCHAR': 'large_string', 'BLOB': 'large_binary'}
#: The setting whose statements :func:`large_buffers` reads.
LARGE_BUFFER_SETTING = 'arrow_large_buffer_size'
#: The one canonical statement an element naming the setting may be
#: (E112, fix 3, the orchestrator's strict rule): ``SET`` with ``=``,
#: ``TO`` or neither and ``true`` or ``false``, unquoted or single-quoted,
#: or ``RESET``; any case; no scope.
_LARGE_BUFFERS = re.compile(
    r'SET\s+arrow_large_buffer_size(?:\s*=\s*|\s+TO\s+|\s+)'
    r"(?P<value>true|false|'true'|'false')"
    r'|RESET\s+arrow_large_buffer_size',
    re.IGNORECASE,
)


class UnsupportedSetting(ValueError):
    r"""A base or setup element naming :data:`LARGE_BUFFER_SETTING` that
    is not one canonical statement: failing the comparison by name, never
    guessed."""


def large_buffers(
    case: dict[str, Any], bases: dict[str, list[str]] | None = None
) -> bool:
    r"""Whether DuckDB writes Arrow's large buffers for the case's
    statements: the last canonical statement on
    :data:`LARGE_BUFFER_SETTING` in its base's statements (``bases``) and
    then its own setup, as DuckDB runs them, says ``true``; a ``RESET`` or
    ``false`` after it turns it off. Strict (E112, fix 3): an element whose
    raw text names the setting anywhere, in any case, must be, after
    trimming its white space and one trailing ``;``, exactly one statement
    :data:`_LARGE_BUFFERS` matches, or :class:`UnsupportedSetting` is
    raised (comments, several statements, a scope, another value or a read
    included); an element not naming it is ignored. No SQL is lexed."""
    spec = case.get('input') or {}
    steps = [
        *(bases or {}).get(spec.get('base'), []),
        *(spec.get('setup') or []),
    ]
    found = False
    for step in steps:
        if not isinstance(step, str):
            continue
        if LARGE_BUFFER_SETTING not in step.lower():
            continue
        text = step.strip()
        if text.endswith(';'):
            text = text[:-1].rstrip()
        setting = _LARGE_BUFFERS.fullmatch(text)
        if setting is None:
            raise UnsupportedSetting(
                f'unsupported setting statement on {LARGE_BUFFER_SETTING}'
            )
        found = (setting.group('value') or '').strip("'").lower() == 'true'
    return found


#: The Arrow types ADBC's SQLite driver reads a column as (it infers each
#: from the values: integers, floats, text, bytes, an empty or all-null
#: column as integers); its cursor's description names each by its Arrow
#: name, so a SQLite row read's types are those names, each its own Arrow
#: name. Any other is not one, and is never taken as equal to one.
SQLITE_ARROW = {name: name for name in ('int64', 'double', 'string', 'binary')}


#: The Arrow types the Python host's Postgres provider builds a column of
#: each Postgres type as (``sdm_client._rust.providers.postgres``, T2.3),
#: by the type's name as psycopg's description gives it, which the driver
#: capture records: a type it reads as text, and a type the driver does
#: not name (a composite or enum), as ``string``. ``numeric`` with a
#: precision, an array, and a ``varchar`` or ``bpchar`` of a length are
#: read by :func:`postgres_arrow_name`. A ``timestamptz`` is built in UTC,
#: whatever the session's zone, so its name says its Arrow type.
POSTGRES_ARROW = {
    'bool': 'bool',
    'int2': 'int16',
    'int4': 'int32',
    'int8': 'int64',
    'oid': 'int64',
    'float4': 'double',
    'float8': 'double',
    'bytea': 'binary',
    'date': 'date32[day]',
    'time': 'time64[us]',
    'timestamp': 'timestamp[us]',
    'timestamptz': 'timestamp[us, tz=UTC]',
    'interval': 'month_day_nano_interval',
    'numeric': 'string',
    **{
        name: 'string'
        for name in (
            'text',
            'varchar',
            'bpchar',
            'char',
            'name',
            'uuid',
            'xml',
            'inet',
            'cidr',
            'macaddr',
            'macaddr8',
            'json',
            'jsonb',
            'money',
            'bit',
            'varbit',
            'timetz',
        )
    },
}
#: A composite type the contract loader created, which the driver does not
#: name and reads as text (``contract/runners/databases.py``'s
#: ``composite_name``).
_POSTGRES_COMPOSITE = re.compile(r'sdm_[0-9a-f]{32}')
_POSTGRES_NUMERIC = re.compile(r'numeric\((\d+),(\d+)\)')
_POSTGRES_LENGTH = re.compile(r'(varchar|bpchar|bit|varbit)\(\d+\)')


def postgres_arrow_name(postgres_type: str) -> str | None:
    r"""The Arrow type the Postgres provider builds a column of
    ``postgres_type`` as, or None when :data:`POSTGRES_ARROW` does not say:
    ``numeric(p,s)`` a ``decimal128`` to 38 digits and a ``decimal256``
    past them; an array a list of its element's type, as pyarrow names it;
    a length's type its type's."""
    if postgres_type.endswith('[]'):
        inner = postgres_arrow_name(postgres_type[:-2])
        return None if inner is None else f'list<item: {inner}>'
    found = _POSTGRES_NUMERIC.fullmatch(postgres_type)
    if found:
        width = 128 if int(found.group(1)) <= 38 else 256
        return f'decimal{width}({found.group(1)}, {found.group(2)})'
    found = _POSTGRES_LENGTH.fullmatch(postgres_type)
    if found:
        return POSTGRES_ARROW[found.group(1)]
    if _POSTGRES_COMPOSITE.fullmatch(postgres_type):
        return 'string'
    return POSTGRES_ARROW.get(postgres_type)


def row_arrow_name(
    database: str, row_type: str, large: bool = False
) -> str | None:
    r"""The Arrow type a row read's recorded type stands for, by the
    database's driver: DuckDB's name of its type (:func:`arrow_name`) on
    DuckDB, ADBC's Arrow name on SQLite (:data:`SQLITE_ARROW`); None where
    the driver's vocabulary does not say, which never compares equal; on
    Postgres, the Arrow type its provider builds the column as
    (:func:`postgres_arrow_name`)."""
    if database == 'duckdb':
        return arrow_name(row_type, large)
    if database == 'sqlite':
        return SQLITE_ARROW.get(row_type)
    if database == 'postgres':
        return postgres_arrow_name(row_type)
    return None


def arrow_name(duckdb_type: str, large: bool = False) -> str | None:
    r"""The Arrow type DuckDB reads a column of ``duckdb_type`` as, or
    None when :data:`DUCKDB_ARROW` does not say; with ``large``
    (:func:`large_buffers`), as its large buffers name it."""
    if duckdb_type.endswith('[]'):
        inner = arrow_name(duckdb_type[:-2], large)
        if inner is None:
            return None
        return f'large_list<l: {inner}>' if large else f'list<l: {inner}>'
    found = _DECIMAL.fullmatch(duckdb_type)
    if found and int(found.group(1)) <= 38:
        return f'decimal128({found.group(1)}, {found.group(2)})'
    if large and duckdb_type in DUCKDB_ARROW_LARGE:
        return DUCKDB_ARROW_LARGE[duckdb_type]
    return DUCKDB_ARROW.get(duckdb_type)


def _evidence(report: Report, at: str, who: str, read: dict[str, Any]) -> bool:
    r"""The read carries the type evidence its runner and method give: a
    Python driver's read DuckDB's type names (its description), an Arrow
    read its Arrow schema, each one per column; only the pure-Rust host,
    which has no driver description, lacks the type names, and only a row
    read lacks a schema. Missing evidence fails, never skips the check."""
    columns = len(read['values']['columns'])
    method = read['method']
    good = True
    if method not in ARROW_READS + ROW_READS:
        report.fail(f'{at}: {who} read it by {method!r}', 'sql.result')
        return False
    types = read['values']['types']
    if who == 'rust-host':
        if method != 'query_arrow' or types is not None:
            report.fail(f'{at}: rust-host read it otherwise', 'sql.result')
            good = False
    elif not isinstance(types, list) or len(types) != columns:
        report.fail(f'{at}: {who} recorded no DuckDB types', 'sql.result')
        good = False
    schema = read.get('schema')
    if method in ARROW_READS and (
        not isinstance(schema, list) or len(schema) != columns
    ):
        report.fail(f'{at}: {who} recorded no Arrow schema', 'sql.result')
        good = False
    return good


def _results(
    report: Report,
    where: str,
    mine: dict[str, Any],
    theirs: dict[str, Any],
    ordered: bool,
    who: tuple[str, str],
    large: bool = False,
    database: str = 'duckdb',
) -> None:
    r"""One statement's results against another's: ``sql.result``. ``who``
    names the two runners, whose evidence :func:`_evidence` requires;
    ``large``, :func:`large_buffers` of the case; ``database``, whose
    driver's vocabulary a row read's types are in (:func:`row_arrow_name`)."""
    if (mine['error'] is None) != (theirs['error'] is None):
        report.fail(f'{where}: one statement failed', 'sql.result')
        return
    if len(mine['results']) != len(theirs['results']):
        report.fail(
            f'{where}: {len(mine["results"])} results read, against '
            f'{len(theirs["results"])}',
            'sql.result',
        )
        return
    for read, (a, b) in enumerate(
        zip(mine['results'], theirs['results'], strict=True)
    ):
        at = f'{where} result {read}'
        if not (
            _evidence(report, at, who[0], a) & _evidence(report, at, who[1], b)
        ):
            continue
        va, vb = a['values'], b['values']
        if va['unencodable'] or vb['unencodable']:
            # A value the canonical form does not hold is never equal.
            report.fail(
                f'{at}: values the canonical form does not hold: '
                f'{sorted(set(va["unencodable"]) | set(vb["unencodable"]))}',
                'sql.result',
            )
            continue
        if va['columns'] != vb['columns']:
            report.fail(f'{at}: columns differ', 'sql.result')
        if (
            va['types'] is not None
            and vb['types'] is not None
            and va['types'] != vb['types']
        ):
            report.fail(f'{at}: DuckDB types differ', 'sql.result')
        _arrow_types(report, at, a, b, large, database)
        if a['rows'] != b['rows']:
            report.fail(
                f'{at}: {a["rows"]} rows, against {b["rows"]}', 'sql.result'
            )
        key = 'ordered' if ordered else 'unordered'
        if va[key] != vb[key]:
            report.fail(
                f'{at}: rows differ '
                + ('in order' if ordered else 'as a multiset'),
                'sql.result',
            )


def _arrow_types(
    report: Report,
    at: str,
    a: dict[str, Any],
    b: dict[str, Any],
    large: bool = False,
    database: str = 'duckdb',
) -> None:
    r"""The two reads' Arrow types: both schemas where both read Arrow;
    where one read rows (the reference's catalog, by ``fetchall``) and the
    other Arrow (a core runner), the row read's types as Arrow names in its
    database's driver's vocabulary (:func:`row_arrow_name`: DuckDB's type
    names, ADBC SQLite's Arrow names) against the Arrow read's schema, a
    type with no Arrow name failing."""
    sa, sb = a.get('schema'), b.get('schema')
    if sa is not None and sb is not None:
        if sa != sb:
            report.fail(f'{at}: Arrow schema differs', 'sql.result')
        return
    if sa is None and sb is None:
        # Two row reads: their DuckDB types are compared.
        return
    # One read Arrow, the other rows: whether or not the Arrow read carries
    # DuckDB's names too (which the type check compares), its schema is
    # held to the row read's names as the Arrow types DuckDB reads them as.
    arrow, rows = (a, b) if sa is not None else (b, a)
    schema, types = arrow['schema'], rows['values']['types']
    if types is None:
        report.fail(f'{at}: neither read records its types', 'sql.result')
        return
    named = [row_arrow_name(database, kind, large) for kind in types]
    if None in named:
        report.fail(
            f'{at}: {database} types with no Arrow name: '
            f'{sorted({k for k, n in zip(types, named, strict=True) if n is None})}',
            'sql.result',
        )
        return
    if named != [field['type'] for field in schema]:
        report.fail(f'{at}: Arrow schema differs from the types', 'sql.result')


def _complete(report: Report, where: str, observation: dict[str, Any]) -> bool:
    r"""The runner's record of what ran is there and complete, its counts
    DuckDB's own; otherwise nothing of it is compared, and that fails."""
    driver = observation.get('driver')
    if driver is None or driver.get('capture') is None:
        report.fail(f'{where}: no record of the statements run', 'sql.count')
        return False
    problems = _capture_check().complete(observation)
    for problem in problems:
        report.fail(f'{where}: the record of what ran: {problem}', 'sql.count')
    return not problems


def _sql(
    report: Report,
    case: dict[str, Any],
    who: str,
    observation: dict[str, Any],
    reference: dict[str, Any],
    ordered: dict[str, list[int]],
    stop: str | None = None,
    large: bool = False,
    database: str = 'duckdb',
) -> None:
    r"""Decision H9 against the reference: ``sql.count`` and ``sql.result``,
    with ``sql.text``, ``sql.parameters`` and ``sql.bindings`` reported;
    past the reference's ``stop`` (:func:`past_stop`), at that stage the
    reference's statements and bindings each against the runner's at its
    position, and none after."""
    stages = _stages(case, who)
    if stop is not None:
        stages = tuple(
            stage
            for stage in stages
            if STAGES.index(stage) <= STAGES.index(stop)
        )
    if not stages:
        return
    where = f'{case["id"]}: {who} against python-ref'
    if not _complete(report, f'{case["id"]}: python-ref', reference):
        return
    if not _complete(report, where, observation):
        return
    mine, theirs = _by_stage(observation), _by_stage(reference)
    assert mine is not None and theirs is not None
    for stage in stages:
        ours, refs = mine.get(stage, []), theirs.get(stage, [])
        if stage == stop and len(ours) > len(refs):
            # Those after the reference's own came after it stopped.
            ours = ours[: len(refs)]
        if len(ours) != len(refs):
            report.fail(
                f'{where}: {stage}: {len(ours)} statements, the reference '
                f'{len(refs)}',
                'sql.count',
            )
            continue
        for position, (a, b) in enumerate(zip(ours, refs, strict=True)):
            at = f'{where}: {stage} statement {position}'
            if a['sql'] != b['sql']:
                report.fail(f'{at}: text differs', 'sql.text')
            if a['parameters'] != b['parameters']:
                report.fail(f'{at}: parameters differ', 'sql.parameters')
            _results(
                report,
                at,
                a,
                b,
                position in ordered.get(stage, []),
                (who, 'python-ref'),
                large,
                database,
            )
    mine_bound = _by_stage(observation, 'bindings') or {}
    theirs_bound = _by_stage(reference, 'bindings') or {}
    for stage in stages:
        ours, refs = mine_bound.get(stage, []), theirs_bound.get(stage, [])
        if stage == stop and len(ours) > len(refs):
            ours = ours[: len(refs)]
        if len(ours) != len(refs):
            report.fail(
                f'{where}: {stage}: {len(ours)} relations bound, the '
                f'reference {len(refs)}',
                'sql.bindings',
            )
            continue
        for position, (a, b) in enumerate(zip(ours, refs, strict=True)):
            at = f'{where}: {stage} binding {position}'
            va, vb = a.get('values'), b.get('values')
            if a.get('rows') != b.get('rows') or (
                va is None or vb is None or va['unordered'] != vb['unordered']
            ):
                report.fail(
                    f'{at}: its rows differ as written, the order of '
                    'columns included',
                    'sql.bindings',
                )


def _sql_core(
    report: Report,
    case: dict[str, Any],
    python: dict[str, Any],
    rust: dict[str, Any],
    ordered: dict[str, list[int]],
    large: bool = False,
    database: str = 'duckdb',
) -> None:
    r"""The two core runners' statement results, public call by public
    call: ``sql.count`` and ``sql.result``; ``large``, :func:`large_buffers`
    of the case; ``database``, the run's."""
    stages = _stages(case, 'rust-host')
    if not stages:
        return
    where = f'{case["id"]}: python-core against rust-host'
    if not (
        _complete(report, f'{case["id"]}: python-core', python)
        and _complete(report, f'{case["id"]}: rust-host', rust)
    ):
        return
    mine, theirs = _by_stage(python), _by_stage(rust)
    assert mine is not None and theirs is not None
    for stage in stages:
        ours, refs = mine.get(stage, []), theirs.get(stage, [])
        if len(ours) != len(refs):
            report.fail(
                f'{where}: {stage}: {len(ours)} statements, against '
                f'{len(refs)}',
                'sql.count',
            )
            continue
        for position, (a, b) in enumerate(zip(ours, refs, strict=True)):
            _results(
                report,
                f'{where}: {stage} statement {position}',
                a,
                b,
                position in ordered.get(stage, []),
                ('python-core', 'rust-host'),
                large,
                database,
            )


def _cell_pair(
    report: Report,
    case: dict[str, Any],
    core: dict[str, Any] | None,
    old: dict[str, Any] | None,
    stop: str | None = None,
) -> None:
    r"""H1's comparison in a cell other than the reference's: ``python-core``
    and ``python-old`` observed in the same cell, and their host fields
    equal (the ``schema``, ``warnings``, ``result`` and ``payload`` rules),
    in place of the reference's, which are reported. Either runner alone
    there is a failure: the cell's comparison needs both. ``stop``, where
    python-old stopped as the reference did and an approved move took
    python-core past it (E114): held only that far, as :func:`_host_fields`
    holds a runner past the reference's stop."""
    cells = {
        each['identity'].get('cell') for each in (core, old) if each is not None
    }
    other = cells - {None, REFERENCE_CELL}
    if not other:
        return
    where = f'{case["id"]}: cell {sorted(other)[0]}'
    if core is None or old is None:
        report.fail(
            f'{where}: python-core and python-old are both needed: the cell '
            'compares them with each other (H1)'
        )
        return
    if len(cells) != 1:
        report.fail(f'{where}: python-core and python-old in different cells')
        return
    _host_fields(
        report, case, 'python-core', core, old, 'python-old', stop=stop
    )


def compare(
    cases_dir: list[Path],
    reference_file: Path | None,
    files: list[Path],
    policy_file: Path | None = None,
    bases_file: Path | None = None,
    expect_file: Path | None = None,
    register_file: Path = REGISTER,
    improvements: Path | None = None,
    database: str = 'duckdb',
) -> Report:
    check = _capture_check()
    try:
        policy = (
            check.policy() if policy_file is None else check.policy(policy_file)
        )
    except check.PolicyError as problem:
        report = Report()
        report.fail(f'compare-policy.yaml: {problem}')
        return report
    report = Report(policy['rules'], check.excluded(policy))
    report.narrowed = owner_approved(RUNTIME_DEPRECATIONS, register_file)
    report.more_exact = more_exact(improvements or register_file)
    if set(policy['rules']) != set(RULES):
        report.fail(
            'compare-policy.yaml: its rules are not the ones this runs: '
            f'{sorted(set(policy["rules"]) ^ set(RULES))}'
        )
        return report
    try:
        effective = {} if expect_file is None else expectations(expect_file)
    except ValueError as problem:
        report.fail(f'--expect: {problem}')
        return report
    runner = _runner()
    try:
        loaded = runner.load_cases(cases_dir)
    except runner.RunnerFailure as failure:
        report.fail(f'cases: {failure}')
        return report
    if database not in runner.DATABASES:
        report.fail(f'--database: {database} is not one of {runner.DATABASES}')
        return report
    cases = [
        case for case, _ in loaded if database in runner.case_databases(case)
    ]
    try:
        bases = runner.load_bases(bases_file or runner.BASES)
    except runner.RunnerFailure as failure:
        report.fail(f'bases: {failure}')
        return report
    digests = {
        runner.case_key(case['id'], database): digest for case, digest in loaded
    }
    digests[''] = runner.digest((bases_file or runner.BASES).read_bytes())
    reference: dict[str, dict[str, Any]] = {}
    if reference_file is not None:
        who, reference = _observations(reference_file, runner, report)
        if who != 'python-ref':
            report.fail(f'{reference_file.name}: not python-ref observations')
    observed: dict[str, dict[str, dict[str, Any]]] = {}
    for path in files:
        who, by_case = _observations(path, runner, report)
        if who is None:
            report.fail(f'{path.name}: runner failure: no valid observations')
            continue
        if who in observed:
            report.fail(f'{path.name}: a second file of {who}')
        observed[who] = by_case
    # A line's key is its case, or its case and database (``<case>@sqlite``,
    # ``<case>@postgres``; the schema holds its form): the part before ``@``
    # must be a case. A line of another database than this run's is read,
    # and compared by that database's run.
    known = {case['id'] for case, _ in loaded}
    for who, by_case in [('python-ref', reference), *observed.items()]:
        for name in sorted(by_case):
            bare, _, there = name.partition('@')
            if bare not in known:
                report.fail(
                    f'{name}: {who} observed a case that does not exist'
                )
            elif (there or 'duckdb') != database:
                report.note(
                    f'{name}: {who}: a line of {there or "duckdb"}, not '
                    f'compared in the {database} run'
                )
    # A case the policy names must exist, among every case of the
    # repository, not only those compared now.
    every = {
        case['id'] for case, _ in runner.load_cases([runner.CONTRACT / 'cases'])
    }
    for name in sorted(set(policy['reach']) | set(policy['ordered'])):
        if name not in every:
            report.fail(f'compare-policy.yaml: {name} is not a case')
    for case in cases:
        try:
            large = large_buffers(case, bases)
        except UnsupportedSetting as problem:
            # Neither mode is assumed: the rows' types are not taken for
            # either, and the case fails by name.
            report.fail(f'{case["id"]}: {problem}', 'sql.result')
            large = False
        gating = case['status'] == 'acceptance'
        reach = policy['reach'].get(case['id'])
        ordered = policy['ordered'].get(case['id'], {})
        # The case as this database's lines name it; its expect is
        # DuckDB's, so elsewhere the reference's line is the expectation.
        on_duckdb = database == 'duckdb'
        case = {**case, 'id': runner.case_key(case['id'], database)}
        if not on_duckdb:
            # The core is started on this database's dialect.
            case['core'] = {
                op: (
                    {
                        **spec,
                        'provider': {**spec['provider'], 'dialect': database},
                    }
                    if (spec.get('provider') or {}).get('dialect') == 'duckdb'
                    else spec
                )
                for op, spec in case.get('core', {}).items()
            }
        before = len(report.failures)
        ref = reference.get(case['id'])
        stops: dict[str, str | None] = {}
        if reference_file is not None and ref is None:
            report.fail(f'{case["id"]}: no reference observation')
        if ref is not None:
            _shape(report, case, 'python-ref', ref)
            _inputs(report, case, 'python-ref', ref, digests)
            if on_duckdb:
                _expectation(report, case, 'python-ref', ref['outcome'])
            _route(report, case, 'python-ref', ref)
            if not observed:
                # The reference alone (a SQLite or Postgres job before the
                # core's providers, T2.1a fix 5): its record of what ran is
                # held complete, as _sql holds it beside a runner's.
                _complete(report, f'{case["id"]}: python-ref', ref)
            if reach is not None:
                _reach(report, case, 'python-ref', ref, reach)
            found = counts(ref)
            if found is not None:
                report.info.append(f'{case["id"]}: python-ref: {found}')
        for who, by_case in observed.items():
            observation = by_case.get(case['id'])
            if observation is None:
                report.fail(
                    f'{case["id"]}: {who}: runner failure: no observation',
                    'reach' if reach is not None else None,
                )
                continue
            _shape(report, case, who, observation)
            _inputs(report, case, who, observation, digests)
            _core_consistent(report, case, who, observation)
            wanted = effective.get(case['id'], {}).get(who)
            stop = past_stop(
                who,
                wanted,
                who in effective.get(MOVED, {}).get(case['id'], []),
                ref,
            )
            stops[who] = stop
            if wanted is not None:
                _effective(
                    report, case, who, observation['outcome'], wanted, ref
                )
            else:
                if on_duckdb or ref is None:
                    _expectation(report, case, who, observation['outcome'])
                if ref is not None:
                    _against_reference(
                        report, case, who, observation['outcome'], ref
                    )
            _route(report, case, who, observation)
            _case_inputs(report, case, who, observation)
            if reach is not None:
                _reach(report, case, who, observation, reach)
            found = counts(observation)
            if found is not None:
                report.info.append(f'{case["id"]}: {who}: {found}')
            _core_payload(report, case, who, observation, ref, stop)
            if ref is not None:
                cell = observation['identity'].get('cell')
                if who in PYTHON and cell not in (None, REFERENCE_CELL):
                    # H1: in a cell other than the reference's, the host
                    # fields against the reference are reported; python-core
                    # against python-old gates them (below).
                    since = len(report.failures)
                    _host_fields(report, case, who, observation, ref, stop=stop)
                    report.reported(
                        since, f'cell {cell}, against the reference; H1'
                    )
                elif who in PYTHON:
                    _host_fields(report, case, who, observation, ref, stop=stop)
                _sql(
                    report,
                    case,
                    who,
                    observation,
                    ref,
                    ordered,
                    stop,
                    large,
                    database,
                )
                theirs = ref['identity']['versions']['duckdb']
                mine = observation['identity']['versions']['duckdb']
                if mine != theirs:
                    _differ(
                        report,
                        f'{case["id"]}: {who}',
                        'identity.versions.duckdb',
                        theirs,
                        mine,
                        rule='route',
                    )
        python = observed.get('python-core', {}).get(case['id'])
        old = observed.get('python-old', {}).get(case['id'])
        # Past the reference's stop (E114): python-old held python-core
        # only as far as it went, where it stopped as the reference did.
        pair_stop = (
            stops.get('python-core')
            if old is not None
            and ref is not None
            and all(old['outcome'][f] == ref['outcome'][f] for f in OUTCOME)
            else None
        )
        _cell_pair(report, case, python, old, pair_stop)
        rust = observed.get('rust-host', {}).get(case['id'])
        if python is not None and rust is not None:
            _core_agree(report, case, python, rust)
            _sql_core(report, case, python, rust, ordered, large, database)
        if not gating:
            report.demote(before, keep=('reach',) if reach else ())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--cases', type=Path, action='append', required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--bases', type=Path)
    parser.add_argument('--expect', type=Path)
    parser.add_argument(
        '--register',
        type=Path,
        help="the candidate's register, read for its owner-approved "
        "more_exact fields (default: this tree's)",
    )
    parser.add_argument('--database', default='duckdb')
    parser.add_argument('observations', type=Path, nargs='*')
    arguments = parser.parse_args(argv)
    if not arguments.observations and arguments.reference is None:
        # Something is compared: a runner's observations, or with none (a
        # SQLite or Postgres job before the core's providers, T2.1a fix 4)
        # the committed reference itself, held to the cases.
        parser.error('give observations, or --reference alone')
    report = compare(
        arguments.cases,
        arguments.reference,
        arguments.observations,
        arguments.policy,
        arguments.bases,
        arguments.expect,
        improvements=arguments.register,
        database=arguments.database,
    )
    for text in report.info:
        print(f'statements per public call: {text}')
    for text in report.notes:
        print(text)
    for text in report.texts:
        print(f'DIFFERS {text}')
    if report.failures:
        print(f'compare: {len(report.failures)} differences')
        return 1
    print('compare: no differences')
    return 0


if __name__ == '__main__':
    sys.exit(main())
