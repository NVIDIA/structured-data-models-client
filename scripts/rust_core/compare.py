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
  between the core runners, each statement's result likewise. On
  Postgres the reference's two reads whose results it never uses, the
  current schema's name and each table's row count, are not expected
  (KO2-58), every other statement is. A runner's
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
(``public.payloads/0/2/context/...``, ``core.predicted.body/0/...``, or
on a request asked in several calls ``core.predicted.calls/<n>/...``,
T2.1r), ``*``
standing for any one key or index. For that case and runner, the
``schema``, ``result`` and ``payload`` rules leave out the value at each
such path, on both sides, and compare everything else; a field present on
one side only, a list of another length, or any other field still differs.
A value is left out only where both sides hold one value there (a
string, a number or a boolean), never a dict, a list or a null; an item
marked ``null_on_one_side`` (T2.1e) also leaves out, at its fields, one
value on one side and a null or nothing on the other. The
register is the candidate's (``--register``; ``gate.sh`` gives it), but an
entry counts only when its ``decision_ref`` names a row of this harness's
(the checkpoint's) decisions approved by the owner, as a policy relaxation
does; an entry not approved so, or a field that is not such a path,
excuses nothing.

Fewer DeprecationWarnings (T2.1s, KO2-83, approved by the owner on
2026-10-20): a register entry approved by the owner, its ``decision_ref``
a row of this harness's (the checkpoint's) decisions the owner approved
(as ``more_exact`` reads one), may list under ``fewer_deprecations`` a
line's key (the bare case ID on DuckDB, ``<case>@sqlite`` or
``<case>@postgres``) and the core runners. For that line and runner, the
``warnings`` rule takes the runner's warnings as the reference's (or
python-old's) with only repeats of a ``DeprecationWarning`` left out: the
reference's list with some ``DeprecationWarning``s deleted and nothing
else changed, each deleted one preceded in the reference's list by
another, so the first is always kept and no class is added, dropped or
moved. The observations record each
warning's class only, so the rule works on classes; the entry carries, in
text, the evidence that the repeats share one message and source line.
Past the reference's stop nothing changes, and a line or runner no such
entry lists is compared as before.

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
unchanged past the stop. Where that move is to a success and the owner-
approved entry's ``case_outcomes`` row for the line names the stopping
call under ``statements_differ`` (T2.1e), the call answers where the
reference refused: its ``sql.count`` and ``sql.result`` against the
reference's are reported as diagnostic, nothing else; where the move is
from the reference's answer to the runner's refusal at the call the row
names, the reference's statements of that call are its only ones held,
their ``sql.count`` and ``sql.result`` and the line's ``warnings``
reported as diagnostic; where the row keeps the reference's refusal at
that call, its code the same (KO2-68), that call's ``sql.count`` and
``sql.result`` are reported as diagnostic, nothing else; where the row
moves the reference's refusal to the runner's at an earlier call, with
its own code (T2.1r), the line is held to the calls up to that one on
every database, as below: that call's ``sql.count`` and ``sql.result``
and the line's ``warnings`` reported as diagnostic, and the reference's
later calls (their operations, statements, ``result`` and ``payload``)
no expectation of its own. A SQLite or
Postgres line is held, by the ``route``, ``inputs`` and core-consistency
rules, to the case's core operations of the calls before the one it stops
at (its effective expectation's refusal, else the reference line's), and
to any of that call or a later one the runner ran (T2.1e): the case's
core block is DuckDB's run. There the ``core`` rule between the two
core runners, where one went on past the operations both ran into the
call the line stops at or a later one, holds them on the operations
both ran.
Columns only the core keeps (T2.1f, KO2-36): an owner-approved entry,
under a row of this harness's decisions the owner approved (as
``statements_differ``), may list ``core_only_columns``, each
``<table>.<column>`` (the table's name holds no dot; the column's may), for
the lines of its ``cases`` (each database's). For those lines a core
runner's observation is compared with the reference's (and, in a cell
other than the reference's, python-core's with python-old's, which leaves
them out as the reference does) with exactly those
columns taken out of the core side wherever they appear, and the links
and key they make: the public schema's columns, its primary key and time
column naming one and its links with one at either end; each payload
table's columns and its rows' cells, its schema's column and key, and its
relationships naming one, in every call's body where predict made several
(T2.1s); and each statement result's columns, types and
Arrow fields of that name where the reference's result lacks them (the
reference reads a column it then leaves out). The reference's from_source warning that it
leaves a column out, one ``SDMWarning`` per table the entry names, the
first of its warnings of that class, is taken out of the reference side
for each table whose named column the runner kept (a warning's class does
not say its table). A statement's result columns
are taken out where its text names the table, quoted.
A statement result read with such a column cannot have its row digests
compared, which hash the column: its row count is, and its digests are
reported, not gated. Everything else is compared exactly, an extra column
the entry does not name included; between the core runners nothing
changes.
And a row read's DuckDB types name the Arrow
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
import copy
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
#: The runners that make no public call, so observe neither a stage nor an
#: exception class: the pure-Rust host, and Java's host, whose S0 lines leave
#: both null as rust-host's do (T3.2, found by T3.11).
UNOBSERVING = ('rust-host', 'java')
#: An endpoint's refusal of a call as the client raises it (decision E47):
#: the endpoint's own text, under the client's class for it and no code of
#: the client's. The core, told the endpoint's status, reports the same call
#: as ``INVALID_RESPONSE`` (decision E45, ``sdm-request``'s ``bad_response``),
#: and the two agree there (T2.1g); no other pairing of class and code does.
ENDPOINT_REFUSAL = 'NimRequestError'
INVALID_RESPONSE = 'INVALID_RESPONSE'
PYTHON = ('python-ref', 'python-core', 'python-old')
#: The runners that carry the stand-in's verdict on a warehouse leg judged
#: offline (Phase 3 design choice 2): the core's, Python's, Rust's and
#: Java's hosts.
STANDIN_RUNNERS = ('python-core', 'rust-host', 'java')
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
    r'(public\.schema|public\.result|public\.payloads|core\.predicted\.body'
    r'|core\.predicted\.calls)(/[^/]+)+'
)
#: The one warning class ``fewer_deprecations`` lets a core runner show
#: fewer times than the reference (KO2-83).
REPEATED_WARNING = 'DeprecationWarning'
#: What stands for a value ``more_exact`` leaves out, on both sides.
_MORE_EXACT = 'left out: more exact than the reference'
#: How :func:`more_exact` marks a field of an item that also excuses a
#: value present on one side and null or absent on the other
#: (``null_on_one_side``, T2.1e); no path starts with it.
_NULL_SIDE = 'null-on-one-side:'


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
    masked = {
        **predicted,
        'body': _capture_check().masked(predicted['body'], paths),
    }
    if isinstance(predicted.get('calls'), list):
        # Every call's body where there were several (T2.1g), each masked
        # as the last's is.
        masked['calls'] = [
            _capture_check().masked(body, paths) for body in predicted['calls']
        ]
    return masked


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
        #: The lines and core runners that may show fewer
        #: DeprecationWarnings than the reference
        #: (:func:`fewer_deprecations`, KO2-83).
        self.fewer: frozenset[tuple[str, str]] = frozenset()
        #: The public calls whose statements an entry says differ from the
        #: reference's, by case and runner (:func:`statements_differ`).
        self.statements_differ: dict[tuple[str, str], tuple[str, ...]] = {}
        #: The columns only the core keeps, by case (the bare ID) and
        #: table (:func:`core_only_columns`, T2.1f).
        self.core_only: dict[str, dict[str, frozenset[str]]] = {}
        #: The live-only items of a stand-in run (Phase 3 design choice 2),
        #: each line's key with its code: neither passed nor failed
        #: offline, each discharged only by E-1's verdict on the final
        #: candidate (:func:`verdict_problems`).
        self.live_only: dict[str, str] = {}

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
        if (
            isinstance(observation, dict)
            and observation.get('format') == _capture_check().RECORD_FORMAT
        ):
            # A warehouse leg's reference line held in the recording mode's
            # line (Phase 3 design choice 2), read from it whole.
            observation = observation.get('observation')
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


def _standin(
    report: Report,
    key: str,
    who: str,
    observation: dict[str, Any],
    codes: tuple[str, ...],
    offline: bool,
) -> tuple[str | None, tuple[str, ...]]:
    r"""The stand-in's verdict an observation carries (Phase 3 design
    choice 2): its live-only code and its flags, each held to the policy's
    ``standin`` codes. Only a core runner's observation of a warehouse leg
    judged offline carries one, and it must; anywhere else one fails. A
    code not listed, a live-only mark with no stand-in reason among them,
    fails as any difference does, so no case marks itself live only."""
    verdict = observation.get('standin')
    if not offline or who not in STANDIN_RUNNERS:
        if verdict is not None:
            report.fail(
                f'{key}: {who}: a stand-in verdict where no stand-in served '
                'the case'
            )
        return None, ()
    if verdict is None:
        report.fail(
            f"{key}: {who}: a stand-in leg without the stand-in's verdict"
        )
        return None, ()
    code, flagged = verdict['live_only'], tuple(verdict['flagged'])
    for each in (*([] if code is None else [code]), *flagged):
        if each not in codes:
            report.fail(
                f'{key}: {who}: live-only mark {each} is no stand-in reason '
                f'of compare-policy.yaml ({", ".join(codes)})'
            )
            return None, ()
    return code, flagged


def _live_only(
    report: Report, key: str, who: str, code: str, since: int, why: str
) -> None:
    r"""``who``'s comparison of ``key`` since ``since`` made live only:
    listed by its code, its differences under a rule reported, neither
    passed nor failed offline. A failure under no rule (a runner failure,
    a file's problem) still fails."""
    report.live_only.setdefault(key, code)
    _reported_rules(report, since, f'live only, {code}: {why}')


def _reported_rules(report: Report, since: int, why: str) -> None:
    r"""The failures under a rule since ``since`` reported as notes, each
    saying ``why``; a failure under no rule still fails."""
    kept = []
    for rule, text in report.failures[since:]:
        if rule is None:
            kept.append((rule, text))
        else:
            report.notes.append(f'reported ({why}): {rule}: {text}')
    del report.failures[since:]
    report.failures.extend(kept)


def _differences(report: Report, since: int) -> bool:
    r"""Whether a rule found a difference since ``since``."""
    return any(rule is not None for rule, _ in report.failures[since:])


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


def _endpoint_refusal(outcome: dict[str, Any]) -> bool:
    r"""Whether ``outcome`` is the endpoint's refusal of a call at
    ``predict``, as the client raises it: :data:`ENDPOINT_REFUSAL`, no
    code."""
    return (
        outcome.get('result') == 'refusal'
        and outcome.get('stage') == 'predict'
        and outcome.get('error') == ENDPOINT_REFUSAL
        and outcome.get('code') is None
    )


def _rust_host_s_refusal(
    who: str, field: str, wanted: dict[str, Any], seen: Any
) -> bool:
    r"""Whether ``rust-host``'s code ``seen`` agrees with ``wanted``, an
    endpoint's refusal at ``predict`` (:func:`_endpoint_refusal`): the pure-Rust
    host has no transport or client class, so its outcome is its core's
    ``INVALID_RESPONSE`` for that call (T2.1g)."""
    return (
        who == 'rust-host'
        and field == 'code'
        and seen == INVALID_RESPONSE
        and _endpoint_refusal(wanted)
    )


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
        if who in UNOBSERVING and field in UNOBSERVED_BY_RUST_HOST:
            continue
        if _rust_host_s_refusal(
            who, field, {'result': expect['outcome'], **wanted}, seen
        ):
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
        if who in UNOBSERVING and field in UNOBSERVED_BY_RUST_HOST:
            continue
        if _rust_host_s_refusal(who, field, reference['outcome'], mine):
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
    effective = {field: wanted.get(field, base[field]) for field in OUTCOME}
    for field in OUTCOME:
        if who in UNOBSERVING and field in UNOBSERVED_BY_RUST_HOST:
            continue
        value = effective[field]
        if _rust_host_s_refusal(who, field, effective, outcome[field]):
            continue
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


def line_stop(
    wanted: dict[str, Any] | None, reference: dict[str, Any] | None
) -> str | None:
    r"""The call a database line stops at: the refusal of this runner's
    effective expectation (an approved move's), or else of the reference's
    line; None where the line answers or names no call (T2.1e)."""
    theirs = (reference or {}).get('outcome') or {}
    if wanted is not None:
        if wanted.get('result') != 'refusal':
            return None
        stage = wanted.get('stage') or theirs.get('stage')
    elif theirs.get('result') == 'refusal':
        stage = theirs.get('stage')
    else:
        return None
    return str(stage) if stage in STAGES else None


def core_held(
    case: dict[str, Any], observation: dict[str, Any], stop: str | None
) -> dict[str, Any]:
    r"""``case`` with the core operations a database line is held to (T2.1e):
    the case's core block is DuckDB's run, and off DuckDB the reference's
    line may stop at an earlier call (SQLite stores a decimal as text,
    which the reference refuses as a number, KO2-47's types), and a
    refusal there may come before the core's operation of that call runs
    (the client's query check, as the reference's), so an operation of
    the call at ``stop`` or after it is held only where the runner ran it;
    every operation of a call before the stop, and every one it ran, as
    before."""
    if stop is None:
        return case
    ran = set(observation['identity'].get('operations') or [])
    limit = STAGES.index(stop)
    held = {
        op: spec
        for op, spec in case.get('core', {}).items()
        if op in ran or op not in STAGE_OF or STAGES.index(STAGE_OF[op]) < limit
    }
    return {**case, 'core': held}


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


def _later_error(
    case: dict[str, Any], core: dict[str, Any], database: str
) -> bool:
    r"""Whether ``core``'s error, beside an accepted validation, is a later
    call's: off DuckDB every call a case makes runs through its ``until``
    (T2.1d), so an accepted query goes on to ``materialize`` (and
    ``predict``), and the outcome is that of the last call reached, the
    one with no answer; every call before it answered, none after it ran.
    A refused query, a call that both answers and errors, and DuckDB,
    where the calls stop at the case's own, are never one."""
    calls = {'materialize': 1, 'predict': 2}.get(case['until'], 0)
    if database == 'duckdb' or core['accepted'] is not True or not calls:
        return False
    answered = [
        core[field] is not None for field in ('materialized', 'predicted')
    ][:calls]
    return False in answered and not any(answered[answered.index(False) :])


def _core_consistent(
    report: Report,
    case: dict[str, Any],
    who: str,
    observation: dict[str, Any],
    database: str = 'duckdb',
) -> None:
    r"""The core's answer is the outcome, as the route maps it: an accepted
    query succeeds, a refused one is ``INVALID_REQUEST``, and a core error
    is a refusal with its own code. ``describe_source`` answers with a
    description or an error, never both, and its error ends the call there,
    as a refusal with its code; a description lets the call go on, so the
    outcome is the later calls', the user's edits first (``edit_schema``,
    from T1.24b), whose refusal ends it beside the description, as a
    ``validate_query`` error does where the case's core validates its
    query and that call ended it (T2.1r, DT-5). Off DuckDB an accepted
    query goes on through the case's ``until``, and a later call's error
    is the outcome (:func:`_later_error`)."""
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
            # The endpoint's refusal of the call, raised as the client
            # raises it, is the core's INVALID_RESPONSE for it (T2.1g).
            refused = (
                _endpoint_refusal(observation['outcome'])
                and core['error']['code'] == INVALID_RESPONSE
            )
            if seen != wanted and not refused:
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
        operations = observation['identity'].get('operations') or []
        if 'edits' not in core:
            # rust-host records no edits: its operations say the last call
            # was an edit (T2.1e, a line held to the calls it makes).
            edited = operations[-1:] == ['edit_schema']
        # A validate_query error after the description ends the call
        # beside it as a refused edit does (T2.1r; DT-5's packed date
        # literal), where the case validates its query and that call was
        # the last one made.
        last = operations[-1:]
        validated = 'validate_query' in case['core'] and last == [
            'validate_query'
        ]
        if (core['described'] is None) == (core['error'] is None) and not (
            (edited or validated) and core['described'] is not None
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
        if core['accepted'] is not None and not _later_error(
            case, core, database
        ):
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


def shared_operations(
    python: dict[str, Any], rust: dict[str, Any], stop: str | None
) -> list[str] | None:
    r"""On a database line that stops at ``stop`` (T2.1e), the operations
    both core runners ran, where one ran them and then more, every one of
    them of that call or a later one (python-core refused there by the
    client's check, rust-host's core refusing in the call it went on to);
    None otherwise, and the hosts are held to each other whole."""
    if stop is None:
        return None
    mine = python['identity']['operations']
    theirs = rust['identity']['operations']
    short, long = (mine, theirs) if len(mine) <= len(theirs) else (theirs, mine)
    if mine == theirs or long[: len(short)] != short:
        return None
    limit = STAGES.index(stop)
    for op in long[len(short) :]:
        if op not in STAGE_OF or STAGES.index(STAGE_OF[op]) < limit:
            return None
    return list(short)


def _core_agree(
    report: Report,
    case: dict[str, Any],
    python: dict[str, Any],
    rust: dict[str, Any],
    stop: str | None = None,
) -> None:
    r"""The kind and code rule: the two core runners on ``core`` fields.
    On a database line stopping at ``stop`` where one host went on past
    the operations both ran into that call or a later one
    (:func:`shared_operations`, T2.1e), the hosts are held to each other
    on the operations both ran, their fetches the shorter's, and not on
    the results of the operations one alone ran."""
    where = f'{case["id"]}: python-core against rust-host'
    shared = shared_operations(python, rust, stop)
    if shared is not None:
        _core_agree_shared(report, where, python['core'], rust['core'], shared)
        _same_contract(report, where, python, rust)
        return
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
    _same_contract(report, where, python, rust)


def _same_contract(
    report: Report, where: str, python: dict[str, Any], rust: dict[str, Any]
) -> None:
    r"""The two core runners' contract versions, whichever operations each
    ran."""
    if python['identity']['contract'] != rust['identity']['contract']:
        _differ(
            report,
            where,
            'identity.contract',
            python['identity']['contract'],
            rust['identity']['contract'],
            rule='core',
        )


def _core_agree_shared(
    report: Report,
    where: str,
    mine: dict[str, Any] | None,
    theirs: dict[str, Any] | None,
    shared: list[str],
) -> None:
    r""":func:`_core_agree` on the operations both hosts ran (``shared``)."""
    if mine is None or theirs is None:
        return
    count = len(shared)
    fetched = min(len(mine['fetches'] or []), len(theirs['fetches'] or []))
    held = {
        'inputs': (
            (mine['inputs'] or [])[:count],
            (theirs['inputs'] or [])[:count],
        ),
        'fetches': (
            (mine['fetches'] or [])[:fetched],
            (theirs['fetches'] or [])[:fetched],
        ),
    }
    for field, op in (
        ('described', 'describe_source'),
        ('accepted', 'validate_query'),
        ('materialized', 'materialize'),
    ):
        if op in shared:
            held[field] = (mine[field], theirs[field])
    for field, (a, b) in held.items():
        if a != b:
            _differ(
                report,
                where,
                f'core.{field}',
                None,
                None,
                show=False,
                rule='core',
            )


#: The dtype a column reaches a host with on a database that stores its type
#: in a form of its own (the type map's ``<database>_value``): each such
#: form is text there (``TEXT``, or ISO or JSON text), so the column is a
#: string. A stored form not listed here fails closed.
STORED_DTYPES = {
    'canonical': 'string',
    'digits': 'string',
    'iso': 'string',
    'json': 'string',
}


@functools.cache
def _loader() -> ModuleType:
    r"""The contract loader beside the runner, whose type map this reads."""
    path = REPO / 'contract/runners/databases.py'
    spec = importlib.util.spec_from_file_location('t21d_databases', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UnmappedColumn(Exception):
    r"""A column whose type the type map does not place on a database."""


def stored_schema(
    schema: Any,
    case: dict[str, Any],
    bases: dict[str, list[str]],
    database: str,
) -> Any:
    r"""The schema ``validate_query`` is handed for ``case`` on ``database``
    (T2.1d): on DuckDB the case's; elsewhere the case's with each column's
    dtype the one the contract loader's type map gives it there
    (``contract/runners/databases.py``'s ``TYPE_MAP``, the same entries as
    ``contract/type-map-vectors.json``'s ``map``): a type the database
    stores in a form of its own reaches a host as that form's dtype
    (:data:`STORED_DTYPES`), and any other type keeps the case's dtype.
    Each column's type is read from the case's DuckDB base and setup, as
    the loader reads it; a column it cannot place raises
    :class:`UnmappedColumn`."""
    if database == 'duckdb':
        return schema
    import duckdb

    loader = _loader()
    spec = case.get('input') or {}
    out = copy.deepcopy(schema)
    connection = duckdb.connect()
    try:
        for statement in [
            *bases.get(spec.get('base'), []),
            *(spec.get('setup') or []),
        ]:
            connection.execute(statement)
        for table in (out or {}).get('tables') or []:
            for column in table.get('columns') or []:
                where = f'{table.get("name")}.{column.get("name")}'
                try:
                    kind = (
                        connection.execute(
                            f'SELECT {loader.quote(column["name"])} FROM '
                            f'{loader.quote(table["name"])} LIMIT 0'
                        )
                        .to_arrow_table()
                        .schema.field(0)
                        .type
                    )
                    entry = loader.TYPE_MAP[loader.family(kind)]
                except Exception:  # noqa: BLE001 - named below
                    raise UnmappedColumn(
                        f'{where}: its type is not in the type map'
                    ) from None
                stored = entry.get(f'{database}_value')
                if stored is None:
                    continue
                if stored not in STORED_DTYPES:
                    raise UnmappedColumn(
                        f'{where}: no dtype for the stored form {stored}'
                    )
                column['dtype'] = STORED_DTYPES[stored]
    finally:
        connection.close()
    return out


def _case_inputs(
    report: Report,
    case: dict[str, Any],
    who: str,
    observation: dict[str, Any],
    database: str = 'duckdb',
    bases: dict[str, list[str]] | None = None,
) -> None:
    r"""A core runner's recorded inputs are the case's ``core`` inputs, and
    the schema ``materialize`` sampled is the runner's own: the description
    its ``describe_source`` answered, where the case makes no edit to it
    (T1.21d; the case names no schema, and with edits the hosts' agreement
    is the ``core`` rule's). Off DuckDB the schema ``validate_query`` is
    handed holds each column's dtype as the type map stores it there
    (:func:`stored_schema`, T2.1d: SQLite keeps a timestamp as text)."""
    core = observation['core']
    if core is None:
        return
    wanted = []
    if 'validate_query' in case.get('core', {}):
        spec = case['core']['validate_query']
        try:
            schema = stored_schema(spec['schema'], case, bases or {}, database)
        except UnmappedColumn as problem:
            report.fail(f'{case["id"]}: {who}: {problem}', 'inputs')
            return
        wanted.append(
            {
                'op': 'validate_query',
                'query': case['input']['pql'],
                'profile': spec['profile'],
                'schema': schema,
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
    if _null_side(path, excused) and (mine is None) != (theirs is None):
        # A value on one side, null or absent on the other, where an
        # owner-approved item says so (null_on_one_side, T2.1e).
        if _one(mine) or _one(theirs):
            return _MORE_EXACT, _MORE_EXACT
    if any(_matches(path, field) for field in excused):
        if _one(mine) and _one(theirs):
            return _MORE_EXACT, _MORE_EXACT
        return mine, theirs
    if isinstance(mine, dict) and isinstance(theirs, dict):
        pairs = {
            key: _left_out(mine[key], theirs[key], f'{path}/{key}', excused)
            for key in mine.keys() & theirs.keys()
        }
        for key in mine.keys() ^ theirs.keys():
            if _null_side(f'{path}/{key}', excused):
                pairs[key] = _left_out(
                    mine.get(key), theirs.get(key), f'{path}/{key}', excused
                )
        return (
            {
                key: pairs[key][0] if key in pairs else v
                for key, v in {**theirs, **mine}.items()
                if key in mine or key in pairs
            },
            {
                key: pairs[key][1] if key in pairs else v
                for key, v in {**mine, **theirs}.items()
                if key in theirs or key in pairs
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


def _null_side(path: str, excused: tuple[str, ...]) -> bool:
    r"""Whether an item excusing a value null or absent on one side names
    ``path`` (:data:`_NULL_SIDE`)."""
    return any(
        field.startswith(_NULL_SIDE)
        and _matches(path, field[len(_NULL_SIDE) :])
        for field in excused
    )


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
        who not in UNOBSERVING and outcome['stage'] != stage
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
        elif (
            stop is None
            and (case['id'], who) in report.fewer
            and _fewer_deprecations(
                observation['host']['warnings'], reference['host']['warnings']
            )
        ):
            report.note(
                f'{where}: host.warnings: fewer DeprecationWarnings than '
                f'{against} (KO2-83)'
            )
        else:
            report.fail(f'{where}: host.warnings: differ', 'warnings')
    if stop is not None and stop != 'predict':
        return
    # The rows as text (P2-T2.21-1): public.result_text, or a line's
    # result.text_digest, compared where both lines hold one; a line
    # recorded before it holds none and is not held to it, and where the
    # typed digest is excused that is named.
    mine_result, theirs_result = mine['result'], theirs['result']
    texts = []
    for public, result in ((mine, mine_result), (theirs, theirs_result)):
        held = public.get('result_text')
        if held is None and isinstance(result, dict):
            held = result.get('text_digest')
        texts.append(held)
    if isinstance(mine_result, dict):
        mine_result = {
            k: v for k, v in mine_result.items() if k != 'text_digest'
        }
    if isinstance(theirs_result, dict):
        theirs_result = {
            k: v for k, v in theirs_result.items() if k != 'text_digest'
        }
    if None not in texts and texts[0] != texts[1]:
        report.fail(f'{where}: public.result/text_digest: differs', 'result')
    elif None in texts and 'public.result/digest' in excused:
        report.note(
            f'{where}: public.result/text_digest: not compared, one line '
            'was recorded before it; record it again (P2-T2.21-1)'
        )
    found = (
        _path_of(mine_result, theirs_result, 'public.result', excused)
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


def _fewer_deprecations(mine: Any, theirs: Any) -> bool:
    r"""Whether ``mine`` is ``theirs`` with only repeats of a
    ``DeprecationWarning`` left out (KO2-83): ``theirs`` with some
    ``DeprecationWarning``s deleted and nothing else changed, each deleted
    one preceded in ``theirs`` by another ``DeprecationWarning``. So the
    first is always kept, no other class is dropped, added or moved, a
    ``DeprecationWarning`` never moves past another class, and ``mine``
    never holds more of them than ``theirs``. Anything but two lists of
    class names says no."""
    if not (
        isinstance(mine, list)
        and isinstance(theirs, list)
        and all(isinstance(x, str) for x in mine + theirs)
    ):
        return False
    at = 0
    for i, each in enumerate(theirs):
        if at < len(mine) and each == mine[at]:
            at += 1
        elif not (each == REPEATED_WARNING and REPEATED_WARNING in theirs[:i]):
            return False
    return at == len(mine)


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


def statements_differ(
    register: Path,
) -> dict[tuple[str, str], tuple[str, ...]]:
    r"""The public calls each ``case_outcomes`` row of ``register`` names
    under ``statements_differ``, by its case (the line's key) and runner,
    from an entry approved as :func:`more_exact` reads one (by the owner,
    under a row of this harness's decisions the owner approved): the call
    an owner-approved move turns from the reference's refusal into an
    answer, whose statements up to that refusal are no expectation of its
    own (T2.1e). Anything else, and a file it cannot read, opts in
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
            and isinstance(entry.get('case_outcomes'), list)
        ):
            continue
        for item in entry['case_outcomes']:
            if not isinstance(item, dict):
                continue
            case, runner = item.get('case'), item.get('runner')
            calls = item.get('statements_differ')
            if (
                isinstance(case, str)
                and isinstance(runner, str)
                and isinstance(calls, list)
            ):
                kept = tuple(c for c in calls if c in STAGES)
                found[(case, runner)] = found.get((case, runner), ()) + kept
    return found


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
            if item.get('null_on_one_side') is True:
                kept += tuple(_NULL_SIDE + field for field in kept)
            for who in runners:
                if who in MORE_EXACT_RUNNERS:
                    found[case, who] = found.get((case, who), ()) + kept
    return found


def fewer_deprecations(register: Path) -> frozenset[tuple[str, str]]:
    r"""The lines and core runners each entry of ``register`` lists under
    ``fewer_deprecations`` (T2.1s, KO2-83), from an entry approved by the
    owner whose ``decision_ref`` names a row of this harness's (the
    checkpoint's) decisions the owner approved, as :func:`more_exact` reads
    one, so a candidate cannot invent an approval: only a core runner's;
    anything else, and a file it cannot read, names nothing."""
    import yaml

    try:
        loaded = yaml.safe_load(register.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, yaml.YAMLError, RecursionError):
        return frozenset()
    approved = _owner_rows()
    entries = loaded.get('entries') if isinstance(loaded, dict) else None
    found: set[tuple[str, str]] = set()
    for entry in entries if isinstance(entries, list) else []:
        if not (
            isinstance(entry, dict)
            and entry.get('status') == 'approved'
            and entry.get('approved_by') == 'owner'
            and isinstance(entry.get('decision_ref'), dict)
            and isinstance(entry['decision_ref'].get('row'), str)
            and entry['decision_ref']['row'] in approved
            and isinstance(entry.get('fewer_deprecations'), list)
        ):
            continue
        for item in entry['fewer_deprecations']:
            if not isinstance(item, dict):
                continue
            case, runners = item.get('case'), item.get('runners')
            if isinstance(case, str) and isinstance(runners, list):
                found.update(
                    (case, who) for who in runners if who in MORE_EXACT_RUNNERS
                )
    return frozenset(found)


def core_only_columns(register: Path) -> dict[str, dict[str, frozenset[str]]]:
    r"""The columns each entry of ``register`` lists under
    ``core_only_columns``, by each of its ``cases`` (the bare case ID) and
    table, from an entry approved by the owner under a row of this
    harness's decisions the owner approved, as :func:`statements_differ`
    reads one (T2.1f): a column the core keeps and the reference leaves
    out (KO2-36). Each is ``<table>.<column>``, split at the first dot;
    anything else, and a file it cannot read, names nothing."""
    import yaml

    try:
        loaded = yaml.safe_load(register.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, yaml.YAMLError, RecursionError):
        return {}
    approved = _owner_rows()
    entries = loaded.get('entries') if isinstance(loaded, dict) else None
    found: dict[str, dict[str, frozenset[str]]] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not (
            isinstance(entry, dict)
            and entry.get('status') == 'approved'
            and entry.get('approved_by') == 'owner'
            and isinstance(entry.get('decision_ref'), dict)
            and isinstance(entry['decision_ref'].get('row'), str)
            and entry['decision_ref']['row'] in approved
            and isinstance(entry.get('core_only_columns'), list)
            and isinstance(entry.get('cases'), list)
        ):
            continue
        named: dict[str, set[str]] = {}
        for each in entry['core_only_columns']:
            if not isinstance(each, str):
                continue
            table, dot, column = each.partition('.')
            if table and dot and column:
                named.setdefault(table, set()).add(column)
        for case in entry['cases']:
            if not isinstance(case, str):
                continue
            mine = found.setdefault(case, {})
            for table, columns in named.items():
                mine[table] = mine.get(table, frozenset()) | columns
    return found


def _without_in_schema(schema: Any, named: dict[str, frozenset[str]]) -> Any:
    r"""A public schema with the named columns taken out, and the key, time
    column and links they make."""
    if not isinstance(schema, dict):
        return schema
    tables = []
    for table in schema.get('tables') or []:
        out = (
            set()
            if not isinstance(table, dict)
            else named.get(table.get('name'), set())
        )
        if not out:
            tables.append(table)
            continue
        kept = dict(table)
        kept['columns'] = [
            c
            for c in table.get('columns') or []
            if not (isinstance(c, dict) and c.get('name') in out)
        ]
        key = table.get('primary_key')
        if isinstance(key, str) and key in out:
            kept['primary_key'] = None
        elif isinstance(key, list):
            left = [k for k in key if k not in out]
            kept['primary_key'] = (
                None if not left else left[0] if len(left) == 1 else left
            )
        if table.get('time_column') in out:
            kept['time_column'] = None
        tables.append(kept)
    links = [
        link
        for link in schema.get('links') or []
        if not (
            isinstance(link, list)
            and len(link) == 4
            and (
                link[1] in named.get(link[0], set())
                or link[3] in named.get(link[2], set())
            )
        )
    ]
    return {**schema, 'tables': tables, 'links': links}


def _without_in_payload(value: Any, named: dict[str, frozenset[str]]) -> Any:
    r"""A payload with the named columns taken out of each table's columns
    and rows' cells, its schema's columns and key, and the relationships
    naming one."""
    if isinstance(value, list):
        return [_without_in_payload(item, named) for item in value]
    if not isinstance(value, dict):
        return value
    out: dict[str, Any] = {}
    for key, item in value.items():
        if key == 'related_tables' and isinstance(item, dict):
            out[key] = {
                table: _without_in_table(section, named.get(table, set()))
                for table, section in item.items()
            }
        elif key == 'relationships' and isinstance(item, list):
            out[key] = [
                link for link in item if not _relationship_names(link, named)
            ]
        else:
            out[key] = _without_in_payload(item, named)
    return out


def _without_in_table(section: Any, out: frozenset[str] | set[str]) -> Any:
    if not out or not isinstance(section, dict):
        return section
    kept = dict(section)
    columns = section.get('columns')
    if isinstance(columns, list):
        drop = {i for i, name in enumerate(columns) if name in out}
        kept['columns'] = [c for i, c in enumerate(columns) if i not in drop]
        rows = section.get('rows')
        if isinstance(rows, list):
            kept['rows'] = [
                [cell for i, cell in enumerate(row) if i not in drop]
                if isinstance(row, list)
                else row
                for row in rows
            ]
    elif isinstance(columns, dict):
        kept['columns'] = {k: v for k, v in columns.items() if k not in out}
    key = section.get('primary_key')
    if isinstance(key, list):
        kept['primary_key'] = [k for k in key if k not in out]
    elif isinstance(key, str) and key in out:
        kept['primary_key'] = None
    for name in ('time_column', 'end_time_column'):
        if section.get(name) in out:
            kept[name] = None
    return kept


def _relationship_names(link: Any, named: dict[str, frozenset[str]]) -> bool:
    r"""Whether a payload relationship has a named column at either end."""
    if not isinstance(link, dict):
        return False
    for table, columns in (
        (link.get('source_table'), link.get('source_columns')),
        (link.get('target_table'), link.get('target_columns')),
    ):
        if isinstance(columns, list) and any(
            c in named.get(table, set()) for c in columns
        ):
            return True
    return False


#: How a statement result read with a column only the core keeps marks
#: itself, so its row digests, which hash the column, are reported.
_CORE_ONLY = 'core_only'


def _without_in_results(
    statement: Any, named: dict[str, frozenset[str]]
) -> Any:
    r"""A captured statement with each result read with a named column of
    a table its text names (quoted, as every dialect quotes it) marked
    with those columns, which :func:`_results` takes out where the
    reference's result lacks them."""
    if not isinstance(statement, dict) or not statement.get('results'):
        return statement
    text = statement.get('sql') if isinstance(statement.get('sql'), str) else ''
    names = frozenset().union(
        *(
            columns
            for table, columns in named.items()
            if '"' + table.replace('"', '""') + '"' in text
        )
    )
    results = []
    for read in statement['results']:
        values = read.get('values') if isinstance(read, dict) else None
        columns = values.get('columns') if isinstance(values, dict) else None
        if not isinstance(columns, list) or not names & set(columns):
            results.append(read)
            continue
        marked = {**values, _CORE_ONLY: sorted(names & set(columns))}
        results.append({**read, 'values': marked})
    return {**statement, 'results': results}


def _core_only_out(read: dict[str, Any], theirs: dict[str, Any]) -> bool:
    r"""Takes out of ``read`` the columns marked as only the core's
    (:func:`_without_in_results`) that ``theirs`` did not read; whether it
    took any."""
    values = read['values']
    columns = values['columns']
    missing = {
        name
        for name in values.get(_CORE_ONLY) or ()
        if name not in theirs['values']['columns']
    }
    if not missing:
        return False
    drop = {i for i, name in enumerate(columns) if name in missing}
    values['columns'] = [c for i, c in enumerate(columns) if i not in drop]
    if isinstance(values.get('types'), list):
        values['types'] = [
            t for i, t in enumerate(values['types']) if i not in drop
        ]
    if isinstance(read.get('schema'), list):
        read['schema'] = [
            f for i, f in enumerate(read['schema']) if i not in drop
        ]
    return True


def without_core_only(
    observation: dict[str, Any], named: dict[str, frozenset[str]]
) -> dict[str, Any]:
    r"""A core runner's observation with the columns only the core keeps
    taken out, as the reference never had them (T2.1f): from its public
    schema and payloads, the core's predicted body and each of several
    calls' bodies (T2.1s), and its statements' results."""
    if not named:
        return observation
    found = copy.deepcopy(observation)
    public = found.get('public')
    if isinstance(public, dict):
        public['schema'] = _without_in_schema(public.get('schema'), named)
        for each in public.get('payloads') or []:
            if isinstance(each, dict):
                each['parsed'] = _without_in_payload(each.get('parsed'), named)
    predicted = (found.get('core') or {}).get('predicted')
    if isinstance(predicted, dict) and predicted.get('body') is not None:
        predicted['body'] = _without_in_payload(predicted['body'], named)
    if isinstance(predicted, dict) and isinstance(predicted.get('calls'), list):
        # Every call's body where predict made several (T2.1g), each as the
        # single call's is (T2.1s).
        predicted['calls'] = [
            _without_in_payload(body, named) for body in predicted['calls']
        ]
    capture = (found.get('driver') or {}).get('capture')
    if isinstance(capture, dict):
        capture['statements'] = [
            _without_in_results(statement, named)
            for statement in capture.get('statements') or []
        ]
    return found


def without_left_out_warning(
    reference: dict[str, Any],
    named: dict[str, frozenset[str]],
    observation: dict[str, Any],
) -> dict[str, Any]:
    r"""The reference's observation with its from_source warning of the
    columns it leaves out taken out for each table whose named column the
    core runner kept (its public schema holds it): one ``SDMWarning`` per
    such table, the first of its warnings of that class. A runner that
    left the column out, as the reference did, is held to the warning."""
    found = copy.deepcopy(reference)
    schema = (observation.get('public') or {}).get('schema') or {}
    kept = 0
    for table in schema.get('tables') or [] if isinstance(schema, dict) else []:
        if not isinstance(table, dict):
            continue
        names = {
            c.get('name')
            for c in table.get('columns') or []
            if isinstance(c, dict)
        }
        if names & named.get(table.get('name'), frozenset()):
            kept += 1
    host = found.get('host')
    if not isinstance(host, dict):
        return found
    shown = list(host.get('warnings') or [])
    origins = host.get('origins')
    # The warnings' recorded origins, where the runner keeps them, lose the
    # same entries, so the two lists stay aligned (HOST-5 reads them);
    # origins that never matched are left as they are, still refused.
    aligned = isinstance(origins, list) and len(origins) == len(shown)
    kept_origins = list(origins) if aligned else origins
    for _ in range(kept):
        if 'SDMWarning' not in shown:
            break
        at = shown.index('SDMWarning')
        del shown[at]
        if aligned:
            del kept_origins[at]
    host['warnings'] = shown
    if aligned:
        host['origins'] = kept_origins
    return found


def _core_payload(
    report: Report,
    case: dict[str, Any],
    who: str,
    observation: dict[str, Any],
    reference: dict[str, Any] | None,
    stop: str | None = None,
) -> None:
    r"""The calls the core wrote are the requests the reference sent, on
    ``rust-host``, and the requests the endpoint received, on
    ``python-core``, call by call, in order: the ``payload`` rule on the
    core; on ``rust-host``, held to the reference only where it reached
    ``predict`` when past its ``stop`` (:func:`past_stop`). One call is
    ``core.predicted.body``; several are ``core.predicted.calls``, every
    call's body in order (T2.1g)."""
    predicted = (observation.get('core') or {}).get('predicted')
    if not predicted or predicted.get('body') is None:
        return
    several = isinstance(predicted.get('calls'), list)
    named = 'core.predicted.calls' if several else 'core.predicted.body'
    bodies = [
        _payload(body, report.paths)
        for body in (predicted['calls'] if several else [predicted['body']])
    ]
    if who == 'python-core':
        sent = [
            _payload(each['parsed'], report.paths)
            for each in (observation.get('public') or {}).get('payloads', [])
        ]
        found = _path_of(bodies, sent, named)
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
            bodies,
            wanted,
            named,
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


#: The reference's two reads on Postgres whose results it never uses, which
#: the core does not make (KO2-58, approved by the owner on 2026-10-17):
#: the current schema's name, and each table's row count.
_POSTGRES_UNUSED_READS = (
    re.compile(r'SELECT current_schema\(\)'),
    re.compile(r'SELECT COUNT\(\*\) FROM "(?:[^"]|"")+"\."(?:[^"]|"")+"'),
)


def _expected_statements(
    statements: list[dict[str, Any]], database: str
) -> list[dict[str, Any]]:
    r"""The reference's ``statements`` of one public call that a core runner
    is expected to match: on Postgres, all but the two reads whose results
    the reference never uses (:data:`_POSTGRES_UNUSED_READS`, KO2-58), each
    with no parameters; every one elsewhere."""
    if database != 'postgres':
        return statements
    return [
        each
        for each in statements
        if not (
            each['parameters'] is None
            and any(
                shape.fullmatch(each['sql']) for shape in _POSTGRES_UNUSED_READS
            )
        )
    ]


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
        # A column only the core keeps, which the reference's result lacks,
        # is taken out of the core's (T2.1f); its digests hash it.
        marked = False
        if a['values'].get(_CORE_ONLY):
            a = copy.deepcopy(a)
            marked = _core_only_out(a, b)
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
        if marked and va[key] != vb[key]:
            # The digests hash a column only the core keeps (T2.1f).
            report.note(
                f'{at}: rows read with a column only the core keeps; their '
                'digests are not compared'
            )
        elif va[key] != vb[key]:
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
    answered: bool = False,
) -> None:
    r"""Decision H9 against the reference: ``sql.count`` and ``sql.result``,
    with ``sql.text``, ``sql.parameters`` and ``sql.bindings`` reported;
    past the reference's ``stop`` (:func:`past_stop`), at that stage the
    reference's statements and bindings each against the runner's at its
    position, and none after. Where the approved move is to a success and
    its owner-approved entry says that call's statements differ
    (``answered``: ``statements_differ`` on its ``case_outcomes`` row,
    T2.1e), the call answers where the reference refused, so the
    reference's statements up to its refusal are no expectation of its
    own: its ``sql.count`` and ``sql.result`` differences are reported as
    diagnostic, every other rule and every other call as before. Without
    that opt-in, the reference's statements at its stop gate as ever."""
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
        since = len(report.failures)
        ours = mine.get(stage, [])
        refs = _expected_statements(theirs.get(stage, []), database)
        if stage == stop and len(ours) > len(refs):
            # Those after the reference's own came after it stopped.
            ours = ours[: len(refs)]
        if len(ours) != len(refs):
            report.fail(
                f'{where}: {stage}: {len(ours)} statements, the reference '
                f'{len(refs)}',
                'sql.count',
            )
        else:
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
        if answered and stage == stop:
            # That call's own sql.count and sql.result, as diagnostic notes.
            report.demote(
                since,
                keep=tuple(
                    rule
                    for rule in RULES
                    if rule not in ('sql.count', 'sql.result')
                ),
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


def statements_refused(
    report: Report,
    key: str,
    who: str,
    wanted: dict[str, Any] | None,
    reference: dict[str, Any] | None,
) -> str | None:
    r"""The call ``who`` refuses at on the line ``key`` where the reference
    answered, when the owner-approved entry's row moving it there names
    that call under ``statements_differ`` (T2.1e): the reference's
    statements there and after, and its warnings, are no expectation of a
    call the runner refuses by that approval; None otherwise."""
    theirs = (reference or {}).get('outcome') or {}
    if (
        wanted is None
        or wanted.get('result') != 'refusal'
        or theirs.get('result') != 'success'
    ):
        return None
    calls = report.statements_differ.get((key, who), ())
    stage = wanted.get('stage') or (calls[0] if len(calls) == 1 else None)
    return stage if stage in calls else None


def statements_earlier(
    report: Report,
    key: str,
    who: str,
    wanted: dict[str, Any] | None,
    reference: dict[str, Any] | None,
) -> str | None:
    r"""The call ``who`` refuses at on the line ``key``, with its own code,
    before the call the reference refused at, when the owner-approved
    entry's row moving the refusal there names that call under
    ``statements_differ`` (T2.1r): the line is compared up to that call,
    its statements there and its warnings reported as diagnostic, and the
    reference's later calls, never made, are no expectation of its own;
    None otherwise. A move to a later call, and a row naming another
    call, never are."""
    theirs = (reference or {}).get('outcome') or {}
    if (
        wanted is None
        or wanted.get('result') != 'refusal'
        or theirs.get('result') != 'refusal'
        or theirs.get('stage') not in STAGES
    ):
        return None
    calls = report.statements_differ.get((key, who), ())
    stage = wanted.get('stage') or (calls[0] if len(calls) == 1 else None)
    if stage not in calls or stage not in STAGES:
        return None
    if STAGES.index(stage) >= STAGES.index(theirs['stage']):
        return None
    return str(stage)


def statements_kept(
    report: Report,
    key: str,
    who: str,
    wanted: dict[str, Any] | None,
    reference: dict[str, Any] | None,
) -> str | None:
    r"""The call both the reference and ``who`` refuse at on the line
    ``key``, with one code, when the owner-approved entry's row keeping
    that refusal names the call under ``statements_differ`` (KO2-68): the
    runner reads less before the same refusal, so the reference's
    statements there are no expectation of its own; None otherwise. With
    no effective expectation the reference's outcome is the one held."""
    theirs = (reference or {}).get('outcome') or {}
    stage = theirs.get('stage')
    if wanted is None:
        wanted = theirs
    if (
        wanted.get('result') != 'refusal'
        or theirs.get('result') != 'refusal'
        or wanted.get('code') != theirs.get('code')
        or wanted.get('stage') not in (None, stage)
    ):
        return None
    return (
        stage if stage in report.statements_differ.get((key, who), ()) else None
    )


def statements_opted_in(
    report: Report,
    key: str,
    who: str,
    stop: str | None,
    wanted: dict[str, Any] | None,
) -> bool:
    r"""Whether ``who``'s statements at the reference's ``stop`` on the line
    ``key`` are no expectation of its own (T2.1e): its effective
    expectation is an approved move past that stop (:func:`past_stop`) to
    a success, and the owner-approved entry's row names that call under
    ``statements_differ``. A move to a later refusal, and a row that does
    not opt in, never are."""
    return (
        stop is not None
        and (wanted or {}).get('result') == 'success'
        and stop in report.statements_differ.get((key, who), ())
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
    on_duckdb: bool = True,
    earlier: bool = False,
) -> None:
    r"""H1's comparison in a cell other than the reference's: ``python-core``
    and ``python-old`` observed in the same cell, and their host fields
    equal (the ``schema``, ``warnings``, ``result`` and ``payload`` rules),
    in place of the reference's, which are reported. Either runner alone
    there is a failure: the cell's comparison needs both. ``stop``, where
    python-old stopped as the reference did and an approved move took
    python-core past it (E114): held only that far, as :func:`_host_fields`
    holds a runner past the reference's stop. ``earlier``, where an
    owner-approved row moved python-core's refusal to an earlier call
    (:func:`statements_earlier`, T2.1r) and ``stop`` is that call: the
    warnings are reported, not gated, as against the reference, since
    python-old stands in for the reference's later calls.

    Off DuckDB python-old does not run (its SQLite sampling lacks the
    reference's seeding, main's f0f324ce; T2.1a fix 4), so there
    python-core alone in another cell is no failure: its host fields stay
    reported against the reference, and every core rule still gates it
    (T2.4: the SQLite and Postgres jobs in every cell)."""
    cells = {
        each['identity'].get('cell') for each in (core, old) if each is not None
    }
    other = cells - {None, REFERENCE_CELL}
    if not other:
        return
    if old is None and not on_duckdb:
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
    # python-old leaves out what only the core keeps, as the reference does
    # (T2.1f): taken out of the core's side, and its warning of leaving it
    # out out of python-old's, as against the reference.
    named = report.core_only.get(case['id'].partition('@')[0], {})
    if named:
        old = without_left_out_warning(old, named, core)
        core = without_core_only(core, named)
    since = len(report.failures)
    _host_fields(
        report, case, 'python-core', core, old, 'python-old', stop=stop
    )
    if earlier:
        report.demote(since, keep=tuple(r for r in RULES if r != 'warnings'))


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
    live: bool = False,
) -> Report:
    r"""The comparison of ``files``' observations on ``database``, a
    database or a warehouse leg. A warehouse leg is judged offline, on the
    stand-in, unless ``live`` (E-1, H9 in full): offline its core runners
    carry the stand-in's verdict, the policy's ``standin`` rules are only
    reported, and a live-only item is listed rather than passed or failed
    (:func:`_standin`)."""
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
    report.fewer = fewer_deprecations(improvements or register_file)
    report.statements_differ = statements_differ(improvements or register_file)
    report.core_only = core_only_columns(improvements or register_file)
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
    if database not in runner.LEGS:
        report.fail(f'--database: {database} is not one of {runner.LEGS}')
        return report
    offline = database in runner.WAREHOUSES and not live
    codes = tuple(policy['standin']['codes'])
    if offline:
        # H9 narrowed on a stand-in leg, and only there (KO3-45).
        report.rules = {
            **report.rules,
            **{rule: 'diagnostic' for rule in policy['standin']['diagnostic']},
        }
    elif live and database not in runner.WAREHOUSES:
        report.fail(f'--live: {database} is not a warehouse leg')
        return report
    cases = [case for case, _ in loaded if database in runner.case_legs(case)]
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
        live_only: dict[str, str] = {}
        stops: dict[str, str | None] = {}
        earliers: dict[str, str | None] = {}
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
            since = len(report.failures)
            code, flagged = _standin(
                report, case['id'], who, observation, codes, offline
            )
            _shape(report, case, who, observation)
            _inputs(report, case, who, observation, digests)
            wanted = effective.get(case['id'], {}).get(who)
            # An approved move of the reference's refusal to an earlier
            # call (T2.1r) holds the line to the calls up to that one, on
            # every database; off DuckDB the line is held to the calls it
            # makes (T2.1e).
            earlier = statements_earlier(report, case['id'], who, wanted, ref)
            if earlier is not None:
                held = core_held(case, observation, earlier)
            elif on_duckdb:
                held = case
            else:
                held = core_held(case, observation, line_stop(wanted, ref))
            _core_consistent(report, held, who, observation, database)
            stop = past_stop(
                who,
                wanted,
                who in effective.get(MOVED, {}).get(case['id'], []),
                ref,
            )
            stops[who] = earlier or stop
            earliers[who] = earlier
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
            _route(report, held, who, observation)
            _case_inputs(report, held, who, observation, database, bases)
            if reach is not None:
                _reach(report, case, who, observation, reach)
            found = counts(observation)
            if found is not None:
                report.info.append(f'{case["id"]}: {who}: {found}')
            # Against the reference, what only the core keeps is taken out
            # of the core's side, and the reference's warning of leaving it
            # out of the reference's (T2.1f); between the core runners,
            # below, nothing is.
            seen, against = observation, ref
            named = report.core_only.get(case['id'].partition('@')[0], {})
            if named and who in MORE_EXACT_RUNNERS and ref is not None:
                seen = without_core_only(observation, named)
                against = without_left_out_warning(ref, named, observation)
            _core_payload(report, case, who, seen, against, earlier or stop)
            refused = (
                statements_refused(report, case['id'], who, wanted, ref)
                or earlier
            )
            kept = statements_kept(report, case['id'], who, wanted, ref)
            if against is not None:
                cell = observation['identity'].get('cell')
                if who in PYTHON and cell not in (None, REFERENCE_CELL):
                    # H1: in a cell other than the reference's, the host
                    # fields against the reference are reported; python-core
                    # against python-old gates them (below).
                    since = len(report.failures)
                    _host_fields(
                        report, case, who, seen, against, stop=earlier or stop
                    )
                    report.reported(
                        since, f'cell {cell}, against the reference; H1'
                    )
                elif who in PYTHON:
                    since = len(report.failures)
                    _host_fields(
                        report, case, who, seen, against, stop=earlier or stop
                    )
                    if refused is not None:
                        # The warnings of a call refused by the approval.
                        report.demote(
                            since,
                            keep=tuple(r for r in RULES if r != 'warnings'),
                        )
                _sql(
                    report,
                    case,
                    who,
                    seen,
                    against,
                    ordered,
                    refused or kept or stop,
                    large,
                    database,
                    answered=refused is not None
                    or kept is not None
                    or statements_opted_in(
                        report, case['id'], who, stop, wanted
                    ),
                )
                theirs = ref['identity']['versions'].get('duckdb')
                mine = observation['identity']['versions'].get('duckdb')
                # Java's host opens DuckDB only where its leg does.
                if mine != theirs and not offline and mine is not None:
                    # Offline, a warehouse leg's reference ran on the
                    # warehouse, and DuckDB is only the stand-in's engine.
                    _differ(
                        report,
                        f'{case["id"]}: {who}',
                        'identity.versions.duckdb',
                        theirs,
                        mine,
                        rule='route',
                    )
            if code is not None:
                _live_only(
                    report,
                    case['id'],
                    who,
                    code,
                    since,
                    'the stand-in could not answer',
                )
                live_only[who] = code
            elif flagged and _differences(report, since):
                # A flagged answer that differs may be the local engine's
                # casts, not the core's: live only. One that matches passes.
                _live_only(
                    report,
                    case['id'],
                    who,
                    flagged[0],
                    since,
                    'a flagged answer differs from the reference',
                )
                live_only[who] = flagged[0]
        paired = len(report.failures)
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
        _cell_pair(
            report,
            case,
            python,
            old,
            pair_stop,
            on_duckdb,
            earlier=pair_stop is not None
            and pair_stop == earliers.get('python-core'),
        )
        rust = observed.get('rust-host', {}).get(case['id'])
        if python is not None and rust is not None:
            _core_agree(
                report,
                case,
                python,
                rust,
                None
                if on_duckdb
                else line_stop(
                    effective.get(case['id'], {}).get('python-core'), ref
                ),
            )
            _sql_core(report, case, python, rust, ordered, large, database)
        if live_only:
            # The core runners' agreement on a live-only case is reported:
            # one side's answer there is the local engine's or none.
            _reported_rules(
                report, paired, 'live only: ' + ', '.join(sorted(live_only))
            )
        if not gating:
            report.demote(before, keep=('reach',) if reach else ())
    return report


def verdict_problems(
    live_only: dict[str, str], manifests: list[Path], candidate: str
) -> list[str]:
    r"""Why the locked run fails on the live-only items (Phase 3 design
    choice 2; locked check 1): each needs E-1's verdict ``equal`` for its
    key in a capture manifest of the core's live run (``contract/captures/``,
    ``{"recording_commit": SHA, "verdicts": {"<case>@<leg>": "equal" |
    "differs"}}``) recorded on ``candidate``, the final candidate; none, one
    recorded on another commit, or ``differs`` fails, naming the item."""
    found: dict[str, list[tuple[str, str]]] = {}
    problems = []
    for path in manifests:
        try:
            manifest = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError, RecursionError) as error:
            problems.append(f'{path.name}: unreadable ({type(error).__name__})')
            continue
        verdicts = (
            manifest.get('verdicts') if isinstance(manifest, dict) else None
        )
        if not isinstance(verdicts, dict):
            continue
        commit = str(manifest.get('recording_commit'))
        for key, verdict in verdicts.items():
            found.setdefault(str(key), []).append((commit, str(verdict)))
    for key, code in sorted(live_only.items()):
        mine = [
            verdict
            for commit, verdict in found.get(key, [])
            if commit == candidate
        ]
        if not mine:
            problems.append(
                f'{key}: live only ({code}), and no E-1 verdict for the final '
                f'candidate {candidate}'
                + (' (only for another commit)' if found.get(key) else '')
            )
        elif any(verdict != 'equal' for verdict in mine):
            problems.append(
                f'{key}: live only ({code}), and E-1 found it differs'
            )
    return problems


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
    parser.add_argument(
        '--live',
        action='store_true',
        help='a warehouse leg run live (E-1), judged by every rule, with no '
        'stand-in verdict',
    )
    parser.add_argument(
        '--verdicts',
        type=Path,
        nargs='+',
        default=[],
        metavar='MANIFEST',
        help="the core's live capture manifests (E-1), read with --candidate",
    )
    parser.add_argument(
        '--candidate',
        help='the final candidate: each live-only item needs an E-1 verdict '
        'equal recorded on it (the locked run)',
    )
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
        live=arguments.live,
    )
    for text in report.info:
        print(f'statements per public call: {text}')
    for text in report.notes:
        print(text)
    for text in report.texts:
        print(f'DIFFERS {text}')
    for key, code in sorted(report.live_only.items()):
        print(f'LIVE ONLY {key}: {code}')
    if report.live_only:
        print(
            f'compare: {len(report.live_only)} live-only items, neither passed '
            'nor failed offline'
        )
    if arguments.candidate is not None:
        refused = verdict_problems(
            report.live_only, arguments.verdicts, arguments.candidate
        )
        for text in refused:
            print(f'NO VERDICT {text}')
        if refused:
            print(
                f'compare: {len(refused)} live-only items E-1 has not discharged'
            )
            return 1
    if report.failures:
        print(f'compare: {len(report.failures)} differences')
        return 1
    print('compare: no differences')
    return 0


if __name__ == '__main__':
    sys.exit(main())
