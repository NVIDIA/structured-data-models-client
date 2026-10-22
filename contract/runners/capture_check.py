#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Checks the driver capture on a runner's observations (design, "Driver
capture"; decision Q2):

    python contract/runners/capture_check.py --on FILE --off FILE
        [--require CASE]...

``--on`` holds a run with the capture, ``--off`` the same cases run with
``--no-capture``; each ``--require`` names a case both must hold (T1.11a's
evidence is ``--require base.regression``). Empty files, a case id twice in
one file, the two files holding different cases, or a capture-off
observation in ``--on`` fail. It holds, case by case:

- completeness: the capture recorded every statement it saw and could
  record every result read, its statement and binding counts equal
  DuckDB's own query log, and each statement's text equals the logged
  text where DuckDB logs it as written (every statement but a ``PRAGMA``,
  which DuckDB logs as its table function);
- conservation: capture on and off give the same public projection (the
  outcome, warnings, public ``Schema``, each request the scripted endpoint
  received, and the public result) and the same query-log counts, so the
  capture changed no value and no execution count.

Three more read the warehouse legs' captures (Phase 3, design choice 2):

    python contract/runners/capture_check.py captures --cases PATH...
        [--leg LEG]... [--locked] MANIFEST...
    python contract/runners/capture_check.py recording --cases PATH...
        --capture FILE... [--leg LEG]... [--locked]
    python contract/runners/capture_check.py commit --candidate SHA [REV]

``captures`` reads the capture manifests T3.3's writer writes beside the
recorded tables (``{"format": "sdm-capture/1", "recording_commit",
"reference_commit", "legs", "cases": {"<case>@<leg>": {"input_hash",
...}}}``): a manifest of another format version, or missing a part, is
refused by name; each case and warehouse leg the cases run on with no
capture of its current input (``run_python.input_hash``) is listed
``pending live: <case>@<leg>``, and fails the run with ``--locked``
(locked check 1). ``recording`` holds the recording mode's own output
(``run_python.py --record``, one line per case and leg, the writer's
input) complete (:func:`capture_problems`). ``commit`` holds a
captures-only commit after the final run (design, "The live-test plan"):
REV (default ``HEAD``) changes only ``contract/captures/**``, deletes
nothing, and every manifest or capture line it adds or changes names the
final candidate SHA as its recording commit.

A payload is compared parsed. Only the value of ``metadata.request_id``,
which the client draws at random for every request, is left out
(:data:`RANDOM_PATHS`): the field must be present or absent in both runs
alike, with a value of the same JSON type. The payload's raw SHA-256
differs with that value, so it is not compared. The exclusion is the one
entry of ``contract/compare-policy.yaml``'s ``exclusions``, approved by the
owner on 2026-10-10 (decision E35), which :func:`policy` reads and checks.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

POLICY = Path(__file__).resolve().parents[1] / 'compare-policy.yaml'
#: The repository this file is in.
ROOT = Path(__file__).resolve().parents[2]
#: Where every phase's decisions file is, ``phase-N/decisions.md``: the
#: decisions an exclusion must name, each by a row whose status reads
#: "approved by owner", are read from all of them (T2.0), Phase 1's rows in
#: force and each later phase's added (:func:`phase_decisions`).
DECISIONS_DIR = 'docs/design/rust-core'
#: A phase's decisions file, as a repository path; the phase's number.
DECISIONS_FILE = re.compile(
    r'docs/design/rust-core/phase-(0|[1-9][0-9]*)/decisions\.md'
)
#: A phase's design, whose phase must have a decisions file.
DESIGN_FILE = re.compile(
    r'docs/design/rust-core/phase-(0|[1-9][0-9]*)/design\.md'
)
#: The decisions file of Phase 1, the first phase with a ratchet: every
#: tree since holds it, and its rows stay in force.
FIRST_DECISIONS = 'docs/design/rust-core/phase-1/decisions.md'
#: The policy's format, and what a rule may be.
POLICY_FORMAT = 'sdm-compare-policy/1'
RULE_CLASSES = ('gate', 'diagnostic')
STAGES = ('from_source', 'materialize', 'predict')
#: The policy's sections: the rules, the exclusions, the ordered statements
#: and the cases whose reach gates (T1.11e); from T1.14 the projection the
#: baselines hold (``baseline``, read by scripts/rust_core/ratchet.py) and
#: the relaxations of this file an owner-approved row allows.
SECTIONS = (
    'format',
    'rules',
    'exclusions',
    'ordered',
    'reach',
    'baseline',
    'relaxations',
    'standin',
    'sampled',
)
#: A field path of an observation, dotted (``outcome.code``).
FIELD = re.compile(r'[a-z_]+(\.[a-z_]+)*')


#: The exclusions the owner has approved, by decision, each with the one
#: path it leaves the value of out: an allow-list the loader holds every
#: exclusion to, beside the decision's own row. Adding one needs an
#: owner-approved decisions row (decisions.md's preamble) and a review of
#: this protected file.
APPROVED_EXCLUSIONS: dict[str, tuple[str, ...]] = {
    # The request id the client draws at random (approved 2026-10-10).
    'E35': ('metadata', 'request_id'),
}


#: What the policy's ``standin`` section may hold, by the owner's
#: approval: the stand-in's reason codes, exactly the design's blind-spot
#: rows B1, B2 and B4 (Phase 3 design choice 2), and the rules only reported
#: on a stand-in leg, H9's statement count and results (KO3-45, approved by
#: owner 2026-10-22). Widening either needs the owner and a review of this
#: protected file.
APPROVED_STANDIN = {
    'codes': ('B1', 'B2', 'B4'),
    'diagnostic': ('sql.count', 'sql.result'),
}


#: What the policy's ``sampled`` section may hold, by the owner's approval
#: ("2026-10-22 evening: Snowflake sampling, option C"): one case, on the
#: Snowflake leg, whose comparison holds only its sampling statement and the
#: result's row count, since the core keeps the old client's unseeded
#: ``SAMPLE ROW (n ROWS)``. A second case or another leg needs the owner and
#: a review of this protected file.
APPROVED_SAMPLED = {
    'decision': '2026-10-22 evening: Snowflake sampling, option C',
    'leg': 'snowflake',
    'cases': 1,
}


class PolicyError(Exception):
    r"""``compare-policy.yaml`` is not a policy this checker can apply."""


class DecisionsError(Exception):
    r"""The phases' decisions files cannot be read as a whole: each problem
    named by its file (and row), so nothing they approve counts."""


#: A decision row's start: its ID cell.
ROW = re.compile(r'\| ([A-Z][A-Z0-9-]*) \|')


def phase_decisions(
    paths: Iterable[str], read: Callable[[str], str | None]
) -> str:
    r"""Every phase's decisions, as one text :func:`decision_rows` reads:
    each ``phase-N/decisions.md`` among ``paths`` (repository paths under
    :data:`DECISIONS_DIR`), in phase order, ``read`` giving a path's text
    (None when it is missing). It fails closed, raising
    :class:`DecisionsError` that names each problem's file, when:

    - a file a phase needs is missing: Phase 1's, any phase's below the
      highest phase with one (a gap), and a phase's that has a
      ``design.md``;
    - a file is not UTF-8 text or cannot be read (``read`` raising) or
      reads as None though listed;
    - a decision row is malformed: no status in its last cell;
    - a row's ID is in two places, in one file or in two, so which row
      approves would be a guess."""
    listed = set(paths)
    files: dict[int, str] = {}
    designs: set[int] = set()
    for path in listed:
        found = DECISIONS_FILE.fullmatch(path)
        if found:
            files[int(found.group(1))] = path
        found = DESIGN_FILE.fullmatch(path)
        if found:
            designs.add(int(found.group(1)))
    needed = {1} | set(range(1, max(files, default=1) + 1)) | designs
    problems = [
        f'{DECISIONS_DIR}/phase-{number}/decisions.md: missing'
        for number in sorted(needed - set(files))
    ]
    texts: list[str] = []
    seen: dict[str, str] = {}
    for number in sorted(files):
        path = files[number]
        try:
            text = read(path)
        except UnicodeDecodeError:
            problems.append(f'{path}: not UTF-8')
            continue
        except Exception as error:  # the reader's own, named: fail closed
            problems.append(f'{path}: unreadable ({error})')
            continue
        if text is None:
            problems.append(f'{path}: missing')
            continue
        for line in text.splitlines():
            found = ROW.match(line)
            if not found:
                continue
            row = found.group(1)
            cells = [
                cell.strip() for cell in line.strip().strip('|').split('|')
            ]
            if len(cells) < 2 or not cells[-1]:
                problems.append(f'{path}: row {row} is malformed: no status')
            if row in seen:
                problems.append(
                    f'{path}: row {row} is also in {seen[row]}'
                    if seen[row] != path
                    else f'{path}: row {row} is there twice'
                )
            seen.setdefault(row, path)
        texts.append(text)
    if problems:
        raise DecisionsError('; '.join(problems))
    return '\n'.join(texts)


def repository_decisions(root: Path = ROOT) -> str:
    r""":func:`phase_decisions` of the files under ``root``'s
    :data:`DECISIONS_DIR`."""
    base = root / DECISIONS_DIR
    paths = [
        path.relative_to(root).as_posix()
        for path in sorted(base.glob('phase-*/*.md'))
        if path.is_file()
    ]

    def read(path: str) -> str | None:
        file = root / path
        return file.read_text(encoding='utf-8') if file.is_file() else None

    return phase_decisions(paths, read)


def _decision_rows(path: Path | None) -> dict[str, tuple[str, str]]:
    r"""The rows by ID of every phase's decisions file (``None``: this
    repository's, :func:`repository_decisions`), or of the one file
    ``path`` names: each row's whole text and its status, the last cell."""
    if path is None:
        try:
            return decision_rows(repository_decisions())
        except DecisionsError as error:
            raise PolicyError(f'the decisions: {error}') from None
    return decision_rows(path.read_text(encoding='utf-8'))


def decision_rows(text: str) -> dict[str, tuple[str, str]]:
    r"""The rows of a decisions file's text by ID, as
    :func:`_decision_rows`; the ratchet reads the checkpoint's this way."""
    rows = {}
    for line in text.splitlines():
        found = ROW.match(line)
        if found:
            cells = [
                cell.strip() for cell in line.strip().strip('|').split('|')
            ]
            rows[found.group(1)] = (line, cells[-1])
    return rows


def _exclusion_problem(
    exclusion: dict[str, Any], rows: dict[str, tuple[str, str]]
) -> str | None:
    r"""Why an exclusion is not one the owner approved, if it is not: its
    decision and path must be an entry of :data:`APPROVED_EXCLUSIONS`, and
    its decision a row of decisions.md whose status reads its own
    "approved by owner <date>", and its path the one that row approves,
    written there in backticks as dotted names, never a strict prefix of a
    path the row names (``[metadata]`` for ``metadata.request_id``), which
    would leave out more than was approved."""
    status = str(exclusion['status'])
    if not re.fullmatch(r'approved by owner \d{4}-\d{2}-\d{2}', status):
        return 'not owner-approved'
    decision = exclusion['decision']
    if not isinstance(decision, str) or decision not in APPROVED_EXCLUSIONS:
        return 'its decision approves no exclusion'
    if decision not in rows:
        return 'its decision is not a row of decisions.md'
    text, row_status = rows[decision]
    if status not in row_status:
        return f'{decision} does not read {status!r}'
    path = exclusion['path']
    if (
        not isinstance(path, list)
        or not path
        or not all(isinstance(part, str) and part for part in path)
    ):
        return 'its path'
    if tuple(path) != APPROVED_EXCLUSIONS[decision]:
        return f'{decision} approves another path'
    dotted = '.'.join(path)
    named = set(re.findall(r'`([A-Za-z0-9_.]+)`', text))
    if dotted not in named:
        return f'{decision} does not name {dotted}'
    if any(other.startswith(dotted + '.') for other in named):
        return f'{dotted} leaves out more than {decision} approves'
    return None


def policy(
    path: Path = POLICY, decisions: Path | None = None
) -> dict[str, Any]:
    r"""The comparator's policy, checked: every rule a gate or diagnostic;
    every exclusion value only (``keeps: [presence, type]``), of a payload,
    naming a decision whose row in ``decisions`` (by default every phase's
    decisions file, :func:`repository_decisions`) reads "approved by owner"
    with the exclusion's own status, and the one path that row approves
    (:func:`_exclusion_problem`; an exclusion waits for the owner,
    decisions.md's preamble, and an unapproved or widened one is refused
    here, never applied); ``ordered`` statement positions by case and
    stage; ``reach`` stages by case."""
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(loaded, dict) or set(loaded) != set(SECTIONS):
        raise PolicyError('not a policy: its sections')
    if loaded['format'] != POLICY_FORMAT:
        raise PolicyError(f'not {POLICY_FORMAT}')
    rows = _decision_rows(decisions)
    rules = loaded['rules']
    if not isinstance(rules, dict) or any(
        value not in RULE_CLASSES for value in rules.values()
    ):
        raise PolicyError('a rule is neither a gate nor diagnostic')
    for exclusion in loaded['exclusions']:
        if set(exclusion) != {
            'id',
            'decision',
            'status',
            'in',
            'path',
            'keeps',
        }:
            raise PolicyError('an exclusion: its fields')
        if exclusion['in'] != 'payload':
            raise PolicyError(f'exclusion {exclusion["id"]}: not of a payload')
        if exclusion['keeps'] != ['presence', 'type']:
            raise PolicyError(f'exclusion {exclusion["id"]}: not value only')
        problem = _exclusion_problem(exclusion, rows)
        if problem is not None:
            raise PolicyError(f'exclusion {exclusion["id"]}: {problem}')
    for case, stages in loaded['ordered'].items():
        if not isinstance(stages, dict) or any(
            stage not in STAGES
            or not isinstance(positions, list)
            or not all(isinstance(p, int) and p >= 0 for p in positions)
            for stage, positions in stages.items()
        ):
            raise PolicyError(f'ordered: {case}')
    for case, stage in loaded['reach'].items():
        if stage not in STAGES:
            raise PolicyError(f'reach: {case}')
    fields = loaded['baseline']
    if (
        not isinstance(fields, list)
        or not fields
        or len(set(map(str, fields))) != len(fields)
        or not all(
            isinstance(field, str) and FIELD.fullmatch(field)
            for field in fields
        )
    ):
        raise PolicyError('baseline: not a list of field paths')
    if not isinstance(loaded['relaxations'], list) or any(
        not isinstance(item, dict)
        or set(item) != {'change', 'decision'}
        or not all(isinstance(value, str) and value for value in item.values())
        for item in loaded['relaxations']
    ):
        raise PolicyError('relaxations: not a list of changes and decisions')
    standin = loaded['standin']
    if not isinstance(standin, dict) or set(standin) != set(APPROVED_STANDIN):
        raise PolicyError('standin: not its codes and diagnostic rules')
    if list(standin['codes'] or []) != list(APPROVED_STANDIN['codes']):
        raise PolicyError(
            "standin: its codes are not the design's B1, B2 and B4"
        )
    diagnostic = standin['diagnostic']
    if (
        not isinstance(diagnostic, list)
        or any(
            rule not in APPROVED_STANDIN['diagnostic'] for rule in diagnostic
        )
        or any(rule not in rules for rule in diagnostic)
    ):
        raise PolicyError(
            'standin: a diagnostic rule the owner did not approve (KO3-45)'
        )
    sampled = loaded['sampled']
    if (
        not isinstance(sampled, dict)
        or len(sampled) > APPROVED_SAMPLED['cases']
    ):
        raise PolicyError(
            'sampled: more than the one Snowflake case the owner approved '
            f'({APPROVED_SAMPLED["decision"]})'
        )
    for case, entry in sampled.items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {'leg', 'stage', 'statement', 'decision'}
            or entry['leg'] != APPROVED_SAMPLED['leg']
            or entry['decision'] != APPROVED_SAMPLED['decision']
            or entry['stage'] not in STAGES
            or not isinstance(entry['statement'], int)
            or isinstance(entry['statement'], bool)
            or entry['statement'] < 0
        ):
            raise PolicyError(
                f'sampled: {case}: not a statement of a call on the Snowflake '
                f'leg under the owner\'s "{APPROVED_SAMPLED["decision"]}"'
            )
    return loaded


Paths = tuple[tuple[str, ...], ...]


def excluded(checked: dict[str, Any]) -> Paths:
    r"""A checked policy's payload exclusions, as the paths whose value
    (not presence or type) is left out."""
    return tuple(
        tuple(exclusion['path'])
        for exclusion in checked['exclusions']
        if exclusion['in'] == 'payload'
    )


#: The repository policy's payload exclusions (decision E35, approved by the
#: owner on 2026-10-10), the default where no policy is given; compare.py
#: passes the policy it was given instead.
RANDOM_PATHS: Paths = excluded(policy())


def _json_type(value: Any) -> str:
    r"""A parsed value's JSON type; a number is ``{"number": text}``."""
    if isinstance(value, dict) and set(value) == {'number'}:
        return 'number'
    if isinstance(value, dict):
        return 'object'
    if isinstance(value, list):
        return 'array'
    if isinstance(value, bool):
        return 'boolean'
    if value is None:
        return 'null'
    return 'string'


def masked(parsed: Any, paths: Paths | None = None) -> Any:
    r"""A parsed prediction payload with the value at each of ``paths``
    (by default :data:`RANDOM_PATHS`) replaced by its JSON type, as
    conservation compares it. The same rule holds the core's call (the payload it writes for the
    endpoint, ``core.predicted.body``) across the core runners and to its
    pin (T1.11d): one value, ``metadata.request_id``, under E35, approved by
    the owner on 2026-10-10."""
    parsed = copy.deepcopy(parsed)
    for path in RANDOM_PATHS if paths is None else paths:
        node = parsed
        for part in path[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
        # Only the value: an absent field stays absent, and a present
        # one keeps its type, so a deleted or retyped field differs.
        if isinstance(node, dict) and path[-1] in node:
            node[path[-1]] = {
                'excluded value of type': _json_type(node[path[-1]])
            }
    return parsed


def _payloads(public: dict[str, Any]) -> list[Any]:
    return [
        (payload['method'], payload['path'], masked(payload['parsed']))
        for payload in public['payloads']
    ]


def projection(observation: dict[str, Any]) -> dict[str, Any]:
    r"""What the capture must not change of one observation."""
    public = observation['public']
    return {
        'outcome': observation['outcome'],
        'warnings': observation['host']['warnings'],
        'schema': public['schema'],
        'payloads': _payloads(public),
        'result': public['result'],
        'query_log': None
        if observation['driver']['query_log'] is None
        else {
            key: observation['driver']['query_log'][key]
            for key in ('statements', 'bindings')
        },
    }


def complete(observation: dict[str, Any]) -> list[str]:
    r"""Why a captured observation's capture is not complete, if it is
    not."""
    driver = observation['driver']
    capture, logged = driver['capture'], driver['query_log']
    if capture is None:
        return ['the capture was off']
    problems = [f'could not record: {what}' for what in capture['incomplete']]
    if logged is None:
        # Every line has its log: DuckDB's own, or off DuckDB the driver's
        # statement log (T2.1a fix 4); a capture held to none is incomplete.
        problems.append('an observation without its query log')
        return problems
    if len(capture['statements']) != logged['statements']:
        problems.append(
            f'{len(capture["statements"])} statements captured, '
            f'{logged["statements"]} in the query log'
        )
    if len(capture['bindings']) != logged['bindings']:
        problems.append(
            f'{len(capture["bindings"])} bindings captured, '
            f'{logged["bindings"]} in the query log'
        )
    if logged['differing']:
        problems.append(
            f'statements {logged["differing"]} ran as text other than the '
            'capture recorded'
        )
    return problems


def _first_difference(on: Any, off: Any, path: str) -> str | None:
    if isinstance(on, dict) and isinstance(off, dict):
        for key in sorted(set(on) | set(off)):
            if key not in on or key not in off:
                return f'{path}/{key}: present in one run only'
            found = _first_difference(on[key], off[key], f'{path}/{key}')
            if found:
                return found
        return None
    if isinstance(on, (list, tuple)) and isinstance(off, (list, tuple)):
        if len(on) != len(off):
            return f'{path}: {len(on)} items against {len(off)}'
        for index, (a, b) in enumerate(zip(on, off, strict=True)):
            found = _first_difference(a, b, f'{path}/{index}')
            if found:
                return found
        return None
    return None if on == off else f'{path}: differs'


def _cases(ids: list[str], name: str) -> list[str]:
    problems = []
    if not ids:
        problems.append(f'the {name} run holds no case')
    repeated = sorted({case for case in ids if ids.count(case) > 1})
    if repeated:
        problems.append(f'the {name} run holds {repeated} more than once')
    return problems


#: One captured observation and its capture-off twin (``None`` when the
#: capture-off run lacks the case).
Pair = tuple[dict[str, Any], dict[str, Any] | None]


def _check(
    on_ids: list[str],
    off_ids: list[str],
    required: tuple[str, ...] | list[str],
    pairs: Iterable[Pair],
) -> list[str]:
    r""":func:`check` over the two runs' case ids and their observations
    in pairs, taken one pair at a time, so a caller reading files holds
    one pair in memory, never a whole run."""
    problems = _cases(on_ids, 'captured') + _cases(off_ids, 'uncaptured')
    if on_ids != off_ids:
        problems.append('the two runs hold different cases')
    on_set, off_set = set(on_ids), set(off_ids)
    for case in required:
        if case not in off_set or case not in on_set:
            problems.append(f'{case}: required, and not in both runs')
    for item, other in pairs:
        case = item['case']
        problems.extend(f'{case}: {p}' for p in complete(item))
        if other is None:
            continue
        if other['driver']['capture'] is not None:
            problems.append(f'{case}: the run without capture captured')
        found = _first_difference(
            projection(item), projection(other), 'projection'
        )
        if found:
            problems.append(f'{case}: capture on and off differ at {found}')
    return problems


def check(
    on: list[dict[str, Any]],
    off: list[dict[str, Any]],
    required: tuple[str, ...] | list[str] = (),
) -> list[str]:
    r"""Every problem with the capture over two runs of the same cases,
    each case in ``required`` among them."""
    by_case = {item['case']: item for item in off}
    return _check(
        [item['case'] for item in on],
        [item['case'] for item in off],
        required,
        ((item, by_case.get(item['case'])) for item in on),
    )


def _lines(path: Path) -> Iterator[tuple[int, bytes]]:
    r"""Each non-empty line of ``path`` with the offset it starts at."""
    with path.open('rb') as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                return
            if line.strip(b'\r\n'):
                yield offset, line


def _index(path: Path) -> list[tuple[str, int]]:
    r"""Each observation's case id and the offset of its line, in file
    order; one line is parsed at a time."""
    return [(json.loads(line)['case'], offset) for offset, line in _lines(path)]


def _at(path: Path, offset: int) -> dict[str, Any]:
    with path.open('rb') as handle:
        handle.seek(offset)
        loaded: dict[str, Any] = json.loads(handle.readline())
        return loaded


def _summary(item: dict[str, Any]) -> str | None:
    capture = item['driver']['capture']
    logged = item['driver']['query_log']
    if capture is None:
        return None
    if logged is None:
        return (
            f'{item["case"]}: {len(capture["statements"])} statements and '
            f'{len(capture["bindings"])} bindings captured on '
            f'{capture["cursors"]} derived cursors; no query log'
        )
    return (
        f'{item["case"]}: {len(capture["statements"])} statements and '
        f'{len(capture["bindings"])} bindings captured on '
        f'{capture["cursors"]} derived cursors; query log '
        f'{logged["statements"]} and {logged["bindings"]}; '
        f'{logged["compared"]} statement texts compared with it'
    )


def check_files(
    on: Path, off: Path, required: tuple[str, ...] | list[str] = ()
) -> tuple[list[str], list[str]]:
    r""":func:`check` over two observation files, and each captured
    case's summary line, reading one pair of observations at a time: a
    family's runs reach hundreds of megabytes, which parsed whole took
    gigabytes (T1.32e's time and feature family: 6.6 GB on Linux)."""
    on_index, off_index = _index(on), _index(off)
    off_at = dict(off_index)
    summaries: list[str] = []

    def pairs() -> Iterator[Pair]:
        for case, offset in on_index:
            item = _at(on, offset)
            summary = _summary(item)
            if summary is not None:
                summaries.append(summary)
            where = off_at.get(case)
            yield item, None if where is None else _at(off, where)

    problems = _check(
        [case for case, _ in on_index],
        [case for case, _ in off_index],
        required,
        pairs(),
    )
    return problems, summaries


# --- the warehouse captures (Phase 3, design choice 2) ------------------------------

#: Where the captures are committed: a protected path (contract/protected-
#: paths.txt), changed only in a landing whose trailer names it.
CAPTURES = 'contract/captures/'
#: The capture format this checker reads.
CAPTURE_FORMAT = 'sdm-capture/1'
#: The recording mode's lines (run_python.py's RECORD_FORMAT), which T3.3's
#: captures.py import turns into the capture.
RECORD_FORMAT = 'sdm-record/1'
#: A commit's full SHA.
SHA = re.compile(r'[0-9a-f]{40}')
#: A statement that answers rows: one with no result read and no error
#: lost its result.
_QUERY = re.compile(r'\s*(SELECT|WITH|DESCRIBE|SHOW|VALUES|CALL|TABLE)\b', re.I)
#: The versions every capture names, beside each leg's driver's
#: (run_python.py's ``_DRIVERS``, one at least).
_VERSIONS = ('python', 'pandas', 'pyarrow')


def runner() -> Any:
    r"""``run_python.py``, beside this file: the one definition of the
    case format, the legs, the input hash and the draws."""
    import importlib.util

    if 'run_python' in sys.modules:
        return sys.modules['run_python']
    spec = importlib.util.spec_from_file_location(
        'run_python', Path(__file__).resolve().parent / 'run_python.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['run_python'] = module
    spec.loader.exec_module(module)
    return module


def _table(text: Any, where: str, problems: list[str]) -> Any:
    r"""The Arrow table a capture holds as IPC text, or None with the
    problem named."""
    if not isinstance(text, str):
        problems.append(f'{where}: no Arrow table')
        return None
    try:
        return runner().read_arrow_ipc(text)
    except Exception as error:  # noqa: BLE001 - named, never raised
        problems.append(f'{where}: not an Arrow table ({type(error).__name__})')
        return None


def _duck_tables(
    case: dict[str, Any], bases: dict[str, list[str]]
) -> list[str]:
    r"""The tables of ``case``'s DuckDB data, as the recording mode reads
    them back through each leg's driver."""
    import duckdb

    connection = duckdb.connect()
    try:
        for statement in [
            *bases[case['input']['base']],
            *case['input'].get('setup', []),
        ]:
            connection.execute(statement)
        return [
            str(row[0])
            for row in connection.execute(
                'SELECT table_name FROM information_schema.tables '
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
        ]
    finally:
        connection.close()


def capture_problems(
    line: dict[str, Any],
    case: dict[str, Any],
    tables: list[str] | None = None,
) -> list[str]:
    r"""Why one capture line is not the complete capture of ``case`` on its
    leg (design choice 2), each problem named; ``tables`` are the case's
    tables (:func:`_duck_tables`), read when not given. Complete is: its
    format, case key, leg and recording and reference commits; the
    versions of the runtime and the leg's driver; the session zone; a
    valid python-ref reference line of the same key whose capture recorded
    everything it saw; every statement that capture saw, in order, each
    with its text and every result read, each held whole as Arrow of the
    rows it counted, and a query neither read nor failed counted as lost;
    every random draw its statements made (:func:`run_python.draws_of`),
    no other, each with the rows its statement picked; every table of the
    case's data once and no other; and each view-derived relation a
    statement read, evaluated or refused."""
    run = runner()
    key = line.get('case')
    problems: list[str] = []
    if line.get('format') != RECORD_FORMAT:
        return [f'{key}: not {RECORD_FORMAT}']
    leg = line.get('leg')
    if leg not in run.WAREHOUSES:
        return [f'{key}: leg {leg!r} is not a warehouse leg']
    if key != run.case_key(case['id'], leg):
        problems.append(f'{key}: not the key of {case["id"]} on {leg}')
    if leg not in run.case_legs(case):
        problems.append(f'{key}: {case["id"]} does not run on {leg}')
    recording = line.get('recording') or {}
    for what in ('commit', 'reference'):
        if not SHA.fullmatch(str(recording.get(what))):
            problems.append(f'{key}: no {what} commit recorded')
    versions = line.get('versions') or {}
    missing = [name for name in _VERSIONS if not versions.get(name)]
    if missing:
        problems.append(f'{key}: no {missing} version recorded')
    if not any(versions.get(name) for name in run._DRIVERS[leg]):
        problems.append(f"{key}: no version of {leg}'s driver recorded")
    if not isinstance(line.get('zone'), str) or not line['zone']:
        problems.append(f'{key}: no session zone recorded')
    observation = line.get('observation')
    validate = run.validator('observation')
    if not isinstance(observation, dict):
        return [*problems, f'{key}: no reference line']
    invalid = run._invalid(validate, observation)
    if invalid is not None:
        return [*problems, f'{key}: its reference line is not valid: {invalid}']
    if observation['case'] != key:
        problems.append(
            f'{key}: its reference line is of {observation["case"]}'
        )
    if observation['identity']['runner'] != 'python-ref':
        problems.append(f"{key}: its reference line is not python-ref's")
    captured = observation['driver']['capture']
    if captured is None:
        return [*problems, f'{key}: recorded with the capture off']
    for what in [*captured['incomplete'], *(line.get('incomplete') or [])]:
        problems.append(f'{key}: could not record: {what}')
    statements = line.get('statements')
    if not isinstance(statements, list):
        return [*problems, f'{key}: no statements']
    if [s.get('sql') for s in statements] != [
        s['sql'] for s in captured['statements']
    ]:
        problems.append(
            f'{key}: its statements are not the {len(captured["statements"])} '
            'its reference line captured'
        )
    else:
        # Each statement whole: its parameters, its error and every read,
        # as the reference line captured them, the Arrow aside (Codex r1,
        # T3.2 fix 5: a lost parameter or read passed on the text alone).
        for position, (raw, seen) in enumerate(
            zip(statements, captured['statements'], strict=True)
        ):
            mine = {
                **raw,
                'results': [
                    {k: v for k, v in result.items() if k != 'arrow'}
                    if isinstance(result, dict)
                    else result
                    for result in raw.get('results') or []
                ],
            }
            for field in sorted(set(mine) | set(seen)):
                if mine.get(field) != seen.get(field):
                    problems.append(
                        f'{key}: statement {position}: its {field} is not what '
                        'its reference line captured'
                    )
    expected_draws = []
    for position, statement in enumerate(statements):
        where = f'{key}: statement {position}'
        if not isinstance(statement.get('sql'), str):
            problems.append(f'{where}: no text')
            continue
        results = statement.get('results') or []
        if (
            not results
            and statement.get('error') is None
            and _QUERY.match(statement['sql'])
        ):
            problems.append(f'{where}: a query with no result read')
        for read, result in enumerate(results):
            table = _table(
                result.get('arrow'), f'{where} read {read}', problems
            )
            if table is not None and table.num_rows != result.get('rows'):
                problems.append(
                    f'{where} read {read}: {table.num_rows} rows held, '
                    f'{result.get("rows")} read'
                )
            columns = (result.get('values') or {}).get('columns')
            if table is not None and columns is not None:
                if table.column_names != columns:
                    problems.append(
                        f'{where} read {read}: columns {table.column_names} '
                        f'held, {columns} read'
                    )
            schema = result.get('schema')
            if table is not None and isinstance(schema, list):
                held_schema = [
                    {'name': field.name, 'type': str(field.type)}
                    for field in table.schema
                ]
                if held_schema != schema:
                    problems.append(
                        f'{where} read {read}: the Arrow held is not the '
                        'schema read'
                    )
        for draw in run.draws_of(statement['sql']):
            expected_draws.append({'statement': position, **draw})
            if not results and statement.get('error') is None:
                problems.append(f'{where}: a random draw without its rows')
    if (line.get('draws') or []) != expected_draws:
        problems.append(
            f'{key}: its draws are not the {len(expected_draws)} its '
            'statements made'
        )
    want = sorted(
        _duck_tables(case, run.load_bases()) if tables is None else tables
    )
    held = [item.get('name') for item in line.get('tables') or []]
    if sorted(held) != want:
        problems.append(f'{key}: its tables are {sorted(held)}, not {want}')
    for item in line.get('tables') or []:
        _table(item.get('arrow'), f'{key}: table {item.get("name")}', problems)
    derived = [
        position
        for position, statement in enumerate(statements)
        if isinstance(statement.get('sql'), str)
        and statement.get('error') is None
        and not statement.get('parameters')
        and run._DERIVED.search(statement['sql'])
        and run._LIMIT.search(statement['sql'])
    ]
    relations = {
        item.get('statement'): item for item in line.get('relations') or []
    }
    seen: set[str] = set()
    for position in derived:
        whole = run._LIMIT.sub('', statements[position]['sql'])
        if whole in seen:
            continue
        seen.add(whole)
        relation = relations.get(position)
        if relation is None:
            problems.append(
                f'{key}: statement {position} read a view-derived relation '
                'the capture did not evaluate'
            )
        elif 'error' not in relation:
            _table(
                relation.get('table'), f'{key}: relation {position}', problems
            )
    return problems


def recordings(
    cases: list[dict[str, Any]],
    lines: Iterable[dict[str, Any]],
    legs: Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    r"""Every problem with the capture lines over ``cases`` (each line
    complete, no case and leg twice, none of a case or leg not named), and
    every case and leg, of ``legs`` (default every warehouse leg), with no
    capture of its current input: pending live, the first a capture's own
    failure and the second a listing (design choice 2)."""
    run = runner()
    bases = run.load_bases()
    legs = tuple(run.WAREHOUSES if legs is None else legs)
    by_key = {
        run.case_key(case['id'], leg): case
        for case in cases
        for leg in run.case_legs(case)
        if leg in legs
    }
    problems: list[str] = []
    found: dict[str, str] = {}
    for line in lines:
        if not isinstance(line, dict):
            problems.append(f'a capture line is a JSON {type(line).__name__}')
            continue
        key = line.get('case')
        if key in found:
            problems.append(f'{key}: captured twice')
            continue
        case = by_key.get(key)
        if case is None:
            problems.append(f'{key}: no case of these runs on that leg')
            continue
        try:
            problems.extend(capture_problems(line, case))
        except RecursionError:
            problems.append(f'{key}: nested past what can be read')
        found[key] = line.get('input')
    pending = sorted(
        f'{key}: '
        + (
            'no capture'
            if key not in found
            else 'its input changed after its capture'
        )
        for key, case in by_key.items()
        if found.get(key) != run.input_hash(case, bases)
    )
    return problems, pending


def recording_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog='capture_check.py recording')
    parser.add_argument('--cases', type=Path, nargs='+', required=True)
    parser.add_argument('--capture', type=Path, nargs='*', default=[])
    parser.add_argument('--leg', action='append', default=None)
    parser.add_argument('--locked', action='store_true')
    arguments = parser.parse_args(argv)
    run = runner()
    try:
        cases = [case for case, _ in run.load_cases(arguments.cases)]
        lines = [
            json.loads(text)
            for path in arguments.capture
            for text in path.read_text(encoding='utf-8').splitlines()
            if text.strip()
        ]
    except (OSError, ValueError, RecursionError, run.RunnerFailure) as error:
        print(f'captures: unreadable: {type(error).__name__}: {error}')
        return 1
    if arguments.leg and any(
        leg not in run.WAREHOUSES for leg in arguments.leg
    ):
        print(f'captures: --leg is one of {run.WAREHOUSES}')
        return 1
    problems, pending = recordings(cases, lines, arguments.leg)
    for problem in problems:
        print(f'FAILS {problem}')
    for item in pending:
        print(f'{"FAILS" if arguments.locked else "PENDING LIVE"} {item}')
    failed = bool(problems) or (arguments.locked and bool(pending))
    print(
        f'recording: {"FAILED" if failed else "complete"}; '
        f'{len(lines)} recorded, {len(pending)} pending live'
    )
    return 1 if failed else 0


# --- the capture manifests (T3.3's writer; design choice 2) -------------------


def manifest_problem(manifest: Any, name: str) -> str | None:
    r"""Why a capture manifest is not one this checker reads, or None: a
    JSON object of format :data:`CAPTURE_FORMAT` naming its recording and
    reference commits, its legs, and its cases by key, each with its input
    hash (``{"format", "recording_commit", "reference_commit", "legs":
    {LEG: {...versions}}, "cases": {"<case>@<leg>": {"input_hash", ...}}}``,
    the capture writer's, T3.3)."""
    if not isinstance(manifest, dict):
        return f'{name}: not a capture manifest'
    if manifest.get('format') != CAPTURE_FORMAT:
        return (
            f'{name}: format {manifest.get("format")!r} is not '
            f'{CAPTURE_FORMAT}, the version this checker reads'
        )
    for what in ('recording_commit', 'reference_commit'):
        if not SHA.fullmatch(str(manifest.get(what))):
            return f'{name}: no {what}'
    legs, cases = manifest.get('legs'), manifest.get('cases')
    if not isinstance(legs, dict) or not isinstance(cases, dict):
        return f'{name}: no legs or cases'
    for key, entry in cases.items():
        if not isinstance(entry, dict) or not re.fullmatch(
            r'[0-9a-f]{64}', str(entry.get('input_hash'))
        ):
            return f'{name}: {key} has no input hash'
    return None


def captures(
    cases: list[dict[str, Any]],
    manifests: list[tuple[str, Any]],
    legs: Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    r"""Every manifest refused (:func:`manifest_problem`), and every case
    and warehouse leg (of ``legs``, default every one) the cases run on
    with no capture of its current input (``run_python.input_hash``):
    pending live, a case added or changed after the capture or a leg the
    capture does not hold (design choice 2)."""
    run = runner()
    bases = run.load_bases()
    legs = tuple(run.WAREHOUSES if legs is None else legs)
    problems: list[str] = []
    held: dict[str, str] = {}
    for name, manifest in manifests:
        problem = manifest_problem(manifest, name)
        if problem is not None:
            problems.append(problem)
            continue
        for key, entry in manifest['cases'].items():
            held[str(key)] = str(entry['input_hash'])
    pending = sorted(
        run.case_key(case['id'], leg)
        for case in cases
        for leg in run.case_legs(case)
        if leg in legs
        and held.get(run.case_key(case['id'], leg))
        != run.input_hash(case, bases)
    )
    return problems, pending


def captures_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog='capture_check.py captures')
    parser.add_argument('--cases', type=Path, action='append', required=True)
    parser.add_argument('--locked', action='store_true')
    parser.add_argument('--leg', action='append', default=None)
    parser.add_argument('manifests', type=Path, nargs='*')
    arguments = parser.parse_args(argv)
    run = runner()
    try:
        cases = [case for case, _ in run.load_cases(arguments.cases)]
        manifests = [
            (str(path), json.loads(path.read_text(encoding='utf-8')))
            for path in arguments.manifests
        ]
    except (OSError, ValueError, RecursionError, run.RunnerFailure) as error:
        print(f'captures: unreadable: {type(error).__name__}: {error}')
        return 1
    if arguments.leg and any(
        leg not in run.WAREHOUSES for leg in arguments.leg
    ):
        print(f'captures: --leg is one of {run.WAREHOUSES}')
        return 1
    problems, pending = captures(cases, manifests, arguments.leg)
    for problem in problems:
        print(f'FAILS {problem}')
    for key in pending:
        print(f'pending live: {key}')
    failed = bool(problems) or (arguments.locked and bool(pending))
    print(
        f'captures: {"FAILED" if failed else "ok"}; {len(manifests)} '
        f'manifests read, {len(pending)} cases and legs waiting for a live run'
        + (' (the locked run fails on any)' if arguments.locked else '')
    )
    return 1 if failed else 0


def _git_lines(repo: Path, *arguments: str) -> list[str]:
    import subprocess

    done = subprocess.run(
        ['git', '-C', str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    if done.returncode != 0:
        raise ValueError(f'git {arguments[0]} failed')
    return [line for line in done.stdout.splitlines() if line]


def captures_only(repo: Path, final: str, commit: str = 'HEAD') -> list[str]:
    r"""Why the commits from ``final`` to ``commit`` are not a captures-only
    change recorded from ``final`` (design, "The live-test plan": E-0's
    reference lines and the core's live capture committed on the final
    candidate run no new campaign): ``final`` an ancestor of ``commit``,
    at least one commit between, every path they change under
    :data:`CAPTURES`, none deleted, and every line of every file they add
    or change naming ``final`` as its recording commit."""
    try:
        final = _git_lines(
            repo, 'rev-parse', '--verify', f'{final}^{{commit}}'
        )[0]
        commit = _git_lines(
            repo, 'rev-parse', '--verify', f'{commit}^{{commit}}'
        )[0]
        _git_lines(repo, 'merge-base', '--is-ancestor', final, commit)
    except (ValueError, IndexError):
        return [f'{final} is not a commit below {commit}']
    if final == commit:
        return ['no commit after the final candidate']
    problems = []
    changes = _git_lines(
        repo, 'diff', '--no-renames', '--name-status', final, commit
    )
    for change in changes:
        status, _, path = change.partition('\t')
        if not path.startswith(CAPTURES):
            problems.append(f'{path}: changed outside {CAPTURES}')
            continue
        if status.startswith('D'):
            problems.append(f'{path}: deleted')
            continue
        if not path.endswith(('.json', '.jsonl')):
            continue  # a recorded table, named by its manifest
        text = '\n'.join(_git_lines(repo, 'show', f'{commit}:{path}'))
        documents = [text] if path.endswith('.json') else text.splitlines()
        for number, raw in enumerate(documents, 1):
            try:
                document = json.loads(raw)
                recorded = document.get('recording_commit') or (
                    document.get('recording') or {}
                ).get('commit')
            except (ValueError, AttributeError, RecursionError):
                problems.append(f'{path}:{number}: not a capture')
                continue
            if recorded != final:
                problems.append(
                    f'{path}:{number}: recorded from {recorded}, not the final '
                    f'candidate {final}'
                )
    if not changes:
        problems.append('the commits change nothing')
    return problems


def commit_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog='capture_check.py commit')
    parser.add_argument('--candidate', required=True)
    parser.add_argument('commit', nargs='?', default='HEAD')
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    problems = captures_only(
        arguments.repo, arguments.candidate, arguments.commit
    )
    for problem in problems:
        print(f'FAILS {problem}')
    print(f'commit: {"FAILED" if problems else "ok"}')
    return 1 if problems else 0


def accepted_recordings(repo: Path, head: str = 'HEAD') -> list[str]:
    r"""The commits whose live campaign stands for ``head`` (design, "The
    live-test plan"): ``head`` itself, and each first-parent ancestor
    below a run of captures-only commits up to ``head``
    (:func:`captures_only`), the final candidate those captures were
    recorded from."""
    try:
        sha = _git_lines(repo, 'rev-parse', '--verify', f'{head}^{{commit}}')[0]
    except (ValueError, IndexError):
        return []
    found = [sha]
    for parent in _git_lines(repo, 'rev-list', '--first-parent', f'{sha}~1')[
        :50
    ]:
        if captures_only(repo, parent, sha):
            break
        found.append(parent)
    return found


def main(argv: list[str] | None = None) -> int:
    given = sys.argv[1:] if argv is None else argv
    commands = {
        'captures': captures_main,
        'recording': recording_main,
        'commit': commit_main,
    }
    if given[:1] and given[0] in commands:
        return commands[given[0]](given[1:])
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--on', type=Path, required=True)
    parser.add_argument('--off', type=Path, required=True)
    parser.add_argument('--require', action='append', default=[])
    arguments = parser.parse_args(argv)
    problems, summaries = check_files(
        arguments.on, arguments.off, arguments.require
    )
    for problem in problems:
        print(problem, file=sys.stderr)
    for summary in summaries:
        print(summary)
    print('capture: ' + ('FAILED' if problems else 'complete and conserving'))
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
