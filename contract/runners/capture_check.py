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


def main(argv: list[str] | None = None) -> int:
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
