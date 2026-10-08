# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What the Phase 8.7 checks share (``CONTRACT.md`` beside this file).

- The request coverage-gap list (``scripts/parity/request_coverage_gaps.json``)
  as it stands, and as it stood at the phase's base (``base.json``, with the
  fixture exclusions of then).
- The entries Phase 8.6's coverage-gap verifier called wrong
  (``investigations/P86-coverage-gaps-verifier.md`` in the migration
  harness): group A, reached by a public call and held by nothing, and group
  B, still unreachable but given a false reason or the wrong guard.
- Two measures of today's Python under branch coverage of every module the
  list names: the recorded cases (``_corpus.py``), and the host tests under
  ``tests/rust_core`` (``_host.py``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUST_CORE = HERE.parent
REPO = HERE.parents[2]
PARITY = REPO / 'tests' / 'parity'
SCRIPTS = REPO / 'scripts' / 'parity'
GAPS = SCRIPTS / 'request_coverage_gaps.json'
EXCLUSIONS = SCRIPTS / 'request_fixture_exclusions.json'
PHASE8_GAPS = PARITY / 'COVERAGE_GAPS.md'
BASE = HERE / 'base.json'

#: The packages measured, as ``scripts/parity/coverage.py`` measures them.
PACKAGES = (
    'sdm_relational_engine',
    'sdm_client',
    'sdm_connectors',
    'predictive_query',
)
#: The operations whose fixture cases the core is held to (Phase 8.5,
#: ``tests/rust_core/request/CONTRACT.md``, "Names").
REQUEST_OPERATIONS = ('client.', 'decompose.', 'payload.', 'tabular.', 'wire.')

ENGINE = 'packages/sdm-relational-engine/src/sdm_relational_engine/'
CLIENT = 'packages/sdm-client/src/sdm_client/'
PAYLOAD = ENGINE + 'rfm/inference/payload.py'
BASE_TABLE = ENGINE + 'rfm/base/table.py'
KEYS = CLIENT + 'wire/keys.py'
RELATIONAL = CLIENT + 'materializers/relational.py'
SOURCES = CLIENT + 'materializers/sources.py'
RESULT = CLIENT + 'result.py'
ADAPTER = CLIENT + 'adapters/relational.py'
CLIENT_PY = CLIENT + 'client.py'
REQUEST = CLIENT + 'request.py'

#: Group A: a public call reaches the branch and nothing held it.
A = 'A'
#: Group B: still unreachable, but the reason is false or names the wrong
#: guard.
B = 'B'


@dataclass(frozen=True)
class Entry:
    module: str
    branch: tuple[int, int]
    group: str
    what: str
    #: Clauses of the base reason the verifier showed false; none may stay
    #: (case-insensitive, runs of whitespace as one space).
    false: tuple[str, ...] = ()
    #: For group B: patterns (case-insensitive regular expressions) the
    #: rewritten reason must each match, naming the real guard.
    names: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return f'{self.module.rsplit("/src/", 1)[-1]} {list(self.branch)}'


def _a(module: str, branch: tuple[int, int], what: str, *false: str) -> Entry:
    return Entry(module, branch, A, what, false=false)


_EDITED = 'an edited envelope through Request.materialized(payloads=...)'

ENTRIES: tuple[Entry, ...] = (
    # Group A: an edited envelope through Request.materialized, then predict
    # or fit.
    *(
        _a(ADAPTER, branch, f'{_EDITED}: {what}')
        for branch, what in (
            ((652, 653), 'the target column renamed, a constant target'),
            ((687, 688), "task.kind='binary_classification'"),
            ((709, 710), "schema.related_tables['orders']=None"),
            ((746, 747), "the predict instance table's entity ref renamed"),
            ((780, 781), "task.kind='forecasting', no forecast_step"),
            ((781, 782), "task.kind='forecasting', no forecast_step"),
            ((781, 794), "task.kind='forecasting', with forecast_step"),
            ((805, 820), "the predict instance table's instance_id renamed"),
            ((851, 865), "the reference relationship's source_columns"),
            ((852, 853), "the reference relationship's source_columns"),
            ((856, 857), "the reference relationship's target_columns"),
        )
    ),
    *(
        _a(KEYS, branch, f'{_EDITED}, then fit: {what}')
        for branch, what in (
            ((84, 85), 'no entity ref column, primary_key renamed'),
            ((89, 91), 'a windowed COUNT without __kumo_anchor_time'),
            ((307, 308), 'no context, holdout keys'),
            ((312, 313), 'context rows as a tuple, holdout keys'),
            ((330, 331), 'a context row cut short, holdout keys'),
            ((352, 353), 'related tables with rows=None, holdout keys'),
            ((354, 355), 'no instance_id in the context instance table'),
        )
    ),
    # Group A: Request.materialized without entity_table.
    *(
        _a(
            REQUEST,
            branch,
            'Request.materialized(...) called without entity_table, then '
            'predict',
        )
        for branch in ((840, 841), (843, 844), (840, 845), (843, 840))
    ),
    # Group A: other public routes.
    _a(
        CLIENT_PY,
        (800, 801),
        'request.entity_table = None on a request with nothing to score',
        'unreachable from sdm_client through any call',
    ),
    _a(
        RELATIONAL,
        (474, 477),
        'an int-named Decimal column added to a frame and its schema',
        'calls the engine only with keyword arguments it names itself',
        'unreachable from sdm_client',
    ),
    _a(
        RELATIONAL,
        (501, 502),
        'an int-named Decimal column added to a frame and its schema',
        'sdm_client calls the engine correctly',
        'unreachable from sdm_client',
    ),
    _a(
        KEYS,
        (240, 241),
        "the key column deleted from a frame request's schema",
        'the dtypes are always read before keys are taken from a frame',
    ),
    _a(
        KEYS,
        (277, 278),
        'FittedModel.predict after a fit over an envelope with no instance '
        'table',
        'a mapping for every envelope either adapter writes',
        'unreachable from sdm_client',
    ),
    _a(
        RESULT,
        (231, 232),
        "a forecast result's items reassigned, then to_pandas",
        'unreachable from sdm_client',
    ),
    _a(
        BASE_TABLE,
        (495, 496),
        'a DuckDB column named like the derived key, left out, then a '
        'composite key',
        "the table's source never holds",
        'unreachable from sdm_client',
    ),
    _a(
        BASE_TABLE,
        (542, 544),
        'a DuckDB column named like the derived key, left out, then a '
        'composite key',
        'not reached offline',
    ),
    # Group B: still unreachable; the reason is false or names the wrong
    # guard.
    Entry(
        ADAPTER,
        (737, 738),
        B,
        'payload is None never holds: both callers of normalize pass a dict',
        false=('given other envelopes, reaches it',),
        names=(r'client\.py:\s*(16[3-8]|41[4-6])\b',),
    ),
    Entry(
        RELATIONAL,
        (497, 503),
        B,
        'a caught TypeError always has a __traceback__',
        false=(
            'only reached by a TypeError raised in this module',
            'sdm_client calls the engine correctly',
        ),
        names=(r'traceback',),
    ),
    Entry(
        SOURCES,
        (59, 60),
        B,
        'both routes need a live warehouse: the Schema(..., source=...) '
        'constructor, and a table another semantic-model schema described put '
        'into schema.tables',
        names=(r'Schema\(|constructor', r'schema\.tables', r'warehouse'),
    ),
    *(
        Entry(
            BASE_TABLE,
            branch,
            B,
            'each gather sets each key once on a fresh copy; over a source a '
            'composite key folds while the graph is built',
            false=(
                'refused by the reserved-prefix check before the graph is '
                'built',
            ),
            names=(r'sources\.py|_copy_for_one_gather',),
        )
        for branch in ((447, 449), (449, 450), (449, 451))
    ),
    Entry(
        PAYLOAD,
        (451, 452),
        B,
        "a link prediction's second entity table is kept by the refusal at "
        'rfm.py:2529-2536',
        names=(r'rfm\.py:\s*25[23]\d|historical target entities',),
    ),
    Entry(
        PAYLOAD,
        (520, 521),
        B,
        "a link prediction's second entity table is kept by the refusal at "
        'rfm.py:2529-2536',
        false=('every instance is seeded from it',),
        names=(r'rfm\.py:\s*25[23]\d|historical target entities',),
    ),
    *(
        Entry(
            PAYLOAD,
            branch,
            B,
            'the entity table and its key are refused earlier, by '
            '_instance_dataframe and _entity_values',
            false=(
                'every sampled link joins tables of the subgraph',
                'its pairs are never empty',
            ),
            names=(
                r'_instance_dataframe|_entity_values|'
                r'payload\.py:\s*(4[45]\d|5[12]\d)\b',
            ),
        )
        for branch in ((610, 611), (616, 617))
    ),
)

GROUP_A = tuple(entry for entry in ENTRIES if entry.group == A)
GROUP_B = tuple(entry for entry in ENTRIES if entry.group == B)

#: A host test, cited by its pytest node id.
NODE_ID = re.compile(r'tests/rust_core/[\w./-]+\.py::\w+(?:\[[^\]\s]+\])?')
REACHABLE_ONLY = 'reachable only'


def squeeze(text: str) -> str:
    r"""``text`` lower-cased, with runs of whitespace as one space."""
    return ' '.join(text.split()).lower()


def says_reachable_only(reason: str) -> bool:
    return REACHABLE_ONLY in squeeze(reason)


def false_clauses(entry: Entry, reason: str) -> list[str]:
    said = squeeze(reason)
    return [clause for clause in entry.false if squeeze(clause) in said]


def unnamed(entry: Entry, reason: str) -> list[str]:
    r"""The patterns of ``entry.names`` the reason does not match."""
    return [
        pattern
        for pattern in entry.names
        if re.search(pattern, reason, re.IGNORECASE) is None
    ]


def base() -> dict[str, Any]:
    return json.loads(BASE.read_text('utf-8'))


def _entries_of(gaps: Any) -> dict[tuple[str, tuple[int, int]], str]:
    out: dict[tuple[str, tuple[int, int]], str] = {}
    if not isinstance(gaps, dict):
        return out
    for module, entries in gaps.items():
        module = _historical_path(module)
        for entry in entries if isinstance(entries, list) else ():
            branch = entry.get('branch') if isinstance(entry, dict) else None
            if (
                isinstance(branch, list)
                and len(branch) == 2
                and all(isinstance(x, int) for x in branch)
            ):
                reason = entry.get('reason')
                key = (module, (branch[0], branch[1]))
                assert key not in out, 'duplicate projected branch identity'
                out[key] = reason if isinstance(reason, str) else ''
    return out


def listed() -> dict[tuple[str, tuple[int, int]], str]:
    r"""Every entry of today's gap list: its reason, by module and branch."""
    if not GAPS.is_file():
        return {}
    return _entries_of(json.loads(GAPS.read_text('utf-8')))


def listed_at_base() -> dict[tuple[str, tuple[int, int]], str]:
    return _entries_of(base()['request_coverage_gaps'])


def cited(reason: str) -> list[str]:
    r"""The host tests a reason cites, by node id, each once, in order."""
    return list(dict.fromkeys(NODE_ID.findall(reason)))


def exclusions() -> dict[str, Any]:
    if not EXCLUSIONS.is_file():
        return {}
    found = json.loads(EXCLUSIONS.read_text('utf-8'))
    return found if isinstance(found, dict) else {}


def modules() -> list[str]:
    r"""Every module the gap list names, now or at the base, and every module
    of a named entry."""
    historical = (
        {module for module, _ in listed()}
        | {module for module, _ in listed_at_base()}
        | {entry.module for entry in ENTRIES}
    )
    current = [_current_path(module) for module in historical]
    assert len(current) == len(set(current)), (
        'duplicate projected source module'
    )
    assert all((REPO / module).is_file() for module in current), (
        'missing current source module'
    )
    return sorted(current)


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    # Not ``scripts/parity``, whose ``coverage.py`` would shadow coverage.
    paths = [str(HERE)]
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    env.setdefault('PYTHONHASHSEED', '0')
    return env


def _as_sets(found: dict[str, Any]) -> dict[str, set[tuple[int, int]]]:
    projected = {}
    for module, branches in found.items():
        key = _historical_path(module)
        assert key not in projected, 'duplicate projected coverage module'
        projected[key] = {(a, b) for a, b in branches}
    return projected


@cache
def corpus_measure() -> dict[str, Any]:
    r"""What ``_corpus.py`` reports: by module, the branches the held cases
    take (``held``), and those every transcript and fixture case take
    (``all``), with the transcripts and cases it ran and any transcript that
    did not replay."""
    with tempfile.TemporaryDirectory(prefix='sdm-p87-corpus-') as tmp:
        out = Path(tmp) / 'corpus.json'
        env = _environment()
        env['SDM_P87_MODULES'] = json.dumps(modules())
        proc = subprocess.run(
            [sys.executable, str(HERE / '_corpus.py'), '--out', str(out)],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,
        )
        assert proc.returncode == 0, (
            f'_corpus.py failed ({proc.returncode}):\n'
            f'{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}'
        )
        found = json.loads(out.read_text('utf-8'))
    found['held'] = _as_sets(found['held'])
    found['all'] = _as_sets(found['all'])
    return found


def _build_binding() -> None:
    r"""``sdm_core._native`` built from this checkout and installed into this
    environment, so the host tests run against the core as it stands."""
    for path in (RUST_CORE / 'request', RUST_CORE / 'sampling', RUST_CORE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from _request import native

    native()


@cache
def host_measure() -> dict[str, Any]:
    r"""Every test pytest collects under ``tests/rust_core``, run under branch
    coverage of today's Python (``_host.py``): by node id, its outcomes, the
    calls it made into ``sdm_core._native``, and the branches it took in the
    modules ``modules()`` names."""
    _build_binding()
    with tempfile.TemporaryDirectory(prefix='sdm-p87-host-') as tmp:
        out = Path(tmp) / 'host.json'
        env = _environment()
        env['SDM_P7_WHEEL_INSTALLED'] = '1'
        env['SDM_P87_HOST_OUT'] = str(out)
        env['SDM_P87_MODULES'] = json.dumps(modules())
        proc = subprocess.run(
            [
                sys.executable,
                '-m',
                'pytest',
                str(RUST_CORE),
                '-q',
                '-p',
                'no:cacheprovider',
                '-p',
                '_host',
                '--rootdir',
                str(REPO),
            ],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if not out.is_file():
            raise AssertionError(
                f'the host tests did not run ({proc.returncode}):\n'
                f'{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}'
            )
        found = json.loads(out.read_text('utf-8'))
    for body in found.values():
        body['executed'] = _as_sets(body.get('executed', {}))
    return found


def host_problems(
    entry_name: str,
    branch: tuple[int, int],
    module: str,
    node: str,
    hosts: dict[str, Any],
) -> list[str]:
    r"""Why ``node`` does not hold the branch: it was not collected, did not
    pass, made no call into ``sdm_core._native``, or did not take it."""
    body = hosts.get(node)
    if body is None:
        return [
            f'{entry_name}: cites {node}, which pytest does not collect under '
            f'tests/rust_core'
        ]
    problems = []
    outcomes = {when: outcome for when, outcome in body['outcomes']}
    if outcomes.get('call') != 'passed' or any(
        outcome != 'passed' for outcome in outcomes.values()
    ):
        problems.append(f'{entry_name}: {node} gave {outcomes}')
    if body['native_calls'] < 1:
        problems.append(
            f'{entry_name}: {node} made no call into sdm_core._native'
        )
    if branch not in body['executed'].get(module, set()):
        problems.append(
            f"{entry_name}: {node} does not take the branch in today's Python"
        )
    return problems


def _current_path(path: str) -> str:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'switch'))
    from _switch import current_package_path

    return current_package_path(path)


def _historical_path(path: str) -> str:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'switch'))
    from _switch import PACKAGE_RENAMES

    for old, current in PACKAGE_RENAMES.items():
        prefix = f'packages/{current}/'
        if path.startswith(prefix):
            return f'packages/{old}/' + path[len(prefix) :]
    return path
