# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What the Phase 8.6 checks share (``CONTRACT.md`` beside this file).

- The entries of ``scripts/parity/request_coverage_gaps.json`` that Phase
  8.5's coverage-gap verifier called wrong, each with the group the verifier
  put it in.
- The gap list and the fixture exclusions as they stood at the phase's base
  (``base.json``).
- Two measures of today's Python under branch coverage: the held recorded
  cases (``_held.py``), and the host tests a gap entry cites
  (``_host_plugin.py``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUST_CORE = HERE.parent
REPO = HERE.parents[2]
PARITY = REPO / 'tests' / 'parity'
SCRIPTS = REPO / 'scripts' / 'parity'
GAPS = SCRIPTS / 'request_coverage_gaps.json'
EXCLUSIONS = SCRIPTS / 'request_fixture_exclusions.json'
COVERAGE_JSON = PARITY / 'coverage.json'
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
TASK = ENGINE + 'rfm/inference/task.py'
RFM = ENGINE + 'rfm/rfm.py'
LOCAL_TABLE = ENGINE + 'rfm/backend/local/table.py'
BASE_TABLE = ENGINE + 'rfm/base/table.py'
KEYS = CLIENT + 'wire/keys.py'
TABULAR = CLIENT + 'materializers/tabular.py'
RELATIONAL = CLIENT + 'materializers/relational.py'
SOURCES = CLIENT + 'materializers/sources.py'
RESULT = CLIENT + 'result.py'
DECOMPOSE = CLIENT + 'decompose.py'
ADAPTER = CLIENT + 'adapters/relational.py'
TABULAR_ADAPTER = CLIENT + 'adapters/tabular.py'
CLIENT_PY = CLIENT + 'client.py'
REQUEST = CLIENT + 'request.py'

#: A branch that ordinary public calls reach: a recorded case must take it.
PUBLIC = 'public'
#: Reached by a route the entry did not name: a recorded case takes it, or
#: the entry says "Reachable only" and cites a host test for the route.
ROUTE = 'route'
#: A "Reachable only" entry whose routes are too few: a recorded case takes
#: it, or the entry cites a host test for each route.
NARROW = 'narrow'
#: Untaken, but the reason given is false or cites the wrong guard.
FALSE = 'false'


@dataclass(frozen=True)
class Entry:
    module: str
    branch: tuple[int, int]
    group: str
    what: str
    #: For a NARROW entry, how many routes the verifier found to it.
    routes: int = 1
    #: Clauses of the base reason the verifier showed false; none may stay.
    false: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return f'{self.module.rsplit("/src/", 1)[-1]} {list(self.branch)}'


ENTRIES: tuple[Entry, ...] = (
    # Ordinary public calls (the verifier's first list).
    Entry(PAYLOAD, (918, 919), PUBLIC, 'quantile outputs'),
    Entry(PAYLOAD, (924, 925), PUBLIC, 'return_embeddings, then predict'),
    Entry(PAYLOAD, (980, 981), PUBLIC, 'a bool target gathered as multiclass'),
    Entry(
        PAYLOAD,
        (541, 542),
        PUBLIC,
        "plain link prediction's context entity references",
    ),
    Entry(
        PAYLOAD,
        (208, 209),
        PUBLIC,
        'SDMClient.fit answered 400 "categorical cardinality ..."',
    ),
    Entry(KEYS, (318, 319), PUBLIC, 'tabular fit with holdout keys'),
    Entry(KEYS, (334, 336), PUBLIC, 'tabular fit with holdout keys'),
    Entry(KEYS, (323, 324), PUBLIC, 'tabular fit with holdout keys'),
    Entry(KEYS, (81, 82), PUBLIC, 'tabular fit with holdout keys'),
    Entry(TABULAR, (227, 228), PUBLIC, 'a tabular target whose stype is ID'),
    Entry(RESULT, (132, 133), PUBLIC, 'result.absent with an unknown name'),
    Entry(
        DECOMPOSE,
        (289, 290),
        PUBLIC,
        'a ranking over a table whose name needs backticks',
    ),
    Entry(
        DECOMPOSE,
        (292, 293),
        PUBLIC,
        'a ranking over a column whose name needs backticks',
    ),
    Entry(
        LOCAL_TABLE,
        (74, 75),
        PUBLIC,
        'a frame given a multi-index after Schema.from_data',
    ),
    Entry(
        LOCAL_TABLE,
        (76, 77),
        PUBLIC,
        'a frame given repeated column names after Schema.from_data',
    ),
    Entry(
        LOCAL_TABLE,
        (78, 79),
        PUBLIC,
        'a frame given an empty column name after Schema.from_data',
    ),
    Entry(
        BASE_TABLE,
        (404, 405),
        PUBLIC,
        'a part of a composite key dropped from its frame after '
        'Schema.from_data',
    ),
    Entry(TABULAR, (338, 339), PUBLIC, 'a SourceTable put into schema.tables'),
    # Wrong or missing route, and no host test (the second list).
    Entry(
        ADAPTER,
        (90, 96),
        ROUTE,
        'Request.materialized(...) without requested_inference',
    ),
    Entry(
        CLIENT_PY,
        (887, 888),
        ROUTE,
        'FittedModel.predict of an envelope with no instance_table',
    ),
    Entry(
        PAYLOAD, (211, 213), ROUTE, 'an edited envelope, Request.materialized'
    ),
    Entry(
        PAYLOAD, (214, 206), ROUTE, 'an edited envelope, Request.materialized'
    ),
    Entry(
        PAYLOAD, (216, 215), ROUTE, 'an edited envelope, Request.materialized'
    ),
    Entry(
        PAYLOAD, (251, 252), ROUTE, 'an edited envelope, Request.materialized'
    ),
    Entry(
        PAYLOAD, (256, 257), ROUTE, 'an edited envelope, Request.materialized'
    ),
    Entry(
        REQUEST,
        (216, 217),
        ROUTE,
        'request.entity_table = None on a request built from frames',
    ),
    # "Reachable only" claims that are too narrow (the third list).
    Entry(
        RELATIONAL,
        (383, 382),
        NARROW,
        'a frame changed in place; a column added to schema[t].columns; '
        'a hand-built Table',
        routes=3,
    ),
    Entry(
        RELATIONAL,
        (728, 729),
        NARROW,
        'a frame changed in place; a column added to schema[t].columns; '
        'a hand-built Table',
        routes=3,
    ),
    Entry(
        RELATIONAL,
        (1475, 1479),
        NARROW,
        'a frame changed in place; a relationship over a column added to '
        'schema[t].columns',
        routes=2,
    ),
    Entry(
        TABULAR,
        (304, 305),
        NARROW,
        'a frame changed in place; a target added to schema[t].columns',
        routes=2,
    ),
    Entry(
        SOURCES,
        (58, 59),
        NARROW,
        'the Schema(..., source=...) constructor; a SourceTable put into a '
        'Schema.from_source schema.tables',
        routes=2,
    ),
    Entry(
        SOURCES,
        (59, 63),
        NARROW,
        'the Schema(..., source=...) constructor; a SourceTable put into a '
        'Schema.from_source schema.tables',
        routes=2,
    ),
    # Still unreachable, but the reason is false (the fourth list).
    Entry(
        PAYLOAD,
        (926, 927),
        FALSE,
        'explain is never forwarded to the engine',
        false=('never gathers',),
    ),
    Entry(
        PAYLOAD,
        (1001, 1002),
        FALSE,
        'the LIST_DISTINCT filter drops entities with no values',
        false=('decomposed before gathering',),
    ),
    Entry(
        TASK,
        (380, 382),
        FALSE,
        'link prediction is gathered',
        false=('decomposed before gathering',),
    ),
    Entry(
        RFM,
        (279, 280),
        FALSE,
        'link prediction is gathered',
        false=('decomposed before gathering',),
    ),
    Entry(
        TASK,
        (211, 212),
        FALSE,
        "anchor_time='entity' gathers a task without an anchor time column",
        false=('has an anchor time column',),
    ),
    Entry(
        TASK,
        (228, 227),
        FALSE,
        "anchor_time='entity' gathers a task without an anchor time column",
        false=('has an anchor time column',),
    ),
    # Minor: the wrong guard or rationale.
    Entry(
        ADAPTER,
        (766, 767),
        FALSE,
        'the guard is PredictionItem.from_dict',
        false=('the engine always writes an entity reference and row indices',),
    ),
    Entry(
        TABULAR_ADAPTER,
        (504, 505),
        FALSE,
        'the guard is PredictionItem.from_dict',
        false=(
            'the tabular envelope always carries a predict table and row '
            'indices',
        ),
    ),
    Entry(
        BASE_TABLE,
        (473, 474),
        FALSE,
        'SQL composite keys fold before the reserved-prefix refusal',
        false=('a SQL composite key is refused by the reserved-prefix check',),
    ),
    *(
        Entry(
            BASE_TABLE,
            branch,
            FALSE,
            'SQL composite keys fold before the reserved-prefix refusal',
            false=('which the reserved-prefix check refuses',),
        )
        for branch in ((495, 496), (497, 499), (538, 539), (542, 544))
    ),
    Entry(
        PAYLOAD,
        (1246, 1247),
        FALSE,
        'the .item() call earlier in _json_value',
        false=('columns hold Timestamp',),
    ),
)

#: A host test, cited by its pytest node id.
NODE_ID = re.compile(r'tests/rust_core/[\w./-]+\.py::\w+(?:\[[^\]\s]+\])?')
REACHABLE_ONLY = 'Reachable only'


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
    historical = {entry.module for entry in ENTRIES}
    current = [_current_path(module) for module in historical]
    assert len(current) == len(set(current)), (
        'duplicate projected source module'
    )
    assert all((REPO / module).is_file() for module in current), (
        'missing current source module'
    )
    return sorted(current)


def _branches(report: dict[str, Any]) -> dict[str, set[tuple[int, int]]]:
    out: dict[str, set[tuple[int, int]]] = {}
    for module, body in report.get('files', {}).items():
        key = _historical_path(module)
        assert key not in out, 'duplicate projected coverage module'
        out[key] = {(a, b) for a, b in body.get('executed_branches', []) or []}
    return out


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    # Not ``scripts/parity``, whose ``coverage.py`` would shadow coverage.
    paths = [str(HERE)]
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    env.setdefault('PYTHONHASHSEED', '0')
    return env


def held_measure() -> dict[str, Any]:
    r"""What ``_held.py`` reports: the branches of the entries' modules that
    the held recorded cases take in today's Python, the cases it ran, and
    any transcript that did not replay."""
    with tempfile.TemporaryDirectory(prefix='sdm-p86-held-') as tmp:
        out = Path(tmp) / 'held.json'
        proc = subprocess.run(
            [sys.executable, str(HERE / '_held.py'), '--out', str(out)],
            cwd=REPO,
            env=_environment(),
            capture_output=True,
            text=True,
            timeout=7200,
        )
        assert proc.returncode == 0, (
            f'_held.py failed ({proc.returncode}):\n'
            f'{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}'
        )
        found = json.loads(out.read_text('utf-8'))
    found['executed'] = _branches(found['coverage'])
    return found


def host_measure(
    node_ids: list[str], extra: dict[str, str] | None = None
) -> dict[str, Any]:
    r"""Each node id run by pytest under branch coverage of today's Python:
    its outcome, the calls made into ``sdm_core._native`` while it ran, and
    the branches of the entries' modules it took (``_host_plugin.py``)."""
    if not node_ids:
        return {}
    with tempfile.TemporaryDirectory(prefix='sdm-p86-host-') as tmp:
        out = Path(tmp) / 'host.json'
        env = {**_environment(), **(extra or {})}
        env['SDM_P86_HOST_OUT'] = str(out)
        env['SDM_P86_HOST_MODULES'] = json.dumps(modules())
        proc = subprocess.run(
            [
                sys.executable,
                '-m',
                'pytest',
                *node_ids,
                '-q',
                '-p',
                'no:cacheprovider',
                '-p',
                '_host_plugin',
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
                f'the cited host tests did not run ({proc.returncode}):\n'
                f'{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}'
            )
        found = json.loads(out.read_text('utf-8'))
    for body in found.values():
        executed = {}
        for module, branches in body.get('executed', {}).items():
            key = _historical_path(module)
            assert key not in executed, 'duplicate projected coverage module'
            executed[key] = {(a, b) for a, b in branches}
        body['executed'] = executed
    return found


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
