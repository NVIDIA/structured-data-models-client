#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Offline L7 (plan section 6, Phase 2 check 11): a one-time report at the
phase end against KO2-29's limits (KO2-53 and KO2-76: a report read by
both reviewers, not a gate step; no seeded-fault suite).

    python -I contract/tools/l7.py measure --mode python-ref --reference DIR
        --out REF.json [--repeats N] [--database D] CASES...
    python -I contract/tools/l7.py measure --mode python-core
        --out CORE.json [--repeats N] [--database D] CASES...
    python -I contract/tools/l7.py report --ref REF.json --core CORE.json
        [--observations FILE.jsonl ...] [--wheel WHL --baseline-wheel WHL]
        [--client DIR] --out l7.md

``measure`` runs from the interpreter of the environment it measures (the
reference's for ``python-ref``, one holding the candidate's wheel for
``python-core``), in the same cell for both. Each scenario is a case that
expects a success: its public calls (``Schema.from_source``,
``PredictiveQuery.materialize``, ``SDMClient.predict``) made ``--repeats``
times (5 by default) as ``run_python.py`` makes them, with no driver
capture, the scripted endpoint answering; the median of each call's
process CPU (``time.process_time``) is recorded, the case's setup and data
load not counted. Also recorded: ``import sdm_client``'s process CPU in a
new interpreter, ``--repeats`` times (on ``python-core`` with
``sdm_client._native`` imported too, which a core route loads), and on
``python-core`` the boundary, per scenario: the control messages the host
encodes and reads (``sdm_client._rust.codec``), their count, the largest's
size, and the host's CPU in the codec, plus the CPU of one crossing into
the native library (the median of a no-op call, ``reset_log_cache``,
timed in a loop) so the crossings' own cost is counted.

``report`` holds them to KO2-29's limits and writes ``l7.md``:

1. CPU: the median scenario's core-to-reference ratio at most 1.0, and
   every scenario's at most 1.1, over the scenarios complete on both
   sides: a scenario any repetition of which stopped before the stage its
   case runs to (a refusal, say) did less work, so it is listed as
   incomplete and left out of the verdict, as it is of item 4;
2. import time: the core's at most the reference client's;
3. native library: the candidate wheel's ``_native`` library at most 1.25
   times the baseline wheel's (Phase 1's merged wheel, built at 91a00319;
   design, "Gates");
4. in-memory boundary overhead per scenario (the codec's CPU and each
   crossing's): at most 5% of the scenario's CPU or 2 ms, whichever is
   larger;
5. control messages: none larger than the 64 KB cap (a larger one travels
   attached, decision H7, and is not a control message);
6. statements: each scenario's statement count per stage on each core
   runner's observations equal to python-ref's (``--observations``, each a
   runner's ``.jsonl``; ``driver.capture.statements``);
7. no pandas in providers: no provider module of the client imports or
   calls pandas (``--client``, the installed ``sdm_client`` by default).

Zero-copy is not measured here: ``tests/test_arrow_lifetimes.py`` holds it
(every buffer back at the input's address), run by CI's ``native-test-ops``
job on the exact SHA; the report says so. An item whose input is not given
is reported "not measured", never passed. Exit status: 0 when every
measured limit holds, 1 when one fails, 2 on a usage error or a run that
cannot be read. No value of a case is printed or written: ids, counts,
sizes and times only.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import re
import statistics
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CONTRACT = HERE.parent
REPO = CONTRACT.parent
FORMAT = 'sdm-l7/1'
#: KO2-29's limits (approved by owner 2026-10-16).
MEDIAN_CPU_RATIO = 1.0
SCENARIO_CPU_RATIO = 1.1
NATIVE_SIZE_RATIO = 1.25
BOUNDARY_SHARE = 0.05
BOUNDARY_FLOOR = 0.002
#: The control cap (plan section 3; ``sdm_client._rust.codec.CONTROL_CAP``).
CONTROL_CAP = 64 * 1024
#: The public calls a scenario's CPU is counted in, by stage.
CALLS = (
    ('from_source', 'Schema', 'from_source'),
    ('materialize', 'PredictiveQuery', 'materialize'),
    ('predict', 'SDMClient', 'predict'),
)
#: pandas in a provider's source: an import or a conversion call.
PANDAS = re.compile(
    r'^\s*(import\s+pandas|from\s+pandas\b)|\bto_pandas\(|\bfrom_pandas\('
    # DuckDB's own frames (T2.16a fix 3, the Claude review's P3).
    r'|\.(df|fetchdf|fetch_df|fetch_df_chunk)\(',
    re.M,
)
IMPORT_PROBE = (
    'import time\n'
    't = time.process_time()\n'
    'import sdm_client\n'
    '{native}'
    'print(time.process_time() - t)\n'
)


class Refused(Exception):
    r"""An input this report cannot read, named."""


def _runner() -> Any:
    r"""``contract/runners/run_python.py``, loaded by path: the case loader
    and the calls exactly as the runners make them."""
    path = CONTRACT / 'runners' / 'run_python.py'
    spec = importlib.util.spec_from_file_location('l7_run_python', path)
    if spec is None or spec.loader is None:
        raise Refused(f'{path}: cannot be loaded')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- measure -----------------------------------------------------------------


class _Clock:
    r"""The process CPU spent in each wrapped call, by stage, for one run of
    a case; a call made inside another is counted once, in the outer."""

    def __init__(self) -> None:
        self.spent: dict[str, float] = {}
        self._depth = 0

    def wrap(self, stage: str, function: Callable[..., Any]) -> Any:
        clock = self

        def timed(*args: Any, **kwargs: Any) -> Any:
            if clock._depth:
                return function(*args, **kwargs)
            clock._depth += 1
            start = time.process_time()
            try:
                return function(*args, **kwargs)
            finally:
                clock.spent[stage] = clock.spent.get(stage, 0.0) + (
                    time.process_time() - start
                )
                clock._depth -= 1

        return timed


class _Codec:
    r"""The control messages the Python host encodes and reads in one run of
    a case, their sizes, and its CPU in doing so."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.messages = 0
        self.largest = 0
        self.cpu = 0.0

    def wrap(self, function: Callable[..., Any], encodes: bool) -> Any:
        codec = self

        def timed(*args: Any, **kwargs: Any) -> Any:
            start = time.process_time()
            try:
                out = function(*args, **kwargs)
            finally:
                codec.cpu += time.process_time() - start
            message = out if encodes else (args[0] if args else None)
            if isinstance(message, (bytes, bytearray, memoryview)):
                codec.messages += 1
                codec.largest = max(codec.largest, len(message))
            return out

        return timed


def _crossing_cpu(repeats: int) -> float:
    r"""The process CPU of one crossing into the native library: the median
    over ``repeats`` loops of 2,000 no-op calls."""
    from sdm_client import _native

    loops = []
    for _ in range(repeats):
        start = time.process_time()
        for _ in range(2000):
            _native.reset_log_cache()
        loops.append((time.process_time() - start) / 2000)
    return statistics.median(loops)


def _import_cpu(core: bool, repeats: int) -> float:
    r"""``import sdm_client``'s process CPU in a new interpreter, the median
    of ``repeats``; with the native library too on the core."""
    probe = IMPORT_PROBE.format(
        native='import sdm_client._native\n' if core else ''
    )
    times = []
    for _ in range(repeats):
        done = subprocess.run(
            [sys.executable, '-I', '-c', probe],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        if done.returncode != 0:
            raise Refused('import sdm_client failed in a new interpreter')
        times.append(float(done.stdout.strip().splitlines()[-1]))
    return statistics.median(times)


def scenarios(cases: list[tuple[dict[str, Any], str]]) -> list[dict]:
    r"""The cases that are scenarios: each that expects a success."""
    return [
        case
        for case, _ in cases
        if (case.get('expect') or {}).get('outcome') == 'success'
    ]


def cmd_measure(args: argparse.Namespace) -> int:
    run = _runner()
    if (args.mode == 'python-ref') != (args.reference is not None):
        raise Refused('--reference is given in python-ref mode, and only there')
    guard = None
    if args.reference is not None:
        guard = run._Guard(args.reference)
        guard.check('before the first case')
        run._require_reference_client(args.reference)
    import sdm_client

    clock = _Clock()
    for stage, owner, name in CALLS:
        cls = getattr(sdm_client, owner)
        original = inspect.getattr_static(cls, name)
        if isinstance(original, classmethod):
            wrapped: Any = classmethod(clock.wrap(stage, original.__func__))
        elif isinstance(original, staticmethod):
            wrapped = staticmethod(clock.wrap(stage, original.__func__))
        else:
            wrapped = clock.wrap(stage, original)
        setattr(cls, name, wrapped)
    codec = _Codec()
    if args.mode == 'python-core':
        from sdm_client._rust import codec as host_codec

        host_codec.encode = codec.wrap(host_codec.encode, encodes=True)
        host_codec.read = codec.wrap(host_codec.read, encodes=False)
    cases = scenarios(run.load_cases(args.cases))
    if not cases:
        raise Refused('no case expects a success: no scenario to measure')
    bases = run.load_bases()
    measured = {}
    for case in cases:
        if args.database not in run.case_databases(case):
            continue
        runs: list[dict[str, float]] = []
        boundary: list[dict[str, Any]] = []
        outcomes: list[dict[str, Any]] = []
        for _ in range(args.repeats):
            clock.spent = {}
            codec.reset()
            if args.mode == 'python-core':
                os.environ['SDM_CORE_ROUTES'] = ','.join(
                    run.routes_on(case, args.database)
                )
            else:
                os.environ['SDM_CORE_ROUTES'] = ''
            try:
                outcomes.append(run._call(case, bases, False, args.database)[0])
            finally:
                os.environ.pop('SDM_CORE_ROUTES', None)
            runs.append(dict(clock.spent))
            boundary.append(
                {
                    'messages': codec.messages,
                    'largest': codec.largest,
                    'cpu': codec.cpu,
                }
            )
        # A scenario is complete only when every repetition answered
        # through the stage the case runs to: one that stopped early did
        # less work, and its CPU is no comparison (T2.16a fix 1).
        seen = sorted(
            {f'{o.get("result")} at {o.get("stage")}' for o in outcomes}
        )
        entry: dict[str, Any] = {
            'outcomes': seen,
            'complete': seen == [f'success at {case["until"]}'],
            'cpu': {
                stage: statistics.median(r.get(stage, 0.0) for r in runs)
                for stage, _, _ in CALLS
            },
        }
        entry['cpu']['total'] = statistics.median(sum(r.values()) for r in runs)
        if args.mode == 'python-core':
            entry['boundary'] = {
                'messages': max(b['messages'] for b in boundary),
                'largest': max(b['largest'] for b in boundary),
                'codec_cpu': statistics.median(b['cpu'] for b in boundary),
            }
        measured[run.case_key(case['id'], args.database)] = entry
    if guard is not None:
        guard.check('after the last case')
    versions = run._versions()
    document = {
        'format': FORMAT,
        'mode': args.mode,
        'database': args.database,
        'repeats': args.repeats,
        'cell': run._cell(versions),
        'versions': versions,
        # The client measured: the reference checkout's, or this tree's.
        'commit': run._git(args.reference or run.REPO),
        'import_cpu': _import_cpu(args.mode == 'python-core', args.repeats),
        'crossing_cpu': (
            _crossing_cpu(args.repeats) if args.mode == 'python-core' else None
        ),
        'scenarios': measured,
    }
    args.out.write_text(json.dumps(document, indent=1, sort_keys=True) + '\n')
    print(f'{args.mode}: {len(measured)} scenarios measured into {args.out}')
    return 0


# --- report ------------------------------------------------------------------


def _read(path: Path, mode: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise Refused(f'{path}: not a measurement it reads') from None
    if (
        not isinstance(document, dict)
        or document.get('format') != FORMAT
        or document.get('mode') != mode
        or not isinstance(document.get('scenarios'), dict)
    ):
        raise Refused(f'{path}: not a {mode} measurement ({FORMAT})')
    return document


def native_size(wheel: Path) -> int:
    r"""The uncompressed size of the wheel's ``sdm_client/_native`` library."""
    try:
        with zipfile.ZipFile(wheel) as archive:
            sizes = [
                info.file_size
                for info in archive.infolist()
                if re.fullmatch(
                    r'sdm_client/_native[^/]*\.(so|pyd|dylib)', info.filename
                )
            ]
    except (OSError, zipfile.BadZipFile):
        raise Refused(f'{wheel}: not a wheel it reads') from None
    if len(sizes) != 1:
        raise Refused(f'{wheel}: not one sdm_client/_native library')
    return sizes[0]


def statement_counts(line: dict[str, Any]) -> dict[str, int] | None:
    r"""A line's statements per stage, or None where it recorded none."""
    capture = (line.get('driver') or {}).get('capture')
    if not isinstance(capture, dict):
        return None
    counts: dict[str, int] = {}
    for statement in capture.get('statements') or []:
        stage = statement.get('stage') or '-'
        counts[stage] = counts.get(stage, 0) + 1
    return counts


def _lines(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        raise Refused(f'{path}: cannot be read') from None
    for line in text.splitlines():
        if line.strip():
            try:
                found = json.loads(line)
            except ValueError:
                raise Refused(f'{path}: a line is not JSON') from None
            out[found['case']] = found
    return out


def providers_with_pandas(client: Path) -> list[str]:
    r"""The provider modules of the client at ``client`` that use pandas."""
    folder = client / '_rust' / 'providers'
    if not folder.is_dir():
        raise Refused(f'{folder}: no providers to read')
    return sorted(
        path.name
        for path in folder.glob('*.py')
        if PANDAS.search(path.read_text(encoding='utf-8'))
    )


def _ratio(core: float, ref: float) -> float:
    return core / ref if ref > 0 else (1.0 if core <= 0 else float('inf'))


def judge(
    ref: dict[str, Any],
    core: dict[str, Any],
    observations: dict[str, dict[str, dict[str, Any]]],
    sizes: tuple[int, int] | None,
    pandas: list[str] | None,
) -> tuple[list[str], bool]:
    r"""``l7.md``'s lines and whether every measured limit holds."""
    if ref.get('cell') != core.get('cell') or ref.get('cell') is None:
        raise Refused(
            'the two measurements are not of one known cell: '
            f'{ref.get("cell")} and {core.get("cell")}'
        )
    held = True
    lines = [
        '# Offline L7 against KO2-29',
        '',
        f'Cell {core["cell"]}, database {core.get("database")}, '
        f'{core.get("repeats")} repeats; core commit '
        f'{(core.get("commit") or {}).get("sha", "?")[:12]}, reference '
        f'commit {(ref.get("commit") or {}).get("sha", "?")[:12]}.',
        '',
    ]
    both = sorted(set(ref['scenarios']) & set(core['scenarios']))
    only = sorted(set(ref['scenarios']) ^ set(core['scenarios']))
    # Only a scenario every repetition of both completed is judged.
    incomplete = [
        key
        for key in both
        if ref['scenarios'][key].get('complete') is not True
        or core['scenarios'][key].get('complete') is not True
    ]
    judged = [key for key in both if key not in incomplete]
    rows = []
    for key in judged:
        r, c = ref['scenarios'][key], core['scenarios'][key]
        rows.append((key, r['cpu']['total'], c['cpu']['total']))
    ratios = [_ratio(c, r) for _, r, c in rows]
    median = statistics.median(ratios) if ratios else None
    slow = [
        (key, ratio)
        for (key, _, _), ratio in zip(rows, ratios, strict=True)
        if ratio > SCENARIO_CPU_RATIO
    ]
    cpu_ok = bool(ratios) and median <= MEDIAN_CPU_RATIO and not slow
    held &= cpu_ok
    lines += [
        '## 1. CPU per scenario',
        '',
        f'- scenarios measured on both: {len(both)}'
        + (f'; on one only (not judged): {only}' if only else ''),
        f'- incomplete on either side, a repetition stopping before the '
        f'stage the case runs to (not judged): {len(incomplete)}',
        *(
            f'  - {key}: reference {ref["scenarios"][key].get("outcomes")}, '
            f'core {core["scenarios"][key].get("outcomes")}'
            for key in incomplete
        ),
        f'- judged: {len(judged)}',
        '- median core-to-reference ratio: '
        + (f'{median:.3f}' if median is not None else 'none')
        + f' (limit {MEDIAN_CPU_RATIO})',
        f'- scenarios above {SCENARIO_CPU_RATIO}: {len(slow)}',
        *(f'  - {key}: {ratio:.3f}' for key, ratio in slow),
        f'- {"holds" if cpu_ok else "FAILS"}',
        '',
        '| scenario | reference CPU (ms) | core CPU (ms) | ratio |',
        '|---|---|---|---|',
        *(
            f'| {key} | {r * 1e3:.2f} | {c * 1e3:.2f} | {ratio:.3f} |'
            for (key, r, c), ratio in zip(rows, ratios, strict=True)
        ),
        '',
    ]
    imports = (core['import_cpu'], ref['import_cpu'])
    import_ok = imports[0] <= imports[1]
    held &= import_ok
    lines += [
        '## 2. Import time',
        '',
        f'- core (with `sdm_client._native`): {imports[0] * 1e3:.1f} ms; '
        f'reference: {imports[1] * 1e3:.1f} ms; '
        f'{"holds" if import_ok else "FAILS"}',
        '',
        '## 3. Native library size',
        '',
    ]
    if sizes is None:
        lines += ['- not measured (no --wheel and --baseline-wheel)', '']
    else:
        size_ok = sizes[0] <= NATIVE_SIZE_RATIO * sizes[1]
        held &= size_ok
        lines += [
            f'- candidate {sizes[0]:,} bytes; baseline {sizes[1]:,} bytes; '
            f'ratio {sizes[0] / sizes[1]:.3f} (limit {NATIVE_SIZE_RATIO}); '
            f'{"holds" if size_ok else "FAILS"}',
            '',
        ]
    crossing = core.get('crossing_cpu') or 0.0
    over, largest, rows_b = [], 0, []
    for key in judged:
        c = core['scenarios'][key]
        b = c.get('boundary') or {}
        overhead = b.get('codec_cpu', 0.0) + b.get('messages', 0) * crossing
        bound = max(BOUNDARY_SHARE * c['cpu']['total'], BOUNDARY_FLOOR)
        largest = max(largest, b.get('largest', 0))
        rows_b.append((key, overhead, bound, b.get('messages', 0)))
        if overhead > bound:
            over.append(key)
    boundary_ok = bool(judged) and not over
    held &= boundary_ok
    worst = max(rows_b, key=lambda row: row[1] / row[2], default=None)
    lines += [
        '## 4. In-memory boundary overhead',
        '',
        f"- per crossing: {crossing * 1e6:.2f} us; overhead = the codec's "
        "CPU + messages x the crossing's",
        f'- scenarios over max(5% of their CPU, 2 ms): {len(over)} {over}',
        (
            f'- the closest: {worst[0]}, {worst[1] * 1e3:.3f} ms of '
            f'{worst[2] * 1e3:.3f} ms ({worst[3]} messages)'
            if worst
            else '- none measured'
        ),
        f'- {"holds" if boundary_ok else "FAILS"}',
        '',
    ]
    cap_ok = largest <= CONTROL_CAP
    held &= cap_ok
    lines += [
        '## 5. Control messages',
        '',
        f'- the largest: {largest:,} bytes (cap {CONTROL_CAP:,}); '
        f'{"holds" if cap_ok else "FAILS"}',
        '',
        '## 6. Statements per scenario',
        '',
    ]
    reference = observations.get('python-ref')
    if not reference:
        lines += ['- not measured (no python-ref observations given)', '']
    else:
        differ: list[str] = []
        compared = 0
        for runner in sorted(set(observations) - {'python-ref'}):
            for key in both:
                if key not in reference or key not in observations[runner]:
                    continue
                want = statement_counts(reference[key])
                got = statement_counts(observations[runner][key])
                if want is None or got is None:
                    continue
                compared += 1
                if want != got:
                    differ.append(f'{runner} {key}: {got} against {want}')
        statements_ok = compared > 0 and not differ
        held &= statements_ok
        lines += [
            f'- scenario lines compared: {compared}; differing: {len(differ)}',
            *(f'  - {text}' for text in differ),
            f'- {"holds" if statements_ok else "FAILS"}'
            + ('' if compared else ' (nothing compared)'),
            '',
        ]
    lines += ['## 7. No pandas in providers', '']
    if pandas is None:
        lines += ['- not measured (no client to read)', '']
    else:
        held &= not pandas
        lines += [
            f'- provider modules using pandas: {len(pandas)} {pandas}; '
            f'{"holds" if not pandas else "FAILS"}',
            '',
        ]
    lines += [
        '## Zero-copy',
        '',
        '- not measured here: `tests/test_arrow_lifetimes.py` holds it, in '
        "CI's `native-test-ops` job on the exact SHA.",
        '',
        f'**L7: {"every measured limit holds" if held else "a limit FAILS"}**',
    ]
    return lines, held


def cmd_report(args: argparse.Namespace) -> int:
    ref = _read(args.ref, 'python-ref')
    core = _read(args.core, 'python-core')
    observations: dict[str, dict[str, dict[str, Any]]] = {}
    for path in args.observations:
        lines = _lines(path)
        runners = {
            ((line.get('identity') or {}).get('runner'))
            for line in lines.values()
        }
        if len(runners) != 1 or None in runners:
            raise Refused(f"{path}: not one runner's observations")
        observations[runners.pop()] = lines
    sizes = None
    if (args.wheel is None) != (args.baseline_wheel is None):
        raise Refused('give --wheel and --baseline-wheel together')
    if args.wheel is not None:
        sizes = (native_size(args.wheel), native_size(args.baseline_wheel))
    client = args.client
    if client is None:
        spec = importlib.util.find_spec('sdm_client')
        if spec is not None and spec.origin:
            client = Path(spec.origin).parent
    pandas = providers_with_pandas(client) if client is not None else None
    lines, held = judge(ref, core, observations, sizes, pandas)
    args.out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(lines[-1])
    print(f'written: {args.out}')
    return 0 if held else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('measure')
    p.set_defaults(function=cmd_measure)
    p.add_argument(
        '--mode', choices=('python-ref', 'python-core'), required=True
    )
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--database', default='duckdb')
    p.add_argument('cases', type=Path, nargs='+')
    p = sub.add_parser('report')
    p.set_defaults(function=cmd_report)
    p.add_argument('--ref', type=Path, required=True)
    p.add_argument('--core', type=Path, required=True)
    p.add_argument('--observations', type=Path, nargs='*', default=[])
    p.add_argument('--wheel', type=Path)
    p.add_argument('--baseline-wheel', type=Path)
    p.add_argument('--client', type=Path)
    p.add_argument('--out', type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if getattr(args, 'repeats', 1) < 1:
        print('l7: --repeats must be at least 1', file=sys.stderr)
        return 2
    try:
        return args.function(args)
    except Refused as refused:
        print(f'l7: {refused}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
