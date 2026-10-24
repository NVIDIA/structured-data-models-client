#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Tools for a held-out case set: author it, freeze it, run it once, count.

A held-out set is a directory of ``sdm-case/1`` files, one per case, named
``<id>.json``, every id under one prefix (Phase 1's was ``h11.``), written
by someone who implements nothing and kept where implementers cannot read
it until its one run (docs/design/rust-core/orchestration/briefs/
heldout-author.md). ``heldout_run.sh`` drives the run; these are its steps.

    heldout.py validate DIR --prefix P [--contract DIR] [--verbose]
    heldout.py digests  DIR --prefix P [--bundle-sha HEX] [--inputs-sha HEX]
                            [--cases N] [--write]
    heldout.py expect   ORIG_DIR REF_JSONL OUT_DIR --prefix P [--guesses FILE]
    heldout.py same     REF_A_JSONL REF_B_JSONL
    heldout.py report   RUN_DIR --prefix P [--missing RUNNER=REASON]
    heldout.py counts   RUN_DIR --prefix P --threshold N [--approved FILE]
                            [--header FILE] [--missing RUNNER=REASON]
    heldout.py convert  ORIG_DIR OUT_DIR --prefix P [--contract DIR]
    heldout.py dummy    KIND OUT_DIR --prefix P [--threshold N]

- ``validate``: every case file holds to ``contract/case.schema.json``, its
  id is under the prefix, unique and its file's name, its indices are
  integers or text, and its base and ``setup`` build on a fresh in-memory
  DuckDB (needs ``duckdb``, ``jsonschema`` and ``pyyaml``).
- ``digests``: the set's two digests. ``BUNDLE_SHA256`` is the SHA-256 of
  the lines ``name\0sha256hex(content)\n`` of every file in DIR but the two
  digest files, in byte order of name; ``INPUTS_SHA256`` the same over the
  case files with their top-level ``expect`` removed, as canonical JSON
  (sorted keys, no spaces), so a set whose ``expect`` was regenerated keeps
  it. Each is checked against the value given and the digest file in DIR,
  where either is there; ``--write`` freezes a set by writing the files,
  and never overwrites one.
- ``expect``: writes a copy of ORIG_DIR to the empty OUT_DIR with each
  case's ``expect`` the outcome a ``python-ref`` run observed (the
  author's ``expect`` is a prediction), and the author's into ``--guesses``
  (``OUT_DIR/../writer-guesses.json`` by default).
- ``same``: two ``python-ref`` runs observed the same outcome of every case.
- ``report``: the per-case table of a run directory (``cases/``,
  ``python-ref.jsonl``, a ``.jsonl`` per runner, ``compare.txt``), written
  to ``report.md`` and ``differences.json``. One runner leg may be missing
  (``--missing python-old='exit 1'``): its column says so, and the counts
  are partial.
- ``counts``: per core runner, the panics (an ``INTERNAL_ERROR`` or a leg
  with no observation), the new refusals of an input the reference accepts
  and the legs whose outcome differs from the reference's, written to
  ``counts.md``, against ``--threshold``: more than N differing cases with
  no approval, any panic or any unapproved new refusal is a no-go
  (exit 1); a missing runner leg makes the counts partial (exit 3).
  ``--approved`` names what was approved before the run, one a line: a
  case id approves every leg of the case, ``<id>@<database>`` that leg.
  ``--databases`` (``report`` too) names the databases the run ran on, so
  a leg there that no file observed is counted, as missing; without it
  the legs are those the observations hold.
- ``convert``: a set in the held-out format (c8075715, T2.1j's landing)
  written into the empty OUT_DIR in the head's case format
  (:data:`FORMAT_MOVES`), each case held to the head's case schema.
- ``dummy``: a synthetic run directory ``counts`` must find a no-go
  (check 9's two dummy bundles): ``panic`` or ``over-threshold``.

A case runs on each database its file names (all three by default): its
observation on DuckDB is keyed by its bare id, elsewhere by
``<id>@<database>`` (``run_python.case_key``); each key is a leg of the
case. H11's threshold reads per case (T2.16a, the orchestrator's reading
of 2026-10-21): a case counts once however many of its legs differ; it
differs when any leg's outcome class differs from python-ref's line of the
same key, and it is approved only when every differing leg is; any leg's
panic, a leg python-ref observed and a core runner did not among them,
and any leg's unapproved new refusal is a no-go.

Nothing here prints a case's contents: ``validate`` prints ids and the
place of each problem (``--verbose`` adds the message, which may quote the
case), ``digests`` digests, ``expect`` and ``convert`` counts and the place
of each problem; ``report`` and ``counts`` print outcomes, which is the
point of the run. Nothing deletes a file, and nothing writes into a case
directory but ``expect``, ``convert`` and ``dummy`` (each into an empty
one) and ``digests --write`` (its two files, once).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DIGEST_FILES = ('BUNDLE_SHA256', 'INPUTS_SHA256')
RUNNERS = ('python-core', 'python-old', 'rust-host')
CORE_RUNNERS = ('python-core', 'rust-host')


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def case_files(directory: Path, prefix: str) -> list[Path]:
    """The case files under the prefix, in byte order of name."""
    return sorted(
        (
            p
            for p in directory.iterdir()
            if p.is_file()
            and p.name.startswith(prefix)
            and p.name.endswith('.json')
        ),
        key=lambda p: p.name.encode(),
    )


def bundle_digest(directory: Path) -> str:
    names = sorted(
        (
            p.name
            for p in directory.iterdir()
            if p.is_file() and p.name not in DIGEST_FILES
        ),
        key=lambda n: n.encode(),
    )
    return _sha(
        b''.join(
            n.encode()
            + b'\0'
            + _sha((directory / n).read_bytes()).encode()
            + b'\n'
            for n in names
        )
    )


def inputs_digest(directory: Path, prefix: str) -> str:
    lines = b''
    for path in case_files(directory, prefix):
        data = json.loads(path.read_text(encoding='utf-8'))
        canon = json.dumps(
            {k: v for k, v in data.items() if k != 'expect'},
            sort_keys=True,
            separators=(',', ':'),
        ).encode()
        lines += path.name.encode() + b'\0' + _sha(canon).encode() + b'\n'
    return _sha(lines)


def _observations(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            o = json.loads(line)
            if o['case'] in out:
                raise SystemExit(f'{path.name}: duplicate observation')
            out[o['case']] = o
    return out


# --- validate ----------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        import duckdb
        import jsonschema
        import pyarrow  # noqa: F401 (each table is read as Arrow)
        import yaml
    except ImportError as missing:
        print(f'validate needs duckdb, pyarrow, jsonschema, pyyaml: {missing}')
        return 2
    contract: Path = args.contract
    schema = json.loads((contract / 'case.schema.json').read_text())
    bases = yaml.safe_load((contract / 'bases.yaml').read_text())['bases']
    validator = jsonschema.Draft202012Validator(schema)
    pinned = _pinned_duckdb(contract)
    if pinned and duckdb.__version__ != pinned:
        print(
            f'note: duckdb {duckdb.__version__} here, the reference '
            f'environment pins {pinned}: types may differ from the run'
        )
    problems, ids = 0, set()
    stray = [
        p.name
        for p in args.dir.glob('*.json')
        if not p.name.startswith(args.prefix)
    ]
    for name in sorted(stray):
        print(f'NAME {name}: not under the prefix {args.prefix!r}')
        problems += 1
    for path in case_files(args.dir, args.prefix):
        try:
            case = json.loads(path.read_text(encoding='utf-8'))
        except ValueError as error:
            print(f'JSON {path.name}: {type(error).__name__}')
            problems += 1
            continue
        for error in validator.iter_errors(case):
            where = error.json_path
            detail = f': {error.message[:200]}' if args.verbose else ''
            print(f'SCHEMA {path.name} {where} ({error.validator}){detail}')
            problems += 1
        if not isinstance(case, dict):
            continue
        cid = case.get('id')
        if not isinstance(cid, str) or not cid.startswith(args.prefix):
            print(f'ID {path.name}: id not under the prefix {args.prefix!r}')
            problems += 1
        elif cid in ids:
            print(f'ID {path.name}: id used twice')
            problems += 1
        elif path.name != f'{cid}.json':
            print(f'ID {path.name}: the file is not named <id>.json')
            problems += 1
        if isinstance(cid, str):
            ids.add(cid)
        given = case.get('input') if isinstance(case.get('input'), dict) else {}
        indices = (given.get('materialize') or {}).get('indices', [])
        if not isinstance(indices, list) or not all(
            isinstance(i, (int, str)) and not isinstance(i, bool)
            for i in indices
        ):
            print(f'INDICES {path.name}: an index is not an integer or text')
            problems += 1
        problems += _builds(duckdb, bases, given, path.name, args.verbose)
    print(f'cases {len(ids)} problems {problems}')
    return 0 if problems == 0 and ids else 1


def _pinned_duckdb(contract: Path) -> str | None:
    env = contract / 'reference-env.txt'
    if not env.is_file():
        return None
    for line in env.read_text().splitlines():
        if line.startswith('duckdb=='):
            return line.split('==', 1)[1].strip()
    return None


def _builds(duckdb: Any, bases: dict, given: dict, name: str, verbose: bool):
    base = given.get('base')
    if base not in bases:
        print(f'BASE {name}: no such base in bases.yaml')
        return 1
    con = duckdb.connect()
    try:
        for statement in bases[base] + list(given.get('setup', [])):
            con.execute(statement)
        tables = [
            t
            for (t,) in con.execute(
                'SELECT DISTINCT table_name FROM information_schema.columns'
            ).fetchall()
        ]
        for table in tables:  # every column reaches Arrow, as a runner reads it
            result = con.execute(f'SELECT * FROM "{table}"')
            (
                getattr(result, 'to_arrow_table', None)
                or result.fetch_arrow_table
            )()
    except Exception as error:  # any failure is the finding
        detail = f': {repr(error)[:300]}' if verbose else ''
        print(f'SQL {name}: {type(error).__name__}{detail}')
        return 1
    finally:
        con.close()
    print(f'OK {name}')
    return 0


# --- digests -----------------------------------------------------------------


def cmd_digests(args: argparse.Namespace) -> int:
    computed = {
        'BUNDLE_SHA256': bundle_digest(args.dir),
        'INPUTS_SHA256': inputs_digest(args.dir, args.prefix),
    }
    given = {
        'BUNDLE_SHA256': args.bundle_sha,
        'INPUTS_SHA256': args.inputs_sha,
    }
    count = len(case_files(args.dir, args.prefix))
    bad = args.cases is not None and count != args.cases
    print(f'cases: {count}' + (f' (expected {args.cases})' if bad else ''))
    for name, value in computed.items():
        recorded = args.dir / name
        wanted = {given[name]} if given[name] else set()
        if recorded.is_file():
            wanted.add(recorded.read_text().strip())
        if not wanted:
            verdict = 'not recorded'
        elif wanted == {value}:
            verdict = 'ok'
        else:
            verdict = 'MISMATCH'
            bad = True
        print(f'{name} {value} {verdict}')
    if args.write:
        if bad:
            print('not written: a digest or the count differs')
            return 1
        for name, value in computed.items():
            recorded = args.dir / name
            if recorded.exists():
                print(f'{name}: already recorded, left as it is')
                continue
            recorded.write_text(value + '\n')
            print(f'written: {recorded}')
    return 1 if bad else 0


# --- expect ------------------------------------------------------------------


def expect_of(outcome: dict[str, Any]) -> dict[str, Any]:
    if outcome['stage'] is None:
        raise SystemExit(
            'python-ref observed no stage: not a reference outcome'
        )
    expect = {'outcome': outcome['result'], 'stage': outcome['stage']}
    if outcome['result'] == 'refusal':
        expect['error'] = outcome['error']
        expect['code'] = outcome['code']
    return expect


def cmd_expect(args: argparse.Namespace) -> int:
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f'{out} holds files; write into an empty directory')
    guesses_file = args.guesses or out.parent / 'writer-guesses.json'
    if guesses_file.exists():
        raise SystemExit(f'{guesses_file} exists; it is never overwritten')
    observed = _observations(args.ref_jsonl)
    loader = _head_loader(ROOT / 'contract')
    guesses, changed = {}, 0
    for path in case_files(args.orig_dir, args.prefix):
        case = json.loads(path.read_text(encoding='utf-8'))
        keys = [
            loader.case_key(case['id'], leg) for leg in loader.case_legs(case)
        ]
        key = next((key for key in keys if key in observed), None)
        if key is None:
            raise SystemExit(f'{path.name}: no python-ref observation')
        new = expect_of(observed[key]['outcome'])
        old = case.get('expect') or {}
        guesses[case['id']] = case.get('expect')
        changed += {k: v for k, v in old.items() if k in new} != new
        case['expect'] = new
        (out / path.name).write_text(
            json.dumps(case, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
    guesses_file.write_text(
        json.dumps(guesses, indent=2, sort_keys=True) + '\n'
    )
    print(
        f'regenerated {len(guesses)} expects; {changed} differ from the '
        "author's prediction"
    )
    before = inputs_digest(args.orig_dir, args.prefix)
    after = inputs_digest(out, args.prefix)
    print(f'INPUTS_SHA256 {after} {"ok" if after == before else "MISMATCH"}')
    return 0 if after == before and guesses else 1


def cmd_same(args: argparse.Namespace) -> int:
    x, y = _observations(args.ref_a), _observations(args.ref_b)
    bad = sorted(
        k
        for k in set(x) | set(y)
        if (x.get(k) or {}).get('outcome') != (y.get(k) or {}).get('outcome')
    )
    print(
        f'python-ref outcomes: {len(x)} vs {len(y)} cases, {len(bad)} differ'
        + (f': {bad}' if bad else '')
    )
    return 1 if bad or len(x) != len(y) else 0


# --- report and counts -------------------------------------------------------

#: The databases a leg may be on, in the order a case's legs are listed:
#: DuckDB's leg keyed by the bare case id, every other ``<id>@<database>``
#: (``run_python.case_key``).
DATABASES = (
    'duckdb',
    'sqlite',
    'postgres',
    'databricks-sql',
    'databricks-spark',
    'snowflake',
    'lakebase',
)
#: The runners that run on DuckDB alone (the old client, T2.1a fix 4): their
#: legs elsewhere are not expected.
DUCKDB_ONLY = ('python-old',)


def case_of(key: str) -> str:
    r"""The case id of a leg's key, ``<id>`` or ``<id>@<database>``."""
    return key.partition('@')[0]


def database_of(key: str) -> str:
    r"""The database of a leg's key: DuckDB's is the bare case id."""
    return key.partition('@')[2] or 'duckdb'


def _leg_order(key: str) -> tuple[int, str]:
    database = database_of(key)
    known = DATABASES.index(database) if database in DATABASES else 99
    return known, key


def _missing(
    value: str | None, runners: tuple[str, ...] = RUNNERS
) -> tuple[str | None, str]:
    if value is None:
        return None, ''
    runner, _, why = value.partition('=')
    if runner not in runners:
        raise SystemExit(f'--missing: {runner!r} is not one of {runners}')
    return runner, why or 'no reason given'


def _legs(
    run: Path, gone: str | None, runners: tuple[str, ...] = RUNNERS
) -> dict[str, dict[str, dict]]:
    legs = {}
    for runner in runners:
        if runner == gone:
            legs[runner] = {}
            continue
        path = run / f'{runner}.jsonl'
        if not path.is_file():
            raise SystemExit(
                f'{path.name}: missing; if the leg failed, say so with '
                f"--missing {runner}='REASON'"
            )
        legs[runner] = _observations(path)
    return legs


def leg_keys(
    cases: list[str],
    *observed: dict[str, Any],
    expected: dict[str, set[str]] | None = None,
) -> dict[str, list[str]]:
    r"""Each case's legs: every key of the case any of ``observed`` holds
    (python-ref's and each runner's lines) and every key ``expected``
    names for it (the databases the run named, ``--databases``), its
    DuckDB leg (the bare id) where none is, in :data:`DATABASES` order."""
    found: dict[str, set[str]] = {
        c: set((expected or {}).get(c, ())) for c in cases
    }
    for lines in observed:
        for key in lines:
            if case_of(key) in found:
                found[case_of(key)].add(key)
    return {c: sorted(keys or {c}, key=_leg_order) for c, keys in found.items()}


def _expected(
    runner: str, key: str, java: frozenset[str] = frozenset()
) -> bool:
    r"""Whether ``runner`` is expected to observe the leg ``key``."""
    if runner == 'java':
        return key in java
    return runner not in DUCKDB_ONLY or database_of(key) == 'duckdb'


def _differences(compare_log: Path, prefix: str) -> dict[str, list[str]]:
    leg = re.compile(
        rf'(?<![\w.-]){re.escape(prefix)}[A-Za-z0-9_.-]+(?:@[a-z][a-z0-9-]*)?'
    )
    found: dict[str, list[str]] = {}
    if not compare_log.is_file():
        return found
    for line in compare_log.read_text(encoding='utf-8').splitlines():
        if not line.startswith(('DIFFERS ', 'diagnostic')):
            continue
        m = leg.search(line)
        if m:
            key = m.group(0)
            if '@' not in key:
                key = key.rstrip('.:')
            found.setdefault(key, []).append(line)
    return found


def _class(o: dict | None) -> str:
    if o is None:
        return 'NO OBSERVATION'
    out = o['outcome']
    if out['result'] == 'success':
        return 'success'
    return f'refusal {out.get("code") or out.get("error")}'


def expected_legs(
    run: Path, prefix: str, databases: list[str] | None
) -> dict[str, set[str]] | None:
    r"""Each case's legs on ``databases`` (the run's ``--databases``), as
    far as its file's ``databases`` lets it run there; None where the run
    named none, the legs then known from the observations alone."""
    if databases is None:
        return None
    out: dict[str, set[str]] = {}
    loader = _head_loader(ROOT / 'contract')
    for path in case_files(run / 'cases', prefix):
        case = json.loads(path.read_text(encoding='utf-8'))
        runs_on = loader.case_legs(case)
        out[case['id']] = {
            case['id'] if d == 'duckdb' else f'{case["id"]}@{d}'
            for d in databases
            if d in runs_on
        }
    return out


def _databases(value: str | None) -> list[str] | None:
    if value is None:
        return None
    named = [d for d in value.split(',') if d]
    unknown = sorted(set(named) - set(DATABASES))
    if unknown or not named:
        raise SystemExit(f'--databases: {value!r} is not of {DATABASES}')
    return named


def _case_ids(run: Path, prefix: str) -> list[str]:
    return sorted(
        json.loads(p.read_text(encoding='utf-8'))['id']
        for p in case_files(run / 'cases', prefix)
    )


def _partial_note(
    gone: str | None, why: str, runners: tuple[str, ...] = RUNNERS
) -> list[str]:
    if gone is None:
        return []
    kept = ', '.join(r for r in runners if r != gone)
    return [
        f'**Partial: the {gone} leg is missing ({why}); the owner rules.** '
        f'The counts below cover {kept} only.',
        '',
    ]


def _per_database(keys: list[str]) -> str:
    count = Counter(database_of(k) for k in keys)
    order = sorted(
        count, key=lambda d: (DATABASES.index(d) if d in DATABASES else 99, d)
    )
    return ', '.join(f'{d} {count[d]}' for d in order)


def _cell(
    runner: str,
    key: str,
    gone: str | None,
    obs: dict,
    java: frozenset[str] = frozenset(),
) -> str:
    r"""A runner's cell of a leg's row in ``report.md``."""
    if runner == gone:
        return 'NOT RUN'
    if key not in obs[runner] and not _expected(runner, key, java):
        return 'not run there'
    return _class(obs[runner].get(key))


def _java_legs(args: argparse.Namespace) -> frozenset[str]:
    """The authored subset, admitted by the actual Java runner's policy."""
    path = getattr(args, 'java_case_ids', None)
    if path is None:
        return frozenset()
    names = path.read_text().splitlines()
    ids = {line.strip() for line in names if line.strip()}
    if not ids or len(ids) != len([line for line in names if line.strip()]):
        raise SystemExit('--java-case-ids must hold unique case IDs')
    chosen = {}
    for path in case_files(args.run_dir / 'cases', args.prefix):
        case = json.loads(path.read_text())
        chosen[case['id']] = case
    if not ids <= chosen.keys():
        raise SystemExit('an authored Java case is missing from the bundle')
    runner = _head_loader(ROOT / 'contract', name='run_java')
    loader = _head_loader(ROOT / 'contract')
    databases = _databases(args.databases) or list(DATABASES)
    found = set()
    for cid in sorted(ids):
        legs = [leg for leg in databases if runner.on_java(chosen[cid], leg)]
        if not legs:
            raise SystemExit(f'{cid}: authored Java case has no admitted leg')
        found.update(loader.case_key(cid, leg) for leg in legs)
    return frozenset(found)


def cmd_report(args: argparse.Namespace) -> int:
    run: Path = args.run_dir
    java = _java_legs(args)
    runners = (
        (*RUNNERS, 'java') if getattr(args, 'java_case_ids', None) else RUNNERS
    )
    gone, why = _missing(args.missing, runners)
    cases = _case_ids(run, args.prefix)
    ref = _observations(run / 'python-ref.jsonl')
    obs = _legs(run, gone, runners)
    if 'java' in obs and set(obs['java']) - java:
        raise SystemExit('Java observed a case outside its authored subset')
    legs = leg_keys(
        cases,
        ref,
        *obs.values(),
        expected=expected_legs(run, args.prefix, _databases(args.databases)),
    )
    diffs = _differences(run / 'compare.txt', args.prefix)
    rows = []
    for c in cases:
        for key in legs[c]:
            cells = (_cell(w, key, gone, obs, java) for w in runners)
            rows.append(
                (key, _class(ref.get(key)), *cells, len(diffs.get(key, [])))
            )
    differing = sorted({case_of(k) for k in diffs if case_of(k) in legs})
    every = [k for c in cases for k in legs[c]]
    lines = [
        *_partial_note(gone, why, runners),
        '| case (leg) | python-ref | '
        + ' | '.join(runners)
        + ' | compare lines '
        '| entry approved before the run |',
        '|---' * (len(runners) + 4) + '|',
        *('| ' + ' | '.join(str(x) for x in row) + ' | |' for row in rows),
        '',
        f'cases: {len(cases)}; legs: {len(every)} ({_per_database(every)}); '
        f'cases with a compare line: {len(differing)}',
        'Fill the last column from the register as it was at the run start '
        'only; counts.md holds the outcome counts.',
    ]
    (run / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (run / 'differences.json').write_text(
        json.dumps(diffs, indent=2, sort_keys=True) + '\n'
    )
    print('\n'.join(lines[-2:]))
    print(f'written: {run / "report.md"}, {run / "differences.json"}')
    return 0


def _approved(path: Path | None) -> set[str]:
    approved = set()
    if path:
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.split('#', 1)[0].strip()
            if line:
                approved.add(line)
    return approved


def cmd_compare(args: argparse.Namespace) -> int:
    """The normal comparator on all cases, plus the authored Java subset."""
    import importlib.util
    import shutil

    run: Path = args.run_dir
    java = _java_legs(args)
    spec = importlib.util.spec_from_file_location(
        'heldout_compare', ROOT / 'scripts/rust_core/compare.py'
    )
    assert spec is not None and spec.loader is not None
    comparator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = comparator
    spec.loader.exec_module(comparator)
    regular = [run / (who + '.jsonl') for who in RUNNERS]
    java_files = []
    subset = run / 'java-compare'
    if java:
        subset.mkdir()
        (subset / 'cases').mkdir()
        ids = {case_of(key) for key in java}
        for path in case_files(run / 'cases', args.prefix):
            if json.loads(path.read_text())['id'] in ids:
                shutil.copyfile(path, subset / 'cases' / path.name)
        for who in ('python-ref', 'python-core', 'java'):
            rows = _observations(run / (who + '.jsonl'))
            selected = [
                value for key, value in rows.items() if case_of(key) in ids
            ]
            target = subset / (who + '.jsonl')
            target.write_text(
                ''.join(json.dumps(row) + '\n' for row in selected)
            )
            if who != 'python-ref':
                java_files.append(target)
    failed = False
    for database in _databases(args.databases) or list(DATABASES):
        expected = expected_legs(run, args.prefix, [database]) or {}
        if not any(expected.values()):
            continue
        comparisons = [(run / 'cases', run / 'python-ref.jsonl', regular)]
        if any(database_of(key) == database for key in java):
            comparisons.append(
                (subset / 'cases', subset / 'python-ref.jsonl', java_files)
            )
        for cases, reference, files in comparisons:
            report = comparator.compare(
                [cases],
                reference,
                files,
                register_file=run / 'register-at-run-start.yaml',
                improvements=run / 'register-at-run-start.yaml',
                database=database,
            )
            for line in report.notes:
                print(line)
            for line in report.texts:
                print('DIFFERS ' + line)
            failed |= bool(report.failures)
    return int(failed)


def cmd_counts(args: argparse.Namespace) -> int:
    run: Path = args.run_dir
    java = _java_legs(args)
    runners = (
        (*RUNNERS, 'java') if getattr(args, 'java_case_ids', None) else RUNNERS
    )
    core_runners = (*CORE_RUNNERS, 'java') if java else CORE_RUNNERS
    gone, why = _missing(args.missing, runners)
    cases = _case_ids(run, args.prefix)
    ref = _observations(run / 'python-ref.jsonl')
    obs = _legs(run, gone, runners)
    if 'java' in obs and set(obs['java']) - java:
        raise SystemExit('Java observed a case outside its authored subset')
    legs = leg_keys(
        cases,
        ref,
        *obs.values(),
        expected=expected_legs(run, args.prefix, _databases(args.databases)),
    )
    every = [k for c in cases for k in legs[c]]
    if any(key not in ref for key in every):
        raise SystemExit('a required reference observation is missing')
    approved = _approved(args.approved)

    def ok(key: str) -> bool:
        return key in approved or case_of(key) in approved

    compare_log = run / 'compare.txt'
    gating = []
    if compare_log.is_file():
        log = compare_log.read_text(encoding='utf-8').splitlines()
        if any(line.startswith('Traceback ') for line in log):
            raise SystemExit('comparison failed; this is not held-out evidence')
        gating = [line for line in log if line.startswith('DIFFERS ')]
        parsed = _differences(compare_log, args.prefix)
        if any(
            not any(line in texts for texts in parsed.values())
            for line in gating
        ):
            raise SystemExit(
                'comparison has an invalid input or global failure'
            )
    header = args.header.read_text().rstrip() if args.header else ''
    out = [header, ''] if header else []
    out += [*_partial_note(gone, why, runners)]
    out += [
        f'cases: {len(cases)}; legs: {len(every)} ({_per_database(every)}); '
        f'gating DIFFERS lines in compare.txt: {len(gating)} (held-out '
        'cases are not baselined, so outcome differences are counted here '
        'from the observations)',
        '',
    ]
    panics, new_refusals, differing = set(), set(), set()
    for key, texts in _differences(compare_log, args.prefix).items():
        if key in every and any(
            re.search(
                rf'{re.escape(key)}:\s*(?:python-core|rust-host|java)\b', text
            )
            for text in texts
        ):
            differing.add(key)
    for name in runners:
        if name == gone:
            out += [f'## {name}', f'- not run: {why}', '']
            continue
        o = obs[name]
        mine = [k for k in every if k in o or _expected(name, k, java)]
        crash = [
            k
            for k in mine
            if k not in o or o[k]['outcome'].get('code') == 'INTERNAL_ERROR'
        ]
        diff = [
            k for k in mine if k in o and _class(o[k]) != _class(ref.get(k))
        ]
        newref = [
            k
            for k in diff
            if k in ref
            and ref[k]['outcome']['result'] == 'success'
            and o[k]['outcome']['result'] == 'refusal'
        ]
        title = name
        if name in core_runners:
            panics |= set(crash)
            new_refusals |= set(newref)
            differing |= set(diff)
        else:
            title += ' (the old client: reported, not counted)'
        out += [
            f'## {title}',
            f'- legs: {len(mine)} ({_per_database(mine)})',
            f'- panics / INTERNAL_ERROR / missing: {len(crash)} {crash}',
            f'- new refusals of an input the reference accepts: '
            f'{len(newref)} {newref}',
            f'- outcome class differs from the reference: {len(diff)}',
            *(
                f'  - {k}: reference `{_class(ref.get(k))}`, '
                f'{name} `{_class(o[k])}`'
                + (' (approved before the run)' if ok(k) else '')
                for k in diff
            ),
            '',
        ]
    cases_differing = sorted({case_of(k) for k in differing})
    unapproved = sorted({case_of(k) for k in differing if not ok(k)})
    refusals = sorted((k for k in new_refusals if not ok(k)), key=_leg_order)
    crashed = sorted(panics, key=_leg_order)
    nogo = bool(panics) or bool(refusals) or len(unapproved) > args.threshold
    summary = [
        'On the core runners ('
        + ', '.join(core_runners)
        + '), on every database '
        '(a case counts once, however many of its legs differ):',
        f'- panics: {len(crashed)} {crashed}',
        f'- new refusals with no approval: {len(refusals)} {refusals}',
        f'- cases differing from the reference: {len(cases_differing)} '
        f'({len(differing)} legs); with no approval: {len(unapproved)} '
        f'{unapproved}',
        f'- threshold: more than {args.threshold} differing cases with no '
        'approval, any panic, or any new refusal with no approval = no-go',
        '',
        (
            'counts: PARTIAL (a leg is missing; the owner rules)'
            if gone in core_runners
            else f'counts: {"NO-GO" if nogo else "within the threshold"}'
        )
        + (
            ''
            if args.approved
            else ' (no --approved list: every difference counted as unapproved)'
        ),
    ]
    out += summary
    (run / 'counts.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
    print('\n'.join(summary))
    print(f'written: {run / "counts.md"}')
    if gone in core_runners:
        return 3
    return 1 if nogo else 0


# --- convert -----------------------------------------------------------------

#: The commit whose case format the held-out set was written in (T2.1j's
#: landing, heldout-record.md).
FORMAT_COMMIT = 'c80757150ab6daae04e94b1455fb9a5e912779e4'
#: Each field of the held-out format that the head's case format moved
#: since :data:`FORMAT_COMMIT`, as (where it was, where it is now), a path
#: of object keys from the case's top level. Empty at the head (T2.16a):
#: every field added since (T2.1m's ``lag_timesteps``, ``outputs`` and
#: ``quantiles``, T2.1n's ``explanation`` and ``item_metadata``, T2.1t's
#: sampling options, ``context_anchor_time`` and ``embeddings``) is
#: optional, its absence the call as the held-out format made it, and the
#: ranking and multi-call fields (``candidates``, ``top_k``,
#: ``rows_per_call``) are still ``input.materialize``'s. A move added here
#: is applied by :func:`converted`.
FORMAT_MOVES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = ()


def converted(
    case: dict[str, Any],
    moves: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = FORMAT_MOVES,
) -> tuple[dict[str, Any], int]:
    r"""``case`` in the head's format, and how many moves applied: each
    field at a move's old path taken to its new one (made where it is not
    there), never over a field already at the new path."""
    out = json.loads(json.dumps(case))
    applied = 0
    for old, new in moves:
        holder = out
        for key in old[:-1]:
            holder = holder.get(key) if isinstance(holder, dict) else None
        if not isinstance(holder, dict) or old[-1] not in holder:
            continue
        value = holder.pop(old[-1])
        target = out
        for key in new[:-1]:
            target = target.setdefault(key, {})
            if not isinstance(target, dict):
                raise ValueError(f'{"/".join(new)}: not an object on the way')
        if new[-1] in target:
            raise ValueError(f'{"/".join(new)}: already given')
        target[new[-1]] = value
        applied += 1
    return out, applied


def _head_loader(contract: Path, name: str = 'run_python') -> Any:
    r"""The head's case loader (``run_python.load_cases``): the case schema
    and the checks past it, as every runner reads a case."""
    import importlib.util

    path = contract / 'runners' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'heldout_{name}', path)
    if spec is None or spec.loader is None:
        raise SystemExit(f'{path}: cannot be loaded')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cmd_convert(args: argparse.Namespace) -> int:
    orig: Path = args.orig_dir
    out: Path = args.out_dir
    if orig.resolve() == out.resolve():
        raise SystemExit('convert writes a new directory, never the original')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f'{out} holds files; write into an empty directory')
    runner = _head_loader(args.contract)
    cases = case_files(orig, args.prefix)
    moved = problems = 0
    for path in sorted(orig.iterdir(), key=lambda p: p.name.encode()):
        if not path.is_file() or path.name in DIGEST_FILES:
            continue
        data = path.read_bytes()
        if path not in cases:
            (out / path.name).write_bytes(data)
            continue
        try:
            case, applied = converted(json.loads(data))
        except ValueError as error:
            print(f'MOVE {path.name}: {error}')
            problems += 1
            continue
        if applied:
            moved += 1
            data = (
                json.dumps(case, indent=2, ensure_ascii=False) + '\n'
            ).encode()
        (out / path.name).write_bytes(data)
        try:
            runner.load_cases([out / path.name])
        except runner.RunnerFailure as error:
            # The loader names the place and the rule, never a value.
            print(f'HEAD {error}')
            problems += 1
    print(
        f'cases {len(cases)}; moved {moved} (format {FORMAT_COMMIT[:8]} to '
        f'the head, {len(FORMAT_MOVES)} moves); problems {problems}'
    )
    for name, value in (
        ('BUNDLE_SHA256', bundle_digest(out)),
        ('INPUTS_SHA256', inputs_digest(out, args.prefix)),
    ):
        print(f'{name} {value} (converted)')
    for name, value in (
        ('BUNDLE_SHA256', bundle_digest(orig)),
        ('INPUTS_SHA256', inputs_digest(orig, args.prefix)),
    ):
        print(f'{name} {value} (original)')
    return 0 if problems == 0 and cases else 1


# --- dummy -------------------------------------------------------------------

DUMMY_KINDS = ('panic', 'over-threshold')


def _line(runner: str, key: str, outcome: dict[str, Any]) -> str:
    return json.dumps(
        {'case': key, 'identity': {'runner': runner}, 'outcome': outcome},
        sort_keys=True,
    )


def cmd_dummy(args: argparse.Namespace) -> int:
    r"""A synthetic run directory (check 9's dummy bundles): cases on DuckDB
    and SQLite, every leg as python-ref observed it but for the fault its
    kind holds. ``panic``: rust-host's SQLite leg of one case an
    ``INTERNAL_ERROR``. ``over-threshold``: python-core differs from the
    reference on one leg of each of threshold + 1 cases (DuckDB and SQLite
    in turn), each an input the reference refuses and it accepts, so
    neither a panic nor a new refusal: only the threshold makes it a no-go.
    One more case differs nowhere."""
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f'{out} holds files; write into an empty directory')
    (out / 'cases').mkdir()
    count = args.threshold + 2
    ids = [f'{args.prefix}dummy-{n:02d}' for n in range(count)]
    refused = {
        'result': 'refusal',
        'stage': 'materialize',
        'error': 'ValidationError',
        'code': 'INVALID_INPUT',
    }
    accepted = {'result': 'success', 'stage': 'predict'}
    lines: dict[str, list[str]] = {r: [] for r in ('python-ref', *RUNNERS)}
    for n, cid in enumerate(ids):
        (out / 'cases' / f'{cid}.json').write_text(
            json.dumps(
                {
                    'format': 'sdm-case/1',
                    'id': cid,
                    'why': f'a synthetic {args.kind} bundle (check 9)',
                },
                indent=2,
            )
            + '\n'
        )
        for database in ('duckdb', 'sqlite'):
            key = cid if database == 'duckdb' else f'{cid}@{database}'
            ref = refused
            lines['python-ref'].append(_line('python-ref', key, ref))
            for runner in RUNNERS:
                if runner in DUCKDB_ONLY and database != 'duckdb':
                    continue
                mine = ref
                if (
                    args.kind == 'panic'
                    and runner == 'rust-host'
                    and n == 0
                    and database == 'sqlite'
                ):
                    mine = {
                        'result': 'refusal',
                        'stage': 'materialize',
                        'error': 'StructuredDataError',
                        'code': 'INTERNAL_ERROR',
                    }
                if (
                    args.kind == 'over-threshold'
                    and runner == 'python-core'
                    and n <= args.threshold
                    and database == ('duckdb', 'sqlite')[n % 2]
                ):
                    mine = accepted
                lines[runner].append(_line(runner, key, mine))
    for runner, written in lines.items():
        (out / f'{runner}.jsonl').write_text('\n'.join(written) + '\n')
    print(f'written: a synthetic {args.kind} run of {count} cases in {out}')
    print(
        f'then: heldout.py counts {out} --prefix {args.prefix} '
        f'--threshold {args.threshold} (a no-go exits 1)'
    )
    return 0


# --- command line ------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    def command(name: str, function: Any) -> argparse.ArgumentParser:
        p = sub.add_parser(name)
        p.set_defaults(function=function)
        return p

    def prefix(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            '--prefix', required=True, help="the case-id prefix, e.g. 'h11.'"
        )

    p = command('validate', cmd_validate)
    p.add_argument('dir', type=Path)
    prefix(p)
    p.add_argument('--contract', type=Path, default=ROOT / 'contract')
    p.add_argument('--verbose', action='store_true')

    p = command('digests', cmd_digests)
    p.add_argument('dir', type=Path)
    prefix(p)
    p.add_argument('--bundle-sha')
    p.add_argument('--inputs-sha')
    p.add_argument('--cases', type=int)
    p.add_argument('--write', action='store_true')

    p = command('expect', cmd_expect)
    p.add_argument('orig_dir', type=Path)
    p.add_argument('ref_jsonl', type=Path)
    p.add_argument('out_dir', type=Path)
    prefix(p)
    p.add_argument('--guesses', type=Path)

    p = command('same', cmd_same)
    p.add_argument('ref_a', type=Path)
    p.add_argument('ref_b', type=Path)

    p = command('report', cmd_report)
    p.add_argument('run_dir', type=Path)
    prefix(p)
    p.add_argument('--missing', metavar='RUNNER=REASON')
    p.add_argument('--databases', metavar='LIST')
    p.add_argument('--java-case-ids', type=Path)

    p = command('compare', cmd_compare)
    p.add_argument('run_dir', type=Path)
    prefix(p)
    p.add_argument('--databases', metavar='LIST')
    p.add_argument('--java-case-ids', type=Path)

    p = command('counts', cmd_counts)
    p.add_argument('run_dir', type=Path)
    prefix(p)
    p.add_argument('--threshold', type=int, required=True)
    p.add_argument('--approved', type=Path)
    p.add_argument('--header', type=Path)
    p.add_argument('--missing', metavar='RUNNER=REASON')
    p.add_argument('--databases', metavar='LIST')
    p.add_argument('--java-case-ids', type=Path)

    p = command('convert', cmd_convert)
    p.add_argument('orig_dir', type=Path)
    p.add_argument('out_dir', type=Path)
    prefix(p)
    p.add_argument('--contract', type=Path, default=ROOT / 'contract')

    p = command('dummy', cmd_dummy)
    p.add_argument('kind', choices=DUMMY_KINDS)
    p.add_argument('out_dir', type=Path)
    prefix(p)
    p.add_argument('--threshold', type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser())
    if getattr(args, 'prefix', None) == '':
        raise SystemExit('--prefix must not be empty')
    return args.function(args)


if __name__ == '__main__':
    sys.exit(main())
