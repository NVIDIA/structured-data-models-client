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
- ``counts``: per core runner, the panics (an ``INTERNAL_ERROR`` or a case
  with no observation), the new refusals of an input the reference accepts
  and the cases whose outcome differs from the reference's, written to
  ``counts.md``, against ``--threshold``: more than N differing cases with
  no approval, any panic or any unapproved new refusal is a no-go
  (exit 1); a missing leg makes the counts partial (exit 3). ``--approved``
  names the case ids approved before the run, one a line.

Nothing here prints a case's contents: ``validate`` prints ids and the
place of each problem (``--verbose`` adds the message, which may quote the
case), ``digests`` digests, ``expect`` counts; ``report`` and ``counts``
print outcomes, which is the point of the run. Nothing deletes a file, and
nothing writes into a case directory but ``expect`` (into an empty one) and
``digests --write`` (its two files, once).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
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
    guesses, changed = {}, 0
    for path in case_files(args.orig_dir, args.prefix):
        case = json.loads(path.read_text(encoding='utf-8'))
        if case['id'] not in observed:
            raise SystemExit(f'{path.name}: no python-ref observation')
        new = expect_of(observed[case['id']]['outcome'])
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


def _missing(value: str | None) -> tuple[str | None, str]:
    if value is None:
        return None, ''
    runner, _, why = value.partition('=')
    if runner not in RUNNERS:
        raise SystemExit(f'--missing: {runner!r} is not one of {RUNNERS}')
    return runner, why or 'no reason given'


def _legs(run: Path, gone: str | None) -> dict[str, dict[str, dict]]:
    legs = {}
    for runner in RUNNERS:
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


def _differences(compare_log: Path, prefix: str) -> dict[str, list[str]]:
    case = re.compile(rf'(?<![\w.-]){re.escape(prefix)}[A-Za-z0-9_.-]+')
    found: dict[str, list[str]] = {}
    if not compare_log.is_file():
        return found
    for line in compare_log.read_text(encoding='utf-8').splitlines():
        if not line.startswith(('DIFFERS ', 'diagnostic')):
            continue
        m = case.search(line)
        if m:
            found.setdefault(m.group(0).rstrip('.:'), []).append(line)
    return found


def _class(o: dict | None) -> str:
    if o is None:
        return 'NO OBSERVATION'
    out = o['outcome']
    if out['result'] == 'success':
        return 'success'
    return f'refusal {out.get("code") or out.get("error")}'


def _case_ids(run: Path, prefix: str) -> list[str]:
    return sorted(
        json.loads(p.read_text(encoding='utf-8'))['id']
        for p in case_files(run / 'cases', prefix)
    )


def _partial_note(gone: str | None, why: str) -> list[str]:
    if gone is None:
        return []
    kept = ', '.join(r for r in RUNNERS if r != gone)
    return [
        f'**Partial: the {gone} leg is missing ({why}); the owner rules.** '
        f'The counts below cover {kept} only.',
        '',
    ]


def cmd_report(args: argparse.Namespace) -> int:
    run: Path = args.run_dir
    gone, why = _missing(args.missing)
    cases = _case_ids(run, args.prefix)
    ref = _observations(run / 'python-ref.jsonl')
    obs = _legs(run, gone)
    diffs = _differences(run / 'compare.txt', args.prefix)
    rows = []
    for c in cases:
        cells = (
            'NOT RUN' if w == gone else _class(obs[w].get(c)) for w in RUNNERS
        )
        rows.append((c, _class(ref.get(c)), *cells, len(diffs.get(c, []))))
    differing = [c for c in cases if diffs.get(c)]
    lines = [
        *_partial_note(gone, why),
        '| case | python-ref | ' + ' | '.join(RUNNERS) + ' | compare lines '
        '| entry approved before the run |',
        '|---' * (len(RUNNERS) + 4) + '|',
        *('| ' + ' | '.join(str(x) for x in row) + ' | |' for row in rows),
        '',
        f'cases: {len(cases)}; with a compare line: {len(differing)}',
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


def cmd_counts(args: argparse.Namespace) -> int:
    run: Path = args.run_dir
    gone, why = _missing(args.missing)
    cases = _case_ids(run, args.prefix)
    ref = _observations(run / 'python-ref.jsonl')
    obs = _legs(run, gone)
    approved = set()
    if args.approved:
        for line in args.approved.read_text(encoding='utf-8').splitlines():
            line = line.split('#', 1)[0].strip()
            if line:
                approved.add(line)
    compare_log = run / 'compare.txt'
    gating = []
    if compare_log.is_file():
        gating = [
            line
            for line in compare_log.read_text(encoding='utf-8').splitlines()
            if line.startswith('DIFFERS ')
        ]
    header = args.header.read_text().rstrip() if args.header else ''
    out = [header, ''] if header else []
    out += [*_partial_note(gone, why)]
    out += [
        f'cases: {len(cases)}; gating DIFFERS lines in compare.txt: '
        f'{len(gating)} (held-out cases are not baselined, so outcome '
        'differences are counted here from the observations)',
        '',
    ]
    panics, new_refusals, differing = set(), set(), set()
    for name in RUNNERS:
        if name == gone:
            out += [f'## {name}', f'- not run: {why}', '']
            continue
        o = obs[name]
        crash = [
            c
            for c in cases
            if c not in o or o[c]['outcome'].get('code') == 'INTERNAL_ERROR'
        ]
        diff = [
            c for c in cases if c in o and _class(o[c]) != _class(ref.get(c))
        ]
        newref = [
            c
            for c in diff
            if c in ref
            and ref[c]['outcome']['result'] == 'success'
            and o[c]['outcome']['result'] == 'refusal'
        ]
        title = name
        if name in CORE_RUNNERS:
            panics |= set(crash)
            new_refusals |= set(newref)
            differing |= set(diff)
        else:
            title += ' (the old client: reported, not counted)'
        out += [
            f'## {title}',
            f'- panics / INTERNAL_ERROR / missing: {len(crash)} {crash}',
            f'- new refusals of an input the reference accepts: '
            f'{len(newref)} {newref}',
            f'- outcome class differs from the reference: {len(diff)}',
            *(
                f'  - {c}: reference `{_class(ref.get(c))}`, '
                f'{name} `{_class(o[c])}`'
                + (' (approved before the run)' if c in approved else '')
                for c in diff
            ),
            '',
        ]
    unapproved = sorted(differing - approved)
    refusals = sorted(new_refusals - approved)
    nogo = bool(panics) or bool(refusals) or len(unapproved) > args.threshold
    summary = [
        'On either core runner (python-core, rust-host):',
        f'- panics: {len(panics)} {sorted(panics)}',
        f'- new refusals with no approval: {len(refusals)} {refusals}',
        f'- cases differing from the reference: {len(differing)}; with no '
        f'approval: {len(unapproved)} {unapproved}',
        f'- threshold: more than {args.threshold} differing cases with no '
        'approval, any panic, or any new refusal with no approval = no-go',
        '',
        (
            'counts: PARTIAL (a leg is missing; the owner rules)'
            if gone in CORE_RUNNERS
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
    if gone in CORE_RUNNERS:
        return 3
    return 1 if nogo else 0


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

    p = command('counts', cmd_counts)
    p.add_argument('run_dir', type=Path)
    prefix(p)
    p.add_argument('--threshold', type=int, required=True)
    p.add_argument('--approved', type=Path)
    p.add_argument('--header', type=Path)
    p.add_argument('--missing', metavar='RUNNER=REASON')
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
