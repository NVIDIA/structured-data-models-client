#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Real RelBench data through the old-against-core differential (T1.32f):
a local, informational run, never a CI gate, whose data is never committed.

    python -I contract/tools/relbench.py fetch --cache DIR [DATASET...]
    python -I contract/tools/relbench.py sample --cache DIR --out SPEC.json
    python -I contract/tools/relbench.py run --cache DIR \
        --reference-python REF --candidate-python CAND --out DIR

``fetch`` downloads the tables each recipe below reads, at the commit and
SHA-256 the engine's RelBench loader pins
(``rfm/assets/relbench_registry.json``), into ``DIR`` laid out as that
loader's cache is (``<repository>/<revision>/<dataset>/db/<table>.parquet``),
so either fills the other; a file whose digest differs is refused and
deleted, never read. Only the public Hugging Face host is reached.

``sample`` turns each dataset into a bounded DuckDB source: a seed of
entities that have facts in the recipe's window, each child table's rows of
those entities in the window up to a row cap, then every row a kept row
refers to, repeated until no foreign key is left dangling (a self-reference
included), each table written as parquet under ``DIR/samples``. Its spec is
``differential.py``'s: each question one source, whose ``setup`` reads the
samples with ``read_parquet``, so the spec names local paths and is never
committed either.

``--cache`` and ``--out`` must name places outside the worktree this
tool is in, in any spelling (absolute, relative or through ``..``, read
from the working directory, links resolved): the cache holds RelBench's
tables and the samples, ``--out`` the spec or every probe's payloads, and
none of them is ever committed. Each command refuses such a path, with exit
status 2, before it reads, writes, fetches or starts an environment.

``run`` samples, then runs ``differential.py probe`` on each route, the
reference corrected for register P1B-13 among them (real data holds null
foreign keys, which the reference takes for the last key it looked up),
and ``differential.py compare --corrected`` over the spec, as
``differential.py run --corrected`` does over generated sources; its exit
status is ``compare``'s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REGISTRY = (
    ROOT / 'packages/structured-data-models-relational-engine/src'
    '/sdm_relational_engine/rfm/assets/relbench_registry.json'
)
HUB = 'https://huggingface.co'

#: Per dataset: the tables read (the rest are never fetched), the entity
#: table and the fact table whose rows choose the seed, how many entities,
#: the window (``since``, ``until``) the facts are kept in, each child
#: table's row cap, and the questions, each one source of the spec. A
#: question's ``inferred`` asks ``Schema.from_source`` with nothing given,
#: so the source's own description decides keys, times and links.
#:
#: ``exclude`` names the columns the core refused to describe when T1.32f
#: first measured them (long or free text, read as multi-category since
#: T1.32e; text its formats do not read, ``text_time``, register P1B-2).
#: Each question with them is asked twice: as the data is, and narrowed,
#: the columns left out, so materialize and predict are compared too. Most
#: caps keep a timestamped table under the 1,000 rows of a preview;
#: rel-avito's go past it, so a time column's range is read over the whole
#: table (T1.32e) on real data too.
RECIPES: dict[str, dict[str, Any]] = {
    # A fact table with no key and two links, zero-padded text-like ids
    # held as integers, a leading run of one date (t_dat).
    'rel-hm': {
        'tables': ['customer', 'article', 'transactions'],
        'seed': ('customer', 'transactions', 'customer_id'),
        'entities': 120,
        'window': ('2020-05-01', '2020-09-22'),
        'caps': {'transactions': 900},
        'exclude': {'article': ['prod_name']},
        'questions': [
            {
                'id': 'hm-sales-7d',
                'pql': 'PREDICT SUM(transactions.price, 0, 7, days) '
                'FOR EACH customer.customer_id',
                'anchor_time': '2020-08-15',
            },
            {
                'id': 'hm-churn-7d',
                'pql': 'PREDICT COUNT(transactions.*, 0, 7, days) = 0 '
                'FOR EACH customer.customer_id',
                'anchor_time': '2020-08-15',
            },
            {
                'id': 'hm-club-status',
                'pql': 'PREDICT customer.club_member_status '
                'FOR EACH customer.customer_id',
            },
            {
                'id': 'hm-sales-7d-inferred',
                'pql': 'PREDICT SUM(transactions.price, 0, 7, days) '
                'FOR EACH customer.customer_id',
                'anchor_time': '2020-08-15',
                'inferred': True,
            },
        ],
    },
    # A self-reference (posts.ParentId), two links from one table to the
    # same table (postLinks), a time column on the entity table.
    'rel-stack': {
        'tables': ['users', 'posts', 'postLinks', 'votes', 'badges'],
        'seed': ('users', 'posts', 'OwnerUserId'),
        'entities': 80,
        'window': ('2020-01-01', '2021-01-01'),
        'caps': {'posts': 600, 'votes': 900, 'badges': 900, 'postLinks': 300},
        'exclude': {
            'users': ['WebsiteUrl', 'AboutMe'],
            'posts': ['Title', 'Body'],
        },
        'questions': [
            {
                'id': 'stack-engagement-90d',
                'pql': 'PREDICT COUNT(posts.*, 0, 90, days) > 0 '
                'FOR EACH users.Id',
                'anchor_time': '2020-07-01',
            },
            {
                'id': 'stack-post-votes-90d',
                'pql': 'PREDICT COUNT(votes.*, 0, 90, days) FOR EACH posts.Id',
                'anchor_time': '2020-07-01',
                'indices_from': 'posts',
            },
            {
                'id': 'stack-badges-inferred',
                'pql': 'PREDICT COUNT(badges.*, 0, 90, days) FOR EACH users.Id',
                'anchor_time': '2020-07-01',
                'inferred': True,
            },
        ],
    },
    # Keyless event tables, entity tables with no time column, CamelCase
    # table names, a table reached at the second hop.
    'rel-avito': {
        'tables': [
            'UserInfo',
            'AdsInfo',
            'Location',
            'Category',
            'VisitStream',
            'SearchInfo',
            'SearchStream',
        ],
        'seed': ('UserInfo', 'VisitStream', 'UserID'),
        'entities': 100,
        'window': ('2015-05-01', '2015-05-21'),
        'caps': {'VisitStream': 3000, 'SearchInfo': 2000, 'SearchStream': 3000},
        'questions': [
            {
                'id': 'avito-user-visits-4d',
                'pql': 'PREDICT COUNT(VisitStream.*, 0, 4, days) > 0 '
                'FOR EACH UserInfo.UserID',
                'anchor_time': '2015-05-14',
            },
            {
                'id': 'avito-ad-clicks-4d',
                'pql': 'PREDICT SUM(SearchStream.IsClick, 0, 4, days) '
                'FOR EACH AdsInfo.AdID',
                'anchor_time': '2015-05-14',
                'indices_from': 'AdsInfo',
            },
        ],
    },
    # Many fact tables on one entity, two bridge tables to lookups, a
    # static target with nulls and nine classes (studies.phase).
    'rel-trial': {
        'tables': [
            'studies',
            'outcomes',
            'drop_withdrawals',
            'designs',
            'conditions_studies',
            'conditions',
            'sponsors_studies',
            'sponsors',
        ],
        'seed': ('studies', 'outcomes', 'nct_id'),
        'entities': 120,
        'window': ('2015-01-01', '2021-01-01'),
        'caps': {
            'outcomes': 900,
            'drop_withdrawals': 900,
            'designs': 600,
            'conditions_studies': 900,
            'sponsors_studies': 900,
        },
        'exclude': {
            'studies': [
                'baseline_population',
                'brief_title',
                'official_title',
                'detailed_descriptions',
                'brief_summaries',
            ],
            'outcomes': [
                'title',
                'description',
                'time_frame',
                'population',
                'units',
            ],
            'designs': [
                'time_perspective',
                'masking_description',
                'intervention_model_description',
            ],
        },
        'questions': [
            {
                'id': 'trial-outcomes-1y',
                'pql': 'PREDICT COUNT(outcomes.*, 0, 365, days) '
                'FOR EACH studies.nct_id',
                'anchor_time': '2018-01-01',
            },
            {
                'id': 'trial-withdrawals-1y',
                'pql': 'PREDICT COUNT(drop_withdrawals.*, 0, 365, days) > 0 '
                'FOR EACH studies.nct_id',
                'anchor_time': '2018-01-01',
            },
            {
                'id': 'trial-phase',
                'pql': 'PREDICT studies.phase FOR EACH studies.nct_id',
            },
        ],
    },
}


def inside_worktree(path: Path) -> bool:
    r"""Whether ``path``, read from the working directory with ``..`` and
    links resolved, is the worktree this tool is in or below it."""
    root = ROOT.resolve()
    resolved = path.expanduser().resolve()
    return resolved == root or root in resolved.parents


def registry() -> dict[str, Any]:
    return json.loads(REGISTRY.read_text())


def _tables(name: str) -> dict[str, dict[str, Any]]:
    r"""The recipe's tables, as the registry pins them."""
    pinned = registry()['datasets'][name]['tables']
    return {table: pinned[table] for table in RECIPES[name]['tables']}


def directory(cache: Path, name: str) -> Path:
    found = registry()
    repository = found['datasets'][name]['repository']
    return cache / repository / found['revisions'][repository] / name / 'db'


# --- fetch ------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(cache: Path, names: list[str]) -> None:
    found = registry()
    for name in names:
        repository = found['datasets'][name]['repository']
        revision = found['revisions'][repository]
        where = directory(cache, name)
        where.mkdir(parents=True, exist_ok=True)
        for table, spec in _tables(name).items():
            path = where / f'{table}.parquet'
            if path.exists() and _sha256(path) == spec['sha256']:
                print(f'{name}/{table}: cached')
                continue
            url = (
                f'{HUB}/datasets/{repository}/resolve/{revision}'
                f'/{name}/db/{table}.parquet'
            )
            part = path.with_suffix('.part')
            print(f'{name}/{table}: fetching {url}', flush=True)
            with urllib.request.urlopen(url, timeout=60) as answer:
                with part.open('wb') as out:
                    for block in iter(lambda: answer.read(1 << 20), b''):
                        out.write(block)
            if _sha256(part) != spec['sha256']:
                part.unlink()
                raise SystemExit(f'{name}/{table}: not the pinned SHA-256')
            part.replace(path)


# --- sample -----------------------------------------------------------------


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _links(name: str) -> list[tuple[str, str, str]]:
    r"""``(table, column, parent)`` for each foreign key between the
    recipe's tables."""
    tables = _tables(name)
    return [
        (table, column, parent)
        for table, spec in tables.items()
        for column, parent in spec['foreign_keys'].items()
        if parent in tables
    ]


def _exists(con: Any, table: str) -> bool:
    return bool(
        con.execute(
            'SELECT count(*) FROM duckdb_tables() WHERE table_name = ?',
            [table],
        ).fetchone()[0]
    )


def sample_dataset(con: Any, cache: Path, name: str) -> dict[str, Path]:
    r"""One dataset's bounded, foreign-key-closed sample, each table written
    as parquet; the paths by table."""
    recipe = RECIPES[name]
    tables = _tables(name)
    links = _links(name)
    source = directory(cache, name)
    since, until = recipe['window']
    for table in tables:
        path = source / f'{table}.parquet'
        con.execute(
            f'CREATE VIEW {_q("src_" + table)} AS '
            f'SELECT * FROM read_parquet({_text(str(path))})'
        )

    def window(table: str) -> str:
        time = tables[table]['time_column']
        if time is None:
            return 'TRUE'
        return (
            f'{_q(time)} >= TIMESTAMP {_text(since)} '
            f'AND {_q(time)} < TIMESTAMP {_text(until)}'
        )

    # The seed: entities with a fact in the window, chosen by hash.
    entity, fact, column = recipe['seed']
    key = tables[entity]['primary_key']
    con.execute(
        f'CREATE TABLE {_q("s_" + entity)} AS SELECT * FROM '
        f'{_q("src_" + entity)} WHERE {_q(key)} IN (SELECT {_q(column)} '
        f'FROM {_q("src_" + fact)} WHERE {_q(column)} IS NOT NULL AND '
        f'{window(fact)} GROUP BY 1 ORDER BY hash({_q(column)}) '
        f'LIMIT {int(recipe["entities"])})'
    )
    # Children: a capped table's rows in the window that refer to a row
    # already kept, by any of its links, until no capped table is left.
    pending = [t for t in recipe['tables'] if t in recipe['caps']]
    while pending:
        progressed = False
        for table in list(pending):
            parents = [
                (column, parent)
                for child, column, parent in links
                if child == table and _exists(con, 's_' + parent)
            ]
            if not parents:
                continue
            refers = ' OR '.join(
                f'{_q(column)} IN (SELECT '
                f'{_q(tables[parent]["primary_key"])} FROM {_q("s_" + parent)})'
                for column, parent in parents
            )
            con.execute(
                f'CREATE TABLE {_q("s_" + table)} AS SELECT * FROM '
                f'{_q("src_" + table)} t WHERE ({refers}) AND {window(table)} '
                f'ORDER BY hash(t) LIMIT {int(recipe["caps"][table])}'
            )
            pending.remove(table)
            progressed = True
        if not progressed:
            raise SystemExit(f'{name}: {pending} reach no kept table')
    # Closure: every row a kept row refers to, until none is missing.
    for table in tables:
        if not _exists(con, 's_' + table):
            con.execute(
                f'CREATE TABLE {_q("s_" + table)} AS SELECT * FROM '
                f'{_q("src_" + table)} LIMIT 0'
            )
    for _ in range(20):
        added = 0
        for child, column, parent in links:
            pk = _q(tables[parent]['primary_key'])
            added += con.execute(
                f'INSERT INTO {_q("s_" + parent)} SELECT * FROM '
                f'{_q("src_" + parent)} WHERE {pk} IN (SELECT {_q(column)} '
                f'FROM {_q("s_" + child)}) AND {pk} NOT IN (SELECT {pk} '
                f'FROM {_q("s_" + parent)})'
            ).fetchone()[0]
        if not added:
            break
    else:
        raise SystemExit(f'{name}: the closure did not settle')
    out = cache / 'samples' / name
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for table in tables:
        rows = con.execute(
            f'SELECT count(*) FROM {_q("s_" + table)}'
        ).fetchone()[0]
        if not rows:
            raise SystemExit(f'{name}: {table} is empty in the sample')
        paths[table] = out / f'{table}.parquet'
        con.execute(
            f'COPY {_q("s_" + table)} TO {_text(str(paths[table]))} '
            '(FORMAT parquet)'
        )
        print(f'{name}/{table}: {rows} rows', file=sys.stderr)
    return paths


def _indices(con: Any, name: str, table: str, n: int = 5) -> list[Any]:
    key = _tables(name)[table]['primary_key']
    return [
        row[0]
        for row in con.execute(
            f'SELECT {_q(key)} FROM {_q("s_" + table)} '
            f'ORDER BY hash({_q(key)}) LIMIT {n}'
        ).fetchall()
    ]


def _setup(paths: dict[str, Path], exclude: dict[str, list[str]]) -> list[str]:
    r"""The statements that load the samples, ``exclude``'s columns left
    out."""
    out = []
    for table, path in paths.items():
        left = exclude.get(table)
        columns = f'* EXCLUDE ({", ".join(map(_q, left))})' if left else '*'
        out.append(
            f'CREATE TABLE {_q(table)} AS SELECT {columns} FROM '
            f'read_parquet({_text(str(path))})'
        )
    return out


def spec(cache: Path, names: list[str]) -> dict[str, Any]:
    r"""``differential.py``'s spec: one source per question."""
    import duckdb

    sources = []
    for name in names:
        con = duckdb.connect()
        paths = sample_dataset(con, cache, name)
        tables = _tables(name)
        exclude = RECIPES[name].get('exclude', {})

        described = {
            'primary_keys': {
                t: s['primary_key']
                for t, s in tables.items()
                if s['primary_key'] is not None
            },
            'time_columns': {
                t: s['time_column']
                for t, s in tables.items()
                if s['time_column'] is not None
            },
            'relationships': [
                {
                    'source_table': child,
                    'source_column': column,
                    'target_table': parent,
                    'target_column': tables[parent]['primary_key'],
                }
                for child, column, parent in _links(name)
            ],
        }
        for question in RECIPES[name]['questions']:
            entity = question.get('indices_from', RECIPES[name]['seed'][0])
            for narrowed in (False, True) if exclude else (False,):
                source: dict[str, Any] = {
                    'id': question['id'] + ('.narrowed' if narrowed else ''),
                    'dataset': name,
                    'setup': _setup(paths, exclude if narrowed else {}),
                    'from_source': (
                        {} if question.get('inferred') else described
                    ),
                    'edits': [],
                    'pql': question['pql'],
                    'indices': _indices(con, name, entity),
                }
                if 'anchor_time' in question:
                    source['anchor_time'] = question['anchor_time']
                sources.append(source)
    return {'relbench': names, 'sources': sources}


# --- run --------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    me = str(Path(__file__).resolve())
    differential = str(HERE / 'differential.py')
    spec_path = args.out / 'spec.json'
    # The sampler needs DuckDB, which the reference environment has.
    subprocess.run(
        [
            args.reference_python,
            '-I',
            me,
            'sample',
            '--cache',
            str(args.cache),
            '--out',
            str(spec_path),
            *args.datasets,
        ],
        check=True,
    )
    for python, route in (
        (args.reference_python, 'old'),
        (args.candidate_python, 'core'),
        (args.reference_python, 'corrected'),
    ):
        subprocess.run(
            [
                python,
                '-I',
                differential,
                'probe',
                str(spec_path),
                str(args.out / f'{route}.json'),
                '--route',
                route,
            ],
            check=True,
        )
    return subprocess.run(
        [
            args.reference_python,
            '-I',
            differential,
            'compare',
            str(args.out / 'old.json'),
            str(args.out / 'core.json'),
            '--spec',
            str(spec_path),
            '--corrected',
            str(args.out / 'corrected.json'),
            '--register',
            str(args.register),
        ],
        check=False,
    ).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('fetch')
    f.add_argument('--cache', type=Path, required=True)
    f.add_argument('datasets', nargs='*', default=list(RECIPES))
    s = sub.add_parser('sample')
    s.add_argument('--cache', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('datasets', nargs='*', default=list(RECIPES))
    r = sub.add_parser('run')
    r.add_argument('--cache', type=Path, required=True)
    r.add_argument('--reference-python', required=True)
    r.add_argument('--candidate-python', required=True)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument(
        '--register',
        type=Path,
        default=ROOT / 'docs/design/rust-core/register.yaml',
    )
    r.add_argument('datasets', nargs='*', default=list(RECIPES))
    args = parser.parse_args(argv)
    unknown = sorted(set(args.datasets) - set(RECIPES))
    if unknown:
        parser.error(f'no recipe for {unknown}')
    for option in ('cache', 'out'):
        given = getattr(args, option, None)
        if given is not None and inside_worktree(given):
            parser.error(
                f'--{option} {given} is inside the worktree ({ROOT}): '
                'RelBench data, samples, specs and payloads are never '
                'committed, so they live outside it'
            )
    if args.command == 'fetch':
        fetch(args.cache, args.datasets)
        return 0
    if args.command == 'sample':
        args.out.write_text(
            json.dumps(spec(args.cache, args.datasets), indent=1) + '\n'
        )
        return 0
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
