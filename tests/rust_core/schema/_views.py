# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Metric and semantic views: today's Python's reading of one, the Rust
core's, the views the engine's tests and the corpus read, and generated
ones.

A generated metric view is YAML text and a trust flag, compared through
``parse_metric_view``. It mixes what the parser keeps with every way it drops
or refuses a part, and spells some scalars as only YAML 1.1 reads them:
``on`` as a key, ``yes``, ``0x1A``, ``1:30``, ``2021-01-01``.

A generated semantic view is YAML text, the tables behind it and a trust
flag, read through ``Schema.from_source`` over the stand-in warehouse.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import functools
import json
import os
import random
import subprocess
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ENGINE = REPO / 'packages' / 'structured-data-models-relational-engine'

METRIC_SEED = 20261008
SEMANTIC_SEED = 20261009

#: Scalars as YAML 1.1 reads them, and PyYAML with it: each is written
#: unquoted where a generated view puts it.
YAML11 = (
    'yes',
    'No',
    'on',
    'OFF',
    '~',
    'null',
    'true',
    '0x1A',
    '017',
    '1_000',
    '1:30',
    '+.inf',
    '.NaN',
    '1.5e+3',
    '1e3',
    '2021-01-01',
    '2021-01-01 10:00:00',
    '0o17',
)
_RAW = '__raw_{}__'


def _raw(rng: random.Random) -> str:
    return _RAW.format(rng.randrange(len(YAML11)))


def _dump(cfg: Any, rng: random.Random) -> str:
    text = yaml.safe_dump(
        cfg, sort_keys=False, allow_unicode=True, width=10_000
    )
    for i, spelling in enumerate(YAML11):
        text = text.replace(_RAW.format(i), spelling)
    if rng.random() < 0.5:
        text = text.replace("'on':", 'on:')
    return text


# --- Metric views ------------------------------------------------------------------

_ALIASES = (
    'users',
    'products',
    'customer',
    'nation',
    'region',
    'j1',
    'J2',
    'c',
)
_COLUMNS = ('id', 'user_id', 'amount', 'ts', 'category', 'x', 'Name', 'n_name')
_FUNCTIONS = (
    'UPPER',
    'lower',
    'coalesce',
    'date_trunc',
    'abs',
    'my_udf',
    'secret',
)


def _ident(rng: random.Random, name: str) -> str:
    roll = rng.random()
    if roll < 0.1:
        return '`' + name + '`'
    if roll < 0.13:
        return '`' + name + '-x y`'
    if roll < 0.18:
        return name.upper()
    return name


def _table_name(rng: random.Random) -> Any:
    roll = rng.random()
    if roll < 0.8:
        parts = rng.choice((3, 3, 3, 2, 1))
        names = ['cat', 'sch', rng.choice(('orders', 'users', 't'))][-parts:]
        return '.'.join(_ident(rng, n) for n in names)
    if roll < 0.88:
        return '`example-catalog`.parity_shop.' + rng.choice(('orders', 'users'))
    if roll < 0.9:
        return rng.choice(('  cat . sch.t ', ' cat.sch.t '))
    if roll < 0.93:
        return 'SELECT * FROM cat.sch.t'
    if roll < 0.95:
        return 'cat.sch.t.u'
    if roll < 0.98:
        return _raw(rng)
    return rng.choice((5, ['cat.sch.t'], {'table': 't'}, None))


def _column_ref(rng: random.Random, qualifiers: list[str]) -> str:
    column = _ident(rng, rng.choice(_COLUMNS))
    roll = rng.random()
    if roll < 0.2 or not qualifiers:
        return column
    qualifier = rng.choice(qualifiers)
    if rng.random() < 0.15:
        qualifier = qualifier.upper()
    space = rng.choice(('', '', '', ' '))
    return f'{qualifier}{space}.{space}{column}'


def _good_table_name(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.85:
        parts = rng.choice((3, 3, 3, 2, 1))
        names = ['cat', 'sch', rng.choice(('orders', 'users', 't'))][-parts:]
        return '.'.join(_ident(rng, n) for n in names)
    if roll < 0.95:
        return '`example-catalog`.parity_shop.' + rng.choice(('orders', 'users'))
    return ' cat.sch.t '


def _defect(
    rng: random.Random, join: dict[str, Any], parent: str, alias: str
) -> None:
    r"""Make ``join`` one the parser skips, by its name, source, cardinality
    or condition."""
    roll = rng.random()
    if roll < 0.15:
        join['name'] = 'source'
    elif roll < 0.4:
        join['source'] = rng.choice(
            (
                'SELECT * FROM cat.sch.t',
                'cat.sch.t.u',
                '  cat . sch.t ',
                5,
                None,
                _raw(rng),
            )
        )
    elif roll < 0.55:
        join['cardinality'] = rng.choice(
            ('many_to_many', 'MANY_TO_ONE', 'one', _raw(rng))
        )
    elif roll < 0.7:
        join.pop('on', None)
        join['using'] = rng.choice(
            (list(_COLUMNS[:2]), 5, [], {'a': 1}, _raw(rng))
        )
    else:
        join.pop('using', None)
        left = _column_ref(rng, [parent])
        right = _column_ref(rng, [alias])
        join['on'] = rng.choice(
            (
                f'{left} = {right} AND {_column_ref(rng, [parent])} = {right}',
                f'{left} > {right}',
                f'{left} = 1',
                None,
                5,
                [left, right],
                _raw(rng),
            )
        )


def _joins(
    rng: random.Random,
    depth: int,
    names: list[str],
    parent: str,
    paths: list[str],
    budget: list[int],
) -> list[Any]:
    r"""Joins under ``parent``. At most one named join of the whole view is
    skipped (``budget``), and it has no named joins under it, so the parser's
    set of skipped names holds one name at most and no answer depends on the
    order of a set. Items that are not mappings, joins with no name, and
    repeats of a kept name are skipped without being named, and may come any
    number of times."""
    out: list[Any] = []
    for _ in range(
        rng.choice((0, 1, 1, 2, 2, 3) if depth == 0 else (0, 0, 1, 2))
    ):
        roll = rng.random()
        if roll < 0.04:
            out.append(rng.choice(('users', 5, ['a'], _raw(rng))))
            continue
        if roll < 0.08:
            out.append(
                {
                    'name': rng.choice((5, None, ['users'], _raw(rng))),
                    'source': _good_table_name(rng),
                    'joins': [
                        {'name': rng.choice(_ALIASES), 'source': 'cat.sch.t'}
                    ],
                }
            )
            continue
        if roll < 0.11:
            out.append(
                {'source': _good_table_name(rng), 'on': 'source.x = y.x'}
            )
            continue
        if roll < 0.15 and names:
            out.append(
                {
                    'name': rng.choice(names),
                    'source': _good_table_name(rng),
                    'on': f'{parent}.id = {rng.choice(names)}.id',
                }
            )
            continue
        free = [a for a in _ALIASES if a not in names]
        if not free:
            break
        alias = rng.choice(free)
        join: dict[str, Any] = {'name': alias, 'source': _good_table_name(rng)}
        if rng.random() < 0.2:
            join['using'] = rng.choice(
                (rng.choice(_COLUMNS), [rng.choice(_COLUMNS)], ['`user id`'])
            )
        else:
            left = _column_ref(rng, [parent, *paths[-1:]])
            right = _column_ref(rng, [alias])
            join['on'] = rng.choice(
                (
                    f'{left} = {right}',
                    f'{right}={left}',
                    f'  {left}  =  {right} ',
                )
            )
        if rng.random() < 0.25:
            join['cardinality'] = rng.choice(('many_to_one', 'one_to_many'))
        names.append(alias)
        if budget[0] and rng.random() < 0.3:
            budget[0] = 0
            _defect(rng, join, parent, alias)
        elif depth < 2 and rng.random() < 0.35:
            join['joins'] = _joins(
                rng, depth + 1, names, alias, [*paths, alias], budget
            )
        out.append(join)
    return out


def _expr(rng: random.Random, aliases: list[str], defined: list[str]) -> Any:
    roll = rng.random()
    qualifiers = ['source', *aliases]
    if roll < 0.35:
        return _column_ref(rng, qualifiers)
    if roll < 0.5:
        return f'{rng.choice(_FUNCTIONS)}({_column_ref(rng, qualifiers)})'
    if roll < 0.6:
        return (
            f'{_column_ref(rng, qualifiers)} + {_column_ref(rng, qualifiers)}'
        )
    if roll < 0.68 and defined:
        return f'{rng.choice(defined)} * 2'
    if roll < 0.74 and len(aliases) >= 1:
        path = '.'.join(rng.sample(aliases, min(len(aliases), 2)))
        return f'{path}.{rng.choice(_COLUMNS)}'
    if roll < 0.8:
        return f"CASE WHEN {_column_ref(rng, qualifiers)} = 'users.x' THEN 1 ELSE 0 END"
    if roll < 0.85:
        return f'{_column_ref(rng, qualifiers)} -- note'
    if roll < 0.9:
        return f'(SELECT max(x) FROM {_column_ref(rng, [])})'
    if roll < 0.95:
        return _raw(rng)
    return rng.choice((None, 5, ['x']))


def _dimensions(rng: random.Random, aliases: list[str]) -> list[Any]:
    out: list[Any] = []
    defined: list[str] = []
    for i in range(rng.choice((0, 1, 2, 3, 4, 5))):
        if rng.random() < 0.04:
            out.append(rng.choice(('d', 5, _raw(rng), dt.date(2021, 1, 2))))
            continue
        name = rng.choice((*_COLUMNS, f'd{i}', 'total', 'Total'))
        dim: dict[str, Any] = {}
        if rng.random() > 0.04:
            dim['name'] = name if rng.random() > 0.03 else _raw(rng)
        if rng.random() > 0.04:
            dim['expr'] = _expr(rng, aliases, defined)
        out.append(dim)
        defined.append(name)
    return out


def _filter(rng: random.Random, aliases: list[str]) -> Any:
    roll = rng.random()
    if roll < 0.5:
        return None
    source_column = _column_ref(rng, ['source'])
    if roll < 0.7:
        return f"{source_column} > 0 AND {_column_ref(rng, [])} <> 'x'"
    if roll < 0.76 and aliases:
        return f'{_column_ref(rng, [rng.choice(aliases)])} = 1'
    if roll < 0.82:
        return f'{rng.choice(_FUNCTIONS)}({source_column}) = 1'
    if roll < 0.86:
        return f"{source_column} = 'users.x'"
    if roll < 0.9:
        return f'{source_column} = 1; DROP TABLE t'
    if roll < 0.94:
        return rng.choice(('', '   ', _raw(rng)))
    return rng.choice((5, True, ['x']))


def metric_view(rng: random.Random) -> tuple[str, bool]:
    r"""One generated metric view: its YAML and whether its expressions are
    trusted. A filter names at most one joined table, so no answer depends
    on the order of a set."""
    cfg: dict[str, Any] = {}
    if rng.random() < 0.8:
        cfg['version'] = rng.choice(('0.1', 0.1, 1.1, '1.1'))
    if rng.random() < 0.98:
        cfg['source'] = _table_name(rng)
    names: list[str] = []
    joins = _joins(rng, 0, names, 'source', [], [1])
    if joins or rng.random() < 0.3:
        cfg['joins'] = joins
    row_filter = _filter(rng, names)
    if row_filter is not None or rng.random() < 0.05:
        cfg['filter'] = row_filter
    dims = _dimensions(rng, names)
    key = 'fields' if rng.random() < 0.15 else 'dimensions'
    if dims or rng.random() < 0.2:
        cfg[key] = dims
    if rng.random() < 0.1:
        cfg['fields' if key == 'dimensions' else 'dimensions'] = _dimensions(
            rng, names
        )
    if rng.random() < 0.3:
        cfg['measures'] = [{'name': 'm', 'expr': 'SUM(source.amount)'}]
    for section in ('parameters', 'materialization'):
        if rng.random() < 0.07:
            cfg[section] = [{'name': 'p'}]
    return _dump(cfg, rng), rng.random() < 0.5


def metric_views(
    count: int, seed: int = METRIC_SEED
) -> Iterator[tuple[str, bool]]:
    rng = random.Random(seed)
    for _ in range(count):
        yield metric_view(rng)


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def python_metric_view(definition: str, trust: bool) -> dict[str, Any]:
    r"""``parse_metric_view``'s answer, as ``metric_view_parse`` gives it."""
    from sdm_relational_engine.rfm.backend.databricks.metric_view import (
        parse_metric_view,
    )

    try:
        spec = parse_metric_view(definition)
    except Exception as error:  # noqa: BLE001 - every refusal is an answer
        return {'error': {'code': type(error).__name__, 'message': str(error)}}
    return {'spec': _plain(dataclasses.asdict(spec))}


def is_yaml_error(definition: str) -> bool:
    try:
        yaml.safe_load(definition)
    except yaml.YAMLError:
        return True
    return False


# --- Semantic views -----------------------------------------------------------------

_TYPES = (
    'NUMBER(38,0)',
    'NUMBER(10,2)',
    'FLOAT',
    'VARCHAR(16777216)',
    'TIMESTAMP_NTZ(9)',
    'DATE',
    'BOOLEAN',
)


def _sv_name(rng: random.Random, base: str) -> tuple[str, str]:
    r"""A name as a semantic view's YAML writes it, and the name it spells."""
    roll = rng.random()
    if roll < 0.6:
        return base.upper(), base.upper()
    if roll < 0.85:
        return f'"{base.lower()}"', base.lower()
    if roll < 0.95:
        mixed = base[:1].upper() + base[1:].lower()
        return f'"{mixed}"', mixed
    return f'"{base.lower()} x"', f'{base.lower()} x'


def semantic_view(
    rng: random.Random,
) -> tuple[str, dict[tuple[str, str, str], dict[str, str]], bool]:
    r"""One generated semantic view: its YAML, the tables behind it, and
    whether its expressions are trusted."""
    count = rng.choice((1, 2, 2, 3, 3))
    bases = rng.sample(
        ('orders', 'users', 'products', 'stores', 'events'), count
    )
    tables: dict[tuple[str, str, str], dict[str, str]] = {}
    written: list[dict[str, Any]] = []
    logical: list[str] = []
    keys: list[str] = []
    for base in bases:
        name, _ = _sv_name(rng, base)
        logical.append(name)
        table_written, table_spelled = (
            _sv_name(rng, base)
            if rng.random() < 0.85
            else _sv_name(rng, base + '_raw')
        )
        key_written, key_spelled = _sv_name(rng, 'id')
        keys.append(key_written)
        columns: dict[str, str] = {key_spelled: 'NUMBER(38,0)'}
        table: dict[str, Any] = {
            'name': name,
            'base_table': {
                'database': 'TEST',
                'schema': rng.choice(
                    ('PARITY_SHOP', 'PARITY_SHOP', '"PARITY_SHOP"')
                ),
                'table': table_written,
            },
        }
        roll = rng.random()
        if roll < 0.75:
            table['primary_key'] = {'columns': [key_written]}
        elif roll < 0.85:
            other_written, other_spelled = _sv_name(rng, 'code')
            columns[other_spelled] = 'VARCHAR(16777216)'
            table['primary_key'] = {'columns': [key_written, other_written]}
        elif roll < 0.9:
            table['primary_key'] = {'columns': []}
        sections: dict[str, list[dict[str, Any]]] = {}
        for i in range(rng.choice((0, 1, 2, 3, 4))):
            kind = rng.choice(_TYPES)
            column_written, column_spelled = _sv_name(
                rng, f'c{i}_{kind[:3].lower()}'
            )
            section = rng.choice(
                ('dimensions', 'dimensions', 'facts', 'time_dimensions')
            )
            if section == 'time_dimensions':
                kind = rng.choice(
                    ('TIMESTAMP_NTZ(9)', 'DATE', 'VARCHAR(16777216)')
                )
            column: dict[str, Any] = {'name': column_written}
            roll = rng.random()
            qualifier = rng.choice(
                (name, name.lower(), f'"{name.strip(chr(34))}"')
            )
            if roll < 0.3:
                column['expr'] = column_written
                columns[column_spelled] = kind
            elif roll < 0.4:
                columns[column_spelled] = kind
            elif roll < 0.5:
                column['expr'] = f'{qualifier}.{column_written}'
                columns[column_spelled] = kind
            elif roll < 0.6:
                inner = rng.choice(list(columns))
                quoted = '"' + inner.replace('"', '""') + '"'
                column['expr'] = rng.choice(
                    (
                        f'UPPER({quoted})',
                        f'{quoted} * 2',
                        f"CASE WHEN {quoted} IS NULL THEN 'x.y' ELSE 'z' END",
                        f'COALESCE({qualifier}.{quoted}, 0)',
                    )
                )
            elif roll < 0.7 and len(logical) > 1:
                column['expr'] = (
                    f'{rng.choice(logical[:-1])}.{rng.choice(keys)}'
                )
            elif roll < 0.8:
                column['expr'] = rng.choice(
                    (
                        'MY_UDF(1)',
                        'SYSTEM$WAIT(1)',
                        '(SELECT 1)',
                        '1; DROP TABLE T',
                        'X -- c',
                        f"'{rng.choice(logical)}.x'",
                    )
                )
            else:
                column['expr'] = rng.choice(('1', "'k'", 'CURRENT_DATE()'))
            if rng.random() < 0.85:
                column['data_type'] = kind
            sections.setdefault(section, []).append(column)
        table.update(sections)
        if rng.random() < 0.2:
            table['metrics'] = [{'name': 'M', 'expr': 'SUM(1)'}]
        written.append(table)
        if rng.random() > 0.04:
            schema = identifier_of(table['base_table']['schema'])
            tables[('TEST', schema, table_spelled)] = columns
    links: list[dict[str, Any]] = []
    if count > 1:
        for i in range(rng.choice((0, 1, 1, 2, 3))):
            left, right = rng.sample(range(count), 2)
            fk_written, fk_spelled = _sv_name(rng, f'{bases[right]}_id')
            link: dict[str, Any] = {
                'name': f'L{i}',
                'left_table': logical[left],
                'right_table': logical[right],
            }
            roll = rng.random()
            if roll < 0.6:
                pairs = [
                    {'left_column': fk_written, 'right_column': keys[right]}
                ]
            elif roll < 0.68:
                pairs = [
                    {'left_column': fk_written, 'right_column': keys[right]},
                    {'left_column': keys[left], 'right_column': keys[right]},
                ]
            elif roll < 0.76:
                pairs = [
                    {'left_column': keys[left], 'right_column': keys[right]}
                ]
            elif roll < 0.84:
                pairs = [
                    {'left_column': fk_written, 'right_column': fk_written}
                ]
            elif roll < 0.9:
                link['right_table'] = 'NOT_A_TABLE'
                pairs = [
                    {'left_column': fk_written, 'right_column': keys[right]}
                ]
            else:
                pairs = [
                    {'left_column': fk_written, 'right_column': keys[right]}
                ]
            link['relationship_columns'] = pairs
            links.append(link)
            schema = identifier_of(written[left]['base_table']['schema'])
            left_key = (
                'TEST',
                schema,
                identifier_of(written[left]['base_table']['table']),
            )
            if left_key in tables and rng.random() < 0.9:
                tables[left_key].setdefault(
                    fk_spelled,
                    rng.choice(
                        ('NUMBER(38,0)', 'NUMBER(38,0)', 'VARCHAR(16777216)')
                    ),
                )
    cfg: dict[str, Any] = {'name': 'SV', 'tables': written}
    if links or rng.random() < 0.3:
        cfg['relationships'] = links
    text = yaml.safe_dump(
        cfg, sort_keys=False, allow_unicode=True, width=10_000
    )
    return text, tables, rng.random() < 0.3


def identifier_of(text: str) -> str:
    from _warehouse import identifier

    return identifier(text)


def semantic_views(
    count: int, seed: int = SEMANTIC_SEED
) -> Iterator[tuple[str, dict[tuple[str, str, str], dict[str, str]], bool]]:
    rng = random.Random(seed)
    for _ in range(count):
        yield semantic_view(rng)


# --- Views the engine's tests and the corpus read ---------------------------------

_PLUGIN = r"""
import json, os, sys

_OUT = os.environ['SDM_P65_VIEW_CAPTURE']
_SEEN = {'metric': [], 'semantic': []}


def _keep():
    with open(_OUT, 'w') as handle:
        json.dump(_SEEN, handle)


def pytest_configure(config):
    import yaml
    from sdm_relational_engine.rfm.backend.databricks import metric_view

    parse = metric_view.parse_metric_view

    def parse_metric_view(definition, *args, **kwargs):
        trust = kwargs.get('trust_expressions', args[0] if args else False)
        if isinstance(definition, str):
            _SEEN['metric'].append([definition, bool(trust)])
            _keep()
        return parse(definition, *args, **kwargs)

    metric_view.parse_metric_view = parse_metric_view
    load = yaml.safe_load

    def safe_load(stream):
        if isinstance(stream, str):
            frame = sys._getframe(1)
            while frame is not None:
                if frame.f_code.co_name == 'from_snowflake_semantic_view':
                    _SEEN['semantic'].append(stream)
                    _keep()
                    break
                frame = frame.f_back
        return load(stream)

    yaml.safe_load = safe_load
    _keep()
"""


def engine_test_views(tmp: Path) -> dict[str, list[Any]]:
    r"""Every metric view definition the engine's tests parse, with its
    trust, and every semantic view YAML they read: the view fixtures."""
    tests = sorted(
        str(path.relative_to(ENGINE))
        for path in (ENGINE / 'tests').rglob('test_*.py')
        if any(
            word in path.read_text()
            for word in (
                'metric_view',
                'semantic_view',
                'metric view',
                'semantic view',
            )
        )
    )
    assert tests, 'no engine test reads a view'
    plugin = tmp / '_p65_view_capture.py'
    plugin.write_text(_PLUGIN)
    out = tmp / 'views.json'
    env = dict(os.environ)
    env['SDM_P65_VIEW_CAPTURE'] = str(out)
    env['PYTHONPATH'] = os.pathsep.join(
        [str(tmp), *filter(None, [env.get('PYTHONPATH')])]
    )
    proc = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            '-q',
            '-p',
            'no:cacheprovider',
            '-p',
            '_p65_view_capture',
            '-W',
            'ignore::DeprecationWarning',
            *tests,
        ],
        cwd=ENGINE,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert out.is_file(), (
        f'the engine tests captured nothing (exit {proc.returncode})\n'
        f'{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}'
    )
    seen = json.loads(out.read_text())
    metric: list[Any] = []
    for item in seen['metric']:
        if item not in metric:
            metric.append(item)
    semantic: list[str] = []
    for item in seen['semantic']:
        if item not in semantic:
            semantic.append(item)
    return {'metric': metric, 'semantic': semantic}


@contextlib.contextmanager
def _capturing_metric_views(found: list[list[Any]]) -> Iterator[None]:
    from sdm_relational_engine.rfm.backend.databricks import metric_view

    parse = metric_view.parse_metric_view

    @functools.wraps(parse)
    def captured(definition: Any, *args: Any, **kwargs: Any) -> Any:
        trust = kwargs.get('trust_expressions', args[0] if args else False)
        if (
            isinstance(definition, str)
            and [definition, bool(trust)] not in found
        ):
            found.append([definition, bool(trust)])
        return parse(definition, *args, **kwargs)

    metric_view.parse_metric_view = captured
    try:
        yield
    finally:
        metric_view.parse_metric_view = parse


def recorded_views() -> dict[str, list[Any]]:
    r"""Every metric view the corpus's scenarios parse, and every semantic
    view YAML the corpus recorded."""
    import pyarrow.ipc as ipc
    from _schema import TRANSCRIPTS, scenario_calls

    metric: list[list[Any]] = []
    semantic: list[str] = []
    for path in sorted(TRANSCRIPTS.glob('**/transcript.json')):
        transcript = json.loads(path.read_text())
        if transcript['data'].get('view') == 'metric-view':
            with _capturing_metric_views(metric), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                scenario_calls(path)
        for step in transcript['steps']:
            if 'SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW' in (step.get('sql') or ''):
                table = ipc.open_file(path.parent / step['result']).read_all()
                text = table.column(0)[0].as_py()
                if text not in semantic:
                    semantic.append(text)
    return {'metric': metric, 'semantic': semantic}


def json_of(value: Any) -> Any:
    return json.loads(json.dumps(value))
