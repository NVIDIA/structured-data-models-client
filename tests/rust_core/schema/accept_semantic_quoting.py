# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P65.semantic-quoting: a Snowflake semantic view whose YAML keeps quoted
names can be read, by today's Python and by the Rust core.

Phase 5.6 found that a semantic view over tables written with quoted
lower-case names can't be read: its YAML keeps the quotes (``name:
'"orders"'``), and ``Graph.from_snowflake_semantic_view`` quotes them again
(``TableNotFoundError: Table 'TEST.PARITY_SHOP."orders"' does not exist``).

The view here is the recorded semantic view's YAML with every name of the
shop written so, over the shop written with quoted lower-case names, served
by the stand-in warehouse (``_warehouse.py``). Read, it must be the recorded
view's schema with every name in lower case: the tables ``orders``,
``products`` and ``users``, the shop's columns, keys and links, as every
other backend has them, each name quoted once in the SQL. The core must give
the same outcome from the same statements.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from typing import Any

import pyarrow.ipc as ipc
import pytest
import yaml
from _schema import TRANSCRIPTS, difference, python_outcome, rust_from_source
from _warehouse import (
    DATABASE,
    SCHEMA,
    Warehouse,
    identifier,
    source,
    tables_of,
)

SCENARIO = TRANSCRIPTS / 'snowflake' / 'semantic-view' / 'count'
VIEW = 'PARITY_SHOP_SEMANTICS'
QUALIFIED = f'{DATABASE}.{SCHEMA}.{VIEW}'


def _recorded_yaml() -> str:
    transcript = json.loads((SCENARIO / 'transcript.json').read_text())
    for step in transcript['steps']:
        if step.get(
            'op'
        ) == 'execute' and 'SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW' in (
            step.get('sql') or ''
        ):
            table = ipc.open_file(SCENARIO / step['result']).read_all()
            return table.column(0)[0].as_py()
    raise AssertionError(f'{SCENARIO} reads no semantic view')


def _upper(name: str) -> str:
    return identifier(name).upper()


def _quoted(name: str) -> str:
    return '"' + identifier(name).lower().replace('"', '""') + '"'


_REFERENCE = re.compile(r'("(?:[^"]|"")*"|\w+)\.("(?:[^"]|"")*"|\w+)')


def _written(cfg: dict[str, Any], write: Any) -> dict[str, Any]:
    r"""The view with every name of the shop written by ``write``: unquoted
    in upper case, as Snowflake writes a view over tables made unquoted, or
    quoted in lower case, as it writes one over tables made so. The recorded
    view may be written either way."""
    cfg = copy.deepcopy(cfg)
    for table in cfg['tables']:
        table['name'] = write(table['name'])
        table['base_table']['table'] = write(table['base_table']['table'])
        key = table.get('primary_key')
        if key:
            key['columns'] = [write(c) for c in key['columns']]
        for section in ('dimensions', 'time_dimensions', 'facts', 'metrics'):
            for column in table.get(section) or []:
                column['name'] = write(column['name'])
                if section == 'metrics':
                    column['expr'] = _REFERENCE.sub(
                        lambda m: f'{write(m.group(1))}.{write(m.group(2))}',
                        column['expr'],
                    )
                elif column.get('expr') is not None:
                    column['expr'] = write(column['expr'])
    for link in cfg.get('relationships') or []:
        link['name'] = write(link['name'])
        link['left_table'] = write(link['left_table'])
        link['right_table'] = write(link['right_table'])
        for pair in link['relationship_columns']:
            pair['left_column'] = write(pair['left_column'])
            pair['right_column'] = write(pair['right_column'])
    return cfg


def _quote_names(cfg: dict[str, Any]) -> dict[str, Any]:
    r"""The view with every name of the shop quoted and in lower case."""
    return _written(cfg, _quoted)


def _lower(value: Any) -> Any:
    r"""Names, in lists and as keys too, in lower case."""
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, dict):
        return {k.lower(): _lower(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_lower(v) for v in value]
    return value


def _lower_names(outcome: dict[str, Any]) -> dict[str, Any]:
    r"""``outcome`` with the names in its schema lower-cased, and nothing
    else: types, semantic types and warnings stay as they are."""
    schema = outcome['schema']
    out: dict[str, Any] = {'tables': {}, 'relationships': []}
    for name, table in schema['tables'].items():
        out['tables'][name.lower()] = {
            **table,
            'primary_key': _lower(table['primary_key']),
            'time_column': _lower(table['time_column']),
            'source_name': table['source_name'].lower(),
            'columns': {c.lower(): v for c, v in table['columns'].items()},
        }
    out['relationships'] = _lower(schema['relationships'])
    definition = schema['definition']
    out['definition'] = {
        'tables': {
            name.lower(): {
                **table,
                'cols': [[c[0].lower(), *c[1:]] for c in table['cols']],
                'pkey': _lower(table['pkey']),
                'time_col': _lower(table['time_col']),
            }
            for name, table in definition['tables'].items()
        },
        'col_groups': _lower(definition['col_groups']),
        'warnings': definition['warnings'],
    }
    return {'schema': out, 'warnings': outcome['warnings']}


def _read(text: str) -> tuple[Warehouse, Any]:
    cfg = yaml.safe_load(text)
    warehouse = Warehouse({QUALIFIED: text}, tables_of(cfg))
    return warehouse, source(warehouse).semantic_model(VIEW)


@pytest.fixture(scope='module')
def views() -> tuple[str, str]:
    r"""The recorded view written unquoted in upper case, and quoted in lower
    case."""
    cfg = yaml.safe_load(_recorded_yaml())
    upper = yaml.safe_dump(_written(cfg, _upper), sort_keys=False)
    quoted = yaml.safe_dump(_quote_names(cfg), sort_keys=False)
    assert 'name: ORDERS' in upper
    assert 'name: \'"orders"\'' in quoted
    return upper, quoted


def _python(text: str) -> tuple[dict[str, Any], Warehouse]:
    from sdm_client import Schema

    warehouse, model = _read(text)
    outcome = python_outcome(lambda: Schema.from_source(model))
    assert not warehouse.unknown, warehouse.unknown
    return outcome, warehouse


def test_the_recorded_view_reads_over_the_stand_in(
    views: tuple[str, str],
) -> None:
    outcome, _ = _python(views[0])
    assert 'schema' in outcome, outcome
    assert list(outcome['schema']['tables']) == ['ORDERS', 'PRODUCTS', 'USERS']


def test_python_reads_a_view_of_quoted_names(views: tuple[str, str]) -> None:
    upper, _ = _python(views[0])
    quoted, warehouse = _python(views[1])
    assert 'schema' in quoted, (
        f'Schema.from_source refused it: {quoted.get("error")}'
    )
    assert difference(_lower_names(upper), quoted) is None, difference(
        _lower_names(upper), quoted
    )
    described = [
        sql for sql, _ in warehouse.log if sql.startswith('DESCRIBE TABLE')
    ]
    assert sorted(described) == [
        f'DESCRIBE TABLE "{DATABASE}"."{SCHEMA}"."{name}"'
        for name in ('orders', 'products', 'users')
    ], described


def test_the_core_reads_a_view_of_quoted_names(views: tuple[str, str]) -> None:
    want, by_python = _python(views[1])
    warehouse, model = _read(views[1])
    got, asked = rust_from_source(model, {})
    assert difference(want, got) is None, difference(want, got)

    def statements(log: list[Any]) -> Counter[str]:
        return Counter(json.dumps([sql, params or None]) for sql, params in log)

    assert statements(asked) == statements(by_python.log), (
        'the core asked other statements than Python did'
    )
