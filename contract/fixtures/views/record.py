# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Records the old client's view answers as vectors for the core (T3.5).

A pytest plugin, run inside the reference checkout at 43e55f4f with its own
environment. It wraps the old client's expression checks
(``rfm/view_expr.py``), the Snowflake qualifier helpers (``rfm/graph.py``)
and ``yaml.safe_load``, runs the old client's own tests of them, and writes
every distinct call with its answer. So each vector is an answer the old
client gave to an input its own tests chose; nothing is computed here but
by the old client's code. The YAML vectors also take the inputs of
``YAML_INPUTS`` below, PyYAML 1.1's value rules.

Run from the reference checkout::

    PYTHONPATH=<this directory> .venv/bin/python -m pytest -p record \
        -q -o addopts= -W ignore <the test files in TESTS> \
        --record-out=<this repository>/contract \
        --record-cases=<the view cases of rust-core-phase-3-topic-T3.5-tests2>

which also reads the semantic-view definitions the old tests hold as text
(``SEMANTIC_TEST_FILES``, whose tests need the Snowflake connector the
reference's environment lacks) and each view case's definition through
``yaml.safe_load``, and each metric-view case's through ``parse_metric_view``,
and writes ``view-expr-vectors.json``, ``view-metric-vectors.json``
(each definition ``parse_metric_view`` read, with its whole answer) and
``view-yaml-vectors.json`` there.
"""

from __future__ import annotations

import base64
import datetime
import functools
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

# The old client's tests this records, relative to the reference checkout.
_ENGINE = 'packages/structured-data-models-relational-engine/tests/backend'
TESTS = (
    f'{_ENGINE}/test_view_expr_calls.py',
    f'{_ENGINE}/test_view_expr_names.py',
    f'{_ENGINE}/test_view_expr_import.py',
    f'{_ENGINE}/test_view_qualifier.py',
    f'{_ENGINE}/databricks/test_metric_view.py',
    'packages/structured-data-models-client/tests/test_semantic_model_snowflake.py',
    'packages/structured-data-models-client/tests/test_a_metric_view_filter_holds.py',
)

# PyYAML 1.1's value rules (design T3.5 part 5, old-behaviour row 69), each a
# plain scalar unless quoted, plus the forms a view definition uses.
YAML_INPUTS = (
    # Booleans: every case form of yes, no, on, off, true, false; y and n
    # stay text, as do mixed cases.
    *(
        f'v: {word}'
        for base in ('yes', 'no', 'on', 'off', 'true', 'false')
        for word in (base, base.capitalize(), base.upper())
    ),
    'v: y', 'v: n', 'v: Y', 'v: N', 'v: yES', 'v: tRUE', 'v: oN',
    # Null.
    'v: ~', 'v: null', 'v: Null', 'v: NULL', 'v:', 'v: nULL', "v: ''",
    # Integers: decimal, octal, hexadecimal, binary, base 60, underscores.
    'v: 0', 'v: 17', 'v: -17', 'v: +17', 'v: 017', 'v: -017', 'v: 0o17',
    'v: 0x1F', 'v: 0x1f', 'v: -0x1F', 'v: 0b101', 'v: 1_000', 'v: 0_17',
    'v: 1:30', 'v: 190:20:30', 'v: -1:30', 'v: 1:60', 'v: 08', 'v: 09',
    'v: 00', 'v: 12345678901234567890123', 'v: 0b', 'v: 0x',
    # Floats: only a signed exponent makes one of 1e3.
    'v: 1e3', 'v: 1.0e3', 'v: 1.0e+3', 'v: 1.0E-3', 'v: 1.5', 'v: -1.5',
    'v: .5', 'v: 1.', 'v: 1_000.5', 'v: .inf', 'v: -.Inf', 'v: +.INF',
    'v: .nan', 'v: .NaN', 'v: 1:30.5', 'v: 6.8523015e+5', 'v: 1e+3',
    'v: 0.1', 'v: 1e16', 'v: 1.0e+16', 'v: 1.0e-5', 'v: 123456789.123',
    'v: 1.0e+400', 'v: -0.0',
    # Dates and timestamps.
    'v: 2024-01-01', 'v: 2024-1-1', 'v: 2024-01-01 10:00:00',
    'v: 2024-01-01T10:00:00Z', 'v: 2024-01-01t10:00:00.5',
    'v: 2001-12-14t21:59:43.10-05:00', 'v: 2001-12-14 21:59:43.10 -5',
    'v: 2024-1-1 1:02:03', 'v: 2024-01-01 10:00:00.1234567',
    'v: 2024-13-01',
    # Text, quoted or not.
    "v: 'yes'", 'v: "on"', 'v: "017"', 'v: abc', 'v: =', 'v: <<',
    'v: source.price > 50', "v: 'it''s'", 'v: "a\\tb"', 'v: a: b',
    # Keys: a bare on is the key true, as a join's on is.
    'on: x', 'On: x', 'yes: 1', '1: one', '017: x', '~: x', 'null: x',
    '2024-01-01: x', '1.5: x',
    # Structure: sequences, nested mappings, block text, anchors, merges,
    # tags and duplicate keys.
    'v: [1, yes, ~, 1e3]',
    'v: {a: 1, b: on}',
    'v: |\n  line one\n  line two\n',
    'v: >\n  folded\n  text\n',
    'a: &x {k: 1}\nb: *x',
    'base: &b {k: 1, j: 2}\nv:\n  <<: *b\n  j: 3',
    'v:\n  <<: [{a: 1}, {a: 2, b: 2}]',
    'v: !!str 017', 'v: !!int "017"', 'v: !!float "1"', 'v: !!bool "yes"',
    'v: !!null ""', 'v: !!binary aGVsbG8=', 'v: !custom x',
    'v: !!timestamp 2024-01-01',
    'a: 1\na: 2',
    '[1, 2]', 'just text', '', '# only a comment', '---\na: 1\n---\nb: 2',
    '- a\n- b', '{[1]: x}', 'v: "\\x01"', 'v: [1, [2, {a: b}]]',
    # A whole metric view, as the cases write one.
    'version: 0.1\nsource: "`c`.`s`.orders"\nfilter: source.price > 50\n'
    'joins:\n  - name: users\n    source: "`c`.`s`.users"\n'
    '    on: source.user_id = users.user_id\n'
    'dimensions:\n  - name: order_id\n    expr: source.order_id\n'
    'measures:\n  - name: revenue\n    expr: SUM(source.price)\n',
)  # fmt: skip


# The old tests whose semantic-view definitions are read as text, since
# they need the Snowflake connector, which the reference's environment
# lacks.
SEMANTIC_TEST_FILES = (
    'packages/structured-data-models-client/tests/test_semantic_model_snowflake.py',
    f'{_ENGINE}/test_view_expr_import.py',
    f'{_ENGINE}/snow/test_snow_graph.py',
)


# Semantic-view shapes the old tests do not write: a composite and an empty
# primary key, a composite relationship, a column with no expression, time
# dimensions and facts, and no relationships at all.
SEMANTIC_INPUTS = (
    'tables:\n'
    '  - name: A\n'
    '    base_table: {database: D, schema: S, table: A_T}\n'
    '    primary_key: {columns: [K1, K2]}\n'
    '    dimensions:\n'
    '      - name: K1\n'
    '      - name: X\n'
    '        expr: a.X + 1\n'
    '        data_type: NUMBER(38,0)\n'
    '    time_dimensions:\n'
    '      - name: T\n'
    '        expr: T\n'
    '        data_type: TIMESTAMP_NTZ\n'
    '    facts:\n'
    '      - name: F\n'
    '        expr: "B.Y || \'B.\'"\n'
    '  - name: B\n'
    '    base_table: {database: D, schema: S, table: B}\n'
    '    primary_key: {columns: []}\n'
    'relationships:\n'
    '  - name: R1\n'
    '    left_table: A\n'
    '    right_table: B\n'
    '    relationship_columns:\n'
    '      - {left_column: K1, right_column: K}\n'
    '      - {left_column: K2, right_column: L}\n'
    '  - name: R2\n'
    '    left_table: A\n'
    '    right_table: B\n'
    '    relationship_columns: []\n',
    'tables:\n'
    '  - name: ONLY\n'
    '    base_table: {database: D, schema: S, table: ONLY}\n',
)


def semantic_texts() -> list[str]:
    r"""The semantic-view definitions the old tests write as string
    constants, a ``{extra}`` slot left empty.
    """
    import ast

    texts = list(SEMANTIC_INPUTS)
    for name in SEMANTIC_TEST_FILES:
        tree = ast.parse(Path(name).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and 'base_table' in node.value
                and 'tables' in node.value
            ):
                texts.append(node.value.replace('{extra}', ''))
    return texts


def case_texts(directory: str | None) -> list[tuple[str, str]]:
    r"""Each view case's kind and definition, from ``directory`` (the
    cases of ``rust-core-phase-3-topic-T3.5-tests2``), the ``{catalog}``
    and ``{schema}`` placeholders written ``c`` and ``s``.
    """
    if directory is None:
        return []
    found = []
    for path in sorted(Path(directory).glob('view.*.json')):
        view = json.loads(path.read_text(encoding='utf-8'))['input'].get('view')
        if view and view.get('yaml'):
            text = (
                view['yaml'].replace('{catalog}', 'c').replace('{schema}', 's')
            )
            found.append((view['kind'], text))
    return found


def value(item: Any) -> Any:
    r"""``item``, a loaded YAML value, as typed JSON."""
    if item is None:
        return {'t': 'null'}
    if isinstance(item, bool):
        return {'t': 'bool', 'v': item}
    if isinstance(item, int):
        return {'t': 'int', 'v': str(item)}
    if isinstance(item, float):
        return {'t': 'float', 'v': repr(item)}
    if isinstance(item, str):
        return {'t': 'str', 'v': item}
    if isinstance(item, bytes):
        return {'t': 'bytes', 'v': base64.b64encode(item).decode()}
    if isinstance(item, datetime.datetime):
        return {'t': 'datetime', 'v': item.isoformat()}
    if isinstance(item, datetime.date):
        return {'t': 'date', 'v': item.isoformat()}
    if isinstance(item, list):
        return {'t': 'list', 'v': [value(entry) for entry in item]}
    if isinstance(item, dict):
        return {
            't': 'map',
            'v': [[value(key), value(entry)] for key, entry in item.items()],
        }
    if isinstance(item, set):
        return {'t': 'set', 'v': sorted(repr(entry) for entry in item)}
    raise TypeError(f'no vector form for {type(item).__name__}')


def yaml_vector(text: str) -> dict[str, Any]:
    r"""What ``yaml.safe_load`` makes of ``text``: the value, its ``repr``
    and ``str``, or the error's class.
    """
    try:
        loaded = yaml.safe_load(text)
    except Exception as error:  # noqa: BLE001 - every failure is an answer
        return {'yaml': text, 'error': type(error).__name__}
    finite = not (isinstance(loaded, float) and not math.isfinite(loaded))
    vector = {'yaml': text, 'value': value(loaded), 'repr': repr(loaded)}
    if finite:
        vector['str'] = str(loaded)
    return vector


class _Recorder:
    def __init__(self) -> None:
        self.calls: dict[str, dict[str, dict[str, Any]]] = {}
        self.yaml_texts: dict[str, None] = {}

    def add(self, kind: str, key: Any, vector: dict[str, Any]) -> None:
        self.calls.setdefault(kind, {}).setdefault(
            json.dumps(key, sort_keys=True), vector
        )


RECORDER = _Recorder()


def _wrap(module: Any, name: str, record: Any) -> tuple[Any, Any]:
    original = getattr(module, name)

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return record(original, *args, **kwargs)

    return original, wrapped


def _unsafe(original: Any, expr: Any, dialect: Any = None) -> Any:
    answer = original(expr, dialect)
    if isinstance(expr, str):
        RECORDER.add(
            'unsafe',
            [expr, dialect],
            {'expr': expr, 'dialect': dialect, 'reason': answer},
        )
    return answer


def _calls(original: Any, expr: Any, dialect: Any, *, columns: Any) -> Any:
    answer = original(expr, dialect, columns=columns)
    names = sorted({str(column) for column in columns})
    RECORDER.add(
        'calls',
        [expr, dialect, names],
        {'expr': expr, 'dialect': dialect, 'columns': names, 'reason': answer},
    )
    return answer


def _names(original: Any, expr: Any) -> Any:
    from sdm_relational_engine.rfm.view_expr import UnreadableExpressionError

    try:
        answer = original(expr)
    except UnreadableExpressionError as error:
        RECORDER.add('names', expr, {'expr': expr, 'error': str(error)})
        raise
    RECORDER.add(
        'names',
        expr,
        {
            'expr': expr,
            'references': [
                {
                    'parts': list(reference.parts),
                    'text': expr[reference.span[0] : reference.span[1]],
                    'bare': reference.bare,
                }
                for reference in answer
            ],
        },
    )
    return answer


def _strip(original: Any, expr: Any) -> Any:
    answer = original(expr)
    RECORDER.add('strip_comments', expr, {'expr': expr, 'out': answer})
    return answer


def _literals(original: Any, expr: Any) -> Any:
    answer = original(expr)
    RECORDER.add(
        'string_literals',
        expr,
        {'expr': expr, 'literals': [expr[s:e] for s, e in answer]},
    )
    return answer


class _Qualifier:
    r"""A Snowflake qualifier pattern that says which table it was made for."""

    def __init__(self, pattern: Any, table: str) -> None:
        self.pattern, self.table = pattern, table

    def search(self, text: str, *args: Any) -> Any:
        found = self.pattern.search(text, *args)
        if not args:
            RECORDER.add(
                'qualifier_found',
                [self.table, text],
                {'table': self.table, 'text': text, 'found': bool(found)},
            )
        return found

    def sub(self, *args: Any, **kwargs: Any) -> Any:
        return self.pattern.sub(*args, **kwargs)


def _qualifier(original: Any, table: str) -> Any:
    return _Qualifier(original(table), table)


def _sub_outside(original: Any, pattern: Any, expr: str) -> Any:
    answer = original(pattern, expr)
    if isinstance(pattern, _Qualifier):
        RECORDER.add(
            'qualifier_stripped',
            [pattern.table, expr],
            {'table': pattern.table, 'expr': expr, 'out': answer},
        )
    return answer


def _mask(original: Any, expr: str) -> Any:
    answer = original(expr)
    RECORDER.add('mask_literals', expr, {'expr': expr, 'out': answer})
    return answer


def _message(original: Any, *args: Any) -> Any:
    answer = original(*args)
    column, expr, view, dialect, trust, columns = args
    names = sorted({str(name) for name in columns})
    key = [column, expr, view, dialect, trust, names]
    RECORDER.add(
        'expr_message',
        key,
        {
            'column': column,
            'expr': expr,
            'view': view,
            'dialect': dialect,
            'trust_expressions': trust,
            'columns': names,
            'message': answer,
        },
    )
    return answer


def _spec(spec: Any) -> dict[str, Any]:
    r"""The old client's ``MetricViewSpec`` as JSON, every field."""
    from sdm_relational_engine.rfm.backend.databricks.metric_view import (
        MetricViewWildcard,
    )

    def key(ref: Any) -> dict[str, Any]:
        return {'qualifier': ref.qualifier, 'column': ref.column}

    return {
        'source': list(spec.source),
        'joins': [
            {
                'alias': join.alias,
                'table': list(join.table),
                'parent_alias': join.parent_alias,
                'path': list(join.path),
                'condition': [key(side) for side in join.condition],
                'cardinality': join.cardinality,
            }
            for join in spec.joins
        ],
        'columns': [
            {'wildcard': column.alias, 'excluded': list(column.excluded)}
            if isinstance(column, MetricViewWildcard)
            else {
                'name': column.name,
                'alias': column.alias,
                'expr': column.expr,
            }
            for column in spec.columns
        ],
        'messages': list(spec.messages),
        'skipped_joins': [
            {
                'alias': skipped.alias,
                'reason': skipped.reason,
                'nested': list(skipped.nested),
                'dimensions': list(skipped.dimensions),
                'message': str(skipped),
            }
            for skipped in spec.skipped_joins
        ],
        'notes': [[note.parameter, note.default] for note in spec.notes],
        'filter': None if spec.filter is None else spec.filter.trusted(),
        'window_orders': list(spec.window_orders),
        'measures': [
            {
                'name': measure.name,
                'inputs': [list(entry) for entry in measure.inputs],
                'aggregation': (
                    None
                    if measure.aggregation is None
                    else list(measure.aggregation)
                ),
            }
            for measure in spec.measures
        ],
    }


def _parse_metric(
    original: Any,
    definition: Any,
    parameter_types: Any = None,
    source_columns: Any = None,
) -> Any:
    read: list[list[str]] = []

    def columns(parts: Any) -> Any:
        answer = list(source_columns(parts))
        read.append([str(column) for column in answer])
        return answer

    vector: dict[str, Any] = {
        'definition': definition,
        'parameter_types': dict(parameter_types or {}),
    }
    try:
        answer = original(
            definition,
            parameter_types,
            None if source_columns is None else columns,
        )
    except Exception as error:
        vector['source_columns'] = read[0] if read else None
        vector['error'] = {'type': type(error).__name__, 'message': str(error)}
        RECORDER.add('metric_views', vector, vector)
        raise
    vector['source_columns'] = read[0] if read else None
    vector['spec'] = _spec(answer)
    RECORDER.add('metric_views', vector, vector)
    return answer


def _safe_load(original: Any, stream: Any) -> Any:
    if isinstance(stream, str):
        RECORDER.yaml_texts.setdefault(stream, None)
    return original(stream)


_PATCHES: list[tuple[Any, Any]] = []


def _install() -> None:
    import sdm_relational_engine.rfm.backend.databricks.metric_view as metric
    import sdm_relational_engine.rfm.graph as graph
    import sdm_relational_engine.rfm.view_expr as view_expr

    wanted = [
        (view_expr, '_unsafe_expr_reason', _unsafe),
        (view_expr, '_untrusted_call_reason', _calls),
        (view_expr, 'name_references', _names),
        (view_expr, 'strip_comments', _strip),
        (view_expr, 'string_literals', _literals),
        (graph, '_qualifier_pattern', _qualifier),
        (graph, '_sub_outside_literals', _sub_outside),
        (graph, '_mask_literals', _mask),
        (graph, '_view_expr_message', _message),
        (metric, 'parse_metric_view', _parse_metric),
        (yaml, 'safe_load', _safe_load),
    ]
    for module, name, record in wanted:
        original, wrapped = _wrap(module, name, record)
        _PATCHES.append((original, wrapped))
        setattr(module, name, wrapped)
    _sweep()


def _sweep() -> None:
    r"""Points every loaded module's copy of a wrapped name at the wrapper,
    since the old client imports these by name.
    """
    for module in list(sys.modules.values()):
        attributes = getattr(module, '__dict__', None)
        if not isinstance(attributes, dict):
            continue
        for name, attribute in list(attributes.items()):
            for original, wrapped in _PATCHES:
                if attribute is original:
                    attributes[name] = wrapped


def pytest_addoption(parser: Any) -> None:
    parser.addoption('--record-out', required=True)
    parser.addoption('--record-cases', default=None)


def pytest_configure(config: Any) -> None:
    _install()


def pytest_collection_finish(session: Any) -> None:
    _sweep()


def pytest_runtest_setup(item: Any) -> None:
    _sweep()


def _write(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=1, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )


def _probe_words() -> None:
    r"""Puts each word of the old client's word tables to its checks, so
    every function on an allow-list, keyword, type name and session word
    has a vector of its own, read by the wrapped (recording) functions.
    """
    import sdm_relational_engine.rfm.view_expr as view_expr

    words = sorted(
        view_expr._DATABRICKS_FUNCTIONS
        | view_expr._SNOWFLAKE_FUNCTIONS
        | view_expr._KEYWORDS
        | view_expr._TYPE_NAMES
        | view_expr._SESSION_WORDS
        | view_expr._GRAMMAR_WORDS
        | view_expr._NEVER_BEFORE_AN_EXPRESSION
        | view_expr._WORD_OPERATORS
    )
    for word in words:
        for dialect in ('databricks', 'snowflake'):
            for expr in (f'{word}(x)', f'x + {word}', f'{word}'):
                for columns in (('x',), ('x', word)):
                    view_expr._untrusted_call_reason(
                        expr, dialect, columns=columns
                    )
        for expr in (f'{word}', f'{word} + 1', f'x = {word}', f'{word}(x)'):
            try:
                view_expr.name_references(expr)
            except view_expr.UnreadableExpressionError:
                pass


def pytest_sessionfinish(session: Any) -> None:
    out = Path(session.config.getoption('--record-out'))
    _probe_words()
    from sdm_relational_engine.rfm.backend.databricks import metric_view

    cases = case_texts(session.config.getoption('--record-cases'))
    for kind, text in cases:
        if kind == 'metric_view':
            try:
                metric_view.parse_metric_view(text)
            except Exception:  # noqa: BLE001, S110 - recorded by the wrapper
                pass
    source = (
        'recorded by contract/fixtures/views/record.py from the old '
        'client at 43e55f4f, running its own tests: ' + ', '.join(TESTS)
    )
    calls = RECORDER.calls
    _write(
        out / 'view-expr-vectors.json',
        {
            'format': 'sdm-view-expr-vectors/1',
            'source': source,
            **{
                kind: list(calls.get(kind, {}).values())
                for kind in (
                    'unsafe',
                    'calls',
                    'names',
                    'strip_comments',
                    'string_literals',
                    'mask_literals',
                    'qualifier_found',
                    'qualifier_stripped',
                    'expr_message',
                )
            },
        },
    )
    _write(
        out / 'view-metric-vectors.json',
        {
            'format': 'sdm-view-metric-vectors/1',
            'source': source,
            'metric_views': list(calls.get('metric_views', {}).values()),
        },
    )
    texts = dict.fromkeys(YAML_INPUTS)
    texts.update(dict.fromkeys(semantic_texts()))
    texts.update(dict.fromkeys(text for _, text in cases))
    texts.update(RECORDER.yaml_texts)
    _write(
        out / 'view-yaml-vectors.json',
        {
            'format': 'sdm-view-yaml-vectors/1',
            'source': (
                f'PyYAML {yaml.__version__} safe_load, '
                + source
                + ', and the inputs of YAML_INPUTS there'
            ),
            'vectors': [yaml_vector(text) for text in texts],
        },
    )
