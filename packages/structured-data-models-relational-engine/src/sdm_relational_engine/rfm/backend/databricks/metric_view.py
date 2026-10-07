# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, NoReturn

from sdm_relational_engine.exceptions import (
    TableNotFoundError,
    ViewConversionError,
)
from sdm_relational_engine.rfm.view_conversion import (
    ParameterNote,
    by_folded_name,
    quoted_list,
)
from sdm_relational_engine.rfm.view_expr import (
    BARE_NAME,
    UnreadableExpressionError,
    _unsafe_expr_reason,
    _untrusted_call_reason,
    name_references,
    string_literals,
    strip_comments,
)

if TYPE_CHECKING:
    from sdm_relational_engine.rfm.backend.databricks import AnyConnection

SOURCE_ALIAS = 'source'

_CARDINALITIES = {'many_to_one', 'one_to_many'}
_MEASURE_SUFFIX = ' measure'

_IDENT = rf'`(?:[^`]|``)+`|{BARE_NAME}'

_TABLE_REFERENCE_RE = re.compile(
    rf'\s*({_IDENT})(?:\.({_IDENT}))?(?:\.({_IDENT}))?\s*$'
)
_CHAIN = rf'(?:{_IDENT})(?:\s*\.\s*(?:{_IDENT}))*'
_JOIN_CONDITION_RE = re.compile(rf'\s*({_CHAIN})\s*=\s*({_CHAIN})\s*$')
_IDENT_RE = re.compile(_IDENT)
# `- expr: source.*` or `- expr: customers.* EXCEPT (a, b)`, which Databricks
# expands into one dimension per column, each named as the column.
_WILDCARD_RE = re.compile(
    rf'\s*({_CHAIN})\s*\.\s*\*'
    rf'(?:\s*EXCEPT\s*\(\s*({_CHAIN}(?:\s*,\s*{_CHAIN})*)\s*\))?\s*$',
    flags=re.IGNORECASE,
)

_INTEGER_TYPES = {
    'tinyint',
    'byte',
    'smallint',
    'short',
    'int',
    'integer',
    'bigint',
    'long',
}
_FLOAT_TYPES = {'float', 'real', 'double'}
_TIMESTAMP_TYPES = {'timestamp', 'timestamp_ntz', 'timestamp_ltz'}
_DECIMAL_TYPE_RE = re.compile(
    r'(?:decimal|dec|numeric)(?:\(\s*\d+\s*(?:,\s*\d+\s*)?\))?'
)
_STRING_TYPE_RE = re.compile(r'string|(?:var)?char\(\s*\d+\s*\)')

# What a timestamp dimension may wrap a column in and still say when the row
# happened, only less precisely: a truncation or a cast to a date or time.
_TIME_TYPE = r'(?:date|timestamp(?:_ntz|_ltz)?)'
_DERIVATIONS = tuple(
    re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
    for pattern in (
        r'\(\s*(?P<inner>.+)\s*\)',
        r"date_trunc\s*\(\s*(?:'\w+'|\"\w+\")\s*,\s*(?P<inner>.+)\s*\)",
        r"trunc\s*\(\s*(?P<inner>.+?)\s*,\s*(?:'\w+'|\"\w+\")\s*\)",
        rf'cast\s*\(\s*(?P<inner>.+)\s+as\s+{_TIME_TYPE}\s*\)',
        r'(?:to_date|date|to_timestamp)\s*\(\s*(?P<inner>.+)\s*\)',
        rf'(?P<inner>.+?)\s*::\s*{_TIME_TYPE}',
    )
)


def derived_column(expr: str) -> str | None:
    r"""The column ``expr`` truncates or casts to a date or time, such as
    ``order_date`` in ``DATE_TRUNC('MONTH', order_date)``; ``None`` for
    anything else, a bare column included.
    """
    column = _column_under(expr)
    if column is None or _IDENT_RE.fullmatch(expr.strip().strip('() ')):
        return None
    return column


def _column_under(expr: str) -> str | None:
    expr = expr.strip()
    if _IDENT_RE.fullmatch(expr):
        return unquote_ident(expr)
    for pattern in _DERIVATIONS:
        match = pattern.fullmatch(expr)
        if match is None:
            continue
        inner = match.group('inner')
        # `.+` reaches past a parenthesis that closes early, as in
        # `date(a) - date(b)`, so what it holds must balance.
        depth = 0
        for char in _masked_strings(inner):
            depth += {'(': 1, ')': -1}.get(char, 0)
            if depth < 0:
                break
        if depth == 0 and (column := _column_under(inner)) is not None:
            return column
    return None


# A measure a predictive query aggregation says as is: one aggregate over one
# column, or a count of rows.
_SIMPLE_MEASURE_RE = re.compile(
    rf'\s*(sum|avg|min|max|count)\s*\(\s*(distinct\s+)?'
    rf'(\*|1|{_CHAIN})\s*\)\s*',
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SkippedJoin:
    r"""A join left out of the graph, with the joins and dimensions that go
    with it, which a question may have been asked of.

    Args:
        alias: The join's name.
        reason: Why, as a clause, such as ``its table is empty``.
        nested: The joins nested in it, left out with it.
        dimensions: The dimensions on it or on those, left out with it.
    """

    alias: str
    reason: str
    nested: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()

    def __str__(self) -> str:
        lost: list[str] = []
        if self.nested:
            noun = 'join' if len(self.nested) == 1 else 'joins'
            lost.append(f'the {noun} {quoted_list(self.nested)} nested in it')
        if self.dimensions:
            noun = 'dimension' if len(self.dimensions) == 1 else 'dimensions'
            lost.append(f'the {noun} {quoted_list(self.dimensions)}')
        with_it = f', and with it {" and ".join(lost)},' if lost else ''
        return f"Failed to add join '{self.alias}'{with_it} since {self.reason}"


class UnsupportedJoinError(ViewConversionError):
    r"""Raised when a metric view join cannot be converted into a graph
    edge.

    Attributes:
        alias: The join's name.
        reason: Why, as a clause.
    """

    def __init__(self, alias: str, reason: str) -> None:
        super().__init__(str(SkippedJoin(alias, reason)))
        self.alias = alias
        self.reason = reason


class UnsupportedFilterError(ViewConversionError):
    r"""Raised when a metric view's filter cannot be applied to its source
    table.
    """


@dataclass(frozen=True)
class JoinKeyRef:
    r"""One side of a metric view join condition.

    Args:
        qualifier: The (dot-joined) table alias or join path qualifying the
            column, if any.
        column: The referenced column name.
    """

    qualifier: str | None
    column: str


@dataclass(frozen=True)
class MetricViewJoin:
    r"""A join of a metric view definition.

    Args:
        alias: The join name, used as the graph table name.
        table: The parsed name parts of the joined table.
        parent_alias: The alias of the table this join is declared on, either
            :obj:`SOURCE_ALIAS` or the alias of an enclosing join.
        path: The aliases of all enclosing joins followed by ``alias``.
            Dimensions reference nested joins by this full path, *e.g.*,
            ``customer.nation.n_name``.
        condition: The two sides of the single-column equality join condition.
        cardinality: Either ``'many_to_one'`` or ``'one_to_many'``.
    """

    alias: str
    table: tuple[str, ...]
    parent_alias: str
    path: tuple[str, ...]
    condition: tuple[JoinKeyRef, JoinKeyRef]
    cardinality: str


@dataclass(frozen=True)
class MetricViewColumn:
    r"""A metric view dimension resolved onto a single table.

    Args:
        name: The dimension name, used as the column name.
        alias: The alias of the table the dimension refers to, either
            :obj:`SOURCE_ALIAS` or a join alias.
        expr: The dimension expression with the alias qualifier stripped.
    """

    name: str
    alias: str
    expr: str


@dataclass(frozen=True)
class MetricViewWildcard:
    r"""A metric view dimension ``<alias>.*``, which stands for every column
    of its table but those it excepts, each named as the column.

    Args:
        alias: The alias of the table, either :obj:`SOURCE_ALIAS` or a join
            alias.
        excluded: The columns named by its ``EXCEPT``, if any.
    """

    alias: str
    excluded: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetricViewMeasure:
    r"""A measure of a metric view, which the graph leaves to a predictive
    query to aggregate.

    Args:
        name: The measure's name.
        inputs: The columns it reads, each with the alias of its table,
            either :obj:`SOURCE_ALIAS` or a join alias, as written. A name
            that turns out to be no column of its table reads nothing.
        aggregation: The predictive query aggregation it is, with the name
            it aggregates or ``None`` for a count of rows, when it is one
            plain aggregate over the source table.
    """

    name: str
    inputs: tuple[tuple[str, str], ...]
    aggregation: tuple[str, str | None] | None = None


@dataclass
class MetricViewSpec:
    r"""The graph-relevant parts of a metric view definition.

    Args:
        source: The parsed name parts of the source (fact) table.
        joins: All convertible joins, parents before children.
        columns: All dimensions resolved onto a single table, in order. A
            wildcard is expanded once its table's columns are known.
        messages: Human-readable messages for every dropped element but a
            join.
        skipped_joins: Every join left out.
        notes: Every parameter read as its default.
        filter: The view's ``filter``, a condition on the source table's
            columns, if it has one.
        window_orders: The dimensions a window measure orders by, which
            the view's author made its time axis.
        measures: The measures, in order.
    """

    source: tuple[str, ...]
    joins: list[MetricViewJoin]
    columns: list[MetricViewColumn | MetricViewWildcard]
    messages: list[str]
    skipped_joins: list[SkippedJoin] = field(default_factory=list)
    notes: list[ParameterNote] = field(default_factory=list)
    filter: MetricViewFilter | None = None
    window_orders: list[str] = field(default_factory=list)
    measures: list[MetricViewMeasure] = field(default_factory=list)


@dataclass(frozen=True)
class MetricViewFilter:
    r"""A metric view's ``filter``, held back until what it calls is checked.

    Every read of the source table runs the filter under the caller's
    credentials, and which of its words are columns is known only once the
    source table's columns are read. So the SQL is handed out only by
    :meth:`checked`, given those columns, or by :meth:`trusted`.

    Args:
        _sql: The condition, which already references no joined table and
            holds no sub-query, comment or statement separator.
    """

    _sql: str

    def checked(self, columns: Collection[str]) -> str:
        r"""Returns the filter if it calls only pure built-in functions.

        Args:
            columns: The columns of the source table.

        Raises:
            UnsupportedFilterError: If the filter calls a function that is
                not trusted, or reads a session value.
        """
        reason = _untrusted_call_reason(
            self._sql, 'databricks', columns=columns
        )
        if reason is not None:
            _refuse_filter(self._sql, reason, trusting_helps=True)
        return self._sql

    def trusted(self) -> str:
        r"""Returns the filter unchecked, for a caller who trusts whoever
        authored the view.
        """
        return self._sql


def unquote_ident(name: str) -> str:
    r"""Removes backtick quoting from an identifier, if present."""
    name = name.strip()
    if len(name) >= 2 and name.startswith('`') and name.endswith('`'):
        return name[1:-1].replace('``', '`')
    return name


def parse_table_reference(name: Any) -> tuple[str, ...] | None:
    r"""Parses a table reference into its unquoted name parts.

    Returns ``None`` if the reference is not a plain (optionally qualified)
    table name, *e.g.*, a SQL query.
    """
    if not isinstance(name, str):
        return None
    match = _TABLE_REFERENCE_RE.match(name)
    if match is None:
        return None
    return tuple(
        unquote_ident(part) for part in match.groups() if part is not None
    )


def parse_join_condition(
    condition: Any,
) -> tuple[JoinKeyRef, JoinKeyRef] | None:
    r"""Parses a single-column equality join condition.

    Column references may be qualified by an alias or a (dotted) join path.
    Returns ``None`` for any other condition, *e.g.*, composite or
    non-equality joins.
    """
    if not isinstance(condition, str):
        return None
    match = _JOIN_CONDITION_RE.match(condition)
    if match is None:
        return None

    def _ref(chain: str) -> JoinKeyRef:
        parts = [unquote_ident(part) for part in _IDENT_RE.findall(chain)]
        return JoinKeyRef(
            qualifier='.'.join(parts[:-1]) or None,
            column=parts[-1],
        )

    left, right = match.groups()
    return _ref(left), _ref(right)


def _quoted_variants(name: str) -> str:
    quoted = re.escape('`' + name.replace('`', '``') + '`')
    if re.fullmatch(BARE_NAME, name):
        return f'{quoted}|{re.escape(name)}'
    return quoted


def _to_path(alias: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(alias, str):
        return (alias,)
    return tuple(alias)


def _path_pattern(path: Sequence[str]) -> re.Pattern[str]:
    chain = r'\s*\.\s*'.join(f'(?:{_quoted_variants(part)})' for part in path)
    return re.compile(
        rf'(?<![A-Za-z0-9_`$.]){chain}\s*\.',
        flags=re.IGNORECASE,
    )


def _sub_outside_strings(
    pattern: re.Pattern[str],
    repl: str | Callable[[re.Match[str]], str],
    expr: str,
) -> str:
    r"""``pattern.sub(repl, expr)`` with every string literal left as is.

    A literal holding `source.` or a field's name is data, and rewriting it
    changes the values the view declares.
    """
    out: list[str] = []
    last = 0
    for start, end in string_literals(expr):
        out.append(pattern.sub(repl, expr[last:start]))
        out.append(expr[start:end])
        last = end
    out.append(pattern.sub(repl, expr[last:]))
    return ''.join(out)


def _masked_strings(expr: str) -> str:
    r"""``expr`` with each string literal emptied, so no text in one reads
    as SQL.
    """
    out: list[str] = []
    last = 0
    for start, end in string_literals(expr):
        out += [expr[last:start], "''"]
        last = end
    out.append(expr[last:])
    return ''.join(out)


def references_alias(expr: str, alias: str | Sequence[str]) -> bool:
    r"""Returns ``True`` if the expression references a column of the given
    table alias or join path outside its string literals.
    """
    masked = _masked_strings(expr)
    return _path_pattern(_to_path(alias)).search(masked) is not None


def strip_alias_qualifier(expr: str, alias: str | Sequence[str]) -> str:
    r"""Removes all qualified references to the given table alias or join
    path from the expression, leaving its string literals as they are.
    """
    pattern = _path_pattern(_to_path(alias))
    return _sub_outside_strings(pattern, '', expr).strip()


@dataclass(frozen=True)
class _Parameter:
    r"""A metric view parameter, as a query of the view without arguments
    reads it.

    Args:
        name: The parameter's name.
        literal: Its default as a typed SQL literal, or ``None``.
        text: Its default as text, or why it has no ``literal``.
    """

    name: str
    literal: str | None
    text: str


def _sql_string(text: str) -> str:
    r"""A Databricks string literal holding ``text``.

    Databricks reads two adjacent literals as one, so ``'it''s'`` is ``its``.
    A quote is therefore escaped with a backslash, as is a backslash.
    """
    escaped = text.replace('\\', '\\\\').replace("'", "\\'")
    return f"'{escaped}'"


def _default_text(data_type: str, default: Any) -> str:
    r"""The text a parameter's default casts from, checked against its type
    so that nothing but the value reaches the SQL.

    Raises:
        ValueError: If the default is not a value of the type, or the type
            has no literal here.
    """
    folded = ' '.join(data_type.lower().split())
    if isinstance(default, bool):
        if folded == 'boolean' or _STRING_TYPE_RE.fullmatch(folded):
            return 'true' if default else 'false'
    elif folded == 'boolean':
        if str(default).strip().lower() in ('true', 'false'):
            return str(default).strip().lower()
    elif folded in _INTEGER_TYPES:
        if isinstance(default, int) or re.fullmatch(
            r'[+-]?\d+', str(default).strip()
        ):
            return str(int(default))
    elif folded in _FLOAT_TYPES or _DECIMAL_TYPE_RE.fullmatch(folded):
        try:
            number = Decimal(str(default).strip())
        except InvalidOperation:
            number = None
        if number is not None and number.is_finite():
            return format(number, 'f')
    elif _STRING_TYPE_RE.fullmatch(folded):
        if isinstance(default, datetime.date):
            return default.isoformat()
        return str(default)
    elif folded == 'date':
        if isinstance(default, datetime.date) and not isinstance(
            default, datetime.datetime
        ):
            return default.isoformat()
        try:
            day = datetime.date.fromisoformat(str(default).strip())
            return day.isoformat()
        except ValueError:
            pass
    elif folded in _TIMESTAMP_TYPES:
        if isinstance(default, datetime.datetime):
            return default.isoformat(' ')
        if isinstance(default, datetime.date):
            return default.isoformat()
        try:
            moment = datetime.datetime.fromisoformat(str(default).strip())
            return moment.isoformat(' ')
        except ValueError:
            pass
    else:
        raise ValueError(f"its type '{data_type}' has no literal here")
    raise ValueError(f'its default {default!r} is not of type {data_type}')


def _read_parameters(
    parameter_cfgs: Any, parameter_types: Mapping[str, str]
) -> dict[str, _Parameter]:
    r"""Reads the ``parameters`` section, by folded name.

    Args:
        parameter_cfgs: The ``parameters`` section, if any.
        parameter_types: Each parameter's type as Databricks describes it,
            which wins over the one written in the definition.
    """
    types = {name.lower(): type_ for name, type_ in parameter_types.items()}
    parameters: dict[str, _Parameter] = {}
    if not isinstance(parameter_cfgs, list):
        return parameters
    for parameter_cfg in parameter_cfgs:
        if not isinstance(parameter_cfg, dict):
            continue
        name = parameter_cfg.get('name')
        if not isinstance(name, str):
            continue
        data_type = types.get(name.lower(), parameter_cfg.get('data_type'))
        default = parameter_cfg.get('default')
        if default is None:
            parameter = _Parameter(name, None, 'has no default')
        elif not isinstance(data_type, str):
            parameter = _Parameter(name, None, 'has no type')
        else:
            try:
                text = _default_text(data_type, default)
            except ValueError as error:
                parameter = _Parameter(name, None, str(error))
            else:
                sql_type = ' '.join(data_type.split()).upper()
                literal = f'CAST({_sql_string(text)} AS {sql_type})'
                parameter = _Parameter(name, literal, text)
        parameters[name.lower()] = parameter
    return parameters


def _substitute_parameters(
    expr: str, parameters: Mapping[str, _Parameter]
) -> tuple[str, list[_Parameter]]:
    r"""Replaces each reference to a parameter with its default.

    A parameter is referenced by its bare name where the expression reads a
    value, so a qualified or backticked name, and a word that is grammar,
    such as a datetime unit or a lambda's parameter, is left as is.

    Returns:
        The expression, and every parameter it references, in order.
    """
    found = [
        (reference.span, parameter)
        for reference in name_references(expr)
        if reference.bare
        and (parameter := parameters.get(reference.parts[0].lower()))
    ]
    for (start, end), parameter in reversed(found):
        if parameter.literal is not None:
            expr = f'{expr[:start]}{parameter.literal}{expr[end:]}'
    used = {parameter.name: parameter for _, parameter in found}
    return expr, list(used.values())


# The first part of a name, backticked or not.
_FIRST_PART = re.compile(r'`(?:``|[^`])*`|[A-Za-z0-9_]+')


def _substitute_fields(
    expr: str,
    fields: Mapping[str, str | None],
    qualifiers: Collection[tuple[str, ...]],
) -> tuple[str, str | None]:
    r"""Replaces each reference to an earlier field with its expression.

    A field is referenced by its name, bare or backticked, where the
    expression reads a value, and may be followed by a member, as in
    ``address.city``. So a function, type, keyword, datetime unit or
    qualifier of the same name is left as is: a field named ``upper`` leaves
    ``UPPER(source.z)`` alone. A name that a table's alias or join path
    begins qualifies a column instead, as Databricks resolves a qualifier
    before a struct's field.

    Args:
        expr: The expression, its parameters substituted.
        fields: The expression of each earlier field, by folded name, or
            ``None`` for one refused.
        qualifiers: Every join path or alias, folded.

    Returns:
        The expression, and the name of a refused field it references, as
        written, if any.

    Raises:
        UnreadableExpressionError: If which of its words are values cannot
            be settled.
    """
    refused = None
    for reference in reversed(name_references(expr)):
        parts = tuple(part.lower() for part in reference.parts)
        qualified = any(
            len(parts) > len(path) and parts[: len(path)] == path
            for path in qualifiers
        )
        if parts[0] not in fields or qualified:
            continue
        field = fields[parts[0]]
        if field is None:
            refused = reference.parts[0]
            continue
        start = reference.span[0]
        first = _FIRST_PART.match(expr, start)
        assert first is not None
        expr = f'{expr[:start]}({field}){expr[first.end() :]}'
    return expr, refused


def _unusable_parameter(used: Sequence[_Parameter]) -> str | None:
    r"""Why an expression referencing ``used`` cannot be read, or ``None``."""
    for parameter in used:
        if parameter.literal is None:
            return (
                f"it references the parameter '{parameter.name}', which "
                f'{parameter.text}'
            )
    return None


def _sql(value: Any) -> Any:
    r"""An expression of the definition without its comments, which
    Databricks drops, or ``value`` itself if it is no string.
    """
    return strip_comments(value) if isinstance(value, str) else value


def parse_metric_view(
    definition: str,
    parameter_types: Mapping[str, str] | None = None,
    source_columns: Callable[[tuple[str, ...]], Collection[str]] | None = None,
) -> MetricViewSpec:
    r"""Parses a metric view YAML definition into its graph-relevant parts.

    Measures define aggregations rather than row-level values, and are
    therefore never part of the returned specification. Elements that cannot
    be represented in a graph (SQL query join sources, composite or
    non-equality join conditions, dimensions referencing multiple tables,
    and materializations) are dropped and reported in
    :attr:`MetricViewSpec.messages`, or as a join in
    :attr:`MetricViewSpec.skipped_joins`. Each parameter read as its
    default is in :attr:`MetricViewSpec.notes`.

    A comment in an expression is dropped, as Databricks drops it.

    A parameter a dimension or the filter references is read as its default,
    as a query of the view without arguments reads it. A dimension
    referencing a parameter with no default is dropped. Databricks reads a
    bare name as a column of the source table where it has one, and as a
    parameter only otherwise.

    The ``filter`` decides which rows the view holds, so it is never dropped:
    it is returned to be applied to the source table, or the view is refused.
    Which functions it calls depends on the source table's columns, so it
    comes back as a :class:`MetricViewFilter` that gives out its SQL only
    once checked against them.

    Args:
        definition: The YAML definition.
        parameter_types: The type of each parameter as Databricks describes
            it, which wins over the one the definition names.
        source_columns: Reads the columns of the source table, given its
            name parts. Called only if the view has parameters. If set to
            ``None``, no column shadows a parameter.

    Raises:
        ViewConversionError: If the definition is not a YAML mapping,
            misses a source, or defines a SQL query as its source.
        UnsupportedFilterError: If the filter references a joined table or a
            parameter with no default, or is not a safe scalar condition.
    """
    import yaml

    cfg = yaml.safe_load(definition)
    if not isinstance(cfg, dict):
        raise ViewConversionError(
            'The metric view definition is not a YAML mapping'
        )

    source = cfg.get('source')
    if source is None:
        raise ViewConversionError(
            "The metric view definition misses a 'source'"
        )
    source_ref = parse_table_reference(source)
    if source_ref is None:
        raise ViewConversionError(
            f'Unsupported metric view source {source!r}. Only plain table '
            f'names are supported as the source of a metric view when '
            f'converting it into a graph. Create a regular view with this '
            f"query and point the metric view's source at it"
        )

    messages: list[str] = []
    if 'materialization' in cfg:
        messages.append(
            "Ignored the 'materialization' section since it cannot "
            'be represented in a graph'
        )
    parameters = _read_parameters(cfg.get('parameters'), parameter_types or {})
    if parameters and source_columns is not None:
        for column in source_columns(source_ref):
            parameters.pop(column.lower(), None)
    used: dict[str, _Parameter] = {}

    joins: list[MetricViewJoin] = []
    aliases: set[str] = set()
    kept: set[str] = set()
    skipped: list[SkippedJoin] = []
    # The position in `skipped` of the join each skipped alias goes with.
    skipped_with: dict[str, int] = {}

    def _lose(alias: str, dimension: str) -> None:
        at = skipped_with[alias]
        lost = (*skipped[at].dimensions, dimension)
        skipped[at] = replace(skipped[at], dimensions=lost)

    def _collect_aliases(join_cfgs: Any, nested: list[str]) -> None:
        for join_cfg in join_cfgs or []:
            if isinstance(join_cfg, dict):
                if isinstance(join_cfg.get('name'), str):
                    aliases.add(join_cfg['name'])
                    nested.append(join_cfg['name'])
                _collect_aliases(join_cfg.get('joins'), nested)

    def _add_joins(join_cfgs: Any, parent_path: tuple[str, ...]) -> None:
        parent_alias = parent_path[-1] if parent_path else SOURCE_ALIAS
        for join_cfg in join_cfgs or []:
            if not isinstance(join_cfg, dict):
                messages.append(
                    f'Failed to add join {join_cfg!r} since it is not a mapping'
                )
                continue

            alias = join_cfg.get('name')
            if not isinstance(alias, str):
                messages.append(
                    f'Failed to add join {join_cfg!r} since it misses a name'
                )
                continue

            def _skip(reason: str) -> None:
                nested: list[str] = []
                _collect_aliases(join_cfg.get('joins'), nested)
                for name in (alias, *nested):
                    skipped_with.setdefault(name, len(skipped))
                skipped.append(SkippedJoin(alias, reason, tuple(nested)))
                aliases.add(alias)

            if alias == SOURCE_ALIAS or alias in kept:
                _skip(f"the name '{alias}' is already in use")
                continue

            table = parse_table_reference(join_cfg.get('source'))
            if table is None:
                _skip(
                    'only plain table names are supported as join sources. '
                    'Create a regular view with its query and join that '
                    'instead'
                )
                continue

            cardinality = join_cfg.get('cardinality', 'many_to_one')
            if cardinality not in _CARDINALITIES:
                _skip(f"of its unsupported cardinality '{cardinality}'")
                continue

            using = join_cfg.get('using')
            if isinstance(using, str):
                using = [using]
            if using is not None:
                if not isinstance(using, list) or len(using) != 1:
                    _skip(
                        'it joins on more than one column, which a graph '
                        'link cannot express'
                    )
                    continue
                column = unquote_ident(str(using[0]))
                condition = (
                    JoinKeyRef(qualifier=parent_alias, column=column),
                    JoinKeyRef(qualifier=alias, column=column),
                )
            else:
                on_cfg = join_cfg.get('on', join_cfg.get(True))
                parsed = parse_join_condition(_sql(on_cfg))
                if parsed is None:
                    _skip(
                        'only single-column equality join conditions are '
                        'supported'
                    )
                    continue
                condition = parsed

            joins.append(
                MetricViewJoin(
                    alias=alias,
                    table=table,
                    parent_alias=parent_alias,
                    path=parent_path + (alias,),
                    condition=condition,
                    cardinality=cardinality,
                )
            )
            aliases.add(alias)
            kept.add(alias)
            _add_joins(
                join_cfg.get('joins'), parent_path=parent_path + (alias,)
            )

    _add_joins(cfg.get('joins'), parent_path=())

    candidates: list[tuple[tuple[str, ...], str]] = []
    for join in joins:
        candidates.append((join.path, join.alias))
        if len(join.path) > 1:
            candidates.append(((join.alias,), join.alias))
    for alias in aliases - kept:
        candidates.append(((alias,), alias))
    candidates.append(((SOURCE_ALIAS,), SOURCE_ALIAS))
    candidates.sort(key=lambda candidate: len(candidate[0]), reverse=True)
    qualifiers = {
        tuple(part.lower() for part in path) for path, _ in candidates
    }

    row_filter = _sql(cfg.get('filter'))
    # What the guard refuses, such as a sub-query, is not read but left for
    # the check below to refuse, saying why.
    if (
        isinstance(row_filter, str)
        and _unsafe_expr_reason(row_filter, 'databricks') is None
    ):
        try:
            substituted, referenced = _substitute_parameters(
                row_filter, parameters
            )
        except UnreadableExpressionError as error:
            _refuse_filter(row_filter, str(error))
        if reason := _unusable_parameter(referenced):
            _refuse_filter(row_filter, reason, default_helps=True)
        row_filter = substituted
        used.update((parameter.name, parameter) for parameter in referenced)
    row_filter = _check_filter(row_filter, candidates)

    columns: list[MetricViewColumn | MetricViewWildcard] = []
    # Each field's expression by folded name, or `None` for one refused,
    # so a field naming it is refused too rather than read as a column.
    defined: dict[str, str | None] = {}
    # `fields` is the newer name of `dimensions`, and Databricks reads only
    # `dimensions` when a view has both.
    dim_cfgs = list(cfg.get('dimensions') or cfg.get('fields') or [])
    if cfg.get('dimensions') and cfg.get('fields'):
        messages.append(
            "Ignored the 'fields' section since the view also has "
            "'dimensions', and Databricks reads only those"
        )
    for dim_cfg in dim_cfgs:
        if not isinstance(dim_cfg, dict):
            messages.append(
                f'Failed to add dimension {dim_cfg!r} since it is not a mapping'
            )
            continue

        name = dim_cfg.get('name')
        expr = _sql(dim_cfg.get('expr'))
        if isinstance(expr, str) and (wildcard := _WILDCARD_RE.match(expr)):
            path = tuple(
                unquote_ident(part).lower()
                for part in _IDENT_RE.findall(wildcard.group(1))
            )
            aliases_of = {
                tuple(part.lower() for part in path_): alias_
                for path_, alias_ in candidates
            }
            wildcard_alias = aliases_of.get(path)
            if wildcard_alias is None:
                messages.append(
                    f"Failed to add dimension '{expr.strip()}' since it "
                    f'references no table of the view'
                )
            elif wildcard_alias != SOURCE_ALIAS and wildcard_alias not in kept:
                _lose(wildcard_alias, expr.strip())
            else:
                excluded = tuple(
                    unquote_ident(_IDENT_RE.findall(chain)[-1])
                    for chain in re.findall(_CHAIN, wildcard.group(2) or '')
                )
                columns.append(MetricViewWildcard(wildcard_alias, excluded))
            continue
        if not isinstance(name, str) or not isinstance(expr, str):
            messages.append(
                f'Failed to add dimension {dim_cfg!r} since it '
                f'misses a name or expression'
            )
            continue

        # What the guard refuses, such as a sub-query, is not read but left
        # for the guard to drop, saying why.
        if _unsafe_expr_reason(expr, 'databricks') is not None:
            defined[name.lower()] = None
        else:
            try:
                expr, referenced = _substitute_parameters(expr, parameters)
                expr, refused = _substitute_fields(expr, defined, qualifiers)
            except UnreadableExpressionError as error:
                reason = str(error)
            else:
                reason = _unusable_parameter(referenced)
                if reason is None and refused is not None:
                    reason = (
                        f"it names the dimension '{refused}', which is refused"
                    )
            if reason is not None:
                messages.append(
                    f"Failed to add dimension '{name}' since {reason}"
                )
                defined[name.lower()] = None
                continue
            used.update((parameter.name, parameter) for parameter in referenced)
            defined[name.lower()] = expr

        refs: set[str] = set()
        for path, path_alias in candidates:
            if references_alias(expr, path):
                refs.add(path_alias)
                expr = strip_alias_qualifier(expr, path)
        if len(refs) > 1:
            messages.append(
                f"Failed to add dimension '{name}' since its "
                f'expression references multiple tables'
            )
            continue

        alias = refs.pop() if refs else SOURCE_ALIAS
        if alias != SOURCE_ALIAS and alias not in kept:
            _lose(alias, name)
            continue

        columns.append(
            MetricViewColumn(
                name=name,
                alias=alias,
                expr=expr,
            )
        )

    measures = _read_measures(
        cfg.get('measures'),
        candidates,
        kept,
        dimensions={
            column.name.lower(): column
            for column in columns
            if isinstance(column, MetricViewColumn)
        },
        parameters=set(parameters),
        messages=messages,
    )
    return MetricViewSpec(
        source=source_ref,
        joins=joins,
        columns=columns,
        messages=messages,
        skipped_joins=skipped,
        notes=[
            ParameterNote(parameter.name, parameter.text)
            for parameter in used.values()
        ],
        filter=row_filter,
        window_orders=_window_orders(cfg.get('measures')),
        measures=measures,
    )


def _read_measures(
    measure_cfgs: Any,
    candidates: Sequence[tuple[tuple[str, ...], str]],
    kept: Collection[str],
    dimensions: Mapping[str, MetricViewColumn],
    parameters: Collection[str],
    messages: list[str],
) -> list[MetricViewMeasure]:
    r"""Reads which columns each measure aggregates, never running it.

    A name that is a dimension, a measure or a parameter is no column to
    add, and a measure over more than one table reads none, since no one
    table's columns can express it.

    Args:
        measure_cfgs: The ``measures`` section, if any.
        candidates: Every join path or alias a column can be qualified by,
            with the alias it resolves to.
        kept: The aliases of the joins converted.
        dimensions: The dimensions converted, by folded name.
        parameters: The folded names of the parameters.
        messages: Where to say which measure was left out, and why.
    """
    paths = {
        tuple(part.lower() for part in path): alias
        for path, alias in candidates
    }
    measure_names = {
        cfg['name'].lower()
        for cfg in (measure_cfgs if isinstance(measure_cfgs, list) else [])
        if isinstance(cfg, dict) and isinstance(cfg.get('name'), str)
    }
    measures: list[MetricViewMeasure] = []
    for cfg in measure_cfgs if isinstance(measure_cfgs, list) else []:
        if not isinstance(cfg, dict):
            continue
        name, expr = cfg.get('name'), _sql(cfg.get('expr'))
        if not isinstance(name, str) or not isinstance(expr, str):
            continue
        try:
            names = name_references(expr)
        except UnreadableExpressionError as error:
            messages.append(f"Failed to add measure '{name}' since {error}")
            continue
        inputs: list[tuple[str, str]] = []
        read: set[str] = set()  # The aliases of every table read.
        for named in names:
            parts = named.parts
            if len(parts) == 1:
                folded = parts[0].lower()
                if (
                    folded in measure_names
                    or folded in parameters
                    or (folded,) in paths
                ):
                    continue
                if (dimension := dimensions.get(folded)) is not None:
                    # Already a column of the graph.
                    read.add(dimension.alias)
                    continue
                reference = (SOURCE_ALIAS, parts[0])
            else:
                alias = paths.get(tuple(part.lower() for part in parts[:-1]))
                if alias is None:
                    continue  # A field of a struct, or nothing known.
                reference = (alias, parts[-1])
            read.add(reference[0])
            if reference not in inputs:
                inputs.append(reference)
        if len(read) > 1 or not read <= {SOURCE_ALIAS, *kept}:
            inputs = []
        measures.append(
            MetricViewMeasure(
                name=name,
                inputs=tuple(inputs),
                aggregation=_aggregation(cfg, expr, inputs, dimensions),
            )
        )
    return measures


def _aggregation(
    cfg: Mapping[str, Any],
    expr: str,
    inputs: Sequence[tuple[str, str]],
    dimensions: Mapping[str, MetricViewColumn],
) -> tuple[str, str | None] | None:
    r"""The predictive query aggregation a measure is, if it is one plain
    aggregate of a column or dimension of the source table, or a count of
    its rows.
    """
    match = _SIMPLE_MEASURE_RE.fullmatch(_masked_strings(expr))
    if match is None or 'window' in cfg:
        return None
    function, distinct, argument = match.groups()
    function = function.upper()
    if argument in ('*', '1'):
        if function != 'COUNT' or distinct:
            return None
        return 'COUNT', None
    dimension = dimensions.get(unquote_ident(argument).lower())
    if dimension is not None and dimension.alias == SOURCE_ALIAS:
        column = dimension.name
    elif len(inputs) == 1 and inputs[0][0] == SOURCE_ALIAS:
        column = inputs[0][1]
    else:
        return None
    if distinct:
        if function != 'COUNT':
            return None
        function = 'COUNT_DISTINCT'
    return function, column


def _window_orders(measure_cfgs: Any) -> list[str]:
    r"""The dimensions the window measures order by, in order."""
    orders: list[str] = []
    for measure_cfg in measure_cfgs if isinstance(measure_cfgs, list) else []:
        windows = (
            measure_cfg.get('window') if isinstance(measure_cfg, dict) else None
        )
        if isinstance(windows, dict):
            windows = [windows]
        for window in windows if isinstance(windows, list) else []:
            order = window.get('order') if isinstance(window, dict) else None
            if isinstance(order, str) and unquote_ident(order) not in orders:
                orders.append(unquote_ident(order))
    return orders


def _check_filter(
    row_filter: Any,
    candidates: Sequence[tuple[tuple[str, ...], str]],
) -> MetricViewFilter | None:
    r"""Returns the filter if it can be applied to the source table alone.

    The graph keeps each join as a table of its own, so a condition on a
    joined table has no single table to apply to. Dropping it instead would
    let predictions read rows the view leaves out.

    Args:
        row_filter: The ``filter`` section, if any.
        candidates: Every join path or alias a column can be qualified by,
            with the alias it resolves to.
    """
    if row_filter is None:
        return None

    reason: str | None = None
    if not isinstance(row_filter, str) or not row_filter.strip():
        reason = 'it is not a SQL condition'
    else:
        reason = _unsafe_expr_reason(row_filter, 'databricks')
    if reason is None:
        assert isinstance(row_filter, str)
        masked = _masked_strings(row_filter)
        for path, alias in candidates:
            if alias != SOURCE_ALIAS and references_alias(masked, path):
                reason = f"it references the joined table '{alias}'"
                break
    if reason is not None:
        _refuse_filter(row_filter, reason)
    return MetricViewFilter(row_filter.strip())


def _refuse_filter(
    row_filter: Any,
    reason: str,
    *,
    trusting_helps: bool = False,
    default_helps: bool = False,
) -> NoReturn:
    r"""Refuses the view for its filter, and says the ways out.

    Args:
        row_filter: The filter.
        reason: Why, as a clause.
        trusting_helps: Whether ``trust_expressions=True`` would accept it,
            as for a call to a function other than a pure built-in.
        default_helps: Whether a default for the parameter it reads would
            let it be read.
    """
    hatch = ''
    if default_helps:
        hatch = ', or give the parameter a default of its type'
    if trusting_helps:
        hatch = (
            ', or pass trust_expressions=True to apply it if you trust '
            'whoever authored the view'
        )
    raise UnsupportedFilterError(
        f'Cannot apply the metric view filter {row_filter!r} since '
        f'{reason}. The graph would otherwise read rows the view leaves '
        f'out. Create a regular view with this WHERE clause and point the '
        f"metric view's source at it{hatch}"
    )


def resolve_join_keys(
    join: MetricViewJoin,
    child_columns: Collection[str],
    parent_columns: Collection[str],
) -> tuple[str, str, str]:
    r"""Resolves a join condition into ``(other_alias, other_key,
    child_key)``.

    ``other_alias`` is the alias of the non-child side of the join, either
    :attr:`MetricViewJoin.parent_alias` or :obj:`SOURCE_ALIAS`. Unqualified
    column references are resolved by their (unambiguous) membership in the
    child or parent table, the child first: Databricks reads an unqualified
    column of a join's ``on`` as one of the joined table. Names are compared
    without regard to case, as Databricks compares them, and the keys come
    back as written.

    Raises:
        UnsupportedJoinError: If the condition cannot be resolved.
    """
    child_names = {join.alias.lower(), '.'.join(join.path).lower()}
    parent_names = {join.parent_alias.lower()}
    if len(join.path) > 1:
        parent_names.add('.'.join(join.path[:-1]).lower())
    child_folded = by_folded_name(child_columns)
    parent_folded = by_folded_name(parent_columns)

    def _side(ref: JoinKeyRef) -> str | None:
        if ref.qualifier is not None:
            qualifier = ref.qualifier.lower()
            if qualifier in child_names:
                return join.alias
            if qualifier in parent_names:
                return join.parent_alias
            if qualifier == SOURCE_ALIAS:
                return SOURCE_ALIAS
            return None
        if ref.column.lower() in child_folded:
            return join.alias
        if ref.column.lower() in parent_folded:
            return join.parent_alias
        return None

    first, second = join.condition
    first_side, second_side = _side(first), _side(second)

    if first_side == join.alias and second_side not in (None, join.alias):
        assert second_side is not None
        return second_side, second.column, first.column
    if second_side == join.alias and first_side not in (None, join.alias):
        assert first_side is not None
        return first_side, first.column, second.column

    raise UnsupportedJoinError(
        join.alias,
        f'its join condition could not be resolved into a key reference '
        f"between '{join.parent_alias}' and '{join.alias}'",
    )


@dataclass(frozen=True)
class MetricViewDescription:
    r"""What ``DESCRIBE TABLE EXTENDED`` says about a metric view.

    Args:
        definition: The YAML definition.
        dtypes: The data types of all dimensions, by name.
        parameters: The data types of all parameters, by name. Their
            defaults are in the definition only.
        catalog: The catalog of the metric view, if given.
        schema: The schema of the metric view, if given.
    """

    definition: str
    dtypes: dict[str, str]
    parameters: dict[str, str]
    catalog: str | None
    schema: str | None


# The section of `DESCRIBE TABLE EXTENDED` that comes before any header.
_COLUMNS_SECTION = ''


def _describe_sections(rows: Sequence[Any]) -> dict[str, list[tuple[str, str]]]:
    r"""Splits the rows of ``DESCRIBE TABLE EXTENDED`` into their sections,
    keyed by the lower-case header without its ``#``.

    The columns come first, then sections such as ``# Parameters`` and
    ``# Detailed Table Information``. A header names no type, which tells it
    from a column named ``#...``. Row width differs between a SQL warehouse
    (four) and a Spark session (three), and so does whether an empty cell is
    ``''`` or ``None``, so only the first two cells are read.
    """
    sections: dict[str, list[tuple[str, str]]] = {_COLUMNS_SECTION: []}
    current: str | None = _COLUMNS_SECTION
    for row in rows:
        name = str(row[0]).strip() if row[0] is not None else ''
        value = str(row[1]) if len(row) > 1 and row[1] is not None else ''
        if name.startswith('#') and not value.strip():
            header = name.lstrip('#').strip().lower()
            # Spark heads the rows of some sections with `# col_name`.
            if header != 'col_name':
                current = header
                sections.setdefault(current, [])
            continue
        if not name:
            # A blank row ends the columns; rows up to the next header belong
            # to no section.
            if current == _COLUMNS_SECTION:
                current = None
            continue
        if current is not None:
            sections[current].append((name, value))
    return sections


def read_metric_view_definition(
    connection: AnyConnection,
    quoted_name: str,
) -> MetricViewDescription:
    r"""Reads a metric view definition from a Databricks SQL warehouse or
    Spark session.

    Each section of the description is read on its own, so a parameter named
    like a detail, such as ``Type`` or ``View Text``, cannot stand in for it.

    Raises:
        TableNotFoundError: If there is no such relation, or you may not see
            it.
        ViewConversionError: If the relation is not a metric view or
            exposes no definition.
    """
    from sdm_relational_engine.rfm.backend.databricks.errors import sql_state

    try:
        with connection.cursor() as cursor:
            cursor.execute(f'DESCRIBE TABLE EXTENDED {quoted_name}')
            rows = cursor.fetchall()
    except Exception as error:
        # A mistyped name is the caller's to fix, as a missing table is.
        if sql_state(error) != '42P01':
            raise
        raise TableNotFoundError(
            f"Metric view '{quoted_name}' does not exist or you may not see it"
        ) from error

    sections = _describe_sections(rows)
    dtypes = {
        name: data_type
        for name, data_type in sections[_COLUMNS_SECTION]
        if not data_type.endswith(_MEASURE_SUFFIX)
    }
    parameters = dict(sections.get('parameters', []))
    details: dict[str, str] = {}
    for name, value in sections.get('detailed table information', []):
        details.setdefault(name, value)

    if details.get('Type') != 'METRIC_VIEW':
        raise ViewConversionError(
            f"'{quoted_name}' is not a metric view "
            f"(got type '{details.get('Type', 'UNKNOWN')}')"
        )

    definition = details.get('View Text')
    if not definition:
        raise ViewConversionError(
            f"Could not read the definition of metric view '{quoted_name}'"
        )

    return MetricViewDescription(
        definition=definition,
        dtypes=dtypes,
        parameters=parameters,
        catalog=details.get('Catalog'),
        schema=details.get('Database'),
    )
