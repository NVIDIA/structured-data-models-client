# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import warnings
from typing import Any

import pandas as pd
import pytest
from sdm_relational_engine.exceptions import (
    EmptyTableError,
    GraphConstructionError,
    KumoRelationalError,
    TableNotFoundError,
)
from sdm_relational_engine.rfm import Edge, Graph, ViewConversionWarning
from sdm_relational_engine.rfm.view_conversion import (
    MeasureNote,
    ParameterNote,
    TimeColumnNote,
)

try:
    import pyarrow as pa
    from sdm_connectors.backends.databricks import Connection
    from sdm_relational_engine.rfm.backend.databricks.metric_view import (
        SOURCE_ALIAS,
        JoinKeyRef,
        MetricViewFilter,
        MetricViewJoin,
        MetricViewWildcard,
        UnsupportedFilterError,
        UnsupportedJoinError,
        parse_join_condition,
        parse_metric_view,
        parse_table_reference,
        read_metric_view_definition,
        references_alias,
        resolve_join_keys,
        strip_alias_qualifier,
        unquote_ident,
    )
except ImportError:
    pytest.skip("'databricks' extension not installed", allow_module_level=True)


def test_unquote_ident() -> None:
    assert unquote_ident('name') == 'name'
    assert unquote_ident('`name`') == 'name'
    assert unquote_ident('`my name`') == 'my name'
    assert unquote_ident('`my``name`') == 'my`name'
    assert unquote_ident(' name ') == 'name'


def test_parse_table_reference() -> None:
    assert parse_table_reference('tab') == ('tab',)
    assert parse_table_reference('sch.tab') == ('sch', 'tab')
    assert parse_table_reference('cat.sch.tab') == ('cat', 'sch', 'tab')
    assert parse_table_reference('`c at`.sch.`my tab`') == (
        'c at',
        'sch',
        'my tab',
    )
    assert parse_table_reference('  cat.sch.tab  ') == ('cat', 'sch', 'tab')

    assert parse_table_reference(None) is None
    assert parse_table_reference(42) is None
    assert parse_table_reference('SELECT * FROM cat.sch.tab') is None
    assert parse_table_reference('cat.sch.tab AS t') is None
    assert parse_table_reference('cat.sch.tab.col') is None


def test_parse_join_condition() -> None:
    assert parse_join_condition('source.a = customers.b') == (
        JoinKeyRef(qualifier='source', column='a'),
        JoinKeyRef(qualifier='customers', column='b'),
    )
    assert parse_join_condition('a = b') == (
        JoinKeyRef(qualifier=None, column='a'),
        JoinKeyRef(qualifier=None, column='b'),
    )
    assert parse_join_condition('customers.a=b') == (
        JoinKeyRef(qualifier='customers', column='a'),
        JoinKeyRef(qualifier=None, column='b'),
    )
    assert parse_join_condition('`source` . `my a` = x.`my b`') == (
        JoinKeyRef(qualifier='source', column='my a'),
        JoinKeyRef(qualifier='x', column='my b'),
    )
    assert parse_join_condition('source.2024_id = c.2024_id') == (
        JoinKeyRef(qualifier='source', column='2024_id'),
        JoinKeyRef(qualifier='c', column='2024_id'),
    )

    assert parse_join_condition(None) is None
    assert parse_join_condition('a.x = b.y AND a.z = b.w') is None
    assert parse_join_condition('a.x >= b.y') is None
    assert parse_join_condition('a.x = b.y + 1') is None


def test_parse_join_condition_paths() -> None:
    assert parse_join_condition('customer.nation.n_nationkey = source.x') == (
        JoinKeyRef(qualifier='customer.nation', column='n_nationkey'),
        JoinKeyRef(qualifier='source', column='x'),
    )
    assert parse_join_condition('`a`.`b c`.key = other') == (
        JoinKeyRef(qualifier='a.b c', column='key'),
        JoinKeyRef(qualifier=None, column='other'),
    )


def test_references_alias() -> None:
    assert references_alias('products.generation', 'products')
    assert references_alias("DATE_TRUNC('DAY', products.at)", 'products')
    assert references_alias('`products`.generation', 'products')
    assert references_alias('PRODUCTS.generation', 'products')
    assert references_alias('UPPER(`my join`.name)', 'my join')

    assert not references_alias('other_products.generation', 'products')
    assert not references_alias('products_x.generation', 'products')
    assert not references_alias('a.products.generation', 'products')
    assert not references_alias('products', 'products')


def test_strip_alias_qualifier() -> None:
    assert (
        strip_alias_qualifier('products.generation', 'products') == 'generation'
    )
    assert (
        strip_alias_qualifier('COALESCE(source.a, source.`b c`)', 'source')
        == 'COALESCE(a, `b c`)'
    )
    assert (
        strip_alias_qualifier('`my join`.a + `my join`.b', 'my join') == 'a + b'
    )
    assert strip_alias_qualifier('other.a', 'products') == 'other.a'


def test_parse_metric_view_star_schema() -> None:
    spec = parse_metric_view("""
version: "0.1"
source: cat.sch.order_lines
joins:
  - name: customers
    source: cat.sch.customers
    on: source.customer_id = customers.customer_id
  - name: products
    source: cat.sch.products
    on: products.product_id = source.product_id
dimensions:
  - name: Order Month
    expr: DATE_TRUNC('MONTH', source.order_date)
  - name: product_line
    expr: source.product_line
  - name: Customer Segment
    expr: customers.segment
measures:
  - name: Total Revenue
    expr: SUM(source.amount)
""")

    assert spec.source == ('cat', 'sch', 'order_lines')
    assert spec.messages == []

    assert len(spec.joins) == 2
    assert spec.joins[0] == MetricViewJoin(
        alias='customers',
        table=('cat', 'sch', 'customers'),
        parent_alias=SOURCE_ALIAS,
        path=('customers',),
        condition=(
            JoinKeyRef(qualifier='source', column='customer_id'),
            JoinKeyRef(qualifier='customers', column='customer_id'),
        ),
        cardinality='many_to_one',
    )
    assert spec.joins[1].condition == (
        JoinKeyRef(qualifier='products', column='product_id'),
        JoinKeyRef(qualifier='source', column='product_id'),
    )

    assert [(c.name, c.alias, c.expr) for c in spec.columns] == [
        ('Order Month', 'source', "DATE_TRUNC('MONTH', order_date)"),
        ('product_line', 'source', 'product_line'),
        ('Customer Segment', 'customers', 'segment'),
    ]


def test_parse_metric_view_quoted_on_key() -> None:
    spec = parse_metric_view("""
source: cat.sch.facts
joins:
  - name: dim
    source: cat.sch.dim
    'on': source.dim_id = dim.id
""")
    assert len(spec.joins) == 1
    assert spec.messages == []


def test_parse_metric_view_fields_key() -> None:
    spec = parse_metric_view("""
source: cat.sch.facts
fields:
  - name: month
    expr: DATE_TRUNC('MONTH', source.at)
""")
    assert [(c.name, c.expr) for c in spec.columns] == [
        ('month', "DATE_TRUNC('MONTH', at)"),
    ]


def test_parse_metric_view_field_references() -> None:
    spec = parse_metric_view("""
source: cat.sch.facts
joins:
  - name: dim
    source: cat.sch.dim
    on: source.dim_id = dim.id
dimensions:
  - name: order_month
    expr: DATE_TRUNC('MONTH', source.order_date)
  - name: order_quarter
    expr: DATE_TRUNC('QUARTER', order_month)
  - name: Dim Name
    expr: UPPER(dim.name)
  - name: dim_prefix
    expr: LEFT(`Dim Name`, 2)
  - name: mixed
    expr: CONCAT(order_month, `Dim Name`)
""")

    columns = {c.name: c for c in spec.columns}
    assert columns['order_quarter'].alias == 'source'
    assert columns['order_quarter'].expr == (
        "DATE_TRUNC('QUARTER', (DATE_TRUNC('MONTH', order_date)))"
    )
    assert columns['dim_prefix'].alias == 'dim'
    assert columns['dim_prefix'].expr == 'LEFT((UPPER(name)), 2)'
    assert 'mixed' not in columns
    assert spec.messages == [
        "Failed to add dimension 'mixed' since its expression references "
        'multiple tables',
    ]


def test_a_field_replaces_only_its_name_where_a_value_stands() -> None:
    r"""A field named like a function, type, datetime unit or qualifier is
    substituted where its name reads a value alone: a field `upper` turned
    `UPPER(source.z)` into `(x)(z)`. A member of a field is read through it,
    unless the name begins a join's path.
    """
    spec = parse_metric_view("""
version: 1.1
source: cat.sch.t
joins:
  - name: customers
    source: cat.sch.customers
    on: source.customer_id = customers.id
fields:
  - name: upper
    expr: source.x
  - name: y
    expr: UPPER(source.z) || `upper`
  - name: day
    expr: source.day_of_order
  - name: next
    expr: source.ts + INTERVAL 1 DAY
  - name: dom
    expr: extract(DAY FROM source.ts) + day
  - name: int
    expr: source.n
  - name: c
    expr: CAST(source.z AS INT) + int
  - name: source
    expr: source.w
  - name: q
    expr: source.z || source
  - name: address
    expr: named_struct('city', upper(source.address.city))
  - name: city
    expr: address.city || `address`['city']
  - name: customers
    expr: source.customer_id
  - name: segment
    expr: customers.segment
""")

    assert [(c.name, c.expr) for c in spec.columns] == [
        ('upper', 'x'),
        ('y', 'UPPER(z) || (x)'),
        ('day', 'day_of_order'),
        ('next', 'ts + INTERVAL 1 DAY'),
        ('dom', 'extract(DAY FROM ts) + (day_of_order)'),
        ('int', 'n'),
        ('c', 'CAST(z AS INT) + (n)'),
        ('source', 'w'),
        ('q', 'z || (w)'),
        ('address', "named_struct('city', upper(address.city))"),
        (
            'city',
            "(named_struct('city', upper(address.city))).city "
            "|| (named_struct('city', upper(address.city)))['city']",
        ),
        ('customers', 'customer_id'),
        ('segment', 'segment'),
    ]
    assert spec.columns[-1].alias == 'customers'
    assert spec.messages == []


def test_a_field_whose_names_cannot_be_settled_is_refused() -> None:
    r"""Where its words cannot be told apart, substituting a field could
    rewrite one that is not a value.
    """
    spec = parse_metric_view("""
source: cat.sch.t
dimensions:
  - name: a
    expr: source.x
  - name: b
    expr: a AT TIME ZONE 'UTC'
""")

    assert [column.name for column in spec.columns] == ['a']
    assert spec.messages == [
        "Failed to add dimension 'b' since it has syntax that is not "
        'understood, so which of its words name columns cannot be settled'
    ]


def test_parse_metric_view_nested_joins() -> None:
    spec = parse_metric_view("""
source: cat.sch.facts
joins:
  - name: customer
    source: cat.sch.customers
    on: source.customer_id = customer.customer_id
    joins:
      - name: nation
        source: cat.sch.nations
        on: c_nationkey = n_nationkey
""")

    assert spec.messages == []
    assert [j.alias for j in spec.joins] == ['customer', 'nation']
    assert spec.joins[1].parent_alias == 'customer'
    assert spec.joins[1].path == ('customer', 'nation')
    assert spec.joins[1].condition == (
        JoinKeyRef(qualifier=None, column='c_nationkey'),
        JoinKeyRef(qualifier=None, column='n_nationkey'),
    )


def test_parse_metric_view_nested_full_path_dimensions() -> None:
    spec = parse_metric_view("""
source: cat.sch.facts
joins:
  - name: customer
    source: cat.sch.customers
    on: source.customer_id = customer.customer_id
    joins:
      - name: nation
        source: cat.sch.nations
        on: customer.c_nationkey = nation.n_nationkey
dimensions:
  - name: Nation Name
    expr: customer.nation.n_name
  - name: Customer Name
    expr: UPPER(customer.c_name)
  - name: Nation Lax
    expr: nation.n_comment
  - name: Mixed Path
    expr: CONCAT(customer.nation.n_name, customer.c_name)
""")

    columns = {c.name: (c.alias, c.expr) for c in spec.columns}
    assert columns['Nation Name'] == ('nation', 'n_name')
    assert columns['Customer Name'] == ('customer', 'UPPER(c_name)')
    assert columns['Nation Lax'] == ('nation', 'n_comment')
    assert 'Mixed Path' not in columns
    assert spec.messages == [
        "Failed to add dimension 'Mixed Path' since its expression "
        'references multiple tables',
    ]


def test_resolve_join_keys_full_path_qualifiers() -> None:
    join = MetricViewJoin(
        alias='nation',
        table=('cat', 'sch', 'nations'),
        parent_alias='customer',
        path=('customer', 'nation'),
        condition=(
            JoinKeyRef(qualifier='customer.nation', column='n_nationkey'),
            JoinKeyRef(qualifier='customer', column='c_nationkey'),
        ),
        cardinality='many_to_one',
    )
    assert resolve_join_keys(join, {'n_nationkey'}, {'c_nationkey'}) == (
        'customer',
        'c_nationkey',
        'n_nationkey',
    )


def test_parse_metric_view_using() -> None:
    spec = parse_metric_view("""
source: cat.sch.facts
joins:
  - name: products
    source: cat.sch.products
    using: [product_id]
  - name: composite
    source: cat.sch.other
    using: [a, b]
""")

    assert len(spec.joins) == 1
    assert spec.joins[0].condition == (
        JoinKeyRef(qualifier=SOURCE_ALIAS, column='product_id'),
        JoinKeyRef(qualifier='products', column='product_id'),
    )
    assert spec.messages == []
    (skipped,) = spec.skipped_joins
    assert skipped.alias == 'composite' and 'more than one column' in str(
        skipped
    )


def test_parse_metric_view_cardinality() -> None:
    spec = parse_metric_view("""
source: cat.sch.facts
joins:
  - name: refunds
    source: cat.sch.refunds
    on: refunds.fact_id = source.id
    cardinality: one_to_many
  - name: broken
    source: cat.sch.broken
    on: broken.fact_id = source.id
    cardinality: many_to_many
""")

    assert [j.alias for j in spec.joins] == ['refunds']
    assert spec.joins[0].cardinality == 'one_to_many'
    (skipped,) = spec.skipped_joins
    assert skipped.alias == 'broken' and "'many_to_many'" in skipped.reason


def test_parse_metric_view_skipped_joins() -> None:
    spec = parse_metric_view("""
source: cat.sch.facts
filter: source.status = 'VALID'
joins:
  - name: promos
    source: SELECT * FROM cat.sch.promos
    on: source.promo_id = promos.promo_id
    joins:
      - name: nested
        source: cat.sch.nested
        on: promos.nested_id = nested.id
  - name: composite
    source: cat.sch.other
    on: source.a = composite.a AND source.b = composite.b
dimensions:
  - name: promo_kind
    expr: promos.kind
  - name: nested_name
    expr: nested.name
  - name: composite_flag
    expr: composite.flag = 1
  - name: valid
    expr: source.amount
""")

    assert spec.joins == []
    assert [(c.name, c.alias) for c in spec.columns] == [
        ('valid', 'source'),
    ]
    assert spec.filter == MetricViewFilter("source.status = 'VALID'")
    assert spec.messages == []
    assert [
        (join.alias, join.nested, join.dimensions)
        for join in spec.skipped_joins
    ] == [
        ('promos', ('nested',), ('promo_kind', 'nested_name')),
        ('composite', (), ('composite_flag',)),
    ]


_FILTERED_YAML = """
source: cat.sch.facts
filter: {filter}
joins:
  - name: customers
    source: cat.sch.customers
    on: source.customer_id = customers.customer_id
    joins:
      - name: regions
        source: cat.sch.regions
        on: customers.region_id = regions.region_id
  - name: promos
    source: SELECT * FROM cat.sch.promos
    on: source.promo_id = promos.promo_id
dimensions:
  - name: amount
    expr: source.amount
"""


@pytest.mark.parametrize(
    'row_filter',
    [
        "source.status = 'VALID'",
        "status = 'VALID' AND amount > 0",
        "`source`.`status` = 'VALID'",
        "`order status` IN ('a', 'b')",
        "source.note <> 'customers.segment'",
        "UPPER(source.status) = 'VALID' AND source.amount BETWEEN 1 AND 9",
        # A raw string takes no escapes, so this one ends in a backslash.
        "source.note <> r'C:\\dir\\'",
    ],
)
def test_parse_metric_view_source_filter(row_filter: str) -> None:
    spec = parse_metric_view(
        _FILTERED_YAML.format(filter=json.dumps(row_filter))
    )

    assert spec.filter is not None
    assert spec.filter.checked(_FACT_COLUMNS) == row_filter
    assert not any('filter' in message for message in spec.messages)


@pytest.mark.parametrize(
    ('row_filter', 'reason'),
    [
        ("customers.segment = 'gold'", "the joined table 'customers'"),
        ("`customers`.`segment` = 'gold'", "the joined table 'customers'"),
        ("customers.regions.name = 'EMEA'", "the joined table 'regions'"),
        ("regions.name = 'EMEA'", "the joined table 'regions'"),
        ('promos.kind IS NULL', "the joined table 'promos'"),
        (
            'source.customer_id IN (SELECT customer_id FROM cat.sch.vip)',
            "keyword 'SELECT'",
        ),
        ('source.customer_id IN (TABLE cat.sch.vip)', "keyword 'TABLE'"),
        ('source.amount > 0; DROP TABLE x', "contains ';'"),
        ('source.note = $$x$$', "'$$'"),
        (
            "r'x\\' = source.note OR source.customer_id IN "
            "(SELECT customer_id FROM cat.sch.vip) OR 'a\\'b' = source.note",
            "keyword 'SELECT'",
        ),
    ],
)
def test_parse_metric_view_refuses_a_filter(
    row_filter: str, reason: str
) -> None:
    definition = _FILTERED_YAML.format(filter=json.dumps(row_filter))

    with pytest.raises(UnsupportedFilterError) as caught:
        parse_metric_view(definition)

    message = str(caught.value)
    assert reason in message
    assert (
        'Create a regular view with this WHERE clause and point the metric '
        "view's source at it"
    ) in message
    assert isinstance(caught.value, ValueError)


# The columns of `cat.sch.facts`, which `_FILTERED_YAML` reads.
_FACT_COLUMNS = ('status', 'amount', 'order status', 'note', 'token')


@pytest.mark.parametrize(
    ('row_filter', 'reason'),
    [
        ("secret('scope', 'key') = source.token", "calls 'secret'"),
        ('my_udf(source.amount) > 0', "calls 'my_udf'"),
        ('current_date > source.order_date', 'current_date'),
        ('escape(source.note)', "calls 'escape'"),
        ('note FOO escape(token)', "calls 'escape'"),
        ('upper(note) FILTER (WHERE from(token))', "calls 'from'"),
        (
            "r'x\\' IS NOT NULL AND secret('scope', 'key') IS NOT NULL "
            "AND 'a\\'b' IS NOT NULL",
            "calls 'secret'",
        ),
    ],
)
def test_a_checked_filter_refuses_a_call(row_filter: str, reason: str) -> None:
    with pytest.raises(UnsupportedFilterError) as caught:
        MetricViewFilter(row_filter).checked(_FACT_COLUMNS)

    assert reason in str(caught.value)
    assert 'Create a regular view with this WHERE clause' in str(caught.value)


def test_a_checked_filter_reads_a_keyword_after_a_column_as_grammar() -> None:
    # `status` is a column of the source table, so `IN (` is grammar.
    row_filter = MetricViewFilter("status IN ('VALID')")
    assert row_filter.checked(_FACT_COLUMNS) == "status IN ('VALID')"
    with pytest.raises(UnsupportedFilterError, match="calls 'IN'"):
        row_filter.checked(())


@pytest.mark.parametrize(
    'comment',
    [
        "/* secret('s', 'k') */",
        # Databricks nests one block comment in another,
        "/* a /* secret('s', 'k') */ b */",
        # and goes on with a line comment past a backslash ending its line.
        "-- c \\\nAND secret('s', 'k') IS NULL",
    ],
    ids=['block', 'nested', 'continued'],
)
def test_a_comment_is_dropped_and_hides_no_call(comment: str) -> None:
    r"""Databricks drops a comment in a filter, dimension or measure, so
    one is read as dropped, and only what is left is checked and runs.
    """
    spec = parse_metric_view(f"""
source: cat.sch.facts
filter: {json.dumps("status = 'VALID' " + comment)}
dimensions:
  - name: amount
    expr: |
      source.amount -- before tax
""")

    assert spec.filter is not None
    assert spec.filter.checked(_FACT_COLUMNS) == "status = 'VALID'"
    (column,) = spec.columns
    assert column.expr == 'amount'

    hiding = "status = 'VALID' -- it's\nAND secret('s', 'k') IS NULL"
    spec = parse_metric_view(_FILTERED_YAML.format(filter=json.dumps(hiding)))
    assert spec.filter is not None
    with pytest.raises(UnsupportedFilterError, match="calls 'secret'"):
        spec.filter.checked(_FACT_COLUMNS)


def test_parse_metric_view_refuses_a_filter_that_is_not_sql() -> None:
    with pytest.raises(UnsupportedFilterError, match='not a SQL condition'):
        parse_metric_view(_FILTERED_YAML.format(filter='true'))


def test_parsing_leaves_calls_to_the_check() -> None:
    def parse(row_filter: str) -> Any:
        return parse_metric_view(
            _FILTERED_YAML.format(filter=json.dumps(row_filter))
        )

    # Trusted mode skips the check, and nothing else refuses this:
    assert parse('my_udf(source.amount) > 0').filter == MetricViewFilter(
        'my_udf(source.amount) > 0'
    )
    with pytest.raises(UnsupportedFilterError, match='joined table'):
        parse("customers.segment = 'gold'")
    with pytest.raises(UnsupportedFilterError, match='SELECT'):
        parse('source.amount > (SELECT 1)')
    # A stray `)` would close the filtered subquery and append to it:
    with pytest.raises(UnsupportedFilterError, match="closes a '\\('"):
        parse('source.amount > 0) OR (1 = 1')


def test_a_parsed_filter_gives_out_its_sql_only_through_a_check() -> None:
    r"""The parsed filter is no string, so it cannot reach a query unless a
    caller checks it against the source table's columns or trusts the view.
    """
    row_filter = "my_udf(status) = 'VALID'"
    spec = parse_metric_view(
        _FILTERED_YAML.format(filter=json.dumps(row_filter))
    )

    assert isinstance(spec.filter, MetricViewFilter)
    assert not isinstance(spec.filter, str)
    public = {name for name in dir(spec.filter) if not name.startswith('_')}
    assert public == {'checked', 'trusted'}
    with pytest.raises(UnsupportedFilterError, match="calls 'my_udf'"):
        spec.filter.checked(_FACT_COLUMNS)
    assert spec.filter.trusted() == row_filter


def test_parse_metric_view_duplicate_aliases() -> None:
    spec = parse_metric_view("""
source: cat.sch.facts
joins:
  - name: source
    source: cat.sch.other
    on: source.a = source.b
  - name: dim
    source: cat.sch.dim
    on: source.dim_id = dim.id
  - name: dim
    source: cat.sch.dim2
    on: source.dim2_id = dim.id
""")

    assert [j.alias for j in spec.joins] == ['dim']
    assert spec.joins[0].table == ('cat', 'sch', 'dim')
    assert [join.alias for join in spec.skipped_joins] == ['source', 'dim']
    assert all('already in use' in j.reason for j in spec.skipped_joins)


def test_parse_metric_view_errors() -> None:
    with pytest.raises(ValueError, match='not a YAML mapping'):
        parse_metric_view('[]')
    with pytest.raises(ValueError, match="misses a 'source'"):
        parse_metric_view('version: "0.1"')
    with pytest.raises(ValueError, match='Only plain table names'):
        parse_metric_view('source: SELECT * FROM cat.sch.tab')


def _join(
    condition: tuple[JoinKeyRef, JoinKeyRef],
    parent_alias: str = SOURCE_ALIAS,
) -> MetricViewJoin:
    if parent_alias == SOURCE_ALIAS:
        path: tuple[str, ...] = ('dim',)
    else:
        path = (parent_alias, 'dim')
    return MetricViewJoin(
        alias='dim',
        table=('cat', 'sch', 'dim'),
        parent_alias=parent_alias,
        path=path,
        condition=condition,
        cardinality='many_to_one',
    )


def test_resolve_join_keys_qualified() -> None:
    join = _join(
        (
            JoinKeyRef(qualifier='source', column='dim_id'),
            JoinKeyRef(qualifier='dim', column='id'),
        )
    )
    assert resolve_join_keys(join, {'id'}, {'dim_id'}) == (
        'source',
        'dim_id',
        'id',
    )

    join = _join(
        (
            JoinKeyRef(qualifier='dim', column='id'),
            JoinKeyRef(qualifier='source', column='dim_id'),
        )
    )
    assert resolve_join_keys(join, {'id'}, {'dim_id'}) == (
        'source',
        'dim_id',
        'id',
    )


def test_resolve_join_keys_nested_source_reference() -> None:
    join = _join(
        (
            JoinKeyRef(qualifier='source', column='dim_id'),
            JoinKeyRef(qualifier='dim', column='id'),
        ),
        parent_alias='parent',
    )
    assert resolve_join_keys(join, {'id'}, {'other'}) == (
        'source',
        'dim_id',
        'id',
    )


def test_resolve_join_keys_unqualified() -> None:
    join = _join(
        (
            JoinKeyRef(qualifier=None, column='dim_id'),
            JoinKeyRef(qualifier=None, column='id'),
        )
    )
    assert resolve_join_keys(
        join,
        child_columns={'id', 'name'},
        parent_columns={'dim_id', 'amount'},
    ) == ('source', 'dim_id', 'id')


def test_resolve_join_keys_unqualified_means_the_joined_table() -> None:
    r"""In `source.CustomerID = CustomerID` Databricks reads `CustomerID` as
    the joined table's, though the source has one too, and both of the
    view's entity tables were dropped as unresolvable.
    """
    join = _join(
        (
            JoinKeyRef(qualifier='source', column='CustomerID'),
            JoinKeyRef(qualifier=None, column='customerid'),
        )
    )
    assert resolve_join_keys(
        join,
        child_columns={'CustomerID', 'name'},
        parent_columns={'CustomerID', 'amount'},
    ) == ('source', 'CustomerID', 'customerid')


def test_resolve_join_keys_unresolvable() -> None:
    ambiguous = _join(
        (
            JoinKeyRef(qualifier=None, column='id'),
            JoinKeyRef(qualifier=None, column='id'),
        )
    )
    with pytest.raises(UnsupportedJoinError, match='could not be resolved'):
        resolve_join_keys(ambiguous, {'id'}, {'id'})

    unknown = _join(
        (
            JoinKeyRef(qualifier='other', column='a'),
            JoinKeyRef(qualifier='dim', column='id'),
        )
    )
    with pytest.raises(UnsupportedJoinError, match='could not be resolved'):
        resolve_join_keys(unknown, {'id'}, {'a'})

    no_child = _join(
        (
            JoinKeyRef(qualifier='source', column='a'),
            JoinKeyRef(qualifier='source', column='b'),
        )
    )
    with pytest.raises(UnsupportedJoinError, match='could not be resolved'):
        resolve_join_keys(no_child, {'a', 'b'}, {'a', 'b'})


_CATALOG = 'cat'
_SCHEMA = 'sch'

_DATA = {
    'order_lines': pd.DataFrame(
        {
            'line_id': [f'L{i}' for i in range(100)],
            'order_id': [f'O{i % 40}' for i in range(100)],
            'customer_id': [f'C{i % 10}' for i in range(100)],
            'product_id': [f'P{i % 5}' for i in range(100)],
            'promo_id': [f'M{i % 3}' for i in range(100)],
            'order_date': pd.date_range('2025-01-01', periods=100, freq='D'),
            'product_line': [['GPU', 'CPU', 'DPU'][i % 3] for i in range(100)],
            'status': ['VALID'] * 100,
            'amount': [float(10 + i) for i in range(100)],
        }
    ),
    'customers': pd.DataFrame(
        {
            'customer_id': [f'C{i}' for i in range(10)],
            'region_id': [f'R{i % 3}' for i in range(10)],
            'segment': [['gold', 'silver'][i % 2] for i in range(10)],
        }
    ),
    'regions': pd.DataFrame(
        {
            'region_id': ['R0', 'R1', 'R2'],
            'name': ['AMER', 'EMEA', 'APAC'],
        }
    ),
    'products': pd.DataFrame(
        {
            'product_id': [f'P{i}' for i in range(5)],
            'product_line': [['GPU', 'CPU', 'DPU'][i % 3] for i in range(5)],
        }
    ),
    'refunds': pd.DataFrame(
        {
            'refund_id': [f'F{i}' for i in range(20)],
            'order_line_id': [f'L{i * 5}' for i in range(20)],
            'refund_date': pd.date_range('2025-02-01', periods=20, freq='D'),
            'amount': [float(i) for i in range(20)],
        }
    ),
    'payments': pd.DataFrame(
        {
            'payment_id': [f'Y{i}' for i in range(10)],
            'order_id': [f'O{i}' for i in range(10)],
            'amount': [float(i) for i in range(10)],
        }
    ),
}

_COLUMN_TYPES = {
    'order_lines': {
        'line_id': 'string',
        'order_id': 'string',
        'customer_id': 'string',
        'product_id': 'string',
        'promo_id': 'string',
        'order_date': 'timestamp',
        'product_line': 'string',
        'status': 'string',
        'amount': 'double',
    },
    'customers': {
        'customer_id': 'string',
        'region_id': 'string',
        'segment': 'string',
    },
    'regions': {
        'region_id': 'string',
        'name': 'string',
    },
    'products': {
        'product_id': 'string',
        'product_line': 'string',
    },
    'refunds': {
        'refund_id': 'string',
        'order_line_id': 'string',
        'refund_date': 'timestamp',
        'amount': 'double',
    },
    'payments': {
        'payment_id': 'string',
        'order_id': 'string',
        'amount': 'double',
    },
}

# The relations `information_schema.tables` lists as other than `MANAGED`.
_TABLE_TYPES: dict[str, str] = {}

_EXPR_DATA = {
    'order_lines': pd.DataFrame(
        {
            'order_month': _DATA['order_lines']['order_date']
            .dt.to_period('M')
            .dt.to_timestamp(),
            'order_quarter': _DATA['order_lines']['order_date']
            .dt.to_period('Q')
            .dt.to_timestamp(),
        }
    ),
    'customers': pd.DataFrame(
        {
            'customer_segment': _DATA['customers']['segment'],
        }
    ),
    'regions': pd.DataFrame(
        {
            'region_name': _DATA['regions']['name'],
            'region_code': _DATA['regions']['name'].str[:2],
        }
    ),
}

_METRIC_VIEW_YAML = """
version: "1.1"
source: cat.sch.order_lines
filter: source.status = 'VALID'
joins:
  - name: customers
    source: cat.sch.customers
    on: source.customer_id = customers.customer_id
    joins:
      - name: regions
        source: regions
        on: customers.region_id = regions.region_id
  - name: products
    source: cat.sch.products
    using: [product_id]
  - name: refunds
    source: cat.sch.refunds
    on: refunds.order_line_id = line_id
    cardinality: one_to_many
  - name: promos
    source: SELECT * FROM cat.sch.promos
    on: source.promo_id = promos.promo_id
  - name: payments
    source: cat.sch.payments
    on: payments.order_id = source.missing_col
dimensions:
  - name: order_month
    expr: DATE_TRUNC('MONTH', source.order_date)
  - name: order_quarter
    expr: DATE_TRUNC('QUARTER', order_month)
  - name: product_line
    expr: source.product_line
  - name: product_id
    expr: UPPER(source.product_id)
  - name: customer_segment
    expr: customers.segment
  - name: region_name
    expr: regions.name
  - name: region_code
    expr: LEFT(customers.regions.name, 2)
  - name: promo_kind
    expr: promos.kind
measures:
  - name: total_amount
    expr: SUM(source.amount)
"""

_DESCRIBE_ROWS = [
    ('order_month', 'timestamp', None),
    ('order_quarter', 'timestamp', None),
    ('product_line', 'string', None),
    ('product_id', 'string', None),
    ('customer_segment', 'string', None),
    ('region_name', 'string', None),
    ('region_code', 'string', None),
    ('total_amount', 'double measure', None),
    ('', '', None),
    ('# Detailed Table Information', '', None),
    ('Catalog', _CATALOG, None),
    ('Database', _SCHEMA, None),
    ('Table', 'sales_mv', None),
    ('Type', 'METRIC_VIEW', None),
    ('View Text', _METRIC_VIEW_YAML, None),
    ('Language', 'YAML', None),
]


class _FakeCursor:
    def __init__(
        self, describe_rows: list, sql_log: list[str] | None = None
    ) -> None:
        self._describe_rows = describe_rows
        self._rows: list = []
        self._arrow: pa.Table | None = None
        self._sql_log = sql_log if sql_log is not None else []

    def __enter__(self) -> '_FakeCursor':
        return self

    def __exit__(self, *args: object) -> None:
        pass

    @staticmethod
    def _bound(sql: str, parameters: Any, column: str) -> str | None:
        r"""Returns the value ``column`` is compared against, mimicking
        `information_schema`, which holds canonical (lower-case) names and
        compares them as case-sensitive string literals.

        Names are bound rather than interpolated, so they are read off the
        parameter list in the order their placeholders appear.
        """
        markers = re.findall(
            r'lower\((?:\w+\.)?(\w+)\) = lower\(\?\)'
            r'|(?:\w+\.)?(\w+) = lower\(\?\)'
            r'|(?:\w+\.)?(\w+) = \?',
            sql,
        )
        names = [
            folded or sargable or plain for folded, sargable, plain in markers
        ]
        if column not in names or parameters is None:
            return None
        value = parameters[names.index(column)]
        return str(value).lower()

    def execute(self, sql: str, parameters: Any = None) -> None:
        self._rows = []
        self._arrow = None
        sql = sql.strip()
        self._sql_log.append(sql)

        if sql.upper().startswith('DESCRIBE TABLE EXTENDED'):
            self._rows = list(self._describe_rows)
            return

        if 'information_schema.columns' in sql:
            schema = self._bound(sql, parameters, 'table_schema')
            table = self._bound(sql, parameters, 'table_name')
            column_types = (
                _COLUMN_TYPES.get(table, {}) if schema == _SCHEMA else {}
            )
            self._rows = [
                (
                    name,
                    dtype,
                    'YES',
                    _CATALOG,
                    _SCHEMA,
                    table,
                    _TABLE_TYPES.get(table, 'MANAGED'),
                )
                for name, dtype in column_types.items()
            ]
            return

        if 'information_schema' in sql:
            return

        if sql.upper().startswith('SELECT COUNT(*)'):
            table = re.search(r'FROM `[^`]+`\.`[^`]+`\.`([^`]+)`', sql).group(1)
            self._rows = [(len(_DATA[table]),)]
            return

        if re.match(r'SELECT \d+ AS k, COUNT\(\*\) AS n', sql):
            self._rows = [
                (int(part.split()[1]), *self._key_count(part))
                for part in sql.split('\nUNION ALL\n')
            ]
            return

        if 'HAVING COUNT(*) > 1' in sql:
            key = self._key(sql, r'SELECT `((?:[^`]|``)+)`')
            values = self._frame(sql)[key].dropna()
            self._rows = [(value,) for value in values[values.duplicated()]]
            return

        table = re.search(r'FROM `([^`]+)`\.`([^`]+)`\.`([^`]+)`', sql)
        assert table is not None, f'Unexpected SQL: {sql}'
        assert table.group(1) == _CATALOG and table.group(2) == _SCHEMA
        df = _DATA[table.group(3)]
        expr_df = _EXPR_DATA.get(table.group(3))

        projection = sql.split(' FROM ')[0]
        if ' AS `' in projection:
            names = re.findall(r'AS `((?:[^`]|``)+)`', projection)
        else:
            names = re.findall(r'`((?:[^`]|``)+)`', projection)

        # Databricks matches a name to a column without regard to case:
        folded = {column.lower(): column for column in df.columns}
        columns = {}
        for name in names:
            name = name.replace('``', '`')
            if name.lower() in folded:
                columns[name] = df[folded[name.lower()]]
            else:
                assert expr_df is not None and name in expr_df.columns, (
                    f'Unexpected column {name!r} in SQL: {sql}'
                )
                columns[name] = expr_df[name]
        self._arrow = pa.Table.from_pandas(
            pd.DataFrame(columns), preserve_index=False
        )

    @staticmethod
    def _frame(sql: str) -> pd.DataFrame:
        table = re.search(
            r'FROM \(?(?:SELECT \* FROM )?`[^`]+`\.`[^`]+`\.`([^`]+)`', sql
        )
        assert table is not None, f'Unexpected SQL: {sql}'
        return _DATA[table.group(1)]

    @staticmethod
    def _key(sql: str, pattern: str) -> str:
        key = re.search(pattern, sql)
        assert key is not None, f'Unexpected SQL: {sql}'
        return key.group(1).replace('``', '`')

    def _key_count(self, sql: str) -> tuple[int, int, int]:
        r"""The rows, distinct keys and missing keys of one table, as the
        `UNION ALL` part counting a key reads them. The filter, if any, is
        not applied, as nothing here applies it.
        """
        key = self._key(sql, r'COUNT\(DISTINCT `((?:[^`]|``)+)`\)')
        frame = self._frame(sql)
        values = frame[{c.lower(): c for c in frame.columns}[key.lower()]]
        return len(values), int(values.nunique()), int(values.isna().sum())

    def fetchall(self) -> list:
        return self._rows

    def fetchone(self) -> tuple:
        return self._rows[0]

    def fetchall_arrow(self) -> 'pa.Table':
        assert self._arrow is not None
        return self._arrow


class _FakeConnection(Connection):
    def __init__(self, describe_rows: list = _DESCRIBE_ROWS) -> None:
        self._describe_rows = describe_rows
        self.closed = False
        self.sql_log: list[str] = []

    def cursor(self) -> _FakeCursor:  # type: ignore[override]
        return _FakeCursor(self._describe_rows, self.sql_log)

    def close(self) -> None:
        self.closed = True

    def __del__(self) -> None:
        pass


def test_from_databricks_metric_view() -> None:
    with pytest.warns(UserWarning) as caught:
        graph = Graph.from_databricks_metric_view(
            'sales_mv',
            connection=_FakeConnection(),
            verbose=False,
        )

    assert set(graph.tables) == {
        'order_lines',
        'customers',
        'regions',
        'products',
        'refunds',
    }

    fact = graph['order_lines']
    assert fact.source_name == 'cat.sch.order_lines'
    assert {column.name for column in fact.columns} == {
        'order_month',
        'order_quarter',
        'product_line',
        'line_id',
        'customer_id',
        'product_id',
        'order_date',
        'amount',
    }
    assert fact.primary_key is not None
    assert fact.primary_key.name == 'line_id'
    assert fact.time_column is not None
    assert fact.time_column.name == 'order_date'
    assert fact.time_column_choice is not None
    assert fact.time_column_choice.instead_of == (
        'order_month',
        'order_quarter',
    )
    assert fact['order_month'].expr is not None
    assert str(fact['order_month'].expr) == "DATE_TRUNC('MONTH', order_date)"
    assert str(fact['order_quarter'].expr) == (
        "DATE_TRUNC('QUARTER', (DATE_TRUNC('MONTH', order_date)))"
    )
    assert fact['product_line'].expr is None
    assert fact['product_id'].expr is None

    customers = graph['customers']
    assert customers.source_name == 'cat.sch.customers'
    assert {column.name for column in customers.columns} == {
        'customer_segment',
        'customer_id',
        'region_id',
    }
    assert customers.primary_key is not None
    assert customers.primary_key.name == 'customer_id'
    assert str(customers['customer_segment'].expr) == 'segment'

    regions = graph['regions']
    assert regions.source_name == 'cat.sch.regions'
    assert {column.name for column in regions.columns} == {
        'region_name',
        'region_code',
        'region_id',
    }
    assert regions.primary_key is not None
    assert regions.primary_key.name == 'region_id'
    assert str(regions['region_code'].expr) == 'LEFT(name, 2)'

    products = graph['products']
    assert products.primary_key is not None
    assert products.primary_key.name == 'product_id'
    assert {column.name for column in products.columns} == {'product_id'}

    refunds = graph['refunds']
    assert refunds.primary_key is None
    assert {column.name for column in refunds.columns} == {'order_line_id'}

    assert set(graph.edges) == {
        Edge('order_lines', 'customer_id', 'customers'),
        Edge('customers', 'region_id', 'regions'),
        Edge('order_lines', 'product_id', 'products'),
        Edge('refunds', 'order_line_id', 'order_lines'),
    }
    linked = {edge.src_table for edge in graph.edges} | {
        edge.dst_table for edge in graph.edges
    }
    assert set(graph.tables) == linked

    message = str(caught[0].message)
    assert 'filter' not in message
    assert fact.row_filter == "source.status = 'VALID'"
    assert graph['customers'].row_filter is None
    assert "Failed to add join 'promos'" in message
    assert "the dimension 'promo_kind'" in message
    assert "Failed to add join 'payments'" in message
    assert "Replaced the derived column 'product_id'" in message


def test_from_databricks_metric_view_not_a_metric_view() -> None:
    describe_rows = [
        (name, 'MANAGED' if name == 'Type' else dtype, comment)
        for name, dtype, comment in _DESCRIBE_ROWS
    ]
    with pytest.raises(ValueError, match='is not a metric view'):
        Graph.from_databricks_metric_view(
            'sales_mv',
            connection=_FakeConnection(describe_rows),
            verbose=False,
        )


def test_a_metric_view_that_is_not_there_is_a_missing_table() -> None:
    r"""A mistyped name failed as the warehouse's own error, which the
    client cannot tell from a warehouse that went away.
    """
    connection = _FailingConnection(
        _DESCRIBE_ROWS,
        'DESCRIBE',
        '[TABLE_OR_VIEW_NOT_FOUND] The table or view `cat`.`sch`.`salse_mv` '
        'cannot be found. SQLSTATE: 42P01; line 1 pos 24',
    )

    with pytest.raises(TableNotFoundError):
        _convert(connection)


def test_from_databricks_metric_view_invalid_name() -> None:
    with pytest.raises(ValueError, match='Invalid metric view name'):
        Graph.from_databricks_metric_view(
            'SELECT 1',
            connection=_FakeConnection(),
            verbose=False,
        )


def test_read_metric_view_definition() -> None:
    description = read_metric_view_definition(
        _FakeConnection(), '`cat`.`sch`.`sales_mv`'
    )

    assert description.definition == _METRIC_VIEW_YAML
    assert description.dtypes == {
        'order_month': 'timestamp',
        'order_quarter': 'timestamp',
        'product_line': 'string',
        'product_id': 'string',
        'customer_segment': 'string',
        'region_name': 'string',
        'region_code': 'string',
    }
    assert description.parameters == {}
    assert description.catalog == _CATALOG
    assert description.schema == _SCHEMA


def _parameterized(*parameters: str, width: int = 4, empty: Any = '') -> list:
    r"""The rows a warehouse (four wide, empty cells `''`) or a Spark session
    (three wide, empty cells `None`) describes a metric view with, holding
    the given parameters ahead of the details, as Databricks lists them.
    """
    columns = _DESCRIBE_ROWS[: _DESCRIBE_ROWS.index(('', '', None))]
    rows = [
        *((name, dtype, None, None) for name, dtype, _ in columns),
        ('#tag', 'string', None, None),
        ('', '', '', ''),
        ('# Parameters', '', '', ''),
        *((name, 'string', '', '') for name in parameters),
        ('', '', '', ''),
        ('# Detailed Table Information', '', '', ''),
        ('Catalog', _CATALOG, '', ''),
        ('Database', _SCHEMA, '', ''),
        ('Table', 'sales_mv', '', ''),
        ('Type', 'METRIC_VIEW', '', ''),
        ('View Text', _METRIC_VIEW_YAML, '', ''),
        ('Language', 'YAML', '', ''),
    ]
    return [
        tuple(empty if cell == '' else cell for cell in row[:width])
        for row in rows
    ]


@pytest.mark.parametrize(
    'width, empty', [(4, ''), (3, None)], ids=['warehouse', 'spark']
)
def test_a_parameter_named_like_a_detail_does_not_stand_in_for_it(
    width: int, empty: Any
) -> None:
    r"""Databricks lists the parameters before the details, so reading the
    first row of each name took a parameter named `Type` for the view's type.
    """
    rows = _parameterized(
        'Type', 'View Text', 'Catalog', 'Database', width=width, empty=empty
    )

    description = read_metric_view_definition(
        _FakeConnection(rows), '`cat`.`sch`.`sales_mv`'
    )

    assert description.definition == _METRIC_VIEW_YAML
    assert description.catalog == _CATALOG
    assert description.schema == _SCHEMA
    assert description.parameters == {
        'Type': 'string',
        'View Text': 'string',
        'Catalog': 'string',
        'Database': 'string',
    }
    # A column named `#...` holds a type, so it is no section header:
    assert description.dtypes['#tag'] == 'string'
    assert 'total_amount' not in description.dtypes


def _metric_view(definition: str, dtypes: dict[str, str]) -> list:
    return [(name, dtype, None) for name, dtype in dtypes.items()] + [
        ('', '', None),
        ('# Detailed Table Information', '', None),
        ('Catalog', _CATALOG, None),
        ('Database', _SCHEMA, None),
        ('Table', 'audit_mv', None),
        ('Type', 'METRIC_VIEW', None),
        ('View Text', definition, None),
        ('Language', 'YAML', None),
    ]


_REJECTED_JOIN_YAML = """
version: "1.1"
source: cat.sch.order_lines
joins:
  - name: ord
    source: cat.sch.refunds
    on: source.order_date = ord.order_line_id
dimensions:
  - name: product_line
    expr: source.product_line
measures:
  - name: total_amount
    expr: SUM(source.amount)
"""

_REJECTED_JOIN_DERIVED_KEY_YAML = """
version: "1.1"
source: cat.sch.order_lines
joins:
  - name: ord
    source: cat.sch.refunds
    on: source.product_id = ord.refund_date
dimensions:
  - name: product_id
    expr: UPPER(source.product_id)
measures:
  - name: total_amount
    expr: SUM(source.amount)
"""


def test_from_databricks_metric_view_rejected_join_does_not_leak_column() -> (
    None
):
    # Regression test:
    # a join that is reported as not added must not graft its key column onto
    # the fact table, where it can go on to become the graph's time column.
    describe_rows = _metric_view(
        _REJECTED_JOIN_YAML, {'product_line': 'string'}
    )

    with pytest.warns(UserWarning) as caught:
        graph = Graph.from_databricks_metric_view(
            'audit_mv',
            connection=_FakeConnection(describe_rows),
            verbose=False,
        )

    assert "Failed to add join 'ord'" in str(caught[0].message)

    assert set(graph.tables) == {'order_lines'}
    assert set(graph.edges) == set()

    fact = graph['order_lines']
    assert {column.name for column in fact.columns} == {
        'product_line',
        'amount',
    }
    assert fact.primary_key is None
    assert fact.time_column is None


def test_from_databricks_metric_view_rejected_join_keeps_expression() -> None:
    # Regression test:
    # a join that is reported as not added must not replace a declared
    # dimension with its physical source column.
    describe_rows = _metric_view(
        _REJECTED_JOIN_DERIVED_KEY_YAML, {'product_id': 'string'}
    )

    with pytest.warns(UserWarning) as caught:
        graph = Graph.from_databricks_metric_view(
            'audit_mv',
            connection=_FakeConnection(describe_rows),
            verbose=False,
        )

    message = str(caught[0].message)
    assert "Failed to add join 'ord'" in message
    assert 'Replaced the derived column' not in message

    assert set(graph.tables) == {'order_lines'}

    fact = graph['order_lines']
    assert {column.name for column in fact.columns} == {'product_id', 'amount'}
    assert str(fact['product_id'].expr) == 'UPPER(product_id)'


_HOSTILE_EXPR_YAML = """
version: "1.1"
source: cat.sch.order_lines
dimensions:
  - name: product_line
    expr: source.product_line
  - name: leaked
    expr: (SELECT MAX(amount) FROM cat.sch.payments)
measures:
  - name: total_amount
    expr: SUM(source.amount)
"""


def test_from_databricks_metric_view_drops_a_hostile_expression() -> None:
    # Regression test: a dimension expression comes from whoever authored
    # the view but runs under the caller's warehouse role, so a sub-query that
    # reads a table outside the graph must never be lifted into a `ColumnSpec`.
    describe_rows = _metric_view(
        _HOSTILE_EXPR_YAML,
        {
            'product_line': 'string',
            'leaked': 'double',
        },
    )

    with pytest.warns(UserWarning) as caught:
        graph = Graph.from_databricks_metric_view(
            'audit_mv',
            connection=_FakeConnection(describe_rows),
            verbose=False,
        )

    assert "Failed to add column 'leaked'" in str(caught[0].message)

    fact = graph['order_lines']
    assert not fact.has_column('leaked')
    assert {column.name for column in fact.columns} == {
        'product_line',
        'amount',
    }


_UDF_EXPR_YAML = """
version: "1.1"
source: cat.sch.order_lines
dimensions:
  - name: product_line
    expr: UPPER(source.product_line)
  - name: masked
    expr: my_udf(source.product_line)
measures:
  - name: total_amount
    expr: SUM(source.amount)
"""


def _udf_metric_view(**kwargs: Any) -> tuple[Graph, list[str]]:
    connection = _FakeConnection(
        _metric_view(
            _UDF_EXPR_YAML, {'product_line': 'string', 'masked': 'string'}
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ViewConversionWarning)
        graph = Graph.from_databricks_metric_view(
            'audit_mv', connection=connection, verbose=False, **kwargs
        )
    return graph, connection.sql_log


def test_a_dimension_calling_a_udf_is_dropped_before_it_runs() -> None:
    graph, sql_log = _udf_metric_view()

    fact = graph['order_lines']
    assert not fact.has_column('masked')
    assert str(fact['product_line'].expr) == 'UPPER(product_line)'
    assert not any('my_udf' in sql for sql in sql_log)
    (message,) = [m for m in graph.conversion_messages if "'masked'" in m]
    assert "calls 'my_udf'" in message
    assert 'trust_expressions=True' in message


def test_a_field_reads_through_to_what_an_earlier_field_stands_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""The guard reads a field through the one it names, so a UDF behind it
    never runs.
    """
    yaml = """
version: "1.1"
source: cat.sch.order_lines
fields:
  - name: shouted
    expr: UPPER(source.product_line)
  - name: masked
    expr: my_udf(source.product_line)
  - name: via
    expr: masked
measures:
  - name: total_amount
    expr: SUM(source.amount)
"""
    expr_df = _EXPR_DATA['order_lines'].assign(shouted='A')
    monkeypatch.setitem(_EXPR_DATA, 'order_lines', expr_df)
    connection = _FakeConnection(
        _metric_view(
            yaml,
            {
                'shouted': 'string',
                'masked': 'string',
                'via': 'string',
            },
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ViewConversionWarning)
        graph = Graph.from_databricks_metric_view(
            'audit_mv', connection=connection, verbose=False
        )

    fact = graph['order_lines']
    assert str(fact['shouted'].expr) == 'UPPER(product_line)'
    assert not fact.has_column('masked')
    assert not fact.has_column('via')
    assert not any('my_udf' in sql for sql in connection.sql_log)
    (message,) = [m for m in graph.conversion_messages if "'via'" in m]
    assert "calls 'my_udf'" in message


def test_a_trusted_metric_view_keeps_its_udf_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expr_df = _EXPR_DATA['order_lines'].assign(masked='x')
    monkeypatch.setitem(_EXPR_DATA, 'order_lines', expr_df)

    graph, sql_log = _udf_metric_view(trust_expressions=True)

    assert str(graph['order_lines']['masked'].expr) == 'my_udf(product_line)'
    assert any('my_udf' in sql for sql in sql_log)
    assert _said(graph) == ()


def test_a_trusted_metric_view_still_refuses_a_sub_query() -> None:
    describe_rows = _metric_view(
        _HOSTILE_EXPR_YAML, {'product_line': 'string', 'leaked': 'double'}
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ViewConversionWarning)
        graph = Graph.from_databricks_metric_view(
            'audit_mv',
            connection=_FakeConnection(describe_rows),
            verbose=False,
            trust_expressions=True,
        )

    assert not graph['order_lines'].has_column('leaked')


def test_from_databricks_case_insensitive_identifiers() -> None:
    # Regression test:
    # `information_schema` compares string literals case-sensitively, so
    # identifiers must be canonicalized rather than compared verbatim.
    graph = Graph.from_databricks(
        connection=_FakeConnection(),
        catalog=_CATALOG.upper(),
        schema=_SCHEMA.upper(),
        tables=[
            dict(
                name='CUSTOMERS',
                columns=['customer_id', 'segment'],
                primary_key='customer_id',
            ),
            dict(
                name='ORDER_LINES',
                columns=['line_id', 'customer_id'],
                primary_key='line_id',
            ),
        ],
        edges=[('ORDER_LINES', 'customer_id', 'CUSTOMERS')],
        infer_metadata=False,
        verbose=False,
    )

    assert graph['CUSTOMERS'].source_name == 'cat.sch.customers'
    assert graph['ORDER_LINES'].source_name == 'cat.sch.order_lines'
    assert graph.validate() is graph


def test_from_databricks_unknown_table() -> None:
    with pytest.raises(ValueError, match='does not exist'):
        Graph.from_databricks(
            connection=_FakeConnection(),
            catalog=_CATALOG,
            schema=_SCHEMA,
            tables=['does_not_exist'],
            infer_metadata=False,
            verbose=False,
        )


def test_from_databricks_metric_view_tracks_internal_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression test: a
    # connection the client opened is a connection the client owns and closes.
    connection = _FakeConnection()
    monkeypatch.setattr(
        'sdm_connectors.backends.databricks.connect',
        lambda **kwargs: connection,
    )

    with pytest.warns(UserWarning):
        graph = Graph.from_databricks_metric_view(
            'sales_mv',
            connection=dict(server_hostname='localhost'),
            verbose=False,
        )
    assert graph._connection is connection

    with pytest.warns(UserWarning):
        graph = Graph.from_databricks_metric_view(
            'sales_mv',
            connection=connection,
            verbose=False,
        )
    assert graph._connection is None


def test_from_databricks_closes_internal_connection_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A connection the client opened must not leak in case the graph is never
    # constructed and can therefore never take ownership of it:
    connection = _FakeConnection()
    monkeypatch.setattr(
        'sdm_connectors.backends.databricks.connect',
        lambda **kwargs: connection,
    )

    with pytest.raises(ValueError, match='does not exist'):
        Graph.from_databricks(
            connection=dict(server_hostname='localhost'),
            catalog=_CATALOG,
            schema=_SCHEMA,
            tables=['does_not_exist'],
            infer_metadata=False,
            verbose=False,
        )
    assert connection.closed


def test_from_databricks_keeps_external_connection_open_on_error() -> None:
    connection = _FakeConnection()

    with pytest.raises(ValueError, match='does not exist'):
        Graph.from_databricks(
            connection=connection,
            catalog=_CATALOG,
            schema=_SCHEMA,
            tables=['does_not_exist'],
            infer_metadata=False,
            verbose=False,
        )
    assert not connection.closed


def test_from_databricks_metric_view_closes_internal_connection_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _FakeConnection()
    monkeypatch.setattr(
        'sdm_connectors.backends.databricks.connect',
        lambda **kwargs: connection,
    )

    with pytest.raises(ValueError, match='Invalid metric view name'):
        Graph.from_databricks_metric_view(
            'SELECT * FROM sales_mv',
            connection=dict(server_hostname='localhost'),
            verbose=False,
        )
    assert connection.closed


@pytest.fixture(scope='session')
def metric_view_graph(connection: 'Connection') -> Graph:
    # A live view that converts only partly is the normal case, not a failure,
    # and `error::UserWarning` would turn it into one, failing these tests
    # for a reason that has nothing to do with what they assert. Suppressed
    # here rather than in `filterwarnings`, which cannot name a warning class
    # without importing it before the session starts.
    import warnings

    from sdm_relational_engine.rfm import ViewConversionWarning

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ViewConversionWarning)
        return Graph.from_databricks_metric_view(
            'nvidia_erp_mv',
            connection=connection,
            verbose=False,
        )


def test_live_metric_view_graph(metric_view_graph: Graph) -> None:
    graph = metric_view_graph

    assert set(graph.tables) == {
        'order_lines',
        'customers',
        'products',
        'orders',
    }

    fact = graph['order_lines']
    assert fact.primary_key is None
    # `Order Month` truncates a source column, which is taken instead:
    assert fact.time_column is not None
    assert fact.time_column_choice is not None
    assert fact.time_column_choice.instead_of == ('Order Month',)
    # Beside the columns its measures aggregate:
    assert {column.name for column in fact.columns} >= {
        'Order Month',
        'Product Family',
        'Product Line',
        'customer_id',
        'product_id',
        'order_id',
        fact.time_column.name,
    }
    assert fact[fact.time_column.name].expr is None

    assert graph['customers'].primary_key is not None
    assert graph['customers'].primary_key.name == 'customer_id'
    assert graph['products'].primary_key is not None
    assert graph['products'].primary_key.name == 'product_id'
    assert graph['orders'].primary_key is not None
    assert graph['orders'].primary_key.name == 'order_id'
    assert graph['orders'].source_name.endswith('.sales_orders')

    assert set(graph.edges) == {
        Edge('order_lines', 'customer_id', 'customers'),
        Edge('order_lines', 'product_id', 'products'),
        Edge('order_lines', 'order_id', 'orders'),
    }


def test_live_metric_view_qualified_name(
    connection: 'Connection', catalog: str, schema: str
) -> None:
    graph = Graph.from_databricks_metric_view(
        f'{catalog}.{schema}.nvidia_erp_mv',
        connection=connection,
        verbose=False,
    )
    assert set(graph.tables) == {
        'order_lines',
        'customers',
        'products',
        'orders',
    }


@pytest.mark.live_warehouse
def test_live_filtered_metric_view_counts_the_rows_it_holds(
    connection: 'Connection',
) -> None:
    r"""Names a persistent metric view with a source-only ``filter`` and a
    ``COUNT(1)`` measure through ``DATABRICKS_FILTERED_METRIC_VIEW``. The
    graph's fact table holds as many rows as the view counts itself.
    """
    import os

    import yaml

    name = os.getenv('DATABRICKS_FILTERED_METRIC_VIEW')
    if not name:
        pytest.skip('DATABRICKS_FILTERED_METRIC_VIEW names no metric view')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ViewConversionWarning)
        graph = Graph.from_databricks_metric_view(
            name, connection=connection, verbose=False
        )

    from sdm_connectors import quote_ident

    parts = parse_table_reference(name)
    assert parts is not None
    quoted = '.'.join(quote_ident(part, '`') for part in parts)
    definition, *_ = read_metric_view_definition(connection, quoted)
    cfg = yaml.safe_load(definition)
    counts = [
        measure['name']
        for measure in cfg.get('measures') or []
        if re.fullmatch(r'COUNT\(\s*(?:1|\*)\s*\)', measure['expr'], re.I)
    ]
    if not cfg.get('filter') or not counts:
        pytest.skip(f'{name} has no filter or no COUNT(1) measure')

    fact = next(
        table
        for table in graph.tables.values()
        if getattr(table, 'row_filter', None) is not None
    )
    with connection.cursor() as cursor:
        cursor.execute(f'SELECT MEASURE(`{counts[0]}`) FROM {quoted}')
        (expected,) = cursor.fetchone()

    assert fact._get_num_rows() == expected


def test_conversion_messages_survive_a_warnings_as_errors_policy() -> None:
    """Partial conversion is the normal outcome for a metric view, and the
    diagnostics used to exist only as one aggregated ``UserWarning``: under
    ``-W error`` the constructor raised instead of returning, and the fully
    built graph was unrecoverable. They are now on the graph as well, and the
    warning has its own category so a project can silence just this one.
    """
    import warnings

    from sdm_relational_engine.rfm import ViewConversionWarning

    with warnings.catch_warnings():
        warnings.simplefilter('error', ViewConversionWarning)
        with pytest.raises(ViewConversionWarning):
            Graph.from_databricks_metric_view(
                'sales_mv', connection=_FakeConnection(), verbose=False
            )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ViewConversionWarning)
        graph = Graph.from_databricks_metric_view(
            'sales_mv', connection=_FakeConnection(), verbose=False
        )

    assert issubclass(ViewConversionWarning, UserWarning)
    assert len(graph.conversion_messages) > 0
    assert any(
        "Failed to add join 'promos'" in message
        for message in graph.conversion_messages
    )


def test_conversion_messages_are_empty_for_a_plain_graph() -> None:
    graph = Graph.from_data(
        {'users': pd.DataFrame({'user_id': [1, 2]})}, verbose=False
    )

    assert graph.conversion_messages == ()


def test_from_databricks_rejects_a_schema_with_no_tables() -> None:
    """A mistyped schema name discovers nothing, and used to yield a
    valid-looking empty graph that fails much later without naming the schema.
    """
    with pytest.raises(ValueError, match='No tables found'):
        Graph.from_databricks(
            _FakeConnection(),
            catalog=_CATALOG,
            schema='no_such_schema',
            verbose=False,
        )


def test_escalated_conversion_warning_closes_an_owned_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Under warnings-as-errors the warning leaves the constructor, so the
    caller never receives the graph that would have owned this connection.
    Same contract as every other failure path here.

    ``Graph.__del__`` does close it eventually, so this is a matter of *when*:
    a caller that holds the exception, which any ``except ... as error`` does,
    keeps the constructor's frame, and therefore the unreachable graph,
    alive with it. The connection then stays open for as long as the exception
    is held rather than being released where it was opened.
    """
    import warnings

    from sdm_relational_engine.rfm import ViewConversionWarning

    connection = _FakeConnection()
    monkeypatch.setattr(
        'sdm_connectors.backends.databricks.connect',
        lambda **kwargs: connection,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error', ViewConversionWarning)
        with pytest.raises(ViewConversionWarning) as raised:
            Graph.from_databricks_metric_view(
                'sales_mv',
                connection=dict(server_hostname='localhost'),
                verbose=False,
            )

        # `raised` pins the traceback, so nothing here has been collected.
        assert raised.traceback is not None
        assert connection.closed


def test_escalated_conversion_warning_keeps_a_borrowed_connection_open() -> (
    None
):
    import warnings

    from sdm_relational_engine.rfm import ViewConversionWarning

    connection = _FakeConnection()

    with warnings.catch_warnings():
        warnings.simplefilter('error', ViewConversionWarning)
        with pytest.raises(ViewConversionWarning):
            Graph.from_databricks_metric_view(
                'sales_mv', connection=connection, verbose=False
            )

    assert not connection.closed


def test_conversion_warning_is_attributed_to_the_caller() -> None:
    r"""The message is about the caller's view, so the traceback has to point
    at their call and not at this module's own `warnings.warn` line.
    """
    import warnings

    from sdm_relational_engine.rfm import ViewConversionWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        Graph.from_databricks_metric_view(
            'sales_mv', connection=_FakeConnection(), verbose=False
        )

    escalated = [
        w for w in caught if issubclass(w.category, ViewConversionWarning)
    ]
    assert len(escalated) == 1
    assert escalated[0].filename == __file__


def test_information_schema_lookups_stay_pushdown_friendly() -> None:
    """Folding the *column* is case-insensitive but not sargable: Unity Catalog
    cannot push the predicate down, so a metadata point look-up degrades into a
    scan of every column of every table in the catalog, measured live at 31s
    per table against 0.5s, growing with catalog size. Fold the literal
    instead; identifiers are stored lower-cased.
    """
    connection = _FakeConnection()
    Graph.from_databricks(
        connection=connection,
        catalog=_CATALOG,
        schema=_SCHEMA,
        tables=[
            dict(
                name='customers',
                columns=['customer_id', 'segment'],
                primary_key='customer_id',
            )
        ],
        infer_metadata=False,
        verbose=False,
    )

    lookups = [sql for sql in connection.sql_log if 'information_schema' in sql]
    assert lookups

    for sql in lookups:
        assert (
            re.search(r'lower\(\s*(?:\w+\.)?table_(?:schema|name)\s*\)', sql)
            is None
        ), f'unsargable predicate in SQL: {sql}'

    columns = next(
        sql for sql in lookups if 'information_schema.columns' in sql
    )
    assert 'table_schema = lower(?)' in columns
    assert 'table_name = lower(?)' in columns
    assert f"'{_SCHEMA}'" not in columns
    assert "'customers'" not in columns


def test_metric_view_drops_a_type_mismatched_relationship(monkeypatch) -> None:
    """A view constructor converts partially by contract. A relationship whose
    keys have incompatible data types is one more unconvertible element: it is
    dropped and reported, not raised, so the tables and the well-formed edges
    of the view survive.
    """
    from sdm_relational_engine.rfm import ViewConversionWarning

    products = _DATA['products'].copy()
    products['product_id'] = range(len(products))
    monkeypatch.setitem(_DATA, 'products', products)
    monkeypatch.setitem(
        _COLUMN_TYPES,
        'products',
        {**_COLUMN_TYPES['products'], 'product_id': 'bigint'},
    )

    with pytest.warns(ViewConversionWarning):
        graph = Graph.from_databricks_metric_view(
            'sales_mv',
            connection=_FakeConnection(),
            verbose=False,
        )

    assert set(graph.tables) == {
        'order_lines',
        'customers',
        'regions',
        'products',
        'refunds',
    }
    assert Edge('order_lines', 'product_id', 'products') not in graph.edges
    assert graph.edges

    dropped = [
        msg
        for msg in graph.conversion_messages
        if 'incompatible data types' in msg
    ]
    assert len(dropped) == 1
    assert "'order_lines'" in dropped[0] and "'products'" in dropped[0]
    assert graph.validate() is graph


class _RaisingConnection(_FakeConnection):
    r"""A connection whose driver fails the way a real one does."""

    def cursor(self) -> Any:
        raise _ServerOperationError('METRIC_VIEW_MISSING_MEASURE_FUNCTION')


class _ServerOperationError(Exception):
    r"""Stands in for ``databricks.sql.exc.ServerOperationError``.

    Defined here rather than imported so the test states what matters: the
    exception belongs to the driver's hierarchy and to nothing this client owns.
    """


def test_from_databricks_wraps_a_driver_exception() -> None:
    with pytest.raises(GraphConstructionError) as caught:
        Graph.from_databricks_metric_view(
            'sales_mv',
            connection=_RaisingConnection(),
            verbose=False,
        )

    assert isinstance(caught.value, KumoRelationalError)
    assert isinstance(caught.value.__cause__, _ServerOperationError)
    assert 'Databricks' in str(caught.value)


def test_the_filter_wraps_every_read_of_the_fact_table() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ViewConversionWarning)
        graph = Graph.from_databricks_metric_view(
            'sales_mv', connection=_FakeConnection(), verbose=False
        )

    fact = graph['order_lines']
    filtered = (
        '(SELECT * FROM `cat`.`sch`.`order_lines` AS source '
        "WHERE (source.status = 'VALID'))"
    )
    assert fact._quoted_source_name == filtered
    assert fact.source_name == 'cat.sch.order_lines'
    assert graph['customers']._quoted_source_name == '`cat`.`sch`.`customers`'

    # A shallow copy, as every gathering takes, keeps the filter:
    copied = fact.copy('lines')
    assert copied.row_filter == fact.row_filter
    assert copied._quoted_source_name == filtered

    # Reading other rows is another graph to anything that caches by name:
    from sdm_relational_engine.rfm.graph_fingerprint import graph_fingerprint

    name = graph_fingerprint(graph)
    fact._row_filter = None
    assert graph_fingerprint(graph) != name


def test_a_row_filter_is_set_only_before_a_row_is_read() -> None:
    from sdm_relational_engine.rfm.backend.databricks import DatabricksTable

    table = DatabricksTable(
        _FakeConnection(),
        name='order_lines',
        catalog='cat',
        schema='sch',
        columns=[],
        primary_key=None,
    )
    table.row_filter = "status = 'VALID'"
    assert "WHERE (status = 'VALID'))" in table._quoted_source_name

    table.add_columns(['status'])
    with pytest.raises(ValueError, match='already been read'):
        table.row_filter = None
    assert table.row_filter == "status = 'VALID'"


def test_a_column_named_like_a_session_word_is_named_through_the_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Databricks reads `user` as the session user once the column is gone,
    so every read names the columns it was given through the table, where a
    renamed or dropped one fails the statement instead.
    """
    from sdm_relational_engine.rfm.backend.databricks import DatabricksTable

    monkeypatch.setitem(
        _COLUMN_TYPES, 'logins', {'login_id': 'string', 'USER': 'string'}
    )
    table = DatabricksTable(
        _FakeConnection(),
        name='logins',
        catalog='cat',
        schema='sch',
        columns=[],
        primary_key=None,
    )
    # Nothing was read yet, so nothing could rest on a column:
    assert table._quoted_source_name == '`cat`.`sch`.`logins`'

    assert 'USER' in table._source_column_dict
    assert table._quoted_source_name == (
        '(SELECT source.`login_id`, source.`USER` '
        'FROM `cat`.`sch`.`logins` AS source)'
    )
    table.row_filter = "user = 'a'"
    assert table._quoted_source_name == (
        '(SELECT source.`login_id`, source.`USER` '
        "FROM `cat`.`sch`.`logins` AS source WHERE (user = 'a'))"
    )

    # Other tables are read as before:
    regions = DatabricksTable(
        _FakeConnection(), name='regions', catalog='cat', schema='sch'
    )
    assert 'name' in regions._source_column_dict
    assert regions._quoted_source_name == '`cat`.`sch`.`regions`'


def test_a_join_filter_refuses_the_metric_view() -> None:
    definition = _METRIC_VIEW_YAML.replace(
        "filter: source.status = 'VALID'", "filter: customers.segment = 'gold'"
    )
    describe_rows = [
        ('View Text', definition, None) if row[0] == 'View Text' else row
        for row in _DESCRIBE_ROWS
    ]
    connection = _FakeConnection(describe_rows)

    with pytest.raises(UnsupportedFilterError, match='regular view'):
        Graph.from_databricks_metric_view(
            'sales_mv', connection=connection, verbose=False
        )
    assert not any('order_lines' in sql for sql in connection.sql_log)


_KEYWORD_YAML = """
version: "1.1"
source: cat.sch.order_lines
filter: {filter}
joins:
  - name: customers
    source: cat.sch.customers
    on: source.customer_id = customers.customer_id
dimensions:
  - name: product_line
    expr: source.product_line
  - name: is_gold
    expr: customers.segment IN ('gold')
  - name: misplaced
    expr: segment IN ('gold')
"""


def _keyword_view(row_filter: str) -> list:
    return _metric_view(
        _KEYWORD_YAML.format(filter=json.dumps(row_filter)),
        {
            'product_line': 'string',
            'is_gold': 'boolean',
            'misplaced': 'boolean',
        },
    )


def test_a_dimension_is_checked_against_the_columns_of_its_own_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""`segment` is a column of the joined table only, so `IN (` after it is
    grammar there and a call on the fact table.
    """
    expr_df = _EXPR_DATA['customers'].assign(is_gold=[True, False] * 5)
    monkeypatch.setitem(_EXPR_DATA, 'customers', expr_df)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ViewConversionWarning)
        graph = Graph.from_databricks_metric_view(
            'audit_mv',
            connection=_FakeConnection(_keyword_view("status = 'VALID'")),
            verbose=False,
        )

    assert str(graph['customers']['is_gold'].expr) == "segment IN ('gold')"
    assert not graph['order_lines'].has_column('misplaced')
    (message,) = graph.conversion_messages
    assert "'misplaced'" in message and "calls 'IN'" in message
    assert graph['order_lines'].row_filter == "status = 'VALID'"


@pytest.mark.parametrize('internal', [True, False])
def test_a_filter_is_checked_against_the_columns_of_the_source_table(
    monkeypatch: pytest.MonkeyPatch, internal: bool
) -> None:
    r"""`segment` belongs to the joined table, so on the fact table the
    filter's `IN (` would call a function. It is refused before a row is read.
    """
    connection = _FakeConnection(_keyword_view("segment IN ('gold')"))
    monkeypatch.setattr(
        'sdm_connectors.backends.databricks.connect',
        lambda **kwargs: connection,
    )

    with pytest.raises(UnsupportedFilterError, match="calls 'IN'"):
        Graph.from_databricks_metric_view(
            'audit_mv',
            connection=dict(server_hostname='x') if internal else connection,
            verbose=False,
        )

    assert connection.closed is internal
    assert connection.sql_log
    assert all(
        sql.startswith('DESCRIBE') or 'information_schema' in sql
        for sql in connection.sql_log
    ), connection.sql_log


def test_a_trusted_metric_view_keeps_a_filter_calling_a_udf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for table, column in (
        ('customers', 'is_gold'),
        ('order_lines', 'misplaced'),
    ):
        expr_df = _EXPR_DATA[table].assign(**{column: False})
        monkeypatch.setitem(_EXPR_DATA, table, expr_df)
    row_filter = "my_udf(status) = 'VALID'"
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ViewConversionWarning)
        graph = Graph.from_databricks_metric_view(
            'audit_mv',
            connection=_FakeConnection(_keyword_view(row_filter)),
            verbose=False,
            trust_expressions=True,
        )

    assert graph['order_lines'].row_filter == row_filter
    with pytest.raises(UnsupportedFilterError, match="calls 'my_udf'"):
        Graph.from_databricks_metric_view(
            'audit_mv',
            connection=_FakeConnection(_keyword_view(row_filter)),
            verbose=False,
        )


def test_a_string_literal_is_data_to_a_dimension() -> None:
    r"""A literal holding `source.`, a join's name or a field's name is data,
    so it neither moves the dimension to a join nor gets rewritten.
    """
    spec = parse_metric_view("""
source: cat.sch.orders
joins:
  - name: customers
    source: cat.sch.customers
    on: source.customer_id = customers.customer_id
dimensions:
  - name: gold
    expr: customers.segment = 'gold'
  - name: tag
    expr: CONCAT('source.', source.channel, "customers.x", 'gold')
  - name: label
    expr: CASE WHEN gold THEN 'gold' END
""")

    assert spec.messages == []
    assert [(c.name, c.alias, c.expr) for c in spec.columns] == [
        ('gold', 'customers', "segment = 'gold'"),
        (
            'tag',
            'source',
            "CONCAT('source.', channel, \"customers.x\", 'gold')",
        ),
        ('label', 'customers', "CASE WHEN (segment = 'gold') THEN 'gold' END"),
    ]
    assert not references_alias("'customers.x'", 'customers')
    assert strip_alias_qualifier("source.a || 'source.b'", 'source') == (
        "a || 'source.b'"
    )


_LAST_LOGIN_YAML = """
version: "1.1"
source: cat.sch.order_lines
joins:
  - name: customers
    source: cat.sch.customers
    on: source.customer_id = customers.customer_id
dimensions:
  - name: product_line
    expr: source.product_line
  - name: last_login
    expr: customers.last_login
measures:
  - name: total_amount
    expr: SUM(source.amount)
"""


def test_from_databricks_metric_view_refuses_an_edit_stamp_on_a_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The view joins `customers` to its source, so a customer is an entity
    edited in place and its `last_login` is no customer's time.
    """
    logins = pd.date_range('2025-01-01', periods=10, freq='D')
    monkeypatch.setitem(
        _DATA, 'customers', _DATA['customers'].assign(last_login=logins)
    )
    monkeypatch.setitem(
        _COLUMN_TYPES,
        'customers',
        {**_COLUMN_TYPES['customers'], 'last_login': 'timestamp'},
    )
    describe_rows = _metric_view(
        _LAST_LOGIN_YAML,
        {
            'product_line': 'string',
            'last_login': 'timestamp',
            'total_amount': 'double measure',
        },
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        graph = Graph.from_databricks_metric_view(
            'audit_mv',
            connection=_FakeConnection(describe_rows),
            verbose=False,
        )

    assert any(edge.dst_table == 'customers' for edge in graph.edges)
    choice = graph['customers'].time_column_choice
    assert choice is not None and choice.fallback == 'last_login'
    assert graph['customers'].time_column is None


_JOINED_YAML = """
version: "1.1"
source: cat.sch.order_lines
joins:
  - name: customers
    source: cat.sch.{customers}
    on: source.customer_id = customers.customer_id
  - name: products
    source: cat.sch.products
    on: source.product_id = products.product_id
dimensions:
  - name: product_line
    expr: source.product_line
  - name: customer_segment
    expr: customers.segment
{extra}
"""


def _joined_view(customers: str = 'customers', extra: str = '') -> list:
    return _metric_view(
        _JOINED_YAML.format(customers=customers, extra=extra),
        {'product_line': 'string', 'customer_segment': 'string'},
    )


class _SparkError(Exception):
    r"""Stands in for a Spark session's error, which answers for its state
    and class rather than writing them into its message.
    """

    def getSqlState(self) -> str:  # noqa: N802 - Spark's name
        return '42501'

    def getCondition(self) -> str:  # noqa: N802 - Spark's name
        return 'INSUFFICIENT_PERMISSIONS'


# Stands in for `pyspark.errors.PySparkRuntimeError`, a driver's error.
_PySparkRuntimeError = type(
    'PySparkRuntimeError',
    (RuntimeError,),
    {'__module__': 'pyspark.errors.exceptions.base'},
)


class _FailingConnection(_FakeConnection):
    r"""A warehouse that fails every statement holding ``marker``, in its SQL
    or as a parameter, the way a driver does, and answers the rest.
    """

    def __init__(
        self,
        describe_rows: list,
        marker: str,
        error: str | Exception = 'Failed',
    ) -> None:
        super().__init__(describe_rows)
        self._marker = marker
        self._error = (
            _ServerOperationError(error) if isinstance(error, str) else error
        )

    def cursor(self) -> Any:
        cursor = super().cursor()
        execute = cursor.execute

        def _execute(sql: str, parameters: Any = None) -> None:
            if self._marker in sql or self._marker in (parameters or ()):
                raise self._error
            execute(sql, parameters)

        cursor.execute = _execute  # type: ignore[method-assign]
        return cursor


def _convert(connection: Any) -> Graph:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ViewConversionWarning)
        return Graph.from_databricks_metric_view(
            'audit_mv', connection=connection, verbose=False
        )


@pytest.mark.parametrize(
    'unusable, reason',
    [
        ('missing', 'does not exist'),
        ('empty', 'is empty'),
        ('metric-view', 'is a metric view'),
        ('permission-denied', 'may not read'),
        ('spark-permission-denied', 'may not read'),
    ],
)
def test_a_join_whose_table_cannot_be_used_is_left_out(
    unusable: str, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""With the join nested in it and the dimensions on both, which the
    message names, since a question may have been asked of them.
    """
    connection = _FakeConnection()
    if unusable == 'missing':
        monkeypatch.delitem(_COLUMN_TYPES, 'customers')
    elif unusable == 'empty':
        monkeypatch.setitem(_DATA, 'customers', _DATA['customers'].iloc[:0])
    elif unusable == 'metric-view':
        monkeypatch.setitem(_TABLE_TYPES, 'customers', 'METRIC_VIEW')
    elif unusable == 'permission-denied':
        connection = _FailingConnection(
            _DESCRIBE_ROWS,
            '`sch`.`customers`',
            '[INSUFFICIENT_PERMISSIONS] Insufficient privileges: User does '
            'not have SELECT on Table. SQLSTATE: 42501',
        )
    else:
        connection = _FailingConnection(
            _DESCRIBE_ROWS, '`sch`.`customers`', _SparkError('denied')
        )

    graph = _convert(connection)

    assert set(graph.tables) == {'order_lines', 'products', 'refunds'}
    assert not any('customers' in edge for edge in graph.edges)
    said = graph.conversion_messages
    (customers,) = [m for m in said if "'customer_segment'" in m]
    (regions,) = [m for m in said if "'region_name'" in m]
    assert reason in customers and customers.count("'customers'") == 1
    assert "'customers'" in regions and "'region_code'" in regions
    # The join nested in it is never read, so cannot fail the conversion.
    assert not any('`regions`' in sql for sql in connection.sql_log)


@pytest.mark.parametrize(
    'marker, error',
    [
        ('customers', 'Failed'),
        ('TRY_CAST', 'Failed'),
        # The state of the failure is the last one, not one its message
        # quotes from a nested error or a value.
        (
            '`sch`.`customers`',
            '[INTERNAL_ERROR] Query could not be scheduled: upstream said '
            '[TABLE_OR_VIEW_NOT_FOUND] SQLSTATE: 42P01. SQLSTATE: XX000',
        ),
        (
            '`sch`.`customers`',
            "[CAST_INVALID_INPUT] The value 'SQLSTATE: 42P01' of the type "
            '"STRING" cannot be cast to "INT". SQLSTATE: 22018',
        ),
        # A Spark session's errors are RuntimeError and ValueError.
        ('TRY_CAST', _PySparkRuntimeError('[SESSION_NOT_ACTIVE] gone')),
    ],
    ids=['columns', 'expression', 'nested-state', 'quoted-state', 'spark'],
)
def test_a_warehouse_failure_while_reading_a_join_fails_the_conversion(
    marker: str, error: str | Exception
) -> None:
    r"""A failure that says nothing of the table or the expression may be
    the warehouse's own, so leaving out what it was reading would build a
    smaller graph that looks whole.
    """
    extra = (
        '  - name: segment_code\n    expr: TRY_CAST(customers.segment AS INT)'
    )
    connection = _FailingConnection(_joined_view(extra=extra), marker, error)

    with pytest.raises(GraphConstructionError) as caught:
        _convert(connection)

    assert isinstance(
        caught.value.__cause__, _ServerOperationError | _PySparkRuntimeError
    )


@pytest.mark.parametrize('refused', ['expression', 'type'])
def test_a_column_the_warehouse_refuses_is_left_out_alone(
    refused: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A table's expressions are sampled in one query, so the one refused
    used to fail the others, and the conversion, with it.
    """
    if refused == 'expression':
        name = 'segment_code'
        extra = (
            f'  - name: {name}\n    expr: TRY_CAST(customers.segment AS INT)'
        )
        connection = _FailingConnection(
            _joined_view(extra=extra),
            'TRY_CAST',
            '[CAST_INVALID_INPUT] The value \'gold\' of the type "STRING" '
            'cannot be cast to "INT". SQLSTATE: 22018',
        )
    else:
        name = 'attrs'
        monkeypatch.setitem(
            _COLUMN_TYPES,
            'customers',
            {**_COLUMN_TYPES['customers'], name: 'struct<a:int>'},
        )
        monkeypatch.setitem(
            _DATA, 'customers', _DATA['customers'].assign(attrs='x')
        )
        extra = f'  - name: {name}\n    expr: customers.{name}'
        connection = _FakeConnection(_joined_view(extra=extra))

    graph = _convert(connection)

    customers = graph['customers']
    assert {column.name for column in customers.columns} == {
        'customer_segment',
        'customer_id',
    }
    assert str(customers['customer_segment'].expr) == 'segment'
    assert Edge('order_lines', 'customer_id', 'customers') in graph.edges
    (message,) = graph.conversion_messages
    assert f"'{name}'" in message


@pytest.mark.parametrize('failing', ['empty', 'driver'])
def test_a_fact_table_that_cannot_be_read_fails_and_closes_the_connection(
    failing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if failing == 'empty':
        lines = _DATA['order_lines'].iloc[:0]
        monkeypatch.setitem(_DATA, 'order_lines', lines)
        connection = _FakeConnection(_joined_view())
    else:
        connection = _FailingConnection(_joined_view(), '`sch`.`order_lines`')
    monkeypatch.setattr(
        'sdm_connectors.backends.databricks.connect',
        lambda **kwargs: connection,
    )

    with pytest.raises(KumoRelationalError) as caught:
        Graph.from_databricks_metric_view(
            'audit_mv',
            connection=dict(server_hostname='localhost'),
            verbose=False,
        )

    if failing == 'empty':
        assert isinstance(caught.value, EmptyTableError)
    else:
        assert isinstance(caught.value, GraphConstructionError)
        assert isinstance(caught.value.__cause__, _ServerOperationError)
    assert connection.closed


@pytest.mark.parametrize(
    'rows, reason',
    [
        ('repeats', "1 of 11 rows repeat a value, e.g. 'C3'"),
        ('no-value', "'customer_id' holds no value"),
        ('some-missing', None),
    ],
)
def test_a_join_key_is_counted_where_the_rows_are(
    rows: str, reason: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""Neither Databricks nor the view checks that a many-to-one join meets
    one row, so a key that repeats made one customer of two rows. No join
    reaches a row with no key, and sampling leaves it out, so such rows
    are no reason to leave the join out.
    """
    customers = _DATA['customers']
    if rows == 'repeats':
        customers = pd.concat([customers, customers.iloc[[3]]])
    elif rows == 'no-value':
        customers = customers.assign(customer_id=None)
    else:
        customers = customers.copy()
        customers.loc[len(customers)] = [None, 'R0', 'gold']
    monkeypatch.setitem(_DATA, 'customers', customers)

    graph = _convert(_FakeConnection(_joined_view()))

    if reason is None:
        assert graph['customers'].primary_key is not None
        assert Edge('order_lines', 'customer_id', 'customers') in graph.edges
        assert graph.conversion_messages == ()
    else:
        assert set(graph.tables) == {'order_lines', 'products'}
        assert not graph['order_lines'].has_column('customer_id')
        (message,) = graph.conversion_messages
        assert "'customers'" in message and reason in message


def test_join_keys_are_counted_in_one_statement_through_the_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""One-to-many makes the parent the table keyed, here the fact table, so
    its key is counted over the rows its filter holds.
    """
    lines = _DATA['order_lines']
    monkeypatch.setitem(
        _DATA, 'order_lines', lines.assign(line_id=lines['order_id'])
    )
    connection = _FakeConnection()

    graph = _convert(connection)

    (counted,) = [sql for sql in connection.sql_log if 'COUNT(DISTINCT' in sql]
    assert 'UNION ALL' in counted
    assert "WHERE (source.status = 'VALID')" in counted
    assert 'refunds' not in graph.tables
    assert graph['order_lines'].primary_key is None
    (message,) = [m for m in graph.conversion_messages if "join 'refunds'" in m]
    assert "'line_id' is not unique in table 'order_lines'" in message


def test_a_join_whose_key_cannot_be_counted_is_kept_unchecked() -> None:
    r"""An unchecked key may build wrong entities, so the join says so, but
    a warehouse failing to count it is no reason to lose the join.
    """
    connection = _FailingConnection(_joined_view(), 'COUNT(DISTINCT')

    graph = _convert(connection)

    assert set(graph.edges) == {
        Edge('order_lines', 'customer_id', 'customers'),
        Edge('order_lines', 'product_id', 'products'),
    }
    assert graph['customers'].primary_key.name == 'customer_id'
    assert graph['products'].primary_key.name == 'product_id'
    customers, products = sorted(graph.conversion_messages)
    assert "'customers'" in customers and "'customer_id'" in customers
    assert "'products'" in products and "'product_id'" in products


def test_parse_metric_view_wildcards() -> None:
    spec = parse_metric_view("""
source: cat.sch.orders
joins:
  - name: customers
    source: cat.sch.customers
    on: source.customer_id = customers.customer_id
    joins:
      - name: regions
        source: cat.sch.regions
        on: customers.region_id = regions.region_id
  - name: promos
    source: SELECT * FROM cat.sch.promos
    on: source.promo_id = promos.promo_id
dimensions:
  - expr: source.*
  - name: tier
    expr: UPPER(customers.segment)
  - expr: customers.* except (`segment`, customers.region_id)
  - expr: customers . regions.*
  - expr: Source.* EXCEPT(amount, 2024_total)
  - expr: promos.*
  - expr: nowhere.*
measures:
  - expr: source.*
""")

    assert spec.columns == [
        MetricViewWildcard('source'),
        spec.columns[1],
        MetricViewWildcard('customers', ('segment', 'region_id')),
        MetricViewWildcard('regions'),
        MetricViewWildcard('source', ('amount', '2024_total')),
    ]
    assert spec.columns[1].name == 'tier'
    (skipped,) = spec.skipped_joins
    assert (skipped.alias, skipped.dimensions) == ('promos', ('promos.*',))
    (message,) = spec.messages
    assert "'nowhere.*'" in message


_WILDCARD_YAML = """
version: 0.1
source: cat.sch.order_lines
joins:
  - name: customers
    source: cat.sch.customers
    on: source.customer_id = customers.customer_id
dimensions:
  - name: order_month
    expr: DATE_TRUNC('MONTH', source.order_date)
  - expr: source.* EXCEPT (STATUS, `promo_id`)
  - expr: customers.* EXCEPT (customer_id)
"""


def test_a_wildcard_dimension_stands_for_the_columns_of_its_table() -> None:
    graph = _convert(
        _FakeConnection(
            _metric_view(_WILDCARD_YAML, {'order_month': 'timestamp'})
        )
    )

    fact = graph['order_lines']
    assert [column.name for column in fact.columns] == [
        'order_month',
        'line_id',
        'order_id',
        'customer_id',
        'product_id',
        'order_date',
        'product_line',
        'amount',
    ]
    assert all(fact[name].expr is None for name in ('line_id', 'amount'))
    assert {column.name for column in graph['customers'].columns} == {
        'region_id',
        'segment',
        'customer_id',
    }
    assert Edge('order_lines', 'customer_id', 'customers') in graph.edges
    assert graph.conversion_messages == ()
    assert graph.conversion_notes == (
        TimeColumnNote(
            'order_lines', 'order_date', instead_of=('order_month',)
        ),
    )


_PARAMETER_YAML = """
version: 1.1
source: cat.sch.orders
parameters:
  - name: min_qty
    data_type: INT
    default: 1
  - name: since
    data_type: DATE
    default: 2024-01-31
  - name: required
    data_type: INT
  - name: unused
    data_type: INT
filter: {filter}
dimensions:
  - name: big
    expr: source.qty >= MIN_QTY
  - name: recent
    expr: order_date >= since
  - name: untouched
    expr: CONCAT('min_qty', source.min_qty, `min_qty`, s . min_qty)
  - name: needy
    expr: qty > required
"""


def test_a_parameter_is_read_as_its_default() -> None:
    r"""As the type Databricks describes it, which wins over the one the
    definition names. A parameter nothing reads says nothing.
    """
    spec = parse_metric_view(
        _PARAMETER_YAML.format(filter='source.qty >= min_qty'),
        {'min_qty': 'bigint'},
    )

    assert spec.filter is not None
    assert spec.filter.trusted() == "source.qty >= CAST('1' AS BIGINT)"
    assert [(c.name, c.expr) for c in spec.columns] == [
        ('big', "qty >= CAST('1' AS BIGINT)"),
        ('recent', "order_date >= CAST('2024-01-31' AS DATE)"),
        ('untouched', "CONCAT('min_qty', min_qty, `min_qty`, s . min_qty)"),
    ]
    assert spec.messages == [
        "Failed to add dimension 'needy' since it references the parameter "
        "'required', which has no default",
    ]
    assert spec.notes == [
        ParameterNote('min_qty', '1'),
        ParameterNote('since', '2024-01-31'),
    ]


def test_a_filter_on_a_parameter_with_no_default_is_refused() -> None:
    with pytest.raises(UnsupportedFilterError) as caught:
        parse_metric_view(_PARAMETER_YAML.format(filter='qty > required'))

    assert "parameter 'required', which has no default" in str(caught.value)
    assert 'give the parameter a default' in str(caught.value)


@pytest.mark.parametrize(
    'data_type, default, literal',
    [
        ('bigint', "'-7'", "CAST('-7' AS BIGINT)"),
        ('BOOLEAN', 'true', "CAST('true' AS BOOLEAN)"),
        ('STRING', '2024-01-31', "CAST('2024-01-31' AS STRING)"),
        ('DECIMAL(10, 2)', "'3.10'", "CAST('3.10' AS DECIMAL(10, 2))"),
        ('STRING', '"O\'Brien \\\\ co"', "CAST('O\\'Brien \\\\ co' AS STRING)"),
        (
            'timestamp_ntz',
            "'2024-01-31T10:00'",
            "CAST('2024-01-31 10:00:00' AS TIMESTAMP_NTZ)",
        ),
    ],
)
def test_a_default_is_a_typed_literal(
    data_type: str, default: str, literal: str
) -> None:
    spec = parse_metric_view(f"""
source: cat.sch.orders
parameters:
  - name: p
    data_type: {data_type}
    default: {default}
filter: x = p
""")

    assert spec.filter is not None
    assert spec.filter.checked(['x']) == f'x = {literal}'


@pytest.mark.parametrize(
    'data_type, default, reason',
    [
        ('INT', 'abc', "its default 'abc' is not of type INT"),
        ('INT', 'true', 'its default True is not of type INT'),
        ('DATE', 'yesterday', "its default 'yesterday' is not of type DATE"),
        ('DOUBLE', '.nan', 'its default nan is not of type DOUBLE'),
        ('ARRAY<INT>', '[1]', "its type 'ARRAY<INT>' has no literal here"),
    ],
)
def test_a_default_that_is_no_value_of_its_type_is_refused(
    data_type: str, default: str, reason: str
) -> None:
    spec = parse_metric_view(f"""
source: cat.sch.orders
parameters:
  - name: p
    data_type: {data_type}
    default: {default}
dimensions:
  - name: d
    expr: x = p
""")

    assert spec.columns == []
    assert spec.messages == [
        f"Failed to add dimension 'd' since it references the parameter "
        f"'p', which {reason}"
    ]


def test_a_substituted_filter_is_still_checked() -> None:
    r"""A default is the view author's text, and is escaped into a literal,
    but what runs is still held to the same rules.
    """
    yaml_ = """
source: cat.sch.orders
parameters:
  - name: p
    data_type: STRING
    default: "x'); DROP TABLE t; --"
filter: status = p
"""
    with pytest.raises(UnsupportedFilterError, match="contains ';'"):
        parse_metric_view(yaml_)


def test_a_metric_view_on_a_metric_view_is_refused_for_what_it_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""The catalog's own advice for a metric view read as a table, to read
    it as a metric view, is what the caller already did.
    """
    monkeypatch.setitem(_TABLE_TYPES, 'order_lines', 'METRIC_VIEW')

    with pytest.raises(ValueError) as caught:
        _convert(_FakeConnection(_joined_view()))

    assert str(caught.value) == (
        "'cat.sch.order_lines' is a metric view; a metric view built on "
        'another metric view is not supported, point its source at the '
        'underlying table'
    )


def _said(graph: Graph) -> tuple[object, ...]:
    r"""The conversion notes but those on measures."""
    assert graph.conversion_messages == ()
    return tuple(
        note
        for note in graph.conversion_notes
        if not isinstance(note, MeasureNote)
    )


def _time_view(dimensions: str, measures: str = '') -> list:
    definition = f"""
version: 1.1
source: cat.sch.order_lines
dimensions:
  - name: product_line
    expr: source.product_line
{dimensions}
measures:
  - name: total_amount
    expr: SUM(source.amount)
{measures}
"""
    return _metric_view(
        definition,
        {
            'product_line': 'string',
            'order_month': 'timestamp',
            'order_quarter': 'timestamp',
            'order_date': 'timestamp',
        },
    )


_MONTH = """  - name: order_month
    expr: DATE_TRUNC('MONTH', source.order_date)"""
_QUARTER = """  - name: order_quarter
    expr: CAST(DATE_TRUNC('QUARTER', order_date) AS DATE)"""


@pytest.mark.parametrize(
    'shape', ['month-only', 'named-as-the-date', 'window-axis']
)
def test_a_truncated_date_gives_way_to_the_column_it_truncates(
    shape: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""`order_month` starts at or before `order_date`, so the shared rule
    took the truncation, and every window counted months. Named as the date
    it truncates, the source column could not be added under that name, so
    the truncation was kept, silently. A window measure ordering by the
    month makes it the time axis, though `shipped` starts earlier.
    """
    measures = ''
    if shape == 'month-only':
        dimensions = _MONTH + '\n' + _QUARTER
        note = TimeColumnNote(
            'order_lines',
            'order_date',
            instead_of=('order_month', 'order_quarter'),
            added=True,
        )
    elif shape == 'named-as-the-date':
        expr_df = _EXPR_DATA['order_lines'].assign(
            ORDER_DATE_RAW=_DATA['order_lines']['amount'],
            order_date_raw_2=_DATA['order_lines']['order_date'],
        )
        monkeypatch.setitem(_EXPR_DATA, 'order_lines', expr_df)
        # A column already named as the first choice, in another case:
        dimensions = (
            "  - name: order_date\n    expr: DATE_TRUNC('MONTH', "
            'source.order_date)\n  - name: ORDER_DATE_RAW\n    expr: '
            'source.amount'
        )
        note = TimeColumnNote(
            'order_lines',
            'order_date_raw_2',
            instead_of=('order_date',),
            source_column='order_date',
        )
    else:
        early = _DATA['order_lines']['order_date'] - pd.Timedelta(days=400)
        lines = _DATA['order_lines'].assign(shipped=early)
        monkeypatch.setitem(_DATA, 'order_lines', lines)
        monkeypatch.setitem(
            _COLUMN_TYPES,
            'order_lines',
            {**_COLUMN_TYPES['order_lines'], 'shipped': 'timestamp'},
        )
        dimensions = _MONTH + '\n  - name: shipped\n    expr: source.shipped'
        measures = (
            '  - name: trailing\n    expr: SUM(source.amount)\n'
            '    window:\n      - order: order_month\n'
            '        range: trailing 3 month\n        semiadditive: last'
        )
        note = TimeColumnNote(
            'order_lines',
            'order_date',
            instead_of=('order_month',),
            added=True,
            window_order=True,
        )

    graph = _convert(_FakeConnection(_time_view(dimensions, measures)))

    fact = graph['order_lines']
    assert fact.time_column is not None
    assert fact.time_column.name == note.column
    assert _said(graph) == (note,)
    assert f"'{note.column}'" in str(note)


_MEASURE_INPUT_YAML = """
source: c.s.lines
parameters:
  - name: floor
    data_type: INT
    default: 1
joins:
  - name: products
    source: c.s.products
    on: source.pid = products.pid
  - name: promos
    source: SELECT * FROM c.s.promos
    on: source.promo_id = promos.promo_id
dimensions:
  - name: qty
    expr: source.OrderQty
measures:
  - name: m
    expr: "{expr}"
"""


@pytest.mark.parametrize(
    'expr, inputs',
    [
        (
            'SUM(CAST(products.ListPrice AS DOUBLE))',
            [('products', 'ListPrice')],
        ),
        (
            'COUNT(DISTINCT CAST(order_ts AS DATE)) '
            '+ SIZE(CAST(tags AS ARRAY<STRUCT<a: INT>>))',
            ['order_ts', 'tags'],
        ),
        (
            'SUM(x::double) + '
            'SUM(aggregate(filter(items, z -> z > 0), 0, (acc, y) -> acc + y))',
            ['x', 'items'],
        ),
        (
            'SUM(aggregate(items, 0, (acc, y) -> acc + y, acc -> acc * y))',
            ['items', 'y'],
        ),
        (
            "MAX(order_ts) - INTERVAL 1 DAY - INTERVAL '1' DAY TO HOUR "
            '- INTERVAL (lag) DAYS',
            ['order_ts', 'lag'],
        ),
        (
            'SUM(LineTotal) FILTER (WHERE UnitPrice > floor) '
            '/ COUNT(DISTINCT SalesOrderID)',
            ['LineTotal', 'UnitPrice', 'SalesOrderID'],
        ),
        (
            'SUM(datediff(day, shipped) + datediff(DAY, a, b)) '
            '+ COUNT(DISTINCT struct(order_ts AS f))',
            ['day', 'shipped', 'a', 'b', 'order_ts'],
        ),
        (
            'SUM(date_diff(DAY, a, b) + timediff(HOUR, a, b)) '
            '+ COUNT(extract(DOW FROM order_ts))',
            ['a', 'b', 'order_ts'],
        ),
        # A dimension, a measure and two tables: no one table's columns.
        ('SUM(qty * products.ListPrice) / MEASURE(m) + SUM(promos.d)', []),
        # Databricks reads a column named like a keyword where an operand is.
        (
            'SUM(filter) / COUNT(DISTINCT case) + MAX(from) * any',
            ['filter', 'case', 'from', 'any'],
        ),
        (
            'SUM(x) OVER (ORDER BY last ROWS BETWEEN UNBOUNDED PRECEDING '
            'AND CURRENT ROW) + SUM(y) OVER (ROWS BETWEEN 1 PRECEDING AND '
            'CURRENT ROW)',
            ['x', 'last', 'y'],
        ),
        # A keyword before NOT is grammar, unless NOT negates IN or LIKE.
        ('COUNT(x IS NOT DISTINCT FROM y)', ['x', 'y']),
        ('SUM(CASE WHEN NOT x THEN 1 END)', ['x']),
        ('SUM(a) FILTER (WHERE NOT y)', ['a', 'y']),
        ('COUNT(a AND NOT NOT b)', ['a', 'b']),
        ("COUNT(filter NOT IN (1) OR case NOT LIKE 'a')", ['filter', 'case']),
        # CASE WHEN opens a searched case; DISTINCT and ALL take a sign.
        ('SUM(CASE WHEN filter THEN 1 END)', ['filter']),
        ('COUNT(DISTINCT -x) + SUM(ALL +y)', ['x', 'y']),
        # A CASE stays a CASE in a cast to a type with parameters.
        (
            'SUM(CAST(CASE WHEN source.refunded THEN 0 ELSE source.amount '
            'END AS DECIMAL(38,2)))',
            ['refunded', 'amount'],
        ),
        # A signed exponent ends where a name would go on.
        ('SUM(CASE WHEN ok THEN 1e+5ELSE amount END)', ['ok', 'amount']),
        ('SUM(amount) /* gross */ -- switch to net_amount later', ['amount']),
        # A name may lead with a digit, where a number is no name.
        (
            'SUM(2024_total) * 1e5 / 2d + 10L - SUM(source.1_000)',
            ['2024_total', '1_000'],
        ),
    ],
    ids=[
        'cast-joined',
        'cast-date',
        'lambda',
        'finishing-lambda',
        'interval',
        'filter',
        'date-function',
        'unit-first',
        'mixed',
        'keyword-named',
        'window-frame',
        'is-not-distinct',
        'when-not',
        'where-not',
        'not-not',
        'negated-comparison',
        'case-when',
        'signed-operand',
        'case-in-cast',
        'signed-exponent',
        'comments',
        'digit-led',
    ],
)
def test_a_measure_reads_the_columns_it_names(
    expr: str, inputs: list[Any]
) -> None:
    r"""A type, a datetime unit or a lambda's parameter was read as a column
    of the source table, which then read two tables and so nothing.
    """
    spec = parse_metric_view(_MEASURE_INPUT_YAML.format(expr=expr))

    (measure,) = spec.measures
    assert list(measure.inputs) == [
        name if isinstance(name, tuple) else (SOURCE_ALIAS, name)
        for name in inputs
    ]


@pytest.mark.parametrize(
    'expr, read',
    [
        (
            'order_date + INTERVAL day DAY',
            "order_date + INTERVAL CAST('1' AS INT) DAY",
        ),
        (
            'aggregate(items, x, (acc, x) -> acc + x)',
            "aggregate(items, CAST('2' AS INT), (acc, x) -> acc + x)",
        ),
        ('payload:region = region', "payload:region = CAST('EU' AS STRING)"),
        (
            "concat(r'\\', region, 'x')",
            "concat(r'\\', CAST('EU' AS STRING), 'x')",
        ),
    ],
    ids=['interval', 'lambda', 'json-path', 'raw-string'],
)
def test_a_parameter_is_read_only_where_it_is_a_value(
    expr: str, read: str
) -> None:
    spec = parse_metric_view(f"""
source: cat.sch.orders
parameters:
  - name: day
    data_type: INT
    default: 1
  - name: x
    data_type: INT
    default: 2
  - name: region
    data_type: STRING
    default: EU
dimensions:
  - name: d
    expr: {json.dumps(expr)}
""")

    (column,) = spec.columns
    assert column.expr == read


# Nested past what the name scanner settles, so it reads no name of it.
_TOO_DEEP = 'NOT (' * 64 + '(CASE WHEN x THEN not END)' + ')' * 64


def test_an_expression_too_deep_to_read_is_refused_not_misread() -> None:
    r"""Its names cannot be told from its grammar within the scan's bounds,
    and a name dropped would substitute or add the wrong columns.
    """
    view = f"""
source: cat.sch.orders
parameters:
  - name: p
    data_type: INT
    default: 1
dimensions:
  - name: d
    expr: {json.dumps(_TOO_DEEP)}
  - name: kept
    expr: x + p
measures:
  - name: m
    expr: {json.dumps(f'SUM(x) FILTER (WHERE {_TOO_DEEP})')}
"""
    spec = parse_metric_view(view)

    assert [column.name for column in spec.columns] == ['kept']
    assert spec.measures == []
    assert spec.messages == [
        "Failed to add dimension 'd' since it is too deeply nested or too "
        'long to read safely',
        "Failed to add measure 'm' since it is too deeply nested or too "
        'long to read safely',
    ]
    with pytest.raises(UnsupportedFilterError, match='too deeply nested'):
        parse_metric_view(f'{view}filter: {json.dumps(_TOO_DEEP)}\n')


def test_a_measure_with_syntax_not_understood_is_refused_not_misread() -> None:
    r"""Syntax the scan does not know failed the reading of the `NOT` before
    it, which then read `NOT` as a column of the source.
    """
    expr = "SUM(x) FILTER (WHERE NOT flag AND x AT TIME ZONE 'UTC')"
    spec = parse_metric_view(_MEASURE_INPUT_YAML.format(expr=expr))

    assert spec.measures == []
    assert spec.messages == [
        "Failed to add measure 'm' since it has syntax that is not "
        'understood, so which of its words name columns cannot be settled'
    ]


_MEASURE_YAML = """
version: 1.1
source: cat.sch.order_lines
joins:
  - name: customers
    source: cat.sch.customers
    on: source.customer_id = customers.customer_id
dimensions:
  - name: product_line
    expr: source.product_line
  - name: status
    expr: LOWER(source.status)
measures:
  - name: total_amount
    expr: SUM(AMOUNT)
  - name: lines
    expr: COUNT(1)
  - name: buyers
    expr: COUNT(DISTINCT source.customer_id)
  - name: valid_amount
    expr: SUM(amount) FILTER (WHERE source.status = 'VALID' AND promo_id IS NOT NULL)
  - name: segments
    expr: COUNT(DISTINCT customers.segment)
"""


def test_the_columns_a_measure_aggregates_become_columns() -> None:
    r"""`SUM(amount)` is only a measure, so `amount` was in no table and the
    predictive query that would say it had nothing to aggregate.
    """
    rows = _metric_view(
        _MEASURE_YAML, {'product_line': 'string', 'status': 'string'}
    )

    graph = _convert(_FakeConnection(rows))

    fact = graph['order_lines']
    assert [column.name for column in fact.columns] == [
        'product_line',
        'status',
        'amount',
        'customer_id',
        'promo_id',
    ]
    assert str(fact['status'].expr) == 'LOWER(status)'
    assert fact['amount'].expr is None
    assert {column.name for column in graph['customers'].columns} == {
        'segment',
        'customer_id',
    }
    assert graph.conversion_messages == (
        "Did not add the column 'status', which the measure 'valid_amount' "
        "reads, since the dimension 'status' has its name",
    )
    assert graph.conversion_notes == (
        MeasureNote('total_amount', 'order_lines', 'SUM', 'amount'),
        MeasureNote('lines', 'order_lines', 'COUNT', None),
        MeasureNote('buyers', 'order_lines', 'COUNT_DISTINCT', 'customer_id'),
    )


_UPPER_YAML = """
version: 0.1
source: cat.sch.order_lines
joins:
  - name: CUSTOMERS
    source: cat.sch.customers
    on: source.CUSTOMER_ID = customers.Customer_Id
dimensions:
  - name: PRODUCT_LINE
    expr: source.PRODUCT_LINE
  - name: line
    expr: source.LINE_ID
  - name: CUSTOMER_ID
    expr: CUSTOMER_ID
  - name: SEGMENT
    expr: CUSTOMERS.SEGMENT
"""


def test_names_are_matched_without_regard_to_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Databricks reads `source.PRODUCT_LINE` as the column `product_line`,
    and so must the conversion, which refused `PRODUCT_LINE` as no column.
    """
    lines = _EXPR_DATA['order_lines']
    monkeypatch.setitem(
        _EXPR_DATA,
        'order_lines',
        lines.assign(line=_DATA['order_lines']['line_id']),
    )
    rows = _metric_view(
        _UPPER_YAML,
        {
            'PRODUCT_LINE': 'string',
            'line': 'string',
            'CUSTOMER_ID': 'string',
            'SEGMENT': 'string',
        },
    )

    connection = _FakeConnection(rows)
    graph = _convert(connection)

    fact = graph['order_lines']
    assert [column.name for column in fact.columns] == [
        'PRODUCT_LINE',
        'line',
        'CUSTOMER_ID',
    ]
    assert fact['PRODUCT_LINE'].expr is None
    assert str(fact['line'].expr) == 'LINE_ID'
    # Every read of its rows names the column as the view does.
    reads = [
        sql for sql in connection.sql_log if '`order_lines` AS source' in sql
    ]
    assert reads
    assert all(
        'source.`product_line` AS `PRODUCT_LINE`' in sql for sql in reads
    )
    customers = graph['CUSTOMERS']
    # Its key is named as the join's condition spells it:
    assert {column.name for column in customers.columns} == {
        'SEGMENT',
        'Customer_Id',
    }
    assert customers.primary_key is not None
    assert customers.primary_key.name == 'Customer_Id'
    assert Edge('order_lines', 'CUSTOMER_ID', 'CUSTOMERS') in graph.edges
    assert graph.conversion_messages == ()


@pytest.mark.parametrize(
    'note',
    [
        {},
        {'instead_of': ('order_month',), 'added': True, 'source_column': 'd'},
    ],
    ids=['no-reason', 'added-twice'],
)
def test_a_time_column_note_holds_only_what_can_be_said(
    note: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        TimeColumnNote('order_lines', 'order_date', **note)


@pytest.mark.parametrize(
    'name, expr, noted',
    [
        ('Order Amount', 'source.amount', True),
        ('amount', 'amount * 100', False),
    ],
    ids=['rename', 'shadow'],
)
def test_a_measure_note_holds_only_where_the_column_is_the_source_column(
    name: str, expr: str, noted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A dimension renaming `amount` stands for it in a measure. One with
    its name that holds something else shadows it, and the note said the
    measure aggregated the dimension.
    """
    lines = _EXPR_DATA['order_lines']
    monkeypatch.setitem(
        _EXPR_DATA,
        'order_lines',
        lines.assign(**{name: _DATA['order_lines']['amount']}),
    )
    definition = f"""
version: 1.1
source: cat.sch.order_lines
dimensions:
  - name: {name}
    expr: {expr}
measures:
  - name: total
    expr: SUM(source.amount)
  - name: bare
    expr: SUM(amount)
"""
    rows = _metric_view(definition, {name: 'double'})

    graph = _convert(_FakeConnection(rows))

    assert [column.name for column in graph['order_lines'].columns] == [name]
    if noted:
        assert graph.conversion_notes == tuple(
            MeasureNote(measure, 'order_lines', 'SUM', name)
            for measure in ('total', 'bare')
        )
        assert graph.conversion_messages == ()
    else:
        assert graph.conversion_notes == ()
        (message,) = graph.conversion_messages
        assert "'total'" in message and f"'{name}'" in message


def test_a_decimal_key_is_read_as_the_integer_its_type_says(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""A `DECIMAL(38, 0)` key, as H&M's, has an integer data type but is
    read as decimals, which no type inference takes, so the sampled keys
    failed as soon as a question read them.
    """
    import decimal

    from sdm_relational_engine.rfm.base import Table

    def _decimals(values: pd.Series) -> list:
        return [decimal.Decimal(int(value[1:])) for value in values]

    lines, products = _DATA['order_lines'], _DATA['products']
    monkeypatch.setitem(
        _DATA,
        'order_lines',
        lines.assign(product_id=_decimals(lines['product_id'])),
    )
    monkeypatch.setitem(
        _DATA,
        'products',
        products.assign(product_id=_decimals(products['product_id'])),
    )
    for table in ('order_lines', 'products'):
        monkeypatch.setitem(
            _COLUMN_TYPES,
            table,
            {**_COLUMN_TYPES[table], 'product_id': 'decimal(38,0)'},
        )

    graph = _convert(_FakeConnection(_joined_view()))

    for table in ('order_lines', 'products'):
        sample = graph[table]._source_sample_df['product_id']
        assert sample.dtype == pd.ArrowDtype(pa.int64())
    assert Edge('order_lines', 'product_id', 'products') in graph.edges

    # As the sampler reads rows, by each column's data type:
    from predictive_query.schema.typing import Dtype

    df = pa.table(
        {
            'key': pa.array([decimal.Decimal(7)], pa.decimal128(38, 0)),
            'price': pa.array([decimal.Decimal('1.25')], pa.decimal128(9, 2)),
            'huge': pa.array([decimal.Decimal(10**30)], pa.decimal128(38, 0)),
        }
    ).to_pandas(types_mapper=pd.ArrowDtype)
    df = Table._sanitize(df, {'key': Dtype.int, 'price': Dtype.float})
    assert df['key'].tolist() == [7]
    assert df['price'].dtype == pd.ArrowDtype(pa.float64())
    # Past 64 bits, left as it was:
    assert df['huge'].dtype == pd.ArrowDtype(pa.decimal128(38, 0))


def test_a_view_with_fields_and_dimensions_reads_only_dimensions() -> None:
    r"""Databricks reads `fields` only when there are no `dimensions`, so
    reading both added dimensions no query of the view can see.
    """
    spec = parse_metric_view("""
source: cat.sch.orders
dimensions:
  - name: status
    expr: source.status
fields:
  - name: channel
    expr: source.channel
""")

    assert [column.name for column in spec.columns] == ['status']
    assert spec.messages == [
        "Ignored the 'fields' section since the view also has 'dimensions', "
        'and Databricks reads only those'
    ]


def test_a_filter_refused_for_a_call_names_both_ways_out() -> None:
    r"""`current_date()` reads the session, which trusting the view accepts,
    and the refusal did not say so.
    """
    row_filter = parse_metric_view(
        'source: cat.sch.orders\n'
        'filter: order_date > current_date() - INTERVAL 20 YEARS\n'
    ).filter
    assert row_filter is not None

    with pytest.raises(UnsupportedFilterError) as caught:
        row_filter.checked(['order_date'])

    message = str(caught.value)
    assert 'Create a regular view with this WHERE clause' in message
    assert message.endswith(
        'or pass trust_expressions=True to apply it if you trust whoever '
        'authored the view'
    )

    with pytest.raises(UnsupportedFilterError) as caught:
        parse_metric_view(
            _KEYWORD_YAML.format(filter="customers.segment = 'a'")
        )
    assert 'trust_expressions' not in str(caught.value)


def test_a_view_that_loses_nothing_converts_without_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""A view with measures, a parameter default and a truncated date
    loses nothing, yet warned "Could not fully convert", so a caller
    running with warnings as errors could convert no such view. How it
    was read is noted instead. A parameter is read as its default, as the
    type Databricks describes it, but a source column of its name, here
    `AMOUNT`, wins over it, bare or in another case, as a warehouse reads
    it.
    """
    expr_df = _EXPR_DATA['order_lines'].assign(big=True)
    monkeypatch.setitem(_EXPR_DATA, 'order_lines', expr_df)
    definition = f"""
version: 1.1
source: cat.sch.order_lines
parameters:
  - name: AMOUNT
    data_type: INT
    default: 20
  - name: floor
    data_type: INT
    default: 1
filter: amount >= floor
dimensions:
  - name: product_line
    expr: source.product_line
  - name: big
    expr: amount >= Amount * floor
{_MONTH}
measures:
  - name: total_amount
    expr: SUM(source.amount)
"""
    rows = _metric_view(
        definition,
        {
            'product_line': 'string',
            'big': 'boolean',
            'order_month': 'timestamp',
        },
    )
    at = rows.index(('', '', None)) + 1
    rows[at:at] = [('# Parameters', '', None), ('floor', 'bigint', None)]

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        graph = Graph.from_databricks_metric_view(
            'audit_mv', connection=_FakeConnection(rows), verbose=False
        )

    fact = graph['order_lines']
    assert fact.row_filter == "amount >= CAST('1' AS BIGINT)"
    assert str(fact['big'].expr) == "amount >= Amount * CAST('1' AS BIGINT)"
    assert graph.conversion_messages == ()
    notes = graph.conversion_notes
    assert notes == (
        ParameterNote('floor', '1'),
        MeasureNote('total_amount', 'order_lines', 'SUM', 'amount'),
        TimeColumnNote(
            'order_lines', 'order_date', instead_of=('order_month',), added=True
        ),
    )
    parameter, measure, _ = map(str, notes)
    assert "'floor'" in parameter
    assert 'SUM(order_lines.amount, 0, 30, days)' in measure


def test_a_join_key_keeps_the_spelling_the_view_names_it_by() -> None:
    r"""A wildcard names `customer_id` as the table spells it, as Databricks
    does, but the join's `source.CUSTOMER_ID` renamed the column, and the
    wildcard's own column then failed to be added as no column of the table.
    """
    definition = """
version: 1.1
source: cat.sch.order_lines
joins:
  - name: customers
    source: cat.sch.customers
    on: source.CUSTOMER_ID = customers.customer_id
dimensions:
  - expr: source.*
"""
    graph = _convert(_FakeConnection(_metric_view(definition, {})))

    fact = graph['order_lines']
    assert fact.has_column('customer_id')
    assert not fact.has_column('CUSTOMER_ID')
    assert Edge('order_lines', 'customer_id', 'customers') in graph.edges
    assert graph.conversion_messages == ()


def test_a_conversion_told_not_to_warn_says_it_only_on_the_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""A caller that says what a view loses itself had to filter the
    warnings for every thread to keep the engine from saying it too. Told
    not to, the tables it built stopped warning for good.
    """
    from sdm_relational_engine.rfm import TimeColumnWarning

    lines = pd.concat([_DATA['order_lines']] * 10, ignore_index=True)
    monkeypatch.setitem(
        _DATA,
        'order_lines',
        lines.assign(order_date=pd.Timestamp('2025-01-01')),
    )
    extra = (
        '  - name: order_date\n    expr: source.order_date\n'
        'materialization:\n  mode: relaxed\n'
    )
    # The whole table cannot be read, so the constant date is unconfirmed.
    connection = _FailingConnection(_joined_view(extra=extra), 'MIN(')

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        graph = Graph.from_databricks_metric_view(
            'audit_mv', connection=connection, verbose=False, warn=False
        )

    assert any('materialization' in m for m in graph.conversion_messages)
    fact = graph['order_lines']
    assert fact.time_column is not None
    fact.time_column = None
    with pytest.warns(TimeColumnWarning):
        fact.infer_time_column(verbose=False)
