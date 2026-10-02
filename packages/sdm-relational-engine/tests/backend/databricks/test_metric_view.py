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
    GraphConstructionError,
    KumoRelationalError,
)
from sdm_relational_engine.rfm import Edge, Graph, ViewConversionWarning

try:
    import pyarrow as pa
    from sdm_connectors.backends.databricks import Connection
    from sdm_relational_engine.rfm.backend.databricks.metric_view import (
        SOURCE_ALIAS,
        JoinKeyRef,
        MetricViewJoin,
        UnsupportedFilterError,
        UnsupportedJoinError,
        check_filter_calls,
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
    assert spec.messages == [
        "Failed to add join 'composite' since composite key references are "
        'not yet supported',
    ]


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
    assert spec.messages == [
        "Failed to add join 'broken' since of its unsupported cardinality "
        "'many_to_many'",
    ]


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
  - name: valid
    expr: source.amount
""")

    assert spec.joins == []
    assert [(c.name, c.alias) for c in spec.columns] == [
        ('valid', 'source'),
    ]
    assert spec.filter == "source.status = 'VALID'"
    assert spec.messages == [
        "Failed to add join 'promos' since only plain table names are "
        'supported as join sources',
        "Failed to add join 'composite' since only single-column equality "
        'join conditions are supported',
        "Failed to add dimension 'promo_kind' since it references the "
        "skipped join 'promos'",
        "Failed to add dimension 'nested_name' since it references the "
        "skipped join 'nested'",
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
    ],
)
def test_parse_metric_view_source_filter(row_filter: str) -> None:
    spec = parse_metric_view(
        _FILTERED_YAML.format(filter=json.dumps(row_filter))
    )

    assert spec.filter == row_filter
    assert not any('filter' in message for message in spec.messages)
    check_filter_calls(spec.filter, _FACT_COLUMNS)


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
    ],
)
def test_check_filter_calls_refuses_a_call(
    row_filter: str, reason: str
) -> None:
    with pytest.raises(UnsupportedFilterError) as caught:
        check_filter_calls(row_filter, _FACT_COLUMNS)

    assert reason in str(caught.value)
    assert 'Create a regular view with this WHERE clause' in str(caught.value)


def test_check_filter_calls_reads_a_keyword_after_a_column_as_grammar() -> None:
    # `status` is a column of the source table, so `IN (` is grammar.
    check_filter_calls("status IN ('VALID')", _FACT_COLUMNS)
    with pytest.raises(UnsupportedFilterError, match="calls 'IN'"):
        check_filter_calls("status IN ('VALID')", ())


def test_parse_metric_view_refuses_a_filter_that_is_not_sql() -> None:
    with pytest.raises(UnsupportedFilterError, match='not a SQL condition'):
        parse_metric_view(_FILTERED_YAML.format(filter='true'))


def test_parsing_leaves_calls_to_check_filter_calls() -> None:
    def parse(row_filter: str) -> Any:
        return parse_metric_view(
            _FILTERED_YAML.format(filter=json.dumps(row_filter))
        )

    # Trusted mode skips `check_filter_calls`, and nothing else refuses this:
    assert parse('my_udf(source.amount) > 0').filter == (
        'my_udf(source.amount) > 0'
    )
    with pytest.raises(UnsupportedFilterError, match='joined table'):
        parse("customers.segment = 'gold'")
    with pytest.raises(UnsupportedFilterError, match='SELECT'):
        parse('source.amount > (SELECT 1)')
    # A stray `)` would close the filtered subquery and append to it:
    with pytest.raises(UnsupportedFilterError, match="closes a '\\('"):
        parse('source.amount > 0) OR (1 = 1')


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
    assert spec.messages == [
        "Failed to add join 'source' since the name 'source' is already in use",
        "Failed to add join 'dim' since the name 'dim' is already in use",
    ]


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
                (name, dtype, 'YES', _CATALOG, _SCHEMA, table)
                for name, dtype in column_types.items()
            ]
            return

        if 'information_schema' in sql:
            return

        if sql.upper().startswith('SELECT COUNT(*)'):
            table = re.search(r'FROM `[^`]+`\.`[^`]+`\.`([^`]+)`', sql).group(1)
            self._rows = [(len(_DATA[table]),)]
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

        columns = {}
        for name in names:
            name = name.replace('``', '`')
            if name in df.columns:
                columns[name] = df[name]
            else:
                assert expr_df is not None and name in expr_df.columns, (
                    f'Unexpected column {name!r} in SQL: {sql}'
                )
                columns[name] = expr_df[name]
        self._arrow = pa.Table.from_pandas(
            pd.DataFrame(columns), preserve_index=False
        )

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
    }
    assert fact.primary_key is not None
    assert fact.primary_key.name == 'line_id'
    assert fact.time_column is not None
    assert fact.time_column.name in {'order_month', 'order_quarter'}
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
    assert "Failed to add dimension 'promo_kind'" in message
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


def test_from_databricks_metric_view_invalid_name() -> None:
    with pytest.raises(ValueError, match='Invalid metric view name'):
        Graph.from_databricks_metric_view(
            'SELECT 1',
            connection=_FakeConnection(),
            verbose=False,
        )


def test_read_metric_view_definition() -> None:
    definition, dtypes, catalog, schema = read_metric_view_definition(
        _FakeConnection(), '`cat`.`sch`.`sales_mv`'
    )

    assert definition == _METRIC_VIEW_YAML
    assert dtypes == {
        'order_month': 'timestamp',
        'order_quarter': 'timestamp',
        'product_line': 'string',
        'product_id': 'string',
        'customer_segment': 'string',
        'region_name': 'string',
        'region_code': 'string',
    }
    assert catalog == _CATALOG
    assert schema == _SCHEMA


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
    assert {column.name for column in fact.columns} == {'product_line'}
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
    assert {column.name for column in fact.columns} == {'product_id'}
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
    assert {column.name for column in fact.columns} == {'product_line'}


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


def test_a_trusted_metric_view_keeps_its_udf_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expr_df = _EXPR_DATA['order_lines'].assign(masked='x')
    monkeypatch.setitem(_EXPR_DATA, 'order_lines', expr_df)

    graph, sql_log = _udf_metric_view(trust_expressions=True)

    assert str(graph['order_lines']['masked'].expr) == 'my_udf(product_line)'
    assert any('my_udf' in sql for sql in sql_log)
    assert graph.conversion_messages == ()


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
    assert fact.time_column is not None
    assert fact.time_column.name == 'Order Month'
    assert {column.name for column in fact.columns} == {
        'Order Month',
        'Product Family',
        'Product Line',
        'customer_id',
        'product_id',
        'order_id',
    }

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
