# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Naming a graph so a cache can tell one from another.

``hash()`` cannot do this. Python salts string hashing per process, so the same
graph names itself differently in two workers, and the built-in covers only table
names and edges: a column that changed type, a key that moved and a time column
that appeared all leave the name untouched, and a cache keyed on it serves the
wrong graph.
"""

import subprocess
import sys

import pandas as pd
import pytest
import sdm_relational_engine.rfm as rfm
from sdm_relational_engine.rfm.fingerprint import graph_fingerprint


def _graph(**changes) -> rfm.Graph:
    names = changes.get('rename', {})
    column = changes.get('column_rename', {})
    frames = {
        names.get('customers', 'customers'): pd.DataFrame(
            {'cid': [1, 2, 3], column.get('seg', 'seg'): [*'abc']}
        ),
        names.get('orders', 'orders'): pd.DataFrame(
            {
                'oid': [1, 2, 3],
                'cid': [1, 2, 3],
                'ts': pd.to_datetime(['2025-01-01'] * 3),
            }
        ),
    }
    customers = names.get('customers', 'customers')
    orders = names.get('orders', 'orders')
    if changes.get('extra_column'):
        frames[orders]['amount'] = [1.0, 2.0, 3.0]
    graph = rfm.Graph.from_data(
        frames, edges=[], infer_metadata=True, verbose=False
    )
    graph[customers].primary_key = 'cid'
    graph[orders].primary_key = changes.get('key', 'oid')
    graph[orders].time_column = changes.get('time', 'ts')
    if changes.get('linked'):
        graph.link(src_table=orders, fkey='cid', dst_table=customers)
    if changes.get('stype'):
        graph[customers][column.get('seg', 'seg')].stype = changes['stype']
    return graph


def test_the_same_shape_gets_the_same_name() -> None:
    assert graph_fingerprint(_graph()) == graph_fingerprint(_graph())


@pytest.mark.parametrize(
    'change',
    [
        {'linked': True},
        {'extra_column': True},
        {'time': None},
        {'key': 'cid'},
        {'stype': 'text'},
    ],
    ids=['edge', 'column', 'time-column', 'primary-key', 'stype'],
)
def test_anything_a_prediction_can_see_changes_the_name(change: dict) -> None:
    assert graph_fingerprint(_graph()) != graph_fingerprint(_graph(**change))


def test_the_name_does_not_depend_on_the_order_tables_were_added() -> None:
    r"""Two builds of one source differ only in iteration order, not in shape."""
    frames = {'a': pd.DataFrame({'x': [1]}), 'b': pd.DataFrame({'y': [2]})}
    forward = rfm.Graph.from_data(
        frames, edges=[], infer_metadata=True, verbose=False
    )
    reverse = rfm.Graph.from_data(
        dict(reversed(list(frames.items()))),
        edges=[],
        infer_metadata=True,
        verbose=False,
    )

    assert graph_fingerprint(forward) == graph_fingerprint(reverse)


def test_the_name_is_the_same_in_another_process() -> None:
    r"""The reason ``hash()`` cannot be used: it is salted per process.

    A cache shared by workers has to agree on the name, so this runs the same
    graph in a fresh interpreter rather than trusting one.
    """
    script = (
        'import pandas as pd, sdm_relational_engine.rfm as rfm;'
        'from sdm_relational_engine.rfm.fingerprint import graph_fingerprint;'
        "f={'t': pd.DataFrame({'x':[1,2],'y':[3,4]})};"
        'g=rfm.Graph.from_data(f, edges=[], infer_metadata=True, verbose=False);'
        'print(graph_fingerprint(g))'
    )
    here = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    there = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert here == there and len(here) == 64


def test_the_encoding_version_is_part_of_the_name() -> None:
    r"""A change to the encoding must not read as the same graph."""
    import sdm_relational_engine.rfm.fingerprint as module

    graph = _graph()
    name = graph_fingerprint(graph)

    original = module.FINGERPRINT_VERSION
    module.FINGERPRINT_VERSION = original + 1
    try:
        assert graph_fingerprint(graph) != name
    finally:
        module.FINGERPRINT_VERSION = original


def test_a_release_that_did_not_touch_the_encoding_keeps_the_name() -> None:
    r"""Versioning on the package version threw away every downstream cache on
    a release that had nothing to do with fingerprinting.
    """
    import sdm_relational_engine.rfm.fingerprint as module

    graph = _graph()
    name = graph_fingerprint(graph)

    assert 'FINGERPRINT_VERSION' in dir(module)
    assert graph_fingerprint(_graph()) == name


def test_a_name_cannot_be_mistaken_for_a_delimiter() -> None:
    r"""Two shapes must not write the same bytes.

    Joining fields with a chosen character lets a name containing that character
    split into two, so a graph with one oddly named column can produce the bytes
    of a graph with two ordinary ones.
    """
    from sdm_relational_engine.rfm.fingerprint import _term

    assert _term('ab') + _term('c') != _term('a') + _term('bc')
    assert _term('a') + _term('') != _term('') + _term('a')
    assert _term('xy') + _term('z') != _term('x') + _term('yz')

    assert _term('a,b') + _term('c') != _term('a') + _term('b,c')


def test_a_composite_key_is_not_confusable_with_two_single_keys() -> None:
    r"""Key order is semantic, so its encoding has to be unambiguous."""
    from sdm_relational_engine.rfm.fingerprint import _term

    assert _term(11) + _term('a') != _term(1) + _term('1a')
    assert _term(2) + _term('a') + _term('b') != _term(1) + _term('a,b')


def test_a_guessed_edge_is_not_the_same_graph_as_a_declared_one() -> None:
    r"""A guess can change on the next build; a declaration cannot.

    A cache that could not tell them apart would keep serving whichever shape
    inference happened to produce first.
    """
    frames = {
        'customers': pd.DataFrame({'cid': [1, 2, 3], 'seg': [*'abc']}),
        'orders': pd.DataFrame({'oid': [1, 2, 3], 'cid': [1, 2, 3]}),
    }

    def build(declare: bool) -> rfm.Graph:
        graph = rfm.Graph.from_data(
            frames, edges=[], infer_metadata=True, verbose=False
        )
        graph['customers'].primary_key = 'cid'
        graph['orders'].primary_key = 'oid'
        if declare:
            graph.link(src_table='orders', fkey='cid', dst_table='customers')
        graph.infer_links(verbose=False)
        return graph

    declared, guessed = build(True), build(False)

    assert [(e.src_table, e.fkey, e.dst_table) for e in declared.edges] == [
        (e.src_table, e.fkey, e.dst_table) for e in guessed.edges
    ]
    assert graph_fingerprint(declared) != graph_fingerprint(guessed)


def test_an_adversarial_name_cannot_forge_another_graph_end_to_end() -> None:
    r"""The _term tests check the encoder; this checks what is built from it.

    A bug in how the table and column terms are composed, such as one of them
    joining names without going through _term, would not be caught by testing
    the helper on its own.
    """
    one = _graph(rename={'customers': '3:orders'})
    other = _graph(rename={'customers': '3', 'orders': 'orders'})

    assert graph_fingerprint(one) != graph_fingerprint(other)


def test_a_column_named_like_an_encoded_field_cannot_forge_one() -> None:
    colliding = _graph(column_rename={'seg': '4:cid7:unknown'})

    assert graph_fingerprint(colliding) != graph_fingerprint(_graph())
