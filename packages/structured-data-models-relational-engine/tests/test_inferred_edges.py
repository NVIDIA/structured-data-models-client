# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Recording which edges were guessed rather than declared.

Link inference reads names and values to guess a relationship, so a graph built
without catalog edges is partly a guess. Nothing said which parts, which leaves a
caller unable to explain why a prediction came out as it did, and leaves a cache
unable to tell one guessed shape from another.
"""

import itertools

import pandas as pd
import sdm_relational_engine.rfm as rfm
from sdm_relational_engine.rfm.graph_fingerprint import graph_fingerprint


def _graph(edges: list | None = None) -> rfm.Graph:
    frames = {
        'customers': pd.DataFrame({'customer_id': [1, 2, 3], 'seg': [*'abc']}),
        'orders': pd.DataFrame(
            {'order_id': [1, 2, 3], 'customer_id': [1, 2, 3]}
        ),
    }
    graph = rfm.Graph.from_data(
        frames,
        edges=[] if edges is None else edges,
        infer_metadata=True,
        verbose=False,
    )
    graph['customers'].primary_key = 'customer_id'
    graph['orders'].primary_key = 'order_id'
    return graph


def test_nothing_is_reported_before_inference_runs() -> None:
    assert _graph().inferred_edges == ()


def test_an_inferred_edge_is_reported() -> None:
    graph = _graph()

    graph.infer_links(verbose=False)

    assert graph.inferred_edges == (('orders', 'customer_id', 'customers'),)


def test_a_declared_edge_is_not_reported_as_a_guess() -> None:
    r"""The distinction the record exists to make."""
    graph = _graph()
    graph.link(src_table='orders', fkey='customer_id', dst_table='customers')

    graph.infer_links(verbose=False)

    assert graph.inferred_edges == ()


def test_the_record_is_ordered_so_two_runs_can_be_compared() -> None:
    graph = _graph()
    graph.infer_links(verbose=False)

    assert list(graph.inferred_edges) == sorted(graph.inferred_edges)


def test_inferring_twice_does_not_un_guess_the_first_guess() -> None:
    r"""``from_data(infer_metadata=True)`` already infers, so a later call adds nothing.

    Recording only what THIS call added would clear the earlier mark, and a
    guessed edge would then be indistinguishable from a declared one.
    """
    graph = _graph()
    graph.infer_links()
    once = graph.inferred_edges

    graph.infer_links()

    assert once
    assert graph.inferred_edges == once


def test_a_guessed_edge_and_a_declared_one_are_not_the_same_graph() -> None:
    r"""Same tables, same edge, different confidence in why the edge is there."""
    guessed = _graph()
    guessed.infer_links()

    declared = _graph(edges=[])
    declared.link('orders', 'customer_id', 'customers')

    assert [(e.src_table, e.fkey, e.dst_table) for e in guessed.edges] == [
        (e.src_table, e.fkey, e.dst_table) for e in declared.edges
    ]
    assert declared.inferred_edges == ()
    assert graph_fingerprint(guessed) != graph_fingerprint(declared)


def test_inference_provenance_only_ever_grows() -> None:
    r"""Once inference has chosen an edge, a later pass cannot un-choose it.

    Stated as its own invariant because everything downstream rests on it: a
    fingerprint that could drop a mark would let a guessed graph be served for a
    declared one after nothing more than a second inference pass.
    """
    graph = _graph()
    marks = []

    for _ in range(4):
        graph.infer_links()
        marks.append(set(graph.inferred_edges))

    assert marks[0]
    for earlier, later in itertools.pairwise(marks):
        assert earlier <= later
    assert marks[0] == marks[-1]


def test_a_declared_edge_is_never_marked_by_a_later_inference() -> None:
    r"""Inference running afterwards must not claim an edge it did not choose."""
    graph = _graph(edges=[])
    graph.link('orders', 'customer_id', 'customers')

    graph.infer_links()

    assert graph.inferred_edges == ()


def test_the_mark_reaches_the_fingerprint() -> None:
    r"""The provenance is only worth recording if it changes the graph's identity."""
    graph = _graph(edges=[])
    graph.link('orders', 'customer_id', 'customers')
    declared = graph_fingerprint(graph)

    graph._inferred_edges = (('orders', 'customer_id', 'customers'),)

    assert graph_fingerprint(graph) != declared


def test_an_edge_that_was_removed_stops_being_reported() -> None:
    r"""A mark that outlives its edge fingerprints a link the graph no longer has."""
    graph = _graph()
    graph.infer_links()
    assert graph.inferred_edges

    graph.unlink('orders', 'customer_id', 'customers')

    assert graph.inferred_edges == ()


def test_a_graph_that_lost_a_guessed_edge_matches_one_that_never_had_it() -> (
    None
):
    r"""Otherwise removing an edge leaves a graph that cannot equal its own shape."""
    removed = _graph()
    removed.infer_links()
    removed.unlink('orders', 'customer_id', 'customers')

    never = _graph()

    assert graph_fingerprint(removed) == graph_fingerprint(never)


def test_dropping_a_table_drops_the_marks_of_its_edges() -> None:
    graph = _graph()
    graph.infer_links()
    assert graph.inferred_edges

    graph.remove_table('orders')

    assert graph.inferred_edges == ()


def test_declaring_an_inferred_edge_again_clears_the_guess() -> None:
    """A caller who removes an inferred link and declares it themselves has
    stated the relationship, so it must stop being reported as a guess.
    """
    graph = _graph()
    graph.infer_links(verbose=False)
    assert graph.inferred_edges == (('orders', 'customer_id', 'customers'),)

    graph.unlink('orders', 'customer_id', 'customers')
    graph.link('orders', 'customer_id', 'customers')

    assert graph.inferred_edges == ()


def test_inference_can_guess_an_edge_again_after_it_was_declared() -> None:
    """Clearing the mark records who declared it last, not that the edge may
    never be guessed again.
    """
    graph = _graph()
    graph.infer_links(verbose=False)
    graph.unlink('orders', 'customer_id', 'customers')
    graph.link('orders', 'customer_id', 'customers')
    graph.unlink('orders', 'customer_id', 'customers')

    graph.infer_links(verbose=False)

    assert graph.inferred_edges == (('orders', 'customer_id', 'customers'),)


def test_declaring_one_edge_leaves_another_guess_alone() -> None:
    frames = {
        'customers': pd.DataFrame({'customer_id': [1, 2, 3], 'seg': [*'abc']}),
        'orders': pd.DataFrame(
            {'order_id': [1, 2, 3], 'customer_id': [1, 2, 3]}
        ),
        'items': pd.DataFrame({'item_id': [1, 2, 3], 'order_id': [1, 2, 3]}),
    }
    graph = rfm.Graph.from_data(
        frames, edges=[], infer_metadata=True, verbose=False
    )
    graph['customers'].primary_key = 'customer_id'
    graph['orders'].primary_key = 'order_id'
    graph['items'].primary_key = 'item_id'
    graph.infer_links(verbose=False)
    assert len(graph.inferred_edges) == 2

    graph.unlink('orders', 'customer_id', 'customers')
    graph.link('orders', 'customer_id', 'customers')

    assert graph.inferred_edges == (('items', 'order_id', 'orders'),)
