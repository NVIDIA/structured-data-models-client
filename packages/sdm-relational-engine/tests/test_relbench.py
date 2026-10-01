# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

import warnings

import pytest


def test_ambiguous_links_are_reported_when_the_graph_is_built() -> None:
    r"""PQL rejects an ambiguous link only once a query fails, which reads as a
    fault in the query rather than in the graph it was written against.
    """
    from sdm_relational_engine.rfm.relbench import _warn_ambiguous_links

    edges = [
        ('item', 'SOLDTOPARTY', 'customer'),
        ('item', 'SHIPTOPARTY', 'customer'),
        ('item', 'doc_id', 'document'),
    ]
    with pytest.warns(UserWarning) as recorded:
        _warn_ambiguous_links(edges)

    message = str(recorded[0].message)
    assert "'item' -> 'customer'" in message
    assert 'SOLDTOPARTY' in message and 'SHIPTOPARTY' in message
    assert "graph.unlink('item', 'SOLDTOPARTY', 'customer')" in message
    assert '; ' in message or message.count('graph.unlink(') == 1
    assert 'document' not in message


def test_a_graph_with_one_key_per_pair_warns_nothing() -> None:
    from sdm_relational_engine.rfm.relbench import _warn_ambiguous_links

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _warn_ambiguous_links(
            [('item', 'doc_id', 'document'), ('doc', 'cust_id', 'customer')]
        )
