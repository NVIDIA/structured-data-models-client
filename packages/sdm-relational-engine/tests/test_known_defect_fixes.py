# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Three defects that predated the Kumo rename, and the checks that keep them
fixed. Each one was reproduced before it was repaired.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest
from sdm_relational_engine.rfm.base.sql_sampler import (
    RESERVED_COLUMN_PREFIX,
    _reject_reserved_columns,
)

RFM_INIT = (
    Path(__file__).resolve().parents[1]
    / 'src'
    / 'sdm_relational_engine'
    / 'rfm'
    / '__init__.py'
)


def test_the_package_root_does_not_import_the_local_backend() -> None:
    r"""Importing it at module scope requires the compiled sampler.

    ``backend.local`` raises at import unless the extension is present, so a
    module-scope import here made every path through the package depend on it,
    including the Databricks and Snowflake serving paths that run inference on
    the endpoint and never sample locally. Asserted structurally because the
    extension is present wherever these tests run, so importing it cannot be
    observed here.
    """
    tree = ast.parse(RFM_INIT.read_text())

    module_scope = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    offenders = [
        node
        for node in module_scope
        if isinstance(node, ast.ImportFrom)
        and (node.module or '').endswith('backend.local')
    ]

    assert not offenders, (
        'rfm/__init__.py imports backend.local at module scope again, which '
        'makes the compiled sampler a hard requirement of every import of '
        'this package'
    )


def test_local_table_is_still_reachable() -> None:
    r"""Deferring the import must not remove the name."""
    import sdm_relational_engine.rfm as rfm

    assert rfm.LocalTable is not None
    assert 'LocalTable' in rfm.__all__


def test_an_unknown_attribute_still_raises_attribute_error() -> None:
    r"""The lazy accessor must not swallow ordinary typos."""
    import sdm_relational_engine.rfm as rfm

    with pytest.raises(AttributeError, match='no attribute'):
        rfm.NoSuchName


class _Graph:
    def __init__(self, tables: dict) -> None:
        self.tables = tables


class _Table:
    def __init__(self, name: str, columns: list[str]) -> None:
        self.name = name
        self.columns = [_Column(c) for c in columns]


class _Column:
    def __init__(self, name: str) -> None:
        self.name = name


def test_a_source_column_using_the_reserved_prefix_is_refused() -> None:
    r"""It collided with the temporary key table the generated joins select
    from, which the warehouse reported only as an ambiguous column name.
    """
    graph = _Graph(
        {
            'users': _Table(
                'users', ['user_id', f'{RESERVED_COLUMN_PREFIX}batch__']
            )
        }
    )

    with pytest.raises(ValueError, match='reserved by the SQL backends'):
        _reject_reserved_columns(graph)


def test_ordinary_column_names_are_left_alone() -> None:
    graph = _Graph({'users': _Table('users', ['user_id', 'age', '_private'])})

    _reject_reserved_columns(graph)


def test_a_response_naming_another_model_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    r"""The contract does not require the response to echo the request, so this
    warns rather than failing; silently returning another model's predictions
    is what it prevents.
    """
    from sdm_relational_engine.client.rfm import _warn_on_unexpected_model

    class _Response:
        model = 'some-other-model'

    with caplog.at_level(logging.WARNING, logger='sdm_relational_engine'):
        _warn_on_unexpected_model(_Response(), 'kumo-relational')

    assert 'some-other-model' in caplog.text
    assert 'kumo-relational' in caplog.text


def test_a_matching_response_says_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from sdm_relational_engine.client.rfm import _warn_on_unexpected_model

    class _Response:
        model = 'kumo-relational'

    with caplog.at_level(logging.WARNING, logger='sdm_relational_engine'):
        _warn_on_unexpected_model(_Response(), 'kumo-relational')

    assert caplog.text == ''


def test_no_requested_model_is_not_a_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    r"""Session predictions do not carry a model; the session pinned it."""
    from sdm_relational_engine.client.rfm import _warn_on_unexpected_model

    class _Response:
        model = 'kumo-relational'

    with caplog.at_level(logging.WARNING, logger='sdm_relational_engine'):
        _warn_on_unexpected_model(_Response(), None)

    assert caplog.text == ''
