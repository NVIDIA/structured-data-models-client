# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native overlap facts preserve the original reached warning boundary."""

from types import SimpleNamespace
import warnings
from unittest.mock import patch

import pandas as pd
import pytest

from sdm_client import Relationship
from sdm_client._rust.frame_facade import _relationship_overlap
from sdm_client.materializers.relational import _warn_of_relationships_that_join_nothing
from sdm_client.warnings import SDMWarning


class Tables(dict):
    @property
    def tables(self):
        return self


def observe(core, child, parent, *, duplicate=False, fail=None):
    link = Relationship('child', 'fk', 'parent', 'id')
    schema = Tables(child=SimpleNamespace(data=pd.DataFrame({'fk': child})),
                    parent=SimpleNamespace(data=pd.DataFrame({'id': parent})))
    schema.relationships = [link, link] if duplicate else [link]
    trace = []
    unique = pd.Series.unique
    isin = pd.Series.isin

    def native_unique(series):
        trace.append(('unique', series.name))
        if fail == series.name:
            raise TypeError('native unique failed')
        return unique(series)

    def native_isin(series, values):
        trace.append(('isin', tuple(series), tuple(values)))
        return isin(series, values)

    with patch.object(pd.Series, 'unique', native_unique), patch.object(pd.Series, 'isin', native_isin):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            error = None
            try:
                if core:
                    _relationship_overlap(schema)
                else:
                    _warn_of_relationships_that_join_nothing(schema)
            except Exception as caught:
                error = (type(caught), str(caught))
            return trace, [(type(w.message), str(w.message)) for w in seen], error


@pytest.mark.parametrize(('child', 'parent'), [([], []), ([], [1]), ([1], []), ([None], [1]), ([1, 1], [1]), ([1], [2]), ([1, None], [2, None])])
@pytest.mark.parametrize('duplicate', [False, True])
def test_complete_overlap_native_schedule_and_warning(child, parent, duplicate):
    assert observe(True, child, parent, duplicate=duplicate) == observe(False, child, parent, duplicate=duplicate)


@pytest.mark.parametrize('failed', ['fk', 'id'])
def test_native_observation_error_prevents_all_warning(failed):
    assert observe(True, [1], [2], fail=failed) == observe(False, [1], [2], fail=failed)


@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt, SDMWarning])
def test_overlap_handler_error_is_the_actual_exception(error_type):
    schema = Tables(child=SimpleNamespace(data=pd.DataFrame({'fk': [1]})),
                    parent=SimpleNamespace(data=pd.DataFrame({'id': [2]})))
    schema.relationships = [Relationship('child', 'fk', 'parent', 'id')]
    actual = error_type('native warning failed')
    with patch.object(warnings, 'showwarning', side_effect=actual), warnings.catch_warnings():
        warnings.simplefilter('always')
        with pytest.raises(error_type) as got:
            _relationship_overlap(schema)
    assert got.value is actual


def test_selected_relationship_rendering_errors_are_not_formatter_defects():
    class NativeLink:
        source_table, source_column = 'child', 'fk'
        target_table, target_column = 'parent', 'id'
        def __str__(self):
            raise ValueError('native relationship render failed')
    schema = Tables(child=SimpleNamespace(data=pd.DataFrame({'fk': [1]})),
                    parent=SimpleNamespace(data=pd.DataFrame({'id': [2]})))
    schema.relationships = [NativeLink()]
    with pytest.raises(ValueError, match='native relationship render failed'):
        _relationship_overlap(schema)
