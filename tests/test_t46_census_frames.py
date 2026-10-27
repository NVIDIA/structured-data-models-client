# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame census inputs are real public-call recipes, never recorded outcomes."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.tooling
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def census() -> Any:
    spec = importlib.util.spec_from_file_location(
        't46_census', ROOT / 'contract/tools/refusal_census.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frame_probes_cover_the_current_constructor_and_consumer_sites(census: Any) -> None:
    probes = census.probes(ROOT)
    frame = [probe for probe in probes if probe.get('frame') is not None]
    assert {probe['frame'] for probe in frame} == set(census.FRAME_RECIPES)
    required = {
        'rule:frame_boolean_key', 'rule:frame_native_family',
        'rule:frame_temporal_aggregation', 'rule:frame_temporal_entity',
        'rule:frame_text_features', 'rule:frame_text_key',
        'rule:frame_time_type', 'rule:frame_untimed_timestamp_features',
        'rule:frame_wide_timestamps', 'rule:direct_numeric_storage',
    }
    assert required <= {probe['site'] for probe in frame}
    assert all(probe['databases'] == ['duckdb'] for probe in frame)
    assert all('outcomes' not in probe for probe in frame)


def test_constructor_recipes_contain_the_intended_native_fault(census: Any) -> None:
    duplicate, _ = census._frame_probe('duplicate-columns')
    assert duplicate['users'].columns.has_duplicates
    nonstrings, _ = census._frame_probe('nonstring-columns')
    assert list(nonstrings['users'].columns) == [1]
    assert census._frame_probe('empty-tables')[0] == {}
    assert 7 in census._frame_probe('invalid-table-name')[0]
    assert isinstance(census._frame_probe('not-a-frame')[0]['users'], list)
    boolean, _ = census._frame_probe('boolean-key')
    assert str(boolean['users']['user_id'].dtype) == 'bool'
    wide, options = census._frame_probe('wide-timestamps')
    assert str(wide['orders']['at'].dtype) == 'datetime64[s]'
    assert options['time_columns'] == {'orders': 'at'}


def test_route_passes_frame_recipe_without_rewriting_case_schema(
    census: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = next(p for p in census.probes(ROOT) if p.get('frame'))
    case = census._case(probe, 0, ['schema.from-data/frames/frames'])
    assert 'frame' not in case['input']
    (tmp_path / '0.json').write_text(json.dumps({
        'case': case, 'options': {}, 'model': 'KumoRelational',
        'frame': probe['frame'],
    }))
    seen = []

    def outcome(*args: Any, **kwargs: Any) -> dict[str, str]:
        seen.append((args, kwargs, os.environ['SDM_CORE_ROUTES']))
        return {'result': 'refusal', 'stage': 'from_data'}

    monkeypatch.setattr(census, '_outcome', outcome)
    monkeypatch.setenv('SDM_CORE_ROUTES', '')
    census.route(tmp_path, tmp_path / 'recorded.json', 'duckdb', core=True)
    assert seen[0][1]['frame'] == probe['frame']
    assert seen[0][2] == 'schema.from-data/frames/frames'
    assert json.loads((tmp_path / 'recorded.json').read_text())[0]['stage'] == 'from_data'


def test_unknown_frame_recipe_fails_closed(census: Any, tmp_path: Path) -> None:
    path = tmp_path / census.PROBES
    path.parent.mkdir()
    probe = dict(next(p for p in census.probes(ROOT) if p.get('frame')))
    probe['frame'] = 'unrecorded-python-expression()'
    path.write_text(json.dumps({'format': 'sdm-refusal-census/1', 'probes': [probe]}))
    with pytest.raises(census.CensusError, match='not a probe'):
        census.probes(tmp_path)
