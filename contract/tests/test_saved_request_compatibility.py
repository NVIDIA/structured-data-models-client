# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Older serialized requests and schemas remain usable by the current client.

Frozen pickle bytes exercise saved-format compatibility, including forecasts,
rankings, direct frames and schema conversion notes.
Loads, round trips and predictions retain independent expected values.
These fixtures must not be regenerated using the current implementation.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PICKLES = ROOT / 'contract/fixtures/request-pickles'
EXPECTED = json.loads((PICKLES / 'expected.json').read_text(encoding='utf-8'))


def writer() -> ModuleType:
    name = 'test_saved_request_compatibility_write_request_pickles'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'tests/support/pickle_observations.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _load(name: str) -> tuple[Any, list[Any]]:
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        loaded = pickle.loads((PICKLES / f'{name}.pkl').read_bytes())
    return loaded, [
        w for w in shown if not issubclass(w.category, ResourceWarning)
    ]


def test_saved_pickle_formats_and_cases_are_covered() -> None:
    """Keep every saved-format fixture and the scenarios it exercises."""
    written = {path.stem for path in PICKLES.glob('*.pkl')}
    assert (
        written
        == set(EXPECTED['objects'])
        == {
            'request-one-call',
            'request-forecast',
            'request-ranking',
            'request-frames',
            'schema-notes',
        }
    )
    assert EXPECTED['protocol'] == 4
    # The fixtures exercise what they name: a forecast's and a ranking's
    # parts, a request of frames with no payloads, and a schema's notes.
    objects = EXPECTED['objects']
    assert len(objects['request-forecast']['projection']['parts']) == 2
    assert len(objects['request-ranking']['projection']['parts']) == 3
    assert objects['request-frames']['projection']['payloads'] is None
    assert objects['schema-notes']['projection']['conversion_notes']


def _stable_projection(projection: dict[str, Any]) -> dict[str, Any]:
    """Keep frozen metadata and wire digests; validate payload-backed views separately."""
    result = dict(projection)
    if result.get('payloads'):
        result.pop('context')
        result.pop('predict')
    if 'parts' in result:
        result['parts'] = [_stable_projection(part) for part in result['parts']]
    return result


def _assert_wire_views(request: Any) -> None:
    for part in getattr(request, 'parts', None) or ():
        _assert_wire_views(part)
    payloads = getattr(request, 'payloads', None)
    if not payloads:
        return
    for half in ('context', 'predict'):
        frame = getattr(request, half).instance_table
        assert len(frame) == sum(
            len(p[half]['instance_table']['rows']) for p in payloads
        )
        if len(payloads) == 1:
            table = payloads[0][half]['instance_table']
            expected = pd.DataFrame(table['rows'], columns=table['columns'])
            target = request.task.target.column_name
            if half == 'predict' and target not in expected:
                assert frame[target].isna().all()
                frame = frame.drop(columns=[target])
            if '__kumo_anchor_time' in expected:
                expected['__kumo_anchor_time'] = pd.to_datetime(
                    expected['__kumo_anchor_time'], utc=True
                ).dt.tz_localize(None)
            pd.testing.assert_frame_equal(
                frame.reset_index(drop=True), expected, check_dtype=False
            )


@pytest.mark.parametrize('name', sorted(EXPECTED['objects']))
def test_a_pickle_the_reference_wrote_loads_as_the_reference_reads_it(
    name: str,
) -> None:
    r"""Design exit "pickle fixtures from the reference load": the object
    loads, saying nothing, as the reference read it back, and pickles again
    to what it was."""
    assert int(pd.__version__.split('.')[0]) >= int(
        EXPECTED['pandas'].split('.')[0]
    )
    loaded, shown = _load(name)
    assert shown == [], [str(w.message) for w in shown]
    want = EXPECTED['objects'][name]['projection']
    assert type(loaded).__name__ == want['kind']
    assert _stable_projection(
        writer().projection(loaded)
    ) == _stable_projection(want)
    _assert_wire_views(loaded)
    again = pickle.loads(pickle.dumps(loaded, protocol=4))
    assert writer().projection(again) == writer().projection(loaded)
    _assert_wire_views(again)


@pytest.mark.parametrize(
    'name',
    sorted(n for n, v in EXPECTED['objects'].items() if 'predicted' in v),
)
def test_a_request_the_reference_pickled_predicts_as_the_reference_s(
    name: str,
) -> None:
    r"""H23: a request the reference gathered and pickled is still a
    request here: predicting it sends the reference's bodies (each but its
    request id) and gives the reference's answer."""
    loaded, _ = _load(name)
    assert writer().predicted(loaded) == EXPECTED['objects'][name]['predicted']


def test_the_schema_s_conversion_notes_and_links_survive_the_pickle() -> None:
    r"""The ``main``-era schema: its conversion notes load as the
    reference wrote them, it holds no source, and its relationships are
    still checked when edited."""
    from sdm_client import Relationship, Schema, StructuredDataError

    loaded, _ = _load('schema-notes')
    assert isinstance(loaded, Schema)
    notes = EXPECTED['objects']['schema-notes']['projection'][
        'conversion_notes'
    ]
    assert list(loaded.conversion_notes) == notes
    assert loaded.source is None
    with pytest.raises(StructuredDataError):
        loaded.relationships.append(
            Relationship('order_lines', 'nothing', 'customers', 'customer_id')
        )
