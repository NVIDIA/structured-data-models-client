# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``Request`` pickles the reference wrote load in this client (H23,
T2.14b's design exit "pickle fixtures from the reference load"), and a
``main``-era ``Schema`` pickle holding ``conversion_notes`` does too.

    python -m pytest contract/tests/test_t214b_pickles.py -q

The pickles and what the reference reads back from each are in
``contract/fixtures/request-pickles``, written by its ``write.py`` from the
reference commit: a request a predictive query gathered in one call, a
forecast gathered as its steps, a ranking gathered as one question per
candidate, a request built from frames, and a schema of ``main``'s semantic
model with its conversion notes. Each loads here as the reference reads it
(its task, query, entities, facts, payloads, halves and parts; the
schema's tables, links and notes), pickles again to the same, and each
gathered request predicts against the scripted endpoint as the reference's
did: the same bodies sent and the same answer.

A request gathered over a source holds the source, which neither client
pickles (``Source.__reduce__``), so a reference-pickled request is one over
frames, which predicts on the route that answers frames. These pass on the
base and hold the package's host clean-up to them; the request state's
version, which a request the core gathers carries, is held by
``test_t214b_host.py``.
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
    name = 't214b_write_request_pickles'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, PICKLES / 'write.py'
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


def test_every_pickle_the_reference_wrote_is_listed() -> None:
    r"""Control: the fixtures are the writer's, of the reference commit the
    contract names, each with what the reference read back."""
    written = {path.stem for path in PICKLES.glob('*.pkl')}
    assert written == set(EXPECTED['objects']) == set(writer().NAMES)
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {EXPECTED["reference"]}' in env
    assert EXPECTED['protocol'] == 4
    # The fixtures exercise what they name: a forecast's and a ranking's
    # parts, a request of frames with no payloads, and a schema's notes.
    objects = EXPECTED['objects']
    assert len(objects['request-forecast']['projection']['parts']) == 2
    assert len(objects['request-ranking']['projection']['parts']) == 3
    assert objects['request-frames']['projection']['payloads'] is None
    assert objects['schema-notes']['projection']['conversion_notes']


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
    assert writer().projection(loaded) == want
    again = pickle.loads(pickle.dumps(loaded, protocol=4))
    assert writer().projection(again) == want


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
