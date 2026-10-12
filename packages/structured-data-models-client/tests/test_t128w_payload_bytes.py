# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T1.28w exit 3, held from outside the implementation (the package's
independent test author): the producer's bytes are the reference's.

For each generated DuckDB source ``t128w_payload_bytes_record.py`` recorded
from the reference (``t128w-payload-bytes.json``: statements, indices, seed
and the reference's payload as its compact JSON), python-core materializes
the same regression, and the payload bytes the core attached to the request
state (``RequestState.payload``, the typed ``sdm-wire`` payload serialized)
are the reference's, byte for byte: members in the reference's order, each
number the reference's token, each text as the reference writes it. A
sample the reference refused is held refused on the core too.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema, _routing
from sdm_client._rust import run as _rust_run
from sdm_client.errors import UnsupportedRequestError

CORE = ','.join(_routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB)
VECTORS = json.loads(
    Path(__file__)
    .with_name('t128w-payload-bytes.json')
    .read_text(encoding='utf-8')
)


@pytest.fixture(autouse=True)
def _core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)


def _core_bytes(
    sample: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> bytes:
    r"""The payload attachment's bytes, as the core's run gave them."""
    taken: list[Any] = []
    take = _rust_run.Finished.take

    def recording(self: Any, attachment: int) -> Any:
        got = take(self, attachment)
        state = self.result.get('state')
        if isinstance(state, dict) and attachment == state.get('payload'):
            taken.append(got)
        return got

    monkeypatch.setattr(_rust_run.Finished, 'take', recording)
    connection = duckdb.connect()
    for statement in sample['statements']:
        connection.execute(statement)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(connection))
        PredictiveQuery(VECTORS['query'], schema).materialize(
            indices=sample['indices'],
            anchor_time=pd.Timestamp(VECTORS['anchor_time']),
            random_seed=sample['seed'],
        )
    assert len(taken) == 1, taken
    return bytes(taken[0])


def _decimal_target(sample: dict[str, Any]) -> bool:
    return any(
        s.startswith('CREATE TABLE orders') and 'price DECIMAL' in s
        for s in sample['statements']
    )


def test_the_vectors_were_recorded_and_say_something() -> None:
    samples = VECTORS['samples']
    assert VECTORS['recorder'] == 't128w_payload_bytes_record.py'
    assert len(samples) >= 20
    assert sum('payload' in s for s in samples) >= 20
    # Most of them reach a payload on the core, not a skip.
    assert sum(not _decimal_target(s) for s in samples) >= 20


@pytest.mark.parametrize(
    'sample', VECTORS['samples'], ids=[s['id'] for s in VECTORS['samples']]
)
def test_the_core_s_payload_bytes_are_the_reference_s(
    sample: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    if 'refused' in sample:
        with pytest.raises(Exception):  # noqa: B017 - any refusal
            _core_bytes(sample, monkeypatch)
        return
    try:
        got = _core_bytes(sample, monkeypatch)
    except UnsupportedRequestError as refused:
        # A sum over a DECIMAL column is T1.27's ``target_type``, refused by
        # name on the core until it lands: not a payload this package
        # writes, so the refusal is held instead. Any other sample the
        # core does not take fails; once T1.27 takes it, its bytes are
        # compared below.
        assert _decimal_target(sample), refused
        assert refused.code == 'UNSUPPORTED_REQUEST'
        return
    want = sample['payload'].encode('utf-8')
    if got != want:
        at = next(
            (
                i
                for i, (a, b) in enumerate(zip(got, want, strict=False))
                if a != b
            ),
            min(len(got), len(want)),
        )
        pytest.fail(
            f'{sample["id"]}: the bytes differ at {at}: core '
            f'{got[max(0, at - 60) : at + 60]!r} reference '
            f'{want[max(0, at - 60) : at + 60]!r}'
        )
