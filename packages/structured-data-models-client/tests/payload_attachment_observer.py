# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe the root payload take of an actual typed materialization result."""

from __future__ import annotations

import json
import weakref
from typing import Any

from sdm_client._rust import results
from sdm_client._rust.run import Finished


def record_payload_bytes(monkeypatch: Any, taken: list[bytes]) -> None:
    original = Finished.take
    payloads: weakref.WeakKeyDictionary[Finished, int] = (
        weakref.WeakKeyDictionary()
    )

    def recording(self: Finished, attachment: int) -> Any:
        got = original(self, attachment)
        result = self.result
        if isinstance(result, results.Gathered):
            payloads[self] = result.state.payload
        elif (
            isinstance(result, results.MaterializePrepared)
            and attachment == result.value
        ):
            value = json.loads(got)
            if value['value'] == 'request':
                payloads[self] = results.Gathered.from_contract(
                    value['data']
                ).state.payload
        if attachment == payloads.get(self):
            taken.append(got)
        return got

    monkeypatch.setattr(Finished, 'take', recording)
