# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Present the core's retained request tables as lossless pandas frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from sdm_client._rust import codec, results
from sdm_client._rust.run import run
from sdm_client.errors import StructuredDataError

if TYPE_CHECKING:
    from sdm_client.request import Split


def read(payloads: Sequence[Mapping[str, Any]], half: str) -> Split:
    """Read all tables in one core call; conversion/rebasing stays in Rust."""
    import pandas as pd
    import pyarrow as pa

    from sdm_client.request import Split

    attachments = {
        at: codec.attachment(payload) for at, payload in enumerate(payloads)
    }
    finished = run(
        {
            'op': 'read_request_split',
            'payloads': list(attachments),
            'half': half,
            'selection': {'kind': 'all'},
        },
        attachments,
    )
    answer = results.expect(finished.result, results.RequestSplitData)

    def frame(attachment: int | None) -> pd.DataFrame:
        if attachment is None:
            raise StructuredDataError(
                'the core omitted an inspected request table',
                code='INTERNAL_ERROR',
            )
        stream = finished.take(attachment)
        with pa.RecordBatchReader._import_from_c_capsule(
            stream.__arrow_c_stream__()
        ) as reader:
            # Arrow-backed columns preserve nullable wide integers, list
            # elements and microsecond timestamps without a float/object detour.
            return reader.read_all().to_pandas(types_mapper=pd.ArrowDtype)

    return Split(
        instance_table=frame(answer.instance),
        related_tables={name: frame(data) for name, data in answer.related},
    )
