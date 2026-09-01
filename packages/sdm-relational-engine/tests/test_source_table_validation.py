# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError
from sdm_relational_engine.api.source_table import (
    S3SourceTableRequest,
    SourceTableDataRequest,
)


@pytest.fixture
def source_table():
    return S3SourceTableRequest(
        s3_root_dir='s3://bucket/',
        s3_path='s3://bucket/table.csv',
        connector_id='connector',
        source_table_names=['table'],
    )


@pytest.mark.parametrize('sample_rows', [1001, 10**9, -1, -5])
def test_an_out_of_range_sample_rows_is_rejected(source_table, sample_rows):
    # The bound used to be checked with `return ValueError(...)` rather than
    # `raise`, so pydantic took the returned exception as the field's value:
    # the request was accepted and `sample_rows` held a ValueError instance.
    with pytest.raises(ValidationError):
        SourceTableDataRequest(
            source_table_request=source_table, sample_rows=sample_rows
        )


@pytest.mark.parametrize('sample_rows', [0, 1, 1000])
def test_a_sample_rows_inside_the_bound_is_kept_as_an_int(
    source_table, sample_rows
):
    request = SourceTableDataRequest(
        source_table_request=source_table, sample_rows=sample_rows
    )

    assert request.sample_rows == sample_rows
    assert isinstance(request.sample_rows, int)
