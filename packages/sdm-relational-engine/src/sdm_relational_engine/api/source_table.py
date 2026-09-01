# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Literal, Union

from predictive_query.schema.common import StrEnum
from predictive_query.schema.typing import (
    Dtype,
    Stype,
    compatible_field_validator,
)
from pydantic import Field
from pydantic.dataclasses import dataclass

from sdm_relational_engine.api.data_source import DataSourceType

TableName = str

# Source Table ================================================================


class FileType(StrEnum):
    r"""Supported file types for file-based source tables."""

    CSV = 'CSV'
    PARQUET = 'PARQUET'


class LLMType(StrEnum):
    r"""Supported LLM types."""

    # Use LLM embeddings as features
    FEATURE = 'feature'


@dataclass(frozen=True)
class UnavailableSourceTable:
    r"""A source table not available during processing, e.g. because it is
    processed by client-side.
    """

    table: TableName
    data_source_type: Literal[DataSourceType.UNAVAILABLE] = (
        DataSourceType.UNAVAILABLE
    )


@dataclass(frozen=True)
class S3SourceTable:
    r"""A source table located on the Amazon S3 object store."""

    # We support two types of table file path:
    # 1. s3_path specifies the whole directory (prefix), ending with "/"
    # 2. s3_path specifies the full path of a single file, ending with file
    #    name suffix that must be one of ".csv" or ".parquet"
    s3_path: str

    # Internal: S3 connector ID, if we are working with a Kumo Relational-owned named S3
    # connector:
    connector_id: str | None = None
    source_table_name: TableName | None = None

    # If not provided, then the file_path must either end in `.csv` or
    # `.parquet`, and we will parse the file type from there. Please use the
    # `validated_file_type` proper to access the parsed & validated file type.
    file_type: FileType | None = None

    data_source_type: Literal[DataSourceType.S3] = DataSourceType.S3

    @property
    def table(self) -> TableName:
        if self.s3_path == '':
            assert self.source_table_name is not None
            return self.source_table_name
        if self.s3_path.endswith('/'):
            return TableName(
                self.s3_path.rstrip('/').rsplit('/', maxsplit=1)[1]
            )
        filename = self.s3_path.rsplit('/', maxsplit=1)[1]
        return TableName(filename.rsplit('.', maxsplit=1)[0])  # strip suffix


@dataclass(frozen=True)
class SnowflakeSourceTable:
    r"""A source table located in the Snowflake data warehouse."""

    snowflake_connector_id: str
    database: str
    schema_name: str
    table: TableName
    data_source_type: Literal[DataSourceType.SNOWFLAKE] = (
        DataSourceType.SNOWFLAKE
    )


@dataclass(frozen=True)
class DatabricksSourceTable:
    r"""A source table located in the Databricks data warehouse."""

    databricks_connector_id: str
    table: TableName
    data_source_type: Literal[DataSourceType.DATABRICKS] = (
        DataSourceType.DATABRICKS
    )


@dataclass(frozen=True)
class GlueSourceTable:
    r"""A source table located in the AWS Glue data warehouse."""

    glue_connector_id: str
    table: TableName
    account: str
    region: str
    database: str
    data_source_type: Literal[DataSourceType.GLUE] = DataSourceType.GLUE


@dataclass(frozen=True)
class BigQuerySourceTable:
    r"""A source table loated in the BigQuery data warehouse."""

    bigquery_connector_id: str
    table_name: TableName
    project_id: str
    dataset_id: str
    data_source_type: Literal[DataSourceType.BIGQUERY] = DataSourceType.BIGQUERY


@dataclass(frozen=True)
class GCSSourceTable:
    r"""A source table located on Google Cloud Storage (GCS)."""

    # We support two types of table file path:
    # 1. gcs_path specifies the whole directory (prefix), ending with "/"
    # 2. gcs_path specifies the full path of a single file, ending with file
    #    name suffix that must be one of ".csv" or ".parquet"
    gcs_path: str

    # Internal: GCS connector ID, if we are working with a Kumo Relational-owned named GCS
    # connector:
    connector_id: str | None = None
    source_table_name: TableName | None = None

    # If not provided, then the file_path must either end in `.csv` or
    # `.parquet`, and we will parse the file type from there.
    file_type: FileType | None = None

    data_source_type: Literal[DataSourceType.GCS] = DataSourceType.GCS

    @property
    def table(self) -> TableName:
        if self.gcs_path == '':
            assert self.source_table_name is not None
            return self.source_table_name
        if self.gcs_path.endswith('/'):
            return TableName(
                self.gcs_path.rstrip('/').rsplit('/', maxsplit=1)[1]
            )
        filename = self.gcs_path.rsplit('/', maxsplit=1)[1]
        return TableName(filename.rsplit('.', maxsplit=1)[0])  # strip suffix


@dataclass(frozen=True)
class ADLSSourceTable:
    r"""A source table located on Azure Data Lake Storage (ADLS)."""

    # We support two types of table file path:
    # 1. adls_path specifies the whole directory (prefix), ending with "/"
    # 2. adls_path specifies the full path of a single file, ending with file
    #    name suffix that must be one of ".csv" or ".parquet"
    adls_path: str

    # Internal: ADLS connector ID, if we are working with a Kumo Relational-owned named
    # ADLS connector:
    connector_id: str | None = None
    source_table_name: TableName | None = None

    # If not provided, then the file_path must either end in `.csv` or
    # `.parquet`, and we will parse the file type from there.
    file_type: FileType | None = None

    data_source_type: Literal[DataSourceType.ADLS] = DataSourceType.ADLS

    @property
    def table(self) -> TableName:
        if self.adls_path == '':
            assert self.source_table_name is not None
            return self.source_table_name
        if self.adls_path.endswith('/'):
            return TableName(
                self.adls_path.rstrip('/').rsplit('/', maxsplit=1)[1]
            )
        filename = self.adls_path.rsplit('/', maxsplit=1)[1]
        return TableName(filename.rsplit('.', maxsplit=1)[0])  # strip suffix


# Deliberately typing.Union rather than `X | Y`: this is the annotation pydantic
# reads to build a *discriminated* union (see the Field(discriminator=...) uses
# in this module and in api/table.py). It is a wire contract that works today,
# and rewriting the spelling buys nothing a caller can see.
SourceTableType = Union[  # noqa: UP007
    S3SourceTable,
    SnowflakeSourceTable,
    DatabricksSourceTable,
    BigQuerySourceTable,
    GCSSourceTable,
    ADLSSourceTable,
    GlueSourceTable,
    UnavailableSourceTable,
]

# Method: Configuration =======================================================


@dataclass
class SourceTableConfigRequest:
    connector_id: str | None
    table_name: str
    source_type: DataSourceType

    root_dir: str | None = None
    file_type: FileType | None = None


@dataclass
class SourceTableConfigResponse:
    source_table: SourceTableType = Field(discriminator='data_source_type')


# Method: Validate ============================================================


@dataclass
class SourceTableValidateRequest:
    table_name: str
    connector_id: str | None
    source_type: DataSourceType
    root_dir: str | None = None


@dataclass
class SourceTableValidateResponse:
    is_valid: bool
    # In case there's a suggestion, a warning, or an error:
    msg: str


# Method: List ================================================================


@dataclass
class SourceTableListRequest:
    connector_id: str | None
    source_type: DataSourceType

    # Only for object store-based connectors:
    root_dir: str | None = None

    def __post_init__(self):
        if self.connector_id is None and self.source_type not in (
            DataSourceType.S3,
            DataSourceType.GCS,
            DataSourceType.ADLS,
        ):
            raise ValueError(
                "A 'None' connector ID is only supported for S3, GCS, or "
                'ADLS-backed tables. Please specify a connector ID to '
                'proceed.'
            )


@dataclass
class SourceTableListResponse:
    table_names: list[str]


# Method: Get Data ============================================================


@dataclass
class SourceColumn:
    r"""The metadata of a column in a source table. Note that a source column
    simply provides a view into the metadata of a source table. To modify
    metadata, please create a Kumo Relational Table and adjust the table's data and
    semantic types.

    .. note::
        Semantic types are inferred based on data types only, and thus may not
        be accurate.

    Args:
        name (str): The name of the column.
        stype (Stype, optional): The semantic type of the column.
        dtype (Dtype): The data type of the column
        is_primary (bool): Whether the column refers to a primary key.
    """

    name: str
    stype: Stype | None  # Kumo Relational-inferred.
    dtype: Dtype
    is_primary: bool


@dataclass
class S3SourceTableRequest:
    r"""A request to fetch a source table located on Amazon S3. This table
    can be located at either
        root_dir/table_name/*.(csv|parquet)
        root_dir/table_name.(csv|parquet)
    """

    s3_root_dir: str
    connector_id: str | None = None
    table_names: list[str] | None = None
    file_type: FileType | None = None
    source_type: Literal[DataSourceType.S3] = DataSourceType.S3


@dataclass
class SnowflakeSourceTableRequest:
    connector_id: str
    table_names: list[str] | None = None
    source_type: Literal[DataSourceType.SNOWFLAKE] = DataSourceType.SNOWFLAKE


@dataclass
class BigQuerySourceTableRequest:
    connector_id: str
    table_names: list[str] | None = None

    # Discriminator:
    source_type: Literal[DataSourceType.BIGQUERY] = DataSourceType.BIGQUERY


@dataclass
class DatabricksSourceTableRequest:
    connector_id: str
    table_names: list[str] | None = None

    # Discriminator:
    source_type: Literal[DataSourceType.DATABRICKS] = DataSourceType.DATABRICKS


@dataclass
class GlueSourceTableRequest:
    connector_id: str
    table_names: list[str] | None = None
    source_type: Literal[DataSourceType.GLUE] = DataSourceType.GLUE


@dataclass
class GCSSourceTableRequest:
    r"""A request to fetch a source table located on Google Cloud Storage. This
    table can be located at either
        root_dir/table_name/*.(csv|parquet)
        root_dir/table_name.(csv|parquet)
    """

    gcs_root_dir: str
    connector_id: str | None = None
    table_names: list[str] | None = None
    file_type: FileType | None = None
    source_type: Literal[DataSourceType.GCS] = DataSourceType.GCS


@dataclass
class ADLSSourceTableRequest:
    r"""A request to fetch a source table located on Azure Data Lake Storage.
    This table can be located at either
        root_dir/table_name/*.(csv|parquet)
        root_dir/table_name.(csv|parquet)
    """

    adls_root_dir: str
    connector_id: str | None = None
    table_names: list[str] | None = None
    file_type: FileType | None = None
    source_type: Literal[DataSourceType.ADLS] = DataSourceType.ADLS


@dataclass
class SourceTableDataRequest:
    # Table request (metadata needed to fetch a table from the connector):
    source_table_request: (
        S3SourceTableRequest
        | BigQuerySourceTableRequest
        | DatabricksSourceTableRequest
        | SnowflakeSourceTableRequest
        | GCSSourceTableRequest
        | ADLSSourceTableRequest
        | GlueSourceTableRequest
    ) = Field(discriminator='source_type')

    # Whether to fetch and include sample rows in the response:
    sample_rows: int = 0

    @compatible_field_validator('sample_rows')
    def _validate_sample_rows(cls, v: int):
        if v > 1000:
            raise ValueError('sample_rows cannot be greater than 1000.')
        if v < 0:
            raise ValueError('sample_rows cannot be negative.')
        return v


@dataclass
class SourceTableDataResponse:
    table_name: TableName
    cols: list[SourceColumn] = field(default_factory=list)

    # Serialized (json) data of sample rows dataframe, if requested:
    sample_rows: str | None = None


# Other =======================================================================


@dataclass
class TableStats:
    r"""Minimal statistics of a :class:`SourceTable`.

    Args:
        size_bytes (int): The size of the table in bytes.
        num_rows (int): The number of rows in the table.
    """

    size_bytes: int
    num_rows: int


@dataclass
class LLMRequest:
    source_table_type: SourceTableType
    template: str
    model: str
    model_api_key: str
    output_dir: str
    output_column_name: str
    output_table_name: str
    dimensions: int | None = None
    llm_type: LLMType = LLMType.FEATURE


@dataclass
class LLMResponse:
    job_id: str
