# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

import builtins
from enum import Enum

import pydantic

from predictive_query.schema.common import StrEnum

WITH_PYDANTIC_V2 = int(pydantic.__version__.split('.')[0]) >= 2

if WITH_PYDANTIC_V2:
    from pydantic import (  # type: ignore[attr-defined]
        field_validator as _compatible_field_validator,
    )
else:
    from pydantic import (  # type: ignore[no-redef]  # ty: ignore
        validator as _compatible_field_validator,
    )

compatible_field_validator = _compatible_field_validator


class Stype(StrEnum):
    r"""The semantic type of a column.

    A semantic type denotes the semantic meaning of a column, and determines
    the preprocessing that is applied to the column. Semantic types can be
    passed to methods in the client as strings (*e.g.* ``"numerical"``).

    .. note::

        For more information about how to select a semantic type, please
        refer to the column-preprocessing documentation.

    Attributes:
        numerical: A numerical column. Typically integers or floats.
        categorical: A categorical column. Typically boolean or string values
            typically a single token in length.
        multicategorical: A multi-categorical column. Typically a concatenation
            of multiple categories under a single string representation.
        ID: A column holding IDs. Typically numerical values used to uniquely
            identify different entities.
        text: A text column. String values typically multiple tokens in length,
            where the actual language content of the value has semantic
            meaning.
        timestamp: A date/time column.
        sequence: A column holding sequences/embeddings. Consists of lists of
            floats, all of equal length, and are typically the output of
            another AI model
        image: A column holding image URLs.
    """

    numerical = 'numerical'
    categorical = 'categorical'
    multicategorical = 'multicategorical'
    ID = 'ID'
    text = 'text'
    timestamp = 'timestamp'
    sequence = 'sequence'
    image = 'image'
    unsupported = 'unsupported'

    def supports_dtype(self, dtype: 'Dtype') -> bool:
        r"""Whether a :class:`Stype` supports a :class:`Dtype`."""
        if self == Stype.numerical:
            return dtype.is_numerical()
        if self == Stype.categorical:
            return dtype.is_bool() or dtype.is_numerical() or dtype.is_string()
        if self == Stype.multicategorical:
            return dtype.is_string() or dtype.is_list()
        if self == Stype.ID:
            return dtype.is_int() or dtype.is_string() or dtype.is_float()
        if self == Stype.text:
            return dtype in {Dtype.string}
        if self == Stype.timestamp:
            return dtype.is_maybe_timestamp()
        if self == Stype.sequence:
            return dtype in {
                Dtype.floatlist,
                Dtype.intlist,
                Dtype.string,
            }
        if self == Stype.image:
            return dtype in {Dtype.string}

        assert self == Stype.unsupported
        return True


class Dtype(StrEnum):
    r"""The data type of a column.

    A data type represents how the data of a column is physically stored. Data
    types can be passed to methods in the client as strings (*e.g.* ``"int"``).

    Attributes:
        bool: A boolean column.
        int: An integer column.
        float: An floating-point column.
        date: A column holding a date.
        time: A column holding a timestamp.
        floatlist: A column holding a list of floating-point values.
        intlist: A column holding a list of integers.
        binary: A column containing binary data.
        stringlist: A column containing list of strings.
    """

    # Booleans:
    bool = 'bool'
    # Integers:
    int = 'int'
    byte = 'byte'
    int16 = 'int16'
    int32 = 'int32'
    int64 = 'int64'
    # Floating point numbers:
    float = 'float'
    float32 = 'float32'
    float64 = 'float64'
    # Strings:
    string = 'string'
    binary = 'binary'
    # Time:
    date = 'date'
    time = 'time'
    timedelta = 'timedelta'
    # Nested lists:
    floatlist = 'floatlist'
    intlist = 'intlist'
    stringlist = 'stringlist'
    # Unsupported:
    unsupported = 'unsupported'

    def is_bool(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds booleans."""
        return self in {Dtype.bool}

    def is_int(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds integers."""
        return self in {
            Dtype.int,
            Dtype.byte,
            Dtype.int16,
            Dtype.int32,
            Dtype.int64,
        }

    def is_float(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds floating point numbers."""
        return self in {Dtype.float, Dtype.float32, Dtype.float64}

    def is_numerical(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds numbers."""
        return self.is_int() or self.is_float() or self == Dtype.timedelta

    def is_string(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds strings."""
        return self in {Dtype.string, Dtype.binary}

    def is_timestamp(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds timestamps."""
        return self in {Dtype.date, Dtype.time}

    def is_maybe_timestamp(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds castable timestamps."""
        return self.is_timestamp() or self in {Dtype.string}

    def is_list(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds nested lists."""
        return self in {Dtype.floatlist, Dtype.intlist, Dtype.stringlist}

    def is_unsupported(self) -> builtins.bool:
        r"""Whether the :class:`Dtype` holds unsupported types."""
        return self in {Dtype.unsupported}

    @property
    def default_stype(self) -> Stype:
        r"""Returns the default semantic type of this data type."""
        if self.is_bool():
            return Stype.categorical
        if self.is_numerical():
            return Stype.numerical
        if self == Dtype.binary:
            return Stype.categorical
        if self == Dtype.string:
            return Stype.text
        if self.is_timestamp():
            return Stype.timestamp
        if self in {Dtype.stringlist}:
            return Stype.multicategorical
        if self in {Dtype.floatlist, Dtype.intlist}:
            return Stype.sequence

        assert self == Dtype.unsupported
        return Stype.unsupported


class TimeUnit(StrEnum):
    r"""Defines the unit of a time."""

    SECONDS = 'seconds'
    MINUTES = 'minutes'
    HOURS = 'hours'
    DAYS = 'days'
    WEEKS = 'weeks'
    MONTHS = 'months'


class ProblemType(StrEnum):
    r"""Defines supported problem types.
    RANK, CLASSIFY, and FORECAST are supported.
    With RANK we internally use a ranking loss while training
    and during batch prediction we output top k targets.
    With CLASSIFY we use a classification loss.
    With FORECAST we use a time-series forecasting approach.
    """

    RANK = 'RANK'
    CLASSIFY = 'CLASSIFY'
    FORECAST = 'FORECAST'


class AggregationType(StrEnum):
    r"""Defines supported aggregations."""

    SUM = 'SUM'
    AVG = 'AVG'
    MIN = 'MIN'
    MAX = 'MAX'
    COUNT = 'COUNT'
    COUNT_DISTINCT = 'COUNT_DISTINCT'
    FIRST = 'FIRST'
    LAST = 'LAST'
    LIST_DISTINCT = 'LIST_DISTINCT'


class RelOp(StrEnum):
    r"""Defines relational operators: :obj:`!=, <=, >=, =, <, >`."""

    NEQ = '!='
    LEQ = '<='
    GEQ = '>='
    EQ = '='
    LT = '<'
    GT = '>'


class MemberOp(StrEnum):
    r"""Defines membership operators: :obj:`IS_IN`."""

    IS_IN = 'IS IN'
    IN = 'IN'


class StrOp(StrEnum):
    r"""Defines string operators: :obj:`STARTS_WITH, ENDS_WITH, CONTAINS,
    NOT_CONTAINS`.
    """

    STARTS_WITH = 'STARTS WITH'
    ENDS_WITH = 'ENDS WITH'
    CONTAINS = 'CONTAINS'
    NOT_CONTAINS = 'NOT CONTAINS'


class BoolOp(StrEnum):
    r"""Defines boolean operators: :obj:`AND, OR, NOT`."""

    AND = 'AND'
    OR = 'OR'
    NOT = 'NOT'


def symbol(value: Enum | str) -> str:
    r"""The written form of an operator, aggregation or problem type.

    Several fields on the AST are declared as ``SomeEnum | str`` because a query
    may be built from text as well as from the enum, and not every one of them
    is normalized on construction. Reading ``.value`` off such a field therefore
    raises :class:`AttributeError` on the string case.

    Args:
        value: An enum member, or the string spelling of one.

    Returns:
        The string spelling, whichever form was given.
    """
    return value.value if isinstance(value, Enum) else value
