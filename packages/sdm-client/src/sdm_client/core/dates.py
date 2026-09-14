# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Reading a date out of text, only where the text says one thing.

Inference does not parse strings for dates, because a reader loose enough for
a warehouse `DATE` column stored as `STRING` also reads part codes as dates and
fails much later on a value it should never have touched. But a caller may name
a string column as a table's time column, and then it has to be read -- so this
is where the reading is done, and where it refuses rather than guesses.
"""

from __future__ import annotations

import re
import warnings

import pandas as pd

#: Year first, always. `02/01/2020` is the 2nd of January to half the world
#: and the 1st of February to the other half, and nothing in the column says
#: which; `2020-01-02` says it.
UNAMBIGUOUS = re.compile(
    r'\d{4}[-/]\d{2}[-/]\d{2}'
    r'([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?'
)

#: Enough rows to be sure the whole column has one shape, few enough that
#: inference stays cheap on a table of millions.
SAMPLE = 1000

#: `4711-01-02` is a part number and a valid date, and pandas reads it as the
#: year 4711 without complaint. Nothing in the value distinguishes the two, so
#: the window does: a column of transactional dates lands inside it, and a code
#: that happens to be date-shaped usually does not.
PLAUSIBLE_YEARS = (1900, 2100)


def looks_unambiguous(values: pd.Series, limit: int = SAMPLE) -> bool:
    r"""Whether every value read is year-first and whole.

    Matches whole rather than by prefix: a part code can open with something
    date-shaped and continue.
    """
    read = values.head(limit) if limit else values
    if read.empty:
        return False
    return all(
        isinstance(value, str) and UNAMBIGUOUS.fullmatch(value)
        for value in read
    )


def conversion_that_works(values: pd.Series) -> str | None:
    r"""The arguments `pd.to_datetime` needs for this column, or nothing.

    Run rather than reasoned about. A column of impossible dates coerces to
    NaT and one of mixed UTC offsets raises instead of coercing, and both are
    date-shaped enough to reach here. A column that converts to years no
    warehouse holds is a code that looks like a date, and telling its owner to
    convert it would turn part numbers into timestamps.
    """
    first, last = PLAUSIBLE_YEARS
    for arguments, named in (({}, ''), ({'utc': True}, ', utc=True')):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                converted = pd.to_datetime(values, errors='coerce', **arguments)
        except (ValueError, TypeError, OverflowError):
            continue
        if converted.isna().any():
            continue
        years = converted.dt.year
        if first <= years.min() and years.max() <= last:
            return named
    return None


def read(values: pd.Series, arguments: str) -> pd.Series:
    r"""``values`` as datetimes, by the call `conversion_that_works` named."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return pd.to_datetime(values, utc=bool(arguments))
