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


#: `dd/mm/yyyy` or `mm/dd/yyyy`, whichever it turns out to be. Anchored, so a
#: part code that opens like a date and continues is not read as one.
_TWO_PART_PATTERN = r'^(\d{1,2})[-/](\d{1,2})[-/]\d{2,4}$'


def day_comes_first(values: pd.Series, limit: int | None = None) -> bool | None:
    r"""Which way round a slash-separated column reads, if it says.

    `02/01/2024` is two dates and the column cannot tell you which. But a
    column holding `15/03/2024` has said: no month is 15, so the first
    component is the day, and every other row reads the same way. The same
    argument the other way round for `03/15/2024`.

    So the question is not whether a single value is ambiguous -- most are --
    but whether anything in the column decides it. `None` means nothing did,
    which is also the answer when the column contradicts itself.

    Every value is read, rather than the sample the shape test takes. One
    deciding value is enough and it can be anywhere: a column of a thousand
    `01/02/2024` followed by one `03/15/2024` is decided by the last of them,
    and sampling the front of it would refuse a column that answers. Read as
    two columns of numbers rather than a loop, so the whole of a large one
    costs about what the sample did.
    """
    read = values.head(limit) if limit else values
    if read.empty:
        return None
    parts = read.astype(str).str.extract(_TWO_PART_PATTERN)
    if parts.empty:
        return None
    one = pd.to_numeric(parts[0], errors='coerce')
    two = pd.to_numeric(parts[1], errors='coerce')
    first_over = bool((one > 12).any())
    second_over = bool((two > 12).any())
    if first_over == second_over:
        return None
    return first_over


def looks_unambiguous(values: pd.Series, limit: int = SAMPLE) -> bool:
    r"""Whether the column says what its dates are.

    Year-first says it outright, and the shape of a sample answers that: the
    column has one shape throughout. A slash-separated column says it when one
    of its own values rules a reading out, which is the ordinary case for any
    column spanning more than twelve days -- and that question is asked of the
    whole column, because the value that answers it can be anywhere.

    Matches whole rather than by prefix: a part code can open with something
    date-shaped and continue.
    """
    read = values.head(limit) if limit else values
    if read.empty:
        return False
    if all(
        isinstance(value, str) and UNAMBIGUOUS.fullmatch(value)
        for value in read
    ):
        return True
    return day_comes_first(values) is not None


def conversion_that_works(values: pd.Series) -> str | None:
    r"""The arguments `pd.to_datetime` needs for this column, or nothing.

    Run rather than reasoned about. A column of impossible dates coerces to
    NaT and one of mixed UTC offsets raises instead of coercing, and both are
    date-shaped enough to reach here. A column that converts to years no
    warehouse holds is a code that looks like a date, and telling its owner to
    convert it would turn part numbers into timestamps.
    """
    first, last = PLAUSIBLE_YEARS
    order = day_comes_first(values)
    extra = {} if order is None else {'dayfirst': order}
    for arguments, named in (({}, ''), ({'utc': True}, ', utc=True')):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                converted = pd.to_datetime(
                    values, errors='coerce', **arguments, **extra
                )
        except (ValueError, TypeError, OverflowError):
            continue
        if converted.isna().any():
            continue
        years = converted.dt.year
        if first <= years.min() and years.max() <= last:
            return named
    return None


def read(values: pd.Series, arguments: str) -> pd.Series:
    r"""``values`` as datetimes, by the call `conversion_that_works` named.

    A slash-separated column is read the way its own values said, since
    reading it the other way is exactly the silent wrong answer this is here
    to prevent.
    """
    first = day_comes_first(values)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if first is None:
            return pd.to_datetime(values, utc=bool(arguments))
        return pd.to_datetime(values, utc=bool(arguments), dayfirst=first)
