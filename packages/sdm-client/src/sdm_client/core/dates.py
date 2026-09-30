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
from dataclasses import dataclass
from typing import Any

import pandas as pd

_ZONE = r'(?:Z|[+-]\d{2}(?::?\d{2})?|[A-Za-z][A-Za-z0-9_+/-]*)'

#: A date that leads with its year, which is one of the two ways a column can
#: say what it holds. `2020-01-02` says it outright; `02/01/2020` is the 2nd of
#: January to half the world and the 1st of February to the other half, and is
#: decided, if at all, by `day_comes_first`.
YEAR_FIRST = re.compile(
    r'\d{4}[-/]\d{2}[-/]\d{2}'
    r'([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?'
    r'( ?' + _ZONE + r')?)?'
)

#: Enough rows to be sure the whole column has one shape, few enough that
#: inference stays cheap on a table of millions.
SAMPLE = 1000

#: `4711-01-02` is a part number and a valid date, and pandas reads it as the
#: year 4711 without complaint. Nothing in the value distinguishes the two, so
#: the window does: a column of transactional dates lands inside it, and a code
#: that happens to be date-shaped usually does not.
PLAUSIBLE_YEARS = (1900, 2100)


#: `dd/mm/yyyy` or `mm/dd/yyyy`, whichever it turns out to be, with or without
#: a time and a zone after it – a warehouse hands back all of them, in the
#: same spellings `YEAR_FIRST` takes, because a column is not written in two
#: shapes just because this client reads a zone on one of them.
_TWO_PART_PATTERN = (
    r'^(\d{1,2})[-/](\d{1,2})[-/]\d{2,4}'
    r'(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?( ?' + _ZONE + r')?)?$'
)


DAY_OR_MONTH_FIRST = re.compile(_TWO_PART_PATTERN)


def shapes_written(values: pd.Series) -> dict[str, str]:
    r"""One example of each shape the column is written in.

    A column written two ways has no single reading: the call that reads its
    slash-separated rows re-dates its year-first ones.

    Every value is read, for the reason `day_comes_first` reads every value:
    the row that makes a column two shapes can be the last one.
    """
    text = values[[isinstance(value, str) for value in values]]
    if text.empty:
        return {}
    text = text.astype(str).str.strip()
    leads_with_its_year = text.str.fullmatch(YEAR_FIRST.pattern, na=False)
    rest = text[~leads_with_its_year]
    shaped = {
        'year-first': text[leads_with_its_year],
        'day-or-month-first': rest[rest.str.match(_TWO_PART_PATTERN, na=False)],
    }
    found: dict[str, str] = {}
    for name, values in shaped.items():
        if values.empty:
            continue
        readable = values[reads_as_a_date(values)]
        if not readable.empty:
            found[name] = readable.iloc[0]
    return found


def reads_as_a_date(values: pd.Series) -> pd.Series:
    r"""Which of ``values`` are a date under some reading of them.

    The one answer to that question. It was asked in four places and answered
    four ways, and three of them could not answer at all for a column of mixed
    offsets, or counted a date the column reads as one it does not.

    A shape counts only where it holds dates. `99/99/2020` is shaped like a
    slash date and is not one, and `2021-13-45` is shaped like a year-first
    one; reporting a column holding either as written two ways sends its owner
    to reconcile two formats when what they have is one bad value.

    Asked of each value rather than of the column: one inferred format does
    not cover a column zoned on some rows and bare on others, and whether a
    value is a date is not a question about its neighbours. Read to a common
    zone, and a reading that will not run at all counts as reading nothing –
    this is asked to find out whether a column reads, so it is the one place
    that must not fail to answer.
    """
    read = pd.Series(False, index=values.index)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for dayfirst in (True, False):
            for mixed in ({}, {'format': 'mixed'}):
                try:
                    read |= pd.to_datetime(
                        values,
                        errors='coerce',
                        dayfirst=dayfirst,
                        utc=True,
                        **mixed,
                    ).notna()
                except (ValueError, TypeError, OverflowError):
                    continue
    return read


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
    parts = read.astype(str).str.strip().str.extract(_TWO_PART_PATTERN)
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
        isinstance(value, str) and YEAR_FIRST.fullmatch(value.strip())
        for value in read
    ):
        return True
    return day_comes_first(values) is not None


@dataclass(frozen=True)
class Conversion:
    r"""The call that reads one column, as arguments rather than as text.

    `pd.to_datetime` takes these; a message shows them. Carrying the decision
    as a value keeps those two from drifting, and keeps the reading that was
    checked as the reading that is applied -- deciding it again at the point
    of use is how the two come to disagree.
    """

    utc: bool = False
    dayfirst: bool | None = None
    mixed: bool = False

    @property
    def arguments(self) -> dict[str, Any]:
        r"""What to pass `pd.to_datetime`."""
        given: dict[str, Any] = {}
        if self.utc:
            given['utc'] = True
        if self.dayfirst is not None:
            given['dayfirst'] = self.dayfirst
        if self.mixed:
            given['format'] = 'mixed'
        return given

    def __str__(self) -> str:
        r"""How to write the call, for a message that tells someone to make it."""
        return ''.join(
            f', {name}={value!r}' for name, value in self.arguments.items()
        )


def conversion_that_works(
    values: pd.Series,
    *,
    only_plausible_years: bool = True,
    per_row: bool = False,
) -> Conversion | None:
    r"""The arguments `pd.to_datetime` needs for this column, or nothing.

    Run rather than reasoned about. A column of impossible dates coerces to
    NaT and one of mixed UTC offsets raises instead of coercing, and both are
    date-shaped enough to reach here. A column that converts to years no
    warehouse holds is a code that looks like a date, and telling its owner to
    convert it would turn part numbers into timestamps.

    That window is for deciding whether an undeclared column is a date at all.
    A caller who has named the column as their time column has answered that,
    and `only_plausible_years=False` takes the answer: a table of nineteenth
    century records is a real table, and refusing it for the year it holds
    would be this package overruling what it was told.

    `per_row` adds `format='mixed'`, after the readings that do not need it. A
    warehouse writes a zone on the rows that have one, so a column can be
    year-first throughout and still need a format per value. It is asked for
    only where the column has been found to be one shape: reading each value
    on its own terms will also read a column written two ways, by re-dating
    the half that is not the one it settles on.
    """
    first, last = PLAUSIBLE_YEARS
    order = day_comes_first(values)
    for mixed in [False, True] if per_row else [False]:
        for utc in (False, True):
            candidate = Conversion(utc=utc, dayfirst=order, mixed=mixed)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    converted = pd.to_datetime(
                        values, errors='coerce', **candidate.arguments
                    )
            except (ValueError, TypeError, OverflowError):
                continue
            if converted.isna().any():
                continue
            if not only_plausible_years:
                return candidate
            years = converted.dt.year
            if first <= years.min() and years.max() <= last:
                return candidate
    return None


def read(values: pd.Series, conversion: Conversion) -> pd.Series:
    r"""``values`` as datetimes, by the call `conversion_that_works` found.

    The conversion is carried rather than worked out again, so the reading
    that was checked against the whole column is the reading applied to it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return pd.to_datetime(values, **conversion.arguments)
