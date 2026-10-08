# Date literals

A predictive query compares a timestamp column with a date literal, such as `orders.ordered_at > 2024-01-31`.
Today's Python reads the literal with `pd.to_datetime(literal)`, and the core reads it the same way without pandas (`src/dates.rs`).
This file states what that reading is, rule by rule.
Each rule has examples, and `tests/rust_core/pql/accept_dates.py` checks every example against pandas 3.0 and against the core.

Results are written `YYYY-MM-DDTHH:MM:SS`, followed by the UTC offset when the literal gives one.
A refused literal makes the query fail with `PQL_VALUE_INVALID_LITERAL`.

## How pandas reads one literal

pandas takes three steps, and the core follows them exactly:

1. It reads the literal with `dateutil`'s parser and tries to write that reading back as a `strptime` format.
   If it can, it parses the literal with that format, and warns when the format puts the day before the month.
2. Otherwise it tries ISO 8601, then its own fixed `MM/DD/YYYY` reading of a ten-character date.
3. Otherwise it reads the literal with `dateutil` again.

The rules below are what those steps give for the shapes a query can hold.

## The rules

**D1** A date literal is three runs of digits with the same separator, `/`, `-` or `.`, between them, optionally followed by one space and a time of day: three runs of digits with `:` between them.
The grammar takes any number of digits in each run; nothing else is a date literal.

**D2** The three date fields are the year, month and day in one of three orders, chosen from the first field.
If the first field is above 31, or is written with three digits or more, the order is year, month, day.
Otherwise, if it is above 12, the order is day, month, year.
Otherwise it is month, day, year, so an ambiguous date is read month first.
The order never depends on the second or third field: `1-2024-5` asks for day 2024 and is refused.

**D3** A month-first reading gives no warning: `01/02/2024` is the 2nd of January.

**D4** A day-first reading warns when pandas can name the literal's format, which it can when the year is written with four digits and every other field with one or two.
The warning is a `UserWarning` that begins "Parsing dates in %d/%m/%Y format when dayfirst=False (the default) was specified", with the format written in the literal's own separators and time part.
When pandas cannot name the format, as with a two-digit year, the date is read the same way without a warning.
For a year below 1000 written with a leading zero, as in `13/01/0999`, pandas can name the format only where Python's `strftime` writes `%Y` with four digits: on macOS, and on Linux from Python 3.14.
On Linux with Python 3.12 or 3.13 it writes `999`, so pandas reads the same date without the warning (see "Linux and macOS" below).
A query gives the warning each time it reads the literal, once while it is parsed and once more for each check that reads it.

**D5** When no date field is written with three digits or more, a year below 100 is moved into the hundred years around the current year: from 50 years before it to 49 after.
In 2026, `00` to `75` are 2000 to 2075, and `76` to `99` are 1976 to 1999.
As soon as one field has three digits or more, every year is taken as written: `100-1-1` is the year 100, and in `02/015/75` the year is 75.

**D6** A time of day sets the hour, minute and second.
A field of up to five digits is read by its value, so `001:002:003` is 01:02:03; longer fields follow D8.
An hour of 24 or a minute or second of 60 is refused.

**D7** A first field of exactly `0000`, followed by a month and a day of one or two digits, gives the year 0, which pandas holds although Python's `datetime` cannot.
Any other way of reaching the year 0 is refused.

**D8** A field of six, eight, twelve or fourteen digits is a packed date as `dateutil` reads it: `YYMMDD`, `YYYYMMDD`, `YYYYMMDDHHMM` or `YYYYMMDDHHMMSS`.
The fields after a packed date are read as times: a field of one, two or four digits after it is an hour (or hour and minute), and a later field after a `-` is a UTC offset in hours.
A six-digit field after a packed date is `HHMMSS`.
A field of five digits or more that is not packed is a year by its value: `00001` is the year 1.

**D9** Everything else is refused: a date that does not exist, such as the 29th of February in a common year or the 31st of April; a month of 0 or above 12; a day of 0; a year above 9999; two date fields of three digits or more.

## Examples

| Rule | Literal | Result |
|---|---|---|
| D1 | `2024-1-5` | `2024-01-05T00:00:00` |
| D1 | `2024.1.5` | `2024-01-05T00:00:00` |
| D1 | `1/2/2024 12:00:00` | `2024-01-02T12:00:00` |
| D2 | `2024-01-31` | `2024-01-31T00:00:00` |
| D2 | `001-2-3` | `0001-02-03T00:00:00` |
| D2 | `1.2.3456` | `3456-01-02T00:00:00` |
| D2 | `1-2024-5` | refused |
| D3 | `01/02/2024` | `2024-01-02T00:00:00` |
| D3 | `01.02.2024` | `2024-01-02T00:00:00` |
| D3 | `12/11/2024` | `2024-12-11T00:00:00` |
| D4 | `29.02.2024` | `2024-02-29T00:00:00`, with a warning |
| D4 | `13/01/2024` | `2024-01-13T00:00:00`, with a warning |
| D4 | `31/12/2024 23:59:59` | `2024-12-31T23:59:59`, with a warning |
| D4 | `13/1/24` | `2024-01-13T00:00:00` |
| D5 | `24-01-05` | `2005-01-24T00:00:00` |
| D5 | `99-1-1` | `1999-01-01T00:00:00` |
| D5 | `1/2/3` | `2003-01-02T00:00:00` |
| D5 | `100-1-1` | `0100-01-01T00:00:00` |
| D5 | `02/015/75` | `0075-02-15T00:00:00` |
| D6 | `2024-01-02 12:30:00` | `2024-01-02T12:30:00` |
| D6 | `2024-01-02 001:002:003` | `2024-01-02T01:02:03` |
| D6 | `2024-01-02 24:00:00` | refused |
| D6 | `2024-01-02 23:59:60` | refused |
| D7 | `0000-01-01` | `0000-01-01T00:00:00` |
| D7 | `0-1-1` | refused |
| D8 | `230203-06-09` | `2003-02-23T06:00:00-09:00` |
| D8 | `20240101/00/123456` | `2024-01-01T12:34:56` |
| D8 | `00001/1/2` | `0001-01-02T00:00:00` |
| D9 | `2023-02-29` | refused |
| D9 | `2024-04-31` | refused |
| D9 | `2024-13-01` | refused |
| D9 | `2024-01-00` | refused |
| D9 | `10000-01-01` | refused |
| D9 | `123-4-567` | refused |

## Linux and macOS

pandas guesses a literal's format by writing its `dateutil` reading back with Python's `strftime`.
For a year below 1000, `%Y` is `0999` on macOS and on Linux from Python 3.14, but `999` on Linux with Python 3.12 or 3.13.
That changes one thing: the D4 warning for a day-first literal whose year is written with four digits and a leading zero.
The date read is the same on both.

- `tests/padded_years.tsv` lists every such literal found among the examples here, the quirk table's dates, the 100,000 literals `P5.dates` generates and every year below 1000 written in each field order: 885 literals, each with its date and its warning.
  `tests/padded_years.py` records it, reading each literal with pandas 3.0.6, numpy 2.5.3 and dateutil 2.9.0.post0 on macOS and in Docker's `python:3.12-slim` (Python 3.12.14), and keeps only those that differ.
- Of 103,288 literals, those 885 are the only ones that differ, and only by that warning.
  Checked the same way, Python 3.13.16 on Linux gives what 3.12 gives, and 3.14.8 gives what macOS gives.
- The core reads a literal as either platform does: `dates::Host` says whether `%Y` is padded.
  The Python binding asks the Python it runs in, so `pql_validate` and `pql_parse_date` match pandas in the same process, on macOS and on Linux.
  Without a Python process (`validate_json`, `parse_date_json`) the core assumes Linux with Python 3.12 or 3.13, where customers run it: no padding.
  `tests/padded_years.rs` checks every listed literal both ways.

## What the core does not follow

- Two-digit years are read against the current year, as `dateutil` reads them; the core takes the year from the caller (the Python binding passes Python's local year), and otherwise from the clock in UTC.
  On New Year's Eve the two can differ by a year for a few hours.
