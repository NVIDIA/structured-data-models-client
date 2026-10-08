# Date rules of the reference (T0.6)

This records which text the reference reads as a date or a timestamp, how it decides an ambiguous reading, what it does with time zones, and what depends on the pandas version or on the host.
The reference is `rust-core-reference` at `70aa159`, run with pandas 3.0.6 as recorded in [reference-environment.md](reference-environment.md); the pandas 2 evidence comes from a scratch environment with pandas 2.3.3, NumPy 2.3.5, pytz 2026.5 and dateutil 2.9.0.post0, which loaded the reference's `dates.py` by path.

Paths are relative to `packages/` on the reference.
`client/` stands for `structured-data-models-client/src/sdm_client/`, `pql/` for `structured-data-models-pql/src/predictive_query/`, and `engine/` for `structured-data-models-relational-engine/src/sdm_relational_engine/`.

## Where text becomes a date

| Path | Public entry | Code | Rule set |
|---|---|---|---|
| A string column named as a frame table's time column | Relational `materialize` over in-memory frames | `client/materializers/relational.py:732`, `_to_time` at `:894-960`, with `client/core/dates.py` | The client rules below |
| A date stored as text, found during schema inference | `Schema.from_data` | `_warn_dates_stored_as_text`, `client/schema.py:2076-2124` | The client rules, plus the plausible-year window; it only warns and never converts |
| Which timestamp column becomes the time column | `Schema.from_data` and sources | `choose_time_column` and `_first_instant`, `pql/time_columns.py:336`, `:493-566` | `pd.to_datetime(values, errors='coerce', utc=True)`; values before 1901 and the epoch at midnight are placeholders |
| A date literal in a predictive query | Any PQL text, such as `orders.placed > 2024-01-31` | Grammar `pql/grammar/PQLGrammar.g4:202-211`, `visitDatetime` at `pql/parser/visitor.py:570-589`, read at `pql/ast/constant.py:136` | The PQL literal rules below |
| A column the client already holds as timestamps in an `object` column | Relational `materialize` | `client/materializers/relational.py:735-757` | `pd.to_datetime(values, utc=True)` |
| A time column the engine reads from a source as text | SQL sources whose time column is a string | `to_datetime`, `engine/rfm/base/utils.py:67-100`, called from `engine/rfm/base/table.py:1135` | `pd.to_datetime(values, unit='ns', errors='coerce')`, value by value |
| A text column the engine has to type itself | Engine tables built without a semantic type for a column | `contains_timestamp`, `engine/rfm/metadata_inference/timestamp.py:13-47` | Name pattern worth 0.3 plus the share of the first 100 non-empty values `pd.to_datetime` reads; a timestamp at 1.0 or more |
| An anchor time | The `anchor_time` and `context_anchor_time` options | `_check_anchor_time`, `engine/rfm/rfm.py:470-487` | Text is refused; only a `pd.Timestamp` or `'entity'` is taken |
| A timestamp cell in a single-table request | Tabular `materialize` | `serialize_cell`, `client/wire/dtypes.py:533-545` | `pd.Timestamp(value)`, then UTC, written with microseconds and `Z` |

The client sets every engine column's semantic type from its own schema (`client/materializers/relational.py:372-384`), so on the client's public paths `contains_timestamp` does not decide a column; it matters for engine use without the client.
A single-table request sends a text column as text: `infer_tfm_dtype` types a column of date strings as `String`.

## The client rules (time columns of frames)

### Accepted shapes

A value is stripped of surrounding white space and must match one of two shapes in full (`client/core/dates.py:23-56`).

- Year first: `YYYY-MM-DD` or `YYYY/MM/DD`, two-digit month and day, optionally followed by a space or `T`, `HH:MM`, optional `:SS` with an optional fraction of any length, and an optional zone with or without one space before it.
- Day or month first: `D/M/YY` to `DD/MM/YYYY` with `/` or `-`, the year of two to four digits, optionally followed by a space or `T`, `H:MM` or `HH:MM`, optional seconds and fraction, and an optional zone.
- A zone is `Z`, `+HH`, `+HHMM`, `+HH:MM` (or with `-`), or a name starting with a letter (`[A-Za-z][A-Za-z0-9_+/-]*`).
  The name is accepted by the shape, but pandas 3 reads only `UTC` and `GMT`; every other name (`EST`, `CET`, `Europe/Berlin`) makes the value unreadable.

A value outside both shapes is read only in one case: the column is otherwise one shape, decided, and read per row (`format='mixed'`), in which case pandas reads it with `dateutil` (see "Per-row reading").

### Deciding the order

- A year-first column says its order outright.
- A slash or dash column is decided by `day_comes_first` (`dates.py:125-156`), over every value, not a sample: if some first field is above 12 and no second field is, the day comes first; if some second field is above 12 and no first field is, the month comes first; otherwise, including a column that contradicts itself, nothing decides it.
- `looks_unambiguous` (`dates.py:159-179`) is true when the first 1,000 values are all year-first strings, or when `day_comes_first` decides.
- There is no default order: an undecided slash column such as `02/01/2020` alone is refused with `INVALID_REQUEST`, telling the caller to convert it with `dayfirst=True` or store it year-first.

### The conversion

`conversion_that_works` (`dates.py:215-267`) runs `pd.to_datetime` over the whole column, trying in order: without `utc`, then with `utc=True`; and, only when asked for per-row reading, the same two again with `format='mixed'`.
`dayfirst` is passed only when the column decided it.
A candidate is taken when it gives a datetime dtype with no missing value.
The time-column path takes any year (`only_plausible_years=False`); the schema warning takes only columns whose years all lie in 1900 to 2100 (`PLAUSIBLE_YEARS`, `dates.py:43`).

`_to_time` (`client/materializers/relational.py:894-960`) applies these steps:

1. Missing values are kept as missing and not read.
2. If the column says its order, try the strict conversion (one format, inferred by pandas from the first value, for every row).
3. Otherwise, or if that fails, a column whose readable values come in both shapes is refused as written two ways.
4. If the column says its order, try the per-row conversion.
5. Otherwise refuse: as values that do not read when no shape or only junk is unread, else as ambiguous.

All refusals are `StructuredDataError` with code `INVALID_REQUEST`.

### Per-row reading

Per-row reading exists for warehouse output that writes a zone on some rows and not others.
It also reads a value outside the accepted shapes, by `dateutil`, if the rest of the column is one decided shape: for example `['15/03/2024', 'March 3 2020']` reads both.
The core must either implement `dateutil`'s reading for such values or register their refusal.

### Two-digit years

A year of two digits is placed in the hundred years around the current year, from 50 years before it to 49 after, as `dateutil` does: in 2026, `24` is 2024, `75` is 2075 and `76` is 1976.
This depends on the host's clock.

### Time zones

- A column whose values carry one offset keeps it: the result dtype is a fixed offset, such as `datetime64[us, UTC+01:00]`.
- A column mixing offsets, or mixing zoned and bare values, is read with `utc=True`.
- A bare value is a wall time with no zone; the engine and the wire treat it as UTC (`engine/rfm/base/utils.py:13-22`, `:46-64`, `client/wire/dtypes.py:536-541`).
- A zoned column is converted to naive UTC before it is used (`engine/rfm/base/utils.py:97-98`); the offset only survives in the client's own result.
- A zone given by name is never resolved from text: pandas 3 refuses every name but `UTC` and `GMT`.
- A frame column that is already zone-aware (`datetime64[ns, America/New_York]`) holds UTC instants, so converting it needs no zone rules.
  Zone rules are needed only to show local wall times, and when a caller localizes a wall time; pandas 3 does both with `zoneinfo` and the host's time zone database.

## The PQL literal rules

The grammar accepts three runs of digits joined by one separator, `/`, `-` or `.`, optionally followed by one space and three runs of digits joined by `:` (`PQLGrammar.g4:204-211`).
The text is read by `pd.to_datetime(text)` with pandas defaults (`pql/ast/constant.py:136`); a failure becomes `TypeError` in `Constant.__post_init__` and the diagnostic `VALUE_INVALID_LITERAL` (`pql/api.py:60-72`).
The first attempt wrote these rules down as D1 to D9 in `rust/crates/pql/DATES.md` on `archive/rust-core-attempt-1`.
All 34 of its examples were re-run on the reference and all hold, values and warnings.
In short:

- The order comes from the first field alone: year first if it is above 31 or has three or more digits, else day first if it is above 12, else month first (D2).
  So `01/02/2024` is the 2nd of January, unlike the client's time-column rule, which refuses it.
- A day-first reading warns, with pandas' `UserWarning` about `dayfirst=False`, when pandas can name the format (D4).
- Two-digit years follow the current year, as above (D5).
- `0000-01-01` is the year 0 under pandas 3 (D7).
- Six, eight, twelve and fourteen digit fields are packed dates read by `dateutil` (D8).
- Impossible dates, a year above 9999 and two long fields are refused (D9).

## Units and ranges

- pandas 3 reads text at microsecond resolution (`datetime64[us]`), so digits past the sixth of a fraction are dropped; pandas 2 reads at nanoseconds.
- pandas 3 reads any year from 1 to 9999 from text; pandas 2 refuses dates outside 1677-09-21 to 2262-04-11 (`OutOfBoundsDatetime`, or `NaT` when coercing).
- The in-memory sampler works in whole seconds by floor division of nanoseconds (`engine/rfm/backend/local/sampler.py:93`, `:234`), and back by multiplying (`:160`).
- Wire timestamps are UTC with exactly six fractional digits and `Z` (`client/wire/dtypes.py:536-545`, `engine/rfm/inference/payload.py:1459-1464`).
- Response columns `ANCHOR_TIMESTAMP` and `TIME` are read as text when their first value is text, else as milliseconds (`engine/rfm/rfm.py:1339-1360`).
- SQLite compares a text time column with bounds spelled like its first stored row: separator, fractional digits and a UTC suffix only (`engine/rfm/backend/sqlite/sampler.py:25-60`, probed at `:262`).

## The engine's own reading

`to_datetime` in the engine (`engine/rfm/base/utils.py:67-100`) reads a text time column of a source value by value, not with the client's rules.
An ambiguous slash date is read month first (`01/02/2024` is the 2nd of January), an impossible month is swapped (`15/03/2024` is the 15th of March), and unreadable values become `NaT` with a warning.
Mixed offsets raise `ValueError` under pandas 3 instead of coercing.
So one text column can read differently as a frame (client rules) and as a SQL source (engine rules); this is today's behavior, and the core should keep it or register the change.
This was checked at the function level, not through a source end to end.

## pandas 2 and pandas 3

The client allows pandas 1.5 and later, the connectors 2.0 and later; pandas 1.5 was not run.
Differences seen between pandas 2.3.3 and 3.0.6 on the same inputs:

| Difference | pandas 2.3.3 | pandas 3.0.6 | Evidence |
|---|---|---|---|
| Resolution of text read as dates | `datetime64[ns]` | `datetime64[us]` | 38 of the first attempt's 121 column cases read the same values with only this dtype difference |
| Years outside 1677 to 2262 | Refused (`OutOfBoundsDatetime` or `NaT`) | Read | `4711-01-02`, `0001-01-01`, and the PQL literals `001-2-3` and `1.2.3456` |
| Zone abbreviations other than the host's | Dropped: the value reads as a bare wall time | The value does not read | The corpus case `time_column.read/cst-zone` is read under pandas 2 and refused under pandas 3 |
| The host's own zone abbreviation (such as `PST` on a Pacific host) | `AttributeError` | Refused, naming `tzlocal` | Run with `TZ=America/Los_Angeles` and `TZ=America/Chicago` |
| Mixed offsets without `utc=True` | An `object` column and a warning | `ValueError` | The comment at `client/core/dates.py:255-257`; both are skipped, so the client answers alike |
| `dayfirst=True` with `format='mixed'` on year-first text | Year, month, day | Year, day, month (`2024-01-02` is the 1st of February) | 183 single-value readings of the corpus |
| Zone database | pytz, which ignores rules after 2037 | `zoneinfo`, which follows them | New York in July 2038 is UTC-5 under pandas 2 and UTC-4 under pandas 3 |

Without `format='mixed'`, both versions read year-first text as year, day, month under `dayfirst=True`, because the inferred format is `%Y-%d-%m`.
The client never passes `dayfirst` for a column that is all year-first, since `day_comes_first` decides only from slash or dash values, so this reading is not reached on the time-column path.

## What depends on the host

- The current year decides two-digit years, in the client's columns and in PQL literals.
- The host's local zone abbreviations are read differently from other abbreviations (see the table).
- The zone database: pandas 3 reads the host's `/usr/share/zoneinfo` (version 2026c on the reference host) and falls back to the `tzdata` package only when the host has none; the reference environment has no `tzdata` or `pytz` installed.
- Linux and macOS differ in whether pandas warns for a day-first PQL literal with a zero-padded year below 1000, because `strftime` pads `%Y` differently (the first attempt's `DATES.md`, "Linux and macOS"); the date read is the same.
- No locale dependence was found: the accepted shapes are numeric, and no path reads month or day names except the per-row `dateutil` fallback, which uses English names regardless of locale.
