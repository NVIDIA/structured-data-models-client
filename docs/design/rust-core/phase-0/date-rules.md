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
The core implemented `dateutil`'s reading for such values (DT-6, owner decision 30) until the owner's answer of 2026-10-11 (E66(h)): from T1.25b such a value is outside the core's list and refused by name ("The core's text formats" below, DT-15).

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

## The core's text formats (T1.25b)

The core reads text as times by this list and nothing else (decision E68, on the owner's answer of 2026-10-11, E66(h)).
`sdm_engine::time::text` follows it; `rust/crates/sdm-engine/tests/text_times_record.py` reads it apart from the Rust reader, with Python's `re` and `datetime`, to record the vectors `text_times.rs` holds the reader to.
No reading here follows pandas or `dateutil`: no format is guessed from a column's first value, no word is read as a date, no second rolls over.
Where the reference reads text otherwise, the difference is in register DT-15 (text outside the list, refused by name), DT-16 (text in the list, read otherwise) and DT-17 (inference's text outside the client's shapes), measured over the inputs T1.25 recorded by `text-times-differential.json`.

### The formats

A value, with its surrounding white space (Unicode's) taken off, is in the list when it is, whole, one of these:

| Format | As `strftime` writes it |
|---|---|
| A date, year first: a four-digit year, a two-digit month and day, `-` or `/` twice | `%Y-%m-%d`, `%Y/%m/%d` |
| A date, day or month first: a day and a month of one or two digits, a year of two to four, `/` or `-` twice | `%d/%m/%Y` or `%m/%d/%Y`, and with `-` |
| Either date, a space or `T`, and a time of day: the hour (two digits after a year-first date, one or two after the other), the minutes, optionally the seconds, optionally a point and one to nine digits | `%H:%M`, `%H:%M:%S`, `%H:%M:%S.%f` |
| A time of day and, with or without one space between, a zone: `Z`, `UTC`, `GMT`, or an offset `+HH`, `+HHMM` or `+HH:MM` (or `-`) | `Z`, `UTC`, `GMT`, `%:::z`, `%z`, `%:z` |

Digits are ASCII digits.
ISO 8601 and RFC 3339 dates and timestamps are the first format with the third and the fourth: `T` or a space, `Z` or an offset.
The list is the client's two shapes ("Accepted shapes" above, `sdm_engine::time::text_shapes`) with one separator twice, no zone named but `UTC` and `GMT`, and every field in range.

### What a value reads as

Its fields are parsed by `jiff`'s `strtime` with the format it is written in, which the value alone gives:

- the date and the time of day must name one: a month of 1 to 12, a day the month has, an hour of 0 to 23, a minute and a second of 0 to 59 (no leap second), and an offset's hours 0 to 23 and minutes 0 to 59;
- a two-digit year is placed in the hundred years around the clock's year, from 50 years before it to 49 after, the client's rule ("Two-digit years" above; the run's `HostFacts.now` in UTC); a year of three or four digits is the year written;
- a value with a zone reads as the instant its wall clock names at that offset (`Z`, `UTC` and `GMT` at zero), one without as its wall clock in UTC;
- the year written must be 1 to 9999, and so must the instant's year, in UTC, so that no offset carries a year outside the list into it (`0000-12-31T23:00-02:00` is outside);
- a value empty once its white space is gone holds no time, as a missing one.

Every other value is outside the list: dates in words (`March 3 2020`), packed digits (`20240101`), dots (`15.01.2024`), fields written short (`2024-1-5`), two separators (`2024/01-05`), a value naming no date (`2024-02-30`, `32-12-24`), a second of 60, a fraction past nine digits, an offset of a day or more or with minutes past 59, another zone name (`EST`, `utc`), the year 0, and words such as `now`, `today` or `NaT`.

### Day or month first

The values read together decide the order, by the client's rule (`day_comes_first`, "Deciding the order" above): among the values written day or month first, in the client's shape, if some first field is past 12 and no second field is, the day comes first; if some second field is past 12 and no first field is, the month comes first; otherwise nothing decides it.
The values read together are a fetch's for `materialize`, the preview's for the time column's choice, and the first 100 non-empty ones for inference's score.
Where nothing decides, a value the two orders read differently is refused (`text_time_order`), and one they read alike (`05/05/2024`, or a year-first value) is read.
Where the rule decides that the day comes first, inference warns (`DATES_READ_DAY_FIRST`), naming the first such value's format as `strftime` writes it, `%y` for a two-digit year.

### The readings

Each refusal is `UNSUPPORTED_REQUEST` naming its rule.

- `materialize` reads a fetched text time column by the reading its provider declares (`text_times`, register DT-7: `value_by_value` is this list): a value outside the list is refused (`text_time`), as are an order left open (`text_time_order`) and offsets mixed (`text_time_zones`, DT-11); a blank value is dropped and warned of (`TIME_VALUES_UNREAD`), as the reference drops it.
- The time column's choice reads a text time column's preview to UTC, offsets mixed taken; a value outside the list (`text_time`), or an order left open (`text_time_order`), is refused only where the choice asks where the column starts.
- Inference's score (`contains_timestamp`, "Where text becomes a date" above): 0.3 for a name that says time and the share of the first 100 non-empty values read as times, a timestamp at 1.0 or more.
  The share is counted at fewest, the fewer of the two orders' counts where the values decide none.
  Where the values read in either order decide the score otherwise, the order is left open and the text is refused (`text_time_order`).
  Otherwise the share is counted at most too, adding the values in the client's shapes the list reads in neither order and, under a name that says time, every value the list does not read; where that decides otherwise the text is refused (`text_time`).
  Offsets mixed among the values read are refused (`text_time_zones`).
  So under a name that does not say time, text outside the client's shapes is no time, as the client's own inference holds (`client/core/dates.py`: inference does not read loose text as dates, which would read part codes as dates): `AW00011205` is no timestamp.

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
- Six, eight, twelve and fourteen digit fields are packed dates read by `dateutil` (D8); the core refuses every literal with a run of five digits or more by name (T1.6c, below).
- Impossible dates, a year above 9999 and two long fields are refused (D9).

### As the core reads them (T1.6b)

The core reads a literal by these rules over its fields, written down from the reference's answers and held to them (`sdm-pql`'s `tests/dates.json`, `dates.rs`; decision E69); no pandas or `dateutil` code is ported.

- A literal whose every run of digits has four digits or fewer: D2's order; a year below 100 placed around the current year where no date field has three digits or more (D5); the year 0 only as a first field of exactly `0000`, a month and a day of one or two digits, and a time of day, if any, of fields of one or two digits (D7); a time of day by value, an hour to 23, a minute and a second to 59 (D6).
- A literal with a run of five digits or more (D8) is refused by name from T1.6c, before any field is read: `validate_query` ends with `INVALID_REQUEST` naming the rule `date_literal_packed` and the literal's span, never its text (`Reading::Packed`; Phase 1 decision E78, register DT-5, the owner's decision of 2026-10-12). The reference reads such a literal through `dateutil`, as the core did from T1.6b to T1.6c, by these rules, kept as the record of what is refused:
  - a first field of six digits is a year, month and day of two digits each, in D2's order, the year placed by D5; of eight, a four-digit year, a month and a day; of twelve, then an hour and minutes; of fourteen, then seconds;
  - after such a field, each of the other two date fields, in order: with `-` as the separator once an hour is known, a UTC offset west of Greenwich (`HHMM`, or one or two digits of hours), the later one overriding; otherwise two or four digits are an hour (and minutes) when no hour is known yet, six digits an `HHMMSS` time; anything else is refused;
  - any other first field: the three date fields in D2's order, a field of three digits or more a year as written, two such refused (D9), the year placed by D5 only where no date field has three digits or more (the long run then is in the time of day);
  - a time of day after the date replaces the hour, minutes and seconds, each by value, except an hour of six, eight, twelve or fourteen digits, which is refused;
  - the year 0 is refused, and an offset of a day or more.
- D4: a reading day first (a first field of one or two digits, 13 to 31) warns when its format can be named: a four-digit year last, a month of one or two digits, a time of day of fields of one or two digits; a year below 1000 written with four digits only where the platform's `strftime` pads it (`HostFacts.platform.strftime_pads_year`). The format is written in the literal's separator: `%d/%m/%Y`, then ` %H:%M:%S`. pandas also guesses `%d/%Y/%m` for a literal whose middle field is written as its year (`13/0001/1`) and warns with it; the core names no format there (register PQL-4, proposed).
- The current year is the clock's (`HostFacts.now`) in the host's local zone (`HostFacts.local_zone`), as `dateutil` places a two-digit year by the machine's local year; where the host names no zone, UTC's. `dateutil` reads that year once, when it is imported; the core reads it each run.
- The warning is given once per day-first literal by the `validate_query` run that reads it (`DATE_LITERAL_DAY_FIRST`); the reference gives it each time it reads the literal (register PQL-3, proposed).

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
From T1.25b the core reads a source's text by its own list, "The core's text formats" below, with the client's order and two-digit year rules, and not by this reading (DT-7, DT-16).

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
  The core does the same, with a bundled fallback (DT-2, owner decision 30), and keeps applying each zone's rules after 2099 (DT-1).
- Linux and macOS differ in whether pandas warns for a day-first PQL literal with a zero-padded year below 1000, because `strftime` pads `%Y` differently (the first attempt's `DATES.md`, "Linux and macOS"); the date read is the same.
- No locale dependence was found: the accepted shapes are numeric, and no path reads month or day names except the per-row `dateutil` fallback, which uses English names regardless of locale.
