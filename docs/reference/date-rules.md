<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Date and timestamp rules

These are the maintained Rust reader contracts, not instructions to reproduce pandas parsing.
Python/Java adapt input types; Rust decides date meaning using explicit host facts and provider capabilities.
Older pandas investigations remain in Git history at `0b3afe8fa:docs/design/date-rules.md`.

## Which rules apply?

| Input | Rule owner |
|---|---|
| Text time columns | [`sdm-engine` text reader](../../rust/crates/sdm-engine/src/time/text.rs), with the formats and column-order rules below |
| Time-column selection and inference | [`sdm-schema`](../../rust/crates/sdm-schema/src/infer.rs), using the text reader and bounded inference rules |
| PQL date literals | [`sdm-pql` date reader](../../rust/crates/sdm-pql/src/dates.rs), reached through [`sdm-core`](../../rust/crates/sdm-core/src/pql.rs) |
| Already typed Arrow timestamps | Their declared unit and timezone; no text guessing |
| Provider data | The provider's declared `text_times` capability, never a guess based on connector name |

A string column is not automatically a timestamp merely because its values resemble dates.
Timestamp inference and declared timestamp roles decide whether these conversions apply.

## The core's text formats

### Accepted shapes

Surrounding Unicode whitespace is removed; digits must be ASCII.
A missing or blank value holds no time.
Every nonblank value must match one complete format:

| Part | Accepted spelling |
|---|---|
| Year-first date | `YYYY-MM-DD` or `YYYY/MM/DD`, with a four-digit year and two-digit month/day |
| Day/month-first date | One- or two-digit day/month, two- to four-digit year; `/` or `-`, using the same separator twice |
| Optional time | Space or `T`, then `HH:MM`, optional `:SS`, optional fraction of one to nine digits; the hour may have one digit after a day/month-first date |
| Optional zone after time | `Z`, `UTC`, `GMT`, `+HH`, `+HHMM`, `+HH:MM`, or the negative offsets; optionally separated by one space |

Fields must name a valid calendar date and time: hour 0–23, minute/second 0–59, offset strictly less than one day with minutes 0–59.
The written year and resulting UTC year must both be 1–9999.
Unzoned values mean UTC; zoned values preserve the instant.
Two-digit years fall between 50 years before and 49 years after the run's UTC year supplied by `HostFacts.now`.
Three- or four-digit years mean the year written.

Reject words such as `March 3 2020`, `now` or `NaT`, packed dates, dotted dates, short year-first fields, mixed separators, impossible dates, leap seconds, fractions longer than nine digits, other zone names and year zero.
This is deliberately narrower and more deterministic than historical pandas/dateutil fallback parsing.

### Day or month first

Across values in the day/month-first shape, some first field above 12 and no second field above 12 means day first; the reverse means month first.
Otherwise the order is undecided.
An undecided value is accepted only if both orders give the same date, such as `05/05/2024`; `02/01/2024` is refused with `text_time_order`.
Year-first values do not depend on that decision.
The relevant values are the fetched column for materialization, the preview for time-column selection, and nonempty text values within the first 100 sampled rows for inference scoring.
Day-first inference can emit `DATES_READ_DAY_FIRST`, using `%y` for a two-digit year.

### Provider and inference behavior

| Reading | Behavior |
|---|---|
| `TextTimes::ValueByValue` materialization | Fixed format list; mixed offsets or mixed zoned/unzoned values are refused with `text_time_zones`. |
| `TextTimes::ClientRules` and in-memory frame materialization | Same formats and order rules, but normalize mixed offsets to UTC. |
| Time-column selection | Preview normalized to UTC; invalid/ambiguous text is refused when the selection needs that column's first time. |
| Inference score | 0.3 for a time-like name plus the readable share of nonempty text values within the first 100 sampled rows; timestamp threshold 1.0. |

Inference compares the least-readable date ordering with the share readable in either ordering; a difference that changes the type decision is refused rather than guessed.
Values outside the supported formats do not contribute to the readable share.
They can therefore leave a column as ordinary text; mixed offsets are refused in this score path.
Under a name that does not suggest time, unshaped text such as a part code is not counted as a timestamp.
Materialization reports unsupported formats as `text_time`, ambiguous order as `text_time_order`, and incompatible offsets as `text_time_zones`, using `UNSUPPORTED_REQUEST`.
Unread blank time rows may be dropped with `TIME_VALUES_UNREAD`; null-key/time sanitization is a separate graph-preparation policy.

## The PQL literal rules

These retain the rule identifiers used in reader comments and regression cases.
They differ from text-column parsing: `01/02/2024` means January 2 in a query literal but is ambiguous as a text-column value.

| Rule | Meaning |
|---|---|
| D1 | Three digit runs with one repeated `/`, `-` or `.` separator; optional space followed by three `:`-separated time fields. |
| D2 | First field above 31 or at least three digits means year first; otherwise above 12 means day first; otherwise month first. |
| D4 | A day-first literal emits `DATE_LITERAL_DAY_FIRST` when its format can be named: four-digit final year, one/two-digit month and time fields. Years below 1000 warn only when the host reports zero-padded `strftime` years. |
| D5 | If no date field has three digits and the year is below 100, place it in the 100-year window around the host's current local year; use UTC when no local zone is supplied. Missing required clock facts cannot silently guess the year. |
| D6 | Hour 0–23, minute/second 0–59; no fraction or timezone suffix in this literal grammar. |
| D7 | Year zero is accepted only as first field `0000`, followed by one/two-digit month/day and, if present, one/two-digit time fields. |
| D8 | Any digit run of five or more is refused before parsing, with `INVALID_REQUEST` and `date_literal_packed`; diagnostics name the span, not the literal's content. |
| D9 | Impossible dates, two date fields of at least three digits, years above 9999 and values outside a declared host date-literal range are refused. |

A day-first warning is emitted once per literal in its validation run.
The current year is read per run, not cached at import time.
Historical packed-date and natural-language behavior is intentionally unsupported.

## Timezone and range ownership

The core resolves named zones through host-specified directories, then its bundled database; typed zoned timestamps already identify instants.
Named-zone handling is separate from the text-format list, which admits only UTC/GMT and explicit offsets.
Arrow conversion checks the range of the requested unit; the text reader's calendar range does not promise that every year fits signed nanoseconds.
PQL additionally honors `HostFacts.date_literal_range` where a host representation is narrower.

Regression coverage: [`text_times.rs`](../../rust/crates/sdm-engine/tests/text_times.rs), [`PQL date cases`](../../rust/crates/sdm-pql/tests/dates.rs) and [`materialization tests`](../../rust/crates/sdm-core/tests/materialize.rs).
Do not refresh expected vectors merely to make a dependency change pass.
