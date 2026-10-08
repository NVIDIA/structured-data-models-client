# chrono and chrono-tz against the reference (T0.6)

A throwaway Rust probe answered the date operations the core will need with `chrono` and `chrono-tz`, and its answers were compared with the reference's Python answers for the same inputs.
The probe is kept outside the repository, at `~/.local/share/sdm-dbx/prototypes/chrono-probe/`: `gen_ref.py` writes the reference's answers, `src/main.rs` is the probe, and `compare.py` classifies the differences.
The rules being compared are in [date-rules.md](date-rules.md).

## Versions

| Component | Version |
|---|---|
| `chrono` | 0.4.45 (features `std`, `clock`) |
| `chrono-tz` | 0.10.4, built-in IANA database 2025b |
| `regex` | 1.13.1 |
| `serde_json` | 1.0.151 |
| Rust toolchain | cargo 1.97.0 |
| Reference | `rust-core-reference` at `70aa159`, Python 3.12.13, pandas 3.0.6, NumPy 2.5.3, dateutil 2.9.0.post0, pyarrow 25.0.1, no `pytz` or `tzdata` package |
| Host zone database (read by `zoneinfo`) | 2026c, macOS 26.7.1 arm64 |
| pandas 2 comparison | pandas 2.3.3, NumPy 2.3.5, pytz 2026.5, pyarrow 20.0.0, in a scratch environment |

`chrono-tz` 0.10.4 is the newest release; its database is chosen when the crate is built, not read from the host.

## Inputs

- The first attempt's date corpus, `tests/parity/fixtures/dates.json` on `archive/rust-core-attempt-1`: 185 cases over 1,004 distinct strings.
- 30 added strings: zone spellings (`UTC`, `GMT`, `EST`, `Europe/Berlin`, `+05`, `+0530`, `+05:30`), nanosecond fractions, 2038 and later instants, the pandas nanosecond bounds, the years 1 and 9999, and an impossible 29th of February.
- The 34 examples of the first attempt's PQL date-literal rules (`rust/crates/pql/DATES.md`), plus 11 added literals: two-digit years around the pivot, the nanosecond bounds, and day-first and impossible dates.
- 16 nanosecond and 8 millisecond values, including negatives, the 2038 second, and the int64 extremes.
- 18 zones (New York, Los Angeles, Berlin, London, Dublin, Kyiv, Kolkata, Tokyo, Lord Howe, Chatham, Apia, Santiago, Sao Paulo, Mexico City, Ciudad Juarez, Casablanca, Tehran, UTC) at six instants in each of 13 years from 1970 to 2200, including wall times inside spring gaps and autumn folds.
- 13 base dates (month ends, leap days, 2037 and 2038, 2100) with 16 `DateOffset` values, each added, subtracted and tripled.

## Operations and agreement

| Operation | What is compared | Agree / total |
|---|---|---|
| Reading one value | The UTC instant `pd.to_datetime([v], utc=True, format='mixed', dayfirst=d, errors='coerce')` gives, for `d` both ways, against the probe's reading of the accepted shapes | 1,843 / 2,068 |
| Reading a time column | The values and the read-or-refuse outcome of the client's `_to_time` over each corpus column, against the probe's reading with the order the column decides | 121 / 121 (38 read, 83 refused) |
| PQL literals | The value, refusal and day-first warning of `pd.to_datetime(literal)` | 43 / 45 |
| Nanoseconds | Seconds by floor division, the wire text with six fractional digits and `Z`, and the seconds back as a timestamp | 15 / 16 |
| Milliseconds | `pd.to_datetime(ms, unit='ms')` | 8 / 8 |
| Zone, from an instant | Local wall time and UTC offset (`tz_convert`) | 1,297 / 1,404 |
| Zone, from a wall time | The instant, or a refusal for a gap or fold (`tz_localize` with pandas' defaults) | 1,295 / 1,404 |
| `DateOffset` arithmetic | `base + offset`, `base - offset` and `base + 3 * offset`, months clamped to the month's end before days and times are added | 208 / 208 |

In all, 4,830 of 5,274 answers agree.
Every corpus column case also matches the first attempt's recorded answer, so the corpus is still the reference's behavior.
Month arithmetic with `checked_add_months` and `checked_sub_months` agrees everywhere, including the 31st into short months, the 29th of February into common years, and `months` combined with `days`.

## Mismatch classes

| # | Class | Count | Example | Expected or decision |
|---|---|---|---|---|
| M1 | pandas 3 reads year-first text as year, day, month under `dayfirst=True` | 225 single values | `2024-01-02` with `dayfirst=True` is the 1st of February in pandas, the 2nd of January in the probe | Expected: the time-column path never passes `dayfirst` for a year-first column, and the column reads all agree; the only other caller (`reads_as_a_date`) asks whether a value reads, not which date; proposed: the core always reads year-first as year, month, day, with no register entry since no public answer changes |
| M2 | PQL packed dates (rule D8) | 2 | `230203-06-09` is `2003-02-23T06:00:00-09:00` in pandas | Not a `chrono` limit: the probe does not model `dateutil`'s packed reading, and PQL literals come from the PQL team's crate (owner decision 6), so this goes to the PQL asks |
| M3 | int64 wrap at the smallest timestamp | 1 | `-9223372036854775807` ns (`pd.Timestamp.min`): pandas' seconds round trip wraps to `2262-04-11T23:47:16.709551616`, the probe gives `1677-09-21T00:12:43` | Decision: the core should not reproduce the wrap; it touches only the `Timestamp.min` sentinel that `packages/structured-data-models-relational-engine/src/sdm_relational_engine/rfm/inference/payload.py:469-471` on the reference compares against, so whether that comparison still sees the sentinel after the sampler's round trip needs a look in Phase 1 |
| M4 | No daylight saving time from 2100 on | 128 zone answers | New York on 2100-07-01 12:00 UTC: pandas 3 gives 08:00 at UTC-4, the probe 07:00 at UTC-5 | Decision: `chrono-tz` 0.10.4 expands each zone's rules only to 2099 (New York, Berlin and Lord Howe all show their last summer offset in 2099), while `zoneinfo` applies the zone's rule for every later year |
| M5 | Zone database version | 88 zone answers, all Africa/Casablanca, in every tested year from 2036 on | 2036-01-15 12:00 UTC is 12:00 (UTC+0) with the host's 2026c and pytz 2026.5, 13:00 (UTC+1) with `chrono-tz`'s 2025b; Python's `zoneinfo` over the `tzdata` 2025b package gives 13:00 too | Decision: which database the core uses, and whether its version is a pinned build fact or follows the host as today |

Within 2038 to 2099, outside Casablanca, the probe agrees with pandas 3 on all 816 zone answers.
The same 408 instant-to-local answers agree with pandas 2 (pytz) on 278, and with pyarrow's `local_timestamp` on 278: both keep the offset of the last transition their files list, which is standard time from November 2037 on.
So the register's dropped rule, "time zone rules ignored after 2037", is what pandas 2 and pyarrow do; the reference on pandas 3 already follows the rules after 2037, and so does `chrono-tz`, up to 2099.
The reference makes no `pyarrow.compute` zone call on its public paths (a search of the four packages finds none), so pyarrow's behavior reaches the reference only through a user's own data.

The nine wall times in a gap or fold that pandas refuses (`ValueError`) before 2100 (London and Dublin in 2036 and 2087, and Apia's skipped 30 December 2011) are also refused by `chrono-tz` (`LocalResult::None` or `Ambiguous`); the four in 2200 are not, by M4.

## What the probe did not cover

- The probe reads the accepted shapes with its own regular expressions and builds dates with `NaiveDate::from_ymd_opt`; `chrono`'s format strings were not used for parsing, since pandas' readings are not `strftime` formats.
- The client's refusal messages and which refusal applies were not compared, only read or refused.
- `dateutil`'s free-form reading, reached by per-row columns (see date-rules.md) and by PQL's packed literals, has no `chrono` counterpart.
- pandas 1.5 was not run.

## Decisions needed

1. M3: drop the int64 wrap, after checking the `Timestamp.min` sentinel in Phase 1.
2. M4: accept that zone rules stop after 2099 (a register entry next to the post-2037 one), or choose a zone library that applies the rule of a zone's last year indefinitely.
3. M5: the zone database's source and version: `chrono-tz`'s built-in copy (pinned, 2025b today) or the host's, recorded as a host fact either way.
4. M2 goes to the PQL asks; M1 needs no entry.
