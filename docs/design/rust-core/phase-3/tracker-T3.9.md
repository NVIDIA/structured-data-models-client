# T3.9: parity and CI cleanup

## Production corrections

Postgres zoned text uses canonical UTC on Python and Java under the explicit owner decision T39-OWNER-2026-10-24-UTC-TEXT.
The public materialize path previously emitted different range feature strings for the same instants on UTC and Los Angeles sessions.
The conversion preserves microseconds, nulls, empty and open ranges, and readable wall dates whose UTC year is outside Python's datetime range.
Scalar timestamp columns and timezone-free values retain their existing conversion, and caller sessions are unchanged.

Postgres floating seeds are bound as exact numeric values, repairing six float/integer key-pair materialize refusals.
Whole floats use their actual integer digits rather than a shortest decimal that can name a nearby integer.
Fractional seeds are not truncated, and integer columns are not rounded to double precision for matching.
This changes the binding representation of already admitted values; it does not relax entity-key admission.
The dialect matrix is regenerated canonically, changing only four PostgreSQL answers and preserving all 2,972 inputs and every other dialect.

Metadata refresh retains the accepted tier of a held big-integer column.
Fresh nullability, uniqueness and catalog facts remain current, while a later fetch wider than the held tier produces the established VALUE_OUT_OF_RANGE error.
Ordinary numeric-width compatibility remains unchanged.
The existing public DuckDB regression reproduced the lost refusal before the fix.

## Validation state

Regression tests precede each production change.
Only narrowly scoped suspected-bug reproductions and the canonical matrix writer ran during implementation.
Both independent approvals of the final integrated SHA are required before lint and package checks.
Full CI remains the phase-end check.
