# T2.1d: the judging path T2.3 needs

On T2.1b's head on lane d (60b07bbe), before T2.1b lands; the judging-path changes T2.3 found, moved out of T2.3 so the gate judges T2.3 with a checkpoint that holds them (the orchestrator's decision).
Each change has one focused test, which fails before it.

## What changed

- **`more_exact` takes a database's line key** (`register.schema.json`, `register.py`): `<case>@sqlite` and `<case>@postgres`, as a case outcome has since T2.0f, the case part one of the entry's cases; `compare.py` already looks the fields up by the line's key (`test_t21d_more_exact_database.py`).
- **The comparator reads a Postgres row read's types** (`compare.postgres_arrow_name`): each type name psycopg's description gives, as the Arrow type the Python host's Postgres provider builds; `timestamptz` as `timestamp[us, tz=UTC]`, since the provider builds it in UTC whatever the session's zone (`test_t21d_postgres_row_types.py`).
- **An interval has one canonical cell** (`run_python.canonical_cell`): pyarrow's `MonthDayNano`, as the Postgres provider hands an interval over, is `["interval", months, days, nanoseconds]`, the cell Arrow gives (`test_t21d_interval_cell.py`).
- **rust-host's Postgres replay** (KO2-20, `sdm-testkit/src/replay.rs`, `sdm-contract-run --replay`): each fetch answered from a recording in order, a fetch other than the one recorded failing the run by name; the statements recorded as run, so rust-host's query log and driver record are complete; the test host lists Postgres among its providers (the replay's unit test).
- **python-core's recording** (`run_python.py --record-fetches`): each fetch with its answer and the statements the driver ran for it (`test_t21d_record_fetches.py`); in CI an artifact handed from the Postgres python-core job to rust-host's (T2.3b), never committed.

## Probes

- With T2.3's product, all 86 cases on Postgres 17: python-core recorded, rust-host replayed every fetch as recorded, and the comparator found no difference between the two hosts' core fields.

## Deviations

- The replay lives in `sdm-testkit`, the rust-host runner, so it is judging and lands here; T2.3 brings the Postgres dialect it answers for.
