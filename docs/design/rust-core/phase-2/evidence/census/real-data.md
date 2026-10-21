# The two kept rules on real data (check 2)

Phase 2 keeps two of E92's refusal rules until Phase 4: `text_time_rules` and `model` (decision KO2-22).
Check 2 accepts a kept rule only under an owner answer that quotes its real-data count.
The owner pre-approved accepting both if each is hit by under 5% of real queries.

Both counts are zero by construction on DuckDB, SQLite and Postgres 17, for both real datasets (AdventureWorks at full size and RelBench H&M, as `scripts/live/real_data_benchmark.py` loads them).
Neither dataset was loaded: the counts follow from the code, not from the data (the orchestrator's decision, 2026-10-21).
A guard test holds the code to it, so the counts cannot go stale silently.

## Counts

| Dataset | Database | `text_time_rules` hits | `model` hits |
| --- | --- | --- | --- |
| adventureworks-full | DuckDB | 0, by construction | 0 of 3 tasks |
| adventureworks-full | SQLite | 0, by construction | 0 of 3 tasks |
| adventureworks-full | Postgres 17 | 0, by construction | 0 of 3 tasks |
| rel-hm | DuckDB | 0, by construction | 0 of 3 tasks |
| rel-hm | SQLite | 0, by construction | 0 of 3 tasks |
| rel-hm | Postgres 17 | 0, by construction | 0 of 3 tasks |

Both rules are under the owner's 5% condition on every database.

## Why `text_time_rules` is never hit

The core refuses `text_time_rules` only for a time column that arrives as text from a provider that declares it reads text times by the client's rules (`TextTimes::ClientRules`): `rust/crates/sdm-core/src/materialize.rs:1047` and `:1163`.
Every SQL provider declares the other reading, each value on its own (`TextTimes::ValueByValue`): `rust/crates/sdm-data/src/duckdb.rs:156`, `rust/crates/sdm-data/src/sqlite.rs:180` and `rust/crates/sdm-data/src/postgres.rs:184`.
No provider in the tree declares `ClientRules`; it is named only by its definition (`rust/crates/sdm-contract/src/data.rs:665`) and the core's two refusals.
So on these three databases the rule cannot fire, whatever the data holds.

Exposure, not a hit: SQLite has no time type, so the contract loader stores every time as ISO 8601 text (`contract/runners/databases.py`, `sqlite_value: iso`), and the SQLite provider reads that text value by value; on DuckDB and Postgres the benchmark's time columns are typed (AdventureWorks' dates load as timestamps, H&M's are parquet timestamps).

## Why `model` is never hit

The core refuses `model` only for a request to the tabular model (`KumoTabular`): `rust/crates/sdm-request/src/predict.rs:465`.
That is a property of the call, not of the data.
The real-data benchmark's six tasks (three per dataset, `TASKS` in `scripts/live/real_data_benchmark.py`) ask every question of `KumoRelational` (`scripts/live/real_data_benchmark.py:766`), so none is a tabular request.

## The guard

`tests/test_census_real_data.py` fails when any provider in the tree (`rust/crates/*/src`, `rust/bindings/*/src`, `packages/*/src`) names `ClientRules` or `client_rules` beyond the definition and the core's two refusals, when a SQL provider stops declaring `ValueByValue`, or when the benchmark asks the tabular model.
If it fails, these counts are to be measured again.

## Commands that regenerate this

    python -m pytest -q tests/test_census_real_data.py
    git grep -n 'ClientRules\|client_rules' -- rust packages
    git grep -n 'text_times: Some(TextTimes::' -- rust/crates/sdm-data/src
    git grep -n 'not_yet("text_time_rules")\|not_yet("model")' -- rust/crates
    git grep -n 'KumoTabular\|model=KumoRelational()' -- scripts/live/real_data_benchmark.py
