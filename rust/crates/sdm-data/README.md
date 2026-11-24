# sdm-data

The engine-neutral data layer.
It holds `parts`, a request asked in parts of at most the keys its provider declares it takes in one request, and `timeseries`: aligned history expanded into ordinary relational input tables.
Every dialect's SQL, catalog reading and key-set binding is `sdm-sql`'s.
Must not hold: SQL, a dialect, a database driver or I/O.

```text
used by:     sdm-core
     |
sdm-data
     |
depends on:  sdm-contract
```
