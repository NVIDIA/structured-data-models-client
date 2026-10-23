# sdm-engine

The rules materialize and sampling follow, over Arrow that has already been fetched.
Modules: `time` (`Instant`, `Span`, civil dates, the `Zones` port), `random` (numpy-compatible `default_rng` and shuffle), `graph` (`Graph`, `Table`, `Edge`: the tables a request samples), `sample` (examples, seeds, neighbor reads `by_pkey`, `by_fkey`, `by_time`), `labels` (label aggregation), `keys`, `words`.
Depends on `sdm-schema` for the described tables and on `sdm-contract` for types and errors.
Must not hold: SQL, a provider match, I/O, or payload layout (that is `sdm-request`).
