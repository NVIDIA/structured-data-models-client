# sdm-engine

The rules materialize and sampling follow, over Arrow that has already been fetched.
Modules: `time` (`Instant`, `Span`, civil dates, the `Zones` port, and the reading of text times), `random` (numpy-compatible `default_rng` and shuffle), `graph` (`Graph`, `Table`, `Edge`: the tables a request samples, built from `sdm-contract`'s `SourceDescription`), `sample` (examples, seeds, neighbor reads `by_pkey`, `by_fkey`, `by_time`), `labels` (label aggregation), `keys`, `words`.
It depends on `sdm-contract` for the described schema, types and errors.
It depends on `sdm-schema` only to implement its `infer::TextTimes` trait (`time/text.rs`), which is how inference reads text as times.
Must not hold: SQL, a provider match, I/O, or payload layout (that is `sdm-request`).

```text
used by:     sdm-request, sdm-core
     |
sdm-engine
     |
depends on:  sdm-schema (for infer::TextTimes), sdm-contract
```
