# sdm-schema

Schema inference over Arrow, the schema's assembly, and its edits.
The schema's types themselves (`SourceDescription`, `DescribedTable`, `DescribedColumn`) are `sdm-contract`'s, in `ops.rs`; this crate decides their contents.
Modules: `infer` (each column's data and semantic types, the primary key, the time column: `table`, `InferredTable`), `assemble` (the tables into one schema, inferred links, warnings, the consistency check), `edit` (checks and makes the user's edits), `held` (the schema as a host holds it, as plain data, read into the schema the core works on and written back), `catalog` (reads the key counts).
It reads the catalog answers in the dialect-neutral shape `sdm-data` produced, plus preview rows as Arrow.
Reading text as times is asked through the `infer::TextTimes` trait, which `sdm-engine` implements, so this crate needs no time library.
It depends only on `sdm-contract`, for the schema and catalog types.
Warnings are codes and fields; each host words them.
Must not hold: SQL, a dialect's or provider's name, or I/O.

```text
used by:     sdm-engine (implements infer::TextTimes), sdm-core
     |
sdm-schema
     |
depends on:  sdm-contract
```
