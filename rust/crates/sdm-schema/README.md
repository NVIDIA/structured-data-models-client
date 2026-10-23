# sdm-schema

The schema of the user's tables and its inference over Arrow.
Modules: `infer` (each column's data and semantic types, the primary key, the time column: `table`, `InferredTable`), `assemble` (the tables into one schema, inferred links, warnings, the consistency check), `edit` (checks and makes the user's edits), `catalog` (reads the key counts).
It reads the catalog answers in the dialect-neutral shape `sdm-data` produced, plus preview rows as Arrow.
Depends only on `sdm-contract`, for the schema and catalog types.
Warnings are codes and fields; each host words them.
Must not hold: SQL, a dialect's or provider's name, or I/O.
