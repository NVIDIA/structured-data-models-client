# Owner decisions

Every decision the owner makes, newest last.
The plan's section 2 holds the standing decisions; this file records how and when they were made, and any made during Phase 0.

| # | Decision | Where it applies |
|---|---|---|
| 1 | Python is the main interface and comes first; Java follows and never holds Python back | Plan section 2, decision 1 |
| 2 | Python keeps every backend it supports today; Java starts with DuckDB, SQLite, Postgres and Databricks | Decision 2 |
| 3 | Refusing an input the old Python client accepted is a regression (Java on Snowflake not held to this yet) | Decision 3 |
| 4 | Quirks follow common sense with a small bias towards today's Python | Decision 4, section 5 |
| 5 | The reference is `rust-core-reference`: `main` plus only the minimum determinism fixes, pushed to GitHub, compared locally | Decision 5, T0.3 |
| 6 | PQL comes from the PQL team's C++ core through its Rust crate; the hand-written parser is a stopgap | Decision 6 |
| 7 | The data layer is swappable; a Pyramid provider is not in scope | Decision 7 |
| 8 | Java's first consumer brings its own data layer; Java loads the core as a native library; no WebAssembly | Decision 8 |
| 9 | No calendar budget and no time estimates | Decision 9 |
| 10 | This team owns the core long term | Decision 10 |
| 11 | Work lives on the epic branch `rust-core`; each phase on its own branch, pushed, then merged locally once verified; no pull requests | Decision 11 |
| 12 | The owner is the only human approver; a fresh agent is the independent reviewer | Decision 12 |
| 13 | The first attempt is archived on GitHub as `archive/rust-core-attempt-1`, as one squashed snapshot commit with no earlier history | Decision 13, T0.2 |
| 14 | Phases are named Phase 0 to Phase 5; each has a design doc and tracker in the branch | Plan section 9 |
| 15 | The old engine is deleted when the epic is ready to merge, with no switch-back release | Plan section 7 |
