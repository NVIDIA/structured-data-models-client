# Phase 2 tracker: T2.21

The design branch owns `tracker.md`; this file holds T2.21's section until it folds in.
Lane g, on the phase head 1d85bce3 (after T2.9's landing), then the test author's acceptance tests (`T2.21 tests:`, from `t221-tests`), then this package (`T2.21:` commits).

## T2.21

Exact numeric keys follow the effective keys and relationships, from the mid-phase design check (codex-reports/T2.0-mid-r1.md; learnings lesson 19, design choice 5, KO2-35).

### What changed

- A Postgres `numeric` declared without a precision is handed over as its exact text, marked `sdm:exact_number`.
  The core still reads the table for inference by the role its catalog declares, as the reference does: a declared primary, unique or foreign key as text, any other column as a double (`sdm-core/src/exact_number.rs`, `settle`).
- The description records which columns are such numbers, and whether the catalog declares each a key (`DescribedColumn.exact_number`, contract v27, the phase's working entry).
- The schema then reads each by its role in the schema as it stands, at `describe_source`, at every edit and at `materialize` (`sdm_schema::assemble::read_exact_numbers`, called by `mark_keys` and before a given or edited link is checked).
  A column is its text where its catalog declares it a key, or where the schema keys it or joins it and every column it joins to is text too, both ends of a link decided together.
  Otherwise it is a double, a feature included.
  A link to an integer column keeps the reference's reading, and links inferred by name are judged by the reference's types, so no link is added.
- The Python host stores the data type the core answers after an edit (`_rust/edits.py`, `_store`); it decides nothing of it.

### Register

- P2G-2 (KO2-78, the owner's): a key or link the caller gives or edits on such a column is answered where the reference refuses it (`pk.given`, `fk.given`, `fk.edited`).
- P2G-3 (KO2-35, the orchestrator's decision under it): on Postgres a declared `numeric(p,s)` key is int64 by its declared type, where the reference uses its text.
  The id is exact on both; the dtype differs in the public schema's two key columns, the request's three key columns and the result's id column, with the result's digest.

### Round 1

- Codex P2 and Claude P2-1: python-old's baselines for the four `key.past-2-53.*` cases (fix 1).
- Codex P3: `contract/tests/test_t221_exact_keys.py` runs in the Postgres vector step of every Postgres cell, with `SDM_REQUIRE_POSTGRES` (fix 1).
- Claude P2-2: P2G-3 listed only the request's dtype fields.
  Fix 2 adds the public schema's and the result's, found by comparing python-core's run of the four cases on Postgres with the reference line field by field; rust-host, replaying python-core's recorded fetches, has no public schema or result and shows no difference.
- Claude P3-2, noted, not changed: when a key the caller gave is moved by an edit to another column, the reference keeps the link that joined the old key, and so does the core.
  The old key's column then stays text in the core, since a link still joins it, where the reference shows it as a double.
  This follows the rule as written ("keyed or joined"), and both clients refuse the follow-up materialize the same way.
- Claude P3-1, for the test author: the per-case python-core test compares the outcome and the payloads only, not `public.schema` or `public.result`, which is why P2-2 passed locally.
