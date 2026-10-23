# T3.17: the core picks every refusal's and warning's sentence; each host fills it

Lane E.
Design: the Phase 3 structure design (round 3, 2026-10-23), section 2's fix B and invariant 8.
Started on the phase head 775b84be (T3.16 landed).

## Decision

| # | Decision | Amends | Source |
|---|---|---|---|
| T3.17-D1 | Sentences are shared; values and advice are spelled per host. The core picks one sentence of a closed list for every error and warning (`Sentence`, `sdm-contract/src/error.rs`), and each sentence has one template every host fills (`sdm-contract/src/wording.rs`, exported as `_native.templates()`). A host formats the values a template names with its one value formatter, spells each advice the template names (`Advice`) once in its own language, and says a sentence that relays a library of its own (pandas, NumPy, Python) in that library's words. No host chooses a sentence. | The migration plan's "presentation in the host" (section 3's table, "Object behavior, exceptions, presentation"), and E59(d)'s host-held templates and variants | The structure design's fix B, which names this row as T3.17's |

## Tests first

The `T3.17 tests` commit came first, each test failing on the phase head:

- `sdm-contract/tests/wording.rs` did not compile: `Sentence`, `WarningCode`, `wording` and `CoreError::new(sentence, message)` did not exist.
- `test_t317_wording.py`: `_native.templates()` and `errors.render` did not exist; the census's probes on the core route said "the request is valid, and not one the core supports yet" and "the request is not one the core accepts"; a primary key linked as a foreign key said the latter where the old client says "Cannot treat the primary key of table 'orders' as a foreign key. Remove either the primary key or the link before proceeding."
- The key-check advice and the grouped schema warnings passed on the head: they are guards that the move keeps the old client's words.

## What changed

| Where | Change |
|---|---|
| `sdm-contract/src/error.rs` | `Sentence` (294 sentences), `WarningCode`, `Code`; `CoreError` and `CoreWarning` carry their sentence, `CoreError::new(sentence, message)` takes the code's kind and the sentence's rule, `holds_to_table` checks the sentence is of the code. Contract v28's working entry rewritten (H4). |
| `sdm-contract/src/wording.rs` | `template(sentence)`, exhaustive; `Advice` (21) and the two groups; `table_json()`. The databases' names a sentence reads are `sdm_data::DATABASE_NAMES`, which `sdm-host` adds to the exported table. |
| Every crate | Each refusal and warning site names its sentence; the warning-code tables (`WARNINGS`, `warning_codes()`, `_native.WARNING_CODES`) are gone. |
| `_rust/errors.py` | The code-to-class table, the warning classes, the emission policy, the value formatter, `ADVICE`, `HOST_OWNED`; grouping from the table's data. The generic fallback, `WORDED`, `SAID`, `GROUPED` and `_derived` are gone. |
| `client.py`, `decompose.py` | The ranking's and the parts' refusals the core now words are no longer said again here; a ranking's refusal is said again from the core's sentence with the candidates as given. |
| `assemble.rs` `check_links` | The primary key linked as a foreign key says the old client's sentence (T3.16 review r4, P2-2). |

Register: P3E-5.

## Fix 1 (review round 1)

- The values only the host holds that a sentence names (a candidate as given, its type's name and its decimal written as the column's type, the question as written, the source described) are `{host.<name>}` placeholders, which the call in progress supplies (`errors.holding`); `decompose.py`, `client.py` and `_rust/schema.py` no longer choose or write a sentence.
- `num_context_samples` past the endpoint's most and below 1 say the old client's words, the value as the caller gave it; the shapes a model lacks read `temporal` as the reference's `required_capabilities` does (the task's options); several reserved columns are paired as the old client pairs them.
- No shared template spells a Python value: `None` and `'NaT'` are advice.
- The census's test compares each refusal the core says on DuckDB with the old client's words, each difference listed with its entry; P3E-5 lists every remaining difference and waits for the owner.

## Fix 2 (review round 2)

- The part of a question asked in parts a refusal happened in is the core's: `CoreError.part`, a sentence of the part kind (`part_forecast_step`, `part_forecast_step_outran`, `part_ranking_candidate`), said after the refusal's own, with `forecast_steps` beside `forecast_step`; the candidate as given is a host value. `decompose.py` and `client.py` keep only the details as the old route's; a refusal the Python host raises itself, which the core never saw, is still said of its part by `about_the_part`. Contract v28's working entry rewritten.
- The old client's census words are recorded in `contract/reference/census-words/duckdb.jsonl` by `contract/tools/census_words_record.py`, which the census's test compares with, so it runs wherever the suite does.
- `run_mode` names the size as given itself; a materialize refusal's `candidate` detail is the candidate as given, as predict's is.
