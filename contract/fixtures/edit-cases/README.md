# Edit cases

Eight cases in the case format (`sdm-case/1`) that make the schema edits
T1.24b added to the format (`set_primary_key`, `set_time_column`,
`set_stype`, `set_relationships`): seven the reference refuses at the
assignment and one whose edits it takes, then materializes (decision E67).
`reference.jsonl` holds their reference observations, recorded on
`python-ref` under T1.13's rules from the reference checkout at b413e9a.

They are kept here rather than in `contract/cases` because the gate judges
a candidate by its checkpoint's harness, whose case loader does not take
the new edits until the package that adds them has landed. Held by
`contract/tests/test_edit_cases.py` (python-core, against the reference
and `contract/pinned/edits.core.json`) and `sdm-testkit`'s
`tests/edit_cases.rs` (rust-host, against the same pin). A later package
moves them, with their reference lines, into `contract/cases` and the
reference file once the format is the checkpoint's.
