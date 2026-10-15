# H11: the reviewers' classification and verdict

Hand-written from both reviewers' final classifications; the generated
`relegs-274d075/report.md` is left as the run wrote it, its last column
blank. This file fills that column: for each case whose outcome class differs
from the reference on either core runner, the entry approved before the run
that states it, read from the run-start register (`register-at-run-start.yaml`,
SHA-256 in [register-at-run-start.sha256](register-at-run-start.sha256)).

Sources: the independent subagent's and Codex's classifications of the
Python legs (H11-PYTHON-PASS, both), of the rust-host leg under the owner's
item 49 (H11-FAIL, both, on `h11.task.forecast-sum` alone), and their final
classifications at 8295049c after the owner's item 50 moved forecasts into
the core (H11-PASS, both). The final evidence at 274d0755
([relegs-274d075/](relegs-274d075/), NOTES.md item 8) has the same outcomes on
identical core code, so the verdict carries to it.

## The cases that differ from the reference

| case | runners differing | entry approved before the run |
|---|---|---|
| h11.decimal.key-38-0-large | python-core, rust-host | P1B-33 (owner) |
| h11.empty.entity-table | python-core, rust-host | P1B-11 (owner; E66(b), E97) |
| h11.key.composite-primary-key | python-core, rust-host | E92's Phase 2 class through P1B-2, rule `composite_key` (owner) |
| h11.nested.list-of-list | python-core, rust-host | P1B-51 (owner; E93) |
| h11.nested.struct-on-fact-table | python-core, rust-host | P1B-10 (owner; E66(b)) |
| h11.null.nan-and-infinity-target | python-core, rust-host | E92's Phase 2 class through P1B-2, rule `float_nan` (owner) |
| h11.task.ranking-without-candidates | rust-host | E92's Phase 2 class through P1B-3, rule `problem_type` for a ranking (owner) |
| h11.key.explicit-tables-and-link | rust-host | **none: unapproved.** The reference and python-core fail with the reference's own AttributeError before any core call (the case passes `tables` as a list); rust-host is given the case's normalised core input and succeeds. Both reviewers keep it as one unapproved difference |

The other cases with compare lines in `relegs-274d075/report.md` differ only
in diagnostic lines (SQL text and bindings as written, host-drawn request ids,
sampled statement results, and the contract-22 input conversion of item 49,
listed apart in `inputs-format-only.txt`); both reviewers found each core
runner's public outcome equal to the reference's there, so none counts
against H11.

## Verdict

**H11-PASS**, by both reviewers, at 8295049c, carried to 274d0755 with
identical outcomes:

- panics (`INTERNAL_ERROR` from a core runner, or a runner failure): 0;
- new refusals of an input the reference accepts: 0;
- cases differing without an entry approved before the run: 1, against H11's
  no-go threshold of more than 4.

**Disclosure: `h11.task.forecast-sum`.** On the one-shot run's rust-host leg
(owner item 49, at 88ccc265) the core refused this forecast
(`UNSUPPORTED_REQUEST`, rule `problem_type`), a new refusal of an input the
reference accepts, and both reviewers ruled H11-FAIL on it. The owner chose a
fix over an exception (item 50): T1.36 fix 21 moved forecast decomposition
into the core, and the two core legs were rerun on the same sealed cases,
unchanged and not inspected (`relegs-f744de6/`, `relegs-8295049/`,
`relegs-274d075/`). From 8295049c the case succeeds on every runner, as the
reference does. The forecast was therefore seen by the core in more than one
run of the bundle; this is why the bundle is spent ([../heldout.md](../heldout.md)).
