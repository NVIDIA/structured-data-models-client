# H11 counts, all legs (rust-host leg completed under owner item 49)

Candidate 0a8a74125d8372b9bc71363239322a785430cc5c; 42 held-out cases; register snapshot register-at-run-start.yaml (4 owner-pending entries open at run start: P1B-80, P1B-86, P1B-87, P1B-89; all four since decided by the owner, items 45-48, AFTER the run start, so under H11 they do not count as approved before the run).
python-ref, python-core and python-old ran once, starting 2026-10-15T14:02:54Z (the one-shot run; identity.txt). rust-host failed then (runner failure on h11.key.composite-primary-key: core.describe_source.source is not a SourceSpec; no observations) and was completed alone under owner item 49 with sdm-contract-run --pre-given-specs (see rust-host-leg.md).

gating DIFFERS lines (compare.py, all four legs): 0; held-out cases are not baselined, so outcome differences show as diagnostic lines and are counted here from the observations.

## python-core
- panics / INTERNAL_ERROR / missing: 0 []
- new refusals of an input the reference accepts: 0 []
- outcome class differs from the reference: 6
  - h11.decimal.key-38-0-large: reference `refusal INTERNAL_ERROR`, python-core `success`
  - h11.empty.entity-table: reference `refusal RuntimeError`, python-core `refusal INVALID_REQUEST`
  - h11.key.composite-primary-key: reference `refusal INVALID_REQUEST`, python-core `refusal UNSUPPORTED_REQUEST`
  - h11.nested.list-of-list: reference `refusal ValueError`, python-core `refusal UNSUPPORTED_REQUEST`
  - h11.nested.struct-on-fact-table: reference `refusal RuntimeError`, python-core `refusal UNSUPPORTED_REQUEST`
  - h11.null.nan-and-infinity-target: reference `refusal INVALID_REQUEST`, python-core `refusal UNSUPPORTED_REQUEST`

## rust-host
- panics / INTERNAL_ERROR / missing: 0 []
- new refusals of an input the reference accepts: 1 ['h11.task.forecast-sum']
- outcome class differs from the reference: 9
  - h11.decimal.key-38-0-large: reference `refusal INTERNAL_ERROR`, rust-host `success`
  - h11.empty.entity-table: reference `refusal RuntimeError`, rust-host `refusal INVALID_REQUEST`
  - h11.key.composite-primary-key: reference `refusal INVALID_REQUEST`, rust-host `refusal UNSUPPORTED_REQUEST`
  - h11.key.explicit-tables-and-link: reference `refusal AttributeError`, rust-host `success`
  - h11.nested.list-of-list: reference `refusal ValueError`, rust-host `refusal UNSUPPORTED_REQUEST`
  - h11.nested.struct-on-fact-table: reference `refusal RuntimeError`, rust-host `refusal UNSUPPORTED_REQUEST`
  - h11.null.nan-and-infinity-target: reference `refusal INVALID_REQUEST`, rust-host `refusal UNSUPPORTED_REQUEST`
  - h11.task.forecast-sum: reference `success`, rust-host `refusal UNSUPPORTED_REQUEST`
  - h11.task.ranking-without-candidates: reference `refusal INVALID_REQUEST`, rust-host `refusal UNSUPPORTED_REQUEST`

## python-old (the old client; its INTERNAL_ERROR on h11.decimal.key-38-0-large is the reference's own crash, reproduced)
- panics / INTERNAL_ERROR / missing: 1 ['h11.decimal.key-38-0-large']
- new refusals of an input the reference accepts: 0 []
- outcome class differs from the reference: 0

Cases differing from the reference on either core runner: 9 ['h11.decimal.key-38-0-large', 'h11.empty.entity-table', 'h11.key.composite-primary-key', 'h11.key.explicit-tables-and-link', 'h11.nested.list-of-list', 'h11.nested.struct-on-fact-table', 'h11.null.nan-and-infinity-target', 'h11.task.forecast-sum', 'h11.task.ranking-without-candidates']
format-only (contract-22 spec converted under owner item 49) core.inputs lines: 3 [('h11.key.composite-primary-key', 'python-core'), ('h11.key.composite-primary-key', 'rust-host'), ('h11.key.explicit-tables-and-link', 'rust-host')] (inputs-format-only.txt). These are not outcome differences; they are listed apart and are not counted above.

H11 (E66(e)): any panic, any new refusal of an input the reference accepts, or more than 4 cases differing from the reference without an entry approved before the run = no-go.
Fill report.md's last column from register-at-run-start.yaml only; the reviewer checks it. The executor does not judge the verdict here.
