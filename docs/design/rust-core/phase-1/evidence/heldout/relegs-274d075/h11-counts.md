# H11 counts, all legs: python-core and rust-host rerun at 274d075 (owner items 49, 50)

Sealed cases of heldout-run-0a8a741 (INPUTS_SHA256 unchanged); python-ref and python-old are the one-shot run's, unchanged.
python-core: local client wheel from 274d075; rust-host: sdm-contract-run --pre-given-specs from 274d075 (relegs-274d075/identity.txt, binaries.txt).

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
- new refusals of an input the reference accepts: 0 []
- outcome class differs from the reference: 8
  - h11.decimal.key-38-0-large: reference `refusal INTERNAL_ERROR`, rust-host `success`
  - h11.empty.entity-table: reference `refusal RuntimeError`, rust-host `refusal INVALID_REQUEST`
  - h11.key.composite-primary-key: reference `refusal INVALID_REQUEST`, rust-host `refusal UNSUPPORTED_REQUEST`
  - h11.key.explicit-tables-and-link: reference `refusal AttributeError`, rust-host `success`
  - h11.nested.list-of-list: reference `refusal ValueError`, rust-host `refusal UNSUPPORTED_REQUEST`
  - h11.nested.struct-on-fact-table: reference `refusal RuntimeError`, rust-host `refusal UNSUPPORTED_REQUEST`
  - h11.null.nan-and-infinity-target: reference `refusal INVALID_REQUEST`, rust-host `refusal UNSUPPORTED_REQUEST`
  - h11.task.ranking-without-candidates: reference `refusal INVALID_REQUEST`, rust-host `refusal UNSUPPORTED_REQUEST`

## python-old (the old client; an INTERNAL_ERROR equal to the reference's is its own crash, reproduced)
- panics / INTERNAL_ERROR / missing: 1 ['h11.decimal.key-38-0-large']
- new refusals of an input the reference accepts: 0 []
- outcome class differs from the reference: 0

Cases differing from the reference on either core runner: 8 ['h11.decimal.key-38-0-large', 'h11.empty.entity-table', 'h11.key.composite-primary-key', 'h11.key.explicit-tables-and-link', 'h11.nested.list-of-list', 'h11.nested.struct-on-fact-table', 'h11.null.nan-and-infinity-target', 'h11.task.ranking-without-candidates']
format-only core.inputs pairs (contract-22 spec converted under owner item 49): 3 [('h11.key.composite-primary-key', 'python-core'), ('h11.key.composite-primary-key', 'rust-host'), ('h11.key.explicit-tables-and-link', 'rust-host')] (inputs-format-only.txt); not outcome differences, not counted above.

H11 (E66(e)): any panic, any new refusal of an input the reference accepts, or more than 4 cases differing from the reference without an entry approved before the run = no-go.
Fill report.md's last column from register-at-run-start.yaml only; the reviewer checks it. The executor gives no verdict.
