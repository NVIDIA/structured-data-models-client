cases: 47; legs: 140 (duckdb 47, sqlite 46, postgres 47); gating DIFFERS lines in compare.txt: 0 (held-out cases are not baselined, so outcome differences are counted here from the observations)

## python-core
- legs: 140 (duckdb 47, sqlite 46, postgres 47)
- panics / INTERNAL_ERROR / missing: 0 []
- new refusals of an input the reference accepts: 0 []
- outcome class differs from the reference: 0

## python-old (the old client: reported, not counted)
- legs: 47 (duckdb 47)
- panics / INTERNAL_ERROR / missing: 0 []
- new refusals of an input the reference accepts: 0 []
- outcome class differs from the reference: 0

## rust-host
- legs: 140 (duckdb 47, sqlite 46, postgres 47)
- panics / INTERNAL_ERROR / missing: 0 []
- new refusals of an input the reference accepts: 0 []
- outcome class differs from the reference: 12
  - h2.calls.last-of-three-calls-fails: reference `refusal NimRequestError`, rust-host `refusal INVALID_RESPONSE` (approved before the run)
  - h2.calls.last-of-three-calls-fails@sqlite: reference `refusal NimRequestError`, rust-host `refusal INVALID_RESPONSE` (approved before the run)
  - h2.calls.last-of-three-calls-fails@postgres: reference `refusal NimRequestError`, rust-host `refusal INVALID_RESPONSE` (approved before the run)
  - h2.calls.only-call-fails: reference `refusal NimRequestError`, rust-host `refusal INVALID_RESPONSE`
  - h2.calls.only-call-fails@sqlite: reference `refusal NimRequestError`, rust-host `refusal INVALID_RESPONSE`
  - h2.calls.only-call-fails@postgres: reference `refusal NimRequestError`, rust-host `refusal INVALID_RESPONSE`
  - h2.forecast.second-step-call-fails: reference `refusal NimRequestError`, rust-host `refusal INVALID_RESPONSE` (approved before the run)
  - h2.forecast.second-step-call-fails@sqlite: reference `refusal NimRequestError`, rust-host `refusal INVALID_RESPONSE` (approved before the run)
  - h2.forecast.second-step-call-fails@postgres: reference `refusal NimRequestError`, rust-host `refusal INVALID_RESPONSE` (approved before the run)
  - h2.rank.second-candidate-call-fails: reference `refusal NimRequestError`, rust-host `refusal INVALID_RESPONSE`
  - h2.rank.second-candidate-call-fails@sqlite: reference `refusal NimRequestError`, rust-host `refusal INVALID_RESPONSE`
  - h2.rank.second-candidate-call-fails@postgres: reference `refusal NimRequestError`, rust-host `refusal INVALID_RESPONSE`

On either core runner (python-core, rust-host), on every database (a case counts once, however many of its legs differ):
- panics: 0 []
- new refusals with no approval: 0 []
- cases differing from the reference: 4 (12 legs); with no approval: 2 ['h2.calls.only-call-fails', 'h2.rank.second-candidate-call-fails']
- threshold: more than 4 differing cases with no approval, any panic, or any new refusal with no approval = no-go

counts: within the threshold
