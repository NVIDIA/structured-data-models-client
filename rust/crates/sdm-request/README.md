# sdm-request

Turns what was sampled into the model request, and the model's answer into result rows.
Modules: `payload` (`Sampled` in, `build` out to `sdm-wire`'s typed payload), `options` (run mode, statistic, inference options), `predict` (checks before the call, the call body, reading the answer), `ranking` (a ranking split into the questions the model serves and put back together).
It depends on `sdm-wire` for the payload types, on `sdm-engine` for sampled data, and on `sdm-contract`.
The boundary check would also allow `sdm-schema`, but nothing uses it today.
Must not hold: SQL, a provider match, HTTP, or the order of a run's steps (that is `sdm-core`).

```text
used by:     sdm-core
     |
sdm-request
     |
depends on:  sdm-engine, sdm-wire, sdm-contract
```
