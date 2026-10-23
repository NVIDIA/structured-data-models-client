# sdm-request

Turns what was sampled into the model request, and the model's answer into result rows.
Modules: `payload` (`Sampled` in, `build` out to `sdm-wire`'s typed payload), `options` (run mode, statistic, inference options), `predict` (checks before the call, the call body, reading the answer), `ranking` (a ranking split into the questions the model serves and put back together).
Depends on `sdm-wire` for the payload types, on `sdm-engine` and `sdm-schema` for sampled data and tables, and on `sdm-contract`.
Must not hold: SQL, a provider match, HTTP, or the order of a run's steps (that is `sdm-core`).
