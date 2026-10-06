# structured-data-models-api

The Universal TFM API as the NVIDIA Structured Data Models client speaks it:
generated bindings and the wire vocabulary, and nothing else.

This package has **no third-party dependencies**. Anything that only needs to
know the shape of a request (a NIM, a test harness, a future model engine)
can depend on it without pulling in a parser runtime or pandas.

Everything under `sdm_api.generated` is produced from the canonical contract in
`structured-data-api` by `scripts/generate_sdm_api.py`. Do not edit it by hand;
regenerate. `generated/PROVENANCE.json` records the contract revision and the
digests of both the spec and the generated file, and a test asserts they agree.
