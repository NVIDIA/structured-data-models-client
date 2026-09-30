# Support

## Support level

**Maintained.** The client is developed and released by NVIDIA alongside the
structured data foundation model NIMs it calls. Issues and pull requests are
reviewed, and fixes ship in the next release of all five packages.

It is not a supported NVIDIA product under a commercial agreement. There is no
response-time commitment, and nothing here changes the warranty terms of the
Apache-2.0 licence the code is released under.

## Getting help

| What you have | Where to take it |
| --- | --- |
| A bug, or behaviour that contradicts the documentation | A GitHub issue |
| A question about usage | A GitHub issue with the `question` label |
| A feature request | A GitHub issue describing the problem, not only the proposed fix |
| A suspected security vulnerability | **Not GitHub.** Follow [SECURITY.md](SECURITY.md) |
| A problem with the NIM itself rather than the client | Your NVIDIA support channel for that NIM |

A bug report is most useful with the package versions (`pip show sdm-client
sdm-api sdm-pql sdm-connectors sdm-relational-engine`), the Python version and
platform, the smallest script that reproduces the problem, and the full
traceback. `StructuredDataError` and its subclasses carry a stable `code`
field; include it.

## Which package

Five packages are released together and share a version:

- `sdm-client`: the client, the schema, and the model handles
- `sdm-api`: the generated wire types for the Universal TFM API, and nothing else
- `sdm-pql`: the Predictive Query Language, with parser, AST and schema types
- `sdm-connectors`: reading source tables from warehouses and object stores
- `sdm-relational-engine`: the relational driver, graph building and sampling

If you are unsure which is at fault, file against the repository; triage is our
job, not yours.

## Versions

Only the latest release receives fixes. There are no long-term-support
branches. See [SECURITY.md](SECURITY.md) for the same statement as it applies to
security fixes.
