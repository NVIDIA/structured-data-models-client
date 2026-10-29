# Phase 4 decisions

This file records only the owner's own Phase 4 decisions.
The decisions in [Phase 1](../phase-1/decisions.md), [Phase 2](../phase-2/decisions.md) and [Phase 3](../phase-3/decisions.md) remain in force unless an actual later owner decision changes them.
Cite those rows by ID rather than copying their approvals into this phase.
A status `approved by owner <date>` records an answer the owner wrote; an agent never assigns that status to its own choice.
Implementation and reviewer decisions belong in [design.md](design.md) and package records.

| # | Decision | Owner's words | Status |
|---|---|---|---|
| T42-PANDAS-ADVICE-2026-10-26 | Keep only the existing pandas dates-as-text advisory and its presentation order in Python; Rust retains all data, schema and prediction rules, with no pandas parser emulation, semantic callback or data-driving metadata. | "Keep pandas advice in Python (recommended)" | approved by owner 2026-10-26 |
| T4-PANDAS-WARNINGS-2026-10-29 | Pandas warnings generally and their warning-related effects do not block Phase 4; this includes native pandas/pyarrow conversion warnings across classes, runtime cells and routes, rather than only the measured Pandas4Warning case list. Retain truthful raw receipts and focus acceptance on actual requests, results, substantive failures and architecture. No additional warning-effect parity probes or artificial conversions are required. | "Idk what this means. Don't overindex on warnings/errors. We don't care about pandas warnings in general / and what impact they have, since they're just warnings" | approved by owner 2026-10-29 |
