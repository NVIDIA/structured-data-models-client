# Phase 4 decisions

This file records only the owner's own Phase 4 decisions.
The decisions in [Phase 1](../phase-1/decisions.md), [Phase 2](../phase-2/decisions.md) and [Phase 3](../phase-3/decisions.md) remain in force unless an actual later owner decision changes them.
Cite those rows by ID rather than copying their approvals into this phase.
A status `approved by owner <date>` records an answer the owner wrote; an agent never assigns that status to its own choice.
Implementation and reviewer decisions belong in [design.md](design.md) and package records.

| # | Decision | Owner's words | Status |
|---|---|---|---|
| T42-PANDAS-ADVICE-2026-10-26 | Keep only the existing pandas dates-as-text advisory and its presentation order in Python; Rust retains all data, schema and prediction rules, with no pandas parser emulation, semantic callback or data-driving metadata. | "Keep pandas advice in Python (recommended)" | approved by owner 2026-10-26 |
| T44-BOUNDARY-METHOD-2026-10-28 | Formally plan and implement the identified boundary correction and implementation-method repair, preserving the existing public behavior and review gates. | "You said there's a real boundary flaw? Plan and fix it. Same with the methodology failure. Plan this officially and then implement" | approved by owner 2026-10-28 |
