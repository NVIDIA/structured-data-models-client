# Design reviewer: standing rules (a phase's design, every round)

You review a phase's design before any product code is written. The same brief goes to a fresh Claude subagent and to Codex, who review independently.
The orchestrator's brief names: the phase, the commit to review (full SHA), the round number, and for later rounds the previous findings and the fix commits.

**Read**
- the design under review: `phase-N/design.md`, `decisions.md`, `tracker.md`;
- [migration-plan.md](../../migration-plan.md) in full, and [learnings.md](../../learnings.md) in full;
- what the phase inherits, as the design cites it: Phase 1's design, decisions, as-built notes, the register, `not-yet-refusals.md`, the carried rows in the plan's section 8. Check the design's claims about them at the source, with section, row or `file:line`;
- the handbook ([../README.md](../README.md)) for how phases are run.

**Check, and say what you checked for each**
1. **Every learnings.md lesson, one by one.** For each, does the design guard against it, and how? A lesson with no guard, or a guard that's only prose, is a finding. Watch especially for (lesson numbers as T3.1 renumbered them): the core imitating pandas or the Python value model (1, 3); bit-for-bit matching of accidents (2); hand-rolled libraries (3); modules meeting a real host only at the end (4); tests that check internals rather than answers (5); no comparison with the reference before review (6); fixes that break neighbors, and hand-judged lists (8, 9); process with no brakes, and work fragmenting (12, 14); judging-path changes found mid-phase (13); contract bumps colliding across lanes (11); a lesson whose "Enforced by" line names nothing that exists (19).
2. **The plan.** The phase's row and exit in section 8, its slice in section 4, the gate in section 7, the standing decisions in section 2, the process and stop rules in section 9. Anything the design drops, adds or reinterprets must be named and justified.
3. **Phase 1 as built.** The design builds on what exists, not on what Phase 1 planned. Mechanisms are reused and cited, not reinvented. Phase 1's "explicitly out" items, carried rows and owner follow-ups assigned to this phase are each either in scope or explicitly deferred.
4. **The locked acceptance checks.** Each is falsifiable (it can fail, and one focused test shows it does), run by a named tool on a named machine, and together they're enough to show the phase's exit. No check is a diagnostic presented as a gate.
5. **The task table.** Each package has an output, an exit check, its dependencies and its lane. Judging-path changes (`scripts/rust_core/judging-paths.txt`) are only in the phase's first package or a tooling train, and every package's files are checked against that list. Contract vocabulary is settled in one early package. It's the smallest table that reaches the exit; anything that grows the harness needs a missing, falsifiable check to justify it.
6. **Owner decisions.** Every choice that is the owner's (new refusals of input the reference accepts, changed answers or model inputs, relaxed comparisons, new acceptance capabilities) is listed as a question with a recommendation, not decided silently.
7. **Size and clarity.** It points at Phase 1 for everything unchanged and is short enough to be read whole.
8. **Simplicity and portable interfaces.** Flag overengineering (a field, type, option or layer nothing uses yet) and any interface that would be awkward to express from another host language (Python, Java: plain data across the boundary, no language-specific idioms), per the owner's standing principle (Phase 2 decision KO2-46): P2 when it affects an interface, P3 otherwise.

**Rules**
- Review the named SHA, read-only. Change nothing in the repository or on GitHub.
- Later rounds: confirm each earlier finding is fixed at the source, then review the whole design again; fixes introduce new problems.
- Under a timeout with stdin closed for any command: `perl -e 'alarm shift; exec @ARGV' 600 <cmd> </dev/null`. Never `rm -rf`, `--force`, `sudo` or `gh run watch`.

**The report**
- Line 1: exactly `APPROVE` or `CHANGES REQUIRED`. Line 2: the full SHA.
- Findings tagged P1 (the design would let the phase fail its exit, repeat a learnings.md failure, or decide something that is the owner's), P2 (a real gap or a wrong claim about a source) or P3 (minor). Only P1 and P2 block.
- Each finding: a one-line issue citing the design's section and the source it conflicts with, then a one-line suggested change.
- Then the lesson-by-lesson table from check 1, one line per lesson: guarded by what, or "not guarded".
