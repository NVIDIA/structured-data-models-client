# Reviewer: standing rules (the same brief goes to both reviewers)

The orchestrator's brief names: the package ID, its design row and decision rows, the exact `BASE..CANDIDATE` range, the round number, the findings already known or in progress (skip them), and the report file to write.

**How to review**
- Review the fixed SHA, in a throwaway worktree or with `git show`, never the live lane worktree.
- Round 1 reviews the whole range. Later rounds review only the fix commits and rerun the earlier rounds' repros.
- Check code and behavior, not prose. Run the package's tests and the targeted gate checks. Try to break it: edge inputs, empty and missing evidence, the reference's behavior on the same input.
- Check `git log` to confirm no implementer commit changed a test author's file.
- Flag overengineering (a field, type, option or layer nothing uses yet) and any interface that would be awkward to express from another host language (Python, Java: plain data across the boundary, no language-specific idioms), per the owner's standing principle (Phase 2 decision KO2-46): P2 when it affects an interface, P3 otherwise.
- Every measured difference from the reference must have a register entry. You may approve an entry the brief says is reviewer-approvable (for example a reference crash or internal error becoming a named error) by writing `APPROVE-<entry id>` with your reasons; the rest go to the owner.
- Don't wait for CI. Report with CI's state as you find it; the orchestrator checks the exact-SHA run at landing.
- Run probes under a timeout with stdin closed: `perl -e 'alarm shift; exec @ARGV' 600 <cmd> </dev/null`. Never `gh run watch`, `rm -rf`, `--force` or `sudo`. Change nothing in the repository or on GitHub.

**The report** (`land.py` reads its first two lines)
- Line 1: exactly `APPROVE` or `CHANGES REQUIRED`.
- Line 2: the full candidate SHA.
- Then the findings, each tagged P1 (wrong results, a crash, a check that passes when it shouldn't), P2 (a real defect that's less severe, or a missing required test) or P3 (minor). Only P1 and P2 block.
- Each finding: a one-line issue with `file:line` and how to reproduce it, then a one-line suggested fix.
- Then one short paragraph on what you verified and how.
