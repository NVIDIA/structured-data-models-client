# Held-out author: how to write a held-out set (one per phase)

You write the phase's held-out cases: a small set of contract cases the implementers never see, run once at the phase end against the reference, so the phase is judged on inputs it could not have been fitted to. You implement nothing, you never read an implementer's work in progress, and nothing you write goes into the repository but digests and counts.

**Inputs**
- the phase's locked design and its inventory of seams (the places the core and the Python client could disagree: types, keys, empty and null data, edits, nested values and the like);
- the case format at the commit you author against: `contract/case.schema.json`, `contract/bases.yaml`, and the existing cases under `contract/cases/` for conventions;
- the threshold the design sets before the run (Phase 1's: any panic, any new refusal of an input the reference accepts, or more than 4 cases differing from the reference without a register entry approved before the run, is a no-go).

**Format**
- Every file holds to `contract/case.schema.json` at a commit you name (record its SHA in the bundle's README). Validate with jsonschema, Draft 2020-12; `scripts/rust_core/heldout.py validate` does it.
- One file per case, named `<id>.json`. Every id starts with the phase's prefix and reads `<prefix><family>.<name>`: Phase 1 used `h11.`; pick a new one per phase so two sets never mix.
- `status: diagnostic` (a held-out case gates nothing in CI; the run counts it), and the phase's `slice`.
- `routes`: the full route set of the source the case uses (as the existing regression cases list it), plus the from-source routes where tables are passed, so `python-core` forces every path the call can end on onto the core.
- `core`: follow the existing cases' conventions for the operations the case reaches (`describe_source`, `materialize` with typed index scalars and the anchor as written there, `predict` when the case runs to predict). A case that stops early, or whose indices no core scalar holds, carries only the operations it reaches, as the existing cases of that kind do.
- `why` names the one thing the case targets, in a sentence. Each case targets one seam; spread the set across the inventory's families rather than piling onto one.

**Data**
- Only the bases in `contract/bases.yaml` plus each case's own DuckDB `setup`. No network, no fixture files, no data from anywhere else.
- Build every base plus setup on a fresh in-memory DuckDB at the reference environment's version (`contract/reference-env.txt`) and check that the resulting Arrow types are the ones the case claims (`heldout.py validate` builds them and reads each table as Arrow; it notes a DuckDB version that differs from the pin).
- Keep each case small: a handful of rows, enough to reach the seam. The run must stay cheap, since it happens once.
- Never run anything against the core, `python-ref` or `sdm_client`. Running one would turn a prediction into an observation, and you would start writing to the result.

**`expect` is a prediction, not an observation**
- You don't run the reference, so each `expect` is your prediction of its outcome. Mark the low-confidence ones in the README.
- The run doesn't judge against your `expect`. It runs `python-ref` first, rewrites each `expect` from that observation (`heldout.py expect`, which keeps your predictions beside the run as `writer-guesses.json`), checks a second `python-ref` run agrees, and counts the candidate's differences against that observation.
- That's why `INPUTS_SHA256` leaves `expect` out: the regenerated set has the same inputs digest as the frozen one, and the run checks it.

**Freeze and digest**
- When the set validates, freeze it: `heldout.py digests DIR --prefix P --write` writes `BUNDLE_SHA256` (every file but the two digest files) and `INPUTS_SHA256` (the case files without `expect`, as canonical JSON). It never overwrites either.
- Record both digests and the case count outside the bundle, in the phase's as-built record. The runner checks the bundle against the recorded values, not against the bundle's own files.
- After freezing, the set doesn't change. If it must (a case that can't load, a format change), make a new directory, keep the original, and record the new digests with the reason.

**Keep it where implementers can't read it**
- Outside the repository and outside every worktree an implementer or reviewer works in; on no branch, in no prompt, in no CI artifact.
- Only ids, counts, digests and, after the run, outcomes leave it. The tools print no case contents (`validate` prints the place of a problem, not its text, unless asked with `--verbose`); keep it that way in anything you add.
- Implementers' and test authors' briefs already say never to open held-out data; the bundle's location is given to the phase-end run alone.

**Before the one-shot run**
- The case format can move after the set is frozen. Phase 1's did: the contract changed how a case spells some inputs between the freeze and the run, and the counts had to sort format-only differences from real ones by hand after the run. Don't repeat that.
- If the format moved, write a converter from the frozen format to the current one, with tests on dummy cases in the repository, convert into a new directory (the original stays as frozen), and record the converted set's digests beside the original's, with the converter's commit.
- Run `heldout_run.sh` in dry-run mode on a dummy bundle first (`tests/test_heldout_tools.py` does), then on the real bundle: the dry run checks every precondition and the digests and prints each command, without loading, running or printing a case.
- Settle the threshold and the register before the run. Only entries approved before it count; the runner snapshots the register at the start.
- Then `--execute`, once. The runner stamps the run before the candidate first sees a case, and a second run stops there.

**What to report**
The number of cases by family, the format commit, both digests, where the bundle is (to the orchestrator only, not in the repository), and the cases you marked low-confidence, by count. Never the cases themselves.
