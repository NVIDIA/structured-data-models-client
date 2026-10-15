# T1.35 part 2: run notes (read before the generated files)

Phase commit `0a8a74125d8372b9bc71363239322a785430cc5c` (landing of T1.36 part 2; Base-SHA and checkpoint `c505788f`).
The checkpoint's runner files equal the phase commit's, so the gate's observations are from a clean tree; no empty evidence commit was needed.

1. **Locked gates on the exact final SHA 0a8a7412: 1a and 1b both passed** (`final_gate.sh ... 38746039117 --execute`; `gate/final-gate-1a-1b-summary.txt`).
   - CI run 38746039117 (rust-core-phase-1, 0a8a7412) completed green before the gate.
   - 1a: checks 1, 2, 3 and 6 ok. Seeded suite ok. Wheel-smoke ok in both lanes, on CI's native-wheel and native-wheel-test-ops from run 38746039117, whose identity commit is 0a8a7412.
   - 1b: checks 4, 5 and 6 ok.
   - Both halves were judged by c505788f's harness in ref-pandas3. The one evidence JSON `<git-common-dir>/sdm-gate-evidence/gate/0a8a74125d8372b9bc71363239322a785430cc5c.json` holds both halves as passed. Its SHA-256 is in `gate/evidence-json.sha256`; it stays outside the tree.
   - The earlier T1.35 `--locked 1b` run (`gate/gate-1b-summary.txt`) also passed. Its observations are the ones in this pack.
   - python-core and python-old were also run in dbr17-pandas2 and floor-pandas2.2 (76 cases each, clean tree). `cases.md`: 25 correct success, 13 approved improvement, 3 expected refusal, 0 regression.
2. **Four owner-pending register entries were open at the held-out run's start:** P1B-80, P1B-86, P1B-87, P1B-89 (`heldout/proposed-at-run-start.txt`). Under H11 they do not count as approved before the run. The run used `--allow-proposed`.
3. **H11: the rust-host leg is now complete (owner item 49).**
   - The one-shot run (42 cases, bundle digests verified) recorded python-ref, python-core and python-old. rust-host stopped then: the bundle predates contract 23's SourceSpec shape, and the test host refuses that shape by design.
   - Under owner item 49, only the rust-host leg was run afterwards, with `sdm-contract-run --pre-given-specs` from t136-ck 88ccc265, which differs from 0a8a7412 in sdm-testkit only (`heldout/rust-host-leg.md`). Nothing else was rerun and no case was changed (INPUTS_SHA256 is unchanged).
   - Counts (`heldout/h11-counts.md`). Gating differences on all legs: 0.
     - python-core: 0 panics, 0 new refusals, 6 cases whose outcome class differs from the reference.
     - rust-host: 0 panics, **1 new refusal (h11.task.forecast-sum: reference success, rust-host UNSUPPORTED_REQUEST; python-core succeeds)**, 9 cases differing.
     - 9 distinct cases differ on either core runner.
   - Format-only `inputs` lines (the contract-22 spec converted) are listed apart in `heldout/inputs-format-only.txt` and `report.md`.
   - The verdict needs `report.md`'s last column filled from `register-at-run-start.yaml` (the reviewer's job) and the owner's ruling on the new refusal. The executor gives no verdict.
   - The partial step's files are kept as `*-partial*`.
4. **DBR 14.3 LTS one-time install check** (`dbr143/t135-after.public.json`): run 644976276138616, TERMINATED SUCCESS, notebook `passed: false`.
   12 of 13 checks pass. These include NumPy 1.26.4, pandas 2.3.3, pyarrow 25.0.1, the Volume's native wheel installed and loading, Spark, the engine import and the offline journey to `materialize`.
   The only failing check is `import mlflow` (ModuleNotFoundError). `mlflow-skinny` is not installed on the standard (non-ML) 14.3 runtime, so the check's assumption was wrong; the install did not cause it. It was not resubmitted (O1).
5. Not done: aarch64 (CI builds x86_64 only); 15.4 LTS and serverless stay "pending"; the archive column is not measured (C9).
6. **Owner item 50: python-core and rust-host legs rerun at f744de6b98fe19edc3eb29346365ecda377a9e48** (T1.36 fix 21, forecasts split into per-step regressions in the core; on phase head bbb6c67f, which descends from 0a8a7412). Files are in `heldout/relegs-f744de6/`.
   - **What ran:** only the two core legs, on the one-shot run's sealed `cases/` (INPUTS_SHA256 unchanged, nothing inspected or changed). python-ref and python-old are the one-shot run's observations, unchanged. rust-host ran with `--pre-given-specs` (owner item 49), as in item 3.
   - **Binaries** (`binaries.txt`):
     - python-core: local client wheel `structured_data_models_client-1.0.0-cp310-abi3-macosx_11_0_arm64.whl`, built at f744de6 from a clean tree with production features only. sha256 `a77efe57e0af34c85fdd81489afed7aab3eb1a423ce4f69392e42d925eba5335`, native `6815a2ad213fde0bbd8ea9dfe3ef9be9251e1728ff12b9c86852c3fe3c2877ef`, contract 25.
     - rust-host: `sdm-contract-run` (debug) built at f744de6, sha256 `d502a4a1b54399bceadba43c24af21322026ca4cf29cb9cfb22be006d31d3767`.
     - compare.py is 0a8a7412's (unchanged at f744de6).
   - **Counts** (`h11-counts.md`):
     - python-core: 0 panics, 0 new refusals, 6 cases differing from the reference (the same 6 as item 3).
     - rust-host: 0 panics, 0 new refusals, 8 differing. h11.task.forecast-sum is now `success`, as the reference.
     - python-old: unchanged.
     - 8 distinct cases differ on either core runner.
     - 3 format-only `inputs` pairs (item 49's conversion) are listed apart.
   - **Open: one gating line.** `DIFFERS rust-host.jsonl:32: runner failure: not a valid observation: core/predicted/body fails 'type'`.
     - It is h11.task.forecast-sum's rust-host observation. Its `core.predicted.body` is a list (one body per forecast step), but `contract/observation.schema.json`, unchanged at f744de6, requires an object.
     - The outcome matches the reference (success), but compare.py cannot compare that observation's fields. Fix 21 needs the observation schema (or the test kit's observation writer) to allow per-step bodies, and the rust-host leg needs rerunning after that.
   - The verdict still needs `report.md`'s last column (reviewer) and the owner. The executor gives no verdict.
7. **Owner items 49 and 50: python-core and rust-host legs rerun at 8295049c95f4e23dfa63d8805f36aa88427dc183** (T1.36 fixes 22–24 on f744de6b). This supersedes item 6's open schema failure. Files are in `heldout/relegs-8295049/`.
   - **What ran:** only the two core legs, on the one-shot run's sealed `cases/` (INPUTS_SHA256 unchanged, nothing inspected or changed). python-ref and python-old are the one-shot run's, unchanged. rust-host ran with `--pre-given-specs` (owner item 49).
   - **run_python.py:** fix 24 changed `contract/runners/run_python.py`: each model call is now checked against its own forecast step's request id, not predict's start id. The python-core leg ran this SHA's own `run_python.py`, from a clean worktree at 8295049c. All 42 python-core and 42 rust-host observations name commit 8295049c, clean.
   - **compare.py** is 0a8a7412's, identical at 8295049c.
   - **Binaries** (`binaries.txt`):
     - python-core: local client wheel `structured_data_models_client-1.0.0-cp310-abi3-macosx_11_0_arm64.whl`, built at 8295049c from a clean tree with production features only. sha256 `a41595d14376b2dacb8d98a13f7e5019b9c6501ecade01f2a48f4fd7fc343447`, native `b79f6386e897634d2c1d73b889b483cda754332b8bea8f5b4cb4de4093b9d77a`, contract 25.
     - rust-host: `sdm-contract-run` (debug) built at 8295049c, sha256 `fa4bdb7b05782df057863ab5efe05d969ad4794bde557862ec2f7f204ff452d3`.
   - **Result:** `compare: no differences` (0 gating lines) and no runner failure.
   - **Counts** (`h11-counts.md`):
     - python-core: 0 panics, 0 new refusals, 6 cases differing from the reference.
     - rust-host: 0 panics, 0 new refusals, 8 differing. h11.task.forecast-sum is `success`, as the reference.
     - python-old: unchanged (its one INTERNAL_ERROR is the reference's own crash, reproduced).
     - 8 distinct cases differ on either core runner. They are listed in `h11-counts.md` and need `report.md`'s last column filled from `register-at-run-start.yaml` (reviewer).
     - 3 format-only `inputs` pairs (item 49's conversion) are listed apart.
     - A non-gating payload diagnostic remains on forecast-sum: the recorded `core.predicted.body` holds 1 item against the reference's 3.
   - The executor gives no verdict.
8. **Owner items 49 and 50: final held-out evidence at 274d0755942432f943ba5404cdb9ed49d3f42155** (T1.36 fixes 25–27 on 8295049c, approved by both reviewers; the code that lands). Files are in `heldout/relegs-274d075/`. It supersedes items 6 and 7 as the held-out evidence.
   - **Core code is identical to 8295049c's:**
     - 8295049c..274d0755 changes only `contract/runners/run_python.py` (the per-step request-id check), `contract/refusal-maps/empty-null.json` (a citation) and `contract/tests/test_t136_forecast_capture.py`. Nothing under `rust/` or `packages/` changed.
     - The client's native module SHA-256 is the same in both builds: `b79f6386e897634d2c1d73b889b483cda754332b8bea8f5b4cb4de4093b9d77a`.
   - **What ran:** only python-core and rust-host, on the one-shot run's sealed `cases/` (INPUTS_SHA256 unchanged). python-ref and python-old are the one-shot run's. rust-host ran with `--pre-given-specs` (owner item 49). python-core ran 274d0755's own `run_python.py` from a clean worktree; all 42 observations of each core leg name 274d0755, clean.
   - **compare.py** is 0a8a7412's, identical at 274d0755.
   - **Binaries** (`binaries.txt`):
     - client wheel `structured_data_models_client-1.0.0-cp310-abi3-macosx_11_0_arm64.whl`, sha256 `61f60cd7e50877362cba4d24bfd279ed407979348e497e7a73be0486ab927bfb`, native `b79f6386…` (as 8295049c), contract 25;
     - `sdm-contract-run` (debug) sha256 `8e861969f17d8870ce4bccfbc8011dc5d2bacb9560482216ab6585cf85c2f170`.
     - Both files hash differently from 8295049c's, although the native module is identical: the wheel records its commit, and a debug binary embeds its build path.
   - **Result:** `compare: no differences` (0 gating lines) and no runner failure.
   - **Counts** (`h11-counts.md`):
     - python-core: 0 panics, 0 new refusals, 6 differing from the reference.
     - rust-host: 0 panics, 0 new refusals, 8 differing.
     - python-old: unchanged.
     - 8 distinct cases differ on either core runner; `report.md`'s last column must be filled from `register-at-run-start.yaml` (reviewer).
     - 3 format-only `inputs` pairs are listed apart.
   - **Run-to-run differences against 8295049c's observations** are confined to:
     - host-drawn request ids and the payload digests that carry them (python-core);
     - the captured statement result values and their digests (both legs, 30 cases). The materialize seed is unset in these cases, so sampling differs run to run.
     - No outcome changed.
   - The executor gives no verdict.
