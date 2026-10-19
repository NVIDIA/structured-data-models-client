# Phase 2 tracker: T2.0h

T2.0h has `run_python.py` name the floor cell at T2.0c's new pins (Python 3.10, pandas 2.2.2, pyarrow 14.0.1, NumPy 1.23.5; KO2-37), and still at the earlier ones (pandas 2.2.0, NumPy 1.26.4) through `EARLIER_PINS`, which the phase branch's `ci.yml` installs until T2.0c lands; the previous package, T2.0g, landed at `95259022`.
Judging-path file touched: `contract/runners/run_python.py` only; the `CELLS` line is byte-identical to T2.0c's, so its merge with lane b is clean (`git merge-tree`).
`contract/tests/test_t20h_floor_pins.py`: both pin sets resolve to `floor-pandas2.2` and a third NumPy to no cell; without the fix the new-pins case fails.
