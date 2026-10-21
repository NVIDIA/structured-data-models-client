# Offline L7 against KO2-29

Cell ref-pandas3, database duckdb, 21 repeats; core commit be55838a14da, reference commit 43e55f4f5892.

## 1. CPU per scenario

- scenarios measured on both: 2
- incomplete on either side, a repetition stopping before the stage the case runs to (not judged): 0
- judged: 2
- median core-to-reference ratio: 1.005 (limit 1.0)
- scenarios above 1.1: 0
- FAILS

| scenario | reference CPU (ms) | core CPU (ms) | ratio |
|---|---|---|---|
| s0.accept.sum-window | 85.85 | 87.74 | 1.022 |
| s0.boundary.deepest | 99.39 | 98.27 | 0.989 |

## 2. Import time

- core (with `sdm_client._native`): 1131.2 ms; reference: 1150.8 ms; holds

## 3. Native library size

- not measured (no --wheel and --baseline-wheel)

## 4. In-memory boundary overhead

- per crossing: 0.13 us; overhead = the codec's CPU + messages x the crossing's
- scenarios over max(5% of their CPU, 2 ms): 0 []
- the closest: s0.boundary.deepest, 0.049 ms of 4.913 ms (2 messages)
- holds

## 5. Control messages

- the largest: 397 bytes (cap 65,536); holds

## 6. Statements per scenario

- not measured (no python-ref observations given)

## 7. No pandas in providers

- provider modules using pandas: 0 []; holds

## Zero-copy

- not measured here: `tests/test_arrow_lifetimes.py` holds it, in CI's `native-test-ops` job on the exact SHA.

**L7: a limit FAILS**
