# Shor's Factoring Algorithm in Qiskit

A learning project that builds Shor's algorithm from first principles, one tested layer at a time, and runs it on a simulator. The QFT, phase estimation and modular arithmetic are written by hand from basic gates rather than taken from Qiskit's circuit library.

The full design, build order and exit tests are in [SHOR_IMPLEMENTATION_PLAN.md](SHOR_IMPLEMENTATION_PLAN.md).

## Status

| Milestone | Component | State |
|---|---|---|
| M0 | Project setup, conventions | Done |
| M1 | Classical shell: number theory, continued fractions, factor driver | Done |
| M2 | QFT and inverse QFT | Done |
| M3 | Quantum phase estimation (textbook and iterative) | Textbook (M3a) done; iterative (M3b) postponed until after M4 |
| M4 | First end-to-end factoring: N = 15 | Done |
| M5 | Oracle order finding for N ≤ 1000 | Partly done: tabulated multiplier with the textbook circuit, N ≤ 127. N ≤ 1000 needs M3b |
| M6 | Reversible modular arithmetic (Draper / Beauregard) | Not started |
| M7 | Full Shor for N ≤ 50 | Not started |
| M8 | Analysis and optional extensions | Not started |

Factoring runs end to end on a simulated quantum circuit in two forms. For N = 15 the multipliers are built from gates by hand. For any valid N up to 127 (21, 33, 35, …) the multiplier is tabulated classically as a matrix, which tests phase estimation and the post-processing but is not a scalable circuit. Both use textbook phase estimation. Still to come: the iterative phase estimation of M3b, which takes the tabulated version to N = 1000, and the real modular arithmetic of M6 and M7.

## Environment setup

Requirements: Python 3.10 or newer. The project was set up and tested on Python 3.14.7 (macOS) with `qiskit` 2.6.2 and `qiskit-aer` 0.18.4.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins `qiskit` to 2.6.x and `qiskit-aer` to 0.18.x, the versions the plan's measurements were taken with. `numpy` and `pytest` are unpinned.

To check the installation, run the smoke test, which simulates a Bell circuit on `AerSimulator`:

```bash
pytest tests/test_smoke.py
```

## Project structure

```
shor/
  __init__.py        # conventions (bit order, QFT definition, notation)
  classical.py       # M1: order, prime-power check, continued fractions
  factor.py          # M1: shor_factor driver, takes any order finder
  qft.py             # M2: QFT and inverse QFT, with and without swaps
  qpe.py             # M3: textbook phase estimation (iterative still to come)
  oracles.py         # M4: multiplier for N = 15 from gates; M5: tabulated multiplier for any N
  order_finding.py   # M4: find_order_15; M5: find_order_oracle; batching of shots
main.py              # command-line entry point: python main.py N
tests/
  test_smoke.py      # M0: Bell circuit on AerSimulator
  test_classical.py  # M1
  test_factor.py     # M1
  test_qft.py        # M2
  test_qpe.py        # M3
  test_oracles.py    # M4, M5
  test_order_finding.py  # M4, M5
  test_main.py       # main.py
  helpers.py         # test doubles shared by the test files
experiments/         # M8 scripts and results (empty for now)
pytest.ini           # test discovery; puts the project root on the import path
requirements.txt
LICENSE              # Apache-2.0
NOTICE               # copyright notice
SHOR_IMPLEMENTATION_PLAN.md
```

Modules still to come, one per milestone:

| File | Milestone | Contents |
|---|---|---|
| `shor/arithmetic.py` | M6 | Fourier-space adders, modular multiplier, controlled-U_a |
| `shor/order_finding.py` (additions) | M5, M7 | iterative `find_order_oracle` for N ≤ 1000, `find_order_arith` |

## Running the tests

With the virtual environment active, from the project root:

```bash
pytest                                # everything
pytest tests/test_classical.py        # one file
pytest -k order_from                  # tests whose name matches
pytest -v                             # list each test
```

The suite currently has 398 tests and runs in about 15 seconds. Each milestone ends with an exit test from the plan; the next milestone does not start until it passes.

## Running the project

With the virtual environment active, run `main.py` from the project root:

```bash
python main.py 989                        # classical order finder, random bases
python main.py 989 --seed 1               # repeatable
python main.py 15 --method n15 --base 7   # quantum order finder for N = 15
python main.py 21 --method oracle --base 2 # quantum order finder, tabulated multiplier
python main.py --help
```

It prints each base it tries, the order found and what the driver does with it:

```
Factoring N = 21 (n = 5 bits)
Order finder: classical (brute-force classical order finder, no quantum circuit)
  a = 10: order r = 6, a^(r/2) ≡ 13 (mod N), gcd(13 ± 1, N) = 3 and 7
21 = 3 × 7
```

`--method` selects the order finder:

| Method | What it runs | Valid N |
|---|---|---|
| `classical` (default) | Brute-force order, no quantum circuit | any valid N |
| `n15` | Textbook phase estimation on a 12-qubit simulated circuit, multipliers built from gates, 10 shots per batch | 15 only |
| `oracle` | Textbook phase estimation on 3n simulated qubits, multiplier tabulated as a matrix, 10 shots per batch | up to 127 |

`--base a` tries the base a first. For small N many random bases share a factor with N; the factor then comes from a gcd, no order finding is needed and the circuit never runs. A base coprime to N, such as 2, always goes through the order finder:

```
$ python main.py 21 --method oracle --base 2
Factoring N = 21 (n = 5 bits)
Order finder: oracle (textbook phase estimation on 3n simulated qubits, multiplier tabulated as a matrix, N up to 127)
  circuit: 15 qubits, 10 phase bits
  a = 2: order r = 6, a^(r/2) ≡ 8 (mod N), gcd(8 ± 1, N) = 7 and 3
21 = 3 × 7
```

The `oracle` method takes well under a second per order for N up to 63 (18 qubits) and about 5 seconds for N from 65 to 127 (21 qubits). N = 143 would need 24 qubits and took 93 seconds per order, which is why the limit is 127. The order finder of M7 will be added to `ORDER_FINDERS` in `main.py` when it is built.

The QFT of M2 is a circuit builder rather than something to run on its own:

```python
from shor.qft import qft

print(qft(3))                                    # H, controlled-phase and SWAP gates
print(qft(3, do_swaps=False, inverse=True))      # inverse of the no-swap form
```

Phase estimation (M3) takes the unitary as a function that returns controlled-U^(2^j). Estimating the phase 5/16 of a phase gate with 4 bits:

```python
from math import pi
from qiskit import QuantumCircuit
from shor.qpe import qpe_circuit, measure_phases

def controlled_power(j):                 # controlled-U^(2^j) for U = P(2π·5/16)
    qc = QuantumCircuit(2)               # qubit 0 is the control
    qc.cp(2 * pi * (5 / 16) * 2**j, 0, 1)
    return qc

prepare = QuantumCircuit(1)
prepare.x(0)                             # the eigenstate |1>

circuit = qpe_circuit(controlled_power, 4, 1, prepare)
measure_phases(circuit, shots=5, seed=1)     # [5, 5, 5, 5, 5], and 5 / 2^4 = 5/16
```

The same functions can be called from Python. Factor a number, using the classical order finder:

```python
import random
from shor.classical import classical_order
from shor.factor import shor_factor

shor_factor(989, classical_order, random.Random(1))   # (23, 43)
```

`shor_factor(N, find_order, rng)` accepts any function `find_order(a, N)` that returns the order of a mod N, or `None` on failure. The quantum order finders take extra parameters such as `shots` and `seed`, with defaults so that they can be passed in the same way: `shor_factor(15, find_order_15, random.Random(2))` with `find_order_15` from `shor.order_finding`. If a finder returns a wrong value or a multiple of the order, the driver notices and tries another base.

Valid inputs are odd composites that are not prime powers (15, 21, 33, 35, 39, 45, …). Anything else raises `ValueError`.

Recover an order from phase estimates, which is the post-processing the quantum circuit will feed:

```python
from shor.classical import classical_order, order_from_samples

classical_order(2, 989)                             # 154
# Two ideal samples for r = 154 with t = 20 phase bits. Alone they give the
# denominators 11 and 14; their lcm is the order.
order_from_samples([95325, 74898], 20, 2, 989)      # 154
order_from_samples([95325], 20, 2, 989)             # None: one shot is not enough here
```

`order_from_samples` only uses a sample that lies next to a peak s·2^t / r of the phase-estimation distribution. This keeps it from finding r by classical trial and error: from 20 random samples it returns r = 154 in under 1% of trials, and from 5 samples of an ideal circuit in over 99%. Its result is always a multiple of r and almost always r itself.

## How the algorithm is put together

1. Pick a random base a. If gcd(a, N) > 1, that is already a factor.
2. Find the order r of a mod N, the smallest r with a^r ≡ 1 (mod N). This is the only quantum step: phase estimation on the unitary U_a|y⟩ = |a·y mod N⟩ returns an integer y with y / 2^t ≈ s / r for a random s.
3. Recover r from y with continued fractions, combining several shots when needed.
4. If r is even and a^(r/2) ≢ −1 (mod N), then gcd(a^(r/2) ± 1, N) are non-trivial factors. Otherwise pick another a.

Targets, from the plan:

| Target | What runs | Qubits |
|---|---|---|
| Primary: N ≤ 50 | Complete Shor circuit with reversible arithmetic | 2n+3 = 15 at n = 6 |
| Secondary: N ≤ 1000 | Phase estimation with U_a as a permutation matrix | n+1 = 11 at n = 10 |

To keep the qubit count this low, order finding uses iterative phase estimation with one recycled control qubit instead of a 2n-qubit counting register. The N = 15 warm-up in M4 is the exception and uses the textbook form.

## Conventions

- **Bit order:** Qiskit little-endian. Qubit 0 of a register is the least significant bit.
- **Notation:** N is the number to factor, n = `N.bit_length()`, a is the base, r is the order of a mod N, t = 2n is the number of phase bits.
- **Phase estimate:** every phase-estimation variant returns an integer y with y / 2^t ≈ phase.
- **Hand-built blocks:** the QFT and the adders are not replaced with library gates.
- **Commits:** one commit per passed exit test, made with `git commit -s`, on `main`.
- **Evidence:** a correct factorisation is weak evidence that a circuit works, because a random base often shares a factor with a small N and the classical checks are forgiving. Each quantum milestone is tested on its measured distribution as well.

Section 3 of the plan has the full list, including the exact QFT definition.

## License

Copyright 2026 Md Solaiman Hosen. Licensed under the [Apache License, Version 2.0](LICENSE); see also [NOTICE](NOTICE).

## References

- P. Shor, "Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer", arXiv:quant-ph/9508027.
- M. Nielsen and I. Chuang, *Quantum Computation and Quantum Information*, chapter 5.
- T. Draper, "Addition on a Quantum Computer", arXiv:quant-ph/0008033.
- S. Beauregard, "Circuit for Shor's algorithm using 2n+3 qubits", arXiv:quant-ph/0205095.
- R. Griffiths and C.-S. Niu, "Semiclassical Fourier Transform for Quantum Computation", arXiv:quant-ph/9511007.
