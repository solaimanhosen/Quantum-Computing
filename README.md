# Shor's Factoring Algorithm in Qiskit

A learning project that builds Shor's algorithm from first principles, one tested layer at a time, and runs it on a simulator. The QFT, phase estimation and modular arithmetic are written by hand from basic gates rather than taken from Qiskit's circuit library.

The full design, build order and exit tests are in [SHOR_IMPLEMENTATION_PLAN.md](SHOR_IMPLEMENTATION_PLAN.md).

## Status

| Milestone | Component | State |
|---|---|---|
| M0 | Project setup, conventions | Done |
| M1 | Classical shell: number theory, continued fractions, factor driver | Done |
| M2 | QFT and inverse QFT | Not started |
| M3 | Quantum phase estimation (textbook and iterative) | Not started |
| M4 | First end-to-end factoring: N = 15 | Not started |
| M5 | Oracle order finding for N ≤ 1000 | Not started |
| M6 | Reversible modular arithmetic (Draper / Beauregard) | Not started |
| M7 | Full Shor for N ≤ 50 | Not started |
| M8 | Analysis and optional extensions | Not started |

Only the classical half exists so far. Factoring already works end to end, but with a brute-force classical order finder standing in for the quantum one.

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
tests/
  test_smoke.py      # M0: Bell circuit on AerSimulator
  test_classical.py  # M1
  test_factor.py     # M1
experiments/         # M8 scripts and results (empty for now)
pytest.ini           # test discovery; puts the project root on the import path
requirements.txt
SHOR_IMPLEMENTATION_PLAN.md
```

Modules still to come, one per milestone:

| File | Milestone | Contents |
|---|---|---|
| `shor/qft.py` | M2 | QFT and inverse QFT, with and without swaps |
| `shor/qpe.py` | M3 | Textbook and iterative phase estimation |
| `shor/oracles.py` | M4, M5 | N = 15 circuits, permutation oracle |
| `shor/arithmetic.py` | M6 | Fourier-space adders, modular multiplier, controlled-U_a |
| `shor/order_finding.py` | M5, M7 | `find_order_oracle`, `find_order_arith` |

## Running the tests

With the virtual environment active, from the project root:

```bash
pytest                                # everything
pytest tests/test_classical.py        # one file
pytest -k order_from                  # tests whose name matches
pytest -v                             # list each test
```

The suite currently has 51 tests and runs in about a second. Each milestone ends with an exit test from the plan; the next milestone does not start until it passes.

## Running the project

There is no command-line entry point yet; the code is used from Python. Start `python` from the project root with the virtual environment active.

Factor a number, using the classical order finder:

```python
import random
from shor.classical import classical_order
from shor.factor import shor_factor

shor_factor(989, classical_order, random.Random(1))   # (23, 43)
```

`shor_factor(N, find_order, rng)` accepts any function `find_order(a, N)` that returns the order of a mod N, or `None` on failure. The quantum order finders from M4, M5 and M7 will plug into the same argument.

Valid inputs are odd composites that are not prime powers (15, 21, 33, 35, 39, 45, …). Anything else raises `ValueError`.

Recover an order from phase estimates, which is the post-processing the quantum circuit will feed:

```python
from shor.classical import classical_order, order_from_samples

classical_order(2, 989)                             # 154
# Two ideal samples for r = 154 with t = 20 phase bits. Alone they give the
# denominators 11 and 14; their lcm is the order.
order_from_samples([95325, 74898], 20, 2, 989)      # 154
```

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

To keep the qubit count this low, order finding uses iterative phase estimation with one recycled control qubit instead of a 2n-qubit counting register.

## Conventions

- **Bit order:** Qiskit little-endian. Qubit 0 of a register is the least significant bit.
- **Notation:** N is the number to factor, n = `N.bit_length()`, a is the base, r is the order of a mod N, t = 2n is the number of phase bits.
- **Phase estimate:** every phase-estimation variant returns an integer y with y / 2^t ≈ phase.
- **Hand-built blocks:** the QFT and the adders are not replaced with library gates.
- **Commits:** one commit per passed exit test, made with `git commit -s`, on `main`.

Section 3 of the plan has the full list, including the exact QFT definition.

## References

- P. Shor, "Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer", arXiv:quant-ph/9508027.
- M. Nielsen and I. Chuang, *Quantum Computation and Quantum Information*, chapter 5.
- T. Draper, "Addition on a Quantum Computer", arXiv:quant-ph/0008033.
- S. Beauregard, "Circuit for Shor's algorithm using 2n+3 qubits", arXiv:quant-ph/0205095.
- R. Griffiths and C.-S. Niu, "Semiclassical Fourier Transform for Quantum Computation", arXiv:quant-ph/9511007.
