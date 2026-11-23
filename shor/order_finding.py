"""Quantum order finders: given a and N, find the order r of a mod N.

Each one runs phase estimation on U_a |y> = |a * y mod N> and hands the
measured integers to the classical post-processing of M1.

Why this gives r. The state |1> is an equal superposition of the r eigenstates
of U_a, whose eigenvalues are exp(2 pi i s / r) for s = 0 .. r-1. Phase
estimation with the target in |1> therefore returns, in each shot, an integer
y with y / 2^t ~ s / r for a uniformly random s. Continued fractions recover
the denominator r / gcd(s, r) from y, and a few shots together give r.

Every finder can be called as find_order(a, N) and returns a Python int or
None (plan section 3), so it can be passed to shor_factor.

So far:
  - M4: find_order_15, with multipliers built from gates, for N = 15.
  - First half of M5: find_order_oracle, with the tabulated multiplier, for
    N up to MAX_ORACLE_N. It uses the textbook circuit; M5 proper replaces
    that with the iterative circuit of M3b and reaches N = 1000.
M7 adds the finder that uses real modular arithmetic.
"""

import numpy as np
from qiskit import QuantumCircuit

from shor.classical import order_from_samples
from shor.oracles import controlled_power_15, controlled_power_oracle
from shor.qpe import measure_phases, qpe_circuit

# A finder takes up to this many batches of shots for one base before giving up.
MAX_BATCHES = 3

# Largest N for the oracle finder while it uses the textbook circuit. That
# circuit has 3n qubits (2n counting + n target) and 2n dense matrices of size
# 2^(n+1). Measured time per order: 0.06 s at n = 5, 0.4 s at n = 6, 5 s at
# n = 7 and 93 s at n = 8. So n = 7 is the limit: N <= 2^7 - 1.
MAX_ORACLE_N = 127


def order_from_batches(sample, t, a, N):
    """Return the order of a mod N from repeated batches of shots, or None.

    sample() returns one batch of measured y values. If the shots so far do
    not determine the order, another batch is taken and all shots are
    post-processed together, up to MAX_BATCHES batches. Giving up earlier
    would make shor_factor move on to a new base and waste the shots already
    taken for this one.
    """
    ys = []
    for _ in range(MAX_BATCHES):
        ys += sample()
        r = order_from_samples(ys, t, a, N)
        if r is not None:
            return r
    return None


def order_finding_circuit_15(a, t=8):
    """Return the textbook QPE circuit for the order of a mod 15.

    t counting qubits and 4 target qubits prepared in |1>: 12 qubits for the
    default t = 8 = 2n. Every order mod 15 divides 4, hence divides 2^t, so
    each phase s / r is an exact t-bit fraction and the measured y is exactly
    s * 2^t / r.
    """
    one = QuantumCircuit(4)
    one.x(0)  # |0001> = |1>
    return qpe_circuit(controlled_power_15(a), t, 4, one)


def find_order_15(a, N=15, shots=10, seed=None):
    """Return the order of a mod 15 found by phase estimation, or None.

    Uses the hand-built multipliers of oracles.py, so N must be 15. seed is an
    int or a numpy Generator; every batch is drawn from the same generator.
    """
    if N != 15:
        raise ValueError(f"find_order_15 only handles N = 15, got N = {N}")
    t = 8
    circuit = order_finding_circuit_15(a, t)  # raises ValueError if gcd(a, 15) != 1
    rng = np.random.default_rng(seed)
    return order_from_batches(lambda: measure_phases(circuit, shots, rng), t, a, N)


# ----------------------------------------------------------------------------
# Any N up to MAX_ORACLE_N, with the tabulated multiplier
# ----------------------------------------------------------------------------


def order_finding_circuit_oracle(a, N):
    """Return the textbook QPE circuit for the order of a mod N.

    n = N.bit_length() target qubits prepared in |1> and t = 2n counting
    qubits: 3n in total (15 for N = 21). t = 2n is what guarantees that the
    continued-fraction step can recover r from y (classical.py).

    Unlike N = 15, the order usually does not divide 2^t. The phases s / r
    are then not exact t-bit fractions, and y is spread around s * 2^t / r,
    with about 90% of the probability on the two integers next to it.
    """
    if N > MAX_ORACLE_N:
        raise ValueError(
            f"N = {N} is too large for the textbook circuit: the limit is {MAX_ORACLE_N} "
            f"({3 * N.bit_length()} qubits would be needed)"
        )
    n = N.bit_length()
    one = QuantumCircuit(n)
    one.x(0)  # |0...01> = |1>
    return qpe_circuit(controlled_power_oracle(a, N), 2 * n, n, one)


def sample_phases_oracle(a, N, shots, seed=None):
    """Return the y measured in each of `shots` runs of the oracle circuit.

    This is the raw output of the quantum part, before any post-processing.
    """
    return measure_phases(order_finding_circuit_oracle(a, N), shots, seed)


def find_order_oracle(a, N, shots=10, seed=None):
    """Return the order of a mod N found by phase estimation, or None.

    Works for any N up to MAX_ORACLE_N with a coprime to N. seed is an int or
    a numpy Generator; every batch is drawn from the same generator.
    """
    t = 2 * N.bit_length()
    circuit = order_finding_circuit_oracle(a, N)  # raises ValueError for bad a or N
    rng = np.random.default_rng(seed)
    return order_from_batches(lambda: measure_phases(circuit, shots, rng), t, a, N)
