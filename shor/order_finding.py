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

So far: M4, the order finder for N = 15. M5 and M7 add the general ones.
"""

import numpy as np
from qiskit import QuantumCircuit

from shor.classical import order_from_samples
from shor.oracles import controlled_power_15
from shor.qpe import measure_phases, qpe_circuit

# A finder takes up to this many batches of shots for one base before giving up.
MAX_BATCHES = 3


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
