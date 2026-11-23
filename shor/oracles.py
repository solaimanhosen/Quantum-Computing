"""Modular multipliers U_a |y> = |a * y mod N> for phase estimation.

Two kinds live here:

  - M4, built by hand from gates, for N = 15 only (below).
  - The tabulated multiplier for any N (end of this file): a permutation
    matrix computed classically. It is the first half of M5.

The hand-built multiplier for N = 15
------------------------------------

This only works for N = 15, because it uses two facts about that number.
The target register holds y in four qubits, y = q_0 + 2 q_1 + 4 q_2 + 8 q_3.

  1. Multiplying by 2 is a cyclic rotation of the bits. Doubling moves every
     bit up one place, and the bit that leaves the top has the value
     16 = 1 (mod 15), so it comes back in at the bottom:

         2 * (q_0 + 2 q_1 + 4 q_2 + 8 q_3) = q_3 + 2 q_0 + 4 q_1 + 8 q_2  (mod 15).

  2. Multiplying by -1 is a bitwise NOT. Flipping all four bits turns y into
     15 - y, which is -y (mod 15).

Every a coprime to 15 is 2^k or -2^k (mod 15) for some k = 0, 1, 2, 3:

    1 = 2^0,   2 = 2^1,   4 = 2^2,   8 = 2^3,
   14 = -2^0, 13 = -2^1, 11 = -2^2,  7 = -2^3,

so U_a is a rotation by k places, followed by NOT for the second row. A
rotation is a few SWAP gates and NOT is four X gates; their controlled
versions are CSWAP and CX.

The states y = 0 and y = 15 are not residues coprime to 15 and do not occur in
order finding. The rotation leaves both alone and NOT exchanges them, so U_a
is a permutation of all 16 basis states and therefore unitary.

The tabulated multiplier for any N
----------------------------------

For general N there is no such shortcut; a multiplier made of gates is the
modular arithmetic of M6. Until then U_a is written down as a matrix: for
every basis state y we compute a * y mod N classically and put a 1 in that
row. This is an "oracle": it lets phase estimation and the post-processing be
tested for any N, but it is not a quantum algorithm for multiplying. Building
the matrix takes 2^n steps, which is the exponential cost Shor's algorithm
is meant to avoid.
"""

from math import gcd

import numpy as np
from qiskit import QuantumCircuit

# SWAPs on (q_0, q_1, q_2, q_3) that move the bit in place j to place j + k
# (mod 4), in the order they are applied. Check for k = 1: (q_0, q_1, q_2, q_3)
# -> swap(2,3) -> (q_0, q_1, q_3, q_2) -> swap(1,2) -> (q_0, q_3, q_1, q_2)
# -> swap(0,1) -> (q_3, q_0, q_1, q_2), the rotation of fact 1.
_ROTATION_SWAPS = {
    0: [],
    1: [(2, 3), (1, 2), (0, 1)],
    2: [(0, 2), (1, 3)],
    3: [(0, 1), (1, 2), (2, 3)],  # three places up = one place down
}


def _as_signed_power_of_two(a):
    """Return (k, negative) with a = 2^k or a = -2^k (mod 15), k in 0..3."""
    for k in range(4):
        if a == 2**k % 15:
            return k, False
        if a == -(2**k) % 15:
            return k, True
    raise AssertionError(f"unreachable for a = {a} coprime to 15")


def controlled_multiplier_15(a):
    """Return controlled-U_a for N = 15 as a circuit on five qubits.

    Qubit 0 is the control and qubits 1..4 hold y, least significant bit
    first. With the control at 1 the circuit maps |y> to |a * y mod 15> for
    y = 1..14; with the control at 0 it does nothing.
    """
    a %= 15
    if gcd(a, 15) != 1:
        raise ValueError(f"a = {a} shares a factor with 15, so multiplying by it is not reversible")
    k, negative = _as_signed_power_of_two(a)

    control, target = 0, [1, 2, 3, 4]
    qc = QuantumCircuit(5, name=f"c-U_{a}")
    for i, j in _ROTATION_SWAPS[k]:  # multiply by 2^k
        qc.cswap(control, target[i], target[j])
    if negative:  # multiply by -1
        for qubit in target:
            qc.cx(control, qubit)
    return qc


def controlled_power_15(a):
    """Return the function j -> controlled-U_a^(2^j), in the form QPE expects.

    Applying U_a 2^j times multiplies by a^(2^j), so the power is itself a
    multiplier: U_a^(2^j) = U_b with b = a^(2^j) mod 15. Every order mod 15
    divides 4, so b is a for j = 0, a^2 (which is 1 or 4) for j = 1, and 1
    from j = 2 on, where the circuit is empty.

    Computing a^(2^j) mod N classically like this is legitimate for any N: it
    is repeated squaring and needs no knowledge of the order.
    """
    return lambda j: controlled_multiplier_15(pow(a, 2**j, 15))


# ----------------------------------------------------------------------------
# The tabulated multiplier for any N
# ----------------------------------------------------------------------------


def controlled_permutation(a, N):
    """Return the matrix of controlled-U_a for any N, with a coprime to N.

    It acts on n + 1 qubits, n = N.bit_length(): qubit 0 is the control and
    qubits 1..n hold y, so the basis state |y>|c> has index c + 2 * y.

      - control 0 (even index): nothing happens;
      - control 1 (odd index):  y -> a * y mod N        for y < N,
                                y -> y                  for N <= y < 2^n.

    The values y >= N are not residues mod N and never occur in order finding.
    Leaving them alone makes the map a permutation of all basis states: a is
    invertible mod N, so y -> a * y mod N already permutes 0..N-1. A
    permutation matrix is unitary.
    """
    if N < 2:
        raise ValueError(f"N = {N} must be at least 2")
    a %= N
    if gcd(a, N) != 1:
        raise ValueError(f"a = {a} shares a factor with {N}, so multiplying by it is not reversible")

    n = N.bit_length()
    matrix = np.zeros((2 ** (n + 1), 2 ** (n + 1)))
    for y in range(2**n):
        image = a * y % N if y < N else y
        matrix[2 * y, 2 * y] = 1  # control 0: column 2y -> row 2y
        matrix[1 + 2 * image, 1 + 2 * y] = 1  # control 1: column 1+2y -> row 1+2*image
    return matrix


def controlled_power_oracle(a, N):
    """Return the function j -> controlled-U_a^(2^j), in the form QPE expects.

    As for N = 15, the power is a single multiplier: U_a^(2^j) = U_b with
    b = a^(2^j) mod N, found by repeated squaring. Each power is therefore
    one matrix of the same size, not 2^j copies of U_a.
    """
    n = N.bit_length()

    def controlled_power(j):
        b = pow(a, 2**j, N)
        qc = QuantumCircuit(n + 1, name=f"c-U_{b}")
        qc.unitary(controlled_permutation(b, N), range(n + 1), label=f"c-U_{b}")
        return qc

    return controlled_power
