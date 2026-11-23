"""M2: the quantum Fourier transform, built from H and controlled-phase gates.

Definition (plan section 3), on m qubits with dim = 2^m:

    QFT|x> = dim^(-1/2) * sum_k exp(2 pi i x k / dim) |k>

Product form. Write k in bits, k = sum_i k_i 2^i (qubit i has weight 2^i).
The exponent splits into one term per output qubit,

    x k / dim = sum_i k_i * x / 2^(m-i),

so the sum over k factorises into a product of single-qubit states:

    QFT|x> = tensor_i (|0> + exp(2 pi i x / 2^(m-i)) |1>) / sqrt(2)   on qubit i.

Output qubit i carries the phase x / 2^(m-i). The top qubit (i = m-1) holds
x / 2, which depends only on the lowest bit of x; the bottom qubit (i = 0)
holds x / 2^m, which depends on every bit. The order is reversed with respect
to the input bits.

The circuit computes the reversed arrangement first, because it can be done in
place: the "no-swap QFT" leaves

    qubit j in (|0> + exp(2 pi i x / 2^(j+1)) |1>) / sqrt(2),

and swapping qubit j with qubit m-1-j then gives the QFT above. The Fourier
adders of M6 work directly in the no-swap arrangement and skip the swaps.

Reading the gates off the no-swap form. Only the fractional part of
x / 2^(j+1) matters, and with x = sum_i x_i 2^i it is

    x_j / 2  +  sum_{i<j} x_i / 2^(j-i+1).

  - H on qubit j maps |x_j> to (|0> + exp(2 pi i x_j / 2) |1>) / sqrt(2):
    the first term.
  - A controlled phase of angle 2 pi / 2^(j-i+1) = pi / 2^(j-i) with control
    qubit i adds x_i / 2^(j-i+1): one gate for each lower qubit i < j.

Qubit j needs the lower qubits still in their original basis state, so the
qubits are processed from the top (j = m-1) down to j = 0.
"""

from math import pi

from qiskit import QuantumCircuit


def qft(m, do_swaps=True, inverse=False):
    """Return the QFT on m qubits as a circuit of H, CP and (optionally) SWAP gates.

    do_swaps=False returns the no-swap QFT described in the module docstring.
    inverse=True returns the inverse of whichever variant was selected.
    """
    if m < 1:
        raise ValueError(f"the QFT needs at least one qubit, got m = {m}")

    qc = QuantumCircuit(m, name="QFT")
    for j in reversed(range(m)):
        qc.h(j)  # phase x_j / 2
        for i in reversed(range(j)):
            qc.cp(pi / 2 ** (j - i), i, j)  # phase x_i / 2^(j-i+1)

    if do_swaps:
        for j in range(m // 2):
            qc.swap(j, m - 1 - j)

    if inverse:
        # The same gates in reverse order with every angle negated
        # (H and SWAP are their own inverses).
        qc = qc.inverse()
        qc.name = "QFT†"
    return qc
