"""Shor's factoring algorithm, built layer by layer (see SHOR_IMPLEMENTATION_PLAN.md).

Conventions used by every module (plan section 3):

- Bit order: Qiskit little-endian. Qubit 0 of a register is the least
  significant bit, so a register holds the integer sum_j q_j * 2**j.
- QFT on m qubits: QFT|x> = 2**(-m/2) * sum_k exp(2*pi*i*x*k / 2**m) |k>.
- No-swap QFT leaves qubit j in (|0> + exp(2*pi*i*x / 2**(j+1)) |1>) / sqrt(2).
- Phase estimate: every QPE variant returns an integer y with y / 2**t ~ phase.
- Notation: N is the number to factor, n = N.bit_length(), a is the base,
  r is the order of a mod N, t = 2n is the number of phase bits.
"""
