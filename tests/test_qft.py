"""M2 tests for shor/qft.py (quantum Fourier transform, built from H and CP)."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator, Statevector

from shor.qft import qft

SIZES = [1, 2, 3, 4, 5]


def dft_matrix(m):
    """The definition from plan section 3: F[k, x] = exp(2 pi i x k / 2^m) / sqrt(2^m)."""
    dim = 2**m
    x = np.arange(dim)
    return np.exp(2j * np.pi * np.outer(x, x) / dim) / np.sqrt(dim)


def bit_reversal(m):
    """Circuit that swaps qubit j with qubit m-1-j."""
    qc = QuantumCircuit(m)
    for j in range(m // 2):
        qc.swap(j, m - 1 - j)
    return qc


def matrix(circuit):
    return Operator(circuit).data


# -------------------------------------------------------------------- exit test 1


@pytest.mark.parametrize("m", SIZES)
def test_qft_equals_the_dft_matrix(m):
    # Exact equality, global phase included.
    assert np.allclose(matrix(qft(m)), dft_matrix(m))


# -------------------------------------------------------------------- exit test 2


@pytest.mark.parametrize("m", SIZES)
def test_qft_equals_the_library_gate(m):
    assert np.allclose(matrix(qft(m)), matrix(QFTGate(m)))


# -------------------------------------------------------------------- exit test 3


@pytest.mark.parametrize("m", SIZES)
@pytest.mark.parametrize("do_swaps", [True, False])
def test_inverse_undoes_the_qft(m, do_swaps):
    forward = qft(m, do_swaps=do_swaps)
    backward = qft(m, do_swaps=do_swaps, inverse=True)
    identity = np.eye(2**m)
    assert np.allclose(matrix(forward.compose(backward)), identity)
    assert np.allclose(matrix(backward.compose(forward)), identity)


@pytest.mark.parametrize("m", SIZES)
def test_inverse_is_the_conjugate_transpose_of_the_dft(m):
    assert np.allclose(matrix(qft(m, inverse=True)), dft_matrix(m).conj().T)


# -------------------------------------------------------------------- exit test 4


@pytest.mark.parametrize("m", SIZES)
def test_no_swap_qft_gives_the_product_state(m):
    # Qubit j ends in (|0> + exp(2 pi i x / 2^(j+1)) |1>) / sqrt(2).
    circuit = qft(m, do_swaps=False)
    for x in range(2**m):
        expected = np.array([1.0 + 0j])
        for j in range(m):
            qubit_j = np.array([1, np.exp(2j * np.pi * x / 2 ** (j + 1))]) / np.sqrt(2)
            expected = np.kron(qubit_j, expected)  # little-endian: qubit 0 is rightmost
        state = Statevector.from_int(x, 2**m).evolve(circuit)
        assert np.allclose(state.data, expected), x


@pytest.mark.parametrize("m", SIZES)
def test_no_swap_qft_then_bit_reversal_is_the_qft(m):
    with_reversal = qft(m, do_swaps=False).compose(bit_reversal(m))
    assert np.allclose(matrix(with_reversal), dft_matrix(m))


@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_no_swap_qft_differs_from_the_qft(m):
    # Guards against do_swaps being ignored. (For m = 1 there is nothing to swap.)
    assert not np.allclose(matrix(qft(m, do_swaps=False)), dft_matrix(m))


# ------------------------------------------------------------------ construction


@pytest.mark.parametrize("m", SIZES)
@pytest.mark.parametrize("inverse", [False, True])
def test_built_from_h_and_controlled_phase_only(m, inverse):
    counts = dict(qft(m, do_swaps=False, inverse=inverse).count_ops())
    expected = {"h": m, "cp": m * (m - 1) // 2}
    assert counts == {gate: n for gate, n in expected.items() if n}

    counts = dict(qft(m, do_swaps=True, inverse=inverse).count_ops())
    expected["swap"] = m // 2
    assert counts == {gate: n for gate, n in expected.items() if n}


def test_returns_a_circuit_on_m_qubits_without_classical_bits():
    circuit = qft(4)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 4 and circuit.num_clbits == 0


@pytest.mark.parametrize("m", [0, -1])
def test_rejects_empty_register(m):
    with pytest.raises(ValueError):
        qft(m)


# ----------------------------------------------------- how M6 will use the no-swap QFT


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_phases_in_the_fourier_basis_add_a_constant(m):
    # In the no-swap basis qubit j carries the phase x / 2^(j+1). Adding
    # a / 2^(j+1) to it with P(2 pi a / 2^(j+1)) turns x into x + a (mod 2^m).
    # This is the Draper adder of M6a, checked here as a test of the convention.
    dim = 2**m
    for a in range(dim):
        qc = qft(m, do_swaps=False)
        for j in range(m):
            qc.p(2 * np.pi * a / 2 ** (j + 1), j)
        qc.compose(qft(m, do_swaps=False, inverse=True), inplace=True)
        add_a = np.zeros((dim, dim))
        for x in range(dim):
            add_a[(x + a) % dim, x] = 1
        assert np.allclose(matrix(qc), add_a), a
