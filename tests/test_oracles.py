"""M4 tests for shor/oracles.py: controlled multiplication by a modulo 15."""

from math import gcd

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from shor.oracles import (
    controlled_multiplier_15,
    controlled_permutation,
    controlled_power_15,
    controlled_power_oracle,
)

UNITS = [1, 2, 4, 7, 8, 11, 13, 14]  # everything coprime to 15
BASES = [2, 4, 7, 8, 11, 13, 14]  # the seven bases of the plan


def image(matrix, column):
    """The basis state a permutation matrix sends basis state `column` to."""
    assert np.isclose(np.abs(matrix[:, column]).sum(), 1.0)  # a single entry ...
    row = int(np.argmax(np.abs(matrix[:, column])))
    assert np.isclose(matrix[row, column], 1.0)  # ... equal to 1, no stray phase
    return row


# Index convention: the control is qubit 0 and the target register holds y in
# qubits 1..4, so a basis state has index c + 2 * y.


# -------------------------------------------------------------------- exit test 1


@pytest.mark.parametrize("a", UNITS)
def test_multiplies_by_a_when_the_control_is_one(a):
    matrix = Operator(controlled_multiplier_15(a)).data
    for y in range(1, 15):
        assert image(matrix, 1 + 2 * y) == 1 + 2 * (a * y % 15), y


@pytest.mark.parametrize("a", UNITS)
def test_identity_when_the_control_is_zero(a):
    matrix = Operator(controlled_multiplier_15(a)).data
    for y in range(16):
        assert image(matrix, 2 * y) == 2 * y, y


@pytest.mark.parametrize("a", UNITS)
def test_states_outside_the_residues_stay_outside(a):
    # 15 is not a residue mod 15 and 0 is not a unit; U_a only has to permute
    # them among themselves so that the whole map stays unitary.
    matrix = Operator(controlled_multiplier_15(a)).data
    assert {image(matrix, 1 + 2 * y) for y in (0, 15)} == {1, 31}


# ----------------------------------------------------------------- construction


@pytest.mark.parametrize("a", UNITS)
def test_built_from_cswap_and_cx_only(a):
    circuit = controlled_multiplier_15(a)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 5 and circuit.num_clbits == 0
    assert set(circuit.count_ops()) <= {"cswap", "cx"}


def test_multiplying_by_one_is_the_empty_circuit():
    assert len(controlled_multiplier_15(1).data) == 0


def test_base_is_reduced_modulo_15():
    assert Operator(controlled_multiplier_15(22)) == Operator(controlled_multiplier_15(7))


@pytest.mark.parametrize("a", [0, 3, 5, 6, 9, 10, 12, 15, 30])
def test_rejects_bases_that_share_a_factor_with_15(a):
    assert gcd(a, 15) != 1
    with pytest.raises(ValueError):
        controlled_multiplier_15(a)


# ----------------------------------------------------------------------- powers


@pytest.mark.parametrize("a", BASES)
def test_power_j_is_the_multiplier_applied_2_to_the_j_times(a):
    # U_a^(2^j) is built as the single multiplier U_(a^(2^j) mod 15). Compare
    # it with U_a literally repeated 2^j times.
    controlled_power = controlled_power_15(a)
    once = controlled_multiplier_15(a)
    repeated = QuantumCircuit(5)
    repeated.compose(once, inplace=True)  # 2^0 = 1 application
    for j in range(5):
        assert Operator(controlled_power(j)) == Operator(repeated), j
        repeated = repeated.compose(repeated)  # double the number of applications


@pytest.mark.parametrize("a", BASES)
def test_powers_from_the_fourth_on_are_the_identity(a):
    # Every order mod 15 divides 4, so a^(2^j) = 1 for j >= 2.
    controlled_power = controlled_power_15(a)
    for j in range(2, 8):
        assert len(controlled_power(j).data) == 0, j


# =============================================================================
# The tabulated multiplier for general N (first half of M5)
# =============================================================================


def units(N):
    return [a for a in range(1, N) if gcd(a, N) == 1]


@pytest.mark.parametrize("N", [15, 21, 33, 35])
def test_permutation_matrix_multiplies_by_a(N):
    n = N.bit_length()
    for a in units(N):
        matrix = controlled_permutation(a, N)
        assert matrix.shape == (2 ** (n + 1), 2 ** (n + 1))
        for y in range(2**n):
            # Control 1 (odd index): y -> a * y mod N below N, unchanged from N up.
            assert image(matrix, 1 + 2 * y) == 1 + 2 * (a * y % N if y < N else y), (a, y)
            # Control 0 (even index): unchanged.
            assert image(matrix, 2 * y) == 2 * y, (a, y)


@pytest.mark.parametrize("N", [15, 21, 35])
def test_permutation_matrix_is_unitary(N):
    for a in units(N):
        matrix = controlled_permutation(a, N)
        assert np.array_equal(matrix @ matrix.T, np.eye(len(matrix)))


@pytest.mark.parametrize("a", UNITS)
def test_agrees_with_the_hand_built_multiplier_for_15(a):
    # Two independent constructions of the same map. They may differ on y = 0
    # and y = 15, which are outside the residues 1..14 (see oracles.py).
    by_hand = Operator(controlled_multiplier_15(a)).data
    tabulated = controlled_permutation(a, 15)
    columns = [2 * y for y in range(16)] + [1 + 2 * y for y in range(1, 15)]
    assert np.allclose(by_hand[:, columns], tabulated[:, columns])


def test_base_is_reduced_modulo_N():
    assert np.array_equal(controlled_permutation(23, 21), controlled_permutation(2, 21))


@pytest.mark.parametrize("a, N", [(3, 21), (7, 21), (14, 21), (0, 21), (21, 21), (5, 35)])
def test_permutation_rejects_bases_that_share_a_factor(a, N):
    with pytest.raises(ValueError):
        controlled_permutation(a, N)


@pytest.mark.parametrize("N", [1, 0, -21])
def test_permutation_rejects_modulus_below_two(N):
    with pytest.raises(ValueError):
        controlled_permutation(1, N)


@pytest.mark.parametrize("a, N", [(2, 21), (5, 21), (2, 35), (3, 35)])
def test_oracle_power_j_is_the_matrix_to_the_power_2_to_the_j(a, N):
    n = N.bit_length()
    controlled_power = controlled_power_oracle(a, N)
    once = controlled_permutation(a, N)
    for j in range(6):
        circuit = controlled_power(j)
        assert circuit.num_qubits == n + 1 and circuit.num_clbits == 0
        expected = np.linalg.matrix_power(once, 2**j)
        assert np.allclose(Operator(circuit).data, expected), j


def test_oracle_powers_use_one_gate_each():
    # U^(2^j) is a single multiplier by a^(2^j) mod N, not 2^j copies of U.
    controlled_power = controlled_power_oracle(2, 21)
    for j in range(10):
        assert dict(controlled_power(j).count_ops()) == {"unitary": 1}
