"""M4 tests for shor/oracles.py: controlled multiplication by a modulo 15."""

from math import gcd

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from shor.oracles import controlled_multiplier_15, controlled_power_15

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
