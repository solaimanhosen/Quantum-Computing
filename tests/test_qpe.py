"""M3 tests for shor/qpe.py (quantum phase estimation). M3a: the textbook circuit."""

from math import pi

import numpy as np
import pytest
from qiskit import QuantumCircuit

from shor.qpe import exact_phase_distribution, measure_phases, qpe_circuit


def phase_gate_powers(phi):
    """controlled_power for U = P(2 pi phi): U|1> = exp(2 pi i phi)|1>, U|0> = |0>.

    U^(2^j) is the phase gate with 2^j times the angle, and a controlled phase
    gate is cp. Qubit 0 is the control, qubit 1 the target.
    """

    def controlled_power(j):
        qc = QuantumCircuit(2)
        qc.cp(2 * pi * phi * 2**j, 0, 1)
        return qc

    return controlled_power


def one():
    """prepare_target for the eigenstate |1> of the phase gate."""
    qc = QuantumCircuit(1)
    qc.x(0)
    return qc


def theory(phi, t):
    """P(y) = sin^2(pi 2^t d) / (4^t sin^2(pi d)) with d = phi - y / 2^t."""
    d = phi - np.arange(2**t) / 2**t
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.sin(pi * 2**t * d) ** 2 / (4.0**t * np.sin(pi * d) ** 2)
    probs[np.isnan(probs)] = 1.0  # d is an integer: phi is exactly y / 2^t
    return probs


def phase_qpe(phi, t, prepare=one):
    return qpe_circuit(phase_gate_powers(phi), t, 1, prepare() if prepare else None)


# -------------------------------------------------------------------- exit test 1


def test_exact_phase_is_certain():
    # phi = 5/16 with t = 4: y = 5 with probability 1.
    probs = exact_phase_distribution(phase_qpe(5 / 16, 4), 4)
    assert np.isclose(probs[5], 1.0)


def test_exact_phase_is_measured_in_every_shot():
    # The same through the simulator, which also checks the wiring of the
    # measurements into y (5 = 0101 reads 10 when the bit order is reversed).
    ys = measure_phases(phase_qpe(5 / 16, 4), shots=500, seed=1)
    assert len(ys) == 500
    assert set(ys) == {5}


@pytest.mark.parametrize("t", [1, 2, 3, 4, 5])
def test_every_t_bit_phase_is_recovered_exactly(t):
    for y in range(2**t):
        probs = exact_phase_distribution(phase_qpe(y / 2**t, t), t)
        assert np.isclose(probs[y], 1.0), y


# -------------------------------------------------------------------- exit test 2


def test_nearest_estimate_of_one_third():
    # phi = 1/3 with t = 6: 64/3 = 21.33, so the nearest estimate is y = 21.
    probs = exact_phase_distribution(phase_qpe(1 / 3, 6), 6)
    assert probs[21] >= 4 / pi**2  # the general bound, about 0.405
    assert np.isclose(probs[21], 0.684, atol=0.001)  # the value for this phase
    assert np.argmax(probs) == 21
    # The two integers next to the peak together: at least 8 / pi^2 = 0.81.
    assert probs[21] + probs[22] >= 8 / pi**2


@pytest.mark.parametrize("phi, t", [(1 / 3, 6), (1 / 3, 3), (0.1, 5), (0.77, 4), (5 / 16, 4), (0.0, 3)])
def test_distribution_matches_the_theory(phi, t):
    probs = exact_phase_distribution(phase_qpe(phi, t), t)
    assert np.isclose(probs.sum(), 1.0)
    assert np.allclose(probs, theory(phi, t))


def test_sampled_distribution_matches_the_exact_one():
    # 4000 shots, 64 outcomes: sampling noise alone gives a total variation
    # distance of about 0.02, and at most 0.03 in 99.9% of runs.
    t, shots = 6, 4000
    circuit = phase_qpe(1 / 3, t)
    exact = exact_phase_distribution(circuit, t)
    sampled = np.bincount(measure_phases(circuit, shots, seed=7), minlength=2**t) / shots
    assert 0.5 * np.abs(sampled - exact).sum() < 0.05


# ------------------------------------------------------------- the target register


def test_unprepared_target_measures_phase_zero():
    # Without prepare_target the target is |0>, an eigenstate with eigenvalue 1.
    probs = exact_phase_distribution(phase_qpe(5 / 16, 4, prepare=None), 4)
    assert np.isclose(probs[0], 1.0)


def test_superposition_of_eigenstates_gives_each_phase_with_its_weight():
    # (|0> + |1>) / sqrt(2) mixes the phases 0 and 5/16 equally. Order finding
    # works the same way: |1> is an equal mix of the r eigenstates of U_a.
    plus = QuantumCircuit(1)
    plus.h(0)
    probs = exact_phase_distribution(qpe_circuit(phase_gate_powers(5 / 16), 4, 1, plus), 4)
    assert np.isclose(probs[0], 0.5) and np.isclose(probs[5], 0.5)


@pytest.mark.parametrize("ones, y", [((0,), 3), ((1,), 1), ((0, 1), 4), ((), 0)])
def test_two_qubit_target(ones, y):
    # U = P(2 pi 3/8) on target qubit 0 and P(2 pi 1/8) on target qubit 1, so
    # the basis states are eigenstates with phases 0, 3/8, 1/8 and 4/8.
    t = 3

    def controlled_power(j):
        qc = QuantumCircuit(3)  # qubit 0 is the control
        qc.cp(2 * pi * (3 / 8) * 2**j, 0, 1)
        qc.cp(2 * pi * (1 / 8) * 2**j, 0, 2)
        return qc

    prepare = QuantumCircuit(2)
    for qubit in ones:
        prepare.x(qubit)
    probs = exact_phase_distribution(qpe_circuit(controlled_power, t, 2, prepare), t)
    assert np.isclose(probs[y], 1.0)


# ----------------------------------------------------------------- construction


def test_circuit_layout():
    t, num_target = 4, 1
    circuit = phase_qpe(5 / 16, t)
    assert circuit.num_qubits == t + num_target
    assert circuit.num_clbits == t
    assert len(circuit.cregs) == 1  # one register, so y = int(memory_string, 2)
    # t Hadamards to start, t(t-1)/2 + t controlled phases, and our own inverse
    # QFT (t more Hadamards, t // 2 swaps): no library QFT.
    assert dict(circuit.count_ops()) == {"h": 8, "cp": 4 + 6, "swap": 2, "measure": 4, "x": 1}


def test_counting_qubit_j_controls_the_power_2_to_the_j():
    t = 5
    requested = []

    def controlled_power(j):
        requested.append(j)
        qc = QuantumCircuit(2, name=f"power {j}")
        qc.cp(2 * pi * (1 / 32) * 2**j, 0, 1)
        return qc

    circuit = qpe_circuit(controlled_power, t, 1, one())
    assert sorted(requested) == list(range(t))
    # The controlled phases between a counting qubit and the target (qubit t).
    angle_by_control = {
        circuit.find_bit(inst.qubits[0]).index: inst.operation.params[0]
        for inst in circuit.data
        if inst.operation.name == "cp" and circuit.find_bit(inst.qubits[1]).index == t
    }
    assert angle_by_control == pytest.approx({j: 2 * pi * 2**j / 32 for j in range(t)})


def test_rejects_controlled_power_of_the_wrong_size():
    with pytest.raises(ValueError):
        qpe_circuit(lambda j: QuantumCircuit(3), 4, 1)


def test_rejects_prepare_target_of_the_wrong_size():
    with pytest.raises(ValueError):
        qpe_circuit(phase_gate_powers(0.25), 4, 1, QuantumCircuit(2))


@pytest.mark.parametrize("t, num_target", [(0, 1), (-1, 1), (3, 0)])
def test_rejects_empty_registers(t, num_target):
    with pytest.raises(ValueError):
        qpe_circuit(phase_gate_powers(0.25), t, num_target)


# ----------------------------------------------------------------- measure_phases


def test_measure_phases_returns_python_integers_in_range():
    ys = measure_phases(phase_qpe(1 / 3, 6), shots=200, seed=3)
    assert len(ys) == 200
    assert all(type(y) is int and 0 <= y < 64 for y in ys)
    assert len(set(ys)) > 1  # shots are not copies of one another


def test_measure_phases_is_repeatable_with_a_seed():
    circuit = phase_qpe(1 / 3, 6)
    assert measure_phases(circuit, 100, seed=11) == measure_phases(circuit, 100, seed=11)
    assert measure_phases(circuit, 100, seed=11) != measure_phases(circuit, 100, seed=12)


def test_batches_drawn_from_one_generator_differ():
    # Plan section 3, seeding: a second batch must not be a copy of the first.
    circuit = phase_qpe(1 / 3, 6)
    rng = np.random.default_rng(5)
    first = measure_phases(circuit, 100, seed=rng)
    second = measure_phases(circuit, 100, seed=rng)
    assert first != second
    # ... and the pair is still repeatable from the same starting seed.
    rng = np.random.default_rng(5)
    assert [measure_phases(circuit, 100, seed=rng), measure_phases(circuit, 100, seed=rng)] == [first, second]
