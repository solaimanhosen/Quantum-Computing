"""M3: quantum phase estimation (QPE).

Given a unitary U and an eigenstate |u> with U|u> = exp(2 pi i phi)|u>, QPE
measures a t-bit integer y with y / 2^t ~ phi.

M3a, the textbook circuit, on a counting register of t qubits:

  1. H on every counting qubit: the register holds every k = 0 .. 2^t - 1
     with equal amplitude.
  2. Counting qubit j controls U^(2^j) on the target. The target is an
     eigenstate, so it does not change; the phase goes to the control instead
     ("phase kickback"), which becomes (|0> + exp(2 pi i phi 2^j)|1>) / sqrt(2).
     Qubit j has weight 2^j in k, so the register as a whole is

         2^(-t/2) * sum_k exp(2 pi i phi k) |k>.

  3. If phi = y / 2^t exactly, that state is QFT|y> (the definition in
     qft.py), so the inverse QFT turns it into |y> and the measurement returns
     y with certainty. Otherwise the inverse QFT gives a distribution peaked
     at the integers next to phi * 2^t:

         P(y) = sin^2(pi 2^t d) / (4^t sin^2(pi d)),    d = phi - y / 2^t.

     The nearest integer has probability at least 4 / pi^2 = 0.405, and the
     two integers on either side of phi * 2^t together at least 8 / pi^2.

If the target is a superposition of eigenstates, each shot behaves as if one
of them had been picked at random with its weight in the superposition.

M3b, the iterative circuit with one recycled control qubit, is not here yet.
"""

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from shor.qft import qft


def qpe_circuit(controlled_power, t, num_target, prepare_target=None):
    """Return the textbook QPE circuit with t counting qubits.

    controlled_power(j) returns a circuit on 1 + num_target qubits that
    implements controlled-U^(2^j), with the control as its qubit 0.
    prepare_target, if given, is a circuit on num_target qubits that prepares
    the target from |0...0>.

    Layout: counting qubits 0 .. t-1, then the target. Counting qubit j is
    measured into bit j of the classical register "y", so the measured
    integer is y = int(memory_string, 2).
    """
    if t < 1 or num_target < 1:
        raise ValueError(f"need t >= 1 and num_target >= 1, got t = {t}, num_target = {num_target}")

    count = QuantumRegister(t, "count")
    target = QuantumRegister(num_target, "target")
    y = ClassicalRegister(t, "y")
    qc = QuantumCircuit(count, target, y, name="QPE")

    if prepare_target is not None:
        if prepare_target.num_qubits != num_target:
            raise ValueError(
                f"prepare_target acts on {prepare_target.num_qubits} qubits, expected {num_target}"
            )
        qc.compose(prepare_target, qubits=target, inplace=True)

    # 1. Equal superposition of all k.
    qc.h(count)

    # 2. Counting qubit j controls U^(2^j).
    for j in range(t):
        power = controlled_power(j)
        if power.num_qubits != 1 + num_target:
            raise ValueError(
                f"controlled_power({j}) acts on {power.num_qubits} qubits, expected {1 + num_target}"
            )
        qc.compose(power, qubits=[count[j], *target], inplace=True)

    # 3. Inverse QFT (with swaps, so that qubit j ends up with weight 2^j), then measure.
    qc.compose(qft(t, inverse=True), qubits=count, inplace=True)
    qc.measure(count, y)
    return qc


def exact_phase_distribution(circuit, t):
    """Return the exact probability of each outcome y of a textbook QPE circuit.

    The final measurements are removed and the state is computed exactly, so
    the result is the array P(y) for y = 0 .. 2^t - 1 with no sampling noise.
    The counting register must be qubits 0 .. t-1, as qpe_circuit builds it.
    """
    without_measurements = circuit.remove_final_measurements(inplace=False)
    return Statevector(without_measurements).probabilities(range(t))


def measure_phases(circuit, shots, seed=None):
    """Run a QPE circuit on the simulator and return the measured y of each shot.

    The circuit must measure into one classical register whose bit j is the
    bit of weight 2^j, as both QPE variants do. Returns a list of Python ints.

    seed is an int or a numpy Generator (plan section 3). Each call draws a
    fresh simulator seed from the generator, so two batches taken from the
    same Generator are different samples, not copies of each other.
    """
    rng = np.random.default_rng(seed)
    simulator = AerSimulator(method="statevector")
    job = simulator.run(
        transpile(circuit, simulator, optimization_level=0),
        shots=shots,
        memory=True,
        seed_simulator=int(rng.integers(2**32)),
    )
    return [int(bits, 2) for bits in job.result().get_memory()]
