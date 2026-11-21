"""M0 smoke test: the environment can build and simulate a circuit."""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def test_bell_state_counts():
    # |00> -> (|00> + |11>) / sqrt(2): H on qubit 0, then CNOT 0 -> 1.
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    shots = 2000
    sim = AerSimulator(seed_simulator=1234)
    counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()

    # Only the correlated outcomes appear, each about half the time.
    assert set(counts) == {"00", "11"}
    assert sum(counts.values()) == shots
    # Std dev of a fair-coin count is sqrt(shots)/2 ~ 22; 150 is > 6 sigma.
    assert abs(counts["00"] - shots / 2) < 150
