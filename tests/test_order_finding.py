"""Tests for shor/order_finding.py. M4: order finding for N = 15."""

from functools import partial
from math import gcd

import numpy as np
import pytest
from helpers import CountingOrder, FixedBases, order_finding_theory

from shor.classical import classical_order, is_prime_power
from shor.factor import shor_factor
from shor.order_finding import (
    MAX_BATCHES,
    MAX_ORACLE_N,
    find_order_15,
    find_order_oracle,
    order_finding_circuit_15,
    order_finding_circuit_oracle,
    order_from_batches,
    sample_phases_oracle,
)
from shor.qpe import exact_phase_distribution, measure_phases

BASES = [2, 4, 7, 8, 11, 13, 14]
T = 8


# -------------------------------------------------------------------- the circuit


def test_circuit_size():
    circuit = order_finding_circuit_15(7)
    assert circuit.num_qubits == 12  # 8 counting qubits + 4 target qubits
    assert circuit.num_clbits == 8


def test_target_starts_in_the_state_one():
    # |2>, |4>, ... would give the same outcomes, since they are superpositions
    # of the same eigenstates, so this has to be checked on the circuit itself.
    circuit = order_finding_circuit_15(7)
    x_gates = [inst for inst in circuit.data if inst.operation.name == "x"]
    assert [circuit.find_bit(inst.qubits[0]).index for inst in x_gates] == [T]


# -------------------------------------------------------------------- exit test 2


def test_base_7_gives_four_outcomes_with_probability_one_quarter():
    probs = exact_phase_distribution(order_finding_circuit_15(7), T)
    assert np.allclose(probs[[0, 64, 128, 192]], 0.25)
    assert np.isclose(probs.sum(), 1.0)


@pytest.mark.parametrize("a", BASES + [1])
def test_outcomes_are_the_multiples_of_256_over_r(a):
    # The phases are s / r, and r divides 256 for every base, so each y is
    # exactly s * 256 / r, with all r values of s equally likely.
    r = classical_order(a, 15)
    expected = np.zeros(2**T)
    expected[[s * 2**T // r for s in range(r)]] = 1 / r
    assert np.allclose(exact_phase_distribution(order_finding_circuit_15(a), T), expected)


def test_simulator_shots_land_on_the_four_peaks():
    # Through the measurements, which the exact distribution bypasses.
    ys = measure_phases(order_finding_circuit_15(7), shots=400, seed=4)
    assert set(ys) == {0, 64, 128, 192}
    assert min(ys.count(y) for y in (0, 64, 128, 192)) > 60  # 100 expected each


# -------------------------------------------------------------------- exit test 3


@pytest.mark.parametrize("a", BASES + [1])
def test_order_agrees_with_classical_order(a):
    for seed in range(5):
        r = find_order_15(a, 15, seed=seed)
        assert r == classical_order(a, 15)
        assert type(r) is int


def test_shor_factor_15_through_the_quantum_order_finder():
    # Base fixed to 7 so that the gcd shortcut cannot hide the order finder.
    find_order = CountingOrder(partial(find_order_15, seed=1))
    assert shor_factor(15, find_order, FixedBases(7)) == (3, 5)
    assert find_order.calls == [7]


@pytest.mark.parametrize("a", [2, 4, 7, 8, 11, 13])
def test_every_coprime_base_the_driver_can_pick_factors_15(a):
    # None of these has a^(r/2) = -1 (mod 15), so one attempt is enough.
    find_order = CountingOrder(partial(find_order_15, seed=a))
    assert shor_factor(15, find_order, FixedBases(a)) == (3, 5)
    assert find_order.calls == [a]


# ----------------------------------------------------------------------- inputs


def test_callable_as_find_order_with_two_arguments():
    assert find_order_15(7, 15) == 4


@pytest.mark.parametrize("N", [21, 35, 16])
def test_rejects_other_moduli(N):
    with pytest.raises(ValueError):
        find_order_15(2, N)


@pytest.mark.parametrize("a", [3, 5, 6, 10])
def test_rejects_bases_that_share_a_factor_with_15(a):
    with pytest.raises(ValueError):
        find_order_15(a, 15)


def test_same_seed_same_shots():
    circuit = order_finding_circuit_15(7)
    assert measure_phases(circuit, 20, seed=9) == measure_phases(circuit, 20, seed=9)


def test_every_batch_is_drawn_from_one_generator(monkeypatch):
    # Plan section 3, seeding: passing the integer seed to every batch would
    # make the second and third batches copies of the first.
    seeds = []

    def fake_measure_phases(circuit, shots, seed):
        seeds.append(seed)
        return [0] * shots  # never enough, so all batches are taken

    monkeypatch.setattr("shor.order_finding.measure_phases", fake_measure_phases)
    assert find_order_15(7, 15, shots=4, seed=5) is None
    assert len(seeds) == MAX_BATCHES
    assert isinstance(seeds[0], np.random.Generator)
    assert all(seed is seeds[0] for seed in seeds)
    # ... and that generator is the one the caller's seed defines.
    assert seeds[0].integers(2**32) == np.random.default_rng(5).integers(2**32)


# --------------------------------------------------------------------- batching


class Batches:
    """A sampler that returns prepared batches and counts how often it is asked."""

    def __init__(self, *batches):
        self._batches = iter(batches)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return next(self._batches)


def test_first_batch_is_enough_when_it_determines_the_order():
    sample = Batches([64, 0])  # 64/256 = 1/4
    assert order_from_batches(sample, T, 7, 15) == 4
    assert sample.calls == 1


def test_takes_another_batch_for_the_same_base_when_needed():
    # y = 0 says nothing. The shots of the first batch are kept, not discarded.
    sample = Batches([0, 0], [0, 192])
    assert order_from_batches(sample, T, 7, 15) == 4
    assert sample.calls == 2


def test_batches_are_combined():
    # r = 154 for a = 2, N = 989: 95325 gives the denominator 11 and 74898
    # gives 14. Neither batch is enough alone; together lcm(11, 14) = 154.
    sample = Batches([95325], [74898])
    assert order_from_batches(sample, 20, 2, 989) == 154
    assert sample.calls == 2


def test_gives_up_after_max_batches():
    sample = Batches(*[[0, 0]] * MAX_BATCHES)
    assert order_from_batches(sample, T, 7, 15) is None
    assert sample.calls == MAX_BATCHES == 3


# =============================================================================
# Order finding with the tabulated multiplier (first half of M5)
# =============================================================================


def coprime_bases(N):
    """The bases shor_factor can pick that do not already share a factor with N."""
    return [a for a in range(2, N - 1) if gcd(a, N) == 1]


def next_to_a_peak(y, r, t):
    """True if y is one of the two integers beside some s * 2^t / r."""
    return any(abs(y * r - s * 2**t) < r for s in range(r + 1))


# -------------------------------------------------------------------- the circuit


@pytest.mark.parametrize("N, qubits", [(15, 12), (21, 15), (35, 18)])
def test_oracle_circuit_size(N, qubits):
    # t = 2n counting qubits and n target qubits: 3n in total.
    circuit = order_finding_circuit_oracle(2, N)
    assert circuit.num_qubits == qubits == 3 * N.bit_length()
    assert circuit.num_clbits == 2 * N.bit_length()


def test_oracle_target_starts_in_the_state_one():
    circuit = order_finding_circuit_oracle(2, 21)
    x_gates = [inst for inst in circuit.data if inst.operation.name == "x"]
    assert [circuit.find_bit(inst.qubits[0]).index for inst in x_gates] == [10]


# ------------------------------------------------------------- the y distribution


@pytest.mark.parametrize("a", [2, 4, 5, 8, 13, 20])
def test_distribution_for_21_matches_the_theory(a):
    # The test that shows the circuit works. Orders mod 21 are 2, 3 and 6;
    # 3 and 6 do not divide 2^10, so the peaks are not exact integers.
    t = 10
    r = classical_order(a, 21)
    probs = exact_phase_distribution(order_finding_circuit_oracle(a, 21), t)
    assert np.allclose(probs, order_finding_theory(r, t))


def test_distribution_for_35_matches_the_theory():
    t = 12
    probs = exact_phase_distribution(order_finding_circuit_oracle(2, 35), t)
    assert np.allclose(probs, order_finding_theory(12, t))


def test_oracle_reproduces_the_hand_built_circuit_for_15():
    for a in BASES:
        by_hand = exact_phase_distribution(order_finding_circuit_15(a), T)
        tabulated = exact_phase_distribution(order_finding_circuit_oracle(a, 15), T)
        assert np.allclose(by_hand, tabulated), a


def test_sampled_shots_for_21_cluster_at_the_six_peaks():
    # Through the simulator and the measurements. a = 2 has r = 6, t = 10.
    r, t, shots = 6, 10, 400
    ys = sample_phases_oracle(2, 21, shots, seed=8)
    assert len(ys) == shots and all(type(y) is int and 0 <= y < 2**t for y in ys)
    # Theory: 90.3% of shots are next to a peak. Uniformly random y: about 1%.
    assert sum(next_to_a_peak(y, r, t) for y in ys) >= 0.75 * shots
    # Every s occurs about equally often.
    per_peak = np.bincount([round(y * r / 2**t) % r for y in ys], minlength=r) / shots
    assert np.all(np.abs(per_peak - 1 / r) < 0.07)


def test_sample_phases_oracle_is_repeatable_with_a_seed():
    assert sample_phases_oracle(2, 21, 50, seed=3) == sample_phases_oracle(2, 21, 50, seed=3)
    assert sample_phases_oracle(2, 21, 50, seed=3) != sample_phases_oracle(2, 21, 50, seed=4)


# ------------------------------------------------------------ orders and factors


def test_oracle_order_agrees_with_classical_order_for_every_base_of_21():
    for a in coprime_bases(21):
        r = find_order_oracle(a, 21, seed=a)
        assert r == classical_order(a, 21), a
        assert type(r) is int


@pytest.mark.parametrize("a, N", [(2, 33), (5, 33), (10, 33), (2, 35), (3, 35), (6, 35), (29, 35), (2, 51)])
def test_oracle_order_at_18_qubits(a, N):
    # Orders 10, 10, 2, 12, 12, 2, 2 and 8. About 0.4 s each at n = 6.
    assert find_order_oracle(a, N, seed=a) == classical_order(a, N)


def test_shor_factor_21_through_the_oracle_order_finder():
    # a = 2: r = 6, 2^3 = 8, gcd(7, 21) = 7 and gcd(9, 21) = 3.
    find_order = CountingOrder(partial(find_order_oracle, seed=1))
    assert shor_factor(21, find_order, FixedBases(2)) == (3, 7)
    assert find_order.calls == [2]


@pytest.mark.parametrize("N", [15, 21, 33, 35, 39, 45, 51, 55, 57, 63])
def test_oracle_factors_every_valid_N_below_64(N):
    # Only bases coprime to N are offered, so every factor comes from an order.
    assert N % 2 == 1 and not is_prime_power(N)
    find_order = CountingOrder(partial(find_order_oracle, seed=np.random.default_rng(N)))
    p, q = shor_factor(N, find_order, FixedBases(*coprime_bases(N)))
    assert p * q == N and 1 < p <= q < N
    assert find_order.calls


# ----------------------------------------------------------------------- inputs


def test_oracle_callable_as_find_order_with_two_arguments():
    assert find_order_oracle(2, 21) == 6


@pytest.mark.parametrize("a, N", [(3, 21), (7, 21), (5, 35)])
def test_oracle_rejects_bases_that_share_a_factor(a, N):
    with pytest.raises(ValueError):
        find_order_oracle(a, N)


@pytest.mark.parametrize("N", [129, 143, 989])
def test_oracle_rejects_N_too_large_for_the_textbook_circuit(N):
    # 3n qubits and 2n dense matrices: N = 143 already takes over a minute
    # per order. Larger N has to wait for the iterative circuit of M3b.
    assert MAX_ORACLE_N == 127
    with pytest.raises(ValueError, match="127"):
        find_order_oracle(2, N)
    with pytest.raises(ValueError, match="127"):
        sample_phases_oracle(2, N, 10)


def test_oracle_draws_every_batch_from_one_generator(monkeypatch):
    seeds = []

    def fake_measure_phases(circuit, shots, seed):
        seeds.append(seed)
        return [0] * shots

    monkeypatch.setattr("shor.order_finding.measure_phases", fake_measure_phases)
    assert find_order_oracle(2, 21, shots=4, seed=5) is None
    assert len(seeds) == MAX_BATCHES
    assert isinstance(seeds[0], np.random.Generator)
    assert all(seed is seeds[0] for seed in seeds)
    assert seeds[0].integers(2**32) == np.random.default_rng(5).integers(2**32)
