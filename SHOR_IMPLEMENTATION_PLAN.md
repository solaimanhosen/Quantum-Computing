# Shor's Factoring Algorithm in Qiskit — Implementation Plan

Revised 2026-11-20; M1 and the exit tests that depend on it were revised again the same day after a code review (see M1). M2–M8 were revised on 2026-11-21 after a plan review that prototyped each milestone; no step was found to fail, and the changes make tests and interfaces precise. Learning project: build Shor's algorithm from first principles, one tested layer at a time, on a simulator.

## 1. Scope and targets

| Target | What runs | Qubits | Cost (measured on 2 CPU cores; Aer statevector with `max_parallel_threads=2` unless noted) |
|---|---|---|---|
| **Primary: N ≤ 50, full reversible arithmetic** | Complete Shor circuit, no shortcuts | 2n+3 = 15 at n = 6 | ~22.7k gates, ~1.3 s per shot |
| Secondary: N ≤ 1000, oracle U_a | QPE + post-processing with U_a as a permutation matrix | n+1 = 11 at n = 10 | ~0.3 s per shot with dense matrices on `qiskit.quantum_info.Statevector` |
| Optional capstone: larger N, full arithmetic | Same code as primary | 19 / 21 / 23 | N=143: 45 s/shot; N=323: 284 s/shot; N=989: ~25–30 min/shot (extrapolated, not run) |
| Optional: IBM hardware | Compiled N = 15 only | ≤ 5 | The general circuit is far too deep for current devices at any N |

Notation: N is the number to factor, n = N.bit_length(), a is the base with gcd(a, N) = 1, r is the order of a mod N, t = 2n is the number of phase bits.

Valid inputs are odd composites that are not prime powers. For N ≤ 50 these are 15, 21, 33, 35, 39, 45.

**Design decision that everything depends on:** the textbook layout (2n counting qubits plus arithmetic registers, 4n+2 total) needs 26 qubits at n = 6 and 42 at n = 10. We therefore use *iterative* phase estimation with one recycled control qubit (the semiclassical QFT) for all order finding beyond the N = 15 warm-up. Textbook QPE is still implemented first, because it is how the method is understood and tested.

## 2. Build order at a glance

| Milestone | Component | Depends on | First used by |
|---|---|---|---|
| M0 | Project setup, conventions | — | everything |
| M1 | Classical shell: number theory, continued fractions, factor driver | M0 | M4, M5, M7 |
| M2 | **QFT and inverse QFT** (with and without swaps) | M0 | M3 (inverse QFT), M6 (no-swap QFT) |
| M3 | **Quantum phase estimation**: textbook, then iterative | M2 | M4, M5, M7 |
| M4 | First end-to-end factoring: N = 15 with hand-built U_a | M1, M3 | — |
| M5 | Oracle order finding for any N ≤ 1000 | M1, M3 | M7 (same interface) |
| M6 | Reversible modular arithmetic (Draper / Beauregard) | M2 | M7 |
| M7 | Full Shor for N ≤ 50 | M1, M3, M5, M6 | M8 |
| M8 | Analysis and optional extensions | M7 | — |

M2 is implemented exactly once and reused twice: its inverse closes textbook QPE in M3, and its no-swap form is the basis change for the Fourier-space adder in M6. The iterative QPE of M3 contains no explicit QFT circuit; the inverse QFT is replaced by classically controlled phase corrections.

## 3. Conventions (fix in M0, never change)

- **Bit order:** Qiskit little-endian. Qubit 0 of a register is the least significant bit; a register holds the integer Σ_j q_j 2^j.
- **QFT definition:** QFT|x⟩ = 2^(−m/2) Σ_k exp(2πi·x·k / 2^m) |k⟩ on m qubits. This matches `qiskit.circuit.library.QFTGate` (verified on Qiskit 2.6.2).
- **No-swap QFT:** `qft(m, do_swaps=False)` leaves qubit j in (|0⟩ + exp(2πi·x / 2^(j+1)) |1⟩)/√2. Following it with the bit-reversal swaps gives the QFT above (verified numerically).
- **Phase estimate:** every QPE variant returns an integer y with y / 2^t ≈ phase.
- **Environment:** Python ≥ 3.10, `qiskit` 2.6.x, `qiskit-aer` 0.18.x (the versions the measurements above were taken with), `numpy`, `pytest`.
- **Commits:** one commit per passed exit test, made with `git commit -s` on `main`. No `Co-Authored-By` or session trailer lines in commit messages; the `Signed-off-by` line that `-s` adds is wanted.
- **Order finder interface:** every order finder can be called as `find_order(a, N)` and returns a Python `int` or `None`. Extra parameters such as `shots` have defaults, so the function can be passed to `shor_factor` unchanged (`functools.partial` changes a default). The value returned is a multiple of r, almost always r itself; `None` means the shots did not determine it.
- **Seeding:** each `find_order_*` and `sample_phases_*` call takes `seed` (an int or a `numpy.random.Generator`), creates one `rng = numpy.random.default_rng(seed)` and draws every shot and every batch from it. For `Statevector`, pass the Generator itself to `sv.seed(rng)`; for Aer, pass `seed_simulator=int(rng.integers(2**32))` per run. Never reuse one integer seed per shot or per batch: that makes every shot identical, or every later batch a copy of the first, and no exit test would notice.
- **What counts as evidence:** a correct order or factorisation is weak evidence that a circuit works. The classical post-processing checks a^c ≡ 1 (mod N), so for small N it can reach r from poor samples, and `shor_factor` returns a factor without any order finding when the random base shares one with N (half the bases for N = 15). Every quantum milestone therefore has an exit test on the measured y distribution, and every end-to-end test asserts that the order finder was called.

## 4. Suggested layout

```
shor/
  classical.py       # M1
  qft.py             # M2
  qpe.py             # M3
  oracles.py         # M4 (N=15 circuits), M5 (permutation oracle)
  arithmetic.py      # M6
  order_finding.py   # M4, M5, M7: find_order_15, find_order_oracle, find_order_arith
  factor.py          # M1 driver, takes any find_order
tests/               # one test file per module
experiments/         # M8 scripts and results
requirements.txt
```

## 5. Milestones

Each milestone ends with an exit test. Do not start the next milestone until it passes. One agreed exception (2026-11-23): M3b is postponed and M4 was built first, since M4 uses only the textbook circuit of M3a. M3b and its exit test 3 must be done before the rest of M5. The first half of M5 was also built ahead of M3b (see M5).

### M0 — Setup

- Create the virtual environment, `requirements.txt`, package skeleton, and `pytest` configuration.
- Add a smoke test: build a Bell circuit, run it on `AerSimulator`, and check the counts.

### M1 — Classical shell

Implement in `classical.py` and `factor.py`:

- `classical_order(a, N)`: brute-force order, used as ground truth in every later test. Raises `ValueError` if N < 2 or gcd(a, N) ≠ 1.
- `is_prime_power(N)`: true for p^k with k ≥ 1, so primes are included.
- `convergents(y, t, N)`: the continued-fraction convergents (p, q) of y / 2^t with q < N. Raises `ValueError` unless 0 ≤ y < 2^t, which catches a wrong t or bit order in the caller.
- `order_candidates(y, t, N)`: the denominators q of those convergents that satisfy |y·q − p·2^t| < q, meaning y is one of the two integers next to the peak p·2^t / q. A y that is next to no peak gives no candidates.
- `order_from_samples(ys, t, a, N)`: take the candidates of every shot, add their lcms across shots (below N), then the multiples k·d for k = 2, 3, 4 and d > 1. Return the smallest candidate c with a^c ≡ 1 (mod N), or `None`. Inputs may be numpy integers; the result is a Python `int`.
- `shor_factor(N, find_order, rng, max_attempts=100)`: reject even N and prime powers, pick random a in 2..N−2, return gcd(a, N) if it is non-trivial, call `find_order(a, N)`, retry if it returns `None` or a value with a^r ≢ 1, if r is odd, or if a^(r/2) ≡ −1 (mod N); otherwise return gcd(a^(r/2) ± 1, N) as (p, q) with p ≤ q. If both gcds are trivial (r was an even multiple of the order), retry. Raise `RuntimeError` after `max_attempts`.

Properties of `order_from_samples` to keep in mind later:

- **The window in `order_candidates` is essential.** The first version took every convergent of every shot as a candidate. Combined with the lcm step, the multiples and the a^c ≡ 1 check, that was a classical search: from 20 uniformly random y it returned r = 154 for N = 989 in 96% of trials. With the window the figure is below 1%, while ideal QPE samples still give r in about 72% of trials from one shot, 98% from three and over 99% from five (measured for N = 21, 35, 143 and 989).
- **It cannot be made blind for very small N.** For N = 15 there are only 14 possible denominators, so 20 random shots still return 4 in about 19% of trials and a wrong multiple (8 or 12) in about 67%. Results for N = 15 say little about the circuit that produced the samples.
- **The result is a multiple of r, not always r.** It equals r whenever r is among the candidates. A y halfway between two peaks looks like a peak of the order 2r, so a proper multiple is possible when no other shot leads to r; in ideal QPE samples this was not observed beyond 0.3% of single-shot trials. The multiple is deliberately not reduced to r by dividing out primes: that would turn every multiple of r below N into a hit and make random samples succeed far more often. `shor_factor` detects the case and retries.
- **y = 0 is not discarded.** It yields only the candidate 1, which passes only for a ≡ 1 (order 1), and its multiples are not tried.

Exit tests:

1. With `find_order = classical_order`, `shor_factor` factors every valid N ≤ 1000, both with random bases and when offered only bases coprime to N (so that every factorisation comes from an order).
2. For several (a, N), feed the ideal samples y = round(s·2^t / r) for all s into `order_from_samples`; it returns r.
3. Both integers next to s·2^t / r give the candidate r / gcd(s, r), for every r < N / 2.
4. Samples drawn from the exact QPE distribution give r from 5 shots in at least 95% of trials, and 20 uniformly random samples give r = 154 for N = 989 in fewer than 5% of trials.

Concept checks:

- Why does t = 2n guarantee that s/r appears among the convergents of y / 2^t when y is the integer nearest to s·2^t / r?
- Show that r < N / 2 for every valid input, and that this extends the guarantee to both neighbours of the peak.

### M2 — QFT and inverse QFT

Implement `qft(m, do_swaps=True, inverse=False) -> QuantumCircuit` in `qft.py` from H and controlled-phase gates only. Do not call the library QFT inside it.

Exit tests (m = 1 to 5):

1. `Operator(qft(m))` equals the DFT matrix from the definition in section 3.
2. It equals `Operator(QFTGate(m))`.
3. `qft(m, inverse=True)` composed with `qft(m)` is the identity.
4. For every basis state |x⟩, `qft(m, do_swaps=False)` produces the product state stated in section 3.

Concept check: derive the product form of QFT|x⟩ and read the circuit off it.

### M3 — Quantum phase estimation

Both variants take the same arguments, `(controlled_power, t, num_target, prepare_target=None)`:

- `controlled_power(j)` returns a circuit on 1 + num_target qubits, control first (qubit 0, as in M5), that implements controlled-U^(2^j).
- `prepare_target`, if given, is a circuit applied to the target register first (X on its qubit 0 to get |1⟩).

**M3a, textbook QPE.** `qpe_circuit(...)`:

- Counting register of t qubits, each prepared with H.
- Counting qubit j controls U^(2^j) on the target register.
- Apply `qft(t, inverse=True)` (with swaps) to the counting register and measure it; the little-endian integer is y.

Two helpers in `qpe.py` are shared with later milestones:

- `exact_phase_distribution(circuit, t)`: the exact P(y) of a textbook circuit, from `Statevector` with the final measurements removed. It does not apply to M3b, which measures mid-circuit.
- `measure_phases(circuit, shots, seed=None)`: run either variant on Aer and return the measured y of each shot as Python ints, following the seeding rule of section 3. The exact distribution bypasses the measurements, so at least one test per circuit must go through this function to check the wiring of the classical bits.

**M3b, iterative QPE.** `iterative_qpe_circuit(...)`. One control qubit, one t-bit classical register m, t rounds. In round k = 0, …, t−1:

1. Reset the control and apply H.
2. Apply controlled-U^(2^(t−1−k)).
3. For each j < k, inside `with qc.if_test((m[j], 1)):` apply P(−2π / 2^(k−j+1)) to the control. Together these apply P(−2π · Σ_{j<k} m_j / 2^(k−j+1)). A gate angle cannot depend on a measured bit at run time, so it has to be one `if_test` per earlier bit (190 of them at t = 20, which Aer handles without trouble).
4. Apply H and measure into classical bit m_k.

The estimate is y = Σ_k m_k 2^k = `int(memory_string, 2)`. Round 0 measures the least significant phase bit.

Put the angle arithmetic in a helper `correction_angle(k, bits)` and test it here: M5 runs the same rounds without this circuit and reuses it.

Exit tests, with U = P(2πφ) acting on |1⟩:

1. φ = 5/16, t = 4: both variants return y = 5 with probability 1.
2. φ = 1/3, t = 6: the nearest estimate y = 21 has probability ≥ 4/π² ≈ 0.405.
3. For φ = 1/3, t = 6, the histogram of 4000 shots of M3b is within total variation distance 0.05 of the exact M3a probabilities (from `Statevector`, final measurements removed). Sampling noise alone gives about 0.02, and at most 0.03 in 99.9% of runs.

Concept check: show that the round-k correction phase is exactly what the inverse QFT would have applied to that qubit.

### M4 — First factoring: N = 15

- In `oracles.py`, build controlled-U_a (a control plus 4 target qubits) for a ∈ {2, 4, 7, 8, 11, 13, 14} from CSWAP and CX gates. Hint: multiplying by 2 mod 15 is a cyclic rotation of the 4 bits, and multiplying by −1 mod 15 is a bitwise NOT. The bases 2, 4, 8 need rotations only; 14, 13, 11, 7 are a rotation followed by NOT.
- The powers are U^(2^j) = U_{a^(2^j) mod 15}, and a^(2^j) mod 15 is always a, 4 or 1; U_1 is the identity.
- Run textbook QPE with t = 8 on the target state |1⟩ (12 qubits total).
- Add `find_order_15(a, N, shots=10, seed=None)` to `order_finding.py`: sample this circuit and post-process with M1.
- `order_from_batches(sample, t, a, N)` in `order_finding.py` holds the batching rule (up to three batches for the same base, all shots post-processed together). `find_order_15` uses it, and the finders of M5 and M7 reuse it.

Exit tests:

1. Each U_a maps |y⟩ to |a·y mod 15⟩ for y = 1, …, 14.
2. For a = 7, the measured y is 0, 64, 128 or 192, each with probability exactly 1/4 (computed with `Statevector`, final measurements removed). For each of the seven bases, the only outcomes are the multiples of 256 / r. This is the test that shows the circuit works.
3. The order returned agrees with `classical_order` for each of the seven bases, and `shor_factor(15, …)` returns 3 × 5 when the base is fixed to a = 7 and the order finder is confirmed to have been called. On its own this test proves little at N = 15 (see M1).

### M5 — Oracle order finding for N ≤ 1000

- `controlled_permutation(a, N)`: the 2^(n+1)-dimensional matrix of controlled-U_a with the control as qubit 0 (index = c + 2y). States y ≥ N map to themselves so the matrix is unitary.
- `find_order_oracle(a, N, shots=10, seed=None)`: iterative QPE (M3b) with U^(2^j) = U_{a^(2^j) mod N}, target initialised to |1⟩, post-processed with M1. If `order_from_samples` returns `None`, take another batch of `shots` for the same a and post-process all shots together, up to three batches, before returning `None`; a new base would throw the earlier shots away.
- `sample_phases_oracle(a, N, shots, seed=None)`: the same circuit without the post-processing, returning the list of measured y. `find_order_oracle` is built on it, and the distribution tests use it directly.

This oracle is tabulated classically, so it validates QPE and post-processing; it is not a scalable circuit.

Status (2026-11-23): the first half is built, on the textbook circuit of M3a instead of M3b. `controlled_permutation`, `controlled_power_oracle`, `order_finding_circuit_oracle`, `sample_phases_oracle` and `find_order_oracle` exist with the signatures above and are tested for N = 15 to 63, including the exact y distribution for N = 21 and 35. The textbook circuit needs 3n qubits and 2n dense matrices, so it is capped at `MAX_ORACLE_N = 127`: measured time per order was 0.06 s at n = 5, 0.4 s at n = 6, 5 s at n = 7 and 93 s at n = 8. What remains for M5 is to run the M3b rounds on a `Statevector` as described below, raise the cap to 1000, and pass exit tests 1 to 3; the function names and signatures stay the same.

M5 does not run the M3b circuit: `Statevector` cannot execute `if_test`. It runs the M3b rounds in a Python loop on a `Statevector` and takes the correction angle from `correction_angle(k, bits)` of M3. After each round, continue from the state that the measurement returns, `outcome, sv = sv.measure([0])`, and apply X to the control if the outcome was 1. Discarding that state and letting a reset re-sample the control loses the correlation with the target: in a prototype the fraction of shots next to a peak fell from 0.90 to 0.28.

Known pitfalls (hit during prototyping):

- `UnitaryGate(P).control()` fails untranspiled in Aer with "unknown instruction: c-unitary". Transpiling synthesises the whole unitary into elementary gates (236 CX already at 4 target qubits), which is infeasible at n = 10. Build the controlled matrix directly.
- Twenty dense 2048×2048 matrices in one Aer circuit exceeded 7 GB at n = 10 (2.5 GB at n = 9). Simulate round by round with `Statevector.evolve` and `Statevector.measure` instead; that used 0.22 GB.
- Speed, optional: evolving with the dense matrix costs 0.2–0.3 s per shot at n = 10, so exit test 3 takes one to two minutes and exit test 2 several. Applying the permutation to the amplitude array with numpy indexing instead took 0.003 s per shot in a prototype.

Exit tests:

1. N = 989, a = 2 returns r = 154 and the factors 23 × 43.
2. `shor_factor` with this order finder factors every valid N ≤ 100 and a random sample of 50 valid N in (100, 1000], offering only bases coprime to N so that the order finder is called for every N.
3. Distribution: for N = 35, a = 2 (r = 12) and for N = 989, a = 2, at least 75% of 400 measured y are one of the two integers next to a peak s·2^t / r (theory: at least 8/π² ≈ 81%; the exact value is 90.3% for both), and for N = 35 all 12 values of s occur.

Concept check: for r = 154, which eigenphases s/r yield a candidate smaller than r, and what fraction of shots return r exactly?

### M6 — Reversible modular arithmetic

Registers: control c (1), x (n), b (n+1, the extra qubit catches overflow), ancilla (1). All adders act on b in the Fourier basis of the no-swap QFT from M2: every "QFT" in this milestone means `qft(n+1, do_swaps=False)` on b, and every "inverse QFT" means `qft(n+1, do_swaps=False, inverse=True)`. The default swap version breaks every case. Build and test bottom-up in `arithmetic.py`.

**M6a, φADD(a).** Add a classical constant: apply P(2π·a / 2^(j+1)) to qubit j of b. Subtraction is the inverse. Provide singly and doubly controlled versions (`cp`, `mcp`).

**M6b, φADD(a) mod N** with two controls, valid for a, b < N:

1. Doubly controlled φADD(a).
2. Inverse φADD(N), uncontrolled.
3. Inverse QFT; CNOT from b's top qubit to the ancilla; QFT.
4. φADD(N) controlled by the ancilla.
5. Inverse doubly controlled φADD(a).
6. Inverse QFT; X on b's top qubit; CNOT from it to the ancilla; X again; QFT.
7. Doubly controlled φADD(a).

Only the three φADD(a) gates (steps 1, 5, 7) are doubly controlled. Steps 2, 3 and 6 are uncontrolled, and step 4 is controlled by the ancilla only. With the controls off, steps 2–6 must cancel on their own; controlling step 2 as well passes every test with the controls on and fails every case with them off. Steps 5–7 uncompute the ancilla. This is the hardest block of the project.

**M6c, CMULT(a) mod N.** QFT on b; for i = 0, …, n−1 apply φADD(2^i·a mod N) mod N controlled by c and x_i; inverse QFT. Effect: |x⟩|b⟩ → |x⟩|b + a·x mod N⟩ when c = 1.

**M6d, controlled-U_a.** CMULT(a); controlled-SWAP between x and the low n qubits of b; inverse of CMULT(a⁻¹ mod N). Effect: |x⟩|0⟩ → |a·x mod N⟩|0⟩ when c = 1, identity when c = 0.

Exit tests, exhaustive over basis states with `Statevector` for N = 5, 7 and 15. In each, check that the output is the expected basis state with amplitude 1 to within 1e-9, phase included; a probability check misses stray phases (for example from `crz` used in place of `cp`). M6a and M6b act in the Fourier basis, so wrap them in the no-swap QFT and its inverse on b for the test.

1. M6a: b → b + a mod 2^(n+1) for every a < 2^(n+1), for the uncontrolled version and for the controlled versions with all controls 1; b unchanged when any control is 0.
2. M6b: b → (a + b) mod N for all a, b < N when both controls are 1, and b unchanged for the other three control settings. In every case the ancilla and b's top qubit return to 0.
3. M6c: |x⟩|b⟩ → |x⟩|b + a·x mod N⟩ for all x, b < N when c = 1, and unchanged when c = 0.
4. M6d: x → a·x mod N for all x < N and every a coprime to N; b and the ancilla return to 0; identity when c = 0.

The exhaustive tests are slow with `Statevector`: about 3.5 minutes in total in a prototype, of which M6c at N = 15 over all eight bases took 141 s. Mark the N = 15 cases with a `slow` pytest marker, or run M6c at N = 15 for two bases only.

Concept check: why does the ancilla have to be uncomputed, and what would go wrong in QPE if it were left entangled?

### M7 — Full Shor for N ≤ 50

- `find_order_arith(a, N, shots=10, seed=None)` and `sample_phases_arith(a, N, shots, seed=None)`: `iterative_qpe_circuit` of M3b with the controlled-U_a of M6d, on 2n+3 qubits: num_target = 2n+2 (x, b, ancilla), and `prepare_target` is X on x_0. Same signatures, batching and seeding as the oracle versions in M5.
- Run on `AerSimulator(method="statevector")`, transpiled with `optimization_level=0`, `memory=True`. Resets, mid-circuit measurements and `if_test` all run natively.
- Speed, optional: `shot_branching_enable=True` ran 400 shots at N = 21 in 1.5 s instead of 17 s with an identical histogram. Its memory comes back grouped by outcome, so shuffle it with the seeded rng before splitting it into batches.

Exit tests:

1. For N = 15 and 21, the order returned agrees with `classical_order` for every a coprime to N in 2..N−2. Fix the simulator seed so the test is repeatable.
2. `shor_factor` factors each of 15, 21, 33, 35, 39, 45, offering only bases coprime to N so that the order finder is called.
3. For N = 21, a = 2 (r = 6, t = 10), take 400 shots from `sample_phases_arith` with a fixed seed. At least 75% are next to a peak (theory 90.3%). Assign every shot to its nearest peak, s = round(y·r / 2^t) mod r; each of the 6 values of s receives 1/6 ± 0.07 of the shots. Apply the same two checks to 400 shots of `sample_phases_oracle`. Do not compare the raw 1024-bin histograms: two 400-shot samples of the same distribution differ by a total variation distance of about 0.13. This is the test that shows the arithmetic circuit works; tests 1 and 2 would pass with a mediocre circuit at these sizes.

Reference numbers (a = 2): N = 15 uses 11 qubits and 6.0k gates; N = 21 uses 13 qubits and 12.3k gates; N = 33–45 use 15 qubits and about 22.7k gates at roughly 1.3 s per shot. For r = 12 only about a third of shots give r directly, so the lcm step from M1 matters. Test costs: N = 21 runs at about 0.05 s per shot, so test 3 takes about 17 s; test 2 takes roughly a minute. A 400-shot distribution test at n = 6 would take 8–9 minutes, so do not add one without shot branching.

### M8 — Analysis and optional extensions

Pick any; none is required for the primary target.

1. **Scaling:** tabulate qubits, gate count and depth against n; compare with the expected polynomial growth. Also transpile to {cx, rz, sx, x} and count CNOTs (N = 989 gave about 265k CNOTs at depth 240k).
2. **Capstone:** run full arithmetic at N = 143 or 323; N near 1000 is an overnight job.
3. **Approximate QFT:** the primary path (M3b) contains no QFT circuit, so apply the cut in both places it has an effect: in M3b skip the corrections with k−j+1 > d, and in M6 drop rotations smaller than 2π / 2^d in the adders' QFTs and in φADD. Plot the success probability against d.
4. **Noise:** add an Aer depolarising noise model and find the error rate at which N = 15 stops working, using M4's U_a with iterative QPE on 5 qubits (the circuit of item 5).
5. **Hardware:** run a compiled N = 15 circuit (M4's U_a with iterative QPE) on an IBM device and compare with the noise model. This demonstrates QPE on hardware, not scalable factoring, because the compiled U_a uses knowledge of the problem's structure.

## 6. Working through this plan with Claude Code

- One milestone per session, in order. Write the tests for a milestone before its implementation.
- Stop at each exit test and review the code and the concept check before moving on.
- Keep functions small and commented with the maths they implement; this is a study project, so readability beats cleverness.
- Do not replace hand-built blocks (QFT, adders) with library gates. Note that Qiskit's `ModularAdderGate` adds two registers modulo 2^n, which is not addition of a constant modulo N.

## 7. References

- P. Shor, "Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer", arXiv:quant-ph/9508027.
- M. Nielsen and I. Chuang, *Quantum Computation and Quantum Information*, chapter 5 (QFT, phase estimation, order finding).
- T. Draper, "Addition on a Quantum Computer", arXiv:quant-ph/0008033.
- S. Beauregard, "Circuit for Shor's algorithm using 2n+3 qubits", arXiv:quant-ph/0205095.
- R. Griffiths and C.-S. Niu, "Semiclassical Fourier Transform for Quantum Computation", arXiv:quant-ph/9511007.
- J. Smolin, G. Smith and A. Vargo, "Oversimplifying quantum factoring", Nature 499, 163 (2013); arXiv:1301.7007, where it is titled "Pretending to factor large numbers on a quantum computer" (on compiled hardware demonstrations).
