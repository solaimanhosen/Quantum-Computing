"""M1: the factoring driver. It reduces factoring N to finding orders mod N.

The order finder is a parameter, so the same driver runs with the classical
brute force (M1), the N = 15 circuits (M4), the oracle (M5) and the full
arithmetic circuit (M7).
"""

from math import gcd

from shor.classical import is_prime_power


def shor_factor(N, find_order, rng, max_attempts=100):
    """Return a non-trivial factorisation (p, q) of N with p <= q.

    find_order(a, N) returns the order r of a mod N, or None if it failed.
    A quantum order finder may also return a wrong value or a proper multiple
    of r; both are detected here and cost one more attempt, never a wrong
    answer. rng supplies the random bases through rng.randrange (e.g.
    random.Random).

    A base that shares a factor with N returns that factor without calling
    find_order at all. For small N this is common (half the bases for N = 15),
    so a test of an order finder must check that it was actually called.

    Why it works: if r is even then a^r - 1 = (a^(r/2) - 1)(a^(r/2) + 1) is a
    multiple of N. N cannot divide the first bracket, because r is the smallest
    exponent with a^r = 1. If N does not divide the second bracket either,
    i.e. a^(r/2) != -1 (mod N), then each bracket shares a proper factor with
    N. For N with at least two distinct odd prime factors, at least half of
    the bases a coprime to N are good in this sense.
    """
    if N < 2 or N % 2 == 0:
        raise ValueError(f"N = {N} must be an odd number greater than 1")
    if is_prime_power(N):
        raise ValueError(f"N = {N} is a prime power; Shor's algorithm does not apply")

    for _ in range(max_attempts):
        a = rng.randrange(2, N - 1)  # a = 1 and a = N - 1 can never work

        # A base that shares a factor with N hands us that factor directly.
        g = gcd(a, N)
        if g != 1:
            return _ordered(g, N // g)

        r = find_order(a, N)
        if r is None:
            continue  # order finding failed
        r = int(r)  # pow() rejects numpy integers
        if r < 1 or pow(a, r, N) != 1:
            continue  # order finding returned a wrong value
        if r % 2 == 1:
            continue  # a^(r/2) is not defined
        half = pow(a, r // 2, N)
        if half == N - 1:
            continue  # a^(r/2) = -1 (mod N): both gcds below would be trivial

        for g in (gcd(half - 1, N), gcd(half + 1, N)):
            if 1 < g < N:
                return _ordered(g, N // g)
        # Only reachable if r was an even multiple of the true order: then
        # a^(r/2) = 1 and both gcds are trivial. Try again.

    raise RuntimeError(f"no factor of {N} found in {max_attempts} attempts")


def _ordered(p, q):
    return (p, q) if p <= q else (q, p)
