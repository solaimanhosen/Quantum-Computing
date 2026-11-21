"""M1: the classical half of Shor's algorithm.

Order finding is the only quantum step. Everything here is ordinary number
theory: a brute-force order (the ground truth for later tests), the input
check, and the continued-fraction post-processing that turns phase estimates
y / 2^t ~ s / r into the order r.
"""

from math import gcd, lcm

# order_from_samples also tries k * d for k up to this value. A shot with
# gcd(s, r) = k yields the denominator r / k, so small k recovers r directly.
MAX_MULTIPLE = 4


def classical_order(a, N):
    """Return the order of a mod N: the smallest r >= 1 with a^r = 1 (mod N).

    Takes up to N steps, which is exponential in n = N.bit_length(). This is
    the function the quantum order finder replaces.
    """
    if gcd(a, N) != 1:
        raise ValueError(f"a = {a} has no order mod {N}: gcd(a, N) != 1")
    r, power = 1, a % N
    while power != 1:
        power = power * a % N
        r += 1
    return r


def is_prime_power(N):
    """Return True if N = p^k for a prime p and some k >= 1 (primes included).

    Shor's algorithm needs two distinct odd prime factors: for N = p^k an even
    order with a^(r/2) != -1 is not guaranteed to exist for half the bases.
    """
    if N < 2:
        return False
    # The smallest divisor > 1 of N is prime.
    p = 2
    while p * p <= N and N % p != 0:
        p += 1
    if p * p > N:
        return True  # no divisor up to sqrt(N): N itself is prime
    # N is a power of p exactly when dividing out p leaves nothing.
    while N % p == 0:
        N //= p
    return N == 1


def convergent_denominators(y, t, N):
    """Return the denominators q < N of the convergents of y / 2^t, in order.

    Writing y / 2^t = [a_0; a_1, a_2, ...] as a continued fraction, the
    denominators of its convergents p_i / q_i satisfy

        q_i = a_i * q_(i-1) + q_(i-2),    q_(-1) = 0,  q_(-2) = 1,

    and the a_i are the quotients of Euclid's algorithm on (y, 2^t). The list
    always starts with q_0 = 1 and is non-decreasing.
    """
    num, den = y, 2**t
    q_before, q = 1, 0  # q_(-2), q_(-1)
    denominators = []
    while den != 0:
        a_i = num // den
        q_before, q = q, a_i * q + q_before
        if q >= N:
            break
        denominators.append(q)
        num, den = den, num - a_i * den  # Euclid step: num / den -> its remainder's reciprocal
    return denominators


def _close_under_lcm(values, bound):
    """Return values together with every lcm of a subset that is below bound."""
    closed = set(values)
    todo = list(closed)
    while todo:
        c = todo.pop()
        for d in list(closed):
            m = lcm(c, d)
            if m < bound and m not in closed:
                closed.add(m)
                todo.append(m)
    return closed


def order_from_samples(ys, t, a, N):
    """Recover the order r of a mod N from phase estimates ys, or return None.

    Each y is an integer with y / 2^t ~ s / r for a random s in 0..r-1. The
    convergent of y / 2^t equal to s / r has denominator r / gcd(s, r), so one
    shot gives a divisor of r, and only gives r itself when gcd(s, r) = 1.

    Candidates for r, all below N because r < N:
      1. every convergent denominator of every shot (y = 0 says nothing: s = 0);
      2. lcms of those across shots, e.g. r/2 and r/3 combine to r;
      3. small multiples k * d of the above, for k = 2..MAX_MULTIPLE.

    Any candidate c with a^c = 1 (mod N) is a multiple of r, so the smallest
    one that passes is r whenever r is among the candidates.
    """
    denominators = set()
    for y in ys:
        if y != 0:
            denominators.update(convergent_denominators(y, t, N))

    candidates = _close_under_lcm(denominators, N)
    candidates |= {
        k * c for c in candidates for k in range(2, MAX_MULTIPLE + 1) if k * c < N
    }

    for c in sorted(candidates):
        if pow(a, c, N) == 1:
            return c
    return None
