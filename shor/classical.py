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
    if N < 2:
        raise ValueError(f"N = {N} must be at least 2")
    if gcd(a, N) != 1:
        raise ValueError(f"a = {a} has no order mod {N}: gcd(a, N) != 1")
    r, power = 1, a % N
    while power != 1:
        power = power * a % N
        r += 1
    return r


def is_prime_power(N):
    """Return True if N = p^k for a prime p and some k >= 1 (primes included).

    Shor's algorithm needs two distinct odd prime factors. For an odd prime
    power the units mod N form a cyclic group, whose only square roots of 1
    are +1 and -1. So whenever r is even, a^(r/2) = -1 (mod N), and the gcd
    step of shor_factor never produces a factor.
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


def convergents(y, t, N):
    """Return the convergents (p, q) of y / 2^t with q < N, in order.

    Writing y / 2^t = [a_0; a_1, a_2, ...] as a continued fraction, its
    convergents p_i / q_i satisfy

        p_i = a_i * p_(i-1) + p_(i-2),    p_(-1) = 1,  p_(-2) = 0,
        q_i = a_i * q_(i-1) + q_(i-2),    q_(-1) = 0,  q_(-2) = 1,

    and the a_i are the quotients of Euclid's algorithm on (y, 2^t). Each
    p_i / q_i is in lowest terms. The list always starts with (0, 1), and the
    denominators are non-decreasing.
    """
    y = int(y)  # a numpy integer would overflow or break pow() later
    if not 0 <= y < 2**t:
        raise ValueError(f"y = {y} is not a {t}-bit phase estimate")
    num, den = y, 2**t
    p_before, p = 0, 1  # p_(-2), p_(-1)
    q_before, q = 1, 0  # q_(-2), q_(-1)
    result = []
    while den != 0:
        a_i = num // den
        p_before, p = p, a_i * p + p_before
        q_before, q = q, a_i * q + q_before
        if q >= N:
            break
        result.append((p, q))
        num, den = den, num - a_i * den  # Euclid step
    return result


def order_candidates(y, t, N):
    """Return the denominators q that one shot y supports as divisors of r.

    Phase estimation of the phase s / r puts at least 8/pi^2 = 81% of the
    probability on the two integers next to the peak s * 2^t / r. So a
    convergent p / q of y / 2^t is kept only if y is that close to its peak:

        |y - p * 2^t / q| < 1,    i.e.    |y * q - p * 2^t| < q.

    Without this window every convergent would be a candidate, and the
    a^c = 1 check in order_from_samples would find r from random y by trial
    and error. A y that is not next to any peak returns the empty set.

    Is s / r in lowest terms always among the convergents when y is next to
    its peak? Legendre's theorem says yes if |y / 2^t - s / r| < 1 / (2 r^2).
    With t = 2n we have 2^t > N^2, so this holds
      - for the nearest integer (error <= 2^-(t+1)) whenever r < N;
      - for both neighbours (error < 2^-t) whenever r < N / 2, which is true
        for every valid Shor input: N has two distinct odd prime factors, so
        its unit group is not cyclic and r <= phi(N) / 2 < N / 2.

    y = 0 is the exact peak of s = 0 and returns {1}: it only says that r is
    a multiple of 1.
    """
    y = int(y)
    return {q for p, q in convergents(y, t, N) if abs(y * q - p * 2**t) < q}


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

    Each y is an integer with y / 2^t ~ s / r for a random s in 0..r-1. A shot
    next to its peak yields the denominator r / gcd(s, r) (order_candidates),
    so one shot gives a divisor of r, and r itself only when gcd(s, r) = 1.

    Candidates for r, all below N because r < N:
      1. the denominators supported by each shot;
      2. lcms of those across shots, e.g. r/2 and r/3 combine to r;
      3. small multiples k * d of the above, k = 2..MAX_MULTIPLE, for d > 1.
         (d = 1 comes only from y = 0, which carries no information; its
         multiples would be a blind search for small orders.)

    The result is the smallest candidate c with a^c = 1 (mod N). Such a c is
    always a multiple of r, and it equals r whenever r is among the
    candidates. It can be a proper multiple of r only when no shot led to r
    and a stray shot led to a multiple, for example a y halfway between two
    peaks, which looks like a peak of the order 2r. shor_factor detects that
    case and retries.
    """
    a, N = int(a), int(N)
    supported = set()
    for y in ys:
        supported |= order_candidates(y, t, N)

    candidates = _close_under_lcm(supported, N)
    candidates |= {
        k * d
        for d in candidates
        if d > 1
        for k in range(2, MAX_MULTIPLE + 1)
        if k * d < N
    }

    for c in sorted(candidates):
        if pow(a, c, N) == 1:
            return c
    return None
