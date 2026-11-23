"""Run the project: factor an integer with Shor's algorithm.

    python main.py 989
    python main.py 21 --seed 3
    python main.py 989 --method classical

Only the classical order finder exists so far (milestone M1). The quantum
order finders of M4, M5 and M7 are added to ORDER_FINDERS as they are built.
"""

import argparse
import random
import sys
from math import gcd

from shor.classical import classical_order
from shor.factor import shor_factor

# name -> (find_order, description). Every finder can be called as
# find_order(a, N) and returns the order of a mod N or None (plan section 3).
ORDER_FINDERS = {
    "classical": (classical_order, "brute-force classical order finder, no quantum circuit"),
}

def describe_order(a, r, N):
    """Say what shor_factor will do with the order r reported for the base a."""
    if r is None:
        return "order finding failed; trying another base"
    r = int(r)
    if r < 1 or pow(a, r, N) != 1:
        return f"got r = {r}, but a^r ≢ 1 (mod N); trying another base"
    if r % 2 == 1:
        return f"order r = {r} is odd; trying another base"
    half = pow(a, r // 2, N)
    if half == N - 1:
        return f"order r = {r}, but a^(r/2) ≡ −1 (mod N); trying another base"
    g_minus, g_plus = gcd(half - 1, N), gcd(half + 1, N)
    text = f"order r = {r}, a^(r/2) ≡ {half} (mod N), gcd({half} ± 1, N) = {g_minus} and {g_plus}"
    if not (1 < g_minus < N or 1 < g_plus < N):
        text += "; no factor, trying another base"  # r was a multiple of the order
    return text

def traced(find_order, calls):
    """Wrap an order finder so that each call is printed and counted."""

    def wrapper(a, N):
        r = find_order(a, N)
        calls.append(a)
        print(f"  a = {a}: {describe_order(a, r, N)}")
        return r

    return wrapper

def main(argv=None):
    parser = argparse.ArgumentParser(description="Factor N with Shor's algorithm.")
    parser.add_argument("N", type=int, help="odd composite that is not a prime power, e.g. 15, 21, 989")
    parser.add_argument(
        "--method",
        choices=sorted(ORDER_FINDERS),
        default="classical",
        help="order finder to use (default: classical)",
    )
    parser.add_argument("--seed", type=int, default=None, help="seed for the random bases; omit for a different run each time")
    args = parser.parse_args(argv)

    find_order, description = ORDER_FINDERS[args.method]
    N = args.N
    print(f"Factoring N = {N} (n = {N.bit_length()} bits)")
    print(f"Order finder: {args.method} ({description})")

    calls = []
    try:
        p, q = shor_factor(N, traced(find_order, calls), random.Random(args.seed))
    except ValueError as error:
        parser.error(str(error))  # prints usage and exits with status 2
    except RuntimeError as error:
        print(error)
        return 1

    if not calls:
        print("  the first random base already shared a factor with N; no order finding needed")
    print(f"{N} = {p} × {q}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
