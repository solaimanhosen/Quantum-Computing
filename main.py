"""Run the project: factor an integer with Shor's algorithm.

    python main.py 989
    python main.py 21 --seed 3
    python main.py 15 --method n15

Order finders (--method):
  classical  brute force, no quantum circuit; any valid N (milestone M1)
  n15        textbook phase estimation with hand-built multipliers on a
             simulator; N = 15 only (milestone M4)
The order finders of M5 and M7 are added to ORDER_FINDERS as they are built.
"""

import argparse
import random
import sys
from functools import partial
from math import gcd

import numpy as np

from shor.classical import classical_order
from shor.factor import shor_factor
from shor.order_finding import find_order_15

# name -> (make_finder, description). make_finder(seed) returns a function
# find_order(a, N) that gives the order of a mod N or None (plan section 3).
# The quantum finders draw all their shots from one generator, so a run is
# repeatable with --seed and successive calls still get different shots.
ORDER_FINDERS = {
    "classical": (
        lambda seed: classical_order,
        "brute-force classical order finder, no quantum circuit",
    ),
    "n15": (
        lambda seed: partial(find_order_15, seed=np.random.default_rng(seed)),
        "textbook phase estimation on 12 simulated qubits, hand-built multipliers, N = 15 only",
    ),
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
    parser.add_argument("--seed", type=int, default=None, help="seed for the random bases and the simulator; omit for a different run each time")
    args = parser.parse_args(argv)

    make_finder, description = ORDER_FINDERS[args.method]
    find_order = make_finder(args.seed)
    N = args.N
    if args.method == "n15" and N != 15:
        parser.error("--method n15 only factors N = 15")
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
