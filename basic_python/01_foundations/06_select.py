"""
06 · select: a 0/1 mask times a runtime value (big-M)

    x if m else 0          m is 0 or 1, x is a runtime number

The obvious data-flow answer is  m * x.  It is FORBIDDEN here: both m and x
are runtime values, and a network can only multiply a runtime value by a
CONSTANT (that is what a weight is). So we build the product from relu:

    select(m, x) = relu(x - M*(1 - m)) - relu(-x - M*(1 - m))      |x| <= M

M is a constant that is at least as big as any |x| that can occur ("big-M").
    m = 1:  the shift M*(1 - m) is 0:   relu(x) - relu(-x) = x        (kept)
    m = 0:  the shift is M:             relu(x - M) - relu(-x - M)
            x - M <= 0 and -x - M <= 0 because |x| <= M, so both are 0  (zeroed)

If x is known to be >= 0, the second relu is not needed:
    select_nonneg(m, x) = relu(x - M*(1 - m))

And an if/else with two runtime values:
    x if m else y  =  y + select(m, x - y)          |x - y| <= M

Inputs: m in {0, 1}; n in [0, 6]; x, y in [-6, 6]  (so M = 6, and M = 12 for x - y).
Run:    uv run 01_foundations/06_select.py
"""

import numpy as np

M = 6  # bound on |x|, |y| and n
M2 = 2 * M  # bound on |x - y|


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes.


def control_flow(m, n, x, y):
    return (
        n if m else 0,  # nonnegative select
        x if m else 0,  # signed select
        x if m else y,  # if / else on two runtime values
    )


# ── Stage B · branchless arithmetic ─────────────────────────────────────
# Rules: only  +  -  (multiply by a constant)  relu.
#        No `if`, no comparisons, no loops whose length depends on the input.
#        In particular NEVER write  m * x  with both m and x runtime.


def relu(x):
    return max(0, x)  # the one non-linear primitive


def select_nonneg(m, x, big_m):
    # Requires 0 <= x <= big_m.   m = 1: relu(x) = x.   m = 0: relu(x - big_m) = 0.
    return relu(x - big_m * (1 - m))  # big_m is a constant, (1 - m) is runtime: allowed


def select(m, x, big_m):
    # Requires |x| <= big_m.  m = 1: relu(x) - relu(-x) = x.  m = 0: relu(x - big_m) - relu(-x - big_m) = 0.
    shift = big_m * (1 - m)
    return relu(x - shift) - relu(-x - shift)


def branchless(m, n, x, y):
    return (
        select_nonneg(m, n, M),
        select(m, x, M),
        y + select(m, x - y, M2),  # |x - y| can reach 2 * M
    )


def branchless_two_selects(m, x, y):
    # Another way to write if/else: keep x when m = 1, keep y when m = 0.
    # Each select only needs |value| <= M (not 2 * M), and nothing has to be carried.
    return select(m, x, M) + select(1 - m, y, M)


def too_small_demo():
    # What goes wrong when M is smaller than |x|: select(0, 5, M=3).
    # relu(5 - 3) - relu(-5 - 3) = 2 - 0 = 2, not 0. The mask leaks.
    return select(0, 5, 3)


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
#   v = [m, n, x, y]
#   h = relu(W1 @ v + b1)       7 neurons. M*(1 - m) = M - M*m, and relu(x - M + M*m)
#                               is an ordinary "weight M on m, bias -M".
#        h0 = relu( n + M*m - M)                 nonneg select of n
#        h1 = relu( x + M*m - M)                 signed select of x, positive part
#        h2 = relu(-x + M*m - M)                 signed select of x, negative part
#        h3 = relu( (x - y) + M2*m - M2)         select of (x - y), positive part
#        h4 = relu(-(x - y) + M2*m - M2)         select of (x - y), negative part
#        h5 = relu( y),  h6 = relu(-y)           carry y to the output: y = h5 - h6
#   out = W2 @ h               [h0,  h1 - h2,  h3 - h4 + h5 - h6]

W1 = np.array(
    [
        #  m     n     x     y
        [M, 1.0, 0.0, 0.0],
        [M, 0.0, 1.0, 0.0],
        [M, 0.0, -1.0, 0.0],
        [M2, 0.0, 1.0, -1.0],
        [M2, 0.0, -1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, -1.0],
    ]
)
b1 = np.array([-M, -M, -M, -M2, -M2, 0.0, 0.0], dtype=float)

#                h0    h1    h2    h3    h4    h5    h6
W2 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # n if m else 0
        [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],  # x if m else 0
        [0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0],  # x if m else y
    ]
)
b2 = np.zeros(3)


def network(m, n, x, y):
    v = np.array([m, n, x, y], dtype=float)
    h = np.maximum(0, W1 @ v + b1)
    out = W2 @ h + b2
    return tuple(int(t) for t in out)


# ── Check: all three agree on every input ───────────────────────────────

if __name__ == "__main__":
    count = 0
    for m in (0, 1):
        for n in range(0, M + 1):
            for x in range(-M, M + 1):
                for y in range(-M, M + 1):
                    expected = control_flow(m, n, x, y)
                    assert branchless(m, n, x, y) == expected, (m, n, x, y)
                    assert network(m, n, x, y) == expected, (m, n, x, y, network(m, n, x, y))
                    assert branchless_two_selects(m, x, y) == expected[2], (m, x, y)
                    count += 1

    print(" m  n   x   y | n?  x?0  x?y     (m ? value : 0 or y)")
    for m, n, x, y in ((1, 4, -5, 2), (0, 4, -5, 2), (1, 0, 6, -6), (0, 3, 6, -6)):
        o = network(m, n, x, y)
        print(f"{m:2} {n:2} {x:3} {y:3} | {o[0]:2}  {o[1]:3}  {o[2]:3}")

    print()
    print("Demo (not asserted): M too small. select(m=0, x=5, M=3) should be 0:")
    print(f"  got {too_small_demo()}  <- the mask leaked, because |x| = 5 > M = 3")
    print(f"OK: all {count} inputs agree")
