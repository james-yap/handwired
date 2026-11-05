"""
05 · max, min, abs, clamp, and stacking them

    max(a, b)    = b + relu(a - b)
    min(a, b)    = a - relu(a - b)
    abs(x)       = relu(x) + relu(-x)
    clamp(x, lo, hi) = lo + relu(x - lo) - relu(x - hi)        (lo <= hi)

These four work for ANY numbers, not just integers, because they use relu
directly, never step(). Part 1 of this file computes all of them from [a, b]
in one hidden layer; note that relu(a - b) is shared by max, min and |a - b|.

Part 2 stacks them: the median of three numbers
    median(a, b, c) = max(min(a, b), min(max(a, b), c))
is a composition, and a composition of helpers is a stack of layers: the output
of one relu layer feeds the next. The price: values that must survive a layer
have to be carried through it (see "carrying" in the network comments).

Inputs: part 1: a, b integers in [-8, 8].  part 2: a, b, c integers in [-6, 6].
Run:    uv run 01_foundations/05_min_max_abs_clamp.py
"""

import numpy as np

LOW, HIGH = -8, 8  # part 1
LO, HI = -3, 4  # part 1: the constant clamp bounds
S = 6  # part 2: inputs live in [-S, S]
NAMES = ("max(a,b)", "min(a,b)", "|a-b|", f"clamp(a,{LO},{HI})")


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes.


def control_flow(a, b):
    return (max(a, b), min(a, b), abs(a - b), min(max(a, LO), HI))


def median3_control_flow(a, b, c):
    return sorted([a, b, c])[1]


# ── Stage B · branchless arithmetic ─────────────────────────────────────
# Rules: only  +  -  (multiply by a constant)  relu.
#        No `if`, no comparisons, no loops whose length depends on the input.


def relu(x):
    return max(0, x)  # the one non-linear primitive


def max2(a, b):
    return b + relu(a - b)  # a >= b: b + (a - b) = a.  a < b: b + 0 = b.


def min2(a, b):
    return a - relu(a - b)  # a >= b: a - (a - b) = b.  a < b: a - 0 = a.


def abs1(x):
    return relu(x) + relu(-x)  # exactly one of the two relus is non-zero


def clamp(x, lo, hi):
    # x <= lo: 0 - 0 -> lo.   lo <= x <= hi: (x - lo) - 0 -> x.
    # x >= hi: (x - lo) - (x - hi) = hi - lo -> hi.  Needs lo <= hi.
    return lo + relu(x - lo) - relu(x - hi)


def branchless(a, b):
    return (max2(a, b), min2(a, b), abs1(a - b), clamp(a, LO, HI))


def median3_branchless(a, b, c):
    # Composition: the result of one helper is the argument of the next.
    return max2(min2(a, b), min2(max2(a, b), c))


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
# Part 1.  x = [a, b]
#   h = relu(W1 @ x + b1)       8 hidden neurons:
#        h0 = relu(a - b)    h1 = relu(b - a)
#        h2 = relu(a)        h3 = relu(-a)      a = h2 - h3   <- a "carried" copy of a
#        h4 = relu(b)        h5 = relu(-b)      b = h4 - h5   <- a "carried" copy of b
#        h6 = relu(a - LO)   h7 = relu(a - HI)
#   y = W2 @ h + b2
#        max   = b + relu(a - b) = h4 - h5 + h0
#        min   = a - relu(a - b) = h2 - h3 - h0
#        |a-b| = h0 + h1
#        clamp = LO + h6 - h7
# The output layer only sees the hidden neurons, never x itself. So to ADD b to
# something the network must first smuggle b through the hidden layer; the trick
# is  b = relu(b) - relu(-b)  (it works for negative b too).

W1 = np.array(
    [
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
b1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -LO, -HI])

#                h0    h1    h2    h3    h4    h5    h6    h7
W2 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],  # max
        [-1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],  # min
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # |a - b|
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],  # clamp (minus the constant LO)
    ]
)
b2 = np.array([0.0, 0.0, 0.0, float(LO)])


def network(a, b):
    x = np.array([a, b], dtype=float)
    h = np.maximum(0, W1 @ x + b1)
    y = W2 @ h + b2
    return tuple(int(v) for v in y)


# Part 2.  median of three = 3 hidden layers.
#
#   Write hi = max(a, b),  lo = min(a, b),  d = hi - lo = |a - b|.
#   u = relu(a - b):          hi = b + u,   lo = a - u
#   v = relu(hi - c):         mid = min(hi, c) = hi - v
#   median = max(lo, mid) = mid + relu(lo - mid)
#          = hi - v + relu(v - d)                   (lo - mid = v - d)
#
#   Each relu takes the layer before it as input, so  u -> v -> relu(v - d)  is
#   three layers deep. Carrying: every value used later must be kept alive.
#   Instead of the pair  x = relu(x) - relu(-x)  we use ONE neuron per value
#   by shifting by S so it is never negative:  relu(x + S) = x + S  for x in [-S, S].
#
#   x  = [a, b, c]
#   h1 = relu(...)  [ u, u2, B', C' ]      u = relu(a-b), u2 = relu(b-a),
#                                          B' = b + S,    C' = c + S
#   h2 = relu(...)  [ v, d, H' ]           v = relu(B' + u - C')   (= relu(hi - c))
#                                          d = relu(u + u2)        (= |a - b| >= 0)
#                                          H' = relu(B' + u)       (= hi + S)
#   h3 = relu(...)  [ w, m' ]              w  = relu(v - d)
#                                          m' = relu(H' - v)       (= mid + S)
#   y  = m' + w - S                        (= mid + relu(lo - mid) = median)

W1_med = np.array(
    [
        [1.0, -1.0, 0.0],  # u  = relu(a - b)
        [-1.0, 1.0, 0.0],  # u2 = relu(b - a)
        [0.0, 1.0, 0.0],  # B' = relu(b + S)
        [0.0, 0.0, 1.0],  # C' = relu(c + S)
    ]
)
b1_med = np.array([0.0, 0.0, float(S), float(S)])

#                  u     u2    B'    C'
W2_med = np.array(
    [
        [1.0, 0.0, 1.0, -1.0],  # v  = relu(B' + u - C')
        [1.0, 1.0, 0.0, 0.0],  # d  = relu(u + u2)
        [1.0, 0.0, 1.0, 0.0],  # H' = relu(B' + u)
    ]
)
b2_med = np.zeros(3)

#                  v     d     H'
W3_med = np.array(
    [
        [1.0, -1.0, 0.0],  # w  = relu(v - d)
        [-1.0, 0.0, 1.0],  # m' = relu(H' - v)
    ]
)
b3_med = np.zeros(2)

#                  w    m'
W4_med = np.array([[1.0, 1.0]])
b4_med = np.array([-float(S)])


def median3_network(a, b, c):
    x = np.array([a, b, c], dtype=float)
    h1 = np.maximum(0, W1_med @ x + b1_med)
    h2 = np.maximum(0, W2_med @ h1 + b2_med)
    h3 = np.maximum(0, W3_med @ h2 + b3_med)
    y = W4_med @ h3 + b4_med
    return int(y[0])


# ── Check: all three agree on every input ───────────────────────────────

if __name__ == "__main__":
    n1 = 0
    for a in range(LOW, HIGH + 1):
        for b in range(LOW, HIGH + 1):
            expected = control_flow(a, b)
            assert branchless(a, b) == expected, (a, b, branchless(a, b), expected)
            assert network(a, b) == expected, (a, b, network(a, b), expected)
            n1 += 1

    n2 = 0
    for a in range(-S, S + 1):
        for b in range(-S, S + 1):
            for c in range(-S, S + 1):
                expected = median3_control_flow(a, b, c)
                assert median3_branchless(a, b, c) == expected, (a, b, c)
                assert median3_network(a, b, c) == expected, (a, b, c, median3_network(a, b, c))
                n2 += 1

    print("  a   b | " + "  ".join(f"{n:>10}" for n in NAMES))
    for a, b in ((3, 5), (5, 3), (-2, 1), (-8, 8), (6, 6)):
        print(f"{a:3} {b:3} | " + "  ".join(f"{v:10}" for v in network(a, b)))
    print()
    print("  a   b   c | median")
    for a, b, c in ((1, 5, 3), (5, 1, 3), (4, 5, -2), (-6, 6, 0), (2, 2, 2)):
        print(f"{a:3} {b:3} {c:3} | {median3_network(a, b, c):6}")
    print(f"OK: {n1} (a, b) pairs and {n2} (a, b, c) triples agree")
