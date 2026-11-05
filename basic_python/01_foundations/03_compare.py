"""
03 · comparisons as numbers

    a > b,  a >= b,  a < b,  a <= b,  a == b,  a != b

In control flow a comparison produces True/False and an `if` reacts to it. In
data flow a comparison is a plain 0/1 NUMBER that later arithmetic can use.
Everything reduces to step() of the difference d = a - b (integers!):

    a >  b  =  step(d)            d >= 1
    a >= b  =  step(d + 1)        d + 1 >= 1
    a <  b  =  step(-d)           -d >= 1
    a <= b  =  step(-d + 1)       -d + 1 >= 1
    a == b  =  1 - (a > b) - (a < b)
    a != b  =  (a > b) + (a < b)

Writing step(z) = relu(z) - relu(z - 1) out, only SIX different relus appear,
and several are shared between comparisons. One hidden layer computes all six
results at once.

Inputs: a and b are integers in [-8, 8].
Run:    uv run 01_foundations/03_compare.py
"""

import numpy as np

LOW, HIGH = -8, 8
NAMES = ("a>b", "a>=b", "a<b", "a<=b", "a==b", "a!=b")


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes.


def control_flow(a, b):
    # Python's True/False are 0/1 as ints.
    return (int(a > b), int(a >= b), int(a < b), int(a <= b), int(a == b), int(a != b))


# ── Stage B · branchless arithmetic ─────────────────────────────────────
# Rules: only  +  -  (multiply by a constant)  relu.
#        No `if`, no comparisons, no loops whose length depends on the input.


def relu(x):
    return max(0, x)  # the one non-linear primitive


def step(x):
    # For an integer x:  step(x) = 1 if x >= 1, else 0.
    return relu(x) - relu(x - 1)


def branchless(a, b):
    d = a - b
    gt = step(d)  # d >= 1
    ge = step(d + 1)  # d + 1 >= 1, i.e. d >= 0
    lt = step(-d)  # -d >= 1
    le = step(-d + 1)  # -d + 1 >= 1, i.e. d <= 0
    eq = 1 - gt - lt  # neither greater nor less
    ne = gt + lt  # greater or less (they can never both be 1)
    return (gt, ge, lt, le, eq, ne)


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
#   x = [a, b]                       d = a - b
#   h = relu(W1 @ x + b1)            6 hidden neurons, shared by all outputs:
#        h0 = relu( d + 1)    h3 = relu(-d + 1)
#        h1 = relu( d    )    h4 = relu(-d    )
#        h2 = relu( d - 1)    h5 = relu(-d - 1)
#   y = W2 @ h + b2                  each row is "step(z) = relu(z) - relu(z-1)",
#                                    taken from two neighbouring hidden neurons:
#        a>b  = h1 - h2           a<b  = h4 - h5
#        a>=b = h0 - h1           a<=b = h3 - h4
#        a==b = 1 - (a>b) - (a<b) a!=b = (a>b) + (a<b)

W1 = np.array(
    [
        [1.0, -1.0],
        [1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
    ]
)
b1 = np.array([1.0, 0.0, -1.0, 1.0, 0.0, -1.0])

#                 h0    h1    h2    h3    h4    h5
W2 = np.array(
    [
        [0.0, 1.0, -1.0, 0.0, 0.0, 0.0],  # a >  b
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],  # a >= b
        [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],  # a <  b
        [0.0, 0.0, 0.0, 1.0, -1.0, 0.0],  # a <= b
        [0.0, -1.0, 1.0, 0.0, -1.0, 1.0],  # a == b   (1 - gt - lt)
        [0.0, 1.0, -1.0, 0.0, 1.0, -1.0],  # a != b   (gt + lt)
    ]
)
b2 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def network(a, b):
    x = np.array([a, b], dtype=float)
    h = np.maximum(0, W1 @ x + b1)
    y = W2 @ h + b2
    return tuple(int(v) for v in y)


# ── Check: all three agree on every input ───────────────────────────────

if __name__ == "__main__":
    for a in range(LOW, HIGH + 1):
        for b in range(LOW, HIGH + 1):
            expected = control_flow(a, b)
            assert branchless(a, b) == expected, (a, b, branchless(a, b), expected)
            assert network(a, b) == expected, (a, b, network(a, b), expected)

    print("  a   b | " + "  ".join(f"{n:>4}" for n in NAMES))
    for a, b in ((3, 5), (4, 4), (5, 3), (-2, 1), (-8, -8)):
        print(f"{a:3} {b:3} | " + "  ".join(f"{v:4}" for v in network(a, b)))
    print(f"OK: all {(HIGH - LOW + 1) ** 2} (a, b) pairs agree")
