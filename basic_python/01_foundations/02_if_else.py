"""
02 · if / else

    result = 10 if a > 4 else 12

Control flow runs ONE branch. Data flow computes BOTH branches and uses a
0/1 number (a "mask") to keep the one we want:

    above4 = step(a - 4)                       # 1 when a > 4, else 0
    result = above4 * 10 + (1 - above4) * 12   # = 12 - 2 * above4

Inputs: a is an integer in [-20, 20].
Run:    uv run 01_foundations/02_if_else.py
"""

import numpy as np

LOW, HIGH = -20, 20


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes.


def control_flow(a):
    if a > 4:
        return 10
    else:
        return 12


# ── Stage B · branchless arithmetic ─────────────────────────────────────
# Rules: only  +  -  (multiply by a constant)  relu.
#        No `if`, no comparisons, no loops whose length depends on the input.


def relu(x):
    return max(0, x)  # the one non-linear primitive


def step(x):
    # For an integer x:  step(x) = 1 if x >= 1, else 0.
    #   x <= 0:  relu(x) = 0,  relu(x - 1) = 0          -> 0
    #   x >= 1:  relu(x) = x,  relu(x - 1) = x - 1      -> 1
    return relu(x) - relu(x - 1)


def branchless(a):
    above4 = step(a - 4)  # a > 4  <=>  a - 4 >= 1  (integers)
    return above4 * 10 + (1 - above4) * 12  # both branches are constants, so this is affine


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
#   x  = [a]
#   h  = relu(W1 @ x + b1)      h = [relu(a - 4), relu(a - 5)]
#   y  = W2 @ h + b2            y = 12 - 2 * (h[0] - h[1])

W1 = np.array([[1.0], [1.0]])
b1 = np.array([-4.0, -5.0])
W2 = np.array([[-2.0, 2.0]])
b2 = np.array([12.0])


def network(a):
    x = np.array([a], dtype=float)
    h = np.maximum(0, W1 @ x + b1)
    y = W2 @ h + b2
    return y[0]


# ── Check: all three agree on every input ───────────────────────────────

if __name__ == "__main__":
    for a in range(LOW, HIGH + 1):
        expected = control_flow(a)
        assert branchless(a) == expected, (a, branchless(a), expected)
        assert network(a) == expected, (a, network(a), expected)

    print("  a | A  B  C")
    for a in (2, 3, 4, 5, 6, 7):
        print(f"{a:3} | {control_flow(a)} {branchless(a)} {network(a):.0f}")
    print(f"OK: all {HIGH - LOW + 1} inputs agree")
