"""
01 · relu and step

    step(x) = 1 if x >= 1 else 0

The only non-linear building block we get is relu(x) = max(0, x): flat on the
left of 0, a straight line on the right. A "step" (a 0/1 switch) looks nothing
like that, but two relus subtract to one, as long as x is an INTEGER:

    step(x) = relu(x) - relu(x - 1)

Both relus are 0 for x <= 0 (0 - 0 = 0). Both are on their sloped part for
x >= 1 (x - (x - 1) = 1). The only place where one is sloped and the other is
flat is 0 < x < 1, and no integer lives there. For a non-integer x the formula
returns the in-between value x. The demo at the bottom prints that.

Inputs: x is an integer in [-10, 10].
Run:    uv run 01_foundations/01_relu_and_step.py
"""

import numpy as np

LOW, HIGH = -10, 10


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes.


def relu_control_flow(x):
    return x if x > 0 else 0


def control_flow(x):
    return 1 if x >= 1 else 0


# ── Stage B · branchless arithmetic ─────────────────────────────────────
# Rules: only  +  -  (multiply by a constant)  relu.
#        No `if`, no comparisons, no loops whose length depends on the input.


def relu(x):
    return max(0, x)  # the one non-linear primitive


def branchless(x):
    # For an integer x:  step(x) = 1 if x >= 1, else 0.
    #   x <= 0:  relu(x) = 0,  relu(x - 1) = 0          -> 0 - 0 = 0
    #   x >= 1:  relu(x) = x,  relu(x - 1) = x - 1      -> x - (x - 1) = 1
    return relu(x) - relu(x - 1)


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
#   x = [x]                          1 input
#   h = relu(W1 @ x + b1)            2 hidden neurons: h = [relu(x), relu(x - 1)]
#   y = W2 @ h + b2                  1 output:         y = h[0] - h[1]

W1 = np.array([[1.0], [1.0]])
b1 = np.array([0.0, -1.0])
W2 = np.array([[1.0, -1.0]])
b2 = np.array([0.0])


def network(x):
    v = np.array([x], dtype=float)
    h = np.maximum(0, W1 @ v + b1)
    y = W2 @ h + b2
    return y[0]


# relu itself is the smallest possible network: 1 input, 1 hidden, 1 output.
W1_relu = np.array([[1.0]])
b1_relu = np.array([0.0])
W2_relu = np.array([[1.0]])
b2_relu = np.array([0.0])


def network_relu(x):
    v = np.array([x], dtype=float)
    h = np.maximum(0, W1_relu @ v + b1_relu)
    return (W2_relu @ h + b2_relu)[0]


# ── Check: all three agree on every input ───────────────────────────────

if __name__ == "__main__":
    for x in range(LOW, HIGH + 1):
        assert relu(x) == relu_control_flow(x) == network_relu(x), x

        expected = control_flow(x)
        assert branchless(x) == expected, (x, branchless(x), expected)
        assert network(x) == expected, (x, network(x), expected)

    print("  x | relu  step: A  B  C")
    for x in (-2, -1, 0, 1, 2, 3):
        print(f"{x:3} | {relu(x):4}        {control_flow(x)}  {branchless(x)}  {network(x):.0f}")

    # Not asserted: what happens between the integers.
    print()
    print("Not an integer? step(x) is only a ramp between 0 and 1:")
    for x in (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        print(f"  step({x:5}) = {branchless(x)}")

    # If the inputs live on a finer grid you can rescale: here x is a multiple
    # of 1/4, so 4 * x is an integer and the ramp becomes exact again.
    print("Inputs on a 1/4 grid: relu(4x) - relu(4x - 1) is exact (1 when x >= 1/4):")
    for x in (-0.25, 0.0, 0.25, 0.5):
        print(f"  step_quarter({x:5}) = {relu(4 * x) - relu(4 * x - 1)}")

    print(f"OK: all {HIGH - LOW + 1} inputs agree")
