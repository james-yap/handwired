"""
08 · while loop with a data-dependent length   (GCD by repeated subtraction)

    while a != b:
        if a > b: a -= b
        else:     b -= a

Control flow stops when the condition fails. Data flow cannot stop, so we:
  1. work out a number T that is >= the iterations ANY input needs,
  2. run exactly T identical steps,
  3. compute halted = (a == b) at the start of each step; once halted is 1 the
     step is made a no-op:   state = select(halted, old_state, new_state).

Every step uses the very same weights: the loop body is ONE layer applied T
times (like an RNN). We also count the real iterations as the sum of (1 - halted).

Inputs: a and b are integers in [1, 12].
Run:    uv run 02_loops/08_while_halt.py
"""

import math

import numpy as np

LOW, HIGH = 1, 12

# Why T = HIGH - 1 = 11 steps always suffice:
#   Each step subtracts the smaller number from the larger one, so max(a, b)
#   drops by at least 1 (the new larger value is <= old larger - 1).
#   The loop ends when a == b == gcd >= 1, so max(a, b) never goes below 1.
#   Start <= 12, end >= 1  =>  at most 11 steps.  (12, 1) really needs 11.
T = HIGH - 1
M = HIGH  # big-M for every select below: all values it masks lie in [-12, 12]


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes.


def control_flow(a, b):
    iters = 0
    while a != b:
        if a > b:
            a -= b
        else:
            b -= a
        iters += 1
    return a, iters


# ── Stage B · branchless arithmetic ─────────────────────────────────────
# Rules: only  +  -  (multiply by a constant)  relu.
#        No `if`, no comparisons, loops only with a count fixed up front.


def relu(x):
    return max(0, x)


def step(x):
    # For an integer x:  step(x) = 1 if x >= 1, else 0.
    return relu(x) - relu(x - 1)


def eq(x, y):
    return 1 - step(x - y) - step(y - x)


def select(m, x):
    # `x if m else 0` for a 0/1 mask m and a runtime value |x| <= M.
    return relu(x - M * (1 - m)) - relu(-x - M * (1 - m))


def branchless(a, b):
    iters = 0
    for _ in range(T):  # always T steps
        halted = eq(a, b)  # the loop condition, as a 0/1 number
        gt = step(a - b)  # the `if` inside the body

        # The body, run unconditionally. Both branches are computed
        # (na, nb = the values the body proposes):
        na = a - select(gt, b)  # `a -= b`  only when a > b
        nb = b - select(1 - gt, a)  # `b -= a`  otherwise (even when a == b, giving 0!)

        # Freeze: once halted, keep the OLD state.  x if m else y = y + select(m, x - y)
        # |old - new| <= 12 = M, so the select is exact.
        undo_a = select(halted, a - na)
        undo_b = select(halted, b - nb)
        a, b = na + undo_a, nb + undo_b
        iters = iters + 1 - halted  # counts the steps in which the body really ran
    return a, iters


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
# State s = [a, b, count].   One loop step = THREE hidden relu layers plus a
# final affine map, with the same constants (W1..W4, b1..b4) every time.
# Every state entry is >= 0, so passing it through a relu is exact (one neuron).
#
# h1 (7 neurons)  from s = [a, b, count]
#   0,1:  relu(a - b), relu(a - b - 1)      -> gt = h[0] - h[1]    = step(a - b)
#   2,3:  relu(b - a), relu(b - a - 1)      -> lt = h[2] - h[3]    = step(b - a)
#   4,5,6: relu(a), relu(b), relu(count)    (pass-through)
# h2 (8 neurons)  halted = 1 - gt - lt;  the two select()s of the body
#   0,1:  relu( b - M + M*gt), relu(-b - M + M*gt)     = select(gt, b)
#   2,3:  relu( a - M*gt),     relu(-a - M*gt)         = select(1 - gt, a)
#   4,5,6: a, b, count (pass-through)
#   7:    relu(1 - gt - lt)  = halted
#   so  new_a = a - select(gt, b),   new_b = b - select(1 - gt, a)
# h3 (8 neurons)  the freeze:  d_a = a - new_a = select(gt, b),  d_b = b - new_b
#   0,1:  relu( d_a - M + M*halted), relu(-d_a - M + M*halted)   = select(halted, d_a)
#   2,3:  the same for d_b
#   4,5:  relu(new_a), relu(new_b)   (pass-through; both are >= 0)
#   6,7:  count, halted
# output  s' = [ new_a + select(halted, d_a),  new_b + select(halted, d_b),  count + 1 - halted ]

W1 = np.array(
    [
        [1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
b1 = np.array([0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0])

W2 = np.array(
    [
        [M, -M, 0, 0, 0, 1, 0],
        [M, -M, 0, 0, 0, -1, 0],
        [-M, M, 0, 0, 1, 0, 0],
        [-M, M, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [-1, 1, -1, 1, 0, 0, 0],
    ],
    dtype=float,
)
b2 = np.array([-M, -M, 0, 0, 0, 0, 0, 1], dtype=float)

W3 = np.array(
    [
        [1, -1, 0, 0, 0, 0, 0, M],
        [-1, 1, 0, 0, 0, 0, 0, M],
        [0, 0, 1, -1, 0, 0, 0, M],
        [0, 0, -1, 1, 0, 0, 0, M],
        [-1, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, -1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=float,
)
b3 = np.array([-M, -M, -M, -M, 0, 0, 0, 0], dtype=float)

W4 = np.array(
    [
        [1, -1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, -1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1],
    ],
    dtype=float,
)
b4 = np.array([0.0, 0.0, 1.0])


def loop_body(s):
    h1 = np.maximum(0, W1 @ s + b1)
    h2 = np.maximum(0, W2 @ h1 + b2)
    h3 = np.maximum(0, W3 @ h2 + b3)
    return W4 @ h3 + b4


def network(a, b):
    s = np.array([a, b, 0], dtype=float)
    for _ in range(T):  # the SAME weights, T times: weight tying
        s = loop_body(s)
    return s[0], s[2]


# ── Check: all three agree on every input ───────────────────────────────

if __name__ == "__main__":
    worst = 0
    count = 0
    for a in range(LOW, HIGH + 1):
        for b in range(LOW, HIGH + 1):
            expected = control_flow(a, b)
            assert expected[0] == math.gcd(a, b), (a, b)
            assert expected[1] <= T, (a, b, expected[1])  # T really is safe
            worst = max(worst, expected[1])
            assert branchless(a, b) == expected, (a, b, branchless(a, b), expected)
            got = network(a, b)
            assert got == expected, (a, b, got, expected)
            count += 1
    assert worst == T  # and it is tight: (12, 1) needs all T steps

    print("  a   b | A: gcd iters | B: gcd iters | C: gcd iters")
    for a, b in [(12, 8), (7, 7), (9, 6), (12, 1), (5, 11)]:
        ra, rb, rc = control_flow(a, b), branchless(a, b), network(a, b)
        print(f"{a:3} {b:3} | {ra[0]:7} {ra[1]:5} | {rb[0]:7} {rb[1]:5} | {rc[0]:7.0f} {rc[1]:5.0f}")
    print(f"OK: all {count} inputs agree; most iterations needed = {worst} = T")
