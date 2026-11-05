"""
07 · for i in range(n)   (n is an input)

    total = 0
    for i in range(n):
        total += xs[i]

Control flow runs the body exactly n times. Data flow cannot: a network has a
fixed number of layers. So we always run N copies of the body (the loop is
"unrolled") and give each copy a 0/1 mask that says "is this iteration real?":

    active_i = step(n - i)                 # 1 when i < n, else 0
    total   += select(active_i, xs[i])     # add xs[i], or add nothing

Inputs: n is an integer in [0, 5];  xs has 5 entries, each an integer in [-2, 2].
Run:    uv run 02_loops/07_bounded_for.py
"""

import itertools

import numpy as np

N = 5  # the loop can run at most N times (fixed before we see any input)
XB = 2  # every xs[i] is in [-XB, XB]


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes.


def control_flow(n, xs):
    total = 0
    for i in range(n):
        total += xs[i]
    return total


# ── Stage B · branchless arithmetic ─────────────────────────────────────
# Rules: only  +  -  (multiply by a constant)  relu.
#        No `if`, no comparisons, loops only with a count fixed up front.


def relu(x):
    return max(0, x)


def step(x):
    # For an integer x:  step(x) = 1 if x >= 1, else 0.
    return relu(x) - relu(x - 1)


def select(m, x, M):
    # `x if m else 0` for a 0/1 mask m and a runtime value |x| <= M.
    #   m = 1:  relu(x) - relu(-x)        = x
    #   m = 0:  relu(x - M) - relu(-x - M) = 0 - 0   (because |x| <= M)
    return relu(x - M * (1 - m)) - relu(-x - M * (1 - m))


def branchless(n, xs):
    total = 0
    for i in range(N):  # always N iterations, whatever n is
        active = step(n - i)  # i < n  <=>  n - i >= 1   (integers)
        # M = XB: the only runtime value we mask is xs[i], and |xs[i]| <= XB.
        total = total + select(active, xs[i], XB)
    return total


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
# One "layer" below = affine -> relu -> affine:   s_next = V @ relu(W @ s + b) + c
# so the state s between layers is a plain (signed) vector.
#
# Layer 0 turns the inputs  s = [n, x_0 .. x_4]  into the state
#       s = [m_0 .. m_4,  x_0 .. x_4,  total]       (m_i = "iteration i is real")
# using m_i = step(n - i) = relu(n - i) - relu(n - i - 1), and copying the x's.
#
# Layers 1..N are the unrolled loop. Layer i+1 adds xs[i] to total if m_i = 1.
# Its hidden neurons are (all pre-activations are affine in the state s):
#       0:  relu( x_i - XB + XB*m_i)      positive half of select(m_i, x_i)
#       1:  relu(-x_i - XB + XB*m_i)      negative half
#       then pass-throughs, because a relu layer destroys anything it doesn't copy:
#       m_j (0/1, so relu(m_j) = m_j),   x_j  (relu(x_j + XB) - XB = x_j since x_j >= -XB),
#       total (relu(total + TB) - TB = total since |total| <= TB = N * XB)
# and the output adds  h[0] - h[1]  to the total.

TB = N * XB  # |total| can never exceed N * XB

# Layer 0: hidden = [relu(n - i), relu(n - i - 1) for each i] + [relu(x_j + XB) for each j]
W0 = np.zeros((3 * N, 1 + N))
b0 = np.zeros(3 * N)
V0 = np.zeros((2 * N + 1, 3 * N))
c0 = np.zeros(2 * N + 1)
for i in range(N):
    W0[2 * i, 0] = 1.0
    b0[2 * i] = -i
    W0[2 * i + 1, 0] = 1.0
    b0[2 * i + 1] = -i - 1
    V0[i, 2 * i] = 1.0  # m_i = relu(n - i) - relu(n - i - 1)
    V0[i, 2 * i + 1] = -1.0
    W0[2 * N + i, 1 + i] = 1.0  # x_i + XB, nonnegative
    b0[2 * N + i] = XB
    V0[N + i, 2 * N + i] = 1.0  # x_i = relu(x_i + XB) - XB
    c0[N + i] = -XB
# total starts at 0: its row of V0 and c0 is all zeros.


def loop_layer(i):
    """The weights of iteration i. Same shape and same pattern for every i;
    only the column that picks out m_i and x_i moves."""
    H = 3 + 2 * N
    W = np.zeros((H, 2 * N + 1))
    b = np.zeros(H)
    V = np.zeros((2 * N + 1, H))
    c = np.zeros(2 * N + 1)
    # neuron 0 / 1: the two halves of select(m_i, x_i)
    W[0, N + i], W[0, i], b[0] = 1.0, XB, -XB
    W[1, N + i], W[1, i], b[1] = -1.0, XB, -XB
    for j in range(N):
        W[2 + j, j] = 1.0  # pass m_j
        V[j, 2 + j] = 1.0
        W[2 + N + j, N + j] = 1.0  # pass x_j (shifted by XB to keep it >= 0)
        b[2 + N + j] = XB
        V[N + j, 2 + N + j] = 1.0
        c[N + j] = -XB
    W[2 + 2 * N, 2 * N] = 1.0  # pass total (shifted by TB)
    b[2 + 2 * N] = TB
    V[2 * N, 2 + 2 * N] = 1.0
    c[2 * N] = -TB
    V[2 * N, 0], V[2 * N, 1] = 1.0, -1.0  # total += select(m_i, x_i)
    return W, b, V, c


LAYERS = [loop_layer(i) for i in range(N)]


def network(n, xs):
    s = np.array([n, *xs], dtype=float)
    s = V0 @ np.maximum(0, W0 @ s + b0) + c0
    for W, b, V, c in LAYERS:
        s = V @ np.maximum(0, W @ s + b) + c
    return s[2 * N]  # the last entry of the state is total


# ── Check: all three agree on every input ───────────────────────────────

if __name__ == "__main__":
    count = 0
    for n in range(N + 1):
        for xs in itertools.product(range(-XB, XB + 1), repeat=N):
            expected = control_flow(n, xs)
            assert branchless(n, xs) == expected, (n, xs, branchless(n, xs), expected)
            assert network(n, xs) == expected, (n, xs, network(n, xs), expected)
            count += 1

    xs = (2, -1, 1, 2, -2)
    print(f"xs = {list(xs)}")
    print("  n | A  B  C")
    for n in range(N + 1):
        print(f"{n:3} | {control_flow(n, xs):2} {branchless(n, xs):2} {network(n, xs):2.0f}")
    print(f"OK: all {count} inputs agree (every n in [0, {N}] x every xs in [-{XB}, {XB}]^{N})")
