"""
10 · array read

    y = xs[i]          # xs: 4 numbers, i: which one -- both known only at runtime

Memory addressing in data flow = a ONE-HOT vector. Turn the address i into
a vector with a single 1 at position i (equality bits), then keep each cell
only where its bit is 1 and add everything up:

    onehot_j = (i == j)                          # 1 for exactly one j
    y        = sum_j select(onehot_j, xs[j])     # all other terms are 0

Because xs is runtime data, "bit * xs[j]" is a product of two runtime values,
which we cannot do. So each term is a big-M select (see README toolbox).
If xs were a CONSTANT table, no select is needed: onehot @ table is a plain
matrix product. The weights themselves are the memory.

Inputs: xs is 4 integers, each in [0, 7]; i is an integer in [0, 3].
Run:    uv run 03_memory/10_array_read.py
"""

import itertools

import numpy as np

N = 4  # number of cells
VMAX = 7  # cell values are integers in [0, VMAX]
M = VMAX  # big-M: |xs[j]| <= M, so a masked-off term x - M <= 0 is killed by relu


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes.


def control_flow(xs, i):
    return xs[i]


# ── Stage B · branchless arithmetic ─────────────────────────────────────
# Rules: only  +  -  (multiply by a constant)  relu.
#        No `if`, no comparisons, no indexing by a runtime value,
#        no loops whose length depends on the input.


def relu(x):
    return max(0, x)


def step(x):
    return relu(x) - relu(x - 1)  # 1 if x >= 1 else 0 (integers)


def select(m, x):
    # x if m else 0, for m in {0, 1} and |x| <= M
    return relu(x - M * (1 - m)) - relu(-x - M * (1 - m))


def onehot(i):
    # j is a constant (the loop counter), i is runtime.
    # (i == j) = 1 - step(i - j) - step(j - i)
    return [1 - step(i - j) - step(j - i) for j in range(N)]


def branchless(xs, i):
    oh = onehot(i)
    total = 0
    for j in range(N):  # fixed count
        total += select(oh[j], xs[j])
    return total


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
#   input  x  = [i, xs[0], ..., xs[3]]
#   layer 1:  for each cell j, four relus that encode (i == j):
#                 relu(i-j), relu(i-j-1), relu(j-i), relu(j-i-1)
#             plus relu(xs[j]) to carry the data along (xs >= 0, so exact).
#             onehot_j = 1 - (r0 - r1) - (r2 - r3)
#   layer 2:  a_j = relu(xs[j] + M*onehot_j - M)
#                 onehot_j = 1:  relu(xs[j])      = xs[j]
#                 onehot_j = 0:  relu(xs[j] - M)  = 0     (xs[j] <= M)
#             (xs >= 0, so the second half of select, relu(-x - M(1-m)), is
#              always 0 and is dropped.)  The onehot is substituted as an
#              affine function of layer 1's outputs; the "+M*1 - M" cancels.
#   layer 3:  y = sum_j a_j


def build_read_network():
    W1 = np.zeros((5 * N, 1 + N))
    b1 = np.zeros(5 * N)
    for j in range(N):
        r = 4 * j
        W1[r + 0, 0], b1[r + 0] = 1, -j  # relu(i - j)
        W1[r + 1, 0], b1[r + 1] = 1, -j - 1  # relu(i - j - 1)
        W1[r + 2, 0], b1[r + 2] = -1, j  # relu(j - i)
        W1[r + 3, 0], b1[r + 3] = -1, j - 1  # relu(j - i - 1)
        W1[4 * N + j, 1 + j] = 1  # relu(xs[j])
    W2 = np.zeros((N, 5 * N))
    for j in range(N):
        r = 4 * j
        # M*onehot_j - M = -M*r0 + M*r1 - M*r2 + M*r3
        W2[j, r : r + 4] = [-M, M, -M, M]
        W2[j, 4 * N + j] = 1  # + xs[j]
    W3 = np.ones((1, N))
    return W1, b1, W2, W3


W1, b1, W2, W3 = build_read_network()


def network(xs, i):
    x = np.array([i, *xs], dtype=float)
    h1 = np.maximum(0, W1 @ x + b1)
    h2 = np.maximum(0, W2 @ h1)
    return (W3 @ h2)[0]


# ── Contrast: a CONSTANT table needs no select ──────────────────────────
# If the memory is fixed, the weights are the memory: y = TABLE @ onehot(i).
# Everything below is affine in the layer-1 outputs, so no big-M at all.

TABLE = np.array([3.0, 1.0, 4.0, 7.0])

OH = np.zeros((N, 5 * N))  # onehot = 1 - r0 + r1 - r2 + r3  (per cell)
for _j in range(N):
    OH[_j, 4 * _j : 4 * _j + 4] = [-1, 1, -1, 1]


def table_read_network(i):
    x = np.array([i, 0, 0, 0, 0], dtype=float)  # the xs slots are unused
    h1 = np.maximum(0, W1 @ x + b1)
    oh = OH @ h1 + 1
    return TABLE @ oh  # constant row of memory contents


# ── Check: all three agree on every input ───────────────────────────────

if __name__ == "__main__":
    count = 0
    for xs in itertools.product(range(VMAX + 1), repeat=N):
        for i in range(N):
            expected = control_flow(xs, i)
            assert branchless(xs, i) == expected, (xs, i)
            assert network(xs, i) == expected, (xs, i)
            count += 1
    for i in range(N):
        assert table_read_network(i) == TABLE[i], i

    print("  xs           i | A  B  C")
    for xs, i in [((5, 0, 7, 2), 0), ((5, 0, 7, 2), 2), ((5, 0, 7, 2), 3), ((1, 1, 1, 6), 3)]:
        print(f"{str(xs):12} {i} | {control_flow(xs, i)}  {branchless(xs, i)}  {network(xs, i):.0f}")
    print(f"OK: all {count} (xs, i) pairs agree; constant-table read matches for all {N} indices")
