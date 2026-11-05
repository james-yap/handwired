"""
11 · array write

    xs[i] = v          # then use the new list
    return xs

Data flow cannot "modify a cell". It computes a whole NEW list in which
every cell is rewritten: cell j takes the new value v if j is the address,
and otherwise keeps what it had.

    onehot_j = (i == j)
    new[j]   = xs[j] + select(onehot_j, v - xs[j])
             = v if j == i else xs[j]

Cost: every write reads and rewrites ALL cells (read-modify-write): N
selects and N equality tests, however small the change. The difference
v - xs[j] is a runtime-minus-runtime value in [-VMAX, VMAX], so the big-M
select needs both of its relus.

Inputs: xs is 4 integers, each in [0, 5]; v is an integer in [0, 5];
        i is an integer in [0, 3].
Run:    uv run 03_memory/11_array_write.py
"""

import itertools

import numpy as np

N = 4
VMAX = 5  # cells and v are integers in [0, VMAX]
M = VMAX  # big-M: |v - xs[j]| <= VMAX


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes.


def control_flow(xs, v, i):
    xs = list(xs)
    xs[i] = v
    return xs


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
    return [1 - step(i - j) - step(j - i) for j in range(N)]


def branchless(xs, v, i):
    oh = onehot(i)
    new = []
    for j in range(N):  # fixed count
        new.append(xs[j] + select(oh[j], v - xs[j]))
    return new


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
#   input  x  = [i, v, xs[0], ..., xs[3]]
#   layer 1:  4 relus per cell for onehot_j (as in 10_array_read), plus
#             relu(v) and relu(xs[j]) to carry the data (all >= 0, exact).
#   layer 2:  per cell j, with onehot_j = 1 - r0 + r1 - r2 + r3:
#               p_j = relu( (v - xs[j]) + M*onehot_j - M)      # positive part
#               q_j = relu(-(v - xs[j]) + M*onehot_j - M)      # negative part
#               c_j = relu(xs[j])                              # carried
#   layer 3:  new[j] = c_j + p_j - q_j                         # affine, no relu
#
#   Width: layer 1 = 5N + 1 = 21, layer 2 = 3N = 12, output = N = 4.

W1_ROWS = 5 * N + 1  # 4N onehot relus, N copies of xs, 1 copy of v
V_ROW = 5 * N  # row of relu(v) in layer 1; relu(xs[j]) is row 4N + j


def build_write_network():
    W1 = np.zeros((W1_ROWS, 2 + N))
    b1 = np.zeros(W1_ROWS)
    for j in range(N):
        r = 4 * j
        W1[r + 0, 0], b1[r + 0] = 1, -j
        W1[r + 1, 0], b1[r + 1] = 1, -j - 1
        W1[r + 2, 0], b1[r + 2] = -1, j
        W1[r + 3, 0], b1[r + 3] = -1, j - 1
        W1[4 * N + j, 2 + j] = 1
    W1[V_ROW, 1] = 1

    W2 = np.zeros((3 * N, W1_ROWS))
    for j in range(N):
        r = 4 * j
        gate = [-M, M, -M, M]  # = M*onehot_j - M in terms of r0..r3
        W2[3 * j + 0, r : r + 4] = gate
        W2[3 * j + 0, V_ROW] = 1  # + v
        W2[3 * j + 0, 4 * N + j] = -1  # - xs[j]
        W2[3 * j + 1, r : r + 4] = gate
        W2[3 * j + 1, V_ROW] = -1  # - v
        W2[3 * j + 1, 4 * N + j] = 1  # + xs[j]
        W2[3 * j + 2, 4 * N + j] = 1  # carry xs[j]

    W3 = np.zeros((N, 3 * N))
    for j in range(N):
        W3[j, 3 * j : 3 * j + 3] = [1, -1, 1]  # p_j - q_j + c_j
    return W1, b1, W2, W3


W1, b1, W2, W3 = build_write_network()


def network(xs, v, i):
    x = np.array([i, v, *xs], dtype=float)
    h1 = np.maximum(0, W1 @ x + b1)
    h2 = np.maximum(0, W2 @ h1)
    return W3 @ h2


# ── Check: all three agree on every input ───────────────────────────────

if __name__ == "__main__":
    count = 0
    for xs in itertools.product(range(VMAX + 1), repeat=N):
        for v in range(VMAX + 1):
            for i in range(N):
                expected = control_flow(xs, v, i)
                assert branchless(xs, v, i) == expected, (xs, v, i)
                assert list(network(xs, v, i)) == expected, (xs, v, i)
                count += 1

    print("  xs            v i | A             B             C")
    for xs, v, i in [((5, 0, 3, 2), 4, 0), ((5, 0, 3, 2), 4, 2), ((5, 0, 3, 2), 0, 3)]:
        a = control_flow(xs, v, i)
        b = branchless(xs, v, i)
        c = [int(t) for t in network(xs, v, i)]
        print(f"{str(xs):13} {v} {i} | {str(a):13} {str(b):13} {c}")
    print(f"OK: all {count} (xs, v, i) triples agree")
