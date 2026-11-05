"""
09 · break / early return   (linear search)

    for i in range(5):
        if xs[i] == target:
            return i          # break out: later elements are never looked at
    return -1

Control flow stops at the first match. Data flow looks at EVERY element and
uses a "found already" flag to make sure only the first match counts:

    match_i = (xs[i] == target)
    first_i = match_i AND NOT found       # a one-hot vector: 1 only at the first match
    found   = found + first_i             # (= found OR match_i)
    answer  = sum(i * first_i) + found - 1     # found = 0  ->  -1

Inputs: xs has 5 entries and target is an integer, all in [0, 3].
Run:    uv run 02_loops/09_break_first_match.py
"""

import itertools

import numpy as np

N = 5  # list length
LOW, HIGH = 0, 3  # every value (and the target) is in [LOW, HIGH]


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes.


def control_flow(xs, target):
    for i in range(N):
        if xs[i] == target:
            return i
    return -1


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


def AND(p, q):
    return relu(p + q - 1)  # p, q in {0, 1}


def branchless(xs, target):
    found = 0  # 0/1: has an earlier element matched?
    answer = 0
    for i in range(N):  # never stops early
        match = eq(xs[i], target)
        first = AND(match, 1 - found)  # match, and nothing matched before
        answer = answer + i * first  # i is a loop counter (a constant), so this is affine
        found = found + first  # = found OR match, because first = match AND NOT found
    return answer + found - 1  # nothing found: answer = 0, found = 0  ->  -1


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
# State s = [m_0 .. m_4,  found,  f_0 .. f_4]      (11 numbers)
#   m_i = match bit of element i,   found = "a match was already seen",
#   f_i = first_i, the one-hot "this is the first match" vector (all 0 at the start).
#
# Layer M (hidden = 4N neurons) turns the inputs [x_0 .. x_4, target] into the state:
#   for each i:  relu(x_i - t), relu(x_i - t - 1), relu(t - x_i), relu(t - x_i - 1)
#   m_i = 1 - step(x_i - t) - step(t - x_i)
#
# Layer i (the unrolled loop, 12 hidden neurons) - all values are >= 0 so pass-through is exact:
#   0:        relu(m_i - found)              = first_i  (m_i AND NOT found, both 0/1)
#   1..5:     relu(m_j)                      pass m
#   6:        relu(found)                    pass found
#   7..11:    relu(f_j)                      pass f
#   output:   found' = found + first_i,   f_i' = f_i + first_i (f_i was 0), the rest copied.
#
# Readout (affine, no relu):  answer = sum_i i * f_i + found - 1
#   i.e. one-hot -> index is  f @ arange(N),  and `found - 1` supplies the -1.

D = 2 * N + 1  # state size
FOUND = N  # index of `found` in the state

WM = np.zeros((4 * N, N + 1))
VM = np.zeros((D, 4 * N))
cM = np.zeros(D)
for i in range(N):
    for k, sign in enumerate([1, 1, -1, -1]):  # biases (below): 0, -1, 0, -1
        WM[4 * i + k, i] = sign  # sign * x_i
        WM[4 * i + k, N] = -sign  # - sign * t
    VM[i, 4 * i + 0] = -1.0  # m_i = 1 - (h0 - h1) - (h2 - h3)
    VM[i, 4 * i + 1] = 1.0
    VM[i, 4 * i + 2] = -1.0
    VM[i, 4 * i + 3] = 1.0
    cM[i] = 1.0
bM = np.tile([0.0, -1.0, 0.0, -1.0], N)


def loop_layer(i):
    H = 2 * N + 2
    W = np.zeros((H, D))
    V = np.zeros((D, H))
    W[0, i], W[0, FOUND] = 1.0, -1.0  # first_i = relu(m_i - found)
    for j in range(N):
        W[1 + j, j] = 1.0  # pass m_j
        V[j, 1 + j] = 1.0
        W[N + 2 + j, N + 1 + j] = 1.0  # pass f_j
        V[N + 1 + j, N + 2 + j] = 1.0
    W[N + 1, FOUND] = 1.0  # pass found
    V[FOUND, N + 1] = 1.0
    V[FOUND, 0] = 1.0  # found' = found + first_i
    V[N + 1 + i, 0] = 1.0  # f_i' = f_i + first_i
    return W, V


LAYERS = [loop_layer(i) for i in range(N)]

READOUT = np.zeros(D)
READOUT[FOUND] = 1.0
READOUT[N + 1 :] = np.arange(N)  # one-hot @ arange(N) = the index
READOUT_BIAS = -1.0


def run_network(xs, target):
    x = np.array([*xs, target], dtype=float)
    s = VM @ np.maximum(0, WM @ x + bM) + cM
    for W, V in LAYERS:
        s = V @ np.maximum(0, W @ s)
    return s


def network(xs, target):
    return READOUT @ run_network(xs, target) + READOUT_BIAS


# ── Check: all three agree on every input ───────────────────────────────

if __name__ == "__main__":
    count = 0
    for xs in itertools.product(range(LOW, HIGH + 1), repeat=N):
        for target in range(LOW, HIGH + 1):
            expected = control_flow(xs, target)
            assert branchless(xs, target) == expected, (xs, target, branchless(xs, target), expected)
            assert network(xs, target) == expected, (xs, target, network(xs, target), expected)
            count += 1

    xs = (2, 0, 3, 0, 3)
    print(f"xs = {list(xs)}")
    print("  target | A  B  C | one-hot first_i")
    for target in range(LOW, HIGH + 1):
        s = run_network(xs, target)
        onehot = [int(v) for v in s[N + 1 :]]
        print(f"{target:7} | {control_flow(xs, target):2} {branchless(xs, target):2} {network(xs, target):2.0f} | {onehot}")
    print(f"OK: all {count} inputs agree (every xs in [{LOW}, {HIGH}]^{N} x every target)")
