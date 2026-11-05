"""
12 · argmax / argmin as a one-hot vector

    i = max(range(n), key=lambda k: xs[k])     # index of the biggest value
    # ties: the LOWEST index wins (same as xs.index(max(xs)))

Data flow returns the answer as a ONE-HOT vector (a single 1 at the winner)
and the value itself as a masked sum. Index i "wins" when it beats everyone:

    beats_ij = (xs[i] >  xs[j])   for j < i     # strict: an earlier tie wins
    beats_ij = (xs[i] >= xs[j])   for j > i     # non-strict: we win later ties
    won_i    = AND_j beats_ij = relu(sum_j beats_ij - (n - 2))     # n-1 bits
    value    = sum_i select(won_i, xs[i])

Exactly one i wins, the lowest index among the maxima. argmin is the same
with the comparison flipped (negate the differences); ties still go to the
lowest index. Cost: n(n-1) comparisons, so O(n^2) neurons.

Inputs: xs is 4 integers, each in [0, 5].
Run:    uv run 03_memory/12_argmax.py
"""

import itertools

import numpy as np

N = 4
VMAX = 5  # values are integers in [0, VMAX]
M = VMAX  # big-M for the masked sum: |xs[i]| <= M


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes. Returns (one-hot list, extreme value).


def control_flow(xs, sign=+1):
    # sign=+1: argmax, sign=-1: argmin.  max()/min() return the FIRST best.
    pick = max if sign > 0 else min
    best = pick(range(len(xs)), key=lambda k: xs[k])
    return [1 if k == best else 0 for k in range(len(xs))], xs[best]


# ── Stage B · branchless arithmetic ─────────────────────────────────────
# Rules: only  +  -  (multiply by a constant)  relu.
#        No `if`, no comparisons, no max/min/argmax on runtime values,
#        no loops whose length depends on the input.


def relu(x):
    return max(0, x)


def step(x):
    return relu(x) - relu(x - 1)  # 1 if x >= 1 else 0 (integers)


def select(m, x):
    return relu(x - M * (1 - m)) - relu(-x - M * (1 - m))


def branchless(xs, sign=+1):
    onehot = []
    for i in range(N):  # fixed counts
        bits = 0
        for j in range(N):
            if j == i:  # constants only: j and i are loop counters
                continue
            d = sign * (xs[i] - xs[j])
            # earlier j (j < i): strict  d >= 1  -> step(d)
            # later  j (j > i): ties ok  d >= 0  -> step(d + 1)
            bits += step(d) if j < i else step(d + 1)
        onehot.append(relu(bits - (N - 2)))  # AND of the N-1 bits
    value = 0
    for i in range(N):
        value += select(onehot[i], xs[i])
    return onehot, value


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
#   input  x = xs                                   (N values in [0, VMAX])
#   layer 1: for each ordered pair (i, j), i != j, with d = sign*(xs[i]-xs[j]) + c
#            (c = 0 if j < i, c = 1 if j > i):   relu(d), relu(d - 1)
#            beats_ij = relu(d) - relu(d - 1) = step(d).
#            plus relu(xs[i]) to carry the data.            (2*N*(N-1) + N rows)
#   layer 2: won_i = relu( sum_j beats_ij - (N - 2) )       (this IS the one-hot)
#            plus a carried copy of xs[i].
#   layer 3: a_i = relu(xs[i] + M*won_i - M)   (xs >= 0, so one relu is enough)
#   output:  onehot = layer-2 won,  value = sum_i a_i


def build(sign):
    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    P = len(pairs)
    W1 = np.zeros((2 * P + N, N))
    b1 = np.zeros(2 * P + N)
    for p, (i, j) in enumerate(pairs):
        c = 0 if j < i else 1
        W1[2 * p, i] += sign
        W1[2 * p, j] -= sign
        b1[2 * p] = c  # relu(d)
        W1[2 * p + 1, i] += sign
        W1[2 * p + 1, j] -= sign
        b1[2 * p + 1] = c - 1  # relu(d - 1)
    for i in range(N):
        W1[2 * P + i, i] = 1  # carry xs[i]

    W2 = np.zeros((2 * N, 2 * P + N))
    b2 = np.zeros(2 * N)
    for p, (i, j) in enumerate(pairs):
        W2[i, 2 * p] = 1  # + relu(d)
        W2[i, 2 * p + 1] = -1  # - relu(d - 1)
    b2[:N] = -(N - 2)
    for i in range(N):
        W2[N + i, 2 * P + i] = 1  # carry xs[i]

    W3 = np.zeros((N, 2 * N))
    b3 = np.full(N, -M, dtype=float)
    for i in range(N):
        W3[i, i] = M  # M * won_i
        W3[i, N + i] = 1  # + xs[i]
    return W1, b1, W2, b2, W3, b3


NETS = {+1: build(+1), -1: build(-1)}


def network(xs, sign=+1):
    W1, b1, W2, b2, W3, b3 = NETS[sign]
    x = np.array(xs, dtype=float)
    h1 = np.maximum(0, W1 @ x + b1)
    h2 = np.maximum(0, W2 @ h1 + b2)
    h3 = np.maximum(0, W3 @ h2 + b3)
    return h2[:N], h3.sum()  # one-hot, value  (sum is a final affine readout)


# ── Check: all three agree on every input ───────────────────────────────

if __name__ == "__main__":
    count = 0
    for xs in itertools.product(range(VMAX + 1), repeat=N):
        for sign in (+1, -1):
            oh, val = control_flow(xs, sign)
            boh, bval = branchless(xs, sign)
            coh, cval = network(xs, sign)
            assert boh == oh and bval == val, (xs, sign)
            assert list(coh) == oh and cval == val, (xs, sign)
            # decoding the one-hot at the very end is affine: onehot @ arange
            assert int(coh @ np.arange(N)) == (xs.index(max(xs)) if sign > 0 else xs.index(min(xs)))
            count += 1

    print("  xs            | argmax (A)  value | argmin (A)  value | C agrees")
    for xs in [(2, 5, 5, 1), (3, 3, 3, 3), (0, 4, 1, 0), (1, 2, 3, 4)]:
        mx = control_flow(xs, +1)
        mn = control_flow(xs, -1)
        cm, cv = network(xs, +1)
        cn, cw = network(xs, -1)
        ok = list(cm) == mx[0] and list(cn) == mn[0]
        print(f"{str(xs):13} | {mx[0]} {mx[1]}      | {mn[0]} {mn[1]}      | {ok}")
    print(f"OK: all {count} (xs, argmax/argmin) cases agree, ties go to the lowest index")
