"""
04 · logic on bits

    not p,  p and q,  p or q,  p xor q,  p implies q

Bits are the numbers 0 and 1. For two bits the only thing that matters is the
SUM s = p + q  (0, 1 or 2):

    p and q  = relu(s - 1)             only s = 2 survives
    p or  q  = step(s)                 s >= 1
    not p    = 1 - p
    p -> q   = 1 - relu(p - q)         false only for p = 1, q = 0

XOR is the interesting one: no single "weighted sum then threshold" can do
it. The classic fix is two layers:  p xor q = (p or q) and not (p and q).
Layer 1 computes OR and AND, layer 2 combines them.

For n bits, with s = sum of the bits:   AND = relu(s - (n - 1)),  OR = step(s).
The second half of this file does that for n = 4.

Inputs: p and q are bits (0 or 1); the n-bit part uses all 16 four-bit inputs.
Run:    uv run 01_foundations/04_logic.py
"""

import numpy as np

N = 4  # number of bits in the n-ary part
NAMES = ("not p", "p&q", "p|q", "p^q", "p->q")


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes.


def control_flow(p, q):
    return (
        int(not p),
        int(p and q),
        int(p or q),
        int(p != q),
        int((not p) or q),
    )


def nary_control_flow(bits):
    return (int(all(bits)), int(any(bits)))


# ── Stage B · branchless arithmetic ─────────────────────────────────────
# Rules: only  +  -  (multiply by a constant)  relu.
#        No `if`, no comparisons, no loops whose length depends on the input.


def relu(x):
    return max(0, x)  # the one non-linear primitive


def step(x):
    # For an integer x:  step(x) = 1 if x >= 1, else 0.
    return relu(x) - relu(x - 1)


def branchless(p, q):
    s = p + q
    not_p = 1 - p
    or_ = step(s)  # layer 1
    and_ = relu(s - 1)  # layer 1
    nand = 1 - and_
    xor = relu(or_ + nand - 1)  # layer 2: AND(or_, nand), see "p and q" above
    implies = 1 - relu(p - q)
    return (not_p, and_, or_, xor, implies)


def nary_branchless(bits):
    s = 0
    for i in range(N):  # fixed count: N is a constant, not the input
        s = s + bits[i]
    return (relu(s - (N - 1)), step(s))


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
#   x  = [p, q]
#   h1 = relu(W1 @ x + b1)      4 neurons:  [relu(s), relu(s - 1), relu(p - q), relu(p)]
#        With p, q >= 0 these are:  s (a copy of the sum), AND, "p and not q", and a copy of p.
#        OR = relu(s) - relu(s - 1) is not a neuron, it is the difference of two.
#   h2 = relu(W2 @ h1 + b2)     5 neurons, layer 2.
#        Row 0 is  AND(OR, NAND) = relu(OR + NAND - 1) = relu(h1[0] - 2 * h1[1])  -> XOR
#        Rows 1-4 only COPY layer-1 values forward: relu(v) = v when v >= 0.
#   y  = W3 @ h2 + b3           reorder + the two "1 - ..." outputs

W1 = np.array(
    [
        [1.0, 1.0],  # relu(p + q)
        [1.0, 1.0],  # relu(p + q - 1)
        [1.0, -1.0],  # relu(p - q)
        [1.0, 0.0],  # relu(p)
    ]
)
b1 = np.array([0.0, -1.0, 0.0, 0.0])

#                h0    h1    h2    h3
W2 = np.array(
    [
        [1.0, -2.0, 0.0, 0.0],  # xor      = relu(OR + NAND - 1) = relu(h0 - 2 * h1)
        [1.0, -1.0, 0.0, 0.0],  # or       = h0 - h1 >= 0, copied
        [0.0, 1.0, 0.0, 0.0],  # and      = h1 >= 0, copied
        [0.0, 0.0, 1.0, 0.0],  # p and not q, copied
        [0.0, 0.0, 0.0, 1.0],  # p, copied
    ]
)
b2 = np.zeros(5)

#                g0=xor g1=or g2=and g3=p&~q g4=p
W3 = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, -1.0],  # not p     = 1 - p
        [0.0, 0.0, 1.0, 0.0, 0.0],  # p and q
        [0.0, 1.0, 0.0, 0.0, 0.0],  # p or q
        [1.0, 0.0, 0.0, 0.0, 0.0],  # p xor q
        [0.0, 0.0, 0.0, -1.0, 0.0],  # p implies q = 1 - (p and not q)
    ]
)
b3 = np.array([1.0, 0.0, 0.0, 0.0, 1.0])


def network(p, q):
    x = np.array([p, q], dtype=float)
    h1 = np.maximum(0, W1 @ x + b1)
    h2 = np.maximum(0, W2 @ h1 + b2)
    y = W3 @ h2 + b3
    return tuple(int(v) for v in y)


# XOR does not need the second layer here: OR and AND are each built from the same
# two relus of s (OR = relu(s) - relu(s - 1), AND = relu(s - 1)), so
#   xor = OR - AND = relu(s) - 2 * relu(s - 1)   (s = 0 -> 0, s = 1 -> 1, s = 2 -> 2 - 2 = 0)
# Same layer-1 neurons, a different output matrix: ONE hidden layer is enough.
W_shallow = np.array(
    [
        [0.0, 0.0, 0.0, -1.0],  # not p     = 1 - relu(p)
        [0.0, 1.0, 0.0, 0.0],  # p and q   = relu(s - 1)
        [1.0, -1.0, 0.0, 0.0],  # p or q    = relu(s) - relu(s - 1)
        [1.0, -2.0, 0.0, 0.0],  # p xor q   = relu(s) - 2 relu(s - 1)
        [0.0, 0.0, -1.0, 0.0],  # p -> q    = 1 - relu(p - q)
    ]
)
b_shallow = np.array([1.0, 0.0, 0.0, 0.0, 1.0])


def network_shallow(p, q):
    x = np.array([p, q], dtype=float)
    h = np.maximum(0, W1 @ x + b1)
    y = W_shallow @ h + b_shallow
    return tuple(int(v) for v in y)


# n-ary AND / OR: one sum, then
#   h = [relu(s), relu(s - 1), relu(s - (N - 1))],  AND = h[2],  OR = h[0] - h[1].
Wn1 = np.ones((3, N))
bn1 = np.array([0.0, -1.0, -(N - 1.0)])
Wn2 = np.array(
    [
        [0.0, 0.0, 1.0],  # AND of all N bits
        [1.0, -1.0, 0.0],  # OR of all N bits
    ]
)
bn2 = np.zeros(2)


def nary_network(bits):
    x = np.array(bits, dtype=float)
    h = np.maximum(0, Wn1 @ x + bn1)
    y = Wn2 @ h + bn2
    return tuple(int(v) for v in y)


# ── Check: all three agree on every input ───────────────────────────────

if __name__ == "__main__":
    for p in (0, 1):
        for q in (0, 1):
            expected = control_flow(p, q)
            assert branchless(p, q) == expected, (p, q, branchless(p, q), expected)
            assert network(p, q) == expected, (p, q, network(p, q), expected)
            assert network_shallow(p, q) == expected, (p, q, network_shallow(p, q), expected)

    for k in range(2**N):
        bits = [(k >> i) & 1 for i in range(N)]
        expected = nary_control_flow(bits)
        assert nary_branchless(bits) == expected, (bits, nary_branchless(bits), expected)
        assert nary_network(bits) == expected, (bits, nary_network(bits), expected)

    print(" p q | " + "  ".join(f"{n:>5}" for n in NAMES))
    for p in (0, 1):
        for q in (0, 1):
            print(f" {p} {q} | " + "  ".join(f"{v:5}" for v in network(p, q)))
    print(f"n = {N} bits:  AND is 1 only for 1111, OR is 0 only for 0000")
    print(f"OK: 4 two-bit inputs (2 networks) and {2**N} {N}-bit inputs agree")
