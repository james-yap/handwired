"""
13 · state machine (a DFA as a repeated layer)

    Does the bit string contain "101" ?

A finite automaton keeps one STATE and updates it once per input symbol.
In data flow the state is a ONE-HOT vector (a single 1: which state we're
in) and each symbol has a constant 0/1 transition matrix T_sym, so that
T_sym @ state is the one-hot of the next state:

    next = sum_sym  AND(is_sym, T_sym @ state)
         = sum_sym  relu(T_sym @ state + is_sym - 1)        (both are 0/1 bits)

Only the term of the symbol actually read can be 1; the others are killed
by `is_sym = 0`. The very same layer (same weights) is applied once per
character, like unrolling a recurrent network.

States: 0 = nothing yet, 1 = seen "1", 2 = seen "10", 3 = seen "101" (stays).
Symbols: 0 = '0', 1 = '1', 2 = padding (strings shorter than MAXLEN are
padded on the right; padding is the identity transition).

Inputs: every string over {0, 1} of length 0..6 (127 strings), padded to 6.
Run:    uv run 03_memory/13_state_machine.py
"""

import itertools

import numpy as np

S = 4  # states
K = 3  # symbols: '0', '1', pad
L = 6  # fixed max string length = number of layers
ACCEPT = [0, 0, 0, 1]  # which states accept

# DELTA[symbol][state] = next state.  Fixed structure, known in advance.
DELTA = {
    0: [0, 2, 0, 3],  # '0'
    1: [1, 1, 3, 3],  # '1'
    2: [0, 1, 2, 3],  # pad: stay where we are
}

# T[k] is a constant S x S 0/1 matrix with T[k][next, cur] = 1 iff DELTA[k][cur] == next
T = np.zeros((K, S, S))
for _k in range(K):
    for _s in range(S):
        T[_k, DELTA[_k][_s], _s] = 1


def encode(s):
    # '0' -> 0, '1' -> 1, then pad with symbol 2 up to length L
    return [int(ch) for ch in s] + [2] * (L - len(s))


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes.


def control_flow(s):
    state = 0
    for ch in s:
        if ch == "0":
            state = {0: 0, 1: 2, 2: 0, 3: 3}[state]
        else:
            state = {0: 1, 1: 1, 2: 3, 3: 3}[state]
    return 1 if state == 3 else 0


# ── Stage B · branchless arithmetic ─────────────────────────────────────
# Rules: only  +  -  (multiply by a constant)  relu.
#        No `if` on a runtime value, no comparisons, no table lookup by a
#        runtime value, loops only with a fixed count.


def relu(x):
    return max(0, x)


def step(x):
    return relu(x) - relu(x - 1)  # 1 if x >= 1 else 0 (integers)


def equals(a, b):
    return 1 - step(a - b) - step(b - a)


def branchless(codes):
    state = [1, 0, 0, 0]  # one-hot: start in state 0
    for t in range(L):  # one layer per character
        is_sym = [equals(codes[t], k) for k in range(K)]
        new = [0] * S
        for k in range(K):
            for s_next in range(S):
                # (T[k] @ state)[s_next], with T[k] constant 0/1 entries
                moved = 0
                for s_cur in range(S):
                    moved += int(T[k][s_next][s_cur]) * state[s_cur]
                new[s_next] += relu(moved + is_sym[k] - 1)  # AND of two bits
        state = new
    accept = 0
    for s in range(S):
        accept += ACCEPT[s] * state[s]
    return accept


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
# Encoder (two small layers, runs once):
#   input  c = [c_0 .. c_{L-1}]   symbol codes
#   E1: for each (t, k): relu(c_t-k), relu(c_t-k-1), relu(k-c_t), relu(k-c_t-1)
#   h = WE2 @ E1 + bE2:   window bit (t,k) = is_sym = 1 - r0 + r1 - r2 + r3,
#                         state = [1, 0, 0, 0]
#   so  h = [ state (S) | window of L*K bits, position-major ]
#
# Step layer (the SAME A, b, B for every character):
#   hidden = relu(A @ h + b)
#       rows (s, k):  (T[k] @ state)[s] + window[0][k] - 1   <- AND of two bits
#       plus window carried unchanged (bits are >= 0, so relu is exact)
#   h = B @ hidden
#       state'  = sum over k of the (s, k) rows
#       window' = window shifted left by one symbol (zero filled; never read)
#
# Readout: accept = ACCEPT @ state


def build_encoder():
    WE1 = np.zeros((4 * L * K, L))
    bE1 = np.zeros(4 * L * K)
    for t in range(L):
        for k in range(K):
            r = 4 * (t * K + k)
            WE1[r + 0, t], bE1[r + 0] = 1, -k
            WE1[r + 1, t], bE1[r + 1] = 1, -k - 1
            WE1[r + 2, t], bE1[r + 2] = -1, k
            WE1[r + 3, t], bE1[r + 3] = -1, k - 1
    WE2 = np.zeros((S + L * K, 4 * L * K))
    bE2 = np.zeros(S + L * K)
    bE2[0] = 1  # initial one-hot state: state 0
    for m in range(L * K):
        WE2[S + m, 4 * m : 4 * m + 4] = [-1, 1, -1, 1]
        bE2[S + m] = 1
    return WE1, bE1, WE2, bE2


def build_step():
    H = S * K + L * K
    A = np.zeros((H, S + L * K))
    b = np.zeros(H)
    for k in range(K):
        for s in range(S):
            row = s * K + k
            A[row, :S] = T[k][s]  # (T[k] @ state)[s]
            A[row, S + k] = 1  # + window[0][k]  (the is_sym bit for symbol k)
            b[row] = -1
    for m in range(L * K):
        A[S * K + m, S + m] = 1  # carry the window
    B = np.zeros((S + L * K, H))
    for s in range(S):
        for k in range(K):
            B[s, s * K + k] = 1  # sum over symbols
    for m in range(L * K - K):
        B[S + m, S * K + m + K] = 1  # shift the window left by K
    return A, b, B


WE1, bE1, WE2, bE2 = build_encoder()
A, b, B = build_step()
ACCEPT_ROW = np.array(ACCEPT + [0] * (L * K), dtype=float)


def network(codes):
    c = np.array(codes, dtype=float)
    h = WE2 @ np.maximum(0, WE1 @ c + bE1) + bE2
    for _ in range(L):  # same weights every time
        h = B @ np.maximum(0, A @ h + b)
    return ACCEPT_ROW @ h


# ── Check: all three agree on every input ───────────────────────────────

if __name__ == "__main__":
    strings = [""]
    for n in range(1, L + 1):
        strings += ["".join(bits) for bits in itertools.product("01", repeat=n)]
    for s in strings:
        expected = control_flow(s)
        assert branchless(encode(s)) == expected, s
        assert network(encode(s)) == expected, s
        assert expected == (1 if "101" in s else 0), s  # sanity: A is right

    print("string   | A B C")
    for s in ["", "101", "1001", "0101", "110100", "111111"]:
        print(f"{s or '(empty)':8} | {control_flow(s)} {branchless(encode(s))} {network(encode(s)):.0f}")
    print(f"OK: all {len(strings)} strings of length 0..{L} agree")
