"""
18 · minimax for a real (tiny) game: subtraction Nim

    value, move = best_move(pile)

A pile of n objects (0 <= n <= 12). Players alternate; on your turn you take
1, 2 or 3 objects (never more than the pile holds). Whoever takes the LAST
object wins. So a player who faces an empty pile has already lost.

Scores are from MY point of view (I am the player to move at the root):
+1 = I win, -1 = I lose, 0 = not decided yet. Two data-flow versions:

  1. TREE   (depth-limited minimax, D = 4 plies).  The tree always has the
            same 3^4 = 81 leaves. Every node's pile is  n - (takes so far),
            which is affine in n. "Is this move legal?" and "is the game
            over?" are 0/1 masks made with step(). Illegal moves are kept out
            of max/min with the big-M select. Cost: 1426 neurons.
  2. TABLE  (full solve). win[p] depends only on p, not on the input, so it
            is computed once, bottom-up (win[p] = OR over k of NOT win[p-k]).
            The network is then a lookup: one-hot(n) @ table. 15 neurons.

Which Stage A function checks which version:
  version 1 (tree)  <->  control_flow(n)       depth-limited recursion, D = 4
  version 2 (table) <->  control_flow_full(n)  memoized, searches to the end

Inputs: the pile n is an integer in [0, 12].
Run:    uv run 05_game_search/18_nim_minimax.py
"""

from functools import cache

import numpy as np

N = 12  # pile sizes 0 .. N
D = 4  # lookahead in plies for the tree version
TAKES = (1, 2, 3)
M = 4  # big-M for the select; derived below


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes. Every function returns (value, move) where
# value is from MY point of view and move = how many to take (0 = no move,
# the game is already over). Ties go to the smallest take.


def minimax(pile, depth, me_to_move):
    """Depth-limited minimax: look `depth` more plies ahead."""
    if pile == 0:
        return (-1 if me_to_move else +1), 0  # the previous player took the last one
    if depth == 0:
        return 0, 0  # out of lookahead: unresolved
    best_value, best_move = None, None
    for k in TAKES:
        if k > pile:
            break  # illegal
        value, _ = minimax(pile - k, depth - 1, not me_to_move)
        if best_move is None or (value > best_value if me_to_move else value < best_value):
            best_value, best_move = value, k  # strict comparison keeps the smaller k on ties
    return best_value, best_move


def control_flow(n):
    return minimax(n, D, True)


@cache  # memoization: each (pile, who) is solved once, then looked up
def solve(pile, me_to_move):
    """Full-depth minimax. The game always ends, so no depth limit is needed."""
    if pile == 0:
        return (-1 if me_to_move else +1), 0
    best_value, best_move = None, None
    for k in TAKES:
        if k > pile:
            break
        value, _ = solve(pile - k, not me_to_move)
        if best_move is None or (value > best_value if me_to_move else value < best_value):
            best_value, best_move = value, k
    return best_value, best_move


def control_flow_full(n):
    return solve(n, True)


# ── Stage B · branchless arithmetic ─────────────────────────────────────
# Rules: only  +  -  (multiply by a constant)  relu.
#        No `if`, no comparisons, no max/min/argmax on runtime values,
#        no loops whose length depends on the input.
# (Loops over the FIXED tree and `j % 2` on loop counters are constants.)


def relu(x):
    return max(0, x)


def step(x):
    return relu(x) - relu(x - 1)  # 1 if x >= 1 else 0 (integers)


def max2(a, b):
    return b + relu(a - b)


def min2(a, b):
    return a - relu(a - b)


def select(m, x):
    # x if m else 0, valid when m is 0/1 and |x| <= M
    return relu(x - M * (1 - m)) - relu(-x - M * (1 - m))


def argmax_first(v):
    """One-hot of the largest entry of v; ties go to the lowest index (see 17)."""
    n = len(v)
    ge = {(i, j): step(v[i] - v[j] + 1) for i in range(n) for j in range(i + 1, n)}
    onehot = []
    for i in range(n):
        total = 0
        for j in range(i):
            total += 1 - ge[(j, i)]
        for j in range(i + 1, n):
            total += ge[(i, j)]
        onehot.append(relu(total - (n - 2)))
    return onehot


# ---- version 1: the tree --------------------------------------------------
#
# Node (level j, index i) means: j plies played; the path to it is the digits
# of i in base 3, digit d meaning "took d + 1". Its children are 3i, 3i+1, 3i+2
# on level j + 1, so the shape never depends on n. Who moves at level j is
# fixed too: me when j is even (MAX level), you when j is odd (MIN level).
#
#   pile      = n - (sum of takes on the path)             affine in n
#   legal     = step(pile + 1)    "pile >= 0": the move into this node was
#                                  legal (and no earlier node ended the game,
#                                  because piles only shrink along a path)
#   zero      = 1 - step(pile) - step(-pile)      "pile == 0": game over here
#
# Values, with s = +1 on MAX levels and -1 on MIN levels:
#   leaf (level D):  value = -s * zero            (nobody has won yet -> 0)
#   inner node:      value = combine(entries), where combine is max (MAX level)
#                    or min (MIN level), over 4 entries:
#       entry 0      "no move" = fill + s * zero   (pile 0: -1 on a MAX level
#                                                   = I already lost; +1 on a
#                                                   MIN level = you already lost)
#       entry 1..3   take k:   fill + select(legal_child, value_child - fill)
#   with fill = -2 s: the value that loses every comparison. An illegal child
#   is replaced by `fill`, so it can never be chosen.
#
# Bounds for the select (why M = 4): every node value, even for nodes that do
# not exist (pile < 0), lies in [-2, 2]. Leaves are in {-1, 0, 1}; an entry is
# either `fill` (+-2) or a real value in [-1, 1]; max/min of entries stays in
# [-2, 2]. So |value_child - fill| <= 2 + 2 = 4 = M, and select is exact.


def branchless(n):
    piles = [[n]]
    for j in range(D):
        piles.append([p - k for p in piles[j] for k in TAKES])  # level j+1
    legal = [[step(p + 1) for p in level] for level in piles]
    zero = [[1 - step(p) - step(-p) for p in level] for level in piles]

    s = 1 - 2 * (D % 2)
    values = [-s * z for z in zero[D]]  # leaves
    for j in range(D - 1, -1, -1):
        s = 1 - 2 * (j % 2)  # +1: my turn (max), -1: your turn (min)
        fill = -2 * s
        combine = max2 if s > 0 else min2
        parents, root_entries = [], None
        for i in range(3**j):
            entries = [fill + s * zero[j][i]]
            for c in range(3 * i, 3 * i + 3):
                entries.append(fill + select(legal[j + 1][c], values[c] - fill))
            acc = entries[0]
            for e in entries[1:]:
                acc = combine(acc, e)
            parents.append(acc)
            root_entries = entries
        values = parents
    onehot = argmax_first(root_entries)  # root_entries = the root's 4 entries
    move = 0 * onehot[0] + 1 * onehot[1] + 2 * onehot[2] + 3 * onehot[3]
    return values[0], move


# ---- version 2: the table -------------------------------------------------
# win[p] = 1 if the player to move with a pile of p wins (plays perfectly).
#   win[p] = OR over legal k of NOT win[p - k]        (win[0] = 0: no move left)
# Nothing here depends on the input n, so these are CONSTANTS, computed once.


def build_table():
    win, move = [], []
    for p in range(N + 1):
        found = 0  # 1 once a winning take has been seen
        first = 0  # the smallest winning take
        total = 0
        for k in TAKES:
            if k <= p:  # p and k are constants here, not inputs
                beats = 1 - win[p - k]  # NOT win[p - k]
                total += beats
                take = relu(beats - found)  # beats AND NOT found
                first += k * take
                found += take
        win.append(step(total))  # OR
        move.append(first + step(p) * (1 - found))  # no winning take: smallest legal (1), or 0 at p = 0
    value = [2 * w - 1 for w in win]  # +1 win, -1 loss
    return win, value, move


WIN, VALUE, MOVE = build_table()


def branchless_table(n):
    # one-hot(n) from three relus per position:  onehot_i = r[i-1] - 2 r[i] + r[i+1],  r[t] = relu(n - t)
    r = {t: relu(n - t) for t in range(-1, N + 2)}
    onehot = [r[i - 1] - 2 * r[i] + r[i + 1] for i in range(N + 1)]
    value = move = 0
    for i in range(N + 1):
        value += VALUE[i] * onehot[i]
        move += MOVE[i] * onehot[i]
    return value, move


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
# ---- version 1: the tree, level by level ---------------------------------
# Every layer below is  relu(constant matrix @ previous values + constant).
# n >= 0, so it can be carried unchanged from layer to layer by ONE neuron
# relu(n) = n; masks are recomputed from n just before they are needed.
#
#   layer 0        flags of the leaves (5 neurons per node) + carried n
#   per level j = D-1 .. 1, three layers:
#     SELECT       2 neurons per child (the select), flags of level j, carried n
#     MAX/MIN #1   the 4 entries of each node in pairs (3 neurons per pair)
#     MAX/MIN #2   the two pair results (3 neurons) + flags of level j again
#                  (their `legal` readout feeds the next level's select)
#   root (level 0): SELECT; then layer R1 = the 6 comparisons (12 neurons) and the
#                   first round of the max (6); layer R2 = 4 AND gates (the
#                   one-hot of the best take) and the last max round (3).

MAX_IN = np.array([[1.0, -1.0], [0.0, 1.0], [0.0, -1.0]])  # relu(a-b), relu(b), relu(-b)
MAX_OUT = np.array([[1.0, 1.0, -1.0]])  # max(a,b)
MIN_IN = np.array([[1.0, -1.0], [1.0, 0.0], [-1.0, 0.0]])  # relu(a-b), relu(a), relu(-a)
MIN_OUT = np.array([[-1.0, 1.0, -1.0]])  # min(a,b)

# five neurons per node, from its pile p:  relu(p+1), relu(p), relu(p-1), relu(-p), relu(-p-1)
FLAG_IN = np.array([[1.0], [1.0], [1.0], [-1.0], [-1.0]])
FLAG_B = np.array([1.0, 0.0, -1.0, 0.0, -1.0])
LEGAL_OUT = np.array([[1.0, -1.0, 0.0, 0.0, 0.0]])  # step(p+1) = relu(p+1) - relu(p)
ZERO_OUT = np.array([[0.0, -1.0, 1.0, -1.0, 1.0]])  # 1 - step(p) - step(-p), the "1" is added by hand


def blocks(count, block):
    return np.kron(np.eye(count), block)  # `count` copies of `block` on the diagonal


def level_offsets(j):
    """OFFSET[j][i] = total taken on the path to node (j, i)."""
    off = np.array([0.0])
    for _ in range(j):
        off = (off[:, None] + np.array([1.0, 2.0, 3.0])[None, :]).reshape(-1)
    return off


OFFSET = [level_offsets(j) for j in range(D + 1)]
ONES = [np.ones((3**j, 1)) for j in range(D + 1)]
FLAG_W = [blocks(3**j, FLAG_IN) for j in range(D + 1)]
FLAG_BIAS = [np.tile(FLAG_B, 3**j) for j in range(D + 1)]
LEGAL_R = [blocks(3**j, LEGAL_OUT) for j in range(D + 1)]
ZERO_R = [blocks(3**j, ZERO_OUT) for j in range(D + 1)]

# For an inner level j with g = 3^j nodes: 4 entries per node, in the order
# (no move, take 1, take 2, take 3), then two rounds of pairwise max/min.
E_NOMOVE = [blocks(3**j, np.array([[1.0], [0.0], [0.0], [0.0]])) for j in range(D)]  # 4g x g
E_CHILD = [blocks(3**j, np.array([[0.0] * 3, [1, 0, 0], [0, 1, 0], [0, 0, 1]])) for j in range(D)]  # 4g x 3g
PAIR_IN = [MAX_IN if j % 2 == 0 else MIN_IN for j in range(D)]  # which combine this level uses
PAIR_OUT = [MAX_OUT if j % 2 == 0 else MIN_OUT for j in range(D)]
ROUND1_IN = [blocks(2 * 3**j, PAIR_IN[j]) for j in range(D)]  # 4g -> 6g neurons
ROUND1_OUT = [blocks(2 * 3**j, PAIR_OUT[j]) for j in range(D)]  # 6g neurons -> 2g values
ROUND2_IN = [blocks(3**j, PAIR_IN[j]) for j in range(D)]  # 2g -> 3g neurons
ROUND2_OUT = [blocks(3**j, PAIR_OUT[j]) for j in range(D)]  # 3g neurons -> g values

# root: all 6 comparisons ge_ij = step(e_i - e_j + 1)  (i < j), 2 relus each
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
CMP_IN = np.zeros((12, 4))
CMP_B = np.zeros(12)
CMP_OUT = np.zeros((6, 12))
for p, (i, j) in enumerate(PAIRS):
    CMP_IN[2 * p, i], CMP_IN[2 * p, j], CMP_B[2 * p] = 1, -1, 1  # relu(e_i - e_j + 1)
    CMP_IN[2 * p + 1, i], CMP_IN[2 * p + 1, j] = 1, -1  # relu(e_i - e_j)
    CMP_OUT[p, 2 * p], CMP_OUT[p, 2 * p + 1] = 1, -1
# AND gates: win_i = relu( sum of 3 bits - 2 ), bits = ge_ij (j later) or 1 - ge_ji (j earlier)
AND_IN = np.zeros((4, 6))
AND_B = np.zeros(4)
for p, (i, j) in enumerate(PAIRS):
    AND_IN[i, p] += 1  # i beats j on a tie
    AND_IN[j, p] -= 1  # j must be strictly better than i ...
    AND_B[j] += 1  # ... which is 1 - ge_ij
AND_B -= 2
ROOT_PAIR_IN = blocks(2, MAX_IN)  # entries (e0,e1), (e2,e3) -> 6 neurons
ROOT_PAIR_OUT = blocks(2, MAX_OUT)
FINAL_IN, FINAL_OUT = MAX_IN, MAX_OUT
MOVE_OUT = np.array([[0.0, 1.0, 2.0, 3.0]])  # one-hot -> take; affine decode at the very end


def relu_(z):
    return np.maximum(0, z)


def flags(x, j):
    """Layer of 5 relu neurons per node of level j, computed from n alone."""
    pile = ONES[j] @ x - OFFSET[j]
    return relu_(FLAG_W[j] @ pile + FLAG_BIAS[j])


def entries_layer(j, x, v_c, legal_c):
    """The SELECT layer of level j. Returns (carried n, the 4 entries of every node)."""
    s = 1 - 2 * (j % 2)
    fill = -2 * s
    z = v_c - fill
    both = lambda a: np.concatenate([a, a])  # noqa: E731  (concat is allowed)
    h_sel = relu_(np.concatenate([z, -z]) - M + M * both(legal_c))  # relu(z - M(1-m)), relu(-z - M(1-m))
    h_zero = flags(x, j)
    masked = fill + h_sel[: len(z)] - h_sel[len(z) :]  # fill + select(legal, v - fill)
    zero = ZERO_R[j] @ h_zero + 1
    entries = E_NOMOVE[j] @ (fill + s * zero) + E_CHILD[j] @ masked
    return relu_(x), entries


def leaf_layer(x):
    h = flags(x, D)
    s = 1 - 2 * (D % 2)
    v = -s * (ZERO_R[D] @ h + 1)
    return relu_(x), v, LEGAL_R[D] @ h


def level_layers(j, x, v_c, legal_c):
    x, e = entries_layer(j, x, v_c, legal_c)  # SELECT
    h1 = relu_(ROUND1_IN[j] @ e)  # MAX/MIN #1
    m = ROUND1_OUT[j] @ h1
    x = relu_(x)
    h2 = relu_(ROUND2_IN[j] @ m)  # MAX/MIN #2, plus flags for the next level's select
    h_flags = flags(x, j)
    return relu_(x), ROUND2_OUT[j] @ h2, LEGAL_R[j] @ h_flags


def root_layers(x, v_c, legal_c):
    _, e = entries_layer(0, x, v_c, legal_c)  # SELECT: 4 entries (n is not needed after this)
    h_cmp = relu_(CMP_IN @ e + CMP_B)  # R1: comparisons ...
    h_pair = relu_(ROOT_PAIR_IN @ e)  # ... and first max round
    ge = CMP_OUT @ h_cmp
    m = ROOT_PAIR_OUT @ h_pair
    h_and = relu_(AND_IN @ ge + AND_B)  # R2: one-hot of the best take ...
    h_final = relu_(FINAL_IN @ m)  # ... and last max round
    return (FINAL_OUT @ h_final)[0], (MOVE_OUT @ h_and)[0]


def network(n):
    x = np.array([n], dtype=float)
    x, v, legal = leaf_layer(x)
    for j in range(D - 1, 0, -1):
        x, v, legal = level_layers(j, x, v, legal)
    return root_layers(x, v, legal)


# ---- version 2: the table -------------------------------------------------
#   layer 1:  r_t = relu(n - t) for t = -1 .. N+1                     (N + 3 neurons)
#   readout:  onehot_i = r_{i-1} - 2 r_i + r_{i+1}                    (a "hat": 1 only at n = i)
#   output:   [value, move] = [VALUE; MOVE] @ onehot                  (the table, as a matrix)

T_IN = np.ones((N + 3, 1))
T_B = -np.arange(-1, N + 2, dtype=float)
HAT = np.zeros((N + 1, N + 3))
for i in range(N + 1):
    HAT[i, i], HAT[i, i + 1], HAT[i, i + 2] = 1, -2, 1
TABLE = np.array([VALUE, MOVE], dtype=float)  # 2 x (N+1)


def network_table(n):
    x = np.array([n], dtype=float)
    r = relu_(T_IN @ x + T_B)
    onehot = HAT @ r
    y = TABLE @ onehot
    return y[0], y[1]


# ── Neuron counts (formulas; the layer comments above say where they come from) ──


def tree_neurons(depth):
    total = 5 * 3**depth + 1  # leaf flags + carried n
    for j in range(depth - 1, 0, -1):
        g = 3**j
        total += (2 * 3 * g + 5 * g + 1) + (6 * g + 1) + (3 * g + 5 * g + 1)
    return total + (2 * 3 + 5) + (12 + 6) + (4 + 3)  # root: select, R1, R2


def table_neurons(pile_max):
    return pile_max + 3


# ── Check: all versions agree on every input ────────────────────────────

if __name__ == "__main__":
    assert WIN == [0 if p % 4 == 0 else 1 for p in range(N + 1)]  # the known answer: lose iff p % 4 == 0
    unresolved = []
    for n in range(N + 1):
        expected = control_flow(n)
        assert branchless(n) == expected, (n, branchless(n), expected)
        assert network(n) == expected, (n, network(n), expected)

        full = control_flow_full(n)
        assert branchless_table(n) == full, (n, branchless_table(n), full)
        assert network_table(n) == full, (n, network_table(n), full)

        # how does the depth-limited answer relate to the full solve?
        if expected[0] != 0:
            assert expected[0] == full[0], n  # a resolved lookahead value is always right
        else:
            unresolved.append(n)  # the lookahead could not see the end

    print("  n | tree (D=4)   | full solve  | table (B, C)")
    for n in range(N + 1):
        a, f = control_flow(n), control_flow_full(n)
        t = network_table(n)
        print(f"{n:3} | {a[0]:+d}, take {a[1]} | {f[0]:+d}, take {f[1]} | {t[0]:+.0f}, take {t[1]:.0f}")
    print(f"depth-{D} lookahead is unresolved (value 0) for n = {unresolved}; all other n match the full solve")
    print(f"neurons: tree D={D}: {tree_neurons(D)}; table: {table_neurons(N)}")
    print(f"OK: all {N + 1} piles agree (tree: A == B == C; table: full solve == B == C)")
