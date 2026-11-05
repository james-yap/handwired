"""
17 · minimax over a fixed game tree

    value, move = minimax(leaves)

A tiny two-player game with a FIXED shape. I move first and pick one of
3 moves, then you pick one of 2, then I pick one of 2; then the game ends
and the leaf holds the score for me:

    level 0   me   (MAX)   3 moves
    level 1   you  (MIN)   2 moves each      3 * 2     =  6 positions
    level 2   me   (MAX)   2 moves each      3 * 2 * 2 = 12 leaves

Minimax: every node takes the max (my turn) or the min (your turn) of its
children. We also want my best FIRST move (ties: lowest move number).

Data flow: one layer of max/min neurons per tree level, and an argmax
(the one-hot trick from 12_argmax.py) over the root's three children.
Alpha-beta pruning skips leaves, which only helps control flow; data flow
always evaluates all 12 leaves (in parallel), so pruning disappears.

Inputs: the 12 leaf scores, each an integer in [-3, 3].
Run:    uv run 05_game_search/17_minimax_tree.py
"""

import random

import numpy as np

LOW, HIGH = -3, 3
BRANCH = (3, 2, 2)  # moves available at level 0, 1, 2
LEAVES = 3 * 2 * 2  # = 12; leaf i is reached by moves (i // 4, (i // 2) % 2, i % 2)


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes.


def minimax(leaves, depth=0, start=0):
    """Value of the node at `depth` whose leaves begin at leaves[start]."""
    if depth == len(BRANCH):
        return leaves[start], None
    size = 1  # how many leaves hang under ONE child of this node
    for b in BRANCH[depth + 1 :]:
        size *= b
    maximizing = depth % 2 == 0
    best_value, best_move = None, None
    for move in range(BRANCH[depth]):
        value, _ = minimax(leaves, depth + 1, start + move * size)
        if best_move is None or (value > best_value if maximizing else value < best_value):
            best_value, best_move = value, move  # strict '>' keeps the lowest move on ties
    return best_value, best_move


def control_flow(leaves):
    return minimax(leaves)  # (value of the game, my best first move)


def alpha_beta(leaves, counter, depth=0, start=0, alpha=-99, beta=99):
    """Same answer as minimax, but skips children that cannot matter.

    This is the control-flow trick: `break` = "stop looking, the result can
    no longer change". `counter[0]` counts the leaves actually looked at.
    """
    if depth == len(BRANCH):
        counter[0] += 1
        return leaves[start], None
    size = 1
    for b in BRANCH[depth + 1 :]:
        size *= b
    maximizing = depth % 2 == 0
    best_value, best_move = None, None
    for move in range(BRANCH[depth]):
        value, _ = alpha_beta(leaves, counter, depth + 1, start + move * size, alpha, beta)
        if best_move is None or (value > best_value if maximizing else value < best_value):
            best_value, best_move = value, move
        if maximizing:
            alpha = max(alpha, best_value)
        else:
            beta = min(beta, best_value)
        if alpha >= beta:
            break  # the opponent will never let the game get here
    return best_value, best_move


# ── Stage B · branchless arithmetic ─────────────────────────────────────
# Rules: only  +  -  (multiply by a constant)  relu.
#        No `if`, no comparisons, no max/min/argmax on runtime values,
#        no loops whose length depends on the input.


def relu(x):
    return max(0, x)


def step(x):
    return relu(x) - relu(x - 1)  # 1 if x >= 1 else 0 (integers)


def max2(a, b):
    return b + relu(a - b)


def min2(a, b):
    return a - relu(a - b)


def level_up(values, branching, combine):
    """One tree level: every group of `branching` children becomes one parent."""
    parents = []
    for g in range(len(values) // branching):
        acc = values[g * branching]
        for i in range(1, branching):
            acc = combine(acc, values[g * branching + i])
        parents.append(acc)
    return parents


def argmax_first(v):
    """One-hot of the LARGEST entry of v; ties go to the lowest index.

    ge[i][j] = (v[i] >= v[j]) for i < j, as a 0/1 number.
    Entry i wins iff it beats every earlier j (strictly: NOT ge[j][i])
    and is at least as good as every later j (ge[i][j]).
    """
    n = len(v)
    ge = {(i, j): step(v[i] - v[j] + 1) for i in range(n) for j in range(i + 1, n)}
    onehot = []
    for i in range(n):
        total = 0
        for j in range(i):
            total += 1 - ge[(j, i)]  # v[i] > v[j]
        for j in range(i + 1, n):
            total += ge[(i, j)]  # v[i] >= v[j]
        onehot.append(relu(total - (n - 2)))  # AND of n-1 bits: (sum) - (n-1) + 1
    return onehot


def branchless(leaves):
    values = list(leaves)
    # leaves -> level 2 (MAX of 2) -> level 1 (MIN of 2) -> root children
    values = level_up(values, BRANCH[2], max2)  # 6 values
    values = level_up(values, BRANCH[1], min2)  # 3 values: what each first move is worth
    root = values[0]
    for v in values[1:]:
        root = max2(root, v)  # the root is MAX of its 3 children
    onehot = argmax_first(values)
    move = 0 * onehot[0] + 1 * onehot[1] + 2 * onehot[2]
    return root, move


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
#   x  = the 12 leaves
#   layer 1 (relu, 18 neurons): for each leaf pair (a, b)       [relu(a-b), relu(b), relu(-b)]
#            readout: max(a, b) = relu(a-b) + relu(b) - relu(-b)   -> 6 values
#   layer 2 (relu,  9 neurons): for each value pair (a, b)      [relu(a-b), relu(a), relu(-a)]
#            readout: min(a, b) = -relu(a-b) + relu(a) - relu(-a)  -> 3 values v0, v1, v2
#   layer 3 (relu, 11 neurons): ge01, ge02, ge12 (2 relus each), max(v0, v1) (3 relus),
#            plus v2 carried (relu(v2), relu(-v2))
#   layer 4 (relu,  6 neurons): the 3 AND gates (= the one-hot) and max(max(v0,v1), v2)
#   output: value = readout of the max neurons,  move = onehot @ [0, 1, 2]
#
# Every pair of siblings gets the same tiny 3x2 block; np.kron(I, block) puts
# one block per pair on the diagonal.

MAX_IN = np.array([[1.0, -1.0], [0.0, 1.0], [0.0, -1.0]])  # rows: relu(a-b), relu(b), relu(-b)
MAX_OUT = np.array([[1.0, 1.0, -1.0]])  # max(a,b)
MIN_IN = np.array([[1.0, -1.0], [1.0, 0.0], [-1.0, 0.0]])  # rows: relu(a-b), relu(a), relu(-a)
MIN_OUT = np.array([[-1.0, 1.0, -1.0]])  # min(a,b)


def blocks(group_count, block):
    return np.kron(np.eye(group_count), block)  # one copy of `block` per group, side by side


W1, R1 = blocks(6, MAX_IN), blocks(6, MAX_OUT)  # 12 -> 18 neurons -> 6 values
W2, R2 = blocks(3, MIN_IN), blocks(3, MIN_OUT)  # 6 -> 9 neurons -> 3 values

# layer 3 reads v = (v0, v1, v2). Neuron rows:
#   0,1: ge01 = step(v0 - v1 + 1) = relu(v0-v1+1) - relu(v0-v1)
#   2,3: ge02,  4,5: ge12 likewise
#   6,7,8: max(v0, v1) = relu(v0-v1) + relu(v1) - relu(-v1)
#   9,10:  v2 carried: relu(v2), relu(-v2)
W3 = np.array(
    [
        [1, -1, 0], [1, -1, 0],
        [1, 0, -1], [1, 0, -1],
        [0, 1, -1], [0, 1, -1],
        [1, -1, 0], [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ],
    dtype=float,
)  # fmt: skip
b3 = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)

# readouts of layer 3: ge = (ge01, ge02, ge12), m01 = max(v0, v1), v2
GE_OUT = np.zeros((3, 11))
for p in range(3):
    GE_OUT[p, 2 * p], GE_OUT[p, 2 * p + 1] = 1, -1
M01_OUT = np.array([[0, 0, 0, 0, 0, 0, 1, 1, -1, 0, 0]], dtype=float)
V2_OUT = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]], dtype=float)

# layer 4 reads (ge01, ge02, ge12, m01, v2). Neuron rows:
#   0,1,2: AND gates  win_i = relu(sum of 2 bits - 1)
#       win_0 = ge01 + ge02 - 1                (v0 >= v1 and v0 >= v2)
#       win_1 = (1 - ge01) + ge12 - 1          (v1 >  v0 and v1 >= v2)
#       win_2 = (1 - ge02) + (1 - ge12) - 1    (v2 >  v0 and v2 >  v1)
#   3,4,5: max(m01, v2) = relu(m01 - v2) + relu(v2) - relu(-v2)
W4 = np.array(
    [
        [1, 1, 0, 0, 0],
        [-1, 0, 1, 0, 0],
        [0, -1, -1, 0, 0],
        [0, 0, 0, 1, -1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, -1],
    ],
    dtype=float,
)
# biases: win_0 = ge01 + ge02 - 1,  win_1 = -ge01 + ge12 + 0,  win_2 = -ge02 - ge12 + 1
b4 = np.array([-1, 0, 1, 0, 0, 0], dtype=float)
VALUE_OUT = np.array([[0, 0, 0, 1, 1, -1]], dtype=float)
MOVE_OUT = np.array([[0, 1, 2, 0, 0, 0]], dtype=float)  # onehot @ [0, 1, 2]


def network(leaves):
    x = np.array(leaves, dtype=float)
    h1 = np.maximum(0, W1 @ x)
    h2 = np.maximum(0, W2 @ (R1 @ h1))
    v = R2 @ h2
    h3 = np.maximum(0, W3 @ v + b3)
    h4 = np.maximum(0, W4 @ np.concatenate([GE_OUT @ h3, M01_OUT @ h3, V2_OUT @ h3]) + b4)
    return (VALUE_OUT @ h4)[0], (MOVE_OUT @ h4)[0]


# ── Check: all three agree ──────────────────────────────────────────────
# 7^12 = 13.8 billion leaf assignments, too many to enumerate. Instead:
#   * every assignment of 0/1 leaves (4096 cases, lots of ties),
#   * 20000 random assignments in [-3, 3] (fixed seed),
#   * hand-picked edge cases (all equal, all extreme, ...).
# Also checks alpha-beta gives the same answer (it only skips work).

if __name__ == "__main__":
    rng = random.Random(17)
    cases = [[(bits >> i) & 1 for i in range(LEAVES)] for bits in range(2**LEAVES)]
    cases += [[rng.randint(LOW, HIGH) for _ in range(LEAVES)] for _ in range(20000)]
    cases += [[v] * LEAVES for v in range(LOW, HIGH + 1)]
    cases += [[LOW, HIGH] * (LEAVES // 2), [HIGH, LOW] * (LEAVES // 2), list(range(-3, 3)) * 2]

    visited = 0
    for leaves in cases:
        expected = control_flow(leaves)
        assert branchless(leaves) == expected, (leaves, branchless(leaves), expected)
        assert network(leaves) == expected, (leaves, network(leaves), expected)
        counter = [0]
        assert alpha_beta(leaves, counter) == expected, leaves
        visited += counter[0]

    example = [3, 1, 2, 0, 0, -1, 2, 2, 1, 3, 0, 0]
    print("leaves:", example)
    print("  A (value, move):", control_flow(example))
    print("  B (value, move):", branchless(example))
    print("  C (value, move):", tuple(int(t) for t in network(example)))
    counter = [0]
    alpha_beta(example, counter)
    print(f"  alpha-beta looked at {counter[0]} of {LEAVES} leaves; data flow looks at all {LEAVES}")
    print(f"OK: all {len(cases)} cases agree (A == B == C == alpha-beta); "
          f"alpha-beta averaged {visited / len(cases):.1f} of {LEAVES} leaves")
