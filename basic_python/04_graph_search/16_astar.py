"""
16 · A*: pick the best open node, expand it, repeat

    heap = [(h[start], start)]
    while heap:
        f, u = heappop(heap)              # best f = g + h, lowest index on ties
        close u
        if u == goal: stop
        for each free neighbour n of u:
            if g[u] + 1 < g[n]:  g[n] = g[u] + 1;  parent[n] = u;  push n

A* is sequential by nature: each step depends on the previous one. We do not
change that. We turn ONE step into a fixed layer and repeat it N times (a 4x4
grid has 16 nodes, so at most 16 expansions). The step is built from tricks
we already have:

    key[v]   = g[v] + h[v], pushed far away if v is closed or never reached
    sel      = one-hot of the lowest key, lowest index on ties   (argmin, ch. 3)
    sel      = sel AND "something is open" AND "not done yet"    (the halt flag)
    closed  += sel
    cand[n]  = n touches sel AND n is no wall AND n is not closed
    better   = cand AND (g[sel] + 1 < g[n])
    g[n]     = g[n] if not better else g[sel] + 1                (big-M select)
    parent[n]= one-hot(sel) if better else unchanged             (big-M select)
    done     = 1 - sum(sel) + sel[goal]    (goal expanded, or nothing left to expand)

A path is then read back by following the parent one-hots N times.

Grid: 4x4, nodes numbered row by row 0..15. Start = node 0 (top-left), goal =
node 3 (top-right), 4-neighbour unit-cost moves. The 14 other cells are
runtime inputs: 1 = wall. (A wall bit on the start is ignored: the start is
closed first. A wall on the goal makes it unreachable.)

Output (all three stages): (cost or -1, order in which nodes were expanded, path
from the goal back to the start); lists are padded with -1 up to 16 entries.
Run:    uv run 04_graph_search/16_astar.py
"""

import heapq
import random
from collections import deque

import numpy as np

ROWS = COLS = 4
N = ROWS * COLS
START, GOAL = 0, 3


def neighbours(v):
    r, c = divmod(v, COLS)
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        if 0 <= r + dr < ROWS and 0 <= c + dc < COLS:
            out.append((r + dr) * COLS + (c + dc))
    return out


NEIGH = [neighbours(v) for v in range(N)]
# Manhattan distance to the goal: the goal is fixed, so h is a constant vector.
H = [abs(v // COLS - GOAL // COLS) + abs(v % COLS - GOAL % COLS) for v in range(N)]

# Bounds that make the arithmetic exact:
#   g of a real node <= N - 1 = 15 (a path visits each node at most once), h <= 6,
#   so every real f = g + h <= 21.
#   BIG stands for "no / not allowed": it must exceed every real f and every g + 1.
G_MAX, H_MAX = N - 1, (ROWS - 1) + (COLS - 1)
BIG = G_MAX + H_MAX + 1  # 22
#   key = g + h + BIG*closed  is < BIG for open nodes, >= BIG for closed / unreached ones.
#   key' = N*key + index is unique per node: the argmin has no ties left to break.


def pad(xs):
    return xs + [-1] * (N - len(xs))


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes. walls[v] = 1 for a wall.


def control_flow(walls):
    g = {START: 0}
    parent = {}
    closed = set()
    order = []
    heap = [(H[START], START)]  # (f, node): ties on f go to the lower node index
    while heap:
        f, u = heapq.heappop(heap)
        if u in closed:
            continue  # stale entry: u was pushed again with a lower f and already expanded
        closed.add(u)
        order.append(u)
        if u == GOAL:
            break
        for n in NEIGH[u]:
            if walls[n] or n in closed:
                continue
            if g[u] + 1 < g.get(n, BIG):
                g[n] = g[u] + 1
                parent[n] = u
                heapq.heappush(heap, (g[n] + H[n], n))
    if GOAL not in closed:
        return -1, pad(order), pad([])
    path, cur = [GOAL], GOAL
    while cur != START:
        cur = parent[cur]
        path.append(cur)
    return g[GOAL], pad(order), pad(path)


# ── Stage B · branchless arithmetic ─────────────────────────────────────
# Rules: only  +  -  (multiply by a constant)  relu.
#        No `if`, no comparisons, no heap, no loops whose length depends on
#        the input (N expansion steps, N path steps, always).


def relu(x):
    return max(0, x)


def step(x):
    return relu(x) - relu(x - 1)  # 1 if x >= 1 else 0, for integers


def select(m, x, M):
    # x if m else 0, for m in {0, 1} and |x| <= M
    return relu(x - M * (1 - m)) - relu(-x - M * (1 - m))


E_GOAL = [1 if v == GOAL else 0 for v in range(N)]  # constant one-hot of the goal


def decode(rows):
    # one-hot -> node number (affine read-out: sum of (v+1)*onehot[v], minus 1; all-zero -> -1)
    return [sum((v + 1) * row[v] for v in range(N)) - 1 for row in rows]


def branchless(walls):
    g = [0 if v == START else BIG for v in range(N)]  # BIG = not reached yet
    closed = [0] * N
    parent = [[0] * N for _ in range(N)]  # parent[n] = one-hot of n's parent (all 0: none)
    done = 0
    log = []  # the one-hot chosen at each step
    for _ in range(N):
        # 1. keys, then argmin as a one-hot (rank 0 = nobody has a smaller key)
        key = [N * (g[v] + H[v] + BIG * closed[v]) + v for v in range(N)]
        rank = [sum(step(key[v] - key[u]) for u in range(N) if u != v) for v in range(N)]
        reached = [step(BIG - g[v]) for v in range(N)]  # g < BIG
        # 2. selected = rank 0 AND open (reached, not closed) AND not done
        sel = [relu(reached[v] - closed[v] - done - rank[v]) for v in range(N)]
        # 3. g of the selected node: mask each g[v] by sel[v] (g >= 0, so one relu is enough)
        g_sel = sum(relu(g[v] - BIG * (1 - sel[v])) for v in range(N))
        # 4. neighbours of the selected node (graph structure is constant; the goal never expands)
        near = [sum(sel[u] for u in NEIGH[n] if u != GOAL) for n in range(N)]
        cand = [relu(near[n] - walls[n] - closed[n]) for n in range(N)]
        # 5. better route found?  g_sel + 1 < g[n]  <=>  g[n] - g_sel - 1 >= 1
        better = [relu(cand[n] + step(g[n] - g_sel - 1) - 1) for n in range(N)]
        # 6. update the registers (all from the OLD values)
        new_g = [g[n] + select(better[n], g_sel + 1 - g[n], BIG) for n in range(N)]
        new_parent = [
            [parent[n][u] + select(better[n], sel[u] - parent[n][u], 1) for u in range(N)]
            for n in range(N)
        ]
        done = 1 - sum(sel) + sel[GOAL]
        closed = [closed[v] + sel[v] for v in range(N)]
        g, parent = new_g, new_parent
        log.append(sel)
    found = closed[GOAL]
    # Path: start at the goal (if found) and follow parent pointers N times.
    cur = [found * E_GOAL[v] for v in range(N)]
    path_log = []
    for _ in range(N):
        path_log.append(cur)
        # parent of cur: for every pair (v, u), AND(cur[v], parent[v][u]), summed over v
        cur = [sum(relu(parent[v][u] + cur[v] - 1) for v in range(N)) for u in range(N)]
    cost = g[GOAL] - (BIG + 1) * (1 - found)  # BIG -> -1
    return cost, decode(log), decode(path_log)


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
# Every layer is  x' = W2 @ relu(W1 @ x + b1) + b2  on one big state vector.
# All state entries are >= 0, so relu(x_i) = x_i copies an entry through a layer.
# One A* step is 6 layers (the numbers are the steps of stage B):
#   L1  pairwise key comparisons (2*N*(N-1) units), reached flags     -> RANK, REACHED
#   L2  sel = relu(REACHED - CLOSED - DONE - RANK)                    -> SEL
#   L3  g of the selected node (masked sum), neighbour/free flags     -> GSEL, CAND
#   L4  step(g[n] - GSEL - 1)                                         -> STP
#   L5  better = relu(CAND + STP - 1)                                 -> IMP
#   L6  big-M selects for g and parent; closed, done, log updates     (scratch cleared)
# Then 1 affine map and N layers for the path walk.


class Layer:
    """x' = W2 @ relu(W1 @ x + b1) + b2, assembled one hidden unit at a time."""

    def __init__(self, n_in, n_out):
        self.n_in, self.n_out = n_in, n_out
        self.rows, self.bias = [], []
        self.out = [[] for _ in range(n_out)]  # out[j] = [(unit, coefficient), ...]
        self.b2 = np.zeros(n_out)
        self.copies = {}

    def unit(self, row, bias=0.0):
        self.rows.append(np.asarray(row, dtype=float))
        self.bias.append(bias)
        return len(self.rows) - 1

    def copy(self, i):  # relu(x_i) = x_i because every x_i >= 0
        if i not in self.copies:
            e = np.zeros(self.n_in)
            e[i] = 1.0
            self.copies[i] = self.unit(e)
        return self.copies[i]

    def output(self, j, terms, const=0.0):
        self.out[j] = list(terms)
        self.b2[j] = const

    def keep(self, indices):
        for i in indices:
            self.output(i, [(self.copy(i), 1)])

    def finish(self):
        self.W1 = np.array(self.rows)
        self.b1 = np.array(self.bias)[:, None]  # column: works for a batch of inputs too
        self.W2 = np.zeros((self.n_out, len(self.rows)))
        for j, terms in enumerate(self.out):
            for unit, coef in terms:
                self.W2[j, unit] += coef
        self.b2 = self.b2[:, None]
        return self

    def __call__(self, x):  # x: (n_in, batch)
        return self.W2 @ np.maximum(0, self.W1 @ x + self.b1) + self.b2


# State layout: persistent registers first, then scratch registers.
SIZES = [("G", N), ("CLOSED", N), ("DONE", 1), ("WALL", N), ("P", N * N), ("LOG", N * N),
         ("RANK", N), ("REACHED", N), ("SEL", N), ("GSEL", 1), ("CAND", N), ("STP", N), ("IMP", N)]
OFF, NS = {}, 0
for name, size in SIZES:
    OFF[name] = NS
    NS += size
PERSIST = list(range(OFF["RANK"]))  # G .. LOG


def at(name, i=0):
    return OFF[name] + i


def row_of(**terms):
    # e.g. row_of(G=(3, 1.0), SEL=(3, -1.0)): a coefficient row over the state vector
    row = np.zeros(NS)
    for name, (i, coef) in terms.items():
        row[at(name, i)] += coef
    return row


def key_row(v):
    # key'[v] = N*(g[v] + H[v] + BIG*closed[v]) + v : affine in G and CLOSED (bias added by caller)
    row = np.zeros(NS)
    row[at("G", v)] = N
    row[at("CLOSED", v)] = N * BIG
    return row


def build_step():
    # L1: RANK[v] = sum_u step(key'[v] - key'[u]);  REACHED[v] = step(BIG - g[v])
    L1 = Layer(NS, NS)
    L1.keep(PERSIST)
    for v in range(N):
        terms = []
        for u in range(N):
            if u != v:
                d = key_row(v) - key_row(u)
                const = N * (H[v] - H[u]) + (v - u)
                a = L1.unit(d, const)  # relu(d)
                b = L1.unit(d, const - 1)  # relu(d - 1)
                terms += [(a, 1), (b, -1)]
        L1.output(at("RANK", v), terms)
        ra = L1.unit(-row_of(G=(v, 1)), BIG)  # relu(BIG - g)
        rb = L1.unit(-row_of(G=(v, 1)), BIG - 1)  # relu(BIG - g - 1)
        L1.output(at("REACHED", v), [(ra, 1), (rb, -1)])
    # L2: SEL[v] = relu(REACHED - CLOSED - DONE - RANK)   (open, not done, nobody smaller)
    L2 = Layer(NS, NS)
    L2.keep(PERSIST)
    for v in range(N):
        row = row_of(REACHED=(v, 1), CLOSED=(v, -1), DONE=(0, -1), RANK=(v, -1))
        L2.output(at("SEL", v), [(L2.unit(row), 1)])
    # L3: GSEL = sum_v relu(g[v] - BIG*(1 - sel[v]));   CAND[n] = relu(near[n] - wall[n] - closed[n])
    L3 = Layer(NS, NS)
    L3.keep(PERSIST + [at("SEL", v) for v in range(N)])
    gs_terms = []
    for v in range(N):
        gs_terms.append((L3.unit(row_of(G=(v, 1), SEL=(v, BIG)), -BIG), 1))
    L3.output(at("GSEL"), gs_terms)
    for n in range(N):
        row = row_of(WALL=(n, -1), CLOSED=(n, -1))
        for u in NEIGH[n]:
            if u != GOAL:
                row[at("SEL", u)] += 1  # near[n] = sum of sel over n's neighbours
        L3.output(at("CAND", n), [(L3.unit(row), 1)])
    # L4: STP[n] = step(g[n] - GSEL - 1)
    L4 = Layer(NS, NS)
    L4.keep(PERSIST + [at("SEL", v) for v in range(N)] + [at("GSEL")] + [at("CAND", n) for n in range(N)])
    for n in range(N):
        row = row_of(G=(n, 1), GSEL=(0, -1))
        a, b = L4.unit(row, -1), L4.unit(row, -2)
        L4.output(at("STP", n), [(a, 1), (b, -1)])
    # L5: IMP[n] = relu(CAND + STP - 1)   (AND of two 0/1 values)
    L5 = Layer(NS, NS)
    L5.keep(PERSIST + [at("SEL", v) for v in range(N)] + [at("GSEL")])
    for n in range(N):
        L5.output(at("IMP", n), [(L5.unit(row_of(CAND=(n, 1), STP=(n, 1)), -1), 1)])
    # L6: updates. Scratch registers are not copied, so they are 0 again afterwards.
    L6 = Layer(NS, NS)
    L6.keep([at("WALL", v) for v in range(N)])
    for n in range(N):
        # g' = g + select(IMP, GSEL + 1 - g, M=BIG)
        x = row_of(GSEL=(0, 1), G=(n, -1))
        u1 = L6.unit(x + row_of(IMP=(n, BIG)), 1 - BIG)  # relu(x - BIG*(1 - imp))
        u2 = L6.unit(-x + row_of(IMP=(n, BIG)), -1 - BIG)  # relu(-x - BIG*(1 - imp))
        L6.output(at("G", n), [(L6.copy(at("G", n)), 1), (u1, 1), (u2, -1)])
        # parent[n][u] = parent[n][u] + select(IMP, sel[u] - parent[n][u], M=1)
        for u in range(N):
            x = row_of(SEL=(u, 1), P=(n * N + u, -1))
            u1 = L6.unit(x + row_of(IMP=(n, 1)), -1)
            u2 = L6.unit(-x + row_of(IMP=(n, 1)), -1)
            L6.output(at("P", n * N + u), [(L6.copy(at("P", n * N + u)), 1), (u1, 1), (u2, -1)])
    for v in range(N):
        L6.output(at("CLOSED", v), [(L6.copy(at("CLOSED", v)), 1), (L6.copy(at("SEL", v)), 1)])
    # done' = 1 - sum(sel) + sel[goal]
    L6.output(at("DONE"), [(L6.copy(at("SEL", v)), -1) for v in range(N)] + [(L6.copy(at("SEL", GOAL)), 1)], 1.0)
    # log shifts up by one row of N; the new row is sel (so after N steps LOG[t] = sel of step t)
    for i in range(N * (N - 1)):
        L6.output(at("LOG", i), [(L6.copy(at("LOG", i + N)), 1)])
    for v in range(N):
        L6.output(at("LOG", N * (N - 1) + v), [(L6.copy(at("SEL", v)), 1)])
    return [L.finish() for L in (L1, L2, L3, L4, L5, L6)]


# input walls (N) -> initial state: G = [0 at start, BIG elsewhere], WALL = walls, rest 0
INIT_W = np.zeros((NS, N))
INIT_B = np.zeros((NS, 1))
for v in range(N):
    INIT_W[at("WALL", v), v] = 1
    INIT_B[at("G", v)] = 0 if v == START else BIG

# path walk. state Q = [P | CUR | PLOG | LOG | COST]
QO, QN = {}, 0
for name, size in [("P", N * N), ("CUR", N), ("PLOG", N * N), ("LOG", N * N), ("COST", 1)]:
    QO[name] = QN
    QN += size

# affine map: final search state -> Q.   CUR = found * onehot(goal);  COST = g[goal] - (BIG+1)(1-found)
TO_Q = np.zeros((QN, NS))
TO_Q_B = np.zeros((QN, 1))
for i in range(N * N):
    TO_Q[QO["P"] + i, at("P", i)] = 1
    TO_Q[QO["LOG"] + i, at("LOG", i)] = 1
TO_Q[QO["CUR"] + GOAL, at("CLOSED", GOAL)] = 1  # found = closed[goal]
TO_Q[QO["COST"], at("G", GOAL)] = 1
TO_Q[QO["COST"], at("CLOSED", GOAL)] = BIG + 1
TO_Q_B[QO["COST"]] = -BIG  # stored as cost + 1 so that it stays >= 0 (read-out subtracts 1)


def build_walk():
    L = Layer(QN, QN)
    L.keep([QO["P"] + i for i in range(N * N)] + [QO["LOG"] + i for i in range(N * N)] + [QO["COST"]])
    for u in range(N):
        terms = []
        for v in range(N):
            row = np.zeros(QN)
            row[QO["P"] + v * N + u] = 1
            row[QO["CUR"] + v] = 1
            terms.append((L.unit(row, -1), 1))  # AND(parent[v][u], cur[v])
        L.output(QO["CUR"] + u, terms)
    for i in range(N * (N - 1)):
        L.output(QO["PLOG"] + i, [(L.copy(QO["PLOG"] + i + N), 1)])
    for v in range(N):
        L.output(QO["PLOG"] + N * (N - 1) + v, [(L.copy(QO["CUR"] + v), 1)])
    return L.finish()


STEP_LAYERS = build_step()
WALK_LAYER = build_walk()

# Decoding vector: onehot @ (1..N) - 1 gives the node number (or -1 for all zeros).
NODE_NUMBERS = np.arange(1, N + 1, dtype=float)


def network_batch(walls_batch):
    x = INIT_W @ np.array(walls_batch, dtype=float).T + INIT_B  # (NS, batch)
    for _ in range(N):
        for layer in STEP_LAYERS:
            x = layer(x)
    q = TO_Q @ x + TO_Q_B
    for _ in range(N):
        q = WALK_LAYER(q)
    results = []
    for b in range(q.shape[1]):
        col = q[:, b]
        order = col[QO["LOG"] : QO["LOG"] + N * N].reshape(N, N) @ NODE_NUMBERS - 1
        path = col[QO["PLOG"] : QO["PLOG"] + N * N].reshape(N, N) @ NODE_NUMBERS - 1
        results.append((int(col[QO["COST"]]) - 1, order.astype(int).tolist(), path.astype(int).tolist()))
    return results


def network(walls):
    return network_batch([walls])[0]


# ── Check: all three agree ──────────────────────────────────────────────


def bfs_cost(walls):
    dist = {START: 0}
    queue = deque([START])
    while queue:
        u = queue.popleft()
        for n in NEIGH[u]:
            if not walls[n] and n not in dist:
                dist[n] = dist[u] + 1
                queue.append(n)
    return dist.get(GOAL, -1)


def layout(text):
    # "S.#G" style rows -> wall list ('#' = wall)
    cells = "".join(text.split())
    return [1 if ch == "#" else 0 for ch in cells]


def show(walls, result):
    cost, order, path = result
    rank = {v: t for t, v in enumerate(order) if v >= 0}
    for r in range(ROWS):
        cells = []
        for c in range(COLS):
            v = r * COLS + c
            if v == START:
                s = "S"
            elif walls[v]:
                s = "##"
            else:
                s = f"{rank[v]:2}" if v in rank else " ."
            star = "*" if v in path else " "
            cells.append(f"{s:>2}{star}")
        print("    " + " ".join(cells))
    print(f"    cost {cost}, expanded {[v for v in order if v >= 0]}, path {[v for v in reversed(path) if v >= 0]}")


if __name__ == "__main__":
    rng = random.Random(16)
    hand = {
        "empty": layout("....  ....  ....  ...."),
        "wall in the way (U-trap)": layout("..#. ..#. ..#. ...."),
        "goal enclosed": layout("..#. ...# .... ...."),
        "start enclosed": layout(".#.. #... .... ...."),
        "goal is a wall": layout("...# .... .... ...."),
        "long detour": layout(".#.. .#.# ...# #..."),
    }
    cases = list(hand.values())
    for _ in range(1000):
        p = rng.choice([0.1, 0.2, 0.3, 0.4])
        cases.append([1 if rng.random() < p else 0 for _ in range(N)])

    got_c = network_batch(cases)
    reachable = 0
    for walls, c in zip(cases, got_c):
        a = control_flow(walls)
        b = branchless(walls)
        assert b == a, ("B", walls, a, b)
        assert c == a, ("C", walls, a, c)
        walls_wo_start = [0 if v == START else walls[v] for v in range(N)]
        assert a[0] == bfs_cost(walls_wo_start), ("A* cost vs BFS", walls, a)  # A* is optimal here
        reachable += a[0] >= 0

    print("U-trap: S top-left, goal top-right; numbers = order of expansion, * = path")
    show(hand["wall in the way (U-trap)"], network(hand["wall in the way (U-trap)"]))
    print(f"network: {N} steps x {len(STEP_LAYERS)} layers + {N} path layers; "
          f"state {NS} numbers, {sum(len(L.rows) for L in STEP_LAYERS)} hidden neurons per step")
    print(f"OK: {len(cases)} wall layouts agree ({reachable} reachable, {len(cases) - reachable} not): "
          "cost, expansion order and path")
