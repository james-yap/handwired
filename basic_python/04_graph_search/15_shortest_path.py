"""
15 · Shortest paths: Dijkstra's heap vs Bellman-Ford relaxation

Dijkstra keeps a priority queue and settles ONE node per step: inherently
sequential, and "pop the smallest" is a data-dependent branch. Bellman-Ford
has no queue at all. Every layer it relaxes EVERY edge:

    d[v] = min(d[v],  min over u of (d[u] + w[u][v]))

repeated V-1 times (a shortest path has at most V-1 edges). That is already
data flow: the same arithmetic on every node, every layer. "No path yet" is a
big finite number INF, and `min` is built from relu: min(a, b) = a - relu(a - b).

The weights are inputs too. d[u] + w[u][v] is only an addition, so no AND or
select is needed anywhere: the edge weights simply flow into the sum.

Inputs: one weight per possible undirected edge of 5 nodes, an integer in
        [0, 3] where 0 means "no edge", and the start node (one-hot).
Output: shortest distance from the start to every node, -1 if unreachable.
Run:    uv run 04_graph_search/15_shortest_path.py
"""

import heapq
import random

import numpy as np

V = 5
PAIRS = [(u, v) for u in range(V) for v in range(u + 1, V)]  # 10 undirected edges
E = len(PAIRS)
WMAX = 3
ROUNDS = V - 1  # a shortest path has at most V-1 edges

# INF = "no path yet", a finite stand-in. Derivation:
#   * any real walk the algorithm ever measures has at most V-1 edges of weight <= WMAX,
#     plus one more edge being tried:  at most V*WMAX = 15.
#   * so every real value is < INF = 16, and anything >= INF means "unreachable".
#   * "no edge" costs INF too, so  d[u] + INF >= INF  can never beat a real value.
INF = V * WMAX + 1

PAIR_INDEX = {}  # (u, v) -> index into the weight vector, both directions
for k, (u, v) in enumerate(PAIRS):
    PAIR_INDEX[u, v] = PAIR_INDEX[v, u] = k


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes. Two textbook algorithms; they must agree.


def dijkstra(weights, start):
    dist = [-1] * V
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if dist[u] != -1:
            continue  # already settled
        dist[u] = d
        for v in range(V):
            if v != u and weights[PAIR_INDEX[u, v]] > 0 and dist[v] == -1:
                heapq.heappush(heap, (d + weights[PAIR_INDEX[u, v]], v))
    return dist


def bellman_ford(weights, start):
    big = float("inf")
    d = [big] * V
    d[start] = 0
    for _ in range(V - 1):
        for u in range(V):
            for v in range(V):
                w = weights[PAIR_INDEX[u, v]] if u != v else 0
                if w > 0 and d[u] + w < d[v]:
                    d[v] = d[u] + w
    return [x if x != big else -1 for x in d]


control_flow = dijkstra


# ── Stage B · branchless arithmetic ─────────────────────────────────────
# Rules: only  +  -  (multiply by a constant)  relu.
#        No `if`, no comparisons, no heap, no loops whose length depends on
#        the input (V-1 rounds, always).


def relu(x):
    return max(0, x)


def step(x):
    return relu(x) - relu(x - 1)  # 1 if x >= 1 else 0, for integers


def min_tree(xs):
    # min of many values by pairing neighbours: ceil(log2 n) levels of
    # min(a, b) = a - relu(a - b). The lengths are fixed, so this is a fixed loop.
    xs = list(xs)
    while len(xs) > 1:
        paired = [xs[i] - relu(xs[i] - xs[i + 1]) for i in range(0, len(xs) - 1, 2)]
        xs = paired + xs[len(xs) // 2 * 2 :]  # an odd one out just moves up a level
    return xs[0]


def branchless(weights, start):
    # weight 0 ("no edge") becomes cost INF; weights 1..WMAX stay as they are.
    cost = [w + INF * (1 - step(w)) for w in weights]
    d = [INF * (1 - s) for s in start]  # 0 at the start node, INF elsewhere
    for _ in range(ROUNDS):
        d = [
            min_tree([d[v]] + [d[u] + cost[PAIR_INDEX[u, v]] for u in range(V) if u != v])
            for v in range(V)
        ]
    # Read-out: d == INF means unreachable. reach = 1 if d < INF.  d - (INF+1) = -1 there.
    return [d[v] - (INF + 1) * (1 - step(INF - d[v])) for v in range(V)]


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
# Every layer is  x' = W2 @ relu(W1 @ x + b1) + b2.  Layers are built at import
# time by the little `Layer` class below (it only fills in constant matrices).
# All values are >= 0, so relu(x) = x copies a number through a layer.
#
#   pre   : input [w | s] -> state [D | C]      D = INF*(1 - s),  C = w + INF*(1 - step(w))
#   round : D' = min over candidates of group v, group v = { D[v] } + { D[u] + C[uv] }.
#           Pairing neighbours takes 3 hidden layers for 5 candidates (5 -> 3 -> 2 -> 1).
#           Each level: unit relu(a), unit relu(a - b), output a - relu(a - b).
#   post  : out = D - (INF+1)*(1 - step(INF - D))      (-1 for unreachable)


def basis(i, n):
    row = np.zeros(n)
    row[i] = 1.0
    return row


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
            self.copies[i] = self.unit(basis(i, self.n_in))
        return self.copies[i]

    def output(self, j, terms, const=0.0):
        self.out[j] = list(terms)
        self.b2[j] = const

    def finish(self):
        self.W1 = np.array(self.rows)
        self.b1 = np.array(self.bias)
        self.W2 = np.zeros((self.n_out, len(self.rows)))
        for j, terms in enumerate(self.out):
            for unit, coef in terms:
                self.W2[j, unit] += coef
        return self

    def __call__(self, x):
        return self.W2 @ np.maximum(0, self.W1 @ x + self.b1) + self.b2


def pre_layer():
    L = Layer(E + V, V + E)  # input [w | s], state [D | C]
    for k in range(E):
        a = L.unit(basis(k, E + V))  # relu(w) = w
        b = L.unit(basis(k, E + V), -1)  # relu(w - 1)
        # C = w + INF - INF*(a - b)
        L.output(V + k, [(a, 1 - INF), (b, INF)], INF)
    for v in range(V):
        L.output(v, [(L.copy(E + v), -INF)], INF)  # D = INF - INF*s
    return L.finish()


def min_layers(groups, n_in, carry):
    """Layers that take the min of every group (a list of rows over the state).
    The state entries listed in `carry` are copied after the group results."""
    layers = []
    while len(groups[0]) > 1:
        sizes = [(len(g) + 1) // 2 for g in groups]
        n_out = sum(sizes) + len(carry)
        L = Layer(n_in, n_out)
        j = 0
        for g in groups:
            for i in range(0, len(g) - 1, 2):
                a, b = g[i], g[i + 1]
                ua = L.unit(a)  # relu(a) = a
                ud = L.unit(a - b)  # relu(a - b)
                L.output(j, [(ua, 1), (ud, -1)])  # min(a, b) = a - relu(a - b)
                j += 1
            if len(g) % 2:
                L.output(j, [(L.unit(g[-1]), 1)])  # odd one out: copy up
                j += 1
        for c in carry:
            L.output(j, [(L.copy(c), 1)])
            j += 1
        layers.append(L.finish())
        groups = []  # next level reads the new state directly
        j = 0
        for s in sizes:
            groups.append([basis(j + t, n_out) for t in range(s)])
            j += s
        carry = list(range(sum(sizes), n_out))
        n_in = n_out
    return layers


def round_layers():
    n = V + E  # state [D | C]
    groups = []
    for v in range(V):
        cands = [basis(v, n)]  # keep the current distance
        for u in range(V):
            if u != v:
                cands.append(basis(u, n) + basis(V + PAIR_INDEX[u, v], n))  # D[u] + C[uv]
        groups.append(cands)
    return min_layers(groups, n, carry=list(range(V, n)))


def post_layer():
    L = Layer(V + E, V)
    for v in range(V):
        row = -basis(v, V + E)
        a = L.unit(row, INF)  # relu(INF - D)
        b = L.unit(row, INF - 1)  # relu(INF - D - 1)
        # out = D - (INF+1) + (INF+1) * (a - b)
        L.output(v, [(L.copy(v), 1), (a, INF + 1), (b, -(INF + 1))], -(INF + 1))
    return L.finish()


ROUND = round_layers()
LAYERS = [pre_layer()] + ROUND * ROUNDS + [post_layer()]  # the round is the same layers, repeated


def network(weights, start):
    x = np.concatenate([weights, start]).astype(float)
    for layer in LAYERS:
        x = layer(x)
    return x.astype(int).tolist()


# ── Check: all three agree ──────────────────────────────────────────────


def one_hot(i, n):
    return [1 if k == i else 0 for k in range(n)]


def weights_from(edges):
    w = [0] * E
    for (u, v), x in edges.items():
        w[PAIR_INDEX[u, v]] = x
    return w


if __name__ == "__main__":
    rng = random.Random(15)
    cases = []
    # hand-picked: no edges; all weight 1; a path of weight-3 edges (longest distance 12);
    # a trap where the direct edge (0-4, weight 3) loses to two light hops via node 1.
    cases.append([0] * E)
    cases.append([1] * E)
    cases.append(weights_from({(0, 1): 3, (1, 2): 3, (2, 3): 3, (3, 4): 3}))
    trap = weights_from({(0, 4): 3, (0, 1): 1, (1, 4): 1, (1, 2): 2, (2, 3): 1})
    cases.append(trap)
    cases.append([WMAX] * E)
    # random weights: each edge absent with probability p (random per case).
    for _ in range(2000):
        p = rng.choice([0.2, 0.5, 0.8])
        cases.append([0 if rng.random() < p else rng.randint(1, WMAX) for _ in range(E)])

    count = 0
    for w in cases:
        starts = range(V) if count < 5 * V else [rng.randrange(V)]
        for s in starts:
            expected = control_flow(w, s)
            assert bellman_ford(w, s) == expected, (w, s)  # the two textbook versions agree
            assert branchless(w, one_hot(s, V)) == expected, (w, s, branchless(w, one_hot(s, V)), expected)
            assert network(w, one_hot(s, V)) == expected, (w, s, network(w, one_hot(s, V)), expected)
            count += 1

    hidden = [len(L.rows) for L in LAYERS]
    print(f"network: {len(LAYERS)} layers, {sum(hidden)} hidden neurons ({ROUNDS} identical rounds of {len(ROUND)} layers)")
    print("trap graph, start 0:  node | A  B  C")
    s = one_hot(0, V)
    for v in range(V):
        print(f"                       {v:4} | {control_flow(trap, 0)[v]:2} {branchless(trap, s)[v]:2} {network(trap, s)[v]:2}")
    print(f"OK: {count} (weights, start) cases agree (Dijkstra == Bellman-Ford == branchless == network)")
