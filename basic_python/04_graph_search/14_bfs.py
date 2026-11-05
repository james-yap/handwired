"""
14 · BFS: breadth-first search = one matrix-vector product per layer

    dist = [-1] * V;  dist[start] = 0;  queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in neighbours(u):
            if dist[v] == -1:
                dist[v] = dist[u] + 1;  queue.append(v)

A queue handles ONE node at a time. But BFS visits nodes in waves: all nodes at
distance 1, then all at distance 2, ... So process a whole wave per layer and
the queue disappears. The state is three vectors with one entry per node:

    visited   0/1   nodes reached so far
    frontier  0/1   nodes reached by the LAST wave
    dist      int   (distance so far, see below)

One layer (= one wave):

    reached  = A @ frontier                 # A[v][u] = 1 if u-v is an edge: weights!
    new      = step(reached - K * visited)  # touched by the wave AND never seen before
    visited  = visited + new
    frontier = new

K must be >= the largest possible value of `reached` (the max degree), so that a
visited node can never look "new". The graph's longest shortest path has V-1
edges, so V-1 identical layers always suffice. Unreachable nodes get -1.

Part 1 bakes a fixed graph into the weights (the start node is the input).
Part 2 makes the graph's edges inputs too: A @ frontier becomes a sum of
AND(edge, frontier) terms, because now two runtime values meet.

Inputs: start node (one-hot). Part 2: also one bit per possible edge.
Run:    uv run 04_graph_search/14_bfs.py
"""

import itertools
from collections import deque

import numpy as np


# ── Stage A · control flow ──────────────────────────────────────────────
# Normal Python. Anything goes. adj is a 0/1 adjacency matrix.


def control_flow(adj, start):
    dist = [-1] * len(adj)
    dist[start] = 0
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in range(len(adj)):
            if adj[u][v] and dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


# ── Stage B · branchless arithmetic ─────────────────────────────────────
# Rules: only  +  -  (multiply by a constant)  relu.
#        No `if`, no comparisons, no queue, no loops whose length depends on
#        the input (we always run V-1 waves).


def relu(x):
    return max(0, x)


def step(x):
    return relu(x) - relu(x - 1)  # 1 if x >= 1 else 0, for integers


# Part 1: a fixed graph (8 nodes; {6, 7} are cut off from the rest).
V = 8
EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (6, 7)]
ADJ = [[0] * V for _ in range(V)]
for a, b in EDGES:
    ADJ[a][b] = ADJ[b][a] = 1
K = max(sum(row) for row in ADJ)  # max degree = 3 = largest possible `reached`
LAYERS = V - 1  # a shortest path has at most V-1 edges


def branchless(start):
    # start: one-hot list. ADJ[v][u] * frontier[u] is constant * runtime: fine.
    visited, frontier, dist = list(start), list(start), [0] * V
    for _ in range(LAYERS):
        reached = [sum(ADJ[v][u] * frontier[u] for u in range(V)) for v in range(V)]
        new = [step(reached[v] - K * visited[v]) for v in range(V)]
        # Distance without ever multiplying by the layer number: every node that was
        # NOT yet visited when this wave started ages by one. A node found by wave d
        # ages d times; an unreachable node ages V-1 times.
        dist = [dist[v] + 1 - visited[v] for v in range(V)]
        visited = [visited[v] + new[v] for v in range(V)]
        frontier = new
    # Read-out: unreachable (never visited) has dist V-1; shift it to -1.
    return [dist[v] + V * visited[v] - V for v in range(V)]


# Part 2: the edges are runtime bits too (5 nodes, 10 possible undirected edges).
NV = 5
PAIRS = [(u, v) for u in range(NV) for v in range(u + 1, NV)]
NE = len(PAIRS)  # 10
KE = NV - 1  # `reached` counts frontier neighbours: at most NV-1


def branchless_edges(bits, start):
    e = list(bits)  # e[k] = 1 if edge PAIRS[k] is present
    visited, frontier, dist = list(start), list(start), [0] * NV
    for _ in range(NV - 1):
        reached = [0] * NV
        for k, (u, v) in enumerate(PAIRS):
            # Edge bit and frontier bit are BOTH runtime: AND(p, q) = relu(p + q - 1).
            reached[v] += relu(e[k] + frontier[u] - 1)  # wave crosses u -> v
            reached[u] += relu(e[k] + frontier[v] - 1)  # wave crosses v -> u
        new = [step(reached[v] - KE * visited[v]) for v in range(NV)]
        dist = [dist[v] + 1 - visited[v] for v in range(NV)]
        visited = [visited[v] + new[v] for v in range(NV)]
        frontier = new
    return [dist[v] + NV * visited[v] - NV for v in range(NV)]


# ── Stage C · network ───────────────────────────────────────────────────
# Rules: constant matrix @ vector, + constant vector, relu, slicing/concat.
#        Never multiply two runtime vectors together, never compare.
#
# Part 1.  state x = [visited | frontier | dist]      (3V numbers, all >= 0)
#   hidden h = relu(W1 @ x + b1) =
#       [ relu(A@frontier - K*visited)      step(.) = relu(p) - relu(p - 1)
#         relu(A@frontier - K*visited - 1)
#         relu(visited), relu(frontier), relu(dist) ]   <- copies: relu(x) = x for x >= 0
#   x' = W2 @ h + b2:
#       visited'  = visited + (a - b)         frontier' = a - b
#       dist'     = dist + 1 - visited
# The same (W1, b1, W2, b2) is used V-1 times.

I, Z = np.eye(V), np.zeros((V, V))
A = np.array(ADJ, dtype=float)
W1 = np.block([
    [-K * I, A, Z],  # p = A @ frontier - K * visited
    [-K * I, A, Z],  # p - 1  (bias below)
    [I, Z, Z],
    [Z, I, Z],
    [Z, Z, I],
])
b1 = np.concatenate([np.zeros(V), -np.ones(V), np.zeros(3 * V)])
W2 = np.block([
    [I, -I, I, Z, Z],  # visited' = a - b + visited
    [I, -I, Z, Z, Z],  # frontier' = a - b
    [Z, Z, -I, Z, I],  # dist' = dist - visited + 1
])
b2 = np.concatenate([np.zeros(2 * V), np.ones(V)])


def network(start):
    x = np.concatenate([start, start, np.zeros(V)]).astype(float)
    for _ in range(LAYERS):
        x = W2 @ np.maximum(0, W1 @ x + b1) + b2
    visited, dist = x[:V], x[2 * V :]
    return (dist + V * visited - V).astype(int).tolist()  # affine read-out


# Part 2.  state x = [e | visited | frontier | dist]   (NE + 3*NV numbers)
#   block 1 (hidden: one AND unit per edge direction, plus copies of x):
#       and_{k,u->v} = relu(e_k + frontier_u - 1)
#       writes  reached_v = sum of the AND units that point into v
#   block 2 (hidden: the two step units, plus copies of x):
#       new = step(reached - K*visited);  update visited, frontier, dist as in part 1
# So one wave is two hidden layers here instead of one.

NX = NE + 3 * NV
VIS, FRO, DST = NE, NE + NV, NE + 2 * NV  # offsets inside x


def build_edge_net():
    # block 1: x (NX) -> [x | reached] (NX + NV)
    W1a = np.zeros((2 * NE + NX, NX))
    b1a = np.zeros(2 * NE + NX)
    W2a = np.zeros((NX + NV, 2 * NE + NX))
    for k, (u, v) in enumerate(PAIRS):
        for d, (src, dst) in enumerate([(u, v), (v, u)]):
            row = 2 * k + d
            W1a[row, k] = 1  # e_k
            W1a[row, FRO + src] = 1  # frontier_src
            b1a[row] = -1
            W2a[NX + dst, row] = 1  # reached_dst += AND
    for i in range(NX):  # copy x through
        W1a[2 * NE + i, i] = 1
        W2a[i, 2 * NE + i] = 1
    # block 2: [x | reached] -> x
    W1b = np.zeros((2 * NV + NX, NX + NV))
    b1b = np.zeros(2 * NV + NX)
    W2b = np.zeros((NX, 2 * NV + NX))
    b2b = np.zeros(NX)
    for v in range(NV):
        for r, shift in enumerate([0, -1]):  # a = relu(p), b = relu(p - 1)
            W1b[r * NV + v, NX + v] = 1  # reached_v
            W1b[r * NV + v, VIS + v] = -KE  # - K * visited_v
            b1b[r * NV + v] = shift
        W2b[VIS + v, v], W2b[VIS + v, NV + v] = 1, -1  # visited' = visited + a - b
        W2b[FRO + v, v], W2b[FRO + v, NV + v] = 1, -1  # frontier' = a - b
    for i in range(NX):
        W1b[2 * NV + i, i] = 1  # copy of x_i
        if not FRO <= i < DST:  # (the old frontier is replaced, not kept)
            W2b[i, 2 * NV + i] = 1
    for v in range(NV):
        W2b[DST + v, 2 * NV + VIS + v] = -1  # dist' = dist - visited + 1
        b2b[DST + v] = 1
    return W1a, b1a, W2a, W1b, b1b, W2b, b2b


W1a, b1a, W2a, W1b, b1b, W2b, b2b = build_edge_net()


def network_edges(bits, start):
    x = np.concatenate([bits, start, start, np.zeros(NV)]).astype(float)
    for _ in range(NV - 1):
        xr = W2a @ np.maximum(0, W1a @ x + b1a)  # [x | reached]
        x = W2b @ np.maximum(0, W1b @ xr + b1b) + b2b
    visited, dist = x[VIS:FRO], x[DST:]
    return (dist + NV * visited - NV).astype(int).tolist()


# ── Check: all three agree ──────────────────────────────────────────────


def one_hot(i, n):
    return [1 if k == i else 0 for k in range(n)]


if __name__ == "__main__":
    # Part 1: fixed graph, every start node.
    for s in range(V):
        expected = control_flow(ADJ, s)
        assert branchless(one_hot(s, V)) == expected, (s, branchless(one_hot(s, V)), expected)
        assert network(one_hot(s, V)) == expected, (s, network(one_hot(s, V)), expected)
    print(f"fixed graph (V={V}, max degree K={K}, {LAYERS} layers); distances from node 0 and 6:")
    print("  node | start 0: A  B  C | start 6: A  B  C")
    for v in range(V):
        row = []
        for s in (0, 6):
            row += [control_flow(ADJ, s)[v], branchless(one_hot(s, V))[v], network(one_hot(s, V))[v]]
        print(f"  {v:4} |          {row[0]:2} {row[1]:2} {row[2]:2} |          {row[3]:2} {row[4]:2} {row[5]:2}")

    # Part 2: every undirected graph on 5 nodes (2^10 = 1024) x every start node.
    count = 0
    for bits in itertools.product([0, 1], repeat=NE):
        adj = [[0] * NV for _ in range(NV)]
        for k, (u, v) in enumerate(PAIRS):
            adj[u][v] = adj[v][u] = bits[k]
        for s in range(NV):
            expected = control_flow(adj, s)
            got_b = branchless_edges(bits, one_hot(s, NV))
            got_c = network_edges(bits, one_hot(s, NV))
            assert got_b == expected, (bits, s, got_b, expected)
            assert got_c == expected, (bits, s, got_c, expected)
            count += 1
    print(f"runtime edges: {2 ** NE} graphs x {NV} starts, {2 * (NV - 1)} hidden layers each")
    print(f"OK: fixed graph x {V} starts, and {count} (graph, start) cases with runtime edges all agree")
