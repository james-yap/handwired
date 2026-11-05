#import "style.typ": *

#show: booklet.with(
  chapter: "Chapter 4",
  title: "Graph search",
  subtitle: "BFS, shortest paths and A*: queues and heaps become layers",
)

// ── local helpers ───────────────────────────────────────────────────────
#let layer-fill = (rgb("1f2328"), rgb("93c5fd"), rgb("86efac"), rgb("fcd34d"), rgb("f9a8d4"), rgb("c4b5fd"))
#let tiny(body) = text(size: 8.5pt, body)
#let mono(body) = text(font: "Menlo", size: 8pt, body)

// A graph drawn with cetz. pos: node coordinates, edges: (u, v) pairs,
// fills / outlines: one entry per node, weights: optional label per edge.
#let draw-graph(pos, edges, fills, outlines: none, weights: none, scale: 1.0, r: 0.36, font: 9pt) = cetz.canvas(length: scale * 1cm, {
  import cetz.draw: *
  for (k, (u, v)) in edges.enumerate() {
    line(pos.at(u), pos.at(v), stroke: 1pt + muted)
    if weights != none {
      content(
        (pos.at(u), 50%, pos.at(v)), text(size: font, weight: "bold")[#weights.at(k)],
        frame: "rect", fill: white, stroke: none, padding: 1.5pt,
      )
    }
  }
  for (i, p) in pos.enumerate() {
    let dark = fills.at(i) == layer-fill.at(0)
    let outline = if outlines == none { 0.8pt + ink } else { outlines.at(i) }
    circle(p, radius: r, fill: fills.at(i), stroke: outline)
    content(p, text(size: font, fill: if dark { white } else { ink })[#i])
  }
})

// A small feed-forward picture. layers: array of arrays of node labels (content),
// wires: (layer i, node j, layer k, node l, label, colour).
#let nn(layers, wires, r: 0.42, gapx: 3.4, gapy: 1.25, font: 8pt) = {
  let pos(i, j) = (i * gapx, -(j - (layers.at(i).len() - 1) / 2) * gapy)
  cetz.canvas({
    import cetz.draw: *
    for w in wires {
      let (i, j, k, l, lab, col, ..rest) = w
      let a = pos(i, j)
      let b = pos(k, l)
      line(a, b, stroke: 0.9pt + col, mark: (end: ">", size: 0.18))
      if lab != none {
        content((a, if w.len() > 6 { w.at(6) } else { 38% }, b), text(size: font, fill: col)[#lab], frame: "rect", fill: white, stroke: none, padding: 1pt)
      }
    }
    for (i, layer) in layers.enumerate() {
      for (j, lab) in layer.enumerate() {
        circle(pos(i, j), radius: r, fill: white, stroke: 0.9pt + ink)
        content(pos(i, j), text(size: font)[#lab])
      }
    }
  })
}

#let plus = network-color
#let minus = rgb("b91c1c")

#let ex-hint(body) = block(inset: (left: 8pt), stroke: (left: 1.5pt + rule-color), body)

// Two stages side by side, code at 7pt so that ~44 characters fit a column; never split across pages.
#let code-sbs(l, left, r, right) = block(breakable: false, {
  show raw: set text(size: 7pt)
  side-by-side(l, left, r, right)
})

= The shape of a search algorithm

Chapters 1 to 3 gave us the toolbox: masks, `select`, one-hot reads and writes, `argmin`, and loops
unrolled into a fixed number of identical layers. This chapter puts them to work on the three classic
graph-search algorithms. All three have the same skeleton:

#key-idea[
  *State* is a handful of vectors with one entry per node ("is it visited?", "what is its distance so far?").
  One *layer* is one round of the algorithm: *select* some nodes, *expand* them (along edges), *update* the
  vectors. Run the same layer a fixed number of times. A *halt flag* makes the layers after the finish a no-op.
]

What changes between BFS, Dijkstra / Bellman-Ford and A\* is only the *select* step:

#table(
  columns: (auto, 1fr, 1fr),
  align: left,
  [*Algorithm*], [*select*], [*why it becomes data flow*],
  [BFS], [every node of the current layer, at once], [a layer of nodes needs no queue: all of it is the frontier vector],
  [Bellman-Ford], [every edge, every round], [nothing is selected: relax everything, `min` keeps the best],
  [A\*], [the single open node with the lowest f], [it stays sequential: one `argmin` one-hot per layer, N layers],
)

Each example below is the same three stages as before. There are two ways to hold a graph. *In the weights:* the
adjacency matrix is a constant matrix, built at import time (BFS part 1; the grid's neighbours in A\*). *In the input:* the edges
(BFS part 2), the edge weights (shortest paths) or the walls (A\*) are runtime numbers, and the network has to combine them with the
state using ANDs, sums and selects instead of a constant matrix.

#pagebreak()

// ═══════════════════════════════════════════════════════════════════════
#example("14", "BFS: a queue becomes a wave", "04_graph_search/14_bfs.py")

== The problem

Given an undirected graph and a start node, find the number of edges on a shortest path from the start to every
node, and $-1$ for nodes the start cannot reach. Breadth-first search does this by visiting nodes in
order of distance, with a FIFO queue.

#align(center, draw-graph(
  ((0, 1), (1.6, 2), (1.6, 0), (3.2, 1), (4.8, 1), (6.4, 1), (2.4, -1.6), (4.0, -1.6)),
  ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (6, 7)),
  (white,) * 8,
  scale: 0.9,
))

The example graph has 8 nodes. The component ${6, 7}$ is cut off from the rest, so starting from node 0 those
two stay at $-1$. Everything below uses this graph; start node 0 unless stated otherwise.

== A and B side by side

#code-sbs(
  "A",
  [```python
def control_flow(adj, start):
    dist = [-1] * len(adj)
    dist[start] = 0
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in range(len(adj)):
            if adj[u][v] and \
                    dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist
```],
  "B",
  [```python
def branchless(start):  # one-hot
    visited = list(start)
    frontier = list(start)
    dist = [0] * V
    for _ in range(V - 1):
        reached = [
            sum(ADJ[v][u] * frontier[u]
                for u in range(V))
            for v in range(V)]
        new = [step(reached[v]
                    - K * visited[v])
               for v in range(V)]
        dist = [dist[v] + 1 - visited[v]
                for v in range(V)]
        visited = [visited[v] + new[v]
                   for v in range(V)]
        frontier = new
    return [dist[v] + V * visited[v] - V
            for v in range(V)]
```],
)

Nothing in B depends on the *values*: there is no `while queue`, no `if`, no `dist[v] == -1`. The only things
that look at data are `relu` (inside `step`) and arithmetic.

== Why the queue disappears

A queue is a way of processing nodes one at a time while keeping them in distance order. But look at what is
in the queue at any moment: nodes of at most two *consecutive* distances. BFS is secretly
*level-synchronous*: first every node at distance 1 gets processed, then every node at distance 2, and so on.
A level never depends on the order inside the previous level, only on the *set* of nodes in it. So instead of
a queue, keep that set as a 0/1 vector, the *frontier*, and process all of it in one layer.

#key-idea[
  Queue (ordered list, one node at a time) $arrow.r$ *frontier vector* (a set, a whole level at a time).
  The order inside a level never mattered, so nothing is lost.
]

The state is three vectors, one entry per node:

#table(
  columns: (auto, auto, 1fr),
  align: left,
  [*vector*], [*values*], [*meaning*],
  [`visited`], [0/1], [nodes reached so far (the "dist[v] != -1" test, as a vector)],
  [`frontier`], [0/1], [nodes reached by the *last* wave, i.e. the nodes to expand now],
  [`dist`], [integers], [distance accumulator (explained below)],
)

== One layer = one wave

#block(inset: (left: 10pt))[
  $ "reached" &= A thin "frontier" \
    "new" &= step("reached" - K dot "visited") \
    "visited" &arrow.l "visited" + "new" \
    "frontier" &arrow.l "new" $
]

*The adjacency matrix is the weight matrix.* $A$ is the 0/1 matrix with $A_(v u) = 1$ when $u$ and $v$ are
joined by an edge. For a frontier vector $f$, $(A f)_v = sum_u A_(v u) f_u$ counts how many frontier nodes
neighbour $v$. This single matrix-vector product is "for each node in the frontier, look at all its neighbours",
done for all frontier nodes at once. For our graph:

#let adj-rows = (
  (0, 1, 1, 0, 0, 0, 0, 0),
  (1, 0, 1, 1, 0, 0, 0, 0),
  (1, 1, 0, 1, 0, 0, 0, 0),
  (0, 1, 1, 0, 1, 0, 0, 0),
  (0, 0, 0, 1, 0, 1, 0, 0),
  (0, 0, 0, 0, 1, 0, 0, 0),
  (0, 0, 0, 0, 0, 0, 0, 1),
  (0, 0, 0, 0, 0, 0, 1, 0),
)
#align(center, $ A = #math.mat(delim: "[", ..adj-rows.map(r => r.map(x => [#x]))) $)

Row 3 reads "node 3 touches nodes 1, 2 and 4". The largest row sum is 3 (nodes 1, 2 and 3): the *maximum degree*.

*Filtering the already visited: why $K$.* `reached[v] > 0` only says "some frontier node touches $v$". We want
"... and $v$ was never visited". The data-flow way to say "and not" is to subtract a big number when the
bad thing is true:

$ p_v = "reached"_v - K dot "visited"_v quad quad "new"_v = step(p_v) $

#table(
  columns: (auto, auto, auto, 1fr),
  align: (left, center, center, left),
  [*$v$ visited?*], [*reached*], [*$p_v$*], [*$"new"_v = step(p_v)$*],
  [no], [0], [0], [0: nobody touches it],
  [no], [$>= 1$], [$>= 1$], [1: newly discovered],
  [yes], [$0 .. K$], [$<= 0$], [0: never again, as long as $K >= "reached"$],
)

So $K$ must be at least the largest value $"reached"_v$ can take. That is the number of frontier neighbours
of $v$, at most the maximum degree. Here $K = 3$. Too small a $K$ is a real bug, and the trace below shows it.

*Distances without a layer counter.* The textbook writes `dist[v] = dist[u] + 1`; the obvious data-flow version
adds `t * new` in layer $t$, which would need a different constant per layer. We want $V - 1$ *identical*
layers, so use this instead: every layer, every node that is *not yet visited* gets older by one.

$ "dist" arrow.l "dist" + 1 - "visited" quad ("visited" "before" "this" "wave") $

A node discovered by wave $d$ was unvisited at the start of waves $1, ..., d$, so it ages $d$ times and ends
at $d$. The start node is visited from the beginning and stays at 0. A node never reached ages in all $V - 1$
waves and ends at $V - 1$; a final affine read-out $"dist" + V dot "visited" - V$ turns that into $-1$
(reachable nodes have $"visited" = 1$ and are left alone).

*How many layers?* A shortest path has at most $V - 1$ edges, so $V - 1$ waves always suffice. After the last real
wave the frontier is all zeros, so the remaining layers change nothing: *halting is free* in BFS, the wave simply
dies out.

== Worked trace (start node 0)

#let bfs-dist = (0, 1, 1, 2, 3, 4, -1, -1)
#let bfs-snapshot(t) = {
  let fills = bfs-dist.map(d => if d >= 0 and d <= t { layer-fill.at(d) } else { white })
  let outlines = bfs-dist.map(d => if d == t { 2.2pt + ink } else { 0.7pt + muted })
  draw-graph(
    ((0, 1), (1.6, 2), (1.6, 0), (3.2, 1), (4.8, 1), (6.4, 1), (2.4, -1.6), (4.0, -1.6)),
    ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (6, 7)),
    fills, outlines: outlines, scale: 0.4, r: 0.5, font: 7pt,
  )
}
#align(center, grid(
  columns: 5, gutter: 5pt, row-gutter: 3pt,
  ..range(5).map(t => align(center, bfs-snapshot(t))),
  ..range(5).map(t => tiny[after wave #t]),
))
#align(center, tiny[Colour = the wave that discovered the node. The thick outline is the current frontier.])

Numbers, in the order the layers compute them (node numbers 0 to 7 left to right inside each vector):

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (center, left, left, left, left),
  table.header([*wave*], [*frontier $f$*], [*reached $= A f$*], [*new*], [*visited after*]),
  [1], mono[1 0 0 0 0 0 0 0], mono[0 1 1 0 0 0 0 0], mono[0 1 1 0 0 0 0 0], mono[1 1 1 0 0 0 0 0],
  [2], mono[0 1 1 0 0 0 0 0], mono[2 1 1 2 0 0 0 0], mono[0 0 0 1 0 0 0 0], mono[1 1 1 1 0 0 0 0],
  [3], mono[0 0 0 1 0 0 0 0], mono[0 1 1 0 1 0 0 0], mono[0 0 0 0 1 0 0 0], mono[1 1 1 1 1 0 0 0],
  [4], mono[0 0 0 0 1 0 0 0], mono[0 0 0 1 0 1 0 0], mono[0 0 0 0 0 1 0 0], mono[1 1 1 1 1 1 0 0],
  [5], mono[0 0 0 0 0 1 0 0], mono[0 0 0 0 1 0 0 0], mono[0 0 0 0 0 0 0 0], mono[1 1 1 1 1 1 0 0],
)

Wave 2 is the interesting one. Frontier ${1, 2}$ touch each other and node 0 *and* node 3. $A f$ is
$[2, 1, 1, 2, 0, ...]$: node 0 is hit twice and node 3 is hit twice. With $K = 3$ and $"visited" = [1,1,1,0,...]$
we get $p = [-1, -2, -2, 2, 0, ...]$, so only node 3 is new. With $K = 1$ we would get
$p_0 = 2 - 1 = 1$ and node 0 would be "discovered" again: its distance would be corrupted and `visited[0]`
would become 2. A visited node that is hit by several frontier nodes needs $K >= $ that count.

The `dist` accumulator along the way (all 8 nodes): $[0,0,0,0,0,0,0,0]$ $arrow.r$ $[0,1,1,1,1,1,1,1]$
$arrow.r$ $[0,1,1,2,2,2,2,2]$ $arrow.r$ $[0,1,1,2,3,3,3,3]$ $arrow.r$ $[0,1,1,2,3,4,4,4]$, and then
$[0,1,1,2,3,4,5,5]$, $[0,1,1,2,3,4,6,6]$, $[0,1,1,2,3,4,7,7]$ after the idle waves 5 to 7. The read-out
$"dist" + 8 dot "visited" - 8$ gives $[0,1,1,2,3,4,-1,-1]$ since nodes 6 and 7 have $"visited" = 0$ and
$7 + 0 - 8 = -1$.

== Stage C: the weights

State $x = [" visited " | " frontier " | " dist "]$ has $3V = 24$ numbers, all $>= 0$. That matters: for $x >= 0$,
$relu(x) = x$, so a ReLU neuron can *copy* a number through a layer. One layer is $x' = W_2 relu(W_1 x + b_1) + b_2$
with hidden vector $h = [relu(p) | relu(p - 1) | "visited" | "frontier" | "dist"]$:

$ W_1 = mat(-K I, A, 0; -K I, A, 0; I, 0, 0; 0, I, 0; 0, 0, I), quad
  b_1 = vec(0, -1, 0, 0, 0), quad
  W_2 = mat(I, -I, I, 0, 0; I, -I, 0, 0, 0; 0, 0, -I, 0, I), quad
  b_2 = vec(0, 0, 1) $

Each block is $8 times 8$ ($I$ = identity, $0$ = zero block; $b_1, b_2$ list one value per block). Read $W_1$ row
by row: the first block row is $p = A f - K v$, the second is the same with $-1$ in the bias, so
$relu(p) - relu(p-1) = step(p)$; the last three block rows just copy the state. Read $W_2$: the new `visited` is
`visited` plus $relu(p) - relu(p-1)$; the new `frontier` is just $relu(p) - relu(p-1)$; the new `dist` is `dist` $-$ `visited` $+ 1$.

#align(center, diagram(
  node-stroke: 0.7pt, node-inset: 5pt, node-corner-radius: 3pt, spacing: (17mm, 9mm),
  node((0, 0), [frontier $f$], name: <f>),
  node((0, 1.5), [visited $v$], name: <v>),
  node((0, 3), [dist $d$], name: <d>),
  node((1, 0.75), [$p = A f - K v$], name: <p>),
  node((2, 0.35), [$relu(p)$], name: <a>),
  node((2, 1.15), [$relu(p - 1)$], name: <b>),
  node((3, 0.75), [new], name: <n>, fill: tint(network-color)),
  node((4, 0), [$f' = "new"$], name: <f2>),
  node((4, 1.5), [$v' = v + "new"$], name: <v2>),
  node((4, 3), [$d' = d + 1 - v$], name: <d2>),
  edge(<f>, <p>, "->", label: $A$, label-side: left),
  edge(<v>, <p>, "->", label: $-K$, label-side: right),
  edge(<p>, <a>, "->"), edge(<p>, <b>, "->"),
  edge(<a>, <n>, "->", label: $+1$, label-side: left),
  edge(<b>, <n>, "->", label: $-1$, label-side: right),
  edge(<n>, <f2>, "->"), edge(<n>, <v2>, "->"),
  edge(<v>, <v2>, "->", label: $+1$, label-side: right),
  edge(<d>, <d2>, "->", label: $+1$, label-side: right),
  edge(<v>, <d2>, "->", label: $-1$, label-side: left),
))
#align(center, tiny[One BFS layer. "new" is $relu(p) - relu(p - 1) = step(p)$.])

To see actual neurons, here is *one node* of one layer. Take node 3 (neighbours 1, 2, 4) and compute its entry of `new`:

#align(center, nn(
  (
    ($f_1$, $f_2$, $f_4$, $v_3$),
    ($relu(p_3)$, $relu(p_3 - 1)$),
    ($"new"_3$,),
  ),
  (
    (0, 0, 1, 0, $1$, plus), (0, 1, 1, 0, $1$, plus), (0, 2, 1, 0, $1$, plus), (0, 3, 1, 0, $-3$, minus),
    (0, 0, 1, 1, none, plus), (0, 1, 1, 1, none, plus), (0, 2, 1, 1, none, plus), (0, 3, 1, 1, none, minus),
    (1, 0, 2, 0, $+1$, plus), (1, 1, 2, 0, $-1$, minus),
  ),
  r: 0.6, font: 7pt, gapx: 3.6, gapy: 1.4,
))
#align(center, tiny[$p_3 = f_1 + f_2 + f_4 - 3 v_3$; the lower hidden neuron has bias $-1$. The whole layer is 8 of these side by side, plus copy neurons.])

The same $(W_1, b_1, W_2, b_2)$ is applied 7 times. Cost: *40 hidden neurons per layer* ($2V$ for the step, $3V$ for the copies), 7 layers.

== Part 2: the graph is an input too

In part 1 the graph is a constant, so $A f$ is a *constant matrix times a runtime vector*: free. Now let the
edges be runtime bits $e_k in {0, 1}$ (one per possible edge; 5 nodes have 10, so $2^10 = 1024$ graphs). The
product is no longer constant $times$ runtime but runtime $times$ runtime:
"$e_(u v) dot f_u$". Product of two 0/1 numbers is an AND, and AND is one ReLU:

$ AND(e_(u v), f_u) = relu(e_(u v) + f_u - 1) $

For every edge and each of its two directions, one such neuron. $"reached"_v$ is then the sum of the AND neurons
that point into $v$:

$ "reached"_v = sum_(u) relu(e_(u v) + f_u - 1), quad "new"_v = step("reached"_v - K dot "visited"_v) $

#align(center, nn(
  (
    ($e_(u v)$, $f_u$),
    ($relu$,),
    ($r_v$,),
  ),
  (
    (0, 0, 1, 0, $1$, plus), (0, 1, 1, 0, $1$, plus), (1, 0, 2, 0, $1$, plus),
  ),
  gapx: 4.2, r: 0.5,
))
#align(center, tiny[One AND neuron (bias $-1$): fires only when the edge is present *and* $u$ is on the frontier.])

The wave now needs *two* hidden layers instead of one: first the AND neurons (and `reached` as their sum), then
the `step` neurons that need `reached`. $K = V - 1 = 4$ is now the bound ($"reached"_v$ counts at most $V - 1$ incoming edges).
Cost for 5 nodes: block 1 has $2 times 10$ AND neurons plus 25 copies = 45; block 2 has $2 V = 10$ step neurons
plus 25 copies = 35; so 80 hidden neurons per wave, 4 waves, 8 hidden layers. The check runs all $2^10 = 1024$
graphs $times$ 5 start nodes = 5120 cases.

#code-sbs(
  "B",
  [```python
for k, (u, v) in enumerate(PAIRS):
    # both runtime: AND = relu(p + q - 1)
    reached[v] += relu(
        e[k] + frontier[u] - 1)
    reached[u] += relu(
        e[k] + frontier[v] - 1)
```],
  "C",
  [```python
# block 1: one row per edge direction
W1a[row, k] = 1           # e_k
W1a[row, FRO + src] = 1   # frontier_src
b1a[row] = -1
W2a[NX + dst, row] = 1    # reached_dst
```],
)

== What to notice

- *The graph structure is just weights.* A fixed graph is a constant matrix $A$ in $W_1$; to change the graph
  you change numbers, not code. A runtime graph costs one AND neuron per edge direction instead.
- *No queue, no `while`.* The frontier is a set; the number of waves is fixed by $V - 1$.
- *$K$ is a bound you derive,* not a magic number: max degree (fixed graph) or $V - 1$ (runtime graph).
- *Everything is $>= 0$,* which is what lets $relu(x) = x$ serve as a "wire". A negative state variable
  (like a dist of $-1$) would be clipped, so the $-1$ is only produced at the very end by an affine read-out.
- *Cost grows with the graph:* $O(V)$ neurons per layer for a fixed graph, $O(V^2)$ AND neurons for a runtime graph.

== Try it

*Exercise 1 (answer at the end of the chapter).* With $K = 1$ instead of $K = 3$, find a start node and a wave of the example graph where a visited node is
"discovered" again. Then find the start node that really needs $K = 3$.

#pagebreak()
// ═══════════════════════════════════════════════════════════════════════
#example("15", "Shortest paths: relaxation instead of a heap", "04_graph_search/15_shortest_path.py")

== The problem

Same question as BFS, but every edge has a *weight*: a small positive integer (1 to 3 here). Find the cheapest
total weight from the start to every node, and $-1$ if unreachable. BFS is no longer enough: a path with more
edges can be cheaper than one with fewer.

Inputs: one weight per possible undirected edge of 5 nodes (10 numbers in ${0, 1, 2, 3}$, where 0 means *no edge*)
and the start node as a one-hot vector. The graph is an *input* from the very start, not baked into the weights.

#let sp-pos = ((0, 0), (2.2, 0), (4.4, 0), (6.6, 0), (1.1, -1.8))
#let sp-edges = ((0, 1), (0, 4), (1, 4), (1, 2), (2, 3))
#let sp-weights = (1, 3, 1, 2, 1)
#let sp-dist = (0, 1, 3, 4, 2)
#align(center, draw-graph(
  sp-pos, sp-edges, sp-dist.map(d => layer-fill.at(d)),
  weights: sp-weights, scale: 0.9,
))
#align(center, tiny[The "trap" graph used below. Colour = final distance from node 0 (dark 0, blue 1, green 2, amber 3, pink 4).
  The direct edge $0 - 4$ costs 3, but going through node 1 costs $1 + 1 = 2$.])

== Two textbook algorithms

#block(breakable: false, {
  show raw: set text(size: 7pt)
  side-by-side(
    "A",
    [*Dijkstra* (this is `control_flow`)
```python
def dijkstra(weights, start):
    dist = [-1] * V
    heap = [(0, start)]
    while heap:
        d, u = heappop(heap)
        if dist[u] != -1:
            continue
        dist[u] = d
        for v in range(V):
            w = weight(u, v)
            if w > 0 and dist[v] == -1:
                heappush(heap, (d + w, v))
    return dist
```],
    "A",
    [*Bellman-Ford*
```python
def bellman_ford(weights, start):
    d = [inf] * V
    d[start] = 0
    for _ in range(V - 1):
        for u in range(V):
            for v in range(V):
                w = weight(u, v)
                if w > 0 and d[u] + w < d[v]:
                    d[v] = d[u] + w
    return [x if x != inf else -1
            for x in d]
```],
  )
})

#key-idea[
  Dijkstra's heap *pops one node per step* and the pop order depends on everything pushed so far: inherently
  sequential, and "pop the smallest" is a data-dependent branch. Bellman-Ford has no queue. Every round it relaxes
  *every edge*, with the same arithmetic on all of them, so a round is one layer of data flow. It does more work
  ($V - 1$ rounds of $E$ edges instead of $E log V$), but it is the only one of the two whose shape is a fixed
  repetition of the same formula.
]

Why $V - 1$ rounds? After round $r$, $d_v$ is at most the cost of the cheapest path that uses at most $r$ edges
(induction: a path of $r$ edges is a path of $r - 1$ edges plus one relaxed edge). A shortest path never repeats
a node, so it has at most $V - 1$ edges.

== The relaxation layer

$ d'_v = min( d_v, thin min_(u != v) (d_u + c_(u v)) ) $

where $c_(u v)$ is the weight of the edge, or INF when there is none. Every round uses the *old* $d$ for all
nodes at once (Bellman-Ford with all edges relaxed in parallel); A uses the in-place variant, but the final
answers are identical, and the check asserts that Dijkstra, Bellman-Ford, B and C all agree.

*Weights are inputs, and that costs nothing.* $d_u + c_(u v)$ is a plain addition. Unlike BFS part 2 there is no
AND and no `select`: two runtime values meet only in a sum, and sums are affine. The only non-linearity in the
whole algorithm is `min`.

*No edge = INF.* The input weight 0 means "no edge". Turn it into a cost that can never win a `min`:

$ c = w + "INF" dot (1 - step(w)) $

For $w in {1, 2, 3}$, $step(w) = 1$ and $c = w$. For $w = 0$, $step(0) = 0$ and $c = "INF"$. The start node gets
$d = 0$, every other node gets INF: $d = "INF" dot (1 - s)$ for the one-hot $s$.

*How big must INF be?* INF is "a finite stand-in for infinity". Two requirements:

#table(
  columns: (1fr, 1fr),
  align: left,
  [*Every real value is below INF.* A real distance estimate is the length of an actual walk of at most $V$ edges
   of weight $<= 3$, so at most $V dot 3 = 15$. Taking $"INF" = V dot "WMAX" + 1 = 16$ keeps every real value
   strictly below INF, so "$d >= "INF"$" can mean "unreachable".],
  [*Fake values never win.* A candidate built from a missing edge is $d_u + "INF" >= "INF"$. A candidate through an
   unreached node is $"INF" + c >= "INF" + 1$. Either way it is not smaller than $d_v <= "INF"$, so the `min`
   ignores it. An unreachable node stays at exactly INF.],
)

Too small an INF breaks things silently (see "Try it"). Too big is harmless apart from larger numbers
(the biggest value that ever appears is $2 dot "INF" = 32$).

*Final read-out.* Unreachable means $d = "INF"$, so $"reach" = step("INF" - d)$ is 1 for real distances and 0
for INF, and

$ "out" = d - ("INF" + 1)(1 - "reach") $

leaves real distances alone and maps INF to $-1$ (again an affine read-out, no negative number ever sits in the state).

== `min` of many things: a tournament

The toolbox gives `min` of two: $min(a, b) = a - relu(a - b)$. Node $v$ has $V = 5$ candidates ($d_v$ and four
$d_u + c_(u v)$). Pair neighbours, take the min of each pair, then pair the winners:
$5 arrow.r 3 arrow.r 2 arrow.r 1$, so *3 hidden layers per round* ($ceil.l log_2 5 ceil.r = 3$). An odd one out just moves up
a level unchanged. In stage B this is `min_tree`:

```python
def min_tree(xs):
    xs = list(xs)
    while len(xs) > 1:     # the length is fixed: a fixed loop
        paired = [xs[i] - relu(xs[i] - xs[i + 1]) for i in range(0, len(xs) - 1, 2)]
        xs = paired + xs[len(xs) // 2 * 2:]    # the odd one out moves up
    return xs[0]
```

#let min-tree(levels, labels: none, r: 0.42, gap: 1.6) = {
  let xs = (range(levels.at(0).len()).map(i => i * gap),)
  for k in range(levels.len() - 1) {
    let prev = xs.at(k)
    let nxt = ()
    for i in range(0, prev.len() - 1, step: 2) { nxt.push((prev.at(i) + prev.at(i + 1)) / 2) }
    if calc.odd(prev.len()) { nxt.push(prev.last()) }
    xs.push(nxt)
  }
  cetz.canvas({
    import cetz.draw: *
    for k in range(1, levels.len()) {
      for i in range(levels.at(k).len()) {
        let p = (xs.at(k).at(i), -k * 1.3)
        for c in (2 * i, 2 * i + 1) {
          if c < levels.at(k - 1).len() { line((xs.at(k - 1).at(c), -(k - 1) * 1.3), p, stroke: 0.9pt + muted) }
        }
      }
    }
    for (k, lv) in levels.enumerate() {
      for (i, val) in lv.enumerate() {
        let p = (xs.at(k).at(i), -k * 1.3)
        circle(p, radius: r, fill: if k == levels.len() - 1 { tint(network-color) } else { white }, stroke: 0.9pt + ink)
        content(p, text(size: 8.5pt)[#val])
      }
    }
    if labels != none {
      for (i, l) in labels.enumerate() {
        content((xs.at(0).at(i), 0.85), text(size: 7.5pt, fill: muted)[#l])
      }
    }
  })
}

#align(center, min-tree(
  ((3, 3, 2, 32, 32), (3, 2, 32), (2, 32), (2,)),
  labels: ($d_4$, $d_0 + c_(0 4)$, $d_1 + c_(1 4)$, $d_2 + c_(2 4)$, $d_3 + c_(3 4)$),
))
#align(center, tiny[Node 4 in round 2 of the trap graph (INF = 16): $d = [0, 1, 16, 16, 3]$ and
  $c_(0 4) = 3$, $c_(1 4) = 1$, $c_(2 4) = c_(3 4) = 16$. Candidates $3, 3, 2, 32, 32$; the tournament returns 2.])

One `min(a, b)` is three neurons' worth of wiring: a copy of $a$, the neuron $relu(a - b)$, and an output that subtracts:

#align(center, nn(
  (($a$, $b$), ($relu(a)$, $relu(a - b)$), ($min(a, b)$,)),
  (
    (0, 0, 1, 0, $1$, plus),
    (0, 0, 1, 1, $1$, plus), (0, 1, 1, 1, $-1$, minus),
    (1, 0, 2, 0, $+1$, plus), (1, 1, 2, 0, $-1$, minus),
  ),
  r: 0.6, font: 7.5pt, gapx: 3.8,
))
#align(center, tiny[$relu(a) = a$ because every number in the state is $>= 0$.])

== A and B side by side

#code-sbs(
  "B",
  [```python
def branchless(weights, start):
    cost = [w + INF * (1 - step(w))
            for w in weights]
    d = [INF * (1 - s) for s in start]
    for _ in range(V - 1):
        d = [min_tree(
                [d[v]] + [d[u] + cost[idx[u, v]]
                          for u in range(V)
                          if u != v])
             for v in range(V)]
    return [d[v] - (INF + 1)
                 * (1 - step(INF - d[v]))
            for v in range(V)]
```],
  "C",
  [```python
LAYERS = [pre_layer()] \
       + ROUND * (V - 1) \
       + [post_layer()]

def network(weights, start):
    x = np.concatenate([weights, start])
    for layer in LAYERS:
        x = layer(x)   # W2 relu(W1 x + b1) + b2
    return x.astype(int).tolist()
```],
)

(`if u != v` in B compares two *indices*, which are constants fixed by the graph's shape, not runtime data.)

== Worked trace: the trap graph from node 0

Cost vector ($c$ in the order of the pairs $01, 02, 03, 04, 12, 13, 14, 23, 24, 34$), from weights
$[1, 0, 0, 3, 2, 0, 1, 1, 0, 0]$:

$ c = [1, "INF", "INF", 3, 2, "INF", 1, 1, "INF", "INF"] quad ("INF" = 16) $

#table(
  columns: (auto, auto, auto, auto, auto, auto, 1fr),
  align: (center,) * 6 + (left,),
  table.header([*round*], [*$d_0$*], [*$d_1$*], [*$d_2$*], [*$d_3$*], [*$d_4$*], [*what happened*]),
  [0], [0], [16], [16], [16], [16], [start: INF everywhere except the start],
  [1], [0], [1], [16], [16], [3], [neighbours of 0: $d_1 = 0 + 1$, $d_4 = 0 + 3$],
  [2], [0], [1], [3], [16], [2], [$d_2 = 1 + 2$; $d_4 = min(3, d_1 + 1) = 2$: the trap is *fixed*],
  [3], [0], [1], [3], [4], [2], [$d_3 = d_2 + 1$],
  [4], [0], [1], [3], [4], [2], [nothing changes (spare round)],
)

Dijkstra settles the same nodes in cost order 0, 1, 4, 2, 3 using the heap: pop $(0, 0)$; push $(1, 1)$ and $(3, 4)$; pop
$(1, 1)$; push $(2, 4)$ and $(3, 2)$; pop $(2, 4)$ and so on. Bellman-Ford needed no such bookkeeping: it just
improved $d_4$ from 3 to 2 when round 2 saw a better candidate. That is the whole difference: *a better answer
overwrites a worse one* instead of *deciding the order in which answers are final*.

Round 3 repeats the same formula on the same numbers, and so does round 4: when nothing improves, nothing changes.
So, like BFS, *halting is free*: the extra rounds are harmless.

== Stage C: the layers

Every layer is $x' = W_2 relu(W_1 x + b_1) + b_2$, assembled by a small `Layer` class that fills in
constant matrices at import time (no computation on runtime values). Copying a number through a layer uses
$relu(x) = x$, valid because every state number is $>= 0$.

#align(center, diagram(
  node-stroke: 0.7pt, node-inset: 6pt, node-corner-radius: 3pt, spacing: (9mm, 7mm),
  node((0, 0), [input\ $[w | s]$]),
  node((1, 0), [*pre*\ $c$, $d_0$], fill: tint(network-color)),
  node((2, 0), [level 1\ $5 arrow.r 3$], fill: tint(branchless-color)),
  node((3, 0), [level 2\ $3 arrow.r 2$], fill: tint(branchless-color)),
  node((4, 0), [level 3\ $2 arrow.r 1$], fill: tint(branchless-color)),
  node((5, 0), [*post*\ $-1$ read-out], fill: tint(network-color)),
  node((6, 0), [output\ dist]),
  edge((0, 0), (1, 0), "->"), edge((1, 0), (2, 0), "->"), edge((2, 0), (3, 0), "->"),
  edge((3, 0), (4, 0), "->"), edge((4, 0), (5, 0), "->"), edge((5, 0), (6, 0), "->"),
  edge((4, 0), (2, 0), "->", bend: 55deg, label: [repeat the 3 levels $V - 1 = 4$ times], label-side: left),
))

#table(
  columns: (auto, 1fr, auto),
  align: (left, left, right),
  table.header([*layer*], [*hidden neurons and what they compute*], [*count*]),
  [pre], [$relu(w_k)$, $relu(w_k - 1)$ for each of the 10 weights (gives $c_k = (1 - "INF") relu(w_k) + "INF" relu(w_k - 1) + "INF"$),
    and a copy of each start bit], [25],
  [level 1], [for each node: 2 pairs $times$ ($relu(a)$, $relu(a - b)$) + the odd candidate = 5; plus copies of the 10 costs], [$5 times 5 + 10 = 35$],
  [level 2], [for each node: 1 pair (2 neurons) + the odd one = 3; plus 10 copies], [$5 times 3 + 10 = 25$],
  [level 3], [for each node: 1 pair = 2 neurons; plus 10 copies], [$5 times 2 + 10 = 20$],
  [post], [$relu("INF" - d)$, $relu("INF" - d - 1)$, and a copy of $d$, per node], [15],
)

The first level's candidates are *affine functions of the state* $[d | c]$: candidate $d_u + c_(u v)$ is the row
vector with a 1 at $d_u$ and a 1 at $c_(u v)$. So adding the weight and taking the first `min` happen in the same
layer: the matrix $W_1$ of level 1 is built from rows like "$(d_u + c_(u v)) - d_v$".

Total: $25 + 4 times (35 + 25 + 20) + 15 = 360$ hidden neurons in 14 layers. The test compares
Dijkstra, Bellman-Ford, B and C on 25 hand-picked (weights, start) cases (no edges, all weight 1, a path of weight-3
edges whose longest distance is 12, the trap, all edges weight 3) and on 2000 random weight vectors of three densities.

== What to notice

- *Pick the algorithm whose shape is data flow.* Dijkstra and Bellman-Ford compute the same thing; only one of them is a
  repeated formula on all nodes. When you design your own algorithm, look for a "relax everything" formulation first.
- *`min` is the only non-linearity.* Sums are free; so are the weights, even though they are runtime inputs.
- *INF is a bound you derive.* $"INF" > V dot "WMAX"$ is enough; the derivation is two inequalities.
- *Depth vs width of `min`.* A tournament costs $log_2 n$ layers and about $2n$ neurons per node. The "compare every pair"
  approach (used by `argmin` in ch. 3 and in A\* below) costs 2 layers but about $n^2$ neurons. You choose a point on that curve.
- *Negative weights* would still work for Bellman-Ford in control flow, but this network keeps every state value
  $>= 0$ so that copies are free; you would have to offset the distances to keep that true.

== Try it

*Exercise 2 (answer at the end of the chapter).* The weights grow to $"WMAX" = 5$. What are INF and the number of rounds? Then: what would go wrong with
$"INF" = 10$ if the weights form a path $0 - 1 - 2 - 3 - 4$ with all weights 3?

#pagebreak()

// ═══════════════════════════════════════════════════════════════════════
#example("16", "A*: the same sequential algorithm, one step per layer", "04_graph_search/16_astar.py")

== The problem

A 4 $times$ 4 grid. Nodes are numbered row by row, 0 to 15. The *start* is node 0 (top-left), the *goal* is node 3
(top-right). Moves go up, down, left, right at cost 1. Each of the other cells is a runtime input bit, 1 = wall. (A wall bit
on the start is ignored, since the start is expanded first; a wall on the goal makes it unreachable.) A\* finds a cheapest path
using a heuristic $h_v$ = Manhattan distance to the goal, which is a *constant vector* because the goal is fixed:

#let grid-fig(cells, scale: 0.8, path: none) = cetz.canvas(length: scale * 1cm, {
  import cetz.draw: *
  for r in range(4) {
    for c in range(4) {
      rect((c, -r), (c + 1, -r - 1), fill: cells.at(r * 4 + c).fill, stroke: 0.6pt + muted)
    }
  }
  if path != none {
    line(..path.map(i => (calc.rem(i, 4) + 0.5, -calc.floor(i / 4) - 0.5)), stroke: 2.4pt + network-color.transparentize(35%))
  }
  for r in range(4) {
    for c in range(4) {
      let cell = cells.at(r * 4 + c)
      if cell.label != none {
        content((c + 0.5, -r - 0.5), text(size: 9pt, weight: "bold", fill: if cell.fill == ink { white } else { ink })[#cell.label])
      }
      if cell.at("sub", default: none) != none {
        content((c + 0.78, -r - 0.8), text(size: 6.5pt, fill: muted)[#cell.sub])
      }
      if cell.at("mark", default: none) != none {
        content((c + 0.2, -r - 0.22), text(size: 7pt, weight: "bold", fill: control-color)[#cell.mark])
      }
    }
  }
})
#let H-vals = (3, 2, 1, 0, 4, 3, 2, 1, 5, 4, 3, 2, 6, 5, 4, 3)
#let mark-of(i) = if i == 0 { "S" } else if i == 3 { "G" } else { none }
#let grid-numbers = range(16).map(i => (fill: white, label: [#i], sub: [h=#H-vals.at(i)], mark: mark-of(i)))
#let grid-run(walls, order, path-nodes: none, scale: 0.8) = {
  let cells = range(16).map(i => {
    let k = order.position(x => x == i)
    if walls.at(i) == 1 { (fill: ink, label: none, mark: none) }
    else if k != none { (fill: branchless-color.lighten(92% - k * 4%), label: [#(k + 1)], mark: mark-of(i)) }
    else { (fill: white, label: none, mark: mark-of(i)) }
  })
  grid-fig(cells, scale: scale, path: path-nodes)
}

#align(center, grid(
  columns: 2, gutter: 14mm,
  grid-fig(grid-numbers, scale: 1.0),
  align(horizon, tiny[Node numbers and $h_v$. $h_v = |"row"_v - 0| + |"col"_v - 3|$, so $h$ is 0 at the goal and 6 at the bottom-left corner.
    $g$ = cost of the best route found so far, $f = g + h$.]),
))

The result is a triple, identical in all three stages: *(cost or $-1$, the order in which nodes were expanded, the path from goal to start)*.
Lists are padded with $-1$ up to 16 entries so that A, B and C can be compared with `==`.

#let walls-empty = (0,) * 16
#let walls-trap = (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0)
#let walls-maze = (0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0)
#align(center, grid(
  columns: 3, gutter: 10mm, row-gutter: 5pt,
  grid-run(walls-empty, (0, 1, 2, 3), path-nodes: (0, 1, 2, 3), scale: 0.75),
  grid-run(walls-trap, (0, 1, 4, 5, 8, 9, 12, 13, 14, 15, 11, 7, 3), path-nodes: (0, 1, 5, 9, 13, 14, 15, 11, 7, 3), scale: 0.75),
  grid-run(walls-maze, (0, 4, 8, 9, 10, 6, 2, 3), path-nodes: (0, 4, 8, 9, 10, 6, 2, 3), scale: 0.75),
  tiny[no walls: A\* walks straight to the goal],
  tiny[a wall with a gap at the bottom: A\* first runs into the wall],
  tiny[a small maze],
))
#align(center, tiny[Numbers = order in which A\* expanded each cell (1 = the start). Green line = the returned path.
  Blank cells were never expanded.])

== What stays sequential, and what does not

A\* is *inherently sequential*: each expansion depends on the previous one (which node has the best $f$ right now?).
We do not fight that. The control-flow algorithm does at most one expansion per loop trip and there are at
most 16 nodes, so the data-flow version is *16 identical steps*, each step one expansion. Inside one step everything is parallel
over nodes: all keys at once, all neighbours at once.

What has to be replaced:

#table(
  columns: (1fr, 1fr),
  align: left,
  table.header([*control flow*], [*data flow*]),
  [the heap: "pop the entry with the smallest `(f, node)`"], [keys for all nodes, an `argmin` one-hot ($sigma$)],
  [`closed` set, `g` dict, `parent` dict], [vectors: $"closed"$ (0/1), $g$ (integers), $P$: one one-hot row per node],
  [`if u == GOAL: break`, `while heap:` ends], [a `done` flag that freezes every later step],
  [`order.append(u)`], [a *log*: the one-hot chosen at step $t$ is row $t$ of a $16 times 16$ matrix],
  [`for n in NEIGH[u]` with ifs], [a constant adjacency matrix times $sigma$, then 0/1 masks],
  [stale duplicates in the heap, `if u in closed: continue`], [nothing: there are no duplicates, one key per node],
)

== One A\* step in six layers

#align(center, diagram(
  node-stroke: 0.7pt, node-inset: 5pt, node-corner-radius: 3pt, spacing: (4mm, 7mm), node-fill: tint(branchless-color),
  node((0, 0), [*L1*\ keys, ranks\ reached]),
  node((1, 0), [*L2*\ $sigma$\ (halt-aware)]),
  node((2, 0), [*L3*\ $g_(sigma)$\ candidates]),
  node((3, 0), [*L4*\ $step(g_n - g_(sigma) - 1)$]),
  node((4, 0), [*L5*\ better\ $= AND$]),
  node((5, 0), [*L6*\ update $g$, $P$,\ closed, done, log], fill: tint(network-color)),
  edge((0, 0), (1, 0), "->"), edge((1, 0), (2, 0), "->"), edge((2, 0), (3, 0), "->"),
  edge((3, 0), (4, 0), "->"), edge((4, 0), (5, 0), "->"),
  edge((5, 0), (0, 0), "->", bend: 50deg, label: [repeat 16 times], label-side: left),
))

#let step-note(n, title, body) = block(breakable: false, inset: (left: 10pt, y: 3pt), stroke: (left: 2pt + branchless-color), [*#n. #eval(title, mode: "markup")* #body])

Stage B, line by line (C is the same arithmetic as matrices). Notation: $N = 16$ nodes, $v, u, n$ are node indices.

#step-note(1, "Key and rank (argmin).")[
  $ k_v = N dot (g_v + h_v + "BIG" dot "closed"_v) + v, quad "rank"_v = sum_(u != v) step(k_v - k_u) $
  $"BIG"$ pushes closed nodes (and unreached ones, whose $g = "BIG"$) far away. This is _affine_: $"closed"_v$ is a 0/1 number
  multiplied by the *constant* BIG, so no `select` is needed. Adding $v$ to the key makes all keys *different*, so
  ties are gone: the lowest index wins automatically, exactly like the heap's `(f, node)` tuples.
  With unique keys, $step(k_v - k_u)$ is exactly "$u$ has a smaller key than $v$", and $"rank"_v$ counts the nodes that beat $v$.
  Rank 0 means "I am the minimum". Cost: two neurons per ordered pair, $2 N (N - 1) = 480$.
]

*Bounds.* A path never repeats a node, so $g <= 15$; $h <= 6$; so every real $f = g + h <= 21$. We need an open node to always beat
closed and unreached ones: $"BIG" = 15 + 6 + 1 = 22$. Then $k$ stays below $16 dot 43 + 15 = 703$ and all differences are small integers,
so `step` is exact.

#table(
  columns: (auto, auto, 1fr),
  align: left,
  table.header([*node is*], [*$g + h + "BIG" dot "closed"$*], [*compared with the open nodes*]),
  [open], [$<= 21$], [it is the competition],
  [closed], [$>= 22$], [always loses],
  [unreached ($g = "BIG"$)], [$>= 22$], [always loses],
)

#step-note(2, "Select, with the halt flag.")[
  $ "open"_v = step("BIG" - g_v) - "closed"_v, quad
    sigma_v = relu("open"_v - "done" - "rank"_v) $
  $step("BIG" - g_v)$ is 1 for nodes that have been reached ($g < "BIG"$). Then $sigma_v = 1$ exactly when *all three* hold:
  rank 0, open, and `done` is 0. Why one ReLU is enough: $relu("open" - "done" - "rank")$. If the rank is 0 this is $relu("open" - "done")$, which is 1
  only for open = 1 and done = 0 (an AND). If the rank is 1 or more, then $"open" - "done" - "rank" <= 1 - 0 - 1 = 0$, so the output is 0.
]

Why this one line is the whole halting mechanism:

- *Nothing is open* (the goal is walled in): the argmin picks some closed or unreached node, but its $"open"$ is 0, so $sigma = 0$ everywhere.
- *Already done*: $"done" = 1$ turns every $sigma_v$ off.
- Whenever $sigma = 0$, every later formula (closing, relaxing, logging) is applied with a zero vector: *the state is frozen.*
  That is the same "halted flag freezes the state" idea as chapter 2, but here it costs no extra layers.

#step-note(3, "Update `done`.")[
  $ "done" arrow.l 1 - sum_v sigma_v + sigma_("goal") $
]

#table(
  columns: (auto, auto, auto, 1fr),
  align: center,
  table.header([*situation*], [*$sum sigma$*], [*$sigma_"goal"$*], [*new done*]),
  [already done], [0], [0], [$1 - 0 + 0 = 1$],
  [nothing open], [0], [0], [1],
  [expanded a non-goal node], [1], [0], [$1 - 1 + 0 = 0$],
  [expanded the goal], [1], [1], [$1 - 1 + 1 = 1$],
)

#step-note(4, "Read $g$ of the selected node.")[
  $ g_(sigma) = sum_v relu(g_v - "BIG" dot (1 - sigma_v)) $
  This is chapter 3's masked read (`xs[i]` with a one-hot `i`): the neuron for node $v$ passes $g_v$ only when $sigma_v = 1$.
  Because $g_v >= 0$ one ReLU suffices, with $M = "BIG"$ (the largest $g_v$).
]

#step-note(5, "Neighbours of the selected node, and which may be relaxed.")[
  $ "near" = A_("grid") thin sigma, quad "cand"_n = relu("near"_n - "wall"_n - "closed"_n) $
  $A_"grid"$ is the constant 0/1 adjacency matrix of the grid (the same trick as BFS), with the goal's column zeroed so the goal
  never expands. $"cand"_n = 1$ exactly when $n$ touches the selected node, is not a wall and is not closed
  (three 0/1 conditions: subtract the "bad" ones from the "good" one and clip).
]

#step-note(6, "Did we find a better route?")[
  $ "better"_n = relu("cand"_n + step(g_n - g_(sigma) - 1) - 1) $
  $g_(sigma) + 1 < g_n$ is $g_n - g_(sigma) - 1 >= 1$ (integers), and $relu(p + q - 1)$ is the AND. (Unreached nodes have $g_n = "BIG" >= 22 > 16 >= g_(sigma) + 1$.)
]

#step-note(7, "Overwrite $g$ and the parent with big-M selects.")[
  $ g'_n = g_n + sel_M (b_n, thin g_(sigma) + 1 - g_n), quad
    P'_(n u) = P_(n u) + sel_M (b_n, thin sigma_u - P_(n u)) $
  with $b_n = "better"_n$, $M = "BIG"$ for $g$ and $M = 1$ for $P$, and $sel_M (m, x) = relu(x - M(1 - m)) - relu(-x - M(1 - m))$ from the toolbox.
  Here the 0/1 mask $b_n$ multiplies a *runtime* difference, which is exactly what `select` is for.
  Bounds: $1 <= g_(sigma) + 1 <= 16$ and $0 <= g_n <= 22$, so $|x| <= 22 = M$ for $g$; and $|sigma_u - P_(n u)| <= 1 = M$ for the parent bits.
  The parent register $P$ has one one-hot row per node: row $n$ says which node is the parent of $n$ (all zeros = none yet).
]

#align(center, nn(
  (($m$, $x$), ($R_1$, $R_2$), ($"out"$,)),
  (
    (0, 0, 1, 0, $M$, plus, 30%), (0, 1, 1, 0, $1$, plus, 70%),
    (0, 0, 1, 1, $M$, plus, 70%), (0, 1, 1, 1, $-1$, minus, 30%),
    (1, 0, 2, 0, $+1$, plus), (1, 1, 2, 0, $-1$, minus),
  ),
  r: 0.6, font: 7.5pt, gapx: 3.8,
))
#align(center, tiny[Big-M select: $R_1 = relu(x + M m - M)$ and $R_2 = relu(-x + M m - M)$. Used 16 times for $g$ and 256 times for $P$ per step.])

#step-note(8, "Close, log.")[
  $ "closed"' = "closed" + sigma, quad "log"' = ["rows 2 to 16 of log", sigma] $
  The log is a *shift register*: drop the oldest row, append $sigma$ as the newest. After 16 steps, row $t$ is the node expanded at step $t$
  (an all-zero row if the search had already stopped). Shifting keeps all 16 steps identical, no layer needs to know its step number.
]

== Path reconstruction

After the search, follow the parent pointers from the goal. The pointer chase is itself a fixed loop (a path has at most 16 nodes):

$ "cur"_0 = "found" dot e_("goal"), quad "cur"_(t+1, u) = sum_v AND(P_(v u), "cur"_(t, v)) = sum_v relu(P_(v u) + "cur"_(t, v) - 1) $

where $"found" = "closed"_("goal")$ (0/1) times the *constant* one-hot $e_"goal"$. $"cur"$ is a one-hot "where am I"; the sum over $v$
is "parent of the node I am on", an array read with the one-hot as address, done with one AND neuron per $(v, u)$ pair, 256 per step.
The start has no parent (its row of $P$ is zero), so $"cur"$ becomes the zero vector and stays there: padding for free. The 16 $"cur"$ vectors
are logged with the same shift register. The *cost* is read as $g_("goal") - ("BIG" + 1)(1 - "found")$, which is $-1$ if the goal was never
expanded (then $g_"goal" = "BIG"$). In the network it is stored as cost $+ 1$ so that it stays $>= 0$.

== Worked trace: the wall with a gap

Grid: walls at 2, 6, 10 (the third column, top three rows). Start 0, goal 3 is right behind the wall. The first steps of A\* (this is
what each of the 16 layers computes; open nodes are written `node: g + h = f`):

#table(
  columns: (auto, 1fr, auto, 1fr),
  align: (center, left, center, left),
  table.header([*step*], [*open set before the step*], [*selected*], [*relaxed (new $g$)*]),
  [1], mono[0: 0+3=3], [0], mono[1: 1+2=3, 4: 1+4=5],
  [2], mono[1: 3, 4: 5], [1], mono[5: 2+3=5],
  [3], mono[4: 5, 5: 5], [4 (tie)], mono[8: 2+5=7],
  [4], mono[5: 5, 8: 7], [5], mono[9: 3+4=7],
  [5], mono[8: 7, 9: 7], [8 (tie)], mono[12: 3+6=9],
  [6], mono[9: 7, 12: 9], [9], mono[13: 4+5=9],
  [7], mono[12: 9, 13: 9], [12 (tie)], [none: 13 already has $g = 4$],
  [8], mono[13: 9], [13], mono[14: 5+4=9],
  [9 to 12], [one open node each time, $f = 9$], [14, 15, 11, 7], [15 (g 6), 11 (g 7), 7 (g 8), 3 (g 9)],
  [13], mono[3: 9], [3 = goal], [no relax; `done` becomes 1],
  [14 to 16], [(frozen)], [none], [nothing: $sigma = 0$],
)

A\* follows the heuristic straight into the wall: cells 0, 1, 4, 5, 8, 9 are cheap ($f$ = 3 to 7). Then the $f = 9$ plateau opens: every
remaining step has the same $f$, and ties go to the lowest index (12 before 13). It then walks the long way round the bottom of the wall and up. The returned
cost is 9; the path is $3, 7, 11, 15, 14, 13, 9, 5, 1, 0$ goal to start.

Zoom in on step 3, the first tie. Open: node 4 ($g = 1, h = 4$) and node 5 ($g = 2$, $h = 3$), both $f = 5$. The keys are
$k_4 = 16 dot 5 + 4 = 84$ and $k_5 = 16 dot 5 + 5 = 85$; everything else is above 350 (closed nodes carry $+ 22$ inside the
bracket, unreached ones have $g = 22$). So $"rank"_4 = 0$ and $"rank"_5 = 1$. Then $sigma_4 = relu(1 - 0 - 0 - 0) = 1$ and
$sigma_5 = relu(1 - 0 - 0 - 1) = 0$: node 4 is selected, same as the heap's `(5, 4) < (5, 5)`.

== Stage C: the network

The whole state is one vector of 658 numbers, all $>= 0$ (so copies are free, as before). *Persistent* registers come first, *scratch* registers (reset to 0
every step) after them:

#table(
  columns: (auto, auto, 1fr),
  align: (left, right, left),
  table.header([*register*], [*size*], [*role*]),
  [$g$], [16], [best cost so far; BIG = "not reached"],
  [closed], [16], [0/1],
  [done], [1], [0/1 halt flag],
  [wall], [16], [the runtime input, carried along],
  [$P$], [256], [parent one-hot per node],
  [log], [256], [expansion order, shift register],
  [scratch: rank, reached, $sigma$, $g_sigma$, cand, stp, better], [97], [intermediate values between the six layers],
)

Each layer is $x' = W_2 relu(W_1 x + b_1) + b_2$ and is built at import time, one hidden neuron per formula above, by the same small
`Layer` class as in example 15. For instance $sigma_v = relu("reached"_v - "closed"_v - "done" - "rank"_v)$ is one row of $W_1$ in layer L2:

#align(center, nn(
  (($"reached"_v$, $"closed"_v$, $"done"$, $"rank"_v$), ($relu$,), ($sigma_v$,)),
  (
    (0, 0, 1, 0, $1$, plus), (0, 1, 1, 0, $-1$, minus), (0, 2, 1, 0, $-1$, minus), (0, 3, 1, 0, $-1$, minus),
    (1, 0, 2, 0, $1$, plus),
  ),
  r: 0.55, font: 7pt, gapx: 4.0, gapy: 1.1,
))

```python
# layer L2, as written in the file: sel[v] = relu(reached - closed - done - rank)
for v in range(N):
    row = row_of(REACHED=(v, 1), CLOSED=(v, -1), DONE=(0, -1), RANK=(v, -1))
    L2.output(at("SEL", v), [(L2.unit(row), 1)])
```

Neuron counts (hidden neurons per layer): L1 1073 (480 pairwise + 32 reached + 561 copies), L2 577, L3 609, L4 626, L5 594, L6 1104 (with the 16 $g$ selects, 256 $P$
selects, and copies). One step is 4583 neurons. Most of them are *copies* of the persistent registers ($P$ and the log are 512 numbers carried through
every layer); a residual connection would remove them, but then the layer would no longer be a plain "matrix, ReLU, matrix". With 16 steps and the 16-layer path walk
(1025 neurons per layer): about $16 times 4583 + 16 times 1025 approx 90 000$ neurons in $16 times 6 + 16 = 112$ layers.

#code-sbs(
  "A",
  [```python
f, u = heappop(heap)
if u in closed:
    continue
closed.add(u)
order.append(u)
if u == GOAL:
    break
```],
  "B",
  [```python
key = [N * (g[v] + H[v]
            + BIG * closed[v]) + v
       for v in range(N)]
rank = [sum(step(key[v] - key[u])
            for u in range(N) if u != v)
        for v in range(N)]
sel = [relu(reached[v] - closed[v]
            - done - rank[v])
       for v in range(N)]
```],
)
#code-sbs(
  "A",
  [```python
for n in NEIGH[u]:
    if walls[n] or n in closed:
        continue
    if g[u] + 1 < g.get(n, BIG):
        g[n] = g[u] + 1
        parent[n] = u
        heappush(heap,
                 (g[n] + H[n], n))
```],
  "B",
  [```python
near = [sum(sel[u] for u in NEIGH[n]
            if u != GOAL) for n in ...]
cand = [relu(near[n] - walls[n]
             - closed[n]) for n in ...]
better = [relu(cand[n]
    + step(g[n] - g_sel - 1) - 1)
    for n in range(N)]
g[n] += select(better[n],
               g_sel + 1 - g[n], BIG)
P[n][u] += select(better[n],
                  sel[u] - P[n][u], 1)
```],
)

The check runs 1006 wall layouts: six hand-picked (no walls, the wall with a gap, goal walled in, start walled in, the goal itself a wall, a small maze) and 1000 random
ones with wall probability 0.1 to 0.4. For every one, A, B and C agree on cost, expansion order and path, and the cost also equals plain BFS's.

== What to notice

- *A sequential algorithm becomes a fixed number of identical layers.* The price is a step count you must bound ahead of time (16 expansions at most), and
  the fact that the layers after the end still exist (frozen by `sel = 0`).
- *Pick-the-best is `argmin`, and the tie-break is part of the key.* $N dot "key" + "index"$ makes keys unique; any tie rule is just a different low-order term.
- *"Push it far away" is affine when a 0/1 bit multiplies a constant;* `select` is only needed when a 0/1 bit multiplies a runtime value.
- *Every subtraction of a bad flag is a hidden AND/NOT.* $relu("near" - "wall" - "closed")$ is one neuron instead of three.
- *Derive BIG and every $M$ from the bounds on $g$ and $h$;* write them next to the code. They are the first thing to check when a test fails.
- *Consistent heuristic:* Manhattan distance with unit moves never needs a closed node to be reopened, which is why the `n in closed` filter is
  safe. With an inconsistent $h$ you would remove that filter (the `better` test alone then decides).
- *Making the goal a runtime input* only changes $h$: read the goal's row and column with a one-hot and compute $|r_v - r_g| + |c_v - c_g|$
  with $abs(x) = relu(x) + relu(-x)$, about 4 more neurons per node. Everything else stays.

== Try it

*Exercise 3 (answer at the end of the chapter).* Change the tie-break to prefer the *highest* node index among equal $f$. Which places in stage A and in the
key formula change? And what changes if you set $h = 0$ everywhere (and what can BIG shrink to)?

#pagebreak()

// ═══════════════════════════════════════════════════════════════════════
= A recipe for your own search algorithm

Every algorithm in this chapter fits one template. To design BFS, Dijkstra, greedy best-first, a maze solver or anything similar by hand:

#let recipe(n, title, body) = block(breakable: false, below: 8pt, grid(
  columns: (22pt, 1fr), gutter: 6pt,
  box(width: 20pt, height: 20pt, radius: 10pt, fill: ink, align(center + horizon, text(fill: white, size: 9pt, weight: "bold")[#n])),
  [*#title* #body],
))

#recipe(1, "Choose the state vectors.")[
  One entry per node for every fact the algorithm remembers: `visited`/`closed` (0/1), `frontier`/`open` (0/1), a cost (`dist`, `g`), a
  parent (one-hot per node), a log of what happened in order. *Keep every number $>= 0$ and bounded,* store offsets
  if you must ("cost + 1"), and write the bounds down: they give you $K$, INF, BIG and every $M$.
]

#recipe(2, "Choose the select rule: what does a layer process?")[
  This is the only thing that really differs between algorithms.
  #table(
    columns: (auto, auto, 1fr),
    align: left,
    table.header([*algorithm*], [*selected set*], [*how it is built*]),
    [BFS], [the whole frontier], [the frontier vector itself; no search needed],
    [Bellman-Ford], [every edge], [nothing: relax everything, `min` decides],
    [Dijkstra], [one node: smallest $g$], [argmin one-hot, key $= g$ (+ BIG on closed)],
    [A\*], [one node: smallest $f$], [argmin one-hot, key $= g + h$],
    [greedy best-first], [one node: smallest $h$], [argmin one-hot, key $= h$],
    [DFS], [one node: newest open], [argmax of "discovery time" (argmin of its negative)],
  )
  For a single node, build keys, make them unique with $N dot "key" + "index"$, and use $"rank"_v = sum_(u != v) step(k_v - k_u)$,
  $sigma_v = relu(1 - "rank"_v)$ (2 layers, $O(N^2)$ neurons). For a whole set, skip all that.
]

#recipe(3, "Expand along the edges.")[
  Fixed graph: $"near" = A thin sigma$ with the adjacency matrix as weights. Graph as input: one AND neuron
  $relu(e_(u v) + sigma_u - 1)$ per edge direction. Then *filter*: for each node, subtract the 0/1 flags that forbid it
  ($relu("near" - "wall" - "closed")$), or use $AND = relu(p + q - 1)$ for two flags.
]

#recipe(4, "Update the state with select.")[
  A cost update is `new = old if not better else candidate`: compute $"better" = step("old" - "candidate")$ and
  $"old" + sel_M ("better", "candidate" - "old")$ with $M$ from the bounds. Parents are the same select on one-hot rows. Visited-style
  flags are just $+ sigma$ or $+ "new"$.
]

#recipe(5, "Add the halt flag.")[
  Put $-"done"$ inside the ReLU that builds $sigma$, so a finished search selects nothing, and set
  $"done" arrow.l 1 - sum sigma + sigma_("goal")$. A zero selection leaves the whole state unchanged, so extra layers are harmless
  and you only need an *upper bound* on the number of steps. (For BFS and Bellman-Ford the process dies out on its own.)
]

#recipe(6, "Count layers and neurons.")[
  Steps $times$ layers per step (2 to 6) $times$ neurons per layer. The pairwise `argmin` is $2N(N - 1)$; an adjacency matrix is $N$
  neurons per layer for a fixed graph and $O(N^2)$ AND neurons for a runtime graph; copying a register is one neuron per entry.
]

#recipe(7, "Read out with affine maps, and test.")[
  Decode one-hots with $"onehot" dot (1, ..., N) - 1$ so that "none" is $-1$; map "never reached" markers (INF, BIG) to $-1$ with an affine
  correction, never a comparison. Then do what every example here does: write the plain Python version first, compare against
  it on *every* input you can afford and on a fixed-seed random sample plus the awkward cases (no path, blocked start, ties).
]

Skeleton of one step, in the notation of this chapter:

```python
state = initial(inputs)                          # affine in the inputs
for _ in range(MAX_STEPS):                       # fixed count
    key    = affine(state)                       # push excluded nodes to BIG
    sel    = relu(open - done - rank(key))       # select-best AND not done
    near   = ADJ @ sel                           # expand: constant matrix
    cand   = relu(near - forbidden)              # filter: 0/1 flags
    better = relu(cand + step(old - new) - 1)    # improves?
    state  = update(state, sel, better)          # big-M selects, closed += sel
    done   = 1 - sum(sel) + sel[goal]            # halt flag
```

*Pitfalls checklist.*
- A state value that can go negative is clipped by a "copy" neuron. Offset it, or avoid carrying it through a layer.
- `step` is exact only for integers. Keep every cost an integer.
- Ties silently change the answer. Make the tie-break part of the key and use the same one in stage A.
- `select` with a too-small $M$ gives wrong answers only on rare inputs. Derive $M$ from the bounds.
- Never multiply two runtime values. AND them (0/1 bits), select them (0/1 mask $times$ value), or add them.

= Answers to the exercises

*1 (BFS, $K = 1$).* Start node 0, wave 2. The frontier is ${1, 2}$, $A f = [2, 1, 1, 2, 0, ...]$ and $"visited" = [1, 1, 1, 0, ...]$.
With $K = 1$: $p = [1, 0, 0, 2, 0, ...]$, so $step(p) = [1, 0, 0, 1, 0, ...]$: node 0, already visited, is discovered again
(and `visited[0]` becomes 2). With $K = 3$: $p = [-1, -2, -2, 2, ...]$, correct. $K$ must be at least the largest number of frontier neighbours
a visited node can have. From start node 3, wave 2: the frontier is ${1, 2, 4}$, all three touch node 3, so $"reached"_3 = 3$ and
$K = 3$ (the maximum degree) is really needed; $K = 2$ would rediscover node 3.

*2 (shortest path).* $"INF" = V dot "WMAX" + 1 = 5 dot 5 + 1 = 26$; the rounds stay $V - 1 = 4$ (they depend on the number of nodes, not the weights). With
$"INF" = 10$ and a path of weight-3 edges, node 4 has the true distance $4 dot 3 = 12 >= 10$. Node 4 starts at $d = "INF" = 10$, and the
candidate $d_3 + 3 = 12$ is not smaller than 10, so the `min` never writes it: node 4 stays at INF and the read-out reports $-1$ (unreachable), although
a path exists. The bound $"INF" > V dot "WMAX"$ is what keeps real values and fake values separable.

*3 (A\*).* Prefer the highest index on ties: in stage A push $(f, -"node")$ (both the initial entry and the relaxation push) and negate again after
popping; in the key use $N dot "key" + (N - 1 - v)$ instead of $+ v$. The selection formulas stay the same. With $h = 0$ everywhere, the key is just
$g$ (plus the closed offset): A\* becomes Dijkstra on a unit-cost grid, which expands cells in BFS layers. Then $h_max = 0$, so
$"BIG" = 15 + 0 + 1 = 16$ already suffices.
