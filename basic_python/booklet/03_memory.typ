#import "style.typ": *
#show: booklet.with(
  chapter: "Chapter 3",
  title: "Memory",
  subtitle: "Arrays, argmax and state machines without indexing, comparing or branching",
)

// ── local helpers ───────────────────────────────────────────────────────
#let cell(x, fill: none) = table.cell(fill: fill, align: center)[#x]
#let hit = rgb("fde68a") // highlight colour for "the selected cell"
#let neuron(c) = (shape: rect, fill: tint(c), stroke: 0.7pt + c, corner-radius: 3pt, inset: 5pt)

= Memory is a one-hot vector

A Python program reads and writes memory with an _index_: `xs[i]`. A network has no
indexing instruction. It has vectors, matrices and ReLU, and nothing else. So how can a
network "look something up"?

The trick is to turn the address into a vector with exactly one `1` in it, a *one-hot*
vector. For example, with four cells the address $i = 2$ becomes

$ "onehot"(2) = (0, 0, 1, 0). $

The one-hot vector is the address. Reading and writing then become arithmetic with it:

- *read:* multiply each cell by its one-hot bit and add everything up. Cells whose bit is $0$ vanish.
- *write:* use the bit to decide, cell by cell, "take the new value" or "keep the old one".

There are two cases, and they cost very differently:

#table(
  columns: (1fr, 1fr, 1.4fr), align: (left, left, left),
  table.header[*memory is...*][*example*][*how the lookup works*],
  [a *constant* table, known when you design the network], [a lookup table of costs, a graph, a DFA transition table],
  [`onehot @ table`: an ordinary matrix product. *The weights are the memory.* No big-M needed.],
  [*runtime data*, an input or a computed value], [the array in `xs[i]`, the open list in A\*],
  [`sum_j select(onehot_j, xs[j])`. The bit and the cell are both runtime values, so each product needs a big-M select.],
)

Chapters 4 and 5 use both kinds: the graph is constant memory (weights), while "how far
have we got" is runtime memory (activations).

This chapter builds the machinery: reading (example 10), writing (11), finding the best cell
(12, `argmax`) and a machine that carries a one-hot _state_ through a string (13).

#key-idea[
  *Address = one-hot vector.* Everything in this chapter is "make a one-hot, then use it".
  A constant memory is read with a matrix product; a runtime memory needs a big-M select per
  cell, so reading $N$ cells costs $N$ selects.
]

#pagebreak()

// ════════════════════════════════════════════════════════════════════════
#example("10", "Array read", "03_memory/10_array_read.py")

*Problem.* Given a list of four integers `xs` (each in $[0, 7]$) and an index `i` in
$[0, 3]$, compute `xs[i]` with no indexing and no branching. Both `xs` and `i` are only known
at runtime.

#side-by-side(
  "A",
  [
    ```python
    def control_flow(xs, i):
        return xs[i]
    ```
    The machine jumps straight to cell `i`.
  ],
  "B",
  [
    ```python
    def branchless(xs, i):
        oh = onehot(i)
        total = 0
        for j in range(N):
            total += select(oh[j], xs[j])
        return total
    ```
    Touch *every* cell; all but one contribute 0.
  ],
)

== Step 1: the address as a one-hot

For one cell $j$ we need $[i = j]$. From the toolbox,
$a = b <=> 1 - step(a - b) - step(b - a)$, and $step(x) = relu(x) - relu(x - 1)$. Written out,

$ [i = j] = 1 - underbrace(relu(i - j), r_0) + underbrace(relu(i - j - 1), r_1) - underbrace(relu(j - i), r_2) + underbrace(relu(j - i - 1), r_3). $

So each cell needs four ReLU neurons. As a function of a real $i$ the result is a "tent" with
its peak at $i = j$. Because $i$ is an integer, we only ever sample the tent at whole numbers, where it is exactly $0$ or $1$:

#align(center, cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let col = network-color
  // axes
  line((-0.4, 0), (6.6, 0), mark: (end: ">", fill: ink), stroke: 0.6pt + ink)
  line((0, -0.3), (0, 2.1), mark: (end: ">", fill: ink), stroke: 0.6pt + ink)
  // tent for j = 2 : 0 up to i=1, peak at i=2, 0 from i=3
  line((0, 0), (1, 0), (2, 1.5), (3, 0), (6, 0), stroke: 1.2pt + col)
  for k in range(0, 7) {
    line((k, -0.06), (k, 0.06), stroke: 0.6pt + ink)
    content((k, -0.4), text(size: 8pt)[#k])
  }
  circle((2, 1.5), radius: 0.07, fill: col, stroke: none)
  for k in (0, 1, 3, 4, 5, 6) {
    circle((k, 0), radius: 0.07, fill: col, stroke: none)
  }
  content((0, 2.35), text(size: 8pt)[$[i = 2]$])
  content((6.4, -0.4), text(size: 8pt)[$i$])
  content((4.6, 1.3), text(size: 8pt, fill: muted)[dots = the only values $i$ can take])
}))

Worked example for $i = 2$ (the four neuron outputs per cell, then the result):

#align(center, table(
  columns: 7, align: center,
  table.header[*cell $j$*][*$relu(i-j)$*][*$relu(i-j-1)$*][*$relu(j-i)$*][*$relu(j-i-1)$*][*$1 - r_0 + r_1 - r_2 + r_3$*][*bit*],
  [0], [2], [1], [0], [0], [$1 - 2 + 1 - 0 + 0$], [0],
  [1], [1], [0], [0], [0], [$1 - 1 + 0 - 0 + 0$], [0],
  cell[2], cell[0], cell[0], cell[0], cell[0], cell[$1 - 0 + 0 - 0 + 0$], cell[1],
  [3], [0], [0], [1], [0], [$1 - 0 + 0 - 1 + 0$], [0],
))

== Step 2: keep one cell, kill the rest

Now `xs[j]` times its bit. Both are runtime values, so we cannot multiply them. Use the
big-M select from the toolbox:

$ sel(m, x) = relu(x - M(1 - m)) - relu(-x - M(1 - m)). $

If $m = 1$ the shift $M(1-m)$ is $0$ and we get $relu(x) - relu(-x) = x$. If $m = 0$ both ReLUs see
$plus.minus x - M <= 0$ (this needs $|x| <= M$) and the result is $0$. Here $M = 7$ because
cells are in $[0, 7]$.

*Trace*, `xs = (5, 0, 7, 2)`, $i = 2$, bits $(0,0,1,0)$:

#align(center, table(
  columns: 6, align: center,
  table.header[*$j$*][*bit $m$*][*$x = $ `xs[j]`*][*$relu(x - 7(1-m))$*][*$relu(-x - 7(1-m))$*][*$sel$*],
  [0], [0], [5], [$relu(-2) = 0$], [$relu(-12) = 0$], [0],
  [1], [0], [0], [$relu(-7) = 0$], [$relu(-7) = 0$], [0],
  cell[2], cell[1], cell[7], cell[$relu(7) = 7$], cell[$relu(-7) = 0$], cell[7],
  [3], [0], [2], [$relu(-5) = 0$], [$relu(-9) = 0$], [0],
))

The sum is $0 + 0 + 7 + 0 = 7 =$ `xs[2]`. Correct.

#note("Why M must be big enough")[
  The select only kills a masked-off cell if $x - M <= 0$. With $M = 4$ and a cell holding
  $7$, a "masked off" cell would leak $relu(7 - 4) = 3$ into the sum. *Rule:* $M$ is any upper
  bound on $|x|$. Bigger is safe but not free in a real network (it makes weights large).
  So always state the bound in a comment and derive it from the input range.
]

== Step 3: the network

Because `xs` $>= 0$, the second half $relu(-x - M(1-m))$ is always $0$ and we can drop it.
That leaves *three layers*. The input is $x = (i, "xs"[0], dots, "xs"[3])$.

- *Layer 1* (21 neurons): per cell, the four tent ReLUs, plus $relu("xs"[j])$ to carry the data forward (exact, since $"xs"[j] >= 0$).
- *Layer 2* (4 neurons): $a_j = relu("xs"[j] + M dot "bit"_j - M)$. Substituting $"bit"_j = 1 - r_0 + r_1 - r_2 + r_3$, the constants cancel and what is left is a plain weighted sum:
  $ a_j = relu(-M r_0 + M r_1 - M r_2 + M r_3 + "xs"[j]). $
- *Layer 3:* $y = sum_j a_j$ (a row of ones).

For cell $j = 2$ the weights of layer 1 (columns are $i, "xs"_0, ..., "xs"_3$) and of layer 2 are:

$
  W_1^((2)) = mat(
    1, 0, 0, 0, 0;
    1, 0, 0, 0, 0;
    -1, 0, 0, 0, 0;
    -1, 0, 0, 0, 0
  ), quad
  b_1^((2)) = vec(-2, -3, 2, 1), quad
  W_2^((2)) = mat(-7, 7, -7, 7, 1) #h(4pt) "(over" r_0 .. r_3, relu("xs"_2) ")"
$

The full $W_1$ is five such blocks stacked (cells 0..3 and the $relu("xs")$ carries); $W_2$ has one such
row per cell; $W_3$ is $(1, 1, 1, 1)$. In the code they are filled in by a loop at import time.

#align(center, diagram(
  node-stroke: 0.6pt, spacing: (24mm, 9mm),
  node((0, 1.5), [$i$], shape: circle, fill: tint(branchless-color), stroke: 0.7pt + branchless-color, inset: 5pt),
  node((0, 4), [xs$[j]$], shape: circle, fill: tint(branchless-color), stroke: 0.7pt + branchless-color, inset: 5pt),
  node((1, 0), [$relu(i - j)$], ..neuron(network-color)),
  node((1, 1), [$relu(i - j - 1)$], ..neuron(network-color)),
  node((1, 2), [$relu(j - i)$], ..neuron(network-color)),
  node((1, 3), [$relu(j - i - 1)$], ..neuron(network-color)),
  node((1, 4), [$relu("xs"[j])$], ..neuron(network-color)),
  node((2, 2), [$a_j = relu(dots)$], ..neuron(network-color)),
  node((3, 2), [$sum_j$], shape: circle, fill: tint(control-color), stroke: 0.7pt + control-color, inset: 5pt),
  edge((0, 1.5), (1, 0), "->", label: [$+1$], label-side: left),
  edge((0, 1.5), (1, 1), "->", label: [$+1$], label-side: left),
  edge((0, 1.5), (1, 2), "->", label: [$-1$], label-side: right),
  edge((0, 1.5), (1, 3), "->", label: [$-1$], label-side: right),
  edge((0, 4), (1, 4), "->", label: [$1$], label-side: left),
  edge((1, 0), (2, 2), "->", label: [$-M$], label-side: left),
  edge((1, 1), (2, 2), "->", label: [$+M$], label-side: left),
  edge((1, 2), (2, 2), "->", label: [$-M$], label-side: right),
  edge((1, 3), (2, 2), "->", label: [$+M$], label-side: right),
  edge((1, 4), (2, 2), "->", label: [$1$], label-side: right),
  edge((2, 2), (3, 2), "->"),
))

#align(center, text(size: 9pt, fill: muted)[The wiring for *one* cell $j$. The real network has four of these side by side, all fed by the same $i$, all feeding the same final sum.])

== Contrast: a constant table is just a matrix

Suppose the array is not an input but fixed forever, say `TABLE = [3, 1, 4, 7]`. Then
$"table"[i] = "onehot"(i) dot "TABLE"$, and since the one-hot is affine in layer 1 and $"TABLE"$ is a constant row, the whole read is
*one layer of ReLUs followed by affine maps*. There is no second layer, no $M$, no select:

$ y = "TABLE" dot (bold(1) + O h_1), $ where $h_1$ is the layer-1 output and $O$ is the constant $N times 5N$ matrix that has the row $(-1, 1, -1, 1)$ in each cell's four columns.

Compare the costs for $N$ cells:

#align(center, table(
  columns: 4, align: center,
  table.header[][*neurons*][*layers of ReLU*][*needs big-M*],
  [constant table], [$4N$], [1], [no],
  [runtime array], [$5N + N$], [2], [yes, $M >=$ bound of the values],
))

The code checks both. That's why graphs and cost tables in the next chapters become *weights*,
while only things that change during the computation need selects.

#note("Pitfall: out of range")[
  For $i = 7$ every tent $[i = j]$ is $0$, so the result is silently $0$. A network can't
  raise an `IndexError`. If an address might be invalid, decide what `0` should mean and
  make sure that is acceptable.
]

#pagebreak()

// ════════════════════════════════════════════════════════════════════════
#example("11", "Array write", "03_memory/11_array_write.py")

*Problem.* `xs[i] = v`, then use the list. Data flow has no "modify in place"; it returns a
*new* list. Each cell $j$ becomes `v` if $j$ is the address, else keeps its old value.

#side-by-side(
  "A",
  [
    ```python
    def control_flow(xs, v, i):
        xs = list(xs)
        xs[i] = v
        return xs
    ```
  ],
  "B",
  [
    ```python
    def branchless(xs, v, i):
        oh = onehot(i)
        new = []
        for j in range(N):
            new.append(
                xs[j] + select(oh[j], v - xs[j]))
        return new
    ```
  ],
)

== Derivation

For one cell, "$v$ if $m$ else $x$" is the toolbox's `x if m else y` row:

$ "new"_j = x_j + sel(m_j, v - x_j), quad m_j = [i = j]. $

Check both cases: $m_j = 1$: $x_j + (v - x_j) = v$. $m_j = 0$: $x_j + 0 = x_j$.

The thing being selected is the *difference* $d_j = v - x_j$. It is a runtime value that can be
negative, so here we need *both* ReLUs of the select, and $M$ must bound $|v - x_j|$. With values in
$[0, 5]$ the difference is in $[-5, 5]$, so $M = 5$.

*Trace*, `xs = (5, 0, 3, 2)`, $v = 4$, $i = 2$, bits $(0, 0, 1, 0)$:

#align(center, table(
  columns: 6, align: center,
  table.header[*$j$*][*$m_j$*][*$d_j = v - x_j$*][*$sel(m_j, d_j)$*][*$x_j + sel$*][*new*],
  [0], [0], [$4 - 5 = -1$], [$relu(-1-5) - relu(1-5) = 0$], [$5 + 0$], [5],
  [1], [0], [$4 - 0 = 4$], [$relu(4-5) - relu(-4-5) = 0$], [$0 + 0$], [0],
  cell[2], cell[1], cell[$4 - 3 = 1$], cell[$relu(1) - relu(-1) = 1$], cell[$3 + 1$], cell[4],
  [3], [0], [$4 - 2 = 2$], [$relu(2-5) - relu(-2-5) = 0$], [$2 + 0$], [2],
))

Result $(5, 0, 4, 2)$. Only cell 2 changed.

== The network

Input $x = (i, v, "xs"_0, dots, "xs"_3)$, three layers:

- *Layer 1* ($5N + 1 = 21$ neurons): the four tent ReLUs per cell, plus $relu(v)$ and $relu("xs"_j)$ as carried copies (everything is $>= 0$ so the copy is exact).
- *Layer 2* ($3N = 12$ neurons): for each cell, $ p_j = relu((v - "xs"_j) + M "bit"_j - M), quad q_j = relu(-(v - "xs"_j) + M "bit"_j - M), quad c_j = relu("xs"_j). $
  With $"bit"_j = 1 - r_0 + r_1 - r_2 + r_3$ the constants cancel, so $p_j$'s weight row is $(-M, M, -M, M)$ on the four tent neurons, $+1$ on $relu(v)$ and $-1$ on $relu("xs"_j)$. The $q_j$ row is the same with the last two signs flipped.
- *Layer 3* (affine only): $"new"_j = c_j + p_j - q_j$.

#align(center, diagram(
  node-stroke: 0.6pt, spacing: (26mm, 10mm),
  node((0, 0), [tents of $i$ \ (4 relus)], ..neuron(network-color)),
  node((0, 1), [$relu(v)$], ..neuron(network-color)),
  node((0, 2), [$relu("xs"_j)$], ..neuron(network-color)),
  node((1, 0), [$p_j$], ..neuron(network-color)),
  node((1, 1), [$q_j$], ..neuron(network-color)),
  node((1, 2), [$c_j$], ..neuron(network-color)),
  node((2, 1), [$"new"_j$], shape: circle, fill: tint(control-color), stroke: 0.7pt + control-color, inset: 5pt),
  edge((0, 0), (1, 0), "->", label: [$plus.minus M$], label-side: left),
  edge((0, 0), (1, 1), "->", label: [$plus.minus M$], label-side: right),
  edge((0, 1), (1, 0), "->", label: [$+1$], label-side: left),
  edge((0, 1), (1, 1), "->", label: [$-1$], label-side: right),
  edge((0, 2), (1, 0), "->", label: [$-1$], label-pos: 0.25, label-side: left),
  edge((0, 2), (1, 1), "->", label: [$+1$], label-pos: 0.25, label-side: right),
  edge((0, 2), (1, 2), "->", label: [$1$]),
  edge((1, 0), (2, 1), "->", label: [$+1$], label-side: left),
  edge((1, 1), (2, 1), "->", label: [$-1$], label-side: right),
  edge((1, 2), (2, 1), "->", label: [$+1$], label-side: right),
))

#align(center, text(size: 9pt, fill: muted)[The wiring for *one* cell $j$; the "tents" box stands for the four neurons of example 10.])

== Cost: read-modify-write

A write touches *every* cell. For $N$ cells you pay $N$ equality tests and $N$ selects
(about $5N$ neurons in layer 1 and $3N$ in layer 2), regardless of how little actually changes.
In a normal CPU a write is $O(1)$; in a network "addressing" costs a full pass over memory.
That is the price of having no pointers, and why we keep arrays tiny (a frontier of at most
a dozen nodes, not a million).

#note("A cheaper special case")[
  If the new value is a *constant* the select disappears. For `xs[i] += 1` you get
  $"new"_j = "xs"_j + [i = j]$: the one-hot is added directly (no big-M, one layer fewer). The select is only
  needed when *both* the mask and the value being written are runtime values you must combine.
]

#pagebreak()

// ════════════════════════════════════════════════════════════════════════
#example("12", "argmax and argmin", "03_memory/12_argmax.py")

*Problem.* Which cell is biggest? Return a one-hot vector marking it, plus the maximum
value. If several cells tie, the *lowest index wins*, exactly like
`xs.index(max(xs))` or `max(range(n), key=lambda k: xs[k])` in Python.

#side-by-side(
  "A",
  [
    ```python
    def control_flow(xs, sign=+1):
        pick = max if sign > 0 else min
        best = pick(range(len(xs)),
                    key=lambda k: xs[k])
        return ([1 if k == best else 0
                 for k in range(len(xs))],
                xs[best])
    ```
  ],
  "B",
  [
    ```python
    for i in range(N):
        bits = 0
        for j in range(N):
            if j == i: continue   # constants
            d = sign * (xs[i] - xs[j])
            bits += step(d) if j < i \
                    else step(d + 1)
        onehot.append(relu(bits - (N-2)))
    value = sum_i select(onehot[i], xs[i])
    ```
  ],
)

== The idea: every index runs a tournament

Index $i$ is *the* winner exactly when it beats every other index $j$. "Beats" is defined so
that ties are resolved towards the *smaller* index:

#align(center, table(
  columns: 3, align: (left, left, left),
  table.header[*opponent*][*$i$ beats $j$ when...*][*as a bit*],
  [$j < i$ (earlier)], [$x_i > x_j$ \ _strictly_: on a tie the earlier one wins], [$step(x_i - x_j)$],
  [$j > i$ (later)], [$x_i >= x_j$ \ on a tie $i$ wins, because it is earlier], [$step(x_i - x_j + 1)$],
))

#key-idea[
  *Tie-break rule:* strict `>` against earlier indices, `>=` against later ones. Then exactly
  one index wins every contest, and it is the *lowest index among the maxima*. Remember this
  rule: A\* (chapter 4) uses argmin as "pick the best node", and two nodes with equal cost must be
  resolved the same way Python resolves them, otherwise the network and the algorithm
  silently disagree.
]

Why exactly one winner? Let $m$ be the maximum value and $k$ the smallest index holding it.
Index $k$ is $>= $ everything after it and strictly $>$ everything before it (nothing earlier equals $m$), so it wins. Any other $i$ loses to someone: if $x_i < m$ it loses to $k$; if $x_i = m$ then $i > k$, and the contest against the earlier $k$ is strict, so $i$ loses.

== AND over all opponents

Each index has $n - 1$ opponents, so $n - 1$ bits. Their AND is 1 only when all are 1. Generalising
$"AND"(p, q) = relu(p + q - 1)$:

$ "won"_i = relu(sum_(j != i) "beats"_(i j) - (n - 2)). $

The sum is at most $n - 1$; subtracting $n - 2$ leaves $1$ only if every bit is $1$, and $<= 0$
(killed by the ReLU) otherwise.

*Trace*, `xs = (2, 5, 5, 1)`, argmax. Each row is $i$, each column an opponent $j$; the entry is the bit "$i$ beats $j$":

#align(center, table(
  columns: 8, align: center,
  table.header[$i$][$x_i$][$j=0$][$j=1$][$j=2$][$j=3$][*sum*][*$relu("sum" - 2)$*],
  [0], [2], [--], [$2 >= 5$: 0], [$2 >= 5$: 0], [$2 >= 1$: 1], [1], [0],
  cell(fill: hit)[1], cell(fill: hit)[5], cell(fill: hit)[$5 > 2$: 1], cell(fill: hit)[--], cell(fill: hit)[$5 >= 5$: 1], cell(fill: hit)[$5 >= 1$: 1], cell(fill: hit)[3], cell(fill: hit)[*1*],
  [2], [5], [$5 > 2$: 1], [$5 > 5$: *0*], [--], [$5 >= 1$: 1], [2], [0],
  [3], [1], [$1 > 2$: 0], [$1 > 5$: 0], [$1 > 5$: 0], [--], [0], [0],
))

The one-hot is $(0, 1, 0, 0)$: index 1 wins the tie against index 2, because index 2's contest
against the earlier index 1 is strict ($5 > 5$ is false). The value is
$sum_i sel("won"_i, x_i) = 0 + 5 + 0 + 0 = 5$ (big-M select with $M = 5$).

*Argmin* is the same machine with the differences negated: $d = -(x_i - x_j)$, with the same
strict/non-strict rule. For `xs = (2, 5, 5, 1)` index 3 beats everyone; for an all-equal list
`(3, 3, 3, 3)`, index 0 wins in both argmax and argmin.

#align(center, cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let xs = (2, 5, 5, 1)
  line((-0.3, 0), (4.6, 0), stroke: 0.6pt + ink)
  for (k, v) in xs.enumerate() {
    let col = if k == 1 { network-color } else { rgb("9aa5b1") }
    rect((k + 0.1, 0), (k + 0.9, v * 0.4), fill: col, stroke: none)
    content((k + 0.5, v * 0.4 + 0.25), text(size: 8pt)[#v])
    content((k + 0.5, -0.3), text(size: 8pt)[xs\[#k\]])
  }
  content((1.5, 2.9), text(size: 8pt, fill: network-color)[tie: lowest index wins])
  line((1.5, 2.75), (1.5, 2.5), mark: (end: ">"), stroke: 0.5pt + network-color)
}))

== The network

Three layers, built from loops over the (constant) pairs $(i, j)$:

- *Layer 1* ($2 n (n - 1) + n = 28$ neurons): for each ordered pair, $d = "sign" (x_i - x_j) + c$ with $c = 0$ if $j < i$, $c = 1$ if $j > i$; two neurons $relu(d)$, $relu(d - 1)$ whose difference is $step(d)$. Plus a carried $relu(x_i)$ per index.
- *Layer 2* ($2n = 8$): $"won"_i = relu(sum_j (relu(d_(i j)) - relu(d_(i j) - 1)) - (n - 2))$. This vector *is* the one-hot answer. Plus carried $x_i$.
- *Layer 3* ($n = 4$): $a_i = relu(x_i + M "won"_i - M)$; the value is $sum_i a_i$.

For the pair $(i, j) = (1, 0)$ ($j < i$, strict, sign $+1$) layer 1 contains the rows
$ mat(1, -1, 0, 0; 1, -1, 0, 0) x + vec(0, -1) = vec(relu(x_1 - x_0), relu(x_1 - x_0 - 1)). $
For the pair $(1, 2)$ ($j > i$, non-strict) the biases are $(+1, 0)$ instead and the columns are $(0, 1, -1, 0)$. For argmin, flip the sign of every weight in the $x$ columns; nothing else changes.

Layer 2's row for $i = 1$ adds the three beats-bits: $(+1, -1)$ for each of its three pairs, bias $-(n - 2) = -2$.

#align(center, scale(80%, reflow: true, diagram(
  node-stroke: 0.6pt, spacing: (20mm, 8mm),
  node((0, 0), [$x_0$], shape: circle, fill: tint(branchless-color), stroke: 0.7pt + branchless-color, inset: 5pt),
  node((0, 1), [$x_1$], shape: circle, fill: tint(branchless-color), stroke: 0.7pt + branchless-color, inset: 5pt),
  node((0, 2), [$x_2$], shape: circle, fill: tint(branchless-color), stroke: 0.7pt + branchless-color, inset: 5pt),
  node((0, 3), [$x_3$], shape: circle, fill: tint(branchless-color), stroke: 0.7pt + branchless-color, inset: 5pt),
  node((1, 0.5), [pairs $(i, j)$ \ $2 n (n - 1)$ relus], ..neuron(network-color)),
  node((1, 2.5), [carried $relu(x_i)$], ..neuron(network-color)),
  node((2, 0.5), [$"won"_i$ \ $= "AND"_j$ ($n$ relus)], ..neuron(network-color)),
  node((2, 2.5), [carried $x_i$], ..neuron(network-color)),
  node((3, 1.5), [$a_i = relu(x_i + M "won"_i - M)$], ..neuron(network-color)),
  node((4, 0.5), [*one-hot* \ $("won"_0 dots "won"_3)$], shape: rect, fill: tint(control-color), stroke: 0.7pt + control-color, corner-radius: 3pt, inset: 5pt),
  node((4, 2.5), [*value* \ $sum_i a_i$], shape: rect, fill: tint(control-color), stroke: 0.7pt + control-color, corner-radius: 3pt, inset: 5pt),
  edge((0, 0), (1, 0.5), "->"), edge((0, 1), (1, 0.5), "->"),
  edge((0, 2), (1, 0.5), "->"), edge((0, 3), (1, 0.5), "->"),
  edge((0, 1.5), (1, 2.5), "->"),
  edge((1, 0.5), (2, 0.5), "->", label: [$plus.minus 1$], label-side: left),
  edge((1, 2.5), (2, 2.5), "->"),
  edge((2, 0.5), (3, 1.5), "->", label: [$M$], label-side: left),
  edge((2, 2.5), (3, 1.5), "->", label: [$1$], label-side: right),
  edge((2, 0.5), (4, 0.5), "->", bend: 0deg),
  edge((3, 1.5), (4, 2.5), "->"),
)))

== Cost: $n^2$

Every pair of elements is compared, so layer 1 has $2 n (n - 1)$ neurons. That is quadratic:

#align(center, table(
  columns: 5, align: center,
  table.header[*$n$*][*pairs $n(n-1)$*][*layer-1 neurons*][*layers*][*note*],
  [4], [12], [28], [3], [the example],
  [10], [90], [190], [3], [],
  [100], [9 900], [19 900], [3], [still shallow, but wide],
))

The depth stays at 3 no matter how big $n$ is. Everything is compared in *parallel*; what you
pay is width. (Tournament-style `max` of pairs would need only $n - 1$ comparisons but
$log_2 n$ layers, and it makes the one-hot and the tie-break harder.)

#note("Pitfalls")[
  - *Candidates that must not win.* If some cells are not eligible (e.g. A\* nodes not in the open list), do not exclude them with an `if`: add a large constant to their value first (for argmin, add a constant `BIG` where the mask is 0, via a select). Make `BIG` larger than any real value so they can never win, and widen $M$ accordingly.
  - *The tie rule is part of the algorithm.* Swap the strictness and you get the _last_ maximum instead; both are valid but only one matches the reference program.
  - *Integers only.* The `step(d)` and `step(d + 1)` trick for `>` and `>=` is exact only for integers. Use scaled integer costs (e.g. multiples of 10), not floats.
]

#pagebreak()

// ════════════════════════════════════════════════════════════════════════
#example("13", "State machine", "03_memory/13_state_machine.py")

*Problem.* Does a bit string contain the substring `101`? Strings have length 0 to 6. A
finite automaton (DFA) solves this with four states; here it is as a network.

#align(center, diagram(
  node-stroke: 0.7pt, spacing: (30mm, 20mm),
  node((0, 0), [$q_0$], name: <q0>, shape: circle, inset: 7pt, fill: tint(network-color)),
  node((1, 0), [$q_1$], name: <q1>, shape: circle, inset: 7pt, fill: tint(network-color)),
  node((2, 0), [$q_2$], name: <q2>, shape: circle, inset: 7pt, fill: tint(network-color)),
  node((3, 0), [$q_3$], name: <q3>, shape: circle, inset: 7pt, fill: tint(network-color), extrude: (0, 3)),
  edge(<q0>, <q0>, "->", bend: 130deg, label: [`0`]),
  edge(<q0>, <q1>, "->", label: [`1`], label-side: left),
  edge(<q1>, <q1>, "->", bend: 130deg, label: [`1`]),
  edge(<q1>, <q2>, "->", label: [`0`], label-side: left),
  edge(<q2>, <q3>, "->", label: [`1`], label-side: left),
  edge(<q2>, <q0>, "->", bend: -40deg, label: [`0`], label-side: right),
  edge(<q3>, <q3>, "->", bend: 130deg, label: [`0, 1`]),
  node((0, -0.9), [start], stroke: none),
  edge((0, -0.9), <q0>, "->"),
))

#align(center, text(size: 9pt, fill: muted)[$q_0$: nothing useful yet. $q_1$: just saw `1`. $q_2$: just saw `10`. $q_3$: saw `101` (accepting, never leaves).])

#side-by-side(
  "A",
  [
    ```python
    ZERO = {0: 0, 1: 2, 2: 0, 3: 3}
    ONE  = {0: 1, 1: 1, 2: 3, 3: 3}

    def control_flow(s):
        state = 0
        for ch in s:
            table = ZERO if ch == "0" else ONE
            state = table[state]
        return 1 if state == 3 else 0
    ```
    A loop as long as the string, and two table lookups.
  ],
  "B",
  [
    ```python
    state = [1, 0, 0, 0]       # one-hot
    for t in range(L):         # L = 6, fixed
        is_sym = [equals(codes[t], k)
                  for k in range(K)]
        new = [0] * S
        for k in range(K):
            for s_next in range(S):
                moved = T[k][s_next] @ state
                new[s_next] += relu(
                    moved + is_sym[k] - 1)
        state = new
    ```
  ],
)

== Step 1: the state becomes a one-hot vector

A state is a number $0..3$; we store it as a vector with a single $1$:
$q_2 = (0, 0, 1, 0)$. "Which state am I in?" is now an address, exactly like in example 10.

== Step 2: one constant matrix per symbol

For each symbol $k$ build a constant $4 times 4$ matrix $T_k$ with $T_k[s', s] = 1$ when the
DFA goes from state $s$ to state $s'$ on symbol $k$. Then $T_k$ times a one-hot state is the one-hot
of the next state, since multiplying a matrix by $e_s$ picks out column $s$.

The code has three symbols: `0`, `1`, and a padding symbol (code 2) that fills strings shorter than 6 with
"do nothing", so every input has the same length. Rows are the *next* state, columns the *current* state.

#align(center, grid(
  columns: 3, gutter: 14pt, align: center,
  [$T_0 = mat(1, 0, 1, 0; 0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 1)$ \ #text(size: 8pt, fill: muted)[on `0`: $q_0 -> q_0$, $q_1 -> q_2$, $q_2 -> q_0$, $q_3 -> q_3$]],
  [$T_1 = mat(0, 0, 0, 0; 1, 1, 0, 0; 0, 0, 0, 0; 0, 0, 1, 1)$ \ #text(size: 8pt, fill: muted)[on `1`: $q_0 -> q_1$, $q_1 -> q_1$, $q_2 -> q_3$, $q_3 -> q_3$]],
  [$T_"pad" = I = mat(1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1)$ \ #text(size: 8pt, fill: muted)[padding: stay]],
))

Read the matrices column by column against the diagram: column 1 of $T_0$ is $(1,0,0,0)$ ($q_0$ on a `0` goes to $q_0$), column 2 is $(0,0,1,0)$ ($q_1$ on `0` goes to $q_2$), and so on. Each column contains exactly one $1$, because the automaton is deterministic. That property matters below.

== Step 3: use the symbol that was actually read

Which matrix applies depends on the runtime symbol, so we cannot just pick one. Instead we apply *all
three* and keep the right result using a 0/1 flag $"is"_k = [c_t = k]$ (computed with the equality tent from example 10):

$ "next" = sum_k "AND"("is"_k, T_k "state") = sum_k relu(T_k "state" + "is"_k - 1). $

Everything here is a 0/1 bit, so the AND is exact. Only the term whose $"is"_k = 1$ can be nonzero; for the others
$T_k "state" + 0 - 1 <= 0$ and the ReLU zeroes them. No big-M is needed because nothing is a large number:
both factors are bits.

*Trace*, string `0101`, padded to `0101··` (codes $0,1,0,1,2,2$):

#align(center, table(
  columns: 5, align: center,
  table.header[*step $t$*][*symbol*][*state before*][*the winning term*][*state after*],
  [0], [`0`], [$q_0$], [$T_0 e_0 = e_0$], [$q_0$],
  [1], [`1`], [$q_0$], [$T_1 e_0 = e_1$], [$q_1$],
  [2], [`0`], [$q_1$], [$T_0 e_1 = e_2$], [$q_2$],
  [3], [`1`], [$q_2$], [$T_1 e_2 = e_3$], [$q_3$],
  [4], [pad], [$q_3$], [$I e_3 = e_3$], [$q_3$],
  [5], [pad], [$q_3$], [$I e_3 = e_3$], [$q_3$],
))

At the end, accepting is a dot product: $"accept" = (0, 0, 0, 1) dot "state"$, giving $1$.

At step 1 in detail (state $e_0$, symbol `1`, so $"is" = (0, 1, 0)$):
$k=0$: $relu(T_0 e_0 + 0 - 1) = relu(e_0 - 1) = 0$. $k=1$: $relu(T_1 e_0 + 1 - 1) = relu(e_1) = e_1$. $k=2$: $relu(I e_0 - 1) = 0$. Sum: $e_1$.

#note("Why determinism matters")[
  Each column of every $T_k$ has exactly one $1$, so $T_k "state"$ is again a one-hot (entries $0$ or
  $1$, never $2$). If two states could move to the same target the product would still be a
  bit, but a _non_-deterministic automaton (one state going to two targets) could create
  sums $> 1$ and you would need an OR (`step`) after the AND. The DFA avoids that.
]

== The network: one layer, repeated

Concatenate the state with a "window" holding the not-yet-read flags: $h = ("state" | "is"_(0,0..2) | "is"_(1,0..2) | dots)$, 4 + 18 entries.
Then *one* step layer, with the *same* weights each time:

$ "hidden" = relu(A h + b), quad h' = B "hidden". $

- Rows $(s, k)$ of $A$: $[ "row " s " of " T_k | "pick the first window flag for symbol " k ]$ with bias $-1$. That is exactly $T_k "state" + "is"_k - 1$ for output $s$.
- Extra rows of $A$ carry the window forward unchanged (flags are $>= 0$ so ReLU is exact).
- $B$ sums over $k$ to get the new state, and *shifts the window left* by one symbol so that the next character's flags are at the front. After $L = 6$ shifts all flags were used; the zero fill is never read.

A tiny slice: the three hidden rows for next-state $q_1$ (state columns first, then the first window position's flags for `0`, `1`, pad):

$
  mat(
    0,0,0,0, 1,0,0;
    1,1,0,0, 0,1,0;
    0,1,0,0, 0,0,1
  ) vec("state", "is"_0, "is"_1, "is"_"pad") + vec(-1, -1, -1)
$

(First row: $T_0$'s row 1, the third: $I$'s row 1.) $B$ adds these three outputs together to form the new state's $q_1$ component.

The *encoder* that creates the flags runs once: for each position $t$ and symbol $k$, four ReLUs
$relu(c_t - k), relu(c_t - k - 1), relu(k - c_t), relu(k - c_t - 1)$ and then $"is" = 1 - r_0 + r_1 - r_2 + r_3$ (72 neurons).

#align(center, diagram(
  node-stroke: 0.6pt, spacing: (10.5mm, 12mm),
  node((0, 0), [codes \ $c_0 dots c_5$], shape: rect, fill: tint(branchless-color), stroke: 0.7pt + branchless-color, corner-radius: 3pt, inset: 5pt),
  node((1.4, 0), [encoder \ (once)], ..neuron(control-color)),
  node((2.8, 0), [step \ $A, B$], ..neuron(network-color)),
  node((4, 0), [step \ $A, B$], ..neuron(network-color)),
  node((5.1, 0), [$dots$], stroke: none),
  node((6.2, 0), [step \ $A, B$], ..neuron(network-color)),
  node((7.6, 0), [accept \ $= (0,0,0,1) dot$], shape: rect, fill: tint(control-color), stroke: 0.7pt + control-color, corner-radius: 3pt, inset: 5pt),
  edge((0, 0), (1.4, 0), "->"),
  edge((1.4, 0), (2.8, 0), "->", label: [$h$]),
  edge((2.8, 0), (4, 0), "->", label: [$h$]),
  edge((4, 0), (5.1, 0), "->"),
  edge((5.1, 0), (6.2, 0), "->"),
  edge((6.2, 0), (7.6, 0), "->", label: [$h$]),
  node((4.5, 1), [same weights, 6 times], stroke: none, inset: 0pt),
))

#note("Cost and trade-offs")[
  Each step layer has $S K + L K = 12 + 18 = 30$ neurons; the whole network is the encoder (72) plus
  six identical steps. A longer string needs more repetitions, so the maximum length is fixed
  in advance, as in chapter 2's bounded loops. A bigger alphabet or more states makes the
  $T_k$ bigger, but the structure stays the same. Padding with an identity symbol is how a
  fixed-depth network handles shorter inputs.
]

#pagebreak()

// ════════════════════════════════════════════════════════════════════════
= Try it

+ *(10)* What does the read network return for $i = 7$ (outside $0..3$)? Why is that dangerous?
+ *(11)* Write `xs[i] += 1` as a network. Do you need the big-M select?
+ *(12)* Change the tie-break so that the *last* maximal index wins.
+ *(13)* Design a DFA for "contains `11`" and give `DELTA` and the accepting set.

#pagebreak()

= Answers

+ *(10)* Every tent $[i = j]$ is $0$ for $i = 7$, so all four $a_j$ are $0$ and $y = 0$. No error is raised; an invalid address silently reads as $0$. If $0$ is also a legitimate stored value you cannot tell the difference, so make sure addresses stay in range, or reserve a value for "invalid".

+ *(11)* No select is needed: $"new"_j = "xs"_j + [i = j]$. The added amount is a constant ($1$) multiplied by a bit, which is just the bit. In the network, layer 1 computes the tents as before, and the output layer is affine: $"new"_j = relu("xs"_j) + (1 - r_0 + r_1 - r_2 + r_3)$. Two layers of weights and no $M$. (Selects are only needed when the written value is itself a runtime quantity.)

+ *(12)* Swap the strictness: strict ($step(d)$) against *later* indices ($j > i$), non-strict ($step(d + 1)$) against *earlier* ones ($j < i$). Then on a tie the later index wins every contest. In the code: `step(d) if j > i else step(d + 1)`; in the weights, swap which pairs get bias $c = 0$ and which get $c = 1$.

+ *(13)* Three states: $q_0$ nothing, $q_1$ just saw `1`, $q_2$ saw `11` (accepting, stays).
  `DELTA = {0: [0, 0, 2], 1: [1, 2, 2], 2: [0, 1, 2]}` (rows: symbol `0`, symbol `1`, pad; entries: next state for $q_0, q_1, q_2$). Accepting vector $(0, 0, 1)$. Set `S = 3`; $T_k$ are rebuilt from `DELTA` automatically. Nothing else in the network changes: the step layer is regenerated with $3 K + L K$ rows.
