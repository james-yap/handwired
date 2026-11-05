#import "style.typ": *
#show: booklet.with(
  chapter: "Chapter 1",
  title: "Foundations",
  subtitle: "relu, step, comparisons, logic, min / max / clamp, and select",
)

// ── local helpers (diagrams, plots, matrices) ───────────────────────────
#show table: set block(breakable: false)

#let pos-color = rgb("2563eb")
#let neg-color = rgb("dc2626")
#let fnt = "Avenir Next"

// A matrix / vector from nested arrays of numbers.
#let mx(rows) = math.mat(..rows.map(r => r.map(v => [#v])))
#let vc(items) = math.vec(..items.map(v => [#v]))

// Edge list (l1, j, l2, i, w) from a weight matrix W (rows = destination neurons).
#let edges-of(W, l1, l2) = {
  let out = ()
  for (i, row) in W.enumerate() {
    for (j, w) in row.enumerate() {
      if w != 0 { out.push((l1, j, l2, i, w)) }
    }
  }
  out
}

// Node constructors: label inside the circle, optional note beside it and a bias tag.
#let nin(label, note: none) = (kind: "in", label: label, note: note, bias: none)
#let nrelu(label, bias: none, note: none) = (kind: "relu", label: label, note: note, bias: bias)
#let nout(label, bias: none, note: none) = (kind: "out", label: label, note: note, bias: bias)

// Layered network picture.  layers: array of arrays of nodes.  edges: (l1, j, l2, i, w).
// An unlabeled edge has weight +1 (blue) or -1 (red, dashed); other weights are written on it.
#let net(layers, edges, dx: 3.2, dy: 0.9, r: 0.3, label-at: 0.72, unit: 1cm) = {
  cetz.canvas(length: unit, {
    import cetz.draw: *
    let pos(l, i) = (l * dx, ((layers.at(l).len() - 1) / 2 - i) * dy)
    for e in edges {
      let (l1, j, l2, i, w) = e
      let p1 = pos(l1, j)
      let p2 = pos(l2, i)
      let c = if w > 0 { pos-color } else { neg-color }
      line(p1, p2, stroke: (paint: c, thickness: 0.55pt, dash: if w < 0 { "dashed" } else { none }))
    }
    for e in edges {
      let (l1, j, l2, i, w) = e
      if calc.abs(w) != 1 {
        let p1 = pos(l1, j)
        let p2 = pos(l2, i)
        let t = label-at
        let q = (p1.at(0) + (p2.at(0) - p1.at(0)) * t, p1.at(1) + (p2.at(1) - p1.at(1)) * t)
        content(q, box(fill: white, inset: 1.2pt, radius: 1pt, text(size: 6.5pt, fill: if w > 0 { pos-color } else { neg-color })[#w]))
      }
    }
    for (l, layer) in layers.enumerate() {
      for (i, nd) in layer.enumerate() {
        let p = pos(l, i)
        let (fill-c, stroke-c) = if nd.kind == "in" {
          (white, ink)
        } else if nd.kind == "relu" {
          (tint(network-color), network-color)
        } else {
          (rgb("eceff3"), muted)
        }
        circle(p, radius: r, fill: fill-c, stroke: 0.8pt + stroke-c)
        content(p, text(size: 7pt)[#nd.label])
        if nd.bias != none {
          content((p.at(0), p.at(1) - r - 0.17), text(size: 6pt, fill: muted)[#nd.bias])
        }
        if nd.note != none {
          let side = if nd.kind == "in" { "east" } else { "west" }
          // notes sit outside: left of inputs, right of outputs / neurons
          if nd.kind == "in" {
            content((p.at(0) - r - 0.1, p.at(1)), anchor: "east", text(size: 7pt, fill: muted)[#nd.note])
          } else {
            content((p.at(0) + r + 0.1, p.at(1)), anchor: "west", text(size: 7pt, fill: muted)[#nd.note])
          }
        }
      }
    }
  })
}

// Simple function plot.  curves: array of (pts: ((x, y), ...), color: c, dash: ..) ;
// dots: array of (x, y, color) ; texts: array of (x, y, content).
#let plot2d(
  xs: (-4, 4), ys: (-1, 3), size: (6.5, 3.2),
  curves: (), dots: (), texts: (), xticks: auto, yticks: auto, xlabel: [], ylabel: [],
) = {
  cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let (x0, x1) = xs
    let (y0, y1) = ys
    let sx = size.at(0) / (x1 - x0)
    let sy = size.at(1) / (y1 - y0)
    let P(x, y) = ((x - x0) * sx, (y - y0) * sy)
    let xt = if xticks == auto { range(calc.ceil(x0), calc.floor(x1) + 1) } else { xticks }
    let yt = if yticks == auto { range(calc.ceil(y0), calc.floor(y1) + 1) } else { yticks }
    for t in xt { line(P(t, y0), P(t, y1), stroke: 0.3pt + rule-color) }
    for t in yt { line(P(x0, t), P(x1, t), stroke: 0.3pt + rule-color) }
    let ax = if y0 <= 0 and y1 >= 0 { 0 } else { y0 }
    let ay = if x0 <= 0 and x1 >= 0 { 0 } else { x0 }
    line(P(x0, ax), P(x1, ax), stroke: 0.7pt + ink, mark: (end: ">", scale: 0.5))
    line(P(ay, y0), P(ay, y1), stroke: 0.7pt + ink, mark: (end: ">", scale: 0.5))
    for t in xt {
      if t != 0 { content(P(t, ax), anchor: "north", padding: 0.07, text(size: 6.5pt, fill: muted)[#t]) }
    }
    for t in yt {
      if t != 0 { content(P(ay, t), anchor: "east", padding: 0.07, text(size: 6.5pt, fill: muted)[#t]) }
    }
    content(P(x1, ax), anchor: "west", padding: 0.1, text(size: 7.5pt)[#xlabel])
    content(P(ay, y1), anchor: "south", padding: 0.06, text(size: 7.5pt)[#ylabel])
    for c in curves {
      let pts = c.pts.map(p => P(p.at(0), p.at(1)))
      line(..pts, stroke: (paint: c.color, thickness: 1.3pt, dash: c.at("dash", default: none)))
    }
    for d in dots { circle(P(d.at(0), d.at(1)), radius: 0.07, fill: d.at(2), stroke: none) }
    for t in texts { content(P(t.at(0), t.at(1)), text(size: 7.5pt)[#t.at(2)]) }
  })
}

#let rl(x) = calc.max(0, x)

// A raw code block for use inside stage boxes.
#let code(body) = raw(body, lang: "python", block: true)

= Before the examples: the ground rules

== The idea

A normal program _decides_. It reaches an `if`, follows one branch and never looks at the other. A neural network cannot decide anything: every neuron computes its number every time, and the only operations available are multiply by a weight, add, and one tiny non-linearity, $relu(x) = max(0, x)$.

This book series shows that nothing is lost. A program on small integers can be rewritten so that it computes *every* branch and then multiplies the wrong branches by 0. The 0 and the 1 are ordinary numbers (a "mask") produced by the same arithmetic. Written this way, the program is a stack of `relu(W x + b)` layers: a feed-forward network with weights you wrote by hand. Later chapters do this to loops, arrays, BFS, A\* and minimax. This chapter builds the six tools everything else is made from.

#figure(
  grid(
    columns: (1fr, 1fr), gutter: 14pt, align: horizon,
    diagram(
      node-stroke: 0.6pt, spacing: (13mm, 9mm), node-inset: 5pt,
      node((0, 0), text(size: 8pt)[`a > 4 ?`], name: <c>),
      node((1, -0.6), text(size: 8pt)[`10`], name: <t>),
      node((1, 0.6), text(size: 8pt, fill: muted)[`12`], name: <f>),
      edge(<c>, <t>, "->", label: text(size: 7pt)[yes], label-side: left),
      edge(<c>, <f>, "-->", label: text(size: 7pt)[no], label-side: right),
    ),
    diagram(
      node-stroke: 0.6pt, spacing: (13mm, 9mm), node-inset: 5pt,
      node((0, 0), text(size: 8pt)[$step(a - 4)$], name: <m>),
      node((1, -0.6), text(size: 8pt)[`10`], name: <t>),
      node((1, 0.6), text(size: 8pt)[`12`], name: <f>),
      node((2, 0), text(size: 8pt)[mix by mask], name: <o>),
      edge(<m>, <o>, "->", label: text(size: 7pt)[mask 0 / 1], label-side: left),
      edge(<t>, <o>, "->"),
      edge(<f>, <o>, "->"),
    ),
  ),
  caption: [Left: control flow takes one road. Right: data flow takes every road and lets a 0/1 number choose the answer.],
)

== Three stages, one rule each

Every example is written three times. The three versions must agree on every input, and each file proves it by running all three on every input in range (or on a fixed random sample plus edge cases when the range is too large).

#table(
  columns: (auto, 1fr, 1fr),
  table.header[*Stage*][*What you may write*][*What you may never write*],
  [#text(fill: control-color)[*A · control flow*]], [Normal Python. Anything.], [Nothing is forbidden. It is the specification.],
  [#text(fill: branchless-color)[*B · branchless*]],
  [Scalar Python: `+`, `-`, multiplying by a *constant*, `relu`, helpers built from those, loops whose count is fixed before the input is seen (`range(N)` with a constant `N`).],
  [`if` on a runtime value, comparisons, `min`/`max`/`abs` on runtime values, multiplying two runtime values, a loop whose length depends on the input.],
  [#text(fill: network-color)[*C · network*]],
  [numpy: constant-matrix `@`, `+`/`-`, scalar-constant `*`, `np.maximum(0, ·)`, slicing, `np.concatenate`, fixed loops.],
  [The same list, plus `np.where`, `argmax`, `max` on runtime data, and elementwise products of two runtime arrays.],
)

A *runtime value* is anything that depends on the input. A *constant* is known before the input arrives: a literal, `N`, or a weight. The central rule is the one about products: _a weight multiplies a runtime value by a constant, and nothing else multiplies._ Example 06 shows how a mask "multiplies" a runtime value anyway.

Stage B is the algebra: it is where you find the formulas and check them in plain Python. Stage C is the same algebra with the bookkeeping done by matrices: it shows which neurons exist, what they are connected to, and how deep the network is.

== Why only integers

All the 0/1 decisions come from one trick (example 01): $step(x) = relu(x) - relu(x - 1)$. This is exactly 0 for $x <= 0$ and exactly 1 for $x >= 1$, but in between, for $0 < x < 1$, it is a ramp that returns $x$ itself. There is a hole between 0 and 1 where the "decision" is undecided. If every value that can reach $step$ is an *integer*, no input ever lands in the hole, and the ramp is invisible: $step$ is an exact 0/1 switch. That is why every example declares an integer range up front.

Two consequences. First, "$a > b$" is only a statement about integers: the test is $a - b >= 1$. Second, there is no fuzziness to worry about: small integers are represented exactly by float64, so the files compare A, B and C with plain `==`. (If your inputs live on a finer grid, say multiples of $1/4$, scale them by 4 first; example 01 shows it.)

== What "a network" means here

A *layer* is the function
$ h = relu(W x + b) $
where $x$ is a vector of numbers coming in, $W$ is a matrix of constants, $b$ is a vector of constants and $relu$ is applied to every entry. *Row $i$ of $W$ together with $b_i$ is one neuron*: it takes a weighted sum of the inputs, adds its bias, and clips at zero. A network is layers stacked, with a final layer that has no $relu$, only the affine part:
$ y = W_2 thin relu(W_1 x + b_1) + b_2 $

For instance, with $x = (3, 1)$:

#grid(
  columns: (1fr, auto), gutter: 12pt, align: horizon,
  [
    $ W_1 x + b_1 = #mx(((1, -1), (-1, 1), (1, 1))) #vc((3, 1)) + #vc((0, 0, -2)) = #vc((2, -2, 2)) $
    $ h = relu(dots) = #vc((2, 0, 2)) $
    $ y = #mx(((1, 1, 1),)) #vc((2, 0, 2)) + (-1) = 3 $
  ],
  net(
    (
      (nin($x_0$), nin($x_1$)),
      (nrelu($h_0$), nrelu($h_1$), nrelu($h_2$, bias: $-2$)),
      (nout($y$, bias: $-1$),),
    ),
    edges-of(((1, -1), (-1, 1), (1, 1)), 0, 1) + edges-of(((1, 1, 1),), 1, 2),
    dx: 2.4,
  ),
)

The diagrams in this booklet draw exactly this. A circle is a neuron; a line is one entry of $W$ (blue: positive, red dashed: negative; a line with no number has weight $plus.minus 1$; zero entries are not drawn). The number under a neuron is its bias. White circles are inputs, green circles apply $relu$, grey circles are output neurons that only add.

*Stage C is a real feed-forward neural network.* If you want to see it with your own eyes, the second example's weights drop straight into any deep-learning library:

```python
net = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1))
net[0].weight.data = torch.tensor([[1.0], [1.0]]);  net[0].bias.data = torch.tensor([-4.0, -5.0])
net[2].weight.data = torch.tensor([[-2.0, 2.0]]);   net[2].bias.data = torch.tensor([12.0])
```

Nothing here is trained. Every weight is chosen by reasoning, which is the whole point: the matrices _are_ the program, and you can read the program back out of them.

#pagebreak(weak: true)
== The toolbox

Everything in this chapter is a row of this table. Inputs are integers where $step$ appears.

#table(
  columns: (auto, 1fr, auto, auto),
  table.header[*Want*][*Data-flow formula*][*Holds when*][*Example*],
  [$x >= 1$ as 0/1], [$step(x) = relu(x) - relu(x - 1)$], [$x$ integer], [01],
  [`a > b`], [$step(a - b)$], [integers], [03],
  [`a >= b`], [$step(a - b + 1)$], [integers], [03],
  [`a == b`], [$1 - step(a - b) - step(b - a)$], [integers], [03],
  [`not p`], [$1 - p$], [$p in {0, 1}$], [04],
  [`p and q`], [$relu(p + q - 1)$], [bits], [04],
  [`p or q`], [$step(p + q)$], [bits], [04],
  [`max(a, b)`], [$b + relu(a - b)$], [always], [05],
  [`min(a, b)`], [$a - relu(a - b)$], [always], [05],
  [`abs(x)`], [$relu(x) + relu(-x)$], [always], [05],
  [`clamp(x, lo, hi)`], [$"lo" + relu(x - "lo") - relu(x - "hi")$], [$"lo" <= "hi"$], [05],
  [`x if m else 0`], [$relu(x - M(1 - m)) - relu(-x - M(1 - m))$], [$m in {0, 1}$, $|x| <= M$], [06],
  [`x if m else y`], [$y + sel(m, x - y)$], [same, $|x - y| <= M$], [06],
)

#pagebreak(weak: true)
#example("01", "relu and step", "01_foundations/01_relu_and_step.py")

== The problem

Write `1 if x >= 1 else 0` for an integer `x` without an `if`, using only $relu(x) = max(0, x)$ and arithmetic. This one gadget turns a number into a 0/1 switch, and every later example is built from it.

== relu, then step

$relu$ is flat at 0 for negative input and the line $y = x$ for positive input: it is the identity with the negative half chopped off. Slide it right by one, $relu(x - 1)$, and the kink moves from $x = 0$ to $x = 1$. Subtract the second from the first and the slopes cancel everywhere the two lines are both sloped: $x - (x - 1) = 1$.

#figure(
  grid(
    columns: (1fr, 1fr, 1fr), gutter: 6pt, align: bottom,
    plot2d(
      xs: (-3, 3), ys: (-1, 3), size: (4.4, 2.4), xlabel: $x$,
      curves: ((pts: ((-3, 0), (0, 0), (3, 3)), color: network-color),),
    ),
    plot2d(
      xs: (-3, 3), ys: (-1, 3), size: (4.4, 2.4), xlabel: $x$,
      curves: ((pts: ((-3, 0), (1, 0), (3, 2)), color: network-color),),
    ),
    plot2d(
      xs: (-3, 3), ys: (-1, 3), size: (4.4, 2.4), xlabel: $x$,
      curves: (
        (pts: ((-3, 0), (0, 0)), color: network-color),
        (pts: ((0, 0), (1, 1)), color: neg-color, dash: "dashed"),
        (pts: ((1, 1), (3, 1)), color: network-color),
      ),
      dots: range(-3, 4).map(k => (k, calc.min(1, calc.max(0, k)), ink)),
    ),
  ),
  caption: [Left to right: $relu(x)$, $relu(x - 1)$, and their difference $step(x) = relu(x) - relu(x - 1)$. The black dots are the integers; the dashed red ramp between 0 and 1 is the only place the formula is not 0 or 1, and no integer sits on it.],
)

#side-by-side(
  "A",
  [```python
def control_flow(x):
    return 1 if x >= 1 else 0
```],
  "B",
  [```python
def relu(x):
    return max(0, x)  # the primitive

def branchless(x):
    return relu(x) - relu(x - 1)
```],
)

`relu` is the one place `max` is allowed: it is the building block we are given, not a comparison we smuggled in. Everything else in stage B is `+`, `-` and calls to `relu`.

== Why it works: a table, then a proof

#let step-table(xs) = table(
  columns: xs.len() + 1,
  align: center,
  [$x$], ..xs.map(v => [#v]),
  [$relu(x)$], ..xs.map(v => [#rl(v)]),
  [$relu(x - 1)$], ..xs.map(v => [#rl(v - 1)]),
  table.cell(fill: tint(network-color))[$step(x)$], ..xs.map(v => table.cell(fill: tint(network-color))[#(rl(v) - rl(v - 1))]),
)
#step-table((-3, -2, -1, 0, 1, 2, 3, 4))

*Proof by cases.* If $x <= 0$, both relus return 0, so the difference is $0 - 0 = 0$. If $x >= 1$, both are on their sloped part, so the difference is $x - (x - 1) = 1$. The only case left is $0 < x < 1$, which contains no integer. So on integers, $step(x) = 1$ exactly when $x >= 1$.

*Moving the threshold.* To test $x >= k$, evaluate $step(x - k + 1)$. To test $a > 4$, that is $a - 4 >= 1$, evaluate $step(a - 4)$: this is exactly what example 02 does.

== What happens at 0.5 (the reason for "integers")

Feed the same formula numbers that are not integers:

#table(
  columns: 8, align: center,
  [$x$], ..(-0.5, 0, 0.25, 0.5, 0.75, 1, 1.5).map(v => [#v]),
  [$relu(x)$], ..(-0.5, 0, 0.25, 0.5, 0.75, 1, 1.5).map(v => [#rl(v)]),
  [$relu(x - 1)$], ..(-0.5, 0, 0.25, 0.5, 0.75, 1, 1.5).map(v => [#rl(v - 1)]),
  table.cell(fill: tint(neg-color))[$step(x)$], ..(-0.5, 0, 0.25, 0.5, 0.75, 1, 1.5).map(v => table.cell(fill: tint(neg-color))[#(rl(v) - rl(v - 1))]),
)

Between 0 and 1 the "switch" returns $x$ itself: $step(0.5) = 0.5$, neither "no" nor "yes". Arithmetic on its own cannot repair this: $relu$ is continuous, so any combination of relus is continuous and cannot jump from 0 to 1 at a point. With these unit weights the ramp is exactly one unit wide, and the cure is to make sure inputs never land inside it. Integers guarantee that, because the gap between consecutive integers is exactly the width of the ramp.

If your values live on a finer grid, rescale instead. For multiples of $1/4$, multiply by 4 (the weights become 4 instead of 1): $relu(4x) - relu(4x - 1)$ is 1 exactly when $x >= 1/4$. The ramp is now only $1/4$ wide and fits between grid points. `01_relu_and_step.py` prints both tables.

== Stage C: the smallest interesting network

One input, two hidden neurons, one output. Hidden neuron 0 computes $relu(x)$ (weight 1, bias 0); hidden neuron 1 computes $relu(x - 1)$ (weight 1, bias $-1$); the output neuron subtracts them.

#grid(
  columns: (1fr, auto), gutter: 14pt, align: horizon,
  [
    $ W_1 = #mx(((1,), (1,))), quad b_1 = #vc((0, -1)) $
    $ W_2 = #mx(((1, -1),)), quad b_2 = (0) $
    $ y = W_2 relu(W_1 x + b_1) + b_2 $
  ],
  net(
    (
      (nin($x$),),
      (nrelu($h_0$, bias: $0$), nrelu($h_1$, bias: $-1$)),
      (nout($y$, bias: $0$),),
    ),
    edges-of(((1,), (1,)), 0, 1) + edges-of(((1, -1),), 1, 2),
    dx: 2.6, dy: 1.2,
  ),
)

Worked trace, one column per input:

#table(
  columns: 5, align: center,
  [], [$x = -2$], [$x = 0$], [$x = 1$], [$x = 3$],
  [$W_1 x + b_1$], [$(-2, -3)$], [$(0, -1)$], [$(1, 0)$], [$(3, 2)$],
  [$h = relu(dots)$], [$(0, 0)$], [$(0, 0)$], [$(1, 0)$], [$(3, 2)$],
  [$y = h_0 - h_1$], [$0$], [$0$], [$1$], [$1$],
)

Look at the last column: neuron 0 reports 3, neuron 1 reports 2, and the output $3 - 2 = 1$ is independent of how large $x$ is. That is a *saturating* unit: the output stops growing because the second relu "catches up" with the first. We will use that shape constantly.

== What to notice

- *Where the 0/1 comes from.* Nothing in the network is a threshold. The "jump" is the difference of two slopes cancelling. Two neurons per step.
- *Cost:* 2 neurons, 1 hidden layer. Every comparison later costs this much (or shares it).
- *Pitfall:* feeding $step$ a non-integer. The code prints $0.25$ and $0.5$ without complaint. Garbage in, silent garbage out.
- *Pitfall:* off-by-one in the threshold. $step(x)$ means $x >= 1$, not $x > 0$ (these only agree on integers).

#note("Try it")[
  Build `1 if x >= 3 else 0` as a network with 1 input, 2 hidden neurons and 1 output. Which weights and biases change compared to $W_1, b_1, W_2, b_2$ above? (Answer in the last section.)
]

#pagebreak(weak: true)
#example("02", "if / else", "01_foundations/02_if_else.py")

== The problem

```python
result = 10 if a > 4 else 12        # a is an integer in [-20, 20]
```

Control flow runs one branch and skips the other. Data flow cannot skip: it evaluates *both* branches (here they are the constants 10 and 12) and turns the condition into a 0/1 number that says how much of each to keep.

#side-by-side(
  "A",
  [```python
def control_flow(a):
    if a > 4:
        return 10
    else:
        return 12
```],
  "B",
  [```python
def branchless(a):
    above4 = step(a - 4)
    return above4 * 10 + (1 - above4) * 12
```],
)

== Derivation

+ *The condition becomes a mask.* For integers, $a > 4$ means $a - 4 >= 1$, which is exactly what $step$ tests: $m = step(a - 4)$. The mask is 1 when the condition holds and 0 otherwise.
+ *Both branches are weighted.* $m dot 10 + (1 - m) dot 12$. This looks like multiplying two runtime values, but 10 and 12 are constants, so $m dot 10$ is "multiply a runtime value by a constant": allowed. If the branches were runtime values, we would have a problem; that is example 06.
+ *Simplify.* $m dot 10 + (1 - m) dot 12 = 12 - 2m$: an affine function of the mask.
+ *Expand the step.* $12 - 2 (relu(a - 4) - relu(a - 5)) = 12 - 2 relu(a - 4) + 2 relu(a - 5)$. Now it is "constant, plus weights times relus": exactly the shape of a network.

#table(
  columns: 8, align: center,
  [$a$], ..range(2, 9).map(v => [#v]),
  [$relu(a - 4)$], ..range(2, 9).map(v => [#rl(v - 4)]),
  [$relu(a - 5)$], ..range(2, 9).map(v => [#rl(v - 5)]),
  [mask $m = step(a - 4)$], ..range(2, 9).map(v => [#(rl(v - 4) - rl(v - 5))]),
  table.cell(fill: tint(network-color))[$12 - 2m$], ..range(2, 9).map(v => table.cell(fill: tint(network-color))[#(12 - 2 * (rl(v - 4) - rl(v - 5)))]),
)

#figure(
  plot2d(
    xs: (1, 8), ys: (9, 13), size: (8.5, 3.2), xlabel: $a$, ylabel: $y$,
    curves: ((pts: ((1, 12), (4, 12), (5, 10), (8, 10)), color: network-color),),
    dots: range(1, 9).map(k => (k, 12 - 2 * calc.min(1, calc.max(0, k - 4)), ink)),
  ),
  caption: [The function the network computes, for real $a$ (curve) and for integer $a$ (dots). The one-unit ramp between 4 and 5 is the same ramp as in example 01; no integer lands on it.],
)

== Stage C: the network

Read the formula $12 - 2 relu(a - 4) + 2 relu(a - 5)$ left to right and it already is a network. Two hidden neurons, one per $relu$; their incoming weights are 1, their biases are the $-4$ and $-5$ inside; the output neuron has weights $-2$, $+2$ and bias $12$.

#grid(
  columns: (1fr, auto), gutter: 14pt, align: horizon,
  [
    $ W_1 = #mx(((1,), (1,))), quad b_1 = #vc((-4, -5)) $
    $ W_2 = #mx(((-2, 2),)), quad b_2 = (12) $
    $ y = W_2 relu(W_1 a + b_1) + b_2 $
  ],
  net(
    (
      (nin($a$),),
      (nrelu($h_0$, bias: $-4$), nrelu($h_1$, bias: $-5$)),
      (nout($y$, bias: $+12$),),
    ),
    edges-of(((1,), (1,)), 0, 1) + edges-of(((-2, 2),), 1, 2),
    dx: 2.8, dy: 1.2, label-at: 0.5,
  ),
)

Worked trace for $a = 7$ (condition true, expected 10):
$ W_1 a + b_1 = #vc((7, 7)) + #vc((-4, -5)) = #vc((3, 2)) quad ==> quad h = #vc((3, 2)) quad ==> quad y = (-2)(3) + 2(2) + 12 = 10. $
and for $a = 4$ (condition false, expected 12):
$ W_1 a + b_1 = #vc((0, -1)) quad ==> quad h = #vc((0, 0)) quad ==> quad y = 0 + 0 + 12 = 12. $
and for $a = 5$, the first value above the threshold: $h = (1, 0)$, so $y = -2 + 0 + 12 = 10$.

#key-idea[
  *Below the threshold both neurons are silent (0), so the output is the bias: the "else" value 12.* Above it both fire and their difference saturates at 1, so the output is $12 - 2 = 10$, the "then" value. The bias holds the else-branch; the weights hold the difference (then − else).
]

== What to notice

- *The general pattern.* For constants $X$ (then) and $Y$ (else): $"result" = Y + (X - Y) dot step("test")$. Both branch values go into weights and the bias. The branch *code* has already run on every input; the mask only selects its result.
- *`>` versus `>=`.* $step(a - 4)$ is $a > 4$. For $a >= 4$ use $step(a - 3)$. Shift the thresholds, not the formula. Off-by-one here is the most common mistake.
- *No range needed.* The weights never mention $[-20, 20]$: the branches are constants, so nothing can overflow. (Contrast with example 06, where the branches are runtime numbers and need a bound $M$.)
- *Cost:* 1 hidden layer, 2 neurons per `if`. Chaining `elif`s adds 2 neurons each (see the exercise).

#note("Try it")[
  Turn `1 if a < 0 else 5 if a < 10 else 9` into one hidden layer. How many neurons, and what is the output formula? (Answer at the end.)
]

#pagebreak(weak: true)
#let st(x) = rl(x) - rl(x - 1)
#example("03", "comparisons as numbers", "01_foundations/03_compare.py")

== The problem

Produce all six comparisons of two integers, `a > b`, `a >= b`, `a < b`, `a <= b`, `a == b`, `a != b`, each as a 0/1 *number* that later arithmetic can use (as a mask, as a count, as a multiplier). Here $a, b in [-8, 8]$.

#side-by-side(
  "A",
  [```python
def control_flow(a, b):
    return (int(a > b), int(a >= b),
            int(a < b), int(a <= b),
            int(a == b), int(a != b))
```],
  "B",
  [```python
def branchless(a, b):
    d = a - b
    gt = step(d)
    ge = step(d + 1)
    lt = step(-d)
    le = step(-d + 1)
    eq = 1 - gt - lt
    ne = gt + lt
    return (gt, ge, lt, le, eq, ne)
```],
)

== Derivation

Only the difference $d = a - b$ matters, and for integers every comparison is a threshold on $d$ (or on $-d$):

#table(
  columns: (auto, auto, 1fr),
  table.header[*Test*][*As a threshold*][*Formula*],
  [`a > b`], [$d >= 1$], [$step(d)$],
  [`a >= b`], [$d >= 0$, i.e. $d + 1 >= 1$], [$step(d + 1)$],
  [`a < b`], [$-d >= 1$], [$step(-d)$],
  [`a <= b`], [$-d >= 0$, i.e. $-d + 1 >= 1$], [$step(-d + 1)$],
  [`a == b`], [neither `>` nor `<`], [$1 - step(d) - step(-d)$],
  [`a != b`], [`>` or `<` (never both)], [$step(d) + step(-d)$],
)

Two ideas do all the work. *Shifting* the argument of $step$ by one turns `>` into `>=`, because $step$ tests "$>= 1$" and shifting by one moves the threshold from 1 to 0. *Negating* the argument mirrors the comparison. For `==` and `!=`, the two strict inequalities cannot both be true and cannot both be false when $a = b$, so adding them counts "not equal" and subtracting from 1 gives "equal". Here is every value, with $d$ running over a few integers:

#table(
  columns: 8, align: center,
  [$d = a - b$], ..range(-3, 4).map(v => [#v]),
  [`a > b`], ..range(-3, 4).map(v => [#st(v)]),
  [`a >= b`], ..range(-3, 4).map(v => [#st(v + 1)]),
  [`a < b`], ..range(-3, 4).map(v => [#st(-v)]),
  [`a <= b`], ..range(-3, 4).map(v => [#st(-v + 1)]),
  table.cell(fill: tint(network-color))[`a == b`], ..range(-3, 4).map(v => table.cell(fill: tint(network-color))[#(1 - st(v) - st(-v))]),
  [`a != b`], ..range(-3, 4).map(v => [#(st(v) + st(-v))]),
)

#figure(
  grid(
    columns: (1fr, 1fr), gutter: 8pt, align: bottom,
    plot2d(
      xs: (-3, 3), ys: (-0.5, 1.5), size: (6.6, 2.6), xlabel: $d$, yticks: (0, 1),
      curves: (
        (pts: ((-3, 0), (0, 0), (1, 1), (3, 1)), color: network-color),
        (pts: ((-3, 0), (-1, 0), (0, 1), (3, 1)), color: branchless-color, dash: "dashed"),
      ),
      dots: range(-3, 4).map(k => (k, st(k), network-color)) + range(-3, 4).map(k => (k, st(k + 1), branchless-color)),
      texts: ((2, 0.75, text(fill: network-color)[`a > b`]), (-2, 0.35, text(fill: branchless-color)[`a >= b`])),
    ),
    plot2d(
      xs: (-3, 3), ys: (-0.5, 1.5), size: (6.6, 2.6), xlabel: $d$, yticks: (0, 1),
      curves: ((pts: ((-3, 0), (-1, 0), (0, 1), (1, 0), (3, 0)), color: control-color),),
      dots: range(-3, 4).map(k => (k, 1 - st(k) - st(-k), control-color)),
      texts: ((2, 0.5, text(fill: control-color)[`a == b`]),),
    ),
  ),
  caption: [Left: `a > b` (solid) and `a >= b` (dashed) as functions of $d = a - b$; the second is the first shifted left by one. Right: `a == b` is a "tent" of height 1 and width 2 around $d = 0$; only integers $d$ (dots) are ever fed in.],
)

== Stage C: six shared neurons

Write each $step$ out as two relus. The four steps use eight relus, but neighbouring ones coincide: $step(d) = relu(d) - relu(d - 1)$ and $step(d + 1) = relu(d + 1) - relu(d)$ share $relu(d)$. That leaves six hidden neurons, three on $d$ and three on $-d$:

$ h_0 = relu(d + 1), quad h_1 = relu(d), quad h_2 = relu(d - 1), quad h_3 = relu(-d + 1), quad h_4 = relu(-d), quad h_5 = relu(-d - 1) $

#table(
  columns: 7, align: center,
  [], [$h_0$], [$h_1$], [$h_2$], [$h_3$], [$h_4$], [$h_5$],
  [`a > b`], [], [$+$], [$-$], [], [], [],
  [`a >= b`], [$+$], [$-$], [], [], [], [],
  [`a < b`], [], [], [], [], [$+$], [$-$],
  [`a <= b`], [], [], [], [$+$], [$-$], [],
  [`a == b`], [], [$-$], [$+$], [], [$-$], [$+$],
  [`a != b`], [], [$+$], [$-$], [], [$+$], [$-$],
)

This table _is_ the output matrix $W_2$. Each comparison picks two neighbouring neurons of the same family (that is a $step$), and `==`/`!=` simply add or subtract the `>` and `<` rows (`==` also adds the constant 1).

#let W1 = ((1, -1), (1, -1), (1, -1), (-1, 1), (-1, 1), (-1, 1))
#let W2 = (
  (0, 1, -1, 0, 0, 0), (1, -1, 0, 0, 0, 0), (0, 0, 0, 0, 1, -1),
  (0, 0, 0, 1, -1, 0), (0, -1, 1, 0, -1, 1), (0, 1, -1, 0, 1, -1),
)
$ W_1 = #mx(W1), quad b_1 = #vc((1, 0, -1, 1, 0, -1)) $
$ W_2 = #mx(W2), quad b_2 = #vc((0, 0, 0, 0, 1, 0)) $

#figure(
  net(
    (
      (nin($a$), nin($b$)),
      (nrelu($d+1$), nrelu($d$), nrelu($d-1$), nrelu($-d+1$), nrelu($-d$), nrelu($-d-1$)),
      (nout($y_0$, note: [`a > b`]), nout($y_1$, note: [`a >= b`]), nout($y_2$, note: [`a < b`]), nout($y_3$, note: [`a <= b`]), nout($y_4$, bias: $+1$, note: [`a == b`]), nout($y_5$, note: [`a != b`])),
    ),
    edges-of(W1, 0, 1) + edges-of(W2, 1, 2),
    dx: 4.4, dy: 1.0, r: 0.36, label-at: 0.5,
  ),
  caption: [Two inputs, one hidden layer of six relu neurons (each labelled with the number it computes from $d = a - b$), six outputs. The biases of the hidden layer are folded into the labels: $+1, 0, -1, +1, 0, -1$.],
)

*Worked trace.* Each column is one input pair.

#table(
  columns: 4, align: center,
  [], [$(a, b) = (5, 3)$], [$(4, 4)$], [$(3, 5)$],
  [$d$], [$2$], [$0$], [$-2$],
  [$W_1 x + b_1$], [$(3, 2, 1, -1, -2, -3)$], [$(1, 0, -1, 1, 0, -1)$], [$(-1, -2, -3, 3, 2, 1)$],
  [$h$], [$(3, 2, 1, 0, 0, 0)$], [$(1, 0, 0, 1, 0, 0)$], [$(0, 0, 0, 3, 2, 1)$],
  [`>` `>=` `<` `<=` `==` `!=`], [$1, 1, 0, 0, 0, 1$], [$0, 1, 0, 1, 1, 0$], [$0, 0, 1, 1, 0, 1$],
)

For $(5, 3)$: `a > b` $= h_1 - h_2 = 2 - 1 = 1$; `a >= b` $= h_0 - h_1 = 3 - 2 = 1$; `a < b` $= h_4 - h_5 = 0$; `a == b` $= 1 - 1 - 0 = 0$.

== What to notice

- *Sharing is free.* Neurons are only worth counting once: 6 neurons produce six answers. If a later network needs several comparisons of the same difference, it should reuse $h_0, dots, h_5$ rather than adding more.
- *A shorter route to `==`.* `a == b` is `a >= b` minus `a > b`, which equals $relu(d + 1) - 2 relu(d) + relu(d - 1)$: a tent made of three neurons (the right-hand plot). The file uses $1 - "gt" - "lt"$ instead because it reads straight off the toolbox; both are right.
- *Integers only.* For $a = 0.5$, $b = 0$: `a > b` gives $0.5$ and `a == b` gives $0.5$. There is no warning: the formulas return numbers between 0 and 1.
- *Mirroring.* `a < b` is `b > a`: swap the inputs and the same neurons apply. In the network the swap is the sign flip in rows 3--5 of $W_1$.
- *Cost:* 1 hidden layer, 6 neurons, for any comparison of any range of integers (no bound needed).

#note("Try it")[
  Build `abs(a - b) <= 2` ("$a$ and $b$ are close") in a single hidden layer. How many neurons? (Answer at the end.)
]

#pagebreak(weak: true)
#example("04", "logic on bits", "01_foundations/04_logic.py")

== The problem

Compute `not p`, `p and q`, `p or q`, `p xor q` and `p implies q` for two bits $p, q in {0, 1}$ (all four inputs are checked), plus `all(bits)` and `any(bits)` for four bits (all sixteen inputs). In data flow a bit is just the number 0 or 1, so the truth tables become arithmetic.

#side-by-side(
  "A",
  [```python
def control_flow(p, q):
    return (int(not p), int(p and q),
            int(p or q), int(p != q),
            int((not p) or q))
```],
  "B",
  [```python
def branchless(p, q):
    s = p + q
    not_p = 1 - p
    or_ = step(s)
    and_ = relu(s - 1)
    nand = 1 - and_
    xor = relu(or_ + nand - 1)
    implies = 1 - relu(p - q)
    return (not_p, and_, or_,
            xor, implies)
```],
)

== Derivation: everything is a function of the sum

For bits, $s = p + q$ can only be 0, 1 or 2, and AND, OR, XOR do not care _which_ bit was set, only how many:

#let st(x) = rl(x) - rl(x - 1)
#let bits = ((0, 0), (0, 1), (1, 0), (1, 1))
#table(
  columns: 10, align: center,
  [$p$], [$q$], [$s$], [`not p`], [`p and q`], [`p or q`], [nand], [`p xor q`], [`p -> q`], [$p - q$],
  ..bits.map(((p, q)) => {
    let s = p + q
    ([#p], [#q], [#s], [#(1 - p)], [#rl(s - 1)], [#st(s)], [#(1 - rl(s - 1))], [#(st(s) - rl(s - 1))], [#(1 - rl(p - q))], [#(p - q)])
  }).flatten(),
)

+ *AND* is 1 only when $s = 2$: $relu(s - 1)$ is $0, 0, 1$ for $s = 0, 1, 2$.
+ *OR* is 1 when $s >= 1$: $step(s)$, or equivalently $relu(s) - relu(s - 1)$.
+ *NOT* is $1 - p$: it flips 0 and 1 and needs no relu.
+ *IMPLIES* ($p -> q$, "if $p$ then $q$") is false only for $p = 1, q = 0$, i.e. when $p - q = 1$. So $1 - relu(p - q)$: $p - q$ is $-1, 0, 0, 1$ and only the last survives the relu. One neuron.

#figure(
  plot2d(
    xs: (0, 2), ys: (-0.3, 1.4), size: (7.5, 2.6), xlabel: $s$, yticks: (0, 1), xticks: (0, 1, 2),
    curves: (
      (pts: ((0, 0), (1, 0), (2, 1)), color: network-color),
      (pts: ((0, 0), (1, 1), (2, 1)), color: branchless-color, dash: "dashed"),
      (pts: ((0, 0), (1, 1), (2, 0)), color: control-color),
    ),
    dots: ((0, 0, network-color), (1, 0, network-color), (2, 1, network-color), (0, 0, branchless-color), (1, 1, branchless-color), (2, 1, branchless-color), (0, 0, control-color), (1, 1, control-color), (2, 0, control-color)),
    texts: ((1.62, 0.2, text(fill: network-color)[AND]), (0.4, 0.78, text(fill: branchless-color)[OR]), (1.65, 0.72, text(fill: control-color)[XOR])),
  ),
  caption: [AND, OR and XOR of two bits, as functions of the sum $s$. AND and OR are "monotone": once $s$ is big enough the answer stays 1. XOR goes up _and back down_.],
)

== XOR is different, and why two layers helped historically

AND and OR are "$s$ above a threshold". XOR is "$s$ is between two thresholds". No single line in the $(p, q)$ plane separates the XOR-true corners from the XOR-false ones: it takes two parallel lines, one for OR (s = 0.5) and one for AND (s = 1.5), with the true corners in the band between.

#figure(
  cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let S = 3.4
    let P(x, y) = (x * S, y * S)
    // band between s = 0.5 and s = 1.5, clipped to the window [-0.3, 1.3]
    line(P(-0.3, 0.8), P(0.2, 1.3), P(1.3, 0.2), P(0.8, -0.3), close: true, fill: tint(network-color), stroke: none)
    line(P(-0.3, 0.8), P(0.8, -0.3), stroke: (paint: branchless-color, thickness: 0.8pt, dash: "dashed"))
    line(P(0.2, 1.3), P(1.3, 0.2), stroke: (paint: control-color, thickness: 0.8pt, dash: "dashed"))
    line(P(-0.3, 0), P(1.35, 0), stroke: 0.6pt + ink, mark: (end: ">", scale: 0.5))
    line(P(0, -0.3), P(0, 1.35), stroke: 0.6pt + ink, mark: (end: ">", scale: 0.5))
    content(P(1.35, 0), anchor: "west", padding: 0.1, text(size: 8pt)[$p$])
    content(P(0, 1.35), anchor: "south", padding: 0.1, text(size: 8pt)[$q$])
    for (p, q) in bits {
      let on = (p + q == 1)
      circle(P(p, q), radius: 0.17, fill: if on { network-color } else { white }, stroke: 0.9pt + network-color)
      content(P(p, q), text(size: 7pt, fill: if on { white } else { ink })[#(if on { 1 } else { 0 })])
    }
    content(P(-0.28, 0.58), anchor: "east", text(size: 7pt, fill: branchless-color)[$s = 0.5$ (OR)])
    content(P(1.3, 0.2), anchor: "west", padding: 0.1, text(size: 7pt, fill: control-color)[$s = 1.5$ (AND)])
  }),
  caption: [Truth values of XOR at the four corners of the $(p, q)$ square. The two dashed lines are the thresholds of OR and AND; XOR is true in the shaded band between them.],
)

The classic construction (older than ReLU) names the band's two edges and combines them with a second layer:

$ p xor q = (p or q) and not (p and q) = AND(OR(p, q), thin "NAND"(p, q)) $

#figure(
  diagram(
    node-stroke: 0.6pt, spacing: (24mm, 11mm), node-inset: 6pt,
    node((0, -0.5), $p$, name: <p>), node((0, 0.5), $q$, name: <q>),
    node((1, -0.5), text(size: 8pt)[OR \ $step(p + q)$], name: <or>),
    node((1, 0.5), text(size: 8pt)[NAND \ $1 - relu(p + q - 1)$], name: <nand>),
    node((2, 0), text(size: 8pt)[AND \ $relu("or" + "nand" - 1)$], name: <and>),
    node((3, 0), text(size: 8pt)[$p xor q$], name: <out>, stroke: none),
    edge(<p>, <or>, "->"), edge(<p>, <nand>, "->"), edge(<q>, <or>, "->"), edge(<q>, <nand>, "->"),
    edge(<or>, <and>, "->"), edge(<nand>, <and>, "->"), edge(<and>, <out>, "->"),
  ),
  caption: [Layer 1 names the band's edges (OR and NAND are both threshold units); layer 2 ANDs them.],
)

Check with $p = q = 1$: $"or" = 1$, $"nand" = 1 - 1 = 0$, $relu(1 + 0 - 1) = 0$. Check with $p = 1, q = 0$: $"or" = 1$, $"nand" = 1$, $relu(1 + 1 - 1) = 1$.

== Stage C: a two-hidden-layer network

The outputs of a layer are _only_ what the next layer can see, so every number needed later must be present in the previous layer's output. Two rules make that work:

- *Copy trick.* If $v >= 0$ then $relu(v) = v$, so a hidden neuron with weight 1 and bias 0 on a non-negative value copies it forward. Bits, sums and the relus themselves are all non-negative.
- *Differences are free.* $"or" = relu(s) - relu(s - 1)$ is not a neuron; it is a weighted sum of two neurons, which the next layer takes as input.

Layer 1 has four neurons, $x = (p, q)$ and $s = p + q$: $h_0 = relu(s)$, $h_1 = relu(s - 1)$ (this is AND), $h_2 = relu(p - q)$ ($p$ and not $q$) and $h_3 = relu(p)$ (a copy of $p$). Layer 2 has five: $g_0 = relu(h_0 - 2 h_1)$ is XOR, because $"or" + "nand" - 1 = (h_0 - h_1) + (1 - h_1) - 1 = h_0 - 2h_1$; the other four copy $"or"$, $"and"$, $p and not q$ and $p$ forward. The last layer only reorders them and applies $1 - dot$ for `not p` and `p -> q`.

#let W1 = ((1, 1), (1, 1), (1, -1), (1, 0))
#let W2 = ((1, -2, 0, 0), (1, -1, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
#let W3 = ((0, 0, 0, 0, -1), (0, 0, 1, 0, 0), (0, 1, 0, 0, 0), (1, 0, 0, 0, 0), (0, 0, 0, -1, 0))
$ W_1 = #mx(W1), quad b_1 = #vc((0, -1, 0, 0)) $
$ W_2 = #mx(W2), quad b_2 = vec(0, 0, 0, 0, 0) $
$ W_3 = #mx(W3), quad b_3 = #vc((1, 0, 0, 0, 1)) $

#figure(
  net(
    (
      (nin($p$), nin($q$)),
      (nrelu($s$), nrelu($s-1$), nrelu($p-q$), nrelu($p$)),
      (nrelu($g_0$, note: none), nrelu($g_1$), nrelu($g_2$), nrelu($g_3$), nrelu($g_4$)),
      (nout($y_0$, bias: $+1$, note: [not p]), nout($y_1$, note: [p and q]), nout($y_2$, note: [p or q]), nout($y_3$, note: [p xor q]), nout($y_4$, bias: $+1$, note: [p implies q])),
    ),
    edges-of(W1, 0, 1) + edges-of(W2, 1, 2) + edges-of(W3, 2, 3),
    dx: 3.0, dy: 0.95, r: 0.38, label-at: 0.5,
  ),
  caption: [Layer 1 (labelled by the quantity inside its relu) holds $s$, AND, "$p$ and not $q$" and $p$. Layer 2: $g_0$ = XOR; $g_1, dots, g_4$ copy OR, AND, "$p$ and not $q$" and $p$ forward. The output layer only reorders and negates.],
)

Worked trace for $(p, q) = (1, 0)$ and $(1, 1)$:

#table(
  columns: 3, align: center,
  [], [$(p, q) = (1, 0)$], [$(1, 1)$],
  [$W_1 x + b_1$], [$(1, 0, 1, 1)$], [$(2, 1, 0, 1)$],
  [$h$], [$(1, 0, 1, 1)$], [$(2, 1, 0, 1)$],
  [$W_2 h + b_2$], [$(1, 1, 0, 1, 1)$], [$(0, 1, 1, 0, 1)$],
  [$g$], [$(1, 1, 0, 1, 1)$], [$(0, 1, 1, 0, 1)$],
  [$y$ (not, and, or, xor, implies)], [$(0, 0, 1, 1, 0)$], [$(0, 1, 1, 0, 1)$],
)

== Shallower: XOR in one hidden layer

Because $"or" = relu(s) - relu(s - 1)$ and $"and" = relu(s - 1)$, XOR is a plain weighted sum of layer 1:
$ p xor q = "or" - "and" = relu(s) - 2 relu(s - 1). $
Check: $s = 0 arrow.r 0$; $s = 1 arrow.r 1 - 0 = 1$; $s = 2 arrow.r 2 - 2 = 0$. The tent of the plot above. So with the same four hidden neurons and a different output matrix the network needs only one hidden layer (`network_shallow` in the file):

#let Ws = ((0, 0, 0, -1), (0, 1, 0, 0), (1, -1, 0, 0), (1, -2, 0, 0), (0, 0, -1, 0))
$ W_"out" = #mx(Ws), quad b_"out" = #vc((1, 0, 0, 0, 1)) $

This works because a ReLU "threshold" is a difference of two neurons, not one: a single ReLU neuron really can't do XOR (it is monotone in $s$), but two of them in one layer can. The two-layer version is still worth knowing: it is the pattern for _any_ function of several decisions, where layer 2 reads the 0/1 decisions produced by layer 1.

== n bits

For $n$ bits let $s$ be their sum, from 0 to $n$. All bits are 1 iff $s = n$; at least one is 1 iff $s >= 1$:

$ AND(b_1, dots, b_n) = relu(s - (n - 1)), quad OR(b_1, dots, b_n) = step(s) = relu(s) - relu(s - 1). $

More generally "at least $k$ of the $n$ bits are 1" is $step(s - k + 1)$: $k = 1$ is OR, $k = n$ is AND, $k = 2$ with $n = 3$ is the majority. For $n = 4$:

#table(
  columns: 6, align: center,
  [$s$], [0], [1], [2], [3], [4],
  [$relu(s - 3)$ = AND], ..range(0, 5).map(v => [#rl(v - 3)]),
  [$step(s)$ = OR], ..range(0, 5).map(v => [#st(v)]),
)

#let Wn1 = ((1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1))
#let Wn2 = ((0, 0, 1), (1, -1, 0))
#grid(
  columns: (1fr, auto), gutter: 12pt, align: horizon,
  [
    $ W_1 = #mx(Wn1), quad b_1 = #vc((0, -1, -3)) $
    $ W_2 = #mx(Wn2), quad b_2 = vec(0, 0) $
  ],
  net(
    (
      (nin($b_0$), nin($b_1$), nin($b_2$), nin($b_3$)),
      (nrelu($s$), nrelu($s-1$), nrelu($s-3$)),
      (nout($y_0$, note: [all]), nout($y_1$, note: [any])),
    ),
    edges-of(Wn1, 0, 1) + edges-of(Wn2, 1, 2),
    dx: 2.6, dy: 0.9,
  ),
)

In stage B the sum is `s = 0; for i in range(N): s = s + bits[i]`: the loop count `N` is a constant, not data.

== What to notice

- *Bits must be exactly 0 or 1.* With $p = 2$ the formula for `not p` returns $-1$ and `p and q` is no longer a bit. These formulas are only correct on their domain, and nothing in the network checks the domain.
- *AND and OR are one neuron-pair; XOR is the first function that needs a "band".* More generally, any function of the sum that goes up and comes down needs more than two neurons.
- *Depth versus width.* The deep XOR (3 layers of weights) and the shallow XOR (2) compute the same function. Depth lets later layers reuse earlier decisions; width lets you flatten it. Both are fine for us; later chapters use depth freely because each loop iteration is another layer.
- *Pass-through costs neurons.* Copying a value across a layer costs one neuron per value (when it is $>= 0$). The deep network above needs 4 copy neurons for that.

#note("Try it")[
  `majority(b0, b1, b2)` is 1 when at least two of three bits are 1. Give the formula and the hidden neurons (one hidden layer). Then say what changes for "at least 3 of 5". (Answer at the end.)
]

#pagebreak(weak: true)
#example("05", "max, min, abs, clamp, and stacking them", "01_foundations/05_min_max_abs_clamp.py")

== The problem

Part 1: compute `max(a, b)`, `min(a, b)`, `abs(a - b)` and `clamp(a, -3, 4)` in one network, for integers $a, b in [-8, 8]$. Part 2: the median of three numbers, $a, b, c in [-6, 6]$, to see what happens when helpers are _nested_: that is where depth comes from.

#side-by-side(
  "A",
  [```python
def control_flow(a, b):
    return (max(a, b), min(a, b),
            abs(a - b),
            min(max(a, LO), HI))

def median3_control_flow(a, b, c):
    return sorted([a, b, c])[1]
```],
  "B",
  [```python
def max2(a, b): return b + relu(a - b)
def min2(a, b): return a - relu(a - b)
def abs1(x):    return relu(x) + relu(-x)
def clamp(x, lo, hi):
    return lo + relu(x - lo) - relu(x - hi)

def median3_branchless(a, b, c):
    return max2(min2(a, b),
                min2(max2(a, b), c))
```],
)

== Derivation of the four helpers

All four are built from a single $relu$ of a difference, so none of them uses $step$. They are exact for any numbers, not just integers.

+ *max.* $max(a, b) = b + relu(a - b)$. If $a >= b$ the relu returns $a - b$ and the sum is $a$. If $a < b$ it returns 0 and the sum is $b$. Geometrically: start at $b$, and "go up by the amount $a$ exceeds $b$, if it does".
+ *min.* $min(a, b) = a - relu(a - b)$. If $a >= b$ we subtract the excess and land on $b$; otherwise $a$ stays. (Check: $max + min = a + b$ always.)
+ *abs.* $|x| = relu(x) + relu(-x)$. Exactly one of the two is non-zero, and it is $|x|$.
+ *clamp.* $"clamp"(x, "lo", "hi") = "lo" + relu(x - "lo") - relu(x - "hi")$, valid for $"lo" <= "hi"$. Below $"lo"$ both relus are 0, so the result is $"lo"$. Between them the first relu is $x - "lo"$ and the second is 0, so the result is $x$. Above $"hi"$ the second relu starts subtracting at slope 1 and cancels the first one's slope exactly, freezing the result at $"lo" + ("hi" - "lo") = "hi"$. (This is the same shape as $step$: a "ramp", then flat. In fact $step = "clamp"(x, 0, 1)$.)

#figure(
  grid(
    columns: (1fr, 1fr), gutter: 10pt, row-gutter: 10pt, align: bottom,
    plot2d(
      xs: (-4, 6), ys: (-4, 6), size: (6.4, 2.9), xlabel: $a$, yticks: (-4, -2, 2, 4, 6),
      curves: ((pts: ((-4, 2), (2, 2), (6, 6)), color: network-color),),
      texts: ((-1.5, 3.4, [$max(a, 2)$]),),
    ),
    plot2d(
      xs: (-4, 6), ys: (-4, 6), size: (6.4, 2.9), xlabel: $a$, yticks: (-4, -2, 2, 4, 6),
      curves: ((pts: ((-4, -4), (2, 2), (6, 2)), color: branchless-color),),
      texts: ((-2.0, 3.0, [$min(a, 2)$]),),
    ),
    plot2d(
      xs: (-4, 6), ys: (-4, 6), size: (6.4, 2.9), xlabel: $x$, yticks: (-4, -2, 2, 4, 6),
      curves: ((pts: ((-4, 4), (0, 0), (6, 6)), color: control-color),),
      texts: ((-2.2, -1.5, [$|x|$]),),
    ),
    plot2d(
      xs: (-4, 6), ys: (-4, 6), size: (6.4, 2.9), xlabel: $x$, yticks: (-4, -2, 2, 4, 6),
      curves: ((pts: ((-4, -3), (-3, -3), (4, 4), (6, 4)), color: neg-color),),
      texts: ((-1.6, 3.0, [clamp$(x, -3, 4)$]),),
    ),
  ),
  caption: [The four helpers as functions of one variable (the other fixed at $b = 2$ for max and min). Each kink is a $relu$: one for max, min and abs; two for clamp.],
)

== Stage C part 1: one hidden layer, with "carried" values

The network sees $x = (a, b)$ and must output four numbers. Notice what the formulas need: $max = b + relu(a - b)$ contains a bare $b$, and $min$ contains a bare $a$. The output layer only sees hidden neurons, never the input itself, so $b$ and $a$ have to be _carried_ through the hidden layer. The identity that carries any signed value:

$ x = relu(x) - relu(-x) $

costs two neurons per value (one if the value is known to be non-negative; see part 2 for a cheaper trick). So eight hidden neurons:

#let W1 = ((1, -1), (-1, 1), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 0), (1, 0))
#let W2 = (
  (1, 0, 0, 0, 1, -1, 0, 0), (-1, 0, 1, -1, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 1, -1),
)
#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*neuron*][*computes*][*used for*],
  [$h_0$, $h_1$], [$relu(a - b)$, $relu(b - a)$], [$h_0$: max, min, $|a - b|$. $h_1$: only $|a - b|$],
  [$h_2$, $h_3$], [$relu(a)$, $relu(-a)$], [carry $a = h_2 - h_3$, for min],
  [$h_4$, $h_5$], [$relu(b)$, $relu(-b)$], [carry $b = h_4 - h_5$, for max],
  [$h_6$, $h_7$], [$relu(a + 3)$, $relu(a - 4)$], [clamp: $-3 + h_6 - h_7$],
)
$ W_1 = #mx(W1), quad b_1 = #vc((0, 0, 0, 0, 0, 0, 3, -4)) $
$ W_2 = #mx(W2), quad b_2 = #vc((0, 0, 0, -3)) $

#figure(
  net(
    (
      (nin($a$), nin($b$)),
      (nrelu($h_0$), nrelu($h_1$), nrelu($h_2$), nrelu($h_3$), nrelu($h_4$), nrelu($h_5$), nrelu($h_6$, bias: $+3$), nrelu($h_7$, bias: $-4$)),
      (nout($y_0$, note: [max]), nout($y_1$, note: [min]), nout($y_2$, note: [$|a - b|$]), nout($y_3$, bias: $-3$, note: [clamp])),
    ),
    edges-of(W1, 0, 1) + edges-of(W2, 1, 2),
    dx: 4.2, dy: 0.78, r: 0.28, label-at: 0.5,
  ),
  caption: [Part 1 network. $h_0$ is shared by three outputs; the pairs $(h_2, h_3)$ and $(h_4, h_5)$ only carry $a$ and $b$ to the output.],
)

#block(sticky: true)[Worked trace, $(a, b) = (7, -2)$:]

#table(
  columns: 2, align: (left, left),
  [$W_1 x + b_1$], [$(9, -9, 7, -7, -2, 2, 10, 3)$],
  [$h = relu(dots)$], [$(9, 0, 7, 0, 0, 2, 10, 3)$],
  [max $= h_0 + h_4 - h_5$], [$9 + 0 - 2 = 7$],
  [min $= -h_0 + h_2 - h_3$], [$-9 + 7 - 0 = -2$],
  [$|a - b| = h_0 + h_1$], [$9 + 0 = 9$],
  [clamp $= -3 + h_6 - h_7$], [$-3 + 10 - 3 = 4$],
)

== Part 2: composition is stacking layers

The median of three is "the one in the middle": drop the smallest and the largest. With helpers, in the order a human would write it:

$ "median"(a, b, c) = max(min(a, b), thin min(max(a, b), c)) $

Each helper is one $relu$, and the inner helpers feed the outer ones. Name what each helper produces:

#table(
  columns: (auto, 1fr, auto),
  align: (center, left, center),
  table.header[*step*][*quantity*][*depends on*],
  [1], [$u = relu(a - b)$, so $"hi" = max(a, b) = b + u$, $"lo" = min(a, b) = a - u$], [inputs],
  [2], [$v = relu("hi" - c)$, so $"mid" = min("hi", c) = "hi" - v$], [step 1],
  [3], [$w = relu("lo" - "mid")$, so $"median" = "mid" + w$], [steps 1, 2],
)

A relu that takes another relu as input is a second layer. The nesting $u arrow.r v arrow.r w$ is three relus deep, so the network has *three hidden layers*, no matter how it is arranged. That is the rule: *composition of helpers = stacking layers; depth equals the nesting depth of relus.* Two simplifications make the arithmetic leaner. First, $"lo" - "mid" = "lo" - "hi" + v = v - d$ where $d = "hi" - "lo" = |a - b| = u + u'$ and $u' = relu(b - a)$ (so $w = relu(v - d)$). Second, $"median" = "hi" - v + w$.

*Carrying with an offset.* Everything a later layer needs must exist in the previous layer. We need $b$ (for $"hi" = b + u$) and $c$ (for $v$) in layer 2. A pair $relu(x) - relu(-x)$ would cost two neurons each; instead, since $a, b, c in [-S, S]$ with $S = 6$, shift: $relu(b + S) = b + S$ is one neuron and never clips. The offset is paid back in the bias of the last layer. Write $B' = b + S$, $C' = c + S$, $H' = "hi" + S$, $m' = "mid" + S$.

#table(
  columns: (auto, 1fr),
  align: (center, left),
  [layer 1], [$u = relu(a - b)$, $u' = relu(b - a)$, $B' = relu(b + S)$, $C' = relu(c + S)$],
  [layer 2], [$v = relu(B' + u - C')$ #h(1em) ($= relu("hi" - c)$),  $d = relu(u + u')$ #h(1em) ($= |a - b|$), $H' = relu(B' + u)$ #h(1em) ($= "hi" + S$)],
  [layer 3], [$w = relu(v - d)$, $m' = relu(H' - v)$ #h(1em) ($= "mid" + S$)],
  [output], [$y = m' + w - S$ #h(1em) ($= "mid" + w = "median"$)],
)

#let W1 = ((1, -1, 0), (-1, 1, 0), (0, 1, 0), (0, 0, 1))
#let W2 = ((1, 0, 1, -1), (1, 1, 0, 0), (1, 0, 1, 0))
#let W3 = ((1, -1, 0), (-1, 0, 1))
$ W_1 = #mx(W1), quad b_1 = #vc((0, 0, 6, 6)) $
$ W_2 = #mx(W2), quad b_2 = vec(0, 0, 0) $
$ W_3 = #mx(W3), quad b_3 = vec(0, 0), quad W_4 = #mx(((1, 1),)), quad b_4 = (-6) $

#figure(
  net(
    (
      (nin($a$), nin($b$), nin($c$)),
      (nrelu($u$), nrelu($u'$), nrelu($B'$, bias: $+6$), nrelu($C'$, bias: $+6$)),
      (nrelu($v$), nrelu($d$), nrelu($H'$)),
      (nrelu($w$), nrelu($m'$)),
      (nout($y$, bias: $-6$),),
    ),
    edges-of(W1, 0, 1) + edges-of(W2, 1, 2) + edges-of(W3, 2, 3) + edges-of(((1, 1),), 3, 4),
    dx: 2.9, dy: 0.95, label-at: 0.5,
  ),
  caption: [Median of three: 3 inputs, hidden layers of 4, 3 and 2 neurons, one output. Three relus deep because the nesting is three deep.],
)

Worked traces:

#table(
  columns: 4, align: center,
  [$(a, b, c)$], [layer 1 $(u, u', B', C')$], [layer 2 $(v, d, H')$], [layer 3 $(w, m')$ #sym.arrow $y$],
  [$(4, 5, -2)$], [$(0, 1, 11, 4)$], [$(7, 1, 11)$], [$(6, 4) arrow.r 6 + 4 - 6 = 4$],
  [$(1, 5, 3)$], [$(0, 4, 11, 9)$], [$(2, 4, 11)$], [$(0, 9) arrow.r 0 + 9 - 6 = 3$],
  [$(5, 1, 3)$], [$(4, 0, 7, 9)$], [$(2, 4, 11)$], [$(0, 9) arrow.r 3$],
)

In $(4, 5, -2)$: $"hi" = 5$, $"lo" = 4$, $c = -2$. $v = relu(5 - (-2)) = 7$ so $"mid" = 5 - 7 = -2$. Then $w = relu("lo" - "mid") = relu(4 + 2) = 6$ and $"median" = -2 + 6 = 4$.

== What to notice

- *No integers needed in part 1.* max, min, abs and clamp use $relu$ directly. Only the offset trick (part 2) and the specific ranges care about bounds.
- *Carrying has a price, and two recipes.* The pair $x = relu(x) - relu(-x)$ needs no bound on $x$ and costs 2 neurons per value. The offset $relu(x + S) = x + S$ costs 1 neuron but only works if $x >= -S$; a value outside the range gets clipped silently. Try $(a, b, c) = (0, -8, 3)$ with $S = 6$: $B' = relu(-8 + 6) = 0$ instead of $-2$, and the network returns 2 instead of the true median 0. (The file does not assert outside $[-S, S]$.)
- *Shared neurons.* $h_0$ in part 1 feeds three outputs. Whenever two formulas contain the same $relu(dots)$, they should share the neuron.
- *Depth = nesting.* max-of-4 by a chain `max(max(max(a,b),c),d)` would be 3 layers deep; by a balanced tree `max(max(a,b), max(c,d))` it is 2. Same function, same number of neurons-ish, less depth.

#note("Try it")[
  Build `max(a, b, c)` as a network. How many hidden layers, and which neurons must be carried? (Hint: $max(a, b, c) = c + relu(max(a, b) - c)$. Answer at the end.)
]

#pagebreak(weak: true)
#example("06", "select: a 0/1 mask times a runtime value", "01_foundations/06_select.py")

== The problem

$ x "if" m "else" 0, quad m in {0, 1}, quad x "a runtime number" $

The obvious data-flow answer is $m dot x$. It is exactly the thing we cannot write. In the examples so far, the values we multiplied by a mask (10 and 12 in example 02) were constants. Now $x$ is itself something the network computes. This tiny example makes masks work on runtime values, and that unlocks everything later: loops that stop, arrays that are written, BFS frontiers.

=== Why `m * x` is forbidden

A layer computes $relu(W x + b)$. Every number going into a $relu$ is a sum of runtime values, each multiplied by a *constant* written in $W$. A product of two runtime values is a different kind of operation. You can see the difference geometrically: a relu network is a _piecewise linear_ function of its inputs, made of flat pieces and kinks. The product $m dot x$ along the line $m = x$ is $x^2$, a parabola, which curves everywhere. No stack of relus reproduces it. (Networks can only approximate it, and we want exactness.)

We do not need the whole product, though. We need it only for $m in {0, 1}$. At two values of $m$ there is plenty of freedom to pick a piecewise linear function that matches. Big-M is one such choice.

#side-by-side(
  "A",
  [```python
def control_flow(m, n, x, y):
    return (n if m else 0,
            x if m else 0,
            x if m else y)
```],
  "B",
  [```python
def select(m, x, big_m):
    shift = big_m * (1 - m)
    return relu(x - shift) - relu(-x - shift)

def branchless(m, n, x, y):
    return (select_nonneg(m, n, M),
            select(m, x, M),
            y + select(m, x - y, M2))
```],
)

Stage B never writes `m * x`. `big_m * (1 - m)` is a runtime value ($m$) times a constant (`big_m`): allowed.

== Derivation of the big-M gate

+ *Start with a non-negative $x$ in $[0, M]$.* We want $g(1, x) = x$ and $g(0, x) = 0$. A single relu already gives the first: $relu(x) = x$. For the second, push the kink of the relu to the right: $relu(x - M) = 0$ for every $x <= M$. So the mask should shift the relu by 0 when $m = 1$ and by $M$ when $m = 0$: the shift $M(1 - m)$ does that, and it is affine in $m$.
  $ "select"^+(m, x) = relu(x - M(1 - m)) = relu(x + M m - M). $
  As a neuron: weight 1 on $x$, weight $M$ on $m$, bias $-M$. Weight times runtime input, nothing else: legal.
+ *Signed $x$ in $[-M, M]$.* Split $x = relu(x) - relu(-x)$ and gate both halves with the same shift:
  $ sel(m, x) = relu(x - M(1 - m)) - relu(-x - M(1 - m)). $
  For $m = 1$ the shift is 0 and this is $relu(x) - relu(-x) = x$. For $m = 0$ the shift is $M$, so the first term is $relu(x - M)$ with $x - M <= 0$, the second is $relu(-x - M)$ with $-x - M <= 0$ (both because $|x| <= M$). Both vanish.
+ *Check the numbers* with $M = 6$:

#let selv(m, x, M) = rl(x - M * (1 - m)) - rl(-x - M * (1 - m))
#table(
  columns: 6, align: center,
  [$x$], ..(-6, -5, 0, 3, 6).map(v => [#v]),
  [$m = 1$: $relu(x) - relu(-x)$], ..(-6, -5, 0, 3, 6).map(v => [#selv(1, v, 6)]),
  [$m = 0$: $relu(x - 6) - relu(-x - 6)$], ..(-6, -5, 0, 3, 6).map(v => [#selv(0, v, 6)]),
)

#figure(
  plot2d(
    xs: (-2, 12), ys: (-2, 8), size: (9.5, 3.4), xlabel: $x$, xticks: (0, 2, 4, 6, 8, 10, 12), yticks: (2, 4, 6, 8),
    curves: (
      (pts: ((-2, 0), (0, 0), (8, 8)), color: network-color),
      (pts: ((-2, 0), (6, 0), (12, 6)), color: neg-color),
      (pts: ((0, -2), (0, 8)), color: muted, dash: "dotted"),
      (pts: ((6, -2), (6, 8)), color: muted, dash: "dotted"),
    ),
    texts: ((2.2, 4.6, text(fill: network-color)[$m = 1$: $relu(x)$]), (9.7, 1.2, text(fill: neg-color)[$m = 0$: $relu(x - 6)$]), (3, 7.3, text(fill: muted)[allowed range $x in [0, M]$])),
  ),
  caption: [The mask slides the kink. For $x$ in $[0, M]$ (between the dotted lines) the red curve ($m = 0$) is exactly zero and the green curve ($m = 1$) is the identity. The big-M rule is "the kink must move out beyond the largest $x$ that can occur".],
)

=== Two runtime values: `x if m else y`

For $m = 1$ we want $x$, for $m = 0$ we want $y$. Write it as "start at $y$ and add the difference $x - y$ only when $m = 1$":

$ x "if" m "else" y = y + sel(m, x - y). $

The select now acts on $x - y$, so its bound must cover the difference: $|x - y| <= M$. With $x, y in [-6, 6]$ that is 12, not 6. (The file calls it `M2`.) The network also needs $y$ at the output, so $y = relu(y) - relu(-y)$ is carried by two more neurons (example 05).

An alternative needs no carrying and a tighter bound: $sel(m, x) + sel(1 - m, y)$. Each select only sees one value, so $M = 6$, and $1 - m$ is an affine function of $m$ (still just a weight and a bias). Both versions are checked in the file. We use the first in stage C because it is the toolbox formula.

== When $M$ is too small

The bound is a promise: the formula is only correct when $|x| <= M$. Here is what breaks it. Take $M = 3$ and feed $x = 5$ with the mask *off* ($m = 0$):

$ relu(5 - 3) - relu(-5 - 3) = 2 - 0 = 2 quad ("expected" 0). $

The relu was shifted by $M = 3$, but the input exceeds it, so the "off" gate still lets $x - M = 2$ through. With $x = -5$ you get $-2$. The mask _leaks_ by $|x| - M$. With the mask on ($m = 1$) the shift is 0 and the result is always right, so tests with $m = 1$ will not catch this bug.

#figure(
  plot2d(
    xs: (-8, 8), ys: (-6, 6), size: (9, 3.2), xlabel: $x$, xticks: (-8, -6, -4, -2, 2, 4, 6, 8), yticks: (-4, -2, 2, 4),
    curves: (
      (pts: ((-8, -5), (-3, 0), (3, 0), (8, 5)), color: neg-color),
      (pts: ((-8, 0), (8, 0)), color: network-color, dash: "dashed"),
    ),
    dots: ((5, 2, neg-color), (-5, -2, neg-color)),
    texts: ((-4.5, 2.5, text(fill: network-color)[correct for $|x| <= M = 3$]), (5.8, 3.5, text(fill: neg-color)[leak]), (-5.9, -3.5, text(fill: neg-color)[leak])),
  ),
  caption: [$sel(0, x)$ with $M = 3$ (red) should be the dashed zero line. Inside $[-3, 3]$ it is. Outside it leaks $x - 3$ or $x + 3$. The dots mark $x = plus.minus 5$.],
)

The same mistake in the if/else form: with $x = 6$, $y = -6$, $m = 0$ and $M = 6$ (forgetting that $|x - y|$ can be 12): $y + sel(0, 12) = -6 + (relu(12 - 6) - relu(-12 - 6)) = -6 + 6 = 0$, but the answer is $y = -6$.

What about $M$ too _big_? The arithmetic is still exact for integers, so correctness survives, but all values pass through numbers of size $M$. In float64, an $M$ like $10^16$ erases the small values: for $m = 1$, $x = 1$ the neuron computes $relu(1 + 10^16 dot 1 - 10^16)$, and $1 + 10^16$ rounds back to $10^16$ (spacing of floats at this size is 2), giving 0 instead of 1. So pick the *tightest* bound you can prove. Derive it, don't guess:

+ if $x$ is an input in $[-6, 6]$: $M = 6$;
+ if it is a difference of two such values: $M = 12$;
+ if it is a sum of values with bounds $M_1, M_2$: $M = M_1 + M_2$ (bounds add); a value multiplied by a constant $k$ gets $M dot |k|$;
+ if it is itself the output of a select: it lies in $[-M, M]$ (or $0$), the same $M$ as before.

== Stage C: the network

The smallest piece is the signed select, $m$ and $x$ in, two neurons, one output:

#grid(
  columns: (1fr, auto), gutter: 14pt, align: horizon,
  [
    $ W_1 = #mx(((6, 1), (6, -1))), quad b_1 = #vc((-6, -6)) $
    $ W_2 = #mx(((1, -1),)), quad b_2 = (0) $
  ],
  net(
    (
      (nin($m$), nin($x$)),
      (nrelu($h_0$, bias: $-6$), nrelu($h_1$, bias: $-6$)),
      (nout($y$),),
    ),
    edges-of(((6, 1), (6, -1)), 0, 1) + edges-of(((1, -1),), 1, 2),
    dx: 2.8, dy: 1.2, label-at: 0.4,
  ),
)

The file stacks all three variants into one network with four inputs $(m, n, x, y)$ and seven hidden neurons, where $n in [0, 6]$ is the non-negative case. $M = 6$ and $M_2 = 12$:

#let W1 = ((6, 1, 0, 0), (6, 0, 1, 0), (6, 0, -1, 0), (12, 0, 1, -1), (12, 0, -1, 1), (0, 0, 0, 1), (0, 0, 0, -1))
#let W2 = ((1, 0, 0, 0, 0, 0, 0), (0, 1, -1, 0, 0, 0, 0), (0, 0, 0, 1, -1, 1, -1))
$ W_1 = #mx(W1), quad b_1 = #vc((-6, -6, -6, -12, -12, 0, 0)) $
$ W_2 = #mx(W2), quad b_2 = vec(0, 0, 0) $

#figure(
  net(
    (
      (nin($m$), nin($n$), nin($x$), nin($y$)),
      (
        nrelu($h_0$, bias: $-6$), nrelu($h_1$, bias: $-6$), nrelu($h_2$, bias: $-6$), nrelu($h_3$, bias: $-12$),
        nrelu($h_4$, bias: $-12$), nrelu($h_5$), nrelu($h_6$),
      ),
      (nout($o_0$, note: [n if m else 0]), nout($o_1$, note: [x if m else 0]), nout($o_2$, note: [x if m else y])),
    ),
    edges-of(W1, 0, 1) + edges-of(W2, 1, 2),
    dx: 4.4, dy: 1.0, label-at: 0.4,
  ),
  caption: [$h_0$: non-negative select of $n$. $(h_1, h_2)$: signed select of $x$. $(h_3, h_4)$: signed select of $x - y$ with $M_2 = 12$. $(h_5, h_6)$: carry $y$. The mask $m$ reaches five neurons through its weights $6$ or $12$.],
)

Worked trace for $(m, n, x, y) = (1, 4, -5, 2)$ and for the same values with $m = 0$:

#table(
  columns: 3, align: center,
  [], [$m = 1$], [$m = 0$],
  [$W_1 v + b_1$], [$(4, -5, 5, -7, 7, 2, -2)$], [$(-2, -11, -1, -19, -5, 2, -2)$],
  [$h = relu(dots)$], [$(4, 0, 5, 0, 7, 2, 0)$], [$(0, 0, 0, 0, 0, 2, 0)$],
  [$o_0 = h_0$], [$4$], [$0$],
  [$o_1 = h_1 - h_2$], [$0 - 5 = -5$], [$0$],
  [$o_2 = h_3 - h_4 + h_5 - h_6$], [$0 - 7 + 2 - 0 = -5$], [$0 - 0 + 2 - 0 = 2$],
)

With the mask on: $o = (4, -5, -5)$, that is, $n$, $x$ and $x$. With the mask off: $o = (0, 0, 2)$, that is, 0, 0 and $y$.

== What to notice

- *Masks shift thresholds.* The mask never multiplies anything. It moves the point at which each relu turns on, far out of range when $m = 0$. This one idea is the whole mechanism for "doing something only if..." in every later chapter.
- *The mask lands in the weights.* Weight $M$ on the $m$ input, bias $-M$. If $m$ is not an input but the result of an earlier layer, the same weights sit on the previous layer's output. The mask must be an exact 0/1 number, which is why example 01 insisted on integers.
- *Cost:* non-negative select 1 neuron; signed select 2; two-value if/else 4 (2 + 2 for carrying $y$). $M$ is a bound you have to prove, not tune.
- *Pitfall: only half the cases catch a bad $M$.* A too-small $M$ is invisible when $m = 1$. Always test the mask-off case at the extremes of the range.
- *Pitfall: bounds on differences and sums.* $|x - y|$ can be twice as big as $|x|$. Compute $M$ for the expression inside the select.

#note("Try it")[
  Write `x if m else 100` for $x in [0, 50]$ as a network with as few hidden neurons as you can. Which value must be carried, and what is $M$? (Answer below.)
]

#pagebreak(weak: true)
= Answers to the exercises

*01 · `x >= 3`.* Shift the threshold: $step(x - 2) = relu(x - 2) - relu(x - 3)$. Only the first-layer bias changes: $b_1 = (-2, -3)$; $W_1 = (1, 1)^T$, $W_2 = (1, -1)$, $b_2 = 0$. Check $x = 2$: hidden $(0, 0)$, output 0. $x = 3$: hidden $(1, 0)$, output 1. $x = 5$: hidden $(3, 2)$, output 1.

*02 · `1 if a < 0 else 5 if a < 10 else 9`.* Start at the first branch's value and add a jump at each boundary. The jump from 1 to 5 happens at $a >= 0$ (that is $step(a + 1)$). The jump from 5 to 9 happens at $a >= 10$ (that is $step(a - 9)$):
$ y = 1 + 4 step(a + 1) + 4 step(a - 9) = 1 + 4 (relu(a + 1) - relu(a)) + 4 (relu(a - 9) - relu(a - 10)). $
Four hidden neurons, biases $(1, 0, -9, -10)$, output weights $(4, -4, 4, -4)$, output bias 1. Check: $a = -1$: all four are 0, $y = 1$. $a = 0$: $(1, 0, 0, 0)$, $y = 5$. $a = 10$: $(11, 10, 1, 0)$, $y = 1 + 4 + 4 = 9$.

*03 · `abs(a - b) <= 2`.* With $d = a - b$, "$d >= -2$" is $step(d + 3)$ and "$d <= 2$" is $step(3 - d)$. They are never both 0, so the AND is just their sum minus 1: $y = step(d + 3) + step(3 - d) - 1$. Four neurons: $relu(d + 3), relu(d + 2), relu(3 - d), relu(2 - d)$, output $h_0 - h_1 + h_2 - h_3 - 1$. Check: $d = 0$: $3 - 2 + 3 - 2 - 1 = 1$; $d = 3$: $(6 - 5) + (0 - 0) - 1 = 0$; $d = -3$: $(0 - 0) + (6 - 5) - 1 = 0$.

*04 · majority of three.* $s = b_0 + b_1 + b_2$. "At least 2" is $s >= 2$, i.e. $step(s - 1) = relu(s - 1) - relu(s - 2)$: two hidden neurons, each with weights $(1, 1, 1)$ and biases $-1$ and $-2$. For "at least 3 of 5": five inputs, weights $(1, 1, 1, 1, 1)$, $step(s - 2) = relu(s - 2) - relu(s - 3)$, so biases $-2, -3$.

*05 · max of three.* $"hi" = max(a, b) = b + u$ with $u = relu(a - b)$, and $max(a, b, c) = c + relu("hi" - c)$. Layer 1: $u = relu(a - b)$, $B' = relu(b + S)$, $C' = relu(c + S)$. Layer 2: $v = relu(B' + u - C')$ and $C'' = relu(C')$ (the carried copy of $c$). Output: $y = C'' + v - S$. Two hidden layers (two nested relus), 3 + 2 neurons. Carried: $b$ (once, into layer 2) and $c$ (through both layers). Check $(a, b, c) = (1, 5, 3)$ with $S = 6$: $u = 0$, $B' = 11$, $C' = 9$; $v = relu(11 + 0 - 9) = 2$, $C'' = 9$; $y = 9 + 2 - 6 = 5$.

*06 · `x if m else 100`, $x in [0, 50]$.* The else-branch is a constant, so it is affine in $m$: $x "if" m "else" 100 = relu(x - 50(1 - m)) + 100 (1 - m)$. The relu is the non-negative select with $M = 50$. But the output layer needs $m$ itself to form $100(1 - m)$, and $m >= 0$, so carry it with one more neuron. Two hidden neurons: $h_0 = relu(x + 50 m - 50)$, $h_1 = relu(m)$. Output: $y = h_0 - 100 h_1 + 100$. Check $x = 30$: $m = 1$: $h = (30, 1)$, $y = 30 - 100 + 100 = 30$. $m = 0$: $h = (0, 0)$, $y = 100$.
