# basic_python: control flow vs data flow

Small programs, each written three ways, with no framework in between:

| Stage | What it is | Allowed |
|---|---|---|
| **A · control flow** | Normal Python | Anything |
| **B · branchless arithmetic** | Scalar Python that never branches on a value | `+`, `-`, multiply by a **constant**, `relu`, loops with a **fixed** count |
| **C · network** | numpy "neural net" with hand-written weights | constant matrix `@` vector, `+`, `relu`, slicing / concatenation, fixed loops |

Never allowed in B and C: `if` on a runtime value, comparisons, `min`/`max`/`abs`/`argmax` on runtime values, multiplying two runtime values together, loops whose length depends on the input.

Every file checks that A, B and C agree on **every** input in a small integer range (or, when that's too many, on a fixed random sample plus hand-picked cases).

```sh
cd basic_python
uv run 01_foundations/02_if_else.py   # one example
./run_all.sh                          # every example
./build_booklets.sh                   # booklet/*.typ -> booklet/*.pdf
```

## The toolbox

Everything is built from `relu(x) = max(0, x)`. Inputs are **integers** in a known range, which is what makes these exact.

| Want | Data-flow formula | Holds when |
|---|---|---|
| `x >= 1` as 0/1 | `step(x) = relu(x) - relu(x - 1)` | `x` is an integer |
| `a > b` | `step(a - b)` | integers |
| `a >= b` | `step(a - b + 1)` | integers |
| `a == b` | `1 - step(a - b) - step(b - a)` | integers |
| `not p` | `1 - p` | `p` is 0/1 |
| `p and q` | `relu(p + q - 1)` | 0/1 |
| `p or q` | `step(p + q)` | 0/1 |
| `max(a, b)` | `b + relu(a - b)` | always |
| `min(a, b)` | `a - relu(a - b)` | always |
| `abs(x)` | `relu(x) + relu(-x)` | always |
| `clamp(x, lo, hi)` | `lo + relu(x - lo) - relu(x - hi)` | `lo <= hi` |
| `x if m else 0` | `relu(x - M*(1 - m)) - relu(-x - M*(1 - m))` | `m` is 0/1, `abs(x) <= M` |
| `x if m else y` | `y + select(m, x - y)` | as above, with `abs(x - y) <= M` |

The last two rows ("big-M select") are how a 0/1 mask multiplies a runtime value without ever multiplying two runtime values. That's what keeps stage C pure matmul + ReLU.

## Examples

| Folder | File | Idea |
|---|---|---|
| `01_foundations/` | `01_relu_and_step.py` | relu, and a step built from two relus |
| | `02_if_else.py` | both branches run; a mask picks one |
| | `03_compare.py` | `>`, `>=`, `==`, `!=` as 0/1 numbers |
| | `04_logic.py` | `and`, `or`, `not`, `xor` on bits |
| | `05_min_max_abs_clamp.py` | the piecewise-linear helpers |
| | `06_select.py` | big-M: mask × runtime value without multiplication |
| `02_loops/` | `07_bounded_for.py` | a loop of unknown length becomes a fixed number of masked steps |
| | `08_while_halt.py` | `while` → fixed steps + a "halted" flag that freezes the state |
| | `09_break_first_match.py` | `break` → a "found already" flag |
| `03_memory/` | `10_array_read.py` | `xs[i]` with a runtime `i`: one-hot + masked sum |
| | `11_array_write.py` | `xs[i] = v` with a runtime `i` |
| | `12_argmax.py` | argmax / argmin as a one-hot vector |
| | `13_state_machine.py` | a finite automaton as a repeated layer |
| `04_graph_search/` | `14_bfs.py` | BFS: frontier vector, adjacency as weights |
| | `15_shortest_path.py` | weighted shortest paths by relaxation |
| | `16_astar.py` | A*: pick-the-best, expand, repeat |
| `05_game_search/` | `17_minimax_tree.py` | minimax over a fixed game tree |
| | `18_nim_minimax.py` | minimax for a real (tiny) game |

## Booklets

`booklet/` has one Typst booklet per folder, each compiled to PDF, which derives every example step by step: the formulas, the weight matrices, and diagrams of the resulting network. Read them in order; each ends with exercises and answers.

| PDF | Covers |
|---|---|
| `booklet/01_foundations.pdf` | the ground rules, then examples 01–06 |
| `booklet/02_loops.pdf` | 07–09: unrolling, halt flags, weight tying |
| `booklet/03_memory.pdf` | 10–13: one-hot addressing, argmax, automata |
| `booklet/04_graph_search.pdf` | 14–16, plus a recipe for your own search algorithm |
| `booklet/05_game_search.pdf` | 17–18, tree vs table, plus a recipe for your own game search |
