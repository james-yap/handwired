// 2048 as one numeric program: every key press runs the same fixed list of weighted sums, products and ReLUs over
// FP32 numbers, with no branch and no loop at runtime. The JavaScript in this file is builder code: it runs once, and
// each line appends the operations of one familiar expression, which the comment above it shows. rules.ts holds the
// same game in plain JavaScript, and game2048.test.ts plays both and checks that they agree after every move.
//
// The program has four parts:
// 1. Turn every move into a move to the left, by reading the board in the order of the pressed arrow.
// 2. Slide each of the 4 rows to the left: pack, merge equal neighbors, pack again.
// 3. The board after the key: write the rows back, spawn a tile, update the score, check for a win or a dead end.
// 4. Animation: every tile slides from where it came from, new tiles grow, merged tiles pop.
import {type AnimatedSpring, type Circuit, type Wire, emitAnimatedObject, emitCompare, emitMaximumExact} from '../src/runtime/numeric-ui-core.ts'
import {springMatrixBound} from '../src/runtime/host-timing.ts'
import {type CellWires, type WireValues, createSourceTable, event, input, inputBit, list, state} from '../src/runtime/cells.ts'
import {type Direction, boardSize, cellCount, directions, lineCell, maxExponent, spawnSlotCount, winningExponent} from './rules.ts'

// The board in px: 4 tiles of 100 px with 12 px gaps around them. page.ts and index.html draw the same sizes.
export const layout = {tileSize: 100, gap: 12, pitch: 112, boardSize: 4 * 112 + 12} as const
export const springConstants = {stiffness: 1000, damping: 60}
// Scale springs count in percent of the tile size: a new tile starts at -100, i.e. scale 0, and a merged tile jumps
// to +15, i.e. scale 1.15, then both spring back to 0, i.e. scale 1.
const spawnJump = -100
const mergeJump = 15

// Ranges are promises that every evaluation checks. A sliding tile starts at most 3 cells from its destination, and
// a key pressed mid-slide starts the next slide from where the tile is drawn, so twice that is room to spare.
const maxSlide = 2 * (boardSize - 1) * layout.pitch
const maxScale = 2 * 100
// The largest score that FP32 holds exactly as a whole number, about 4 times what a real game reaches.
const maxScore = 2 ** 24

const exponentCell = () => state([0, maxExponent], {grid: 1})
// A spring stores its displacement from its destination and its velocity, see emitAnimatedObject.
const springCells = (maxDisplacement: number) => ({
  displacement: state([-maxDisplacement, maxDisplacement]),
  velocity: state([-64 * maxDisplacement, 64 * maxDisplacement]),
})
const matrixBound = springMatrixBound(springConstants)
const matrixEntry = input([-matrixBound, matrixBound])
const spawnDrawCells = () => ({slot: input([0, spawnSlotCount - 1], {grid: 1}), four: inputBit()})

// Every number the program reads, or keeps from one key press to the next.
export const gameCells = {
  // One event per key. The host writes at most one of them per evaluation. {grid: 1} declares them whole numbers.
  keys: {left: event({grid: 1}), right: event({grid: 1}), up: event({grid: 1}), down: event({grid: 1}), newGame: event({grid: 1})},
  // The random numbers that place the tiles a key may spawn, see spawnTile in rules.ts. The host draws them for
  // every key press: a move uses the first, a new game both.
  spawns: [spawnDrawCells(), spawnDrawCells()],
  // The game: 16 exponents in reading order, 0 for an empty cell, and the scores.
  board: list(cellCount, exponentCell),
  score: state([0, maxScore], {grid: 1}),
  best: state([0, maxScore], {grid: 1}),
  // The animation of each cell: its tile's offset in px from the cell and its scale, and a ghost, i.e. the tile that
  // merged into this one, which slides in under it and keeps its old number.
  tiles: list(cellCount, () => ({x: springCells(maxSlide), y: springCells(maxSlide), scale: springCells(maxScale)})),
  ghosts: list(cellCount, () => ({exponent: exponentCell(), x: springCells(maxSlide), y: springCells(maxSlide)})),
  // What advanceFrame plans for each evaluation: the spring matrix of its 4 ms steps, whether springs may come to
  // rest (a settle check), and whether they snap onto their destinations because a frame would need too many steps.
  springMatrix: [matrixEntry, matrixEntry, matrixEntry, matrixEntry] as const,
  settleCheck: inputBit(),
  snap: inputBit(),
}

const range = (start: number, end?: number) =>
  end === undefined ? Array.from({length: start}, (_, index) => index) : Array.from({length: end - start}, (_, index) => start + index)

// The few expressions the game is written in, each built from the circuit's operations. Every bit here is exactly 0
// or 1, and every other number a whole number below 2^24, so every result is exact.
//
// emitCompare builds a short comparison, about 2 operations, when both sides are declared to be whole numbers, i.e.
// to lie on the grid 1, and a long one, about 15, otherwise. Sums and products of whole numbers keep the grid, but a
// ReLU doesn't know that its result is whole, so `and` and `or` declare it.
function createVocabulary(circuit: Circuit) {
  // The circuit folds an `and` that reads the constant 0 into that constant's wire, which other operations already
  // read, so the declaration allows existing readers. Declaring a whole number whole changes nothing for them.
  const whole = (wire: Wire) => {
    circuit.assumeGrid(wire, 1, {allowExistingReaders: true})
    return wire
  }
  // The constant 0 has no grid of its own either, and bit matrices use it for the pairs that can't happen.
  const zero = whole(circuit.constant(0))
  return {
    zero,
    // JS: a + b + c, and 0 for no terms
    sum: (...wires: Wire[]) => wires.length === 0 ? zero : circuit.affine(wires.map(wire => [wire, 1] as const)),
    // JS: bit ? value : 0, as value × max(0, bit), one ReGLU
    gate: (bit: Wire, value: Wire) => circuit.multiply(value, bit),
    // JS: !bit, as 1 - bit
    not: (bit: Wire) => circuit.affine([[bit, -1]], 1),
    // JS: bits.every(Boolean), as max(0, sum - (count - 1))
    and: (...bits: Wire[]) => whole(circuit.reglu([], 1, bits.map(bit => [bit, 1] as const), 1 - bits.length)),
    // JS: bits.some(Boolean), as 1 - max(0, 1 - sum)
    or: (...bits: Wire[]) => circuit.affine([[whole(circuit.reglu([], 1, bits.map(bit => [bit, -1] as const), 1)), -1]], 1),
    // JS: a === b and a >= b, as exact bits. Both sides are wires: a number that builder code knows goes in as
    // circuit.constant(value), which folds into the comparison's bias. (A Wire is a number at runtime, so a helper
    // that also took plain numbers couldn't tell the two apart.)
    equals: (a: Wire, b: Wire) => emitCompare(circuit, a, '===', b),
    atLeast: (a: Wire, b: Wire) => emitCompare(circuit, a, '>=', b),
  }
}

// The row and position at which a direction's lines read a cell, the inverse of lineCell.
function rowPosition(direction: Direction, cell: number) {
  for (const line of range(boardSize)) {
    for (const position of range(boardSize)) {
      if (lineCell(direction, line, position) === cell) return {line, position}
    }
  }
  throw new Error(`cell ${cell} is on no line`)
}

export function buildGame() {
  const table = createSourceTable()
  const cells = table.declare(gameCells)
  table.atMostOne('keys', Object.values(cells.keys))
  const circuit = table.createCircuit()
  const vocabulary = createVocabulary(circuit)
  const {zero, sum, gate, not, and, or, equals, atLeast} = vocabulary
  const {keys} = cells
  const slots = range(boardSize)
  // Where each part of the program ends in circuit.operations, for inspector.ts's overview. Parts follow each other
  // because operations are appended in the order the lines below run.
  const parts: ProgramPart[] = []
  const endPart = (name: string) => { parts.push({name, end: circuit.operations.length}) }

  // Keys arrive one per evaluation (table.atMostOne above), so a sum of key bits is itself a bit. The circuit only
  // sees a sum of 0..4, so keyPressed declares it, which keeps everything that reads it on whole numbers in 0..1.
  const keyPressed = (...keyBits: Wire[]) => {
    const pressed = sum(...keyBits)
    circuit.assumeRange(pressed, 0, 1)
    return pressed
  }

  // 1. Turn every move into a move to the left.
  // JS: moving = key is an arrow, 1 or 0
  const moving = keyPressed(...directions.map(direction => keys[direction]))
  // JS: rows[line][position] = perCell[lineCell(pressed, line, position)], or 0 when no arrow was pressed.
  // The sum has one term per direction, and only the pressed one isn't 0.
  const toRows = (perCell: readonly Wire[]) => slots.map(line => slots.map(position =>
    sum(...directions.map(direction => gate(keys[direction], perCell[lineCell(direction, line, position)]!)))))
  // The inverse, rows back into cells, with each direction's term multiplied by weights[direction].
  const allOnes: Record<Direction, number> = {left: 1, right: 1, up: 1, down: 1}
  const fromRows = (rows: readonly (readonly Wire[])[], weights = allOnes) => range(cellCount).map(cell =>
    circuit.affine(directions.filter(direction => weights[direction] !== 0).map(direction => {
      const {line, position} = rowPosition(direction, cell)
      return [gate(keys[direction], rows[line]![position]!), weights[direction]] as const
    })))

  // 2. Slide each row to the left.
  const rowsIn = toRows(cells.board)
  endPart('read rows')
  const rows = rowsIn.map(row => slideRowLeft(vocabulary, circuit, row))
  endPart('slide rows')

  // 3. The board after the key.
  // JS: afterMove = moving ? slid board : board. The slid board is all zeros when no arrow was pressed.
  const slidBoard = fromRows(rows.map(row => row.line))
  const notMoving = not(moving)
  const afterMove = cells.board.map((exponent, cell) => sum(gate(notMoving, exponent), slidBoard[cell]!))
  // JS: changed = afterMove.some((exponent, cell) => exponent !== board[cell])
  const changed = not(and(...afterMove.map((exponent, cell) => equals(exponent, cells.board[cell]!))))
  // JS: start = newGame ? empty board : afterMove
  const notNewGame = not(keys.newGame)
  const start = afterMove.map(exponent => gate(notNewGame, exponent))
  endPart('write rows back')
  // JS: a move that changed the board spawns one tile, a new game two.
  const firstSpawn = spawnTile(vocabulary, circuit, start, or(changed, keys.newGame), cells.spawns[0]!)
  const secondSpawn = spawnTile(vocabulary, circuit, firstSpawn.board, keys.newGame, cells.spawns[1]!)
  const board = secondSpawn.board
  endPart('spawn')
  // JS: score = newGame ? 0 : score + gain, and best = Math.max(best, score). The gain is 0 unless an arrow was pressed.
  // The score's cell only holds scores up to 2^24, and the evaluation checks it, so the sum can promise the same.
  const score = sum(gate(notNewGame, cells.score), ...rows.map(row => row.gain))
  circuit.assumeRange(score, 0, maxScore)
  const best = emitMaximumExact(circuit, cells.best, score)
  // JS: won = board.some(exponent => exponent >= 11), and gameOver = !canMove(board), see rules.ts.
  const won = or(...board.map(exponent => atLeast(exponent, circuit.constant(winningExponent))))
  const neighbors: [number, number][] = range(cellCount).flatMap(cell => [
    ...(cell % boardSize + 1 < boardSize ? [[cell, cell + 1] as [number, number]] : []),
    ...(cell + boardSize < cellCount ? [[cell, cell + boardSize] as [number, number]] : []),
  ])
  const gameOver = not(or(
    ...board.map(exponent => equals(exponent, zero)),
    ...neighbors.map(([a, b]) => equals(board[a]!, board[b]!)),
  ))
  endPart('score and status')

  // 4. Animation. Each cell's springs belong to the tile drawn in that cell, so when a tile moves to another cell,
  // its springs move with it, and the slide is a jump of the spring's destination: the tile is drawn where it was,
  // and springs toward its new cell. emitAnimatedObject takes that jump as destinationDuringSteps, the destination
  // that the stored displacement is measured from, i.e. the old cell seen from the new one.
  //
  // A row says where each slot's tile came from in two bit matrices: `from` for the tile itself, and `mergedFrom` for
  // the tile that merged into it, which page.ts draws as a ghost under it until the next move.
  type Origins = (row: SlidRow) => Wire[][]
  const tileOrigin: Origins = row => row.from
  const ghostOrigin: Origins = row => row.mergedFrom
  // JS: jump[cell] = where cell's tile came from, in px from the cell, or 0 when no tile came.
  // In a row, a tile can only come from behind its slot: (position - slot) cells to the right of a move left, to the
  // left of a move right, below a move up and above a move down.
  const inX: Record<Direction, number> = {left: layout.pitch, right: -layout.pitch, up: 0, down: 0}
  const inY: Record<Direction, number> = {left: 0, right: 0, up: layout.pitch, down: -layout.pitch}
  const jumps = (origins: Origins, weights: Record<Direction, number>) => fromRows(rows.map(row =>
    origins(row).map((bits, slot) => circuit.affine(bits.map((bit, position) => [bit, position - slot] as const)))), weights)
  // JS: carried[cell] = perCell[the cell that cell's tile came from], or 0 when no tile came.
  const carry = (perCell: readonly Wire[], origins: Origins) => fromRows(toRows(perCell).map((values, line) =>
    origins(rows[line]!).map(bits => sum(...bits.map((bit, position) => gate(bit, values[position]!))))))
  // JS: next = keep ? stored : carried. A tile's spring is the same spring after the move, so its displacement and
  // velocity are carried unchanged, only into another cell.
  const keepOrCarry = (stored: readonly SpringWires[], keep: Wire, source: readonly SpringWires[], origins: Origins) => {
    const displacements = carry(source.map(spring => spring.displacement), origins)
    const velocities = carry(source.map(spring => spring.velocity), origins)
    return stored.map((spring, cell): SpringWires => ({
      displacement: sum(gate(keep, spring.displacement), displacements[cell]!),
      velocity: sum(gate(keep, spring.velocity), velocities[cell]!),
    }))
  }
  // A tile keeps its springs until an arrow carries them along, or a new game resets them to 0.
  // JS: still = !moving && !newGame
  const still = not(keyPressed(moving, keys.newGame))
  // A ghost lasts until the board changes or a new game starts. It was a tile until this key, so it takes over the
  // springs of that tile. JS: keepGhost = !changed && !newGame
  const keepGhost = not(or(changed, keys.newGame))
  const tiles = {
    x: cells.tiles.map(tile => tile.x),
    y: cells.tiles.map(tile => tile.y),
    scale: cells.tiles.map(tile => tile.scale),
  }

  // Every spring goes into one animated object, which steps them all with the host's matrix and settles them together.
  // animate returns the index of the spring's results.
  const springs: AnimatedSpring[] = []
  const animate = ({displacement, velocity}: SpringWires, destinationDuringSteps: Wire) =>
    springs.push({matrix: cells.springMatrix, displacement, velocity, destination: zero, destinationDuringSteps}) - 1
  // An offset in x or y: kept or carried, then jumping to where the tile came from.
  const slide = (stored: readonly SpringWires[], keep: Wire, source: readonly SpringWires[], origins: Origins, weights: Record<Direction, number>) => {
    const jump = jumps(origins, weights)
    return keepOrCarry(stored, keep, source, origins).map((spring, cell) => animate(spring, jump[cell]!))
  }
  const tileX = slide(tiles.x, still, tiles.x, tileOrigin, inX)
  const tileY = slide(tiles.y, still, tiles.y, tileOrigin, inY)
  const ghostX = slide(cells.ghosts.map(ghost => ghost.x), keepGhost, tiles.x, ghostOrigin, inX)
  const ghostY = slide(cells.ghosts.map(ghost => ghost.y), keepGhost, tiles.y, ghostOrigin, inY)
  // A tile's scale is carried like its offsets, except that a new tile restarts at rest at scale 0 and a merged tile
  // at rest at scale 1.15. JS: restart = spawned || merged; next = restart ? at rest : carried
  const mergedHere = fromRows(rows.map(row => row.merged))
  const spawnedHere = range(cellCount).map(cell => sum(firstSpawn.spawned[cell]!, secondSpawn.spawned[cell]!))
  const tileScale = keepOrCarry(tiles.scale, still, tiles.scale, tileOrigin).map((spring, cell) => {
    const keep = not(or(spawnedHere[cell]!, mergedHere[cell]!))
    const restartAt = circuit.affine([[spawnedHere[cell]!, spawnJump], [mergedHere[cell]!, mergeJump]])
    return animate({displacement: gate(keep, spring.displacement), velocity: gate(keep, spring.velocity)}, restartAt)
  })
  const animation = emitAnimatedObject(circuit, springs, {settleCheck: cells.settleCheck, snap: cells.snap})
  const nextSpring = (index: number) => ({displacement: animation.displacements[index]!, velocity: animation.velocities[index]!})
  // JS: ghostExponent = keepGhost ? stored : the exponent of the tile that merged in, before the merge
  const mergedExponents = fromRows(rows.map(row => row.ghost))
  const ghostExponent = cells.ghosts.map((ghost, cell) => sum(gate(keepGhost, ghost.exponent), mergedExponents[cell]!))

  // Commit state: each state cell gets exactly one next value.
  table.next(cells.board, board)
  table.next(cells.score, score)
  table.next(cells.best, best)
  table.next(cells.tiles, range(cellCount).map(cell => ({x: nextSpring(tileX[cell]!), y: nextSpring(tileY[cell]!), scale: nextSpring(tileScale[cell]!)})))
  table.next(cells.ghosts, range(cellCount).map(cell => ({exponent: ghostExponent[cell]!, x: nextSpring(ghostX[cell]!), y: nextSpring(ghostY[cell]!)})))

  // What page.ts draws: each cell's tile and ghost at its position in px on the board, and the scores and status.
  // A spring's position is its destination, 0, plus its displacement.
  const left = (cell: number, spring: number) => circuit.affine([[animation.positions[spring]!, 1]], layout.gap + (cell % boardSize) * layout.pitch)
  const top = (cell: number, spring: number) => circuit.affine([[animation.positions[spring]!, 1]], layout.gap + Math.floor(cell / boardSize) * layout.pitch)
  const output = {
    tiles: range(cellCount).map(cell => ({
      exponent: board[cell]!,
      x: left(cell, tileX[cell]!),
      y: top(cell, tileY[cell]!),
      scale: circuit.affine([[animation.positions[tileScale[cell]!]!, 1 / 100]], 1),
    })),
    ghosts: range(cellCount).map(cell => ({
      exponent: ghostExponent[cell]!,
      x: left(cell, ghostX[cell]!),
      y: top(cell, ghostY[cell]!),
    })),
    score,
    best,
    won,
    gameOver,
    // 1 while anything still moves. The host keeps requesting frames until it's 0.
    active: animation.active,
  }
  endPart('animation')
  // What inspector.ts shows of a key press: each row as read in the arrow's order with every step of its slide, the
  // tiles that spawned, and the board after the key. These are wires the program computes anyway, so returning them
  // adds no operation.
  const inspect: InspectWires = {rows: rows.map((row, line) => ({read: rowsIn[line]!, ...row})), spawned: spawnedHere, board}
  return {cells, output, inspect, parts, ...table.finish()}
}

type SpringWires = {displacement: Wire, velocity: Wire}
type Vocabulary = ReturnType<typeof createVocabulary>
// One row after a move to the left. Bit matrices are indexed [slot][position]: slot is where a tile ends up, position
// where it started.
type SlidRow = {
  // The exponents after the slide.
  line: Wire[]
  // merged[slot] = 1 when line[slot] is the result of a merge.
  merged: Wire[]
  // from[slot][position] = 1 when the tile in slot started at position.
  from: Wire[][]
  // mergedFrom[slot][position] = 1 when the tile that merged into slot started at position.
  mergedFrom: Wire[][]
  // ghost[slot] = the exponent of the tile that merged into slot, before the merge, or 0.
  ghost: Wire[]
  // The points this row scores: the sum of its merged tiles.
  gain: Wire
  // The steps in between, named as in slideRowLeft, for inspector.ts.
  occupied: Wire[]
  tilesBefore: Wire[]
  packedFrom: Wire[][]
  packed: Wire[]
  merges: Wire[]
  survives: Wire[]
  finalSlot: Wire[]
  lands: Wire[][]
}

// One row slid to the left, the same three steps as slideLine in rules.ts. row[position] is the exponent at that
// position, 0 for empty.
function slideRowLeft({zero, sum, gate, not, and, equals, atLeast}: Vocabulary, circuit: Circuit, row: readonly Wire[]): SlidRow {
  const slots = range(boardSize)
  // 1. Pack the tiles to the left. JS: packed = row.filter(exponent => exponent > 0)
  // A tile lands in the slot numbered by how many tiles are before it.
  const occupied = row.map(exponent => atLeast(exponent, circuit.constant(1)))
  const tilesBefore = row.map((_, position) => sum(...occupied.slice(0, position)))
  // packedFrom[slot][position] = 1 when the tile at position lands in slot. A tile never moves right.
  const packedFrom = slots.map(slot => row.map((_, position) =>
    position < slot ? zero : and(occupied[position]!, equals(tilesBefore[position]!, circuit.constant(slot)))))
  // JS: packed[slot], or 0 past the last tile
  const packed = packedFrom.map(bits => sum(...bits.map((bit, position) => gate(bit, row[position]!))))
  // JS: filled[slot] = slot < packed.length
  const filled = packed.map(exponent => atLeast(exponent, circuit.constant(1)))

  // 2. Merge equal neighbors, front first, each tile at most once.
  // JS: merges[slot] = !merges[slot - 1] && packed[slot] === packed[slot + 1]
  // An empty slot holds 0, so the neighbor must also hold a tile, or two empty slots would merge.
  const merges: Wire[] = []
  for (const slot of range(boardSize - 1)) {
    const equalNeighbor = and(filled[slot + 1]!, equals(packed[slot]!, packed[slot + 1]!))
    merges.push(slot === 0 ? equalNeighbor : and(equalNeighbor, not(merges[slot - 1]!)))
  }
  merges.push(zero) // the last slot has no neighbor to its right

  // 3. Drop the tiles that merged away, which moves every later tile left by one per merge before it.
  // JS: survives[slot] = slot holds a tile && !merges[slot - 1]
  const survives = slots.map(slot => slot === 0 ? filled[0]! : and(filled[slot]!, not(merges[slot - 1]!)))
  // JS: finalSlot[slot] = slot - (number of merges among the pairs that end before slot)
  const finalSlot = slots.map(slot => circuit.affine(merges.slice(0, Math.max(0, slot - 1)).map(merge => [merge, -1] as const), slot))
  // lands[slot][final] = 1 when packed[slot] survives and ends in final
  const lands = slots.map(slot => slots.map(final =>
    final > slot ? zero : and(survives[slot]!, equals(finalSlot[slot]!, circuit.constant(final)))))
  const landing = (final: number) => slots.map(slot => lands[slot]![final]!)
  // JS: line[final] = the surviving tile's exponent, plus 1 when its neighbor merged into it
  const line = slots.map(final => sum(...landing(final).map((bit, slot) => gate(bit, sum(packed[slot]!, merges[slot]!)))))
  // JS: merged[final] = whether line[final] is a merged tile
  const merged = slots.map(final => sum(...landing(final).map((bit, slot) => and(bit, merges[slot]!))))
  // from[final][position] = 1 when the tile in final started at position
  const from = slots.map(final => row.map((_, position) =>
    sum(...landing(final).map((bit, slot) => and(bit, packedFrom[slot]![position]!)))))
  // mergedFrom[final][position] = 1 when the tile that merged into final started at position
  const mergedFrom = slots.map(final => row.map((_, position) =>
    sum(...landing(final).map((bit, slot) => slot + 1 < boardSize ? and(bit, merges[slot]!, packedFrom[slot + 1]![position]!) : zero))))
  // JS: ghost[final] = the exponent of the tile that merged into final, before the merge, or 0
  const ghost = slots.map(final => sum(...landing(final).map((bit, slot) => gate(and(bit, merges[slot]!), packed[slot]!))))
  // JS: gain = sum of 2 ** (exponent + 1) over the merges, with the powers of two read from a table. A table read
  // doesn't know that every entry is a whole number, so the read declares it.
  const doubled = range(maxExponent + 1).map(exponent => circuit.constant(2 ** (exponent + 1)))
  const gain = sum(...slots.map(slot => {
    const value = circuit.indexRead(packed[slot]!, doubled)
    circuit.assumeGrid(value, 1)
    return gate(merges[slot]!, value)
  }))
  return {line, merged, from, mergedFrom, ghost, gain, occupied, tilesBefore, packedFrom, packed, merges, survives, finalSlot, lands}
}

// A new tile in a random empty cell, the same as spawnTile in rules.ts, when `enabled` is 1.
function spawnTile({zero, sum, and, equals, atLeast}: Vocabulary, circuit: Circuit, board: readonly Wire[], enabled: Wire, draw: CellWires<ReturnType<typeof spawnDrawCells>>) {
  const empty = board.map(exponent => equals(exponent, zero))
  // JS: k = Math.floor(slot × emptyCount / spawnSlotCount), as the number of multiples of spawnSlotCount that
  // slot × emptyCount reaches. The product is a whole number below 2^24, so FP32 holds it exactly.
  const scaledSlot = circuit.multiply(draw.slot, sum(...empty))
  const k = sum(...range(1, cellCount).map(multiple => atLeast(scaledSlot, circuit.constant(multiple * spawnSlotCount))))
  // JS: spawned[cell] = enabled && cell is the k-th empty cell in reading order
  const spawned = board.map((_, cell) => and(enabled, empty[cell]!, equals(sum(...empty.slice(0, cell)), k)))
  // JS: a 2 (exponent 1), or a 4 (exponent 2) when four is 1
  return {spawned, board: board.map((exponent, cell) => sum(exponent, spawned[cell]!, and(spawned[cell]!, draw.four)))}
}

export type GameOutput = WireValues<ReturnType<typeof buildGame>['output']>
export type ProgramPart = {name: string, end: number}
// A row as read in the arrow's order, and every step of its slide.
export type InspectedRow = SlidRow & {read: Wire[]}
export type InspectWires = {rows: InspectedRow[], spawned: Wire[], board: Wire[]}
export type Inspection = WireValues<InspectWires>
