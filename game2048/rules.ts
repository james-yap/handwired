// The rules of 2048 in plain JavaScript. game.ts builds the same rules as one numeric program, step for step, and
// game2048.test.ts plays both side by side and checks that they agree after every move.
//
// The board is 16 numbers in reading order, row by row from the top left. Each number is an exponent: 0 is an empty
// cell and e is the tile 2^e, so 1 is a 2, 2 is a 4 and 11 is 2048. Exponents keep every number a small integer, and
// "these two tiles are equal" stays a comparison of two small integers.

export const boardSize = 4
export const cellCount = boardSize * boardSize
// The largest tile a 4 × 4 board can hold is 2^17 = 131,072.
export const maxExponent = 17
// The tile that wins the game: 2^11 = 2048.
export const winningExponent = 11

export type Direction = 'left' | 'right' | 'up' | 'down'
export const directions: readonly Direction[] = ['left', 'right', 'up', 'down']

// Every move slides each of the 4 lines toward its front. Position 0 of a line is the cell at the edge the tiles move
// toward, e.g. moving left, line 1 is row 1 from left to right, cells 4, 5, 6, 7, and moving down, line 1 is column 1
// from bottom to top, cells 13, 9, 5, 1. game.ts uses the same function to rearrange the board so that every move
// becomes a move to the left.
export function lineCell(direction: Direction, line: number, position: number): number {
  const fromFarEdge = boardSize - 1 - position
  switch (direction) {
    case 'left': return line * boardSize + position
    case 'right': return line * boardSize + fromFarEdge
    case 'up': return position * boardSize + line
    case 'down': return fromFarEdge * boardSize + line
  }
}

// One line slid toward its front: the line afterwards, and the points scored, i.e. the sum of the merged tiles.
export function slideLine(line: readonly number[]): {line: number[], gain: number} {
  // 1. Pack the tiles toward the front, keeping their order: [0, 2, 0, 2] becomes [2, 2].
  const packed = line.filter(exponent => exponent > 0)
  // 2. Merge equal neighbors, front first, and each tile at most once: [2, 2, 2] merges the first pair only, and
  //    [2, 2, 2, 2] merges both pairs. merges[slot] says that packed[slot + 1] merges into packed[slot].
  const merges = packed.map(() => false)
  for (let slot = 0; slot + 1 < packed.length; slot++) {
    const absorbedByPrevious = slot > 0 && merges[slot - 1]!
    merges[slot] = !absorbedByPrevious && packed[slot] === packed[slot + 1]
  }
  // 3. Drop the tiles that merged away and double the ones they merged into, which packs the line again.
  const result: number[] = []
  let gain = 0
  packed.forEach((exponent, slot) => {
    if (slot > 0 && merges[slot - 1]!) return
    result.push(exponent + (merges[slot]! ? 1 : 0))
    if (merges[slot]!) gain += 2 ** (exponent + 1)
  })
  while (result.length < boardSize) result.push(0)
  return {line: result, gain}
}

export function move(board: readonly number[], direction: Direction): {board: number[], gain: number} {
  const next = [...board]
  let gain = 0
  for (let line = 0; line < boardSize; line++) {
    const cells = Array.from({length: boardSize}, (_, position) => lineCell(direction, line, position))
    const slid = slideLine(cells.map(cell => board[cell]!))
    cells.forEach((cell, position) => { next[cell] = slid.line[position]! })
    gain += slid.gain
  }
  return {board: next, gain}
}

// Where a new tile appears. The program can't call Math.random, so the host draws two numbers per spawn and writes
// them into the program's input cells, and both versions place the tile from the same two numbers:
// - slot, a whole number below spawnSlotCount, picks the k-th empty cell in reading order, where
//   k = floor(slot × emptyCount / spawnSlotCount).
// - four is 1 for a 4, which happens 1 time in 10, and 0 for a 2.
// spawnSlotCount is the least common multiple of 1..16, so every possible emptyCount divides it, every empty cell is
// equally likely, and slot × emptyCount stays below 2^24, where FP32 still holds every whole number exactly.
export const spawnSlotCount = 720_720
export type SpawnDraw = {slot: number, four: number}

export function drawSpawn(random: () => number): SpawnDraw {
  return {slot: Math.floor(random() * spawnSlotCount), four: random() < 0.1 ? 1 : 0}
}

export function spawnTile(board: readonly number[], {slot, four}: SpawnDraw): number[] {
  const empty = board.flatMap((exponent, cell) => exponent === 0 ? [cell] : [])
  if (empty.length === 0) return [...board]
  const next = [...board]
  next[empty[Math.floor(slot * empty.length / spawnSlotCount)]!] = 1 + four
  return next
}

// A move is possible while a cell is empty or two neighbors are equal.
export function canMove(board: readonly number[]): boolean {
  return board.some((exponent, cell) => {
    const column = cell % boardSize
    const row = Math.floor(cell / boardSize)
    return exponent === 0
      || (column + 1 < boardSize && board[cell + 1] === exponent)
      || (row + 1 < boardSize && board[cell + boardSize] === exponent)
  })
}

export type GameState = {board: number[], score: number, best: number}
export type GameKey = Direction | 'newGame'

export const emptyGame: GameState = {board: Array<number>(cellCount).fill(0), score: 0, best: 0}

// One key press. A move that changes the board spawns one tile, and a new game starts from an empty board with two.
// The host draws both spawns for every key press, and a key uses as many as it needs.
export function applyKey(state: GameState, key: GameKey, spawns: readonly [SpawnDraw, SpawnDraw]): GameState {
  let board: number[]
  let score: number
  if (key === 'newGame') {
    board = spawnTile(spawnTile(emptyGame.board, spawns[0]), spawns[1])
    score = 0
  } else {
    const moved = move(state.board, key)
    const changed = moved.board.some((exponent, cell) => exponent !== state.board[cell])
    board = changed ? spawnTile(moved.board, spawns[0]) : moved.board
    score = state.score + moved.gain
  }
  return {board, score, best: Math.max(state.best, score)}
}
