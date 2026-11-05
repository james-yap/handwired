// Plays the numeric program (game.ts, through the host in machine.ts) and the plain JavaScript rules (rules.ts) side
// by side, and checks that they agree after every key, then checks the animation's start and end positions.
import {expect, test} from 'bun:test'
import {layout} from './game.ts'
import {type KeyPress, createGameMachine} from './machine.ts'
import {type Direction, type GameKey, type GameState, applyKey, canMove, cellCount, directions, drawSpawn, emptyGame, lineCell, slideLine, winningExponent} from './rules.ts'

// A seeded random number generator, so a failing game replays the same way.
function seededRandom(seed: number) {
  let state = seed
  return () => {
    state = (state * 16807) % 2147483647
    return state / 2147483647
  }
}

// Steps a machine through key presses 1/64 ms after each frame's start, with one frame per key.
function createPlayer(random: () => number) {
  const machine = createGameMachine(random)
  let timeMs = 1000
  return {
    machine,
    press(key: GameKey, gapMs = 16) {
      timeMs += gapMs
      machine.queue({key, timeStamp: timeMs - gapMs + 1 / 64} satisfies KeyPress)
      return machine.step(timeMs)
    },
    idle(frames: number) {
      let output = machine.step(timeMs)
      for (let frame = 0; frame < frames; frame++) output = machine.step(timeMs += 16)
      return output
    },
    // Puts tiles on the board directly, as exponents in reading order, with the springs at rest.
    setBoard(board: readonly number[]) {
      board.forEach((exponent, cell) => { machine.source[machine.built.cells.board[cell]!] = exponent })
    },
  }
}

const homeX = (cell: number) => layout.gap + (cell % 4) * layout.pitch
const homeY = (cell: number) => layout.gap + Math.floor(cell / 4) * layout.pitch

test('slideLine packs, merges each tile at most once, front first, and scores the merged tiles', () => {
  // Exponents: 1 is a 2, 2 is a 4, 3 is an 8.
  expect(slideLine([0, 0, 0, 1])).toEqual({line: [1, 0, 0, 0], gain: 0})
  expect(slideLine([1, 0, 1, 0])).toEqual({line: [2, 0, 0, 0], gain: 4})
  expect(slideLine([1, 1, 1, 0])).toEqual({line: [2, 1, 0, 0], gain: 4})
  expect(slideLine([1, 1, 1, 1])).toEqual({line: [2, 2, 0, 0], gain: 8})
  expect(slideLine([2, 1, 1, 0])).toEqual({line: [2, 2, 0, 0], gain: 4}) // the new 4 doesn't merge again
  expect(slideLine([1, 1, 2, 2])).toEqual({line: [2, 3, 0, 0], gain: 12})
  expect(slideLine([1, 2, 1, 2])).toEqual({line: [1, 2, 1, 2], gain: 0})
})

test('the program slides the same lines as slideLine, in every direction', () => {
  const lines = [[0, 0, 0, 1], [1, 0, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1], [2, 1, 1, 0], [1, 1, 2, 2], [1, 2, 1, 2], [3, 0, 3, 3]]
  for (const direction of directions) {
    // Each board holds 4 of the lines, written along the direction, so that each line's front is at the edge the
    // tiles move toward.
    for (const cases of [lines.slice(0, 4), lines.slice(4, 8)]) {
      const board = Array<number>(cellCount).fill(0)
      cases.forEach((line, index) => line.forEach((exponent, position) => { board[lineCell(direction, index, position)] = exponent }))
      const player = createPlayer(seededRandom(7))
      player.setBoard(board)
      const output = player.press(direction)
      // Every cell holds what slideLine says, except the one cell where a new tile appeared.
      const differences = cases.flatMap((line, index) => slideLine(line).line.flatMap((exponent, position) => {
        const actual = output.tiles[lineCell(direction, index, position)]!.exponent
        return actual === exponent ? [] : [{expected: exponent, actual}]
      }))
      expect(differences.length).toBe(1)
      expect(differences[0]!.expected).toBe(0)
      expect(output.score).toBe(cases.reduce((total, line) => total + slideLine(line).gain, 0))
    }
  }
})

test('the program and the plain JavaScript rules agree after every key of many random games', () => {
  for (const seed of [1, 2, 3, 4, 5]) {
    const keys = seededRandom(seed * 1000)
    const player = createPlayer(seededRandom(seed))
    const referenceRandom = seededRandom(seed)
    let reference: GameState = emptyGame
    const play = (key: GameKey) => {
      const output = player.press(key, 1 + Math.floor(keys() * 40))
      reference = applyKey(reference, key, [drawSpawn(referenceRandom), drawSpawn(referenceRandom)])
      expect(output.tiles.map(tile => tile.exponent)).toEqual(reference.board)
      expect([output.score, output.best]).toEqual([reference.score, reference.best])
      expect(output.gameOver).toBe(canMove(reference.board) ? 0 : 1)
      expect(output.won).toBe(reference.board.some(exponent => exponent >= winningExponent) ? 1 : 0)
      return output
    }
    play('newGame')
    let moves = 0
    // Biased toward left and down, like a human strategy, so games get further.
    const strategy: Direction[] = ['left', 'down', 'left', 'down', 'right', 'up']
    while (moves < 3000 && play(strategy[Math.floor(keys() * strategy.length)]!).gameOver === 0) moves++
    expect(canMove(reference.board)).toBe(false) // every game ran to its end
    // A new game keeps the best score and starts from two tiles.
    const restarted = play('newGame')
    expect(restarted.tiles.filter(tile => tile.exponent > 0).length).toBe(2)
    expect(restarted.score).toBe(0)
  }
})

test('a sliding tile starts where it was, a merged-away tile becomes a ghost, and everything comes to rest at home', () => {
  const player = createPlayer(seededRandom(11))
  // Row 0: a 2 in the last column. Row 1: two 2s that merge.
  player.setBoard([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  player.idle(0)
  // The key's evaluation and the frame share a timestamp, so no spring has stepped yet.
  player.machine.queue({key: 'left', timeStamp: 2000})
  const atKey = player.machine.step(2000)
  expect([atKey.tiles[0]!.exponent, atKey.tiles[0]!.x, atKey.tiles[0]!.y]).toEqual([1, homeX(3), homeY(0)])
  expect([atKey.tiles[4]!.exponent, atKey.tiles[4]!.x, atKey.tiles[4]!.scale]).toEqual([2, homeX(4), Math.fround(1.15)])
  expect([atKey.ghosts[4]!.exponent, atKey.ghosts[4]!.x]).toEqual([1, homeX(5)])
  const moving = player.machine.step(2050)
  expect(moving.tiles[0]!.x).toBeLessThan(homeX(3))
  expect(moving.tiles[0]!.x).toBeGreaterThan(homeX(0))

  // Keys pressed every 3 ms keep retargeting tiles mid-slide, and every spring stays within its declared range.
  let timeMs = 2100
  const keys = seededRandom(3)
  for (let press = 0; press < 400; press++) {
    player.machine.queue({key: directions[Math.floor(keys() * 4)]!, timeStamp: timeMs})
    if (press % 5 === 0) player.machine.step(timeMs)
    timeMs += 3
  }
  player.machine.step(timeMs)
  const rest = player.machine.step(timeMs + 2000)
  expect(player.machine.clock.animatedUntilMs).toBeNull()
  rest.tiles.forEach((tile, cell) => expect([tile.x, tile.y, tile.scale]).toEqual([homeX(cell), homeY(cell), 1]))
  rest.ghosts.forEach((ghost, cell) => expect([ghost.x, ghost.y]).toEqual([homeX(cell), homeY(cell)]))
})
