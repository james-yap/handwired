// The page: arrow keys, WASD and swipes queue key presses, and each animation frame runs the machine and writes the
// DOM from the program's output numbers only. Nothing here knows the rules of 2048.
import {layout} from './game.ts'
import {cellCount, type Direction} from './rules.ts'
import {createGameMachine} from './machine.ts'

const bestScoreKey = 'handwired-2048-best'
const keyDirections: Record<string, Direction> = {
  ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down',
  a: 'left', d: 'right', w: 'up', s: 'down',
}
// A swipe shorter than this, in px, is a tap and does nothing.
const minimumSwipe = 24

export function mountGame() {
  const machine = createGameMachine()
  const savedBest = Number(localStorage.getItem(bestScoreKey))
  if (Number.isInteger(savedBest) && savedBest > 0 && savedBest <= 2 ** 24) machine.restoreBest(savedBest)

  const board = document.querySelector<HTMLElement>('.board')!
  const score = document.querySelector<HTMLElement>('.score output')!
  const best = document.querySelector<HTMLElement>('.best output')!
  const page = document.body
  // Every cell has a ghost, i.e. the tile that just merged into it, under its tile. Both are reused for whatever
  // tile the cell holds, and hidden while their exponent is 0.
  const makeTiles = (className: string) => Array.from({length: cellCount}, () => {
    const element = document.createElement('div')
    element.className = className
    board.append(element)
    return element
  })
  const ghosts = makeTiles('tile ghost')
  const tiles = makeTiles('tile')
  const drawTile = (element: HTMLElement, exponent: number, x: number, y: number, scale: number) => {
    element.hidden = exponent === 0
    if (exponent === 0) return
    const label = String(2 ** exponent)
    if (element.textContent !== label) element.textContent = label
    element.dataset['exponent'] = String(exponent)
    element.style.transform = `translate(${x}px, ${y}px) scale(${scale})`
  }

  let scheduledFrame: number | null = null
  const scheduleRender = () => {
    scheduledFrame ??= requestAnimationFrame(render)
  }
  let savedScore = savedBest
  function render(frameTime: number) {
    scheduledFrame = null
    const output = machine.step(frameTime)
    output.tiles.forEach((tile, cell) => drawTile(tiles[cell]!, tile.exponent, tile.x, tile.y, tile.scale))
    output.ghosts.forEach((ghost, cell) => drawTile(ghosts[cell]!, ghost.exponent, ghost.x, ghost.y, 1))
    score.textContent = String(output.score)
    best.textContent = String(output.best)
    page.classList.toggle('won', output.won === 1)
    page.classList.toggle('over', output.gameOver === 1)
    if (output.best !== savedScore) {
      savedScore = output.best
      localStorage.setItem(bestScoreKey, String(output.best))
    }
    if (machine.clock.animatedUntilMs !== null) scheduleRender()
  }
  // Key presses carry the event's timestamp, which shares requestAnimationFrame's time origin.
  const press = (key: Direction | 'newGame', timeStamp: number) => {
    machine.queue({key, timeStamp})
    scheduleRender()
  }

  window.addEventListener('keydown', event => {
    if (event.metaKey || event.ctrlKey || event.altKey) return
    const direction = keyDirections[event.key]
    if (direction !== undefined) {
      event.preventDefault()
      press(direction, event.timeStamp)
    } else if (event.key === 'n') {
      press('newGame', event.timeStamp)
    }
  })
  document.querySelector('.new-game')!.addEventListener('click', event => press('newGame', event.timeStamp))
  let swipeStart: {x: number, y: number} | null = null
  board.addEventListener('pointerdown', event => { swipeStart = {x: event.clientX, y: event.clientY} })
  board.addEventListener('pointerup', event => {
    if (swipeStart === null) return
    const dx = event.clientX - swipeStart.x
    const dy = event.clientY - swipeStart.y
    swipeStart = null
    if (Math.max(Math.abs(dx), Math.abs(dy)) < minimumSwipe) return
    press(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 'right' : 'left') : (dy > 0 ? 'down' : 'up'), event.timeStamp)
  })
  board.addEventListener('pointercancel', () => { swipeStart = null })

  // The board is drawn at a fixed size in px, and scaled down to fit a narrow window.
  const fit = () => board.style.setProperty('--fit', String(Math.min(1, (document.documentElement.clientWidth - 24) / layout.boardSize)))
  window.addEventListener('resize', fit)
  fit()
  press('newGame', performance.now())
}
