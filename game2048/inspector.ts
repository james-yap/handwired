// The inspector beside the game: what the program computed for the last arrow key, read from its own wires
// (built.inspect in game.ts, evaluated by machine.ts on the key's table). It draws one line's slideRowLeft as a strip
// of steps, top to bottom, with each bit matrix as a switchboard: a wire from where a tile was to where it goes, lit
// where the bit is 1. Hovering a value shows the JavaScript it stands for and the arithmetic of game.ts's operations
// on the values above it. That arithmetic is redone here only to be read, and ends at the wire's own value.
//
// The overview places the program's operations by depth, i.e. the longest chain of operations from the source table
// to each one, as Circuit counts it, colored by the part of buildGame that appended them.
import type {Circuit, Wire} from '../src/runtime/numeric-ui-core.ts'
import {type InspectWires, type InspectedRow, type ProgramPart, layout} from './game.ts'
import type {InspectedMove} from './machine.ts'
import {type Direction, boardSize, lineCell} from './rules.ts'

type Program = {circuit: Circuit, parts: readonly ProgramPart[], inspect: InspectWires}
type RowValues = InspectedMove['inspection']['rows'][number]

type StageKey = 'read' | 'occupied' | 'tilesBefore' | 'packedFrom' | 'packed' | 'merges' | 'survives' | 'finalSlot' | 'lands' | 'line'
type Stage = {
  key: StageKey
  label: string
  // tiles: 4 exponents drawn as tiles. numbers: 4 small numbers. pairs: 3 bits between neighbors. routing: a bit matrix.
  kind: 'tiles' | 'numbers' | 'pairs' | 'routing'
  // The step of slideRowLeft: 1 pack, 2 merge, 3 pack again, and 0 for the row it reads.
  step: 0 | 1 | 2 | 3
  sub?: string
  about: string
}

const stages: readonly Stage[] = [
  {key: 'read', label: 'row', kind: 'tiles', step: 0,
    about: 'The line as the program reads it, in the order of the arrow: position 0 is the edge the tiles move toward, so every move is a move to the left.'},
  {key: 'occupied', label: 'occupied', kind: 'numbers', step: 1,
    about: 'Whether each position holds a tile: row[p] ≥ 1, an exact comparison of 2 operations because exponents are whole numbers.'},
  {key: 'tilesBefore', label: 'tilesBefore', kind: 'numbers', step: 1,
    about: 'How many tiles are in front of each position, a running sum of the occupied bits. That is the slot the tile packs into.'},
  {key: 'packedFrom', label: 'packedFrom', kind: 'routing', step: 1, sub: 'position → slot',
    about: 'A bit matrix: packedFrom[slot][position] is 1 where the tile at that position packs into that slot. The 6 pairs that would move a tile right are the constant 0 and aren\'t drawn.'},
  {key: 'packed', label: 'packed', kind: 'tiles', step: 1,
    about: 'Each slot\'s exponent, Σ packedFrom[slot][position] × row[position]. At most one bit per slot is 1, so the sum picks one tile.'},
  {key: 'merges', label: 'merges', kind: 'pairs', step: 2,
    about: 'Whether each neighbor pair merges: both hold a tile, they are equal, and the pair before didn\'t merge. A chain of 3 steps instead of a loop.'},
  {key: 'survives', label: 'survives', kind: 'numbers', step: 3,
    about: 'Whether a slot\'s tile is still there after merging: it holds a tile and didn\'t merge into the slot in front of it.'},
  {key: 'finalSlot', label: 'finalSlot', kind: 'numbers', step: 3,
    about: 'Where each slot\'s tile ends up: its slot minus the merges in front of it, which closes the gaps that merges leave.'},
  {key: 'lands', label: 'lands', kind: 'routing', step: 3, sub: 'slot → final',
    about: 'A second bit matrix: lands[slot][final] is 1 where a surviving tile lands in its final slot. Again 6 pairs are the constant 0.'},
  {key: 'line', label: 'line', kind: 'tiles', step: 3,
    about: 'The line after the move, Σ lands[slot][final] × (packed[slot] + merges[slot]). A merge adds 1 to the exponent, which doubles the tile.'},
]
const stepNames = ['', '1. pack', '2. merge', '3. pack again']

const heights = {tiles: 44, numbers: 24, pairs: 24, routing: 60}
const stageGap = 6
const headerHeight = 20
const labelWidth = 104
const cellWidth = 52
const cellGap = 8
const cellX = (index: number) => labelWidth + index * (cellWidth + cellGap)
const cellCenter = (index: number) => cellX(index) + cellWidth / 2
const stripWidth = cellX(boardSize) + 100
const overviewWidth = 440
const overviewHeight = 96
// One color per part of buildGame: read rows, slide rows, write rows back, spawn, score and status, animation.
const partColors = ['#8f7a66', '#f65e3b', '#edc22e', '#5fa55a', '#4a90d9', '#d6cdc4']
// How long the playback stays on each step.
const playbackStepMs = 220

const svgNamespace = 'http://www.w3.org/2000/svg'
function svg(tag: string, attributes: Record<string, string | number>, parent?: Element, text?: string): SVGElement {
  const element = document.createElementNS(svgNamespace, tag)
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, String(value))
  if (text !== undefined) element.textContent = text
  parent?.append(element)
  return element
}

const range = (count: number) => Array.from({length: count}, (_, index) => index)
const relu = (x: number) => Math.max(0, x)
const number = (x: number) => (x < 0 ? `−${-x}` : String(x))
const tileLabel = (exponent: number) => (exponent === 0 ? '·' : String(2 ** exponent))
const lineName = (direction: Direction) => (direction === 'left' || direction === 'right' ? 'row' : 'column')
const arrows: Record<Direction, string> = {left: '←', right: '→', up: '↑', down: '↓'}

// The two expressions the strip's bits are made of, as game.ts's vocabulary builds them: an exact a === b on whole
// numbers is relu(1 − relu(D) − relu(−D)) with D = a − b, and an AND of bits is relu(sum − (count − 1)).
function equalsMath(a: number, b: number, aName: string, bName: string) {
  const difference = a - b
  const value = relu(1 - relu(difference) - relu(-difference))
  return {value, text: `${aName} === ${bName}: relu(1 − relu(D) − relu(−D)), D = ${number(a)} − ${number(b)} = ${number(difference)} → ${value}`}
}
function andMath(bits: readonly number[], names: string) {
  const value = relu(bits.reduce((total, bit) => total + bit, 0) - (bits.length - 1))
  return {value, text: `${names}: relu(${bits.map(number).join(' + ')} − ${bits.length - 1}) = ${value}`}
}

type Detail = {title: string, js: string, math: string[], meaning: string}

// What one value of the strip stands for. `index` is a position, slot or final slot, and `other` the second index of
// a bit matrix: packedFrom[index][other] and lands[index][other].
function explain(stage: StageKey, row: RowValues, move: InspectedMove, line: number, index: number, other = 0): Detail {
  const filled = (slot: number) => (row.packed[slot]! >= 1 ? 1 : 0)
  switch (stage) {
    case 'read': {
      const cell = lineCell(move.direction, line, index)
      const exponent = row.read[index]!
      return {
        title: `row[${index}] = ${exponent}${exponent === 0 ? ', empty' : `, the tile ${2 ** exponent}`}`,
        js: `board[lineCell('${move.direction}', ${line}, ${index})], i.e. board[${cell}]`,
        math: [`Σ key[d] × board[lineCell(d, ${line}, ${index})] over the 4 arrows: only the pressed key is 1, so = 1 × ${move.before[cell]} = ${move.before[cell]}`],
        meaning: `Position ${index} of ${lineName(move.direction)} ${line}, read from cell ${cell}. An exponent: 0 is empty and e is the tile 2^e.`,
      }
    }
    case 'occupied': {
      const exponent = row.read[index]!
      return {
        title: `occupied[${index}] = ${row.occupied[index]}`,
        js: `row[${index}] > 0`,
        math: [`row[${index}] ≥ 1: relu(1 − relu(−D)), D = ${exponent} − 1 = ${number(exponent - 1)} → relu(1 − ${relu(1 - exponent)}) = ${relu(1 - relu(1 - exponent))}`],
        meaning: row.occupied[index] === 1 ? `Position ${index} holds a tile.` : `Position ${index} is empty.`,
      }
    }
    case 'tilesBefore': {
      const before = row.occupied.slice(0, index)
      return {
        title: `tilesBefore[${index}] = ${row.tilesBefore[index]}`,
        js: index === 0 ? '0' : `occupied[0] + … + occupied[${index - 1}]`,
        math: [index === 0 ? 'nothing is in front of position 0: the constant 0, built once' : `${before.join(' + ')} = ${before.reduce((total, bit) => total + bit, 0)}`],
        meaning: `${row.tilesBefore[index]} tile(s) in front of position ${index}, so a tile there packs into slot ${row.tilesBefore[index]}.`,
      }
    }
    case 'packedFrom': {
      const [slot, position] = [index, other]
      const equal = equalsMath(row.tilesBefore[position]!, slot, `tilesBefore[${position}]`, String(slot))
      const both = andMath([row.occupied[position]!, equal.value], 'and')
      const bit = row.packedFrom[slot]![position]!
      return {
        title: `packedFrom[${slot}][${position}] = ${bit}`,
        js: `occupied[${position}] && tilesBefore[${position}] === ${slot}`,
        math: [equal.text, both.text],
        meaning: bit === 1 ? `The tile at position ${position} packs into slot ${slot}.` : `No tile goes from position ${position} to slot ${slot}.`,
      }
    }
    case 'packed': {
      const slot = index
      const positions = range(boardSize).filter(position => position >= slot)
      const terms = positions.map(position => `${row.packedFrom[slot]![position]}×${row.read[position]}`)
      const total = positions.reduce((sum, position) => sum + row.packedFrom[slot]![position]! * row.read[position]!, 0)
      return {
        title: `packed[${slot}] = ${row.packed[slot]}`,
        js: `Σ packedFrom[${slot}][position] × row[position]`,
        math: [`${terms.join(' + ')} = ${total}${slot > 0 ? `, and positions before ${slot} add 0: a tile never moves right` : ''}`],
        meaning: row.packed[slot] === 0 ? `No tile packs into slot ${slot}.` : `Slot ${slot} holds the tile ${2 ** row.packed[slot]!} after packing.`,
      }
    }
    case 'merges': {
      const slot = index
      const equal = equalsMath(row.packed[slot]!, row.packed[slot + 1]!, `packed[${slot}]`, `packed[${slot + 1}]`)
      const neighbor = andMath([filled(slot + 1), equal.value], `filled[${slot + 1}] && equal`)
      const math = [`filled[${slot + 1}] = packed[${slot + 1}] ≥ 1 → ${filled(slot + 1)}`, equal.text, neighbor.text]
      let value = neighbor.value
      if (slot > 0) {
        const previous = row.merges[slot - 1]!
        const result = andMath([neighbor.value, 1 - previous], `… && !merges[${slot - 1}], with !bit = 1 − bit`)
        math.push(result.text)
        value = result.value
      }
      const bit = row.merges[slot]!
      return {
        title: `merges[${slot}] = ${bit}`,
        js: `filled[${slot + 1}] && packed[${slot}] === packed[${slot + 1}]${slot > 0 ? ` && !merges[${slot - 1}]` : ''}`,
        math: value === bit ? math : [...math, `(the wire holds ${bit})`],
        meaning: bit === 1
          ? `Slot ${slot + 1}'s tile merges into slot ${slot}.`
          : neighbor.value === 1 ? `Equal, but slot ${slot} already took a merge, and each tile merges at most once.` : `Slots ${slot} and ${slot + 1} don't merge.`,
      }
    }
    case 'survives': {
      const slot = index
      const math = slot === 0
        ? `filled[0] = packed[0] ≥ 1 → ${filled(0)}`
        : andMath([filled(slot), 1 - row.merges[slot - 1]!], `filled[${slot}] && !merges[${slot - 1}]`).text
      return {
        title: `survives[${slot}] = ${row.survives[slot]}`,
        js: slot === 0 ? 'filled[0]' : `filled[${slot}] && !merges[${slot - 1}]`,
        math: [math],
        meaning: row.survives[slot] === 1 ? `Slot ${slot}'s tile stays.` : filled(slot) === 1 ? `Slot ${slot}'s tile merged into slot ${slot - 1}.` : `Slot ${slot} is empty.`,
      }
    }
    case 'finalSlot': {
      const slot = index
      const before = row.merges.slice(0, Math.max(0, slot - 1))
      return {
        title: `finalSlot[${slot}] = ${row.finalSlot[slot]}`,
        js: slot <= 1 ? String(slot) : `${slot} − (merges[0] + … + merges[${slot - 2}])`,
        math: [slot <= 1 ? 'no merge ends in front of this slot: the constant, built once' : `${slot} − (${before.join(' + ')}) = ${slot - before.reduce((total, bit) => total + bit, 0)}`],
        meaning: row.survives[slot] === 1 ? `Slot ${slot}'s tile ends in slot ${row.finalSlot[slot]}.` : 'Computed for every slot, and unused here, since no tile survives in this slot.',
      }
    }
    case 'lands': {
      const [slot, final] = [index, other]
      const equal = equalsMath(row.finalSlot[slot]!, final, `finalSlot[${slot}]`, String(final))
      const both = andMath([row.survives[slot]!, equal.value], 'and')
      const bit = row.lands[slot]![final]!
      return {
        title: `lands[${slot}][${final}] = ${bit}`,
        js: `survives[${slot}] && finalSlot[${slot}] === ${final}`,
        math: [equal.text, both.text],
        meaning: bit === 1 ? `The tile in slot ${slot} lands in final slot ${final}.` : `No tile goes from slot ${slot} to final slot ${final}.`,
      }
    }
    case 'line': {
      const final = index
      const slots = range(boardSize).filter(slot => slot >= final)
      const terms = slots.map(slot => `${row.lands[slot]![final]}×(${row.packed[slot]}+${row.merges[slot]})`)
      const total = slots.reduce((sum, slot) => sum + row.lands[slot]![final]! * (row.packed[slot]! + row.merges[slot]!), 0)
      return {
        title: `line[${final}] = ${row.line[final]}${row.line[final] === 0 ? ', empty' : `, the tile ${2 ** row.line[final]!}`}`,
        js: `Σ lands[slot][${final}] × (packed[slot] + merges[slot])`,
        math: [`${terms.join(' + ')} = ${total}`],
        meaning: row.merged[final] === 1 ? `A merged tile: its exponent went up by 1, so its value doubled.` : `Position ${final} of the line after the move.`,
      }
    }
  }
}

function explainGain(row: RowValues): Detail {
  const slots = range(boardSize - 1)
  const terms = slots.map(slot => `${row.merges[slot]}×${2 ** (row.packed[slot]! + 1)}`)
  const total = slots.reduce((sum, slot) => sum + row.merges[slot]! * 2 ** (row.packed[slot]! + 1), 0)
  return {
    title: `gain = ${row.gain}`,
    js: 'Σ merges[slot] × 2 ** (packed[slot] + 1)',
    math: [`${terms.join(' + ')} = ${total}, each power of two read from a table at packed[slot]`],
    meaning: 'The points this line scores: the value of every merged tile.',
  }
}

// The wire of one value, to find its depth.
function stageWire(wires: InspectedRow, stage: StageKey, index: number, other = 0): Wire {
  const values = wires[stage]
  return stage === 'packedFrom' || stage === 'lands' ? (values as Wire[][])[index]![other]! : (values as Wire[])[index]!
}

export function mountInspector(program: Program, board: HTMLElement) {
  const {circuit} = program
  const root = document.querySelector<HTMLElement>('.inspector')!
  const summary = root.querySelector<HTMLElement>('.summary')!
  const lineButtons = root.querySelector<HTMLElement>('.lines')!
  const strip = root.querySelector<SVGSVGElement>('.strip')!
  const slider = root.querySelector<HTMLInputElement>('.step')!
  const stepName = root.querySelector<HTMLElement>('.step-name')!
  const detail = {
    title: root.querySelector<HTMLElement>('.detail .title')!,
    js: root.querySelector<HTMLElement>('.detail .js')!,
    math: root.querySelector<HTMLElement>('.detail .math')!,
    meaning: root.querySelector<HTMLElement>('.detail .meaning')!,
  }
  const overview = root.querySelector<SVGSVGElement>('.overview')!
  const layers = root.querySelector<HTMLElement>('.layers')!
  const readingOrder = board.querySelector<SVGSVGElement>('.reading-order')!
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)')
  slider.max = String(stages.length - 1)

  // The overview: operations per depth, stacked by the part of buildGame that appended them.
  const depthOf = (wire: Wire) => circuit.depths[wire]!
  const isConstant = (wire: Wire) => circuit.constants[wire] !== undefined
  const operationCount = circuit.operations.length
  const deepest = Math.max(...circuit.depths)
  const counts = range(deepest + 1).map(() => program.parts.map(() => 0))
  let partIndex = 0
  for (let operation = 0; operation < operationCount; operation++) {
    while (operation >= program.parts[partIndex]!.end) partIndex++
    counts[depthOf((circuit.sourceSize + operation) as Wire)]![partIndex]!++
  }
  const widest = Math.max(...counts.map(perPart => perPart.reduce((total, count) => total + count, 0)))
  const barWidth = overviewWidth / deepest
  const barBottom = overviewHeight - 16
  const barScale = (barBottom - 4) / widest
  overview.setAttribute('viewBox', `0 0 ${overviewWidth} ${overviewHeight}`)
  const band = svg('rect', {class: 'band', x: 0, y: 0, width: 0, height: barBottom}, overview)
  for (let depth = 1; depth <= deepest; depth++) {
    let y = barBottom
    const total = counts[depth]!.reduce((sum, count) => sum + count, 0)
    const bar = svg('g', {}, overview)
    svg('title', {}, bar, `Layer ${depth}: ${total} operations`)
    counts[depth]!.forEach((count, part) => {
      if (count === 0) return
      const height = Math.max(1, count * barScale)
      y -= height
      svg('rect', {x: (depth - 1) * barWidth + 0.5, y, width: barWidth - 1, height, fill: partColors[part]!}, bar)
    })
  }
  const marker = svg('line', {class: 'marker', x1: 0, x2: 0, y1: 0, y2: barBottom, visibility: 'hidden'}, overview)
  svg('text', {x: 0, y: overviewHeight - 2}, overview, 'layer 1')
  svg('text', {x: overviewWidth, y: overviewHeight - 2, 'text-anchor': 'end'}, overview, `layer ${deepest}`)
  const legend = root.querySelector<HTMLElement>('.legend')!
  program.parts.forEach((part, index) => {
    const entry = document.createElement('span')
    entry.style.setProperty('--swatch', partColors[index]!)
    entry.textContent = `${part.name} (${part.end - (index === 0 ? 0 : program.parts[index - 1]!.end)})`
    legend.append(entry)
  })

  let move: InspectedMove | null = null
  let line = 0
  let step = stages.length - 1
  let hovering = false
  let playback: ReturnType<typeof setInterval> | undefined
  const stageGroups: SVGElement[] = []

  const stageDepths = (stage: StageKey) => {
    const wires = program.inspect.rows[line]!
    const depths = stageItems(stage).map(([index, other]) => stageWire(wires, stage, index, other)).filter(wire => !isConstant(wire)).map(depthOf)
    return depths.length === 0 ? null : {low: Math.min(...depths), high: Math.max(...depths)}
  }
  // The values a stage draws: every index, and for a bit matrix every pair that isn't the constant 0.
  function stageItems(stage: StageKey): [number, number][] {
    if (stage === 'merges') return range(boardSize - 1).map(slot => [slot, 0])
    if (stage === 'packedFrom') return range(boardSize).flatMap(slot => range(boardSize).filter(position => position >= slot).map(position => [slot, position] as [number, number]))
    if (stage === 'lands') return range(boardSize).flatMap(slot => range(boardSize).filter(final => final <= slot).map(final => [slot, final] as [number, number]))
    return range(boardSize).map(index => [index, 0])
  }

  function showDetail(value: Detail | null, wire: Wire | null) {
    const stage = stages[step]!
    const shown = value ?? {title: `${stage.label}, step ${step + 1} of ${stages.length}`, js: '', math: [], meaning: stage.about}
    detail.title.textContent = shown.title
    detail.js.textContent = shown.js === '' ? '' : `JS  ${shown.js}`
    detail.math.textContent = shown.math.join('\n')
    detail.meaning.textContent = shown.meaning
    if (wire !== null) {
      band.setAttribute('width', '0')
      if (isConstant(wire)) {
        marker.setAttribute('visibility', 'hidden')
        layers.textContent = 'A constant, computed while building the program, so it runs at no layer.'
      } else {
        const x = String((depthOf(wire) - 0.5) * barWidth)
        marker.setAttribute('visibility', 'visible')
        marker.setAttribute('x1', x)
        marker.setAttribute('x2', x)
        layers.textContent = `This value is computed at layer ${depthOf(wire)} of ${deepest}.`
      }
      return
    }
    marker.setAttribute('visibility', 'hidden')
    const depths = stageDepths(stage.key)
    if (depths === null) {
      band.setAttribute('width', '0')
      layers.textContent = ''
      return
    }
    band.setAttribute('x', String((depths.low - 1) * barWidth))
    band.setAttribute('width', String((depths.high - depths.low + 1) * barWidth))
    layers.textContent = depths.low === depths.high
      ? `${stage.label} is computed at layer ${depths.low} of ${deepest}.`
      : `${stage.label} is computed at layers ${depths.low} to ${depths.high} of ${deepest}.`
  }

  // A hoverable value: its visuals go in the returned group.
  function item(parent: Element, value: () => Detail, wire: Wire) {
    const group = svg('g', {class: 'item'}, parent)
    group.addEventListener('pointerenter', () => showDetail(value(), wire))
    group.addEventListener('pointerleave', () => showDetail(null, null))
    return group
  }

  function drawStrip() {
    if (move === null) return
    const current = move
    const row = current.inspection.rows[line]!
    const wires = program.inspect.rows[line]!
    strip.replaceChildren()
    stageGroups.length = 0
    const defs = svg('defs', {}, strip)
    const arrowHead = svg('marker', {id: 'strip-arrow', viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto'}, defs)
    svg('path', {d: 'M0 0 L10 5 L0 10 z', fill: '#f59563'}, arrowHead)

    svg('text', {class: 'sub', x: 0, y: 14}, strip, 'position')
    for (const index of range(boardSize)) svg('text', {class: 'index', x: cellCenter(index), y: 14, 'text-anchor': 'middle'}, strip, String(index))
    let y = headerHeight
    const stageTops: number[] = []
    stages.forEach(stage => {
      const height = heights[stage.kind]
      const group = svg('g', {class: 'stage'}, strip)
      stageGroups.push(group)
      stageTops.push(y)
      const middle = y + height / 2
      svg('text', {class: 'name', x: 0, y: stage.sub === undefined ? middle + 4 : middle - 2}, group, stage.label)
      if (stage.sub !== undefined) svg('text', {class: 'sub', x: 0, y: middle + 12}, group, stage.sub)
      const explainAt = (index: number, other = 0) => () => explain(stage.key, row, current, line, index, other)
      if (stage.kind === 'tiles') {
        const values = row[stage.key as 'read' | 'packed' | 'line']
        values.forEach((exponent, index) => {
          const cell = item(group, explainAt(index), stageWire(wires, stage.key, index))
          cell.setAttribute('data-exponent', String(exponent))
          svg('rect', {class: exponent === 0 ? 'chip empty' : 'chip', x: cellX(index), y, width: cellWidth, height, rx: 5}, cell)
          const label = tileLabel(exponent)
          svg('text', {class: label.length > 3 ? 'chip-text small' : 'chip-text', x: cellCenter(index), y: middle + 6, 'text-anchor': 'middle'}, cell, exponent === 0 ? '' : label)
          svg('rect', {class: 'halo', x: cellX(index) - 2, y: y - 2, width: cellWidth + 4, height: height + 4, rx: 6}, cell)
        })
      } else if (stage.kind === 'numbers') {
        const values = row[stage.key as 'occupied' | 'tilesBefore' | 'survives' | 'finalSlot']
        const isBit = stage.key === 'occupied' || stage.key === 'survives'
        values.forEach((value, index) => {
          const cell = item(group, explainAt(index), stageWire(wires, stage.key, index))
          if (isBit && value === 1) svg('rect', {class: 'pill', x: cellCenter(index) - 11, y: y + 3, width: 22, height: height - 6, rx: 9}, cell)
          const unused = stage.key === 'finalSlot' && row.survives[index] === 0
          svg('text', {class: unused ? 'unused' : '', x: cellCenter(index), y: middle + 4, 'text-anchor': 'middle'}, cell, String(value))
          svg('rect', {class: 'halo', x: cellX(index), y: y + 1, width: cellWidth, height: height - 2, rx: 4}, cell)
        })
      } else if (stage.kind === 'pairs') {
        for (const slot of range(boardSize - 1)) {
          const x = cellX(slot) + cellWidth + cellGap / 2
          const cell = item(group, explainAt(slot), stageWire(wires, 'merges', slot))
          svg('path', {class: 'wire', d: `M${cellCenter(slot)} ${y + 2} V${middle} H${cellCenter(slot + 1)} V${y + 2}`, fill: 'none'}, cell)
          if (row.merges[slot] === 1) svg('rect', {class: 'pill', x: x - 11, y: y + 3, width: 22, height: height - 6, rx: 9}, cell)
          else svg('rect', {x: x - 11, y: y + 3, width: 22, height: height - 6, rx: 9, fill: '#faf8ef'}, cell)
          svg('text', {x, y: middle + 4, 'text-anchor': 'middle'}, cell, String(row.merges[slot]))
          svg('rect', {class: 'halo', x: x - 13, y: y + 1, width: 26, height: height - 2, rx: 10}, cell)
        }
      } else {
        // A switchboard: the top row's columns are where a tile was, the bottom row's where it goes.
        const isPacking = stage.key === 'packedFrom'
        const pairs = stageItems(stage.key).map(([first, second]) => {
          const [from, to] = isPacking ? [second, first] : [first, second]
          const bit = isPacking ? row.packedFrom[first]![second]! : row.lands[first]![second]!
          return {first, second, from, to, bit}
        })
        // Lit wires last, so they're drawn on top.
        pairs.sort((a, b) => a.bit - b.bit)
        for (const {first, second, from, to, bit} of pairs) {
          const cell = item(group, explainAt(first, second), stageWire(wires, stage.key, first, second))
          const points = {x1: cellCenter(from), y1: y + 4, x2: cellCenter(to), y2: y + height - 4}
          svg('line', {...points, class: 'hit'}, cell)
          svg('line', {...points, class: bit === 1 ? 'wire on' : 'wire', ...(bit === 1 ? {'marker-end': 'url(#strip-arrow)'} : {})}, cell)
        }
      }
      y += height + stageGap
    })
    // The gain, below the line.
    const gainCell = item(strip, () => explainGain(row), wires.gain)
    svg('text', {class: 'name', x: 0, y: y + 14}, gainCell, 'gain')
    svg('text', {x: cellX(0), y: y + 14}, gainCell, `${row.gain} points`)
    svg('rect', {class: 'halo', x: -4, y: y - 2, width: cellX(2), height: 22, rx: 4}, gainCell)
    y += 26

    // Brackets that group the stages into slideRowLeft's three steps.
    for (const stepIndex of [1, 2, 3]) {
      const members = stages.flatMap((stage, index) => (stage.step === stepIndex ? [index] : []))
      const top = stageTops[members[0]!]! + 2
      const last = members.at(-1)!
      const bottom = stageTops[last]! + heights[stages[last]!.kind] - 2
      const x = cellX(boardSize) + 6
      svg('line', {class: 'bracket', x1: x, y1: top, x2: x, y2: bottom}, strip)
      svg('text', {class: 'sub', x: x + 6, y: (top + bottom) / 2 + 4}, strip, stepNames[stepIndex])
    }
    strip.setAttribute('viewBox', `0 0 ${stripWidth} ${y}`)
    showStep()
  }

  function showStep() {
    stageGroups.forEach((group, index) => {
      group.classList.toggle('hidden', index > step)
      group.classList.toggle('current', index === step)
    })
    slider.value = String(step)
    stepName.textContent = stages[step]!.label
    showDetail(null, null)
  }

  // The reading order on the board: an arrow along each line toward its position 0, the shown line stronger and
  // numbered.
  function drawReadingOrder() {
    if (move === null) return
    const direction = move.direction
    readingOrder.replaceChildren()
    const defs = svg('defs', {}, readingOrder)
    const head = svg('marker', {id: 'reading-arrow', viewBox: '0 0 10 10', refX: 6, refY: 5, markerWidth: 4, markerHeight: 4, orient: 'auto'}, defs)
    svg('path', {d: 'M0 0 L10 5 L0 10 z', fill: '#776e65'}, head)
    const center = (cell: number) => ({
      x: layout.gap + (cell % boardSize) * layout.pitch + layout.tileSize / 2,
      y: layout.gap + Math.floor(cell / boardSize) * layout.pitch + layout.tileSize / 2,
    })
    for (const index of range(boardSize)) {
      const start = center(lineCell(direction, index, boardSize - 1))
      const end = center(lineCell(direction, index, 0))
      const shown = index === line
      svg('line', {
        x1: start.x, y1: start.y, x2: end.x, y2: end.y, stroke: '#776e65', 'stroke-width': shown ? 6 : 3,
        'stroke-linecap': 'round', opacity: shown ? 0.85 : 0.35, 'marker-end': 'url(#reading-arrow)',
      }, readingOrder)
    }
    const cells = range(boardSize).map(position => lineCell(direction, line, position))
    const xs = cells.map(cell => (cell % boardSize) * layout.pitch + layout.gap)
    const ys = cells.map(cell => Math.floor(cell / boardSize) * layout.pitch + layout.gap)
    svg('rect', {
      x: Math.min(...xs) - 4, y: Math.min(...ys) - 4, width: Math.max(...xs) - Math.min(...xs) + layout.tileSize + 8,
      height: Math.max(...ys) - Math.min(...ys) + layout.tileSize + 8, rx: 9, fill: 'none', stroke: '#f65e3b', 'stroke-width': 4,
    }, readingOrder)
    cells.forEach((cell, position) => {
      const x = (cell % boardSize) * layout.pitch + layout.gap + 16
      const y = Math.floor(cell / boardSize) * layout.pitch + layout.gap + 16
      svg('circle', {cx: x, cy: y, r: 11, fill: '#f65e3b'}, readingOrder)
      svg('text', {x, y: y + 5, 'text-anchor': 'middle', fill: 'white', 'font-size': 14, 'font-weight': 'bold'}, readingOrder, String(position))
    })
  }

  function drawLineButtons() {
    if (move === null) return
    lineButtons.replaceChildren()
    for (const index of range(boardSize)) {
      const button = document.createElement('button')
      button.type = 'button'
      button.textContent = `${lineName(move.direction)} ${index}`
      button.setAttribute('aria-pressed', String(index === line))
      button.addEventListener('click', () => selectLine(index))
      lineButtons.append(button)
    }
  }

  function selectLine(index: number) {
    line = index
    writeSummary()
    drawLineButtons()
    drawStrip()
    drawReadingOrder()
  }

  const showReading = () => board.classList.toggle('reading', hovering || playback !== undefined)
  function stopPlayback() {
    clearInterval(playback)
    playback = undefined
    showReading()
  }
  function play() {
    stopPlayback()
    if (reducedMotion.matches) {
      step = stages.length - 1
      showStep()
      return
    }
    step = 0
    showStep()
    playback = setInterval(() => {
      step++
      showStep()
      if (step === stages.length - 1) stopPlayback()
    }, playbackStepMs)
    showReading()
  }

  // The most eventful line: merges first, then tiles that moved.
  function pickLine(current: InspectedMove) {
    const interest = (row: RowValues) =>
      10 * row.merges.reduce((total, bit) => total + bit, 0) + row.line.filter((exponent, index) => exponent !== row.read[index]).length
    const scores = current.inspection.rows.map(interest)
    const best = Math.max(...scores)
    if (best > 0) return scores.indexOf(best)
    const withTile = current.inspection.rows.findIndex(row => row.occupied.some(bit => bit === 1))
    return withTile === -1 ? 0 : withTile
  }

  function writeSummary() {
    if (move === null) return
    const current = move
    const tiles = (exponents: readonly number[]) => exponents.map(tileLabel).join(' ')
    const row = current.inspection.rows[line]!
    const gain = current.inspection.rows.reduce((total, each) => total + each.gain, 0)
    const spawned = current.inspection.spawned.findIndex(bit => bit === 1)
    const spawn = spawned === -1
      ? 'Nothing moved, so no tile spawned.'
      : `Then a ${2 ** current.inspection.board[spawned]!} spawned in row ${Math.floor(spawned / boardSize)}, column ${spawned % boardSize}.`
    summary.textContent = `${arrows[current.direction]} ${current.direction}: ${lineName(current.direction)} ${line} went ${tiles(row.read)} → ${tiles(row.line)}. `
      + `${gain > 0 ? `The key scored ${gain} points. ` : ''}${spawn}`
  }

  slider.addEventListener('input', () => {
    stopPlayback()
    step = Number(slider.value)
    showStep()
  })
  root.querySelector('.step-back')!.addEventListener('click', () => {
    stopPlayback()
    step = Math.max(0, step - 1)
    showStep()
  })
  root.querySelector('.step-forward')!.addEventListener('click', () => {
    stopPlayback()
    step = Math.min(stages.length - 1, step + 1)
    showStep()
  })
  root.querySelector('.replay')!.addEventListener('click', play)
  root.addEventListener('pointerenter', () => {
    hovering = true
    showReading()
  })
  root.addEventListener('pointerleave', () => {
    hovering = false
    showReading()
  })

  return {
    // Shows a move once: the host calls this every frame with the machine's last move.
    show(next: InspectedMove | null) {
      if (next === null || next === move) return
      move = next
      root.classList.add('has-move')
      selectLine(pickLine(next))
      play()
    },
  }
}
