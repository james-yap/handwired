// The host: the plain JavaScript around the program. Key handlers only queue key presses with their timestamps. Each
// frame, advanceFrame plans the evaluations: one per key press, which first steps the springs to the key's timestamp,
// settle checks every 64 ms of simulated time, and one at the frame's own timestamp. The host writes each
// evaluation's inputs into the source table, runs the program, and writes the next state back.
import {advanceFrame, defaultSpringClockConfig, springMatrixPower, springStepMatrix, stoppedSpringClock} from '../src/runtime/host-timing.ts'
import {createEvaluator} from '../src/runtime/cells.ts'
import {type GameOutput, buildGame, springConstants} from './game.ts'
import {type GameKey, drawSpawn} from './rules.ts'

export type KeyPress = {key: GameKey, timeStamp: number}

// `random` draws the spawned tiles' places and values. Tests pass a seeded one.
export function createGameMachine(random: () => number = Math.random) {
  const built = buildGame()
  const {cells} = built
  const evaluate = createEvaluator(built)
  // The spring matrix of every step count a frame can plan, see springMatrixPower.
  const step = springStepMatrix(springConstants)
  const matrices = Array.from({length: defaultSpringClockConfig.maxStepsPerFrame + 1}, (_, steps) => springMatrixPower(step, steps))
  const source = new Float32Array(built.initialSource)
  const copy = new Float32Array(source.length)
  let queued: KeyPress[] = []

  const machine = {
    built,
    source,
    // The host requests frames while clock.animatedUntilMs isn't null.
    clock: stoppedSpringClock,
    queue(press: KeyPress) {
      queued.push(press)
    },
    // A best score saved by an earlier visit, a whole number within the best cell's range.
    restoreBest(best: number) {
      source[cells.best] = best
      built.validate(source)
    },
    step(frameTime: number): GameOutput {
      let frameOutput: GameOutput | undefined
      // With 'events and settle checks', a frame's own evaluation runs on a copy of the table, so the stored state
      // depends only on the key presses and their timestamps, not on the frame rate.
      machine.clock = advanceFrame(machine.clock, frameTime, queued, evaluation => {
        const values = evaluation.writesBack ? source : copy
        if (!evaluation.writesBack) copy.set(source)
        const matrix = matrices[evaluation.steps]!
        cells.springMatrix.forEach((wire, index) => { values[wire] = matrix[index]! })
        values[cells.settleCheck] = evaluation.settleCheck ? 1 : 0
        values[cells.snap] = evaluation.snap ? 1 : 0
        // Key presses with the same timestamp share an evaluation, and the program handles one key per evaluation,
        // so the last of them wins. Two key presses rarely share a timestamp.
        const press = evaluation.events.at(-1)
        if (press !== undefined) {
          values[cells.keys[press.key]] = 1
          for (const spawn of cells.spawns) {
            const draw = drawSpawn(random)
            values[spawn.slot] = draw.slot
            values[spawn.four] = draw.four
          }
        }
        const result = evaluate(values)
        if (evaluation.writesBack) source.set(result.nextValues)
        frameOutput = result.output
        return result.output.active > 0
      }, 'events and settle checks')
      queued = []
      return frameOutput!
    },
  }
  return machine
}
