import {
	createSourceTable, createEvaluator, input,
} from './src/runtime/cells.ts'
const table = createSourceTable()
const cells = table.declare({
	viewportWidth: input([100, 2000], { initial: 800, grid: 1 }),
})
const circuit = table.createCircuit()
const left = circuit.affine([[cells.viewportWidth, 0.5]], -50)
const built = { cells, output: { left }, ...table.finish() }
const evaluate = createEvaluator(built)
const source = new Float32Array(built.initialSource)
for (const width of [800, 1000]) {
	source[cells.viewportWidth] = width
	const { output } = evaluate(source)
	if (output.left !== width / 2 - 50) throw Error('wrong position')
	console.log(width, output.left)
}
