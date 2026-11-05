// Shared style for the basic_python booklets.
// Compile from basic_python/:  ./build_booklets.sh   (or: typst compile --root . booklet/01_foundations.typ)
#import "@preview/cetz:0.5.3"
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

// ── palette ─────────────────────────────────────────────────────────────
#let ink = rgb("1f2328")
#let muted = rgb("667080")
#let rule-color = rgb("d0d7de")
#let control-color = rgb("b45309") // stage A: normal control flow (amber)
#let branchless-color = rgb("1d4ed8") // stage B: branchless scalar arithmetic (blue)
#let network-color = rgb("047857") // stage C: matrices + ReLU (green)
#let tint(c) = c.lighten(90%)

// ── math operators ──────────────────────────────────────────────────────
#let relu = math.op("relu")
#let step = math.op("step")
#let sel = math.op("select")
#let AND = math.op("AND")
#let OR = math.op("OR")
#let NOT = math.op("NOT")

// ── blocks ──────────────────────────────────────────────────────────────
#let note(title, body) = block(
  width: 100%, inset: 10pt, breakable: true,
  stroke: (left: 2pt + muted), fill: rgb("f4f6f8"),
)[
  #text(font: "Avenir Next", size: 9.5pt, weight: "semibold")[#title]
  #v(2pt)
  #body
]

#let key-idea(body) = block(
  width: 100%, inset: 10pt, breakable: false, radius: 3pt,
  stroke: 0.8pt + ink, fill: rgb("fffbea"),
)[
  #text(font: "Avenir Next", size: 9.5pt, weight: "semibold")[Key idea]
  #v(2pt)
  #body
]

#let _stage-names = (
  "A": ("A · control flow", control-color),
  "B": ("B · branchless arithmetic", branchless-color),
  "C": ("C · network (matrices + relu)", network-color),
)

// A labelled box for one of the three stages, e.g. #stage("B")[ ... ]
#let stage(letter, body) = {
  let (label, color) = _stage-names.at(letter)
  block(width: 100%, breakable: true, radius: 3pt, stroke: 0.7pt + color, clip: true)[
    #block(width: 100%, fill: color, inset: (x: 8pt, y: 4pt), below: 0pt)[
      #text(font: "Avenir Next", size: 8.5pt, weight: "semibold", fill: white)[#label]
    ]
    #block(width: 100%, inset: 8pt, fill: tint(color), above: 0pt)[#body]
  ]
}

// Two stages next to each other, e.g. #side-by-side("A", [...], "B", [...])
#let side-by-side(l, left, r, right) = grid(
  columns: (1fr, 1fr), gutter: 8pt,
  stage(l, left), stage(r, right),
)

// One example section header: number, title, and the file it lives in.
#let example(number, title, file) = {
  heading(level: 1)[#number #title]
  text(font: "Menlo", size: 8pt, fill: muted)[basic_python/#file]
  v(4pt)
}

// ── whole-document setup ────────────────────────────────────────────────
#let booklet(chapter: "", title: "", subtitle: "", body) = {
  set document(title: chapter + " " + title)
  set page(
    paper: "a4",
    margin: (x: 20mm, top: 20mm, bottom: 20mm),
    numbering: "1",
    header: context {
      if counter(page).get().first() > 1 {
        set text(font: "Avenir Next", size: 8pt, fill: muted)
        [CONTROL FLOW → DATA FLOW #h(1fr) #upper(chapter + " · " + title)]
        line(length: 100%, stroke: 0.4pt + rule-color)
      }
    },
  )
  set text(font: "Libertinus Serif", size: 10.5pt, lang: "en", fill: ink)
  set par(justify: true, leading: 0.62em)
  show heading: set text(font: "Avenir Next", weight: "semibold")
  show heading.where(level: 1): set text(size: 17pt)
  show heading.where(level: 1): set block(above: 1.6em, below: 0.6em)
  show heading.where(level: 2): set text(size: 12pt)
  show heading.where(level: 2): set block(above: 1.3em, below: 0.6em)
  show raw: set text(font: "Menlo", size: 8pt)
  show raw.where(block: true): it => block(
    width: 100%, fill: rgb("f6f8fa"), inset: 8pt, radius: 3pt, breakable: true, it,
  )
  set table(inset: 5pt, stroke: 0.4pt + rule-color)
  show table: set text(size: 9pt)

  // cover
  v(12mm)
  text(font: "Avenir Next", size: 9pt, tracking: 1.5pt, fill: muted)[CONTROL FLOW → DATA FLOW · #upper(chapter)]
  v(4mm)
  text(font: "Avenir Next", size: 30pt, weight: "semibold")[#title]
  v(2mm)
  text(size: 13pt, fill: muted)[#subtitle]
  v(6mm)
  line(length: 100%, stroke: 0.6pt + ink)
  v(4mm)
  body
}
