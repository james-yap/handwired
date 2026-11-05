#!/bin/sh
# Compile every booklet chapter to a PDF next to its .typ source.
# Run from anywhere; --root is basic_python/ so booklets can read the .py files.
set -e
cd "$(dirname "$0")"
for f in booklet/[0-9]*.typ; do
  echo "typst compile $f"
  typst compile --root . "$f"
done
