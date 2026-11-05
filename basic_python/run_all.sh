#!/bin/sh
# Run every example. Each one asserts that stages A, B and C agree on every input.
set -e
cd "$(dirname "$0")"
for f in [0-9]*/[0-9]*.py; do
  echo "── $f"
  uv run --quiet "$f"
done
echo "all examples passed"
