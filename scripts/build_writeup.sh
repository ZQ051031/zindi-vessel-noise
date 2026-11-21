#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${PYTHON:-$ROOT/.venv/bin/python}" "$ROOT/scripts/writeup_metadata.py"
cd "$ROOT/writeup"
pdflatex -interaction=nonstopmode -halt-on-error writeup.tex
pdflatex -interaction=nonstopmode -halt-on-error writeup.tex
pdfinfo writeup.pdf | rg '^Pages:'
printf 'Synthetic disclaimer lines: '
pdftotext writeup.pdf - | grep -c 'SYNTHETIC EXAMPLE'
