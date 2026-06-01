#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CACHE_DIR="$PWD/.tex-cache"
mkdir -p "$CACHE_DIR"
export TEXMFVAR="$CACHE_DIR"
export TEXMFCACHE="$CACHE_DIR"
fmtutil-user --byfmt lualatex >/dev/null
latexmk -g -lualatex -jobname=2611193_HeizoNakai_Assignment1 main.tex
