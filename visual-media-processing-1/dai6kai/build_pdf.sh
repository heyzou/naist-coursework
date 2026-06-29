#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CACHE_DIR="$PWD/.tex-cache"
mkdir -p "$CACHE_DIR"

export TEXMFVAR="$CACHE_DIR"
export TEXMFCACHE="$CACHE_DIR"

# Keep lualatex format in sync with the TeX packages found in TEXMFHOME.
fmtutil-user --byfmt lualatex >/dev/null

latexmk -g -lualatex -jobname=2611193_HeizoNakai_visual_media_processing1_dai6kai main.tex
