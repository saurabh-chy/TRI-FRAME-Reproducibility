#!/usr/bin/env bash
# Rebuilds the statistics, the paper's tables and figures, and the supplementary material from output/full.
# Usage: bash run_analysis.sh <path to output/full>
set -e
R="${1:-../output/full}"; HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="$HERE/../reproduced"
mkdir -p "$OUT"
python "$HERE/statistical_tests.py" "$R" "$OUT/stats"
python "$HERE/make_figures_tables.py" "$R" "$OUT/exhibits" "$OUT/stats"
[ -d "$R/bounds" ] && python "$HERE/bounds_analysis.py" "$R" "$OUT/bounds"
python "$HERE/make_supplement.py" "$R" "$OUT/supplementary_material.md" "$OUT/stats"
command -v pandoc >/dev/null && pandoc "$OUT/supplementary_material.md" -o "$OUT/supplementary_material.docx" || echo "pandoc not found: Markdown supplement only"
echo "Done: $OUT"
