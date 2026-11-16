#!/usr/bin/env bash
# Runs the exact worst-case / certified-bounds analysis (Proposition 1), restarting automatically until it finishes.
# Usage (inside experiments/): bash run_bounds.sh
cd "$(dirname "$0")"
for i in $(seq 1 20); do
  python TRI_FRAME_bounds.py --part both && break
  echo "Restarting ($i) ..."; sleep 10
done
