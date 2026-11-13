#!/usr/bin/env bash
# Linux/macOS: per-feature attribution and hypervolume over all runs (resumable).
cd "$(dirname "$0")"
for i in $(seq 1 30); do
  python -u TRI_FRAME_posthoc.py --part both --hv-runs 25 >> run_posthoc_console.log 2>&1 && exit 0
  sleep 30
done
exit 1
