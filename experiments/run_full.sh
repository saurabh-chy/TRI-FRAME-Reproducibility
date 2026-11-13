#!/usr/bin/env bash
# Linux/macOS equivalent of run_full.bat: runs the full pipeline and restarts it if it stops early.
cd "$(dirname "$0")"
for i in $(seq 1 30); do
  echo "==== attempt $i started $(date) ====" >> run_full_console.log
  python -u TRI_FRAME_full.py >> run_full_console.log 2>&1 && { echo "==== FULL RUN FINISHED $(date) ====" >> run_full_console.log; exit 0; }
  sleep 30
done
echo "Gave up after 30 attempts"; exit 1
