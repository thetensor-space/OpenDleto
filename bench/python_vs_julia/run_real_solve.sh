#!/usr/bin/env bash
# A bounded full null-space solve on a decoded 300-frame grayscale tensor.
# Usage: DLETO_METAL_PROJECT=... ./run_real_solve.sh CASE_FOLDER WIDTH HEIGHT LOG_FOLDER
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
case_folder=${1:?case folder required}
width=${2:?processing width required}
height=${3:?processing height required}
logs=${4:?log folder required}
nd=${5:--1}
maxiter=${DLETO_SOLVE_MAXITER:-50}
time_limit=${DLETO_SOLVE_TIMEOUT:-180s}
: "${DLETO_METAL_PROJECT:?Set DLETO_METAL_PROJECT to a Julia environment with Metal.jl}"
mkdir -p "$logs"
name=$(basename "$case_folder")
set +e
JULIA_NUM_THREADS=12 timeout "$time_limit" julia --compiled-modules=existing \
  --project="$DLETO_METAL_PROJECT" "$root/bench/python_vs_julia/video_solve.jl" \
  "$case_folder" "$width" "$height" 300 4 "$maxiter" 1e-4 20261118 "$nd" \
  > "$logs/$name-nd${nd}-solve.log" 2>&1
rc=$?
set -e
echo "$rc" > "$logs/$name-nd${nd}-solve.exit"
exit "$rc"
