#!/usr/bin/env bash
# Run warmed, bounded QuickDer solves on four 10-second decoded clips.
# Usage: DLETO_METAL_PROJECT=... ./run_real_quickder.sh SMALL_TENSOR_FOLDER LOG_FOLDER
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
data=${1:?small video tensor folder required}
logs=${2:?log folder required}
shift 2
: "${DLETO_METAL_PROJECT:?Set DLETO_METAL_PROJECT to a Julia environment with Metal.jl}"
mkdir -p "$logs"
if (( $# )); then cases=("$@"); else cases=(landscape-720p landscape-1080p portrait-720p portrait-1080p); fi
for case in "${cases[@]}"; do
    if [[ $case == landscape-* ]]; then dims=(80 45 300); else dims=(45 80 300); fi
    set +e
    JULIA_NUM_THREADS=12 timeout 180s julia --compiled-modules=existing \
      --project="$DLETO_METAL_PROJECT" "$root/bench/python_vs_julia/video_quickder.jl" \
      "$data/$case" "${dims[@]}" > "$logs/$case.log" 2>&1
    rc=$?
    set -e
    echo "$rc" > "$logs/$case.exit"
    echo "$case exit=$rc" | tee -a "$logs/status.log"
done
echo complete > "$logs/exit.status"
