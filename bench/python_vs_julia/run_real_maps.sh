#!/usr/bin/env bash
# Time one whole-tensor derivation-map call on decoded 10-second video clips.
# Usage: DLETO_METAL_PROJECT=/path/to/metal/env ./run_real_maps.sh /path/to/prepared /path/to/logs
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
data=${1:?prepared video directory required}
logs=${2:?log directory required}
: "${DLETO_METAL_PROJECT:?Set DLETO_METAL_PROJECT to a Julia environment with Metal.jl}"
mkdir -p "$logs"
trap 'echo failed > "$logs/exit.status"' ERR
for orientation in landscape portrait; do
  for size in 720 1080; do
    case="$orientation-${size}p"
    if [[ $orientation == landscape ]]; then
      if [[ $size == 720 ]]; then dims=(1280 720 300); else dims=(1920 1080 300); fi
    else
      if [[ $size == 720 ]]; then dims=(720 1280 300); else dims=(1080 1920 300); fi
    fi
    JULIA_NUM_THREADS=12 julia --compiled-modules=existing --project="$DLETO_METAL_PROJECT" \
      "$root/bench/python_vs_julia/video_julia.jl" "$data/$case" metal "${dims[@]}" 2 5 \
      > "$logs/$case-julia.log" 2>&1
    "$root/python/.venv/bin/python" -u "$root/bench/python_vs_julia/video_python.py" \
      "$data/$case" --shape "${dims[@]}" --backend mps --host-threads 12 \
      --warmups 2 --repeats 5 > "$logs/$case-python.log" 2>&1
    echo "$case complete" | tee -a "$logs/status.log"
  done
done
echo complete > "$logs/exit.status"
