#!/usr/bin/env bash
# Complete small-tensor CPU sweep. Run from any directory; results go in OUT.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:-/tmp/opendleto-full-sweep}
repeats=${2:-3}
scope=${3:-small}
mkdir -p "$out"
cd "$root"
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1
"$root/python/.venv/bin/python" bench/python_vs_julia/full_sweep_inputs.py "$out" > "$out/inputs.log"
: > "$out/python.jsonl"
: > "$out/julia.log"
if [[ "$scope" == extended ]]; then
  cases=('generic8 8 8 8' 'diagonal8 8 8 8' 'embedded10 10 10 10')
elif [[ "$scope" == small ]]; then
  cases=('generic4 4 4 4' 'diagonal4 4 4 4' 'embedded5 5 5 5')
else
  echo "scope must be small or extended" >&2
  exit 2
fi
for spec in "${cases[@]}"; do
  read -r name d1 d2 d3 <<< "$spec"
  chisels=(universal adjoint)
  [[ "$scope" == extended ]] && chisels=(universal)
  for chisel in "${chisels[@]}"; do
    echo "RUN python $name $chisel" >&2
    "$root/python/.venv/bin/python" bench/python_vs_julia/full_sweep.py "$out" \
      --case "$name" --shape "$d1" "$d2" "$d3" --chisel "$chisel" \
      --repeats "$repeats" >> "$out/python.jsonl"
    echo "RUN julia $name $chisel" >&2
    julia --compiled-modules=existing --project=. bench/python_vs_julia/full_sweep.jl \
      "$out" "$name" "$chisel" "$repeats" "$d1" "$d2" "$d3" >> "$out/julia.log" 2>&1
  done
done
"$root/python/.venv/bin/python" bench/python_vs_julia/summarize_full_sweep.py "$out"
echo "DONE $out/full-sweep.csv" >&2
