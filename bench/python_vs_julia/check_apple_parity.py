"""Compare sampled Julia Metal and Python MPS outputs from the Apple sweep."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    args = parser.parse_args()
    for n in args.sizes:
        jl = np.fromfile(args.folder / f"julia_metal_sample_{n}.bin", dtype="<f4")
        py = np.fromfile(args.folder / f"python_mps_sample_{n}.bin", dtype="<f4")
        if jl.size != 16 or py.size != 16:
            raise ValueError(f"Expected 16 samples for n={n}; got {jl.size}, {py.size}")
        sample_relative = float(np.linalg.norm(jl.astype(float) - py.astype(float)) /
                                max(np.linalg.norm(py.astype(float)), 1.0))
        py_norm = json.loads((args.folder / f"accel_{n}.json").read_text())["output_norm"]
        metal_log = (args.folder / f"metal_{n}.log").read_text()
        match = re.search(r"output_norm=([^\s]+)", metal_log)
        if match is None:
            raise ValueError(f"No output norm for n={n}")
        norm_relative = abs(float(match.group(1)) - py_norm) / max(abs(py_norm), 1.0)
        print(f"n={n}: sample relative difference={sample_relative:.3e}, "
              f"norm relative difference={norm_relative:.3e}")
        if sample_relative > 1e-4 or norm_relative > 1e-4:
            raise SystemExit(f"GPU parity exceeded tolerance at n={n}")


if __name__ == "__main__":
    main()
