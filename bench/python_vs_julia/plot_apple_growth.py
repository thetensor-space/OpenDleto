"""Plot the Apple Accelerate and GPU sweep from per-size checkpoints."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def julia_result(path: Path) -> dict[str, str]:
    lines = [line for line in path.read_text().splitlines() if line.startswith("RESULT ")]
    if len(lines) != 1:
        raise ValueError(f"Expected one result in {path}; found {len(lines)}")
    return dict(re.findall(r"(\w+)=([^\s]+)", lines[0]))


def main(folder: Path) -> None:
    output = Path(__file__).parent
    cpu = {}
    for line in (folder / "julia-cpu-accelerate.log").read_text().splitlines():
        if line.startswith("RESULT "):
            values = dict(re.findall(r"(\w+)=([^\s]+)", line))
            cpu[int(values["n"])] = values
    sizes = sorted(int(path.stem.split("_")[1]) for path in folder.glob("accel_*.json"))
    rows = []
    gpu_julia_threads = set()
    python_torch_threads = set()
    for n in sizes:
        py = json.loads((folder / f"accel_{n}.json").read_text())
        jl_gpu = julia_result(folder / f"metal_{n}.log")
        gpu_julia_threads.add(int(jl_gpu["julia_threads"]))
        python_torch_threads.add(int(py["metadata"]["torch_threads"]))
        row = {
            "n": n,
            "julia_accelerate_cpu_forward_ms": 1000 * float(cpu[n]["forward_seconds"]) if n in cpu else "",
            "python_accelerate_cpu_forward_ms": 1000 * py["cpu_numpy_forward_seconds"] if "cpu_numpy_forward_seconds" in py else "",
            "julia_metal_gpu_host_forward_ms": 1000 * float(jl_gpu["forward_host_seconds"]),
            "python_mps_gpu_host_forward_ms": 1000 * py["gpu_mps_forward_host_seconds"],
            "python_mps_gpu_device_forward_ms": 1000 * py["gpu_mps_forward_seconds"],
            "julia_metal_gpu_host_adjoint_ms": 1000 * float(jl_gpu["adjoint_host_seconds"]),
            "python_accelerate_cpu_adjoint_ms": 1000 * py["cpu_numpy_adjoint_seconds"] if "cpu_numpy_adjoint_seconds" in py else "",
            "julia_accelerate_cpu_adjoint_ms": 1000 * float(cpu[n]["adjoint_seconds"]) if n in cpu else "",
        }
        rows.append(row)
    cpu_julia_threads = {int(values["julia_threads"]) for values in cpu.values()}
    if any(len(counts) != 1 for counts in
           (cpu_julia_threads, gpu_julia_threads, python_torch_threads)):
        raise ValueError("Thread settings changed across benchmark checkpoints")
    cpu_julia_threads = cpu_julia_threads.pop()
    gpu_julia_threads = gpu_julia_threads.pop()
    python_torch_threads = python_torch_threads.pop()
    csv_path = output / "apple-results-2026-11-18.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(10.5, 5.8), layout="constrained")
    series = [
        (f"Julia CPU · Accelerate ({cpu_julia_threads} Julia host; BLAS auto)",
         "julia_accelerate_cpu_forward_ms", "#166534", "o"),
        ("Python CPU · Accelerate (BLAS auto)",
         "python_accelerate_cpu_forward_ms", "#2563eb", "s"),
        (f"Julia GPU · Metal ({gpu_julia_threads} Julia host)",
         "julia_metal_gpu_host_forward_ms", "#d97706", "^"),
        (f"Python GPU · MPS ({python_torch_threads} PyTorch host)",
         "python_mps_gpu_host_forward_ms", "#be185d", "D"),
    ]
    for label, column, color, marker in series:
        points = [(row["n"], row[column]) for row in rows if row[column] != ""]
        ax.plot([x for x, _ in points], [y for _, y in points], label=label,
                color=color, marker=marker, markersize=5, linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([8, 16, 32, 64, 128, 256, 512])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Tensor side length n (n × n × n)")
    ax.set_ylabel("Forward map time (ms, log scale; lower is faster)")
    ax.set_title("OpenDleto on Apple silicon: CPU through 128, GPU through 512")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9,
              title="Configured host threads; device worker counts unmeasured",
              title_fontsize=8)
    fig.savefig(output / "apple-growth-2026-11-18.png", dpi=180)
    plt.close(fig)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
