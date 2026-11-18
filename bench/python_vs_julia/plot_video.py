"""Plot one-channel and RGB video-tensor timing from 5x11 benchmark logs."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def samples(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    julia_log = (folder / "julia-metal-5x11.log").read_text()
    julia_line = next(line for line in julia_log.splitlines() if line.startswith("RESULT "))
    match = re.search(r"seconds=\[([^\]]+)\]", julia_line)
    if match is None:
        raise ValueError(f"Julia timing samples missing in {folder}")
    julia = np.array([float(value) for value in match.group(1).split(",")])
    python_log = (folder / "python-mps-5x11.log").read_text()
    python_line = next(line for line in python_log.splitlines() if line.startswith("RESULT "))
    python = np.array(json.loads(python_line.removeprefix("RESULT "))["seconds"])
    if len(julia) != 11 or len(python) != 11:
        raise ValueError(f"Expected 11 timed calls per backend in {folder}")
    return julia, python


def main(mono_folder: Path, rgb_folder: Path) -> None:
    cases = [("One channel", *samples(mono_folder)),
             ("RGB (three channels)", *samples(rgb_folder))]
    output = Path(__file__).parent
    with (output / "video-results-2026-11-18.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["case", "backend", "trial", "seconds"])
        for case, julia, python in cases:
            for backend, times in (("Julia Metal", julia), ("Python MPS", python)):
                writer.writerows((case, backend, i, value)
                                 for i, value in enumerate(times, start=1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    for ax, (case, julia, python), limits in zip(axes, cases,
                                                 ((0.04, 0.075), (0.19, 0.28))):
        for y, (times, color) in enumerate(((julia, "#d97706"), (python, "#be185d"))):
            q1, median, q3 = np.quantile(times, [0.25, 0.5, 0.75])
            ax.hlines(y, q1, q3, linewidth=16, color=color, alpha=0.32)
            ax.plot(median, y, "o", color=color, markersize=10)
            ax.text(median + (limits[1]-limits[0])*0.025, y,
                    f"{median:.4f} s", color=color, fontsize=9, va="center")
        ax.set_xlim(*limits)
        ax.set_ylim(1.35, -0.35)
        ax.set_yticks([0, 1], ["Julia Metal\n8 Julia host threads",
                                "Python MPS\n12 PyTorch host threads"])
        ax.set_title(case)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Forward-map time for a 640 × 480 × 300-frame tensor", fontsize=14)
    fig.supxlabel("Seconds per whole-tensor call · dots are medians; bands show the middle half")
    fig.savefig(output / "video-2026-11-18.png", dpi=180)
    plt.close(fig)
    for case, julia, python in cases:
        print(f"{case}: Julia {np.median(julia):.6f} s; Python {np.median(python):.6f} s")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
