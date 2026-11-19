# Ten-second windows from real video

We used freely downloadable filmed footage from Pexels: a [landscape countryside clip](https://www.pexels.com/video/serene-countryside-landscape-in-autumn-40276646/) and a [portrait nature clip](https://www.pexels.com/video/nature-and-travel-18440087/). The first 10 seconds of each were converted to 30 frames per second. Each 1080p original also supplied a 720p downsample, giving four **300-frame** clips. The benchmark uses grayscale pixel values in `Float32`, with a tensor shape of **width × height × 300**; it excludes download, decode, and resize time. The clips and binary tensors live outside the repository.

![Line chart of seconds for one whole-window derivation-map call versus decoded grayscale tensor size. At 720p each 300-frame tensor is 1.03 GiB: landscape takes 0.255 seconds in Julia Metal and 0.223 in Python MPS; portrait takes 0.195 and 0.216 seconds. At 1080p each tensor is 2.32 GiB: landscape takes 0.633 and 0.578 seconds; portrait takes 0.698 and 0.569 seconds. Both systems use 12 configured host threads.](real-video-map-2026-11-18.png)

| Clip | Tensor size | Julia Metal | Python MPS |
| --- | ---: | ---: | ---: |
| Landscape 720p | `1280 × 720 × 300` · 1.03 GiB | 0.255 s | 0.223 s |
| Landscape 1080p | `1920 × 1080 × 300` · 2.32 GiB | 0.633 s | 0.578 s |
| Portrait 720p | `720 × 1280 × 300` · 1.03 GiB | 0.195 s | 0.216 s |
| Portrait 1080p | `1080 × 1920 × 300` · 2.32 GiB | 0.698 s | 0.569 s |

Each table value is the median of five timed calls after two warmups. Each call applies the whole-window derivation map and transfers its output back to the host, with GPU synchronization. The same decoded tensor and seeded coordinate vector feed both implementations. Julia uses Metal.jl, Python uses PyTorch MPS, and each is configured for **12 host threads**. That count does not measure GPU worker threads. The [raw timing summary](real-video-map-2026-11-18.csv) and scripts are included.

These timings are for **one map application**, an operation inside a derivation solve. A completed derivation basis requires an iterative null-space solve and a residual check. The map timings therefore show neither full derivation latency nor the ability to update a sliding window at 30 fps. One application to a nonoverlapping 10-second window takes less than a second after setup. Recomputing the whole map after every incoming frame would require 30 windows per second; the measured rate is only about **1.4–5.1 windows per second**. An incremental algorithm would be needed for that sliding-window pattern.

## Full derivation solve

The full solve is measured separately below. Its processing tensor is downsampled from the source clip to `80 × 45 × 300` (4.12 MiB), so the native-resolution map numbers above must not be read as full solves. A Julia Metal matrix-free null-space search on the landscape 720p clip did not finish within a 180-second wall limit. A bounded two-vector request returned in 17.7 seconds, but the vectors had forward residual norms of 0.303 and 0.309 and were **not certified**; those are not usable derivations.

QuickDer on that same processed tensor produced candidate derivations in **19.36 seconds** after a separate 36.57-second compilation warmup. The first two candidates' full-tensor relative residuals were `9.29e-5` and `8.79e-5`; the other candidates had only sampled residual checks. The solver returned 683 vectors, but flagged the nullity decision as **unconverged and uncertified**. Most returned directions are trivial consequences of temporal rank 293 on a 300-frame axis; this does not establish 683 distinct, content-specific video symmetries. Full validation of the first two candidates took another 0.55 seconds. Even the warmed 19.36-second solve exceeds the 10-second window arrival time, before decoding or full validation of every candidate.

## Reproduce

The [preparation script](prepare_real_video.py) downloads the Pexels originals or accepts cached copies, verifies their sizes, creates the four 10-second clips, and writes native-resolution tensors and seeded coordinate vectors. Pexels permits free download, use and modification of its videos under its [license](https://www.pexels.com/license/). The benchmark scripts require the project Python environment, `ffmpeg`/`ffprobe`, and a Julia environment containing Dleto and Metal.jl.

```sh
python/.venv/bin/python bench/python_vs_julia/prepare_real_video.py /tmp/real-video --native-tensor
DLETO_METAL_PROJECT=/path/to/metal-env bash bench/python_vs_julia/run_real_maps.sh /tmp/real-video /tmp/real-video-logs
MPLBACKEND=Agg python3 bench/python_vs_julia/plot_real_video.py /tmp/real-video-logs bench/python_vs_julia/real-video-map-2026-11-18.png
```

Machine: Apple M4 Max, 64 GiB unified memory, macOS 27.2.1, Julia 1.12.3, Metal.jl 1.11.1, PyTorch 2.15.0. The native-resolution RGB equivalent would be three times larger than these grayscale tensors.
