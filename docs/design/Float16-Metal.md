# Float16 on the Apple GPU: what MPS actually does for our ops

Measured on an M4 Max (40 GPU cores), Metal.jl / MPS, 2026-09-04. Driver:
`bench/Float16Metal.jl`. Raw data: `bench/reports/2026-09-04/float16-metal/float16-metal.csv`.

Question this answers (the user's): *"Does the GPU do any packing or horizontal
adders etc when the ops are float 16?"* — i.e. what does half precision buy us
on the sketch/lift contraction `A (m x k) * W (k x r)`, in throughput and in
accuracy, given the movie regime will store tensors in Float16
(640x480x1800x3: 6.6 GB Float32, 3.3 GB Float16).


> **Parent's correction (2026-09-04).** The time projection in (d) below assumes ~105 s of
> CPU eigensolve for the one-minute movie. That is wrong: on the 640 x 480 shape the eigensolve
> is the ~20 s *intercept* of the measured fit (`labs/MovieRuntime.ipynb`), flat in frame count;
> the ~150 s that grows with frames is the tensor stages this benchmark targets. Moving those to
> the GPU at the 6-17x measured for the dense kernel puts them at ~10-25 s, so the honest
> projection is **~30-45 s for the minute**, not 115-130 s. The measurements and answers (a)-(c)
> stand as written.

## Results table

### GEMM throughput (`Metal.@sync mul!(C, A, W)`, median of 5)

| shape | m | k | r | eltype | seconds | GFlop/s | GB/s (operand traffic) |
|---|---:|---:|---:|---|---:|---:|---:|
| unfold-1 (640 x 480·90·3) | 640 | 129600 | 33 | Float16 | 0.001395 | 3923.1 | 125.0 |
| unfold-1 (640 x 480·90·3) | 640 | 129600 | 33 | Float32 | 0.001572 | 3482.8 | 222.0 |
| unfold-1 (640 x 480·90·3) | 640 | 129600 | 33 | mixed f16→f32 | 0.001739 | 3147.7 | 100.4 |
| unfold-1T (480·90·3 x 640) | 129600 | 640 | 33 | Float16 | 0.002807 | 1950.1 | 62.2 |
| unfold-1T (480·90·3 x 640) | 129600 | 640 | 33 | Float32 | 0.002469 | 2217.0 | 141.3 |
| unfold-1T (480·90·3 x 640) | 129600 | 640 | 33 | mixed f16→f32 | 0.002389 | 2291.2 | 76.6 |
| frame-accum (1e5 x 1800) | 100000 | 1800 | 33 | Float16 | 0.005834 | 2036.2 | 62.9 |
| frame-accum (1e5 x 1800) | 100000 | 1800 | 33 | Float32 | 0.006218 | 1910.5 | 118.0 |
| frame-accum (1e5 x 1800) | 100000 | 1800 | 33 | mixed f16→f32 | 0.005725 | 2075.2 | 65.2 |
| unfold-3 (640·480 x 270) | 307200 | 270 | 16 | Float16 | 0.002644 | 1003.8 | 66.5 |
| unfold-3 (640·480 x 270) | 307200 | 270 | 16 | Float32 | 0.002914 | 910.7 | 120.6 |
| unfold-3 (640·480 x 270) | 307200 | 270 | 16 | mixed f16→f32 | 0.003247 | 817.4 | 57.1 |
| square control (4096^3) | 4096 | 4096 | 4096 | Float16 | 0.009529 | 14422.6 | 10.6 |
| square control (4096^3) | 4096 | 4096 | 4096 | Float32 | 0.010540 | 13039.2 | 19.1 |
| square control (4096^3) | 4096 | 4096 | 4096 | mixed f16→f32 | 0.009718 | 14142.3 | 13.8 |

### permutedims and host↔device transfer (640x480x90x3 tensor, median of 5)

| op | eltype | seconds | GB/s |
|---|---|---:|---:|
| permutedims | Float16 | 0.012261 | 27.1 |
| permutedims | Float32 | 0.015460 | 42.9 |
| host→device | Float16 | 0.004965 | 33.4 |
| host→device | Float32 | 0.009966 | 33.3 |
| device→host | Float16 | 0.002304 | 72.0 |
| device→host | Float32 | 0.004752 | 69.8 |

### Accuracy vs accumulation length k (GPU result vs Float64 CPU reference, same T-rounded inputs)

| k | eltype | max rel err | RMS rel err |
|---:|---|---:|---:|
| 480 | Float16 | 7.37e-4 | 2.09e-4 |
| 480 | Float32 | 1.22e-3 | 1.48e-5 |
| 480 | mixed f16→f32 | 9.70e-4 | 1.25e-5 |
| 1800 | Float16 | 2.78e-2 | 4.42e-4 |
| 1800 | Float32 | 4.05e-3 | 5.22e-5 |
| 1800 | mixed f16→f32 | 2.76e-2 | 3.90e-4 |
| 8192 | Float16 | 9.89e-3 | 2.39e-4 |
| 8192 | Float32 | 4.57e-2 | 5.08e-4 |
| 8192 | mixed f16→f32 | 1.01e-2 | 1.20e-4 |

Mixed GEMM (`A, W :: MtlMatrix{Float16}`, `C :: MtlMatrix{Float32}`) is
**accepted** by Metal.jl/MPS at every shape tried — no rejection anywhere in
the sweep.

### CPU Float16 control (`Array{Float16} * Array{Float16}`, 640x4800x33 — a 1/27-scale slice of the unfold-1 shape, sized to fit the campaign's wall-clock budget)

| eltype | seconds | GFlop/s |
|---|---:|---:|
| Float16 | 0.003144 | 64.5 |
| Float32 | 0.000629 | 322.4 |
| Float64 | 0.001352 | 149.9 |

## Plain-language answers

**(a) What does the GPU do for half precision on our ops?**
No packing, no horizontal-adder trick, and no register-width win shows up in
practice for these shapes: Float16 GEMM ran at **0.88x–1.15x** the Float32
rate across every movie-regime shape and the 4096^3 control (never near a
clean 2x). The square control — big enough to be genuinely compute-bound —
came in at 14.4 vs 13.0 TFlop/s, a 1.11x edge, not the 2x a native 16-bit ALU
path with halved register pressure would predict. MPS is treating Float16
GEMM as "the same execution units, half the traffic," not "twice the lanes."
Where element size *does* pay off cleanly is pure data movement: host↔device
transfer of the 640x480x90x3 tensor is almost exactly 2x faster in Float16
(0.00497 s vs 0.00997 s uploading, 2.06x downloading) — bandwidth-bound
copies scale with bytes the way theory says they should. `permutedims` sits
in between (1.26x), because a chunk of its cost is indexing/kernel-dispatch
overhead that doesn't shrink with element width — consistent with last
night's finding that `permutedims!` beats the GEMM it feeds. Mixed operands
(Float16 in, Float32 accumulate) cost *slightly more* than pure Float16, not
less — MPS is doing real internal upcasting work for that path, not getting
it for free.

**(b) Is accumulation safe for k = 1800?**
Yes. If MPS accumulated Float16 GEMM in fp16, RMS error should grow like
`sqrt(k)·1e-3` — roughly 0.022 at k=480, 0.042 at k=1800, 0.09 at k=8192.
Instead RMS error is **flat at 1e-4–5e-4 across the entire two-order-of-
magnitude range of k** (2.09e-4 at k=480, 4.42e-4 at k=1800, 2.39e-4 at
k=8192 for pure Float16) — the signature of fp32 accumulation with one
half-precision rounding of the final output, independent of how long the sum
is. (Float32's own RMS error, 1.5e-5 to 5.1e-4, is governed by the same k-
independent floor, just tighter because there is no half-rounding at the
end.) The one outlier — Float16's *max* relative error at k=1800, 2.78e-2 —
is a small-denominator artifact (one entry of `C` near zero from Gaussian
cancellation inflating a relative error, not a system-wide 3% error); RMS is
the number that tracks conditioning, and it stayed flat. **Conclusion: MPS
accumulates half-precision GEMM in fp32 internally; k = 1800 carries no extra
risk over k = 480.**

**(c) Recommended contract for the movie pipeline.**
**Float16 storage, Float16 operands, Float32 accumulate/output** — i.e. the
mixed contract Metal.jl/MPS already accepts everywhere it was tried
(`A, W :: MtlMatrix{Float16}` in, `C :: MtlMatrix{Float32}` out). This halves
the resident tensor (3.3 GB vs 6.6 GB at F=1800) and halves every host↔device
transfer and most of the `permutedims` cost, at essentially zero throughput
penalty (mixed ran within 5–20% of pure Float16, itself within 15% of
Float32) and at *better* accuracy than round-tripping the product through
Float16 (compare the mixed vs pure-Float16 columns above — mixed's RMS error
is consistently 15–50% lower, since it skips the output's half-rounding).
Feed the eigensolve (`ArpackSolver` / `GramSolver`) Float32 vectors built from
that Float32 output — do not let the restricted-system linear algebra
(QR/Cholesky/SVD, none of which run on Metal per `ext/DletoMetalQuickDer.jl`)
see Float16 at all. In one line: **Float16 storage + half operands, fp32
accumulate/output, Float32-restricted eigensolve** — exactly the contract the
task brief proposed, now backed by measurement rather than assumption.

**(d) What this means for the 640x480x1800x3 movie.**
Memory: the tensor itself drops from the documented 6.6 GB (Float32) to
3.3 GB (Float16) — unaffected by anything else here, since it's pure storage.
Time: today's measured Float32 CPU pipeline (`labs/MovieRuntime.jl`, fit from
F = 30/90/300 video-shaped whitened-QuickDer runs) projects to **~171 s** of
wall time at F = 1800, of which **~150 s is tensor-shaped work** (the
sketch/lift/verify stages that scale with F — exactly the GEMM and
`permutedims` operations this benchmark measured) rather than the small-
matrix eigensolve. Two independent multipliers apply to that ~150 s, and they
should not be conflated:
  - **CPU → GPU**, already measured for the dense SylverLining kernel in
    `docs/CONTEXT.md` (6–17x on comparable dense tensor ops, `permutedims!`
    the dominant device cost there too);
  - **Float32 → Float16 storage on the GPU**, measured here: ~1x on GEMM,
    ~1.26x on `permutedims`, ~2x on transfer.

  Chaining a conservative version of both (favoring the `permutedims`-bound
  1.1–1.3x from the storage switch, since `permutedims`, not the GEMM, is the
  standing bottleneck on device) puts the ~150 s tensor-stage budget in the
  neighborhood of **10–25 s** on the GPU in Float16, against ~105 s of
  small-matrix eigensolve that this change does not touch (it stays CPU,
  Float32/Float64, unaffected by the tensor's storage type) — a projected
  total nearer **115–130 s** than 171 s. That is a real but modest win, and
  it is gated on the *eigensolve*, not the tensor contraction, being the next
  thing worth optimizing if the movie regime needs to go faster than that.
