# The movie regime on Metal: what the tensor stages actually cost

Driver: `bench/GpuMovie.jl`. Raw data: `gpu-movie.csv`. M4 Max (16 CPU / 40 GPU
cores, 64 GB unified), 5 Julia + 5 OpenBLAS threads, 2026-09-04. Every run is
`640 x 480 x F x 3` random, `UniversalOp` / `UniversalChisel(4)`, whitened
QuickDer, **matrix-free restricted branch on both devices** (both dense budgets
zeroed so the two columns compare the same computation), nullity 3 of 3 in
every cell.

Every row is a **warm** run — see "The cold-start tax" below, which is the
single most important number on this page.

---

## 1. The finding that reorders the work

**The frame-linear cost of a movie run is not in the tensor stages.** It was
expected to be: the affine fit over F = 30/90/300 in `docs/CONTEXT.md`
projected ~150 s of the one-minute run into "sketch, lift, verify", against
~20 s of eigensolve. Read per stage instead of by fitting the total, the split
is the other way round.

| F | wall | solve (host eigensolve) | sketch + lift + verify | tensor share |
|---:|---:|---:|---:|---:|
| 30 | 34.79 s | 34.16 s | **0.369 s** | 1.1 % |
| 90 | 21.87 s | 20.40 s | **0.670 s** | 3.1 % |
| 300 | 51.75 s | 44.84 s | **2.459 s** | 4.8 % |

(Float32, CPU.) The tensor stages *are* linear in F — 0.0077 s per frame — but
that slope carries only ~9 % of the slope of the total. The rest of the
observed growth is the restricted eigensolve, which is not flat in F either:
the restriction size `r_3` grows with the frame count (15 → 15 → 16), so the
restricted system grows with it (33480x31819 → 40800x38569), and Arpack /
KrylovKit iteration counts do the rest. The eigensolve is also the noisy term —
17 s and 54 s were seen for the *same* input in the same process — which is why
a fit to wall time attributed its variance to the frame count.

**Consequence: a perfect GPU port of every tensor stage removes at most 5 % of
a movie run.** What is left over is the eigensolve.

---

## 2. Before and after, per stage

`before` is the device path as it stood (`permutedims` of the full tensor on
any middle axis); `after` is the cost-modelled slab route plus
`_qdn_mode_order`. CPU is unchanged by construction and by measurement (the
matrix-free sphere case's apply counts are bit-identical, 11080 / 5528).

### F = 300, Float32 — the largest measured case

| stage | CPU | GPU before | GPU after | after vs CPU |
|---|---:|---:|---:|---:|
| sketch | 0.933 s | 0.084 s | **0.046 s** | 20.1x |
| lift | 1.054 s | 0.899 s | **0.876 s** | 1.20x |
| verify | 0.472 s | 2.535 s * | **0.035 s** | 13.5x |
| **tensor total** | **2.459 s** | 3.52 s * | **0.958 s** | **2.57x** |
| solve (host) | 44.84 s | 30.88 s | 29.86 s | — |
| wall | 51.75 s | — | 34.79 s | — |
| peak host RSS | 5.32 GB | — | 5.46 GB | — |
| peak device alloc | — | — | 2.06 GB | — |

\* the `before` column was measured with only a `10x10x10x3` warm-up, so its
`verify` carries the cold-start tax of §3 rather than a permute cost. The
honest reading of the two columns together is in §4.

### The F sweep (warm, Float32)

| F | dev | wall | solve | sketch | lift | verify | tensor | RSS | device peak |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | cpu | 34.79 | 34.16 | 0.022 | 0.299 | 0.048 | 0.369 | 1.93 GB | — |
| 30 | gpu | 23.19 | 22.66 | 0.011 | 0.238 | 0.016 | **0.265** | 2.19 GB | 0.31 GB |
| 90 | cpu | 21.87 | 20.40 | 0.044 | 0.485 | 0.141 | 0.670 | 2.65 GB | — |
| 90 | gpu | 18.72 | 17.44 | 0.036 | 0.390 | 0.017 | **0.442** | 2.86 GB | 0.47 GB |
| 300 | cpu | 51.75 | 44.84 | 0.933 | 1.054 | 0.472 | 2.459 | 5.32 GB | — |
| 300 | gpu | 34.79 | 29.86 | 0.046 | 0.876 | 0.035 | **0.958** | 5.46 GB | 2.06 GB |

Tensor-stage speedup: 1.39x at F = 30, 1.52x at F = 90, **2.57x at F = 300** —
rising with F, as it must, because the fixed per-stage dispatch cost is being
amortized over more data.

**The lift is now the wall.** At F = 300 it is 0.876 s of the 0.958 s device
tensor budget and it is only 1.20x faster than the CPU, because what it spends
its time on is not the pair tensors — those are device mode products like any
other — but the HOST least-squares that follows them: `qr(A) \ B` with
`A` of `m·R_a x d_a` and `B` of `m·R_a x k·h_a`, assembled in a `k x m` Julia
loop. Neither Metal.jl nor MPS has `qr`, so that stays on the host
(`ext/DletoMetalQuickDer.jl` documents why). Sketch and verify, which are pure
mode products, run 13-20x.

### Float16 (F = 300)

| T | dev | wall | tensor | peak RSS | device peak |
|---|---|---:|---:|---:|---:|
| Float32 | cpu | 51.75 | 2.459 | 5.32 GB | — |
| Float16 | cpu | 56.40 | 1.523 | 5.44 GB | — |
| Float32 | gpu | 34.79 | 0.958 | 5.46 GB | 2.06 GB |
| Float16 | gpu | 56.77 | 0.958 | 5.40 GB | 2.06 GB |

**Float16 buys nothing today, in time or in memory, and the reason is a policy
and not a bug.** `compute_eltype(Float16) = Float32` (`src/solvers/Precision.jl`)
promotes the tensor to Float32 in `derTrOpsReduced` before anything else
happens — there is no CPU BLAS or LAPACK half path — so the host copy, the
upload, the device copy and every GEMM are Float32 whatever the caller stored.
The Float16 and Float32 device rows are identical to three digits, and the peak
device allocation is the same 2.06 GB, because they are running the same
arithmetic on the same bytes. The measured contract of
`docs/design/Float16-Metal.md` (half storage, half operands, fp32 accumulate)
is not reachable without the plumbing described in §5.

---

## 3. The cold-start tax, which dominates everything above

A device run pays Metal pipeline specialization the first time each kernel
meets a new shape class, and on the movie tensor that cost is **seconds**:

| | first movie-shaped GPU run | second, same process |
|---|---:|---:|
| sketch | 0.626 s | 0.034 s |
| verify | 3.257 s | 0.018 s |
| lift | 0.407 s | 0.397 s |

A `10x10x10x3` warm-up in between does **not** cover it. Across the sweep the
discarded warm-up runs spend 13.7-15.1 s in the tensor stages against 0.27-0.96 s
warm.

**So a single one-shot GPU run of one movie is a net loss**: ~14 s of
specialization to save ~1.5 s of tensor work. The device path pays for itself
only in a process that solves several tensors — a stratification sweep, a
parameter scan, a server — and any benchmark that does not warm up on the shape
it measures is measuring the compiler. This is why `bench/GpuMovie.jl` warms up
on the tensor it then times, and why it exists rather than a flag on
`bench/WhitenedRestriction.jl`.

---

## 4. What changed in the kernel, and what the permute was worth

`_qdn_ttm!` on a middle axis used to permute the whole `d^n` tensor to the
front on any device array, on the reading that "`back` separate kernel launches
would trade one transpose for a thousand". For the shape that matters that
reading is wrong: a movie contracted on its **frame** axis has `back = 3` —
three dispatches against a 35 ms permuted copy.

The decision is now a cost model over shapes (`_qdn_slab_is_cheap`,
`_qdn_mode_cost`, `_qdn_mode_order` in `src/solvers/QuickDerN.jl`) built from
three numbers measured in `docs/design/Float16-Metal.md`: a streaming GEMM at
200 GB/s, a `permutedims` round trip at 10 GB/s, a dispatch at 60 us. Two
consequences:

* `_qdn_ttm!` takes slab GEMMs on contiguous device views (Metal.jl accepts a
  `view(G3, :, :, b)` as a GEMM operand) whenever they beat the copy — which
  includes the movie's column axis, at `3F` dispatches against a 35 ms permute.
* `_qdn_mode_order` picks *which* axis meets the full tensor. Mode products
  commute, exactly one pass in each chain is over `d^n`, and on the movie shape
  the greedy order is `[1, 3, 2, 4]`: the axis-1 cross sketch (which cannot use
  axis 1) meets `d^n` on the **frame** axis, 3 dispatches, rather than on the
  column axis at `3F`. Host arrays keep `1:N` so no CPU result moves.

How much of the measured 13-20x on sketch and verify is this change, as against
the warm-up correction of §3? Honestly: **most of it is the warm-up**. `verify`
never permuted the full tensor in the first place — its one big pass is an
axis-1 (edge) GEMM against `Ms[â][:, sel]` — so the 2.5-3.2 s "before" number
was always cold-start, and its warm value is 0.035 s either way. The sketch is
where the reordering genuinely applies, and probing the device primitives
directly at the F = 90 shape puts the ceiling on what was available:

| device primitive, 640x480x90x3 Float32 | median |
|---|---:|
| `mul!(C(4 x 129600), transpose(M(640 x 4)), G(640 x 129600))` | 1.4 ms |
| `norm(G)`, full tensor | 7.6 ms |
| `permutedims(G, (2,1,3,4))` | 16.8 ms |
| `copy(selectdim(G, 1, sel))` | 1.7 ms |

Every primitive is milliseconds. The whole tensor budget at F = 90 is 0.44 s,
i.e. ~25x the sum of the kernels it runs; what is left is dispatch and host
glue, not arithmetic. **The permute was never the bottleneck the night-board
note took it for.** Removing it is right and it is now measured, but it bought
tens of milliseconds on a stage that was already a few percent of the run.

---

## 5. The one-minute projection, from this page's own fit

Affine fits over F = 30/90/300 of the warm tensor totals:

* CPU: `0.137 + 0.00774·F` s → **14.1 s** at F = 1800
* GPU: `0.188 + 0.00257·F` s → **4.8 s** at F = 1800

The eigensolve does not admit an honest fit from three noisy points (34.2 /
20.4 / 44.8 s on the CPU; the restricted system grows only from 33480x31819 to
40800x38569 across the whole range) but it is 90-97 % of every cell, and at
F = 1800 the restriction sizes grow again. Taking it as 30-60 s:

**one minute of colour video ≈ 35-75 s, of which the GPU port saves ~9 s.**

That is a real win and a small one, and it is not the number to optimize next.
Memory is closer to binding: peak device allocation is 1.87x the tensor
(2.06 GB against 1.1 GB at F = 300), so F = 1800 in Float32 projects to ~12 GB
of device allocation next to a 6.6 GB host copy — over the ~10 GB working
budget. **That, not speed, is what the Float16 contract is for**, and reaching
it needs:

1. a mixed-eltype `_qdn_ttm` (`A, W :: MtlMatrix{Float16}` in,
   `C :: MtlMatrix{Float32}` out — the form MPS accepts at every shape), and
   a half-precision device copy of the axis sketches alongside the Float32 one,
   since only the first pass of each chain meets the half-precision tensor and
   everything after it is already Float32 and small;
2. a decision about the HOST copy, which is where the other 6.6 GB is. Holding
   it in Float16 means `compute_eltype`'s promotion no longer applies to the
   tensor itself — a precision-policy change, not a kernel change.

Neither was attempted here.
