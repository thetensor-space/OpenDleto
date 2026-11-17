# Contraction backend options for `sylvesterLM` (dense TTM + sparse TTM)

Date: 2026-09-03. Web-research + doc reading only — no Julia was run, nothing
benchmarked on this machine. "Verified" = stated in the package's own
README/Project.toml/docs. "Inferred" = extrapolated, flagged as such.

Scope: replace ITensor `*` in `src/SylverLining/SylverLining.jl`
(`ester`/`sylve`/`sylvester` closures, ~500 applies/eigensolver run) for
(A) sparse Γ (support ~d^3/6 of d^n) and (B) dense Γ (video, up to
1000×1000×100×3, Float32/64). Budget: Apple silicon, ~4 cores, 10 GB.

---

## 1. Dense (B)

**Baseline (verified):** NDTensors' dense contraction is already
permute-then-`gemm!`, permutations via Strided.jl
([NDTensors GEMM design](https://www.researchgate.net/publication/304758071_Design_of_a_High-Performance_GEMM-like_Tensor-Tensor_Multiplication),
[ITensors.jl](https://github.com/ITensor/ITensors.jl)). So the 5–10 MB/apply
cost is ITensor's `*` allocating fresh objects per call, not a slow GEMM.
There's discussion of contraction-overhead fixes
([NDTensors.jl #32](https://github.com/ITensor/NDTensors.jl/issues/32)) but
**no verified public, in-place, zero-alloc `contract!`/TTM entry point** in
ITensors 0.9.x — don't assume one exists without checking installed
internals.

**1. TensorOperations.jl (primary recommendation).** v5.8.0,
`julia = "1.10"`, no upper bound
([Project.toml](https://github.com/QuantumKitHub/TensorOperations.jl)).
Dense backend is Strided.jl permute + BLAS gemm, same strategy as NDTensors
but public
([backends docs](https://jutho.github.io/TensorOperations.jl/dev/man/backends/)).
`@tensor` supports **preallocated output**: `@tensor out[a,b,c] = ...` where
`out` already exists — the standard zero-alloc idiom. TBLIS backend
available via `TensorOperationsTBLIS.jl` if permute+gemm underperforms for
awkward strides. Using this means operating on plain `Array`s, not
`ITensor`s — a real architecture change (re-derive the `Ωframe`/`engaged`
index bookkeeping ITensor currently does for you), not a drop-in.

```julia
using TensorOperations
Γarr = Array(Γ); Σ = similar(Γarr); tmp = similar(Γarr)   # scratch, reused
function ester!(Σ, Γarr, Xs, tmp)
    fill!(Σ, 0)
    for a in eachindex(Xs)
        @tensor tmp[...] = Xs[a][i,j] * Γarr[..., j, ...]   # mode-a TTM
        Σ .+= tmp
    end
end
```
Build `LinearMaps.LinearMap` with `ismutating=true` so the eigensolver calls
`f!(y,x)` instead of allocating each apply. **Gain:** removes per-apply heap
allocation/GC pressure; TTM FLOP rate is already BLAS-speed either way.
**Risk:** reindexing bugs — budget real test time.

**2. AppleAccelerate.jl BLAS forwarding** (stack on top of #1, cheap to try).
Requires macOS 13.4+, Julia 1.10+. README claims **"6–13× faster
single-threaded GEMM than OpenBLAS on Apple Silicon"** via AMX/SME hardware
only Accelerate can reach
([repo](https://github.com/JuliaLinearAlgebra/AppleAccelerate.jl)). No
independent third-party benchmark found — **this is a vendor/maintainer
claim, not independently reproduced**. Switch is a global
`BLAS.lbt_forward`, no call-site changes. Check threading interaction with
the 4-core budget empirically.

**3. Hand-rolled `permutedims!` + `mul!`** — fallback if #1 has friction.
Mode-n unfold (`reshape(permutedims(Γarr,...), d, :)`), `mul!(out, X_a,
Γ_(a))` is one `gemm!` call, zero allocation with preallocated buffers. More
code, zero new dependencies — worth it given this repo's "fewer
abstractions" bias from prior style-collision history.

**Ruled out:**
- **OMEinsum.jl** (v0.9.4, `julia="1.10"`, actively maintained —
  [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.09886) Mar 2026):
  its value is auto contraction-*order* search across many-tensor networks;
  irrelevant to one fixed small contraction. Useful only as an independent
  correctness cross-check.
- **Tullio.jl/LoopVectorization.jl:** LoopVectorization "currently has
  deprecation warnings on Julia's 1.12 branch, causing test failures," now
  maintained via SciML Small Grants (reduced velocity)
  ([Discourse](https://discourse.julialang.org/t/why-is-loopvectorization-deprecated/109547)).
  Avoid given the 1.12 target.
- **TensorKit.jl:** adds symmetry-sector (`TensorMap`) machinery on top of
  TensorOperations; Γ has no group symmetry, so this is pure overhead
  ([README](https://github.com/QuantumKitHub/TensorKit.jl/blob/main/README.md)).

**Ranking:** TensorOperations.jl (§1) > AppleAccelerate stacked on top (§2)
> hand-rolled fallback (§3) > OMEinsum for validation only > avoid
Tullio/LoopVectorization > TensorKit not applicable.

---

## 2. GPU (Metal)

Verified: Metal.jl is active (v1.10.3, `julia="1.10"`,
[JuliaGPU/Metal.jl](https://github.com/JuliaGPU/Metal.jl)). **Float64 is not
supported on Apple GPU hardware at all** (Metal Shading Language limitation,
Float32 only)
([ITensors RunningOnGPUs.md](https://github.com/ITensor/ITensors.jl/blob/main/docs/src/RunningOnGPUs.md)) —
rules Metal out wherever Float64 accuracy is needed. ITensors/NDTensors
**does** support Metal via a package extension (`mtl()`, analogous to CUDA's
`cu()`), dense contractions run on GPU; but matrix factorizations
(QR/SVD/eigen) still run on CPU with transfers back and forth — "if your
algorithm's cost is dominated by those operations you won't see any
speedup." Block-sparse paths are called out as experimental/CPU-bound (a
different sparsity than regime A's, but a signal of general GPU immaturity
outside dense ITensor). Found but did not read a Jan 2026 ScienceDirect paper
benchmarking Apple-M GEMM
([link](https://www.sciencedirect.com/science/article/pii/S0167739X26000270))
— a lead, not a source I can quote numbers from.

**Recommendation:** defer. Sequence after CPU zero-alloc TTM (§1) is done and
measured at d≈500–1000; only reach for Metal if that doesn't meet budget,
and only for the Float32 dense-video path.

```julia
using Metal, ITensors
Γ_gpu = mtl(Γ)                          # Float32 only
Σ_gpu = Γ_gpu * mtl(Matrix{Float32}(X_a))
```

---

## 3. Sparse (A)

**1. Mode-n unfold to `SparseMatrixCSC` + stdlib sparse-dense `mul!`
(primary recommendation).** Store Γ's COO support once, build per-axis
`d × d^{n-1}` `SparseMatrixCSC` unfoldings, compute `X_a' * Γ_(a)` with
Base's sparse-dense `mul!`. Zero new dependencies, zero Julia-version risk,
proven `O(nnz·d)` complexity (textbook Kolda–Bader mode-n unfolding).
Unfoldings built once per chisel (not per apply), so n=5 axes × nnz triples
is still far below `d^n`. "Boring but safe" — matches this repo's stated
preference for balance over cleverness.

**2. Finch.jl `@einsum` (parallel validation spike, likely better long
term).** v1.4.0, `julia="1.10"`
([finch-tensor/Finch.jl](https://github.com/finch-tensor/Finch.jl)). Compiles
sparsity-aware code with algebraic simplification (`x*0=>0`), so a
sparse×dense contraction skips explicit zeros — exactly regime A's
requirement, and one code path for n=3,4,5 instead of n hand-rolled
unfoldings. **Not verified:** no explicit TTM example found in the README, no
Julia-1.12-specific CI evidence, no independent perf numbers — needs a spike
before committing.

```julia
using Finch
Γf  = Tensor(Sparse(Sparse(Sparse(Dense(Element(0.0))))), Γarr)
out = Tensor(Dense(Dense(Dense(Dense(Element(0.0))))))
@einsum out[a2,a3,a4,i] += Γf[a2,a3,a4,j] * X_a[i,j]
```

**Ruled out:**
- **SparseArrayKit.jl** (v0.4.3, `julia="1.8"`, DOK/`Dict{CartesianIndex}`
  storage, integrates with `@tensor`
  [repo](https://github.com/QuantumKitHub/SparseArrayKit.jl)): good for
  scattered point-access sparsity, not bulk mode-n contraction throughput.
  Keep in back pocket, not primary.
- **TensorToolbox.jl:** has exactly the wanted `ttm(X, A, mode)` API
  ([repo](https://github.com/lanaperisa/TensorToolbox.jl)) but **"will not
  be updated anymore" as of 2023** — no Julia 1.10+/1.12 compat claim.
  Reference for API/semantics only, do not depend on it.
- **ITensors general sparse storage:** NDTensors has `Dense`/`Diag`/
  `BlockSparse`/`DiagBlockSparse` only; `BlockSparse` is QN-block (dense
  blocks placed sparsely), not element-wise COO — won't represent a
  simplex-shaped support (`i+j+k+l=d-1`) efficiently. A roadmap issue
  discusses moving to `BlockSparseArray`
  ([NDTensors #1250](https://github.com/ITensor/ITensors.jl/issues/1250))
  but **no general non-QN sparse storage exists in ITensors today** as far
  as found. Confirms regime A needs a path outside ITensor.

**Ranking:** mode-n unfold + stdlib sparse `mul!` (now) > Finch.jl `@einsum`
(spike, likely long-term answer) > SparseArrayKit (wrong shape) >
TensorToolbox (unmaintained, reference only).

---

## 4. Sylvester/derivation-specific packages

Searched for existing packages computing tensor derivations, stabilizer Lie
algebras, or this Sylvester-type equation. Found nothing on-target: Leibniz.jl
is differential-geometry (Grassmann manifold exterior calculus), not this
sense of "derivation"; TensorKit/GenericTensorNetworks solve unrelated
problems (symmetry-sector contraction; combinatorial optimization via tensor
networks). **No existing package covers the derivation math itself** — only
the TTM/contraction kernel underneath is reusable; the Sylvester/null-space
logic is this codebase's own contribution and should stay hand-written.

---

## 5. Compatibility table

| Package | Version | Julia lower bound | Julia 1.12 | Status |
|---|---|---|---|---|
| TensorOperations.jl | 5.8.0 | 1.10 | not run; no upper bound in Project.toml | Active |
| OMEinsum.jl | 0.9.4 | 1.10 | not checked | Active (paper pub. Mar 2026) |
| Tullio.jl | not checked | — | LoopVectorization has 1.12 deprecation/test failures | Reduced (SciML Small Grants) |
| TensorKit.jl | not checked | — | not checked | Active |
| ITensors.jl/NDTensors.jl | 0.9.x (0.9.15 in repo) | not checked | not confirmed | Active (Flatiron-backed) |
| AppleAccelerate.jl | not pinned | 1.10 | inferred fine | Active; macOS 13.4+ |
| Metal.jl | 1.10.3 | 1.10 | not confirmed | Active |
| Finch.jl | 1.4.0 | 1.10 | not confirmed | Active |
| SparseArrayKit.jl | 0.4.3 | 1.8 | not confirmed | Active |
| TensorToolbox.jl | ~2023 | not checked | not applicable | **Unmaintained since 2023** |

"Not confirmed" = lower bound stated, no explicit upper-bound/CI evidence for
1.12 found — verify with an actual `Pkg.add`/precompile before relying on it.

---

## 6. Bottom line

- **Dense (B):** rewrite `ester`/`sylve` over plain arrays with
  TensorOperations.jl `@tensor` + preallocated output + `ismutating=true`
  LinearMaps to kill per-apply allocation; add AppleAccelerate.jl BLAS
  forwarding on top (cheap to try, vendor-claimed 6–13×). Defer Metal/GPU
  until CPU zero-alloc is measured against budget at d≈500–1000, and only
  for Float32-tolerant paths.
- **Sparse (A):** hand-rolled mode-n unfold to `SparseMatrixCSC` + stdlib
  sparse-dense `mul!` now (no new deps, proven complexity); run a Finch.jl
  `@einsum` spike in parallel as the likely better single-code-path answer
  across n=3,4,5.
- No package reuse exists for the derivation/Sylvester math itself, only for
  the contraction kernel beneath it.
