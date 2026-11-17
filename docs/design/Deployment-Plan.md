# Deployment plan: what does "ship this" look like?

Decision document, 2026-09-04. The question the owner is asking has moved from
"would native be faster" (answered `docs/design/Native-Core-Plan.md`: no, not
now) to "what does this need to look like for large deployment" — a
video-compression consumer calling the derivation solvers on `H x W x T x 3`
tensors, on servers and possibly end-user machines. This document assumes
today's whitening work lands (per-axis QR so the restricted Gram's diagonal
blocks become identity), which collapses the matrix-free branch's
iteration-count wall into: a few dozen tensor contractions (`permutedims!` +
GEMM), a handful of small QRs/Choleskys (`d x d`), and one least-squares lift.
It does not touch that work's code; it only designs around the shape of loop
it produces.

Sources measured directly for this document: the `juliac` experiment in
§1 (`bench/jl`, this worktree, Julia 1.12.3, ~10 minutes of the ~30-minute
budget). Everything else is read from `docs/design/Native-Core-Plan.md`,
`bench/reports/night-2026-09-03/{BOARD.md,*.csv}`, `bench/reports/precision-study.md`,
`ext/DletoMetal*.jl`, `Project.toml`, and the web sources cited inline.

## 1. Shipping Julia itself

Three ways to hand someone a working `stratify` without them installing Julia
and this project's ~130-package dependency tree: `juliac` (executable or
shared library, Julia 1.12's experimental static compiler), PackageCompiler.jl
(`create_library`/`create_app`, a bundled sysimage), or "just tell them to
clone and `Pkg.instantiate`" (today's actual answer, and the one that never
breaks).

### 1a. The `juliac` experiment (ran, not estimated)

Julia 1.12.3 ships `juliac.jl` at
`$(julia --startup-file=no -e 'print(Sys.BINDIR)')/../share/julia/juliac/juliac.jl`
in this install (confirmed by `find`, not inferred). Three builds, all through
`bench/jl` so as not to violate the shared-machine budget:

| build | flags | result | size | startup |
|---|---|---|---|---|
| exe, no trim | `--output-exe` | **builds** | 188 MB | 1.38 s cold, 0.08-0.09 s warm |
| exe, trim, with `println` | `--experimental --trim=safe` | **fails** | — | — |
| exe, trim, no I/O (just `qr(rand(Float32,100,100))`, return an exit code) | `--experimental --trim=safe` | **builds** | 1.23 MB | 0.03 s (cold and warm looked the same) |
| shared lib, trim, `@ccallable` C function | `--output-lib --compile-ccallable --experimental --trim=safe` | **builds** | 1.23 MB | (not separately timed; same runtime as the exe) |

The trim failure is exact and reproducible: `Verifier error #1: unresolved
call from statement Base.print(Base.stdout::IO, x::Int64, "\n")::Nothing` —
`println` alone, nothing else in the file, breaks `--trim=safe`. This is not
a fluke of this build: it is the open, unfixed upstream bug
[JuliaLang/julia#58458](https://github.com/JuliaLang/julia/issues/58458),
filed against a 1.12 beta and still open, "`--trim` removes essential
standard I/O functions." Removing the `println` and returning a plain `Cint`
made the same file trim cleanly to 1.23 MB. The shared-library variant with
`Base.@ccallable function dleto_qr_trivial(n::Cint)::Cint` also trimmed
cleanly to 1.23 MB, and `nm -gU` confirms the exported C symbol
(`_dleto_qr_trivial`) is really there.

Two more things the experiment surfaced, both from `otool -l` on the
untrimmed exe: it links `@rpath/libjulia.1.12.dylib` and
`@rpath/libjulia-internal.dylib` with `LC_RPATH` entries pointing at the
**absolute path** of this machine's `~/.julia/juliaup/...` install — i.e. the
binary is not relocatable as built. `juliac.jl --help` documents a
`--relative-rpath` flag ("lookup all required libraries in an adjacent
`julia/` folder") for exactly this, untested here to stay inside the compute
budget; using it plus bundling `libjulia*.dylib` next to the binary (or inside
a `.app`'s `Frameworks/`) is the realistic path to a distributable binary, not
the default output.

**Reading this against Dleto specifically.** `Native-Core-Plan.md` already
says "Dleto's registry-of-solvers design is dispatch-heavy, so `--trim` is not
realistic" — the experiment makes that concrete rather than asserted. If
`println` (a call two frames from `Base.print`, about as ordinary as Julia
code gets) cannot survive the verifier, then none of these can, unqualified:
`Dleto.SOLVER_REGISTRY[solver]` (`Dict{Symbol,NullSolver}` dispatch,
`NullSolvers.jl:294-319` per Native-Core-Plan's map), every `ext/*.jl`
`__init__` (all of which call `println` today, see `DletoArpackExt.jl`'s
`"Loading Dleto Arpack Extension"`), ITensor's index-based dynamic dispatch,
`CSV`/`DataFrames`/`Plots` I/O, or `IJulia`. **`--trim` on the whole `stratify`
API is not on the table for Julia 1.12.** What survives is exactly the shape
of the whitened hot loop this document is designed around: a hand-written,
allocation-free numeric kernel with no `println`, no `Dict` dispatch, no
`Symbol`-keyed registry, calling only `LinearAlgebra`/BLAS/LAPACK — i.e. a
`libdleto_core` built from a *purpose-written* trim-safe entry point, not from
`src/` as it stands. That is a genuinely different artifact from "compile
Dleto," closer to the native-kernel boundary Native-Core-Plan already drew
(§1a of that document: TTM, Gram/Cholesky, restricted map, lift — never the
policy/orchestration layer).

### 1b. PackageCompiler.jl (estimated from documentation only, not built)

`create_library` bundles a custom sysimage plus every runtime `.dylib`/`.so`
Julia needs into a directory with C headers — the intended "call Julia from
C" story, and unlike `juliac` it tolerates ordinary dynamic Julia code (no
verifier, no `--trim`). The project's own docs example (a nearly-empty demo
package) produces a **97 MB** `.dylib`
([PackageCompiler.jl `libs.html`](https://julialang.github.io/PackageCompiler.jl/dev/libs.html)).
Dleto's dependency graph is far from empty — `Project.toml` carries `ITensors`,
`NDTensors`, `Plots`/`PlotlyJS`/`PlotlyKaleido`, `DataFrames`, `CSV`, `IJulia`,
`KrylovKit`, `IterativeSolvers` — the `Pkg.instantiate` this session triggered
pulled in roughly 130 packages. A sysimage baking in all of that (PlotlyJS
alone drags in a browser-automation-adjacent JS toolchain) plausibly lands in
the **500 MB - 1.5 GB** range; nobody should confirm this by building it on a
shared 64 GB box mid-experiment, so this is an estimate, not a measurement.
Startup is the sysimage's actual selling point: no JIT for anything it baked
in, so a call into the library after the first (\~1 s, mmap+init) should be
low milliseconds — a PackageCompiler sysimage's whole reason to exist is
turning JIT latency (BOARD.md's "2.38 s first call" finding) into a build-time
cost instead of a runtime one. The size cost is the trade: a `libdleto` this
way ships as one heavy artifact instead of a `git clone` plus first-run
compilation, and every dependency the sysimage does *not* need (a video
consumer needs none of `Plots`/`IJulia`/`CSV`) is dead weight worth trimming
from a **narrower build target**, not from `src/Dleto.jl` wholesale.

### 1c. Recommendation for this axis

Do not attempt `--trim` on the package as it exists. If a compiled artifact is
ever wanted, the realistic order is: (1) today's answer — `git clone` +
`Pkg.instantiate`, PrecompileTools workload for warm-up (already Phase 1 of
Native-Core-Plan, unaffected by any of this); (2) a `create_library` build of
a **deliberately small entry module** (`stratify_video(A::Array{Float32,4})`
and nothing else reachable from it — no `Plots`, no `IJulia`, no CSV) for a
server-side C ABI, several hundred MB, first-call-free after the sysimage
loads; (3) only reach for hand-written `juliac --trim` kernels for the narrow
numeric core described in §2, where the payoff is a 1-2 MB artifact with
30 ms startup and no Julia runtime dependency beyond `libjulia`/`libopenblas`
— genuinely appropriate for an end-user machine or an embedded/mobile target,
at the cost of writing that core to be trim-clean by construction (no
`Symbol` dispatch, no `println`, concrete types end to end).

## 2. If a native core were built anyway

Assume the whitened design lands: the restricted solve becomes contractions
+ small dense factorizations + a bounded number of applies, i.e. every regime
starts to look like the dense branch Native-Core-Plan already measured.
Reusing that probe (2 threads, M4, d = 100, valence 3, `bench/jl`):

| kernel | where | measured floor ratio | fraction of the d=100 wall (2.7 s) |
|---|---|---|---|
| mode product / TTM (contraction, `permutedims!` + GEMM) | `QuickDerN.jl:148-181`, `SylverLining.jl:499-616` | 1.0-1.4x GEMM floor | apply itself is 14.5 ms of a 2.7 s solve (~0.5%), but dominates cost at `d^n` scale once the iteration wall is gone |
| per-axis thin QR (the whitening step, `d x d` or `(N/d) x d`) | new, per the whitening design | not yet measured (today's `qr(A)\B` lift is the closest analogue) | small; a handful of `d x d` QRs, not in the hot loop |
| Gram + Cholesky of the restricted system | `NullSolvers.jl:1097-1132` (`syrk`, `cholesky`) | `syrk` 1.0x floor, Cholesky 1.1x floor | **98%** (2.64 s of 2.7 s) |
| block Krylov / LOBPCG driver over the (now well-conditioned) whitened operator | `NullSolvers.jl` escalation loop | apply itself 0.12 ms; today's wall was iteration count (417 s at d=130) | should collapse to "a few dozen applies," i.e. low milliseconds, once whitening removes the ill-conditioning that drove iteration counts into the thousands |
| least-squares lift (`qr(A)\B`, shared QR per axis) | `QuickDerN.jl:555-637` | LAPACK, not separately timed | small (`restricted matrix build` at this size is 0.02 s) |

Reading: **the Gram+Cholesky pair already sits at the BLAS/LAPACK hardware
floor and stays there whatever language calls it** — this was Native-Core-Plan's
core finding and whitening does not change it, because whitening's job is to
fix the *iteration count*, not the *arithmetic*. A native core cannot beat
these two rows. Where a native core has room is exactly where
Native-Core-Plan already said it did: the contraction/permute path (ceiling
1.4x, from the ~30% of an apply that is bookkeeping) and, if the deployment
target's tensors are sparse (a video consumer's are not — `H x W x T x 3` is
dense), the sparse `nnz` TTM kernel (2-4x, unverified generally). Whitening
changes *how much of the wall those two rows are* — before whitening the
matrix-free branch's applies were invisible next to iteration count; after,
with iteration count bounded, the contraction cost (already measured at 1.0-1.4x
floor) becomes a bigger fraction of a now-much-shorter wall, but its *ceiling*
is unchanged.

**Language/library choice, if Phase 3 of Native-Core-Plan ever triggers (a
stage measured ≥ 2x off its hardware floor that Julia cannot close):** the
same recommendation as before, **Rust with a C ABI**, unchanged by this
deployment framing — if anything reinforced by it, because a video-compression
consumer is exactly the kind of downstream that benefits from `Dleto_jll`
being redistributable and from `faer`'s pure-Rust (no external BLAS process,
one thread pool) posture when the deployment target is a server container
where "two BLAS libraries in one process" (Native-Core-Plan §2's stated trap)
is a real operational risk. Concretely: `faer` for the dense QR/Cholesky if
the native core ever needs its own (it should not — see below), `ndarray` for
the contraction/permute kernel, and `candle`/`wgpu` only if the *GPU* portion
of the native core is ever real (unlikely — see §3, Metal.jl and CUDA.jl
already own that ground in Julia with less code). C++ (Eigen + platform BLAS)
only if a specific vendor library forces it (e.g. calling cuBLAS/cuSOLVER
directly rather than through a Rust binding). Plug-in mechanism: unchanged
from Native-Core-Plan §3 — `Dleto_jll` as a `[weakdeps]` entry, a
`DletoNativeExt` extension that calls `Dleto.register_solver!`/sets a
`backend=:native` hook exactly as `DletoArpackExt.jl`'s `__init__` does today
(`Dleto.register_solver!(:ArpackSolver, ArpackSolver())`), so a plain
`git clone` with no Rust toolchain never breaks — `:array`/`:auto` stay the
default and the only thing that exists when the weakdep is absent.

## 3. GPU portability

**What exists today, Julia-native, and already shipped** (`ef0249a`,
2026-09-04 morning): `ext/DletoMetalExt.jl` (device hooks, `Val`-dispatched so
the extension precompiles under Julia 1.12's "no method overwriting during
precompilation" rule), `ext/DletoMetalSylver.jl` (`sylvesterLM(...;
backend=:metal)`), `ext/DletoMetalQuickDer.jl` (`device=:gpu` for QuickDer's
Gram/subspace steps). The design is instructive for what a portable layer
would need to reproduce: MtlArray already gets `mul!`, `permutedims!`,
`cholesky`, `lu`, broadcast, `kron`, `norm` "for free" through Metal.jl 1.10.3
+ GPUArrays — QuickDer's contraction code (`_qdn_ttm`/`_qdn_unfold`/`_qdn_slice`)
runs on the device **unchanged**, which is why `DletoMetalQuickDer.jl`'s own
header calls its shortness "a finding, not an omission." The two things
Metal.jl does *not* provide are `qr` and `svd` (no MPS wrapper exists;
`LinearAlgebra`'s generic fallback throws a private-buffer storage error
rather than degrading gracefully), so those two steps stay on the host in
every GPU path that exists today. Measured end-to-end wins, restated from
`docs/CONTEXT.md`: `sylvesterLM(backend=:metal)` 6-17x on dense applies
(100³×3: 26→4.6 ms; 400³: 889→53 ms), Float32 Arpack derivations 100³×3
372 s→100 s; QuickDer `device=:gpu` 11.6x on Gram/`M*X`, 2.6x end-to-end at
d=200 because Cholesky/QR/SVD stay on the host. The bottleneck the Metal
kernel comment calls out by name is `permutedims!`, not the GEMM: at
(200,200,100,3) one mode product's permute is 3.11 ms against 1.44+1.24 ms of
`mul!` — GPUArrays' generic permute runs under a tenth of the device's memory
bandwidth.

**CUDA vs a portable layer.** Per the
[JuliaGPU 2026 ecosystem report](https://juliagpu.org/post/2026-08-03-gpu_ecosystem/index.html),
CUDA.jl remains "the most mature part of Julia's GPU stack" (full array
abstraction, kernel compilation, extensive library bindings including
cuSOLVER/cuBLAS), while Metal.jl is explicitly still developing with "known
limitations" — matching what this repo's own Metal extension found (no
`qr`/`svd`). **KernelAbstractions.jl / AcceleratedKernels.jl** are the
portable layer: a single `@kernel` definition compiles to CUDA, ROCm, oneAPI,
Metal, OpenCL, and CPU "without any backend-specific rewrites." For Dleto the
practical implication is narrow and cheap, because the kernel surface that
needs a *hand-written* GPU kernel is already down to one function: the
permute. `mul!`, `cholesky`, `lu` already dispatch to the right vendor BLAS
purely through Julia's array-type multiple dispatch (`MtlArray` → MPS,
`CuArray` → cuBLAS/cuSOLVER) with zero Dleto-side code — that part of
"portability" is already free and stays free on an NVIDIA server the day
someone adds `ext/DletoCUDAExt.jl` mirroring `DletoMetalSylver.jl` almost
line-for-line (the file's own comments note the CPU-facing map design is
already device-agnostic: "every null solver ... works against `:metal`
unchanged and unaware"). The one thing worth writing once, portably, is a
`KernelAbstractions.jl` permute kernel to replace GPUArrays' generic one on
every backend — a single afternoon-scale kernel versus writing (and
maintaining) a Metal-specific and a CUDA-specific fast permute separately.
**Least-effort path to "runs on an NVIDIA server too": write `DletoCUDAExt.jl`
as `DletoMetalSylver.jl`'s CUDA twin (mechanical — swap `MtlArray` for
`CuArray`, `MPSMatrixMultiplication` is just `mul!` either way), and defer a
KernelAbstractions rewrite of the permute until profiling on real NVIDIA
hardware says the generic `CuArray` permute (which, unlike GPUArrays-on-Metal,
may already have a workspace-tuned path in CUDA.jl) is actually the
bottleneck there too** — do not assume it without measuring, since the two
vendors' generic-permute performance is not necessarily the same gap. wgpu/
Vulkan compute (candle, wgpu-rs) stays not recommended: Native-Core-Plan
already found `wgpu`'s `SHADER_F64` is Vulkan-only and unlisted for Metal, and
building a from-scratch compute-shader GEMM/Cholesky stack to match what
MPS/cuBLAS/cuSOLVER already give Julia for free is pure cost with no
identified benefit for this workload.

## 4. Precision at deployment

Coordinating with, not editing, whatever the concurrent precision-responsive
work lands: `bench/reports/precision-study.md` (2026-09-03,
`signed-sealed-delivered/der-and-derivation-law`) already ran the relevant
experiment on the sphere benchmark and its verdict is directly usable for a
video consumer's `Float16`/`Float32` data. Summary of what deployment needs
from that work, not a re-derivation:

- **Float16 is a storage/transport format only, never a compute type here.**
  ITensor contracts Float16 1.5x *slower* than Float64 (software emulation,
  no hardware path in the tested stack) and every eigensolver throws
  (`MethodError`, no Float16 LAPACK); the dense SVD's Float16 noise floor
  (`eps16 * ‖L‖ ~ 0.09`) swamps the null threshold outright (nullity 0). But
  Γ *rounded* to Float16 and then stratified in Float32 or Float64 gives the
  correct nullity and `lsq_err` equal to the rounding error alone (2.0e-4) —
  i.e. **ingest the consumer's Float16 video, upconvert immediately, same
  idiom `ext/DletoMetalExt.jl` already uses for Float64→Float32** (`to_gpu(x
  ::AbstractArray{Float64}) = MtlArray{Float32}(x)`); extending that pattern
  to a Float16→Float32 host-side upconvert on ingest is mechanical.
- **The strongest configuration measured is mixed precision, not pure Float32
  or pure Float16.** `ArpackP64`/`KrylovP64` (Float32 Γ and contractions
  feeding a Float64 eigensolver) is as fast as pure Float32, keeps the
  correct nullity on every seed (pure Float32 alone is also usually correct
  but degrades as the spectral gap shrinks with `d`), and recovers 2-3x
  better (5e-6..1e-5 vs 1e-5..1e-4). This is the same two-stage shape
  `GramSolver`'s GPU path already has (`NullSolvers.jl:207-236`, restated in
  Native-Core-Plan §5 Phase 4: fp32 Gram formation to *capture*, fp64
  Rayleigh-Ritz on the host to *resolve*) — deployment inherits this for free
  if the precision-responsive work routes through the same two-stage
  structure rather than picking one precision end to end.
- **Tolerances must scale with `eps(T)`, not be hardcoded.** The precision
  study's concrete fix (`tol` floored at `100 * eps(T)` for both the null
  threshold and KrylovKit's absolute residual test) is what makes Float32
  viable at all; a deployment default that assumes Float64's `1e-6` will
  either reject valid Float32 answers or, worse, silently accept noise.
- **fp16 tensor-core GEMM is future work, not evaluated here.** Apple's MPS
  and NVIDIA's tensor cores both have real fp16/bf16 GEMM speedups over fp32
  (this is general hardware knowledge, not something measured in this repo);
  whether that matters for Dleto's contraction step is a question for
  whoever owns the precision-responsive work, since it interacts with the
  same "which stage tolerates which precision" analysis `precision-study.md`
  already did for Float32 vs Float64 — extend that table with an fp16-compute
  row rather than assuming tensor cores help a workload whose current
  bottleneck (per §1b/§2) is Cholesky/`syrk`, which fp16 does not accelerate
  usefully once you need Float32-grade accuracy back out of it.
- **What deployment needs, concretely:** an ingest boundary that always
  upconverts to at least Float32 before any contraction or factorization
  (mirroring the existing `to_gpu` idiom), a mixed-precision eigensolver path
  as the shipped default rather than an option (it is strictly better on the
  numbers above), and `tol` expressed as a multiple of `eps(T)` everywhere it
  currently reads as an absolute Float64-flavored constant.

## 5. Recommendation and sequencing

The bar for any native-code step below is unchanged from Native-Core-Plan:
**a stage measured ≥ 2x off its hardware floor that Julia cannot close.**
Nothing measured in this document or its predecessor clears that bar today —
the two most expensive kernels (`syrk`, Cholesky) sit at 1.0-1.1x the floor
already, and the two-stage precision design has no native-vs-Julia decision
built into it at all.

| # | step | effort (days) | what it buys | measure that triggers it |
|---|---|---|---|---|
| 1 | Land the whitening work (in progress today) and re-run `bench/reports/night-2026-09-03`'s frontier scripts (`Frontier.jl`, `quickder-restricted-solvers.csv`'s harness) at d=130..250 v3, d=50..100 v4 | 0 (already scheduled) | confirms the iteration-count wall is actually gone before anything else here is worth doing | applies-per-solve drops from thousands to "a few dozen"; the d=130 cliff (32 s → 417 s) disappears |
| 2 | Purpose-built trim-safe entry point: a `libdleto_core` source file with concrete types end to end, no `Symbol` dispatch, no `println`, exposing exactly the whitened kernels (contraction, per-axis QR, Cholesky, one Krylov step, lift) behind a C ABI; build with `juliac --output-lib --compile-ccallable --trim=safe`, ship as a ~1-2 MB `.dylib`/`.so` | 3-5 | an end-user-machine-friendly artifact with no Julia runtime dependency beyond `libjulia`/BLAS, ~30 ms cold start — the shape §1's experiment showed *is* achievable, just not for the package as it stands | the artifact builds clean under `--trim=safe` (zero verifier errors) and its outputs bit-match `:array`'s on the existing 108-config differential suite |
| 3 | `ext/DletoCUDAExt.jl` as `DletoMetalSylver.jl`'s CUDA twin | 3-5 | "runs on an NVIDIA server too" without touching the orchestration layer at all — the CPU-facing map design already supports this | same nullity/Z-law residual as `:array`/`:metal` on the scrambled-sphere suite; a server-side smoke test on an actual CUDA box |
| 4 | `create_library` build of a narrow server-side entry module (`stratify_video` only — no `Plots`/`IJulia`/CSV reachable) | 2-3 | a heavier (several-hundred-MB) but zero-JIT server artifact, ordinary dynamic Julia allowed | first-call latency measured against the "2.38 s JIT" baseline BOARD.md already flagged; size measured against the 97 MB empty-package baseline cited above |
| 5 | Mixed-precision eigensolver (`ArpackP64`/`KrylovP64`-style, Float32 contraction + Float64 resolve) as `stratify`'s shipped default for Float16/Float32 input, `tol` re-expressed as `k * eps(T)` throughout | 2-4 (depends on today's precision-responsive work's starting point) | correctness margin on the consumer's actual data type without giving up Float32's speed/memory | recovery error and nullity on the precision-study's own sphere benchmark, re-run at whatever `d` the whitened solve now reaches in under a minute |
| 6 | KernelAbstractions.jl rewrite of the permute kernel (only if step 3's CUDA smoke test shows the same permute-dominates-the-apply pattern Metal shows) | 2-3 | one portable fast-permute implementation instead of a Metal-specific and a CUDA-specific one | permute time as a fraction of one apply, measured on both backends; only worth doing if both show it, matching Metal's 3.11 ms permute vs 1.44+1.24 ms GEMM |
| 7 | Native (Rust) kernel for contraction/permute, `Dleto_jll` weakdep, `backend=:native` | held, gated | closes the ~1.4x apply gap if and only if Julia (steps 2, 6) cannot | Native-Core-Plan's own exit criterion: ≥2x measured on the targeted stage at the targeted size within two weeks, or delete the branch |

Steps 1-2 and 1-5 can run in parallel once whitening lands; step 3 has no
dependency on whitening at all and could start today if an NVIDIA box is
available to test on. Step 7 is intentionally last and gated — nothing in
this document's measurements moves it up the list.
