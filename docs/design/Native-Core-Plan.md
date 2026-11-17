# Native core: should the derivation solvers leave Julia?

Decision document, 2026-09-04.  Question from the owner: make the core derivation
solvers as fast and reliable as possible, possibly by porting to Rust or C++, while
keeping distribution a plain `git clone` / download-and-install.  Line numbers are
from the working tree of `aint-no-mountain-high-enough/valence-n-stratify` on the
morning of 2026-09-04.  `SylverLining.jl` and `NullSolvers.jl` were being edited
concurrently (a `backend = :metal` GPU kernel was landing, see §5 Phase 4), so their
numbers may be offset by a few dozen lines; the cited function names are the stable
reference.

## TL;DR

Do not port now.  Measured at the size the roadmap cares about (d = 100, valence 3),
QuickDer spends 2.64 s of a 2.7 s run inside `GramSolver`, and of that 1.87 s is one
BLAS `syrk` whose arithmetic floor at the measured 117 GFLOP/s is 1.9 s, plus 0.57 s of
Cholesky against a 0.53 s floor.  The SylverLining operator apply at the same size is
14.5 ms against a 10.5 ms GEMM floor.  A Rust or C++ kernel calling the same BLAS
cannot beat the BLAS, so the ceiling on a port is the ~30 % of an apply that is
permutes and bookkeeping — and that 30 % is reachable from Julia (Strided/in-place
permutes) for a fraction of the cost.  Where the code is genuinely slow — the
matrix-free restricted solve from d ≈ 130 up (417 s at d = 130 vs 32 s at d = 120),
Float32-only GPU, sparse products — the limit is the *iteration count* of the
eigensolver, the *precision* of the hardware, and *single-threaded stdlib sparse
kernels*, none of which a language change touches.  Recommendation: (1) harden the
Julia (zero-alloc QuickDer kernel, precompile workload, threaded sparse mode product,
opt-in Accelerate BLAS); (2) spend the engineering on the restricted solve's
algorithm, where a 10x is plausible; (3) keep a one-file kernel boundary so that a
native backend can be dropped in behind the existing `backend=` switch **if and only
if** Phase 0's floor report shows a stage ≥ 2x off the hardware floor that Julia
cannot close.  If that day comes, Rust with a C ABI, shipped as a `Dleto_jll` through
Yggdrasil, with the pure-Julia kernel kept as the always-works fallback.  Nothing in
the orchestration — verdicts, escalation, chisel bookkeeping, the ITensor boundary —
should ever move.

## 1. What would move, and what it would buy

### 1a. Kernels versus orchestration

| Layer | Where | Would move? |
|---|---|---|
| Mode-n product / TTM, unfold, permute | `src/solvers/QuickDerN.jl:148-181` (`_qdn_ttm`, `_qdn_unfold`); `src/SylverLining/SylverLining.jl:499-526, 563-616` | candidate kernel |
| Sparse unfold × dense | `SylverLining.jl:508-526, 572-609` (`SparseMatrixCSC`, stdlib `mul!`) | candidate kernel |
| Sketch (cross sketches, pair tensors) | `QuickDerN.jl:334-370` | built from TTM; moves only if TTM does |
| Gram + Cholesky + Rayleigh–Ritz null solve | `src/solvers/NullSolvers.jl:1097-1132` (`syrk` 1109, `cholesky` 1115, `svd(M*X)` 1128) | **no** — already LAPACK/BLAS |
| Lift least squares, consistency filter | `QuickDerN.jl:555-637` (`qr(A)\B`, `nullspace`) | **no** — LAPACK |
| Restricted matrix assembly | `QuickDerN.jl:411-434` | no (0.02 s at d = 100) |
| Matrix-free restricted map | `QuickDerN.jl:446-488` | built from TTM |
| Null-solver policy, verdict, escalation, confirmation | `NullSolvers.jl:95-153, 497-539, 598-773` | **never** |
| Solver registry + extensions | `NullSolvers.jl:294-319`, `ext/*.jl` `__init__` | never |
| Restriction sizes, retry, Z-law verify | `QuickDerN.jl:234-259, 693-743, 805-831` | never (policy) |
| Chisel/operator embeddings, engagement reduction | `SylverLining.jl:257-453`, `src/ops/*` | never |
| AutoDer, ITensor boundary | `src/solvers/AutoDer.jl`, `QuickDerN.jl:749-848` | never |

The kernel candidates are already isolated: the QuickDerN header says every
contraction goes through `_qdn_ttm` so "swapping in TensorOperations, Strided, Finch
or a GPU backend is one function" (`QuickDerN.jl:38-41`), and SylverLining already
has a `backend=:auto/:array/:itensor` switch (`SylverLining.jl:660-666`) with the
ITensor version kept as the oracle.  That is the seam any native code would sit on.

### 1b. Where the time actually goes (measured, 2 threads, M4, OpenBLAS via LBT)

Probe run for this document (random dense 100³, `UniversalOp`, `UniversalChisel(3)`,
`bench/jl`, Julia 1.12.3):

| stage | time | hardware floor | ratio |
|---|---|---|---|
| dense GEMM 2000² (calibration) | 0.137 s | — | 117 GFLOP/s |
| one mode product (10000×100)·(100×100) | 2.06 ms | 2.0 ms | 1.0 |
| `sylvester` apply, d = 100 v3 universal | 14.5 ms | 10.5 ms (6 GEMMs) | 1.4 |
| QuickDer sketch, r = 19 | < 10 ms | — | — |
| restricted matrix build (6859×5700) | 0.02 s | — | — |
| `syrk` of it (Gram 5700²) | 1.87 s | 1.9 s | 1.0 |
| Cholesky 5700² | 0.57 s | 0.53 s | 1.1 |
| `GramSolver` total | 2.64 s | ~2.5 s | 1.05 |
| plain `svd` of the same matrix (old path) | 34.4 s | — | — |
| Z-law verify (`:random`, 4 slices) | 0.16 s | — | — |
| **QuickDer end to end** | **2.7 s** | | |
| matrix-free restricted apply (6859×5700) | 0.12 ms | | |

Reading, by regime:

- **Dense branch, d ≲ 130 at valence 3 (≲ 50 at valence 4):** 97 % BLAS-3.  A port
  gains nothing.  The lever is algorithmic — the Gram is `Σ_a` of Kronecker-structured
  blocks (`Perm_a · kron(I_r, S_a(a)ᵗ)`, `QuickDerN.jl:397-400`), which the current
  code densifies and squares generically.
- **Matrix-free branch, d ≳ 130:** one apply is 0.12 ms at d = 100, yet the LSMR
  route took 3.0 s and Arpack 6.0 s there
  (`bench/reports/night-2026-09-03/quickder-restricted-solvers.csv`) — i.e. tens of
  thousands of applies.  The wall is the eigensolver's iteration count (κ of the
  restricted system, the confirmation re-solve at `NullSolvers.jl:710-729`), not the
  apply.  `bench/stratify-scaling-QuickDer.csv`: 32 s at d = 120, **417 s at d = 130**
  — that cliff is the dense→matrix-free switch, a policy/algorithm problem.
- **SylverLining apply:** 1.4x the GEMM floor.  The extra is two `permutedims!` per
  map direction (`SylverLining.jl:583, 606`) and the embeddings; the sylvester-kernel
  entry of the board already estimated "~30 % of a dense apply that is not GEMM"
  (BOARD.md, sylvester-kernel, NOT DONE).  Ceiling of any rewrite: 1.4x.
- **Sparse:** stdlib `SparseMatrixCSC` × dense `mul!` is, to my knowledge,
  single-threaded [unverified for Julia 1.12]; the array kernel's 2–5x on sphere
  inputs (BOARD, sylvester-kernel table) came from picking the unfolding convention
  SparseArrays has a fast path for (`SylverLining.jl:488-498`).  A threaded or
  native nnz kernel could plausibly buy another 2–4x *on sparse inputs only*.  This is
  the one place a native kernel has a real, measurable target.
- **Allocation:** `_sylvesterLM_array` is zero-alloc (384 B/apply, BOARD).  QuickDerN
  is **not**: `_qdn_ttm` allocates `permutedims` plus a fresh output every call
  (`QuickDerN.jl:161-165`), and `_qdn_pair_tensor` (`334-342`) rebuilds `H_ab` per
  axis pair.  At d = 100 this is invisible (2.7 s total); at d = 1000, valence 3, one
  `d^n` Float64 array is 8 GB, so each allocating permute is a memory event.  Fix in
  Julia (`permutedims!` into scratch, as SylverLining already does).
- **JIT latency:** the board's first-call column shows 2.38 s vs 0.006 s warm on a
  (5,4,6,3) tensor (BOARD, quickder-n table, `*` rows).  For a user stratifying one
  tensor this dwarfs the solve.  A native core would not fix it — the orchestration
  stays in Julia — but a precompile workload does.
- **d = 1000, valence 3, in prospect:** the tensor is 8 GB; the sketch is
  `n·r·d^n ≈ 1.7e11` flops ≈ 2 s at 100 GFLOP/s and three streaming passes over 8 GB
  (bandwidth-bound, ~0.1 s each on M4); the restricted map is 185k × 171k (design note
  §2) at ~5 ms/apply.  If the iteration count stays in the thousands the solve is
  minutes; if κ grows with d it is hours.  Again the count, not the kernel.

**Honest summary for question 1:** a port buys ≤ 1.4x on the dense operator, ~0 on
the dense restricted solve, a plausible 2–4x on sparse inputs, and nothing on the two
walls that actually block d ≥ 130 (iteration count, memory).  The wins are algorithmic
and Julia-side.

## 2. Language options

Baseline: Julia 1.12.3 + OpenBLAS through libblastrampoline (measured
`LBTConfig([ILP64] libopenblas64_.dylib)`), 117 GFLOP/s at 2 threads on M4.

| | (i) Julia, hardened | (ii) C++ core, C ABI | (iii) Rust core, C ABI | (iv) hybrid: Julia orchestration + native kernels |
|---|---|---|---|---|
| Ceiling vs baseline | 1.4x on apply (permutes), 1.0 on Gram; 10x+ available algorithmically | same 1.4x/1.0 — it calls the same BLAS | same | same; plus 2–4x on sparse if hand-written nnz kernel |
| Apple GPU (Float32 only) | Metal.jl; `ext/DletoMetalExt.jl` landed 2026-09-04 (`ef0249a`), zero native code | metal-cpp + MPSMatrixMultiplication (fp32 only) | metal-rs / wgpu; wgpu `SHADER_F64` is Vulkan-only, Metal unlisted | no advantage over Metal.jl for GEMM/TTM-shaped work |
| Dev cost (small AI-assisted team) | low; one repo, one language, tests already exist | high: cmake, three toolchains, ABI, CI matrix, two BLAS in one process | medium-high: cargo is pleasant, but same ABI/CI/BLAS issues; `faer` gives Cholesky/QR without LAPACK linkage | medium: only 3–4 functions cross the ABI, but the whole release pipeline still has to exist |
| Correctness risk | low; `:array` is bit-identical to `:itensor` on 108 configs (BOARD) | **high** — FastDer history: a re-implementation "was wrong in all three blocks and a fabrication heuristic masked it" (docs/CONTEXT.md:124-130) | same as C++; Rust removes memory bugs, not index-convention bugs | contained: kernels are pure array ops with an oracle to diff against |
| Julia static compilation | 1.12 ships `juliac` + `--trim` (experimental; JuliaC.jl requires 1.12+, `--trim` needs `--experimental` on some builds; dynamic dispatch unsupported).  Dleto's registry-of-solvers design is dispatch-heavy, so `--trim` is not realistic; a PackageCompiler sysimage is, but is hundreds of MB and unnecessary for a library.  **PrecompileTools workload is the right tool.** | — | — | — |

A trap worth naming for (ii)–(iv): a native kernel that links its own OpenBLAS puts
**two BLAS libraries with two thread pools** in one process next to Julia's; the
sylvester-kernel scratch is also documented as single-apply-at-a-time
(`SylverLining.jl:226-229`).  Any native kernel must either take BLAS function
pointers from libblastrampoline or do only non-BLAS work (fused permute+TTM, sparse
nnz loops).  That constraint alone removes most of the reason to write one.

**Recommendation: (i) now; (iv) held in reserve behind a measured gate.**  If (iv)
triggers: Rust, because BinaryBuilder supports `compilers=[:c, :rust]` (docs: the
toolchain "automatically selects the appropriate target"; only 32-bit Windows is
excluded), `cargo test` gives the differential harness a home, and the same crate
serves Python later.  C++ only if a specific library (metal-cpp, cuBLAS) forces it.

## 3. Distribution (only relevant if Phase 3 triggers)

The constraint "simple GitHub clone or download-and-install" is satisfied today
because everything is Julia.  Three ways to keep it with a binary:

1. **Yggdrasil → `Dleto_jll`** (recommended).  Submit `build_tarballs.jl` to
   Yggdrasil; it cross-compiles from Linux for macOS arm64/x86_64, Linux
   x86_64/aarch64, Windows x86_64 and registers `Dleto_jll` in General.  Users get the
   binary through `Pkg.add`/`Pkg.develop` of Dleto with no toolchain; Dleto lists
   `Dleto_jll` in `[weakdeps]` with an extension `DletoNativeExt` that registers
   `backend=:native` — exactly how `DletoArpackExt` registers `:ArpackSolver` today
   (`ext/DletoArpackExt.jl:81-85`, `Project.toml` `[weakdeps]/[extensions]`).
   Cost: the jll is a *separate* registered package with its own release cadence.
2. **Artifacts.toml pointing at GitHub Releases.**  Pkg's `Artifacts.toml` supports
   platform-keyed `[[name.download]]` entries with `url` + `sha256` and `lazy = true`,
   so a `git clone` + `Pkg.develop` downloads the right tarball on first use with no
   registry involvement.  A GitHub Actions matrix (`macos-14`, `macos-13`,
   `ubuntu-latest` x86_64 + aarch64, `windows-latest`) builds and uploads; a script
   rewrites the hashes.  You own the matrix; good for pre-registration.
3. **Build from source at install** (`deps/build.jl` running cargo/cmake).  Rejected:
   this machine has `cmake` and `clang++` but **no `cargo`/`rustc`**; users' machines
   are worse.

Whichever path: **the pure-Julia `:array` kernel stays and is the default when the
binary is absent**, so a missing/incompatible binary degrades to today's speed, never
to a failure.  That is the actual guarantee behind "simple clone".

Layout if it happens (one repo, so a clone carries everything):

```
OpenDleto/
  src/ ext/ test/ bench/ docs/          # unchanged
  native/dleto-core/                    # Rust crate, C ABI, no BLAS of its own
    Cargo.toml  src/lib.rs  src/ttm.rs  src/sparse_ttm.rs  include/dleto.h
    tests/differential.rs               # reads fixtures test/fixtures/*.npz-like
  native/build_tarballs.jl              # the Yggdrasil recipe, mirrored here
  .github/workflows/native.yml          # matrix build → Release assets (path 2)
```

C ABI, versioned: `int32_t dleto_abi_version(void)` (bump on any signature change;
the Julia extension refuses to register `:native` on mismatch and logs why),
`dleto_ttm_f64(const double* G, const int64_t* dims, int32_t n, int32_t axis,
const double* M, int64_t k, double* out)`, `dleto_sparse_ttm_f64(nnz, rows, cols,
vals, ...)`, Float32 twins.  Julia pins the native version through the jll's
`[compat]` entry, the normal mechanism.  Python later: `ctypes` over the same
`libdleto.so`, or a thin `pyo3` crate in `native/dleto-py/` — the math never gets
reimplemented, only re-bound.

## 4. Maintenance

**The Z-law is the contract** (`test/TestDerivationLaws.jl:34-39, 56-164`): whatever
a solver returns must satisfy `Σ_a P[:,a] ⊗ (Γ·D_a) ≈ 0` relative to
`‖Γ‖·max‖D_a‖`.  Every layer already verifies against it — QuickDer's own `_qdn_verify`
(`QuickDerN.jl:693-743`), the near-degenerate cases (`TestDerivationLaws.jl:186-228`)
that caught the `√tol` bug, the oracle tables on the board.  A native kernel adds one
level below that:

- **Differential tests at the kernel seam**, not at the solver level: for fixed-seed
  random `G` (valence 3/4/5, ragged dims, Float32/64, dense and sphere-sparse) assert
  `native_ttm(G,M,a) ≈ _qdn_ttm(G,M,a)` to `10·eps·‖G‖‖M‖`, and
  `sylvesterLM(...; backend=:native)` bit-close to `:array` on the same 108
  configurations the board already used.  This is cheap and localises a bug to one
  function, which is what the FastDer history says matters.
- **Fuzzing:** the same harness with random dims in `2:9` and random `a`, 10⁴ cases
  per CI run, seeds logged so a failure replays.
- **Fixtures, not re-derivation:** dump a handful of `(G, M, a, expected)` from Julia
  into `test/fixtures/`; `cargo test` reads them so the Rust crate is tested without
  Julia in the loop.
- **CI cost:** Julia suite is already ~6 min (`docs/CONTEXT.md` test table); a native
  matrix adds ~5 build jobs × ~3 min plus the differential run.  Acceptable, but every
  ABI bump reruns all of it.
- **Who fixes what:** the solver-level Z-law failing with `:native` but not `:array`
  → kernel bug, owned by whoever touches `native/`; failing with both → algorithm bug,
  owned in `src/`.  The `backend=` switch is the bisector, as it was designed to be
  (`SylverLining.jl:54-58`).
- **AI-agent loop:** today an agent edits Julia and reruns a `bench/jl` script in
  ~30–60 s including JIT.  With a native layer the loop is `cargo build --release`
  (seconds for a small crate) + a Julia restart to reload the `.dylib` (`ccall`ed
  libraries are not hot-reloadable without `Libdl` gymnastics) + the differential
  test.  Roughly 2–3x slower iteration and two languages of context per change.  For
  a team that is mostly agents, that cost is real and recurring; it is the main
  reason the gate in §5 is strict.

## 5. Roadmap with exit criteria

**Phase 0 — measure and pin (1 session).**  Turn the probe above into
`bench/FloorReport.jl`: per stage (sketch, restricted build, Gram, Cholesky, Ritz,
lift, verify, one operator apply) × size (v3 d = 50..250, v4 d = 20..60, video
100³×3) report time and ratio to the BLAS/bandwidth floor.  Freeze the reference:
the differential fixtures and seeds.  *Exit:* the table exists in
`bench/reports/`; every stage has a ratio.  *Go/no-go for everything below:* a stage
with ratio ≥ 2 at a size the user cares about.

**Phase 1 — harden the Julia (1–2 sessions), regardless of the gate.**
(a) Zero-alloc QuickDerN: `_qdn_ttm!` into caller scratch, `permutedims!`, reuse
`H_ab` across basis vectors — mirror what `_sylvesterLM_array` did.  (b) PrecompileTools
workload covering `der`/`stratify` at valence 3 and 4, Float32/64.  (c) Threaded
sparse mode product (or a `Threads.@threads` over unfolding columns) and measure vs
stdlib.  (d) Opt-in `using AppleAccelerate` via LBT forwarding, benchmarked on the
5700² `syrk` (vendor claims 6–13x GEMM; unverified — contraction-options.md §1.2).
*Exit:* apply ratio ≤ 1.2 dense, ≤ 1.5 sparse; first `stratify` call ≤ 2 s; no
allocation of size `d^n` inside the solve loop.

**Phase 2 — the restricted solve (2–3 sessions; this is where the 10x is).**
The d ≈ 130 cliff.  Candidates, cheapest first: (a) use the Kronecker structure —
the per-axis Gram block is `kron(I_r, S_a(a) S_a(a)ᵗ)`, a `d_a × d_a` Cholesky, as a
block-diagonal preconditioner for LSMR/LOBPCG on the restricted map;
(b) `GramSolver` in Float32 `syrk` + Float64 Rayleigh–Ritz (halves the 1.9 s and the
memory on CPU; on the Metal path of Phase 4 it is the same split; the Ritz step on
`M` keeps the σ precision, `NullSolvers.jl:216-230`);
(c) randomized range finder for the *complement* of the null space, since nullity is
tiny (2–13) and known to be bracketed by the confirmation pass;
(d) raise the dense/matrix-free threshold once (b) halves the bytes.
*Exit:* v3 d = 250 restricted solve ≤ 10 s (was 33 s Arpack); d = 130 stratify
≤ 60 s (was 417 s); d = 1000 v3 `der` under an hour on the owner's machine.

**Phase 3 — native kernels, conditional.**  Only if Phase 0 + Phase 1 leave a stage
≥ 2x off floor.  Realistic candidates: sparse nnz TTM; fused permute+GEMM for the
non-flat axes.  Rust crate, C ABI as in §3, `Dleto_jll` weakdep, `backend=:native`,
`:array` remains default until the differential suite is green on all five platforms.
*Exit:* ≥ 2x on the targeted stage at the targeted size, bit-close on the
108-config differential suite, CI green on the matrix.  *If the 2x is not measured
in the first two weeks, stop and delete the branch* — half-finished native code is
the worst outcome.

**Phase 4 — GPU, Float32 only, already Julia-native.**  While this document was
being written, `ef0249a` landed `ext/DletoMetalExt.jl` (Metal.jl weakdep, device
hooks, `backend = :metal` on `sylvesterLM`) — i.e. exactly "Metal.jl on the kernel
seam, no C++/Rust", which settles the GPU half of the language question.  Its header
claims, on an M4 Max vs 8 CPU threads, fp32 GEMM 12 vs 0.8 TFlop/s, TTM 3–5x, **Gram
formation 9x** — and Gram formation is the stage my probe found dominant.  Not
reproduced here; treat as the hypothesis Phase 0 tests.  The precision gate is the
real question: Apple GPUs have no fp64 (`wgpu` lists `SHADER_F64` for Vulkan only),
the verdict machinery already knows Float32 runs out of margin at d ≥ 45 on the sphere
(`GAP_RATIO` note, `NullSolvers.jl:416-441`), and `GRAM_SHIFT_REL = 1e-10` is below
fp32 eps.  A workable design is fp32 Gram + shifted subspace iteration on the GPU to
*capture* near-null directions, then Float64 Rayleigh–Ritz on `M` on the CPU to
*resolve* them — the two-stage structure `GramSolver` already has
(`NullSolvers.jl:207-236`).  *Exit before `:metal` becomes a default anywhere:* same
nullity and Z-law residual ≤ 1e-8 as `:array` on the scrambled-sphere suite v3
d = 40..100, v4 d = 20..40, including the d = 60 near-derivation at σ = 1.9e-6.

**Explicitly not ported, ever:** `solve_nullspace` and its verdict/escalation/
confirmation logic; the solver registry and extensions; restriction-size policy,
retry, and the lift consistency filter; `AutoDer`; chisel/operator embeddings and
engagement reduction; the ITensor boundary; anything that already calls LAPACK.

## Sources

- Julia 1.12 highlights (juliac, `--trim`): <https://julialang.org/blog/2025/10/julia-1.12-highlights/>;
  JuliaC.jl (requires 1.12+, `--compile-ccallable`, `--trim` experimental on some builds): <https://github.com/JuliaLang/JuliaC.jl>;
  LWN on 1.12 standalone binaries: <https://lwn.net/Articles/1044280/>
- BinaryBuilder / Yggdrasil / jll packages: <https://docs.binarybuilder.org/stable/jll/>, Rust toolchain: <https://docs.binarybuilder.org/stable/build_tips/>, <https://github.com/JuliaPackaging/Yggdrasil>
- Pkg Artifacts (`lazy`, `download` url/sha256): <https://pkgdocs.julialang.org/v1/artifacts/>
- faer (pure-Rust dense LA, NEON SIMD; performance claims not independently verified here): <https://github.com/sarah-quinones/faer-rs>
- wgpu `Features::SHADER_F64` "Supported Platforms: Vulkan": <https://docs.rs/wgpu/latest/wgpu/struct.Features.html>
- MPSMatrixMultiplication data types (no float64): <https://machinethink.net/blog/mps-matrix-multiplication/> [secondary source]
- Metal.jl / ITensors GPU notes and AppleAccelerate claim: `bench/reports/night-2026-09-03/contraction-options.md` §1.2, §2
- Measurements: this document's probe (`bench/jl`, 2 threads, M4, 64 GB), and
  `bench/reports/night-2026-09-03/BOARD.md`, `quickder-restricted-solvers.csv`,
  `bench/stratify-scaling-QuickDer.csv`.
