#
# Strata Dleto: QuickDerN
#   Solve-and-lift derivations for tensors of ANY valence.
#
# NOTICE:
#   The solve-and-lift strategy generalised here was developed in companion
#   work by Chris Liu, Joshua Maglione, and James B. Wilson.  Please retain
#   this attribution when reusing this implementation.
#
# This implements docs/design/QuickDer-valence-n.md.  `FastDer3Valent.jl` is a
# line-for-line transcription of the valence-3 reference and stays as the
# oracle (`:QuickDer3` / `:FastDer3Valent`); this file is the general method
# (`:QuickDer`) and is written from the equations rather than from that code,
# so the two are independent implementations that can be compared.
#
# THE SHAPE OF THE METHOD, in one paragraph.  A derivation is a tuple of
# matrices `M_a` with `Σ_a P[ρ,a] (Γ ×_a M_a) = 0` for every chisel row `ρ`.
# Contract every OUTPUT axis of that equation with a thin orthonormal `W_a`
# (d_a x r_a) and it becomes a much smaller system in the restricted unknowns
# `Y_a = M_a W_a` -- one whose size is governed by `r`, not by `d`.  Solve that,
# then recover the missing half `Z_a = M_a W_a⊥` one axis at a time by least
# squares, then CHECK the answer against the defining equation, because
# solve-and-lift is only *generically* correct at a given `r`.
#
# WHY THE SKETCH IS RANDOM BY DEFAULT.  Liu's valence-3 kernel restricts to a
# CORNER, `W_a = I[:, 1:r_a]`, which is free (the sketches are slices).  That is
# a fine choice on a dense generic tensor and a fragile one on a structured
# tensor, where the corner block can miss the support.  Measured on the
# unscrambled sphere octant (support `i+j+k = d-1`), valence 3, `:corner`:
# d = 10, 12 land the answer; from d = 16 the LIFT operator `S_a(a)ᵗ` loses
# column rank -- the corner cross sketch still meets the support hyperplane,
# because one axis of `S_a` stays full, but the part of it the lift sees no
# longer determines `Z_a` -- and only the minimum-norm fallback below saves the
# answer.  A random orthonormal `W_a` costs one pass over Γ per axis and is
# generic for any Γ, with no such cliff.  Both are available; `:corner` is kept
# because it is genuinely cheaper when the tensor is dense and generic.
#
# THE KERNEL IS PLAIN ARRAYS -- OF WHATEVER KIND.  ITensors appear only at the
# boundary, and every contraction goes through `_qdn_ttm`, so swapping in
# TensorOperations, Strided or Finch is one function (see the contraction study
# in bench/reports/night-2026-09-03/contraction-options.md).  `device = :gpu`
# uses that same property from the other side: `_qdn_ttm`, `_qdn_unfold` and
# `_qdn_slice` are written against `AbstractArray` and allocate with
# `similar(G, ...)`, so handing them an `MtlArray` keeps every pass over the
# `d^n` tensor on the GPU with no second implementation.  Apple GPUs have no
# Float64, so that path is Float32 and exploratory; the certified answer is the
# Float64 CPU one.
#
# WHITENED RESTRICTION (`whiten = true`, "QuickDer-W").  The restricted system
# is `[A_1 ⋯ A_n]` with `A_a = I_{r_a} ⊗ M_a` up to a row permutation, where
# `M_a` is the transposed mode-`a` unfolding of the cross sketch `S_a`.  So the
# diagonal blocks of its Gram are `c_a·(I ⊗ M_aᵗM_a)` with `c_a = Σ_ρ P[ρ,a]²`:
# every bit of the sketch-induced conditioning lives in `d_a x d_a` Grams, and
# the off-diagonal blocks -- which are what encodes the derivation condition --
# are untouched by a per-axis change of variables.  Substituting `Ỹ_a = R_a Y_a`
# for the thin QR `M_a = Q_a R_a` therefore makes every diagonal Gram block
# exactly `c_a·I`, puts the whole spectrum in `[0, Σ_a c_a]`, and leaves the
# null space alone.  Cost: `n` QRs of `(∏_{b≠a} r_b) x d_a`, negligible against
# a pass over `d^n`.  See `_qdn_whiten_axis` for the degenerate-mode case and
# the "Whitened restriction" section of the design note for the measurements.
#

using LinearAlgebra
using LinearMaps
using Random

# ---------------------------------------------------------------------------
# The method
# ---------------------------------------------------------------------------

"""
Restricted systems with at least this many unknowns take the Gram route
(`:GramSolver`) instead of a full SVD in the dense branch.  Below it the SVD
costs well under a second and keeps full precision.  A `Ref` so benchmarks can
flip the route (`QDN_GRAM_MIN_COLS[] = typemax(Int)` forces the SVD).
"""
const QDN_GRAM_MIN_COLS = Ref(1000)

"""
Byte budget for the DENSE restricted matrix on the host route.  Mirrors
`DENSE_BUDGET_BYTES / 2` (`NullSolvers.jl`): the matrix plus the Gram plus the
Cholesky factor all have to fit, so filling the whole null-solver budget with
the matrix alone would leave nothing for the solve.  A `Ref` because it is a
policy number, not a fact about the machine -- benchmarks that want the dense
route at a size the default declines (a Float32 CPU run being compared against
a Float32 GPU run, say) raise it explicitly.

Not a `const` expression of `DENSE_BUDGET_BYTES`: `QuickDerN.jl` is included
before `NullSolvers.jl`, so that name does not exist yet at load time.
"""
const QDN_DENSE_BUDGET_BYTES = Ref(2.5 * 2^30)
# 2.5 GB, not 0.5: the matrix-free branch is the weak spot.  On a STRUCTURED
# tensor its spectrum is clustered and Arpack hits its iteration cap
# (XYAUPD_Exception(1)) on the scrambled sphere at d = 150 and 200, where the
# dense Gram route answers in seconds; random tensors converge there
# (bench/reports/night-2026-09-03/quickder-restricted-solvers.csv).  At 2.5 GB
# the Gram route reaches d = 200 at valence 3 (17576 x 15600, 2.2 GB) with a
# peak of ~3x the matrix, under the machine's 10 GB default budget.

"""
Byte budget for the dense restricted matrix on the DEVICE route
(`device = :gpu`).  Larger than the host budget because the device this was
written for has a 52 GB recommended working set while the process that feeds
it has a 16 GB cap: at valence 3 Float32 the matrix is 0.16 GB at d = 100,
1.1 GB at d = 200 and 3.3 GB at d = 300, and only the last is close to the
host-side ceiling (host copy + device copy + device Gram).
"""
const QDN_GPU_DENSE_BUDGET_BYTES = Ref(6.0 * 2^30)

"""
    _qdn_default_free_solver() -> Symbol

Null solver for the MATRIX-FREE restricted branch when the caller left
`solver = :AutoSolver`.

`:AutoSolver` puts LSMR first for a rectangular map, which is right for
`den`'s ill-conditioned densor map and wrong for this one.  Measured on the
matrix-free restricted branch (random `d^3`, 2 threads,
`bench/reports/night-2026-09-03/quickder-restricted-solvers.csv`):

| d | Arpack | Krylov | CG | LSMR |
|---|---|---|---|---|
| 150 | 10.5 s | 76 s | 210 s | did not finish |
| 200 | 49 s | - | - | - |
| 250 | 33 s | - | - | - |

So ARPACK when its package is loaded, `:KrylovSolver` otherwise (the KrylovKit
extension registers itself unconditionally).  An explicit `solver =` still
wins -- this only replaces the *default*.
"""
_qdn_default_free_solver() =
    haskey(SOLVER_REGISTRY, :ArpackSolver) ? :ArpackSolver : :KrylovSolver

"""
    QDN_STAGE_TIMES :: Ref{Union{Nothing, Dict{Symbol,Float64}}}

Opt-in per-stage wall clock, for benchmarking only.  Set it to an empty `Dict`
and the next `derTrOpsReduced` accumulates seconds under `:upload`, `:sketch`,
`:restricted` (dense matrix assembly), `:solve`, `:lift`, `:filter`, `:verify`
and `:restrict_ops` (plus `:whiten` when `whiten = true`); leave it `nothing`
(the default) and the cost is one `Ref` load per stage.  `GramSolver` adds `:gram`, `:cholesky`, `:subspace` and
`:ritz` through the same dictionary.

Measuring a GPU stage needs a synchronisation point, and every stage here
already ends at one -- a `to_cpu`, or a scalar read like `norm(E)` or
`maximum(abs, diag(G))` -- because the next stage is host work that needs the
device's answer.  That is why the boundaries below sit *after* the transfer
rather than after the last kernel launch: an asynchronous launch would
otherwise charge its time to whichever stage happened to block first.
"""
const QDN_STAGE_TIMES = Ref{Union{Nothing, Dict{Symbol,Float64}}}(nothing)

"""
    QDN_STAGE_BYTES :: Ref{Union{Nothing, Dict{Symbol,NTuple{2,Float64}}}}

Opt-in per-stage MEMORY, the companion of `QDN_STAGE_TIMES` and on the same
stage boundaries.  Set it to an empty `Dict` and each stage records

    (bytes allocated during the stage, `Sys.maxrss()` at the end of it)

so a run says both what it churned and where its peak was.  Allocation is read
from `Base.gc_bytes()`, which counts every allocation whether or not it
survives -- the churn -- while `Sys.maxrss()` is the process peak SO FAR and
therefore monotone across stages: the stage whose RSS entry first reaches the
final value is the one that set the peak.  Two numbers because they answer
different questions, and the memory wall this was written for (d = 1000 at
valence 3) is a peak wall, not a churn wall.

Left `nothing` (the default) the cost is one `Ref` load per stage, the same as
the clock.
"""
const QDN_STAGE_BYTES = Ref{Union{Nothing, Dict{Symbol,NTuple{2,Float64}}}}(nothing)

# Allocation counter at the last stage boundary, so consecutive stages can
# report a difference.  Reset by `_qdn_stage_reset!` at the top of a run.
const _QDN_BYTE_MARK = Ref(0.0)

"""
    _qdn_stage_reset!()

Start a fresh per-stage memory accounting, so the first stage of a run is
charged from the call and not from whenever the counters were last read.
"""
function _qdn_stage_reset!()
    QDN_STAGE_BYTES[] === nothing || (_QDN_BYTE_MARK[] = Float64(Base.gc_bytes()))
    return nothing
end

"""
    _qdn_stage!(name, t0) -> Float64

Charge `time() - t0` to stage `name` when stage timing is on, and return the
current time so consecutive stages can chain (`t0 = _qdn_stage!(:sketch, t0)`).
Also charges the bytes allocated since the previous boundary, and records the
process peak RSS, when `QDN_STAGE_BYTES` is on.
"""
function _qdn_stage!(name::Symbol, t0::Float64)
    t = time()
    d = QDN_STAGE_TIMES[]
    d === nothing || (d[name] = get(d, name, 0.0) + (t - t0))
    b = QDN_STAGE_BYTES[]
    if b !== nothing
        now = Float64(Base.gc_bytes())
        (al, _) = get(b, name, (0.0, 0.0))
        b[name] = (al + (now - _QDN_BYTE_MARK[]), Float64(Sys.maxrss()))
        _QDN_BYTE_MARK[] = now
    end
    return t
end

"""
    _qdn_check_device(device, T)

`device` is `:cpu` or `:gpu`; `:gpu` additionally needs a functional GPU
backend extension (`using Metal` alongside `using Dleto`) and a Float32
tensor, because Apple GPUs have no Float64 at all.  A GPU run is therefore an
exploratory Float32 run; Float64 certification stays on the CPU.
"""
function _qdn_check_device(device::Symbol, ::Type{T}) where {T}
    device === :cpu && return :cpu
    device === :gpu || error(
        "QuickDerMethod: device must be :cpu or :gpu, got :$device.")
    gpu_available() || error(
        "QuickDerMethod(device = :gpu): no GPU backend is loaded or functional. " *
        "Load one alongside Dleto (`using Metal` on Apple silicon) and check " *
        "`Dleto.gpu_available()`.")
    T === Float32 || error(
        "QuickDerMethod(device = :gpu): the GPU path is Float32 only (Apple GPUs " *
        "have no Float64), but eltype(Γ) is $T. Convert Γ to Float32 for an " *
        "exploratory run, or keep device = :cpu for the certified one.")
    return :gpu
end

"""
    QDN_APPLY_COUNT :: Ref{Int}

Opt-in count of MATRIX-FREE restricted applies, for benchmarking only.  Set it
to `0` and the next `derTrOpsReduced` accumulates one per forward and one per
adjoint application of `_qdn_restricted_map`; leave it at `-1` (the default)
and the cost is one `Ref` load per apply.  Iteration count is the only lever
left on that branch (the dense branch is at the BLAS floor), so it is worth
being able to read directly rather than inferring it from wall time.
"""
const QDN_APPLY_COUNT = Ref(-1)

"""
    QDN_LAST_SOLVE_STATUS :: Ref{Symbol}

The `status` of the restricted null solve of the LAST `_qdn_solve_and_lift`
attempt: `:ok`, `:unconverged` or `:capped` (see `NullVerdict`).  Set by the
kernel, reset per attempt, in the idiom `QDN_APPLY_COUNT` and
`QDN_TRIVIAL_FACTORED` already use in this file.

It is out of band because `_qdn_solve_and_lift` returns the lifted
derivations, and the caller only learns that the answer is EMPTY after
`_fastder_restrict_to_ops` has intersected them with `Ω` -- one step past the
kernel.  An empty answer from a solve that did not converge is a failure, not
"Γ conforms to no pattern for this chisel", and `derTrOpsReduced` needs both
facts in the same place to say so.
"""
const QDN_LAST_SOLVE_STATUS = Ref(:ok)

@inline function _qdn_tick!()
    c = QDN_APPLY_COUNT[]
    c < 0 || (QDN_APPLY_COUNT[] = c + 1)
    return nothing
end

"""
    QDN_TRIVIAL_MAX_BYTES :: Ref{Float64}

Budget for MATERIALISING the trivial derivations of a rank-deficient mode as
full operator tuples.  Default 256 MB.

Why there is a budget at all.  A mode-`a` unfolding of rank `rank_a < d_a`
gives a trivial space of dimension `(d_a - rank_a)·d_a` (see
`_qdn_trivial_ders`), and one basis element written out is `n` matrices of
`d_b x d_b`.  At d = 500 with a single degenerate mode that is 500 elements at
6 MB each -- 3 GB, for a space whose complete description is one `d x
(d - rank)` matrix of 2 MB.  So the factored description is always computed and
always published (`QDN_TRIVIAL_FACTORED`), and only as many tuples as fit in
this budget are written out; when that truncates, a `@warn` says so and names
the field.

`Inf` materialises everything, as the code did before the budget existed.
"""
const QDN_TRIVIAL_MAX_BYTES = Ref(256.0 * 2^20)

"""
    QDN_TRIVIAL_FACTORED :: Ref{Vector{<:NamedTuple}}

The trivial derivations of the last `derTrOpsReduced` call, COMPLETE, in the
only form that is affordable to write down: one entry `(; axis, K)` per
rank-deficient axis, where `K` is `d_axis x p` with orthonormal columns and the
meaning is

    every `X_axis = K * v * transpose(e_q)`, for every `v` in `R^p` and every
    basis vector `e_q` of `R^{d_axis}`, with the other axes' matrices zero,
    is a derivation of Γ for every chisel

-- a space of dimension `p · d_axis` per axis.  Empty (the usual case) when
every mode unfolding had full rank.

Set by the kernel, not by the caller, and reset at the start of every attempt,
in the idiom `QDN_STAGE_TIMES` and `QDN_APPLY_COUNT` already use in this file.
It is out of band because `derTrOpsReduced`'s return type is fixed by the
`DerivationMethod` interface (`(Ω, expand_map, ders)`) and `ders` is a dense
matrix of coordinate vectors -- which is exactly the representation that
cannot hold this space at scale.
"""
const QDN_TRIVIAL_FACTORED = Ref{Vector{<:NamedTuple}}(NamedTuple[])

"""
    QDN_LIFT_GAP_RATIO :: Ref{Float64}
    QDN_LIFT_CEILING   :: Ref{Float64}

The lift consistency filter's cut rule (`_qdn_solve_and_lift`), which decides
which combinations of restricted null directions actually lift to derivations.

The filter takes a null space of the lift-residual matrix `Rall`, and it used
to take it at a HARD relative cutoff, `atol = max(tol, sqrt(eps(T)))`.  That
cutoff sits on a cliff, and in Float32 the cliff is on the wrong side of the
answer.  Measured on the scrambled sphere valence 3 in Float32, oracle nullity
3: 3 of 3 at d <= 32, 2 of 3 at d = 48 and 64, 1 of 3 at d = 100, 0 of 3 at
d = 140 -- an undercount that grows with d and is silent.  The residuals of the
GENUINE directions there are 1.3 to 2.9 times `sqrt(eps(Float32))`, flat in d;
the cutoff is at 1.0 times it.

Raising the constant is not the fix: at nullity 13 (the raw, unscrambled
sphere) a looser constant sweeps spurious directions in.  What separates the
two populations is not a threshold, it is a GAP -- the genuine directions leave
a residual at the arithmetic's own noise and the spurious ones leave an O(1)
one -- so the cut is placed by the largest consecutive ratio in `Rall`'s
singular spectrum, the same rule `gap_verdict` applies to a null solver's
spectrum in `NullSolvers.jl`, and it lands in the gap whatever `T` is.

`QDN_LIFT_GAP_RATIO` (100, which is `NullSolvers.GAP_RATIO` written out --
that file is `include`d after this one, so the name is not in scope here) is
the jump a cut has to clear.
`QDN_LIFT_CEILING` (default 32) multiplies `sqrt(eps(T))` to bound which cuts
are eligible at all, so a spectrum with no gap in it cannot cut in the middle
of the spurious cluster: 1.1e-2 relative in Float32, and in Float64 it is
below the default `tol` and so changes nothing there.
"""
const QDN_LIFT_GAP_RATIO = Ref(100.0)
const QDN_LIFT_CEILING = Ref(32.0)

"""
    QuickDerMethod(; restriction = :random, sizes = nothing, solver = :AutoSolver,
                     verify = :random, nslices = 4, seed = nothing,
                     device = :cpu, whiten = true)

Solve-and-lift derivations for a tensor of any valence `n >= 2`, any per-axis
dimensions, any chisel with at least one engaged axis, and any
`IndTransverseOps` operator space.

- `restriction`  `:random` (default) sketches each axis with the Q factor of a
  random square Gaussian; `:corner` uses `I[:, 1:r_a]`, which makes every
  sketch a slice and every lift right-hand side a sub-block -- Liu's original
  choice, and the fast one on a dense generic tensor.  `:corner` is fragile on
  a tensor whose leading corner is degenerate: on the unscrambled sphere its
  lift operator loses column rank from d = 16 up (see the file header), so
  `:random` is the default.
- `sizes`        override the restriction sizes `r_1..r_n` (see
  `_qdn_restriction_sizes` for what is chosen otherwise and why).
- `solver`       null solver for the MATRIX-FREE restricted solve.  Left at
  `:AutoSolver` it means "pick the one that is fastest on THIS map", which is
  `_qdn_default_free_solver()`, not `AutoSolver`'s own LSMR-first rule.  The
  dense branch picks between `SVDSolver` and `GramSolver` by size and ignores
  this (`QDN_GRAM_MIN_COLS`).
- `verify`       `:random` (default) checks the defining equation on `nslices`
  output slices of the largest engaged axis; `:full` checks all of it when
  `prod(dims) <= 2e7`; `:none` skips the check and is for benchmarking only.
- `seed`         makes the sketch (and the choice of verification slices)
  reproducible.  Without it the global RNG is used.
- `device`       `:cpu` (default) or `:gpu`.  `:gpu` needs a functional GPU
  backend extension and a **Float32** tensor (Apple GPUs have no Float64), and
  errors clearly when either is missing.  What moves to the device is the work
  that scales with `d^n` or with the restricted matrix: the cross sketches, the
  pair tensors of the lift, the Gram/Cholesky/subspace stage of the restricted
  solve (`GramSolver(device = :gpu)`) and the verification.  What stays on the
  host is the small linear algebra the device has no kernels for -- the
  restricted matrix assembly (`kron` + row permutation, a memory-bound scatter
  of a `d_a x R_a` block), the per-axis lift QR, the consistency filter, and the
  `qr`/`svd` inside `GramSolver` (Metal.jl has neither).  A `:gpu` run is
  exploratory: Float32 is its ceiling, and the certified answer is the Float64
  CPU one.
- `whiten`       `true` (default) removes the sketch-induced ill-conditioning
  of the restricted system exactly, by a thin QR per axis, before it reaches the
  null solver (see "WHITENED RESTRICTION" in the file header).  Costs `n` QRs of
  `(∏_{b≠a} r_b) x d_a` -- 0.01 s of an 18 s solve at valence 3, d = 200 -- and
  changes no answer.  It is on by default because it is what makes the
  MATRIX-FREE branch converge at all on a structured tensor: measured on the
  scrambled sphere octant, forced matrix-free, ARPACK, Float64, 5 threads
  (bench/reports/2026-09-04/whitened/):

  | d | plain applies | plain verdict | whitened applies | whitened verdict |
  |---|---|---|---|---|
  | 30  | 53632  | ARPACK hit its cap | 23732 | nullity 3, resid 2.8e-13 |
  | 100 | 66900  | ARPACK hit its cap | 34636 | nullity 3, resid 7.8e-11 |
  | 150 | 177142 | ARPACK hit its cap | 39544 | nullity 3, resid 1.7e-12 |
  | 200 | 65182  | ARPACK hit its cap | 38262 | nullity 3, resid 3.3e-10 |

  On a GENERIC tensor, where the mode unfoldings are already well conditioned
  (`cond(M_a) ≈ 3` against 20..150 on the sphere), it is worth 1.4x and no more
  -- `randn(150,150,150)`, 14618 applies -> 10156.  Set `whiten = false` to
  reproduce the pre-2026-09-04 behaviour.

Element type follows `eltype(Γ)`; `tol` is relative and floored at
`sqrt(eps(T))` by `_qd_tolerance`, as in `FastDer3ValentMethod`.

Like `FastDer3ValentMethod`, the kernel solves over ALL matrices and the answer
is intersected with `Ω` afterwards (`_fastder_restrict_to_ops`), so the result
means the same thing `SylverLining` on that `Ω` means: coordinates in `Ω`, of
the derivations that lie in `Ω`.
"""
struct QuickDerMethod <: DerivationMethod
    restriction::Symbol
    sizes::Union{Nothing, Vector{Int}}
    solver::Symbol
    verify::Symbol
    nslices::Int
    seed::Union{Nothing, Int}
    device::Symbol
    whiten::Bool
end

function QuickDerMethod(; restriction::Symbol = :random, sizes = nothing,
                        solver::Symbol = :AutoSolver, verify::Symbol = :random,
                        nslices::Integer = 4, seed = nothing,
                        device::Symbol = :cpu, whiten::Bool = true)
    restriction in (:random, :corner) ||
        error("QuickDerMethod: restriction must be :random or :corner, got :$restriction.")
    verify in (:random, :full, :none) ||
        error("QuickDerMethod: verify must be :random, :full or :none, got :$verify.")
    nslices >= 1 || error("QuickDerMethod: nslices must be at least 1, got $nslices.")
    device in (:cpu, :gpu) ||
        error("QuickDerMethod: device must be :cpu or :gpu, got :$device.")
    return QuickDerMethod(restriction,
                          sizes === nothing ? nothing : Int[Int(s) for s in sizes],
                          solver, verify, Int(nslices),
                          seed === nothing ? nothing : Int(seed), device, whiten)
end

# ---------------------------------------------------------------------------
# The contraction kernel -- the ONE place a backend swap has to touch
# ---------------------------------------------------------------------------

"""
    _qdn_front(N, a) -> NTuple{N,Int}

The permutation of `1:N` that brings axis `a` to the front and leaves the
others in their original order.  Its inverse puts it back.
"""
_qdn_front(N::Integer, a::Integer) =
    ntuple(i -> i == 1 ? Int(a) : (i - 1 < a ? i - 1 : i), N)

# ---------------------------------------------------------------------------
# What a mode product COSTS on the device -- the three numbers that decide the
# two choices below (which route `_qdn_ttm!` takes on a middle axis, and which
# order `_qdn_cross_sketches` and `_qdn_pair_tensor` apply their sketches in)
# ---------------------------------------------------------------------------
#
# All three are measured on the M4 Max in docs/design/Float16-Metal.md, on the
# movie shape this exists for (640 x 480 x F x 3):
#
#   * a GEMM streaming the tensor once runs at 100-220 GB/s of operand traffic
#     (the unfold-1 and unfold-3 rows).  200 GB/s is the optimistic end and
#     that is the right side to err on here: it makes the model reluctant to
#     pay a permute, which is the thing being avoided.
#   * `permutedims` of the 331 MB Float32 movie tensor takes 0.0155 s, i.e.
#     ~21 GB/s counted once over the input, ~10 GB/s counted over the read AND
#     the write.  The round-trip figure is the one to charge, since the copy is
#     pure traffic with no arithmetic hiding behind it.
#   * a kernel dispatch costs ~60 us.  This is the number the "one transpose
#     versus a thousand launches" remark in `_qdn_ttm` was worried about, and
#     it is the only one of the three that is a guess rather than a row in the
#     table -- but the decision it drives is a comparison against a permute
#     that costs tens of milliseconds, so it only has to be right to a factor
#     of a few.
#
# `const`, not `Ref`: these are facts about the device, not policy knobs, and
# the two functions below turn them into a decision rather than a threshold --
# which matters because the right answer genuinely flips with the shape.  On
# the movie tensor at F = 90, contracting axis 2 costs 270 launches (16 ms)
# against a 35 ms permute, so the slabs win; contracting axis 3 costs 3
# launches, so they win by two orders of magnitude; and on the 10x10x10x3
# warm-up tensor the permute is 2.4 us against 30 launches, so it wins.
const QDN_GPU_STREAM_BW  = 200.0e9    # B/s, a GEMM reading the tensor once
const QDN_GPU_PERMUTE_BW = 10.0e9     # B/s, `permutedims` read + write
const QDN_GPU_LAUNCH_S   = 60.0e-6    # s, one kernel dispatch

"""
    _qdn_slab_is_cheap(G, a, k, N) -> Bool

Should the mode-`a` product on a DEVICE array `G` be done as one GEMM per
slab of the trailing axes, rather than by permuting axis `a` to the front?

A middle axis is the one case a mode product cannot express as a single GEMM.
The two ways out both cost one streaming pass over `G` plus an overhead:
`back` kernel launches for the slab route, or a permuted `d^n` copy in and a
smaller one out for the permute route.  This compares those overheads with the
constants above; the streaming pass is common to both and cancels, but it is
kept in so the numbers read as seconds.

The answer is NOT a threshold on `back`.  What decides it is whether a slab is
big enough to be worth a dispatch, and that is a statement about `front * d`
as much as about `back` -- the frame axis of a movie has `back = 3` and the
column axis `back = 3F`, and both are the same tensor.
"""
function _qdn_slab_is_cheap(sz::NTuple{N,Int}, a::Integer, k::Integer,
                            ::Type{T}) where {N,T}
    d = sz[a]
    bytes = float(prod(sz)) * sizeof(T)
    back = prod(ntuple(i -> sz[a + i], N - a))
    slab_s = float(back) * QDN_GPU_LAUNCH_S
    # The permute route copies `G` in and the (k/d)-sized result back out.
    permute_s = bytes * (1 + k / max(d, 1)) / QDN_GPU_PERMUTE_BW
    return slab_s <= permute_s
end

"""
    _qdn_ttm(G, M, a) -> Array

The mode-`a` product `G ×_a M` with `M` of size `(size(G,a), k)`:

    (G ×_a M)[.., i, ..] = Σ_p G[.., p, ..] M[p, i]

M's FIRST index meets the tensor.  That is the convention `applyDerivation` and
`embedITensors` use (an operator ITensor carries the frame index first), so
nothing in this file ever transposes an operator at the ITensor boundary.

Used for EVERY contraction in this file -- the sketches, the restricted solve,
the lift right-hand sides, the verification and the trivial derivations -- so a
later move to TensorOperations/Strided/Finch is a change to this function
alone.

NO `permutedims` OF THE INPUT, ever, and that is a memory result and not a
style one.  A mode product on a `d^n` tensor with a `d x k` matrix produces
`k/d` of the input's bytes, so the only large array it needs to touch is the
one it was handed -- but the obvious "permute axis `a` to the front, one GEMM
on the unfolding" route makes a full transposed COPY first.  At d = 1000
valence 3 that copy is 8 GB, live while the GEMM runs and on top of the
tensor itself; it is what put the lift's pair tensors (`_qdn_pair_tensor`,
`n·(n-1)` mode products on the full tensor) out of reach of a 20 GB process,
and measured on the d = 300 profile it is also the largest single allocation
of a run.  Three cases, none of them permuting:

- `a == 1`: the unfolding IS `reshape(G, d, :)`; one GEMM `Mᵗ · G_(1)`.
- `a == N`: the unfolding with axis `a` LAST is `reshape(G, :, d)`; one GEMM
  `G_(N)ᵗ · M`, and the result reshapes straight back.  This is the case the
  old code paid a `d^n` permute for on every pair tensor.
- `1 < a < N`: reshape to `(front, d, back)` and run `back` GEMMs of
  `(front x d)·(d x k)` on contiguous slices.  `view(G3, :, :, b)` of a host
  `Array` is a strided matrix BLAS takes directly, so this is the same
  arithmetic with none of the traffic: it reads the tensor once and writes only
  the (small) output. At valence 3 axis 2 and d = 1000 that is 1000 GEMMs of
  1000 x 1000 x r, each large enough to keep BLAS's own threads busy.

IT IS ALSO THE WHOLE OF THE GPU PORT.  `reshape`, `similar`, `mul!` and
`permutedims` are all implemented for `MtlArray`, so the function is written
against `AbstractArray` and the output is allocated with `similar(G, ...)`
rather than `Matrix{T}(undef, ...)`: hand it a device array and every
intermediate stays on the device.  Nothing here indexes an element, which is
what a GPU array forbids.  The middle-axis case used to be the one place a
device array still permuted -- "`back` separate kernel launches would trade one
transpose for a thousand" -- but `_qdn_slab_is_cheap` says that trade is worth
making far more often than that reading suggests: on the movie tensor the frame
axis has `back = 3`, so the slab route is three dispatches against a 35 ms
permute, and even the column axis's `3F` dispatches come in under it.  Metal.jl
takes a contiguous `view(G3, :, :, b)` of a device array as a GEMM operand
(`MtlMatrixRangeView`), so the slab loop is the SAME body the host uses.  The
permute route is kept for the shapes where the model says it wins (small
tensors with many trailing slabs, where a dispatch costs more than the copy).
The one thing to keep in mind is the `Array(G)` fallback
below: it is guarded on `DenseArray`, and
`MtlArray <: AbstractGPUArray <: DenseArray`, so a device array never takes it
(taking it would silently move `d^n` bytes to the host).
"""
function _qdn_ttm(G::AbstractArray{T,N}, M::AbstractMatrix{T}, a::Integer) where {T,N}
    GA = G isa DenseArray ? G : Array(G)
    out = similar(GA, T, ntuple(i -> i == a ? size(M, 2) : size(GA, i), N))
    return _qdn_ttm!(out, GA, M, a)
end

"""
    _qdn_ttm!(out, G, M, a, α = 1, β = 0) -> out

`_qdn_ttm` writing into a caller-owned `out` of shape `size(G)` with axis `a`
replaced by `size(M, 2)`: `out <- α·(G ×_a M) + β·out`.  `out` must not alias
`G`.

This is the form the memory-lean pipelines want.  A chain of mode products
`Γ ×_1 M_1 ×_2 M_2 ⋯` with square `M_a` -- the harness's orthogonal scramble
and its `nondeg` change of basis are both exactly that -- allocates a fresh
`d^n` array per step through `_qdn_ttm`, so a valence-3 chain at d = 1000
churns 48 GB and leans on the GC to keep two of those six 8 GB arrays live at
a time.  With two buffers and this function it is 16 GB steady and no churn at
all: `_qdn_ttm!(B, A, M, a); A, B = B, A`.

`α`/`β` are `mul!`'s, and they exist so that a SUM of mode products --
`Σ_a P[ρ,a]·(Γ ×_a M_a)`, which is the derivation equation itself -- can be
accumulated into one array instead of one temporary per term.  On a `d^n`
tensor that is the difference between one extra copy and `n + 1` of them.  The
device PERMUTE fall-through below cannot accumulate (it has to permute its
result into place) and refuses a nonzero `β`; every other branch, the device
slab route included, passes `α`/`β` straight to `mul!`.
"""
function _qdn_ttm!(out::AbstractArray{T,N}, G::AbstractArray{T,N},
                   M::AbstractMatrix{T}, a::Integer,
                   α = one(T), β = zero(T)) where {T,N}
    d = size(G, a)
    size(M, 1) == d || throw(DimensionMismatch(
        "mode-$a product: matrix is $(size(M)) but axis $a has length $d"))
    k = size(M, 2)
    size(out) == ntuple(i -> i == a ? k : size(G, i), N) || throw(DimensionMismatch(
        "mode-$a product: output is $(size(out)) but $(size(G)) with axis $a of " *
        "length $k is $(ntuple(i -> i == a ? k : size(G, i), N))"))
    # Explicit second extents, never `:`: an empty `M` (a saturated axis hands
    # over a `d x 0` block) makes `0 x :` uninferable.
    rest = length(G) ÷ max(d, 1)
    if a == 1
        mul!(reshape(out, k, rest), transpose(M), reshape(G, d, rest), α, β)
    elseif a == N
        mul!(reshape(out, rest, k), reshape(G, rest, d), M, α, β)
    elseif (G isa Array && out isa Array) || _qdn_slab_is_cheap(size(G), a, k, T)
        front = prod(ntuple(i -> size(G, i), a - 1))
        back = prod(ntuple(i -> size(G, a + i), N - a))
        G3 = reshape(G, front, d, back)
        O3 = reshape(out, front, k, back)
        for b in 1:back
            @views mul!(O3[:, :, b], G3[:, :, b], M, α, β)
        end
    else
        iszero(β) || error("_qdn_ttm!: the device path cannot accumulate (β = $β); " *
                           "it permutes its result into `out`.")
        perm = _qdn_front(N, a)
        Gm = reshape(permutedims(G, perm), d, :)
        tmp = similar(G, T, (k, size(Gm, 2)))
        mul!(tmp, transpose(M), Gm, α, zero(T))
        permutedims!(out, reshape(tmp, ntuple(i -> i == 1 ? k : size(G, perm[i]), N)),
                     invperm(collect(perm)))
    end
    return out
end

"""
    _qdn_ttm_square!(G, M, a; block_bytes) -> G

The mode-`a` product `G ×_a M` for a SQUARE `M`, written back over `G`.

The two-buffer form (`_qdn_ttm!`) already holds a chain of mode products to
two copies of the tensor; this holds it to ONE.  It is what makes the harness's
d = 1000 valence-3 build fit: six mode products by square orthogonal matrices
(the scramble and the `nondeg` change of basis) at 8 GB per copy is a 16 GB
peak with two buffers and a 9 GB peak with this, against a 20 GB process.

`G ×_a M` is a separate GEMM for each slice of the axes other than `a`, so the
only thing that cannot be overwritten as it goes is the slice being multiplied.
Take `block_bytes` of slices at a time into a buffer and copy back: the extra
traffic is one read and one write of the tensor per mode product, which against
`2 d^{n+1}` flops of GEMM is a few percent, and the buffer is 64 MB rather than
`d^n`.  The slices are contiguous or strided views in every case (a column
block of `reshape(G, d, :)` for axis 1, a row block of `reshape(G, :, d)` for
axis `N`, a row block of one `(front, d, back)` page in between), so BLAS takes
them directly.

Host `Array` only, and square only -- both are checked.  A rectangular mode
product changes the shape of the tensor and cannot be done in place at all;
`_qdn_ttm!` is the general form.
"""
function _qdn_ttm_square!(G::Array{T,N}, M::AbstractMatrix{T}, a::Integer;
                          block_bytes::Real = 64.0 * 2^20) where {T,N}
    d = size(G, a)
    size(M) == (d, d) || throw(DimensionMismatch(
        "in-place mode-$a product: matrix is $(size(M)), expected ($d, $d)"))
    rest = length(G) ÷ max(d, 1)
    span(n) = clamp(floor(Int, block_bytes / (sizeof(T) * max(d, 1))), 1, max(n, 1))
    if a == 1
        Gm = reshape(G, d, rest)
        cols = span(rest)
        buf = Matrix{T}(undef, d, cols)
        for lo in 1:cols:rest
            hi = min(rest, lo + cols - 1)
            B = view(buf, :, 1:(hi - lo + 1))
            @views mul!(B, transpose(M), Gm[:, lo:hi])
            @views Gm[:, lo:hi] .= B
        end
    elseif a == N
        Gm = reshape(G, rest, d)
        rows = span(rest)
        buf = Matrix{T}(undef, rows, d)
        for lo in 1:rows:rest
            hi = min(rest, lo + rows - 1)
            B = view(buf, 1:(hi - lo + 1), :)
            @views mul!(B, Gm[lo:hi, :], M)
            @views Gm[lo:hi, :] .= B
        end
    else
        front = prod(ntuple(i -> size(G, i), a - 1))
        back = rest ÷ front
        G3 = reshape(G, front, d, back)
        rows = span(front)
        buf = Matrix{T}(undef, rows, d)
        for b in 1:back, lo in 1:rows:front
            hi = min(front, lo + rows - 1)
            B = view(buf, 1:(hi - lo + 1), :)
            @views mul!(B, G3[lo:hi, :, b], M)
            @views G3[lo:hi, :, b] .= B
        end
    end
    return G
end

"""
    _qdn_slice(G, a, idx) -> array

`selectdim(G, a, idx)` materialised on the device `G` lives on.  `copy` of a
view does that for both `Array` and `MtlArray` (GPUArrays lowers it to a
kernel), where `Array(view)` would have moved a device slice to the host.
"""
_qdn_slice(G::AbstractArray, a::Integer, idx) = copy(selectdim(G, a, idx))

"""
    _qdn_host(x) -> Array

`x` as a host `Array`, without a copy when it is one already.  Every result
this file hands back to `_fastder_restrict_to_ops`, to a null solver or to a
host least-squares solve goes through here.
"""
_qdn_host(x::AbstractArray{T,N}) where {T,N} = x isa Array{T,N} ? x : Array(to_cpu(x))

"""
    _qdn_zeros_like(G, dims) -> array

A zero array of shape `dims` on the same device as `G`.
"""
_qdn_zeros_like(G::AbstractArray{T}, dims::Tuple) where {T} =
    fill!(similar(G, T, dims), zero(T))

"""
    _qdn_upload(G, A) -> AbstractMatrix

The host matrix `A` on whichever device `G` lives on, so a mode product against
the full tensor never drags `G` back to the host.
"""
_qdn_upload(G::AbstractArray{T}, A::AbstractMatrix{T}) where {T} =
    G isa Array ? A : to_gpu(A)

"""
    _qdn_unfold(G, a) -> Matrix

The mode-`a` unfolding of `G`: a `size(G,a) x prod(other dims)` matrix whose
columns run over the other axes in their natural order, column-major.  That
column order is the one every row permutation and every lift right-hand side in
this file assumes.

Device-preserving, like `_qdn_ttm`: an `MtlArray` in gives an `MtlMatrix` out
(`reshape` and `permutedims` both keep the array type), and only a
non-`DenseArray` takes the host fallback.
"""
function _qdn_unfold(G::AbstractArray{T,N}, a::Integer) where {T,N}
    GA = G isa DenseArray ? G : Array(G)
    d = size(GA, a)
    a == 1 && return reshape(GA, d, :)
    return reshape(permutedims(GA, _qdn_front(N, a)), d, :)
end

"""
    _qdn_unfold!(buf, G, a) -> Matrix

`_qdn_unfold` with the transposed copy written into `buf`, whose shape must be
`size(G)` permuted by `_qdn_front(N, a)`.  Axis 1 needs no copy at all and
`buf` is left alone there.

For the matrix-free adjoint, which unfolds a small tensor once per chisel row
per axis per apply and would otherwise allocate that copy every time.
"""
function _qdn_unfold!(buf::AbstractArray{T,N}, G::AbstractArray{T,N},
                      a::Integer) where {T,N}
    d = size(G, a)
    rest = length(G) ÷ max(d, 1)
    a == 1 && return reshape(G, d, rest)
    permutedims!(buf, G, _qdn_front(N, a))
    return reshape(buf, d, rest)
end

"""
    _qdn_fold(A, a, sz) -> Array

The inverse of `_qdn_unfold`: the `N`-dimensional array of shape `sz` whose
mode-`a` unfolding is the `sz[a] x prod(sz[b], b≠a)` matrix `A`.  Used only by
the whitened restriction, which replaces a cross sketch by a matrix (`Q_aᵗ`)
and has to hand the matrix-free map a tensor again.
"""
function _qdn_fold(A::AbstractMatrix{T}, a::Integer, sz::NTuple{N,Int}) where {T,N}
    perm = _qdn_front(N, a)
    size(A) == (sz[a], prod(sz) ÷ sz[a]) || throw(DimensionMismatch(
        "fold: matrix is $(size(A)) but shape $sz unfolds at axis $a to " *
        "$((sz[a], prod(sz) ÷ sz[a]))"))
    X = reshape(A, ntuple(i -> sz[perm[i]], N))
    return a == 1 ? Array(X) : permutedims(X, invperm(collect(perm)))
end

"""
    _qdn_row_perm(r, a) -> Vector{Int}

Where the rows of the axis-`a` block land.  Unfolding the restricted equation
with axis `a` LAST orders its rows by `(i_{-a}, i_a)`; the assembled system
orders them column-major over `(i_1..i_n)`.  `perm[t]` is the full-system row
of the `t`-th row in the axis-last order, so `Block[perm, :] = kron(I, Sᵗ)`.
"""
function _qdn_row_perm(r::AbstractVector{<:Integer}, a::Integer)
    n = length(r)
    lin = reshape(collect(1:prod(r)), r...)
    others = [b for b in 1:n if b != a]
    return vec(permutedims(lin, (others..., a)))
end

# ---------------------------------------------------------------------------
# Choosing the restriction sizes
# ---------------------------------------------------------------------------

"""
    _qdn_restriction_sizes(dims, engaged, n = length(dims)) -> Vector{Int}

Per-axis restriction sizes `r_1..r_n`, from the two conditions of
docs/design/QuickDer-valence-n.md section 1:

  (i)  `∏ r_a >= Σ_{a engaged} d_a r_a + slack`
       -- the restricted system has enough equations to be generically full
       column rank, so its null space is the restriction of the true
       derivation space and nothing more;
  (ii) for every engaged axis that will need lifting (`r_a < d_a`),
       `∏_{b≠a} r_b >= d_a + slack`
       -- the lift operator for that axis has full column rank.

The start is the balanced value `r_a = min(d_a, ceil((n·max d)^(1/(n-1))) + 1)`,
which is the smallest `r` that makes `∏r` outgrow `Σ d_a r_a` for equal
dimensions; from there the axis with the smallest `r_a/d_a` that is not already
saturated is bumped until both conditions hold.  Tiny axes (a colour axis of
length 3) saturate immediately, get `W_a = I`, and are never lifted.

The slack is 5% (with a small absolute floor).  Sitting exactly at the
crossover makes the restricted matrix square, and a square system whose true
null space is `k`-dimensional is only `k`-dimensional if it has full rank to
the last row -- which is the assumption most likely to fail on a not-quite-
generic tensor.  Examples: `(100,100,100)` -> 19; `(1000,1000,1000)` ->
`(57,56,56)`; `(100,100,100,3)` -> `(11,10,10,3)`; `(5,4,6,3)` -> `(4,4,4,3)`.

When every axis saturates the "restriction" is the identity, `r == dims`, the
restricted system IS the full system, and the conditions are not required (they
cannot always be met at small `d`, e.g. `(3,3,3)`); the loop stops because
there is nothing left to bump.
"""
function _qdn_restriction_sizes(dims::AbstractVector{<:Integer},
                                engaged::AbstractVector{Bool},
                                n::Integer = length(dims))
    n >= 2 || error("QuickDer needs valence at least 2, got $n.")
    d = Int[Int(x) for x in dims]
    base = ceil(Int, (n * maximum(d))^(1 / (n - 1))) + 1
    r = [min(d[a], base) for a in 1:n]

    unknowns() = sum(Int[d[a] * r[a] for a in 1:n if engaged[a]]; init = 0)
    function conditions_hold()
        u = unknowns()
        prod(r) >= u + max(4, ceil(Int, 0.05 * u)) || return false
        for a in 1:n
            (engaged[a] && r[a] < d[a]) || continue
            (prod(r) ÷ r[a]) >= d[a] + max(2, ceil(Int, 0.05 * d[a])) || return false
        end
        return true
    end

    while !conditions_hold()
        cands = [a for a in 1:n if r[a] < d[a]]
        isempty(cands) && break
        r[cands[argmin([r[c] / d[c] for c in cands])]] += 1
    end
    return r
end

"""Validate a user-supplied `sizes` override against the tensor's dimensions."""
function _qdn_check_sizes(sizes::AbstractVector{<:Integer}, dims::AbstractVector{<:Integer})
    length(sizes) == length(dims) ||
        error("QuickDer: sizes has $(length(sizes)) entries but the tensor has " *
              "$(length(dims)) axes.")
    all(a -> 1 <= sizes[a] <= dims[a], eachindex(sizes)) ||
        error("QuickDer: every restriction size must satisfy 1 <= r_a <= d_a; " *
              "got $(collect(sizes)) for dims $(collect(dims)).")
    return Int[Int(s) for s in sizes]
end

# ---------------------------------------------------------------------------
# The sketch
# ---------------------------------------------------------------------------

"""
Per-axis restriction data: `W` (d x r) and its orthogonal complement `Wp`
(d x (d-r)).  `ident` says `W = I[:, 1:r]` and `Wp = I[:, r+1:d]`, in which
case both are left empty and every contraction with them is a slice.  A
saturated axis (`r == d`) is always `ident`, so `Q = I` and its operator needs
no rotation back.

`W` and `Wp` are typed `AbstractMatrix` rather than `Matrix` so the same struct
can carry device copies for the `d^n` contractions (`_qdn_axes_device`) while
the host copies stay behind for `_qdn_assemble`, which builds the answer the
caller gets.  Both are `d x r`-ish and tiny; the type looseness costs a dynamic
dispatch per mode product, against a GEMM.
"""
struct _QDNAxis{T}
    d::Int
    r::Int
    ident::Bool
    W::AbstractMatrix{T}
    Wp::AbstractMatrix{T}
end

function _qdn_axis(::Type{T}, d::Integer, r::Integer, restriction::Symbol, rng) where {T}
    if restriction === :corner || r == d
        return _QDNAxis{T}(d, r, true, Matrix{T}(undef, d, 0), Matrix{T}(undef, d, 0))
    end
    # A full square QR gives W and W⊥ from one factorisation, and they are
    # exactly orthogonal to each other -- which the lift relies on when it
    # reassembles M_a = [Y_a Z_a] Qᵗ.  Formed on the HOST even for a device
    # run: Metal.jl has no `qr`, and this is `d x d` once per axis.
    Q = Matrix(qr(randn(rng, T, d, d)).Q)
    return _QDNAxis{T}(d, r, false, Q[:, 1:r], Q[:, (r + 1):d])
end

"""
    _qdn_axes_device(axs) -> Vector{_QDNAxis}

The same axes with `W`/`Wp` resident on the GPU, so that every mode product
against the full tensor is a device GEMM.  An `ident` axis is left alone: its
matrices are empty (`d x 0`) and never contracted -- the slice path is used
instead -- and a zero-length device buffer is a needless special case.
"""
_qdn_axes_device(axs::Vector{_QDNAxis{T}}) where {T} =
    _QDNAxis{T}[ax.ident ? ax : _QDNAxis{T}(ax.d, ax.r, false, to_gpu(ax.W), to_gpu(ax.Wp))
                for ax in axs]

_qdn_modeW(G::AbstractArray{T,N}, ax::_QDNAxis{T}, a::Integer) where {T,N} =
    ax.ident ? (ax.r == size(G, a) ? G : _qdn_slice(G, a, 1:ax.r)) :
               _qdn_ttm(G, ax.W, a)

_qdn_modeWp(G::AbstractArray{T,N}, ax::_QDNAxis{T}, a::Integer) where {T,N} =
    ax.ident ? _qdn_slice(G, a, (ax.r + 1):ax.d) : _qdn_ttm(G, ax.Wp, a)

"""
    _qdn_mode_cost(sz, ax, a, T) -> Float64

Seconds for the pass `_qdn_modeW` would make over an array of shape `sz` and
element type `T` when it applies the axis-`a` sketch, under the same model,
DIVIDED by the fraction of the data that pass removes -- a chain wants the axis
that buys the most shrinkage per second first, because every later pass is
charged the size this one leaves behind.

`Inf` for an axis that removes nothing (a saturated `ident` axis, whose mode
product is the identity), so that ordering by cost never puts one first.
"""
function _qdn_mode_cost(sz::NTuple{N,Int}, ax::_QDNAxis, a::Integer,
                        ::Type{T}) where {N,T}
    d = sz[a]
    ax.r >= d && return Inf                       # nothing to remove
    bytes = float(prod(sz)) * sizeof(T)
    stream_s = bytes / QDN_GPU_STREAM_BW
    # An `ident` axis is a slice, an edge axis is one GEMM: no copy either way.
    extra_s = (ax.ident || a == 1 || a == N) ? 0.0 :
              _qdn_slab_is_cheap(sz, a, ax.r, T) ?
              float(prod(ntuple(i -> sz[a + i], N - a))) * QDN_GPU_LAUNCH_S :
              bytes * (1 + ax.r / d) / QDN_GPU_PERMUTE_BW
    return (stream_s + extra_s) / (1 - ax.r / d)
end

"""
    _qdn_mode_order(G, axs, cand) -> Vector{Int}

The order in which to apply the axis sketches `cand` to `G`.  Mode products on
distinct axes COMMUTE, so this is free to choose -- and on the device the
choice is worth more than anything else in this file.

HOST ARRAYS KEEP THE NATURAL ORDER, and that is deliberate: on the host every
axis is one pass with no copy (`_qdn_ttm`'s middle-axis case runs strided GEMMs
BLAS takes directly), so there is nothing to win, and reordering would change
the rounding of every existing CPU result for no gain.

On the device there is one pass over the FULL tensor in each chain and the rest
run on something already small, so what this really decides is which single
axis meets `d^n`.  Greedy on `_qdn_mode_cost`, re-evaluated after each step
because contracting a trailing axis shrinks the `back` of the ones before it.
On a 640 x 480 x F x 3 movie the greedy answer is axis 1 first (an edge axis,
one GEMM) and the FRAME axis second (a middle axis, but with `back = 3`); the
cross sketch of axis 1, which cannot use axis 1, therefore meets the full
tensor on the frame axis rather than on the column axis, trading `3F` launches
for 3.
"""
function _qdn_mode_order(G::AbstractArray{T,N}, axs::Vector{_QDNAxis{T}},
                         cand) where {T,N}
    order = Int[c for c in cand]
    (G isa Array || length(order) <= 1) && return order
    # Only the SHAPE feeds the cost, so the chain is walked as a size tuple and
    # nothing is contracted twice.
    sz = size(G)
    picked = Int[]
    while !isempty(order)
        i = argmin([_qdn_mode_cost(sz, axs[c], c, T) for c in order])
        c = order[i]
        push!(picked, c)
        deleteat!(order, i)
        sz = ntuple(j -> j == c ? min(axs[c].r, sz[j]) : sz[j], N)
    end
    return picked
end

"""
    _qdn_assemble(ax, Y, Z) -> Matrix

`M_a` from its two halves: `M_a = [Y_a Z_a] Q_aᵗ = Y_a W_aᵗ + Z_a W_a⊥ᵗ`.
For the corner restriction `Q_a = I` and the two halves simply sit side by
side, which is why `:corner` never has to rotate anything back.
"""
function _qdn_assemble(ax::_QDNAxis{T}, Y::AbstractMatrix{T}, Z::AbstractMatrix{T}) where {T}
    ax.ident && return Matrix(hcat(Y, Z))
    return Y * transpose(ax.W) + Z * transpose(ax.Wp)
end

# ---------------------------------------------------------------------------
# Restricted solve + lift
# ---------------------------------------------------------------------------

"""
    _qdn_pair_tensor(G, axs, a, b) -> Array

`H_{ab} = Γ ×_a W_a⊥ ×_{c ∉ {a,b}} W_c`, the coefficient of `Y_b` in the lift
equation of axis `a`.  The small `W_c` go on FIRST and `W_a⊥` last: contracting
`Γ ×_a W_a⊥` up front would cost `O(d^{n+1})` on the full tensor, while this
order pays `O(r·d^n)` for the first pass and then works on a tensor that is
already `r` in every axis but two.

WHICH of the `W_c` goes first is `_qdn_mode_order`'s call on the device (the
natural order on the host), for the same reason: exactly one of these passes
meets the full tensor, so it is the one that has to be a copy-free GEMM.
"""
function _qdn_pair_tensor(G::AbstractArray{T,N}, axs::Vector{_QDNAxis{T}},
                          a::Integer, b::Integer)::AbstractArray{T,N} where {T,N}
    X = G
    for c in _qdn_mode_order(G, axs, (c for c in 1:N if c != a && c != b))
        X = _qdn_modeW(X, axs[c], c)
    end
    return _qdn_modeWp(X, axs[a], a)
end

"""
    _qdn_cross_sketches(G, axs, engaged) -> Dict{Int,Array}

`S_a = Γ ×_{b≠a} W_b` for every engaged axis `a`, built through a shared prefix
chain (`Γ ×_{π_1} W_{π_1} ⋯`) so the total cost stays about `2·r·d^n` rather
than the `n²/2` passes the definition suggests.  Disengaged axes carry no
unknown and so need no cross sketch -- but they are still sketched, i.e. their
`W_b` is applied inside every other axis's `S_a`.

`π` is `_qdn_mode_order`'s ordering: `1:N` on the host, and on the device the
one that keeps the full tensor away from a `permutedims`.  The prefix structure
does not care which order it is -- `pre[i] = Γ ×_{π_1} ⋯ ×_{π_{i-1}}` and
`S_{π_i} = pre[i] ×_{π_{i+1}} ⋯ ×_{π_N}` for ANY permutation, because mode
products on distinct axes commute -- and exactly two passes here meet `d^n`:
`pre[2]`, which applies `π_1`, and `S_{π_1}`, which applies `π_2`.  Ordering by
`_qdn_mode_cost` is therefore precisely the statement "let those two be the
cheapest passes available".
"""
function _qdn_cross_sketches(G::AbstractArray{T,N}, axs::Vector{_QDNAxis{T}},
                             engaged::AbstractVector{Bool}) where {T,N}
    ord = _qdn_mode_order(G, axs, 1:N)
    pre = Vector{AbstractArray{T,N}}(undef, N)
    pre[1] = G
    for i in 2:N
        pre[i] = _qdn_modeW(pre[i - 1], axs[ord[i - 1]], ord[i - 1])
    end
    S = Dict{Int, AbstractArray{T,N}}()
    for i in 1:N
        engaged[ord[i]] || continue
        X = pre[i]
        for j in (i + 1):N
            X = _qdn_modeW(X, axs[ord[j]], ord[j])
        end
        S[ord[i]] = X
    end
    return S
end

# ---------------------------------------------------------------------------
# Whitening the restriction -- QuickDer-W
# ---------------------------------------------------------------------------

"""
Per-axis whitening data for one engaged axis.

`rank` is the numerical rank kept, `Q` (`R_a x rank`, orthonormal columns) is
the whitened stand-in for `M_a`, `un` (`d_a x rank`) un-whitens a solved
`Ỹ_a` back to `Y_a`, and `K` (`d_a x (d_a - rank)`, orthonormal columns) spans
what was truncated: the left null space of the mode-`a` unfolding.

The invariant every field is built to satisfy is `M_a * un == Q` exactly (to
rounding), i.e. substituting `Ỹ_a = un⁺ Y_a` leaves the restricted operator
unchanged on the range and drops nothing else.
"""
struct _QDNWhite{T}
    rank::Int
    Q::Matrix{T}
    un::Matrix{T}
    K::Matrix{T}
end

"""
    _qdn_whiten_axis(Ma) -> _QDNWhite

Whiten one axis block of the restricted operator.

THE STRUCTURE THIS EXPLOITS.  In the convention of this file the axis-`a`
column block of the restricted system is, up to the row permutation
`_qdn_row_perm` bookkeeps and the chisel weight `P[ρ,a]`, exactly

    A_a = I_{r_a} ⊗ M_a,     M_a := _qdn_unfold(S_a, a)ᵗ   (R_a x d_a)

because `A_a vec(Y_a) = vec(M_a Y_a)` and the unknown `Y_a` is `d_a x r_a`
with `vec` stacking its COLUMNS (that is the ordering `_qdn_restricted_matrix`
writes and `_qdn_restricted_map` reads).  Hence the diagonal blocks of the
Gram of the whole restricted operator `A = [A_1 ⋯ A_n]` are Kronecker,

    A_aᵗ A_a = c_a · (I_{r_a} ⊗ M_aᵗ M_a),   c_a = Σ_ρ P[ρ,a]²,

so all of the sketch-induced conditioning lives in `d_a x d_a` Grams -- tiny --
while the off-diagonal blocks `A_aᵗ A_b`, which are what actually encodes the
derivation condition, are untouched by any per-axis change of variables.
Verified numerically to 5e-16 at valence 3 and 4, universal/centroid/adjoint
chisels (see docs/design/QuickDer-valence-n.md, "Whitened restriction").

NOTE the transposed convention against the design note: there `Y_a` is
`r_a x d_a` and the substitution reads `Ỹ_a = Y_a R_aᵗ`; here `Y_a` is
`d_a x r_a` and it reads `Ỹ_a = R_a Y_a`, one row of `R_a` per row of `Y_a`.

WHAT IT DOES.  Thin QR `M_a = Q_a R_a` and substitute `Ỹ_a = R_a Y_a`; then
`A_a Y_a = (I ⊗ Q_a) Ỹ_a`, every diagonal Gram block is exactly `c_a·I`, and
the whole Gram has spectrum in `[0, Σ_a c_a]` -- each block has norm ≤ 1.  The
genuine null space is unchanged, and what goes away is the conditioning of the
tensor's own mode unfoldings, which is what made Krylov crawl on smooth video
and on structured spheres.

DEGENERATE MODES.  When `M_a` is rank deficient -- the mode-`a` unfolding of Γ
has rank below `d_a`, which a smooth or separable tensor routinely does -- `R_a`
is singular and there is no substitution to make.  Truncate instead: keep the
range (an SVD, at the standard `min(size)·eps` relative cut, so nothing that is
numerically nonzero is ever dropped) and hand back the dropped directions in
`K`.  Those are the TRIVIAL derivations `X_a ·_a Γ = 0` that any degenerate
tensor has; `_qdn_solve_and_lift` re-injects them after the solve rather than
letting them inflate an iterative null solve, where they are a numerically-zero
eigenvalue cluster of dimension `(d_a - rank)·r_a`.

The QR comes first and the SVD only when it fails, because the QR is the cheap
case and it is the case that holds for every generic tensor: `n` QRs of
`(∏_{b≠a} r_b) x d_a`, e.g. 3200 x 1000 at d = 1000 valence 3.  The test is not
`min |diag(R_a)|` -- that is a heuristic on an unpivoted factorisation -- but
the invariant itself, `‖M_a·R_a⁻¹ - Q_a‖`, which costs one GEMM of the same
shape as the QR and cannot be fooled.
"""
function _qdn_whiten_axis(Ma::AbstractMatrix{T}) where {T}
    RT = real(T)
    Ra, da = size(Ma)
    Ra >= da || return _qdn_whiten_svd(Ma)      # wide: QR gives no d_a x d_a R
    F = qr(Ma)
    Rf = Matrix(F.R)
    dg = abs.(diag(Rf))
    if minimum(dg) > min(Ra, da) * eps(RT) * maximum(dg)
        Q = Matrix(F.Q)
        un = UpperTriangular(Rf) \ Matrix{T}(LinearAlgebra.I, da, da)
        # The invariant, not the diagonal heuristic: an unpivoted R can have a
        # healthy diagonal and still lose the substitution to cancellation.
        if all(isfinite, un) &&
           norm(Ma * un .- Q) <= sqrt(eps(RT)) * max(norm(Q), one(RT))
            return _QDNWhite{T}(da, Q, un, Matrix{T}(undef, da, 0))
        end
    end
    return _qdn_whiten_svd(Ma)
end

function _qdn_whiten_svd(Ma::AbstractMatrix{T}) where {T}
    RT = real(T)
    Ra, da = size(Ma)
    # `full` only when the matrix is WIDE: the thin `V` is already `d_a x d_a`
    # for `R_a >= d_a` (the shape condition (ii) guarantees), and asking for a
    # full `U` there would materialise `R_a x R_a` -- 81 MB at d = 1000 -- for
    # columns nothing reads.
    F = svd(Ma; full = Ra < da)
    s = F.S
    cut = minimum((Ra, da)) * eps(RT) * (isempty(s) ? zero(RT) : s[1])
    k = count(>(cut), s)
    Q = Matrix(F.U[:, 1:k])
    un = F.V[:, 1:k] * Diagonal(one(T) ./ s[1:k])
    K = Matrix(F.V[:, (k + 1):da])
    return _QDNWhite{T}(k, Q, un, K)
end

"""
    _qdn_whiten(Uf, eaxes, dims, r, N) -> (wh, Us, Ss, wdims)

Whiten every engaged axis and repackage the restriction so that the two
solve branches need no change at all:

- `Us[a] = Q_aᵗ` (`rank_a x R_a`) replaces `Uf[a]` (`d_a x R_a`);
- `Ss[a]` is `Us[a]` folded back into a tensor of shape
  `(r_1 .. rank_a .. r_n)`, which is the whitened cross sketch the matrix-free
  map contracts;
- `wdims[a] = rank_a` replaces `dims[a]` in the column bookkeeping.

`_qdn_restricted_matrix` and `_qdn_restricted_map` only ever use `dims[a]` as
"the length of the axis-`a` unknown", so passing `wdims` is the whole of the
plumbing.
"""
function _qdn_whiten(Uf::Dict{Int, Matrix{T}}, eaxes::Vector{Int}, dims::Vector{Int},
                     r::Vector{Int}, N::Integer) where {T}
    wh = Dict{Int, _QDNWhite{T}}()
    Us = Dict{Int, Matrix{T}}()
    Ss = Dict{Int, Array{T,N}}()
    wdims = copy(dims)
    for a in eaxes
        w = _qdn_whiten_axis(Matrix(transpose(Uf[a])))
        wh[a] = w
        Us[a] = Matrix(transpose(w.Q))
        wdims[a] = w.rank
        Ss[a] = _qdn_fold(Us[a], a, ntuple(b -> b == a ? w.rank : r[b], N))
    end
    return (wh, Us, Ss, wdims)
end

"""
    _qdn_trivial_ders(G, haxs, wh, engaged, eaxes, dims, r, atol) -> Vector

The derivations the whitening truncation removed from the restricted solve,
back as full operator tuples.

A rank-deficient mode-`a` unfolding means there are `X_a` with `Γ ×_a X_a = 0`,
and `(0,…,X_a,…,0)` satisfies the derivation equation for every chisel.  The
condition on `X_a` is that its COLUMNS lie in the left null space of the mode-`a`
unfolding, which is exactly `span(wh[a].K)`, so the trivial space of that axis
is `(d_a - rank)·d_a`-dimensional with basis `K[:,p]·e_qᵗ`, and that whole basis
is what goes back.

WHY THE WHOLE OF IT, rather than the part the restricted solve would have
found.  The unwhitened solve sees these directions as `Y_a` with columns in
`ker(M_a)`, i.e. `X_a = Y_a W_aᵗ` -- only `(d_a-rank)·r_a` of them -- and its
lift, a least-squares solve against a rank-deficient operator, returns the
minimum-norm `Z_a`, which has no `K` component at all.  So the unwhitened
branch reports an arbitrary, `W_a`-dependent SLICE of the trivial space and
silently drops the rest: measured on a 12x12x12 tensor with a mode-1 rank of
10, the true derivation space is 26-dimensional and the unwhitened branch
returns 16 of it.  Writing the trivial space down exactly costs nothing here --
`K` is already computed -- and it makes the whitened answer a superset, never a
different subspace.  Degenerate modes are meant to be removed upstream by
`nondeg` anyway; this is the safety net, not the normal path.

Each candidate direction is checked against the tensor itself
(`‖Γ ×_a K[:,p]‖ ≤ atol·‖Γ‖`, the same bound `_qdn_verify` applies) rather than
trusted: `ker(M_a)` contains the left null space of the unfolding and is equal
to it only when the sketch preserved the rank.  The check costs one pass over Γ
with a matrix of `d_a - rank` columns, and is only ever reached on a tensor
that is actually degenerate.

WHAT IS RETURNED AND WHAT IS PUBLISHED.  The complete answer is FACTORED and
goes to `QDN_TRIVIAL_FACTORED`: the verified kernel columns per axis, from
which every element of the space is `K v e_qᵗ`.  What this function RETURNS is
that space written out as operator tuples, and only as much of it as
`QDN_TRIVIAL_MAX_BYTES` allows -- `n` dense `d_b x d_b` matrices per basis
element is 6 MB at d = 500, so a single degenerate mode there asks for 3 GB of
zeros to describe a 2 MB matrix. Truncating warns and names the field. Nothing
is lost that was not already unrepresentable: the caller's `ders` is a dense
`globalDim(Ω) x k` matrix, which at d = 500 is 6 MB per column of its own.
"""
function _qdn_trivial_ders(G::AbstractArray{T,N}, wh::Dict{Int, _QDNWhite{T}},
                           eaxes::Vector{Int}, dims::Vector{Int},
                           atol::Real) where {T,N}
    out = Vector{Vector{Matrix{T}}}()
    QDN_TRIVIAL_FACTORED[] = NamedTuple[]
    any(a -> wh[a].rank < dims[a], eaxes) || return out
    gnorm = norm(G)
    bound = real(T)(atol) * max(gnorm, eps(real(T)))
    factored = NamedTuple[]
    for a in eaxes
        K = wh[a].K
        size(K, 2) == 0 && continue
        E = _qdn_ttm(G, _qdn_upload(G, K), a)
        keep = [p for p in 1:size(K, 2) if norm(_qdn_slice(E, a, p:p)) <= bound]
        length(keep) == size(K, 2) || @warn "QuickDer(whiten): $(size(K,2) - length(keep)) " *
            "of $(size(K,2)) truncated directions on axis $a are not trivial derivations " *
            "-- the cross sketch lost rank there. Retry with a larger `sizes`." maxlog = 1
        isempty(keep) && continue
        push!(factored, (; axis = Int(a), K = K[:, keep]))
    end
    QDN_TRIVIAL_FACTORED[] = factored
    isempty(factored) && return out

    total = sum(f -> size(f.K, 2) * dims[f.axis], factored)
    per = float(sum(db -> db^2, dims)) * sizeof(T)         # one tuple, all axes
    cap = max(1, floor(Int, QDN_TRIVIAL_MAX_BYTES[] / max(per, 1.0)))
    for f in factored, p in axes(f.K, 2), q in 1:dims[f.axis]
        length(out) >= cap && break
        Ms = Vector{Matrix{T}}(undef, N)
        for b in 1:N
            Ms[b] = zeros(T, dims[b], dims[b])
        end
        Ms[f.axis][:, q] = view(f.K, :, p)
        push!(out, Ms)
    end
    @info "QuickDer(whiten): Γ has a rank-deficient mode " *
        "(ranks $([wh[a].rank for a in eaxes]) of dims $(dims[eaxes])); its " *
        "$(total) trivial derivations (X_a ·_a Γ = 0) were kept out of the " *
        "restricted solve, where they are a numerically-zero eigenvalue cluster, and " *
        "written down directly instead." maxlog = 1
    length(out) < total && @warn "QuickDer(whiten): the trivial space of the " *
        "rank-deficient mode is $(total)-dimensional and $(length(out)) of it was " *
        "written out as operator tuples ($(round(per / 2^20; sigdigits = 3)) MB each, " *
        "budget QDN_TRIVIAL_MAX_BYTES = " *
        "$(round(QDN_TRIVIAL_MAX_BYTES[] / 2^20; sigdigits = 3)) MB). The COMPLETE " *
        "space is in Dleto.QDN_TRIVIAL_FACTORED[] as (; axis, K) per axis, meaning " *
        "X_axis = K*v*e_qᵗ for every v and q." maxlog = 1
    return out
end

"""
    _qdn_system_rows(rows, ncols) -> Int

How many rows the restricted system is presented with: `max(rows, ncols)`.

A null space is only reachable through `SVDSolver` if the thin SVD's `V` spans
the whole column space, and for a WIDE matrix (`rows < ncols`) it does not --
`svd` returns `min(rows, ncols)` right singular vectors, so a system with more
unknowns than equations silently reports at most `rows` null directions
instead of the `ncols - rank` it has.  Padding with zero rows costs nothing,
changes no null space, and makes the wide case correct.

The sizes `_qdn_restriction_sizes` picks are never wide -- that is exactly its
condition (i).  The padding is for the two cases that escape it: an explicit
`sizes` override, and a tensor small enough that every axis saturates and the
conditions cannot be met at all (a 3x3 matrix has 9 equations and 18
unknowns).
"""
_qdn_system_rows(rows::Integer, ncols::Integer) = max(Int(rows), Int(ncols))

"""
    _qdn_restricted_matrix(Uf, P, eaxes, r, dims, coff, ncols) -> Matrix

The restricted system as a dense matrix, built directly.

For axis `a` the block is `kron(I_{r_a}, S_a(a)ᵗ)` with its rows permuted from
the axis-last order `(i_{-a}, i_a)` into the column-major order `(i_1..i_n)`
the assembled system uses (`_qdn_row_perm`).  `kron(I, ·)` is block diagonal,
so the blocks are written straight into place rather than materialised.

Never via `Matrix(::FunctionMap)`: that would apply the map once per column,
i.e. `∏r·m` tensor contractions to build something this fills with `n·m·r_a`
copies of an already-computed unfolding.

The matrix is padded with zero rows to at least `ncols` (see
`_qdn_system_rows`), which never happens at the sizes `_qdn_restriction_sizes`
picks and matters when an explicit `sizes` override, or a tensor small enough
that every axis saturates, leaves the system WIDE.
"""
function _qdn_restricted_matrix(Uf::Dict{Int, Matrix{T}}, P::Matrix{T},
                                eaxes::Vector{Int}, r::Vector{Int},
                                dims::Vector{Int}, coff::Dict{Int, Int},
                                ncols::Integer) where {T}
    m = size(P, 1)
    R = prod(r)
    M = zeros(T, _qdn_system_rows(m * R, ncols), ncols)
    for a in eaxes
        Ua = Matrix(transpose(Uf[a]))          # R_a x d_a
        Ra = size(Ua, 1)
        perm = _qdn_row_perm(r, a)
        for rho in 1:m
            c = P[rho, a]
            iszero(c) && continue
            base = (rho - 1) * R
            for j in 1:r[a]
                rows = view(perm, ((j - 1) * Ra + 1):(j * Ra))
                cols = (coff[a] + (j - 1) * dims[a] + 1):(coff[a] + j * dims[a])
                @views M[base .+ rows, cols] .= c .* Ua
            end
        end
    end
    return M
end

"""
    _qdn_restricted_map(S, Uf, P, eaxes, r, dims, coff, ncols) -> LinearMap

The same system, matrix free.

Forward: `y ↦ vec(Σ_a P[ρ,a] (S_a ×_a Y_a))`, `n` mode products per apply.
Adjoint: `E ↦ (Σ_ρ P[ρ,a] · S_a(a) · E_ρ(a)ᵗ)_a`, which is the genuine adjoint,
not a stand-in -- `⟨L y, e⟩ = Σ_ρ P[ρ,a] ⟨Y_aᵗ S_a(a), E_ρ(a)⟩ = ⟨y, Lᵗ e⟩`.
Every solver in `NullSolvers.jl` that never densifies relies on that identity.
"""
function _qdn_restricted_map(S::Dict{Int, Array{T,N}}, Uf::Dict{Int, Matrix{T}},
                             P::Matrix{T}, eaxes::Vector{Int}, r::Vector{Int},
                             dims::Vector{Int}, coff::Dict{Int, Int},
                             ncols::Integer) where {T,N}
    m = size(P, 1)
    R = prod(r)
    rt = ntuple(a -> r[a], N)
    nrows = _qdn_system_rows(m * R, ncols)      # zero-padded when wide

    # SCRATCH, allocated once and reused by every apply.  This is a memory
    # change and it is a large one: written with fresh temporaries the pair of
    # applies allocated about 13 MB at d = 700 valence 3, so a 100k-apply
    # solve churned ~700 GB and the process RSS tracked the GC's heap target
    # rather than the live set (measured: 17 GB at d = 700 for a live set
    # under 8).  Nothing here is ever returned -- the two vectors that ARE
    # (`out` and `y`) stay freshly allocated -- and every null solver in this
    # package applies the map one vector at a time, ARPACK, LOBPCG and block
    # Lanczos alike, so there is no apply running concurrently with another.
    # (That is the invariant to keep if a threaded solver is ever added.)
    Yb = Dict{Int, Matrix{T}}(a => Matrix{T}(undef, dims[a], r[a]) for a in eaxes)
    Ab = Dict{Int, Matrix{T}}(a => Matrix{T}(undef, dims[a], r[a]) for a in eaxes)
    # One mode product's output is `rt` whichever axis it came from, so one
    # buffer serves them all.
    Tb = Array{T}(undef, rt)
    Eb = Array{T}(undef, (rt..., m))
    Sb = Array{T}(undef, rt)
    Pb = Dict{Int, Array{T,N}}(
        a => Array{T}(undef, ntuple(i -> rt[_qdn_front(N, a)[i]], N)) for a in eaxes)

    function fwd(y::AbstractVector)
        _qdn_tick!()
        out = zeros(T, nrows)
        Out = reshape(view(out, 1:(R * m)), rt..., m)
        for a in eaxes
            Ya = Yb[a]
            @views Ya .= reshape(y[(coff[a] + 1):(coff[a] + dims[a] * r[a])],
                                 dims[a], r[a])
            Ta = _qdn_ttm!(Tb, S[a], Ya, a)
            for rho in 1:m
                c = P[rho, a]
                iszero(c) && continue
                sl = selectdim(Out, N + 1, rho)
                sl .+= c .* Ta
            end
        end
        return out
    end

    function adj(e::AbstractVector)
        _qdn_tick!()
        @views Eb .= reshape(e[1:(R * m)], rt..., m)
        y = zeros(T, ncols)
        for a in eaxes
            acc = fill!(Ab[a], zero(T))
            for rho in 1:m
                c = P[rho, a]
                iszero(c) && continue
                copyto!(Sb, selectdim(Eb, N + 1, rho))
                Eu = _qdn_unfold!(Pb[a], Sb, a)                        # r_a x R_a
                # 5-argument `mul!` accumulates `c * Uf[a] * Euᵗ` straight into
                # `acc`, where the loop used to build the product and then add
                # it.
                mul!(acc, Uf[a], transpose(Eu), c, one(T))
            end
            @views y[(coff[a] + 1):(coff[a] + dims[a] * r[a])] .= vec(acc)
        end
        return y
    end

    return LinearMaps.LinearMap{T}(fwd, adj, nrows, ncols; ismutating = false)
end

"""
    _qdn_solve_and_lift(G, P, engaged, r, method, rng, atol, progress, store)
        -> Vector{Vector{Matrix}} or nothing

One attempt at the whole kernel: sketch, restricted solve, lift, consistency
filter.  Returns the surviving derivations as one matrix per axis (zero on the
disengaged axes, which carry no unknown), an EMPTY vector when the restricted
system already has no null space -- a legitimate answer, "Γ conforms to no
pattern for this chisel" -- or `nothing` when there were restricted solutions
but the consistency filter rejected every one of them, which is the caller's
cue to retry with a larger `r`.

WHERE THE WORK RUNS.  `G` arrives on whichever device the caller put it on
(`method.device`).  Everything that touches it -- the cross sketches and the
pair tensors of the lift -- runs there, against the device copies of `W`/`W⊥`.
Everything downstream of the sketches is small (`S_a` is `r^{n-1} x d_a`, a few
MB even at d = 300) and is pulled back to the host, because that is where the
rest of the linear algebra has to happen anyway: the restricted matrix is a
memory-bound scatter of `kron(I, S_a(a)ᵗ)`, the lift is a QR, the consistency
filter is a `nullspace`, and the answer has to be host matrices for
`_fastder_restrict_to_ops`.  The one big piece that does go to the device is
the restricted solve, through `GramSolver(device = :gpu)`.

`store` is the type the TENSOR WAS HANDED IN AS, which is not `T` here: `G`
arrives already promoted to `compute_eltype`, so `T` is the arithmetic's type
and `real(T)` would tell `solve_nullspace` that a Float16 tensor resolves to
`eps(Float32)`.  It is passed explicitly, and positionally, because the wrong
answer to this question is silent: `data_floor(Float32) = 1.2e-7` never binds,
so the verdict on a Float16 tensor certified a cut whose first value above it
sat *inside* the rounding of the input (measured on a 40x40x40 Float16 luma
block: `above = [4.21e-4, ...]` against `eps(Float16) = 9.8e-4`, reported
`certified = true`).  See `Dleto.data_floor` and `NullVerdict`'s `undecidable`.
"""
function _qdn_solve_and_lift(G::AbstractArray{T,N}, P::Matrix{T}, engaged::Vector{Bool},
                             r::Vector{Int}, method::QuickDerMethod, rng,
                             atol::Real, progress, store::Type) where {T,N}
    dims = collect(size(G))
    m = size(P, 1)
    eaxes = [a for a in 1:N if engaged[a]]
    on_gpu = method.device === :gpu
    haxs = [_qdn_axis(T, dims[a], r[a], method.restriction, rng) for a in 1:N]
    # Host axes build the answer (`_qdn_assemble`); device axes do the d^n work.
    axs = on_gpu ? _qdn_axes_device(haxs) : haxs

    tstage = time()
    S = _qdn_cross_sketches(G, axs, engaged)
    # d_a x R_a, and small: unfold on the device, then keep the host copy that
    # the restricted matrix, the lift operator and the adjoint all need.  This
    # is also the synchronisation point of the sketch stage on a device run.
    Uf = Dict{Int, Matrix{T}}(a => _qdn_host(_qdn_unfold(S[a], a)) for a in eaxes)
    tstage = _qdn_stage!(:sketch, tstage)

    # ---- whitening (QuickDer-W).  `Us`/`Ss`/`sdims` are what the two solve
    # branches see; `wh` is what un-whitens the answer afterwards.  Unwhitened,
    # they are the sketch data itself, so there is one code path below.
    wh = method.whiten ? _qdn_whiten(Uf, eaxes, dims, r, N) : nothing
    Us    = wh === nothing ? Uf   : wh[2]
    sdims = wh === nothing ? dims : wh[4]
    if wh !== nothing
        @debug "QuickDer whitening" ranks = [wh[1][a].rank for a in eaxes] dims = dims[eaxes]
        tstage = _qdn_stage!(:whiten, tstage)
    end

    coff = Dict{Int, Int}()
    ncols = 0
    for a in eaxes
        coff[a] = ncols
        ncols += sdims[a] * r[a]
    end
    R = prod(r)

    # Dense while the matrix is a quarter of the null-solver's own budget --
    # the SVD needs the matrix plus its factors, so filling the whole budget
    # with the matrix alone would leave nothing for the solve.  The row count is
    # `m·∏r` except in the padded wide case (`_qdn_system_rows`), and it is the
    # count actually allocated that has to fit.  The GPU route gets its own,
    # larger budget: the Gram and the Cholesky live on a device with a 52 GB
    # working set, so the binding constraint there is the HOST copy, not the
    # solve (`QDN_GPU_DENSE_BUDGET_BYTES`).
    dense_bytes = float(_qdn_system_rows(m * R, ncols)) * ncols * sizeof(T)
    dense_budget = on_gpu ? QDN_GPU_DENSE_BUDGET_BYTES[] : QDN_DENSE_BUDGET_BYTES[]
    if dense_bytes <= dense_budget
        Mres = _qdn_restricted_matrix(Us, P, eaxes, r, sdims, coff, ncols)
        tstage = _qdn_stage!(:restricted, tstage)
        # The SVD is O(m n²) and at d = 100 (6859 x 5700) already 52 s; the
        # Gram route (`GramSolver`, n x n, Cholesky-shifted subspace
        # iteration) is 3.8 s there at half the precision, which the lift's
        # consistency filter and the Z-law check can afford.  Small systems
        # keep the SVD's full precision for free.  `GramSolver` is the one
        # solver that carries the device through: it forms the Gram, the
        # Cholesky and the subspace solves on the GPU when asked.
        dsolver = ncols >= QDN_GRAM_MIN_COLS[] ?
                  GramSolver(device = on_gpu ? :gpu : :cpu) : SVDSolver()
        (vals, vecs, verdict) = solve_nullspace(LinearMaps.LinearMap(Mres), dsolver;
                                                tol = atol, nd = -1, progress = progress,
                                                seed = method.seed,
                                                store_eltype = store,
                                                label = "quickder restricted")
        tstage = _qdn_stage!(:solve, tstage)
    else
        # Matrix free stays on the HOST whatever `device` says: the null
        # solvers here iterate with host vectors, and one apply is
        # `∏r · Σd` flops on sketches of a few MB -- the measured cost at
        # d = 100 is 0.12 ms per apply, i.e. the iteration count is the whole
        # story (see the native-core-plan entry on the night board), and a
        # device round trip per apply would only add latency.
        Sh = wh === nothing ?
             Dict{Int, Array{T,N}}(a => _qdn_host(S[a]) for a in eaxes) : wh[3]
        L = _qdn_restricted_map(Sh, Us, P, eaxes, r, sdims, coff, ncols)
        fsolver = method.solver === :AutoSolver ? _qdn_default_free_solver() :
                                                  method.solver
        tstage = _qdn_stage!(:restricted, tstage)
        # `method.seed` is the run's one source of randomness -- the sketch and
        # the verification slices already come from it -- and it now also fixes
        # the null solver's random start.  Without that, ARPACK's start vector
        # comes from a seed kept inside the Fortran library across calls, so
        # two identical calls in one process could return DIFFERENT restricted
        # null spaces (2, 3, 3 measured at d = 48 in Float32).
        (vals, vecs, verdict) = solve_nullspace(L, fsolver;
                                                tol = atol, nd = -1, progress = progress,
                                                seed = method.seed,
                                                store_eltype = store,
                                                label = "quickder restricted")
        tstage = _qdn_stage!(:solve, tstage)
    end

    QDN_LAST_SOLVE_STATUS[] = verdict.status

    # THE ONE UNDETECTABLE FAILURE MODE, closed.  A null solver that did not
    # converge and returned nothing is reporting "I failed", and on the
    # spectrum alone that is indistinguishable from the legitimate answer
    # "Γ conforms to no pattern for this chisel" -- both are "no values near
    # zero".  `solve_nullspace` now carries the solver's own word as
    # `verdict.status`, and here it decides: decline, so that `derTrOpsReduced`
    # raises and `AutoDerMethod` falls back to SylverLining, which is exact.
    #
    # A NONZERO count from a non-converged solve is kept instead of declined:
    # those vectors are then judged by the lift's consistency filter and by the
    # Z-law check in `_qdn_verify`, which test the answer itself and are a
    # stronger statement than any convergence flag.
    if size(vecs, 2) == 0 && verdict.status !== :ok
        error("QuickDer: the restricted null solve reported :$(verdict.status) and " *
              "found no null directions in the $(_qdn_system_rows(m * R, ncols)) x " *
              "$(ncols) restricted system. That is a FAILED solve, not an empty " *
              "derivation space; smallest values seen (relative to the operator " *
              "norm): $(verdict.above). Try `solver = :ArpackSolver` explicitly, a " *
              "larger `sizes`, or fall back to :SylverLining.")
    end

    # The trivial derivations the whitening truncation kept out of the solve.
    # Appended to whatever the solve returns, INCLUDING an empty answer, so
    # that whitened and unwhitened runs report the same space.
    triv = wh === nothing ? Vector{Vector{Matrix{T}}}() :
           _qdn_trivial_ders(G, wh[1], eaxes, dims, atol)

    k = size(vecs, 2)
    @debug "QuickDer restricted solve" rows = m * R cols = ncols nullity = k certified = verdict.certified rule = verdict.rule below = string(verdict.below) above = string(verdict.above) data_floor = verdict.data_floor undecidable = verdict.undecidable rss_GB = Sys.maxrss() / 2^30
    k == 0 && return triv

    # Un-whiten: the solver worked in `Ỹ_a = R_a Y_a`, the lift and the answer
    # want `Y_a` (`d_a x r_a`).  `wh[1][a].un` is `R_a⁻¹`, or its pseudo-inverse
    # on the kept range when the mode was degenerate.
    Yv = Matrix{Matrix{T}}(undef, N, k)
    for a in eaxes, i in 1:k
        Yt = reshape(T.(vecs[(coff[a] + 1):(coff[a] + sdims[a] * r[a]), i]),
                     sdims[a], r[a])
        Yv[a, i] = wh === nothing ? Yt : wh[1][a].un * Yt
    end

    # ---- the lift, one thin QR per axis, shared by every basis vector
    lift = [a for a in eaxes if r[a] < dims[a]]
    Zv = Matrix{Matrix{T}}(undef, N, k)
    Rblocks = Matrix{T}[]
    scale = zero(real(T))
    # `tlift` chains the per-axis substages `:lift1`, `:lift2`, ...; `:lift`
    # below still carries the wall total (its `t0` is untouched), so only its
    # BYTE entry becomes the remainder after the substages took their share.
    tlift = tstage
    for a in lift
        Ua = Matrix(transpose(Uf[a]))                       # R_a x d_a
        Ra = size(Ua, 1)
        da = dims[a]
        ha = da - r[a]
        A = vcat([P[rho, a] .* Ua for rho in 1:m]...)       # (m R_a) x d_a
        # The pair tensors are the last pass over the full tensor, so they are
        # built on the device; the results are `r^{n-2} x d_b x (d_a - r_a)`
        # (a few MB even at d = 300) and everything downstream of them -- the
        # right-hand sides, the QR, the residual filter -- is host work, so
        # they come back here.
        Hs = Dict{Int, Array{T,N}}(b => _qdn_host(_qdn_pair_tensor(G, axs, a, b))
                                   for b in eaxes if b != a)

        B = zeros(T, m * Ra, k * ha)
        for i in 1:k, rho in 1:m
            acc = zeros(T, Ra, ha)
            for b in eaxes
                (b == a || iszero(P[rho, b])) && continue
                Wt = _qdn_ttm(Hs[b], Yv[b, i], b)
                acc .-= P[rho, b] .* transpose(_qdn_unfold(Wt, a))
            end
            B[((rho - 1) * Ra + 1):(rho * Ra), ((i - 1) * ha + 1):(i * ha)] = acc
        end

        # One thin QR per axis, shared by every basis vector.  Condition (ii)
        # of the design note says A has full column rank -- generically.  On a
        # tensor where it does not (the unscrambled sphere under the CORNER
        # restriction from d = 16 up: the corner block of a support hyperplane
        # is too thin to determine the lift) the triangular solve raises
        # `SingularException` instead of saying anything useful.  Fall back to
        # the minimum-norm solution there: it is still a genuine derivation
        # whenever the system is CONSISTENT, and the residual filter below is
        # exactly the test of that -- so a recoverable case is recovered and an
        # unrecoverable one is filtered out and reported by `derTrOpsReduced`.
        Z = if size(A, 1) < size(A, 2)
            # Condition (ii) fails outright -- only reachable through an
            # explicit `sizes` override that is too small.  Least squares still
            # returns the best available Z; the filter reports the damage.
            pinv(A) * B
        else
            try
                qr(A) \ B
            catch err
                (err isa LinearAlgebra.SingularException ||
                 err isa LinearAlgebra.LAPACKException) || rethrow()
                pinv(A) * B
            end
        end
        all(isfinite, Z) || (Z = pinv(A) * B)
        AZ = A * Z
        for i in 1:k
            Zv[a, i] = Z[:, ((i - 1) * ha + 1):(i * ha)]
        end
        scale = max(scale, norm(B), norm(AZ))

        # The lift residual is LINEAR in y, so its columns (one per restricted
        # basis vector) are what the consistency filter takes a null space of.
        # A thin QR compresses each axis block to k x k first, which is exact
        # -- an orthonormal Q does not change singular values -- and keeps the
        # peak memory at one axis block instead of all of them.
        #
        # `AZ .-= B`, not `AZ .- B`: `B` is dead after `scale` above, and these
        # are the largest arrays in the whole method -- `(m·R_a) x (k·h_a)`,
        # 1.5 GB each at d = 1000 valence 3 with a restricted nullity in the
        # tens -- so a third live copy of one is worth more than the line.
        AZ .-= B
        Rm = reshape(AZ, (m * Ra) * ha, k)
        push!(Rblocks, size(Rm, 1) >= k ? Matrix(qr(Rm).R) : Matrix(Rm))
        tlift = _qdn_stage!(Symbol("lift", a), tlift)
    end

    tstage = _qdn_stage!(:lift, tstage)

    # ---- consistency filter
    #
    # A restricted solution that is not the restriction of a true derivation
    # leaves a residual in the lift equations.  Filtering those combinations
    # out (rather than erroring, as the valence-3 reference does) is what makes
    # the method correct and not merely generic: on a tensor whose restricted
    # null space is legitimately too large, the spurious directions are removed
    # and the genuine ones survive.
    local C::Matrix{T}
    if isempty(Rblocks)
        C = Matrix{T}(LinearAlgebra.I, k, k)                # nothing to lift
    else
        RT = real(T)
        sc = max(scale, eps(RT))
        Rall = vcat(Rblocks...) ./ sc
        # `svd` gives a complete `V` only for a tall matrix; the per-axis QR
        # above already compresses each block to `k x k`, and the padding
        # covers the one case it cannot (a block with fewer than `k` rows).
        size(Rall, 1) >= k || (Rall = vcat(Rall, zeros(T, k - size(Rall, 1), k)))
        F = svd(Rall)
        # A GAP, not a threshold -- see `QDN_LIFT_GAP_RATIO`.  The spectrum is
        # already relative to `sc`, the size of the lift equations themselves,
        # so these are the numbers the cut has to be judged on; ascending,
        # because that is the order `gap_verdict` reads.
        asc = reverse(Float64.(F.S))
        # THE FLOOR IS `sqrt(eps)`, and that is the whole point of the change.
        # A lift residual does not bottom out at rounding, it bottoms out at
        # the accuracy of the triangular solve that produced `Z`, which is
        # `sqrt(eps(T))` -- measured 1.3 to 2.9 times it in Float32 -- and that
        # is exactly why `sqrt(eps)` was the old CUTOFF.  As a cutoff it was on
        # the wrong side of the answer; as a FLOOR it says "everything at or
        # under the lift's own noise is zero", which flattens the ratios inside
        # the genuine cluster so the largest jump is the one that matters.  Use
        # `100*eps` here instead (the floor a null SOLVER's spectrum wants) and
        # the genuine cluster spans decades above the floor, its internal
        # ratios can clear `gap_ratio`, and the cut lands inside it: measured,
        # that undercounts the raw sphere at 11 of 13 in Float64.
        #
        # TODO(precision): `rank_rtol(T, m, n)` in Precision.jl is `max(m,n) *
        # eps(compute_eltype(T))` -- a dimension-scaled bound for a
        # rank-revealing factorization, not the dimension-free `sqrt(eps(T))`
        # this floor is (measured at 1.3-2.9x `sqrt(eps(Float32))`, the
        # accuracy of the triangular solve that produced `Z`).  Not an
        # obvious substitution; left as the measured constant.
        floor_rel = Float64(sqrt(eps(RT)))
        ceil_rel = max(Float64(atol), QDN_LIFT_CEILING[] * sqrt(eps(RT)))
        (_, lv) = gap_verdict(asc, 1.0; threshold = ceil_rel, floor = floor_rel,
                              gap_ratio = QDN_LIFT_GAP_RATIO[])
        cut = lv.nullity
        @debug "QuickDer lift residual spectrum" k svals = string(round.(asc; sigdigits = 3)) cut rule = lv.rule certified = lv.certified gap = lv.gap atol ceil_rel
        cut == 0 && return nothing
        C = Matrix(F.V[:, (k - cut + 1):k])
    end

    tstage = _qdn_stage!(:filter, tstage)

    kc = size(C, 2)
    out = Vector{Vector{Matrix{T}}}(undef, kc)
    for j in 1:kc
        Ms = Vector{Matrix{T}}(undef, N)
        for a in 1:N
            if !engaged[a]
                # No unknown on a disengaged axis: its chisel column is zero, so
                # every matrix satisfies the equation there and zero is the
                # representative `SylverLining` returns after its engagement
                # reduction expands.
                Ms[a] = zeros(T, dims[a], dims[a])
                continue
            end
            Yc = zeros(T, dims[a], r[a])
            for i in 1:k
                Yc .+= C[i, j] .* Yv[a, i]
            end
            if r[a] == dims[a]
                Ms[a] = _qdn_assemble(haxs[a], Yc, zeros(T, dims[a], 0))
            else
                Zc = zeros(T, dims[a], dims[a] - r[a])
                for i in 1:k
                    Zc .+= C[i, j] .* Zv[a, i]
                end
                Ms[a] = _qdn_assemble(haxs[a], Yc, Zc)
            end
        end
        out[j] = Ms
    end
    return isempty(triv) ? out : vcat(out, triv)
end

# ---------------------------------------------------------------------------
# Verification -- the Z-law, always
# ---------------------------------------------------------------------------

"""
    _qdn_verify(G, P, engaged, mats, method, atol, r, rng)

Check the DEFINING equation, because solve-and-lift is only generically correct
at a given `r` and the restricted system cannot tell you when its genericity
assumption failed.

`:full` evaluates `Σ_a P[ρ,a] (Γ ×_a M_a)` outright and is used when
`prod(dims) <= 2e7`; otherwise (and by default) `:random` checks `nslices`
output slices along the largest engaged axis `â`.  For the `a ≠ â` terms the
slice is taken from Γ FIRST, so those contractions run on a tensor that is
`nslices` deep rather than `d_â`; only the `a = â` term needs a pass over the
whole tensor, and it needs only the selected columns of `M_â`.

The residual is measured against `‖Γ‖·Σ_a‖M_a‖`, the size of the data the
equation is built from, so the test is invariant to how Γ and the basis happen
to be scaled.

Runs wherever `G` lives.  On a device run the answer's matrices are host
arrays (that is what the caller gets), so they are uploaded once here; the
accumulator, the mode products and the `norm` are all device operations, and
only the two scalars per check come back.
"""
function _qdn_verify(G::AbstractArray{T,N}, P::Matrix{T}, engaged::Vector{Bool},
                     mats::Vector{Vector{Matrix{T}}}, method::QuickDerMethod,
                     atol::Real, r::Vector{Int}, rng) where {T,N}
    (method.verify === :none || isempty(mats)) && return nothing
    dims = collect(size(G))
    m = size(P, 1)
    gnorm = norm(G)
    RT = real(T)
    up = method.device === :gpu ? to_gpu : identity

    fail(res, bound) = error(
        "QuickDer: the lifted solution does not satisfy the derivation equation " *
        "(relative residual $(res) against the bound $(bound)). The tensor is not " *
        "generic enough for the restriction sizes r = $(r) on dims $(dims); retry " *
        "with a larger `sizes = $(min.(dims, ceil.(Int, 1.5 .* r)))`, with " *
        "restriction = :random if it was :corner, or fall back to :SylverLining.")

    if method.verify === :full && prod(dims) <= 2e7
        for Ms in mats
            bound = atol * gnorm * max(sum(norm, Ms), eps(RT))
            if G isa Array
                # The library's own Z-law check (`Dleto.der_residual`), one
                # squared norm per chisel row: two blocks of `block_bytes`
                # instead of an accumulator and a mode product the size of the
                # tensor.  A disengaged axis has `P[rho, a] == 0` -- that is
                # what disengaged MEANS -- so its zero matrix is skipped there
                # exactly as it was skipped here.
                sq = der_residual_squares(G, Ms, P)
                for rho in 1:m
                    res = sqrt(sq[rho])
                    res <= bound || fail(res, bound)
                end
            else
                # Device tensors keep the direct route: the blocked
                # accumulation is `mul!` with `β = 1`, which `_qdn_ttm!`'s
                # device path cannot do.
                Md = [engaged[a] ? up(Ms[a]) : Ms[a] for a in 1:N]
                for rho in 1:m
                    E = _qdn_zeros_like(G, size(G))
                    for a in 1:N
                        (!engaged[a] || iszero(P[rho, a])) && continue
                        E .+= P[rho, a] .* _qdn_ttm(G, Md[a], a)
                    end
                    res = norm(E)
                    res <= bound || fail(res, bound)
                end
            end
        end
        return nothing
    end

    ahat = argmax([engaged[a] ? dims[a] : -1 for a in 1:N])
    ns = min(method.nslices, dims[ahat])
    sel = randperm(rng, dims[ahat])[1:ns]
    Gslice = _qdn_slice(G, ahat, sel)
    for Ms in mats
        bound = atol * gnorm * max(sum(norm, Ms), eps(RT))
        # `Ms[ahat][:, sel]` is selected on the HOST and uploaded: a device
        # `getindex` with an index vector is a kernel launch for what is a
        # `d x nslices` copy.
        Md = [engaged[a] ? up(a == ahat ? Ms[a][:, sel] : Ms[a]) : Ms[a] for a in 1:N]
        for rho in 1:m
            E = _qdn_zeros_like(Gslice, size(Gslice))
            if !iszero(P[rho, ahat])
                E .+= P[rho, ahat] .* _qdn_ttm(G, Md[ahat], ahat)
            end
            for a in 1:N
                (a == ahat || !engaged[a] || iszero(P[rho, a])) && continue
                E .+= P[rho, a] .* _qdn_ttm(Gslice, Md[a], a)
            end
            res = norm(E)
            res <= bound || fail(res, bound)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# ITensor boundary
# ---------------------------------------------------------------------------

function _qdn_validate(Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor)
    Ω isa IndTransverseOps ||
        error("QuickDerMethod requires IndTransverseOps, got $(typeof(Ω)).")
    ndims(Γ) >= 2 || error("QuickDerMethod requires a tensor of valence at least 2.")
    valency(Ω) == ndims(Γ) ||
        error("QuickDerMethod: Ω has valency $(valency(Ω)) but Γ has $(ndims(Γ)) axes.")
    size(P, 2) == ndims(Γ) ||
        error("QuickDerMethod: the chisel has $(size(P,2)) columns but Γ has " *
              "$(ndims(Γ)) axes.")
    size(P, 1) >= 1 || error("QuickDerMethod: the chisel has no rows.")
    all(i -> hasind(Γ, i), frames(Ω)) ||
        error("QuickDerMethod: Γ does not carry the frame of Ω.")
    any(engaged(Matrix{Float64}(P))) ||
        error("QuickDerMethod requires at least one engaged axis.")
    return nothing
end

"""
    _qdn_empty_result(Ω, T, r, dims)

The empty derivation space -- `(Ω, id, zeros(T, globalDim(Ω), 0))` -- UNLESS
the restricted solve that produced it did not converge, in which case this
raises instead.

"Γ conforms to no pattern for this chisel" is a legitimate answer and cannot
be escalated away; "the solver failed" looks exactly the same from here, and
that is the whole reason `NullVerdict` carries a `status`.
`_qdn_solve_and_lift` already declines when the RESTRICTED solve came back
empty and non-`:ok`; this is the same rule one step later, because an
undercounted restricted null space is not empty -- it lifts to universal
derivations that simply do not meet `Ω`, and the answer collapses to nothing
only after `_fastder_restrict_to_ops`.

Measured, and it is why this exists: scrambled sphere d = 12, forced
matrix-free, unwhitened, seed 4242, no ARPACK in the process.  KrylovKit's
block Lanczos broke down, the single-vector Arnoldi fallback returned
restricted nullity 7 of 13, and QuickDer reported ZERO derivations of a true
three -- silently, and green in the test suite.  With the fallback reporting
`:unconverged` (see ext/DletoKrylovKitExt.jl) this raises, and `AutoDerMethod`
falls back to SylverLining, which is exact.
"""
function _qdn_empty_result(Ω::TransverseOps, ::Type{T}, r::Vector{Int},
                           dims::Vector{Int}) where {T}
    QDN_LAST_SOLVE_STATUS[] === :ok || error(
        "QuickDer: the restricted null solve reported " *
        ":$(QDN_LAST_SOLVE_STATUS[]) and the lifted answer is EMPTY. Read that " *
        "as a FAILED solve, not as an empty derivation space -- a solver that " *
        "undercounts a restricted null space lifts to derivations that miss Ω " *
        "entirely. Restriction sizes r = $(r) on dims $(dims). Try " *
        "`solver = :ArpackSolver`, a larger `sizes`, or fall back to " *
        ":SylverLining.")
    return (Ω,
            LinearMaps.LinearMap(identity, identity, globalDim(Ω), globalDim(Ω);
                                 ismutating = false),
            zeros(T, globalDim(Ω), 0))
end

"""
    derTrOpsReduced(::QuickDerMethod, Ω, P, Γ; tol, nd, progress)
        -> (Ω, id_map, ders)

Solve-and-lift derivations of `Γ` for the chisel `P`, in the coordinates of
`Ω`.  The reduced operator space IS `Ω` and the expand map is the identity, as
in `FastDer3ValentMethod`: the kernel already solves over every axis, including
the disengaged ones (whose operator is zero, the representative `SylverLining`
expands to), so there is nothing to expand.

`nd > 0` caps the number of returned derivations; `tol` is relative and floored
at `sqrt(eps(eltype(Γ)))` by `qd_tolerance` (src/solvers/Precision.jl).

PRECISION.  The arithmetic runs in `compute_eltype(eltype(Γ))`, which is
`eltype(Γ)` except for Float16 (no CPU BLAS or LAPACK has a half-precision
path); the returned coordinates are rounded back to `eltype(Γ)`, so a Float16
tensor gives Float16 derivations at Float32 reliability.  The tolerance is
floored on the STORED type, not the computed one: promoting the arithmetic buys
stability, not information about the data.
"""
function derTrOpsReduced(
    method::QuickDerMethod,
    Ω::TransverseOps,
    P::AbstractMatrix,
    Γ::ITensor;
    tol::Real = TOL_DEFAULT,
    nd = -1,
    progress = false,
    kwargs...,
)::Tuple{TransverseOps, LinearMaps.LinearMap, AbstractMatrix{<:Number}}
    _qdn_validate(Ω, P, Γ)

    # Read Γ in Ω's frame order -- the order the coordinates use, and the order
    # `array` can hand back without a d^n copy when it already matches.
    fr = frames(Ω)
    G0 = ITensors.array(Γ, fr...)
    T = eltype(G0)

    # STORAGE type vs COMPUTE type.  `Tc` is the arithmetic; it differs from `T`
    # only for Float16, which has no BLAS or LAPACK anywhere (see
    # `Dleto.compute_eltype`).  The whole kernel below is written against one
    # element type, so the promotion happens here, once, in the same pass that
    # already materialises `G` -- and it is also what lets a Float16 tensor use
    # `device = :gpu`, since Apple GPUs want exactly Float32.
    Tc = compute_eltype(T)
    G = (T === Tc && G0 isa Array{T}) ? G0 : Array{Tc}(G0)

    # One upload for the whole run: every pass over the full tensor (the cross
    # sketches, the pair tensors of the lift, the verification) then happens on
    # the device, and nothing sends `d^n` bytes back.
    _qdn_check_device(method.device, Tc)
    _qdn_stage_reset!()
    QDN_LAST_SOLVE_STATUS[] = :ok
    tstage = time()
    Gk = method.device === :gpu ? to_gpu(G) : G
    tstage = _qdn_stage!(:upload, tstage)

    # On the STORED type: every check the kernel makes compares a product of two
    # data-sized quantities against zero, and promoting the arithmetic does not
    # make the data finer.  `qd_tolerance` (src/solvers/Precision.jl) has the
    # measured table.
    atol = _qd_tolerance(T, tol)
    dims = collect(size(G))
    n = length(dims)
    eng = engaged(Matrix{Float64}(P))
    Pm = Matrix{Tc}(P)
    rng = method.seed === nothing ? Random.default_rng() : MersenneTwister(method.seed)

    r = method.sizes === nothing ? _qdn_restriction_sizes(dims, eng, n) :
                                   _qdn_check_sizes(method.sizes, dims)

    # One retry with r bumped 50%.  The filter rejecting everything means the
    # restricted system saw solutions that no true derivation restricts to,
    # i.e. r was too small for THIS tensor; a bigger r is the only cure that
    # does not change the method.
    tried = Vector{Int}[]
    mats = nothing
    while true
        push!(tried, copy(r))
        mats = _qdn_solve_and_lift(Gk, Pm, eng, r, method, rng, atol, progress, T)
        mats === nothing || break
        length(tried) >= 2 && break
        bumped = [min(dims[a], max(r[a] + 1, ceil(Int, 1.5 * r[a]))) for a in 1:n]
        bumped == r && break                      # already unrestricted
        r = bumped
    end
    mats === nothing && error(
        "QuickDer: every restricted solution failed the lift consistency check at " *
        "restriction sizes $(tried) on dims $(dims). Γ is not generic enough for " *
        "this restriction; try `sizes = $(dims)`, `restriction = :random` if it " *
        "was :corner, or fall back to :SylverLining.")

    @debug "QuickDer after lift" nbasis = length(mats) rss_GB = Sys.maxrss() / 2^30
    tstage = time()
    _qdn_verify(Gk, Pm, eng, mats, method, atol, r, rng)
    tstage = _qdn_stage!(:verify, tstage)
    @debug "QuickDer after verify" rss_GB = Sys.maxrss() / 2^30

    isempty(mats) && return _qdn_empty_result(Ω, T, r, dims)

    # Universal derivations, cut down to the ones that live in Ω.  Rounded back
    # to the stored type: a Float16 tensor gets Float16 derivations, carrying
    # exactly the precision its data justifies and no more.
    ders = _fastder_restrict_to_ops(Ω, mats, atol)
    ders = eltype(ders) === T ? ders : Matrix{T}(ders)
    tstage = _qdn_stage!(:restrict_ops, tstage)
    @debug "QuickDer after restrict_to_ops" nders = size(ders, 2) rss_GB = Sys.maxrss() / 2^30
    size(ders, 2) == 0 && return _qdn_empty_result(Ω, T, r, dims)
    if nd > 0 && size(ders, 2) > nd
        ders = ders[:, 1:floor(Int, nd)]
    end

    id_map = LinearMaps.LinearMap(identity, identity, globalDim(Ω), globalDim(Ω);
                                  ismutating = false)
    return (Ω, id_map, ders)
end
