#
# StratifyTiming -- runtime of tensor stratification vs. dimension d, for the
# SphereLab problem, with the derivation solve and the stratification itself
# timed SEPARATELY.
#
# THE INPUT is labs/SphereLab.ipynb section 3's construction, at scale: a
# cubical d x d x d tensor holding a hidden sphere octant sampled from the
# densor space (axis values u[i] = x_i^2 - r^2/3, support where the u's sum to
# zero), scrambled by a random ORTHOGONAL change of basis on every axis and
# passed through `nondeg`.  bench/SphereHarness.jl builds and scores it; this
# script only adds the timing split, the precision grid and the drop-out rule.
#
# THE SPLIT.  `stratify(Ω, ch, Γ)` is two phases and they scale differently:
#
#   der    `derTrOpsReduced` -- solve for the derivation space.  This is the
#          null-space problem, O(d^3)-O(d^6) depending on the method, and it
#          is what the solver choice actually changes.
#   strat  embed the chosen derivation and put it in real canonical form, then
#          act on Γ.  An eigendecomposition per axis plus three contractions:
#          method-independent, and the floor the der time is measured against.
#
# The body is `stratify(Ω, ch, Γ)` from src/Densors.jl unrolled, so the two
# phases are the real ones, not a re-implementation.
#
# THE PRECISION GRID.  Float32 and Float64, each solved at the repo-standard
# solver tolerance (`SOLVE_TOL`, default 1e-6) and scored against an ACCURACY
# TARGET of 1e-8 for Float32 and 1e-16 for Float64.  The two are different
# things and the distinction is the whole reason this file has both columns:
#
#   `solve_tol`  the singular-value cutoff handed to the solver.  It is NOT
#                honoured literally -- `qd_tolerance` floors it at
#                `max(tol, sqrt(data_floor(T)), precision_floor(T))`
#                (src/solvers/Precision.jl), so anything under ~1e-7 in Float32
#                is decorative, and on the null-solver path the tolerance is
#                relative to the operator norm and gets SQUARED when the map is
#                squared (src/solvers/NullSolvers.jl).  Handing it 1e-16
#                directly does not buy accuracy; measured here, every Float64
#                row past d = 50 came back with nullity 0, i.e. no derivation
#                found at all.
#   `target`     the accuracy the answer is held to, reported against the
#                measured Z-law residual `Dleto.der_residual` and the
#                reconstruction error.  `meets_target` is residual <= target.
#
# Read `residual` and `lsq_err` to see what each precision actually attains.
# On this input that is around 1e-14 in Float64 and 1e-6 in Float32 -- both
# short of the targets above, which is a property of the conditioning of the
# scrambled sphere, not of any solver.
#
# THE CHISEL AXIS.  Every run uses the universal chisel.  What varies is the
# operator space it is chiseled against:
#
#   universal  UniversalOp()  -- the general, unrestricted chisel
#   symmetric  SymmetricOp()  -- operators restricted to symmetric matrices,
#              which is the space that actually fits an orthogonal scramble
#              (an orthogonal conjugate of a diagonal derivation is symmetric)
#
# THE SOLVER AXIS.  `SylverLining` is the general method: build the derivation
# operator as a `LinearMap` and hand it to a null solver.  WHICH null solver is
# the single biggest lever in this whole benchmark, and several of them live in
# package extensions -- `:ArpackSolver` behind Arpack (a weakdep),
# `:KrylovSolver` behind KrylovKit, `:LanczosSolver`/`:CGSolver`/`:LSMRSolver`
# behind IterativeSolvers.  A script that does not load them does not merely
# lose those rows: `:AutoSolver` picks `first(matrix_free_solvers(L))` from the
# solvers actually in `SOLVER_REGISTRY`, so with Arpack missing it silently
# falls through to Krylov or CG, which the repo has measured at 7-20x slower
# (Arpack 10.5 s vs Krylov 76 s vs CG 210 s at d = 150).  That is why the
# `using` block below is load-bearing and why this file refuses to run without
# Arpack rather than quietly producing a slower table.
#
# The same applies to `QuickDer`, whose `solver` keyword selects the null
# solver for its restricted matrix-free solve.
#
# DROP-OUT.  A (method, ops, eltype) configuration whose total time exceeds
# `budget` (default 60 s) at some d is not run at any larger d.  A second,
# looser guard skips a configuration whose projected time at the next d --
# cubic extrapolation from its last measurement -- is over 10x the budget, so
# a single run cannot overrun the budget by more than about that factor.
#
# JIT.  Every configuration runs twice at d = 8 before any timing.
#
# Usage:
#   bench/jl timing/StratifyTiming.jl [maxd] [budget_s] [solve_tol]
#
# Writes one row per (d, eltype, ops, method, solver) to
# timing/stratify-timing.csv as soon as it is measured, and prints the same row.
#
using Dleto
using ITensors
using LinearAlgebra
using Printf
using Random

# The solver extensions.  See THE SOLVER AXIS above: these are not optional
# imports, they decide which solvers exist and therefore what `:AutoSolver`
# chooses.
using KrylovKit          # :KrylovSolver
using IterativeSolvers   # :LanczosSolver, :CGSolver, :LSMRSolver
using Arpack             # :ArpackSolver -- a weakdep; without it AutoSolver
                         # silently falls through to a much slower solver

include(joinpath(@__DIR__, "..", "bench", "SphereHarness.jl"))

const MAXD      = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 150
const BUDGET    = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 60.0
const SOLVE_TOL = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1e-6
const TARGET    = Dict(Float32 => 1e-8, Float64 => 1e-16)
const ELTYPES   = (Float32, Float64)

# Refuse to measure a table that the missing weakdep would silently slow down.
haskey(Dleto.SOLVER_REGISTRY, :ArpackSolver) ||
    error("ArpackSolver is not registered -- `using Arpack` did not take effect. " *
          "Every :AutoSolver row would silently fall through to a slower solver; " *
          "fix the environment rather than measuring that.")
const CSV    = joinpath(@__DIR__, "stratify-timing.csv")
const DS     = 10:5:MAXD
const VALENCE = 3

# ------------------------------------------------------------ configurations
#
# `nd` is the number of null modes to ask for.  -1 means "the whole space" for
# every method except SymmetricGram, which solves for a fixed count of smallest
# approximate modes and therefore needs a positive one; valence 3 with these
# symmetric operators has nullity 3 (two scalar derivations plus the sphere's).
#
# `ops` is which operator spaces the method can be asked for.  QuickDer and
# QuickDer3 lift through least-squares solves over ALL matrices, so they are
# universal-only; SymmetricGram is symmetric-only by construction.

# Every registered null solver, for the general method.  The ones past the
# first four come from package extensions; see THE SOLVER AXIS above.
const NULL_SOLVERS = [:AutoSolver, :SVDSolver, :LUSolver, :GramSolver,
                      :ArpackSolver, :KrylovSolver, :LanczosSolver,
                      :CGSolver, :LSMRSolver, :ShiftInvertSolver]

const CONFIGS = vcat(
    # The general method, once per null solver.
    [(; name = "SylverLining/$(String(sv)[1:end-6])", method = :SylverLining, solver = sv,
        nd = -1, ops = (:universal, :symmetric)) for sv in NULL_SOLVERS],
    # `:Auto` -- QuickDer where it applies, SylverLining otherwise.  This is
    # what `stratify` uses when nothing is asked for, so it is the default a
    # user actually gets.
    [(; name = "Auto", method = :Auto, solver = :none, nd = -1,
        ops = (:universal, :symmetric)),
    # The solve-and-lift methods.  `QuickDer`'s `solver` is the null solver for
    # its RESTRICTED matrix-free solve, which is a different choice from the
    # one above and worth sweeping separately.
     (; name = "QuickDer/Auto", method = :QuickDer, solver = :AutoSolver, nd = -1,
        ops = (:universal,)),
     (; name = "QuickDer/Arpack", method = :QuickDer, solver = :ArpackSolver, nd = -1,
        ops = (:universal,)),
     (; name = "QuickDer/Gram", method = :QuickDer, solver = :GramSolver, nd = -1,
        ops = (:universal,)),
     (; name = "QuickDer3", method = :QuickDer3, solver = :none, nd = -1,
        ops = (:universal,)),
     (; name = "SymmetricGram", method = :SymmetricGram, solver = :none, nd = 3,
        ops = (:symmetric,))])

const OPSPACE = Dict(:universal => UniversalOp(), :symmetric => SymmetricOp())

# ------------------------------------------------------------------ one run

"""
    timed_stratify(inp, cfg, tol) -> (; der_s, strat_s, total_s, bytes, nullity, ...)

`stratify(Ω, ch, Γ)` with the derivation solve and the stratification timed
apart.  Same body as `src/Densors.jl`, unrolled.
"""
function timed_stratify(inp, cfg, tol::Float64)
    Ω, ch = inp.Ω, inp.ch
    # `nondeg` hands back a `TensorSpace.TensorElement`, so the Float64 path of
    # `build_sphere` (which keeps its output as-is) carries a wrapper where the
    # converted Float32 path carries a bare ITensor.  The solvers want the bare
    # one; unwrap rather than convert, so no copy is made.
    Γ = inp.Γ isa Dleto.TensorSpace.TensorElement ? inp.Γ.value : inp.Γ
    der_s = strat_s = 0.0
    bytes = 0
    nullity = 0
    residual = NaN
    res = nothing
    status = "ok"
    GC.gc()
    try
        quietly() do
            m = cfg.solver === :none ?
                Dleto.get_derivation_method(cfg.method) :
                Dleto.get_derivation_method(cfg.method; solver = cfg.solver)
            d1 = @timed derTrOpsReduced(m, Ω, ch, Γ; tol = tol, nd = cfg.nd)
            der_s = d1.time
            bytes += d1.bytes
            (rΩ, expand_map, ders) = d1.value
            nullity = size(ders, 2)
            nullity == 0 && error("no nontrivial derivations at tol = $tol")
            local δ
            d2 = @timed begin
                δ = embedITensors(Ω, expand_map(ders * randn(eltype(ders), nullity)))
                stratify(Γ, δ)
            end
            strat_s = d2.time
            bytes += d2.bytes
            res = d2.value
            # The accuracy the run actually attained: the Z-law residual of the
            # derivation it stratified along, relative to the size of the data
            # (`Dleto.der_residual`).  Not timed -- it is the check, not the work.
            residual = Float64(Dleto.der_residual(Γ, δ, ch))
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
        res = nothing
    end
    # `stratify` returns Σ wrapped as a `TensorSpace.TensorElement`; the
    # harness's scorer wants the bare ITensor.
    Σ = res === nothing ? nothing :
        (res.Σ isa Dleto.TensorSpace.TensorElement ? res.Σ.value : res.Σ)
    sc = (Σ !== nothing && inp.S !== nothing) ?
         reconstruction(inp, Σ, res.Xs) :
         (; lsq_err = NaN, support = NaN, perm_ok = false)
    return (; der_s, strat_s, total_s = der_s + strat_s, bytes, nullity, residual,
              sc.lsq_err, sc.support, sc.perm_ok, status)
end

# --------------------------------------------------------------------- setup

const HEADER = "d,valence,eltype,solve_tol,target,ops,method,solver,der_seconds," *
               "strat_seconds,total_seconds,bytes,nullity,residual,meets_target," *
               "lsq_err,support,perm_ok,dims,nnz,status"

isfile(CSV) || open(io -> println(io, HEADER), CSV, "w")

function emit(row)
    line = @sprintf("%d,%d,%s,%.0e,%.0e,%s,%s,%s,%.4f,%.4f,%.4f,%d,%d,%.3e,%s,%.3e,%.4f,%s,%s,%d,%s",
                    row.d, VALENCE, row.eltype, SOLVE_TOL, row.target, row.ops,
                    row.method, row.solver, row.der_s, row.strat_s, row.total_s,
                    row.bytes, row.nullity, row.residual,
                    isnan(row.residual) ? "false" : string(row.residual <= row.target),
                    row.lsq_err, row.support, row.perm_ok, row.dims, row.nnz,
                    replace(row.status, ',' => ';'))
    println(line)
    flush(stdout)
    open(io -> println(io, line), CSV, "a")
end

println("# StratifyTiming: d = $(first(DS)):5:$(last(DS)), budget $(BUDGET) s, " *
        "solve_tol $(SOLVE_TOL), targets 1e-8 (Float32) / 1e-16 (Float64)")
println("# solvers: $(join(String.(NULL_SOLVERS), ", "))")
# The warm-up is long -- 27 configurations over four facets, each carrying its
# own specialization -- and it runs before a single row is written.  Report it,
# so a sweep that has not yet produced output is visibly working rather than
# apparently hung, and skip the second pass on anything already slow: a solver
# that takes seconds at d = 8 is not one whose JIT cost will matter at d = 150.
println("# warming up at d = 8 (", sum(length(c.ops) for c in CONFIGS) * length(ELTYPES),
        " configurations; no rows are written until this finishes) ...")
let done = 0, total = sum(length(c.ops) for c in CONFIGS) * length(ELTYPES)
    for T in ELTYPES, opsname in (:universal, :symmetric)
        inp = build_sphere(8; valence = VALENCE, T = T, ops = OPSPACE[opsname])
        for cfg in CONFIGS
            opsname in cfg.ops || continue
            r = timed_stratify(inp, cfg, SOLVE_TOL)
            r.total_s < 5.0 && timed_stratify(inp, cfg, SOLVE_TOL)
            done += 1
            @printf("#   [%2d/%2d] %-8s %-9s %-24s %6.2f s\n",
                    done, total, T, opsname, cfg.name, r.total_s)
            flush(stdout)
        end
    end
end
println(HEADER)

# -------------------------------------------------------------------- sweep
#
# `live` maps (eltype, ops, config name) -> the last (d, total_s) measured.
# Absent means never run; `:dropped` means over budget at some smaller d.

live = Dict{Tuple{DataType,Symbol,String},Any}()
for T in ELTYPES, cfg in CONFIGS, opsname in cfg.ops
    live[(T, opsname, cfg.name)] = nothing
end

for d in DS
    any(v -> v !== :dropped, values(live)) || break
    for T in ELTYPES, opsname in (:universal, :symmetric)
        todo = [cfg for cfg in CONFIGS if opsname in cfg.ops &&
                live[(T, opsname, cfg.name)] !== :dropped]
        isempty(todo) && continue
        inp = build_sphere(d; valence = VALENCE, T = T, ops = OPSPACE[opsname])
        for cfg in todo
            key = (T, opsname, cfg.name)
            # Looser guard: cubic extrapolation from the last measurement.  A
            # config is skipped now (not dropped) if it projects past 10x
            # budget, which bounds how far one run can overrun.
            last = live[key]
            if last !== nothing
                proj = last[2] * (d / last[1])^3
                if proj > 10 * BUDGET
                    live[key] = :dropped
                    continue
                end
            end
            r = timed_stratify(inp, cfg, SOLVE_TOL)
            # At sub-second sizes the first call on a given (eltype, ops,
            # method) path still carries specialization the d = 8 warm-up did
            # not reach, which is the whole spread of the d = 10 row.  Anything
            # under a second is cheap enough to simply run again and keep the
            # second measurement.
            r.total_s < 1.0 && (r = timed_stratify(inp, cfg, SOLVE_TOL))
            emit((; d, eltype = string(T), target = TARGET[T], ops = String(opsname),
                    method = cfg.name, solver = String(cfg.solver),
                    r.der_s, r.strat_s, r.total_s, r.bytes, r.nullity, r.residual,
                    r.lsq_err, r.support, r.perm_ok,
                    dims = join(inp.dims, ' '), inp.nnz, r.status))
            live[key] = r.total_s > BUDGET ? :dropped : (d, r.total_s)
        end
        inp = nothing
        GC.gc()
    end
end

println("# done -> $CSV")
