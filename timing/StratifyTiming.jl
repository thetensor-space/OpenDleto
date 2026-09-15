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
# THE PRECISION GRID.  Float32 at tol = 1e-8 and Float64 at tol = 1e-16, as
# requested.  Both tolerances sit BELOW the respective machine epsilon
# (eps(Float32) = 1.2e-7, eps(Float64) = 2.2e-16), which is deliberate on the
# user's part but worth reading with care: `tol` here is the singular-value
# cutoff that separates the null space from the rest, so a cutoff under eps
# asks the solver to call a mode null only when it is null to the last bit.
# Where that yields nullity 0 the row records it as a failure rather than a
# time, and the `nullity` and `status` columns are the ones to read.  Pass a
# third argument to override the pair, e.g. `1e-6,1e-8`.
#
# THE CHISEL AXIS.  Every run uses the universal chisel.  What varies is the
# operator space it is chiseled against:
#
#   universal  UniversalOp()  -- the general, unrestricted chisel
#   symmetric  SymmetricOp()  -- operators restricted to symmetric matrices,
#              which is the space that actually fits an orthogonal scramble
#              (an orthogonal conjugate of a diagonal derivation is symmetric)
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
#   bench/jl timing/StratifyTiming.jl [maxd] [budget_s] [tol32,tol64]
#
# Writes one row per (d, eltype, ops, method) to timing/stratify-timing.csv as
# soon as it is measured, and prints the same row.
#
using Dleto
using ITensors
using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "..", "bench", "SphereHarness.jl"))

const MAXD   = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 200
const BUDGET = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 60.0
const TOLS   = length(ARGS) >= 3 ?
               (t = parse.(Float64, split(ARGS[3], ',')); (Float32 => t[1], Float64 => t[2])) :
               (Float32 => 1e-8, Float64 => 1e-16)
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

const CONFIGS = [
    (; name = "SylverLining/Auto",  method = :SylverLining, solver = :AutoSolver, nd = -1,
       ops = (:universal, :symmetric)),
    (; name = "SylverLining/SVD",   method = :SylverLining, solver = :SVDSolver,  nd = -1,
       ops = (:universal, :symmetric)),
    (; name = "Auto",               method = :Auto,         solver = :none,       nd = -1,
       ops = (:universal, :symmetric)),
    (; name = "QuickDer",           method = :QuickDer,     solver = :none,       nd = -1,
       ops = (:universal,)),
    (; name = "QuickDer3",          method = :QuickDer3,    solver = :none,       nd = -1,
       ops = (:universal,)),
    (; name = "SymmetricGram",      method = :SymmetricGram, solver = :none,      nd = 3,
       ops = (:symmetric,)),
]

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
            d2 = @timed begin
                δ = embedITensors(Ω, expand_map(ders * randn(eltype(ders), nullity)))
                stratify(Γ, δ)
            end
            strat_s = d2.time
            bytes += d2.bytes
            res = d2.value
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
    return (; der_s, strat_s, total_s = der_s + strat_s, bytes, nullity,
              sc.lsq_err, sc.support, sc.perm_ok, status)
end

# --------------------------------------------------------------------- setup

const HEADER = "d,valence,eltype,tol,ops,method,solver,der_seconds,strat_seconds," *
               "total_seconds,bytes,nullity,lsq_err,support,perm_ok,dims,nnz,status"

isfile(CSV) || open(io -> println(io, HEADER), CSV, "w")

function emit(row)
    line = @sprintf("%d,%d,%s,%.0e,%s,%s,%s,%.4f,%.4f,%.4f,%d,%d,%.3e,%.4f,%s,%s,%d,%s",
                    row.d, VALENCE, row.eltype, row.tol, row.ops, row.method, row.solver,
                    row.der_s, row.strat_s, row.total_s, row.bytes, row.nullity,
                    row.lsq_err, row.support, row.perm_ok, row.dims, row.nnz,
                    replace(row.status, ',' => ';'))
    println(line)
    flush(stdout)
    open(io -> println(io, line), CSV, "a")
end

println("# StratifyTiming: d = $(first(DS)):5:$(last(DS)), budget $(BUDGET) s, " *
        "tol $(TOLS[1][2]) (Float32) / $(TOLS[2][2]) (Float64)")
println("# warming up at d = 8 ...")
for (T, tol) in TOLS, opsname in (:universal, :symmetric)
    inp = build_sphere(8; valence = VALENCE, T = T, ops = OPSPACE[opsname])
    for cfg in CONFIGS
        opsname in cfg.ops || continue
        for _ in 1:2
            timed_stratify(inp, cfg, Float64(tol))
        end
    end
end
println(HEADER)

# -------------------------------------------------------------------- sweep
#
# `live` maps (eltype, ops, config name) -> the last (d, total_s) measured.
# Absent means never run; `:dropped` means over budget at some smaller d.

live = Dict{Tuple{DataType,Symbol,String},Any}()
for (T, tol) in TOLS, cfg in CONFIGS, opsname in cfg.ops
    live[(T, opsname, cfg.name)] = nothing
end

for d in DS
    any(v -> v !== :dropped, values(live)) || break
    for (T, tol) in TOLS, opsname in (:universal, :symmetric)
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
            r = timed_stratify(inp, cfg, Float64(tol))
            # At sub-second sizes the first call on a given (eltype, ops,
            # method) path still carries specialization the d = 8 warm-up did
            # not reach, which is the whole spread of the d = 10 row.  Anything
            # under a second is cheap enough to simply run again and keep the
            # second measurement.
            r.total_s < 1.0 && (r = timed_stratify(inp, cfg, Float64(tol)))
            emit((; d, eltype = string(T), tol = Float64(tol), ops = String(opsname),
                    method = cfg.name, solver = String(cfg.solver),
                    r.der_s, r.strat_s, r.total_s, r.bytes, r.nullity,
                    r.lsq_err, r.support, r.perm_ok,
                    dims = join(inp.dims, ' '), inp.nnz, r.status))
            live[key] = r.total_s > BUDGET ? :dropped : (d, r.total_s)
        end
        inp = nothing
        GC.gc()
    end
end

println("# done -> $CSV")
