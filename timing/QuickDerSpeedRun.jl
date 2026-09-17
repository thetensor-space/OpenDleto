using Dleto
using ITensors
using LinearAlgebra
using Printf
using Random
using CSV
using DataFrames

# Keep Arpack loaded so :ArpackSolver is registered.
using Arpack

include(joinpath(@__DIR__, "..", "bench", "SphereHarness.jl"))

const MIND      = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 100
const MAXD      = length(ARGS) >= 2 ? parse(Int, ARGS[2])     : 500
const STEP      = length(ARGS) >= 3 ? parse(Int, ARGS[3])     : 25
const SOLVE_TOL = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1e-6
const ELTYPES   = (Float32, Float64)
const OPSNAME   = :universal
const VALENCE   = 3
const TARGET    = Dict(Float32 => 1e-8, Float64 => 1e-16)
const DS        = MIND:STEP:MAXD

const CONFIGS = [
    (; name = "QuickDer/Auto", method = :QuickDer, solver = :AutoSolver, nd = -1),
    (; name = "QuickDer/Arpack", method = :QuickDer, solver = :ArpackSolver, nd = -1),
]

haskey(Dleto.SOLVER_REGISTRY, :ArpackSolver) ||
    error("ArpackSolver is not registered; cannot run QuickDer/Arpack.")

const CSV_PATH = joinpath(@__DIR__, "stratify-timing.csv")
const HEADER = "d,valence,eltype,solve_tol,target,ops,method,solver,der_seconds," *
               "strat_seconds,total_seconds,bytes,nullity,residual,meets_target," *
               "lsq_err,support,perm_ok,dims,nnz,status"
isfile(CSV_PATH) || open(io -> println(io, HEADER), CSV_PATH, "w")

function timed_stratify(inp, cfg, tol::Float64)
    Ω, ch = inp.Ω, inp.ch
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
            m = Dleto.get_derivation_method(cfg.method; solver = cfg.solver)
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
            residual = Float64(Dleto.der_residual(Γ, δ, ch))
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
        res = nothing
    end

    Σ = res === nothing ? nothing :
        (res.Σ isa Dleto.TensorSpace.TensorElement ? res.Σ.value : res.Σ)
    sc = (Σ !== nothing && inp.S !== nothing) ?
         reconstruction(inp, Σ, res.Xs) :
         (; lsq_err = NaN, support = NaN, perm_ok = false)

    return (; der_s, strat_s, total_s = der_s + strat_s, bytes, nullity, residual,
              sc.lsq_err, sc.support, sc.perm_ok, status)
end

function emit_row(row)
    line = @sprintf("%d,%d,%s,%.0e,%.0e,%s,%s,%s,%.4f,%.4f,%.4f,%d,%d,%.3e,%s,%.3e,%.4f,%s,%s,%d,%s",
                    row.d, VALENCE, row.eltype, row.solve_tol, row.target, row.ops,
                    row.method, row.solver, row.der_s, row.strat_s, row.total_s,
                    row.bytes, row.nullity, row.residual,
                    isnan(row.residual) ? "false" : string(row.residual <= row.target),
                    row.lsq_err, row.support, row.perm_ok, row.dims, row.nnz,
                    replace(row.status, ',' => ';'))
    println(line)
    open(io -> println(io, line), CSV_PATH, "a")
end

println("# QuickDer speed run: d=$(MIND):$(STEP):$(MAXD), ops=$(OPSNAME), tol=$(SOLVE_TOL)")
println("# methods: ", join(getindex.(CONFIGS, :name), ", "))

# Short warmup at the first size for both configs and both precisions.
for T in ELTYPES
    inp = build_sphere(first(DS); valence = VALENCE, T = T, ops = UniversalOp())
    for cfg in CONFIGS
        timed_stratify(inp, cfg, SOLVE_TOL)
    end
end

for d in DS
    for T in ELTYPES
        inp = build_sphere(d; valence = VALENCE, T = T, ops = UniversalOp())
        for cfg in CONFIGS
            r = timed_stratify(inp, cfg, SOLVE_TOL)
            emit_row((;
                d,
                eltype = string(T),
                solve_tol = SOLVE_TOL,
                target = TARGET[T],
                ops = String(OPSNAME),
                method = cfg.name,
                solver = String(cfg.solver),
                r.der_s, r.strat_s, r.total_s, r.bytes, r.nullity, r.residual,
                r.lsq_err, r.support, r.perm_ok,
                dims = join(inp.dims, ' '), inp.nnz, r.status
            ))
        end
        inp = nothing
        GC.gc()
    end
end

println("# done -> ", CSV_PATH)
