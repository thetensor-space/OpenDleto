#
# StratifyKnobTune -- is QDN_GRAM_MIN_COLS's default (1000) the right place to
# switch the dense restricted solve from :SVDSolver to :GramSolver?
#
# The baseline StratifyMatrix.jl grid crosses this boundary twice (valence 3
# between d = 30 and d = 60; valence 4 between d = 20 and d = 40) and never
# crosses QDN_DENSE_BUDGET_BYTES (every cell in that grid stays in the dense
# branch) or AUTODER_MIN_ENTRIES (every cell's entry count is already 13x-1000x
# over the 2000 threshold), so this is the one auto-selection knob that grid
# has evidence for.  This script forces each solver at every cell NEAR the
# boundary and reports both, so the choice is a measured comparison rather
# than an inference from which one the default happened to run.
#
# Usage:
#   JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 bench/jl \
#       bench/StratifyKnobTune.jl
#
using Dleto
using ITensors
using LinearAlgebra
using Random
using Printf

include(joinpath(@__DIR__, "SphereHarness.jl"))
include(joinpath(@__DIR__, "StratifyMatrix.jl"))   # sphere_cell_input (no auto-run here)

const CELLS = [(3, 30), (3, 60), (3, 100), (4, 20), (4, 40), (4, 60)]
const TYPES2 = (Float64, Float32, Float16)

function try_solver(inp, force_min_cols::Int; tol::Real = 1e-6)
    Ω, ch, Γ = inp.Ω, inp.ch, inp.Γ
    saved = Dleto.QDN_GRAM_MIN_COLS[]
    Dleto.QDN_GRAM_MIN_COLS[] = force_min_cols
    GC.gc()
    status = "ok"
    seconds = NaN
    nullity = 0
    certified = false
    solver_used = Symbol("-")
    ncols = -1
    lsq_err = NaN
    zlaw = NaN
    t0 = time()
    try
        quietly() do
            m = Dleto.get_derivation_method(:Auto)
            (_, expand_map, ders, rep) = derTrOpsReduced(m, Ω, ch, Γ; tol = tol,
                                                          return_diagnostics = true)
            nullity = size(ders, 2)
            certified = rep.certified
            solver_used = rep.solver === nothing ? Symbol("-") : rep.solver
            ncols = rep.restricted_size === nothing ? -1 : rep.restricted_size[2]
            zlaw = (rep.residuals === nothing || isempty(rep.residuals)) ? NaN : maximum(rep.residuals)
            if nullity > 0
                δ = embedITensors(Ω, expand_map(ders * randn(eltype(ders), nullity)))
                res = stratify(Γ, δ)
                if inp.S !== nothing
                    sc = reconstruction(inp, res.Σ, res.Xs)
                    lsq_err = sc.lsq_err
                end
            else
                status = "no nontrivial derivations"
            end
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
    end
    seconds = time() - t0
    Dleto.QDN_GRAM_MIN_COLS[] = saved
    return (; seconds, nullity, oracle = inp.oracle, certified, solver_used, ncols,
              lsq_err, zlaw, status)
end

"""
    try_dense_budget(inp, forced_budget) -> NamedTuple

Same shape as `try_solver`, but forces `QDN_DENSE_BUDGET_BYTES` instead of
`QDN_GRAM_MIN_COLS` -- `0.0` forces the matrix-free branch whatever the
restricted system's size, `Inf` forces dense (never falls back).  Also reports
`apply_count` (`Dleto.QDN_APPLY_COUNT`), which is `0` on the dense branch and
positive on the matrix-free one -- the direct evidence for which branch ran.
"""
function try_dense_budget(inp, forced_budget::Real; tol::Real = 1e-6)
    Ω, ch, Γ = inp.Ω, inp.ch, inp.Γ
    saved = Dleto.QDN_DENSE_BUDGET_BYTES[]
    Dleto.QDN_DENSE_BUDGET_BYTES[] = float(forced_budget)
    Dleto.QDN_APPLY_COUNT[] = 0
    GC.gc()
    status = "ok"
    nullity = 0
    certified = false
    solver_used = Symbol("-")
    ncols = -1
    lsq_err = NaN
    zlaw = NaN
    t0 = time()
    try
        quietly() do
            m = Dleto.get_derivation_method(:Auto)
            (_, expand_map, ders, rep) = derTrOpsReduced(m, Ω, ch, Γ; tol = tol,
                                                          return_diagnostics = true)
            nullity = size(ders, 2)
            certified = rep.certified
            solver_used = rep.solver === nothing ? Symbol("-") : rep.solver
            ncols = rep.restricted_size === nothing ? -1 : rep.restricted_size[2]
            zlaw = (rep.residuals === nothing || isempty(rep.residuals)) ? NaN : maximum(rep.residuals)
            if nullity > 0
                δ = embedITensors(Ω, expand_map(ders * randn(eltype(ders), nullity)))
                res = stratify(Γ, δ)
                if inp.S !== nothing
                    sc = reconstruction(inp, res.Σ, res.Xs)
                    lsq_err = sc.lsq_err
                end
            else
                status = "no nontrivial derivations"
            end
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
    end
    seconds = time() - t0
    apply_count = Dleto.QDN_APPLY_COUNT[]
    Dleto.QDN_APPLY_COUNT[] = -1
    Dleto.QDN_DENSE_BUDGET_BYTES[] = saved
    return (; seconds, nullity, oracle = inp.oracle, certified, solver_used, ncols,
              apply_count, lsq_err, zlaw, status)
end

function main()
    outdir = joinpath(@__DIR__, "reports", "2026-09-08", "stratify-matrix")
    mkpath(outdir)
    csv_path = joinpath(outdir, "knob-gram-min-cols.csv")
    io = open(csv_path, "w")
    println(io, "valence,d,T,forced,ncols,seconds,nullity,oracle,certified,lsq_err,zlaw_resid,status")

    # Warm-up both branches so the timed cells exclude JIT.
    wi = sphere_cell_input(3, 8, Float64)
    try_solver(wi, 0); try_solver(wi, typemax(Int))

    @printf("%-14s %-9s %-8s %8s %10s %8s %10s %10s\n",
            "cell", "T", "forced", "ncols", "seconds", "nullity", "certified", "status")
    for (valence, d) in CELLS, T in TYPES2
        inp = sphere_cell_input(valence, d, T)
        for (forced, label) in ((0, "gram"), (typemax(Int), "svd"))
            r = try_solver(inp, forced)
            @printf("%-14s %-9s %-8s %8d %8.3fs %10s %8s %10s\n",
                    "v$valence d=$d", T, label, r.ncols, r.seconds, "$(r.nullity)/$(r.oracle)",
                    r.certified, r.status)
            flush(stdout)
            println(io, join((valence, d, T, label, r.ncols, r.seconds, r.nullity, r.oracle,
                               r.certified, r.lsq_err, r.zlaw, r.status), ","))
            flush(io)
        end
    end
    close(io)
    @printf("\nwrote %s\n", csv_path)

    # ---- the dense-vs-matrix-free boundary (QDN_DENSE_BUDGET_BYTES) --------
    # The main grid never crosses it (every cell stays dense); these two cells
    # are chosen to actually be large enough for the DEFAULT budget to prefer
    # matrix-free at some (d, T), so the comparison is not entirely forced.
    csv_path2 = joinpath(outdir, "knob-dense-budget.csv")
    io2 = open(csv_path2, "w")
    println(io2, "case,T,forced,ncols,seconds,apply_count,nullity,oracle,certified,lsq_err,zlaw_resid,status")
    dense_cells = [("sphere v3 d=200", () -> sphere_cell_input(3, 200, Float32)),
                   ("video F=60", () -> video_cell_input(60, Float32))]
    @printf("\n%-18s %-9s %-8s %8s %10s %8s %10s %8s %10s\n",
            "case", "T", "forced", "ncols", "seconds", "applies", "nullity", "certified", "status")
    for (name, build) in dense_cells
        inp = build()
        for (forced, label) in ((Inf, "default(dense)"), (0.0, "forced-free"))
            r = try_dense_budget(inp, forced)
            @printf("%-18s %-9s %-14s %8d %8.3fs %8d %10s %8s %10s\n",
                    name, inp.T, label, r.ncols, r.seconds, r.apply_count,
                    "$(r.nullity)/$(r.oracle)", r.certified, r.status)
            flush(stdout)
            println(io2, join((name, inp.T, label, r.ncols, r.seconds, r.apply_count,
                                r.nullity, r.oracle, r.certified, r.lsq_err, r.zlaw, r.status), ","))
            flush(io2)
        end
    end
    close(io2)
    @printf("\nwrote %s\n", csv_path2)
end

main()
