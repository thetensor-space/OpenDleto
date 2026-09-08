#
# StratifyMatrix -- the stratify benchmark MATRIX: one grid, one CSV row per
# cell AS IT COMPLETES, that the auto-selection knobs (QDN_DENSE_BUDGET_BYTES,
# QDN_GRAM_MIN_COLS, AUTODER_MIN_ENTRIES, matrix_free_solvers order) get tuned
# against.  bench/reports/2026-09-08/stratify-matrix/README.md is the write-up;
# this script only produces numbers.
#
# Usage (never bare `julia` -- see the ground rules in the task/README):
#
#   JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 bench/jl \
#       bench/StratifyMatrix.jl [method] [solver] [tag]
#
#   method  default :Auto      -- any get_derivation_method symbol
#   solver  default :AutoSolver -- forwarded to the method's `solver` keyword
#           (SylverLiningMethod's own solver; for :QuickDer and :Auto it lands
#           on QuickDerMethod's `solver`, since AutoDerMethod(; kwargs...)
#           forwards anything that is not `fallback_solver`/`min_entries` to
#           its inner QuickDerMethod -- so this DOES reach the matrix-free
#           branch's null solver for :Auto, not just the SylverLining
#           fallback).  Left at :AutoSolver, nothing is passed and every
#           route's own built-in default runs unmodified -- this is what makes
#           a bare `bench/jl bench/StratifyMatrix.jl` the BASELINE run.
#   tag     default "baseline" -- output file stem.
#
# Output: bench/reports/2026-09-08/stratify-matrix/<tag>.csv (one row per
# cell, written and flushed as it completes -- a killed run keeps every row up
# to that point) and <tag>-summary.md (the same rows as a Markdown table plus
# pass/fail counts).
#
# THE GRID.  Scrambled sphere octant (bench/SphereHarness.jl, SymmetricOp,
# oracle nullity = valence -- the (n-1) scalar derivations plus the Euler
# derivation) at valence 3, d in {30, 60, 100, 150} and valence 4,
# d in {20, 40, 60}; a video-shaped random tensor 120x90xFx3 (UniversalOp,
# oracle nullity 3 -- valence-1 scalar derivations only, since a random dense
# tensor has no structure beyond the scalars) for F in {10, 30}.  Every cell in
# T in {Float64, Float32, Float16}.  A cell is skipped, and recorded as such,
# when its tensor's byte count times 6 (build_sphere's own accounting of how
# many full copies the classic construction path holds at once, see its
# docstring) would exceed 5 GB.
#
# PER CELL: wall time (a tiny warm-up run before the grid excludes JIT), the
# `Sys.maxrss()` delta across the cell (monotone process peak, so this is a
# lower bound on the cell's own contribution when an earlier cell set the
# current peak), nullity found vs the oracle, the reconstruction `lsq_err`
# (sphere cells only -- video has no known ground-truth scramble to score
# against) and the Z-law residual (every cell, from
# `derTrOpsReduced(...; return_diagnostics = true)`'s `DerivationReport`,
# which `run_stratify` does not expose), the certified flag, which route
# answered and which null solver, and (for QuickDer) the per-stage times via
# `Dleto.QDN_STAGE_TIMES`.
#
# ONE SOLVE PER CELL, not two: `run_cell` below inlines `run_stratify`'s body
# (SphereHarness.jl says it is itself "the same body as `stratify(Ω, ch, Γ)`,
# unrolled so the nullity is recorded") with `return_diagnostics = true` added,
# rather than calling `run_stratify` for the timed row and a second,
# undiagnosed call for the report -- that would double the wall time of every
# cell for no reason, since the diagnostic call already IS the timed one once
# `return_diagnostics` is on the same `derTrOpsReduced` call.
#
using Dleto
using ITensors
using LinearAlgebra
using Random
using Printf

include(joinpath(@__DIR__, "SphereHarness.jl"))   # build_sphere, reconstruction, quietly

# --------------------------------------------------------------------- grid

const SPHERE_CASES = [(valence = 3, d = 30), (valence = 3, d = 60),
                       (valence = 3, d = 100), (valence = 3, d = 150),
                       (valence = 4, d = 20), (valence = 4, d = 40),
                       (valence = 4, d = 60)]
const VIDEO_FS = (10, 30)
const TYPES = (Float64, Float32, Float16)
const SKIP_BUDGET_BYTES = 5.0 * 1e9   # 5 GB, decimal as the task states it

cell_dims(family::Symbol, valence::Integer, d, F) =
    family === :sphere ? ntuple(_ -> d, valence) : (120, 90, F, 3)

# ------------------------------------------------------------- cell builders

"""One sphere cell's input, in `run_cell`'s shape (`build_sphere` plus `oracle`
and `tag`)."""
function sphere_cell_input(valence::Integer, d::Integer, T::Type)
    inp = build_sphere(d; valence, T, ops = SymmetricOp())
    return merge(inp, (; oracle = valence, tag = "sphere v$valence d=$d"))
end

"""One video cell's input: a plain random dense tensor, no known scramble
(`S = nothing`), oracle nullity 3 (the valence-4 universal chisel's scalar
derivations -- see `_der_scalar_dim`)."""
function video_cell_input(F::Integer, T::Type)
    dims = (120, 90, F, 3)
    fr = [Index(dims[a], "vid$a") for a in 1:4]
    A = randn(T, dims...)
    Γ = ITensor(A, fr...)
    Ω = IndTransverseOps(fr, UniversalOp())
    ch = UniversalChisel(4)
    return (; S = nothing, fr, Xs = nothing, Es = nothing, Ω, ch, Γ,
              dims = collect(dims), T, oracle = 3, tag = "video F=$F")
end

# ------------------------------------------------------------------ one run

"""
    run_cell(inp; method, solver, tol = 1e-6) -> NamedTuple

One stratification of `inp.Γ` under `method`/`solver`, timed, with
`return_diagnostics = true` for the route, solver, certified flag and
per-stage times, then the reconstruction score when `inp.S !== nothing`.
`solver = :AutoSolver` passes nothing through, so every route's own built-in
default runs (see this file's header for what that means for `:Auto`).
"""
function run_cell(inp; method::Symbol, solver::Symbol, tol::Real = 1e-6)
    Ω, ch, Γ = inp.Ω, inp.ch, inp.Γ
    kw = solver === :AutoSolver ? NamedTuple() : (; solver)

    GC.gc()
    rss0 = Sys.maxrss()
    Dleto.QDN_STAGE_TIMES[] = Dict{Symbol,Float64}()
    Dleto.QDN_APPLY_COUNT[] = 0

    status = "ok"
    nullity = 0
    lsq_err = NaN
    zlaw_resid = NaN
    certified = false
    method_used = Symbol("")
    solver_used = Symbol("-")
    device_used = Symbol("-")
    whitened_used = missing

    t0 = time()
    try
        quietly() do
            m = Dleto.get_derivation_method(method; kw...)
            (_, expand_map, ders, rep) = derTrOpsReduced(m, Ω, ch, Γ; tol = tol,
                                                          return_diagnostics = true)
            nullity = size(ders, 2)
            certified = rep.certified
            method_used = rep.method
            rep.solver === nothing || (solver_used = rep.solver)
            rep.device === nothing || (device_used = rep.device)
            whitened_used = rep.whitened === nothing ? missing : rep.whitened
            zlaw_resid = (rep.residuals === nothing || isempty(rep.residuals)) ?
                         NaN : maximum(rep.residuals)
            if nullity == 0
                status = "no nontrivial derivations"
            else
                δ = embedITensors(Ω, expand_map(ders * randn(eltype(ders), nullity)))
                res = stratify(Γ, δ)
                if inp.S !== nothing
                    sc = reconstruction(inp, res.Σ, res.Xs)
                    lsq_err = sc.lsq_err
                end
            end
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
    end
    seconds = time() - t0

    apply_count = Dleto.QDN_APPLY_COUNT[]
    stages = Dleto.QDN_STAGE_TIMES[]
    Dleto.QDN_STAGE_TIMES[] = nothing
    Dleto.QDN_APPLY_COUNT[] = -1
    rss1 = Sys.maxrss()
    stage_times = (stages === nothing || isempty(stages)) ? "" :
        join(("$k=$(round(v; sigdigits = 4))" for (k, v) in sort(collect(stages); by = first)), ";")

    return (; seconds, rss_delta_mb = (rss1 - rss0) / 2^20, nullity, oracle = inp.oracle,
              lsq_err, zlaw_resid, certified, status, method_used, solver_used,
              device_used, whitened_used, apply_count, stage_times)
end

# -------------------------------------------------------------------- CSV/MD

const COLUMNS = (:family, :valence, :d, :F, :T, :tag, :seconds, :rss_delta_mb,
                  :nullity, :oracle, :lsq_err, :zlaw_resid, :certified, :status,
                  :method_used, :solver_used, :device_used, :whitened_used,
                  :apply_count, :stage_times)

function csv_field(x)
    s = x isa AbstractFloat && isnan(x) ? "" : string(x)
    (occursin(',', s) || occursin('"', s) || occursin('\n', s)) ?
        "\"" * replace(s, "\"" => "\"\"") * "\"" : s
end

write_header(io) = println(io, join(String.(COLUMNS), ","))

function write_row(io, row)
    println(io, join((csv_field(get(row, c, "")) for c in COLUMNS), ","))
    flush(io)
end

function md_table(rows)
    io = IOBuffer()
    println(io, "| ", join(String.(COLUMNS), " | "), " |")
    println(io, "|", repeat("---|", length(COLUMNS)))
    for row in rows
        vals = [csv_field(get(row, c, "")) for c in COLUMNS]
        println(io, "| ", join(vals, " | "), " |")
    end
    return String(take!(io))
end

# ------------------------------------------------------------------ warm-up

"""Tiny run at every method the grid will exercise, so the timed cells
exclude JIT compilation (`warmup!`'s own reason for existing, unrolled here
because the grid also needs `return_diagnostics` and stage timing warmed)."""
function warmup!(method::Symbol, solver::Symbol)
    inp = sphere_cell_input(3, 6, Float64)
    run_cell(inp; method, solver)
    inp4 = sphere_cell_input(4, 5, Float64)
    run_cell(inp4; method, solver)
    vinp = video_cell_input(4, Float64)
    run_cell(vinp; method, solver)
    return nothing
end

# --------------------------------------------------------------------- main

function main()
    method = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :Auto
    solver = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :AutoSolver
    tag    = length(ARGS) >= 3 ? ARGS[3] : "baseline"

    outdir = joinpath(@__DIR__, "reports", "2026-09-08", "stratify-matrix")
    mkpath(outdir)
    csv_path = joinpath(outdir, "$tag.csv")
    md_path = joinpath(outdir, "$tag-summary.md")

    @printf("StratifyMatrix: method = %s, solver = %s -> %s\n", method, solver, csv_path)
    flush(stdout)

    print("warm-up... "); flush(stdout)
    twarm = @elapsed warmup!(method, solver)
    @printf("done (%.2fs, excluded from cells)\n", twarm); flush(stdout)

    io = open(csv_path, "w")
    write_header(io)
    rows = NamedTuple[]

    grid = vcat(
        [(; family = :sphere, valence = c.valence, d = c.d, F = missing) for c in SPHERE_CASES],
        [(; family = :video, valence = 4, d = missing, F = F) for F in VIDEO_FS],
    )

    for c in grid, T in TYPES
        dims = cell_dims(c.family, c.valence, c.d, c.F)
        bytes = prod(Float64.(dims)) * sizeof(T)
        tag_str = c.family === :sphere ? "sphere v$(c.valence) d=$(c.d)" : "video F=$(c.F)"
        @printf("%-20s %-9s ", tag_str, T); flush(stdout)

        if bytes * 6 > SKIP_BUDGET_BYTES
            row = merge((; family = c.family, valence = c.valence, d = c.d, F = c.F, T),
                        (; tag = tag_str, seconds = NaN, rss_delta_mb = NaN, nullity = missing,
                           oracle = missing, lsq_err = NaN, zlaw_resid = NaN, certified = false,
                           status = "skipped (bytes*6 > budget)", method_used = Symbol(""),
                           solver_used = Symbol("-"), device_used = Symbol("-"),
                           whitened_used = missing, apply_count = -1, stage_times = ""))
            println("skipped (bytes*6 > budget)")
            push!(rows, row); write_row(io, row)
            continue
        end

        local r
        try
            inp = c.family === :sphere ? sphere_cell_input(c.valence, c.d, T) :
                                          video_cell_input(c.F, T)
            r = run_cell(inp; method, solver)
        catch e
            r = (; seconds = NaN, rss_delta_mb = NaN, nullity = missing, oracle = missing,
                   lsq_err = NaN, zlaw_resid = NaN, certified = false,
                   status = "build error: " * first(split(sprint(showerror, e), '\n')),
                   method_used = Symbol(""), solver_used = Symbol("-"), device_used = Symbol("-"),
                   whitened_used = missing, apply_count = -1, stage_times = "")
        end
        row = merge((; family = c.family, valence = c.valence, d = c.d, F = c.F, T,
                       tag = tag_str), r)
        @printf("%8.3fs  nullity %s/%s  certified %-5s  %s\n",
                r.seconds, r.nullity, r.oracle, r.certified, r.status)
        flush(stdout)
        push!(rows, row); write_row(io, row)
    end
    close(io)

    ok = count(row -> row.status in ("ok",), rows)
    open(md_path, "w") do io
        println(io, "# StratifyMatrix: $tag (method = $method, solver = $solver)")
        println(io)
        println(io, "$(length(rows)) cells, $ok ok.")
        println(io)
        println(io, md_table(rows))
    end
    @printf("\nwrote %s and %s (%d/%d cells ok)\n", csv_path, md_path, ok, length(rows))
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()
