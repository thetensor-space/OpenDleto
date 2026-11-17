#
# LiftCost -- stage-time and bit-closeness harness for the lift's redundant
# full-tensor work.
#
# Question: docs/beta/review/A-quickder.md F6 ("Redundant mode products in the
# lift RHS") and F7 ("`_qdn_pair_tensor` pays n(n-1) full-tensor passes where 2
# would do"), and docs/beta/REVIEW.md S6 M11, all point at
# `_qdn_solve_and_lift`'s `for a in lift` loop (src/solvers/QuickDerN.jl) as
# the place `_qdn_cross_sketches`'s shared-prefix trick was never applied.
# This script records the `:lift` stage time (and total wall time) from
# `Dleto.QDN_STAGE_TIMES[]` before and after that fix, on the cases named in
# the review and the task: scrambled spheres at valence 3 and 4, a multi-row
# chisel (`CentroidChisel(3)`), and a video-shaped tensor.
#
# It also serialises the returned derivation coordinates per case (under
# `coords/<tag>/`) so a "before" and "after" run can be compared for
# bit-closeness: `maximum(abs.(before .- after))` and the principal angle
# between the two derivation spaces (`principal_angle`,
# test/TestQuickDerDevice.jl:57) once both tags have been run.
#
# Usage:
#   JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 \
#     bench/jl bench/LiftCost.jl <tag>
#       <tag> (default "run") names this run -- pass "before"/"after" to keep
#       two runs' serialized coordinates apart for the bit-closeness check.
#   -> bench/reports/2026-09-08/lift/lift-cost.csv (appended, one row/case/tag)
#   -> bench/reports/2026-09-08/lift/coords/<tag>/<case>.jls (Serialization)
#
using Dleto
using ITensors
using LinearAlgebra
using Logging
using Printf
using Random
using Serialization

include(joinpath(@__DIR__, "SphereHarness.jl"))

const LIFT_DIR = joinpath(@__DIR__, "reports", "2026-09-08", "lift")
mkpath(LIFT_DIR)
const LIFT_CSV = joinpath(LIFT_DIR, "lift-cost.csv")

# Every stage `_qdn_solve_and_lift` / `derTrOpsReduced` mark, in the order they
# happen -- the same list bench/GpuMovie.jl uses, so a row reads left to right
# as the run.
const STAGES = [:upload, :sketch, :whiten, :restricted, :solve, :lift, :filter,
                :verify, :restrict_ops]

const HEADER = "tag,case,dims,valence,eltype,seconds," *
               join(string.(STAGES) .* "_seconds", ",") * ",nullity,status"

csv_header(path, header) = isfile(path) || open(path, "w") do io
    println(io, "# LiftCost: :lift-stage and total-time measurements around the ",
                 "_qdn_pair_tensor / lift-RHS sharing fix (F6/F7). One row per ",
                 "(tag, case); coordinates for bit-closeness live under coords/<tag>/.")
    println(io, header)
end

"""A random dense tensor and the pieces `derTrOpsReduced` wants, UniversalOp
throughout so any chisel (CentroidChisel included) has a large enough operator
space -- the pattern `bench/GpuMovie.jl`/`test/TestQuickDerN.jl`'s multi-row
chisel test already use."""
function random_input(dims::NTuple{N,Int}, T::Type; chisel = nothing, seed = 1) where {N}
    Random.seed!(seed)
    fr = [Index(dims[i], "a$i") for i in 1:N]
    Γ = ITensor(Array{T}(randn(dims...)), fr...)
    Ω = IndTransverseOps(fr, UniversalOp())
    ch = chisel === nothing ? UniversalChisel(N) : chisel
    return (; Ω, ch, Γ)
end

"""
    measure(case_name, inp; tag, tol, seed, record) -> NamedTuple

One `derTrOpsReduced(get_derivation_method(:QuickDer; seed), ...)` call with
the stage clock on (`Dleto.QDN_STAGE_TIMES[]`), timed end to end. `record`
controls whether the row is appended to the CSV / the coordinates are
serialized (`false` for a warm-up call).
"""
function measure(case_name, inp; tag = "run", tol::Real = 1e-6, seed::Integer = 1,
                 record::Bool = true)
    m = Dleto.get_derivation_method(:QuickDer; seed = seed)
    stages = Dict{Symbol,Float64}()
    Dleto.QDN_STAGE_TIMES[] = stages
    status = "ok"
    ders = nothing
    GC.gc()
    t = @elapsed try
        Logging.with_logger(Logging.NullLogger()) do
            (_, _, ders) = Dleto.derTrOpsReduced(m, inp.Ω, inp.ch, inp.Γ; tol = tol)
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
    end
    Dleto.QDN_STAGE_TIMES[] = nothing
    nullity = ders === nothing ? 0 : size(ders, 2)

    if record
        if ders !== nothing
            dir = joinpath(LIFT_DIR, "coords", tag)
            mkpath(dir)
            serialize(joinpath(dir, "$(case_name).jls"), Matrix{Float64}(real.(ders)))
        end

        csv_header(LIFT_CSV, HEADER)
        dims = size(inp.Γ)
        open(LIFT_CSV, "a") do io
            @printf(io, "%s,%s,\"%s\",%d,%s,%.6f", tag, case_name, string(dims),
                    length(dims), eltype(inp.Γ), t)
            for s in STAGES
                @printf(io, ",%.6f", get(stages, s, NaN))
            end
            @printf(io, ",%d,\"%s\"\n", nullity, status)
        end

        @printf("%-24s seconds=%8.3f  lift=%8.4f  sketch=%8.4f  solve=%8.4f  \
verify=%8.4f  nullity=%3d  %s\n",
                case_name, t, get(stages, :lift, NaN), get(stages, :sketch, NaN),
                get(stages, :solve, NaN), get(stages, :verify, NaN), nullity, status)
        flush(stdout)
    end
    return (; case_name, t, stages, nullity, status)
end

function main(tag::AbstractString)
    println("# LiftCost, tag = $(tag)")
    println(rpad("case", 24), "  ", "seconds  lift      sketch    solve     verify    nullity")

    # ---- warm-ups, one per shape family, discarded: JIT only, not the model.
    measure("warmup-v3", build_sphere(8; valence = 3); tag, record = false)
    measure("warmup-v4", build_sphere(6; valence = 4); tag, record = false)
    measure("warmup-centroid",
            random_input((6, 6, 6), Float64; chisel = CentroidChisel(3));
            tag, record = false)
    measure("warmup-video", random_input((8, 8, 4, 3), Float32); tag, record = false)

    # ---- valence 3 scrambled spheres
    for d in (60, 100, 150)
        measure("sphere-v3-d$(d)", build_sphere(d; valence = 3); tag)
    end

    # ---- valence 4 scrambled spheres
    for d in (30, 50)
        measure("sphere-v4-d$(d)", build_sphere(d; valence = 4); tag)
    end

    # ---- CentroidChisel(3) on a random 60^3 tensor
    measure("centroid3-random-60",
            random_input((60, 60, 60), Float64; chisel = CentroidChisel(3)); tag)

    # ---- video-shaped random tensor, Float32
    measure("video-160x120x30x3", random_input((160, 120, 30, 3), Float32); tag)

    println("\nCSV: $(LIFT_CSV)")
    println("Coordinates: $(joinpath(LIFT_DIR, "coords", tag))")
    return nothing
end

main(length(ARGS) >= 1 ? ARGS[1] : "run")
