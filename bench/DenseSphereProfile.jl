# DenseSphereProfile -- controlled storage/density profile for the null solvers.
#
# The sphere benchmark normally starts from a sparse *support*, then applies
# dense orthogonal matrices.  This script makes that distinction observable:
#
#   support    original sphere: Dense storage, but many numerical zeros
#   scrambled  support after Γ * Xs: Dense storage and numerically full
#   random     generic dense tensor (only the inevitable scalar derivations)
#
# It deliberately profiles the derivation map rather than an assembled matrix.
# This is the work that Arpack, KrylovKit, AutoSolver, and the dense solvers
# actually see.  See bench/reports/dense-sphere-profile.md for interpretation.
#
# Usage (defaults are intentionally small):
#   julia -t 2 --project=. bench/DenseSphereProfile.jl
#   julia -t 4 --project=. bench/DenseSphereProfile.jl --dims=10,16,24 --reps=15
#   julia --project=. bench/DenseSphereProfile.jl --smoke
#   julia --project=. bench/DenseSphereProfile.jl --ops=symmetric --solvers=AutoSolver,ArpackSolver
#   julia --project=. bench/DenseSphereProfile.jl --csv=/tmp/dense-sphere.csv

using Dleto
using ITensors
using LinearAlgebra
using LinearMaps
using Logging
using Printf
using Random
using Statistics

# Load optional solver extensions when they are installed.  The profile remains
# useful with only the built-in dense solvers.
for (pkg, load) in ((:Arpack, () -> (@eval using Arpack)),
                    (:KrylovKit, () -> (@eval using KrylovKit)),
                    (:IterativeSolvers, () -> (@eval using IterativeSolvers)))
    try
        load()
    catch err
        @warn "optional solver package unavailable; omitting its solver" package = pkg exception = (err, catch_backtrace())
    end
end

include(joinpath(@__DIR__, "SphereHarness.jl"))

const DEFAULT_SOLVERS = [:AutoSolver, :SVDSolver, :ArpackSolver, :KrylovSolver]
const DEFAULT_DIMS = [10]
const DEFAULT_REPS = 9

argvalue(prefix, default = nothing) = begin
    value = findfirst(arg -> startswith(arg, prefix), ARGS)
    value === nothing ? default : ARGS[value][length(prefix)+1:end]
end

parse_symbols(value) = Symbol.(filter(!isempty, split(value, ',')))
parse_dims(value) = parse.(Int, filter(!isempty, split(value, ',')))

"Run `f` without extension banners or verdict warnings in timing output."
quietly(f) = with_logger(NullLogger()) do
    redirect_stdout(devnull) do
        f()
    end
end

"Exact numerical density; benchmark-only because it scans the tensor."
function tensor_profile(Γ::ITensor)
    fr = collect(inds(Γ))
    entries = prod(ITensors.dim.(fr))
    values = Array(Γ, fr...)
    nonzero = count(!iszero, values)
    stored = length(store(Γ))
    return (; storage = string(typeof(store(Γ))), entries, stored, nonzero,
            stored_fraction = stored / entries, numerical_density = nonzero / entries)
end

function sphere_variants(d::Integer; seed::Integer = d, T::Type = Float64)
    Random.seed!(seed)
    S = sphere_octant(d)
    fr = collect(inds(S))
    scrambled = randomize_tensor(S; type = :orthogonal).Δ
    random_dense = ITensor(randn(T, ntuple(_ -> d, 3)...), fr...)
    return [(:support, S), (:scrambled, scrambled), (:random, random_dense)]
end

function op_from_name(name::Symbol)
    name === :symmetric && return SymmetricOp()
    name === :universal && return UniversalOp()
    error("--ops accepts symmetric and universal, got $name")
end

"A count covers forward and adjoint applications, including norm estimation."
function counted_map(L::LinearMaps.LinearMap)
    calls = Ref(0)
    forward = x -> begin
        calls[] += 1
        L * x
    end
    adjoint = x -> begin
        calls[] += 1
        L' * x
    end
    counted = LinearMaps.LinearMap{eltype(L)}(forward, adjoint, size(L, 1), size(L, 2);
                                                ismutating = false,
                                                issymmetric = issymmetric(L),
                                                isposdef = isposdef(L))
    return counted, calls
end

function median_matvec_seconds(L; reps::Integer = DEFAULT_REPS)
    x = randn(eltype(L), size(L, 2))
    L * x                         # compile and warm the contraction
    times = Float64[]
    for _ in 1:reps
        started = time_ns()
        L * x
        push!(times, (time_ns() - started) / 1e9)
    end
    return median(times)
end

function warm_solver!(L, solver::Symbol)
    # One throwaway solve separates compilation from the reported solve.  The
    # problem is deliberately tiny in the default profile.
    counted, _ = counted_map(L)
    quietly(() -> solve_nullspace(counted, solver; tol = 1e-8, nv0 = 8))
    return nothing
end

function solve_profile(L, solver::Symbol)
    warm_solver!(L, solver)
    counted, calls = counted_map(L)
    calls[] = 0
    elapsed = @elapsed result = quietly(() -> solve_nullspace(counted, solver; tol = 1e-8))
    verdict = result.verdict
    return (; seconds = elapsed, map_calls = calls[], nullity = verdict.nullity,
            rule = verdict.rule, certified = verdict.certified, gap = verdict.gap,
            floor_binding = verdict.floor_binding, requested = verdict.requested)
end

function smoke!()
    variants = sphere_variants(6; seed = 6006)
    support = only(last.(filter(p -> first(p) === :support, variants)))
    scrambled = only(last.(filter(p -> first(p) === :scrambled, variants)))
    ps, pr = tensor_profile(support), tensor_profile(scrambled)
    @assert occursin("NDTensors.Dense", ps.storage)
    @assert occursin("NDTensors.Dense", pr.storage)
    @assert ps.numerical_density < 1
    @assert pr.numerical_density == 1
    println("smoke: support Dense / numerical density $(ps.numerical_density); " *
            "scrambled Dense / numerical density $(pr.numerical_density)")
    return nothing
end

function write_header(io)
    println(io, "d,variant,op,storage,stored_fraction,numerical_density,op_rows,op_cols,matvec_ms,solver,seconds,map_calls,nullity,rule,certified,gap,floor_binding,requested")
end

function write_row(io, d, variant, opname, tensor, rows, cols, matvec, solver, result)
    @printf(io, "%d,%s,%s,\"%s\",%.8g,%.8g,%d,%d,%.8g,%s,%.8g,%d,%d,%s,%s,%.8g,%s,%d\n",
            d, variant, opname, tensor.storage, tensor.stored_fraction, tensor.numerical_density,
            rows, cols, 1e3 * matvec, solver, result.seconds, result.map_calls,
            result.nullity, result.rule, result.certified, result.gap,
            result.floor_binding, result.requested)
end

function print_row(d, variant, opname, tensor, rows, cols, matvec, solver, result)
    @printf("%3d  %-9s %-9s %6.3f %6.3f %5dx%-5d %8.3f %-13s %8.3f %5d %3d %-9s %9.2e\n",
            d, String(variant), String(opname), tensor.stored_fraction, tensor.numerical_density,
            rows, cols, 1e3 * matvec, String(solver), result.seconds, result.map_calls,
            result.nullity, String(result.rule), result.gap)
end

function main()
    smoke!()
    "--smoke" in ARGS && return

    dims = parse_dims(argvalue("--dims=", join(DEFAULT_DIMS, ',')))
    ops = parse_symbols(argvalue("--ops=", "symmetric,universal"))
    requested = parse_symbols(argvalue("--solvers=", join(string.(DEFAULT_SOLVERS), ',')))
    solvers = filter(s -> s in available_solvers(), requested)
    isempty(solvers) && error("none of the requested solvers are available: $requested; available: $(available_solvers())")
    unavailable = setdiff(requested, solvers)
    isempty(unavailable) || @warn "requested solvers unavailable; omitted" unavailable available = available_solvers()
    reps = parse(Int, argvalue("--reps=", string(DEFAULT_REPS)))
    reps >= 3 || error("--reps must be at least 3")
    csv = argvalue("--csv=", nothing)

    io = csv === nothing ? nothing : open(csv, "w")
    io === nothing || write_header(io)
    println(" d  variant   op         stored  numden  operator       mv(ms) solver          sec calls nul rule            gap")
    for d in dims
        for (variant, Γ) in sphere_variants(d)
            profile = tensor_profile(Γ)
            for opname in ops
                Ω = IndTransverseOps(collect(inds(Γ)), op_from_name(opname))
                P = Matrix{eltype(Γ)}(UniversalChisel(3))
                L, _ = Dleto.sylvesterLM(Ω, P, Γ)
                matvec = median_matvec_seconds(L; reps)
                for solver in solvers
                    result = try
                        solve_profile(L, solver)
                    catch err
                        @warn "solver failed for profile row" d variant opname solver exception = (err, catch_backtrace())
                        continue
                    end
                    print_row(d, variant, opname, profile, size(L, 1), size(L, 2), matvec, solver, result)
                    io === nothing || write_row(io, d, variant, opname, profile, size(L, 1), size(L, 2), matvec, solver, result)
                end
            end
        end
    end
    io === nothing || close(io)
    csv === nothing || println("wrote $csv")
end

main()
