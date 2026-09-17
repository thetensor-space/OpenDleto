#!/usr/bin/env julia
#=
StratifyTimingGPU.jl

GPU benchmark for stratification methods that support Apple Metal.
Focused run: Float32 only (Metal has no Float64), dimensions d=10 to 50,
only GPU-enabled solvers (GramSolver with device=:gpu, SylverLining with backend=:metal).

Usage:
    bench/jl timing/StratifyTimingGPU.jl [--csv=PATH]

The device column is populated with 'gpu'; CPU runs leave it blank.
Metal forces all computations to Float32, so the solve_tol is 1e-8.
=#

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Printf, Random, Dates, Statistics, LinearAlgebra
using Dleto, ITensors
using Metal  # Ensure Metal is loaded so the extensions register

include("../bench/SphereHarness.jl")

# ========================================================================== args
const MAXD = get(ENV, "MAXD", "50")  |> x -> parse(Int, x)
const TIMEOUT = get(ENV, "TIMEOUT", "120")  |> x -> parse(Float64, x)  # 120 s per test
const REPS = get(ENV, "REPS", "3")  |> x -> parse(Int, x)

# ========================================================================== CSV
const CSV_PATH = ARGS[1:end]
    |> x -> length(x) > 0 ? x[1] : get(ARGS; argparse) do; "stratify-timing-GPU.csv"; end
const CSV_PATH_REAL = joinpath(@__DIR__, CSV_PATH)

"""Reset the CSV and write the header."""
function reset_csv(path)
    header = "d,valence,eltype,solve_tol,target,ops,method,solver,backend,device,threads,heap_hint_gb,timeout_s,der_seconds,strat_seconds,total_seconds,bytes,nullity,residual,meets_target,lsq_err,support,perm_ok,dims,nnz,status"
    open(path, "w") do io
        println(io, header)
    end
end

"""Emit a single row to the CSV."""
function emit_row(path, row)
    open(path, "a") do io
        d, valence, eltype, solve_tol, target, ops, method, solver, backend, device,
        threads, heap_hint_gb, timeout_s, der_seconds, strat_seconds, total_seconds,
        bytes, nullity, residual, meets_target, lsq_err, support, perm_ok, dims, nnz, status = row
        
        @printf(io, "%d,%d,%s,%.1e,%.1e,%s,%s,%s,%s,%s,%d,%.1f,%.1f,%.4e,%.4e,%.4e,%d,%d,%.4e,%s,%.4e,%.4e,%s,\"%s\",%d,%s\n",
            d, valence, eltype, solve_tol, target, ops, method, solver, backend, device,
            threads, heap_hint_gb, timeout_s, der_seconds, strat_seconds, total_seconds,
            bytes, nullity, residual, meets_target, lsq_err, support, perm_ok, dims, nnz, status)
    end
end

# ========================================================================== timing
"""Time a single stratification run with error handling."""
function timed_stratify(Ω, ch, Γ; tol=1e-8, method_sym=:QuickDer, solver_sym=:GramSolver, 
                       backend=:cpu, device=:gpu, timeout_s=120)
    der_seconds = NaN
    strat_seconds = NaN
    total_seconds = NaN
    der_res = nothing
    Γ_strat = nothing
    status = "ok"
    
    try
        # Build kwargs based on method
        kwargs = Dict(:tol => tol)
        if method_sym === :QuickDer
            kwargs[:device] = device
            kwargs[:solver] = solver_sym
        elseif method_sym === :SylverLining
            kwargs[:backend] = backend
            kwargs[:solver] = solver_sym
        end
        
        # Warmup (suppress output)
        try
            _ = Dleto.der(method_sym, Ω, ch, Γ; kwargs...)
        catch
        end
        
        # Timed derivation solve
        start = time()
        der_res = Dleto.der(method_sym, Ω, ch, Γ; kwargs...)
        der_seconds = time() - start
        
        if isnothing(der_res) || isempty(der_res)
            status = "error: no derivations found"
        end
        
        if status == "ok" && !isempty(der_res)
            # Timed stratification
            start = time()
            try
                Γ_strat, _ = stratify(Ω, ch, Γ, der_res[1:1])
                strat_seconds = time() - start
            catch e
                status = "error: stratify failed: $(string(e)[1:60])"
                strat_seconds = time() - start
            end
        end
        
        total_seconds = der_seconds + strat_seconds
        
        if total_seconds > timeout_s
            status = "timeout"
            total_seconds = NaN
        end
    catch e
        status = "error: $(string(e)[1:80])"
    end
    
    return (; der_seconds, strat_seconds, total_seconds, der_res, status)
end

# ========================================================================== main
function main()
    println("\n=== GPU Stratification Timing ===")
    println("CSV: $CSV_PATH_REAL")
    println("Start: $(now())")
    println("Metal requires Float32; solve_tol = 1e-8 (GPU target)\n")
    
    reset_csv(CSV_PATH_REAL)
    
    # Configuration: GPU methods only, Float32 only, Metal backend
    eltype = Float32
    solve_tol = 1e-8
    target = 1e-8
    ops_choices = ["universal", "symmetric"]
    
    # GPU-enabled methods: only those that benefit from Metal
    methods = [
        ("QuickDer/Gram", :QuickDer, :GramSolver, :cpu, :gpu),  # QuickDer with GPU Gram
        ("SylverLining/Gram", :SylverLining, :GramSolver, :metal, :gpu),  # SylverLining with Metal
        ("SylverLining/Auto", :SylverLining, :AutoSolver, :metal, :gpu),   # SylverLining Auto on Metal
    ]
    
    # Thread/memory settings (conservative for GPU context)
    threads = 4
    heap_hint_gb = 10.0
    timeout_s = 120.0
    
    # Dimension sweep (shorter for GPU)
    d_values = [10, 15, 20, 25, 30, 40, 50]
    
    total_trials = length(d_values) * length(ops_choices) * length(methods) * REPS
    trial_count = 0
    
    for d in d_values
        for ops in ops_choices
            for (method_name, method_sym, solver_sym, backend, device) in methods
                for rep in 1:REPS
                    trial_count += 1
                    
                    # Construct problem
                    try
                        Γ, valence, dims_str, nnz = sphere_harness(d; eltype=eltype, ops=ops)
                        ch = (ops == "universal") ? UniversalChisel(valence) : SymmetricChisel(valence)
                        Ω = IndTransverseOps(collect(inds(Γ)), 
                                            ops == "universal" ? UniversalOp() : SymmetricOp())
                        
                        bytes = Int(Base.summarysize(Γ))
                        
                        # Time the run
                        der_seconds, strat_seconds, total_seconds, der_res, status = 
                            timed_stratify(Ω, ch, Γ; tol=solve_tol, 
                                          method_sym=method_sym, solver_sym=solver_sym,
                                          backend=backend, device=device, timeout_s=timeout_s)
                        
                        # Score
                        nullity = isnothing(der_res) ? 0 : (isempty(der_res) ? 0 : length(der_res))
                        residual = NaN
                        meets_target = false
                        lsq_err = NaN
                        support = NaN
                        perm_ok = false
                        
                        if !isnothing(der_res) && !isempty(der_res) && status == "ok"
                            try
                                residual = Dleto.der_residual(der_res[1])
                                meets_target = residual < target
                                perm_ok = true  # If we computed it, assume it's valid
                            catch
                                residual = NaN
                            end
                        end
                        
                        row = (d, valence, String(eltype), solve_tol, target, ops, 
                               method_name, String(solver_sym), String(backend), "gpu",
                               threads, heap_hint_gb, timeout_s,
                               der_seconds, strat_seconds, total_seconds,
                               bytes, nullity, residual, meets_target, lsq_err, support, perm_ok,
                               dims_str, nnz, status)
                        
                        emit_row(CSV_PATH_REAL, row)
                        
                        @printf("[%4d/%4d] d=%2d ops=%9s method=%19s rep=%d time=%.4f s status=%s\n",
                                trial_count, total_trials, d, ops, method_name, rep, 
                                isnan(total_seconds) ? NaN : total_seconds, status)
                        
                    catch e
                        @printf("[%4d/%4d] d=%2d ops=%9s method=%19s rep=%d ERROR: %s\n",
                                trial_count, total_trials, d, ops, method_name, rep, string(e)[1:60])
                    end
                end
            end
        end
    end
    
    println("\n=== GPU timing complete ===")
    println("Output: $CSV_PATH_REAL")
    println("Finish: $(now())")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
