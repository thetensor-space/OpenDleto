#
# D500Matrix -- dense vs sparse, CPU vs GPU, Float64 vs Float32, at d = 500,
# valence 3, whitened (the default), :AutoSolver (the default), one process
# per case (bench/jl convention).
#
# Answers: "I'd like to see the numbers for d=500. Dense vs sparse, CPU vs GPU
# memory usage, time."  DENSE means the scrambled sphere octant
# (bench/SphereHarness.jl build_sphere -- orthogonal scramble + nondeg, a
# genuinely dense array).  SPARSE-STRUCTURED means the raw, unscrambled
# sphere_octant: mostly zeros (nonzero only on the i_1+...+i_n = d-1 lattice)
# but still stored as a dense Array{T} -- same code path, same memory formula,
# fewer allocations in the BUILD step (one tensor, not build_sphere's three).
# This mirrors bench/Frontier.jl's own `sparse-cpu` task (raw sphere_octant,
# UniversalOp, UniversalChisel) rather than inventing a new convention.
#
# Every d = 500 cell here lands in the MATRIX-FREE branch without forcing it:
# dense_bytes = R * sum_dr * sizeof(T) is ~30.7 GB at Float64 and ~15.4 GB at
# Float32 (R = 64000, sum_dr = 60000, from _qdn_restriction_sizes), which is
# over both QDN_DENSE_BUDGET_BYTES (2.5 GB, CPU) and QDN_GPU_DENSE_BUDGET_BYTES
# (6 GB, GPU) -- so these are the numbers the PRODUCTION default gives at this
# size, not an artificially-forced comparison.
#
# GPU memory is read from Metal.device().currentAllocatedSize, polled on a
# background thread (Threads.@spawn) every 20 ms around the timed call, and
# reported as a peak -- Metal.jl 1.x has no peak-allocated query, only current.
#
# Usage (one case per command):
#   bench/jl bench/D500Matrix.jl estimate
#   bench/jl bench/D500Matrix.jl dense-baseline           # transcribes the
#       already-run Float64 CPU row from whitened.csv; does not run Julia work
#   bench/jl bench/D500Matrix.jl dense  Float32 cpu
#   bench/jl bench/D500Matrix.jl dense  Float32 gpu
#   bench/jl bench/D500Matrix.jl sparse Float64 cpu
#   bench/jl bench/D500Matrix.jl sparse Float32 cpu
#   bench/jl bench/D500Matrix.jl sparse Float32 gpu
#   bench/jl bench/D500Matrix.jl sparse-storage Float64 cpu
#       Optional: wrap the raw sphere in a SparseArrays-backed ITensor (a
#       reshaped SparseMatrixCSC through a combiner) and report whether
#       QuickDer accepts it and what memory that storage takes vs dense.
#
# -> bench/reports/2026-09-04/d500/d500.csv
#
using Arpack
using Dleto
using ITensors
using IterativeSolvers
using LinearAlgebra
using LinearMaps
using Logging
using Printf
using Random
using SparseArrays

try
    using Metal
catch err
    @warn "Metal is not loadable; :gpu cells will error." err
end

include(joinpath(@__DIR__, "Frontier.jl"))   # der_residual, csv_header, branch_info,
                                              # THREAD_NOTE, and (via SphereHarness.jl)
                                              # build_sphere, sphere_octant, quietly

const HAVE_GPU = Dleto.gpu_available()

const DREPORT_DIR = joinpath(@__DIR__, "reports", "2026-09-04", "d500")
mkpath(DREPORT_DIR)
const DCSV = joinpath(DREPORT_DIR, "d500.csv")

const DHEADER = "case,dims,valence,eltype,whiten,solver,r,restricted_rows,restricted_cols," *
                "applies,seconds,solve_seconds,whiten_seconds,sketch_seconds,lift_seconds," *
                "maxrss_GB,bytes,nullity,oracle_nullity,residual,uncertified," *
                "trivial_reinjected,status,device,gpu_mem_gb,storage"

# --------------------------------------------------------------- GPU polling

"""
    gpu_poll_start() -> (peak::Ref{Int}, stop::Ref{Bool}, task_or_nothing)

Start a background poller of `Metal.device().currentAllocatedSize` if a GPU is
available, else return a poller that never runs.  Stop with `gpu_poll_stop!`.
"""
function gpu_poll_start()
    peak = Ref(0)
    stop = Ref(false)
    if !HAVE_GPU
        return (peak, stop, nothing)
    end
    dev = Metal.device()
    t = Threads.@spawn begin
        while !stop[]
            try
                v = Int(dev.currentAllocatedSize)
                v > peak[] && (peak[] = v)
            catch
            end
            sleep(0.02)
        end
    end
    return (peak, stop, t)
end

function gpu_poll_stop!(poll)
    peak, stop, t = poll
    stop[] = true
    t === nothing || wait(t)
    return peak[]
end

# --------------------------------------------------------------- the timed call

"""
    timed_quickder2(Ω, ch, Γ; whiten, solver, device, tol) -> NamedTuple

Same measurement as `bench/WhitenedRestriction.jl`'s `timed_quickder`, plus a
`device` kwarg forwarded to `QuickDerMethod` and, for `device = :gpu`, a peak
Metal allocation reading around the call.
"""
function timed_quickder2(Ω, ch, Γ; whiten::Bool, solver::Symbol, device::Symbol = :cpu,
                         tol::Real = 1e-6)
    method = Dleto.get_derivation_method(:QuickDer; whiten = whiten, solver = solver,
                                         device = device, verify = :random, seed = 20260904)
    io = IOBuffer()
    logger = Logging.ConsoleLogger(io, Logging.Info)
    status = "ok"
    nullity = 0
    resid = NaN
    stages = Dict{Symbol,Float64}()
    Dleto.QDN_STAGE_TIMES[] = stages
    Dleto.QDN_APPLY_COUNT[] = 0
    local ders, expand_map
    GC.gc()
    poll = gpu_poll_start()
    st = @timed try
        Logging.with_logger(logger) do
            redirect_stdout(devnull) do
                (_, expand_map, ders) = Dleto.derTrOpsReduced(method, Ω, ch, Γ; tol = tol)
            end
        end
        nullity = size(ders, 2)
        if nullity > 0
            D = embedITensors(Ω, expand_map(ders[:, 1]))
            resid = der_residual(Γ, D, ch)
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
    finally
        # stopped whether or not the call errored
    end
    peak_bytes = gpu_poll_stop!(poll)
    applies = Dleto.QDN_APPLY_COUNT[]
    Dleto.QDN_APPLY_COUNT[] = -1
    Dleto.QDN_STAGE_TIMES[] = nothing
    logtxt = String(take!(io))
    return (; seconds = st.time, bytes = st.bytes, applies, nullity, resid, status,
              uncertified = occursin("UNCERTIFIED", logtxt),
              trivial = occursin("rank-deficient mode", logtxt),
              solve_s = get(stages, :solve, NaN), whiten_s = get(stages, :whiten, 0.0),
              sketch_s = get(stages, :sketch, NaN), lift_s = get(stages, :lift, NaN),
              upload_s = get(stages, :upload, NaN),
              maxrss_GB = Sys.maxrss() / 2^30,
              gpu_mem_GB = device === :gpu ? peak_bytes / 2^30 : NaN,
              stages = stages)
end

function record!(case, dims, r, T, whiten, solver, oracle, device, storage, res)
    csv_header(DCSV,
        "# d = 500 matrix: dense (scrambled sphere) vs sparse-structured (raw sphere)," *
        " CPU vs GPU, Float64 vs Float32.  Every cell lands in the matrix-free branch " *
        "without forcing it (dense_bytes ~15-31 GB is over both dense budgets at d=500).",
        DHEADER)
    n = length(dims)
    eng = Dleto.engaged(Matrix{Float64}(UniversalChisel(n)))
    rows = prod(r)
    cols = sum(Int[dims[a] * r[a] for a in 1:n if eng[a]])
    @printf("%-24s T=%-7s dev=%-4s storage=%-7s applies=%-7d seconds=%8.2f solve=%8.2f \
whiten_s=%6.3f maxrss=%6.2fGB gpu_mem=%6.2fGB nullity=%d(oracle=%d) resid=%.2e \
uncert=%s status=%s\n",
            case, T, device, storage, res.applies, res.seconds, res.solve_s, res.whiten_s,
            res.maxrss_GB, res.gpu_mem_GB, res.nullity, oracle, res.resid, res.uncertified,
            res.status)
    if device === :gpu
        println("    stage breakdown: ", res.stages)
    end
    flush(stdout)
    open(DCSV, "a") do io
        @printf(io, "%s,\"%s\",%d,%s,%d,%s,\"%s\",%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,\
%.6f,%d,%d,%d,%.6e,%s,%s,\"%s\",%s,%.6f,%s\n",
                case, string(dims), n, T, whiten, solver, string(r), rows, cols,
                res.applies, res.seconds, res.solve_s, res.whiten_s, res.sketch_s,
                res.lift_s, res.maxrss_GB, res.bytes, res.nullity, oracle, res.resid,
                res.uncertified, res.trivial, res.status, device, res.gpu_mem_GB, storage)
    end
    return nothing
end

# --------------------------------------------------------------- cases

"""Dense: the scrambled sphere octant (build_sphere), matching whitened.csv."""
function dense_case(T::Type, device::Symbol; valence::Int = 3, d::Int = 500,
                    solver::Symbol = :AutoSolver, whiten::Bool = true)
    # warm up on a small dense case at the same T/device/solver
    let wd = valence == 3 ? 10 : 6
        inp = build_sphere(wd; valence, T)
        try
            timed_quickder2(inp.Ω, inp.ch, inp.Γ; whiten, solver, device)
        catch e
            @warn "warmup failed (continuing)" exception = (e, catch_backtrace())
        end
    end
    inp = build_sphere(d; valence, T)
    dims = collect(inp.dims)
    r = Dleto._qdn_restriction_sizes(dims, Dleto.engaged(Matrix{Float64}(inp.ch)), valence)
    res = timed_quickder2(inp.Ω, inp.ch, inp.Γ; whiten, solver, device)
    record!("sphere-v$valence-d$d-dense", dims, r, T, whiten, solver, valence, device,
            "dense", res)
    return nothing
end

"""
Sparse-structured: the raw, unscrambled sphere_octant, mostly zeros but stored
as a dense Array{T} -- the shape bench/Frontier.jl's `sparse-cpu` task uses
(UniversalOp, UniversalChisel), extended here with T and device.
"""
function sparse_case(T::Type, device::Symbol; valence::Int = 3, d::Int = 500,
                     solver::Symbol = :AutoSolver, whiten::Bool = true)
    let wd = valence == 3 ? 12 : 8
        Sw = sphere_octant(wd; valence)
        frw = collect(inds(Sw))
        Γw = T === Float64 ? Sw : ITensor(Array{T}(Array(Sw, frw...)), frw...)
        try
            timed_quickder2(IndTransverseOps(frw, UniversalOp()), UniversalChisel(valence),
                            Γw; whiten, solver, device)
        catch e
            @warn "warmup failed (continuing)" exception = (e, catch_backtrace())
        end
    end
    S = sphere_octant(d; valence)
    fr = collect(inds(S))
    Γ = T === Float64 ? S : ITensor(Array{T}(Array(S, fr...)), fr...)
    ch = UniversalChisel(valence)
    Ω = IndTransverseOps(fr, UniversalOp())
    dims = collect(ITensors.dim.(fr))
    r = Dleto._qdn_restriction_sizes(dims, Dleto.engaged(Matrix{Float64}(ch)), valence)
    res = timed_quickder2(Ω, ch, Γ; whiten, solver, device)
    record!("sphere-v$valence-d$d-sparse", dims, r, T, whiten, solver, valence, device,
            "sparse-structured", res)
    return nothing
end

"""
Transcribe the already-run Float64 CPU dense row from
bench/reports/2026-09-04/whitened/whitened.csv (sphere-v3-d500, whiten=1,
AutoSolver) into d500.csv with the extra columns filled in -- per the brief,
this one is NOT rerun.
"""
function dense_baseline()
    dims = [500, 500, 500]
    r = [40, 40, 40]
    res = (; applies = 51116, seconds = 136.134947, solve_s = 129.641613,
             whiten_s = 0.075688, sketch_s = 0.247180, lift_s = 2.965739,
             maxrss_GB = 11.365799, bytes = 592198127392, nullity = 3, resid = 3.175339e-13,
             uncertified = false, trivial = false, status = "ok", gpu_mem_GB = NaN,
             stages = Dict{Symbol,Float64}())
    record!("sphere-v3-d500-dense", dims, r, Float64, true, :AutoSolver, 3, :cpu, "dense",
            res)
    println("(transcribed from bench/reports/2026-09-04/whitened/whitened.csv -- not rerun)")
    return nothing
end

"""
Optional (30-min-effort) probe: does QuickDer accept a SparseArrays-backed
ITensor?  Reshapes the raw sphere into a 2-index SparseMatrixCSC (fold axes
2..n into one combined index via an ITensors combiner), wraps it in an ITensor,
and reports whether `derTrOpsReduced` runs on it or errors -- and, either way,
the memory the sparse storage itself takes vs an equivalent dense Array.
"""
function sparse_storage_case(T::Type, device::Symbol; valence::Int = 3, d::Int = 500,
                             solver::Symbol = :AutoSolver, whiten::Bool = true)
    device === :cpu || error("sparse-storage is CPU only (T = $T, device = $device)")
    S = sphere_octant(d; valence)
    fr = collect(inds(S))
    A = Array(S, fr...)
    nnz_ = count(!=(0.0), A)
    dense_bytes = length(A) * sizeof(T)
    sparse_bytes_est = nnz_ * (sizeof(T) + sizeof(Int)) + (d + 1) * sizeof(Int)  # CSC-ish
    @printf("sparse-storage: d=%d valence=%d nnz=%d / %d entries (%.4f%% dense) \
dense=%.3fGB sparse_est=%.3fGB\n",
            d, valence, nnz_, length(A), 100 * nnz_ / length(A), dense_bytes / 2^30,
            sparse_bytes_est / 2^30)

    c = combiner(fr[2:end]...)
    Mflat = reshape(A, size(A, 1), :)
    Msp = sparse(Mflat)
    ci = combinedind(c)
    Γsp = try
        ITensor(Msp, fr[1], ci) * c
    catch e
        @warn "could not build a sparse-backed ITensor directly; report and stop" exception = (e, catch_backtrace())
        nothing
    end
    accepted = Γsp !== nothing
    storage_kind = accepted ? string(typeof(ITensors.storage(Γsp))) : "n/a (build failed)"
    println("Γsp storage type (after combiner contraction): ", storage_kind)

    if !accepted
        open(DCSV, "a") do io
            csv_header(DCSV,
                "# d = 500 matrix: dense (scrambled sphere) vs sparse-structured (raw sphere)," *
                " CPU vs GPU, Float64 vs Float32.  Every cell lands in the matrix-free branch " *
                "without forcing it (dense_bytes ~15-31 GB is over both dense budgets at d=500).",
                DHEADER)
            r = Dleto._qdn_restriction_sizes(collect(ITensors.dim.(fr)),
                                             Dleto.engaged(Matrix{Float64}(UniversalChisel(valence))),
                                             valence)
            @printf(io, "%s,\"%s\",%d,%s,%d,%s,\"%s\",%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,\
%.6f,%d,%d,%d,%.6e,%s,%s,\"%s\",%s,%.6f,%s\n",
                    "sphere-v$valence-d$d-sparsestorage", string(collect(ITensors.dim.(fr))),
                    valence, T, whiten, solver, string(r), 0, 0, 0, NaN, NaN, NaN, NaN, NaN,
                    Sys.maxrss() / 2^30, 0, 0, valence, NaN, false, false,
                    "not attempted: ITensor storage densifies on construction", :cpu, NaN,
                    "sparse-itensor")
        end
        return nothing
    end

    ch = UniversalChisel(valence)
    Ω = IndTransverseOps(fr, UniversalOp())
    dims = collect(ITensors.dim.(fr))
    rr = Dleto._qdn_restriction_sizes(dims, Dleto.engaged(Matrix{Float64}(ch)), valence)
    res = timed_quickder2(Ω, ch, Γsp; whiten, solver, device)
    record!("sphere-v$valence-d$d-sparsestorage", dims, rr, T, whiten, solver, valence,
            device, "sparse-itensor", res)
    return nothing
end

# --------------------------------------------------------------- estimate

function estimate()
    d = 500
    for valence in (3,)
        dims = fill(d, valence)
        eng = Dleto.engaged(Matrix{Float64}(UniversalChisel(valence)))
        for T in (Float64, Float32)
            bi = branch_info(dims, eng, valence; T = T)
            dense_tensor_GB = 3 * prod(float.(dims)) * sizeof(T) / 2^30   # build_sphere: 3 copies
            sparse_tensor_GB = prod(float.(dims)) * sizeof(T) / 2^30      # raw: 1 copy
            @printf("valence=%d T=%-7s r=%-14s dense_bytes(system)=%.2fGB branch=%s \
dense_tensor(build_sphere)~%.2fGB sparse_tensor(raw)~%.2fGB\n",
                    valence, T, string(bi.r), bi.dense_bytes / 2^30, bi.branch,
                    dense_tensor_GB, sparse_tensor_GB)
        end
    end
    println("CPU dense budget = ", Dleto.QDN_DENSE_BUDGET_BYTES[] / 2^30, " GB; ",
            "GPU dense budget = ", Dleto.QDN_GPU_DENSE_BUDGET_BYTES[] / 2^30, " GB")
    println("HAVE_GPU = ", HAVE_GPU)
    return nothing
end

# --------------------------------------------------------------- CLI

function dmain(args)
    isempty(args) && error("first argument must be estimate, dense-baseline, dense, " *
                           "sparse, or sparse-storage")
    task = args[1]
    if task == "estimate"
        estimate()
    elseif task == "dense-baseline"
        dense_baseline()
    elseif task == "dense"
        T = args[2] == "Float32" ? Float32 : Float64
        dense_case(T, Symbol(args[3]))
    elseif task == "sparse"
        T = args[2] == "Float32" ? Float32 : Float64
        sparse_case(T, Symbol(args[3]))
    elseif task == "sparse-storage"
        T = args[2] == "Float32" ? Float32 : Float64
        sparse_storage_case(T, Symbol(args[3]))
    else
        error("unknown task $task")
    end
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && dmain(ARGS)
