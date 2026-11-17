#
# SylverMetalBench.jl -- `sylvesterLM` on 5 CPU threads against an Apple GPU.
#
# Two sections, selected on the command line so that one invocation stays
# inside a sensible wall-clock budget:
#
#   bench/jl bench/SylverMetalBench.jl apply [case...]   one apply, :array vs :metal
#   bench/jl bench/SylverMetalBench.jl e2e   [case...]    der(...) end to end
#   bench/jl bench/SylverMetalBench.jl list               names of the known cases
#
# Cases are named by shape: `100x100x100x3`, `200^3`, ... (see `CASES`).  With
# no case named, `apply` runs all of them and `e2e` runs the two video shapes
# the night's brief asks for.  Rows are APPENDED to
# bench/reports/night-2026-09-03/sylver-metal.csv (`--csv=` to redirect),
# header written only when the file does not exist yet.
#
# Everything is Float32: Apple GPUs have no fp64, so a `:metal` run is a
# Float32 run by construction and the CPU column is Float32 for a fair fight.
#
using Dleto
using ITensors
using LinearAlgebra
using LinearMaps
using Metal
using Printf
using Random

# `:ArpackSolver` lives in an extension, and the extension only registers once
# Arpack itself is loaded.  Arpack is a weakdep the project cannot carry, so it
# comes from the shared `@dleto-bench` environment that `bench/jl` stacks --
# soft-loaded here so a machine without it reports a clean error per row
# instead of failing at the top of the file.
const HAVE_ARPACK = try
    @eval using Arpack
    true
catch
    @warn "Arpack not available; :ArpackSolver rows will error"
    false
end

const OUTDIR = joinpath(@__DIR__, "reports", "night-2026-09-03")
const T = Float32

# name => tensor shape.  The 3-long last axis is a colour video's channel axis;
# `d^3` cases are the valence-3 reference points.
const CASES = [
    "100x100x100x3" => (100, 100, 100, 3),
    "200x200x100x3" => (200, 200, 100, 3),
    "300x300x300x3" => (300, 300, 300, 3),
    "200^3"         => (200, 200, 200),
    "400^3"         => (400, 400, 400),
]

const E2E_CASES = ["100x100x100x3", "200x200x100x3"]

# --- the Z-law, copied rather than `include`d -------------------------------
# test/TestDerivationLaws.jl defines `der_residual` inside a file that also
# runs @testsets on load, so a bench cannot include it; this is the same three
# lines (see that file's docstring).
function der_residual(Γ::ITensor, D::Vector{ITensor}, P::AbstractMatrix)
    C = Chisel(P, collect(inds(Γ)))
    R = applyDerivation(Γ, D, C)
    scale = norm(Γ) * maximum(norm.(D))
    return norm(R) / max(scale, eps())
end

# --- device bookkeeping ------------------------------------------------------
device_mb() = Metal.device().currentAllocatedSize / 2^20
rss_mb() = Sys.maxrss() / 2^20
function reclaim!()
    GC.gc(true)
    # Metal.jl frees a buffer in the MtlArray finalizer but keeps it in its own
    # pool; `reclaim` hands it back to the driver, which is the only way the
    # per-case device numbers below mean anything.
    isdefined(Metal, :reclaim) && Metal.reclaim()
    return nothing
end

# --- CSV ---------------------------------------------------------------------
const COLS = ["section", "case", "dims", "valence", "backend", "what",
              "seconds", "bytes", "applies", "nullity", "residual",
              "device_mb", "rss_mb", "note"]

function emit(path, row::Dict)
    fresh = !isfile(path)
    open(path, "a") do io
        fresh && println(io, join(COLS, ","))
        println(io, join([string(get(row, c, "")) for c in COLS], ","))
    end
end

# --- one apply ---------------------------------------------------------------

# The box is shared, and the night board already documents identical code
# measuring anywhere from 20 s to 935 s depending on who else is running.  The
# only defence available from inside the process is the MINIMUM over as many
# repeats as the budget allows -- a contended sample can only be slower, never
# faster, so the minimum converges on the uncontended time from above.  The
# load average at the moment of measurement goes in the CSV so a reader can
# see how much to trust each row.
loadavg1() = try
    parse(Float64, split(read(`sysctl -n vm.loadavg`, String), r"[{} ]+";
                         keepempty = false)[1])
catch
    NaN
end

"""
    best_of(f; budget, maxreps, minreps)

Minimum elapsed time of `f()` over repeats, stopping when either `maxreps` runs
or `budget` seconds of wall clock are used up, whichever comes first (but never
before `minreps`).
"""
function best_of(f; budget::Float64 = 6.0, maxreps::Int = 12, minreps::Int = 3)
    best = Inf
    spent = 0.0
    k = 0
    while k < maxreps && (k < minreps || spent < budget)
        t = @elapsed f()
        best = min(best, t); spent += t; k += 1
    end
    return best, k
end

"""
    time_applies(Ω, ch, Γ, backend; budget, maxreps)

Best-of-many seconds for one `ester`, one `sylve` and one `sylvester` apply
through the CPU-facing LinearMaps, plus the bytes a warmed-up apply allocates
and what the device gained while the maps existed.

`mul!`, not `*`: these are `ismutating = true` maps, and `*` would allocate the
output every time and measure the allocator instead of the kernel.
"""
function time_applies(Ω, ch, Γ, backend::Symbol; budget::Float64 = 6.0,
                      maxreps::Int = 12)
    dev0 = device_mb()
    derm, denm = sylvesterLM(Ω, ch, Γ; backend = backend)
    devbuilt = device_mb() - dev0
    rows, cols = size(denm)
    x = randn(T, cols); z = similar(x)
    y = Vector{T}(undef, rows); w = randn(T, rows)
    out = Dict{Symbol,Any}()
    for (name, f) in ((:ester, () -> mul!(y, denm, x)),
                      (:sylve, () -> mul!(z, denm', w)),
                      (:sylvester, () -> mul!(z, derm, x)))
        f()                                     # warm up / compile
        best, k = best_of(f; budget = budget, maxreps = maxreps)
        bytes = @allocated f()
        out[name] = (best, bytes, k, loadavg1())
    end
    derm = nothing; denm = nothing
    x = nothing; y = nothing; z = nothing; w = nothing
    reclaim!()
    return out, devbuilt
end

function run_apply(csv, names)
    @printf("%-16s %-8s %-11s %10s %10s %10s %10s %4s %6s\n",
            "case", "backend", "map", "ms", "bytes", "dev MB", "rss MB",
            "reps", "load")
    for nm in names
        dims = CASES[findfirst(p -> first(p) == nm, CASES)][2]
        val = length(dims)
        fr = [Index(dims[a], "a$a") for a in 1:val]
        Random.seed!(20260904)
        Γ = ITensor(randn(T, dims), fr...)
        Ω = IndTransverseOps(fr, UniversalOp())
        ch = UniversalChisel(val)
        for backend in (:array, :metal)
            big = prod(dims) > 2 * 10^7
            res, devmb = time_applies(Ω, ch, Γ, backend;
                                      budget = big ? 12.0 : 6.0,
                                      maxreps = big ? 6 : 12)
            for what in (:ester, :sylve, :sylvester)
                (secs, bytes, k, la) = res[what]
                @printf("%-16s %-8s %-11s %10.3f %10d %10.1f %10.1f %4d %6.1f\n",
                        nm, backend, what, 1000 * secs, bytes, devmb, rss_mb(),
                        k, la)
                emit(csv, Dict("section" => "apply", "case" => nm,
                               "dims" => join(dims, "x"), "valence" => val,
                               "backend" => backend, "what" => what,
                               "seconds" => secs, "bytes" => bytes,
                               "device_mb" => round(devmb; digits = 1),
                               "rss_mb" => round(rss_mb(); digits = 1),
                               "note" => "min-of-$k loadavg=$(round(la; digits=1))"))
            end
        end
        Γ = nothing; reclaim!()
    end
end

# --- end to end --------------------------------------------------------------

"""
    count_applies(f)

Run `f(progress)` with a progress spec that reports every tick into a buffer,
and pull the application count back out of the last line it printed.

The tracker's count is private to `solve_nullspace`, but it is printed, and
`progress` accepts a `ProgressSpec` -- so `delay = 0` and `interval = 0` turn
the human-facing report into a machine-readable one without touching `src/`.
"""
function count_applies(f)
    buf = IOBuffer()
    spec = progress_spec(true; io = buf, interval = 0.0, delay = 0.0)
    val = f(spec)
    txt = String(take!(buf))
    n = 0
    for mm in eachmatch(r"(\d+) applications", txt)
        n = max(n, parse(Int, mm.captures[1]))
    end
    for mm in eachmatch(r"(\d+)/(\d+) \(", txt)          # densifying stage
        n = max(n, parse(Int, mm.captures[1]))
    end
    return val, n
end

function run_e2e(csv, names; solver::Symbol = :ArpackSolver, tol::Real = 1e-4)
    @printf("%-16s %-8s %10s %8s %11s %8s %9s %9s\n",
            "case", "backend", "seconds", "nullity", "residual", "applies",
            "dev MB", "rss MB")
    for nm in names
        dims = CASES[findfirst(p -> first(p) == nm, CASES)][2]
        val = length(dims)
        fr = [Index(dims[a], "a$a") for a in 1:val]
        Random.seed!(20260904)
        Γ = ITensor(randn(T, dims), fr...)
        Ω = IndTransverseOps(fr, UniversalOp())
        ch = UniversalChisel(val)
        for backend in (:array, :metal)
            dev0 = device_mb()
            local D, napp, secs
            try
                secs = @elapsed ((D, napp) = count_applies(sp ->
                    der(:SylverLining, Ω, ch, Γ; tol = tol, solver = solver,
                        backend = backend, progress = sp)))
            catch err
                msg = replace(sprint(showerror, err), "," => ";")[1:min(end, 120)]
                @printf("%-16s %-8s %10s %8s %11s %8s %9s %9s  ERROR %s\n",
                        nm, backend, "-", "-", "-", "-", "-", "-", msg)
                emit(csv, Dict("section" => "e2e", "case" => nm,
                               "dims" => join(dims, "x"), "valence" => val,
                               "backend" => backend, "what" => solver,
                               "note" => "ERROR " * msg))
                continue
            end
            devmb = device_mb() - dev0
            resid = isempty(D) ? NaN :
                    maximum(der_residual(Γ, Dk, ch) for Dk in D)
            @printf("%-16s %-8s %10.2f %8d %11.3e %8d %9.1f %9.1f\n",
                    nm, backend, secs, length(D), resid, napp, devmb, rss_mb())
            emit(csv, Dict("section" => "e2e", "case" => nm,
                           "dims" => join(dims, "x"), "valence" => val,
                           "backend" => backend, "what" => solver,
                           "seconds" => round(secs; digits = 3),
                           "applies" => napp, "nullity" => length(D),
                           "residual" => resid,
                           "device_mb" => round(devmb; digits = 1),
                           "rss_mb" => round(rss_mb(); digits = 1),
                           "note" => "tol=$tol"))
            D = nothing; reclaim!()
        end
        Γ = nothing; reclaim!()
    end
end

# --- main --------------------------------------------------------------------

function main(args)
    csv = joinpath(OUTDIR, "sylver-metal.csv")
    rest = String[]
    solver = :ArpackSolver
    tol = 1e-4
    for a in args
        if startswith(a, "--csv=")
            csv = a[7:end]
        elseif startswith(a, "--solver=")
            solver = Symbol(a[10:end])
        elseif startswith(a, "--tol=")
            tol = parse(Float64, a[7:end])
        else
            push!(rest, a)
        end
    end
    isempty(rest) && (rest = ["apply"])
    section = rest[1]
    names = length(rest) > 1 ? rest[2:end] : String[]
    mkpath(dirname(csv))

    if section == "list"
        for (nm, dims) in CASES
            @printf("%-16s %s\n", nm, join(dims, "x"))
        end
        return
    end
    println("Metal: ", Metal.functional() ? Metal.device().name : "NOT functional",
            " | gpu_available = ", Dleto.gpu_available(),
            " | julia threads = ", Threads.nthreads(),
            " | BLAS threads = ", BLAS.get_num_threads())
    if section == "apply"
        run_apply(csv, isempty(names) ? first.(CASES) : names)
    elseif section == "e2e"
        run_e2e(csv, isempty(names) ? E2E_CASES : names; solver = solver, tol = tol)
    else
        error("SylverMetalBench: unknown section $section (apply | e2e | list)")
    end
    println("\nwrote $csv")
end

main(ARGS)
