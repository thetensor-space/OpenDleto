#
# QuickDerScaling -- scaling sweeps for :QuickDer (src/solvers/QuickDerN.jl)
# at valence 4 and on video-shaped tensors, always checked against a
# correctness oracle (nullity and/or the Z-law residual).
#
# NOTE ON PROCESS COST.  A fresh `bench/jl` process pays a large *fixed* JIT/
# precompilation bill before any timed work starts -- measured ~580s here
# (Dleto, ITensors, SylverLining, the Arpack/KrylovKit/IterativeSolvers
# extensions, QuickDerN).  Running one size per process (as the board's
# "rules of the road" suggest for the SylverLining sweeps, where each point
# is itself slow) would mean paying that ~10 minute tax dozens of times.
# Instead each task below takes a comma-separated LIST of sizes and loops
# them inside ONE process, so the JIT tax is paid once; the whole invocation
# is then run via `run_in_background` (never in an 8-minute-capped foreground
# call) and polled. Each size's own timing is still what gets recorded and
# compared -- only the process-startup cost is shared. A sweep still stops
# itself at the first point whose QuickDer time exceeds 180s (3 minutes), per
# the assignment's stopping rule; it does not attempt the next larger size in
# its list after that.
#
# Three independent regimes, one CSV each:
#
#   v4dense  bench/jl bench/QuickDerScaling.jl v4dense --d=10,16,20,24,30,40,50,60,80 [--tol=1e-8]
#            Dense scrambled hypersphere octant, valence 4, STRATIFICATION
#            (bench/SphereHarness.jl `build_sphere`/`run_stratify`).
#            Runs :QuickDer restriction=:random and restriction=:corner
#            (SymmetricOp, verify=:random) at every d; also runs
#            :SylverLining/:ArpackSolver as a reference for d <= 30.
#            -> quickder-v4-dense.csv
#
#   video    bench/jl bench/QuickDerScaling.jl video --sizes=30,50,64,80,100 --T=Float64
#            Video-shaped dense random tensors `randn(T,H,W,F,3)` (H=W=F, one
#            per `--sizes` entry), DERIVATION only (no stratify), UniversalOp
#            + UniversalChisel(4), via `Dleto.derTrOpsReduced(
#            get_derivation_method(:QuickDer; verify=:random), Ω, ch, Γ;
#            tol=1e-6)`.  Records seconds, bytes, Sys.maxrss(), nullity
#            (oracle 3), and the Z-law residual of one returned basis element
#            (der_residual, see test/TestDerivationLaws.jl), plus the
#            restriction sizes chosen by `Dleto._qdn_restriction_sizes`.  Run
#            once per element type (Float64, Float32) -- separate processes,
#            so each is warmed up on its own type.
#            -> quickder-video.csv
#
#   sparse   bench/jl bench/QuickDerScaling.jl sparse --valence=4 --d=10,20,30,40,60,80
#            Raw (unscrambled) hypersphere octant `sphere_octant(d;valence)`,
#            DERIVATION only, UniversalOp.  Oracle nullity 13.  Runs
#            restriction=:random and :corner at every d.
#            -> quickder-sparse.csv
#
# Every task warms up once, locally, on a small size before its size loop.
#
using Arpack
using LinearAlgebra
using Printf
using Random
using Dleto
using ITensors
include(joinpath(@__DIR__, "SphereHarness.jl"))

try
    @eval using Arpack
catch e
    @warn "Arpack did not load; :ArpackSolver rows will report the registry's " *
          "\"unavailable\" error instead of a real timing" exception = e
end

BLAS.set_num_threads(Threads.nthreads())

const REPORT_DIR = joinpath(@__DIR__, "reports", "night-2026-09-03")
const SWEEP_STOP_S = 180.0   # stop a sweep at the first point over 3 minutes

# ---------------------------------------------------------------- helpers

"""
    der_residual(Γ, D, P) -> Real

Copied from test/TestDerivationLaws.jl (not `include`d -- this is a bench
script, not a test file).  Relative size of the chisel-weighted sum
Σ_a P[:,a] ⊗ (Γ · D_a).  Zero exactly when D is a P-derivation of Γ.
"""
function der_residual(Γ::ITensor, D::Vector{ITensor}, P::AbstractMatrix)
    C = Chisel(P, collect(inds(Γ)))
    R = applyDerivation(Γ, D, C)
    scale = norm(Γ) * maximum(norm.(D))
    return norm(R) / max(scale, eps())
end

getflag(name, default::T) where {T} =
    let a = filter(startswith("--$name="), ARGS)
        isempty(a) ? default : (T <: AbstractString ? String(a[1][(length(name) + 4):end]) :
                                 parse(T, a[1][(length(name) + 4):end]))
    end
gettype(name, default::Type) =
    let a = filter(startswith("--$name="), ARGS)
        isempty(a) ? default : (a[1][(length(name) + 4):end] == "Float32" ? Float32 : Float64)
    end
getlist(name, default::Vector{Int}) =
    let a = filter(startswith("--$name="), ARGS)
        isempty(a) ? default : parse.(Int, split(a[1][(length(name) + 4):end], ","))
    end
const TASK = isempty(ARGS) ? error("first argument must be v4dense, video, or sparse") :
             ARGS[1]

csv_header(path, header) = isfile(path) || open(path, "w") do io
    println(io, "# 2-thread timings (bench/jl pins 2 Julia + 2 OpenBLAS threads); may be")
    println(io, "# noisy if another agent held the other bench/jl slot concurrently")
    println(io, header)
end

"""One derivation-only QuickDer call, timed, with the Z-law checked on one
basis element.  Returns (; seconds, bytes, maxrss, nullity, resid, status)."""
function timed_der(Ω, ch, Γ; tol, restriction, verify = :random)
    method = get_derivation_method(:QuickDer; restriction, verify)
    status = "ok"
    nullity = 0
    resid = NaN
    GC.gc()
    local ders, expand_map
    st = @timed try
        quietly() do
            (_, expand_map, ders) = Dleto.derTrOpsReduced(method, Ω, ch, Γ; tol = tol)
        end
        nullity = size(ders, 2)
        if nullity > 0
            D = embedITensors(Ω, expand_map(ders[:, 1]))
            resid = der_residual(Γ, D, ch)
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
    end
    return (; seconds = st.time, bytes = st.bytes, maxrss = Sys.maxrss(), nullity, resid,
              status)
end

# ---------------------------------------------------------------- v4dense

function run_v4dense()
    ds   = getlist("d", [10, 16, 20, 24, 30, 40, 50, 60, 80])
    tol  = getflag("tol", 1e-8)
    csv  = joinpath(REPORT_DIR, "quickder-v4-dense.csv")
    csv_header(csv, "d,seed,valence,config,seconds,bytes,nullity,lsq_err,perm_ok,status")

    println("v4dense d=$ds tol=$tol (2-thread, bench/jl)")

    warm_configs = [
        (; method = :QuickDer, restriction = :random, verify = :random),
        (; method = :QuickDer, restriction = :corner, verify = :random),
        (; method = :SylverLining, solver = :ArpackSolver),
    ]
    print("  warm-up (d=8, valence 4) ... "); flush(stdout)
    tw = @elapsed warmup!(warm_configs; d = 8, valence = 4, passes = 2)
    @printf("done (%.1fs, not counted)\n", tw)

    for d in ds
        seed = d
        inp = build_sphere(d; valence = 4, seed)
        configs = Tuple{String, NamedTuple}[
            ("QuickDer/random", (; method = :QuickDer, restriction = :random, verify = :random)),
            ("QuickDer/corner", (; method = :QuickDer, restriction = :corner, verify = :random)),
        ]
        d <= 30 && push!(configs,
            ("SylverLining/Arpack", (; method = :SylverLining, solver = :ArpackSolver)))

        println("== d = $d ==")
        worst_qd = 0.0
        for (label, kw) in configs
            r = run_stratify(inp; kw..., tol = tol)
            @printf("  %-22s %9.3fs  %9.2f MB  null=%2d  lsq_err=%9.2e  perm_ok=%s  %s\n",
                    label, r.seconds, r.bytes / 2^20, r.nullity, r.lsq_err, r.perm_ok, r.status)
            flush(stdout)
            open(csv, "a") do io
                @printf(io, "%d,%d,%d,%s,%.6f,%d,%d,%.6e,%s,\"%s\"\n",
                        d, seed, 4, label, r.seconds, r.bytes, r.nullity, r.lsq_err,
                        r.perm_ok, r.status)
            end
            startswith(label, "QuickDer") && (worst_qd = max(worst_qd, r.seconds))
        end
        if worst_qd > SWEEP_STOP_S
            println("  ** QuickDer exceeded $(SWEEP_STOP_S)s at d=$d; stopping v4dense sweep **")
            break
        end
    end
    println("Finished. Results in $csv")
end

# ---------------------------------------------------------------- video

function run_video()
    sizes = getlist("sizes", [30, 50, 64, 80, 100])
    T = gettype("T", Float64)
    tol = getflag("tol", 1e-6)
    csv = joinpath(REPORT_DIR, "quickder-video.csv")
    csv_header(csv,
        "H,W,F,T,seconds,bytes,maxrss_bytes,nullity,residual,restriction_sizes,status")

    println("video sizes=$sizes T=$T tol=$tol (2-thread, bench/jl)")

    build(dd) = begin
        frame = [Index(dd[i], "a$i") for i in 1:4]
        ITensor(randn(T, dd...), frame...), frame
    end

    # warm-up: small size, same T, not counted
    print("  warm-up ((8,8,8,3), $T) ... "); flush(stdout)
    tw = @elapsed begin
        Γw, frw = build((8, 8, 8, 3))
        Ωw = IndTransverseOps(frw, UniversalOp())
        chw = UniversalChisel(4)
        for _ in 1:2
            timed_der(Ωw, chw, Γw; tol = tol, restriction = :random)
        end
    end
    @printf("done (%.1fs, not counted)\n", tw)

    for s in sizes
        H = W = F = s
        dims = (H, W, F, 3)
        println("== $(H)x$(W)x$(F)x3, $T ==")
        Random.seed!(H + W + F)
        Γ, frame = build(dims)
        Ω = IndTransverseOps(frame, UniversalOp())
        ch = UniversalChisel(4)

        rsizes = try
            eng = engaged(Matrix{Float64}(ch))
            dvec = collect(ITensors.dim.(frame))
            string(Dleto._qdn_restriction_sizes(dvec, eng, length(dvec)))
        catch
            "n/a"
        end

        r = timed_der(Ω, ch, Γ; tol = tol, restriction = :random)
        @printf("  %-8s %9.3fs  %9.2f MB  maxrss=%.2f GB  null=%2d  resid=%9.2e  r=%s  %s\n",
                string(T), r.seconds, r.bytes / 2^20, r.maxrss / 2^30, r.nullity, r.resid,
                rsizes, r.status)
        flush(stdout)
        open(csv, "a") do io
            @printf(io, "%d,%d,%d,%s,%.6f,%d,%d,%d,%.6e,\"%s\",\"%s\"\n",
                    H, W, F, string(T), r.seconds, r.bytes, r.maxrss, r.nullity, r.resid,
                    rsizes, r.status)
        end
        if r.seconds > SWEEP_STOP_S
            println("  ** exceeded $(SWEEP_STOP_S)s at size=$s; stopping video sweep **")
            break
        end
    end
    println("Finished. Results in $csv")
end

# ---------------------------------------------------------------- sparse

function run_sparse()
    valence = getflag("valence", 4)
    ds      = getlist("d", valence == 3 ? [20, 50, 100, 200] : [10, 20, 30, 40, 60, 80])
    tol     = getflag("tol", 1e-6)
    csv = joinpath(REPORT_DIR, "quickder-sparse.csv")
    csv_header(csv, "valence,d,restriction,seconds,bytes,nullity,residual,status")

    println("sparse valence=$valence d=$ds tol=$tol (2-thread, bench/jl)")

    dwarm = valence == 3 ? 8 : 6
    print("  warm-up (raw sphere_octant($dwarm; valence=$valence)) ... "); flush(stdout)
    tw = @elapsed begin
        Sw = sphere_octant(dwarm; valence)
        frw = collect(inds(Sw))
        Ωw = IndTransverseOps(frw, UniversalOp())
        chw = UniversalChisel(valence)
        for restr in (:random, :corner)
            timed_der(Ωw, chw, Sw; tol = tol, restriction = restr)
        end
    end
    @printf("done (%.1fs, not counted)\n", tw)

    for d in ds
        S = sphere_octant(d; valence)
        fr = collect(inds(S))
        Ω = IndTransverseOps(fr, UniversalOp())
        ch = UniversalChisel(valence)

        println("== valence=$valence d=$d ==")
        worst = 0.0
        for restr in (:random, :corner)
            r = timed_der(Ω, ch, S; tol = tol, restriction = restr)
            @printf("  %-8s %9.3fs  %9.2f MB  null=%2d  resid=%9.2e  %s\n",
                    restr, r.seconds, r.bytes / 2^20, r.nullity, r.resid, r.status)
            flush(stdout)
            open(csv, "a") do io
                @printf(io, "%d,%d,%s,%.6f,%d,%d,%.6e,\"%s\"\n",
                        valence, d, restr, r.seconds, r.bytes, r.nullity, r.resid, r.status)
            end
            worst = max(worst, r.seconds)
        end
        if worst > SWEEP_STOP_S
            println("  ** exceeded $(SWEEP_STOP_S)s at d=$d; stopping sparse valence=$valence sweep **")
            break
        end
    end
    println("Finished. Results in $csv")
end

# ---------------------------------------------------------------- dispatch

if TASK == "v4dense"
    run_v4dense()
elseif TASK == "video"
    run_video()
elseif TASK == "sparse"
    run_sparse()
else
    error("unknown task \"$TASK\"; expected v4dense, video, or sparse")
end
