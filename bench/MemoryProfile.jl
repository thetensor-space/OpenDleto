#
# MemoryProfile.jl -- where the bytes go, stage by stage.
#
#   bench/jl bench/MemoryProfile.jl sphere <d> [valence] [Float64|Float32]
#   bench/jl bench/MemoryProfile.jl build  <d> [valence] [Float64|Float32]
#
# `build` profiles the HARNESS alone (`build_sphere`), `sphere` profiles the
# harness and then a whitened matrix-free `derTrOpsReduced` on its output.
#
# WHY THIS EXISTS.  After the whitened restriction (`docs/CONTEXT.md`, session
# 4) the matrix-free branch converges and the wall is memory: 11.4 GB peak at
# d = 500 valence 3 and ~22 GB projected at d = 1000, against a per-process
# kill line well below that.  Two suspects were named from reading the code --
# `build_sphere` holding three copies of the tensor, and `_qdn_pair_tensor`
# doing a full `permutedims` of it once per lift axis -- and a guess is not a
# measurement, so this script measures before anything is changed.
#
# WHAT THE THREE NUMBERS MEAN, because they disagree and each is needed:
#
#   alloc   `Base.gc_bytes()` over the stage: every allocation, survivor or
#           not.  CHURN.  A stage that allocates 8 GB and frees it has a large
#           alloc and may cost no peak at all -- if the GC runs in time.
#   live    `Base.gc_live_bytes()` at the end of the stage: what is still
#           reachable.  RETENTION.  This is what a copy that is kept costs.
#   peak    `Sys.maxrss()`, the process high-water mark, monotone across
#           stages.  The stage where it first reaches its final value is the
#           stage that set the peak, and the peak is the kill-line number.
#
# A stage with large `alloc` and flat `peak` is harmless; a stage where `peak`
# jumps is the one to fix, and `live` says whether the fix is "do not keep it"
# or "do not make it".
#
using Arpack          # so `_qdn_default_free_solver()` is :ArpackSolver, as in production
using Printf

include(joinpath(@__DIR__, "SphereHarness.jl"))

const MP_GB = 1 / 2^30

"""
    mp_probe() -> (alloc, live, peak)

The three counters, in GB.  `gc_bytes` is cumulative for the process, so the
caller differences it; the other two are levels.
"""
mp_probe() = (Float64(Base.gc_bytes()) * MP_GB,
              Float64(Base.gc_live_bytes()) * MP_GB,
              Float64(Sys.maxrss()) * MP_GB)

mutable struct MPLog
    rows::Vector{Tuple{String,Float64,Float64,Float64}}
    mark::Float64
end
MPLog() = MPLog(Tuple{String,Float64,Float64,Float64}[], mp_probe()[1])

"""
    mp_stage!(log, name)

Close a stage: charge the churn since the previous boundary and record the
live heap and the process peak as they stand now.  No `GC.gc()` here -- a
forced collection would flatter `live` and hide exactly the retention this is
looking for.  The callers that do want one call `GC.gc()` themselves and
close a stage after it, so the collection is visible in the table.
"""
function mp_stage!(log::MPLog, name::AbstractString)
    (a, l, p) = mp_probe()
    push!(log.rows, (name, a - log.mark, l, p))
    log.mark = a
    return nothing
end

function mp_table(log::MPLog, title::AbstractString)
    @printf("\n%s\n", title)
    @printf("%-26s %10s %10s %10s\n", "stage", "alloc GB", "live GB", "peak GB")
    for (name, a, l, p) in log.rows
        @printf("%-26s %10.3f %10.3f %10.3f\n", name, a, l, p)
    end
    @printf("%-26s %10.3f %10s %10.3f\n", "TOTAL / final",
            sum(r[2] for r in log.rows; init = 0.0), "",
            isempty(log.rows) ? 0.0 : log.rows[end][4])
    flush(stdout)
    return nothing
end

# ------------------------------------------------------- the harness, unrolled

"""
    profile_build(d; valence, T, seed, lean, keep_S) -> (inp, log)

`build_sphere(d; valence, T)` with a stage boundary between each of its steps,
so the copies it holds are attributed one at a time.  Kept in step with
`SphereHarness.build_sphere` by construction: the same sequence of calls,
returning the same NamedTuple, so the profile is of the real thing and not of
a paraphrase.  `lean` selects which of the two construction paths is unrolled.
"""
function profile_build(d::Integer; valence::Integer = 3, T::Type = Float64,
                       seed::Integer = d, ops::Operator = SymmetricOp(),
                       lean::Bool = false, keep_S::Bool = !lean)
    lean && return profile_build_lean(d; valence, T, seed, ops, keep_S)
    log = MPLog()
    Random.seed!(seed)
    mp_stage!(log, "0 baseline")

    S = sphere_octant(d; valence)
    fr = collect(inds(S))
    mp_stage!(log, "1 sphere_octant")

    nnz = count(!=(0), ITensors.array(S, fr...))
    mp_stage!(log, "2 nnz count")

    rn = randomize_tensor(S; type = :orthogonal)
    Xs = rn.Xs
    Δrn = rn.Δ
    rn = nothing
    mp_stage!(log, "3 randomize_tensor")

    nd = nondeg(Δrn)
    Δrn = nothing
    mp_stage!(log, "4 nondeg")

    fr_nd = collect(inds(nd.Δ))
    Γ = T === Float64 ? nd.Δ : ITensor(Array{T}(Array(nd.Δ, fr_nd...)), fr_nd...)
    Ω = IndTransverseOps(fr_nd, ops)
    ch = UniversalChisel(valence)
    mp_stage!(log, "5 convert + ops")

    inp = (; S = keep_S ? S : nothing, fr, nnz, Xs, Es = nd.Es, Γ, Ω, ch,
             dims = ITensors.dim.(fr_nd), T, lean = false)
    GC.gc(); GC.gc()
    mp_stage!(log, "6 after full GC")
    return (inp, log)
end

"""
    profile_build_lean(d; valence, T, seed, ops, keep_S) -> (inp, log)

`build_sphere_lean` unrolled on the same stage boundaries, so the two paths'
tables line up row for row where they do the same work.
"""
function profile_build_lean(d::Integer; valence::Integer = 3, T::Type = Float64,
                            seed::Integer = d, ops::Operator = SymmetricOp(),
                            keep_S::Bool = false)
    n = valence
    log = MPLog()
    Random.seed!(seed)
    mp_stage!(log, "0 baseline")

    A = zeros(Float64, ntuple(_ -> d, n))
    _fill_sum_lattice!(A, d)
    fr = [Index(d, "a$a") for a in 1:n]
    # `itensor`, not `ITensor`: the capitalised constructor COPIES the array
    # it is handed (+0.93 GB of peak at d = 500, measured), the lowercase one
    # takes ownership of it.
    S = keep_S ? ITensors.itensor(copy(A), fr...) : nothing
    mp_stage!(log, "1 lattice array")

    nnz = count(!=(0), A)
    mp_stage!(log, "2 nnz count")

    Xa = [Dleto.__random_orthogonal(d) for _ in 1:n]
    for a in 1:n
        Dleto._qdn_ttm_square!(A, Xa[a], a)
    end
    mp_stage!(log, "3 scramble (in place)")

    bases = [_tsqr_axis_basis(A, a) for a in 1:n]
    ks = Int[count(>=(1e-10), s) for (_, s) in bases]
    Va = [V[:, 1:k] for ((V, _), k) in zip(bases, ks)]
    mp_stage!(log, "4a nondeg bases (TSQR)")

    for a in 1:n
        if ks[a] == d
            Dleto._qdn_ttm_square!(A, Va[a], a)
        else
            A = Dleto._qdn_ttm!(similar(A, ntuple(i -> i <= a ? ks[i] : d, n)),
                                A, Va[a], a)
        end
    end
    mp_stage!(log, "4b nondeg apply (in place)")

    mid = [Index(d, "a$a,rand") for a in 1:n]
    fr_nd = [Index(ks[a], "a$a,nondeg") for a in 1:n]
    Xs = [ITensor(Xa[a], fr[a], mid[a]) for a in 1:n]
    Es = [ITensor(Va[a], mid[a], fr_nd[a]) for a in 1:n]
    Γ = ITensors.itensor(T === Float64 ? A : Array{T}(A), fr_nd...)
    A = nothing
    Ω = IndTransverseOps(fr_nd, ops)
    mp_stage!(log, "5 convert + ops")

    inp = (; S, fr, nnz, Xs, Es, Γ, Ω, ch = UniversalChisel(n), dims = ks, T,
             lean = true)
    GC.gc(); GC.gc()
    mp_stage!(log, "6 after full GC")
    return (inp, log)
end

# ------------------------------------------------------- the solver

"""
    profile_solve(inp; whiten, solver, tol) -> log

One `derTrOpsReduced` with `QDN_STAGE_TIMES` and `QDN_STAGE_BYTES` on, forced
onto the matrix-free branch, reported on the same three-column layout.  The
alloc column is QuickDer's own per-stage churn; the peak column is the
process, so it carries the harness's tensors underneath it -- which is the
point, since the kill line does too.
"""
function profile_solve(inp; whiten::Bool = true, solver::Symbol = :AutoSolver,
                       tol::Real = 1e-6, verbose::Bool = false, seed::Integer = 20260904)
    Dleto.QDN_DENSE_BUDGET_BYTES[] = 0.0        # matrix-free, as at d >= 300
    # `solve_nullspace` estimates the operator norm with a RANDOMISED power
    # iteration, so the threshold it compares the spectrum against depends on
    # the global RNG state -- and at d = 300 that is enough to flip the
    # reported nullity (see the d = 300 anomaly in the session-4 notes).  Seed
    # here so a profile is reproducible run to run.
    Random.seed!(seed)
    method = Dleto.get_derivation_method(:QuickDer; whiten, solver,
                                         verify = :random, seed = 20260904)
    times = Dict{Symbol,Float64}()
    bytes = Dict{Symbol,NTuple{2,Float64}}()
    Dleto.QDN_STAGE_TIMES[] = times
    Dleto.QDN_STAGE_BYTES[] = bytes
    Dleto.QDN_APPLY_COUNT[] = 0
    nullity = 0
    status = "ok"
    t0 = time()
    try
        with_logger(verbose ? ConsoleLogger(stderr, Logging.Debug) : NullLogger()) do
            redirect_stdout(devnull) do
                (_, _, ders) = Dleto.derTrOpsReduced(method, inp.Ω, inp.ch, inp.Γ;
                                                     tol = tol)
                nullity = size(ders, 2)
            end
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
    end
    secs = time() - t0
    applies = Dleto.QDN_APPLY_COUNT[]
    Dleto.QDN_APPLY_COUNT[] = -1
    Dleto.QDN_STAGE_TIMES[] = nothing
    Dleto.QDN_STAGE_BYTES[] = nothing

    order = [:upload, :sketch, :whiten, :restricted, :solve]
    for k in sort!(collect(keys(bytes)))
        startswith(String(k), "lift") && k != :lift && push!(order, k)
    end
    append!(order, [:lift, :filter, :verify, :restrict_ops])
    @printf("\nQuickDer stages (whiten=%d solver=%s matrix-free)\n", whiten, solver)
    @printf("%-26s %10s %10s %10s %10s\n", "stage", "seconds", "alloc GB", "peak GB", "")
    for k in order
        haskey(bytes, k) || continue
        (al, pk) = bytes[k]
        @printf("%-26s %10.2f %10.3f %10.3f\n", String(k), get(times, k, NaN),
                al * MP_GB, pk * MP_GB)
    end
    @printf("%-26s %10.2f %10s %10.3f  applies=%d nullity=%d status=%s\n",
            "TOTAL", secs, "", Sys.maxrss() * MP_GB, applies, nullity, status)
    flush(stdout)
    return (; seconds = secs, applies, nullity, status,
              peak_GB = Sys.maxrss() * MP_GB)
end

# ------------------------------------------------------- CLI

"""
    compare(d; valence)

The two build paths at one size, and the two things that have to hold between
them.

First, `_tsqr_axis_basis` has to be an SVD: its singular values are checked
against `ITensors.svd` of the same unfolding, which is the factorisation
`nondeg` itself calls.  That is a numerical claim and it is exact up to
rounding.

Second, the two inputs differ by a per-axis orthogonal change of basis (see
`build_sphere_lean`), so what has to agree is what a change of basis cannot
touch: the nullity, the verdict, and the ORDER of the Z-law residual.  The
iteration count and the residual's digits are properties of the actual matrix
and are expected to differ.
"""
function compare(d::Integer; valence::Integer = 3)
    (ci, _) = profile_build(d; valence, lean = false)
    Γs = randomize_tensor(sphere_octant(d; valence); type = :orthogonal).Δ
    for a in 1:valence
        fa = collect(inds(Γs))
        (_, s_tsqr) = _tsqr_axis_basis(Array(Γs, fa...), a)
        (_, S, _) = ITensors.svd(Γs, [e for e in fa if e != fa[a]])
        s_it = sort(diag(Array(S, inds(S)...)); rev = true)
        @printf("axis %d singular values: max |tsqr - itensor| = %.2e (s_max = %.2e)\n",
                a, maximum(abs.(s_tsqr .- s_it)), s_it[1])
    end
    (li, _) = profile_build(d; valence, lean = true, keep_S = false)
    rc = profile_solve(ci)
    rl = profile_solve(li)
    @printf("\nclassic: nullity=%d applies=%d %.1f s | lean: nullity=%d applies=%d %.1f s\n",
            rc.nullity, rc.applies, rc.seconds, rl.nullity, rl.applies, rl.seconds)
    return nothing
end

function mpmain(args)
    isempty(args) && error("first argument must be build, sphere or compare")
    task = args[1]
    d = parse(Int, args[2])
    valence = length(args) >= 3 ? parse(Int, args[3]) : 3
    T = length(args) >= 4 && args[4] == "Float32" ? Float32 : Float64
    whiten = !(length(args) >= 5 && args[5] == "0")
    verbose = length(args) >= 6 && args[6] == "1"
    lean = !(length(args) >= 7 && args[7] == "0")

    # Warm the whole path up at a small size so no stage is charged for JIT.
    if task != "build"
        let (wi, _) = profile_build(valence == 3 ? 10 : 6; valence, T, lean)
            try
                redirect_stdout(devnull) do
                    profile_solve(wi; whiten)
                end
            catch e
                @warn "warmup failed (continuing)" exception = (e, catch_backtrace())
            end
        end
        GC.gc(); GC.gc()
    end

    task == "compare" && return compare(d; valence)

    @printf("d=%d valence=%d eltype=%s lean=%d tensor=%.3f GB per copy\n",
            d, valence, T, lean, prod(fill(float(d), valence)) * sizeof(T) * MP_GB)
    (inp, blog) = profile_build(d; valence, T, lean)
    mp_table(blog, "build_sphere(d=$d; valence=$valence, T=$T, lean=$lean)")
    task == "build" && return nothing
    profile_solve(inp; whiten, verbose)
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && mpmain(ARGS)
