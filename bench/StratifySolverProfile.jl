#
# Stratification profiling across null solvers -- speed, memory, recovery.
#
# One tensor per dimension d, every registered null solver on that same tensor:
#
#   1. Build a sphere octant sampled from the densor space, as SphereLab
#      section 3 does: axis values u[i] = x_i^2 - r^2/3 and support where
#      u + v + w = 0.  Two samplings (third argument):
#        densor  (default) x_i^2 = r^2 * i/(d-1), i.e. equally spaced in x^2.
#                Then u+v+w = 0 holds EXACTLY on the support, the tensor is
#                nondegenerate, and the derivation space is exactly the two
#                scalar derivations plus the sphere derivation (nullity 3).
#                This is the well-posed benchmark.
#        lattice SphereLab's literal construction: x_i = i on 0:r+1 with a
#                shell |u+v+w| < cutoff.  At r = 50 this has ~120 points and
#                the shell derivation sits just under tol = 1e-6; at the sizes
#                swept here it is far sparser than a surface, `nondeg` removes
#                a third of every axis, and the derivation does not survive
#                (its eigenvalue ratio is ~1e-2 with no spectral gap).  Kept
#                so the comparison can be reproduced, not as a benchmark.
#   2. Scramble it by a random ORTHOGONAL change of basis on every axis, then
#      remove degenerate axes with `nondeg` (SVD bases, so still orthogonal).
#   3. Stratify against the universal chisel with SYMMETRIC operators.  An
#      orthogonal conjugate of a diagonal derivation is symmetric, so this is
#      the operator space that fits the scramble.
#   4. Measure, per solver: wall time, bytes allocated, peak RSS, the nullity
#      the solver reported, and how well the stratified tensor reconstructs the
#      input.
#
# RECONSTRUCTION METRIC.  A stratification is only defined up to the order and
# scale of the eigenvectors it puts on each axis, so the stratified tensor can
# at best equal the input with its axes permuted and rescaled.  The metric is
# therefore a least-squares fit inside that ambiguity:
#
#   - The composite transform T_a = X_a E_a Z_a (scramble, nondeg, stratify)
#     is known here, and would be a (partial) monomial matrix on a perfect
#     recovery.  Its column argmaxes give the axis permutations pi_a.
#   - With the permutations fixed, the per-axis diagonal scalings are solved
#     by alternating least squares (each sub-problem is closed form), and the
#     relative residual  |Sigma - S[pi] . (D1,D2,D3)| / |Sigma|  is reported
#     as `lsq_err`.  0 = perfect, 1 = nothing recovered.
#   - `support` is the fraction of Sigma's energy lying on the permuted
#     support of S: a tensor-only check that the sparsity pattern came back.
#
# JIT.  Every solver is run once on a small tensor before any timing, so the
# reported times are full runs of compiled code, not compilation.
#
# Usage:
#   julia -t auto --project=. bench/StratifySolverProfile.jl [maxd] [budget_s] [mode] [cutoff]
#
#   maxd     largest d to try (default 100); d = 10, 15, 20, ...
#   budget_s a solver slower than this at some d is dropped for larger d (900)
#   mode     densor (default) or lattice, see above
#   cutoff   lattice mode only: shell thickness in |u+v+w| (default 1.5)
#
# Prints a row per (d, solver) as soon as it is measured and appends the same
# row to bench/stratify-solver-profile.csv.
#
using Dleto
using ITensors
using LinearAlgebra
using Logging
using Printf
using Random

# Trigger packages for the solver extensions.  Arpack is only a weakdep.
using KrylovKit
using IterativeSolvers
const HAVE_ARPACK = try
    @eval using Arpack
    true
catch
    false
end

const MAXD    = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 100
const BUDGET  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 900.0
const MODE    = length(ARGS) >= 3 ? Symbol(ARGS[3])         : :densor
const CUTOFF  = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.5
MODE in (:densor, :lattice) || error("mode must be densor or lattice, got $MODE")
const TOL     = 1e-6
const WARMUP_D = 8
const CSV = joinpath(@__DIR__, "stratify-solver-profile$(MODE === :lattice ? "-lattice" : "").csv")

# ---------------------------------------------------------------- the tensor

"""
    sphere_octant(d, mode, cutoff) -> S::ITensor

The sphere octant x^2 + y^2 + z^2 = r^2 as a d x d x d tensor, sampled from
the densor space: u[i] = x_i^2 - r^2/3 and support where u + v + w ~ 0.

`:densor` samples x_i^2 = r^2 i/(d-1), which makes the equation exact on the
lattice (the support is i + j + k = d-1) -- a surface's worth of points,
~d^2/2, on every axis index.  `:lattice` is SphereLab's x_i = i on 0:r+1 with
r = d-2 and the given cutoff.
"""
function sphere_octant(d::Integer, mode::Symbol, cutoff::Real)
    if mode === :densor
        r2 = (d - 1.0)^2
        Us = [(0:d-1)...] .|> i -> r2 * (i / (d - 1) - 1 / 3)
        return randSurfaceTensor(Us, Us, Us, 1e-9 * r2)
    else
        r = d - 2.0
        Us = [(0:d-1)...] .|> i -> (i^2 - r^2 / 3.0)
        return randSurfaceTensor(Us, Us, Us, cutoff)
    end
end

# ---------------------------------------------------------------- transforms

"""
    axis_chain(i0, lists...) -> (T, i_end)

Follow original index `i0` through successive lists of two-index ITensors,
contracting them, and return the composite together with its far index.
"""
function axis_chain(i0::Index, lists...)
    T = nothing
    cur = i0
    for lst in lists
        M = only(filter(t -> hasind(t, cur), lst))
        nxt = only(filter(j -> j != cur, collect(inds(M))))
        T = T === nothing ? M : T * M
        cur = nxt
    end
    return T, cur
end

# ---------------------------------------------------------------- the metric

"""
    reconstruction(S, fr, Σ, Xs, Es, Zs) -> (; lsq_err, support, perm_ok)

See the header.  `S` is the original tensor on frame `fr`; `Σ` the stratified
tensor; `Xs`, `Es`, `Zs` the per-axis transforms applied in that order.
"""
function reconstruction(S::ITensor, fr, Σ::ITensor, Xs, Es, Zs)
    val = length(fr)
    perms = Vector{Vector{Int}}(undef, val)
    ds = Vector{Vector{Float64}}(undef, val)
    nd_inds = Vector{Index}(undef, val)
    perm_ok = true
    for a in 1:val
        T, i_nd = axis_chain(fr[a], Xs, Es)
        Z = only(filter(t -> hasind(t, i_nd), Zs))
        i_tmp = only(filter(j -> j != i_nd, collect(inds(Z))))
        Ta = Array(T * Z, fr[a], i_tmp)          # d x d'
        π = [argmax(abs.(Ta[:, c])) for c in axes(Ta, 2)]
        perm_ok &= allunique(π)
        perms[a] = π
        ds[a] = [Ta[π[c], c] for c in axes(Ta, 2)]   # monomial part of T_a
        nd_inds[a] = i_nd
    end
    Σarr = Array(Σ, nd_inds...)
    S0 = Array(S, fr...)
    Sp = S0[perms...]                             # S with axes permuted

    # Alternating least squares for the diagonal scalings, started from the
    # monomial part of the solver's own transform.  Fitting a sign pattern is
    # non-convex, so an all-ones start stalls in a poor local fit; from this
    # start the sweeps leave a perfect recovery at zero residual and refine
    # an imperfect one.
    scaled() = Sp .* reshape(ds[1], :, 1, 1) .* reshape(ds[2], 1, :, 1) .* reshape(ds[3], 1, 1, :)
    for _ in 1:6, a in 1:val
        others = copy(ds); others[a] = ones(size(Sp, a))
        B = Sp .* reshape(others[1], :, 1, 1) .* reshape(others[2], 1, :, 1) .* reshape(others[3], 1, 1, :)
        for c in axes(Sp, a)
            bs = selectdim(B, a, c); ss = selectdim(Σarr, a, c)
            nb = sum(abs2, bs)
            ds[a][c] = nb == 0 ? 0.0 : dot(bs, ss) / nb
        end
    end
    fit = scaled()
    nΣ = norm(Σarr)
    lsq_err = nΣ == 0 ? Inf : norm(Σarr - fit) / nΣ
    support = nΣ == 0 ? 0.0 : sum(abs2, Σarr[Sp .!= 0]) / nΣ^2
    return (; lsq_err, support, perm_ok)
end

# ---------------------------------------------------------------- one run

const ROSTER = [:AutoSolver, :SVDSolver, :LUSolver, :LSMRSolver, :KrylovSolver,
                :LanczosSolver, :CGSolver, :ShiftInvertSolver,
                :ArpackSolver, :ArpackDenseSolver]

"""Silence the solvers' own chatter (`println`s and `@info`) during a run."""
quietly(f) = with_logger(NullLogger()) do
    redirect_stdout(devnull) do
        f()
    end
end

"""
    run_solver(sym, Ω, ch, Γ) -> (; seconds, bytes, Σ, Zs, nullity, status)
"""
function run_solver(sym::Symbol, Ω, ch, Γ)
    GC.gc()
    res = nothing
    nullity = 0
    status = "ok"
    st = @timed try
        quietly() do
            # Same body as `stratify(Ω, ch, Γ)`, unrolled so the nullity the
            # solver reported can be recorded alongside the result.
            m = SylverLiningMethod(; solver = sym)
            (rΩ, expand_map, ders) = derTrOpsReduced(m, Ω, ch, Γ; tol = TOL, nd = -1)
            nullity = size(ders, 2)
            nullity == 0 && error("no nontrivial derivations")
            δ = embedITensors(Ω, expand_map(ders * randn(nullity)))
            res = stratify(Γ, δ)
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
        res = nothing
    end
    return (; seconds = st.time, bytes = st.bytes, res, nullity, status)
end

fmt_gb(b) = @sprintf("%7.3f", b / 2^30)

function header(d, dims, nnz)
    println()
    @printf("==== d = %d   nondeg dims = %s   nnz = %d   op dim = %d   %s\n",
            d, string(dims), nnz, sum(x -> x * (x + 1) ÷ 2, dims),
            Libc.strftime("%H:%M:%S", time()))
    @printf("%-18s %10s %10s %10s %6s %9s %8s %5s  %s\n",
            "solver", "time(s)", "alloc(GB)", "rss(GB)", "null", "lsq_err", "support", "perm", "status")
    flush(stdout)
end

function record(d, sym, r, met)
    rss = Sys.maxrss() / 2^30
    lsq = met === nothing ? NaN : met.lsq_err
    sup = met === nothing ? NaN : met.support
    pk  = met === nothing ? "-" : (met.perm_ok ? "yes" : "no")
    @printf("%-18s %10.3f %10s %10.3f %6d %9.2e %8.4f %5s  %s\n",
            String(sym), r.seconds, fmt_gb(r.bytes), rss, r.nullity, lsq, sup, pk, r.status)
    flush(stdout)
    open(CSV, "a") do io
        @printf(io, "%d,%s,%.6f,%d,%.4f,%d,%.6e,%.6f,%s,\"%s\"\n",
                d, String(sym), r.seconds, r.bytes, rss, r.nullity, lsq, sup, pk, r.status)
    end
end

"""Build the input for dimension d: sphere, orthogonal scramble, nondeg."""
function build(d; seed = d)
    Random.seed!(seed)
    S = sphere_octant(d, MODE, CUTOFF)
    fr = collect(inds(S))
    nnz = count(!=(0), Array(S, fr...))
    rn = randomize_tensor(S; type = :orthogonal)
    nd = nondeg(rn.Δ)
    Γ = nd.Δ
    fr_nd = collect(inds(Γ))
    Ω = IndTransverseOps(fr_nd, SymmetricOp())
    ch = UniversalChisel(3)
    return (; S, fr, nnz, Xs = rn.Xs, Es = nd.Es, Γ, Ω, ch, dims = ITensors.dim.(fr_nd))
end

function profile_d(d, solvers; timed = true)
    inp = build(d)
    timed && header(d, inp.dims, inp.nnz)
    out = Dict{Symbol,Float64}()
    for sym in solvers
        r = run_solver(sym, inp.Ω, inp.ch, inp.Γ)
        met = r.res === nothing ? nothing :
              reconstruction(inp.S, inp.fr, r.res.Σ, inp.Xs, inp.Es, r.res.Xs)
        timed && record(d, sym, r, met)
        out[sym] = r.seconds
    end
    return out
end

# ---------------------------------------------------------------- main

function main()
    solvers = filter(s -> s in available_solvers(), ROSTER)
    println("Dleto stratification solver profile")
    println("  sphere = $MODE$(MODE === :lattice ? " (cutoff $CUTOFF)" : ""), tol = $TOL, max d = $MAXD, budget = $(BUDGET)s per solver")
    println("  threads = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads())")
    println("  Arpack loaded: $HAVE_ARPACK")
    println("  solvers: ", join(String.(solvers), ", "))
    if !isfile(CSV)
        open(CSV, "w") do io
            println(io, "d,solver,seconds,bytes,peak_rss_gb,nullity,lsq_err,support,perm_ok,status")
        end
    end

    print("\nWarming up the JIT on d = $WARMUP_D for every solver ... ")
    flush(stdout)
    tw = @elapsed begin
        profile_d(WARMUP_D, solvers; timed = false)
        profile_d(WARMUP_D, solvers; timed = false)
    end
    @printf("done (%.1fs, not counted).\n", tw)

    d = 10
    while d <= MAXD && !isempty(solvers)
        times = profile_d(d, solvers)
        slow = [s for s in solvers if get(times, s, 0.0) > BUDGET]
        for s in slow
            println("  dropping $(s): $(round(times[s]; digits = 1))s exceeds the $(BUDGET)s budget")
        end
        solvers = filter(s -> !(s in slow), solvers)
        d += 5
    end
    println("\nFinished. Results in $CSV")
end

main()
