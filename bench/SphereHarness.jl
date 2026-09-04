#
# SphereHarness -- one input, one metric, for every stratification experiment.
#
# `include` this file from a bench script or a REPL.  It gives:
#
#   build_sphere(d; T, seed, mode, ops)  the scrambled, nondegenerate sphere
#                                        octant and everything needed to score
#                                        a stratification of it
#   run_stratify(inp; solver, method, tol, nd, ...)
#                                        one timed stratification on that input
#   reconstruction(inp, Σ, Zs)           least-squares recovery score
#   warmup!(...)                         JIT warm-up so timings are full runs
#
# THE INPUT.  A sphere octant x^2 + y^2 + z^2 = r^2 sampled from the densor
# space as labs/SphereLab.ipynb section 3 does: axis values u[i] = x_i^2 - r^2/3
# and support where u + v + w = 0.  The default `:densor` sampling takes
# x_i^2 = r^2 i/(d-1), so the equation is exact on the lattice (support
# i + j + k = d-1, about d^2/2 points, every axis index used), the tensor is
# nondegenerate, and the derivation space with symmetric operators is exactly
# the two scalar derivations plus the sphere derivation: nullity 3, distinct
# eigenvalues, so a perfect stratification returns the input up to per-axis
# signed permutation.  `:lattice` is SphereLab's literal x_i = i shell, which is
# a curve's worth of points at these sizes and ill-posed after `nondeg`; kept
# for reference only.
#
# The tensor is scrambled by random ORTHOGONAL matrices and passed through
# `nondeg` (SVD bases, so still orthogonal).  An orthogonal conjugate of a
# diagonal derivation is symmetric, which is why `SymmetricOp()` is the default
# operator space; pass `ops = UniversalOp()` for methods that need all
# matrices (QuickDer).
#
# ELEMENT TYPE.  `T` converts the tensor handed to the solver (the scrambled,
# nondegenerate Γ).  Construction, scrambling and scoring stay in Float64 so
# the score measures the solver's precision, not the harness's.
#
# THE METRIC.  A stratification is defined only up to the order and scale of
# the eigenvectors on each axis, so the best possible result equals the input
# with axes permuted and rescaled.  `reconstruction` fits inside exactly that
# ambiguity: the composite transform T_a = X_a E_a Z_a (scramble, nondeg,
# stratify) is known here and is monomial on a perfect recovery; its column
# argmaxes give the permutations, its monomial part seeds per-axis diagonal
# scalings, and alternating least squares refines them.  Reported:
#
#   lsq_err   |Σ - S[π]·(D1,D2,D3)| / |Σ|     0 = perfect, ~1 = nothing
#   support   energy fraction of Σ on the permuted support of S (tensor-only)
#   perm_ok   whether the argmax columns form a genuine permutation
#
using Dleto
using ITensors
using LinearAlgebra
using Logging
using Random

# ---------------------------------------------------------------- the tensor

"""
    sphere_octant(d; mode = :densor, cutoff = 1.5) -> ITensor
"""
function sphere_octant(d::Integer; mode::Symbol = :densor, cutoff::Real = 1.5)
    if mode === :densor
        r2 = (d - 1.0)^2
        Us = [(0:d-1)...] .|> i -> r2 * (i / (d - 1) - 1 / 3)
        return randSurfaceTensor(Us, Us, Us, 1e-9 * r2)
    elseif mode === :lattice
        r = d - 2.0
        Us = [(0:d-1)...] .|> i -> (i^2 - r^2 / 3.0)
        return randSurfaceTensor(Us, Us, Us, cutoff)
    end
    error("mode must be :densor or :lattice, got $mode")
end

"""
    build_sphere(d; T = Float64, seed = d, mode = :densor, ops = SymmetricOp())

The input for dimension `d`.  Fields:

- `S`, `fr`        the original sphere tensor and its frame (Float64)
- `nnz`            nonzeros in `S`
- `Xs`, `Es`       the orthogonal scramble and the `nondeg` bases, per axis
- `Γ`              the tensor to stratify, element type `T`, on the nondeg frame
- `Ω`, `ch`        operator space on Γ's frame, and the universal chisel
- `dims`           Γ's axis dimensions
"""
function build_sphere(d::Integer; T::Type = Float64, seed::Integer = d,
                      mode::Symbol = :densor, cutoff::Real = 1.5,
                      ops::Operator = SymmetricOp())
    Random.seed!(seed)
    S = sphere_octant(d; mode, cutoff)
    fr = collect(inds(S))
    nnz = count(!=(0), Array(S, fr...))
    rn = randomize_tensor(S; type = :orthogonal)
    nd = nondeg(rn.Δ)
    fr_nd = collect(inds(nd.Δ))
    Γ = T === Float64 ? nd.Δ : ITensor(Array{T}(Array(nd.Δ, fr_nd...)), fr_nd...)
    Ω = IndTransverseOps(fr_nd, ops)
    ch = UniversalChisel(3)
    return (; S, fr, nnz, Xs = rn.Xs, Es = nd.Es, Γ, Ω, ch,
              dims = ITensors.dim.(fr_nd), T)
end

# ---------------------------------------------------------------- the metric

"""
    axis_chain(i0, lists...) -> (T, i_end)

Follow index `i0` through successive lists of two-index ITensors, contracting
them, and return the composite with its far index.
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

"""
    reconstruction(inp, Σ, Zs) -> (; lsq_err, support, perm_ok)

Score a stratified tensor `Σ` with per-axis transforms `Zs` (as returned by
`stratify`) against the input it came from.  Computed in Float64.
"""
function reconstruction(inp, Σ::ITensor, Zs)
    S, fr, Xs, Es = inp.S, inp.fr, inp.Xs, inp.Es
    val = length(fr)
    perms = Vector{Vector{Int}}(undef, val)
    ds = Vector{Vector{Float64}}(undef, val)
    nd_inds = Vector{Index}(undef, val)
    perm_ok = true
    for a in 1:val
        T, i_nd = axis_chain(fr[a], Xs, Es)
        Z = only(filter(t -> hasind(t, i_nd), Zs))
        i_tmp = only(filter(j -> j != i_nd, collect(inds(Z))))
        Zf = ITensor(Float64.(Array(Z, i_nd, i_tmp)), i_nd, i_tmp)
        Ta = Array(T * Zf, fr[a], i_tmp)             # d x d'
        π = [argmax(abs.(Ta[:, c])) for c in axes(Ta, 2)]
        perm_ok &= allunique(π)
        perms[a] = π
        ds[a] = [Ta[π[c], c] for c in axes(Ta, 2)]   # monomial part of T_a
        nd_inds[a] = i_nd
    end
    Σarr = Float64.(Array(Σ, nd_inds...))
    S0 = Array(S, fr...)
    Sp = S0[perms...]

    # ALS for the diagonal scalings, seeded from the monomial part of the
    # solver's own transform: a sign-pattern fit is non-convex and an all-ones
    # start stalls in a poor local fit.
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

"""Silence solver chatter (`println` and `@info`) during a run."""
quietly(f) = with_logger(NullLogger()) do
    redirect_stdout(devnull) do
        f()
    end
end

"""
    run_stratify(inp; method = :SylverLining, solver = :AutoSolver, tol = 1e-6,
                 nd = -1, score = true, method_kwargs...)
        -> (; seconds, bytes, nullity, lsq_err, support, perm_ok, status, res)

One stratification of `inp.Γ`, timed with `@timed` (wall seconds and bytes
allocated), then scored.  `method_kwargs` go to the derivation method
constructor (for `:SylverLining` that is `solver`; `:QuickDer` takes none).
Same body as `stratify(Ω, ch, Γ)`, unrolled so the nullity is recorded.
`status` is "ok", or the first line of the error.
"""
function run_stratify(inp; method::Symbol = :SylverLining, solver::Symbol = :AutoSolver,
                      tol::Real = 1e-6, nd = -1, score::Bool = true, method_kwargs...)
    Ω, ch, Γ = inp.Ω, inp.ch, inp.Γ
    res = nothing
    nullity = 0
    status = "ok"
    GC.gc()
    st = @timed try
        quietly() do
            m = method === :SylverLining ?
                Dleto.get_derivation_method(method; solver, method_kwargs...) :
                Dleto.get_derivation_method(method; method_kwargs...)
            (rΩ, expand_map, ders) = derTrOpsReduced(m, Ω, ch, Γ; tol = tol, nd = nd)
            nullity = size(ders, 2)
            nullity == 0 && error("no nontrivial derivations")
            δ = embedITensors(Ω, expand_map(ders * randn(eltype(ders), nullity)))
            res = stratify(Γ, δ)
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
        res = nothing
    end
    sc = (res !== nothing && score) ? reconstruction(inp, res.Σ, res.Xs) :
         (; lsq_err = NaN, support = NaN, perm_ok = false)
    return (; seconds = st.time, bytes = st.bytes, nullity, sc.lsq_err, sc.support,
              sc.perm_ok, status, res)
end

"""
    warmup!(configs; d = 8, T = Float64, passes = 2)

Run every `(method, solver)` pair in `configs` on a small input so later
timings are full runs of compiled code.  Each config is a NamedTuple of
`run_stratify` keywords, e.g. `(; solver = :ArpackSolver)`.
"""
function warmup!(configs; d::Integer = 8, T::Type = Float64, passes::Integer = 2,
                 ops::Operator = SymmetricOp())
    inp = build_sphere(d; T, ops)
    for _ in 1:passes, cfg in configs
        run_stratify(inp; cfg...)
    end
    return nothing
end
