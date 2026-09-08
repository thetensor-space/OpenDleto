#
# RestrictedSolveGram -- candidate (b), Native-Core-Plan.md "Phase 2": does
# forming the restricted Gram in Float32 (`GramSolver(gram_eltype = Float32)`,
# src/solvers/NullSolvers.jl) buy time on the DENSE route without losing
# precision against the all-Float64 `GramSolver`?
#
# All six of bench/RestrictedSolve.jl's cases take the dense route under the
# PRODUCTION budget (`bench/jl bench/RestrictedSolve.jl estimate`), with
# `GramSolver` already picked over `SVDSolver` (QDN_GRAM_MIN_COLS), so this is
# the lever that actually applies at these sizes -- unlike candidate 1, no
# forcing is needed.
#
# `included` by bench/RestrictedSolve.jl (`the_case`, `csv_row`, `csv_header`,
# `fmt`, `rcsv`, `THREAD_NOTE` are already in scope there).
#
using Statistics

"""
    build_dense_matrix(c; seed) -> (; Mres, r, dims, T, eaxes, ncols)

`_qdn_solve_and_lift`'s DENSE branch (QuickDerN.jl:1636-1679, sketch -> whiten
-> `_qdn_restricted_matrix`), reusing the same internal functions as
`build_whitened_map` (RestrictedSolve.jl) but assembling the actual matrix
instead of a matrix-free map -- both branches solve the same whitened system.
"""
function build_dense_matrix(c; seed::Integer = 20260908)
    Ω, P, Γ = c.Ω, Matrix{c.T}(c.ch), c.Γ
    fr = frames(Ω)
    G = ITensors.array(Γ, fr...)
    T = eltype(G)
    N = ndims(G)
    dims = collect(size(G))
    eng = Dleto.engaged(Matrix{Float64}(P))
    r = Dleto._qdn_restriction_sizes(dims, eng, N)
    eaxes = [a for a in 1:N if eng[a]]
    rng = MersenneTwister(seed)
    axs = [Dleto._qdn_axis(T, dims[a], r[a], :random, rng) for a in 1:N]
    S = Dleto._qdn_cross_sketches(G, axs, eng)
    Uf = Dict{Int, Matrix{T}}(a => Dleto._qdn_host(Dleto._qdn_unfold(S[a], a)) for a in eaxes)
    wh = Dleto._qdn_whiten(Uf, eaxes, dims, r, N)
    Us, sdims = wh[2], wh[4]
    coff = Dict{Int,Int}(); ncols = 0
    for a in eaxes
        coff[a] = ncols
        ncols += sdims[a] * r[a]
    end
    Mres = Dleto._qdn_restricted_matrix(Us, P, eaxes, r, sdims, coff, ncols)
    return (; Mres, r, dims, T, eaxes, ncols)
end

"""
    run_gram_mixed(name)

`:GramSolver` (Float64 throughout) vs `GramSolver(gram_eltype = Float32)` on
the SAME dense restricted matrix (`build_dense_matrix`): time per stage
(`:gram`, `:cholesky`, `:subspace`, `:ritz` via `Dleto.QDN_STAGE_TIMES`),
nullity at the oracle threshold, and the WORST relative disagreement between
the two solvers' singular values at and above the oracle nullity (the
precision question the design note asks before adopting).
"""
function run_gram_mixed(name::AbstractString)
    c = the_case(name)
    b = build_dense_matrix(c)
    Lmap = LinearMaps.LinearMap(b.Mres)
    nv = c.restricted_oracle + 8

    function timed_gram(solver)
        stages = Dict{Symbol,Float64}()
        Dleto.QDN_STAGE_TIMES[] = stages
        GC.gc()
        st = @timed Dleto.solve(solver, Lmap; nv = nv)
        Dleto.QDN_STAGE_TIMES[] = nothing
        return (; res = st.value, seconds = st.time, stages)
    end

    f64 = timed_gram(Dleto.GramSolver())
    f32 = timed_gram(Dleto.GramSolver(gram_eltype = Float32))

    # Precision: the two solvers' Ritz values, aligned by rank (both ascending).
    k = min(length(f64.res.vals), length(f32.res.vals))
    v64, v32 = f64.res.vals[1:k], f32.res.vals[1:k]
    scale = max(maximum(v64), eps())
    reldiff = abs.(v64 .- v32) ./ scale
    worst_at_oracle = maximum(reldiff[1:min(c.restricted_oracle, k)])
    worst_overall = maximum(reldiff)
    nullity64 = count(v -> v <= 1e-6 * scale, v64)
    nullity32 = count(v -> v <= 1e-6 * scale, v32)

    csv_row(rcsv("gram-mixed.csv"),
            "case,label,restricted_rows,restricted_cols,nv,seconds_f64,seconds_mixed," *
            "gram_s_f64,cholesky_s_f64,subspace_s_f64,ritz_s_f64," *
            "gram_s_mixed,cholesky_s_mixed,subspace_s_mixed,ritz_s_mixed," *
            "nullity_f64,nullity_mixed,oracle_nullity,worst_reldiff_at_oracle,worst_reldiff_overall",
            name, "\"$(c.label)\"", size(b.Mres, 1), size(b.Mres, 2), nv,
            fmt(f64.seconds), fmt(f32.seconds),
            fmt(get(f64.stages, :gram, NaN)), fmt(get(f64.stages, :cholesky, NaN)),
            fmt(get(f64.stages, :subspace, NaN)), fmt(get(f64.stages, :ritz, NaN)),
            fmt(get(f32.stages, :gram, NaN)), fmt(get(f32.stages, :cholesky, NaN)),
            fmt(get(f32.stages, :subspace, NaN)), fmt(get(f32.stages, :ritz, NaN)),
            nullity64, nullity32, c.restricted_oracle, fmt(worst_at_oracle), fmt(worst_overall))
    @printf("%-14s rows=%-7d cols=%-7d f64=%7.3fs mixed=%7.3fs (gram %.3f->%.3f) \
nullity64=%d nullity_mixed=%d(oracle=%d) worst_reldiff@oracle=%.2e worst_overall=%.2e\n",
            name, size(b.Mres, 1), size(b.Mres, 2), f64.seconds, f32.seconds,
            get(f64.stages, :gram, NaN), get(f32.stages, :gram, NaN),
            nullity64, nullity32, c.restricted_oracle, worst_at_oracle, worst_overall)
    return nothing
end
