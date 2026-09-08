#
# RestrictedSolveRangeFinder -- candidate (c), Native-Core-Plan.md "Phase 2":
# a randomized range finder for the small near-null invariant subspace,
# MATRIX-FREE (no dense Gram, unlike `GramSolver`), since the nullity is tiny
# (2-13) and already bracketed by `solve_nullspace`'s own confirmation pass.
#
# THE ALGORITHM.  `GramSolver` (src/solvers/NullSolvers.jl) already does
# randomized subspace iteration for the null space -- shifted inverse power
# iteration on the Gram, Rayleigh-Ritz on the unsquared matrix -- but its
# inverse is a dense Cholesky, which is exactly the byte cost candidate (b)
# addresses.  This prototype does the SAME algorithm with the inverse taken
# matrix-free instead: `Dleto.shift_invert_map(L)` (NullSolvers.jl:1242) gives
# `S = (LᵗL + shift I)⁻¹` as a `LinearMap` whose apply is one CG solve (never
# forming `LᵗL`), and the range finder iterates `X <- S X` on `k + p` random
# columns, then measures the subspace on `L` itself exactly as `GramSolver`'s
# stage 2 does.  Every `S * v` is 2 applies of `L` (one forward, one adjoint)
# per CG STEP, so this trades ARPACK/KrylovKit's Lanczos restarts for CG
# iterations inside a shift-invert -- worth trying because a null space this
# small and this well separated (whitened: spectrum in `[0, sum c_a]`) may
# not need Lanczos's generality at all.
#
# `included` by bench/RestrictedSolve.jl (`the_case`, `build_whitened_map`,
# `csv_row`, `rcsv`, `fmt`, `THREAD_NOTE` already in scope there).

"""
    range_finder(L; k, p, shift_rel, steps, cgtol, cgmaxiter) -> (; vals, vecs)

One randomized range-finding pass for the `k` smallest singular directions of
`L`, oversampled by `p`, matrix-free throughout.  Mirrors `GramSolver`'s
`(vals, vecs)` shape so it drops into the same comparison as any other
solver.
"""
function range_finder(L; k::Integer, p::Integer = max(16, k), shift_rel::Real = 1e-10,
                      steps::Integer = 4, cgtol::Real = 1e-8, cgmaxiter::Integer = 200)
    n = size(L, 2)
    T = eltype(L)
    kp = min(n, k + p)
    (_, S) = Dleto.shift_invert_map(L; shift_rel = shift_rel, cgtol = cgtol,
                                    cgmaxiter = cgmaxiter)
    X = randn(T, n, kp)
    for _ in 1:steps
        X = hcat([S * X[:, j] for j in 1:kp]...)
        X = Matrix(qr(X).Q)[:, 1:kp]
    end
    # `L * X` for a matrix `X` (rather than a vector) is a LAZY
    # `LinearMaps.CompositeMap` -- LinearMaps.jl treats `X` itself as a linear
    # map and composes rather than applies -- so it has to be materialised a
    # column at a time, same as the range-finding loop above.
    MX = hcat([L * X[:, j] for j in 1:size(X, 2)]...)
    F = svd(MX)
    ord = sortperm(F.S)[1:k]
    return (; vals = F.S[ord], vecs = X * F.V[:, ord])
end

"""
    run_range_finder(name)

The range finder against the production default (`_qdn_default_free_solver()`,
ARPACK), on the SAME whitened map (`build_whitened_map`), forced matrix-free.
Fixed `k = oracle + 4` (a single call, no escalation loop -- this measures the
core algorithm, not `solve_nullspace`'s bracket/confirm machinery around it),
applies and time, and whether it lands the same nullity and span.
"""
function run_range_finder(name::AbstractString)
    c = the_case(name)
    m = build_whitened_map(c)
    k = c.restricted_oracle + 4

    Dleto.QDN_APPLY_COUNT[] = 0
    GC.gc()
    st_rf = @timed range_finder(m.L; k = k)
    applies_rf = Dleto.QDN_APPLY_COUNT[]
    Dleto.QDN_APPLY_COUNT[] = -1

    Lsq = m.L' * m.L
    arpack = Dleto.SOLVER_REGISTRY[:ArpackSolver]   # the struct lives in DletoArpackExt
    Dleto.QDN_APPLY_COUNT[] = 0
    GC.gc()
    st_ap = @timed Dleto.solve(arpack, Lsq; nv = k, seed = 20260908 + 1)
    applies_ap = Dleto.QDN_APPLY_COUNT[]
    Dleto.QDN_APPLY_COUNT[] = -1

    rf, ap = st_rf.value, st_ap.value
    # The operator's OWN norm, not `maximum(ap.vals)` -- at this small a `k`
    # (oracle + 4) the returned window can be all-null with no non-null value
    # in it at all, which made `maximum(ap.vals)` a proxy for "the largest of
    # a handful of near-zero numbers" rather than a scale, and undercounted
    # nullity as 0 even when the principal angle below confirms the correct
    # subspace was found.
    scale = max(Dleto.opnorm_estimate(Lsq; iters = 10), eps())
    n_rf = count(v -> v^2 <= 1e-6 * scale, rf.vals)     # rf.vals are sigma; ap.vals are sigma^2
    n_ap = count(v -> v <= 1e-6 * scale, ap.vals)

    kk = min(size(rf.vecs, 2), size(ap.vecs, 2), c.restricted_oracle)
    principal_angle_cos = if kk > 0
        Qr = Matrix(qr(rf.vecs[:, 1:kk]).Q)
        Qa = Matrix(qr(ap.vecs[:, 1:kk]).Q)
        minimum(svdvals(Qr' * Qa))
    else
        NaN
    end

    csv_row(rcsv("range-finder.csv"),
            "case,label,restricted_rows,restricted_cols,k,applies_rangefinder,seconds_rangefinder," *
            "applies_arpack,seconds_arpack,nullity_rangefinder,nullity_arpack,oracle_nullity," *
            "principal_angle_cos",
            name, "\"$(c.label)\"", m.nrows, m.ncols, k, applies_rf, fmt(st_rf.time),
            applies_ap, fmt(st_ap.time), n_rf, n_ap, c.restricted_oracle, fmt(principal_angle_cos))
    @printf("%-14s k=%-4d applies: rangefinder=%-7d arpack=%-7d  seconds: rf=%7.3f ap=%7.3f \
nullity: rf=%d ap=%d(oracle=%d) angle_cos=%.6f\n",
            name, k, applies_rf, applies_ap, st_rf.time, st_ap.time, n_rf, n_ap, c.restricted_oracle,
            principal_angle_cos)
    return nothing
end
