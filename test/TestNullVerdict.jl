#
# TestNullVerdict -- the sigma_(e+1) gap test in src/solvers/NullSolvers.jl.
#
# Two layers:
#   1. `gap_verdict` directly, on hand-built relative spectra, so the cut
#      logic is pinned independent of any particular solver's numerics.
#   2. `solve_nullspace` end-to-end on a diagonal `LinearMap` with a known
#      spectrum, via `:SVDSolver` (no optional extension needed), so the
#      NamedTuple contract -- `(vals, vecs, verdict)`, and the old
#      `(vals, vecs) = solve_nullspace(...)` destructuring -- is exercised
#      for real.
#
using Dleto
using Dleto: gap_verdict, NullVerdict, FLOOR_EPS, GAP_RATIO
using LinearAlgebra
using LinearMaps
using Test

# `Logging` is not in this package's [extras]/test target, so `Base.CoreLogging`
# (part of Base, no package needed) supplies the two level constants used
# below to tell an `@info` from a `@warn` in `Test.collect_test_logs`.
const LogInfo = Base.CoreLogging.Info
const LogWarn = Base.CoreLogging.Warn

@testset "gap_verdict" begin

    @testset "clean nullity-3 cluster, Float64" begin
        # Null cluster at rounding, then a clear jump to O(1) values -- the
        # ordinary case: every seed on the sphere benchmark except d=50/50.
        floorv = FLOOR_EPS * eps(Float64)
        threshold = max(1e-6, floorv)
        rel = Float64[0.0, 3e-16, -2e-16, 1e-2, 0.3, 1.0]
        _, v = gap_verdict(rel, 1.0; threshold, floor = floorv, gap_ratio = GAP_RATIO)
        @test v.nullity == 3
        @test v.rule == :gap
        @test v.certified
        @test v.floor_binding   # the value just below the cut sits under the floor
        @test v.near_null == 0
    end

    @testset "near-derivation below tol is not swallowed" begin
        # A value at 1e-9 is below tol=1e-6 -- the fixed threshold would have
        # counted it null -- but the ratio from the (floored) null cluster up
        # to it, 1e-9/2.2e-14 ~= 4.5e4, is bigger than the ratio from it up to
        # the next real eigenvalue, 1e-6/1e-9 = 1e3 (itself not counted below
        # threshold, since 1e-6 is not STRICTLY less than threshold).  The
        # dominant gap therefore falls at 3, and the near-derivation shows up
        # as `near_null`, not folded into `nullity` -- this is the shape of
        # the sphere benchmark's seed 50, d = 50 case (see gap-verdict.md).
        floorv = FLOOR_EPS * eps(Float64)
        threshold = max(1e-6, floorv)
        rel = Float64[0.0, 0.0, 0.0, 1e-9, 1e-6, 0.3, 1.0]
        _, v = gap_verdict(rel, 1.0; threshold, floor = floorv, gap_ratio = GAP_RATIO)
        @test v.nullity == 3
        @test v.near_null == 1
        @test v.rule == :gap
        @test v.certified
    end

    @testset "no dominant gap falls back to :threshold, uncertified" begin
        # A smoothly graded spectrum with NO genuine null cluster near the
        # precision floor -- consecutive ratios are all ~3x, well under
        # gap_ratio -- so no cut clears the bar and the old fixed-threshold
        # count is used, flagged uncertified.
        floorv = FLOOR_EPS * eps(Float64)
        threshold = 1e-3
        rel = Float64[1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
        _, v = gap_verdict(rel, 1.0; threshold, floor = floorv, gap_ratio = GAP_RATIO)
        @test v.rule == :threshold
        @test !v.certified
        @test v.nullity == count(<(threshold), rel)
    end

    @testset "nullity 0, certified" begin
        floorv = FLOOR_EPS * eps(Float64)
        threshold = 1e-6
        rel = Float64[1e-2, 0.3, 1.0]
        _, v = gap_verdict(rel, 1.0; threshold, floor = floorv, gap_ratio = GAP_RATIO)
        @test v.nullity == 0
        @test v.certified
    end

    @testset "nd > 0 forces :fixed, never certified" begin
        floorv = FLOOR_EPS * eps(Float64)
        threshold = 1e-6
        rel = Float64[0.0, 0.0, 0.0, 1e-2, 1.0]
        _, v = gap_verdict(rel, 1.0; threshold, floor = floorv, gap_ratio = GAP_RATIO, nd = 2)
        @test v.rule == :fixed
        @test v.nullity == 2
        @test !v.certified
    end

    @testset "Float32 floor swallows a sub-floor near-derivation" begin
        # A near-derivation hidden BELOW the Float32 floor: relative value
        # 3e-6 against a floor of 100*eps(Float32) ~= 1.19e-5.  With a wide
        # enough `tol` to admit it as a candidate cut, the gap test can only
        # measure it from the floor, not from itself, so it is floor-bound.
        floorv = FLOOR_EPS * eps(Float32)
        threshold = 1e-3   # wide enough that 3e-6 is a candidate
        rel = Float64[0.0, 0.0, 0.0, 3e-6, 1e-2, 1.0]
        _, v = gap_verdict(rel, 1.0; threshold, floor = floorv, gap_ratio = GAP_RATIO)
        @test v.floor_binding
        # Either the merge is accepted (nullity 4, :gap, floor-bound) or no
        # gap clears and it falls back to :threshold -- both are legitimate,
        # but a silent, unflagged nullity-4 is not.
        @test v.rule == :threshold || (v.rule == :gap && v.floor_binding)
    end

    @testset "Float32 floor raises the effective threshold above tol" begin
        # `solve_nullspace` uses `threshold = max(tol, FLOOR_EPS*eps(T)) *
        # scale`, so in Float32 the floor (~1.19e-5) OVERRIDES the default
        # tol=1e-6 -- deliberately, so Float32 does not lose a null vector to
        # its own rounding (see FLOOR_EPS).  A value at 3e-6 then falls inside
        # the (floor-widened) threshold and IS a candidate cut: it gets
        # folded into `nullity` here, floor-bound, because nothing separates
        # it from the null cluster at Float32 precision.
        floorv = FLOOR_EPS * eps(Float32)
        threshold = max(1e-6, floorv)
        rel = Float64[0.0, 0.0, 0.0, 3e-6, 1e-2, 1.0]
        _, v = gap_verdict(rel, 1.0; threshold, floor = floorv, gap_ratio = GAP_RATIO)
        @test v.nullity == 4
        @test v.floor_binding
        @test v.certified
    end
end

@testset "solve_nullspace end-to-end (diagonal LinearMap, :SVDSolver)" begin
    for (T, label) in ((Float64, "Float64"), (Float32, "Float32"))
        @testset "$label" begin
            # Relative spectrum [0,0,0,1e-2,0.3,1] scaled by an arbitrary
            # operator norm, so `solve_nullspace`'s own `scale` estimate
            # (power iteration on LᵗL) is exercised, not just `gap_verdict`.
            # A clean cluster with no near-derivation subtlety -- that case
            # is covered separately below -- just the contract: nullity 3,
            # `(vals, vecs)` destructures, and the `verdict` field is there.
            relvals = T[0, 0, 0, 1e-2, 0.3, 1]
            top = 7.0
            D = Diagonal(T.(top .* relvals))
            L = LinearMaps.LinearMap(D; issymmetric = true)

            (vals, vecs) = solve_nullspace(L, :SVDSolver; tol = 1e-6, nv0 = 6)
            @test length(vals) == 3
            @test size(vecs) == (6, 3)

            full = solve_nullspace(L, :SVDSolver; tol = 1e-6, nv0 = 6)
            @test full.verdict isa NullVerdict
            @test full.verdict.nullity == 3
            @test full.verdict.rule == :gap
            @test full.verdict.certified
        end
    end

    @testset "Float32 gap below the precision floor is flagged, not silent" begin
        # Relative spectrum [0,0,0,3e-6,1e-2,1]: the near-derivation at 3e-6
        # sits below the Float32 floor (100*eps(Float32) ~= 1.19e-5), which
        # `solve_nullspace` uses to WIDEN the effective threshold past the
        # default tol=1e-6 (see FLOOR_EPS).  So it gets folded into nullity 4
        # -- but must come back `floor_binding`, with an `@info` naming the
        # cut, never a bare, unflagged nullity 4.
        relvals = Float32[0, 0, 0, 3e-6, 1e-2, 1]
        D = Diagonal(Float32.(5.0f0 .* relvals))
        L = LinearMaps.LinearMap(D; issymmetric = true)

        local res
        logs = Test.collect_test_logs() do
            res = solve_nullspace(L, :SVDSolver; tol = 1e-6, nv0 = 6)
        end |> first
        @test res.verdict.nullity == 4
        @test res.verdict.floor_binding
        @test res.verdict.certified
        # Certified-but-floor-bound: an @info, not a @warn -- the routine
        # signal, not an alarm.
        @test any(r -> r.level == LogInfo && occursin("precision floor", r.message),
                  logs)
        @test !any(r -> r.level == LogWarn, logs)
    end
end
