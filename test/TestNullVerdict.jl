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
        # `floorv/2` against a floor of `FLOOR_EPS*eps(Float32)` = 6.0e-7.
        # With a wide enough `tol` to admit it as a candidate cut, the gap
        # test can only measure it from the floor, not from itself, so it is
        # floor-bound.  Stated as a fraction of the floor rather than as a
        # literal, so the test follows `FLOOR_EPS` when it is retuned.
        floorv = FLOOR_EPS * eps(Float32)
        threshold = 1e-3   # wide enough that floorv/2 is a candidate
        rel = Float64[0.0, 0.0, 0.0, floorv / 2, 1e-2, 1.0]
        _, v = gap_verdict(rel, 1.0; threshold, floor = floorv, gap_ratio = GAP_RATIO)
        @test v.floor_binding
        # Either the merge is accepted (nullity 4, :gap, floor-bound) or no
        # gap clears and it falls back to :threshold -- both are legitimate,
        # but a silent, unflagged nullity-4 is not.
        @test v.rule == :threshold || (v.rule == :gap && v.floor_binding)
    end

    @testset "Float32 floor raises the effective threshold above tol" begin
        # `solve_nullspace` uses `threshold = max(tol, precision_floor(T))`,
        # and on the SQUARED (Gram) map it squares `tol` first, so in Float32
        # the floor OVERRIDES the resulting ceiling -- deliberately, so
        # Float32 does not lose a null vector to its own rounding (see
        # `FLOOR_EPS`).  A value below the floor then falls inside the
        # (floor-widened) threshold and IS a candidate cut: it gets folded
        # into `nullity` here, floor-bound, because nothing separates it from
        # the null cluster at Float32 precision.
        floorv = FLOOR_EPS * eps(Float32)
        threshold = max(1e-6^2, floorv)
        rel = Float64[0.0, 0.0, 0.0, floorv / 2, 1e-2, 1.0]
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
        # A near-derivation BELOW the Float32 floor (`FLOOR_EPS*eps(Float32)`
        # = 6.0e-7).  `solve_nullspace` uses that floor to widen the effective
        # threshold, so the value is folded into nullity 4 -- but it must come
        # back `floor_binding`, with an `@info` naming the cut, never a bare
        # unflagged nullity 4.  Written as a fraction of the floor rather than
        # as a literal, so it follows `FLOOR_EPS` when that is retuned.
        floorv = Float32(FLOOR_EPS * eps(Float32))
        relvals = Float32[0, 0, 0, floorv / 3, 1f-2, 1]
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

    @testset "Float32 keeps a near-derivation ABOVE the precision floor" begin
        # The other side of the same line, and the reason `FLOOR_EPS` was
        # retuned from 100 to 5 (see its docstring): a near-derivation at 3e-6
        # relative is FIVE times above the Float32 floor and is a real feature
        # of the operator, so Float32 must report nullity 3 and leave it out,
        # not fold it in.  Under the old floor of 100*eps(Float32) = 1.2e-5 it
        # was swallowed and the answer was a certified 4.
        relvals = Float32[0, 0, 0, 3f-6, 1f-2, 1]
        D = Diagonal(Float32.(5.0f0 .* relvals))
        L = LinearMaps.LinearMap(D; issymmetric = true)
        res = solve_nullspace(L, :SVDSolver; tol = 1e-6, nv0 = 6)
        @test 3f-6 > FLOOR_EPS * eps(Float32)      # the premise
        @test res.verdict.nullity == 3
        # And it is still visible to the caller, as the first value above the
        # cut rather than as a silently-counted derivation.
        @test isapprox(res.verdict.above[1], 3f-6; rtol = 0.1)
        # NOT certified, and that is the point: 3e-6 is 5x the floor, which is
        # a real separation but not the 100x `GAP_RATIO` asks for.  Float32
        # genuinely cannot tell this near-derivation from zero with
        # confidence, and says so instead of guessing either way.
        @test !res.verdict.certified
        @test isapprox(res.verdict.gap, 3f-6 / (FLOOR_EPS * eps(Float32)); rtol = 0.1)

        # A near-derivation FAR above the floor is both kept out and certified.
        far = Float32[0, 0, 0, 1f-3, 1f-1, 1]
        Lf = LinearMaps.LinearMap(Diagonal(Float32.(5.0f0 .* far)); issymmetric = true)
        resf = solve_nullspace(Lf, :SVDSolver; tol = 1e-6, nv0 = 6)
        @test resf.verdict.nullity == 3
        @test resf.verdict.certified
    end
end

# --- the solver's own word, kept apart from the verdict on the spectrum ----
#
# The failure this pins was live and silent: an iterative solver that converges
# to NOTHING returns Ritz values stalled well above zero, the gap test reads
# that spectrum as "nothing near zero", and the answer comes back
# `nullity 0, certified` -- indistinguishable from the legitimate "this tensor
# conforms to no pattern".  Measured on the whitened restricted map at d = 300
# and d = 500 valence 3, block Lanczos, spectrum stalled at 1.1e-9 relative.
#
# Two stub solvers, because the point is the PLUMBING and a real
# non-convergence is neither deterministic nor cheap: one reports
# `converged = false`, one omits the field (a dense factorisation has no
# iteration to fail) and must be taken at its word.
struct StubStalledSolver <: Dleto.NullSolver end
struct StubSilentSolver <: Dleto.NullSolver end

# A spectrum with a clean gap and nothing at zero: exactly what a stalled
# iterative solve produces, and exactly what the gap test certifies at 0.  The
# stall sits at 1.1e-4 relative rather than the measured 1.1e-9 only because
# this stub is handed a SQUARE map, so `tol` is not squared here; on the real
# rectangular restricted map `tol = 1e-6` becomes a ceiling of 1e-12 and 1.1e-9
# lands in exactly this position relative to it.
const STUB_VALS = Float64[1.1e-4, 1.7e-4, 3.2e-4, 1e-2, 0.3, 1.0]

Dleto.solve(::StubStalledSolver, L::LinearMaps.LinearMap; nv::Integer = 10, kwargs...) =
    (; vals = STUB_VALS[1:min(nv, length(STUB_VALS))],
       vecs = zeros(Float64, size(L, 2), min(nv, length(STUB_VALS))),
       converged = false)

Dleto.solve(::StubSilentSolver, L::LinearMaps.LinearMap; nv::Integer = 10, kwargs...) =
    (; vals = STUB_VALS[1:min(nv, length(STUB_VALS))],
       vecs = zeros(Float64, size(L, 2), min(nv, length(STUB_VALS))))

@testset "solve_nullspace: a failed iterative solve says so" begin
    L = LinearMaps.LinearMap(Diagonal(collect(STUB_VALS)); issymmetric = true)

    @testset "converged = false, nullity 0 -> :unconverged and a warning" begin
        local res
        logs = Test.collect_test_logs() do
            res = solve_nullspace(L, StubStalledSolver(); tol = 1e-6, nv0 = 6)
        end |> first
        # The verdict on the SPECTRUM is unchanged -- there is genuinely
        # nothing near zero in it, and certifying that is correct.
        @test res.verdict.nullity == 0
        @test res.verdict.certified
        # The verdict on the SOLVER is the new word, and it is the one that
        # says this is a failure and not an empty null space.
        @test res.verdict.status === :unconverged
        @test any(r -> r.level == LogWarn && occursin("FAILED", r.message), logs)
        @test occursin("solver :unconverged", sprint(show, res.verdict))
    end

    @testset "no `converged` field -> :ok, and no warning about the solver" begin
        local res
        logs = Test.collect_test_logs() do
            res = solve_nullspace(L, StubSilentSolver(); tol = 1e-6, nv0 = 6)
        end |> first
        @test res.verdict.status === :ok
        @test Dleto.solver_converged((; vals = [1.0], vecs = zeros(1, 1)))
        @test !any(r -> r.level == LogWarn, logs)
        @test !occursin("solver :", sprint(show, res.verdict))
    end

    @testset "a real solver still reports :ok" begin
        D = Diagonal([0.0, 0.0, 0.0, 7e-2, 2.1, 7.0])
        Ld = LinearMaps.LinearMap(D; issymmetric = true)
        res = solve_nullspace(Ld, :SVDSolver; tol = 1e-6, nv0 = 6)
        @test res.verdict.nullity == 3
        @test res.verdict.status === :ok
    end
end
