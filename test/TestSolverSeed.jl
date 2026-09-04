#
# TestSolverSeed -- an iterative null solve is a function of its arguments.
#
# THE FAILURE THIS PINS.  Every Krylov and block method here starts from a
# random vector or block that the caller did not choose: ARPACK keeps its start
# vector's seed inside the Fortran library ACROSS calls, KrylovKit's block came
# from the task-local default RNG, and LOBPCG draws its own `X0`.  So the same
# map, tolerance and request could return a different subspace of a multiple
# eigenvalue on the second call in one process -- measured on the whitened
# restricted map of the scrambled sphere at d = 48 in Float32, three identical
# QuickDer calls gave restricted nullity 2, 3, 3.  That is not noise to average
# over: it means no benchmark row and no assertion about a Float32 count is
# reproducible except one call per process.
#
# `solve_nullspace(...; seed = ...)` fixes the start (`Dleto.wants_seed` says
# which solvers take it), and this file is the guarantee.
#
# INCLUDED LAST in runtests.jl, deliberately: it is the only test file that
# loads the solver extensions, and loading them registers matrix-free solvers
# that `AutoSolver` would then be free to choose.  Keeping that at the end
# leaves every other testset running against the registry it was written for.
#
using Dleto
using LinearAlgebra
using LinearMaps
using Random
using Test

# Both of these are in [deps], so the extensions load and register.
using IterativeSolvers      # :CGSolver, :LSMRSolver, :LanczosSolver
using KrylovKit             # :KrylovSolver

# Arpack is a WEAKDEP and is not in this project's manifest (see the "Next up"
# list in docs/CONTEXT.md).  Test it when the environment happens to provide
# it -- `bench/jl` does -- and say so plainly when it does not, rather than
# reporting a pass on a solver that never ran.
const HAVE_ARPACK = try
    @eval using Arpack
    true
catch err
    @info "TestSolverSeed: Arpack is not available in this environment; the " *
          "ARPACK reproducibility testset is skipped. (Arpack is a weakdep and " *
          "is not in the manifest.)"
    false
end

"""
A symmetric operator with a genuine 3-fold zero eigenvalue -- the case where a
start vector matters, because a single-vector Krylov method finds one copy of a
multiple eigenvalue per start vector and the rest by luck.

`n = 240` is above every dense shortcut in the solvers: ARPACK's is `n <= 32`,
block Lanczos's is `n <= max(32, 2*(nev+1))` in Float64.  A dense
factorization has no random start and would make the test vacuous.
"""
function seeded_test_map(n::Int = 240; seed::Int = 20260904)
    rng = MersenneTwister(seed)
    Q = Matrix(qr(randn(rng, n, n)).Q)
    λ = vcat(zeros(3), collect(range(0.05, 1.0; length = n - 3)))
    return LinearMaps.LinearMap(Symmetric(Q * Diagonal(λ) * transpose(Q)) |> Matrix;
                                issymmetric = true)
end

@testset "seeded iterative solves are reproducible" begin
    L = seeded_test_map()

    @testset "the trait says who takes a seed" begin
        # `solve_nullspace` may only forward `seed` to a solver that has the
        # keyword; a dense factorization has no random start to fix.
        @test !Dleto.wants_seed(Dleto.SVDSolver())
        @test !Dleto.wants_seed(Dleto.GramSolver())
        @test Dleto.wants_seed(Dleto.AutoSolver())
        @test Dleto.seed_opts(Dleto.SVDSolver(), 7) === (;)
        @test Dleto.seed_opts(Dleto.AutoSolver(), 7) === (; seed = 7)
        @test Dleto.seed_opts(Dleto.AutoSolver(), nothing) === (;)
        # And a solver without the keyword is not handed one -- this is the
        # call that would throw `MethodError: unsupported keyword argument`
        # if the filtering were dropped.
        @test solve_nullspace(L, :SVDSolver; tol = 1e-6, nv0 = 8,
                              seed = 11).verdict.nullity == 3
    end

    if HAVE_ARPACK
        @testset "ArpackSolver: same seed, same spectrum and same nullity" begin
            arp = Dleto.SOLVER_REGISTRY[:ArpackSolver]
            @test Dleto.wants_seed(arp)

            a = solve_nullspace(L, arp; tol = 1e-6, nv0 = 16, seed = 4242)
            b = solve_nullspace(L, arp; tol = 1e-6, nv0 = 16, seed = 4242)
            # THE ASSERTION: two calls in ONE process, identical arguments.
            # Without `v0` these could differ, because ARPACK's saved seed has
            # moved on between them.
            @test a.verdict.nullity == b.verdict.nullity == 3
            @test length(a.verdict.spectrum) == length(b.verdict.spectrum)
            # The RELATIVE spectrum needs the second random start fixed too:
            # `solve_nullspace` divides by an `opnorm_estimate`, itself a power
            # iteration from a random vector, which is also what the null
            # threshold is multiplied by.  Unseeded, these two lines fail while
            # the `vals` line below passes -- measured 8.4e-4 apart.
            @test maximum(abs.(a.verdict.spectrum .- b.verdict.spectrum)) <= 1e-12
            @test a.verdict.scale == b.verdict.scale
            @test maximum(abs.(a.vals .- b.vals)) <= 1e-12 * max(a.verdict.scale, 1.0)

            # A different seed is a different start vector, so it need not give
            # the same values to 1e-12 -- but it must give the same ANSWER.
            c = solve_nullspace(L, arp; tol = 1e-6, nv0 = 16, seed = 99)
            @test c.verdict.nullity == 3
        end
    end

    @testset "KrylovSolver: same seed, same spectrum and same nullity" begin
        kry = Dleto.SOLVER_REGISTRY[:KrylovSolver]
        @test Dleto.wants_seed(kry)
        a = solve_nullspace(L, kry; tol = 1e-10, nv0 = 8, seed = 4242)
        b = solve_nullspace(L, kry; tol = 1e-10, nv0 = 8, seed = 4242)
        @test a.verdict.nullity == b.verdict.nullity == 3
        @test length(a.verdict.spectrum) == length(b.verdict.spectrum)
        @test maximum(abs.(a.verdict.spectrum .- b.verdict.spectrum)) <= 1e-12
    end

    @testset "CGSolver: same seed, same spectrum and same nullity" begin
        cg = Dleto.SOLVER_REGISTRY[:CGSolver]
        @test Dleto.wants_seed(cg)
        a = solve_nullspace(L, cg; tol = 1e-10, nv0 = 8, seed = 4242)
        b = solve_nullspace(L, cg; tol = 1e-10, nv0 = 8, seed = 4242)
        @test a.verdict.nullity == b.verdict.nullity
        @test length(a.verdict.spectrum) == length(b.verdict.spectrum)
        @test maximum(abs.(a.verdict.spectrum .- b.verdict.spectrum)) <= 1e-12
    end
end

# --- a non-ok status withholds the certificate -----------------------------
#
# `certified` is a claim that the spectrum PROVES the nullity.  A solve that
# did not converge, or that ran out of room before it bracketed the cut, did
# not compute that spectrum: unconverged Ritz values sit ABOVE their true
# eigenvalues, so a stalled cluster produces a textbook gap over nothing.  The
# plumbing is pinned by the stubs in TestNullVerdict.jl; this is the same rule
# on a REAL solver.
#
# HOW `:capped` IS FORCED, and why not with `maxiter`.  A tiny `maxiter` does
# not produce a capped verdict at all -- `Arpack.eigs` raises
# `XYAUPD_Exception` when it exhausts its iterations (measured here at
# maxiter = 1, 2 and 4 on a clustered 400 x 400 operator), which propagates
# and never reaches `gap_verdict`.  The status that IS reachable from the
# escalation is `:capped`: "the request reached the dimension of the space and
# the cut was still not bracketed".  `min_above` is the documented knob that
# decides bracketing, so asking for more values above the cut than the
# operator has drives the loop to `k >= N` with the cut still unbracketed --
# a genuine ARPACK run, a genuine cap, and a spectrum whose gap would
# otherwise certify.
if HAVE_ARPACK
    @testset "a non-ok status withholds the certificate (ARPACK)" begin
        L = seeded_test_map(40)          # above ARPACK's n <= 32 dense shortcut
        arp = Dleto.SOLVER_REGISTRY[:ArpackSolver]

        # The control: the same map, the same solver, bracketing normally.
        okr = solve_nullspace(L, arp; tol = 1e-6, nv0 = 16, seed = 4242)
        @test okr.verdict.status === :ok
        @test okr.verdict.nullity == 3
        @test okr.verdict.certified
        @test okr.verdict.rule === :gap

        # And now the cap: 40 values above the cut are demanded of an operator
        # that has 37, so the escalation runs to k = N and gives up.
        cap = solve_nullspace(L, arp; tol = 1e-6, nv0 = 16, seed = 4242,
                              min_above = 40)
        @test cap.verdict.status === :capped
        # The NUMBERS are untouched -- same cut, same rule, same gap as the
        # control -- so the caller still sees what was computed.
        @test cap.verdict.nullity == okr.verdict.nullity
        @test cap.verdict.rule === :gap
        @test cap.verdict.gap >= Dleto.GAP_RATIO
        # Only the certificate is withheld, and that is the whole change.
        @test !cap.verdict.certified
        @test occursin("UNCERTIFIED", sprint(show, cap.verdict))
        @test occursin("solver :capped", sprint(show, cap.verdict))
    end
end

@testset "status and undecidable are independent vetoes" begin
    # They answer different questions -- the SOLVER versus the DATA -- and
    # either alone is enough to withhold the certificate.  Built directly on
    # `gap_verdict` so both can be set without a solver that has to fail.
    floorv = Dleto.FLOOR_EPS * eps(Float64)
    rel = Float64[0.0, 3e-16, 2e-16, 1e-2, 0.3, 1.0]     # a clean nullity 3

    _, ok = gap_verdict(rel, 1.0; threshold = 1e-6, floor = floorv)
    @test ok.certified && ok.status === :ok && ok.undecidable == 0

    # Solver failed, data fine.
    _, bad_solver = gap_verdict(rel, 1.0; threshold = 1e-6, floor = floorv,
                                status = :unconverged)
    @test !bad_solver.certified
    @test bad_solver.undecidable == 0        # the data had nothing to say
    @test bad_solver.nullity == ok.nullity   # and the cut did not move
    @test bad_solver.gap == ok.gap

    # Data too coarse, solver fine: a Float16 tensor solved in Float32 whose
    # first nonzero value lies inside the rounding of the input.
    _, bad_data = gap_verdict(rel, 1.0; threshold = 1e-6, floor = floorv,
                              data_floor = 0.1)
    @test !bad_data.certified
    @test bad_data.status === :ok
    @test bad_data.undecidable > 0

    # `_with_status` applies the same rule when the status is stamped later,
    # and only ever withdraws: an `:ok` stamp cannot restore a certificate the
    # data floor already refused.
    @test !Dleto._with_status(ok, :capped).certified
    @test Dleto._with_status(ok, :ok).certified
    @test !Dleto._with_status(bad_data, :ok).certified
end
