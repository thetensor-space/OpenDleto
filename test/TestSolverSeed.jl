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
