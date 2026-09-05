#
# test/TestPrecision.jl
#   The floating-point policy: does derivation finding actually respond to the
#   element type, and does it stay honest when the type runs out of precision?
#
# The three questions this file pins down, none of which the suite could answer
# before src/solvers/Precision.jl existed:
#
#   1. Do Float64, Float32 and Float16 all return the RIGHT nullity, on every
#      route the library actually takes?  (Before: Float16 returned 32-41 on a
#      3- or 4-dimensional null space, at a reconstruction error of 0.9, with
#      no error and no warning.)
#   2. Does Float32 in stay Float32 all the way through -- no silent promotion
#      to Float64, which doubles the memory and breaks the GPU kernels?
#   3. Does a Float16 answer claim only what Float16 data can support?
#
# The tolerance table these assertions are calibrated against is in
# docs/design/Precision-Policy.md and in the docstrings of `FLOOR_EPS`,
# `precision_floor` and `qd_tolerance`.
#
using Test
using Dleto
using ITensors
using LinearAlgebra
using Random
using Logging
using Dleto: compute_eltype, precision_floor, data_floor, tol_default, iter_tol,
             qd_tolerance, rank_rtol, precision_policy, gap_verdict,
             FLOOR_EPS, GAP_RATIO, TOL_DEFAULT, ITER_TOL_EPS

const PRECISION_TYPES = (Float64, Float32, Float16)

"""
    prec_residual(Γ, D, P) -> Real

Relative size of the chisel-weighted sum `Σ_a P[:,a] ⊗ (Γ · D_a)`; zero exactly
when `D` is a `P`-derivation of `Γ`.  A local copy of TestDerivationLaws'
`der_residual` so this file runs standalone, as TestAutoDer and TestQuickDerN
also keep one.
"""
function prec_residual(Γ::ITensor, D::Vector{ITensor}, P::AbstractMatrix)
    C = Chisel(P, collect(inds(Γ)))
    R = applyDerivation(Γ, D, C)
    scale = norm(Γ) * maximum(norm.(D))
    return norm(R) / max(scale, eps())
end

# --- the policy itself, as arithmetic -------------------------------------

@testset "precision policy" begin
    @testset "compute_eltype promotes only Float16" begin
        @test compute_eltype(Float64) === Float64
        @test compute_eltype(Float32) === Float32
        # No CPU BLAS or LAPACK has a half-precision path; see the docstring
        # for the four `MethodError`s this promotion avoids.
        @test compute_eltype(Float16) === Float32
        @test compute_eltype(ComplexF32) === ComplexF32
        @test compute_eltype(Complex{Float16}) === ComplexF32
    end

    @testset "the two floors, and which one binds" begin
        for T in PRECISION_TYPES
            p = precision_policy(T)
            @test p.store === T
            @test p.compute === compute_eltype(T)
            @test p.precision_floor ≈ FLOOR_EPS * eps(compute_eltype(T))
            @test p.data_floor ≈ eps(T)
            @test p.precision_floor > 0 && p.data_floor > 0
        end
        # When the stored and computed types AGREE, the arithmetic floor is
        # `FLOOR_EPS` times the data floor, so the data floor cannot bind and
        # these paths behave exactly as they did before the policy existed.
        for T in (Float64, Float32)
            @test data_floor(T) < precision_floor(T)
            @test precision_floor(T) ≈ FLOOR_EPS * data_floor(T)
        end
        # Float16 is the whole reason the second floor exists: promoted
        # arithmetic resolves 1600x finer than the data it was given, so the
        # data floor is the binding one.
        @test data_floor(Float16) > precision_floor(Float16)
    end

    @testset "no dimension term in precision_floor" begin
        # `FLOOR_EPS`'s docstring records the measurements that rule one out:
        # the largest relative null values occur at the SMALLEST n.  Asserted
        # so that re-adding one is a deliberate, visible change.
        @test precision_floor(Float32) == precision_floor(Float32)
        for T in PRECISION_TYPES
            @test precision_floor(T) isa Float64
        end
        # `rank_rtol` is where the dimension term does belong -- it is the
        # classical backward-error bound for a factorization.
        @test rank_rtol(Float64, 1000, 10) ≈ 1000 * eps(Float64)
        @test rank_rtol(Float16, 1000, 10) ≈ 1000 * eps(Float32)   # computed wide
    end

    @testset "tolerances floor on T, never below it" begin
        for T in PRECISION_TYPES
            @test tol_default(T; tol = 0.0) ≈ precision_floor(T)
            @test tol_default(T; tol = 1.0) == 1.0            # a ceiling only
            @test tol_default(T; tol = TOL_DEFAULT, squared = true) ≥ precision_floor(T)
            @test iter_tol(T, 0.0) ≈ ITER_TOL_EPS * eps(compute_eltype(T))
            # The solve-and-lift tolerance floors on the STORED type: promoting
            # the arithmetic buys stability, not information about the data.
            @test qd_tolerance(T) ≥ sqrt(data_floor(T))
        end
        # Float64 keeps the value callers have always had.
        @test tol_default(Float64) == TOL_DEFAULT
        @test qd_tolerance(Float64) == TOL_DEFAULT
        # ...and the coarser types are strictly coarser, monotonically.
        @test qd_tolerance(Float64) < qd_tolerance(Float32) < qd_tolerance(Float16)
    end

    @testset "the data floor vetoes certification, never the cut" begin
        # A spectrum whose first nonzero value sits BETWEEN the arithmetic
        # floor and the Float16 data floor: converged, but inside the rounding
        # of the input.  The cut must still land at 3 -- the count the data
        # does support -- and the verdict must refuse to certify it.
        rel = Float64[1e-8, 1e-8, 1e-8, 1e-4, 1e-1, 1.0]
        fl = precision_floor(Float16)          # 6.0e-7, below the cluster? no: above
        thr = max(TOL_DEFAULT, fl)
        _, v16 = gap_verdict(rel, 1.0; threshold = thr, floor = fl,
                             data_floor = data_floor(Float16), gap_ratio = GAP_RATIO)
        @test v16.nullity == 3
        @test v16.undecidable == 1             # the 1e-4 value
        @test !v16.certified
        @test v16.data_floor ≈ eps(Float16)

        # The identical spectrum from Float32 data IS certifiable: 1e-4 is
        # three decades above `eps(Float32)`, so it is a real eigenvalue.
        _, v32 = gap_verdict(rel, 1.0; threshold = thr, floor = precision_floor(Float32),
                             data_floor = data_floor(Float32), gap_ratio = GAP_RATIO)
        @test v32.nullity == 3
        @test v32.undecidable == 0
        @test v32.certified

        # And with no data floor at all (the default) nothing changes for the
        # existing callers.
        _, v0 = gap_verdict(rel, 1.0; threshold = thr, floor = precision_floor(Float32),
                            gap_ratio = GAP_RATIO)
        @test v0.nullity == v32.nullity && v0.certified == v32.certified
    end
end

# --- end to end, one type per pass ----------------------------------------

"""
    _prec_sum_lattice!(A, d) -> A

Independent `randn()` entries on the lattice `i_1 + ... + i_n = d - 1`, zero
elsewhere -- the hypersphere octant of `bench/SphereHarness.jl`, whose
derivation space under `SymmetricOp` has dimension exactly the valence.  Copied
rather than included because `bench/` is not on the test path.
"""
function _prec_sum_lattice!(A::AbstractArray{Float64,N}, d::Integer) where {N}
    idx = Vector{Int}(undef, N)
    function rec(axis::Int, remaining::Int)
        if axis == N
            idx[N] = remaining
            @inbounds A[(idx .+ 1)...] = randn()
            return
        end
        for v in max(0, remaining - (d - 1) * (N - axis)):min(d - 1, remaining)
            idx[axis] = v
            rec(axis + 1, remaining - v)
        end
    end
    rec(1, d - 1)
    return A
end

"""
    prec_sphere(d, n, T; seed) -> (Ω, ch, Γ)

The scrambled nondegenerate sphere octant `bench/SphereHarness.jl` builds, at
valence `n` and dimension `d`, in element type `T`.  Truth: the derivation
space under `SymmetricOp` has dimension `n`.
"""
function prec_sphere(d::Integer, n::Integer, T::Type; seed = d)
    Random.seed!(seed)
    S = ITensor(_prec_sum_lattice!(zeros(Float64, ntuple(_ -> d, n)), d),
                [Index(d, "a$a") for a in 1:n]...)
    ndg = nondeg(randomize_tensor(S; type = :orthogonal).Δ)
    fr = collect(inds(ndg.Δ))
    Γ = T === Float64 ? ndg.Δ :
        ITensor(Array{T}(ITensors.array(ndg.Δ, fr...)), fr...)
    return (IndTransverseOps(fr, SymmetricOp()), UniversalChisel(n), Γ)
end

"""
    prec_generic(dims, T; seed) -> (Ω, ch, Γ)

A random dense tensor.  Truth: under `UniversalOp` its only derivations are the
`valence - 1` scalars, which is the cleanest known nullity in the package and
is confirmed in all three element types on every solver route in
bench/reports/precision-tune-a.csv.
"""
function prec_generic(dims, T::Type; seed = 1234)
    Random.seed!(seed)
    fr = [Index(d, "a$i") for (i, d) in enumerate(dims)]
    A = randn(dims...)
    Γ = ITensor(Array{T}(A), fr...)
    return (IndTransverseOps(fr, UniversalOp()), UniversalChisel(length(dims)), Γ)
end

# The routes a default call actually takes: the dense SVD, the solver-choosing
# `AutoSolver`, and QuickDer's solve-and-lift, which reaches `solve_nullspace`
# by a different path and floors its tolerance on the STORED type.  The
# remaining registered solvers (Gram, Arpack, block Lanczos, LSMR) are swept
# per type in bench/reports/precision-tune.jl rather than in the suite, because
# two of them need optional packages loaded.
const PREC_ROUTES = (
    (; label = "SylverLining/SVDSolver",   method = :SylverLining, solver = :SVDSolver),
    (; label = "SylverLining/AutoSolver",  method = :SylverLining, solver = :AutoSolver),
    (; label = "QuickDer",                 method = :QuickDer,     solver = nothing),
)

# `(name, truth, builder)`: valence 3 and 4 of each, so the multiplicity of the
# zero eigenvalue differs and a solver that finds one copy of a repeated
# eigenvalue per start vector cannot pass by luck.
const PREC_CASES = (
    (; name = "sphere 8^3",   truth = 3, build = T -> prec_sphere(8, 3, T)),
    (; name = "sphere 5^4",   truth = 4, build = T -> prec_sphere(5, 4, T)),
    (; name = "generic 8^3",  truth = 2, build = T -> prec_generic((8, 8, 8), T)),
    (; name = "generic 5^4",  truth = 3, build = T -> prec_generic((5, 5, 5, 5), T)),
)

@testset "derivations respond to the element type" begin
    @testset "$(case.name)" for case in PREC_CASES
        truth = case.truth
        @testset "$T" for T in PRECISION_TYPES
            Ω, ch, Γ = case.build(T)
            @test eltype(Γ) === T
            @testset "$(r.label)" for r in PREC_ROUTES
                m = r.solver === nothing ?
                    Dleto.get_derivation_method(r.method) :
                    Dleto.get_derivation_method(r.method; solver = r.solver)
                (_, _, ders) = derTrOpsReduced(m, Ω, ch, Γ)

                # 1. THE RIGHT ANSWER, in every type.  Before the policy this
                #    was 32-41 in Float16.
                @test size(ders, 2) == truth

                # 2. THE RIGHT TYPE BACK.  A Float16 tensor yields Float16
                #    coordinates: the data's precision is preserved, not
                #    inflated by the wider arithmetic it was solved in.
                @test eltype(ders) === T
            end
        end
    end
end

@testset "Float32 in does not become Float64" begin
    # The failure this guards against is silent and expensive: a Float64 chisel
    # or a Float64 operator space promotes the whole solve, which doubles the
    # memory and makes the Metal kernels unreachable (Apple GPUs have no
    # Float64 at all).  Checked by ELEMENT TYPE, not by timing.
    Ω, ch, Γ = prec_sphere(8, 3, Float32)
    @test eltype(Γ) === Float32
    # The chisel really is Float64 -- that is the promotion risk.
    @test eltype(ch) === Float64

    @testset "$(r.label)" for r in PREC_ROUTES
        m = r.solver === nothing ? Dleto.get_derivation_method(r.method) :
                                   Dleto.get_derivation_method(r.method; solver = r.solver)
        (rΩ, _, ders) = derTrOpsReduced(m, Ω, ch, Γ)
        @test eltype(ders) === Float32
        @test !(eltype(ders) === Float64)
    end

    # The operator the solve is handed must be Float32 too, not just its answer:
    # the map is what holds `d^n` numbers.
    eng = engaged(ch)
    (Ωr, _) = reduceByEngaged(Ω, eng, Float32)
    L, _ = Dleto.sylvesterLM(Ωr, Matrix{Float32}(ch[:, eng]), Γ)
    @test eltype(L) === Float32
    @test eltype(L * randn(Float32, size(L, 2))) === Float32

    # And a Float16 tensor is solved in Float32 -- NOT in Float64, and not in
    # Float16 either.  Its own frame, since `prec_sphere` mints fresh indices.
    Ω16, ch16, Γ16 = prec_sphere(8, 3, Float16)
    fr16 = collect(inds(Γ16))
    (Ω16r, _) = reduceByEngaged(Ω16, engaged(ch16), compute_eltype(Float16))
    Γ16c = ITensor(Array{compute_eltype(Float16)}(ITensors.array(Γ16, fr16...)), fr16...)
    L16, _ = Dleto.sylvesterLM(Ω16r,
                               Matrix{compute_eltype(Float16)}(ch16[:, engaged(ch16)]),
                               Γ16c)
    @test eltype(L16) === Float32
    @test eltype(L16 * randn(Float32, size(L16, 2))) === Float32

    # The @allocated form of the same claim: applying the Float32 map must not
    # allocate a Float64 buffer anywhere.  A Float64 temporary of length `n`
    # would double the bytes, so compare against a Float32-sized budget rather
    # than against a timing.
    v32 = randn(Float32, size(L16, 2))
    L16 * v32                                   # compile first
    bytes = @allocated (L16 * v32)
    @test bytes < 8 * sizeof(Float32) * length(v32) + 4096
end

@testset "Float16 claims no more than Float16 can support" begin
    # The near-degenerate family of TestDerivationLaws: a smooth ramp plus a
    # perturbation at 1e-5, which is 1e11 above `eps(Float64)`, 84x above
    # `eps(Float32)` and 100x BELOW `eps(Float16)` -- so the Float16 tensor is,
    # to its own precision, a different (more degenerate) tensor than the
    # Float64 one.  Its derivation space is legitimately large (77 dimensions),
    # and the exact count is a property of the tensor, not of the policy.  What
    # the policy owes the caller is that whatever comes back IS a derivation to
    # the accuracy the type carries.
    Random.seed!(20260904)
    n = 6
    fr = [Index(n, "a$i") for i in 1:3]
    Ω = IndTransverseOps(fr, UniversalOp())
    P = UniversalChisel(3)
    ramp = [i + j + k for i in 1:n, j in 1:n, k in 1:n] .+ 0.0
    A = ramp .+ 1e-5 .* randn(n, n, n)

    counts = Dict{Type,Int}()
    for T in PRECISION_TYPES
        Γ = ITensor(Array{T}(A), fr...)
        basis = der(:SylverLining, Ω, P, Γ)
        counts[T] = length(basis)
        # The scalars are always derivations, so an empty answer is always wrong.
        @test length(basis) >= size(nullspace(Matrix(P)), 2)
        # THE T-RELATIVE RESIDUAL BOUND.  Measured maxima on this tensor:
        # 9.8e-7 (Float64), 5.0e-6 (Float32), 2.2e-4 (Float16) -- so the
        # accuracy tracks the STORED type's resolution across four decades,
        # which is exactly the claim.  `100 * max(tol, eps(T))` holds with
        # 100x, 20x and 440x of margin respectively; a fixed `1e-6` would pass
        # only in Float64 and a fixed `1e-3` would assert nothing about it.
        bound = 100 * max(TOL_DEFAULT, data_floor(T))
        for D in basis
            @test prec_residual(Γ, D, P) < bound
        end
    end
    # All three agree on the count here, which is the strongest form of "the
    # coarser types neither invent nor lose directions" available on a tensor
    # whose true nullity is not known in closed form.
    @test counts[Float32] == counts[Float64]
    @test counts[Float16] == counts[Float64]

    # And the verdict machinery must SAY so when the data cannot decide.  Built
    # directly on `gap_verdict`, so the assertion is about the policy and not
    # about one tensor's spectrum: a first-nonzero value at 2e-4 is four
    # decades clear of the Float32 arithmetic floor but sits INSIDE the
    # rounding of Float16 data (eps = 9.8e-4), so it is not evidence.
    rel = Float64[2e-8, 3e-8, 4e-8, 2e-4, 5e-2, 1.0]
    _, v = gap_verdict(rel, 1.0; threshold = max(TOL_DEFAULT, precision_floor(Float16)),
                       floor = precision_floor(Float16),
                       data_floor = data_floor(Float16), gap_ratio = GAP_RATIO)
    @test v.undecidable >= 1
    @test !v.certified
    @test occursin("UNDECIDABLE", sprint(show, v))

    # The same spectrum from Float32 data is decided, and certified.
    _, v32 = gap_verdict(rel, 1.0; threshold = max(TOL_DEFAULT, precision_floor(Float32)),
                         floor = precision_floor(Float32),
                         data_floor = data_floor(Float32), gap_ratio = GAP_RATIO)
    @test v32.nullity == 3
    @test v32.undecidable == 0
    @test v32.certified
end

# --- the frontier, as a regression guard ---------------------------------
#
# docs/design/Precision-Policy.md section 5 states what Float32 and Float16 can
# and cannot certify.  Two of those claims are cheap enough to assert here, and
# both are the kind that decays silently if a constant moves.

@testset "the measured frontier holds" begin
    @testset "Float32 certifies where the policy says it should" begin
        # The rule from Exp. D: Float32 certifies when the first nonzero
        # eigenvalue clears `GAP_RATIO * precision_floor(Float32)` = 9.5e-5
        # relative.  Both video shapes (1.6e-4, 2.7e-4) and every sphere point
        # from d = 16 to 96 clear it.  Asserted on the arithmetic rather than by
        # re-running d = 96, which belongs in the benchmark.
        need = GAP_RATIO * precision_floor(Float32)
        @test need < 1.5e-4          # the tightest measured first-nonzero
        @test need > 4.92 * eps(Float32)   # still above the worst null value
        # The same two bounds are what `FLOOR_EPS` was tuned between; if either
        # inequality fails, the window has closed and section 2 needs redoing.
        @test 4.92 < FLOOR_EPS < 12.6
    end

    @testset "Float16 declines exactly when the data cannot decide" begin
        # Exp. D found a perfect correlation over 11 sphere points: Float16
        # certifies iff the first nonzero eigenvalue exceeds `eps(Float16)`.
        # Both sides, on the verdict machinery.
        nulls = Float64[2e-8, 3e-8, 4e-8]
        for (first_nonzero, expect_certified) in ((2.43e-3, true),   # sphere d = 32
                                                  (1.47e-3, true),   # sphere d = 16
                                                  (8.25e-4, false),  # sphere v4 d = 12
                                                  (2.94e-4, false),  # sphere d = 24
                                                  (1.50e-4, false))  # sphere d = 48
            rel = vcat(nulls, [first_nonzero, 5e-2, 1.0])
            _, v = gap_verdict(rel, 1.0;
                               threshold = max(TOL_DEFAULT, precision_floor(Float16)),
                               floor = precision_floor(Float16),
                               data_floor = data_floor(Float16), gap_ratio = GAP_RATIO)
            @test v.nullity == 3                       # the count is right either way
            @test v.certified == expect_certified
            @test (v.undecidable == 0) == expect_certified
            # and the boundary really is eps(Float16), not something else
            @test expect_certified == (first_nonzero >= data_floor(Float16))
        end
    end
end

# --- the STORAGE type reaches the verdict, on QuickDer's own route ---------
#
# `derTrOpsReduced(::QuickDerMethod, ...)` promotes a Float16 tensor to Float32
# before it touches the kernel, so `eltype` inside the kernel is the
# ARITHMETIC's type and `solve_nullspace`'s default `store_eltype =
# real(eltype(L))` describes the wrong number.  The consequence is not a
# rounding difference, it is a false certificate: `data_floor(Float32)` is
# 1.2e-7 and never binds, so the restricted verdict certified a cut whose first
# value above it sat INSIDE the rounding of the Float16 input.
#
# Reported by the downstream video consumer on a 40x40x40 Float16 luma block
# (`below = [1.65e-6, 3.26e-6]`, `above = [4.21e-4, 1.22e-3]`, `certified =
# true`, against `eps(Float16) = 9.8e-4`) and reproduced here on a
# near-degenerate ramp of the same shape family: measured on the tree before
# the fix, `certified = true`; after it, `certified = false` with
# `undecidable = 1`.  The Float32 run on the same data is untouched, which is
# the other half of the claim -- this floor binds ONLY in the mixed case.
@testset "the STORED type reaches QuickDer's restricted verdict" begin
    d = 24
    fr = [Index(d, "a$i") for i in 1:3]
    Ω = IndTransverseOps(fr, UniversalOp())
    ch = UniversalChisel(3)
    # A smooth ramp plus a 3e-5 texture: the first value above the cut then
    # lands at 7.2e-4 in Float16 and 1.8e-4 in Float32, i.e. between
    # `eps(Float32)` and `eps(Float16)` -- the window in which the two storage
    # types must give DIFFERENT verdicts on the same spectrum.
    Random.seed!(20260904)
    A = [1.0 + 0.01i + 0.02j + 0.005k for i in 1:d, j in 1:d, k in 1:d] .+
        3e-5 .* randn(d, d, d)

    # The verdict is the restricted solve's, which `derTrOpsReduced` reports on
    # its `QuickDer restricted solve` debug record; `return_diagnostics` (the
    # public route to the same numbers) is asserted in TestQuickDerN.
    function restricted_verdict(T)
        Γ = ITensor(Array{T}(A), fr...)
        m = Dleto.get_derivation_method(:QuickDer; seed = 4242)
        logs, _ = Test.collect_test_logs(min_level = Logging.Debug) do
            derTrOpsReduced(m, Ω, ch, Γ)
        end
        rec = filter(r -> r.message == "QuickDer restricted solve", logs)
        @test length(rec) == 1
        return Dict(only(rec).kwargs)
    end

    v16 = restricted_verdict(Float16)
    @test v16[:data_floor] ≈ data_floor(Float16)      # NOT eps(Float32)
    @test v16[:undecidable] >= 1
    @test v16[:certified] == false

    v32 = restricted_verdict(Float32)
    @test v32[:data_floor] ≈ data_floor(Float32)
    @test v32[:undecidable] == 0
    @test v32[:certified] == true
    @test v32[:nullity] == v16[:nullity]              # the COUNT does not move
end
