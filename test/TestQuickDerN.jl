#
# TestQuickDerN.jl -- tests for :QuickDer, the valence-n derivation solver
# described in docs/design/QuickDer-valence-n.md.
#
# Every test verifies the Z-law (`der_residual`, the defining equation of a
# derivation -- see test/TestDerivationLaws.jl) and compares the nullity
# against `:SylverLining` (dense SVD nullspace) where feasible.  The
# `:SylverLining` half of every oracle below can be, and was, validated
# BEFORE `:QuickDer` existed: `Dleto.QuickDerMethod` is not defined yet at the
# time this file was written, so every `:QuickDer`-specific assertion is
# `@test_skip`ped and only the oracle is checked.  Once the valence-n kernel
# lands, `QUICKDER_AVAILABLE` flips true and those assertions start running
# with no further changes needed here.
#
using Test
using Dleto
using ITensors
using LinearAlgebra
using Random

# --- shared helpers ----------------------------------------------------
#
# Copied from test/TestDerivationLaws.jl rather than relied upon, so this
# file has no ordering dependency on where it is `include`d relative to that
# one (both are top-level `include`s from test/runtests.jl, in the same
# module, so redefining the same function twice is harmless -- guarded here
# so it doesn't even happen twice when both files are loaded).

if !isdefined(@__MODULE__, :der_residual)
    """
        der_residual(Γ, D, P) -> Real

    Relative size of the chisel-weighted sum  Σ_a P[:,a] ⊗ (Γ · D_a).  Zero
    exactly when D is a P-derivation of Γ.  See test/TestDerivationLaws.jl.
    """
    function der_residual(Γ::ITensor, D::Vector{ITensor}, P::AbstractMatrix)
        C = Chisel(P, collect(inds(Γ)))
        R = applyDerivation(Γ, D, C)
        scale = norm(Γ) * maximum(norm.(D))
        return norm(R) / max(scale, eps())
    end
end

"""
    flatten_derivation(D, frame) -> Vector{Float64}

Flatten one derivation (a `Vector{ITensor}`, one operator per axis) to a
plain vector in a fixed axis/index order, so two different solvers' bases
(different temporary indices, arbitrary order) can be compared as ordinary
matrices.
"""
function flatten_derivation(D::Vector{ITensor}, frame::Vector{<:Index})
    parts = Vector{Vector{Float64}}(undef, length(D))
    for a in eachindex(D)
        other = only(filter(j -> j != frame[a], collect(inds(D[a]))))
        parts[a] = vec(Array(D[a], frame[a], other))
    end
    return vcat(parts...)
end

"""A basis (`Vector{Vector{ITensor}}`) as a matrix of flattened columns."""
basis_matrix(basis::Vector{Vector{ITensor}}, frame::Vector{<:Index}) =
    isempty(basis) ? zeros(0, 0) : hcat([flatten_derivation(D, frame) for D in basis]...)

"""
    subspace_residual(A, B) -> Real

Relative residual of projecting every column of `B` onto the column space of
`A`; zero iff span(B) ⊆ span(A).  Used to check that two bases of a
derivation space span the SAME subspace without caring which particular
basis vectors either solver picked.
"""
function subspace_residual(A::AbstractMatrix, B::AbstractMatrix)
    size(A, 2) == 0 && return size(B, 2) == 0 ? 0.0 : Inf
    Q = Matrix(qr(A).Q)[:, 1:size(A, 2)]
    worst = 0.0
    for j in axes(B, 2)
        b = @view B[:, j]
        nb = norm(b)
        nb == 0 && continue
        p = Q * (Q' * b)
        worst = max(worst, norm(b - p) / nb)
    end
    return worst
end

const QD_TOL   = 1e-6   # solver tolerance passed to der(...)
const RESID64  = 1e-10  # Z-law residual bound, Float64 tensors
const RESID32  = 1e-3   # Z-law residual bound, Float32 tensors ("works", not "matches Float64")
const SPAN_TOL = 1e-6   # subspace-equality residual bound

# --- detect the new valence-n QuickDer kernel ---------------------------
#
# `:QuickDer` currently resolves to the OLD valence-3 kernel
# (`FastDer3ValentMethod`, docs/design/QuickDer-valence-n.md's transcription
# oracle, aliased `:FastDer3Valent`/eventually `:QuickDer3`).  Once the
# valence-n kernel lands it defines `Dleto.QuickDerMethod`, and
# `get_derivation_method(:QuickDer; ...)` starts returning one of those
# instead.  Detect the TYPE directly (as instructed) rather than calling into
# the solver, so this file loads safely whether or not that work has landed.
const QUICKDER_AVAILABLE = isdefined(Dleto, :QuickDerMethod)

# The old valence-3 kernel's name is in flux (`:FastDer3Valent` today,
# `:QuickDer3` once the rename in the design doc lands) -- resolve whichever
# one actually works, so the "vs QuickDer3" test below doesn't need to be
# rewritten when that happens.
const OLD_V3_SYMBOL = try
    get_derivation_method(:QuickDer3)
    :QuickDer3
catch
    :FastDer3Valent
end

# --- sphere harness: valence-n generalisation may or may not have landed ---
#
# bench/SphereHarness.jl is being generalised (another agent, in parallel) to
# take a `valence` keyword.  Detect it with the cheapest possible call
# (d = 4, valence = 3 -- behaviourally identical to the pre-generalisation
# default) instead of guessing from file contents, per the instruction to use
# an `applicable`/try-catch style probe.
include(joinpath(@__DIR__, "..", "bench", "SphereHarness.jl"))

const SPHERE_VALENCE_KW = try
    build_sphere(4; valence=3)
    true
catch
    false
end

# =========================================================================
# 1. Random dense tensors: nullity = n - 1 (the scalar derivations only)
# =========================================================================
#
# Oracle, measured today with :SylverLining (bench/jl probe run, seed
# 20260903 below reproduces the shapes but not bit-for-bit the seed used to
# first measure this -- the nullity and residual order of magnitude are what
# matter and are seed-independent for a generic tensor):
#   valence 3, dims (4,5,3)       -> nullity 2, max Z-law residual 6.0e-16
#   valence 4, dims (5,4,6,3)     -> nullity 3, max Z-law residual 7.1e-16
#   valence 5, dims (4,3,3,3,2)   -> nullity 4, max Z-law residual 9.1e-16
@testset "random dense tensor: nullity = n - 1 (scalar derivations only)" begin
    Random.seed!(20260903)

    cases = (
        (dims = (4, 5, 3),       nullity = 2),
        (dims = (5, 4, 6, 3),    nullity = 3),
        (dims = (4, 3, 3, 3, 2), nullity = 4),
    )

    @testset "valence $(length(c.dims)), dims $(c.dims)" for c in cases
        n = length(c.dims)
        frame = [Index(c.dims[i], "a_$i") for i in 1:n]
        Γ = ITensor(randn(c.dims...), frame...)
        P = UniversalChisel(n)
        Ω = IndTransverseOps(frame, UniversalOp())

        basis = der(:SylverLining, Ω, P, Γ; tol=QD_TOL)
        @test length(basis) == c.nullity
        for D in basis
            @test der_residual(Γ, D, P) < RESID64
        end

        if QUICKDER_AVAILABLE
            method = get_derivation_method(:QuickDer; restriction=:random, verify=:full)
            qbasis = der(method, Ω, P, Γ; tol=QD_TOL)
            @test length(qbasis) == c.nullity
            for D in qbasis
                @test der_residual(Γ, D, P) < RESID64
            end
        else
            @test_skip length(der(:QuickDer, Ω, P, Γ; tol=QD_TOL)) == c.nullity
        end
    end

    # Float32: the nullity must still be exact; the residual only needs to
    # "work" at Float32 precision, not match Float64.  Measured today:
    # nullity 3, residuals ~3e-7 (well inside sqrt(eps(Float32)) ~ 3.5e-4).
    @testset "Float32 works" begin
        dims = (5, 4, 6, 3)
        frame = [Index(dims[i], "a_$i") for i in 1:4]
        Γ32 = ITensor(randn(Float32, dims...), frame...)
        P = UniversalChisel(4)
        Ω = IndTransverseOps(frame, UniversalOp())

        basis32 = der(:SylverLining, Ω, P, Γ32; tol=1e-3)
        @test length(basis32) == 3
        for D in basis32
            @test der_residual(Γ32, D, P) < RESID32
        end

        if QUICKDER_AVAILABLE
            method = get_derivation_method(:QuickDer; restriction=:random, verify=:full)
            qbasis32 = der(method, Ω, P, Γ32; tol=1e-3)
            @test length(qbasis32) == 3
            for D in qbasis32
                @test der_residual(Γ32, D, P) < RESID32
            end
        else
            @test_skip length(der(:QuickDer, Ω, P, Γ32; tol=1e-3)) == 3
        end
    end
end

# =========================================================================
# 2. Diagonal valence-4 tensor: nullity = (n - 1) * d
# =========================================================================
#
# Oracle, measured today with :SylverLining: d=4 -> 12, d=5 -> 15, d=6 -> 18
# (exactly 3d = (n-1)d for n = 4 in every case).
@testset "diagonal valence-4 tensor: nullity = (n - 1) * d" begin
    function diagonal_tensor4(d::Int)
        frame = [Index(d, "a_$i") for i in 1:4]
        A = zeros(Float64, d, d, d, d)
        for i in 1:d
            A[i, i, i, i] = 1.0
        end
        return ITensor(A, frame...), frame
    end

    @testset "d = $d" for d in (4, 5, 6)
        Γ, frame = diagonal_tensor4(d)
        P = UniversalChisel(4)
        Ω = IndTransverseOps(frame, UniversalOp())

        basis = der(:SylverLining, Ω, P, Γ; tol=QD_TOL)
        @test length(basis) == 3d
        for D in basis
            @test der_residual(Γ, D, P) < RESID64
        end

        if QUICKDER_AVAILABLE
            method = get_derivation_method(:QuickDer; restriction=:random, verify=:full)
            qbasis = der(method, Ω, P, Γ; tol=QD_TOL)
            @test length(qbasis) == 3d
            for D in qbasis
                @test der_residual(Γ, D, P) < RESID64
            end
        else
            @test_skip length(der(:QuickDer, Ω, P, Γ; tol=QD_TOL)) == 3d
        end
    end
end

# =========================================================================
# 3. Valence-3 random tensor: :QuickDer3 (the old kernel) spans the same
#    space as :SylverLining -- this runs today, no QUICKDER_AVAILABLE guard,
#    since the valence-3 transcription already exists.
# =========================================================================
@testset "valence-3 random tensor: old kernel ($OLD_V3_SYMBOL) spans same space as SylverLining" begin
    Random.seed!(20260910)
    dims = (5, 4, 6)
    frame = [Index(dims[i], "a_$i") for i in 1:3]
    Γ = ITensor(randn(dims...), frame...)
    P = UniversalChisel(3)
    Ω = IndTransverseOps(frame, UniversalOp())

    syl = der(:SylverLining, Ω, P, Γ; tol=QD_TOL)
    qd3 = der(OLD_V3_SYMBOL, Ω, P, Γ; tol=QD_TOL)

    @test length(syl) == length(qd3)
    for D in syl
        @test der_residual(Γ, D, P) < RESID64
    end
    for D in qd3
        @test der_residual(Γ, D, P) < RESID64
    end

    A = basis_matrix(syl, frame)
    B = basis_matrix(qd3, frame)
    @test subspace_residual(A, B) < SPAN_TOL   # span(qd3) ⊆ span(syl)
    @test subspace_residual(B, A) < SPAN_TOL   # and conversely -- same span
end

# =========================================================================
# 4. Unscrambled sparse sphere (the corner is all-zero): :random restriction
#    is expected to succeed, :corner is expected to fail or be filtered.
# =========================================================================
#
# Oracle, measured today with :SylverLining and matching
# bench/reports/quickder-sphere.md section 1: the UNIVERSAL derivation
# algebra of the unscrambled sphere (support i_1+...+i_n = d-1) is
# 13-dimensional at d=10 -- and, measured here for the first time, this is
# true at BOTH valence 3 and valence 4 (nnz = 220 / 10^4 at valence 4), not
# just valence 3.  See the "surprise" note in BOARD.md.
@testset "unscrambled sparse sphere: :random restriction sees it, :corner does not" begin
    @testset "valence $valence" for valence in (3, 4)
        S = sphere_octant(10; valence)
        fr_S = collect(inds(S))
        nnz = count(!=(0), Array(S, fr_S...))
        @test nnz < prod(size(S))   # genuinely sparse -- the point of this test

        ndS = nondeg(S)
        Γ = ndS.Δ
        frame = collect(inds(Γ))
        P = UniversalChisel(valence)
        Ω = IndTransverseOps(frame, UniversalOp())

        basis = der(:SylverLining, Ω, P, Γ; tol=QD_TOL)
        @test length(basis) == 13
        for D in basis
            @test der_residual(Γ, D, P) < RESID64
        end

        if QUICKDER_AVAILABLE
            rand_method   = get_derivation_method(:QuickDer; restriction=:random, verify=:full)
            corner_method = get_derivation_method(:QuickDer; restriction=:corner, verify=:full)

            qbasis = der(rand_method, Ω, P, Γ; tol=QD_TOL)
            @test length(qbasis) == 13
            for D in qbasis
                @test der_residual(Γ, D, P) < RESID64
            end

            # :corner sees an all-zero corner sketch on this support (spec
            # section 1); it must either error out, or be filtered down to
            # strictly fewer than the true nullity -- never silently report a
            # wrong 13-dimensional basis as if it were right.
            corner_ok = try
                length(der(corner_method, Ω, P, Γ; tol=QD_TOL)) < 13
            catch
                true
            end
            @test corner_ok
        else
            @test_skip length(der(:QuickDer, Ω, P, Γ; tol=QD_TOL, restriction=:random)) == 13
            @test_skip der(:QuickDer, Ω, P, Γ; tol=QD_TOL, restriction=:corner) isa Vector
        end
    end
end

# =========================================================================
# 5. Scrambled sphere (bench/SphereHarness.jl): with SymmetricOp(), nullity
#    = valence (n-1 scalars + the Euler/sphere derivation), stratification
#    lsq_err < 1e-8.
# =========================================================================
#
# Oracle, measured today with :SylverLining (SymmetricOp is build_sphere's
# default operator space):
#   valence 3, d=12 -> nullity 3, lsq_err ~1.0e-14
#   valence 4, d=10 -> nullity 4, lsq_err ~1.0e-14
# `bench/SphereHarness.jl` already has the `valence` keyword as of this
# writing (SPHERE_VALENCE_KW = true); the valence-4 case is still guarded so
# this file keeps working if that regresses or is mid-edit.
@testset "scrambled sphere: nullity = valence, lsq_err < 1e-8" begin
    cases = (
        (valence = 3, d = 12),
        (valence = 4, d = 10),
    )
    @testset "valence $(c.valence), d = $(c.d)" for c in cases
        if c.valence != 3 && !SPHERE_VALENCE_KW
            @test_skip false   # bench/SphereHarness.jl does not (yet) take `valence`
            continue
        end

        inp = build_sphere(c.d; valence=c.valence)
        res = run_stratify(inp; method=:SylverLining, tol=QD_TOL)
        @test res.status == "ok"
        @test res.nullity == c.valence
        @test res.lsq_err < 1e-8

        if QUICKDER_AVAILABLE
            qres = run_stratify(inp; method=:QuickDer, tol=QD_TOL,
                                 restriction=:random, verify=:full)
            @test qres.status == "ok"
            @test qres.nullity == c.valence
            @test qres.lsq_err < 1e-8
        else
            @test_skip run_stratify(inp; method=:QuickDer, tol=QD_TOL).nullity == c.valence
        end
    end
end

# =========================================================================
# 6. Disengaged axis: P = [1 1 0] on a random valence-3 tensor
# =========================================================================
#
# Oracle, measured today with :SylverLining: nullity 1.
@testset "disengaged axis: P = [1 1 0] on random valence-3 tensor" begin
    Random.seed!(20260911)
    dims = (4, 5, 3)
    frame = [Index(dims[i], "a_$i") for i in 1:3]
    Γ = ITensor(randn(dims...), frame...)
    P = [1.0 1.0 0.0]
    Ω = IndTransverseOps(frame, UniversalOp())

    basis = der(:SylverLining, Ω, P, Γ; tol=QD_TOL)
    @test length(basis) == 1
    for D in basis
        @test der_residual(Γ, D, P) < RESID64
    end

    if QUICKDER_AVAILABLE
        method = get_derivation_method(:QuickDer; restriction=:random, verify=:full)
        qbasis = der(method, Ω, P, Γ; tol=QD_TOL)
        @test length(qbasis) == length(basis)
        for D in qbasis
            @test der_residual(Γ, D, P) < RESID64
        end
    else
        @test_skip length(der(:QuickDer, Ω, P, Γ; tol=QD_TOL)) == length(basis)
    end
end

# =========================================================================
# 7. Multi-row chisel: CentroidChisel(3) on a random (4,4,4) tensor
# =========================================================================
#
# CentroidChisel(3) is a 3-row chisel (one row per axis pair (1,2),(1,3),
# (2,3)), exercising the multi-row case QuickDer's spec (section 1, the
# "for all ρ" in equation (R)) has to handle.  Oracle, measured today with
# :SylverLining: nullity 1.
@testset "multi-row chisel: CentroidChisel(3) on random (4,4,4) tensor" begin
    Random.seed!(20260912)
    dims = (4, 4, 4)
    frame = [Index(dims[i], "a_$i") for i in 1:3]
    Γ = ITensor(randn(dims...), frame...)
    P = CentroidChisel(3)
    Ω = IndTransverseOps(frame, UniversalOp())

    basis = der(:SylverLining, Ω, P, Γ; tol=QD_TOL)
    @test length(basis) == 1
    for D in basis
        @test der_residual(Γ, D, P) < RESID64
    end

    if QUICKDER_AVAILABLE
        method = get_derivation_method(:QuickDer; restriction=:random, verify=:full)
        qbasis = der(method, Ω, P, Γ; tol=QD_TOL)
        @test length(qbasis) == length(basis)
        for D in qbasis
            @test der_residual(Γ, D, P) < RESID64
        end
    else
        @test_skip length(der(:QuickDer, Ω, P, Γ; tol=QD_TOL)) == length(basis)
    end
end
