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

            # :corner on a structured support.  The design note expected the
            # corner sketch to be degenerate here; in the cross-sketch
            # formulation each S_a keeps axis a full, so the corner still meets
            # the support and only the LIFT can degenerate (from d = 16 up),
            # where the min-norm fallback plus the consistency filter recover.
            # What must never happen is a silently WRONG answer: :corner either
            # errors out, or every vector it returns is a genuine derivation
            # and the count does not exceed the oracle.
            corner_ok = try
                cb = der(corner_method, Ω, P, Γ; tol=QD_TOL)
                length(cb) <= 13 && all(D -> der_residual(Γ, D, P) < RESID64, cb)
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

# =========================================================================
# 8. Whitened restriction (`whiten = true`, QuickDer-W)
# =========================================================================
#
# The whitening substitutes `Ỹ_a = R_a Y_a` for the thin QR `M_a = Q_a R_a` of
# the transposed mode-`a` unfolding of the cross sketch.  Because
# `A_a = I_{r_a} ⊗ M_a` up to a row permutation, that makes every diagonal
# block of the restricted Gram exactly `c_a·I` and cannot change the null
# space -- so every answer here must be the SAME SUBSPACE the unwhitened run
# returns, not merely the same dimension.  See `Dleto._qdn_whiten_axis` and
# docs/design/QuickDer-valence-n.md, "Whitened restriction".

# --- the algebraic identity the whitening rests on ---------------------
#
# `A_aᵗ A_a == c_a · (I_{r_a} ⊗ M_aᵗ M_a)`, `c_a = Σ_ρ P[ρ,a]²`, in this
# file's own row/column convention.  Measured to 5e-16 at these sizes; this
# guards the convention itself, which is what the whitening is built on.
@testset "whiten: the restricted Gram has Kronecker diagonal blocks" begin
    if !QUICKDER_AVAILABLE
        @test_skip false
    else
        cases = ((dims = [8, 9, 10],   P = UniversalChisel(3)),
                 (dims = [8, 9, 10],   P = CentroidChisel(3)),
                 (dims = [5, 6, 7, 8], P = UniversalChisel(4)),
                 (dims = [8, 9, 10],   P = AdjointChisel(3, 1, 2)))
        @testset "dims $(c.dims), $(size(c.P, 1))-row chisel" for c in cases
            Random.seed!(20260921)
            dims = c.dims
            N = length(dims)
            P = Matrix{Float64}(c.P)
            G = randn(dims...)
            eng = Dleto.engaged(P)
            r = Dleto._qdn_restriction_sizes(dims, eng, N)
            axs = [Dleto._qdn_axis(Float64, dims[a], r[a], :random, MersenneTwister(7))
                   for a in 1:N]
            S = Dleto._qdn_cross_sketches(G, axs, eng)
            eaxes = [a for a in 1:N if eng[a]]
            Uf = Dict{Int,Matrix{Float64}}(a => Dleto._qdn_unfold(S[a], a) for a in eaxes)
            coff = Dict{Int,Int}(); ncols = 0
            for a in eaxes
                coff[a] = ncols
                ncols += dims[a] * r[a]
            end
            Mres = Dleto._qdn_restricted_matrix(Uf, P, eaxes, r, dims, coff, ncols)
            for a in eaxes
                Bl = Mres[:, (coff[a] + 1):(coff[a] + dims[a] * r[a])]
                Ma = Matrix(transpose(Uf[a]))
                want = sum(P[:, a] .^ 2) .*
                       kron(Matrix{Float64}(I, r[a], r[a]), transpose(Ma) * Ma)
                @test norm(transpose(Bl) * Bl .- want) < 1e-12 * norm(want)
            end

            # ... and the whitened operator's spectrum then lies in [0, Σ c_a].
            (wh, Us, Ss, wdims) = Dleto._qdn_whiten(Uf, eaxes, dims, r, N)
            wcoff = Dict{Int,Int}(); wncols = 0
            for a in eaxes
                wcoff[a] = wncols
                wncols += wdims[a] * r[a]
            end
            Mw = Dleto._qdn_restricted_matrix(Us, P, eaxes, r, wdims, wcoff, wncols)
            @test opnorm(Mw) <= sqrt(sum(sum(P[:, a] .^ 2) for a in eaxes)) + 1e-10
            for a in eaxes
                # M_a * un == Q_a exactly: the substitution loses nothing.
                @test norm(Matrix(transpose(Uf[a])) * wh[a].un .- wh[a].Q) <
                      1e-10 * max(norm(wh[a].Q), 1.0)
                # and the folded whitened sketch unfolds back to Q_aᵗ
                @test Dleto._qdn_unfold(Ss[a], a) ≈ Us[a]
            end
        end
    end
end

# --- whitened and unwhitened return the same subspace ------------------
#
# Oracle in every case is the unwhitened run of the same method with the same
# seed; the Z-law is checked on both.  Scrambled spheres at valence 3 and 4,
# random dense tensors, and a video-shaped tensor -- the shapes the scaling
# sweeps use (bench/WhitenedRestriction.jl).
@testset "whiten: same nullity and same span as unwhitened" begin
    if !QUICKDER_AVAILABLE
        @test_skip false
    else
        cases = Any[]
        push!(cases, (label = "sphere valence 3, d = 12", kind = :sphere,
                      valence = 3, d = 12))
        SPHERE_VALENCE_KW && push!(cases, (label = "sphere valence 4, d = 8",
                                           kind = :sphere, valence = 4, d = 8))
        push!(cases, (label = "random dense (6,7,8)", kind = :random,
                      dims = (6, 7, 8), P = UniversalChisel(3)))
        push!(cases, (label = "random dense (5,6,7,4)", kind = :random,
                      dims = (5, 6, 7, 4), P = UniversalChisel(4)))
        push!(cases, (label = "random dense (6,7,8), centroid chisel", kind = :random,
                      dims = (6, 7, 8), P = CentroidChisel(3)))
        push!(cases, (label = "video-shaped 20x20x10x3", kind = :random,
                      dims = (20, 20, 10, 3), P = UniversalChisel(4)))

        @testset "$(c.label)" for c in cases
            Random.seed!(20260922)
            local Ω, P, Γ, frame
            if c.kind === :sphere
                inp = build_sphere(c.d; valence = c.valence)
                Ω, P, Γ = inp.Ω, Matrix{Float64}(inp.ch), inp.Γ
                frame = collect(Dleto.frames(Ω))
            else
                frame = [Index(c.dims[i], "w_$i") for i in eachindex(c.dims)]
                Γ = ITensor(randn(c.dims...), frame...)
                P = Matrix{Float64}(c.P)
                Ω = IndTransverseOps(frame, UniversalOp())
            end

            plain = der(get_derivation_method(:QuickDer; whiten = false, seed = 4242),
                        Ω, P, Γ; tol = QD_TOL)
            white = der(get_derivation_method(:QuickDer; whiten = true, seed = 4242),
                        Ω, P, Γ; tol = QD_TOL)

            @test length(white) == length(plain)
            for D in white
                @test der_residual(Γ, D, P) < RESID64
            end
            A = basis_matrix(plain, frame)
            B = basis_matrix(white, frame)
            @test subspace_residual(A, B) < 1e-8
            @test subspace_residual(B, A) < 1e-8
        end
    end
end

# --- a deliberately degenerate tensor ----------------------------------
#
# Mode 1 of `Γ` is projected to rank `d-2`, so `Γ ×_1 X = 0` has a
# `2·d`-dimensional space of solutions -- trivial derivations that any
# degenerate tensor has.  The whitening truncates them out of the restricted
# solve (they are a numerically-zero eigenvalue cluster there) and writes
# them down exactly instead, so it reports `2 + 2d`: the two scalar
# derivations plus the whole trivial space.
#
# The unwhitened branch reports only `2 + 2·r_1` of them, because its
# rank-deficient lift returns a minimum-norm `Z_1` with no kernel component
# -- measured 16 of 26 at d = 12.  So the assertion is CONTAINMENT, not
# equality: the whitened answer is a superset, every column of it is a
# genuine derivation by the Z-law, and its dimension is the exact one.
@testset "whiten: degenerate mode is counted exactly (rank d-2 on axis 1)" begin
    if !QUICKDER_AVAILABLE
        @test_skip false
    else
        @testset "d = $d" for d in (8, 12)
            Random.seed!(20260923)
            U = Matrix(qr(randn(d, d)).Q)
            G = Dleto._qdn_ttm(randn(d, d, d),
                               U[:, 1:(d - 2)] * transpose(U[:, 1:(d - 2)]), 1)
            frame = [Index(d, "g_$i") for i in 1:3]
            Γ = ITensor(G, frame...)
            P = Matrix{Float64}(UniversalChisel(3))
            Ω = IndTransverseOps(frame, UniversalOp())
            @test rank(Dleto._qdn_unfold(G, 1)) == d - 2

            plain = der(get_derivation_method(:QuickDer; whiten = false, seed = 4242),
                        Ω, P, Γ; tol = QD_TOL)
            white = der(get_derivation_method(:QuickDer; whiten = true, seed = 4242),
                        Ω, P, Γ; tol = QD_TOL)

            # 2 scalar derivations + the whole (d - rank)·d = 2d trivial space
            @test length(white) == 2 + 2 * d
            @test length(plain) < length(white)
            for D in white
                @test der_residual(Γ, D, P) < RESID64
            end
            A = basis_matrix(plain, frame)
            B = basis_matrix(white, frame)
            @test subspace_residual(B, A) < 1e-8      # whitened contains unwhitened
        end
    end
end

# --- the matrix-free branch, which is what the whitening is FOR ---------
#
# Forcing `QDN_DENSE_BUDGET_BYTES` to zero sends a small case down the
# matrix-free branch, where the whitening has to hold up against an iterative
# null solver rather than a dense SVD.  `QDN_APPLY_COUNT` is the counter the
# scaling sweeps read; check it actually counts.
#
# TWO THINGS ARE PINNED HERE, and neither used to be.
#
# THE SOLVER.  The matrix-free branch takes whatever matrix-free solver is
# REGISTERED, and that depends on the environment: ARPACK is a weakdep and is
# not in this project's manifest, so a plain `runtests.jl` process meets
# KrylovKit's block Lanczos while a `bench/jl` process that stacked an
# environment carrying Arpack meets ARPACK.  Two different code paths, one
# testset, and a green run that means different things in different places.
# `:KrylovSolver` is a hard dep, so pinning it makes this test say the same
# thing everywhere.
#
# THE SEED, which now reaches the null solver (`QuickDerMethod`'s `seed` is
# passed to `solve_nullspace`, which fixes the Krylov start block -- see
# `Dleto.wants_seed`) instead of being decided by wherever the global RNG
# happened to be.  That makes the testset reproducible, and it makes visible
# what was already true: at d = 12 block Lanczos loses null vectors when the
# block is a large fraction of the restricted dimension (see its own dense
# gate in ext/DletoKrylovKitExt.jl).  Sweeping the seed 1..24 on this tensor,
# unwhitened returns the full 3 on 20 of 24 and whitened on 22 of 24.  The
# seed must be one where BOTH branches converge, or this compares an answer
# against a breakdown instead of testing that the two branches agree.  4242 is
# one of the unwhitened failures; 4243 is not.  The failure itself is pinned
# separately, just below.
@testset "whiten: matrix-free branch agrees and counts its applies" begin
    if !QUICKDER_AVAILABLE
        @test_skip false
    else
        Random.seed!(20260924)
        inp = build_sphere(12; valence = 3)
        Ω, P, Γ = inp.Ω, Matrix{Float64}(inp.ch), inp.Γ
        saved = Dleto.QDN_DENSE_BUDGET_BYTES[]
        try
            Dleto.QDN_DENSE_BUDGET_BYTES[] = 0.0
            results = Dict{Bool,Any}()
            for w in (false, true)
                Dleto.QDN_APPLY_COUNT[] = 0
                basis = der(get_derivation_method(:QuickDer; whiten = w, seed = 4243,
                                                   solver = :KrylovSolver),
                            Ω, P, Γ; tol = QD_TOL)
                results[w] = (; basis, applies = Dleto.QDN_APPLY_COUNT[])
                Dleto.QDN_APPLY_COUNT[] = -1
            end
            @test results[false].applies > 0
            @test results[true].applies > 0
            @test length(results[true].basis) == 3
            @test length(results[false].basis) == 3
            for D in results[true].basis
                @test der_residual(Γ, D, P) < RESID64
            end
            A = basis_matrix(results[false].basis, collect(Dleto.frames(Ω)))
            B = basis_matrix(results[true].basis, collect(Dleto.frames(Ω)))
            @test subspace_residual(A, B) < 1e-6
            @test subspace_residual(B, A) < 1e-6
        finally
            Dleto.QDN_DENSE_BUDGET_BYTES[] = saved
            Dleto.QDN_APPLY_COUNT[] = -1
        end
    end
end

# --- an empty answer from a failed solve is a failure, not an answer -----
#
# The silent case, and it was live: at d = 12, forced matrix-free, UNWHITENED,
# seed 4242, KrylovKit's block Lanczos breaks down (`DomainError with
# -3.09e-19` out of its Cholesky-type block normalisation), the single-vector
# Arnoldi fallback returns restricted nullity 7 of 13 -- every value converged,
# so the old code called it `certified` -- those 7 lift to universal
# derivations that miss `Ω` entirely, and QuickDer reported ZERO derivations of
# a true three.  Not an error: a confident empty answer.
#
# Two things now stop it.  The Arnoldi fallback reports `converged = false`,
# because `info.converged` counts Ritz pairs that met a residual test and says
# nothing about how many COPIES of a repeated eigenvalue were found -- and a
# null space IS a repeated eigenvalue.  And `_qdn_empty_result` refuses to
# return an empty derivation space from a non-`:ok` solve, so `:Auto` falls
# back to SylverLining, which is exact.
@testset "QuickDer: an empty answer from a non-converged solve raises" begin
    if !QUICKDER_AVAILABLE
        @test_skip false
    else
        Random.seed!(20260924)
        inp = build_sphere(12; valence = 3)
        Ω, P, Γ = inp.Ω, Matrix{Float64}(inp.ch), inp.Γ
        saved = Dleto.QDN_DENSE_BUDGET_BYTES[]
        try
            Dleto.QDN_DENSE_BUDGET_BYTES[] = 0.0
            # :QuickDer alone must RAISE rather than report an empty space.
            @test_throws ErrorException der(
                get_derivation_method(:QuickDer; whiten = false, seed = 4242,
                                      solver = :KrylovSolver),
                Ω, P, Γ; tol = QD_TOL)
            @test Dleto.QDN_LAST_SOLVE_STATUS[] !== :ok
            # ... and :Auto turns that raise into the right answer.
            auto = der(get_derivation_method(:Auto; whiten = false, seed = 4242,
                                             solver = :KrylovSolver),
                       Ω, P, Γ; tol = QD_TOL)
            @test length(auto) == 3
            for D in auto
                @test der_residual(Γ, D, P) < RESID64
            end
        finally
            Dleto.QDN_DENSE_BUDGET_BYTES[] = saved
        end
    end
end

# --- the trivial space of a degenerate mode, factored -------------------
#
# `(d - rank)*d` operator tuples of `n` dense `d x d` matrices is 3 GB at
# d = 500 with a single degenerate mode, to describe a space whose complete
# statement is one `d x (d - rank)` matrix.  So the complete answer is
# published FACTORED (`QDN_TRIVIAL_FACTORED`) and only what fits in
# `QDN_TRIVIAL_MAX_BYTES` is written out.  What has to hold: the factored form
# is complete whether or not the write-out truncated, the truncation warns and
# names the field, and with the budget restored the full count comes back.
@testset "whiten: the trivial space is published factored, and capped" begin
    if !QUICKDER_AVAILABLE
        @test_skip false
    else
        d = 12
        Random.seed!(20260923)
        U = Matrix(qr(randn(d, d)).Q)
        G = Dleto._qdn_ttm(randn(d, d, d),
                           U[:, 1:(d - 2)] * transpose(U[:, 1:(d - 2)]), 1)
        frame = [Index(d, "t_$i") for i in 1:3]
        Γ = ITensor(G, frame...)
        P = Matrix{Float64}(UniversalChisel(3))
        Ω = IndTransverseOps(frame, UniversalOp())
        method = get_derivation_method(:QuickDer; whiten = true, seed = 4242)
        per = 3 * d^2 * sizeof(Float64)          # one tuple, all three axes
        saved = Dleto.QDN_TRIVIAL_MAX_BYTES[]
        try
            # (a) budget for exactly 5 tuples: the write-out truncates.
            Dleto.QDN_TRIVIAL_MAX_BYTES[] = 5.4 * per
            capped = der(method, Ω, P, Γ; tol = QD_TOL)
            @test length(capped) == 2 + 5           # 2 scalars + the budget

            fac = Dleto.QDN_TRIVIAL_FACTORED[]
            @test length(fac) == 1
            @test fac[1].axis == 1
            @test size(fac[1].K) == (d, 2)          # rank d-2, so 2 truncated
            @test norm(transpose(fac[1].K) * fac[1].K - I) < 1e-12
            # The COMPLETE space is (d - rank)*d = 2d, and every element of it
            # `K[:,p] * e_q'` really does annihilate Γ -- the whole content of
            # the factored statement, checked on the corners of the box.
            @test size(fac[1].K, 2) * d == 2 * d
            for p in 1:2, q in (1, d)
                X = zeros(Float64, d, d)
                X[:, q] = fac[1].K[:, p]
                @test norm(Dleto._qdn_ttm(G, X, 1)) < 1e-10 * norm(G)
            end

            # (b) the same call with the budget restored writes all of it out,
            # and reproduces the count the uncapped code has always returned.
            Dleto.QDN_TRIVIAL_MAX_BYTES[] = saved
            full = der(method, Ω, P, Γ; tol = QD_TOL)
            @test length(full) == 2 + 2 * d
            for D in full
                @test der_residual(Γ, D, P) < RESID64
            end
            @test length(Dleto.QDN_TRIVIAL_FACTORED[]) == 1

            # (c) a tensor with no degenerate mode leaves the field empty, so
            # a stale value can never be read as a trivial space.
            Random.seed!(20260925)
            Γg = ITensor(randn(d, d, d), frame...)
            der(method, Ω, P, Γg; tol = QD_TOL)
            @test isempty(Dleto.QDN_TRIVIAL_FACTORED[])
        finally
            Dleto.QDN_TRIVIAL_MAX_BYTES[] = saved
        end
    end
end

# --- the lift consistency filter cuts on a GAP, not at a cliff ------------
#
# The filter decides which combinations of restricted null directions really
# lift to derivations, by taking a null space of the lift-residual matrix.  It
# used to take it at the hard relative cutoff `atol = max(tol, sqrt(eps(T)))`,
# which is the level a lift residual BOTTOMS OUT at -- measured 1.3 to 2.9
# times `sqrt(eps(Float32))` for genuine directions -- so in Float32 the cutoff
# sits inside the population it is meant to keep.  The rule now floors the
# spectrum at `sqrt(eps(T))` and cuts at the largest consecutive ratio above
# it, `gap_verdict`'s rule from `NullSolvers.jl`.
#
# What this test pins is NEUTRALITY, because that is what can be measured
# reproducibly.  On every case where the whole pipeline is deterministic (the
# dense branch with `SVDSolver`: no randomised subspace, no ARPACK saved state)
# the two rules agree, at the oracle, in both element types -- including the
# nullity-13 raw sphere, which is the case a merely LOOSER constant breaks.
# The Float32 undercount the rule was written for is NOT reproducible on the
# matrix-free branch and is not in this filter at all: at d = 48 Float32 the
# filter passes all 13 of its 13 restricted directions under both rules and
# `_fastder_restrict_to_ops` then returns 2.  See the session-4 notes.
@testset "the lift filter's gap rule is neutral where the pipeline is exact" begin
    if !QUICKDER_AVAILABLE
        @test_skip false
    else
        old!() = (Dleto.QDN_LIFT_GAP_RATIO[] = Inf; Dleto.QDN_LIFT_CEILING[] = 0.0)
        new!() = (Dleto.QDN_LIFT_GAP_RATIO[] = 100.0; Dleto.QDN_LIFT_CEILING[] = 32.0)
        saved = (Dleto.QDN_LIFT_GAP_RATIO[], Dleto.QDN_LIFT_CEILING[],
                 Dleto.QDN_DENSE_BUDGET_BYTES[], Dleto.QDN_GRAM_MIN_COLS[])
        function count_ders(Ω, P, Γ)
            Random.seed!(4242)
            m = get_derivation_method(:QuickDer; verify = :random, seed = 20260904)
            try
                return length(der(m, Ω, P, Γ; tol = QD_TOL))
            catch
                return -1
            end
        end
        try
            # Dense branch, dense SVD: the one configuration in which this
            # pipeline gives the same answer twice.
            Dleto.QDN_DENSE_BUDGET_BYTES[] = 8.0 * 2^30
            Dleto.QDN_GRAM_MIN_COLS[] = 10^9

            @testset "scrambled sphere d = $d $T" for T in (Float32, Float64),
                                                      d in (24, 32)
                inp = build_sphere(d; valence = 3, T = T, lean = false)
                Ω, P, Γ = inp.Ω, Matrix{Float64}(inp.ch), inp.Γ
                new!(); @test count_ders(Ω, P, Γ) == 3
                old!(); @test count_ders(Ω, P, Γ) == 3
            end

            # The nullity-13 case, which is the reason the cut is a gap and not
            # a looser constant: a ceiling 32 times wider than the old cutoff
            # must not sweep a spurious direction in here.
            @testset "raw sphere keeps its 13 in $T" for T in (Float32, Float64)
                S = sphere_octant(24; valence = 3)
                Γ0 = nondeg(S).Δ
                fr = collect(inds(Γ0))
                Γ = T === Float64 ? Γ0 : ITensor(Array{T}(Array(Γ0, fr...)), fr...)
                Ω = IndTransverseOps(fr, UniversalOp())
                P = Matrix{Float64}(UniversalChisel(3))
                new!(); @test count_ders(Ω, P, Γ) == 13
                old!(); @test count_ders(Ω, P, Γ) == 13
            end
        finally
            (Dleto.QDN_LIFT_GAP_RATIO[], Dleto.QDN_LIFT_CEILING[],
             Dleto.QDN_DENSE_BUDGET_BYTES[], Dleto.QDN_GRAM_MIN_COLS[]) = saved
        end
    end
end

# =========================================================================
# 8. `return_diagnostics`: the verdict as a value, not as log text
# =========================================================================
#
# The downstream consumer calls `derTrOpsReduced` directly and needs the
# evidence behind the count -- was it certified, what are the values either
# side of the cut, how far from zero is each returned direction -- without
# parsing `@warn` text that `maxlog = 1` silences on the second block of a
# thousand.  `return_diagnostics = true` appends a `DerivationReport`; the
# default appends nothing, and THAT is the assertion that matters most here,
# because everything downstream destructures a three-tuple.
#
# What is checked: every summary field agrees with the `NullVerdict` it was
# copied from, the counts agree with the matrix actually returned, and the
# residuals are RECOMPUTED here from `ders` -- the report may not simply be
# reporting itself.

@testset "8. return_diagnostics" begin
    if !QUICKDER_AVAILABLE
        @test_skip false
    else
        # `(sphere, truth)`: the scrambled sphere (SymmetricOp, nullity 3) and a
        # random dense tensor (UniversalOp, nullity = valence - 1 = the scalars),
        # which are the two ends of the range -- one with structure to find, one
        # with none.
        function random_case(T)
            Random.seed!(20260904)
            fr = [Index(d, "r$i") for (i, d) in enumerate((9, 8, 7))]
            Γ = ITensor(Array{T}(randn(9, 8, 7)), fr...)
            return (; Ω = IndTransverseOps(fr, UniversalOp()),
                      ch = Matrix{Float64}(UniversalChisel(3)), Γ, truth = 2)
        end
        function sphere_case(T)
            inp = build_sphere(10; valence = 3, T = T)
            return (; inp.Ω, ch = Matrix{Float64}(inp.ch), inp.Γ, truth = 3)
        end

        @testset "$(cname) $T / $mname" for (cname, case) in (("sphere", sphere_case),
                                                              ("random", random_case)),
                                            T in (Float64, Float32),
                                            mname in (:QuickDer, :SylverLining)
            c = case(T)
            m = mname === :QuickDer ?
                get_derivation_method(:QuickDer; seed = 20260904) :
                get_derivation_method(:SylverLining)

            # 1. THE DEFAULT IS UNCHANGED.  Three elements, and the same
            #    coordinates as the diagnosed call.
            plain = derTrOpsReduced(m, c.Ω, c.ch, c.Γ; tol = 1e-8)
            @test length(plain) == 3

            out = derTrOpsReduced(m, c.Ω, c.ch, c.Γ; tol = 1e-8,
                                  return_diagnostics = true)
            @test length(out) == 4
            (rΩ, expand_map, ders, rep) = out
            @test rep isa DerivationReport
            @test size(ders) == size(plain[3])
            @test size(ders, 2) == c.truth

            # 2. THE REPORT DESCRIBES THIS ANSWER.
            @test rep.method === mname
            @test rep.policy === :auto
            @test rep.returned == size(ders, 2)
            @test rep.dims == collect(size(c.Γ))
            @test rep.store_eltype === T
            @test rep.compute_eltype === Dleto.compute_eltype(T)
            # The scalars are `dim ker P`, which is 2 for a 3-valent universal
            # chisel -- the count a tensor with no structure returns, and the
            # baseline the sphere's 3 has to be read against.
            @test rep.scalar_dim == 2

            # 3. THE SUMMARY AGREES WITH THE VERDICT IT WAS COPIED FROM.
            @test rep.verdict isa Dleto.NullVerdict
            v = rep.verdict
            @test rep.nullity == v.nullity
            @test rep.certified == v.certified
            @test rep.rule === v.rule
            @test rep.status === v.status
            @test rep.gap == v.gap || (isnan(rep.gap) && isnan(v.gap))
            @test rep.threshold == v.threshold
            @test rep.undecidable == v.undecidable
            @test rep.near_null == v.near_null
            @test rep.spectrum == v.spectrum
            @test rep.data_floor ≈ Dleto.data_floor(T)
            @test rep.precision_floor ≈ Dleto.precision_floor(T)
            # `selected` is the cut, so exactly `nullity` values, ascending,
            # and `next_value` is the first one above it.
            @test length(rep.selected) == rep.nullity
            @test issorted(rep.selected)
            @test rep.spectrum[1:rep.nullity] == rep.selected
            @test length(rep.spectrum) > rep.nullity ?
                  rep.next_value == rep.spectrum[rep.nullity + 1] :
                  isnan(rep.next_value)

            # 4. THE RESIDUALS, RECOMPUTED.  Not copied from the report: each
            #    returned column is embedded back into operators here and put
            #    through the test file's own `der_residual`, which goes by the
            #    ITensor route rather than `Dleto.der_residual`'s blocked one.
            @test rep.residuals !== nothing
            @test length(rep.residuals) == size(ders, 2)
            for j in 1:size(ders, 2)
                D = embedITensors(c.Ω, expand_map * ders[:, j])
                @test isapprox(rep.residuals[j], der_residual(c.Γ, D, c.ch);
                               rtol = 1e-3, atol = 10 * eps(T))
                # and they really are derivations, which is what the number is
                # for: `sqrt(eps)` is the accuracy a solve-and-lift can carry.
                @test rep.residuals[j] < 100 * sqrt(eps(T))
            end

            # 5. ROUTE-SPECIFIC FIELDS: filled by QuickDer, `nothing` from
            #    SylverLining, which has no sketch and no lift.  A `nothing`
            #    means "this route has no such number", never "zero".
            if mname === :QuickDer
                @test rep.whitened === true
                @test rep.seed == 20260904
                @test rep.device === :cpu
                @test rep.restriction isa Vector{Int}
                @test all(rep.restriction .<= rep.dims)
                @test rep.restricted_size isa Tuple{Int,Int}
                # The lift accepts the UNIVERSAL derivations; the intersection
                # with Ω shrinks the count afterwards, so this is an upper
                # bound on what came back and not an equality (13 against 3 on
                # the sphere).
                @test rep.lift_dim >= rep.returned
                @test length(rep.lift_residuals) == rep.lift_dim
                @test all(isfinite, rep.lift_residuals)
            else
                @test rep.whitened === nothing
                @test rep.restriction === nothing
                @test rep.restricted_size === nothing
                @test rep.lift_dim === nothing
                @test rep.lift_residuals === nothing
                # SylverLining decides on the derivation operator itself, so
                # its cut IS the answer -- no filter runs after it.
                @test rep.nullity == rep.returned
            end

            # 6. It prints without erroring, and says the two things a reader
            #    looks for first.
            txt = sprint(show, MIME"text/plain"(), rep)
            @test occursin(string(mname), txt)
            @test occursin(rep.certified ? "CERTIFIED" : "uncertified", txt)
        end

        # `:Auto` forwards the keyword and reports the route that ANSWERED,
        # which is the question a caller of `:Auto` is asking.
        @testset ":Auto names the route that answered" begin
            Random.seed!(20260904)
            fr = [Index(d, "u$i") for (i, d) in enumerate((10, 10, 10))]
            Γ = ITensor(randn(10, 10, 10), fr...)
            Ω = IndTransverseOps(fr, UniversalOp())
            ch = Matrix{Float64}(UniversalChisel(3))
            out = derTrOpsReduced(get_derivation_method(:Auto; seed = 20260904),
                                  Ω, ch, Γ; tol = 1e-8, return_diagnostics = true)
            @test length(out) == 4
            @test out[4].method in (:QuickDer, :SylverLining)
            @test out[4].method !== :Auto
            @test out[4].returned == size(out[3], 2) == 2
        end
    end
end
