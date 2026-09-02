#
# Equational laws for derivations and densors.
#
# The definitions of derivation and densor ARE equations, so they are the
# tests.  Everything is approximate: these are floating-point solves, so the
# residuals are compared relative to the scale of the inputs.
#
# The same residual serves both sides, because the derivation equation and the
# densor equation are one equation with a different unknown:
#
#   Z-law   D in der(P, Γ)          =>  residual(Γ, D, P) ≈ 0
#   T-law   s in den(P, Δ)          =>  residual(s, ω, P) ≈ 0  for all ω in Δ
#   Galois  S ⊆ T(P,Ω) <=> Ω ⊆ Z(S,P), both directions being that residual
#
# `applyDerivation` already computes that residual -- it weights Γ·X_a by
# column a of the chisel and sums over the axes, carrying the chisel axis so
# every row of a multi-row chisel is checked at once.  It had no caller in the
# package; it is the verifier that was written and never wired up.
#
using Test
using Dleto
using ITensors
using LinearAlgebra
using Random

# --- the residual, relative to the scale of its inputs ----------------------

"""
    der_residual(Γ, D, P) -> Real

Relative size of the chisel-weighted sum  Σ_a P[:,a] ⊗ (Γ · D_a).  Zero
exactly when D is a P-derivation of Γ.
"""
function der_residual(Γ::ITensor, D::Vector{ITensor}, P::AbstractMatrix)
    C = Chisel(P, collect(inds(Γ)))
    R = applyDerivation(Γ, D, C)
    scale = norm(Γ) * maximum(norm.(D))
    return norm(R) / max(scale, eps())
end

"""Random linear combination of a basis of derivations, axis by axis."""
function combine(basis::Vector{Vector{ITensor}}, coefs::Vector{<:Number})
    val = length(first(basis))
    return [ sum(coefs[i] * basis[i][a] for i in eachindex(basis)) for a in 1:val ]
end

const LAW_TOL = 1e-6

# --- Z-law: every solver must return actual derivations ---------------------

# Solvers and the settings they support.  FastDer3Valent is restricted to
# valence 3 with a one-row, fully engaged chisel and universal operators.
const SOLVERS = [:SylverLining, :FastDer3Valent]
const BROKEN_SOLVERS = Symbol[]

@testset "Z-law: applyDerivation of any der result is approximately 0" begin
    Random.seed!(20260902)

    @testset "solver $solver" for solver in SOLVERS
        dims = (4, 5, 3)
        frame = [Index(dims[i], "a_$i") for i in 1:3]
        Γ = ITensor(randn(dims...), frame...)
        P = UniversalChisel(3)

        basis = der(solver, Γ; tol=1e-6)
        broken = solver in BROKEN_SOLVERS

        # dim >= dim null(P) always, by the scalar derivations.
        if broken
            @test_broken length(basis) >= size(nullspace(P), 2)
        else
            @test length(basis) >= size(nullspace(P), 2)
        end

        @testset "each generator" begin
            for D in basis
                if broken
                    @test_broken der_residual(Γ, D, P) < LAW_TOL
                else
                    @test der_residual(Γ, D, P) < LAW_TOL
                end
            end
        end

        # The Z-set is a subspace, so combinations must satisfy it too -- and a
        # random combination is what `stratify` actually consumes.
        @testset "random combinations" begin
            for _ in 1:5
                D = combine(basis, randn(length(basis)))
                if broken
                    @test_broken der_residual(Γ, D, P) < LAW_TOL
                else
                    @test der_residual(Γ, D, P) < LAW_TOL
                end
            end
        end
    end

    # QuickSylver handles adjoint-type chisels -- one row, exactly two engaged
    # axes -- so it is compared against SylverLining on those, for both axis
    # pairs, since the kernel fixes X on axis 1 and the tensor is permuted.
    @testset "QuickSylver on adjoint chisels" begin
        dims = (6, 6, 6)
        frame = [Index(dims[i], "a_$i") for i in 1:3]
        Γ = ITensor(randn(dims...), frame...)

        for (a, b) in ((1, 2), (2, 3), (1, 3))
            @testset "adjoint($a,$b)" begin
                P = AdjointChisel(3, a, b)
                base = der(:SylverLining, P, Γ; tol=1e-6)
                quick = der(:QuickSylver, P, Γ; tol=1e-6)

                # Oracle: scl(C_adj) = {(λI, λI, 0)} is 1-dimensional
                # (null_patterns section 7.2, eq. 18), so a generic tensor has
                # exactly one adjoint derivation.
                @test length(base) == 1
                @test length(quick) == 1

                for D in quick
                    @test der_residual(Γ, D, P) < LAW_TOL
                end
            end
        end

        # Three engaged axes must be refused, not silently mishandled.
        @test_throws ErrorException der(:QuickSylver, UniversalChisel(3), Γ; tol=1e-6)
    end

    # The general solver must satisfy the law for the other chisels too.
    @testset "SylverLining across chisels" begin
        dims = (4, 3, 5)
        frame = [Index(dims[i], "a_$i") for i in 1:3]
        Γ = ITensor(randn(dims...), frame...)

        for (name, P) in (
            ("universal", UniversalChisel(3)),
            ("adjoint",   AdjointChisel(3, 1, 2)),
            ("centroid",  CentroidChisel(3)),
        )
            @testset "$name chisel" begin
                basis = der(P, Γ; tol=1e-6)
                for D in basis
                    @test der_residual(Γ, D, P) < LAW_TOL
                end
            end
        end

        # A Tucker chisel forces D_a into the a-th radical, which is zero for a
        # generic tensor; and the only scalar derivation it admits is supported
        # on its disengaged axis, which the engagement reduction drops.  So the
        # reduced Z-set is genuinely {0} here -- a legitimate answer meaning
        # "Γ conforms to no pattern for this chisel".
        #
        # It used to assert "Not enough eigenvalues computed; increase `tol`
        # parameter", misreporting a mathematical fact as a solver failure.
        # Now it returns an empty basis and the caller decides.
        @testset "tucker chisel: trivial Z-set is not an error" begin
            P = TuckerChisel([true, false, true])
            basis = der(P, Γ; tol=1e-6)
            @test basis isa Vector
            @test isempty(basis)
        end
    end
end

# --- T-law and Galois adjunction --------------------------------------------
#
# These need `den` (the densor / T-set), which is still an abstract
# placeholder that asserts false for every method.  They are written out
# rather than omitted, and marked broken, so that implementing `den` turns
# them green -- and so that an accidental pass is reported loudly as
# "Unexpectedly Pass" rather than going unnoticed.

# A generic tensor has only the scalar derivations, and those satisfy the
# equation for *every* tensor -- for C = [1,1,1] and D_a = δ_a I with Σδ_a = 0
# the residual is (Σδ_a)·s = 0 identically.  So its densor is the whole space
# and both laws below would hold trivially (§5.4: "scalar derivations reveal
# nothing").  A structured tensor is needed to make the T-set a proper
# subspace.  The diagonal tensor Γ_ijk = [i=j=k] admits all diagonal triples
# with a_i + b_i + c_i = 0, so its derivation space is genuinely larger.
function diagonal_tensor(n::Int)
    frame = [Index(n, "a_$i") for i in 1:3]
    A = zeros(Float64, n, n, n)
    for i in 1:n
        A[i, i, i] = 1.0
    end
    return ITensor(A, frame...), frame
end

"""Flatten an ITensor to a vector in a fixed frame order."""
flat(s::ITensor, fr) = vec(Array(s, fr...))

@testset "denLM is an abstract map with a genuine adjoint" begin
    Random.seed!(20260905)
    Γ, frame = diagonal_tensor(3)
    P = UniversalChisel(3)
    Ω = IndTransverseOps(frame, UniversalOp())
    Δ = der(SylverLiningMethod(), Ω, P, Γ; tol=1e-6)

    (A, AtA, fr, dims) = denLM(Ω, P, Δ)

    # Shape: one residual block per element of Δ, each of size rows(P) * prod(dims).
    @test size(A) == (length(Δ) * size(P, 1) * prod(dims), prod(dims))
    @test size(AtA) == (prod(dims), prod(dims))

    # The adjoint must be the real adjoint, not a stand-in: <A s, r> == <s, Aᵗ r>.
    # This is the same law TestSylverLining checks for the sylve/ester pair.
    @testset "adjoint law" begin
        for _ in 1:25
            s = randn(size(A, 2))
            r = randn(size(A, 1))
            @test isapprox(dot(A * s, r), dot(s, A' * r); rtol=1e-9, atol=1e-12)
        end
    end

    # The batch parameter: nd < 0 is a basis, nd > 0 is a cap.
    @testset "batch size" begin
        basis = den(Ω, P, Δ; tol=1e-6, nd=-1)
        @test length(den(Ω, P, Δ; tol=1e-6, nd=2)) == min(2, length(basis))
        @test length(den(Ω, P, Δ; tol=1e-6, nd=10^6)) == length(basis)
    end
end

@testset "T-law: every tensor in the densor satisfies the same equation" begin
    Random.seed!(20260903)
    Γ, frame = diagonal_tensor(3)
    P = UniversalChisel(3)
    Ω = IndTransverseOps(frame, UniversalOp())

    Δ = der(SylverLiningMethod(), Ω, P, Γ; tol=1e-6)
    @test length(Δ) > size(nullspace(P), 2)   # more than just the scalars

    # The default (nd <= 0) must not cap the basis.  It used to be rewritten to
    # the valency, so this tensor's 6-dimensional derivation space came back as
    # 3 vectors.  Oracle: the diagonal tensor admits exactly the diagonal
    # triples with a_i + b_i + c_i = 0, i.e. 2 per index, so dim = 2n.
    @test length(Δ) == length(der(SylverLiningMethod(), Ω, P, Γ; tol=1e-6, nd=10^6))
    @test length(Δ) == 2 * 3

    tset = den(Ω, P, Δ; tol=1e-6, nd=-1)   # nd < 0 asks for a basis
    @test length(tset) >= 1
    # The densor must be a proper subspace here, not the whole tensor space.
    @test length(tset) < prod(size(Array(Γ, frame...)))

    @testset "each densor element admits each derivation" begin
        for s in tset, ω in Δ
            @test der_residual(s, ω, P) < LAW_TOL
        end
    end
end

# --- stratify: a change of frame, so it must behave like one ----------------
#
# There is no residual oracle for stratification yet -- the σ_{e+1} verdict of
# null_patterns.pdf Algorithm 2, which decides whether a pattern is genuinely
# present, is not implemented.  But two laws hold for any change of frame and
# are what the operation is for:
#
#   1. Σ lives in the SAME frame as Γ.  Each X_a carries the pair (a-th index,
#      temporary partner), so contracting consumed Γ's index and left the
#      temporary one: Σ came back on the temporary frame and could not be fed
#      to `der`, `den` or `stratify` again -- the `frames(Ω)` check rejected it
#      with "Incompatable Indexes".  The source carried `# retag the indexes`
#      as an undone TODO.
#   2. A change of frame CONJUGATES the derivation algebra, so it cannot change
#      the dimension of the Z-set.
@testset "stratify returns a usable change of frame" begin
    Random.seed!(20260906)
    n = 5
    generic_frame = [Index(n, "g_$i") for i in 1:3]
    generic = ITensor(randn(n, n, n), generic_frame...)
    structured, structured_frame = diagonal_tensor(n)
    P = UniversalChisel(3)

    # Law 2 below needs a tensor whose derivations are NOT just the scalars.
    #
    # `stratify` picks a random derivation and puts it into real canonical
    # form.  On a generic tensor every derivation is a scalar triple
    # (δ_a·I with Σδ_a = 0), and *every* basis is an eigenbasis of a scalar
    # matrix -- so `realCanonicalForm` returns an arbitrary frame, which can be
    # singular.  Conjugating by a singular frame drops rank and *gains*
    # derivations, so "conjugation preserves dim der" is not a law there: it
    # presupposes an invertible frame.  Measured: dim der went 2 -> 7, and all
    # seven satisfied the defining equation, so the 7 was right and the
    # assertion was wrong.  It only passed before because an absolute
    # tolerance happened to reject the extra directions.
    @testset "$name / $m" for (name, Γ, frame, dim_is_law) in (
                ("structured", structured, structured_frame, true),
                ("generic",    generic,    generic_frame,    false)),
            m in (:SylverLining, :QuickDer)
        Ω = IndTransverseOps(frame, UniversalOp())
        res = stratify(Ω, P, Γ; tol=1e-6, method=m)

        # Law 1: same frame in, same frame out -- the whole point of retagging.
        @test Set(inds(res.Σ)) == Set(frame)

        # Which is exactly what makes this call possible at all:
        Δ_Σ = der(:SylverLining, Ω, P, res.Σ; tol=1e-6)

        # Whatever comes back must satisfy the defining equation.
        for D in Δ_Σ
            @test der_residual(res.Σ, D, P) < LAW_TOL
        end

        # Law 2, where the frame is genuinely a change of basis.
        if dim_is_law
            Δ_Γ = der(:SylverLining, Ω, P, Γ; tol=1e-6)
            @test length(Δ_Σ) == length(Δ_Γ)
        end

        # One frame matrix per axis.
        @test length(res.Xs) == 3
        for X in res.Xs
            @test isfinite(cond(Array(X, inds(X)...)))
        end
    end

    # `:QuickDer` is the name for Liu's derivation solve-and-lift;
    # `:FastDer3Valent` is kept as an alias and must resolve to the same thing.
    @test get_derivation_method(:QuickDer) == get_derivation_method(:FastDer3Valent)
end

# --- progress reporting is optional, tagged, and inert -----------------------
#
# Progress must never change an answer, and must be off unless asked for.  The
# tags are checked eagerly so a typo fails loudly instead of silently
# reporting nothing.
@testset "progress reporting is opt-in, tagged, and changes nothing" begin
    Random.seed!(20260907)
    Γ, frame = diagonal_tensor(4)
    P = UniversalChisel(3)
    Ω = IndTransverseOps(frame, UniversalOp())

    base_der = der(:SylverLining, Ω, P, Γ; tol=1e-6)
    base_den = den(Ω, P, base_der; tol=1e-6, nd=-1)

    @testset "option $(repr(p))" for p in (false, true, :all, :densify, :solve,
                                           [:densify, :solve])
        @test length(der(:SylverLining, Ω, P, Γ; tol=1e-6, progress=p)) == length(base_der)
        @test length(den(Ω, P, base_der; tol=1e-6, nd=-1, progress=p)) == length(base_den)
    end

    # A misspelled tag must be an error, not silence.
    @test_throws ErrorException den(Ω, P, base_der; tol=1e-6, progress=:densify_typo)
    @test_throws ErrorException der(:SylverLining, Ω, P, Γ; tol=1e-6, progress=:nonsense)

    # `progress` is a per-call option and must not be mistaken for a method
    # constructor keyword -- the two were conflated, so every symbol overload
    # forwarded it to `SylverLiningMethod(; progress=...)` and raised
    # "does not support keyword progress".
    @test length(der(:SylverLining, P, Γ; tol=1e-6, progress=:solve)) == length(base_der)
    @test length(der(:SylverLining, Γ; tol=1e-6, progress=:solve)) == length(base_der)
    @test stratify(Ω, P, Γ; tol=1e-6, progress=:densify).Xs isa Vector

    # And it still composes with a genuine constructor keyword.
    @test length(der(:SylverLining, Ω, P, Γ; tol=1e-6, progress=:solve,
                     solver=:SVDSolver)) == length(base_der)
end

@testset "Galois adjunction: S ⊆ T(P,Ω) iff Ω ⊆ Z(S,P)" begin
    Random.seed!(20260904)
    Γ, frame = diagonal_tensor(3)
    P = UniversalChisel(3)
    Ω = IndTransverseOps(frame, UniversalOp())

    Δ = der(SylverLiningMethod(), Ω, P, Γ; tol=1e-6)
    tset = den(Ω, P, Δ; tol=1e-6, nd=-1)   # nd < 0 asks for a basis

    # Forward direction: Γ lies in the densor of its own derivations.  It is in
    # the *span*, not equal to a basis element, so project onto the span.  The
    # nullspace basis is orthonormal, so the projector is Σ_j s_j s_jᵗ.
    @testset "Γ lies in the densor of its own derivations" begin
        B = hcat([flat(s, frame) for s in tset]...)
        g = flat(Γ, frame)
        residual = norm(g - B * (B' * g)) / norm(g)
        @test residual < LAW_TOL
    end

    # Reverse direction: everything in Δ is a derivation of everything in the
    # T-set -- which together with the forward direction is the adjunction
    # S ⊆ T(P,Ω) <=> Ω ⊆ Z(S,P) instantiated at S = {Γ}, Ω = Δ.
    @testset "Δ ⊆ Z(T-set, P)" begin
        for s in tset, ω in Δ
            @test der_residual(s, ω, P) < LAW_TOL
        end
    end
end
