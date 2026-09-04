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

# --- Z-law on NEAR-DEGENERATE tensors ------------------------------------
#
# The Z-law above uses generic and diagonal tensors, both well conditioned,
# and that is why it missed a real bug for a whole session.  What exposes a
# tolerance error is a tensor with many *near*-null directions -- smooth,
# low-rank, quantized data, i.e. exactly what real measurements look like.
#
# The bug: `SylverLining` hands the solver `sylvester = ester∘sylve`, which is
# the Gram operator `AᵗA`, so the values being filtered are `λ = σ²`.  A
# threshold relative to `‖AᵗA‖` therefore admitted every direction with
# `σ/σ_max ≤ √tol` -- ask for 1e-6, get 1e-3.  On real video boxes from
# Boards.mp4 that made `der` report 113--156 derivations per 8x8x8 box at a
# residual of 2e-3: about 130 vectors per box that are not derivations,
# silently, with no error and no warning.
#
# The invariant this pins down is the one a caller actually asks for: whatever
# `der` returns satisfies the defining equation to `tol`.  It is asserted here
# per tensor rather than per dimension, because on degenerate tensors the
# derivation space is legitimately large and its exact dimension is not the
# thing under test.
@testset "Z-law holds on near-degenerate tensors" begin
    Random.seed!(20260908)
    n = 6
    frame = [Index(n, "a_$i") for i in 1:3]
    P = UniversalChisel(3)
    Ω = IndTransverseOps(frame, UniversalOp())

    ramp    = [i + j + k for i in 1:n, j in 1:n, k in 1:n] .+ 0.0
    rank1   = [i * j * k for i in 1:n, j in 1:n, k in 1:n] .+ 0.0
    cases = (
        ("constant",              ones(n, n, n)),
        ("rank-1 outer product",  rank1),
        ("smooth ramp",           ramp),
        ("smooth + tiny noise",   ramp .+ 1e-3 .* randn(n, n, n)),
        ("smooth + tinier noise", ramp .+ 1e-8 .* randn(n, n, n)),
        ("quantized 0:255",       Float64.(rand(0:255, n, n, n))),
        ("scaled up 1e6",         1e6 .* randn(n, n, n)),
        ("scaled down 1e-6",      1e-6 .* randn(n, n, n)),
    )

    @testset "$name" for (name, A) in cases
        Γ = ITensor(A, frame...)
        basis = der(:SylverLining, Ω, P, Γ; tol=1e-6)
        # Not "how many" -- "is every one of them actually a derivation".
        for D in basis
            @test der_residual(Γ, D, P) < LAW_TOL
        end
        # The scalars are always there, so an empty answer is always wrong.
        @test length(basis) >= size(nullspace(P), 2)
    end

    # Scale invariance: the answer must not depend on how the tensor is scaled,
    # which is the property an absolute tolerance breaks and a relative one
    # keeps.  Both failed before -- absolute at 1e6, relative at 1e-6.
    @testset "scale invariance" begin
        A = randn(n, n, n)
        base = length(der(:SylverLining, Ω, P, ITensor(A, frame...); tol=1e-6))
        for c in (1e-6, 1e-3, 1e3, 1e6)
            scaled = length(der(:SylverLining, Ω, P, ITensor(c .* A, frame...); tol=1e-6))
            @test scaled == base
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

    # `:QuickDer` now names the ANY-VALENCE solve-and-lift (QuickDerN.jl); the
    # valence-3 transcription it grew out of stays reachable as `:QuickDer3`,
    # with `:FastDer3Valent` as its alias, so the two can be compared.
    @test get_derivation_method(:QuickDer3) == get_derivation_method(:FastDer3Valent)
    @test get_derivation_method(:QuickDer) isa QuickDerMethod
    @test get_derivation_method(:QuickDer3) isa FastDer3ValentMethod
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

# --- `Dleto.der_residual`: the same law, checked without a Float64 copy -----
#
# The Z-law check is what any CONSUMER of this package runs on an answer, so it
# lives in `src/` now rather than in a benchmark script.  Two things have to
# hold, and neither is a timing.
#
# IT IS THE SAME NUMBER as the `applyDerivation` route above -- that route is
# kept in this file precisely as the independent witness.
#
# AND IT DOES NOT PROMOTE.  `applyDerivation` contracts through ITensor, so a
# Float32 tensor against a Float64 chisel is contracted in Float64 and every
# one of its ~2(n+1) intermediates is twice the tensor.  That bill lands on
# whoever is CHECKING an answer: on a 640x480x1800x3 Float32 movie it is ~66 GB
# to verify something that fits in 6.6.  The blocked route holds two buffers of
# `block_bytes` in the tensor's OWN type, so its allocation is bounded by the
# block and not by the tensor.
@testset "Dleto.der_residual: same law, no promoted copy" begin
    Random.seed!(20260904)
    dims = (60, 60, 40, 3)
    fr = [Index(dims[i], "a$i") for i in 1:4]
    P = Matrix{Float64}(UniversalChisel(4))

    make(T, dm) = begin
        f = [Index(dm[i], "b$i") for i in 1:4]
        G = Array{T}(randn(dm...))
        Ms = [Matrix{T}(randn(dm[a], dm[a])) for a in 1:4]
        (ITensor(G, f...),
         [ITensor(Ms[a], f[a], Index(dm[a], "o$a")) for a in 1:4],
         G)
    end

    @testset "agrees with the applyDerivation route" begin
        # A real derivation (a scalar: X_a = c_a I with Σ_a P[1,a] c_a = 0)
        # and an arbitrary tuple, so both ends of the scale are covered.
        f3 = [Index(6, "s$i") for i in 1:3]
        Γs = ITensor(Array{Float64}(randn(6, 6, 6)), f3...)
        scal = [ITensor(Matrix{Float64}(c * LinearAlgebra.I, 6, 6),
                        f3[a], Index(6, "o$a"))
                for (a, c) in enumerate([1.0, -1.0, 0.0])]
        P3 = Matrix{Float64}(UniversalChisel(3))
        @test Dleto.der_residual(Γs, scal, P3) < LAW_TOL
        @test isapprox(Dleto.der_residual(Γs, scal, P3), der_residual(Γs, scal, P3);
                       atol = 1e-12)

        Γ4, D4, _ = make(Float64, dims)
        @test isapprox(Dleto.der_residual(Γ4, D4, P), der_residual(Γ4, D4, P);
                       rtol = 1e-12)
    end

    @testset "no Float64 array of the tensor's size" begin
        Γ32, D32, G32 = make(Float32, dims)
        # The answer stays in the tensor's type: the clearest single piece of
        # evidence that nothing was promoted on the way.
        @test Dleto.der_residual(Γ32, D32, P) isa Float32

        bb = 2^16
        Dleto.der_residual(Γ32, D32, P; block_bytes = bb)          # compile
        a32 = @allocated Dleto.der_residual(Γ32, D32, P; block_bytes = bb)
        # THE ASSERTION: less than one Float64 array of the tensor's size.
        # Measured 3.6e5 bytes against 3.5e6 here.
        @test a32 < 8 * length(G32)

        # And it is bounded by the BLOCK, not by the tensor: doubling the
        # tensor must not double the bytes.
        Γb, Db, Gb = make(Float32, (60, 60, 80, 3))
        @test length(Gb) == 2 * length(G32)                        # the premise
        Dleto.der_residual(Γb, Db, P; block_bytes = bb)
        ab = @allocated Dleto.der_residual(Γb, Db, P; block_bytes = bb)
        @test ab < 8 * length(Gb)
        @test ab < 1.6 * a32

        # No promotion, stated as a scaling law: at a block big enough to hold
        # the whole tensor the buffers ARE the tensor, and a Float32 run then
        # allocates about half what the same shape does in Float64.  It could
        # only fail to if something Float64-sized were being made either way.
        Γ64, D64, _ = make(Float64, dims)
        Dleto.der_residual(Γ32, D32, P)
        Dleto.der_residual(Γ64, D64, P)
        f32 = @allocated Dleto.der_residual(Γ32, D32, P)
        f64 = @allocated Dleto.der_residual(Γ64, D64, P)
        @test f32 < 0.6 * f64
    end
end
