# 
# TestDletoUtils.jl
#

using Dleto
using ITensors

@testset "realCanonicalForm Tests" begin
    for n = 1:50
        @testset "testing size $n" begin
            for _ = 1:30
                M = rand(n,n)
                rcf = Dleto.realCanonicalForm(M)
                @test !isapprox(LinearAlgebra.det(rcf.T), 0.0)
                @test isapprox(M * rcf.T, rcf.T * rcf.D)
                @test isapprox(M * Matrix(rcf.T), Matrix(rcf.T) * Matrix(rcf.D))
            end
        end
    end

    @testset "pairs are recognised by imag(λ), not by eigenvector proximity" begin
        # Nearly defective: two REAL eigenvalues whose eigenvectors differ by
        # ~1e-6.  The old test (sum of squared differences of the real parts
        # of consecutive eigenvectors < tol) called this a conjugate pair and
        # wrote a zero column into T.
        M = [1.0 1.0; 0.0 1.0 + 1e-6]
        rcf = Dleto.realCanonicalForm(M)
        @test abs(LinearAlgebra.det(rcf.T)) > 1e-8
        @test isapprox(M * rcf.T, rcf.T * rcf.D)
        @test rcf.D isa LinearAlgebra.Diagonal        # no complex block

        # A genuine pair: the rotation by 90 degrees has eigenvalues ±i.
        R = [0.0 -1.0; 1.0 0.0]
        rcf = Dleto.realCanonicalForm(R)
        @test isapprox(R * rcf.T, rcf.T * rcf.D)
        @test !(rcf.D isa LinearAlgebra.Diagonal)
        @test isapprox(abs(rcf.D[1, 2]), 1.0) && isapprox(rcf.D[1, 1], 0.0)

        # Mixed: one real eigenvalue and one pair, in a random frame.
        A = [2.0 0.0 0.0; 0.0 0.5 -3.0; 0.0 3.0 0.5]
        S = randn(3, 3)
        M3 = S * A * inv(S)
        rcf = Dleto.realCanonicalForm(M3)
        @test isapprox(M3 * rcf.T, rcf.T * rcf.D)
        @test abs(LinearAlgebra.det(rcf.T)) > 1e-8
        # exactly one 2x2 block: two off-diagonal entries, one real diagonal entry
        @test count(!iszero, rcf.D - LinearAlgebra.Diagonal(LinearAlgebra.diag(rcf.D))) == 2

        # Symmetric input takes the symmetric path and stays diagonal.
        Sy = LinearAlgebra.Symmetric(randn(4, 4))
        rcf = Dleto.realCanonicalForm(Sy)
        @test rcf.D isa LinearAlgebra.Diagonal
        @test isapprox(Matrix(Sy) * rcf.T, rcf.T * rcf.D)

        @test_throws ArgumentError Dleto.realCanonicalForm([1.0im 0; 0 1.0])
    end
end

@testset "act: one operator per axis, the verb that replaced Base.:*" begin
    fr = [Index(3, "p"), Index(4, "q"), Index(2, "r")]
    Γ = ITensor(randn(3, 4, 2), fr...)
    rn = randomize_tensor(Γ)
    # `randomize_tensor` builds its Δ with `act`; the law it used to state with `*`.
    @test isapprox(act(Γ, rn.Xs), rn.Δ)
    # An array is read into the operators' frame.
    @test isapprox(act(Array(Γ, fr...), rn.Xs), rn.Δ)
    # Plain matrices are embedded against Γ's own frame: same valence, same dims.
    Ms = [randn(3, 3), randn(4, 4), randn(2, 2)]
    out = act(Γ, Ms)
    @test ndims(out) == 3 && sort(collect(ITensors.dim.(inds(out)))) == [2, 3, 4]
    @test_throws DimensionMismatch act(Γ, Ms[1:2])
    @test_throws DimensionMismatch act(Array(Γ, fr...), rn.Xs[1:2])
    # The pirated methods are gone: `*` between an ITensor and a Vector is no
    # longer something this package defines.
    @test !any(m -> m.module === Dleto, methods(*, (ITensor, Vector{ITensor})))
end

@testset "TensorSpace wrapper: sticky +, *, and ⊕" begin
    using Dleto.TensorSpace: Axis, TensorElement, ts, tensor, unwrap, randSurfaceTensor, randFaceCurveTensor, randCurveTensor

    fr = [Axis(3, "p"), Axis(4, "q"), Axis(2, "r")]
    Γ = ITensor(randn(3, 4, 2), fr...)
    Δ = ITensor(randn(3, 4, 2), fr...)
    AΔ = Array(Δ, fr...)
    rn = randomize_tensor(Γ)

    @test ts(Γ) isa TensorElement
    @test tensor(AΔ) isa TensorElement

    # Opt-in action semantics remain available and stay wrapped.
    acted = ts(Γ) * rn.Xs
    @test acted isa TensorElement
    @test isapprox(acted, rn.Δ)

    # randomize_tensor should preserve the active wrapper type if the input is already wrapped.
    Γw = ts(Γ)
    rnw = randomize_tensor(Γw)
    @test rnw.Δ isa TensorElement
    @test isapprox(Γw * rnw.Xs, rnw.Δ)

    # The notebook-style reconstruction law should compare numerically even when
    # the stratified tensor lands in a different internal change-of-basis frame.
    Γw = ts(Γ)
    Γrand = randomize_tensor(Γw)
    Γstrat, Xstrat = stratify(Γrand.Δ)
    @test Γrand.Δ * Xstrat ≈ Γstrat

    # + means tensor addition, and results stay wrapped.
    plus_it = ts(Γ) + Δ
    @test plus_it isa TensorElement
    @test isapprox(plus_it, Γ + Δ)

    plus_arr = ts(Γ) + AΔ
    @test plus_arr isa TensorElement
    @test isapprox(plus_arr, Γ + Δ)

    # * contracts and is sticky in mixed-type expressions.
    prod_it = ts(Γ) * Δ
    @test prod_it isa TensorElement
    @test isapprox(prod_it, Γ * Δ)

    # size should forward cleanly for the wrapper itself.
    @test size(ts(Γ)) == size(Γ)
    @test size(ts(Γ), 2) == size(Γ, 2)

    prod_arr = AΔ * ts(Γ)
    @test prod_arr isa TensorElement
    @test isapprox(prod_arr, Δ * Γ)

    # ⊕ computes direct sums and stays wrapped.
    ds_it = Dleto.:⊕(ts(Γ), Δ)
    @test ds_it isa TensorElement
    ds_ref = Dleto.:⊕(Γ, Δ)
    @test isapprox(Array(unwrap(ds_it), inds(unwrap(ds_it))...), Array(ds_ref, inds(ds_ref)...))

    ds_arr = Dleto.:⊕(AΔ, ts(Γ))
    @test ds_arr isa TensorElement
    ds_ref2 = Dleto.:⊕(Δ, Γ)
    @test isapprox(Array(unwrap(ds_arr), inds(unwrap(ds_arr))...), Array(ds_ref2, inds(ds_ref2)...))

    # The core no-piracy guarantee remains: no Dleto method for ITensor*Vector.
    @test !any(m -> m.module === Dleto, methods(*, (ITensor, Vector{ITensor})))
end

@testset "Notebook helper APIs" begin
    fr = [Index(3, "p"), Index(4, "q"), Index(2, "r")]
    A = ITensor(randn(3, 4, 2), fr...)
    B = ITensor(randn(3, 4, 2), fr...)
    C = ITensor(randn(3, 4, 2), fr...)

    # ASCII helper should match chained direct sums.
    Σ1 = Dleto.direct_sum(A, B, C)
    Σ2 = Dleto.:⊕(Dleto.:⊕(A, B), C)
    @test isapprox(Array(Σ1, inds(Σ1)...), Array(Σ2, inds(Σ2)...))

    # Tutorial defaults should provide a stable, typed setup tuple.
    cfg = Dleto.tutorial_defaults()
    @test cfg.layout == (1, 2)
    @test cfg.pic_size == (900, 400)
    @test cfg.tol == 1e-6

    # One-call notebook warmup should run and return metadata.
    w = Dleto.warmup(dims = (4, 3, 2), tol = 1e-6, verbose = false)
    @test w.dims == (4, 3, 2)
    @test w.tol == 1e-6
    @test w.eltype == Float64
    @test w.stratify
    @test w.nondeg

    # Notebook startup can stay simple: ts is exported at top-level.
    @test isdefined(Dleto, :ts)
    @test Dleto.ts(A) isa Dleto.TensorSpace.TensorElement

    # Tensor-native transverse operator constructors avoid manual frame plumbing.
    Ωu = Dleto.UniversalOps(A)
    Ωs = Dleto.SymmetricOps(A)
    @test valence(Ωu) == ndims(A)
    @test valence(Ωs) == ndims(A)
    @test Set(frames(Ωs)) == Set(inds(A))

    Aw = Dleto.ts(A)
    Ωw = Dleto.SymmetricOps(Aw)
    @test valence(Ωw) == ndims(A)
    @test Set(frames(Ωw)) == Set(inds(A))
end

@testset "TensorSpace synthesis wrappers" begin
    using Dleto.TensorSpace: TensorElement, randSurfaceTensor, randFaceCurveTensor, randCurveTensor, unwrap

    us = collect(range(-1.0, 1.0; length = 6))
    r2 = 1.0
    cutoff = 1e-9 * r2

    S = randSurfaceTensor(us, us, us, cutoff)
    F = randFaceCurveTensor(us, us, us, cutoff)
    C = randCurveTensor(us, us, us, cutoff)

    @test S isa TensorElement
    @test F isa TensorElement
    @test C isa TensorElement

    @test unwrap(S) isa ITensor
    @test unwrap(F) isa ITensor
    @test unwrap(C) isa ITensor

    S_top = Dleto.randSurfaceTensor(us, us, us, cutoff)
    @test S_top isa TensorElement
end

# # Do some very basic tensor contraction yoga to confirm no confusing of frames

# function testMultiplication()
#     passing = true
#     Γ = reshape( 1:8, (2,2,2))

#     # If X's are ITensors then promote AbstractArray Γ to ITensor
#     x = Index(2,"x"); y = Index(2,"y"); z = Index(2,"z");
#     X = ITensor( [ -1.0 0.0; 0.0 1.0], x, x');
#     Y = ITensor( [  1.0 0.0; 0.0 11.0], y, y');
#     Z = ITensor( [  0.0 1.0; 1.0 0.0], z, z');
#     Σ = act(Γ, [X, Y, Z])
#     # @assert asarray(Σ) == reshape( [-5.0 6.0 -77.0 88.0 -1.0 2.0 -33.0 44.0], (2,2,2)) "Failed ITensor promotion multiplication test"
#     if Γ*[X, Y, Z] != [X, Y, Z]*Γ 
#         println("Index frame mismatch in ambidextrous multiplication")
#         passing = false
#     end

#     return passing
# end

# function testRandomization()
#     passing = true
#     Γ = reshape( 1:8, (2,2,2))
    
#     Ξ, Xs = randomize_tensor(Γ)
#     # Check that act(Γ, Xs) == Ξ
#     if !isapprox(act(Γ, Xs), Ξ)
#         println("Randomization test failed: act(Γ, Xs) != Ξ")
#         passing = false
#     end
#     return passing
# end
