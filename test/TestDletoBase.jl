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
