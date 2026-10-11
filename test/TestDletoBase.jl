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
    @testset "complex input is conjugated by T" begin
        # Regression: the real-part projection used for conjugate pairs of a
        # real matrix returned a non-conjugating T on complex input.
        for n in (2, 5, 9), _ in 1:10
            M = randn(ComplexF64, n, n)
            rcf = Dleto.realCanonicalForm(M)
            @test isapprox(M * Matrix(rcf.T), Matrix(rcf.T) * Matrix(rcf.D))
            @test isapprox(Matrix(rcf.D), LinearAlgebra.Diagonal(LinearAlgebra.diag(Matrix(rcf.D))))
        end
    end
end

@testset "nondeg basis spans the column space" begin
    # Regression: the basis used to be the first r grid positions of Array(V, link, a),
    # not the singular vectors; it spanned ~60% of the column space and, for complex
    # input, gave ill-conditioned cores.  Now: E = conj(W), Δ = Γ * E = Wᴴ Γ,
    # Γ = Δ * dag(E), and the core factor matrices are as well conditioned as the
    # planted factors.
    for T in (Float64, ComplexF64), (d, r) in ((12, 4), (30, 3))
        P = randn(T, d, r); rest = randn(T, 11, 10)
        Tl = reshape(P * LinearAlgebra.pinv(P) * reshape(randn(T, d, 11, 10), d, :), d, 11, 10)
        fr = [Index(d, "x"), Index(11, "y"), Index(10, "z")]
        Γ = ITensor(Tl, fr...)
        Δ, Es = nondeg(Γ, [fr[1]]; tol = 1e-8)
        E = only(Es); e2 = only(filter(i -> i != fr[1], inds(E))); Em = Array(E, fr[1], e2)
        @test ITensors.dim(e2) == r
        @test isapprox(LinearAlgebra.norm(Em * LinearAlgebra.pinv(Em) * conj(P)), LinearAlgebra.norm(P); rtol = 1e-8)   # conj(E) spans range(P)
        rec = Δ * dag(E)
        @test isapprox(Array(rec, fr...), Tl; rtol = 1e-8)
        core = transpose(Em) * P
        @test LinearAlgebra.cond(core) < 10 * LinearAlgebra.cond(P)
    end
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
#     Σ = Γ*[X, Y, Z]
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
#     # Check that Γ * Xs == Ξ
#     if !isapprox(Γ * Xs, Ξ)
#         println("Randomization test failed: Γ * Xs != Ξ")
#         passing = false
#     end
#     return passing
# end
