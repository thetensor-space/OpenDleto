using Test
using Random
using LinearAlgebra
using ITensors
using Dleto

include(joinpath(@__DIR__, "SymmetricSphereGram.jl"))

@testset "symmetric sphere Gram prototype" begin
    Random.seed!(20260914)
    for d in (3, 4)
        fr = [Index(d, "g$a") for a in 1:3]
        Γ = ITensor(randn(d, d, d), fr...)
        # Dense eigensolve remains the algebraic oracle; inverse iteration is
        # the default d=50 path and must agree on this small deterministic case.
        out = sphere_symmetric_gram(Γ; nd=3, eigensolver=:eigen)
        fast = sphere_symmetric_gram(Γ; nd=3, seed=11)
        Ω = SymmetricOps(Γ)
        _, E = sylvesterLM(Ω, UniversalChisel(3), Γ)
        Er = Matrix(E)
        raw_normal = transpose(Er) * Er

        # q is orthonormal-Frobenius symmetric coordinates; Dleto's raw
        # coordinate x has x_offdiag = q_offdiag / sqrt(2).
        l = d * (d + 1) ÷ 2
        s = ones(Float64, l)
        p = 0
        for j in 1:d, i in 1:j
            p += 1
            i == j || (s[p] = inv(sqrt(2.0)))
        end
        S = Diagonal(vcat(s, s, s))
        expected = S * raw_normal * S
        @test isapprox(out.normal, expected; rtol=1e-12, atol=1e-12)

        # Probe the normal action independently of a full matrix comparison.
        for _ in 1:5
            q = randn(3l)
            @test isapprox(out.normal * q, S * (transpose(Er) * (Er * (S * q)));
                           rtol=1e-12, atol=1e-12)
        end
        refvals = eigen(Symmetric(expected), 1:3).values
        @test isapprox(out.eigenvalues, refvals; rtol=1e-12, atol=1e-12)
        @test maximum(abs.(out.normal * out.eigenvectors -
                           out.eigenvectors * Diagonal(out.eigenvalues))) < 1e-11
        @test isapprox(fast.eigenvalues, refvals; rtol=1e-8, atol=1e-10)
        @test maximum(abs.(out.normal * fast.eigenvectors -
                           fast.eigenvectors * Diagonal(fast.eigenvalues))) < 1e-7
    end
end
