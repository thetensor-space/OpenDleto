using Test
using ITensors
using LinearAlgebra
using Random
using Dleto

function _symmetric_scale(d)
    l = d * (d + 1) ÷ 2
    s = ones(Float64, l)
    p = 0
    for j in 1:d, i in 1:j
        p += 1
        i == j || (s[p] = inv(sqrt(2.0)))
    end
    return vcat(s, s, s)
end

function _lattice_sphere(d; noise=0.0, seed=17)
    rng = MersenneTwister(seed)
    A = zeros(Float64, d, d, d)
    for i in 1:d, j in 1:d, k in 1:d
        i + j + k == d + 2 && (A[i, j, k] = randn(rng))
    end
    noise == 0 || (A .+= noise .* randn(rng, size(A)))
    fr = [Index(d, "sphere_$a") for a in 1:3]
    return ITensor(A, fr...)
end

@testset "SymmetricGram derivation method" begin
    @test get_derivation_method(:SymmetricGram) isa SymmetricGramMethod
    @test get_derivation_method(:Auto) isa AutoDerMethod

    @testset "normal coordinates agree with rectangular oracle" begin
        rng = MersenneTwister(8)
        for d in (3, 4)
            fr = [Index(d, "g$(d)_$a") for a in 1:3]
            Γ = ITensor(randn(rng, d, d, d), fr...)
            Ω, P = SymmetricOps(Γ), UniversalChisel(3)
            method = SymmetricGramMethod(eigensolver=:eigen, seed=11)
            _, _, ders = derTrOpsReduced(method, Ω, P, Γ; nd=3, tol=1e-8)

            _, E = sylvesterLM(Ω, P, Γ)
            Er = Matrix(E)
            S = Diagonal(_symmetric_scale(d))
            expected = S * (transpose(Er) * Er) * S
            ref = eigen(Symmetric(expected), 1:3)
            q = ders ./ reshape(_symmetric_scale(d), :, 1)
            @test isapprox(q' * q, I; rtol=1e-11, atol=1e-11)
            @test norm(expected * q - q * Diagonal(ref.values)) < 1e-9
        end
    end

    @testset "fixed modes, diagnostics, and wrappers" begin
        Γ = _lattice_sphere(8)
        Ω, P = SymmetricOps(Γ), UniversalChisel(3)
        m = SymmetricGramMethod(eigensolver=:eigen, seed=50)
        rΩ, expand, coords, rep = derTrOpsReduced(m, Ω, P, Γ;
                                                   nd=3, tol=1e-8,
                                                   return_diagnostics=true)
        @test rΩ === Ω
        @test size(coords) == (globalDim(Ω), 3)
        @test rep.method === :SymmetricGram
        @test rep.policy === :fixed_nd
        @test !rep.certified
        @test rep.nullity == 0
        @test rep.returned == 3
        @test rep.residuals !== nothing
        @test maximum(rep.residuals) < 1e-10
        @test expand * coords[:, 1] == coords[:, 1]

        wrapped = der(:SymmetricGram, Ω, P, Γ; nd=3, tol=1e-8,
                      eigensolver=:eigen, seed=50)
        @test length(wrapped) == 3
        @test all(length(D) == 3 for D in wrapped)
        out = stratify(Ω, P, Γ; method=:SymmetricGram, nd=3, tol=1e-8,
                       eigensolver=:eigen, seed=50)
        @test length(out.Xs) == 3

        noisy = _lattice_sphere(8; noise=1e-3)
        _, _, _, noisy_rep = derTrOpsReduced(SymmetricGramMethod(seed=12, iterations=24),
                                              SymmetricOps(noisy), P, noisy;
                                              nd=3, tol=1e-6,
                                              return_diagnostics=true)
        @test noisy_rep.policy === :fixed_nd
        @test noisy_rep.returned == 3
        @test noisy_rep.residuals[1] < 1e-10
        @test noisy_rep.residuals[2] < 1e-10
        @test noisy_rep.residuals[3] > 1e-5
    end

    @testset "inverse iteration is seeded and reports convergence" begin
        Γ = _lattice_sphere(8)
        Ω, P = SymmetricOps(Γ), UniversalChisel(3)
        m = SymmetricGramMethod(seed=50, iterations=48, oversample=8)
        a = derTrOpsReduced(m, Ω, P, Γ; nd=3, tol=1e-6, return_diagnostics=true)
        b = derTrOpsReduced(m, Ω, P, Γ; nd=3, tol=1e-6, return_diagnostics=true)
        @test a[4].status === :ok
        @test isapprox(a[3], b[3]; rtol=0, atol=0)
        @test maximum(a[4].residuals) < 1e-6
        @test haskey(a[4].stage_times, :symmetric_solve)

        # The rectangular map has fewer rows than the oversampled subspace at
        # d=1 (and at d=2 for all nine modes).  The full right SVD must retain
        # its zero-singular-value complement instead of dropping those modes.
        for d in (1, 2)
            Γsmall = _lattice_sphere(d)
            Ωsmall = SymmetricOps(Γsmall)
            k = globalDim(Ωsmall)
            small = derTrOpsReduced(SymmetricGramMethod(seed=5, iterations=12),
                                    Ωsmall, P, Γsmall; nd=k, tol=1e-6,
                                    return_diagnostics=true)
            @test size(small[3]) == (k, k)
            @test small[4].returned == k
        end
    end

    @testset "Float32 smoke" begin
        rng = MersenneTwister(31)
        fr = [Index(3, "f32_$a") for a in 1:3]
        Γ = ITensor(randn(rng, Float32, 3, 3, 3), fr...)
        out = derTrOpsReduced(SymmetricGramMethod(eigensolver=:eigen),
                              SymmetricOps(Γ), UniversalChisel(3), Γ;
                              nd=2, tol=1e-4, return_diagnostics=true)
        @test eltype(out[3]) === Float32
        @test size(out[3]) == (18, 2)
        @test all(isfinite, out[4].residuals)
    end

    @testset "reject unsupported settings before allocation" begin
        Γ = _lattice_sphere(3)
        Ω, P = SymmetricOps(Γ), UniversalChisel(3)
        @test_throws ArgumentError derTrOpsReduced(SymmetricGramMethod(), Ω, P, Γ; nd=0)
        @test_throws ArgumentError derTrOpsReduced(SymmetricGramMethod(), Ω, P, Γ; nd=1.0)
        @test_throws ArgumentError derTrOpsReduced(SymmetricGramMethod(), Ω, P, Γ; nd=1, tol=Inf)
        @test_throws ArgumentError derTrOpsReduced(SymmetricGramMethod(), UniversalOps(Γ), P, Γ; nd=1)
        @test_throws ArgumentError derTrOpsReduced(SymmetricGramMethod(), Ω, [1.0 1.0 2.0], Γ; nd=1)
        @test_throws ArgumentError derTrOpsReduced(SymmetricGramMethod(max_bytes=1), Ω, P, Γ; nd=1)
        @test_throws ArgumentError SymmetricGramMethod(eigensolver=:bad)

        complexΓ = ITensor(complex.(randn(3, 3, 3), randn(3, 3, 3)),
                           Index(3, "cx"), Index(3, "cy"), Index(3, "cz"))
        @test_throws ArgumentError derTrOpsReduced(SymmetricGramMethod(),
                                                   SymmetricOps(complexΓ), P, complexΓ; nd=1)

        badshape = ITensor(randn(Float64, 3, 4, 3), Index(3, "x"), Index(4, "y"), Index(3, "z"))
        @test_throws ArgumentError derTrOpsReduced(SymmetricGramMethod(), SymmetricOps(badshape), P, badshape; nd=1)
        badframes = IndTransverseOps([Index(3, "u"), Index(3, "v"), Index(3, "w")], SymmetricOp())
        @test_throws ArgumentError derTrOpsReduced(SymmetricGramMethod(), badframes, P, Γ; nd=1)
    end
end
