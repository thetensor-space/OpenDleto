using Random

@testset "FastDer3Valent Derivation Method" begin
    Random.seed!(7)

    dims = (4, 4, 4)
    Γ_array = randn(Float64, dims...)
    frame = [Index(dims[i], "a_$i") for i in 1:3]
    Γ = ITensor(Γ_array, frame...)

    Ω = IndTransverseOps(frame, UniversalOp())
    ch = UniversalChisel(3)

    @testset "derTrOpsReduced returns basis coordinates" begin
        method = FastDer3ValentMethod()
        rΩ, expand_map, ders = Dleto.derTrOpsReduced(method, Ω, ch, Γ; tol=1e-6)

        @test rΩ === Ω
        @test size(ders, 1) == globalDim(Ω)
        @test size(ders, 2) >= 1

        # Ensure expanded vectors embed as three derivation matrices.
        v = expand_map(ders[:, 1])
        δ = embedITensors(Ω, v)
        @test length(δ) == 3
    end

    @testset "stratify supports solver selection" begin
        fast = stratify(Γ; method=:FastDer3Valent, tol=1e-6)
        base = stratify(Γ; method=:SylverLining, tol=1e-6)

        @test ndims(Array(fast.Σ, inds(fast.Σ)...)) == 3
        @test ndims(Array(base.Σ, inds(base.Σ)...)) == 3
        @test length(fast.Xs) == 3
        @test length(base.Xs) == 3
    end
end
