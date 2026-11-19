using Test, Dleto, ITensors, LinearAlgebra, Random

@testset "Approximate universal derivations" begin
    Γ = ITensor(randn(MersenneTwister(411), 4, 4, 4),
                Index(4, "a"), Index(4, "b"), Index(4, "c"))
    _, N = Dleto.sylvesterLM(Dleto.UniversalOps(Γ),
                              Dleto.UniversalChisel(3), Γ)
    s = svdvals(Matrix(N))
    relative = s ./ first(s)
    cutoff = (relative[end-2] + relative[end-3]) / 2

    result = Dleto.approx_der(Γ; tol=cutoff)
    @test result.map_shape == size(N)
    @test result.accepted_indices == collect((length(s)-2):length(s))
    @test length(result.laws) == 3
    @test result.scalar_dimension == 2
    @test result.approximate_nonscalar_count == 1
    @test result.threshold_gap > 1
    @test result.relative_singular_values ≈ relative

    adapted = Dleto.stratify(Γ; mode=:approx, tol=cutoff,
                              descending_index=length(s)-2)
    @test size(adapted.Σ) == size(Γ)
    @test adapted.accepted_count == 3
    @test adapted.approximate_nonscalar_count == 1
    @test adapted.relative_residual ≈ relative[end-2]
    @test length(adapted.Xs) == 3
    @test size(Dleto.stratify(Γ; tol=1e-8).Σ) == size(Γ) # exact default still works
    array_view = Dleto.stratify(Array(Γ, inds(Γ)...); mode=:approx,
                                 tol=cutoff, descending_index=length(s)-2)
    @test size(array_view.Σ) == size(Γ)
    @test_throws ArgumentError Dleto.stratify(Γ; mode=:approx, tol=cutoff,
                                               method=:SylverLining)
    @test_throws ArgumentError Dleto.approx_stratify(Γ; tol=cutoff,
                                                      descending_index=length(s))
    @test_throws ArgumentError Dleto.approx_der(Γ; tol=0)
    @test_throws ArgumentError Dleto.approx_der(Γ; tol=cutoff,
                                                max_dense_bytes=1)
    @test_throws ArgumentError Dleto.stratify(Γ; mode=:unknown)
end
