using Random

@testset "orthogonal randomization materializes a dense tensor" begin
    # The input has sparse numerical support but ordinary (non-QN) ITensors
    # allocate Dense storage.  Contracting the dense change-of-basis matrices
    # must materialize a full dense tensor, rather than retain a lazy graph.
    Random.seed!(6006)
    d = 6
    r2 = (d - 1.0)^2
    us = [r2 * (i / (d - 1) - 1 / 3) for i in 0:d-1]
    S = randSurfaceTensor(us, us, us, 1e-9 * r2)
    fr = collect(inds(S))
    Δ = randomize_tensor(S; type = :orthogonal).Δ
    support = Array(S, fr...)
    dense = Array(Δ, collect(inds(Δ))...)

    @test occursin("NDTensors.Dense", string(typeof(store(S))))
    @test occursin("NDTensors.Dense", string(typeof(store(Δ))))
    @test count(!iszero, support) < length(support)
    @test count(!iszero, dense) == length(dense)
end
