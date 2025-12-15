using Test
using Dleto
using LinearAlgebra

@testset "Dleto.jl Tests" begin
    @testset "Basic Tensor Operations" begin
        # Test tensor creation and shape
        t = randn(3, 4, 5)
        @test size(t) == (3, 4, 5)
        @test ndims(t) == 3
    end
    
    @testset "Tensor Transformations" begin
        # Create a simple 3D tensor
        t = randn(5, 5, 5)
        
        # Test randomize
        result = Dleto.randomize(t)
        @test size(result.tensor) == size(t)
        @test length(result.matrices) == 3
        
        # Test act (transform tensor along one axis)
        t2 = randn(4, 5, 6)
        M = randn(4, 4)
        t_transformed = Dleto.act(t2, M, 1)
        @test size(t_transformed) == size(t2)
    end
    
    @testset "Tensor Synthesis" begin
        # Test randomOthogonalMatrix
        R = Dleto.randomOthogonalMatrix(5)
        @test size(R) == (5, 5)
        # Check orthogonality: R'*R ≈ I
        @test norm(R' * R - I) < 1e-10
    end
end

println("✓ All tests passed!")
