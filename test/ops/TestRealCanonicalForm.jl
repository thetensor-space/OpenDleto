# 
# Test realCanonicalForm
#

using Dleto
using ITensors

@testset "realCanonicalForm Tests" begin
    for n = 1:50
        @testset "Testing size $n" begin
            for _ = 1:30
                M = rand(n,n)
                rcf = Dleto.realCanonicalForm(M)
                @test !isapprox(LinearAlgebra.det(rcf.T), 0.0)
                @test isapprox(M * rcf.T, rcf.T * rcf.D)
                @test isapprox(M , rcf.T * rcf.D * inv(rcf.T) )
                @test isapprox(M * Matrix(rcf.T), Matrix(rcf.T) * Matrix(rcf.D))
                @test isapprox(M , Matrix(rcf.T) * Matrix(rcf.D) * inv(Matrix(rcf.T)))
            end
        end
    end
end

