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
                D = Matrix(rcf.D)
                # test that D is tridiagonal...
                if n > 2
                    d = norm([D[i,j] for i=1:n, j=1:n if abs(i-j) > 1])
                    @test Dleto.__isapproxzero(d)
                # else
                #     @show n,rcf.D, D
                #     r = [D[i,j] for i=1:n, j=1:n if abs(i-j) > 1]
                #     no = norm([D[i,j] for i=1:n, j=1:n if abs(i-j) > 1])
                #     @show r,no
                end
            end
        end
    end
end

