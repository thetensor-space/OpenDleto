# Local Operator Testing

# TODO add tests that the dimensions are OK; return nothing etc


LΩs=[ LocalUniversalOps(), LocalDiagonalOps(), LocalSymmetricOps(), LocalAntiSymmetricOps() ];

function testInverse(Ω::LocalOps, dim::Integer, num::Integer)
    localdim = localDim(Ω,dim)
    for i = 1:num
        a = rand(localdim)
        # M = unsafe_embedding(Ω,dim,a)
        M = embedding(Ω,dim,a)
        mat = Matrix(M)
        @assert isapprox(a, coordinates(Ω, M)) "Failed Invertability\r\n $Ω $a $M \r\n"
        @assert isapprox(a, coordinates(Ω, mat)) "Failed Invertability\r\n $Ω $a $mat \r\n"
    end
    return true
end;

function testTranspose(Ω::LocalOps, dim::Integer, num::Integer)
    localdim = localDim(Ω,dim)
    for i = 1:num
        a = rand(localdim)
        # A = unsafe_embedding(Ω,dim,a)
        A = embedding(Ω,dim,a)
        AA = Matrix(A)
        BB = rand(dim,dim)
        # b = unsafe_transposeEmbedding(Ω,BB)
        b = transposeEmbedding(Ω,BB)
        @assert isapprox(LinearAlgebra.dot(a,b), LinearAlgebra.dot(reshape(AA,dim*dim),reshape(BB,dim*dim))) "Failed Transpose\r\n $Ω $a $AA $BB $b\r\n"
    end
    return true
end;


@testset "LocalOperators Tests" begin
    for dim = 1:20
        @testset "Dimension $dim Tests" begin
            for Ω in LΩs
                @test testInverse(Ω, dim, 100)
                @test testTranspose(Ω, dim, 100)
            end 
        end
    end
end