# Local Operator Testing

# TODO add tests that the dimensions are OK; return nothing etc


LΩs=[ UniversalOp(), DiagonalOp(), SymmetricOp(), AntiSymmetricOp(), ScalarOp(), EmptyOp() ];
# LΩs=[ UniversalOp(), DiagonalOp(), SymmetricOp(), AntiSymmetricOp() ];

function testInverse(Ω::Operator, dim::Integer, num::Integer)
    localdim = localDim(Ω,dim)
    for i = 1:num
        a = rand(localdim)
        # @show Ω, a
        M = embed(Ω,dim,a)
        mat = Matrix(M)
        # @show Ω, M, mat
        @assert isapprox(a, coordinates(Ω, M)) "Failed Invertability\r\n $Ω $a $M \r\n"
        @assert isapprox(a, unsafe_coordinates(Ω, M)) "Failed Invertability\r\n $Ω $a $M \r\n"
        @assert isapprox(a, coordinates(Ω, mat)) "Failed Invertability\r\n $Ω $a $mat \r\n"
        @assert isapprox(a, unsafe_coordinates(Ω, mat)) "Failed Invertability\r\n $Ω $a $mat \r\n"
        M = unsafe_embed(Ω,dim,a)
        mat = Matrix(M)
        @assert isapprox(a, coordinates(Ω, M)) "Failed Invertability\r\n $Ω $a $M \r\n"
        @assert isapprox(a, unsafe_coordinates(Ω, M)) "Failed Invertability\r\n $Ω $a $M \r\n"
        @assert isapprox(a, coordinates(Ω, mat)) "Failed Invertability\r\n $Ω $a $mat \r\n"
        @assert isapprox(a, unsafe_coordinates(Ω, mat)) "Failed Invertability\r\n $Ω $a $mat \r\n"
    end
    return true
end;

function testTranspose(Ω::Operator, dim::Integer, num::Integer)
    localdim = localDim(Ω,dim)
    for i = 1:num
        a = rand(localdim)
        BB = rand(dim,dim)
        # A = unsafe_embed(Ω,dim,a)
        A = embed(Ω,dim,a)
        AA = Matrix(A)
        b = transposeEmbed(Ω,BB)
        @assert isapprox(LinearAlgebra.dot(a,b), LinearAlgebra.dot(reshape(AA,dim*dim),reshape(BB,dim*dim))) "Failed Transpose\r\n $Ω $a $AA $BB $b\r\n"
        b = unsafe_transposeEmbed(Ω,BB)
        @assert isapprox(LinearAlgebra.dot(a,b), LinearAlgebra.dot(reshape(AA,dim*dim),reshape(BB,dim*dim))) "Failed Transpose\r\n $Ω $a $AA $BB $b\r\n"

        A = unsafe_embed(Ω,dim,a)
        AA = Matrix(A)
        b = transposeEmbed(Ω,BB)
        @assert isapprox(LinearAlgebra.dot(a,b), LinearAlgebra.dot(reshape(AA,dim*dim),reshape(BB,dim*dim))) "Failed Transpose\r\n $Ω $a $AA $BB $b\r\n"
        b = unsafe_transposeEmbed(Ω,BB)
        @assert isapprox(LinearAlgebra.dot(a,b), LinearAlgebra.dot(reshape(AA,dim*dim),reshape(BB,dim*dim))) "Failed Transpose\r\n $Ω $a $AA $BB $b\r\n"
    end
    return true
end;


@testset "LocalOperators Tests" begin
    for dim = 1:10
        @testset "Dimension $dim Tests" begin
            for Ω in LΩs
                @test testInverse(Ω, dim, 100)
                @test testTranspose(Ω, dim, 100)
            end 
        end
    end
end