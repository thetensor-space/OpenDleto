# Local Operator Testing

# TODO add tests that the dimensions are OK; return nothing etc


LΩs=[ UniversalOp(), DiagonalOp(), SymmetricOp(), AntiSymmetricOp(), ScalarOp(), EmptyOp() ];
# LΩs=[ UniversalOp(), DiagonalOp(), SymmetricOp(), AntiSymmetricOp() ];

function testInverse(Ω::Operator, dim::Integer, num::Integer)
    localdim = localDim(Ω,dim)
    for _ = 1:num
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
    for _ = 1:num
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

function testDualize(Ω::Operator, dim::Integer, num::Integer)
    localdim = localDim(Ω,dim)
    for _ = 1:num
        a = rand(localdim)
        adual = dualize(Ω,dim,a)
        adualdual = dualize(Ω,dim,adual)
        @assert isapprox(a, adualdual)

        A = embed(Ω,dim,a)
        Adual = embed(Ω,dim,adual)
        @assert isapprox(A, LinearAlgebra.transpose(Adual))
        @assert isapprox(Adual, LinearAlgebra.transpose(A))

        BB = rand(dim,dim)
        b = unsafe_transposeEmbed(Ω,BB)
        bdual = unsafe_transposeEmbed(Ω,LinearAlgebra.transpose(BB))
        @assert isapprox(dualize(Ω,dim,b), bdual)
    end
    return true
end;

function testMembership(dim::Integer, num::Integer)
    for _ = 1:num
        M = rand(dim,dim)
        @assert isapprox(M, embed(UniversalOp(), dim, coordinates(UniversalOp(), M))) 
        N = M + M'
        @assert isapprox(N, embed(SymmetricOp(), dim, coordinates(SymmetricOp(), N))) 
        N = LinearAlgebra.Symmetric(M)
        @assert isapprox(N, embed(SymmetricOp(), dim, coordinates(SymmetricOp(), N))) 
        N = M - M'
        @assert isapprox(N, embed(AntiSymmetricOp(), dim, coordinates(AntiSymmetricOp(), N))) 
        D = LinearAlgebra.Diagonal(rand(dim))
        @assert isapprox(D, embed(DiagonalOp(), dim, coordinates(DiagonalOp(), D))) 
        D = zeros(dim,dim)
        [D[i,i] = rand() for i=1:dim ]
        @assert isapprox(D, embed(DiagonalOp(), dim, coordinates(DiagonalOp(), D))) 

        D = rand()* LinearAlgebra.Diagonal([ 1.0 for _ =1:dim])
        @assert isapprox(D, embed(ScalarOp(), dim, coordinates(ScalarOp(), D))) 
        D = zeros(dim,dim)
        d = rand()
        [D[i,i] = d for i=1:dim ]
        @assert isapprox(D, embed(ScalarOp(), dim, coordinates(ScalarOp(), D))) 
        D = zeros(dim,dim)
        @assert isapprox(D, embed(EmptyOp(), dim, coordinates(EmptyOp(), D))) 
    end
    return true;
end

@testset "Operators Tests" begin
    for dim = 1:10
        @testset "Dimension $dim Invertability" begin
            for Ω in LΩs
                @test testInverse(Ω, dim, 100)
            end 
        end
        @testset "Dimension $dim Transpose" begin
            for Ω in LΩs
                @test testInverse(Ω, dim, 100)
            end 
        end
        @testset "Dimension $dim Dualize" begin
            for Ω in LΩs
                @test testDualize(Ω, dim, 100)
            end 
        end
        @test testMembership(dim, 100)
    end
end