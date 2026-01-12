# Local Operator Testing

# TODO add tests that the dimensions are OK; return nothing etc

LinOps=[ Dleto.LinearOperatorsDict[k] for k in keys(Dleto.LinearOperatorsDict)]
# LOps=[ UniversalOp(), DiagonalOp(), SymmetricOp(), AntiSymmetricOp() ];

function testInverse(Op::Dleto.LinearOperator, dim::Integer, num::Integer)
    localdim = Dleto.localDim(Op,dim)
    id = Dleto.trivial(Op,dim)
    idM = Dleto.embed(Op,dim,id)
    @assert isapprox(idM, zeros(dim,dim)) "trivial is not zero matrix"
    for _ = 1:num
        a = Dleto.generate_random(Op,dim)
#        a = rand(localdim)
        # @show Op, a
        M = Dleto.embed(Op,dim,a)
        mat = Matrix(M)
        # @show Op, M, mat
        @assert isapprox(a, Dleto.coordinates(Op, M)) "Failed Invertability\r\n $Op $a $M \r\n"
        @assert isapprox(a, Dleto.unsafe_coordinates(Op, M)) "Failed Invertability\r\n $Op $a $M \r\n"
        @assert isapprox(a, Dleto.coordinates(Op, mat)) "Failed Invertability\r\n $Op $a $mat \r\n"
        @assert isapprox(a, Dleto.unsafe_coordinates(Op, mat)) "Failed Invertability\r\n $Op $a $mat \r\n"
        M = Dleto.unsafe_embed(Op,dim,a)
        mat = Matrix(M)
        @assert isapprox(a, Dleto.coordinates(Op, M)) "Failed Invertability\r\n $Op $a $M \r\n"
        @assert isapprox(a, Dleto.unsafe_coordinates(Op, M)) "Failed Invertability\r\n $Op $a $M \r\n"
        @assert isapprox(a, Dleto.coordinates(Op, mat)) "Failed Invertability\r\n $Op $a $mat \r\n"
        @assert isapprox(a, Dleto.unsafe_coordinates(Op, mat)) "Failed Invertability\r\n $Op $a $mat \r\n"
    end
    return true
end;

function testTranspose(Op::Dleto.LinearOperator, dim::Integer, num::Integer)
    localdim = Dleto.localDim(Op,dim)
    for _ = 1:num
        a = Dleto.generate_random(Op,dim)
        # a = rand(localdim)
        BB = rand(dim,dim)
        # A = unsafe_embed(Op,dim,a)
        A = Dleto.embed(Op,dim,a)
        AA = Matrix(A)
        b = Dleto.transposeEmbed(Op,BB)
        @assert isapprox(LinearAlgebra.dot(a,b), LinearAlgebra.dot(reshape(AA,dim*dim),reshape(BB,dim*dim))) "Failed Transpose\r\n $Op $a $AA $BB $b\r\n"
        b = Dleto.unsafe_transposeEmbed(Op,BB)
        @assert isapprox(LinearAlgebra.dot(a,b), LinearAlgebra.dot(reshape(AA,dim*dim),reshape(BB,dim*dim))) "Failed Transpose\r\n $Op $a $AA $BB $b\r\n"

        A = Dleto.unsafe_embed(Op,dim,a)
        AA = Matrix(A)
        b = Dleto.transposeEmbed(Op,BB)
        @assert isapprox(LinearAlgebra.dot(a,b), LinearAlgebra.dot(reshape(AA,dim*dim),reshape(BB,dim*dim))) "Failed Transpose\r\n $Op $a $AA $BB $b\r\n"
        b = Dleto.unsafe_transposeEmbed(Op,BB)
        @assert isapprox(LinearAlgebra.dot(a,b), LinearAlgebra.dot(reshape(AA,dim*dim),reshape(BB,dim*dim))) "Failed Transpose\r\n $Op $a $AA $BB $b\r\n"
    end
    return true
end;

function testStar(Op::Dleto.LinearOperator, dim::Integer, num::Integer)
    if !Dleto.closedUnderStar(Op)
        println("Not closed under star $Op") 
        return true
    end
    localdim = Dleto.localDim(Op,dim)
    for _ = 1:num
        a = Dleto.generate_random(Op,dim)
        # a = rand(localdim)
        astar = Dleto.star(Op,dim,a)
        astarstar = Dleto.star(Op,dim,astar)
        @assert isapprox(a, astarstar)

        A = Dleto.embed(Op,dim,a)
        Astar = Dleto.embed(Op,dim,astar)
        @assert isapprox(A, LinearAlgebra.transpose(Astar))
        @assert isapprox(Astar, LinearAlgebra.transpose(A))

        BB = Dleto.rand(dim,dim)
        b = Dleto.unsafe_transposeEmbed(Op,BB)
        bstar = Dleto.unsafe_transposeEmbed(Op,LinearAlgebra.transpose(BB))
        @assert isapprox(Dleto.star(Op,dim,b), bstar)
    end
    return true
end;

function testMembership(dim::Integer, num::Integer)
    for _ = 1:num
        M = rand(dim,dim)
        @assert isapprox(M, Dleto.embed(Dleto.UniversalOp(), dim, Dleto.coordinates(Dleto.UniversalOp(), M))) 
        N = M + M'
        @assert isapprox(N, Dleto.embed(Dleto.SymmetricOp(), dim, Dleto.coordinates(Dleto.SymmetricOp(), N))) 
        N = LinearAlgebra.Symmetric(M)
        @assert isapprox(N, Dleto.embed(Dleto.SymmetricOp(), dim, Dleto.coordinates(Dleto.SymmetricOp(), N))) 
        N = M - M'
        @assert isapprox(N, Dleto.embed(Dleto.AntiSymmetricOp(), dim, Dleto.coordinates(Dleto.AntiSymmetricOp(), N))) 
        D = LinearAlgebra.Diagonal(rand(dim))
        @assert isapprox(D, Dleto.embed(Dleto.DiagonalOp(), dim, Dleto.coordinates(Dleto.DiagonalOp(), D))) 
        D = zeros(dim,dim)
        [D[i,i] = rand() for i=1:dim ]
        @assert isapprox(D, Dleto.embed(Dleto.DiagonalOp(), dim, Dleto.coordinates(Dleto.DiagonalOp(), D))) 

        D = rand()* LinearAlgebra.Diagonal([ 1.0 for _ =1:dim])
        @assert isapprox(D, Dleto.embed(Dleto.ScalarOp(), dim, Dleto.coordinates(Dleto.ScalarOp(), D))) 
        D = zeros(dim,dim)
        d = rand()
        [D[i,i] = d for i=1:dim ]
        @assert isapprox(D, Dleto.embed(Dleto.ScalarOp(), dim, Dleto.coordinates(Dleto.ScalarOp(), D))) 
        D = zeros(dim,dim)
        @assert isapprox(D, Dleto.embed(Dleto.EmptyOp(), dim, Dleto.coordinates(Dleto.EmptyOp(), D))) 
    end
    return true;
end

@testset "Linear Operators Tests" begin
    # for Op in LinOps
    #     @show Op
    # end 
    for dim = 1:10
        # @show dim
        @testset "Dimension $dim Invertability" begin
            for Op in LinOps
                @test testInverse(Op, dim, 100)
            end 
        end
        @testset "Dimension $dim Transpose" begin
            for Op in LinOps
                @test testTranspose(Op, dim, 100)
            end 
        end
        @testset "Dimension $dim Star" begin
            for Op in LinOps
                @test testStar(Op, dim, 100)
            end 
        end
        @test testMembership(dim, 100)
    end
end