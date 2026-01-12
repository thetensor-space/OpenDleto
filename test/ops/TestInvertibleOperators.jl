# Local Operator Testing

# TODO add tests that the dimensions are OK; return nothing etc

InvOps=[ Dleto.InvertableOperatorsDict[k] for k in keys(Dleto.InvertableOperatorsDict)]
# LOps=[ UniversalOp(), DiagonalOp(), SymmetricOp(), AntiSymmetricOp() ];

function testInverse(Op::Dleto.InvertableOperator, dim::Integer, num::Integer)
    localdim = Dleto.localDim(Op,dim)
    id = Dleto.trivial(Op,dim)
    idM = Dleto.embed(Op,dim,id)
    @assert isapprox(idM, LinearAlgebra.Diagonal([1.0 for i=1:dim])) "trivial is not identity matrix"

    for _ = 1:num
        a = Dleto.generate_random(Op,dim)
        # a = rand(localdim)
        # if isa(Op, Dleto.OrthogonalOp)
        #     S = randn(dim,dim)
        #     Q = LinearAlgebra.eigen(LinearAlgebra.Symmetric(S*S') + LinearAlgebra.I).vectors
        #     a = Dleto.coordinates(Op, Q)
        # end
        M = Dleto.embed(Op,dim,a)
        if isnothing(M) 
            @show Op, a, M
            @assert false "Improper Generation"
            continue
        end
        mat = Matrix(M)
        @assert isapprox(a, Dleto.coordinates(Op, M)) "Failed Invertability\r\n $Op $a $M \r\n"
        @assert isapprox(a, Dleto.unsafe_coordinates(Op, M)) "Failed Invertability\r\n $Op $a $M \r\n"
        @assert isapprox(a, Dleto.coordinates(Op, mat)) "Failed Invertability\r\n $Op $a $mat \r\n"
        @assert isapprox(a, Dleto.unsafe_coordinates(Op, mat)) "Failed Invertability\r\n $Op $a $mat \r\n"
        M = Dleto.unsafe_embed(Op,dim,a)
        # this is not needed
#        isnothing(M) && continue
        mat = Matrix(M)
        @assert isapprox(a, Dleto.coordinates(Op, M)) "Failed Invertability\r\n $Op $a $M \r\n"
        @assert isapprox(a, Dleto.unsafe_coordinates(Op, M)) "Failed Invertability\r\n $Op $a $M \r\n"
        @assert isapprox(a, Dleto.coordinates(Op, mat)) "Failed Invertability\r\n $Op $a $mat \r\n"
        @assert isapprox(a, Dleto.unsafe_coordinates(Op, mat)) "Failed Invertability\r\n $Op $a $mat \r\n"
    end
    return true
end;

function testStar(Op::Dleto.InvertableOperator, dim::Integer, num::Integer)
    if !Dleto.closedUnderStar(Op)
        println("Not closed under star $Op") 
        return true
    end
    localdim = Dleto.localDim(Op,dim)
    for _ = 1:num
        a = Dleto.generate_random(Op,dim)
        # a = rand(localdim)
        # M = Dleto.embed(Op,dim,a)
        # if isa(Op, Dleto.OrthogonalOp)
        #     S = randn(dim,dim)
        #     Q = LinearAlgebra.eigen(LinearAlgebra.Symmetric(S*S') + LinearAlgebra.I).vectors
        #     a = Dleto.coordinates(Op, Q)
        # end
        M = Dleto.embed(Op,dim,a)
        if isnothing(M) 
            @show Op, a, M
            @assert false "Improper Generation"
            continue
        end
        astar = Dleto.star(Op,dim,a)
        astarstar = Dleto.star(Op,dim,astar)
        @assert isapprox(a, astarstar)

        A = Dleto.embed(Op,dim,a)
        Astar = Dleto.embed(Op,dim,astar)
        @assert isapprox(A, LinearAlgebra.transpose(inv(Astar)))
        @assert isapprox(Astar, LinearAlgebra.transpose(inv(A)))
    end
    return true
end;

function testMembershipInv(dim::Integer, num::Integer)
    for _ = 1:num
        M = rand(dim,dim)
        if ! Dleto.__isapproxzero(LinearAlgebra.det(M))
            @assert isapprox(M, Dleto.embed(Dleto.InvertableOp(), dim, Dleto.coordinates(Dleto.InvertableOp(), M)))
        end 
        S = randn(dim,dim)
        Q = LinearAlgebra.eigen(LinearAlgebra.Symmetric(S*S') + LinearAlgebra.I).vectors
        @assert isapprox(Q, Dleto.embed(Dleto.OrthogonalOp(), dim, Dleto.coordinates(Dleto.OrthogonalOp(), Q))) 
    end
    return true;
end

@testset "Invertible Operators Tests" begin
    # for Op in LinOps
    #     @show Op
    # end 
    for dim = 1:30
        # @show dim
        @testset "Dimension $dim Invertability" begin
            for Op in InvOps
                @test testInverse(Op, dim, 100)
            end 
        end
        @testset "Dimension $dim Star" begin
            for Op in InvOps
                @test testStar(Op, dim, 100)
            end 
        end
        @test testMembership(dim, 100)
    end
end