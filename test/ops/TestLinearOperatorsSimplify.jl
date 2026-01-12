# Local Operator Testing

# TODO add tests that the dimensions are OK; return nothing etc

LinOps=[ Dleto.LinearOperatorsDict[k] for k in keys(Dleto.LinearOperatorsDict)]
# LOps=[ UniversalOp(), DiagonalOp(), SymmetricOp(), AntiSymmetricOp() ];

function testSimplify(Op::Dleto.LinearOperator, dim::Integer, num::Integer)
    localdim = Dleto.localDim(Op,dim)
    Dop = Dleto.simplifyTo(Op).D
    Top = Dleto.simplifyTo(Op).T
    for _ = 1:num
        a = rand(localdim)
        res = Dleto.simplfy(Op,dim,a)
        M = Dleto.embed(Op,dim,a)
        D = Dleto.embed(Dop,dim,res.d)
        T = Dleto.embed(Top,dim,res.t)
        @assert !isapprox(LinearAlgebra.det(T), 0.0)
        @assert isapprox(M * T, T * D)
        @assert isapprox(M, T * D * inv(T))
        @assert isapprox(Matrix(M) * Matrix(T), Matrix(T) * Matrix(D) )
        @assert isapprox(Matrix(M), Matrix(T) * Matrix(D) * inv(Matrix(T)))
        # test for 3 diagonal -- not ok if we introduce other types....
        if dim > 2
            n = norm([D[i,j] for i=1:dim, j=1:dim if abs(i-j) > 1])
            @assert Dleto.__isapproxzero(n)
        end
    end
    return true
end;


@testset "Linear Operators Simplify Tests" begin
    for dim = 1:30
        @testset "Dimension $dim Simplify" begin
            for Op in LinOps
                @test testSimplify(Op, dim, 100)
            end 
        end
    end
end