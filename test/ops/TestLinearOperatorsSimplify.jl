# Local Operator Testing

# TODO add tests that the dimensions are OK; return nothing etc

LinOps=[ Dleto.LinearOperatorsDict[k] for k in keys(Dleto.LinearOperatorsDict)]
# LOps=[ UniversalOp(), DiagonalOp(), SymmetricOp(), AntiSymmetricOp() ];

function testSimplify(Op::Dleto.LinearOperator, dim::Integer, num::Integer)
    localdim = Dleto.localDim(Op,dim)
    (Dop, Top) = Dleto.simplifyTo(Op)
    for _ = 1:num
        a = rand(localdim)
        (d,t) = Dleto.simplfy(Op,dim,a)
        M = Dleto.embed(Op,dim,a)
        D = Dleto.embed(Dop,dim,d)
        T = Dleto.embed(Top,dim,t)
        # if isnothing(T)
        #     @show Op, dim
        #     @show a, d, t
        #     @show M, D, T
        # end
        # @show Matrix(M), Matrix(D), Matrix(T)
        # diff = Matrix(M) * Matrix(T) - Matrix(T) * Matrix(D)
        # diff2 = Matrix(M) - Matrix(T) * Matrix(D) * inv(Matrix(T))
        # @show diff, diff2

        @assert !isapprox(LinearAlgebra.det(T), 0.0)
        # if !isapprox(M * T, T * D)
        #     @show Op, M
        #     @show T, D
        #     @show M * T - T * D
        # end
        @assert isapprox(M * T, T * D)
        @assert isapprox(M, T * D * inv(T))
        @assert isapprox(Matrix(M) * Matrix(T), Matrix(T) * Matrix(D) )
        @assert isapprox(Matrix(M), Matrix(T) * Matrix(D) * inv(Matrix(T)))
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