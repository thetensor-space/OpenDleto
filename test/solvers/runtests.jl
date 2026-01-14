using LinearMaps
using KrylovKit
using Arpack
# Null Solvers Tests
function testSolver(solver::Dleto.NullSolver, n::Integer, m::Integer )
    @assert n >= m "Wrong sizes"
    A = randn(n,m)
    while LinearAlgebra.rank(A) < m
        A = randn(n,m)
    end
    # @show A
    M = A * A'
    Lmap = LinearMaps.LinearMap(M; issymmetric=true, isposdef=false)
    res= Dleto.solve(solver,Lmap)
    # @show res
    # if !all([ isapprox(norm(A'* res.vecs[:,i]),0.0 ) for i=1:length(res.vals)] ) 
    #     for i = 1:length(res.vals)
    #         v= A'* res.vecs[:,i]
    #         @show i norm(v)
    #     end
    # end
    # @show n m
    # @show res.vals
    # @show size(res.vecs)
    @assert length(res.vals) == min(10,n-m) "wrong dimension of kernel"
    @assert all([ norm(A'* res.vecs[:,i]) < 1e-8 for i=1:length(res.vals)] ) "wrong vector in kernel"
    return true
end

@testset "Solvers Test" begin
    for k in keys(Dleto.NullSolversDict)
        # k =:ArpackSolver
        solver = Dleto.NullSolversDict[k]
        # @show k
        # @show solver
        @testset "Testing Solver $k" begin
            for _= 1:50
                n=rand(5:250)
                m=rand(0:n)
                # n=rand(50:100)
                # m=rand(10:30)
                for _= 1:20
                    @test testSolver(solver, n,m)
                end
                n=rand(200:500)
                m=rand((n-20):n)
                # n=rand(50:100)
                # m=rand(10:30)
                for _= 1:5
                    @test testSolver(solver, n,m)
                end
            end
        end
    end
end

