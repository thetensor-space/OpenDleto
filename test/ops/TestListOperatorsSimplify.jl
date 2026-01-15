function testSimplify(LOp::Dleto.ListOperators, num::Integer)
    res = Dleto.simplifyTo(LOp)
    val = Dleto.valency(LOp)
#    @show val LOp res.D res.T
    for _ = 1:num
        a = Dleto.generate_random(LOp)
        MMats = Dleto.embedMatrices(LOp, a)
        aa = Dleto.simplify(LOp,a)
        DMats = Dleto.embedMatrices(res.D, aa.d)
        TMats = Dleto.embedMatrices(res.T, aa.t)
        @assert all( [ !isapprox(LinearAlgebra.det(TMats[i]), 0.0) for i=1:val]) 
        r = [ isapprox(MMats[i] * TMats[i], TMats[i] * DMats[i]) for i=1:val]
        # if !all(r)
        #     @show LOp, r
        # end
        @assert all( [ isapprox(MMats[i] * TMats[i], TMats[i] * DMats[i]) for i=1:val])
        @assert all( [ isapprox(MMats[i], TMats[i] * DMats[i] * inv(TMats[i])) for i=1:val])
        @assert all( [ isapprox(Matrix(MMats[i]) * Matrix(TMats[i]), Matrix(TMats[i]) * Matrix(DMats[i]) ) for i=1:val])
        @assert all( [ isapprox(Matrix(MMats[i]), Matrix(TMats[i]) * Matrix(DMats[i]) * inv(Matrix(TMats[i]))) for i=1:val])
    end
    return true
end



@testset "List Operators Independant Simplify Tests" begin
    for val = 1:10
        @testset "Valency $val Tests" begin
            for _ = 1:10
                axisdim = rand(1:15, val)
                localops = rand(1:length(LinOps), val) .|> (i-> LinOps[i] )
                LOp = Dleto.IndListOperators(axisdim,localops)
                @test testSimplify(LOp, 100)
            end 
        end
    end
end

syms = [ [1,1,1,1], [1,1,3,3,5,5], [1,2,1,2,2,1], [1,2,3,-1,2,3], [1,2,-2,-2,5,-5,-1,-1], [1,-1,3,3,-3,1,7,8],[1,2,3,4,-4,-4,-4,-3,-3,-3,-2,-2,-1,-1] ]

@testset "List Operators Symmetry Simplify Tests" begin
    for val = 1:10
        @testset "Valency $val no symmetry Tests" begin
            for _ = 1:10
                axisdim = rand(1:15, val)
                localops = rand(1:length(LinOps), val) .|> (i-> LinOps[i] )
                LOp = Dleto.SymListOperators(axisdim,localops, [i for i=1:val] )
                @test testSimplify(LOp, 100)
            end 
        end
    end

    val = 20
    for sym in syms 
        @testset "Symmetry $sym Tests" begin
            for _ = 1:30
                axisdim = rand(1:10, val)
                localops = rand(1:length(LinOps), val) .|> (i-> LinOps[i] )
                raxisdim = [ axisdim[abs(sym[i])] for i=1:length(sym)]
                rlocalops = [ localops[abs(sym[i])] for i=1:length(sym)] 
                LOp = Dleto.SymListOperators(raxisdim, rlocalops, sym)
                @test testSimplify(LOp, 100)
                for _=1:5
                    eng = [ rand(Bool) for i in 1:length(sym) ]
                    reducedval= sum(eng)
                    if reducedval==0
                        continue
                    end
                    res = Dleto.reduceBy(LOp, eng)
                    rLOp=res.rLOp
                    @test testSimplify(rLOp, 100)
                end
            end
        end 
    end
end
