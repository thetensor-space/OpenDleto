using ITensors
using Dleto

function testNormispreserved(T::ITensor, num::Integer)
    n = norm(T)
    TOp = Dleto.TransverseOps(T,:OrthogonalOp)
    for _ = 1:num
        rT = Dleto.changeBasisRandom(T,TOp)
        rn = norm(rT.T)
        @assert isapprox(n,rn) "Norm has changed"
    end
    return true
end
@testset "Random Othogonal Changes Preserve Norm" begin
    for val = 2:7
        @testset "Valency $val Tests" begin
            for _ = 1:10 
                axisdim = rand(1:15, val)
                inds = axisdim .|> x -> ITensors.Index(x,"$x")
                T = ITensors.random_itensor(inds...)
                @test testNormispreserved(T, 50)
            end
        end
    end
end
