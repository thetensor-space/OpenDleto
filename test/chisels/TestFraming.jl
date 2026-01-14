# 
# Test TestFraming
using Dleto
using ITensors
using Random

function testFraming(F::Dleto.Framing{<:Any}, num::Integer)
    len = F.len
    for _ = 1:num
        v = randn(len)
        v_dict = Dleto.toDict(F,v)
        v_vec  = Dleto.toVector(F,v_dict)
        @assert isapprox(v,v_vec) "Not the same"
    end
    for _ = 1: num
        D = Dict()
        for k in F.frame
            D[k] = randn()
        end
        D_vec  = Dleto.toVector(F,D)
        D_dict = Dleto.toDict(F,D_vec)
        @assert D == D_dict "Not the same"
    end
    for _ = 1: num
        eng = [rand(Bool) for _=1:len]
        @assert length(Dleto.reduceBy_v(F,eng))==sum(eng) "wrong size"
        F_red = Dleto.reduceBy(F,eng)
    end
    return true   
end


@testset "Tests Framing" begin
    @testset "Frammed by Float64" begin
        for n = 1:1000
            len = rand(1:50)
            f= randn(len)
            F= Dleto.Framing{Float64}(f)
            @test testFraming(F,300) 
        end
    end
    @testset "Frammed by String" begin
        for n = 1:100
            len = rand(1:100)
            f= [ randstring(20) for _=1:len]
            F= Dleto.Framing{String}(f)
            @test testFraming(F,300) 
        end
    end
end

