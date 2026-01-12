# 
# Test TestFraming
using Dleto
using ITensors


function testFraming(F::Dleto.Framing{<:Any}, num::Integer)
    len = F.len
    for _ = 1:num
        v = randn(len)
        v_dict = Dleto.toDict(F,v)
        v_vec  = Dleto.toVector(F,v_dict)
        @assert isapprox(v,v_vec) "Not the same"
    end
    return true   
end


@testset "Tests Framing" begin
    for n = 1:1000
        len = rand(1:50)
        f= randn(len)
        F= Dleto.Framing{Float64}(f)
        @test testFraming(F,300) 
    end
end

