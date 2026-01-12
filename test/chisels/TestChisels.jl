# Chisel Testing.

PossibleAxis = [ ITensors.Index(rand(2:50), "axis $i") for i=1:30]
eng_to_dict(eng::Vector{Bool})=Dict(zip(PossibleAxis[1:length(eng)], eng))  

# MISSING: normalization of chosels not tested

function testUnivesalChiselEngaged(eng::Vector{Bool})
    valence = length(eng)
    # Universal Chisel test.
    uc1= Dleto.UniversalChisel(eng_to_dict(eng))
    uc = Dleto.extend_chisel(uc1,PossibleAxis[1:valence]) 
    @assert Dleto.engaged_axis(uc) == eng "Failed Chisel engaged test:\r\n $eng \r\n $uc"
    @assert size(uc.ch_m, 1) == 1 "UniversalChisel should have one row."
    @assert size(uc.ch_m, 2) == valence "UniversalChisel should have correct number of columns."
    for a in 1:valence
        for e in 1:valence
            if eng[a] && eng[e] 
                u = zeros(valence)
                u[a] = 1.0
                v = zeros( valence) 
                v[e] = 1.0
                @assert isapprox(norm(Dleto.evaluate_chisel(uc,(u - v))), 0.0)  "UniversalChisel nullity incorrect."
            end
        end
    end
    return true
end;    

function testTuckerChiselEngaged(eng::Vector{Bool})
    valence = length(eng)
    # Tucker Chisel test.
    tc1= Dleto.TuckerChisel(eng_to_dict(eng))
    tc = Dleto.extend_chisel(tc1,PossibleAxis[1:valence]) 
    @assert Dleto.engaged_axis(tc) == eng "TuckerChisel engaged axes incorrect."
    @assert size(tc.ch_m, 1) == sum(eng) "TuckerChisel should have correct number of rows."
    @assert size(tc.ch_m, 2) == valence "TuckerChisel should have correct number of columns."
    for a in 1:valence
        u = zeros(valence)
        u[a] = 1.0
        @assert xor(eng[a], isapprox(norm(Dleto.evaluate_chisel(tc,u)) , 0.0 ))  "TuckerChisel image incorrect, $eng."
    end
    return true
end;

function testCentroidChiselEngaged(eng::Vector{Bool})
    valence = length(eng)
    # Centroid Chisel test.
    e = sum(eng)
    if e < 2
        return true
    end
    cc1= Dleto.CentroidChisel(eng_to_dict(eng))
    cc = Dleto.extend_chisel(cc1,PossibleAxis[1:valence]) 
    @assert Dleto.engaged_axis(cc) == eng "CentroidChisel engaged axes incorrect\r\n $cc\r\rn $eng"
    @assert size(cc.ch_m, 1) == e*(e-1) ÷ 2 "CentroidChisel should have correct number of rows."
    @assert size(cc.ch_m, 2) == valence "CentroidChisel should have correct number of columns."
    # @assert LinearAlgebra.rank(cc) == sum(eng)-1 "CentroidChisel nullity incorrect \r\n $cc"
    u = zeros(valence)
    for a in 1:valence
        if eng[a]
            u[a] = 1.0
        end
    end    
    @assert isapprox(norm(Dleto.evaluate_chisel(cc,u)) , 0.0 )  "CentroidChisel nullity incorrect\r\n $cc \r\n $u."
    return true
end;


function testAdjointChisel(valence::Integer)
    # Adjoint Chisel test.
    if valence < 2
        return true
    end
    for a = 2:valence
        for e = 1:(a-1)
            ac1 = Dleto.AdjointChisel(PossibleAxis[a],PossibleAxis[e])
            ac = Dleto.extend_chisel(ac1,PossibleAxis[1:valence]) 
            @assert Dleto.engaged_axis(ac) == [ i == a || i == e for i in 1:valence ] "AdjointChisel engaged axes incorrect."
            @assert size(ac.ch_m, 1) == 1 "AdjointChisel should have one row."
            u = zeros(valence)
            u[a] = 1.0
            v = zeros(valence)
            v[e] = 1.0
            @assert isapprox(norm(Dleto.evaluate_chisel(ac,(u + v))), 0.0)  "AdjointChisel nullity incorrect $u $v."
        end
    end
    return true
end;



@testset "Chisel Tests" begin
    for valence = 3:4
        @testset "Valance $valence" begin
            eng = [ true for i in 1:valence ]
            @test testUnivesalChiselEngaged(eng)
            @test testTuckerChiselEngaged(eng)
            @test testCentroidChiselEngaged(eng)
            @test testAdjointChisel(valence)
            for _ in 1:50
                # test random engaged.
                eng = [ rand(Bool) for i in 1:valence ]
                @test testUnivesalChiselEngaged(eng)
                @test testTuckerChiselEngaged(eng)
                @test testCentroidChiselEngaged(eng)
            end
        end
    end
end;

