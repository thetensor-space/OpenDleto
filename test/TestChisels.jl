# Chisel Testing.

# using Test
# using ITensors
# using Dleto

# MISSING: normalization of chosels not tested

function testUnivesalChiselEngaged(eng::Vector{Bool})
    valence = length(eng)
    # Universal Chisel test.
    uc = UniversalChisel(eng)
    @assert engaged(uc) == eng "Failed Chisel engaged test:\r\n $eng \r\n $uc"
    @assert size(uc, 1) == 1 "UniversalChisel should have one row."
    @assert size(uc, 2) == valence "UniversalChisel should have correct number of columns."
    for a in 1:valence
        for e in 1:valence
            if eng[a] && eng[e] 
                u = zeros(valence)
                u[a] = 1.0
                v = zeros( valence)
                v[e] = 1.0
                @assert isapprox(EvaluateChisel(uc,(u - v)), 0.0)  "UniversalChisel nullity incorrect."
            end
        end
    end
    return true
end;    

function testTuckerChiselEngaged(eng::Vector{Bool})
    valence = length(eng)
    # Tucker Chisel test.
    tc = TuckerChisel(eng)
    @assert engaged(tc) == eng "TuckerChisel engaged axes incorrect."
    @assert size(tc, 1) == sum(eng) "TuckerChisel should have correct number of rows."
    @assert size(tc, 2) == valence "TuckerChisel should have correct number of columns."
    for a in 1:valence
        if !eng[a]
            u = zeros(valence)
            u[a] = 1.0
            @assert isapprox(EvaluateChisel(tc,u) , 0.0 )  "TuckerChisel image incorrect, $eng."
        end
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
    cc = CentroidChisel(eng)
    @assert engaged(cc) == eng "CentroidChisel engaged axes incorrect\r\n $cc\r\rn $eng"
    @assert size(cc, 1) == e*(e-1) ÷ 2 "CentroidChisel should have correct number of rows."
    @assert size(cc, 2) == valence "CentroidChisel should have correct number of columns."
    # @assert LinearAlgebra.rank(cc) == sum(eng)-1 "CentroidChisel nullity incorrect \r\n $cc"
    u = zeros(valence)
    for a in 1:valence
        if eng[a]
            u[a] = 1.0
        end
    end
    @assert isapprox(EvaluateChisel(cc,u), 0.0)  "CentroidChisel nullity incorrect\r\n $cc \r\n $u."
    return true
end;


function testAdjointChisel(valence::Integer)
    # Adjoint Chisel test.
    if valence < 2
        return true
    end
    for a = 1:(valence-1)
        for e = (a+1):valence
            ac = AdjointChisel(valence, a,e)
            @assert engaged(ac) == [ i == a || i == e for i in 1:valence ] "AdjointChisel engaged axes incorrect."
            @assert size(ac, 1) == 1 "AdjointChisel should have one row."
            u = zeros(valence)
            u[a] = 1.0
            v = zeros(valence)
            v[e] = 1.0
            @assert isapprox(EvaluateChisel(ac,u+v), 0.0)  "AdjointChisel nullity incorrect $ac\r\n $u\r\n $v."
        end
    end
    return true
end;



@testset "Chisel Tests" begin
    for valence = 1:20
        @testset "Valance $valence" begin
            eng = [ true for i in 1:valence ]
            @test testUnivesalChiselEngaged(eng)
            @test testTuckerChiselEngaged(eng)
            @test testCentroidChiselEngaged(eng)
            @test testAdjointChisel(valence)
            for _ in 1:10
                # test random engaged.
                eng = [ rand(1:10) % 2 == 1 for i in 1:valence ]
                @test testUnivesalChiselEngaged(eng)
                @test testTuckerChiselEngaged(eng)
                @test testCentroidChiselEngaged(eng)
            end
        end
    end
end;




# function testAllChisels()
#     passing = all([testChisels(valence) for valence in 1:20])
#     if passing
#         println("✓ All Delto Chisel tests passed.")
#     else
#         println("Some Delto Chisel tests failed.")
#     end
#     return passing
# end;

# function testChisels(valence::Integer)
#     passing = true
#     # test numeric.
#     eng = [ true for i in 1:valence ]
#     passing &= testChiselEngaged(eng)
#     # passing = true
#     for _ in 1:10
#         # test random engaged.
#         eng = [ rand(1:10) % 2 == 1 for i in 1:valence ]
#         passing &= testChiselEngaged(eng)
#     end
#     return passing
# end;


# function testChiselEngaged(eng::Vector{Bool})
#     passing = true
#     valence = length(eng)
#     # Universal Chisel test.
#     uc = UniversalChisel(eng)
#     passing &= engaged(uc) == eng
#     if !passing
#         println("Failed Chisel engaged test:\r\n $eng \r\n $uc")
#     end
#     @assert size(uc, 1) == 1 "UniversalChisel should have one row."
#     @assert size(uc, 2) == valence "UniversalChisel should have correct number of columns."
#     for a in 1:valence
#         for e in 1:valence
#             if eng[a] && eng[e] 
#                 u = zeros(valence)
#                 u[a] = 1.0
#                 v = zeros( valence)
#                 v[e] = 1.0
#                 @assert isapprox(EvaluateChisel(uc,(u - v)), 0.0)  "UniversalChisel nullity incorrect."
#             end
#         end
#     end
    

#     # Tucker Chisel test.
#     tc = TuckerChisel(eng)
#     @assert engaged(tc) == eng "TuckerChisel engaged axes incorrect."
#     @assert size(tc, 1) == sum(eng) "TuckerChisel should have correct number of rows."
#     @assert size(tc, 2) == valence "TuckerChisel should have correct number of columns."
#     for a in 1:valence
#         if !eng[a]
#             u = zeros(valence)
#             u[a] = 1.0
#             @assert isapprox(EvaluateChisel(tc,u) , 0.0 )  "TuckerChisel image incorrect, $eng."
#         end
#     end

#     # Adjoint Chisel test.
#     if valence < 2
#         return passing
#     end
#     a = rand(1:(valence-1)); e= rand((a+1):valence)
#     ac = AdjointChisel(valence, a,e)
#     @assert engaged(ac) == [ i == a || i == e for i in 1:valence ] "AdjointChisel engaged axes incorrect."
#     @assert size(ac, 1) == 1 "AdjointChisel should have one row."
#     u = zeros(valence)
#     u[a] = 1.0
#     v = zeros(valence)
#     v[e] = 1.0
#     @assert isapprox(EvaluateChisel(ac,u+v), 0.0)  "AdjointChisel nullity incorrect $ac\r\n $u\r\n $v."


#     # Centroid Chisel test.
#     e = sum(eng)
#     if e < 2
#         return passing
#     end
#     cc = CentroidChisel(eng)
#     @assert engaged(cc) == eng "CentroidChisel engaged axes incorrect\r\n $cc\r\rn $eng"
#     @assert size(cc, 1) == e*(e-1) ÷ 2 "CentroidChisel should have correct number of rows."
#     @assert size(cc, 2) == valence "CentroidChisel should have correct number of columns."
#     # @assert LinearAlgebra.rank(cc) == sum(eng)-1 "CentroidChisel nullity incorrect \r\n $cc"
#     u = zeros(valence)
#     for a in 1:valence
#         if eng[a]
#             u[a] = 1.0
#         end
#     end
#     @assert isapprox(EvaluateChisel(cc,u), 0.0)  "CentroidChisel nullity incorrect\r\n $cc \r\n $u."
#     return true
# end;
    