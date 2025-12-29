# Chisel Testing.

using ITensors
using Dleto

function testAllChisels()
    for valence in 1:20
        testChisels(valence)
    end
    println("All Chisel tests passed.")
end

function testChisels(valence::Integer)
    # test numeric.
    eng = [ true for i in 1:valence ]
    test(eng)

    for _ in 1:5
        # test random engaged.
        eng = [ rand(1:10) % 2 == 1 for i in 1:valence ]
        test(eng)
    end
end

function test(eng::Vector{Bool})
    valence = length(eng)
    # Universal Chisel test.
    uc = UniversalChisel(eng)
    @assert engaged(uc) == eng "UniversalChisel engaged axes incorrect:\r\n $eng \r\n $uc"
    @assert size(uc, 1) == 1 "UniversalChisel should have one row."
    @assert size(uc, 2) == valence "UniversalChisel should have correct number of columns."
    for a in 1:valence
        for e in 1:valence
            if eng[a] && eng[e] 
                u = zeros(valence)
                u[a] = 1.0
                v = zeros( valence)
                v[e] = 1.0
                @assert isapprox(uc*(u - v), zeros(size(uc,1)))  "UniversalChisel nullity incorrect."
            end
        end
    end

    # Tucker Chisel test.
    tc = TuckerChisel(eng)
    @assert engaged(tc) == eng "TuckerChisel engaged axes incorrect."
    @assert size(tc, 1) == sum(eng) "TuckerChisel should have correct number of rows."
    @assert size(tc, 2) == valence "TuckerChisel should have correct number of columns."
    for a in 1:valence
        if !eng[a]
            u = zeros(valence)
            u[a] = 1.0
            @assert isapprox(tc*u, zeros(size(tc,1)))  "TuckerChisel image incorrect, $eng."
        end
    end

    # Adjoint Chisel test.
    if valence < 2
        return
    end
    a = rand(1:(valence-1)); e= rand((a+1):valence)
    ac = AdjointChisel(valence, a,e)
    @assert engaged(ac) == [ i == a || i == e for i in 1:valence ] "AdjointChisel engaged axes incorrect."
    @assert size(ac, 1) == 1 "AdjointChisel should have one row."
    u = zeros(valence)
    u[a] = 1.0
    v = zeros(valence)
    v[e] = 1.0
    @assert isapprox(ac*(u+v), zeros(size(ac,1)))  "AdjointChisel nullity incorrect $ac\r\n $u\r\n $v."

    # Centroid Chisel test.
    e = sum(eng)
    if e < 2
        return
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
    @assert isapprox(cc*u, zeros(size(cc,1)))  "CentroidChisel nullity incorrect\r\n $cc \r\n $u."


end
