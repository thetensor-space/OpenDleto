#
# TestSylverLining
#
using Dleto
using ITensors

function basicTest()
    fr = [ Index(2, "x"), Index(2, "y"), Index(2, "z") ]
    eng = [ true, false, false ]
    Ω = UniversalOps(fr, eng)
    ch = TuckerChisel( eng )
    println("Chisel:\n", ch, "\nEngaged: ", engaged(ch))
    Γ = ITensor( Array(reshape(1:8, (2,2,2))), fr... )
    sylvester, ester = sylvesterLM(Ω, ch, Γ)
    return sylvester, ester
end

function testAllSylverLining()
    Ω = TransverseOps(fr3)
    P_eng = [ true, false, true, true ]
    P = TuckerChisel(P_eng)
    testSylvesterLM(Ω, P)
    println("✓ All Delto SylverLining tests passed.")
    return true
end

function testSylvesterLM(Ω::TransverseOps, P::AbstractMatrix)
    for _ in 1:10 
        # Create a random tensor fitting chisel P
        val = size(P, 2)
        dims = [ rand(5:30) for _ in 1:val ]
        frame = collect([ Index(dims[a], "a_$a") for a in 1:val ])
        Γ = ITensor( Array(randn(dims...)), frame...)

        # build sylvester linear map pair.
        sylvester, ester = sylvesterLM(Ω, P, Γ)
        # test that sylvester = ester' * ester on some inputs.
        for _ in 1:5
            u = randn(dim(ester,2))
            @assert isapprox(sylvester(u), ester'(ester(u))) "Sylvester LM failed to satisfy defining property"
        end
        # confirm trivial derivations are zero.
        eng = engaged(P)
        for a in 1:sum(eng)
            for e in 1:sum(eng)
                Xs = [ITensor(zeros(dim[b], dim[b]), frame[b], frame[b]') for b in 1:val if eng[b] ]        
                X[a] = ITensor(Matrix(I,dims[e],dims[e]), frame[a], frame[a]')
                X[e] = ITensor(Matrix(I,dims[a],dims[a]), frame[e], frame[e]')

                X_Ω = member(Ω, Xs)
                @assert isapprox(sylvester * X_Ω, zeros(size(sylvester,1)))  "Sylvester failed to have scalar derivations in nullspace"
            end
        end
        u = member(Ω, scalar)
    end
    return true
end