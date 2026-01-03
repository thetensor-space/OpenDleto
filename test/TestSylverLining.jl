#
# Test SylverLining
#
using Dleto
using ITensors
using LinearAlgebra
using LinearMaps

LΩs=[ LocalUniversalOps(), LocalDiagonalOps(), LocalSymmetricOps(), LocalAntiSymmetricOps(), LocalScalarOps(), LocalEmptyOps() ];

#only tests the maps are transpose to each other and that the composition is OK
#the test are as function since I am lazy to figure our how linear maps work

function testTranspose(f::Function, fdual::Function, m::Integer, n::Integer, num::Integer)
    for i = 1:num
        a=rand(m)
        B=rand(n)
        A=f(a)
        b=fdual(B)
        @assert isapprox(LinearAlgebra.dot(a,b),LinearAlgebra.dot(A,B)) "Failed Transpose $a $B $A $b" 
    end
    return true; 
end;

function testComposition(f::Function, fdual::Function, g::Function, n::Integer, num::Integer)
    for i = 1:num
        a=rand(n)
        A=f(a)
        AA=fdual(A)
        C=g(a)
        @assert isapprox(C,AA) "Failed Transpose $a $A $AA $C" 
    end
    return true; 
end;

function testGlobalOpChisel(Ω ::AbstractGlobalOps, ch::Matrix, num::Integer)
    # @show ch
    @testset "Testing Ω with ch" begin
        for i = 1:num
            Γ = random_itensor(frames(Ω))
            ester, sylve, sylvester, opdim, densor_dim, derdensor_map, densor_map = sylvesterLM(Ω, ch, Γ)
            @test testTranspose(ester, sylve, opdim, densor_dim, 100)
            @test testComposition( ester, sylve, sylvester, opdim, 100)
        end
    end
end

function testGlobalOp(Ω ::AbstractGlobalOps)
    # @show Ω
    @testset "Testing Ω" begin
        #Random Tucher  
        for i = 1:10
            eng =  [ rand(1:10) % 2 == 1 for x in 1:valency(Ω) ]
            if sum(eng) != 0
                ch = TuckerChisel(eng)
                testGlobalOpChisel(Ω,ch,10) 
            end
        end
        #Random Derivation  
        for i = 1:10
            eng =  [ rand(1:10) % 2 == 1 for x in 1:valency(Ω) ]
            if sum(eng) != 0
                ch = UniversalChisel(eng)
                testGlobalOpChisel(Ω,ch,10) 
            end
        end
        #Random Centroid  
        for i = 1:10
            eng =  [ rand(1:10) % 2 == 1 for x in 1:valency(Ω) ]
            if sum(eng) > 1
                ch = CentroidChisel(eng)
                testGlobalOpChisel(Ω,ch,10) 
            end
        end
    end
end

@testset "SylverLining Tests" begin
    for val = 3:5
        @testset "Valency $val Tests" begin
            for i = 1:3
                axisdim = rand(2:5, val)
                frames = axisdim .|>  (i-> Index(i,"dim $i"))
                localops = rand(1:length(LΩs), val) .|> (i-> LΩs[i] )
                Ω = GlobalOpsIndependant(frames,localops)
                testGlobalOp(Ω)
            end 
        end
    end
end


# function basicTest()
#     fr = [ Index(2, "x"), Index(2, "y"), Index(2, "z") ]
#     eng = [ true, false, false ]
#     Ω = UniversalOps(fr, eng)
#     ch = TuckerChisel( eng )
#     println("Chisel:\n", ch, "\nEngaged: ", engaged(ch))
#     Γ = ITensor( Array(reshape(1:8, (2,2,2))), fr... )
#     sylvester, ester = sylvesterLM(Ω, ch, Γ)
#     return sylvester, ester
# end

# function testAllSylverLining()
#     Ω = TransverseOps(fr3)
#     P_eng = [ true, false, true, true ]
#     P = TuckerChisel(P_eng)
#     testSylvesterLM(Ω, P)
#     println("✓ All Delto SylverLining tests passed.")
#     return true
# end

# function testSylvesterLM(Ω::TransverseOps, P::AbstractMatrix)
#     for _ in 1:10 
#         # Create a random tensor fitting chisel P
#         val = size(P, 2)
#         dims = [ rand(5:30) for _ in 1:val ]
#         frame = collect([ Index(dims[a], "a_$a") for a in 1:val ])
#         Γ = ITensor( Array(randn(dims...)), frame...)

#         # build sylvester linear map pair.
#         sylvester, ester = sylvesterLM(Ω, P, Γ)
#         # test that sylvester = ester' * ester on some inputs.
#         for _ in 1:5
#             u = randn(dim(ester,2))
#             @assert isapprox(sylvester(u), ester'(ester(u))) "Sylvester LM failed to satisfy defining property"
#         end
#         # confirm trivial derivations are zero.
#         eng = engaged(P)
#         for a in 1:sum(eng)
#             for e in 1:sum(eng)
#                 Xs = [ITensor(zeros(dim[b], dim[b]), frame[b], frame[b]') for b in 1:val if eng[b] ]        
#                 X[a] = ITensor(Matrix(I,dims[e],dims[e]), frame[a], frame[a]')
#                 X[e] = ITensor(Matrix(I,dims[a],dims[a]), frame[e], frame[e]')

#                 X_Ω = member(Ω, Xs)
#                 @assert isapprox(sylvester * X_Ω, zeros(size(sylvester,1)))  "Sylvester failed to have scalar derivations in nullspace"
#             end
#         end
#         u = member(Ω, scalar)
#     end
#     return true
# end
