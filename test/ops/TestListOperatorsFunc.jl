# Global Operator Testing

# TODO many things,
# the only implemened tests are that packking and unpacking functions work as expected 
# and satsify the required properties


LinOps=[ Dleto.LinearOperatorsDict[k] for k in keys(Dleto.LinearOperatorsDict)]
InvOps=[ Dleto.InvertableOperatorsDict[k] for k in keys(Dleto.InvertableOperatorsDict)]

AllOps = vcat(LinOps,InvOps)


function testInverse(LOp::Dleto.ListOperators, num::Integer)
    globaldim = Dleto.globalDim(LOp)
    for _ = 1:num
        a = Dleto.generate_random(LOp)
        Mats = Dleto.embedMatrices(LOp,a)
        @assert isapprox(a, Dleto.coordinates(LOp, Mats)) "Failed Invertability ss"
        @assert isapprox(a, Dleto.unsafe_coordinates(LOp, Mats)) "Failed Invertability su"
        Mats = Dleto.unsafe_embedMatrices(LOp,a)
        @assert isapprox(a, Dleto.coordinates(LOp, Mats)) "Failed Invertability\r\n $LOp $a $Mats \r\n"
        @assert isapprox(a, Dleto.unsafe_coordinates(LOp, Mats)) "Failed Invertability\r\n $LOp $a $Mats \r\n"
    end
    return true
end;


function testReducedbyEngaged(LOp::Dleto.ListOperators, num::Integer,testzeros::Bool=true)
    globaldim = Dleto.globalDim(LOp)
    val=Dleto.valency(LOp)
    for _=1:num
        eng = [ rand(Bool) for i in 1:val ]
        reducedval= sum(eng)
        if reducedval==0
            continue
        end
        res = Dleto.reduceBy(LOp, eng)
        rLOp=res.rLOp
        reduce_map=res.reduce_map
        expand_func=res.expand_func
        reduceddim=Dleto.globalDim(rLOp)
        if reducedval==0
            continue
        end
        # test transpose of expand map
        for _ = 1:10
            a=randn(reduceddim)
            b=randn(globaldim)
            BB=reduce_map'(a)
            AA=reduce_map(b)
            if !isapprox( LinearAlgebra.dot(b,BB), LinearAlgebra.dot(a,AA))
                @show LOp
                @show eng
                @show rLOp
                @show globaldim,reduceddim
            end
            @assert isapprox( LinearAlgebra.dot(b,BB), LinearAlgebra.dot(a,AA)) "Transpose failed for expand map"
        end
        for _ = 1:10
            b=Dleto.generate_random(LOp)
            allMats = Dleto.embedMatrices(LOp, b)
            allMatsR = allMats[eng]
            AA=reduce_map(b)
            reducedMats = Dleto.embedMatrices(rLOp, AA)
            @assert all( [isapprox(reducedMats[i],allMatsR[i] ) for i=1:reducedval]) "Matrices are not the same"

            a=Dleto.generate_random(rLOp)
            reducedMats = Dleto.embedMatrices(rLOp, a)
            BB=expand_func(a)
            allMats = Dleto.embedMatrices(LOp, BB)
            allMatsR = allMats[eng]
            @assert all( [isapprox(reducedMats[i],allMatsR[i] ) for i=1:reducedval]) "Matrices are not the same"
        end
    end
    return true;
end;


function testTranspose(LOp::Dleto.ListOperators, num::Integer)
    Dleto.isLinear(LOp) || return true
    globaldim = Dleto.globalDim(LOp)
    adim=Dleto.axisDims(LOp)
    for _ = 1:num
        a = randn(globaldim)
        BBs = [ randn(adim[i],adim[i]) for i=1:length(adim) ]

        AAs = Dleto.embedMatrices(LOp,a)
        b = Dleto.transposeEmbed(LOp,BBs) 
        @assert isapprox(
            LinearAlgebra.dot(a,b), 
            sum([LinearAlgebra.dot(
                        reshape(AAs[i],adim[i]*adim[i]),
                        reshape(BBs[i],adim[i]*adim[i])
                    )  for i=1:length(adim) ] )
        ) "Failed Transpose\r\n $LOp $a $AAs $BBs $b\r\n"
        b = Dleto.unsafe_transposeEmbed(LOp,BBs) 
        @assert isapprox(
            LinearAlgebra.dot(a,b), 
            sum([LinearAlgebra.dot(
                        reshape(AAs[i],adim[i]*adim[i]),
                        reshape(BBs[i],adim[i]*adim[i])
                    )  for i=1:length(adim) ] )
        ) "Failed Transpose\r\n $LOp $a $AAs $BBs $b\r\n"


        AAs = Dleto.unsafe_embedMatrices(LOp,a)
        b = Dleto.transposeEmbed(LOp,BBs) 
        @assert isapprox(
            LinearAlgebra.dot(a,b), 
            sum([LinearAlgebra.dot(
                        reshape(AAs[i],adim[i]*adim[i]),
                        reshape(BBs[i],adim[i]*adim[i])
                    )  for i=1:length(adim) ] )
        ) "Failed Transpose\r\n $LOp $a $AAs $BBs $b\r\n"
        b = Dleto.unsafe_transposeEmbed(LOp,BBs) 
        @assert isapprox(
            LinearAlgebra.dot(a,b), 
            sum([LinearAlgebra.dot(
                        reshape(AAs[i],adim[i]*adim[i]),
                        reshape(BBs[i],adim[i]*adim[i])
                    )  for i=1:length(adim) ] )
        ) "Failed Transpose\r\n $LOp $a $AAs $BBs $b\r\n"
    end
    return true
end;


function testSymmetry(LOp::Dleto.ListOperators, num::Integer, localops::Vector{<:Dleto.Operator}, sym::Vector{<:Integer})
    globaldim = Dleto.globalDim(LOp)
    for _ = 1:num
        a = Dleto.generate_random(LOp)
        Mats = Dleto.embedMatrices(LOp,a)
        for i = 1:length(sym)  
            if sym[i]==i
                continue
            end
            if sym[i] > 0 
                @assert isapprox(Mats[i], Mats[sym[i]]) "Failed Matrices at $i and $sym[i] are not the same"
            else
                if isa(localops[i], Dleto.LinearOperator)
                    if !isapprox(LinearAlgebra.Transpose(Mats[i]), Mats[-sym[i]])
                        @show i sym localops
                        @show Mats[i] Mats[-sym[i]]
                    end 
                    @assert isapprox(LinearAlgebra.Transpose(Mats[i]), Mats[-sym[i]]) "Failed Matrices at $i and $sym[i] are not transposed"
                elseif isa(localops[i], Dleto.InvertableOperator)
                    @assert isapprox(LinearAlgebra.Transpose(inv(Mats[i])), Mats[-sym[i]]) "Failed Matrices at $i and $sym[i] are not inverse-transpose"
                else
                    @assert false "Unknown Operator"
                end
            end
        end
    end
    return true
end;










# LΩs=[ UniversalOp(), DiagonalOp(), SymmetricOp(), AntiSymmetricOp(), ScalarOp(), EmptyOp() ];

# syms = [ [1,1,1,1], [1,1,3,3,5,5], [1,2,1,2,2,1], [1,2,3,-1,2,3], [1,2,-2,-2,5,-5,-1,-1], [1,-1,3,3,-3,1,7,8],[1,2,3,4,-4,-4,-4,-3,-3,-3,-2,-2,-1,-1] ]

# function testInverse(Ω::TransverseOps, num::Integer)
#     globaldim = globalDim(Ω)
#     for _ = 1:num
#         a = randn(globaldim)
#         Mats = embedMatrices(Ω,a)
#         @assert isapprox(a, coordinates(Ω, Mats)) "Failed Invertability ss"
#         @assert isapprox(a, unsafe_coordinates(Ω, Mats)) "Failed Invertability su"
#         Mats = unsafe_embedMatrices(Ω,a)
#         @assert isapprox(a, coordinates(Ω, Mats)) "Failed Invertability\r\n $Ω $a $Mats \r\n"
#         @assert isapprox(a, unsafe_coordinates(Ω, Mats)) "Failed Invertability\r\n $Ω $a $Mats \r\n"

#         ITs = embedITensors(Ω,a)
#         @assert isapprox(a, coordinates(Ω, ITs)) "Failed Invertability\r\n $Ω $a $ITs \r\n"
#         @assert isapprox(a, unsafe_coordinates(Ω, ITs)) "Failed Invertability\r\n $Ω $a $ITs \r\n"
#         ITs = unsafe_embedITensors(Ω,a)
#         @assert isapprox(a, coordinates(Ω, ITs)) "Failed Invertability\r\n $Ω $a $ITs \r\n"
#         @assert isapprox(a, unsafe_coordinates(Ω, ITs)) "Failed Invertability\r\n $Ω $a $ITs \r\n"
#     end
#     return true
# end;

# function testTranspose(Ω::TransverseOps, num::Integer)
#     globaldim = globalDim(Ω)
#     adim=axisDims(Ω)
#     for _ = 1:num
#         a = randn(globaldim)
#         BBs = [ randn(adim[i],adim[i]) for i=1:length(adim) ]

#         AAs = embedMatrices(Ω,a)
#         b = transposeEmbed(Ω,BBs) 
#         @assert isapprox(
#             LinearAlgebra.dot(a,b), 
#             sum([LinearAlgebra.dot(
#                         reshape(AAs[i],adim[i]*adim[i]),
#                         reshape(BBs[i],adim[i]*adim[i])
#                     )  for i=1:length(adim) ] )
#         ) "Failed Transpose\r\n $Ω $a $AAs $BBs $b\r\n"
#         b = unsafe_transposeEmbed(Ω,BBs) 
#         @assert isapprox(
#             LinearAlgebra.dot(a,b), 
#             sum([LinearAlgebra.dot(
#                         reshape(AAs[i],adim[i]*adim[i]),
#                         reshape(BBs[i],adim[i]*adim[i])
#                     )  for i=1:length(adim) ] )
#         ) "Failed Transpose\r\n $Ω $a $AAs $BBs $b\r\n"


#         AAs = unsafe_embedMatrices(Ω,a)
#         b = transposeEmbed(Ω,BBs) 
#         @assert isapprox(
#             LinearAlgebra.dot(a,b), 
#             sum([LinearAlgebra.dot(
#                         reshape(AAs[i],adim[i]*adim[i]),
#                         reshape(BBs[i],adim[i]*adim[i])
#                     )  for i=1:length(adim) ] )
#         ) "Failed Transpose\r\n $Ω $a $AAs $BBs $b\r\n"
#         b = unsafe_transposeEmbed(Ω,BBs) 
#         @assert isapprox(
#             LinearAlgebra.dot(a,b), 
#             sum([LinearAlgebra.dot(
#                         reshape(AAs[i],adim[i]*adim[i]),
#                         reshape(BBs[i],adim[i]*adim[i])
#                     )  for i=1:length(adim) ] )
#         ) "Failed Transpose\r\n $Ω $a $AAs $BBs $b\r\n"
#     end
#     return true
# end;

# function testSymmetry(Ω::TransverseOps, num::Integer, sym::Vector{<:Integer})
#     globaldim = globalDim(Ω)
#     for _ = 1:num
#         a = randn(globaldim)
#         Mats = embedMatrices(Ω,a)
#         for i = 1:length(sym)  
#             if sym[i]==i
#                 continue
#             end
#             if sym[i] > 0 
#                 @assert isapprox(Mats[i], Mats[sym[i]]) "Failed Matrices at $i and $sym[i] are not the same"
#             else
#                 @assert isapprox(LinearAlgebra.Transpose(Mats[i]), Mats[-sym[i]]) "Failed Matrices at $i and $sym[i] are not transposed"
#             end
#         end
#     end
#     return true
# end;

# function testReducedbyEngaged(Ω::TransverseOps, num::Integer,testzeros::Bool=true)
#     globaldim = globalDim(Ω)
#     val=valency(Ω)
#     for _=1:num
#         eng = [ rand(1:10) % 2 == 1 for i in 1:val ]
#         reducedval= sum(eng)
#         if reducedval==0
#             continue
#         end
#         (rΩ, expand_map) = reduceByEngaged(Ω, eng)
#         reduceddim=globalDim(rΩ)
#         if reducedval==0
#             continue
#         end
#         # test transpose of expand map
#         for _ = 1:10
#             a=randn(reduceddim)
#             b=randn(globaldim)
#             BB=expand_map(a)
#             AA=expand_map'(b)
#             @assert isapprox( LinearAlgebra.dot(b,BB), LinearAlgebra.dot(a,AA)) "Transpose failed for expand map"
#             allMats = embedMatrices(Ω, BB)
#             reducedMats = embedMatrices(rΩ, a)
#             allMatsR = allMats[eng]
#             @assert all( [isapprox(reducedMats[i],allMatsR[i] ) for i=1:reducedval]) "Matrices are not the same"
#             if testzeros
#                 @assert all( [eng[i] || all(allMats[i] .|> Dleto.__isapproxzero)  for i=1:val]) "Matrices are not zero"
#             end
#             allMats = embedMatrices(Ω,b )
#             allMatsR = allMats[eng]
#             aa = coordinates(rΩ,allMatsR)
#             @assert isapprox(aa,AA)
#         end
#     end
#     return true;
# end;

# # @testset "TransverseOps Valency 0 Tests" begin
# #     Ω = IndTransverseOps(Index{Int64}[],Operator[])
# # #    @test testInverse(Ω, 100)
# #     @test testTranspose(Ω, 100)
# # end 

# @testset "TransverseOpsIndependant Tests" begin
#     for val = 1:10
#         @testset "Valency $val Tests" begin
#             for _ = 1:30
#                 axisdim = rand(1:15, val)
#                 frames = axisdim .|>  (i-> Index(i,"dim $i"))
#                 localops = rand(1:length(LΩs), val) .|> (i-> LΩs[i] )
#                 Ω = IndTransverseOps(frames,localops)
#                 @test testInverse(Ω, 100)
#                 @test testTranspose(Ω, 100)
#                 @test testReducedbyEngaged(Ω,10)
#             end 
#         end
#     end
# end

# @testset "TransverseOpsSymmetires Tests" begin
#     for val = 1:10
#         @testset "No Symmetry Valency $val Tests" begin
#             for _ = 1:30
#                 axisdim = rand(1:15, val)
#                 frames = axisdim .|>  (i-> Index(i,"dim $i"))
#                 localops = rand(1:length(LΩs), val) .|> (i-> LΩs[i] )
#                 Ω = TransverseOpsSymmetries(frames,localops,[i for i=1:val])
#                 @test testInverse(Ω, 100)
#                 @test testTranspose(Ω, 100)
#                 @test testSymmetry(Ω, 100, [i for i=1:val])
#                 @test testReducedbyEngaged(Ω,10)
#             end 
#         end
#     end
#     val = 20
#     for sym in syms 
#         @testset "Symmetry $sym Tests" begin
#             for _ = 1:5
#                 axisdim = rand(2:15, val)
#                 frames = axisdim .|>  (i-> Index(i,"dim $i"))
#                 localops = rand(1:length(LΩs), val) .|> (i-> LΩs[i] )
#                 Ω = TransverseOpsSymmetries(
#                     [ prime(frames[abs(sym[i])],i) for i=1:length(sym)],
#                     [ localops[abs(sym[i])] for i=1:length(sym)],
#                     sym)
#                 @test testInverse(Ω, 100)
#                 @test testTranspose(Ω, 100)
#                 @test testSymmetry(Ω, 100, sym)
#                 @test testReducedbyEngaged(Ω,30,false)
#             end
#         end 
#     end
# end