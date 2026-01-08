# Global Operator Testing

# TODO many things,
# the only implemened tests are that packking and unpacking functions work as expected 
# and satsify the required properties

# reduceByEngaged not tested

LΩs=[ UniversalOp(), DiagonalOp(), SymmetricOp(), AntiSymmetricOp(), ScalarOp(), EmptyOp() ];

function testInverse(Ω::TransverseOps, num::Integer)
    globaldim = globalDim(Ω)
    for _ = 1:num
        a = randn(globaldim)
        Mats = embedMatrices(Ω,a)
        @assert isapprox(a, coordinates(Ω, Mats)) "Failed Invertability\r\n $Ω $a $Mats \r\n"
        @assert isapprox(a, unsafe_coordinates(Ω, Mats)) "Failed Invertability\r\n $Ω $a $Mats \r\n"
        Mats = unsafe_embedMatrices(Ω,a)
        @assert isapprox(a, coordinates(Ω, Mats)) "Failed Invertability\r\n $Ω $a $Mats \r\n"
        @assert isapprox(a, unsafe_coordinates(Ω, Mats)) "Failed Invertability\r\n $Ω $a $Mats \r\n"

        ITs = embedITensors(Ω,a)
        @assert isapprox(a, coordinates(Ω, ITs)) "Failed Invertability\r\n $Ω $a $ITs \r\n"
        @assert isapprox(a, unsafe_coordinates(Ω, ITs)) "Failed Invertability\r\n $Ω $a $ITs \r\n"
        ITs = unsafe_embedITensors(Ω,a)
        @assert isapprox(a, coordinates(Ω, ITs)) "Failed Invertability\r\n $Ω $a $ITs \r\n"
        @assert isapprox(a, unsafe_coordinates(Ω, ITs)) "Failed Invertability\r\n $Ω $a $ITs \r\n"
    end
    return true
end;

function testTranspose(Ω::TransverseOps, num::Integer)
    globaldim = globalDim(Ω)
    adim=axisDims(Ω)
    for _ = 1:num
        a = randn(globaldim)
        BBs = [ randn(adim[i],adim[i]) for i=1:length(adim) ]

        AAs = embedMatrices(Ω,a)
        b = transposeEmbed(Ω,BBs) 
        @assert isapprox(
            LinearAlgebra.dot(a,b), 
            sum([LinearAlgebra.dot(
                        reshape(AAs[i],adim[i]*adim[i]),
                        reshape(BBs[i],adim[i]*adim[i])
                    )  for i=1:length(adim) ] )
        ) "Failed Transpose\r\n $Ω $a $AAs $BBs $b\r\n"
        b = unsafe_transposeEmbed(Ω,BBs) 
        @assert isapprox(
            LinearAlgebra.dot(a,b), 
            sum([LinearAlgebra.dot(
                        reshape(AAs[i],adim[i]*adim[i]),
                        reshape(BBs[i],adim[i]*adim[i])
                    )  for i=1:length(adim) ] )
        ) "Failed Transpose\r\n $Ω $a $AAs $BBs $b\r\n"


        AAs = unsafe_embedMatrices(Ω,a)
        b = transposeEmbed(Ω,BBs) 
        @assert isapprox(
            LinearAlgebra.dot(a,b), 
            sum([LinearAlgebra.dot(
                        reshape(AAs[i],adim[i]*adim[i]),
                        reshape(BBs[i],adim[i]*adim[i])
                    )  for i=1:length(adim) ] )
        ) "Failed Transpose\r\n $Ω $a $AAs $BBs $b\r\n"
        b = unsafe_transposeEmbed(Ω,BBs) 
        @assert isapprox(
            LinearAlgebra.dot(a,b), 
            sum([LinearAlgebra.dot(
                        reshape(AAs[i],adim[i]*adim[i]),
                        reshape(BBs[i],adim[i]*adim[i])
                    )  for i=1:length(adim) ] )
        ) "Failed Transpose\r\n $Ω $a $AAs $BBs $b\r\n"
    end
    return true
end;

function testReducedbyEngaged(Ω::TransverseOps, num::Integer)
    globaldim = globalDim(Ω)
    val=valency(Ω)
    for _=1:num
        eng = [ rand(1:10) % 2 == 1 for i in 1:val ]
        reducedval= sum(eng)
        if reducedval==0
            continue
        end
        (rΩ, expand_map) = reduceByEngaged(Ω, eng)
        reduceddim=globalDim(rΩ)
        # test transpose of expand map
        for _ = 1:10
            a=randn(reduceddim)
            b=randn(globaldim)
            BB=expand_map(a)
            AA=expand_map'(b)
            @assert isapprox( LinearAlgebra.dot(b,BB), LinearAlgebra.dot(a,AA)) "Transpose failed for expand map"
            allMats = embedMatrices(Ω, BB)
            reducedMats = embedMatrices(rΩ, a)
            allMatsR = allMats[eng]
            @assert all( [isapprox(reducedMats[i],allMatsR[i] ) for i=1:reducedval]) "Matrices are not the same"
            @assert all( [eng[i] || all(allMats[i] .|> Dleto.__isapproxzero)  for i=1:val]) "Matrices are not zero"
        end
    end
    return true;
end;

@testset "TransverseOpsIndependant Tests" begin
    for val = 2:10
        @testset "Valency $val Tests" begin
            for _ = 1:30
                axisdim = rand(1:15, val)
                frames = axisdim .|>  (i-> Index(i,"dim $i"))
                localops = rand(1:length(LΩs), val) .|> (i-> LΩs[i] )
                Ω = IndTransverseOps(frames,localops)
                @test testInverse(Ω, 100)
                @test testTranspose(Ω, 100)
                @test testReducedbyEngaged(Ω,100)
            end 
        end
    end
end