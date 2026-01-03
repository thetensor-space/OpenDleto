# Global Operator Testing

# TODO many things,
# the only implemened tests are that packking and unpacking functions work as expected 
# and satsify the required properties

# reduceByEngaged not tested

LΩs=[ LocalUniversalOps(), LocalDiagonalOps(), LocalSymmetricOps(), LocalAntiSymmetricOps(), LocalScalarOps(), LocalEmptyOps() ];

function testInverse(Ω::AbstractGlobalOps, num::Integer)
    globaldim = globalDim(Ω)
    for i = 1:num
        a = rand(globaldim)
        Mats = embeddingMatrices(Ω,a)
        @assert isapprox(a, coordinates(Ω, Mats)) "Failed Invertability\r\n $Ω $a $Mats \r\n"
        @assert isapprox(a, unsafe_coordinates(Ω, Mats)) "Failed Invertability\r\n $Ω $a $Mats \r\n"
        Mats = unsafe_embeddingMatrices(Ω,a)
        @assert isapprox(a, coordinates(Ω, Mats)) "Failed Invertability\r\n $Ω $a $Mats \r\n"
        @assert isapprox(a, unsafe_coordinates(Ω, Mats)) "Failed Invertability\r\n $Ω $a $Mats \r\n"

        ITs = embeddingITensors(Ω,a)
        @assert isapprox(a, coordinates(Ω, ITs)) "Failed Invertability\r\n $Ω $a $ITs \r\n"
        @assert isapprox(a, unsafe_coordinates(Ω, ITs)) "Failed Invertability\r\n $Ω $a $ITs \r\n"
        ITs = unsafe_embeddingITensors(Ω,a)
        @assert isapprox(a, coordinates(Ω, ITs)) "Failed Invertability\r\n $Ω $a $ITs \r\n"
        @assert isapprox(a, unsafe_coordinates(Ω, ITs)) "Failed Invertability\r\n $Ω $a $ITs \r\n"
    end
    return true
end;

function testTranspose(Ω::AbstractGlobalOps, num::Integer)
    globaldim = globalDim(Ω)
    adim=axisDims(Ω)
    for c = 1:num
        a = rand(globaldim)
        BBs = [ rand(adim[i],adim[i]) for i=1:length(adim) ]

        AAs = embeddingMatrices(Ω,a)
        b = transposeEmbedding(Ω,BBs) 
        @assert isapprox(
            LinearAlgebra.dot(a,b), 
            sum([LinearAlgebra.dot(
                        reshape(AAs[i],adim[i]*adim[i]),
                        reshape(BBs[i],adim[i]*adim[i])
                    )  for i=1:length(adim) ] )
        ) "Failed Transpose\r\n $Ω $a $AAs $BBs $b\r\n"
        b = unsafe_transposeEmbedding(Ω,BBs) 
        @assert isapprox(
            LinearAlgebra.dot(a,b), 
            sum([LinearAlgebra.dot(
                        reshape(AAs[i],adim[i]*adim[i]),
                        reshape(BBs[i],adim[i]*adim[i])
                    )  for i=1:length(adim) ] )
        ) "Failed Transpose\r\n $Ω $a $AAs $BBs $b\r\n"


        AAs = unsafe_embeddingMatrices(Ω,a)
        b = transposeEmbedding(Ω,BBs) 
        @assert isapprox(
            LinearAlgebra.dot(a,b), 
            sum([LinearAlgebra.dot(
                        reshape(AAs[i],adim[i]*adim[i]),
                        reshape(BBs[i],adim[i]*adim[i])
                    )  for i=1:length(adim) ] )
        ) "Failed Transpose\r\n $Ω $a $AAs $BBs $b\r\n"
        b = unsafe_transposeEmbedding(Ω,BBs) 
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


@testset "Independent Global Operators Tests" begin
    for val = 2:10
        @testset "Valency $val Tests" begin
            for i = 1:30
                axisdim = rand(1:15, val)
                frames = axisdim .|>  (i-> Index(i,"dim $i"))
                localops = rand(1:length(LΩs), val) .|> (i-> LΩs[i] )
                Ω = GlobalOpsIndependant(frames,localops)
                @test testInverse(Ω, 100)
                @test testTranspose(Ω, 100)
            end 
        end
    end
end

@testset "Symetries Global Operators Tests" begin
    for val = 2:10
        @testset "NoSymmetry Valency $val Tests" begin
            for i = 1:30
                axisdim = rand(1:15, val)
                frames = axisdim .|>  (i-> Index(i,"dim $i"))
                localops = rand(1:length(LΩs), val) .|> (i-> LΩs[i] )
                Ω = GlobalOpsSymmetries(frames,localops,[i for i=1:val])
                @test testInverse(Ω, 100)
                @test testTranspose(Ω, 100)
            end 
        end
    end
end