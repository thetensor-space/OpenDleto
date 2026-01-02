# Global Operator Testing

# TODO many things, but mose packking and unpacking functions work as expected and satsify the reuired properties


LΩs=[ LocalUniversalOps(), LocalDiagonalOps(), LocalSymmetricOps(), LocalAntiSymmetricOps() ];

function testInverse(Ω::AbstractGlobalOps, num::Integer)
    globaldim = globalDim(Ω)
    for i = 1:num
        a = rand(globaldim)
        # Mats = embeddingMatrices(Ω,a)
        # ITs = embeddingITensors(Ω,a)
        Mats = unsafe_embeddingMatrices(Ω,a)
        ITs = unsafe_embeddingITensors(Ω,a)
        # @assert isapprox(a, coordinates(Ω, Mats)) "Failed Invertability\r\n $Ω $a $Mats \r\n"
        # @assert isapprox(a, coordinates(Ω, ITs)) "Failed Invertability\r\n $Ω $a $ITs \r\n"
        @assert isapprox(a, unsafe_coordinates(Ω, Mats)) "Failed Invertability\r\n $Ω $a $Mats \r\n"
        @assert isapprox(a, unsafe_coordinates(Ω, ITs)) "Failed Invertability\r\n $Ω $a $ITs \r\n"
    end
    return true
end;

function testTranspose(Ω::AbstractGlobalOps, num::Integer)
    globaldim = globalDim(Ω)
    adim=axisDims(Ω)
    for c = 1:num
        a = rand(globaldim)
        # AAs = embeddingMatrices(Ω,a)
        AAs = unsafe_embeddingMatrices(Ω,a)
        BBs = [ rand(adim[i],adim[i]) for i=1:length(adim) ]
        # b = transposeEmbedding(Ω,BBs) 
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


@testset "GlobalOperators Tests" begin
    for val = 2:10
        @testset "Valency $val Tests" begin
            for i = 1:50
                axisdim = rand(3:50, val)
                frames = axisdim .|>  (i-> Index(i,"dim $i"))
                localops = rand(1:4, val) .|> (i-> LΩs[i] )
                Ω = GlobalOps(frames,localops)
                @test testInverse(Ω, 100)
                @test testTranspose(Ω, 100)
            end 
        end
    end
end