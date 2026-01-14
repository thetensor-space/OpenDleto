function testNorm(deltas::Vector{<:Vector{<:Number}}, ch::Matrix, num::Integer) ::Bool
    for _ = 1:num
        IT = randTensorChisel(deltas, 0.1, ch)
        norm = ITensorNormChisel(IT, deltas, ch)
        frame = inds(IT)
        @assert (norm >= 0) && (norm <= 1) "Norm should be between 0 and 1"
        @assert norm < 0.1 "Warning Norm too large"
        rIT = random_itensor(frame)
        rnorm = ITensorNormChisel(rIT, deltas, ch)
        @assert (rnorm >= 0) && (rnorm <= 1) "Norm should be between 0 and 1"
        @assert rnorm > 0.1 "Warning Norm of random tnesor too small"

        l2normIT = (IT*IT).tensor[]
        fch = ChiselFramed(ch, vcat(frame...))
        deltas_as_it = [ 
                            ITensor(
                                LinearAlgebra.Diagonal(deltas[i]), 
                                frame[i], 
                                Dleto.__globalOpsMakeTempIndex(frame[i]) 
                                ) 
                        for i = 1:length(deltas)]
        resIT= applyDerivation(IT,deltas_as_it,fch)
        l2normresIT = (resIT*resIT).tensor[]
        @assert l2normIT >= 0 "L2 norm must be non negative"
        @assert l2normresIT >= 0 "L2 norm must be non negative"
        ratio = l2normresIT/(l2normIT + 1e-15)
        # @show ratio
        @assert ratio < 0.1 "after derivation norm must be smaller"
    end
    return true
end;

@testset "Testing Tensor Synthesis" begin
    for val = 2:6
        # @show val
        @testset "Valency $val Tests" begin
            axisdim = rand(5:10, val)
            deltas = randn.(axisdim) 
            @test testNorm(deltas, UniversalChisel(val), 10)
            @test testNorm(deltas, CentroidChisel(val), 10)
            if val > 4
                for _ = 1:3
                    eng = [ rand(1:10) % 2 == 1 for i in 1:val ]
                    if sum(eng) > 1
                        @test testNorm(deltas, UniversalChisel(eng), 5)
                        @test testNorm(deltas, CentroidChisel(eng), 5)
                    end
                end
            end
            #random chisels
            @test testNorm(deltas, randn(2,val), 5)
            @test testNorm(deltas, randn(3,val), 5)
        end
    end
end
