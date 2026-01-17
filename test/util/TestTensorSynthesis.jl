function testNorm(
        deltas::Vector{<:Vector{<:Number}}, 
        frames::Dleto.Framing{Index}, 
        ch::Dleto.Chisel, 
        num::Integer
    ) ::Bool
    for _ = 1:num
        IT = Dleto.randomTensorChisel(deltas, frames.frame, ch; cutoff=0.1)
        normC = Dleto.normTensorChisel(IT, deltas, ch)
        @assert (normC.norm >= 0) && (normC.norm <= 1) "Norm should be between 0 and 1"
        @assert normC.norm < 0.1 "Warning Norm too large"
        rIT = random_itensor(frames.frame)
        rnorm = Dleto.normTensorChisel(rIT, deltas, ch)
        @assert (rnorm.norm >= 0) && (rnorm.norm <= 1) "Norm should be between 0 and 1"
        if rnorm.norm < 0.1 
            n = rnorm.norm
            println("Warning Norm of random tnesor less than 0.1: $n")
        end
        @assert rnorm.norm > 0.01 "Warning Norm of random tnesor too small"

        # l2normIT = (IT*IT).tensor[]
        # fch = ChiselFramed(ch, vcat(frame...))
        # deltas_as_it = [ 
        #                     ITensor(
        #                         LinearAlgebra.Diagonal(deltas[i]), 
        #                         frame[i], 
        #                         Dleto.__globalOpsMakeTempIndex(frame[i]) 
        #                         ) 
        #                 for i = 1:length(deltas)]
        # resIT= applyDerivation(IT,deltas_as_it,fch)
        # l2normresIT = (resIT*resIT).tensor[]
        # @assert l2normIT >= 0 "L2 norm must be non negative"
        # @assert l2normresIT >= 0 "L2 norm must be non negative"
        # ratio = l2normresIT/(l2normIT + 1e-15)
        # # @show ratio
        # @assert ratio < 0.1 "after derivation norm must be smaller"
    end
    return true
end;

@testset "Testing Tensor Synthesis" begin
    for val = 2:5
        # @show val
        @testset "Valency $val Tests" begin
            for _ = 1:5
                axisdim = rand(5:10, val)
                axisdim = rand(3:7, val)
                frames = (axisdim .|> x -> ITensors.Index(x) ) 
                F = Dleto.Framing{Index}(frames) 
                deltas = randn.(axisdim) 
                @test testNorm(deltas, F, Dleto.UniversalChisel(frames), 10)
                @test testNorm(deltas, F, Dleto.CentroidChisel(frames), 10)
                #random chisels
                @test testNorm(deltas, F, Dleto.Chisel(randn(2,val),frames), 5)
                @test testNorm(deltas, F, Dleto.Chisel(randn(3,val),frames), 5)

                deltas = axisdim .|> (n -> [ i for i= (-n÷ 2):(-n÷ 2 +n -1)] ) 
                @test testNorm(deltas, F, Dleto.UniversalChisel(frames), 10)
                @test testNorm(deltas, F, Dleto.CentroidChisel(frames), 10)
                #random chisels
                @test testNorm(deltas, F, Dleto.Chisel(randn(2,val),frames), 5)
                @test testNorm(deltas, F, Dleto.Chisel(randn(3,val),frames), 5)
            end
        end
    end
end
